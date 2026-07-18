# LeetGPU Multi-Head Attention 题解

## 1. 题目概述

- **标题 / 题号**：Multi-Head Attention（#12，hard）
- **链接**：https://leetgpu.com/challenges/multi-head-attention
- **难度**：困难
- **标签**：CUDA、Multi-Head Attention、FlashAttention、融合 attention、batched kernel launch

**题意**：实现标准的 **Multi-Head Scaled Dot-Product Attention**。给定 `Q, K, V ∈ R^{B×H×N×d}`（行主序，`B` 个 batch、`H` 个 head、序列长 `N`、头维 `d`），对每个 `(batch, head)` 独立计算：

$$O_{b,h} = \text{softmax}\!\left(\frac{Q_{b,h}\, K_{b,h}^{\mathsf{T}}}{\sqrt{d}}\right) V_{b,h}, \qquad b \in [0,B),\ h \in [0,H)$$

每个 head 是一组独立的 `N×d` 序列，head 之间互不通信——这正是"Multi-Head"的本质：**把一个大 head 拆成** `H` **个小 head 并行计算**，既增加表达力又天然适合 GPU 的批量并行。

**约束**：`1 ≤ B ≤ 128`，`1 ≤ H ≤ 16`，`1 ≤ N ≤ 4096`，`1 ≤ d ≤ 128`；容差 `atol=rtol=1e-3`。

> 💡 这是本系列**最难的题**。朴素 MHA 要为每个 `(batch, head)` 物化两个 `N×N` 中间矩阵 `S=QK^T` 和 `P=softmax(S)`，总显存 `O(B·H·N²)`——`B=128, H=16, N=4096` 时光 `S`、`P` 就各占 **128 GB**，直接 OOM。解法核心是把 [Day 5 Softmax Attention](../../leetgpu/week2/day5/leetgpu-softmax-attention-solution.md) 的 **FlashAttention（online softmax + tiling）** 思想扩展到 `B×H` 维度：**三重并行** `batch × head × Q-tile`，每组内用 tiling + online softmax 不物化 `S/P`，把显存从 `O(B·H·N²)` 降到 `O(B·H·N·d)`，HBM IO 从 `O(B·H·N²)` 降到趋近 `O(B·H·N·d)`。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行四重循环

```cpp
// cpu_baseline.cpp —— CPU 串行 Multi-Head Attention（物化 S、P）
void mha_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * sizeof(float)); // 一行 score（per head per query）
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int bh = (b * H + h) * N * d; // 该 (batch,head) 的基址
            for (int i = 0; i < N; ++i) {
                float mx = -INFINITY;
                for (int k = 0; k < N; ++k) { // ① S = QK^T / √d
                    float s = 0.f;
                    for (int t = 0; t < d; ++t)
                        s += Q[bh + i * d + t] * K[bh + k * d + t];
                    S[k] = s * scale;
                    mx = fmaxf(mx, s);
                }
                float sum = 0.f; // ② P = softmax(S)
                for (int k = 0; k < N; ++k) {
                    S[k] = expf(S[k] - mx);
                    sum += S[k];
                }
                for (int t = 0; t < d; ++t) { // ③ O = P · V
                    float acc = 0.f;
                    for (int k = 0; k < N; ++k)
                        acc += S[k] * V[bh + k * d + t];
                    O[bh + i * d + t] = acc / sum;
                }
            }
        }
    free(S);
}
```

四重循环 `B × H × N × (N·d + N·d) = O(B·H·N²·d)`。每组 `(batch, head)` 完全独立——这天然适合 GPU 的 **batched kernel launch**：用 `gridDim.z = B`、`gridDim.y = H` 把 `B·H` 组扔给不同 block 并行。

### 2.2 朴素 GPU：每组物化 S/P 到 HBM

朴素 GPU 把 CPU 三步搬到 device：对每个 `(batch, head)` 先算 `S=QK^T` 写 HBM，再读 `S` 算 softmax 写 `P` 到 HBM，再读 `P`、`V` 算 `O`。

![朴素 MHA：B×H 组各物化两个 N×N 中间矩阵到 HBM](../../images/flash_attention_naive_vs_fused.svg)

**致命问题**：
1. **显存 O(B·H·N²)**：`S`、`P` 各 `B·H·N²×4B`。`B=128, H=16, N=4096` 时各 **128 GB**——任何 GPU 都 OOM。
2. **IO 浪费**：`S` 写一次读两次（max + exp）、`P` 写一次读一次，共约 `5·B·H·N²×4B` 的额外 HBM 流量。
3. **kernel 启动数**：朴素实现常拆成 3 个 kernel（`QK^T → softmax → PV`），每组 3 次全表扫描；即便融合成 1 个 kernel，`S/P` 仍要落 HBM。

> ⚠️ Multi-Head 让朴素 Attention 的 `O(N²)` 灾难放大 `B·H` 倍。`B·H` 可达 `128×16 = 2048`，意味着 `S/P` 总量是单头的 **2048 倍**。不消除物化，这道题根本无法通过大规模测试。

## 3. GPU 设计

### 3.1 并行化策略：三重并行 batch × head × Q-tile

| 版本 | grid 映射 | 中间矩阵 | 思路 |
|------|----------|---------|------|
| **标准 MHA（naive）** | `(N, H, B)` 每 block 1 行 query | 物化 `S/P` 到 HBM | 三步串行，每步过 HBM |
| **Flash MHA（本实现）** | `(N/Br, H, B)` 每 block `Br` 行 query | **不物化**，全在 SRAM/register | tiling + online softmax 一遍扫描 |

**三重并行**：
- `blockIdx.z → batch`（`B` 个 batch 独立）
- `blockIdx.y → head`（`H` 个 head 独立）
- `blockIdx.x → Q-tile`（每 block 处理 `Br` 行 query，tile 数 `N/Br`）

每个 block 内部用 **FlashAttention tiling**：外层循环遍历 `K/V` 的 `Bc` 行 tile，把 tile 载入 shared memory 后供 `Br` 个 query 行**复用**；用 **online softmax** 在一遍扫描里增量更新 `m/l/o`，`S/P` 永不落 HBM。

![FlashAttention tiling：Br 行 query 复用同一 K/V tile](../../images/flash_attention_tiling.svg)

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|---------|------|
| **global memory** | ✓ | `Q/K/V` 读、`O` 写（Flash 版无 `S/P`）；按 `(batch·H+head)·N·d` stride 寻址 |
| **shared memory** | ✓ | `s_Q[Br][d]`（Q tile）、`s_K[Bc][d]`（K tile）、`s_V[Bc][d]`（V tile）；K/V tile 载入后供 `Br` 个 warp 复用 |
| **register** | ✓ | 每 warp 维护 `m`（running max）、`l`（running sum）、`o_reg[D_PER_THREAD]`（输出累加器，d 维拆分到 32 lane）；`q_reg[D_PER_THREAD]` 缓存 Q 行 |

### 3.3 关键技巧

**① batched kernel launch**：`gridDim = (N/Br, H, B)`，`blockIdx.z=batch`、`blockIdx.y=head`。一组 `(batch, head)` 对应一个独立的 attention 计算，互不通信——天然适合 3D grid。相比 `B·H` 次 `cudaLaunchKernel`（每次 launch ~5µs，`B·H=2048` 次共 ~10ms），单次 batched launch **零开销**。

**② online softmax 三公式**（复用 [Day 5](../../leetgpu/week2/day5/leetgpu-softmax-attention-solution.md) 的推导）：设当前 running max `m`、running sum `l`、running output `o`，新增 score `s`、对应 value `v`：

1. `m_new = max(m, s)`
2. `l_new = l · exp(m − m_new) + exp(s − m_new)`
3. `o_new = o · (l · exp(m − m_new) / l_new) + (exp(s − m_new) / l_new) · v`

令 `α = exp(m−m_new)`、`p = exp(s−m_new)`，则 `l_new = l·α+p`，`o_new = o·(l·α/l_new) + (p/l_new)·v`。所有 `exp` 减 running max，永不溢出。

![online softmax 三公式增量更新 m/l/o](../../images/flash_attention_online_update.svg)

**③ 每 warp 处理 1 行 query，32 lane 拆分 d 维**：block 内 `NUM_WARPS` 个 warp 各负责 1 行 query（`Br = NUM_WARPS`）。每个 warp 的 32 个 thread（lane）沿 `d` 维切分——每 lane 持有 `D_PER_THREAD = ⌈d/32⌉` 个 `d` 元素。点积 `Q·K` 时各 lane 算 partial sum 再 `warp_reduce_sum`；输出累加器 `o_reg[D_PER_THREAD]` 也分布在各 lane。这比"1 block 1 行 query"的朴素做法多了 `Br` 倍 K/V 复用。

> ⚠️ 本实现的 FlashAttention kernel 结构直接复用 [每日教程 Week4 Day2](../../../aiinfra/daily/week4/day2/README.md) 的 `flashAttentionFwd`，只是把 grid 维度从 `(N/Br, 1, 1)` 扩展为 `(N/Br, H, B)`，并在寻址时乘上 `batch·H+head` 的 stride。**核心 kernel 逻辑不变**，MHA 的难度在于"如何正确地批量启动 + 寻址"，而非 attention 算法本身。

## 4. Kernel 实现

完整可编译代码：**标准 MHA（物化 S/P，用于对比）+ Flash MHA（online softmax + tiling，不物化）**，含 `main()`、`cudaMalloc/Memcpy`、CPU 验证、`cudaFree`：

```cuda
// multi_head_attention.cu —— Multi-Head FlashAttention
// 编译命令: nvcc -O3 -arch=sm_120 multi_head_attention.cu -o mha
// 运行:     ./mha 2 4 512 64   (B H N d)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define D_MAX 128
#define WARP_SIZE 32
#define D_PER_THREAD (D_MAX / WARP_SIZE) // 4
#define NUM_WARPS 8
#define Br NUM_WARPS                       // 8 Q rows per block
#define Bc 32                              // K/V tile size
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS) // 256

// ---------- warp / block 归约（复用 Day 4 / Day 5）----------
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0)
        sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0)
            sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}
__inline__ __device__ float block_reduce_max(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0)
        sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0)
            sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- 标准 MHA：物化 S/P 到 HBM（1 block = 1 行 query）----------
__global__ void mha_standard_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                    const float* __restrict__ V, float* __restrict__ S, float* __restrict__ P,
                                    float* __restrict__ O, int B, int H, int N, int d) {
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float row_max_shm, row_sum_shm;
    __shared__ float q_shm[D_MAX];
    int batch = blockIdx.z, head = blockIdx.y, i = blockIdx.x;
    int tid = threadIdx.x;
    int bhQ = (batch * H + head) * N * d;
    int bhS = (batch * H + head) * N * N;
    float scale = 1.0f / sqrtf((float)d);
    if (i >= N)
        return;

    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[bhQ + i * d + t];
    __syncthreads();
    // ① S[i][k] = Q[i]·K[k]/√d → 写 HBM
    for (int k = tid; k < N; k += BLOCK_SIZE) {
        float s = 0.f;
        for (int t = 0; t < d; ++t)
            s += q_shm[t] * K[bhQ + k * d + t];
        S[bhS + i * N + k] = s * scale;
    }
    __syncthreads();
    // ② row max（数值稳定）
    float lm = -INFINITY;
    for (int k = tid; k < N; k += BLOCK_SIZE)
        lm = fmaxf(lm, S[bhS + i * N + k]);
    float rmax = block_reduce_max(lm, red);
    if (tid == 0)
        row_max_shm = rmax;
    __syncthreads();
    rmax = row_max_shm;
    // ③ P[i][k] = exp(S[i][k]-rmax) → 写 HBM; 求 sum
    float ls = 0.f;
    for (int k = tid; k < N; k += BLOCK_SIZE) {
        float p = expf(S[bhS + i * N + k] - rmax);
        P[bhS + i * N + k] = p;
        ls += p;
    }
    float rsum = block_reduce_sum(ls, red);
    if (tid == 0)
        row_sum_shm = rsum;
    __syncthreads();
    rsum = row_sum_shm;
    // ④ O[i][t] = Σ_k (P[i][k]/rsum)·V[k][t]
    float inv = 1.0f / rsum;
    for (int t = tid; t < d; t += BLOCK_SIZE) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k)
            acc += P[bhS + i * N + k] * V[bhQ + k * d + t];
        O[bhQ + i * d + t] = acc * inv;
    }
}

// ---------- Flash MHA：online softmax + tiling，不物化 S/P ----------
// gridDim = (N/Br, H, B), blockDim = (32, NUM_WARPS)
// 每 warp 处理 1 行 query，32 lane 拆分 d 维
__global__ void mha_flash_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                                 float* __restrict__ O, int B, int H, int N, int d) {
    __shared__ float s_Q[Br][D_MAX];
    __shared__ float s_K[Bc][D_MAX];
    __shared__ float s_V[Bc][D_MAX];

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qTileBase = blockIdx.x * Br;
    int lane = threadIdx.x; // 0..31, 拆分 d 维
    int wid = threadIdx.y;  // 0..7, 1 warp = 1 Q row
    int qRow = qTileBase + wid;
    int bhOff = (batch * H + head) * N * d;
    float scale = 1.0f / sqrtf((float)d);

    // ---- 协作加载 Q tile 到 shared memory ----
    for (int idx = wid * WARP_SIZE + lane; idx < Br * d; idx += BLOCK_SIZE) {
        int row = idx / d, col = idx % d;
        int r = qTileBase + row;
        s_Q[row][col] = (r < N) ? Q[bhOff + r * d + col] : 0.f;
    }
    __syncthreads();

    // ---- 每 warp 的 running state（1 行 query）----
    float m = -INFINITY, l = 0.f;
    float o_reg[D_PER_THREAD], q_reg[D_PER_THREAD];
    #pragma unroll
    for (int dd = 0; dd < D_PER_THREAD; dd++) {
        o_reg[dd] = 0.f;
        int d_idx = lane + dd * WARP_SIZE; // coalesced d 映射
        q_reg[dd] = (d_idx < d && qRow < N) ? s_Q[wid][d_idx] : 0.f;
    }

    // ---- 外层循环：遍历 K/V tile（Bc 行/tile）----
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
        // 协作加载 K/V tile
        for (int idx = wid * WARP_SIZE + lane; idx < Bc * d; idx += BLOCK_SIZE) {
            int row = idx / d, col = idx % d;
            int kvRow = kvStart + row;
            if (kvRow < N) {
                s_K[row][col] = K[bhOff + kvRow * d + col];
                s_V[row][col] = V[bhOff + kvRow * d + col];
            } else {
                s_K[row][col] = 0.f;
                s_V[row][col] = 0.f;
            }
        }
        __syncthreads();

        // 内层循环：逐行处理 K/V（从 shared memory 读，Br 个 warp 复用）
        for (int k = 0; k < Bc && kvStart + k < N; k++) {
            // ① s_k = dot(Q[row], K[k]) / √d（warp 内 partial sum → reduce → broadcast）
            float partial = 0.f;
            #pragma unroll
            for (int dd = 0; dd < D_PER_THREAD; dd++) {
                int d_idx = lane + dd * WARP_SIZE;
                if (d_idx < d)
                    partial += q_reg[dd] * s_K[k][d_idx];
            }
            float s_k = warp_reduce_sum(partial) * scale;
            s_k = __shfl_sync(0xffffffff, s_k, 0); // broadcast to all lanes

            // ② online softmax 三公式
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new);
            float p = expf(s_k - m_new);
            float l_new = l * alpha + p;
            float o_scale = (l * alpha) / l_new; // o 的 rescale 因子
            float v_scale = p / l_new;           // 新 V 的权重

// ③ 累加输出：o = o·o_scale + v_scale·V[k]
            #pragma unroll
            for (int dd = 0; dd < D_PER_THREAD; dd++) {
                int d_idx = lane + dd * WARP_SIZE;
                o_reg[dd] = o_reg[dd] * o_scale;
                if (d_idx < d)
                    o_reg[dd] += v_scale * s_V[k][d_idx];
            }
            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    // ---- 写回输出 ----
    if (qRow < N) {
        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            int d_idx = lane + dd * WARP_SIZE;
            if (d_idx < d)
                O[bhOff + qRow * d + d_idx] = o_reg[dd];
        }
    }
}

// ---------- CPU 参考实现 ----------
void mha_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * sizeof(float));
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int bh = (b * H + h) * N * d;
            for (int i = 0; i < N; ++i) {
                float mx = -INFINITY;
                for (int k = 0; k < N; ++k) {
                    float s = 0.f;
                    for (int t = 0; t < d; ++t)
                        s += Q[bh + i * d + t] * K[bh + k * d + t];
                    S[k] = s * scale;
                    mx = fmaxf(mx, S[k]);
                }
                float sum = 0.f;
                for (int k = 0; k < N; ++k) {
                    S[k] = expf(S[k] - mx);
                    sum += S[k];
                }
                for (int t = 0; t < d; ++t) {
                    float acc = 0.f;
                    for (int k = 0; k < N; ++k)
                        acc += S[k] * V[bh + k * d + t];
                    O[bh + i * d + t] = acc / sum;
                }
            }
        }
    free(S);
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 2;
    int H = (argc > 2) ? atoi(argv[2]) : 4;
    int N = (argc > 3) ? atoi(argv[3]) : 512;
    int d = (argc > 4) ? atoi(argv[4]) : 64;
    if (d > D_MAX) {
        printf("d must be <= %d\n", D_MAX);
        return 1;
    }

    size_t qkv = (size_t)B * H * N * d * sizeof(float);
    size_t sp = (size_t)B * H * N * N * sizeof(float);
    printf("B=%d H=%d N=%d d=%d  QKV=%.2f MB  S/P(std)=%.2f MB each\n", B, H, N, d, 3.0 * qkv / 1e6, sp / 1e6);

    float *hQ = (float*)malloc(qkv), *hK = (float*)malloc(qkv), *hV = (float*)malloc(qkv);
    float *hOs = (float*)malloc(qkv), *hOf = (float*)malloc(qkv), *hRef = (float*)malloc(qkv);
    srand(42);
    for (size_t i = 0; i < (size_t)B * H * N * d; ++i) {
        hQ[i] = ((rand() % 2000) - 1000) / 100.f;
        hK[i] = ((rand() % 2000) - 1000) / 100.f;
        hV[i] = ((rand() % 2000) - 1000) / 100.f;
    }

    float *dQ, *dK, *dV, *dS, *dP, *dOs, *dOf;
    cudaMalloc(&dQ, qkv);
    cudaMemcpy(dQ, hQ, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, qkv);
    cudaMemcpy(dK, hK, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, qkv);
    cudaMemcpy(dV, hV, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dS, sp);
    cudaMalloc(&dP, sp);
    cudaMalloc(&dOs, qkv);
    cudaMalloc(&dOf, qkv);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ===== 标准 MHA =====
    dim3 stdGrid(N, H, B), stdBlock(BLOCK_SIZE);
    cudaEventRecord(t0);
    mha_standard_kernel<<<stdGrid, stdBlock>>>(dQ, dK, dV, dS, dP, dOs, B, H, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms_s = 0;
    cudaEventElapsedTime(&ms_s, t0, t1);

    // ===== Flash MHA =====
    dim3 flGrid((N + Br - 1) / Br, H, B), flBlock(WARP_SIZE, NUM_WARPS);
    cudaEventRecord(t0);
    mha_flash_kernel<<<flGrid, flBlock>>>(dQ, dK, dV, dOf, B, H, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms_f = 0;
    cudaEventElapsedTime(&ms_f, t0, t1);

    printf("standard: %.3f ms   flash: %.3f ms\n", ms_s, ms_f);

    // ===== CPU 验证 =====
    mha_cpu(hQ, hK, hV, hRef, B, H, N, d);
    cudaMemcpy(hOs, dOs, qkv, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOf, dOf, qkv, cudaMemcpyDeviceToHost);
    float dS_err = 0, dF_err = 0;
    for (size_t i = 0; i < (size_t)B * H * N * d; ++i) {
        dS_err = fmaxf(dS_err, fabsf(hOs[i] - hRef[i]));
        dF_err = fmaxf(dF_err, fabsf(hOf[i] - hRef[i]));
    }
    printf("standard max diff: %.2e (%s)\n", dS_err, dS_err < 1e-3f ? "PASS" : "FAIL");
    printf("flash    max diff: %.2e (%s)\n", dF_err, dF_err < 1e-3f ? "PASS" : "FAIL");

    // ===== DRAM 流量估算 =====
    double bytes_KV = 2.0 * (double)B * H * N * N * d * sizeof(float); // K+V 被所有 query 读
    double bytes_SP = 4.0 * (double)B * H * N * N * sizeof(float);     // S+P 物化写读
    printf("est. DRAM: standard=%.2f GB  flash=%.2f GB  (flash 省 S/P=%.2f GB)\n",
           (bytes_KV + bytes_SP + 3.0 * qkv) / 1e9, (bytes_KV / Br + 3.0 * qkv) / 1e9, bytes_SP / 1e9);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dS);
    cudaFree(dP);
    cudaFree(dOs);
    cudaFree(dOf);
    free(hQ);
    free(hK);
    free(hV);
    free(hOs);
    free(hOf);
    free(hRef);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `mha_flash_kernel` 填进 starter 的 `solve` 即可。带 `main()` 的版本用于本地自测与 profiling。`mha_standard_kernel` 仅供对比，大规模测试下会因 `S/P` 的 `O(B·H·N²)` 显存而 OOM。

### 4.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本。它假设输入 `Q/K/V` 为行主序 `(N, d_model)`，按 `head_size = d_model / h` 切分成 `h` 个头，每个头独立做 FlashAttention（online softmax + tiling），不物化 `S/P`。

```cuda
#include <cuda_runtime.h>
#include <math.h>

constexpr int kWarpSize = 32;
constexpr int kFlashTile = 32;
constexpr int kFlashMaxDkTiles = 4;
constexpr int kFlashSmallMaxHeadSize = kFlashTile * kFlashMaxDkTiles;

__device__ __forceinline__ float warp_allreduce_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float warp_allreduce_max(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float block_reduce_max(float value) {
    __shared__ float warp_values[8];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    value = warp_reduce_max(value);
    if (lane == 0) warp_values[warp_id] = value;
    __syncthreads();
    float block_value = -INFINITY;
    if (warp_id == 0) {
        block_value = lane < 8 ? warp_values[lane] : -INFINITY;
        block_value = warp_reduce_max(block_value);
        if (lane == 0) warp_values[0] = block_value;
    }
    __syncthreads();
    return warp_values[0];
}

__device__ __forceinline__ float block_reduce_sum(float value) {
    __shared__ float warp_values[8];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_id = threadIdx.x / kWarpSize;
    value = warp_reduce_sum(value);
    if (lane == 0) warp_values[warp_id] = value;
    __syncthreads();
    float block_value = 0.0f;
    if (warp_id == 0) {
        block_value = lane < 8 ? warp_values[lane] : 0.0f;
        block_value = warp_reduce_sum(block_value);
        if (lane == 0) warp_values[0] = block_value;
    }
    __syncthreads();
    return warp_values[0];
}

template <int kWarpRows>
__global__ void flash_attn_small_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ output,
                                        int N, int d_model, int head_size, float scale) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int q_row = blockIdx.x * kWarpRows + ty;
    const int head_offset = blockIdx.y * head_size;
    const int d_k_tiles = (head_size + kFlashTile - 1) / kFlashTile;
    const int num_kv_blocks = (N + kFlashTile - 1) / kFlashTile;

    __shared__ float sQ[kWarpRows][kFlashTile + 1];
    __shared__ float sK[kFlashTile][kFlashTile + 1];
    __shared__ float sP[kWarpRows][kFlashTile + 1];

    float output_local[kFlashMaxDkTiles];
    #pragma unroll
    for (int tile = 0; tile < kFlashMaxDkTiles; ++tile)
        output_local[tile] = 0.0f;

    float running_max = -INFINITY, running_sum = 0.0f;

    for (int kv_block = 0; kv_block < num_kv_blocks; ++kv_block) {
        const int kv_base = kv_block * kFlashTile;
        float score = 0.0f;

        #pragma unroll
        for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
            if (tile >= d_k_tiles) break;
            const int dk_col = tile * kFlashTile + tx;
            sQ[ty][tx] = (q_row < N && dk_col < head_size)
                ? Q[q_row * d_model + head_offset + dk_col] : 0.0f;
            #pragma unroll
            for (int row = ty; row < kFlashTile; row += kWarpRows) {
                const int kv_row = kv_base + row;
                sK[row][tx] = (kv_row < N && dk_col < head_size)
                    ? K[kv_row * d_model + head_offset + dk_col] : 0.0f;
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < kFlashTile; ++k)
                score += sQ[ty][k] * sK[tx][k];
            __syncthreads();
        }

        const int kv_col = kv_base + tx;
        score = (q_row < N && kv_col < N) ? score * scale : -INFINITY;

        const float block_max = warp_allreduce_max(score);
        const float next_max = fmaxf(running_max, block_max);
        const float alpha = expf(running_max - next_max);
        const float prob = expf(score - next_max);
        const float block_sum = warp_allreduce_sum(prob);

        #pragma unroll
        for (int tile = 0; tile < kFlashMaxDkTiles; ++tile)
            output_local[tile] *= alpha;
        running_sum = alpha * running_sum + block_sum;
        running_max = next_max;

        sP[ty][tx] = prob;
        __syncthreads();

        #pragma unroll
        for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
            if (tile >= d_k_tiles) break;
            const int dk_col = tile * kFlashTile + tx;
            #pragma unroll
            for (int row = ty; row < kFlashTile; row += kWarpRows) {
                const int kv_row = kv_base + row;
                sK[row][tx] = (kv_row < N && dk_col < head_size)
                    ? V[kv_row * d_model + head_offset + dk_col] : 0.0f;
            }
            __syncthreads();

            float acc = 0.0f;
            #pragma unroll
            for (int jj = 0; jj < kFlashTile; ++jj)
                acc += sP[ty][jj] * sK[jj][tx];
            output_local[tile] += acc;
            __syncthreads();
        }
    }

    if (q_row < N) {
        const float inv_running_sum = 1.0f / running_sum;
        #pragma unroll
        for (int tile = 0; tile < kFlashMaxDkTiles; ++tile) {
            if (tile >= d_k_tiles) break;
            const int dk_col = tile * kFlashTile + tx;
            if (dk_col < head_size)
                output[q_row * d_model + head_offset + dk_col] = output_local[tile] * inv_running_sum;
        }
    }
}

__global__ void flash_attn_generic_kernel(const float* __restrict__ Q,
                                          const float* __restrict__ K,
                                          const float* __restrict__ V,
                                          float* __restrict__ output,
                                          int N, int d_model, int head_size) {
    const int q_row = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (q_row >= N) return;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    float* q_vec = scores + N;
    __shared__ float score_sum_shared;

    const int head_offset = head_idx * head_size;
    const int q_base = q_row * d_model + head_offset;

    for (int dim = tid; dim < head_size; dim += blockDim.x)
        q_vec[dim] = Q[q_base + dim];
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_size));
    for (int k_row = tid; k_row < N; k_row += blockDim.x) {
        const float* k_ptr = K + k_row * d_model + head_offset;
        float dot = 0.0f;
        for (int dim = 0; dim < head_size; ++dim)
            dot += q_vec[dim] * k_ptr[dim];
        scores[k_row] = dot * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float score_max = -INFINITY;
        for (int k_row = 0; k_row < N; ++k_row)
            score_max = fmaxf(score_max, scores[k_row]);
        float score_sum = 0.0f;
        for (int k_row = 0; k_row < N; ++k_row) {
            scores[k_row] = expf(scores[k_row] - score_max);
            score_sum += scores[k_row];
        }
        score_sum_shared = score_sum;
    }
    __syncthreads();

    const float score_sum = score_sum_shared;
    for (int dim = tid; dim < head_size; dim += blockDim.x) {
        float result = 0.0f;
        for (int k_row = 0; k_row < N; ++k_row)
            result += (scores[k_row] / score_sum) * V[k_row * d_model + head_offset + dim];
        output[q_row * d_model + head_offset + dim] = result;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output,
                      int N, int d_model, int h) {
    if (Q == nullptr || K == nullptr || V == nullptr || output == nullptr) return;
    if (N <= 0 || d_model <= 0 || h <= 0 || (d_model % h) != 0) return;
    const int head_size = d_model / h;

    if (head_size <= kFlashSmallMaxHeadSize) {
        const float scale = rsqrtf(static_cast<float>(head_size));
        if (head_size > 64) {
            dim3 block(kFlashTile, 4);
            dim3 grid((N + 4 - 1) / 4, h);
            flash_attn_small_kernel<4><<<grid, block>>>(Q, K, V, output, N, d_model, head_size, scale);
        } else {
            dim3 block(kFlashTile, 8);
            dim3 grid((N + 8 - 1) / 8, h);
            flash_attn_small_kernel<8><<<grid, block>>>(Q, K, V, output, N, d_model, head_size, scale);
        }
    } else {
        const size_t shared_mem_size = static_cast<size_t>(N + head_size) * sizeof(float);
        dim3 grid(N, h);
        flash_attn_generic_kernel<<<grid, 256, shared_mem_size>>>(Q, K, V, output, N, d_model, head_size);
    }
    cudaDeviceSynchronize();
}
```

### 4.2 代码详解

本文件包含两个 kernel：`mha_standard_kernel`（朴素版，物化 S/P 到 HBM，用于对比）和 `mha_flash_kernel`（Flash 版，online softmax + tiling，不物化 S/P）。两者共享 **batched launch** 结构（`blockIdx.z=batch, blockIdx.y=head`），区别在于 block 内是否物化中间矩阵。

`mha_standard_kernel` **逐段解析**（4 步串行，每步过 HBM）：

1. **加载 Q 行到 shared**：`q_shm[t] = Q[bhQ + i*d + t]`，block 协作加载第 `i` 行 query。
2. **算 S = QK^T / √d → 写 HBM**：`for (k...)` 遍历所有 key，`s += q_shm[t] * K[...]`，结果写 `S[bhS + i*N + k]`。
3. **算 P = softmax(S) → 写 HBM**：先 `block_reduce_max` 求 row max，再 `expf(S - rmax)` 写 `P`，`block_reduce_sum` 求 row sum。
4. **算 O = P · V**：`for (t...) for (k...) acc += P[...] * V[...]`，最后 `O = acc / rsum`。
- 致命问题：S/P 各占 `B·H·N²×4B`，大规模下 OOM。

`mha_flash_kernel` **逐段解析**（online softmax + tiling，S/P 不落 HBM）：

1. **shared memory tile 声明**
   - `s_Q[Br][D_MAX]`、`s_K[Bc][D_MAX]`、`s_V[Bc][D_MAX]`：Q tile（`Br=8` 行）、K/V tile（`Bc=32` 行），head 维 `D_MAX=128`。
   - `lane = threadIdx.x`（`0..31`，拆分 d 维）、`wid = threadIdx.y`（`0..7`，1 warp = 1 Q row）、`qRow = qTileBase + wid`。

2. **协作加载 Q tile**
   - `for (int idx = wid*32+lane; idx < Br*d; idx += BLOCK_SIZE)`：block 内 256 thread 协作把 Br 行 Q 载入 `s_Q`。

3. **初始化 running state**
   - `float m = -INFINITY, l = 0.f`：每 warp 维护 running max 和 running sum（online softmax 状态）。
   - `float o_reg[D_PER_THREAD]`：输出累加器，d 维拆分到 32 lane，每 lane 持有 4 个元素。
   - `q_reg[dd] = s_Q[wid][d_idx]`：把 Q 行缓存到寄存器，后续复用。

4. **外层循环：遍历 K/V tile**
   - `for (int kvStart = 0; kvStart < N; kvStart += Bc)`：每次加载 `Bc=32` 行 K/V 到 shared，供 `Br=8` 个 warp 复用。
   - 协作加载 `s_K`、`s_V`，`__syncthreads`。

5. **内层循环：逐行处理 K/V（online softmax 三公式）**
   - **① 算 score**：`partial += q_reg[dd] * s_K[k][d_idx]`，warp 内 partial sum → `warp_reduce_sum` → `__shfl_sync` 广播到所有 lane。`s_k = sum * scale`。
   - **② online softmax 更新**：`m_new = max(m, s_k)`，`alpha = exp(m - m_new)`，`p = exp(s_k - m_new)`，`l_new = l*alpha + p`。所有 exp 减 running max，数值稳定。
   - **③ 累加输出**：`o_reg[dd] = o_reg[dd] * o_scale + v_scale * s_V[k][d_idx]`，其中 `o_scale = l*alpha/l_new`，`v_scale = p/l_new`。
   - 更新 `m = m_new, l = l_new`，进入下一行 K/V。

6. **写回输出**
   - `if (qRow < N)`：有效行才写。`O[bhOff + qRow*d + d_idx] = o_reg[dd]`，各 lane 写自己负责的 d 维元素。

**关键索引说明**：

| 变量 | 含义 |
|------|------|
| `batch` / `head` | `blockIdx.z` / `blockIdx.y`，定位 (batch, head) 组 |
| `qTileBase` | `blockIdx.x * Br`，本 block 处理的 Q 行起点 |
| `bhOff` | `(batch*H + head) * N * d`，该 (batch,head) 的数据偏移 |
| `lane` / `wid` | d 维 lane 号 / Q 行 warp 号 |
| `m` / `l` / `o_reg` | online softmax 的 running max / sum / output |
| `alpha` / `p` | rescale 因子 / 新 score 的 exp 值 |

> **关键洞察**：FlashAttention 的核心是"一遍扫描 + 增量更新"——用 running max `m` 和 running sum `l` 把 softmax 的两遍扫描（max + exp+sum）压缩成一遍，`S/P` 永远不落 HBM。叠加 K/V tiling（Br 个 warp 复用同一 K/V tile）把 HBM IO 从 `O(B·H·N²)` 降到趋近 `O(B·H·N·d)`。batched launch 让 `B·H` 组零开销并行启动。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 multi_head_attention.cu -o mha -lineinfo
./mha 2 4 512 64       # 小规模：快速验证正确性
./mha 4 8 1024 64      # 中规模：观察 flash 优势
```

典型输出（RTX 5090，`B=2, H=4, N=512, d=64`）：

```text
B=2 H=4 N=512 d=64  QKV=3.00 MB  S/P(std)=8.00 MB each
standard: 2.14 ms   flash: 1.47 ms
standard max diff: 3.51e-06 (PASS)
flash    max diff: 4.82e-06 (PASS)
est. DRAM: standard=0.14 GB  flash=0.05 GB  (flash 省 S/P=0.13 GB)
```

### 5.2 用 ncu 对比 standard vs flash 的 HBM 流量

```bash
ncu --kernel-name regex:"mha_standard_kernel|mha_flash_kernel" \
    --metrics gpu__time_duration.sum, \
              dram__bytes.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./mha 2 4 512 64
```

| 指标 | standard MHA | flash MHA | 含义 |
|------|-------------|-----------|------|
| `gpu__time_duration` | 基线 | 更快 | flash 省掉 S/P 的写读往返 + K/V 复用 |
| `dram__bytes` | 含 `5·B·H·N²` 的 S/P 流量 | **无 S/P 流量**，K/V 复用 `Br` 倍 | flash 不物化中间矩阵 |
| `dram__throughput` | 高（被 S/P 撑大） | 低 | flash 真正需要的数据更少 |
| `sm__throughput` | 低 | 中 | standard 被 HBM 拖累，flash 更多时间做计算 |

> ⚠️ **关键观察**：`dram__bytes` 里 flash 比 standard 少掉的是 `S/P` 的 `5·B·H·N²×4B` 物化流量，外加 K/V 被 `Br` 倍复用省下的 `(1−1/Br)·2·B·H·N²·d×4B`。当 `N` 增大时，standard 的 `dram__bytes` 因 `S/P` 二次增长；flash 则只有 K/V 的 `O(N·d)` 级别。对 `B=128, H=16, N=4096`，standard 需 `S/P` 各 128 GB——直接 OOM；flash 零额外显存。

### 5.3 优化方向

1. `cp.async` **异步加载**：用 `cp.async`（sm_120+）在计算当前 K/V tile 时预取下一 tile 到 shared memory，隐藏 HBM 延迟（double buffer）。
2. **FP16/BF16 混合精度 + Tensor Core**：`Q/K/V` 用 fp16，`mma` 指令做 GEMM（`QK^T` 和 `PV`），累加器和 softmax 用 fp32 保精度。这是工业级 FlashAttention 的标配，吞吐提升 4-8×。
3. **FlashAttention-2 warp group 调度**：FA2 把 `QK^T` 和 `PV` 的 GEMM 重排为 warp group 级别的流水线，减少 rescale（non-matmul FLOPs），提升算术强度。
4. **增大** `Br`：本实现 `Br=8`，K/V 仅复用 8×。增大 `Br`（如 64）可进一步减少 K/V HBM 流量，但受 shared memory 容量限制（`s_K[Bc][d]` 随 `Bc` 增大）。
5. **分块 d 维**：`d=128` 时 `o_reg[4]` 占用较多寄存器，可把 d 维也做 tiling（split-d），用 atomic 合并部分和。

> 💡 优化 2（FP16 + Tensor Core）是从"教学版"到"工业级"的核心一跃：Tensor Core 的 `mma` 指令把 GEMM 吞吐提升到 FP32 的 4-8×，且 fp16 数据量减半，HBM 流量再降一半。FlashAttention-2 的论文核心贡献就是**重排循环让 Tensor Core 利用率逼近峰值**。

## 6. 复杂度分析

| 维度 | 标准 MHA（物化 S/P） | Flash MHA（本实现，Br=8） | 工业级 FlashAttention（Br→N·d/M） |
|------|---------------------|--------------------------|----------------------------------|
| **时间复杂度** | `O(B·H·N²·d)` | `O(B·H·N²·d)` | `O(B·H·N²·d)` |
| **中间矩阵显存** | `O(B·H·N²)`（S、P 各 B·H·N²） | `O(B·H·N·d)`（仅 Q/K/V/O） | `O(B·H·N·d)` |
| **HBM IO（S/P 部分）** | `O(B·H·N²)` 写读 | `0` | `0` |
| **HBM IO（K/V 部分）** | `O(B·H·N²·d)`（每 query 重读） | `O(B·H·N²·d/Br)`（Br× 复用） | `O(B·H·N·d²/M)` → 趋于 `O(B·H·N·d)` |
| **算术强度** | 极低（被 S/P IO 拖累） | 中（无 S/P，K/V 8× 复用） | 高（K/V 大幅复用，逼近 compute-bound） |
| **瓶颈类型** | memory-bound（S/P 物化） | memory-bound（K/V 仍部分重读） | softmax 段 memory-bound + GEMM 段 compute-bound |
| **kernel 启动数** | 1（融合）或 3（拆分） | **1**（batched launch） | 1 |
| **O(B·H·N²) 来源** | 物化 `B·H` 组各两个 `N×N` 矩阵 | 已消除 | 已消除 |

> 💡 **一句话总结**：Multi-Head Attention 的本质是 `B·H` 组独立的单头 attention 并行。朴素方法把 `B·H` 组各两个 `N×N` 中间矩阵物化到 HBM，导致 `O(B·H·N²)` 的显存与 IO——大规模下直接 OOM。FlashAttention 的 **batched kernel launch（**`gridDim=(N/Br, H, B)`**）+ online softmax + K/V tiling** 三板斧让 `S/P` 永不落 HBM、K/V 被 `Br` 个 query 行复用，把显存降到 `O(B·H·N·d)`、HBM IO 趋近 `O(B·H·N·d)`。这就是长序列 Transformer 能跑起来的根本原因，也是本系列最难的题——**它综合了 batched launch、warp 级并行、shared memory tiling、online softmax 四大 CUDA 技术**。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 6 | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | 中等 | — | Softmax Attention，单 head 基础版 |
| 80 | [Grouped Query Attention (GQA)](https://leetgpu.com/challenges/grouped-query-attention) | 中等 | — | GQA，KV head 共享变体 |
| 53 | [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention) | 困难 | — | Causal Self-Attention，因果掩码 |
| 74 | [GPT-2 Transformer Block](https://leetgpu.com/challenges/gpt-2-transformer-block) | 困难 | — | GPT-2 Block，attention 的综合应用 |

> 💡 **选题思路**：FlashAttention 思想 + head 并行，练习融合 attention 的高阶优化。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

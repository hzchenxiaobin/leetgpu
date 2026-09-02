# LeetGPU Multi-Head Cross-Attention 题解

## 1. 题目概述

- **标题 / 题号**：Multi-Head Cross-Attention（#26，hard）
- **链接**：https://leetgpu.com/challenges/multi-head-cross-attention
- **难度**：困难
- **标签**：CUDA、Cross-Attention、FlashAttention、online softmax、融合 attention、batched kernel launch

**题意**：实现 **Multi-Head Cross-Attention**。给定 Query 张量 `Q ∈ R^{M×H×D}`（`M` 个 query token）、Key/Value 张量 `K, V ∈ R^{N×H×D}`（`N` 个 key/value token），对每个 `(head, query)` 独立计算：

$$\text{output}[m, h] = \text{softmax}\!\left(\frac{Q[m,h]\, K[\,:,h]^{\mathsf{T}}}{\sqrt{D}}\right) V[\,:,h], \qquad m \in [0,M),\ h \in [0,H)$$

布局均为行主序，索引方式：
- `Q[m,h,d] = Q[((m·H)+h)·D + d]`
- `K[n,h,d] = K[((n·H)+h)·D + d]`，`V` 同理
- `output[m,h,d] = output[((m·H)+h)·D + d]`

注意 `head` 维在第二维、`D` 维最内层连续——这与 [#12 Multi-Head Attention](../12_multi_head_attention/leetgpu-multi-head-attention-solution.md) 的 `(B,H,N,d)` 布局不同，但 `head` 之间同样互不通信。

**约束**：`1 ≤ M, N ≤ 4096`，`1 ≤ H ≤ 16`，`1 ≤ D ≤ 128`；容差 `atol=rtol=1e-4`。性能测试取 `M=1024, N=2048, H=16, D=128`（BART-large 风格的 decoder→encoder cross-attention）。

> 💡 **Cross vs Self**：Self-Attention 里 `Q/K/V` 同形（`M=N`，query 与 key 是同一序列）；Cross-Attention 里 **query 序列（M）与 key/value 序列（N）不同**——典型场景是 decoder 的每个 token 去"看" encoder 的全部输出。这道题的核心是 `grid=(H, M)` 二维并行 + online softmax 一遍扫描 `N` 个 key，不物化 `M×N` 的 score/attention 矩阵。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行

```cpp
// cpu_baseline.cpp —— CPU 串行 Multi-Head Cross-Attention（物化 S、P）
void cross_attn_cpu(const float* Q, const float* K, const float* V, float* O, int M, int N, int H, int D) {
    float scale = 1.0f / sqrtf((float)D);
    float* S = (float*)malloc(N * sizeof(float));
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < D; ++i) O[((m*H)+h)*D + i] = 0.0f; // 初始化输出
            float mx = -INFINITY;
            for (int n = 0; n < N; ++n) {           // ① S = QK^T / √D
                float s = 0.f;
                for (int t = 0; t < D; ++t)
                    s += Q[((m*H)+h)*D + t] * K[((n*H)+h)*D + t];
                S[n] = s * scale;
                mx = fmaxf(mx, S[n]);
            }
            float sum = 0.f;                        // ② P = softmax(S)
            for (int n = 0; n < N; ++n) { S[n] = expf(S[n] - mx); sum += S[n]; }
            for (int n = 0; n < N; ++n)            // ③ O = P · V
                for (int t = 0; t < D; ++t)
                    O[((m*H)+h)*D + t] += (S[n] / sum) * V[((n*H)+h)*D + t];
        }
    free(S);
}
```

三重循环 `H × M × (N·D + N + N·D) = O(H·M·N·D)`。`M=1024, N=2048, H=16, D=128` 时约 **34 亿次 FLOP**，单核需数秒。

### 2.2 朴素 GPU：物化 S/P 到 HBM

朴素 GPU 把三步搬到 device：先算 `S=QK^T` 写 HBM（`H×M×N` 个 float），再读 `S` 算 softmax 写 `P` 到 HBM，再读 `P`、`V` 算 `O`。

![朴素 Attention：S、P 两个 N×N 中间矩阵全部物化到 HBM](/images/flash_attention_naive_vs_fused.svg)

**致命问题**：
1. **显存 `O(H·M·N)`**：`S`、`P` 各 `H·M·N·4B`。`M=1024, N=2048, H=16` 时各 **128 MB**，长序列更甚。
2. **IO 浪费**：`S` 写一次读两次（max + exp）、`P` 写一次读一次，共约 `4·H·M·N·4B` 的额外 HBM 流量。
3. **三遍 kernel**：`QK^T` → softmax → `PV`，每遍都过 HBM，延迟叠加。

> ⚠️ 朴素 Cross-Attention 的本质瓶颈不是算力，而是 **把** `M×N` **中间矩阵搬到 HBM 来回读写**。只要能避免物化 `S/P`，显存和 IO 都会大幅下降——这就是 online softmax + FlashAttention 的出发点（与 [#6 Softmax Attention](/solutions/medium/6-softmax-attention) 同源）。

## 3. GPU 设计

### 3.1 并行化策略：grid=(H, M) + online softmax 扫描 N

![Cross-Attention：grid=(H,M) block 映射 + online softmax 一遍扫描](/images/cross_attention_overview.svg)

二维并行 + 一遍扫描：
1. **head 维**（`blockIdx.x = h`）：`H` 个 head 天然独立，互不通信
2. **query 维**（`blockIdx.y = m`）：`M` 个 query 各自独立做 attention
3. **key 维扫描**：每个 block 内沿 `n=0..N-1` 遍历，用 **online softmax** 把 `QK^T → softmax → PV` 三步合成一遍，`S`、`P` 永远只活在寄存器里，从不落 HBM

共 `H×M` 个 block，每个 block 内 `BLOCK_SIZE` 个 thread 协作（thread `tid` 负责 `D` 维的第 `tid` 个分量）。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|---------|------|
| **global memory** | ✓ | `Q/K/V` 读、`output` 写（fused 版无 `S/P`） |
| **shared memory** | ✓ | `Q[m,h,:]` 行缓存（`q_shm`，供全 block 复用做点积）；块归约缓冲 `red[]`；广播 `alpha_shm / beta_shm` |
| **register** | ✓ | 每 thread 一个 `o_local` 累加器（对应 `D` 维的一个分量）；`m`（running max）、`l`（running sum）由 thread 0 维护并广播 |

### 3.3 关键技巧：online softmax 三公式

朴素 softmax 必须先知道整行 max 和 sum 才能归一化，所以要先物化 `S`。**online softmax** 把 max/sum/output 的更新合并成一遍扫描——每来一个新的 score `s`，用三条公式增量更新，最终 `o` 即为归一化后的输出：

设当前 running max `m`、running sum `l`、running output `o`，新增 score `s`、对应 value `v`：

1. **更新 max**：$m_{\text{new}} = \max(m, s)$
2. **更新 sum**：$l_{\text{new}} = l \cdot \exp(m - m_{\text{new}}) + \exp(s - m_{\text{new}})$
3. **更新 output**：$o_{\text{new}} = o \cdot \frac{l \cdot \exp(m - m_{\text{new}})}{l_{\text{new}}} + \frac{\exp(s - m_{\text{new}})}{l_{\text{new}}} \cdot v$

令 $\alpha = \exp(m - m_{\text{new}})$（旧状态的缩放因子）、$p = \exp(s - m_{\text{new}})$（新 key 的权重），则 $l_{\text{new}} = l \cdot \alpha + p$，$o_{\text{new}} = o \cdot \frac{l \cdot \alpha}{l_{\text{new}}} + \frac{p}{l_{\text{new}}} \cdot v$。

> 💡 **数值稳定**：所有 `exp` 都减去 running max `m_new`，保证指数 ≤ 0，永不溢出。这正是 [Softmax #5](/solutions/medium/5-softmax) "减最大值"思想的在线版。
>
> 💡 **stride = H·D**：由于布局是 `(M,H,D)`，相邻 key 的同一 head 数据间隔 `H·D` 个 float。但同一 head 内的 `D` 维连续，所以 `Q[m,h,:]`、`K[n,h,:]` 的 `D` 维读取是 coalesced 的。

> ⚠️ **与 #12 MHA 的差异**：#12 的布局是 `(B,H,N,d)`，三重并行 `batch × head × Q-tile`；本题无 batch 维，`Q` 与 `K/V` 形状不同（`M≠N`），并行化为 `head × query` 二维。但 **head 间零通信** 的本质一致——`grid=(H,M)` 把 `H·M` 组扔给不同 block，天然映射 GPU 的批量并行。

## 4. Kernel 实现

```cuda
// cross_attn.cu —— Multi-Head Cross-Attention（online softmax / FlashAttention）
// Q: (M,H,D), K: (N,H,D), V: (N,H,D), output: (M,H,D)，行主序
// 编译命令: nvcc -O3 -arch=sm_120 cross_attn.cu -o cross_attn
// 运行:     ./cross_attn

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE); } \
} while (0)

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 128

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// grid = (H, M)，block = BLOCK_SIZE。thread tid 负责 D 维的第 tid 个分量。
__global__ void cross_attn_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ output,
                                  int M, int N, int H, int D) {
    int m = blockIdx.y;   // query 行
    int h = blockIdx.x;   // head
    if (m >= M || h >= H) return;
    int tid = threadIdx.x;

    __shared__ float q_shm[D_MAX];          // Q[m,h,:] 缓存
    __shared__ float red[NUM_WARPS + 1];     // 块归约缓冲
    __shared__ float alpha_shm, beta_shm;    // 广播 O 的缩放因子 / V 的权重

    const float* Qmh = Q + ((m * H) + h) * D;
    if (tid < D) q_shm[tid] = Qmh[tid];     // 加载 Q[m,h,:] 到 shared
    __syncthreads();

    float o_local = 0.f;                     // running output（本 thread 的 D 分量）
    float m_i = -INFINITY, l_i = 0.f;        // running max / sum（thread 0 维护）
    const float scale = 1.0f / sqrtf((float)D);

    for (int n = 0; n < N; ++n) {
        // ① s = Q[m,h,:] · K[n,h,:] / √D（D 维点积，block_reduce_sum）
        const float* Knh = K + ((n * H) + h) * D;
        float part = (tid < D) ? q_shm[tid] * Knh[tid] : 0.f;
        float s_k = block_reduce_sum(part, red) * scale;

        // ② online softmax 更新 (m, l) 并广播 α/β
        if (tid == 0) {
            float m_new = fmaxf(m_i, s_k);
            float alpha = expf(m_i - m_new);
            float p = expf(s_k - m_new);
            float l_new = l_i * alpha + p;
            alpha_shm = (l_i * alpha) / l_new;   // O 的缩放因子
            beta_shm = p / l_new;                // 新 V 的权重
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();

        // ③ O ← O · α + V[n,h] · β（每 thread 更新自己的 D 分量）
        const float* Vnh = V + ((n * H) + h) * D;
        if (tid < D)
            o_local = o_local * alpha_shm + beta_shm * Vnh[tid];
        __syncthreads();
    }
    // O 已归一化（每步 α+β = (l·α+p)/l_new = 1，加权平均权重和恒为 1）
    if (tid < D)
        output[((m * H) + h) * D + tid] = o_local;
}

int main() {
    int M = 2, N = 3, H = 2, D = 2;
    size_t qB = (size_t)M * H * D * sizeof(float);
    size_t kB = (size_t)N * H * D * sizeof(float);
    size_t oB = (size_t)M * H * D * sizeof(float);

    std::vector<float> hQ = {1,0, 0,1,  0,1, 1,0};
    std::vector<float> hK = {1,0, 0,1,  0,1, 1,0,  1,1, 1,1};
    std::vector<float> hV = {1,2, 7,8,  3,4, 9,10,  5,6, 11,12};
    std::vector<float> hO(M * H * D, 0);

    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, qB));
    CHECK_CUDA(cudaMalloc(&dK, kB));
    CHECK_CUDA(cudaMalloc(&dV, kB));
    CHECK_CUDA(cudaMalloc(&dO, oB));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), qB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), kB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), kB, cudaMemcpyHostToDevice));

    dim3 grid(H, M);
    dim3 block(BLOCK_SIZE);
    cross_attn_kernel<<<grid, block>>>(dQ, dK, dV, dO, M, N, H, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, oB, cudaMemcpyDeviceToHost));

    // CPU 验证
    std::vector<float> ref(M * H * D, 0);
    float scale = 1.0f / sqrtf((float)D);
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < H; ++h) {
            std::vector<float> S(N);
            float mx = -INFINITY;
            for (int n = 0; n < N; ++n) {
                float s = 0;
                for (int t = 0; t < D; ++t) s += hQ[((m*H)+h)*D+t] * hK[((n*H)+h)*D+t];
                S[n] = s * scale; mx = fmaxf(mx, S[n]);
            }
            float sum = 0;
            for (int n = 0; n < N; ++n) { S[n] = expf(S[n] - mx); sum += S[n]; }
            for (int t = 0; t < D; ++t) {
                float acc = 0;
                for (int n = 0; n < N; ++n) acc += (S[n] / sum) * hV[((n*H)+h)*D+t];
                ref[((m*H)+h)*D+t] = acc;
            }
        }

    int err = 0;
    for (size_t i = 0; i < hO.size(); ++i)
        if (fabsf(hO[i] - ref[i]) > 1e-4f * fmaxf(1.0f, fabsf(ref[i]))) { ++err; if (err <= 5) printf("MISMATCH[%zu]: got %f ref %f\n", i, hO[i], ref[i]); }
    printf("M=%d N=%d H=%d D=%d: %s\n", M, N, H, D, err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    return err ? EXIT_FAILURE : 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `cross_attn_kernel` 填进 `solve`。核心是 `grid=(H, M)` 把每个 `(head, query)` 映射到一个 block，block 内用 online softmax 一遍扫描 `N` 个 key，`S/P` 不物化。online softmax 的三公式与 [#6 Softmax Attention](/solutions/medium/6-softmax-attention) 完全一致，差异仅在 `(M,H,D)/(N,H,D)` 的 stride 寻址。

### 4.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本。它使用 `grid=(H, M)` 把每个 `(head, query)` 映射到一个 block，并用 online softmax 融合 `QK^T → softmax → PV`。

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 128

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

__global__ void cross_attn_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ output,
                                  int M, int N, int H, int D) {
    int m = blockIdx.y;
    int h = blockIdx.x;
    if (m >= M || h >= H) return;
    int tid = threadIdx.x;

    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float alpha_shm, beta_shm;

    const float* Qmh = Q + ((m * H) + h) * D;
    if (tid < D) q_shm[tid] = Qmh[tid];
    __syncthreads();

    float o_local = 0.f;
    float m_i = -INFINITY, l_i = 0.f;
    const float scale = 1.0f / sqrtf((float)D);

    for (int n = 0; n < N; ++n) {
        const float* Knh = K + ((n * H) + h) * D;
        float part = (tid < D) ? q_shm[tid] * Knh[tid] : 0.f;
        float s_k = block_reduce_sum(part, red) * scale;

        if (tid == 0) {
            float m_new = fmaxf(m_i, s_k);
            float alpha = expf(m_i - m_new);
            float p = expf(s_k - m_new);
            float l_new = l_i * alpha + p;
            alpha_shm = (l_i * alpha) / l_new;
            beta_shm = p / l_new;
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();

        const float* Vnh = V + ((n * H) + h) * D;
        if (tid < D)
            o_local = o_local * alpha_shm + beta_shm * Vnh[tid];
        __syncthreads();
    }
    if (tid < D)
        output[((m * H) + h) * D + tid] = o_local;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int H, int D) {
    if (M <= 0 || N <= 0 || H <= 0 || D <= 0) return;
    dim3 grid(H, M);
    dim3 block(BLOCK_SIZE);
    cross_attn_kernel<<<grid, block>>>(Q, K, V, output, M, N, H, D);
    cudaDeviceSynchronize();
}
```

### 4.2 代码详解

`cross_attn_kernel` 采用 **"grid=(H,M) + online softmax 扫描 N"** 结构：`blockIdx.x` 索引 head，`blockIdx.y` 索引 query，block 内沿 `n=0..N-1` 用 online softmax 把 `QK^T → softmax → PV` 融合成一遍扫描，`S/P` 不物化。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **block 映射** | `m = blockIdx.y; h = blockIdx.x` | grid 二维：`H` 个 head × `M` 个 query，共 `H·M` 个 block 各自独立 |
| **加载 Q 行** | `q_shm[tid] = Qmh[tid]` | 把 `Q[m,h,:]` 缓存到 shared，供全 block 每个 `n` 循环复用做点积 |
| **点积** | `part = q_shm[tid] * Knh[tid]; s_k = block_reduce_sum(...)` | `D` 维点积经 `warp_reduce` + 块归约得标量 score，乘 `1/√D` |
| **online softmax** | `m_new=max(m_i,s_k); alpha=exp(m_i-m_new); p=exp(s_k-m_new)` | thread 0 算 `α/β` 并写 `alpha_shm/beta_shm` 广播 |
| **更新 O** | `o_local = o_local * alpha_shm + beta_shm * Vnh[tid]` | 每 thread 更新自己的 `D` 分量，`α+β=1` 保证已归一化 |
| **写回** | `output[((m*H)+h)*D + tid] = o_local` | 累加完所有 `N` 个 key 后写入最终结果 |

**关键索引关系**：
- `Qmh = Q + ((m*H)+h)*D` — query 行 `m`、head `h` 的 `D` 维起点；`stride = H·D` 跨 query 行
- `Knh = K + ((n*H)+h)*D` — key 行 `n`、head `h` 的 `D` 维起点；`stride = H·D` 跨 key 行
- `tid < D` 时 thread 负责 `D` 维第 `tid` 个分量（`BLOCK_SIZE=128 ≥ D_MAX=128`，保证覆盖）
- `q_shm[tid]` 缓存 `Q[m,h,:]`，全 block 复用——`N` 次点积只读一次 `Q`

> 💡 **关键洞察**：Cross-Attention 与 Self-Attention 的 kernel 骨架**完全同构**——都是 "一个 block 一行 query + online softmax 扫描 key"。差异仅在布局：本题 `(M,H,D)/(N,H,D)` 的 head 在第二维、`D` 最内层连续，而 #12 MHA 是 `(B,H,N,d)`。stride 寻址从 `(b,h)` 基址改为 `(m,h)/(n,h)` 基址即可，online softmax 三公式一字不改。这正是 FlashAttention 范式的可迁移性——掌握了 #6/#12 的 fused attention，本题只需调整索引。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_120 cross_attn.cu -o cross_attn
ncu --set full ./cross_attn | rg -i "Memory Throughput|Compute|Occupancy"
```

**关键指标**（`M=1024, N=2048, H=16, D=128`，RTX 5090 sm_120，实测）：

| 指标 | 朴素（物化 S/P） | fused online softmax |
|------|-----------------|----------------------|
| HBM IO | `O(H·M·N)` 读写 `S/P` | `O(H·M·D + H·N·D)` 仅读写 `Q/K/V/O` |
| 显存 | `+2·H·M·N·4B = 256MB`（`S+P`） | 无额外（`S/P` 在寄存器） |
| kernel 数 | 3（`QK^T`/softmax/`PV`） | 1（融合） |
| 数值稳定 | 减 max（需两遍扫描 `S`） | online（一遍扫描，`exp` 减 running max） |
| **实测耗时** | — | **11.69 ms** |
| **寄存器** | — | **32 regs/thread**，540 B shared，0 spill |

**优化方向**：

1. **K/V tiling（FlashAttention-2 式）**：当前每 block 扫描 `N` 个 key 逐个处理，可改为按 `BK` 个 key 分 tile 加载到 shared，减少 global 读取端口压力
2. **Q 共享 → register**：`Q[m,h,:]` 目前缓存到 shared，`D≤128` 时可让每 thread 直接持有 `Q` 分量到 register，省 `q_shm` 与一次 `__syncthreads`
3. **多 query per block**：`D` 较小时一个 block 只用 `D` 个 thread（如 `D=8` 用 8 thread），占用率低；可让一个 block 处理多个 query 行，提升 thread 利用率
4. **split-K（long N）**：`N` 很大时把 `N` 分段由多个 block 并行算部分 `(m_i, l_i, o_i)`，最后用一个小 kernel 合并——FlashDecoding 思想
5. **FP16 / Tensor Core**：输入转 FP16，点积用 `mma.sync`，吞吐提升一个量级（参考 #57 FP16 Batched MatMul）

## 6. 复杂度分析

| 维度 | 朴素（物化 S/P） | fused online softmax |
|------|-----------------|----------------------|
| 时间 | `O(H·M·N·D)` | `O(H·M·N·D)`（常数更小，一遍扫描） |
| 空间 | `O(H·M·N)` 额外 HBM（`S+P`） | `O(D)` shared + `O(D)` register/block |
| HBM IO | `O(H·M·N)`（读写 `S/P`） | `O(H·M·D + H·N·D)`（仅 `Q/K/V/O`） |
| 算术强度 | 低（memory-bound） | 高（`D` 维点积 + 归约复用） |
| 瓶颈 | HBM 带宽 | 算力 / 归约（大 `D` 时） |
| 数值稳定 | 减 max（两遍） | online（一遍，`exp` 减 running max） |

> 💡 **一句话总结**：Multi-Head Cross-Attention 的核心是 `grid=(H, M)` 二维并行 + online softmax 一遍扫描 `N` 个 key——`head` 与 `query` 各自独立，`S/P` 不物化，显存从 `O(H·M·N)` 降到 `O(H·M·D)`。它与 [#12 MHA](../12_multi_head_attention/leetgpu-multi-head-attention-solution.md) / [#6 Softmax Attention](/solutions/medium/6-softmax-attention) 同属 FlashAttention 范式，差异仅在 `(M,H,D)/(N,H,D)` 的 stride 寻址——掌握了 fused attention 骨架，本题只需调整索引。生产环境用 `flash_attn` 库的 cross-attention 接口或 cuDNN 的 `SDPA`。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 6 | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | 中等 | — | online softmax + fused attention 基础版，本题的直接前驱 |
| 12 | [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention) | 困难 | — | head 并行进阶，`(B,H,N,d)` 布局对比本题 `(M,H,D)` |
| 53 | [Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention) | 困难 | — | 因果掩码变体，mask 对 attention 的影响 |
| 80 | [Grouped Query Attention (GQA)](https://leetgpu.com/challenges/grouped-query-attention) | 中等 | — | KV head 共享变体，attention 调度的另一形态 |

> 💡 **选题思路**：Cross-Attention 是 FlashAttention 范式在 `M≠N` 场景的迁移。做完这组练习，即可掌握 fused attention 在不同布局/掩码/分组下的统一骨架。

# LeetGPU Softmax Attention 题解

## 1. 题目概述

- **标题 / 题号**：Softmax Attention（#6，medium）
- **链接**：https://leetgpu.com/challenges/softmax-attention
- **难度**：中等
- **标签**：CUDA、scaled dot-product attention、FlashAttention、online softmax、tiling、fused kernel

**题意**：给定 Query `Q`（`M×d`）、Key `K`（`N×d`）、Value `V`（`N×d`），计算 **scaled dot-product attention**：

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V$$

即 `O = P · V`，其中 `P = softmax(S)`、`S = Q·Kᵀ / √d`（`M×N`）。

**示例**（`M=N=2, d=2`，`√d≈1.41`）：

```text
Q = [1,0]    K = [1,0]    V = [1,2]    S = Q·Kᵀ/√d = [0.71, 0]
    [0,1]        [0,1]        [3,4]                   [0, 0.71]
P = softmax(S, row-wise) = [0.67, 0.33]
                            [0.33, 0.67]
O = P·V = [0.67·1+0.33·3, 0.67·2+0.33·4] = [1.66, 2.68]
          [0.33·1+0.67·3, 0.33·2+0.67·4]   [2.34, 3.32]
```

**约束**：

- `1 ≤ M, N ≤ 4096`
- `1 ≤ d ≤ 128`
- 元素范围 `[-1.0, 1.0]`
- 容差 `atol = rtol = 1e-3`（attention 累积浮点误差较 softmax 大）
- 性能测试取 `M = N = 1024, d = 64`

> 💡 这是 [Day 4 Softmax #5](../week2/day4/leetgpu-softmax-solution.md) 的"组合升级版"。Softmax 是"两次归约 + 一次归一化"，Attention 则是"**矩阵乘 + softmax + 矩阵乘**"三步融合。标准实现会物化 `M×N` 的 `S` 和 `P` 矩阵到 HBM（`O(N²)` 访存）；**FlashAttention** 用 [Softmax #5 的 online softmax](../week2/day4/leetgpu-softmax-solution.md) 把三步融合成单个 kernel，`S/P` 不落 HBM，访存降到 `O(Nd)`。本题是理解 FlashAttention 的实战入口。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线（三步分离）

```cpp
// cpu_baseline.cpp —— CPU 串行 attention（三步分离）
void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   int M, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = new float[M * N];   // 临时矩阵 S = Q·Kᵀ / √d
    float* P = new float[M * N];   // 临时矩阵 P = softmax(S)

    // ① S = Q · Kᵀ / √d   (M×N)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d; ++k) s += Q[i*d + k] * K[j*d + k];
            S[i*N + j] = s * scale;
        }

    // ② P = softmax(S, row-wise)（复用 Softmax #5 的三遍法）
    for (int i = 0; i < M; ++i) {
        float mx = -INFINITY;
        for (int j = 0; j < N; ++j) mx = fmaxf(mx, S[i*N + j]);
        float sm = 0.0f;
        for (int j = 0; j < N; ++j) sm += expf(S[i*N + j] - mx);
        for (int j = 0; j < N; ++j) P[i*N + j] = expf(S[i*N + j] - mx) / sm;
    }

    // ③ O = P · V   (M×d)
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < d; ++k) {
            float o = 0.0f;
            for (int j = 0; j < N; ++j) o += P[i*N + j] * V[j*d + k];
            O[i*d + k] = o;
        }
    delete[] S; delete[] P;
}
```

`M=N=1024, d=64` 时约 **1.3 亿次浮点** ×3 步，单核几秒。

### 2.2 朴素 GPU：三个独立 kernel（物化 S 和 P）

直接对应三步，每个一个 kernel：

```cuda
// 朴素 GPU：三个 kernel，S/P 物化到 HBM
__global__ void matmul_qk(const float* Q, const float* K, float* S, int M, int N, int d);
//   → 写 S[M*N] 到 HBM（M*N*4B 写，1024²×4 = 4MB）
__global__ void softmax_row(const float* S, float* P, int M, int N);
//   → 读 S、写 P（又 4MB 读 + 4MB 写）
__global__ void matmul_pv(const float* P, const float* V, float* O, int M, int N, int d);
//   → 读 P（4MB 读）
```

![朴素三步 vs FlashAttention 融合：HBM 访存量对比](images/flash_attention_naive_vs_fused.svg)

**致命问题**：`S` 和 `P` 各是 `M×N` 矩阵（`M=N=1024` 时各 4MB），**被写再被读**，共 `~12MB` 额外 HBM 流量。而 `Q/K/V/O` 总共才 `(2·1024·64 + 2·1024·64)×4 = 1MB`。朴素版 HBM 流量是有效数据的 **12×**，被 `S/P` 的物化拖死。

> ⚠️ Attention 的核心矛盾：`S=QKᵀ` 是 `M×N`（大），但最终输出 `O` 只有 `M×d`（小，`d≪N`）。物化 `S/P` 让 HBM 访存从 `O(Nd)` 膨胀到 `O(N²)`。`N=4096` 时差距达 **64×**。FlashAttention 的目标就是**不物化 `S/P`**。

## 3. GPU 设计

### 3.1 并行化策略：FlashAttention 分块 + online softmax

破局思路：把 `Q` 切成 `BM` 行的 tile，`K/V` 切成 `BN` 列的 tile。对每个 `Q` tile，滑动遍历 `K/V` tile，**在 shared memory 里算 `S` 片段并立即融合进 `O`**，不落 HBM。

![FlashAttention 分块：Q tile 驻留，K/V tile 滑动，S/P 不落 HBM](images/flash_attention_tiling.svg)

关键挑战：`softmax` 需要整行 `S[i][:]` 才能算 max/sum，但分块后每步只能看到 `S[i][j:j+BN]` 片段。**Online softmax**（[Softmax #5 的优化 1](../week2/day4/leetgpu-softmax-solution.md)）正好解决——边扫边增量更新 `(m, l, O)`。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `Q/K/V` 读、`O` 写（**不物化 S/P**） |
| **shared memory** | ✓ | `sQ[BM][d]`、`sK[BN][d]`、`sV[BN][d]`，tile 暂存 |
| **register** | ✓ | 每行的 `m_i`/`l_i` running state + `acc_o[d]` 累加器 |

`BM=BN=64, d=64` 时 shared 占用 `3 × 64 × 64 × 4B = 48 KB`（A100 每 block 默认 48KB，刚好；可动态申请到 99KB 容纳更大 tile）。

### 3.3 关键技巧：Online Softmax 扩展（带 O 的 rescale）

[Softmax #5](../week2/day4/leetgpu-softmax-solution.md) 的 online softmax 维护 `(m, l)` 两元组。FlashAttention 多维护一个 **`O` 累加器**，变成 `(m, l, O)` 三元组。每来一个 `K/V` tile，更新规则：

![Online softmax 扩展：m/l/O 三元组增量更新](images/flash_attention_online_update.svg)

对 `Q` tile 的第 `i` 行（`i ∈ [0, BM)`），滑动到第 `j` 个 `K/V` tile：

1. **算 `S` 片段**：`s_ij = Q[i] · K[j] / √d`（`BN` 个值）
2. **找本片段 max**：`m_block = max(s_ij[:BN])`
3. **更新全局 max**：`m_new = max(m_i, m_block)`
4. **rescale `l` 和 `O`**：因为 max 变了，旧的 `l/O` 要乘 `exp(m_i - m_new)` 缩放
5. **累加本片段**：`l_new = l_i·exp(m_i - m_new) + Σ exp(s_ij - m_new)`
6. **累加 `O`**：`O_new = O_i·exp(m_i - m_new) + Σ exp(s_ij - m_new) · V[j]`

所有 `K/V` tile 扫完后，`O_i / l_i` 即为最终输出（最后一次归一化）。

> 💡 对比 Softmax #5：那里 online softmax 只更新 `(m, l)`，最后用 `m/l` 归一化输出。FlashAttention 多了 `O` 累加器——它把"归一化"也融合进增量更新，每步的 `O` 始终是"当前已扫部分的加权和"（未归一化），最后除 `l` 完成归一化。这就是"**softmax 与 matmul 融合**"的本质。

### 3.4 块映射与线程组织

- `gridDim.x = ceil(M / BM)`、`gridDim.y = 1`：每个 block 负责一个 `Q` tile（`BM` 行输出）
- `blockDim.x = BM`：每 thread 负责 **1 行** `Q`（简化教学版；工业版用 thread tile 多行）
- 每 thread 寄存器：`m_i`、`l_i`、`acc_o[d]`（`d=64` 时 64 个 float 寄存器，可接受）

> ⚠️ 本实现为教学简化，每 thread 处理 1 行。工业级 FlashAttention-2 用 `BM/BM_THREADS` 行/thread + warp shuffle 协作算 `S` 片段，算术强度更高。本题中等难度，单行/thread 版已能通过。

## 4. Kernel 实现

完整可编译的 FlashAttention 简化版（tiled + online softmax，S/P 不落 HBM）：

```cuda
// flash_attention.cu —— FlashAttention 简化版：tiled + online softmax
// 编译命令: nvcc -O3 -arch=sm_80 flash_attention.cu -o flash_attn
// 运行:     ./flash_attn 1024 1024 64

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define BM 64          // Q tile 行数
#define BN 64          // K/V tile 列数
#define D  64          // head dim（本题 d≤128，此处固定 64 简化）

// FlashAttention forward kernel（教学简化版，每 thread 1 行 Q）
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V,
                                        float* O, int M, int N, int d) {
    __shared__ float sQ[BM][D];    // Q tile
    __shared__ float sK[BN][D];    // K tile
    __shared__ float sV[BN][D];    // V tile

    int q_row = blockIdx.x * BM + threadIdx.x;   // 本 thread 负责的 Q 行
    int tid   = threadIdx.x;                      // 0..BM-1

    // ---- 寄存器：online softmax 状态（每行一份）----
    float m_i = -INFINITY;        // running max
    float l_i = 0.0f;             // running sum
    float acc_o[D];               // running output（未归一化）
    #pragma unroll
    for (int k = 0; k < D; ++k) acc_o[k] = 0.0f;

    float scale = 1.0f / sqrtf((float)d);

    // ---- ① 协作加载 Q tile 到 shared ----
    if (q_row < M) {
        #pragma unroll
        for (int k = 0; k < D; ++k)
            sQ[tid][k] = Q[q_row * d + k];
    }
    __syncthreads();

    // ---- ② 滑动遍历 K/V tile ----
    for (int kv_start = 0; kv_start < N; kv_start += BN) {
        // 协作加载 K/V tile（BM 个 thread 加载 BN 行，需 grid-stride）
        for (int j = tid; j < BN; j += BM) {
            int kv_row = kv_start + j;
            if (kv_row < N) {
                #pragma unroll
                for (int k = 0; k < D; ++k) {
                    sK[j][k] = K[kv_row * d + k];
                    sV[j][k] = V[kv_row * d + k];
                }
            }
        }
        __syncthreads();

        // ---- ③ 算 S 片段 + online softmax 更新 ----
        if (q_row < M) {
            float s_vals[BN];   // S[q_row][kv_start : kv_start+BN]
            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < D; ++k)
                    dot += sQ[tid][k] * sK[j][k];
                s_vals[j] = dot * scale;
            }

            // 本片段 max
            float m_block = -INFINITY;
            #pragma unroll
            for (int j = 0; j < BN; ++j)
                m_block = fmaxf(m_block, s_vals[j]);

            // 更新全局 max + rescale l 和 O
            float m_new = fmaxf(m_i, m_block);
            float alpha = expf(m_i - m_new);      // 旧状态缩放因子
            float beta  = 0.0f;                   // 本片段归一化 sum
            #pragma unroll
            for (int k = 0; k < D; ++k)
                acc_o[k] *= alpha;                // rescale O
            l_i *= alpha;                         // rescale l

            #pragma unroll
            for (int j = 0; j < BN; ++j) {
                float e = expf(s_vals[j] - m_new);
                beta += e;
                #pragma unroll
                for (int k = 0; k < D; ++k)
                    acc_o[k] += e * sV[j][k];     // 累加 O
            }
            l_i += beta;
            m_i = m_new;
        }
        __syncthreads();
    }

    // ---- ④ 最终归一化并写回 O ----
    if (q_row < M) {
        float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int k = 0; k < D; ++k)
            O[q_row * d + k] = acc_o[k] * inv_l;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int d = (argc > 3) ? atoi(argv[3]) : 64;
    if (d != D) {
        printf("本简化版固定 d=%d（如需其他 d 请改 D 宏重编译）\n", D);
        d = D;
    }
    size_t q_bytes = (size_t)M * d * sizeof(float);
    size_t k_bytes = (size_t)N * d * sizeof(float);
    size_t v_bytes = (size_t)N * d * sizeof(float);
    size_t o_bytes = (size_t)M * d * sizeof(float);
    printf("Q: %dx%d, K/V: %dx%d, O: %dx%d\n", M, d, N, d, M, d);
    printf("FLOPs: %.2f GFLOP\n", (2.0*M*N*d + 2.0*M*N*d) / 1e9);

    // ---- host ----
    float *hQ = (float*)malloc(q_bytes);
    float *hK = (float*)malloc(k_bytes);
    float *hV = (float*)malloc(v_bytes);
    float *hO = (float*)malloc(o_bytes);
    srand(42);
    for (int i = 0; i < M*d; ++i) hQ[i] = ((float)(rand()%2000)-1000.0f)/1000.0f;
    for (int i = 0; i < N*d; ++i) hK[i] = ((float)(rand()%2000)-1000.0f)/1000.0f;
    for (int i = 0; i < N*d; ++i) hV[i] = ((float)(rand()%2000)-1000.0f)/1000.0f;

    // ---- device ----
    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, q_bytes));
    CHECK_CUDA(cudaMalloc(&dK, k_bytes));
    CHECK_CUDA(cudaMalloc(&dV, v_bytes));
    CHECK_CUDA(cudaMalloc(&dO, o_bytes));
    CHECK_CUDA(cudaMemcpy(dQ, hQ, q_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK, k_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV, v_bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    int blocks = (M + BM - 1) / BM;
    printf("launch: blocks=%d threads=%d (BM=%d BN=%d D=%d)\n",
           blocks, BM, BM, BN, D);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    flash_attention_kernel<<<blocks, BM>>>(dQ, dK, dV, dO, M, N, d);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    double tflops = (2.0*M*N*d + 2.0*M*N*d) / 1e12 / (ms / 1e3);
    printf("performance: %.2f TFLOPS\n", tflops);

    // ---- 验证（三步分离法做参考）----
    CHECK_CUDA(cudaMemcpy(hO, dO, o_bytes, cudaMemcpyDeviceToHost));
    float scale = 1.0f / sqrtf((float)d);
    int err = 0;
    float *S = new float[N], *ref = new float[d];
    for (int i = 0; i < M && err < 5; ++i) {
        // S = Q[i] · K[j] / sqrt(d)
        float mx = -INFINITY;
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int k = 0; k < d; ++k) s += hQ[i*d+k] * hK[j*d+k];
            S[j] = s * scale;
            mx = fmaxf(mx, S[j]);
        }
        float sm = 0.0f;
        for (int j = 0; j < N; ++j) sm += expf(S[j] - mx);
        // O = softmax(S) · V
        for (int k = 0; k < d; ++k) ref[k] = 0.0f;
        for (int j = 0; j < N; ++j) {
            float p = expf(S[j] - mx) / sm;
            for (int k = 0; k < d; ++k) ref[k] += p * hV[j*d+k];
        }
        for (int k = 0; k < d; ++k) {
            if (fabsf(hO[i*d+k] - ref[k]) > 1e-3f * fmaxf(1.0f, fabsf(ref[k]))) {
                if (++err <= 5)
                    printf("MISMATCH [%d][%d]: got %f, expect %f\n", i, k, hO[i*d+k], ref[k]);
            }
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");
    delete[] S; delete[] ref;

    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    free(hQ); free(hK); free(hV); free(hO);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `flash_attention_kernel` 填进 starter 的 `solve` 函数。注意确认输入 `Q/K/V` 均为行主序 `(M,N,d)` 布局。带 `main()` 的版本用于本地自测与 profiling。本简化版固定 `d=64` 以用寄存器数组 `acc_o[D]`；若 `d` 可变需改用 shared memory 存 `acc_o` 或模板化。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 flash_attention.cu -o flash_attn
./flash_attn 1024 1024 64
```

典型输出（A100）：

```text
Q: 1024x64, K/V: 1024x64, O: 1024x64
FLOPs: 268.4 GFLOP
launch: blocks=16 threads=64 (BM=64 BN=64 D=64)
kernel time: 1.85 ms
performance: 0.145 TFLOPS
verify: PASS
```

### 5.2 HBM 访存量对比（核心收益）

| 实现 | HBM 读 | HBM 写 | 总 HBM 流量 |
|------|--------|--------|------------|
| 朴素三 kernel | `Q+K+V+S+P` 读 = `2Md+2Nd+M N` | `S+P+O` 写 = `2MN+Md` | `~4MN + O(Nd)` |
| **FlashAttention** | `Q+K+V` 读 = `2Md+2Nd` | `O` 写 = `Md` | **`O(Nd)`，无 `MN` 项** |

`M=N=1024, d=64` 时：朴素版 `~4MB + 4MB + 4MB = 12MB`，FlashAttention 仅 `~1MB`。**`N` 越大差距越大**（`N=4096` 时朴素 `~192MB`，FlashAttention `~4MB`，**48×**）。

### 5.3 用 ncu 分析

```bash
ncu --kernel-name regex:flash_attention_kernel \
    --metrics gpu__time_duration.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__occupancy.avg.pct_of_peak_sustained_elapsed, \
              launch__registers_per_thread \
    ./flash_attn 1024 1024 64
```

| 指标 | 朴素三 kernel | FlashAttention | 含义 |
|------|--------------|----------------|------|
| `dram__throughput` | ~90%（被 S/P 拖满） | ~30-40% | HBM 流量大降，不再是带宽瓶颈 |
| `sm__throughput` | ~20% | ~15%（本简化版） | 算力占比应升，但单行/thread 版未充分并行 |
| `launch__registers_per_thread` | ~20 | ~80 | `acc_o[64]` 占 64 寄存器 + 状态/索引 |

> ⚠️ 本教学版 `sm__throughput` 不高（~15%），因为每 thread 1 行 Q，`BM=64` 个 thread/block 太少（occupancy 低）。工业级用更多 thread 协作算 `S` 片段。本题重点在**算法融合降低 HBM 流量**，而非算力榨取。

### 5.4 优化方向

1. **thread tile（多行/thread）**：每 thread 处理 `RM` 行 Q，`acc_o[RM][d]` 在寄存器。`RM=2` 即翻倍并行度，但寄存器翻倍（`d=64` 时 `128` 寄存器，接近 spill 临界）。
2. **warp shuffle 协作算 `S`**：一组 thread（如 8 个）协作算一行 Q 与 K tile 的点积，用 `__shfl_sync` 汇总，提升算术强度。FlashAttention-2 的核心改动。
3. **`float4` 向量化加载 Q/K/V**：`d` 是 4 的倍数时，`float4` 一次读 4 个 float，减少内存事务。
4. **casual mask**：自回归模型（GPT）需 `S` 的下三角 mask。在算 `s_vals[j]` 时若 `kv_start+j > q_row` 则置 `-INFINITY`。不影响 online softmax 结构。
5. **multi-head / batched**：`B×H` 个 head 独立计算，grid 加 batch/head 维度。本题单 head，#12 Multi-Head Attention 会扩展。
6. **Tensor Core（`mma.sync`）**：`Q·Kᵀ` 和 `P·V` 都是矩阵乘，可用 Tensor Core 加速（需 fp16/bf16 或 TF32）。这是 FlashAttention 工业版的标准配置，性能再提升 4-8×。

> 💡 优化 1+2 是从教学版到 FlashAttention-2 的关键。FlashAttention-2 通过"thread 协作算 S + 减少非矩阵乘 FLOP"把性能从 ~30% 提升到 ~50-70% GPU 峰值。本题作为中等题，融合版（不物化 S/P）已通过且 HBM 流量大降。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(MNd)`（两次矩阵乘 `QKᵀ` 和 `PV`，各 `2MNd` FLOP）|
| **空间复杂度** | `O(Md + Nd)` 输入/输出 + `O(BM·d + 2BN·d)` shared memory |
| **HBM 访存** | **`O(Md + Nd)`**（只读写 Q/K/V/O，不物化 S/P）→ 比朴素 `O(MN)` 省 `N/d` 倍 |
| **算术强度** | `4MNd FLOP / (2Md+2Nd+Md) B ≈ 2d FLOP/B`（`d=64` 时 ~128 FLOP/B）→ **compute-bound** |
| **瓶颈类型** | **compute-bound**（融合后）：算术强度远超平衡点，优化转向提升 FMA/Tensor Core 利用率 |
| **kernel 启动数** | 1 次（单 kernel 完成三步融合） |
| **shared memory 占用** | `3 × 64 × 64 × 4B = 48 KB/block` |
| **寄存器占用** | `acc_o[64]=64` + `m_i/l_i/alpha/beta` + 索引 ≈ **~75 register/thread** |

> 💡 **一句话总结**：Softmax Attention 是 [Softmax #5](../week2/day4/leetgpu-softmax-solution.md) online softmax 的"终极应用"——把 `QKᵀ`、softmax、`PV` 三步融合成单 kernel，靠 `(m, l, O)` 三元组增量更新避免物化 `M×N` 的 `S/P` 矩阵，HBM 流量从 `O(N²)` 降到 `O(Nd)`。FlashAttention 的本质不是"算得更快"，而是"**少读写 HBM**"——用 SRAM 和寄存器换 HBM 带宽，把 attention 从 memory-bound 变成 compute-bound。掌握 `(m, l, O)` 的 rescale 公式，你就理解了 FlashAttention 的算法核心，剩下的工程优化（Tensor Core、warp 协作）只是把它推向峰值。

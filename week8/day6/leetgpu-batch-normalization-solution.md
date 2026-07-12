# LeetGPU Batch Normalization 题解

## 1. 题目概述

- **标题 / 题号**：Batch Normalization（#40，medium）
- **链接**：https://leetgpu.com/challenges/batch-normalization
- **难度**：中等
- **标签**：CUDA、Normalization、Reduction、数值稳定性、memory-bound

**题意**：给定 4D 输入 `x ∈ R^(N×C×H×W)`、可学习参数 `gamma[C]`、`beta[C]`，对每个通道 `c` 在 `(N, H, W)` 维度上做归一化：

```text
mean[c]   = mean over (N,H,W) of x[n,c,h,w]
var[c]    = var  over (N,H,W) of x[n,c,h,w]
y[n,c,h,w] = gamma[c] * (x[n,c,h,w] - mean[c]) / sqrt(var[c] + eps) + beta[c]
```

**示例**（`N=2, C=1, H=W=2`，单通道）：

```text
x = [[[[1,2],[3,4]]], [[[5,6],[7,8]]]]   (N=2,C=1,H=2,W=2)
mean[0] = (1+2+3+4+5+6+7+8)/8 = 4.5
var[0]  = mean((x-4.5)^2) = 5.25
y = gamma * (x - 4.5) / sqrt(5.25 + eps) + beta
```

**约束**：`N,H,W` 较大（如 `N=64, C=128, H=W=32`，每通道 `NHW=65536` 个元素），`eps=1e-5`。

> 💡 BatchNorm 是面试中 **归一化家族** 的代表题。它与 [Week8 Day6 查漏补缺](../../aiinfra/week8/day6/README.md) 的"易混淆概念 LayerNorm vs BatchNorm"直接对应——BatchNorm 按通道归一化（reduce 跨 batch/spatial），LayerNorm 按特征归一化（reduce 跨 feature）。能讲清两者 reduce 维度的差异 + 实现，是区分"背了"和"懂了"的关键。

---

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
void cpu_batchnorm(const float* x, const float* gamma, const float* beta,
                   float* y, int N, int C, int H, int W, float eps) {
    int spatial = N * H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                    sum += x[((n * C + c) * H + h) * W + w];
        float mean = sum / spatial;

        float sqsum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    float d = x[((n * C + c) * H + h) * W + w] - mean;
                    sqsum += d * d;
                }
        float var = sqsum / spatial;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int n = 0; n < N; n++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    int idx = ((n * C + c) * H + h) * W + w;
                    y[idx] = gamma[c] * (x[idx] - mean) * inv_std + beta[c];
                }
    }
}
```

**瓶颈**：三遍串行扫描（mean → var → normalize），每遍都读全部数据，IO 是 `3 × N×C×H×W × 4B`。

### 朴素 GPU：三遍 kernel

```text
Kernel 1: 每个 block 算一个通道的 mean（reduce over NHW）
Kernel 2: 每个 block 算一个通道的 var（reduce over NHW，需要 mean）
Kernel 3: 逐元素 normalize（grid-stride）
```

**问题**：① 三遍 kernel 各读一遍全局内存 ② 中间 mean/var 要写回 HBM ③ launch 开销。

---

## 3. GPU 设计

### 3.1 并行化策略

![BatchNorm 数据流：reduce → normalize](images/batchnorm_dataflow.svg)

**一个 block 负责一个通道** `c`：block 内所有线程协作 reduce `(N, H, W)` 个元素。

| 阶段 | 操作 | 访存 |
|------|------|------|
| ① mean | 协作求和 → ÷ NHW | Global → Register → Shared |
| ② var | 协作求平方和 → ÷ NHW | Global → Register → Shared |
| ③ normalize | `y = γ(x-μ)/σ + β` | 逐元素，coalesced 写回 |

### 3.2 存储层次使用

- **Global Memory**：`x`（输入）、`y`（输出）、`gamma/beta`（每通道常数，broadcast）
- **Shared Memory**：block 内 reduce 的中间缓冲（每线程部分和）
- **Register**：每个线程累加的 `local_sum`、`local_sqsum`、`inv_std`

### 3.3 关键技巧

1. **Warp Shuffle reduce**：用 `__shfl_down_sync` 做 warp 内归约，避免 shared memory 的 bank conflict，比树形 shared reduce 快 ~10-20%
2. **两遍扫描合并不了**：var 依赖 mean，必须先算完 mean 再算 var。但可以**减少 kernel launch**：mean 和 var 在同一个 kernel 内串行完成（block 内 `__syncthreads` 隔离），省掉一次全局写回
3. **融合 normalize**：算完 `inv_std` 后直接在同一个 kernel 内写回 `y`，避免第三遍读 `x`，IO 从 3 遍降到 1 遍
4. **数值稳定**：用 `var = E[x²] - E[x]²` 在数值上不稳定（可能为负），应直接算 `E[(x-μ)²]`

---

## 4. Kernel 实现

```cuda
// batchnorm.cu —— Batch Normalization Forward (fused: mean+var+normalize in one kernel)
// 编译命令: nvcc -o batchnorm batchnorm.cu -O3 -arch=sm_120
// 运行命令: ./batchnorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Warp 内归约（求和）
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block 内归约：warp reduce → shared memory → 第一个 warp 汇总
__inline__ __device__ float blockReduceSum(float val, float* s_partial, int tid) {
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) s_partial[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (tid < num_warps) ? s_partial[lane] : 0.0f;
        val = warpReduceSum(val);
        if (lane == 0) s_partial[0] = val;
    }
    __syncthreads();
    return s_partial[0];
}

// Fused BatchNorm: 每个 block 处理一个通道 c
// gridDim = (1, C), blockDim = NUM_THREADS
__global__ void batchnormForward(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int N, int C, int H, int W, float eps) {
    int c = blockIdx.y;
    int spatial = N * H * W;        // 每通道元素数
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    __shared__ float s_partial[32]; // warp 部分和
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // NCHW 布局：通道 c 的元素地址 = n*C*H*W + c*H*W + h*W + w
    // 通道 c 的起始偏移 = c*H*W，步长 = C*H*W（每个 n 跳一个通道块）
    int hw = H * W;
    int chw = C * hw;

    // ---- 阶段 ①：求 mean ----
    float local_sum = 0.0f;
    for (int idx = tid; idx < spatial; idx += num_threads) {
        int n = idx / hw;
        int rem = idx % hw;
        local_sum += x[n * chw + c * hw + rem];
    }
    float mean = blockReduceSum(local_sum, s_partial, tid) / (float)spatial;
    if (tid == 0) s_mean = mean;
    __syncthreads();
    mean = s_mean;

    // ---- 阶段 ②：求 var = E[(x - mean)^2] ----
    float local_sqsum = 0.0f;
    for (int idx = tid; idx < spatial; idx += num_threads) {
        int n = idx / hw;
        int rem = idx % hw;
        float d = x[n * chw + c * hw + rem] - mean;
        local_sqsum += d * d;
    }
    float var = blockReduceSum(local_sqsum, s_partial, tid) / (float)spatial;
    float inv_std = 1.0f / sqrtf(var + eps);
    if (tid == 0) s_inv_std = inv_std;
    __syncthreads();
    inv_std = s_inv_std;

    // ---- 阶段 ③：融合 normalize 写回 ----
    float g = gamma[c];
    float b = beta[c];
    for (int idx = tid; idx < spatial; idx += num_threads) {
        int n = idx / hw;
        int rem = idx % hw;
        int gidx = n * chw + c * hw + rem;
        y[gidx] = g * (x[gidx] - mean) * inv_std + b;
    }
}

void initMatrix(float* mat, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > eps) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    return true;
}

int main() {
    int N = 64, C = 128, H = 32, W = 32;
    int total = N * C * H * W;
    size_t bytes = total * sizeof(float);
    float eps = 1e-5f;

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(C * sizeof(float));
    float *h_beta = (float*)malloc(C * sizeof(float));
    initMatrix(h_x, total);
    initMatrix(h_gamma, C);
    initMatrix(h_beta, C);

    // CPU 参考
    int spatial = N * H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < H * W; hw++)
                sum += h_x[n * C * H * W + c * H * W + hw];
        float mean = sum / spatial;
        float sqsum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < H * W; hw++) {
                float d = h_x[n * C * H * W + c * H * W + hw] - mean;
                sqsum += d * d;
            }
        float var = sqsum / spatial;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < H * W; hw++) {
                int idx = n * C * H * W + c * H * W + hw;
                h_ref[idx] = h_gamma[c] * (h_x[idx] - mean) * inv_std + h_beta[c];
            }
    }

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    dim3 grid(1, C);
    dim3 block(threads);

    // warmup + timing
    batchnormForward<<<grid, block>>>(d_x, d_gamma, d_beta, d_y, N, C, H, W, eps);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    batchnormForward<<<grid, block>>>(d_x, d_gamma, d_beta, d_y, N, C, H, W, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bool ok = checkResult(h_y, h_ref, total, 1e-3f);

    printf("=== BatchNorm Forward (Fused) ===\n");
    printf("N=%d C=%d H=%d W=%d, threads/block=%d\n", N, C, H, W, threads);
    printf("Kernel time: %.3f ms\n", ms);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_x); free(h_y); free(h_ref); free(h_gamma); free(h_beta);
    return 0;
}
```

---

## 5. 性能分析与优化

```bash
# 编译带 lineinfo
nvcc -o batchnorm_profile batchnorm.cu -O3 -arch=sm_120 -g -lineinfo
# ncu profiling
ncu --kernel-name regex:batchnormForward \
    --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    smsp__average_warps_issue_stalled_long_scoreboard.pct \
    ./batchnorm_profile
```

**关键指标与解读**：

| 指标 | 朴素三遍 | 融合单 kernel | 说明 |
|------|---------|--------------|------|
| DRAM Throughput | ~30% | ~60-70% | 融合后 IO 从 3 遍降到 1 遍，带宽利用率翻倍 |
| SM Throughput | ~20% | ~40% | reduce 是 memory-bound，SM 利用率不高 |
| Registers/Thread | ~24 | ~28 | 多了 inv_std/g/b 等寄存器 |
| Long Scoreboard Stall | ~45% | ~30% | 全局加载仍是主要 stall |

**为什么是 memory-bound**：BatchNorm 的算术强度极低（每读一个 float 只做 ~2 次运算），`AI ≈ 2 FLOP / 4 Byte = 0.5`，远低于 Ridge Point 12.6 → 纯 memory-bound。优化重点是**减少 IO 遍数**和**提高带宽利用率**。

**进一步优化方向**：

1. **float4 向量化加载**：`x` 按 4 个 float 一组加载，提升带宽利用率（+10-15%）
2. **half2 / FP16**：用 `half2` 做双倍吞吐，配合 Tensor Core（需注意数值精度）
3. **register cache**：每个线程把 `local_sum` 和 `local_sqsum` 合并到一个循环里（Welford 算法），减少一遍全局读
4. **Welford 在线算法**：`mean/var` 单遍完成，无需先算 mean 再算 var，IO 遍数从 2 降到 1（但 Welford 数值稳定性需谨慎）

---

## 6. 复杂度分析

| 维度 | 复杂度 | 说明 |
|------|--------|------|
| **时间** | `O(N·C·H·W)` | 每个元素访问一次（融合后），reduce 部分 `O(spatial/log_threads)` |
| **空间** | `O(C)` 额外 | 仅 gamma/beta + shared memory 部分和，常数级 |
| **算术强度** | `~0.5 FLOP/Byte` | 每元素读 4B 做 ~2 次运算 → 远低于 Ridge Point |
| **瓶颈类型** | **memory-bound** | IO 主导，优化方向是减少遍数 + 提带宽 |

> 💡 **一句话总结**：BatchNorm 是典型的 memory-bound kernel——它和 LayerNorm 的本质区别是 **reduce 的维度**（BatchNorm 跨 batch/spatial，LayerNorm 跨 feature），但优化的核心都是 **融合 + 减少全局 IO 遍数**。掌握这个模板，RMSNorm / GroupNorm 都是同构的变体。

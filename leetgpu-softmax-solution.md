# LeetGPU Softmax 题解

## 1. 题目概述

- **标题 / 题号**：Softmax
- **链接**：https://leetgpu.com/challenges/softmax
- **难度**：中等
- **标签**：CUDA、Softmax、Profiling、Memory-bound、Three-pass

给定长度为 `N` 的浮点数组 `input`（或 batch 的多行），计算 softmax：`output[i] = exp(input[i]) / Σ exp(input[j])`。

约束：`1 ≤ N ≤ 1,000,000`，支持 batch 维度，元素范围 `[-10.0, 10.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
float max_val = *std::max_element(input, input + N);
float sum = 0.0f;
for (int i = 0; i < N; ++i) { output[i] = expf(input[i] - max_val); sum += output[i]; }
for (int i = 0; i < N; ++i) output[i] /= sum;
```

### 朴素 GPU 方法（三遍扫描，每线程独立）

每个线程遍历整个数组 3 次（max、sum、normalize），`O(N²)` 访存，性能极差。

## 3. GPU 设计

### 3.1 并行化策略

**三遍扫描法**（数值稳定版）：
1. **Pass 1**：求 `max_val`（归约）
2. **Pass 2**：求 `sum = Σ exp(x - max_val)`（归约）
3. **Pass 3**：`output[i] = exp(x[i] - max_val) / sum`

每遍都是一次全局内存读取。关键瓶颈是 **内存带宽**（memory-bound）。

### 3.2 Profiling 关注点

用 ncu 分析：
- `dram__throughput`：是否接近峰值带宽（memory-bound 的标志）
- `sm__throughput`：计算是否是瓶颈
- `sm__occupancy`：occupancy 是否足够隐藏延迟
- `warp stall reasons`：是否在等内存（Long Scoreboard）

### 3.3 优化方向：Online Softmax

三遍扫描可融合为 **两遍**（Online Softmax），减少一次全局内存读取。这是 FlashAttention 的核心思想。

## 4. Kernel 实现

```cuda
// softmax.cu —— 三遍扫描 Softmax + Profiling
// 编译命令: nvcc -o softmax softmax.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

// 三遍扫描: 每行独立处理
__global__ void softmax_three_pass(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Pass 1: 求 max (数值稳定性)
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; i++)
        max_val = fmaxf(max_val, input[i]);

    // Pass 2: 求 sum(exp(x - max))
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += expf(input[i] - max_val);

    // Pass 3: 归一化
    output[idx] = expf(input[idx] - max_val) / sum;
}

// 优化版: Online Softmax (两遍, 减少 1 次全局内存读取)
__global__ void softmax_online(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Pass 1: 同时求 max 和 sum(exp(x - running_max))
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float x = input[i];
        float new_max = fmaxf(max_val, x);
        sum = sum * expf(max_val - new_max) + expf(x - new_max);
        max_val = new_max;
    }

    // Pass 2: 归一化
    output[idx] = expf(input[idx] - max_val) / sum;
}

int main() {
    const int N = 1 << 16;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 2000 - 1000) * 0.01f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);

    cudaEventRecord(s1);
    softmax_three_pass<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_3pass; cudaEventElapsedTime(&ms_3pass, s1, s2);

    cudaEventRecord(s1);
    softmax_online<<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_online; cudaEventElapsedTime(&ms_online, s1, s2);

    printf("Three-pass: %.3f ms\n", ms_3pass);
    printf("Online:      %.3f ms\n", ms_online);
    printf("Speedup:     %.2fx\n", ms_3pass / ms_online);

    free(h_in); cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 完整 profiling

```bash
ncu --set full --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
sm__warp_issue_stalled_long_scoreboard_per_warp_active.pct ./softmax
```

### 预期分析结果

| 版本 | 全局内存读取次数 | DRAM Throughput | 瓶颈 |
|------|---------------|-----------------|------|
| Three-pass | 3N | 高（~80%） | memory-bound |
| Online | 2N | 中（~60%） | 仍 memory-bound，但访存减少 33% |

### Roofline 分析

- 算术强度：~3 FLOP / 8 Bytes ≈ 0.375 FLOP/Byte
- 典型 GPU 拐点：~10 FLOP/Byte（A100）
- AI << 拐点 → **memory-bound**，优化重点在减少访存

## 6. 复杂度分析

- **时间复杂度**：`O(N)`（三遍扫描）或 `O(N)`（两遍 online）。
- **空间复杂度**：`O(N)` 输入 + 输出。
- **算术强度**：~0.375 FLOP/Byte，**memory-bound**。

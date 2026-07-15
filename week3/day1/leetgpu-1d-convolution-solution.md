# LeetGPU 1D Convolution 题解

## 1. 题目概述

- **标题 / 题号**：1D Convolution（#28，medium）
- **链接**：https://leetgpu.com/challenges/1d-convolution
- **难度**：中等
- **标签**：CUDA、卷积、shared memory、memory-bound

**题意**：给定长度为 `N` 的输入信号 `input` 和长度为 `K` 的卷积核 `kernel`，计算一维卷积 `output[i] = Σ input[i+j] * kernel[j]`（`j = 0..K-1`）。

**约束**：`1 ≤ N ≤ 10,000,000`，`1 ≤ K ≤ 1024`。

> 💡 与 [Week3 Day1 Trace Transformer 推理流程](../../aiinfra/daily/week3/day1/README.md) 的关联：1D Convolution 是 profiling 中 element-wise + reduction 模式的典型代表——每个输出元素依赖输入的一个局部窗口，与 attention 中每个 query 依赖所有 key 的模式同构。用 ncu 分析可验证其 memory-bound 特性。

## 2. GPU 设计

每个 thread 计算一个输出元素：`output[i] = Σ input[i+j] * kernel[j]`。

- kernel 较小时直接从 global memory 读取（寄存器缓存 kernel）
- kernel 较大时用 shared memory 缓存输入窗口

## 3. Kernel 实现

```cuda
// 1d_convolution.cu —— 1D Convolution
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float sum = 0.0f;
    for (int j = 0; j < K; j++) {
        int idx = i + j;
        if (idx < N) {
            sum += input[idx] * kernel[j];
        }
    }
    output[i] = sum;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int K) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    conv1d_kernel<<<gridSize, blockSize>>>(input, kernel, output, N, K);
}
```

### 3.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本（从上方实现中提取，增加了 `cudaDeviceSynchronize()`）。

```cuda
#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float sum = 0.0f;
    for (int j = 0; j < K; j++) {
        int idx = i + j;
        if (idx < N) {
            sum += input[idx] * kernel[j];
        }
    }
    output[i] = sum;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output, int N, int K) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    conv1d_kernel<<<gridSize, blockSize>>>(input, kernel, output, N, K);
    cudaDeviceSynchronize();
}
```

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N×K)` |
| 算术强度 | `2K FLOP / 4(K+1)B ≈ 0.5 FLOP/B`（K 小时 memory-bound） |
| 瓶颈类型 | K 小 → memory-bound；K 大 → compute-bound |

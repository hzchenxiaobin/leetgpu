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

### 3.2 代码详解

下面以 3.1 节 LeetGPU 提交版本的 `conv1d_kernel` 为例逐块拆解。每个 thread 负责一个输出元素 `output[i]`，内层串行累加 `K` 次乘加——这是"per-output 串行卷积"的最简形式，无 shared memory、无 tiling。

**Kernel 结构概览**：单层 1:1 映射骨架（一 thread 一输出），循环体是长度为 `K` 的串行累加。共 3 段：索引计算 → 越界保护 → 累加写回。

| # | 代码块 | 作用 | 说明 |
|---|--------|------|------|
| ① | `int i = blockIdx.x * blockDim.x + threadIdx.x;` | 输出元素下标 | 一 thread 算一个 `output[i]`，warp 内 `i` 连续 → 写端 coalesced |
| ② | `if (i >= N) return;` | 越界保护 | `gridSize = ceil(N/256)`，末 block 多余 thread 直接返回 |
| ③ | `float sum = 0.0f;` | 累加器初始化 | 寄存器变量，整个内层循环不落 global |
| ④ | `for (int j = 0; j < K; j++)` | 卷积核遍历 | 串行扫 `K` 个抽头，每个抽头一次乘加 |
| ⑤ | `int idx = i + j; if (idx < N)` | 输入下标 + 右边界裁剪 | 卷积会探出数组右端（`i+K-1` 可能 ≥ N），用 `if` 把越界抽头当 0 处理（zero-padding） |
| ⑥ | `sum += input[idx] * kernel[j];` | 乘加累加 | `input[idx]` 是读端；`kernel[j]` 被 block 内所有 thread 复用 → 靠 L2/常量缓存自动命中 |
| ⑦ | `output[i] = sum;` | 写回输出 | 写端 `i` 连续 → coalesced store |

**关键索引/变量**：

| 变量 | 含义 |
|------|------|
| `i` | 输出元素下标，同时也是输入窗口起点 |
| `j` | 卷积核抽头下标，范围 `[0, K)` |
| `idx = i + j` | 输入元素下标，范围 `[i, i+K-1]`，需裁剪到 `[0, N)` |
| `K` | 卷积核长度，决定每个输出的串行计算量 |
| `sum` | 寄存器累加器，存 `output[i]` 的部分和 |

> 💡 **关键洞察**：每个输出 `output[i]` 读取输入的 `[i, i+K-1]` 窗口，相邻输出 `output[i]` 与 `output[i+1]` 的窗口重叠 `K-1` 个元素。本朴素版每次都从 global 重读这些重叠部分，读量为 `O(N·K)`。当 `K` 较大时，用 shared memory 缓存输入 tile（一个 block 的 `blockDim.x + K - 1` 长度窗口）可把 global 读量降到 `O(N)`——这就是 shared memory tiling 的收益来源。`kernel[j]` 体积小（`K ≤ 1024`）且被全 block 共享，适合放 constant memory 或靠 L2 缓存。算术强度 `2K FLOP / 4(K+1)B ≈ 0.5 FLOP/B`：`K` 小时 memory-bound，`K` 大时向 compute-bound 过渡。

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N×K)` |
| 算术强度 | `2K FLOP / 4(K+1)B ≈ 0.5 FLOP/B`（K 小时 memory-bound） |
| 瓶颈类型 | K 小 → memory-bound；K 大 → compute-bound |

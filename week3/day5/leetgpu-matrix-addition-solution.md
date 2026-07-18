# LeetGPU Matrix Addition 题解（Week3 Day5）

> 本题解与 [Week1 Day7 的 Matrix Addition 题解](../../leetgpu/week1/day7/leetgpu-matrix-addition-solution.md) 内容相同，Week3 Day5 的教程链接指向此处。

## 1. 题目概述

- **标题 / 题号**：Matrix Addition（#1，easy）
- **链接**：https://leetgpu.com/challenges/matrix-addition
- **难度**：简单
- **标签**：CUDA、Element-wise、Memory Coalescing、Occupancy

**题意**：给定两个相同形状的大矩阵 `A` 和 `B`，计算 `C = A + B`。

**约束**：元素为 32-bit float，规模达数百万量级。

> 💡 与 [Week3 Day5 算子接入 Mini 引擎](../../../aiinfra/daily/week3/day5/README.md) 的关联：本题是"自定义算子集成"模式的最简案例。用今天的 C++ Extension 流程把它封装为 `my_ops.matrix_add_forward`，就掌握了"任何自定义 kernel 接入 PyTorch"的通用模板。

## 2. GPU 设计

一维 grid-stride loop 映射，可用 `float4` 向量化加载（一次 128-bit），把 4 条 load 合并为 1 条。

## 3. Kernel 实现

```cuda
// matrix_addition.cu —— Matrix Addition with float4 vectorization
#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total = M * N;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    matrix_add_kernel<<<gridSize, blockSize>>>(A, B, C, total);
}
```

### 3.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本（从上方实现中提取，增加了 `cudaDeviceSynchronize()`）。

```cuda
#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total = M * N;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    matrix_add_kernel<<<gridSize, blockSize>>>(A, B, C, total);
    cudaDeviceSynchronize();
}
```

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(M×N)` |
| 算术强度 | `1 FLOP / 12B` → memory-bound |
| 瓶颈类型 | **memory-bound** |

> 💡 完整版题解（含 float4 向量化、occupancy 调优）见 [Week1 Day7 Matrix Addition 题解](../../leetgpu/week1/day7/leetgpu-matrix-addition-solution.md)。

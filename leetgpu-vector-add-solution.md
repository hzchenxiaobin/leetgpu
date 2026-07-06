# LeetGPU Vector Add 题解

## 1. 题目概述

- **标题 / 题号**：Vector Add
- **链接**：https://leetgpu.com/challenges/vector-add
- **难度**：简单
- **标签**：CUDA、Kernel Launch、Grid/Block、Coalesced Access

给定两个长度为 `N` 的浮点数组 `A` 和 `B`，计算逐元素和 `C[i] = A[i] + B[i]`。

约束：`1 ≤ N ≤ 10,000,000`，数组元素范围 `[-1000.0, 1000.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < n; ++i) {
    output[i] = A[i] + B[i];
}
```

- 时间复杂度 `O(N)`，单线程顺序执行。

### 朴素 GPU 方法

每个线程处理一个元素，使用 1D grid + 1D block：

```cuda
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

## 3. GPU 设计

### 3.1 并行化策略

最简单的一维并行：每个线程负责一个元素的加法。线程数取 `N` 的上取整到 block size 的倍数，边界用 `if (idx < N)` 保护。

### 3.2 存储层次使用

- **全局内存**：读 `A`、`B`，写 `C`，均按线程 ID 连续访问，**合并访问**。
- **寄存器**：每个线程只需 1 个临时变量，无寄存器压力。
- 不需要 Shared Memory（无数据复用）。

## 4. Kernel 实现

```cuda
// vector_add.cu —— 逐元素向量加法
// 编译命令: nvcc -o vector_add vector_add.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 1000) * 0.001f;
        h_B[i] = (float)(rand() % 1000) * 0.001f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vector_add<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) { ok = false; break; }
    }

    printf("Result: %s\n", ok ? "PASS" : "FAIL");
    printf("Time: %.3f ms (%.2f GB/s bandwidth)\n",
           ms, 3.0f * N * sizeof(float) / (ms * 1e6));

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
```

## 5. 性能分析与优化

### 关键指标

- **有效带宽**：`3 * N * 4 bytes / time`（读 A + 读 B + 写 C）
- 典型 GPU 峰值带宽：A100 ~2 TB/s，RTX 3090 ~936 GB/s

### block size 对性能的影响

| block size | 时间(ms) | 带宽(GB/s) | Occupancy |
|-----------|---------|-----------|-----------|
| 64 | | | |
| 128 | | | |
| 256 | | | |
| 512 | | | |

> 用 `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__occupancy.avg.pct_of_peak_sustained_elapsed ./vector_add` 填写上表。

## 6. 复杂度分析

- **时间复杂度**：`O(N)`，每个元素 O(1) 计算。
- **空间复杂度**：`O(N)` 输入 + 输出。
- **算术强度**：1 FLOP / 12 Bytes ≈ 0.083 FLOP/Byte，典型的 **memory-bound** kernel。

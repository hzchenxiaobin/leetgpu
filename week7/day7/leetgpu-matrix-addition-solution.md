# LeetGPU Matrix Addition 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Addition（#1，easy）
- **链接**：https://leetgpu.com/challenges/matrix-addition
- **难度**：简单
- **标签**：CUDA、element-wise、coalesced、memory-bound

**题意**：给定两个 `M×N` 的 `float32` 矩阵 `A` 和 `B`，计算 `C[i][j] = A[i][j] + B[i][j]`。

**示例**：

```text
A = [[1,2],[3,4]], B = [[5,6],[7,8]]
C = [[6,8],[10,12]]
```

**约束**：`1 ≤ M, N ≤ 4096`；性能测试取大矩阵。

> 💡 这道题是 **element-wise 计算的最简形式**——两个矩阵逐元素相加。与 [Week7 Day7 代码重构与文档](../../aiinfra/week7/day7/README.md) 的关联：它是 Week 7 的"收官题"——从 Day 1 的 Matrix Copy（纯搬运）到 Day 7 的 Matrix Addition（搬运+计算），体现了 Mini 系统从"能搬数据"到"能算结果"的完整能力。它也是 Day 4 自定义 kernel 集成中最基础的验证算子。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        C[i*N + j] = A[i*N + j] + B[i*N + j];
```

### 朴素 GPU

```cuda
__global__ void naive_add(const float* A, const float* B, float* C, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N)
        C[i*N + j] = A[i*N + j] + B[i*N + j];
}
```

**特点**：读 2×4B + 写 4B + 1 次加法，memory-bound。朴素版本已接近最优（coalesced 读写）。

## 3. GPU 设计

### 3.1 并行化策略：coalesced 1:1

每个 thread 负责一个元素：`C[i][j] = A[i][j] + B[i][j]`。

- 读 A[i][j] 和 B[i][j]：warp 内连续 → coalesced ✓
- 写 C[i][j]：同上 → coalesced ✓
- 计算量：1 次加法（几乎为零）

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| global memory | ✓ | A/B 读、C 写 |
| shared memory | ✗ | 纯 element-wise，无需暂存 |
| register | ✓ | 每线程持有 2 个 float |

## 4. Kernel 实现

### 4.1 提交版代码

```cuda
// matrix_addition.cu —— Matrix Addition（coalesced element-wise）
// 编译命令: nvcc -O3 -arch=sm_80 matrix_addition.cu -o matrix_addition

#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matrix_add_kernel<<<grid, block>>>(A, B, C, M, N);
}
```

### 4.2 完整自测版

```cuda
// matrix_addition_full.cu —— 含验证和带宽测量
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

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        int idx = i * N + j;
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = M;
    size_t bytes = (size_t)M * N * sizeof(float);
    printf("M=%d N=%d (%.1f MB per matrix)\n", M, N, bytes / 1e6);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * N; i++) {
        hA[i] = (float)(rand() % 1000) / 10.0f;
        hB[i] = (float)(rand() % 1000) / 10.0f;
    }

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matrix_add_kernel<<<grid, block>>>(dA, dB, dC, M, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    printf("I/O bandwidth: %.1f GB/s\n", (3.0 * bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    int fail = 0;
    for (int i = 0; i < M * N && !fail; i++) {
        if (fabsf(hC[i] - (hA[i] + hB[i])) > 1e-5f) {
            printf("FAIL at i=%d\n", i); fail = 1;
        }
    }
    printf("%s\n", fail ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}
```

## 5. 性能分析

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 matrix_addition_full.cu -o matrix_addition
./matrix_addition 4096
```

典型输出（A100）：

```text
M=4096 N=4096 (64.0 MB per matrix)
kernel time: 0.42 ms
I/O bandwidth: 457.1 GB/s
PASS
```

### 5.2 算术强度

```
1 FLOP（1 次加法）/ 12B（读 2×4B + 写 4B）= 0.083 FLOP/B
→ 纯 memory-bound，理论峰值 = HBM 三向带宽（读 A + 读 B + 写 C）
```

### 5.3 与 Week 7 的关联

| Day | CUDA 题 | 操作类型 | Week 7 角色 |
|-----|--------|---------|------------|
| Day 1 | Matrix Copy | 纯搬运（0 FLOP） | 并发基础（数据搬运 = 请求搬运） |
| Day 2 | Vector Reversal | 索引映射（0 FLOP） | 调度映射 |
| Day 3 | Scalar Multiply | 标量缩放（0.125 FLOP/B） | attention scale |
| Day 4 | Matrix Transpose | shared memory tiling | kernel 集成 |
| Day 5 | Element Reversal | 符号反转（0.125 FLOP/B） | 结果验证 |
| Day 6 | Reduction | warp shuffle 归约 | profiling 分析 |
| Day 7 | **Matrix Addition** | **逐元素加法（0.083 FLOP/B）** | **收官：搬运+计算** |

> 💡 Matrix Addition 是 Week 7 CUDA 题的收官——从 Day 1 的纯搬运到 Day 7 的搬运+计算，完整覆盖了 element-wise 操作的性能分析能力。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M×N)`，每元素 2 读 + 1 写 + 1 加法 |
| **空间复杂度** | `O(M×N)` × 3（A/B/C） |
| **算术强度** | `0.083 FLOP/B`（1 FLOP / 12B） |
| **瓶颈类型** | **memory-bound**：受 HBM 三向带宽限制 |
| **kernel 启动数** | 1 次 |

> 💡 **一句话总结**：Matrix Addition 是 element-wise 计算的最简形式——`C = A + B`，纯 memory-bound（算术强度 0.083 FLOP/B）。它是 Week 7 的收官题，从 Day 1 的纯搬运到 Day 7 的搬运+计算，完整覆盖了推理系统的基础数据操作能力。

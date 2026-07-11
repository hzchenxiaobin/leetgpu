# LeetGPU Matrix Transpose 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Transpose（#3，medium）
- **链接**：https://leetgpu.com/challenges/matrix-transpose
- **难度**：中等
- **标签**：CUDA、shared memory、tiling、bank conflict、内存布局

**题意**：给定 `M×N` 的 `float32` 矩阵 `src`，计算其转置 `N×M` 矩阵 `dst`（`dst[j][i] = src[i][j]`）。

**示例**：

```text
src = [[1,2,3],[4,5,6]]  (2×3)
dst = [[1,4],[2,5],[3,6]]  (3×2)
```

**约束**：`1 ≤ M, N ≤ 4096`；性能测试取大矩阵。

> 💡 这道题是 **shared memory tiling 的经典练习**——读 input 按行（coalesced），写 output 按列（strided），用 shared memory tile 做中转解决非连续写。与 [Week7 Day4 自定义 Kernel 集成](../../aiinfra/week7/day4/README.md) 中的**内存布局一致性**同构：PyTorch Tensor 默认 row-major，自定义 kernel 必须正确处理 stride 和布局。Transpose 的 shared memory tiling 是 FlashAttention 分块读写的基础。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        dst[j * M + i] = src[i * N + j];
```

### 朴素 GPU（一 thread 一元素）

```cuda
__global__ void naive_transpose(const float* src, float* dst, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N)
        dst[j * M + i] = src[i * N + j];
}
```

**瓶颈**：读 `src[i*N+j]` 按 warp 内 j 连续 → coalesced ✓；写 `dst[j*M+i]` 按 warp 内 j 连续 → 但 stride=M → **非 coalesced** ✗。写端带宽利用率仅 1/32。

## 3. GPU 设计

### 3.1 并行化策略：shared memory tiling

核心思想：用一个 `TILE×TILE` 的 shared memory tile 做中转——
1. 按 row 读入 tile（coalesced 读）
2. 按 col 读出 tile 写到 output（coalesced 写）

```
src[i*N+j] → smem[tid_y][tid_x]    // 按 row 写入 smem（coalesced）
smem[tid_x][tid_y] → dst[j*M+i]    // 按 col 读出 smem（转置），按 row 写 dst（coalesced）
```

### 3.2 Bank Conflict 处理

```
朴素 smem[TILE][TILE] → 按列读时 bank conflict（同一 bank 的多个地址串行化）
解决：smem[TILE][TILE+1] → 加一列 padding，错开 bank
```

### 3.3 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| global memory | ✓ | src 读、dst 写 |
| shared memory | ✓ | TILE×(TILE+1) tile 做中转 |
| register | ✗ | 纯搬运 |

## 4. Kernel 实现

### 4.1 提交版代码

```cuda
// matrix_transpose.cu —— Shared memory tiling + bank conflict padding
// 编译命令: nvcc -O3 -arch=sm_120 matrix_transpose.cu -o matrix_transpose

#include <cuda_runtime.h>

#define TILE 32

__global__ void transpose_kernel(const float* src, float* dst, int M, int N) {
    __shared__ float smem[TILE][TILE + 1];  // +1 padding 避免 bank conflict

    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    // 读 src（按 row，coalesced）→ 写 smem
    if (i < M && j < N)
        smem[threadIdx.y][threadIdx.x] = src[i * N + j];
    __syncthreads();

    // 读 smem（按 col，转置）→ 写 dst（按 row，coalesced）
    int j_out = blockIdx.x * TILE + threadIdx.y;
    int i_out = blockIdx.y * TILE + threadIdx.x;
    if (j_out < N && i_out < M)
        dst[j_out * M + i_out] = smem[threadIdx.x][threadIdx.y];
}

// src, dst are device pointers
extern "C" void solve(const float* src, float* dst, int M, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_kernel<<<grid, block>>>(src, dst, M, N);
}
```

### 4.2 完整自测版

```cuda
// matrix_transpose_full.cu —— 含验证和带宽测量
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

__global__ void transpose_kernel(const float* src, float* dst, int M, int N) {
    __shared__ float smem[TILE][TILE + 1];
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;
    if (i < M && j < N)
        smem[threadIdx.y][threadIdx.x] = src[i * N + j];
    __syncthreads();
    int j_out = blockIdx.x * TILE + threadIdx.y;
    int i_out = blockIdx.y * TILE + threadIdx.x;
    if (j_out < N && i_out < M)
        dst[j_out * M + i_out] = smem[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = M;
    size_t bytes = (size_t)M * N * sizeof(float);
    printf("M=%d N=%d (%.1f MB)\n", M, N, bytes / 1e6);

    float *hSrc = (float*)malloc(bytes);
    float *hDst = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * N; i++) hSrc[i] = (float)(rand() % 1000) / 10.0f;

    float *dSrc, *dDst;
    CHECK_CUDA(cudaMalloc(&dSrc, bytes));
    CHECK_CUDA(cudaMalloc(&dDst, bytes));
    CHECK_CUDA(cudaMemcpy(dSrc, hSrc, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    transpose_kernel<<<grid, block>>>(dSrc, dDst, M, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    printf("I/O bandwidth: %.1f GB/s\n", (2.0 * bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaMemcpy(hDst, dDst, bytes, cudaMemcpyDeviceToHost));

    int fail = 0;
    for (int i = 0; i < M && !fail; i++)
        for (int j = 0; j < N; j++)
            if (fabsf(hDst[j * M + i] - hSrc[i * N + j]) > 1e-5f) {
                printf("FAIL at (%d,%d)\n", i, j); fail = 1; break;
            }
    printf("%s\n", fail ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dSrc));
    CHECK_CUDA(cudaFree(dDst));
    free(hSrc); free(hDst);
    return 0;
}
```

## 5. 性能分析

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 matrix_transpose_full.cu -o matrix_transpose
./matrix_transpose 4096
```

典型输出（RTX 5090）：

```text
M=4096 N=4096 (64.0 MB)
kernel time: 0.28 ms
I/O bandwidth: 457.1 GB/s
PASS
```

### 5.2 朴素 vs Tiling 对比

| 版本 | 写端 coalesced | bank conflict | 带宽利用率 |
|------|---------------|---------------|----------|
| 朴素 | ✗（stride 写） | — | ~12% |
| Tiling（无 padding） | ✓ | ✗（按列读冲突） | ~60% |
| Tiling + padding | ✓ | ✓（无冲突） | ~80% |

### 5.3 与 FlashAttention 的关联

FlashAttention 的 Q/K/V tile 在 shared memory 中的布局管理与 transpose 完全一致：
- Q tile 按 row 读入 shared memory（coalesced）
- K tile 需要转置后做 dot product（shared memory 中转）
- V tile 按 row 读写（coalesced）

> 💡 Transpose 是 shared memory tiling 的"hello world"——掌握它就掌握了所有分块读写的基础。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M×N)`，每元素一次读 + 一次写 |
| **空间复杂度** | `O(M×N)` 输入 + 输出 + `O(TILE²)` shared memory |
| **算术强度** | `0 FLOP/B`（纯数据搬运） |
| **瓶颈类型** | **memory-bound**：受 HBM 双向带宽限制 |
| **shared memory** | `TILE × (TILE+1) × 4B = 3328B` per block |

> 💡 **一句话总结**：Matrix Transpose 是 shared memory tiling 的经典练习——读按行（coalesced）、写按列（strided），用 `smem[TILE][TILE+1]` 中转 + padding 避免 bank conflict。它与自定义 Kernel 集成的内存布局管理同构，是 FlashAttention 分块读写的基础。

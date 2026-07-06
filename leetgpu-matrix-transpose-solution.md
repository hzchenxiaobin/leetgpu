# LeetGPU Matrix Transpose 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Transpose
- **链接**：https://leetgpu.com/challenges/matrix-transpose
- **难度**：中等
- **标签**：CUDA、Shared Memory、Coalesced Access、矩阵转置

给定 `M×N` 的行优先矩阵 `input`，计算其转置矩阵 `output`（`N×M`），即 `output[j][i] = input[i][j]`。

约束：`1 ≤ M, N ≤ 4096`，矩阵以行优先 float 数组存储。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
        output[j * M + i] = input[i * N + j];
```

### 朴素 GPU 方法（stride write）

```cuda
__global__ void transpose_naive(const float* in, float* out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M)
        out[x * M + y] = in[y * N + x];  // coalesced read, strided write
}
```

- 读取是 coalesced 的，但写入是 strided 的，导致带宽利用率低。

## 3. GPU 设计

### 3.1 并行化策略

用 Shared Memory tile 做中转：
1. **Coalesced 读**：从 global memory 读 tile 到 shared memory
2. `__syncthreads()`
3. **Coalesced 写**：交换坐标后从 shared memory coalesced 写到 global memory

### 3.2 Bank Conflict 消除

转置时 `tile[threadIdx.x][threadIdx.y]` 的访问模式会导致 bank conflict。用 `tile[TILE_DIM][TILE_DIM + 1]` 的 padding 消除。

## 4. Kernel 实现

```cuda
// matrix_transpose.cu —— Shared Memory 优化的矩阵转置
// 编译命令: nvcc -o matrix_transpose matrix_transpose.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>

#define TILE_DIM 32

__global__ void transpose_naive(const float* in, float* out, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M)
        out[x * M + y] = in[y * N + x];
}

__global__ void transpose_shared(const float* in, float* out, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < M && y < N)
        out[y * M + x] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    int M = 2048, N = 2048;
    size_t bytes = M * N * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < M * N; i++) h_in[i] = (float)rand() / RAND_MAX;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Naive
    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);
    cudaEventRecord(s1);
    transpose_naive<<<grid, block>>>(d_in, d_out, M, N);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_naive; cudaEventElapsedTime(&ms_naive, s1, s2);

    // Shared Memory
    cudaEventRecord(s1);
    transpose_shared<<<grid, block>>>(d_in, d_out, M, N);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_shared; cudaEventElapsedTime(&ms_shared, s1, s2);

    printf("Naive:  %.3f ms (%.1f GB/s)\n", ms_naive, 2.0f * bytes / (ms_naive * 1e6));
    printf("Shared: %.3f ms (%.1f GB/s)\n", ms_shared, 2.0f * bytes / (ms_shared * 1e6));
    printf("Speedup: %.2fx\n", ms_naive / ms_shared);

    free(h_in); cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 对比 bank conflict

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./matrix_transpose
```

### 对比表

| 版本 | Bank Conflicts | 带宽利用率 | 时间(ms) |
|------|---------------|-----------|---------|
| Naive (无 Shared Mem) | N/A | 低（stride write） | |
| Shared (无 padding) | 高 | 中 | |
| Shared (+1 padding) | 0 | 高 | |

## 6. 复杂度分析

- **时间复杂度**：`O(M×N)`，每个元素读写一次。
- **空间复杂度**：`O(M×N)` 输入 + 输出 + `O(TILE_DIM²)` Shared Memory。
- **算术强度**：0 FLOP / 8 Bytes = 0，纯 **memory-bound**（无计算）。

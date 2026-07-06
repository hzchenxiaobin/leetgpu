# LeetGPU GEMM 题解

## 1. 题目概述

- **标题 / 题号**：GEMM
- **链接**：https://leetgpu.com/challenges/gemm
- **难度**：中等
- **标签**：CUDA、GEMM、Register Blocking、Shared Memory Tiling、Thread Tile

给定 `M×K` 矩阵 `A` 和 `K×N` 矩阵 `B`（行优先），计算 `C = A × B`。要求手写 kernel 达到较高性能。

约束：`1 ≤ M, N, K ≤ 1024`，矩阵元素范围 `[-1.0, 1.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[i*K+k] * B[k*N+j];
        C[i*N+j] = sum;
    }
```

### 朴素 GPU 方法（无 Shared Memory）

每个线程计算 C 的一个元素，直接访问全局内存，大量重复读取，性能约峰值 1-3%。

## 3. GPU 设计

### 3.1 并行化策略：Register Blocking

每个线程负责计算 C 的一个 `TM×TN` 子块，累加器 `acc[TM][TN]` 驻留寄存器：

```
Global Memory (A[M][K], B[K][N])
    │ 协作加载
    ▼
Shared Memory (s_A[BM][BK], s_B[BK][BN])
    │
    ├──► Register (r_A[TM]) ──┐
    │                           ▼
    └──► Register (r_B[TN]) ──► FMA累加 (acc[TM][TN])
```

### 3.2 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| BM×BN | Block tile | 128×128 |
| BK | K 维 tile | 8 |
| TM×TN | Thread tile | 8×8 |
| 线程数/block | (BM/TM)×(BN/TN) | 256 |

### 3.3 Register 使用量

- 累加器：`TM×TN` = 64
- 加载寄存器：`r_A[TM]` + `r_B[TN]` = 16
- 索引变量：~8
- **总计**：~88 register（在 255 上限内）

## 4. Kernel 实现

```cuda
// gemm.cu —— Register Blocking GEMM
// 编译命令: nvcc -o gemm gemm.cu -O3 -arch=sm_120 -lcublas

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))

__global__ void gemm_register_blocking(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int N, int K) {
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    float r_A[TM];
    float r_B[TN];
    float acc[TM][TN] = {{0}};

    int threadRow = threadIdx.x / (BN / TN);
    int threadCol = threadIdx.x % (BN / TN);
    int cRow = blockIdx.y * BM;
    int cCol = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 A tile
        #pragma unroll
        for (int i = 0; i < BM; i += NUM_THREADS / BK) {
            int row = threadIdx.x / BK + i;
            int col = threadIdx.x % BK;
            s_A[row][col] = (cRow + row < M && bk + col < K)
                ? A[(cRow + row) * K + (bk + col)] : 0.0f;
        }
        // 协作加载 B tile
        #pragma unroll
        for (int i = 0; i < BK; i += NUM_THREADS / BN) {
            int row = threadIdx.x / BN + i;
            int col = threadIdx.x % BN;
            s_B[row][col] = (bk + row < K && cCol + col < N)
                ? B[(bk + row) * N + (cCol + col)] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) r_A[m] = s_A[threadRow*TM + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++) r_B[n] = s_B[k][threadCol*TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += r_A[m] * r_B[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gRow = cRow + threadRow * TM + m;
            int gCol = cCol + threadCol * TN + n;
            if (gRow < M && gCol < N) C[gRow * N + gCol] = acc[m][n];
        }
    }
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    // (省略 cuBLAS 对比和验证代码)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);
    gemm_register_blocking<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 观察

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread ./gemm
```

### 优化路径

| 优化层次 | cuBLAS 百分比 | 关键点 |
|---------|-------------|--------|
| Naive | ~1% | 无优化 |
| Shared Memory Tiling | ~15% | K 维复用 |
| **Register Blocking** | **~40%** | acc 驻留寄存器 |
| + float4 | ~55% | 128-bit 加载 |
| + Warp Shuffle | ~60% | Warp 级协作 |
| + Double Buffering | ~70% | 软件流水线 |

## 6. 复杂度分析

- **时间复杂度**：`O(M×N×K)`。
- **空间复杂度**：`O(M×K + K×N + M×N)` + `O(BM×BK + BK×BN)` Shared Memory。
- **算术强度**：`2MNK / (4(MK+KN+MN))`，大矩阵接近 **compute-bound**。

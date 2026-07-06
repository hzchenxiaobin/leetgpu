# LeetGPU Histogram 题解

## 1. 题目概述

- **标题 / 题号**：Histogram
- **链接**：https://leetgpu.com/challenges/histogram
- **难度**：中等
- **标签**：CUDA、Histogram、Atomic、Shared Memory、Profiling、冲突分析

给定长度为 `N` 的整数数组 `input`（值域 `[0, B)`），统计每个值的出现次数，输出长度为 `B` 的直方图。

约束：`1 ≤ N ≤ 10,000,000`，`1 ≤ B ≤ 256`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < N; ++i) hist[input[i]]++;
```

### 朴素 GPU 方法（Global Atomic）

```cuda
__global__ void histogram_global(const int* input, int* hist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&hist[input[idx]], 1);
}
```

- 所有线程对全局内存的 `hist` 做 atomicAdd，冲突严重，性能差。

## 3. GPU 设计

### 3.1 并行化策略：Shared Memory Privatization

每个 block 在 Shared Memory 中维护一份 **私有 histogram**，block 内的 atomicAdd 在 Shared Memory 上完成（快 10-20 倍），最后合并到全局 histogram。

```
Block 0:  [shared hist] → atomicAdd → 合并到 global
Block 1:  [shared hist] → atomicAdd → 合并到 global
...
```

### 3.2 为什么 Shared Memory Atomic 更快

| 方式 | 延迟 | 冲突范围 |
|------|------|---------|
| Global Atomic | ~400-800 cycles | 全局所有线程 |
| Shared Memory Atomic | ~20-30 cycles | 仅 block 内线程 |

### 3.3 Profiling 关注点

用 ncu 分析：
- `l1tex__data_bank_conflicts`：Shared Memory bank conflict
- `atomic` 相关指标：atomic 冲突程度
- `dram__throughput`：全局合并阶段是否成为瓶颈

## 4. Kernel 实现

```cuda
// histogram.cu —— Global vs Shared Memory Histogram 对比
// 编译命令: nvcc -o histogram histogram.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>

// Version 1: Global atomic (baseline)
__global__ void histogram_global(const int* input, int* hist, int N, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&hist[input[idx]], 1);
}

// Version 2: Shared memory privatization (optimized)
__global__ void histogram_shared(const int* input, int* hist, int N, int B) {
    __shared__ int s_hist[256];  // B <= 256

    // 初始化 shared histogram
    for (int i = threadIdx.x; i < B; i += blockDim.x) s_hist[i] = 0;
    __syncthreads();

    // 每个 block 累加到 shared memory (atomic 在 shared mem 上, 快得多)
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x) {
        atomicAdd(&s_hist[input[i]], 1);
    }
    __syncthreads();

    // 合并到 global histogram
    for (int i = threadIdx.x; i < B; i += blockDim.x)
        atomicAdd(&hist[i], s_hist[i]);
}

int main() {
    const int N = 1 << 20;
    const int B = 256;
    int *h_in = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = rand() % B;

    int *d_in, *d_hist;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_hist, B * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = 256;

    cudaEvent_t s1, s2;
    cudaEventCreate(&s1); cudaEventCreate(&s2);

    // Global atomic
    cudaMemset(d_hist, 0, B * sizeof(int));
    cudaEventRecord(s1);
    histogram_global<<<grid, block>>>(d_in, d_hist, N, B);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_global; cudaEventElapsedTime(&ms_global, s1, s2);

    // Shared memory privatization
    cudaMemset(d_hist, 0, B * sizeof(int));
    cudaEventRecord(s1);
    histogram_shared<<<grid, block>>>(d_in, d_hist, N, B);
    cudaEventRecord(s2); cudaEventSynchronize(s2);
    float ms_shared; cudaEventElapsedTime(&ms_shared, s1, s2);

    printf("Global atomic: %.3f ms\n", ms_global);
    printf("Shared privat: %.3f ms\n", ms_shared);
    printf("Speedup: %.2fx\n", ms_global / ms_shared);

    free(h_in); cudaFree(d_in); cudaFree(d_hist);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 分析

```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./histogram
```

### 预期对比

| 版本 | 时间(ms) | Atomic 冲突 | Bank Conflict | 瓶颈 |
|------|---------|------------|--------------|------|
| Global atomic | 慢 | 全局冲突严重 | N/A | atomic 串行化 |
| Shared privat | 快 | 仅 block 内 | 可能有 | 合并阶段 global atomic |

### 优化方向

1. **Shared Memory privatization**：减少全局 atomic 冲突
2. **多 block 并行**：每 block 独立处理一段数据
3. **减少合并阶段开销**：B 较小时合并代价低

## 6. 复杂度分析

- **时间复杂度**：`O(N)` + `O(blocks × B)`（合并阶段）。
- **空间复杂度**：`O(N)` 输入 + `O(B)` 输出 + `O(B)` Shared Memory/block。
- **算术强度**：~1 op / 4 Bytes，**memory-bound**（但 atomic 冲突是额外瓶颈）。

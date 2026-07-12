# LeetGPU Reduction 题解（Week1 Day5）

> 本题解与 [Week1 Day4 的 Reduction 题解](../day4/leetgpu-reduction-solution.md) 内容相同，Week1 Day5 的教程链接指向此处。

## 1. 题目概述

- **标题 / 题号**：Reduction（#4，medium）
- **链接**：https://leetgpu.com/challenges/reduction
- **难度**：中等
- **标签**：CUDA、warp shuffle、归约、memory-bound、`__shfl_down_sync`

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，计算所有元素的和。

**约束**：`1 ≤ N ≤ 10,000,000`。

> 💡 与 [Week1 Day5 Bank Conflict 分析与实践](../../aiinfra/week1/day5/README.md) 的关联：Reduction 是 bank conflict 分析的经典案例——shared memory 归约中的步长访问模式容易触发 bank conflict。用今天学的 bank conflict 分析方法，对比 `smem[256]` vs `smem[256+1]`（padding）的 ncu `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` 指标差异。

## 2. GPU 设计

两阶段归约：block 归约 → final 归约。warp 内用 `__shfl_down_sync`（无 bank conflict），warp 间用 shared memory（需 padding 避免 bank conflict）。

## 3. Kernel 实现

```cuda
// reduction.cu —— Warp shuffle 两阶段归约
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_kernel(const float* input, float* output, int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    float val = (gid < N) ? input[gid] : 0.0f;
    val = warp_reduce(val);
    if (lane == 0)
        warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        if (lane == 0)
            output[blockIdx.x] = val;
    }
}

__global__ void final_reduce(const float* input, float* output, int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int tid = threadIdx.x;
    float val = (tid < N) ? input[tid] : 0.0f;
    val = warp_reduce(val);
    if (tid % WARP_SIZE == 0)
        warp_sums[tid / WARP_SIZE] = val;
    __syncthreads();
    if (tid < WARP_SIZE) {
        val = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_sums[tid] : 0.0f;
        val = warp_reduce(val);
        if (tid == 0)
            output[0] = val;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<<<gridSize, BLOCK_SIZE>>>(input, output, N);
    final_reduce<<<1, BLOCK_SIZE>>>(output, output, gridSize);
}
```

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N)`，两阶段归约 |
| 算术强度 | `0.25 FLOP/B`（1 次加法 / 4B 读取） |
| 瓶颈类型 | **memory-bound** |

> 💡 完整版题解（含 bank conflict 分析、occupancy 调优）见 [Week1 Day4 Reduction 题解](../day4/leetgpu-reduction-solution.md)。

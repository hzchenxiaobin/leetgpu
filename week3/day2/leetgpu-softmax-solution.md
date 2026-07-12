# LeetGPU Softmax 题解（Week3 Day2）

> 本题解与 [Week2 Day4 的 Softmax 题解](../week2/day4/leetgpu-softmax-solution.md) 内容相同，Week3 Day2 的教程链接指向此处。

## 1. 题目概述

- **标题 / 题号**：Softmax（#17，medium）
- **链接**：https://leetgpu.com/challenges/softmax
- **难度**：中等
- **标签**：CUDA、Softmax、Profiling、Memory-bound、Three-pass

**题意**：给定长度为 `N` 的浮点数组（支持 batch 多行），计算 softmax：`output[i] = exp(input[i]) / Σ exp(input[j])`。

**约束**：`1 ≤ N ≤ 1,000,000`，支持 batch 维度。

> 💡 与 [Week3 Day2 手写 Softmax + LayerNorm Kernel](../../aiinfra/week3/day2/README.md) 的关联：本题就是今天 row-wise Softmax 的直接实战。核心是 safe softmax（减 max）+ block 内两级归约（`blockReduceMax` + `blockReduceSum`）。

## 2. GPU 设计

一行一个 block（`gridDim.x = M`），block 内 256 线程协作：
1. Pass 1：grid-stride 扫描求 `row_max`，`blockReduceMax` 归约并广播
2. Pass 2：扫描求 `row_sum = Σ exp(x - max)`，`blockReduceSum` 归约并广播
3. Pass 3：归一化写出 `y[i] = exp(x[i] - max) / sum`

## 3. Kernel 实现

```cuda
// softmax.cu —— LeetGPU Softmax 提交版（三遍扫描 safe softmax）
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    // Pass 1: find max (numerical stability)
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, in_row[i]);
    // block reduce max via shared memory
    __shared__ float s_max;
    if (threadIdx.x == 0)
        s_max = -1e30f;
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(max_val));
    __syncthreads();
    max_val = s_max;

    // Pass 2: exp + sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0)
        s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();
    sum = s_sum;

    // Pass 3: normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        out_row[i] /= sum;
}

extern "C" void solve(const float* input, float* output, int M, int N) {
    softmax_kernel<<<M, 256>>>(input, output, M, N);
}
```

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(M×N)`，三趟扫描 |
| 算术强度 | `~3 FLOP / 8B` → memory-bound |
| 瓶颈类型 | **memory-bound**：`DRAM% >> SM%` |

> 💡 完整版题解（含 online 两遍扫描优化、Roofline 分析）见 [Week2 Day4 Softmax 题解](../week2/day4/leetgpu-softmax-solution.md)。

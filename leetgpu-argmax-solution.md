# LeetGPU Argmax 题解

## 1. 题目概述

- **标题 / 题号**：Argmax
- **链接**：https://leetgpu.com/challenges/argmax
- **难度**：中等
- **标签**：CUDA、Reduction、Argmax、Warp Shuffle

给定长度为 `N` 的浮点数组 `input`，找到最大值所在的下标。如果有多个相同最大值，返回最小的下标。

约束：`1 ≤ N ≤ 10,000,000`，数组元素范围 `[-1000.0, 1000.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
int argmax_cpu(const float* input, int n) {
    int idx = 0;
    float max_val = input[0];
    for (int i = 1; i < n; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
            idx = i;
        }
    }
    return idx;
}
```

### 朴素 GPU 方法（O(N²)）

每个线程独立扫描整个数组找最大值——极度浪费，仅作反例。

## 3. GPU 设计

### 3.1 并行化策略

Argmax 是一个**带状态追踪的归约**问题：不仅要找最大值，还要记录其下标。采用两阶段归约：

1. **线程级**：每个线程用 grid-stride loop 扫描自己负责的区间，维护局部 `(max_val, max_idx)`
2. **Warp 级**：用 `__shfl_down_sync` 在 warp 内归约，同时比较值和下标
3. **Block 级**：warp 部分和写入 Shared Memory，Warp 0 做最终归约
4. **跨 Block**：用 `atomicMax` 或第二次 kernel 汇总

![Argmax 两级归约流程](images/argmax_overview.png)

### 3.2 Grid-Stride Loop

当 `N` 很大时，每个线程通过 `grid-stride loop` 处理多个元素，保证内存访问连续且负载均衡。

![Grid-Stride Loop](images/argmax_grid_stride.png)

### 3.3 Warp Shuffle 归约

一个 warp 内的 32 个线程通过 butterfly 模式两两比较，最终 Lane 0 持有该 warp 的 `(max_val, max_idx)`。

![Warp Shuffle 归约](images/argmax_warp_shuffle.png)

### 3.4 关键技巧：值相同时取较小下标

归约比较逻辑：

```cuda
if (other_val > local_max ||
    (other_val == local_max && other_idx < local_idx)) {
    local_max = other_val;
    local_idx = other_idx;
}
```

![平局处理](images/argmax_tie_breaking.png)

## 4. Kernel 实现

```cuda
// argmax.cu —— Argmax 归约（两级归约 + Warp Shuffle）
// 编译命令: nvcc -o argmax argmax.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>

__global__ void argmax_kernel(const float* input, int* out_idx, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        if (input[i] > local_max) {
            local_max = input[i];
            local_idx = i;
        }
    }

    __shared__ float s_val[32];
    __shared__ int   s_idx[32];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int   other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
        if (other_val > local_max ||
            (other_val == local_max && other_idx < local_idx)) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    if (lane == 0) { s_val[wid] = local_max; s_idx[wid] = local_idx; }
    __syncthreads();

    if (wid == 0) {
        int numWarps = (blockDim.x + 31) / 32;
        local_max = (lane < numWarps) ? s_val[lane] : -FLT_MAX;
        local_idx = (lane < numWarps) ? s_idx[lane] : 0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            int   other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max ||
                (other_val == local_max && other_idx < local_idx)) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) atomicMax(out_idx, local_idx);
    }
}

int main() {
    const int N = 1 << 20;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) * 0.001f;
    h_in[N / 2] = 999.0f;  // 确保最大值在 N/2

    float *d_in; cudaMalloc(&d_in, N * sizeof(float));
    int *d_out; cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(int));

    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);
    argmax_kernel<<<blocks, threads>>>(d_in, d_out, N);

    int gpu_idx;
    cudaMemcpy(&gpu_idx, d_out, sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU argmax idx = %d (expected %d) %s\n",
           gpu_idx, N / 2, gpu_idx == N / 2 ? "PASS" : "FAIL");

    free(h_in); cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 观察

```bash
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread ./argmax
```

### 优化方向

1. **消除 atomicMax**：改用两次 kernel launch（第一次输出每 block 的 argmax，第二次汇总）
2. **grid-stride 负载均衡**：每个线程处理的元素数相近
3. **初始值选择**：用 `-FLT_MAX` 而非 `0.0f`（数组可能有负数）

## 6. 复杂度分析

- **时间复杂度**：`O(N)`，每个元素被访问一次。
- **空间复杂度**：`O(N)` 输入 + `O(blocks)` 临时输出。
- **算术强度**：~1 comparison / 4 Bytes，**memory-bound**。

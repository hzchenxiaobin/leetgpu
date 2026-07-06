# LeetGPU Reduction 题解

## 1. 题目概述

- **标题 / 题号**：Reduction
- **链接**：https://leetgpu.com/challenges/reduction
- **难度**：中等
- **标签**：CUDA、Parallel Reduction、Shared Memory、Warp Shuffle、Bank Conflict

给定长度为 `N` 的浮点数组 `input`，计算所有元素的总和：`sum = input[0] + input[1] + ... + input[N-1]`。

约束：`1 ≤ N ≤ 100,000,000`，数组元素范围 `[-1000.0, 1000.0]`，结果不会溢出 float。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
float sum = 0.0f;
for (int i = 0; i < n; ++i) sum += input[i];
```

### 朴素 GPU 方法（O(N²)）

每个线程独立计算前缀和——极度浪费。

## 3. GPU 设计

### 3.1 并行化策略

两阶段归约（与 Week 2 Day 1 的 Warp Reduce 一致）：

1. **线程级**：grid-stride loop 做线程级累加
2. **Warp 级**：`__shfl_down_sync` butterfly 归约
3. **Block 级**：warp 部分和写入 Shared Memory，Warp 0 做最终归约
4. **跨 Block**：第二次 kernel launch 汇总

![Reduction 两级归约流程](images/reduction_overview.svg)

**流程说明**：

- **第一次 kernel**：启动足够多的 block 覆盖输入数组。每个 block 内部先让每个线程通过 grid-stride loop 累加自己负责的一段 input，得到线程局部和；接着在 warp 内用 `__shfl_down_sync` 把 32 个线程的局部和归约成 1 个 warp 和；每个 warp 的 lane 0 把 warp 和写入 `warpSums[wid]`；最后由 warp 0 读取所有 `warpSums` 再做一次 shuffle 归约，得到该 block 的部分和，写入 `d_temp[blockIdx.x]`。
- **第二次 kernel**：只启动 1 个 block，把 `d_temp` 里的 block 部分和再用同样的逻辑归约成全局总和，写入 `d_out`。

### 3.2 Grid-Stride Loop 详解

**Grid-stride loop** 是 CUDA 中一种常见的线程遍历模式，让一个线程处理数组中的多个元素，而不是只处理一个。

典型写法：

```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;

for (int i = tid; i < N; i += stride) {
    sum += input[i];
}
```

![Grid-Stride Loop 工作原理](images/reduction_grid_stride.svg)

**关键概念**：

- `tid`：线程在 grid 中的全局索引（从 0 开始）
- `stride = gridDim.x * blockDim.x`：grid 中线程总数，也就是每次前进的步长
- 线程 `tid` 负责的元素下标为：`tid, tid + stride, tid + 2*stride, ...`

**为什么用 grid-stride loop？**

1. **处理 N 远大于线程数的情况**：即使只启动几百个线程，也能处理上亿个元素
2. **负载均衡**：每个线程负责的元素数量大致相同
3. **合并访问（coalesced access）**：相邻线程访问相邻地址，内存访问效率高
4. **可扩展性**：增加 block 数量就能覆盖更大的 N，无需修改 kernel 逻辑

**在 Reduction 中的作用**：

每个线程先通过 grid-stride loop 把自己负责的那一段 input 元素累加起来，得到一个"线程局部和"。这样后续 warp/block 级归约只需要处理 `blockDim.x` 个局部和，而不是直接处理 N 个元素。

### 3.3 单个 Block 内部执行过程

![单个 Block 内部归约过程](images/reduction_block_internal.svg)

上图以 256 线程（8 warps）为例展示了单个 block 的 4 个执行步骤：

1. **Thread-level 累加**：256 个线程各自负责 input 中不相交的一段元素，使用 grid-stride loop 求出线程局部和。
2. **Warp-level 归约**：每个 warp 内部 32 个线程通过 `__shfl_down_sync` 进行 butterfly 归约，最终每个 warp 得到 1 个和（`sum0 ~ sum7`）。
3. **写入 Shared Memory**：每个 warp 的 lane 0 把自己 warp 的和写入 `warpSums[wid]`。由于不同 warp 的 lane 0 写入不同索引 `wid`，对应不同的 bank，因此**无 bank conflict**。
4. **Block-level 最终归约**：warp 0 的 32 个线程读取 `warpSums[0..7]`（不足 32 的 lane 补 0），再做一次 warp shuffle 归约，lane 0 得到 block 部分和并写入 `output[blockIdx.x]`。

### 3.4 `__shfl_down_sync` 详解

`__shfl_down_sync` 是 CUDA 提供的 **warp shuffle** 原语，它让同一个 warp 内的线程直接互相读取寄存器数据，**不经过 Shared Memory**，因此非常适合 warp 级归约。

#### 函数原型

```cuda
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
```

参数说明：

- `mask`：参与 shuffle 的线程掩码。通常写 `0xFFFFFFFF`，表示 warp 内 32 个线程全部参与。
- `var`：当前 lane 要传递下去的变量。
- `delta`：目标 lane 相对于当前 lane 的偏移量。`lane i` 会从 `lane i + delta` 读取 `var`；若 `i + delta` 越界，则返回 `lane i` 自己的值。
- `width`：参与 shuffle 的线程数，默认 32。

#### Butterfly 归约过程

在 `warpReduceSum` 中，我们通过不断折半偏移量来完成归约：

```cuda
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

#### `#pragma unroll` 的作用

```cuda
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

`#pragma unroll` 是 NVCC 编译器指令，指示编译器对紧随其后的 `for` 循环进行**循环展开（Loop Unrolling）**。本例中循环只有固定的 5 次迭代（`offset = 16, 8, 4, 2, 1`），展开后编译器可以直接生成 5 条连续的 `__shfl_down_sync` 指令，带来以下好处：

1. **消除循环控制开销**：不再需要每次迭代中的条件判断（`offset > 0`）和循环变量更新（`offset >>= 1`），减少分支指令。
2. **提升指令级并行（ILP）**：展开后编译器能更自由地调度指令，让多条 `__shfl_down_sync` 之间的空闲周期被其他指令填充，提高流水线利用率。
3. **编译期已知迭代次数**：由于 `offset` 从编译期常量 `16` 开始，`#pragma unroll` 让编译器在编译阶段就确定展开次数，生成最优的机器码。

> **注意**：当循环次数不确定或非常大时，完全展开可能导致指令缓存压力增大（instruction cache miss），反而降低性能。本例 5 次迭代是最理想的场景。

以 8 线程简化示例，初始值为 `[a0, a1, a2, a3, a4, a5, a6, a7]`：

1. `offset = 4`：`lane 0` 读取 `lane 4`，`lane 1` 读取 `lane 5`……得到 `[a0+a4, a1+a5, a2+a6, a3+a7, ..., ...]`
2. `offset = 2`：`lane 0` 读取 `lane 2`……得到 `[a0+a4+a2+a6, a1+a5+a3+a7, ..., ...]`
3. `offset = 1`：`lane 0` 读取 `lane 1`，最终 `lane 0` 持有 `a0+a1+...+a7`

实际 warp 有 32 线程，循环从 `offset=16` 开始，经过 16→8→4→2→1 共 5 步，`lane 0` 得到 32 个线程局部和的总和。

#### 为什么用 Shuffle 而不是 Shared Memory？

| 方式 | 是否需要 Shared Memory | 是否需要 `__syncthreads()` | Bank Conflict | 延迟 |
|---|---|---|---|---|
| Shared Memory 归约 | 是 | 是 | 可能有（取决于访问模式） | 较高 |
| `__shfl_down_sync` | 否 | 否 | 无 | 低 |

使用 `__shfl_down_sync` 的优势：

1. **避免 bank conflict**：warp shuffle 在寄存器级别交换数据，不访问 Shared Memory。
2. **无需同步**：同一个 warp 内的线程天然同步执行（SIMT），不需要 `__syncthreads()`。
3. **更少的内存占用**：不需要为 warp 级归约分配 Shared Memory，节省 `32 * sizeof(float)` 甚至更多。

#### 注意事项

- `__shfl_down_sync` 要求目标线程必须处于**活跃（active）**状态。在现代 GPU 上，只要 warp 内所有线程都执行到同一条 shuffle 指令（即 `0xFFFFFFFF` 掩码且没有分支发散），就是安全的。
- 该原语只能用于**同一个 warp 内部**的线程通信。跨 warp 通信仍然需要 Shared Memory 或全局内存。
- 从 Maxwell 架构（sm_50）开始支持 warp shuffle；本题的优化实现通常面向 sm_70 及以上。

### 3.5 Bank Conflict 分析

关键观察：Warp Shuffle 归约**不经过 Shared Memory**，因此 warp 级归约**无 bank conflict**。

Bank conflict 发生在 **Step 3**：warp 的 lane 0 写入 `warpSums[wid]`。由于不同 warp 的 lane 0 写入不同 bank（wid 0~31 对应 bank 0~31），这里是**无 conflict** 的。

但如果用 Shared Memory 做归约（而非 Shuffle），conflict 会很严重。

### 3.6 为什么 `__shared__ float warpSums[32];` 是 32？

代码中 block size 是 256 线程，即 8 个 warp，但 `warpSums` 却开成了 32。原因如下：

1. **兼容最大 block size**：CUDA 一个 block 最多 1024 线程，即 32 个 warp。`warpSums[32]` 能通吃所有合法 block size，无需动态计算共享内存。
2. **适配 warp 0 最后的 32-lane shuffle 归约**：在 block 级最终归约中，warp 0 的 32 个 lane 都会读取 `warpSums[lane]`，其中 `lane < numWarps` 的线程读实际值，`lane >= numWarps` 的线程读 `0.0f` 作为填充。`warpReduceSum` 内部固定对 32 个 lane 做 `__shfl_down_sync`，因此数组至少要覆盖 `lane = 0~31`，否则会越界。
3. **内存开销可忽略**：`32 * sizeof(float) = 128 bytes / block`，代价极小。

一句话总结：`warpSums[32]` 是为了兼容最大 block size，并保证 warp 0 的 32 个 lane 都能安全参与最后的 shuffle 归约。

## 4. Kernel 实现

```cuda
// reduction.cu —— 并行归约（Warp Shuffle + 两级归约）
// 编译命令: nvcc -o reduction reduction.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void reduction_kernel(const float* input, float* output, int N) {
    __shared__ float warpSums[32];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float sum = 0.0f;
    for (int i = tid; i < N; i += gridDim.x * blockDim.x)
        sum += input[i];

    sum = warpReduceSum(sum);

    if (lane == 0) warpSums[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        int numWarps = (blockDim.x + 31) / 32;
        sum = (lane < numWarps) ? warpSums[lane] : 0.0f;
        sum = warpReduceSum(sum);
        if (lane == 0) output[blockIdx.x] = sum;
    }
}

int main() {
    const int N = 1 << 22;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) * 0.001f;

    float *d_in, *d_temp, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_temp, 1024 * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = min((N + threads - 1) / threads, 1024);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduction_kernel<<<blocks, threads>>>(d_in, d_temp, N);
    reduction_kernel<<<1, 256>>>(d_temp, d_out, blocks);

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) cpu_sum += h_in[i];

    printf("GPU=%.4f CPU=%.4f diff=%.6f %s\n",
           gpu_sum, (float)cpu_sum, fabs(gpu_sum - (float)cpu_sum),
           fabs(gpu_sum - (float)cpu_sum) < 1e-3 ? "PASS" : "FAIL");
    printf("Time: %.3f ms (%.1f GB/s)\n", ms, N * sizeof(float) / (ms * 1e6));

    free(h_in); cudaFree(d_in); cudaFree(d_temp); cudaFree(d_out);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 观察 bank conflict

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__occupancy.avg.pct_of_peak_sustained_elapsed ./reduction
```

### 优化方向

1. **用 Warp Shuffle 替代 Shared Memory 归约**（已实现）：消除 bank conflict
2. **grid-stride loop**：处理 N >> total_threads 的情况
3. **第二次 kernel 汇总**：避免 atomicAdd 的竞争

## 6. 复杂度分析

- **时间复杂度**：`O(N)`，每个元素被访问一次。
- **空间复杂度**：`O(N)` 输入 + `O(blocks)` 临时。
- **算术强度**：1 FLOP / 4 Bytes = 0.25 FLOP/Byte，**memory-bound**。

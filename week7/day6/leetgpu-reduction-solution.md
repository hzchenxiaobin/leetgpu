# LeetGPU Reduction 题解

## 1. 题目概述

- **标题 / 题号**：Reduction（#4，medium）
- **链接**：https://leetgpu.com/challenges/reduction
- **难度**：中等
- **标签**：CUDA、warp shuffle、归约、memory-bound、`__shfl_down_sync`

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，计算所有元素的和 `sum = input[0] + input[1] + ... + input[N-1]`。

**示例**：

```text
输入：[1.0, 2.0, 3.0, 4.0, 5.0]
输出：15.0
```

**约束**：`1 ≤ N ≤ 10,000,000`；性能测试取大数组。

> 💡 这道题是 **warp shuffle 归约的经典练习**——`__shfl_down_sync` 把 warp 内 32 个 lane 的值逐级归约到 lane 0。与 [Week7 Day6 全链路 Profiling](../../aiinfra/week7/day6/README.md) 的关联在于：Reduction 是 profiling 中最常分析的 memory-bound kernel，LayerNorm（求 mean/var）和 Softmax（求 max/sum）内部都包含 reduction。理解它的性能特征是分析这些 kernel 瓶颈的基础。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
float sum = 0;
for (int i = 0; i < N; i++) sum += input[i];
```

### 朴素 GPU（共享内存归约）

```cuda
__global__ void naive_reduce(const float* input, float* output, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
// 需要两遍：第一遍 block 归约，第二遍对 block 结果再归约
```

**瓶颈**：共享内存归约有大量 `__syncthreads`，且每步一半线程空闲。warp 内最后 5 步可以用 `__shfl_down_sync` 替代，无需 sync。

## 3. GPU 设计

### 3.1 并行化策略：两阶段归约

1. **阶段一**：每个 block 归约 `BLOCK_SIZE` 个元素 → 输出 `numBlocks` 个部分和
2. **阶段二**：对 `numBlocks` 个部分和再做一次归约 → 最终结果

### 3.2 Warp 级归约：`__shfl_down_sync`

```cuda
// warp 内 32 lane 归约，5 步完成，无需 __syncthreads
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// lane 0 持有 warp 内 32 个值的和
```

### 3.3 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| global memory | ✓ | input 读、output 写 |
| shared memory | ✓ | block 内 warp 间归约 |
| register | ✓ | warp shuffle 直接在寄存器交换 |

## 4. Kernel 实现

### 4.1 提交版代码

```cuda
// reduction.cu —— Warp shuffle 两阶段归约
// 编译命令: nvcc -O3 -arch=sm_80 reduction.cu -o reduction

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduce_kernel(const float* input, float* output, int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // 每线程加载一个元素
    float val = (gid < N) ? input[gid] : 0.0f;

    // warp 内归约
    val = warp_reduce(val);

    // warp 的 lane 0 写入 shared memory
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // 第一个 warp 对 warp_sums 做归约
    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        if (lane == 0) output[blockIdx.x] = val;
    }
}

// 最终归约 kernel（numBlocks 很小时用）
__global__ void final_reduce(const float* input, float* output, int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int tid = threadIdx.x;
    float val = (tid < N) ? input[tid] : 0.0f;
    val = warp_reduce(val);
    if (tid % WARP_SIZE == 0) warp_sums[tid / WARP_SIZE] = val;
    __syncthreads();
    if (tid < WARP_SIZE) {
        val = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_sums[tid] : 0.0f;
        val = warp_reduce(val);
        if (tid == 0) output[0] = val;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blockSize = BLOCK_SIZE;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 两阶段：block 归约 → final 归约
    reduce_kernel<<<gridSize, blockSize>>>(input, output, N);
    final_reduce<<<1, blockSize>>>(output, output, gridSize);
}
```

### 4.2 ncu 分析要点

```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__sass_thread_inst_executed_op_fadd_pred_on.sum \
    --kernel-name regex:"reduce" \
    ./reduction
```

**预期指标**：

| 指标 | 预期值 | 含义 |
|------|--------|------|
| `dram__throughput` | > 80% | memory-bound（读 N 个 float） |
| `sm__throughput` | < 20% | 算力利用率低（只有加法） |
| `gpu__time_duration` | ~0.1ms (N=10M) | 带宽受限 |

> 💡 Reduction 是典型的 memory-bound kernel——算术强度 = 1 FLOP / 4B = 0.25 FLOP/B，远低于 roofline 拐点。

## 5. 性能分析

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 reduction.cu -o reduction
./reduction 10000000
```

### 5.2 朴素 vs Warp Shuffle 对比

| 版本 | sync 次数 | 空闲线程 | 带宽利用 |
|------|----------|---------|---------|
| 朴素（全 shared memory） | 8 次 | 每步减半 | ~60% |
| Warp shuffle | 1 次（仅 warp 间） | 仅最后 5 步减半 | ~85% |

### 5.3 与 Profiling 的关联

| Reduction 在哪里 | 关联 kernel | Profiling 指标 |
|-----------------|------------|---------------|
| LayerNorm 的 mean/var | `layernorm_kernel` | DRAM throughput 高 |
| Softmax 的 max/sum | `softmax_kernel` | DRAM throughput 高 |
| Attention 的 score sum | `flash_attention_kernel` | 内部 online softmax |
| 独立 reduction | `reduce_kernel` | 典型 memory-bound |

> 💡 理解 reduction 的 memory-bound 特征，就能理解为什么 LayerNorm/Softmax 在 ncu 中表现为 DRAM 高、SM 低——它们的内部都有 reduction 操作。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`，两阶段归约 |
| **空间复杂度** | `O(N)` 输入 + `O(numBlocks)` 中间 + `O(BLOCK)` shared |
| **算术强度** | `0.25 FLOP/B`（1 次加法 / 4B 读取） |
| **瓶颈类型** | **memory-bound**：受 HBM 读带宽限制 |
| **kernel 启动数** | 2 次（block 归约 + final 归约） |

> 💡 **一句话总结**：Reduction 是 warp shuffle 归约的经典练习——`__shfl_down_sync` 在寄存器内交换，比 shared memory 归约减少 sync。它是 memory-bound 的代表（算术强度 0.25 FLOP/B），也是 LayerNorm/Softmax 内部操作的基础，理解它的性能特征是全链路 profiling 的关键。

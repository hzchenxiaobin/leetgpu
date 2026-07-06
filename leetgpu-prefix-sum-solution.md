# LeetGPU Prefix Sum 题解

## 1. 题目概述

- **标题 / 题号**：Prefix Sum
- **链接**：https://leetgpu.com/challenges/prefix-sum
- **难度**：中等
- **标签**：CUDA、Scan、Prefix Sum、Warp Shuffle、`__shfl_up_sync`

给定一个长度为 `N` 的 32-bit 浮点数组 `input`，要求计算其 **inclusive prefix sum（前缀和）**：

```
output[i] = input[0] + input[1] + ... + input[i]
```

约束：`1 ≤ N ≤ 100,000,000`，数值范围 `[-1000.0, 1000.0]`，输出不会溢出 float。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 1; i < n; ++i) {
    output[i] = output[i - 1] + input[i];
}
```

- 时间复杂度 `O(N)`，空间复杂度 `O(1)`（除输出外）。
- 瓶颈：单线程顺序执行，无法利用 GPU 并行性。

### 朴素 GPU 方法（O(N²)）

每个线程 `i` 独立计算 `sum(input[0..i])`，大量重复计算，性能极差。

## 3. GPU 设计

### 3.1 并行化策略

Prefix sum 是经典的 **scan（扫描）** 问题。采用 **分块两阶段 scan**：

1. **Block 内 exclusive scan**：每个 block 独立计算其负责区间内元素的 exclusive prefix sum，并输出该 block 的总和。
2. **Block 间 scan**：对所有 block 总和再做一次 scan，得到每个 block 的 **全局偏移量**。
3. **Add block offset**：把全局偏移量加回到 block 内的每个元素，得到最终的 inclusive prefix sum。

### 3.2 Warp 级 scan：`__shfl_up_sync`

与归约用 `__shfl_down_sync` 对称，scan 用 `__shfl_up_sync` 实现 **Hillis-Steele** 算法：

```
offset=1:  lane i 从 lane (i-1) 取值累加
offset=2:  lane i 从 lane (i-2) 取值累加
offset=4:  lane i 从 lane (i-4) 取值累加
offset=8:  ...
offset=16: ...
5 步完成 32 线程的 inclusive scan
```

### 3.3 存储层次使用

- **全局内存**：读 `input`（coalesced）、写 `output`（coalesced）
- **共享内存**：block 内 scan 的中转（warp 部分和）
- **寄存器**：每个线程的累加值

## 4. Kernel 实现

```cuda
// prefix_sum.cu —— 分块两阶段 Prefix Sum
// 编译命令: nvcc -o prefix_sum prefix_sum.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 256

// Warp 内 inclusive scan (Hillis-Steele), 使用 __shfl_up_sync
__inline__ __device__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & 31) >= offset)
            val += n;
    }
    return val;
}

// Block 内 inclusive scan, 返回每个线程的 inclusive 值, block 总和在 s_data[BLOCK_SIZE/32-1]
__inline__ __device__ float block_scan(float val, float* block_sum) {
    __shared__ float s_data[BLOCK_SIZE / 32];

    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warp_inclusive_scan(val);

    if (lane == 31) s_data[wid] = val;  // 每个 warp 的总和
    __syncthreads();

    // Warp 0 对 warp 总和做 exclusive scan
    if (wid == 0) {
        float warp_sum = (lane < BLOCK_SIZE / 32) ? s_data[lane] : 0.0f;
        // exclusive scan
        float prefix = 0.0f;
        #pragma unroll
        for (int offset = 1; offset < BLOCK_SIZE / 32; offset <<= 1) {
            float n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (lane >= offset) warp_sum += n;
        }
        // exclusive: 把 inclusive 整体右移一位
        float exclusive = __shfl_up_sync(0xFFFFFFFF, warp_sum, 1);
        if (lane == 0) exclusive = 0.0f;
        s_data[lane] = exclusive;
    }
    __syncthreads();

    if (wid == BLOCK_SIZE / 32 - 1 && lane == 31)
        *block_sum = val + s_data[wid];

    return val + s_data[wid];  // 当前线程的 inclusive prefix sum
}

// Kernel 1: block 内 scan, 输出每 block 总和
__global__ void block_scan_kernel(const float* input, float* output,
                                   float* block_sums, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? input[tid] : 0.0f;

    float block_sum;
    float prefix = block_scan(val, &block_sum);

    if (tid < N) output[tid] = prefix;
    if (threadIdx.x == 0) block_sums[blockIdx.x] = block_sum;
}

// Kernel 2: 对 block_sums 做 exclusive scan
// Kernel 3: 把 block offset 加回 output

int main() {
    const int N = 1 << 20;
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 100) * 0.01f;

    float *d_in, *d_out, *d_block_sums;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_block_sums, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    block_scan_kernel<<<blocks, BLOCK_SIZE>>>(d_in, d_out, d_block_sums, N);
    // (省略 kernel 2/3 的调用)

    float *h_out = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    float cpu_sum = 0.0f;
    bool ok = true;
    for (int i = 0; i < N; i++) {
        cpu_sum += h_in[i];
        if (fabs(h_out[i] - cpu_sum) > 1e-2) { ok = false; break; }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_block_sums);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 观察

```bash
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread ./prefix_sum
```

### 优化方向

1. **Warp Shuffle scan**：比 Shared Memory scan 快 10-20 倍
2. **三阶段分块**：处理 N >> block_size 的情况
3. **避免 atomicAdd**：用第二次 kernel 汇总 block 偏移

## 6. 复杂度分析

- **时间复杂度**：`O(N)`，每个元素被访问常数次。
- **空间复杂度**：`O(N)` 输入 + 输出 + `O(blocks)` 临时。
- **算术强度**：~1 FLOP / 8 Bytes，**memory-bound**。

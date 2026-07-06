# LeetGPU ReLU 题解

## 1. 题目概述

- **标题 / 题号**：ReLU
- **链接**：https://leetgpu.com/challenges/relu
- **难度**：简单
- **标签**：CUDA、Element-wise、Activation、Register Pressure

给定长度为 `N` 的浮点数组 `input`，对其逐元素应用 ReLU 激活函数：`output[i] = max(0, input[i])`。

约束：`1 ≤ N ≤ 10,000,000`，数组元素范围 `[-1000.0, 1000.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < n; ++i) {
    output[i] = std::max(0.0f, input[i]);
}
```

### 朴素 GPU 方法

```cuda
__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}
```

## 3. GPU 设计

### 3.1 并行化策略

一维并行，每个线程处理一个元素。kernel 逻辑极简（单条 `fmaxf`），性能完全取决于内存带宽。

### 3.2 与 Occupancy 的关系

本题代码寄存器用量极低（~8 个），因此：
- 不同 block size 下理论 occupancy 差异不大（都接近 100%）
- 但 **achieved occupancy** 可能因 block 调度策略不同而有差异
- 适合用 ncu 观察 `sm__occupancy.avg.pct_of_peak_sustained_elapsed` 随 block size 的变化

## 4. Kernel 实现

```cuda
// relu.cu —— ReLU 激活函数
// 编译命令: nvcc -o relu relu.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

int main() {
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_in[i] = (float)(rand() % 2000 - 1000) * 0.001f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // 测试不同 block size
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int bs : block_sizes) {
        int grid = (N + bs - 1) / bs;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        relu_kernel<<<grid, bs>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("block=%4d  time=%.3f ms  bw=%.1f GB/s\n",
               bs, ms, 2.0f * N * sizeof(float) / (ms * 1e6));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != fmaxf(h_in[i], 0.0f)) { ok = false; break; }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

## 5. 性能分析与优化

### ncu 观察 Occupancy

```bash
ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_elapsed,\
launch__registers_per_thread,\
dram__throughput.avg.pct_of_peak_sustained_elapsed ./relu
```

### block size 对 Occupancy 的影响

| block size | 寄存器/线程 | 理论 Occupancy | Achieved Occupancy | 带宽利用率 |
|-----------|-----------|--------------|-------------------|-----------|
| 32 | | | | |
| 128 | | | | |
| 256 | | | | |
| 512 | | | | |
| 1024 | | | | |

> 观察：由于 ReLU 寄存器用量极低，所有配置下理论 occupancy 都接近 100%。差异主要来自 block 调度粒度。

## 6. 复杂度分析

- **时间复杂度**：`O(N)`，每个元素 O(1)。
- **空间复杂度**：`O(N)`。
- **算术强度**：1 FLOP / 8 Bytes = 0.125 FLOP/Byte，**memory-bound**。

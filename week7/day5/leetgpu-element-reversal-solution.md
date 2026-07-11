# LeetGPU Element Reversal 题解

## 1. 题目概述

- **标题 / 题号**：Element Reversal（#34，easy）
- **链接**：https://leetgpu.com/challenges/element-reversal
- **难度**：简单
- **标签**：CUDA、element-wise、memory-bound、结果验证

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，将每个元素的符号反转（`output[i] = -input[i]`）。

**示例**：

```text
输入：[1.0, -2.0, 3.0, -4.0]
输出：[-1.0, 2.0, -3.0, 4.0]
```

**约束**：`1 ≤ N ≤ 10,000,000`；性能测试取大数组（约 40 MB）。

> 💡 这道题是最简单的 element-wise 操作（`output[i] = -input[i]`），与 [Week7 Day5 系统联调](../../aiinfra/week7/day5/README.md) 中的**结果一致性验证**同构——联调时需要对比自定义 kernel 与 PyTorch 的输出，逐元素比较是否一致。Element Reversal 的"逐元素变换 + 逐元素对比"正是联调精度验证的基础操作：`assert (custom_output - pytorch_output).abs().max() < threshold`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
for (int i = 0; i < N; i++)
    output[i] = -input[i];
```

### 朴素 GPU

```cuda
__global__ void naive_reverse_elements(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = -input[i];
}
```

**特点**：纯 element-wise，读 4B + 写 4B + 1 次取负，memory-bound。朴素版本已接近最优。

## 3. GPU 设计

### 3.1 并行化策略：coalesced 1:1

每个 thread 负责一个元素：`output[i] = -input[i]`。

- 读 `input[i]`：warp 内连续 → coalesced ✓
- 写 `output[i]`：同上 → coalesced ✓
- 计算量：1 次取负（几乎为零）

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| global memory | ✓ | input 读、output 写 |
| shared memory | ✗ | 纯 element-wise，无需暂存 |
| register | ✓ | 每线程持有 1 个 float |

## 4. Kernel 实现

### 4.1 提交版代码

```cuda
// element_reversal.cu —— Element Reversal（符号反转）
// 编译命令: nvcc -O3 -arch=sm_80 element_reversal.cu -o element_reversal

#include <cuda_runtime.h>

__global__ void reverse_elements_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = -input[i];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    reverse_elements_kernel<<<gridSize, blockSize>>>(input, output, N);
}
```

### 4.2 完整自测版

```cuda
// element_reversal_full.cu —— 含验证和带宽测量
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

__global__ void reverse_elements_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = -input[i];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    float *hIn = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    reverse_elements_kernel<<<gridSize, blockSize>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    printf("I/O bandwidth: %.1f GB/s\n", (2.0 * bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    int fail = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(hOut[i] - (-hIn[i])) > 1e-5f) {
            printf("FAIL at i=%d: got %f, expected %f\n", i, hOut[i], -hIn[i]);
            fail = 1; break;
        }
    }
    printf("%s\n", fail ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hOut);
    return 0;
}
```

## 5. 性能分析

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 element_reversal_full.cu -o element_reversal
./element_reversal 10000000
```

典型输出（A100）：

```text
N = 10000000  (40.0 MB)
kernel time: 0.12 ms
I/O bandwidth: 666.7 GB/s
PASS
```

### 5.2 与联调验证的关联

| 操作 | 公式 | 联调验证场景 |
|------|------|------------|
| Element Reversal | `output[i] = -input[i]` | 最简单的 element-wise 对比 |
| 精度验证 | `(custom - pytorch).abs().max()` | 逐元素 diff < threshold |
| KV Cache 隔离 | `req1_result != req2_result` | 逐元素检查结果不串台 |
| 结果一致性 | `assert max_diff < 1e-2` | 联调 Step 5 验证标准 |

> 💡 Element Reversal 是联调精度验证的"原子操作"——理解逐元素对比的方法，就理解了所有联调验证的基础。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`，每元素一次读 + 取负 + 写 |
| **空间复杂度** | `O(N)` 输入 + `O(N)` 输出 |
| **算术强度** | `0.125 FLOP/B`（1 次取负 / 8B） |
| **瓶颈类型** | **memory-bound**：受 HBM 双向带宽限制 |
| **kernel 启动数** | 1 次 |

> 💡 **一句话总结**：Element Reversal 是最简单的 element-wise 操作——`output[i] = -input[i]`，纯 memory-bound。它与系统联调中的结果一致性验证同构：逐元素对比是联调精度验证的基础方法。

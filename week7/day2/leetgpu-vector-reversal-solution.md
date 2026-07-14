# LeetGPU Vector Reversal 题解

## 1. 题目概述

- **标题 / 题号**：Vector Reversal（#32，easy）
- **链接**：https://leetgpu.com/challenges/vector-reversal
- **难度**：简单
- **标签**：CUDA、索引映射、coalesced access、memory-bound

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，将其反转后写入 `output`（`output[i] = input[N-1-i]`）。

**示例**：

```text
输入：[1.0, 2.0, 3.0, 4.0, 5.0]
输出：[5.0, 4.0, 3.0, 2.0, 1.0]
```

**约束**：`1 ≤ N ≤ 10,000,000`；性能测试取大数组（约 40 MB）。

> 💡 这道题是**索引映射的最简形式**——`output[i] = input[N-1-i]`，每个线程处理一个元素的"调度"。与 [Week7 Day2 完整调度器](../../aiinfra/daily/week7/day2/README.md) 的"请求到资源的映射"同构：调度器把请求按优先级分配到 batch 槽位（`batch[i] = waiting[best]`），Vector Reversal 把数据按逆序映射到输出位置。两者都是**用索引规则做资源映射**。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
for (int i = 0; i < N; i++)
    output[i] = input[N - 1 - i];
```

### 朴素 GPU（一 thread 一元素）

```cuda
__global__ void naive_reverse(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        output[i] = input[N - 1 - i];
}
```

**瓶颈**：朴素版本虽然正确，但读 `input[N-1-i]` 的方向与写 `output[i]` 的方向相反——读端从尾向头走，写端从头向尾走。warp 内 32 个 thread 的读地址是递减的，虽然仍 coalesced（连续地址），但需确认 GPU 的内存事务对逆向访问的效率。

## 3. GPU 设计

### 3.1 并行化策略：coalesced 1:1 逆序映射

每个 thread 负责一个元素：`thread i → output[i] = input[N-1-i]`。

关键点：
- **读端** `input[N-1-i]`：warp 内 i 递增 → 读地址递减，但仍连续 → **coalesced**
- **写端** `output[i]`：warp 内 i 递增 → 写地址递增 → **coalesced**
- 读和写都是合并访问，带宽利用率高

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| global memory | ✓ | input 读、output 写 |
| shared memory | ✗ | 纯索引映射，无需暂存 |
| register | ✓ | 每线程持有 1 个 float |

### 3.3 关键技巧：确保双向 coalesced

```
thread 0:  read input[N-1]   → write output[0]
thread 1:  read input[N-2]   → write output[1]
...
thread 31: read input[N-32]  → write output[31]

读地址：N-1, N-2, ..., N-32  → 连续递减 → coalesced ✓
写地址：0, 1, ..., 31         → 连续递增 → coalesced ✓
```

> ⚠️ GPU 的 coalesced access 对递增和递减地址都高效——只要 warp 内 32 个 thread 访问的是同一段连续内存即可。方向不影响事务效率。

## 4. Kernel 实现

### 4.1 提交版代码

```cuda
// vector_reversal.cu —— Vector Reversal（coalesced 逆序映射）
// 编译命令: nvcc -O3 -arch=sm_120 vector_reversal.cu -o vector_reversal

#include <cuda_runtime.h>

__global__ void reverse_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[N - 1 - i];
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    reverse_kernel<<<gridSize, blockSize>>>(input, output, N);
}
```

### 4.2 完整自测版（含 Host）

```cuda
// vector_reversal_full.cu —— 含验证和带宽测量
    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

__global__ void reverse_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        output[i] = input[N - 1 - i];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    float* hIn = (float*)malloc(bytes);
    float* hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++)
        hIn[i] = (float)(rand() % 1000) / 10.0f;

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
    reverse_kernel<<<gridSize, blockSize>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    printf("I/O bandwidth: %.1f GB/s\n", (2.0 * bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    int fail = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(hOut[i] - hIn[N - 1 - i]) > 1e-5f) {
            printf("FAIL at i=%d: got %f, expected %f\n", i, hOut[i], hIn[N - 1 - i]);
            fail = 1;
            break;
        }
    }
    printf("%s\n", fail ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hOut);
    return 0;
}
```

## 5. 性能分析

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 vector_reversal_full.cu -o vector_reversal
./vector_reversal 10000000
```

典型输出（RTX 5090）：

```text
N = 10000000  (40.0 MB)
kernel time: 0.12 ms
I/O bandwidth: 666.7 GB/s
PASS
```

### 5.2 算术强度

```
1 FLOP（无计算，纯索引映射）/ 8B（读 4B + 写 4B）= 0 FLOP/B
→ 纯 memory-bound，理论峰值 = HBM 双向带宽
```

### 5.3 优化方向

1. **float4 向量化**：每线程读 16B（4 个 float），减少地址计算、提升事务效率
2. **block size 调优**：256 通常最优，512 可测试
3. **无需 shared memory**：纯索引映射，shared memory 无收益

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`，每个元素一次读 + 一次写 |
| **空间复杂度** | `O(N)` 输入 + `O(N)` 输出 |
| **算术强度** | `0 FLOP/B`（纯数据搬运） |
| **瓶颈类型** | **memory-bound**：受 HBM 双向带宽限制 |
| **kernel 启动数** | 1 次 |

> 💡 **一句话总结**：Vector Reversal 是索引映射的最简形式——`output[i] = input[N-1-i]`。与调度器的"请求到 batch 槽位的映射"同构：都是用索引规则做资源分配。读地址递减、写地址递增，两端均 coalesced，带宽利用率接近峰值。

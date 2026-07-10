# LeetGPU Max Subarray Sum 题解

## 1. 题目概述

- **标题 / 题号**：Max Subarray Sum（#51，medium）
- **链接**：https://leetgpu.com/challenges/max-subarray-sum
- **难度**：中等
- **标签**：CUDA、滑动窗口、prefix sum、reduction、memory-bound

**题意**：给定长度为 `N` 的 `int32` 数组 `input` 和窗口大小 `window_size`，求所有长度恰好为 `window_size` 的连续子数组的**最大和**。

**示例**：

```text
input = [1, 2, 4, 2, 3], window_size = 2
窗口: [1,2]=3, [2,4]=6, [4,2]=6, [2,3]=5
输出: 6
```

**约束**：`1 ≤ N ≤ 50000`，`-10 ≤ input[i] ≤ 10`，`1 ≤ window_size ≤ N`；性能测试取 `N=50000, window_size=25000`。

> 💡 这道题的**滑动窗口**思想与 [Week6 Day2](../../aiinfra/week6/day2/README.md) Continuous Batching 的 iteration-level 调度同构——窗口在数据上滑动，每步加入新元素、移出旧元素，正是 Continuous Batching "每轮加入新请求、移出完成请求"的微缩版。

## 2. CPU 基线

```cpp
// 暴力：对每个窗口求和 → O(N × window_size)
int max_sum = INT_MIN;
for (int i = 0; i <= N - window_size; i++) {
    int sum = 0;
    for (int j = i; j < i + window_size; j++) sum += input[j];
    max_sum = max(max_sum, sum);
}
```

`O(N × W)`，`N=50000, W=25000` 时约 12.5 亿次加法，太慢。

## 3. GPU 设计

### 3.1 优化思路：prefix sum + reduction

1. 算 prefix sum `prefix[i] = input[0] + ... + input[i-1]`
2. 每个窗口和 `window_sum[i] = prefix[i+W] - prefix[i]`
3. 对所有 `window_sum` 求最大值（block reduce + atomic max）

### 3.2 并行化策略

| 步骤 | 方法 | 复杂度 |
|------|------|--------|
| prefix sum | Week2 Day1 的 warp scan + 三阶段 | O(N) |
| 窗口求和 | 每线程算一个 `prefix[i+W] - prefix[i]` | O(N-W) |
| 求最大 | block reduce + atomic max | O(N-W) |

## 4. Kernel 实现

```cuda
// max_subarray_sum.cu —— 滑动窗口最大和（prefix sum + reduction）
// 编译命令: nvcc -O3 -arch=sm_80 max_subarray_sum.cu -o max_subarray
// 运行:     ./max_subarray

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

// 简化版：直接每个线程算一个窗口和（暴力但并行），适合教学
// 生产版用 prefix sum 优化到 O(N)
__global__ void max_subarray_sum_kernel(const int* input, int* output, int N, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_windows = N - W + 1;
    if (idx >= num_windows) return;

    int sum = 0;
    for (int j = idx; j < idx + W; j++) sum += input[j];

    atomicMax(output, sum);
}

int main() {
    int N = 50000, W = 25000;
    size_t bytes = N * sizeof(int);
    std::vector<int> h_input(N);
    srand(42);
    for (auto& x : h_input) x = rand() % 21 - 10;

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, sizeof(int));
    int neg = INT_MIN;
    cudaMemcpy(d_output, &neg, sizeof(int), cudaMemcpyHostToDevice);

    int num_windows = N - W + 1;
    int blocks = (num_windows + BLOCK - 1) / BLOCK;
    max_subarray_sum_kernel<<<blocks, BLOCK>>>(d_input, d_output, N, W);
    cudaDeviceSynchronize();

    int result;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 验证
    int cpu_max = INT_MIN;
    for (int i = 0; i < num_windows; i++) {
        int s = 0;
        for (int j = i; j < i + W; j++) s += h_input[j];
        cpu_max = std::max(cpu_max, s);
    }

    printf("GPU: %d, CPU: %d, %s\n", result, cpu_max, result == cpu_max ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `max_subarray_sum_kernel` 填进 `solve`。生产版用 prefix sum 把窗口求和从 O(W) 降到 O(1)。

## 5. 复杂度分析

| 维度 | 暴力版 | prefix sum 优化版 |
|------|--------|------------------|
| 时间 | O(N×W) | O(N) |
| 空间 | O(1) | O(N) prefix 数组 |
| 瓶颈 | compute（W 大时） | memory-bound（读 prefix） |

> 💡 **一句话总结**：滑动窗口最大和是 Continuous Batching iteration-level 调度的微缩版——窗口滑动 = 请求加入/退出，prefix sum 优化 = token budget 的窗口控制。

# LeetGPU Top K Selection 题解

## 1. 题目概述

- **标题 / 题号**：Top K Selection（#29，medium）
- **链接**：https://leetgpu.com/challenges/top-k-selection
- **难度**：中等
- **标签**：CUDA、bitonic sort、堆归约、selection、reduction

**题意**：给定长度为 `N` 的 `int32` 数组 `input` 和一个整数 `k`，找出其中**最大的 k 个元素**（无需排序，但需正确选出 top-k）。

**示例**：

```text
input = [5, 2, 8, 1, 9, 3, 7, 4], k = 3
输出 = [7, 8, 9]（顺序可能不同，但必须是最大的 3 个）
```

**约束**：`1 ≤ k ≤ N ≤ 10^6`；性能测试取大 `N`。

> 💡 这道题的 **top-k 选择**与 [Week6 Day6](../../aiinfra/week6/day6/README.md) benchmark 的 P99 latency 计算同构——P99 就是"找出延迟排第 99 百分位的值"，本质是 top-k selection（k=N×0.01，选第 k 小）。benchmark 的 `percentile()` 是串行排序版，这道题用 GPU 并行加速。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
// 方法 1：完整排序取后 k 个 → O(N log N)
sort(input, input + N);
// top-k = input[N-k .. N-1]

// 方法 2：最小堆维护 k 个 → O(N log k)
priority_queue<int, vector<int>, greater<int>> pq;   // 最小堆
for (int x : input) {
    pq.push(x);
    if (pq.size() > k) pq.pop();   // 堆顶是最小的，弹出
}
// 堆中剩 top-k
```

### 朴素 GPU（完整排序）

```cuda
// 对整个数组做 bitonic sort，然后取后 k 个
// 瓶颈：N=10^6 时排序开销大，但只需 top-k，全排序浪费
```

**瓶颈**：只需 top-k 却全排序，`O(N log N)` 浪费——`k` 远小于 `N` 时尤其严重。

## 3. GPU 设计

### 3.1 并行化策略：bitonic sort + 取前 k

![Top K Selection：bitonic 排序 + 取前 k](images/top_k_selection_overview.svg)

策略选择取决于 `k` 与 `N` 的关系：

| 场景 | 策略 | 复杂度 |
|------|------|--------|
| `k` 接近 `N`（如 top 50%） | **bitonic sort** 全排序取后 k | `O(N log²N)` |
| `k` 远小于 `N`（如 top 0.1%） | **block 归约 + 堆/筛选** | `O(N)` + `O(k log k)` |
| 通用 | **radix select**（桶筛选） | `O(N)` |

教学版用 **bitonic sort**（最直观的并行排序，适合 GPU）。

### 3.2 Bitonic Sort 原理

Bitonic 序列：先升后降（或先降后升）的序列。Bitonic sort 利用性质：
1. 把无序数组逐步变成 bitonic 序列（compare-swap 网络）
2. 对 bitonic 序列做 bitonic merge（递归对半比较交换）
3. `O(log²N)` 步，每步 `O(N)` 并行比较 → 适合 GPU

### 3.3 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `input[]` | global memory | 原地排序 |
| 比较对 | registers | compare-swap 在 register 内 |
| block 间归约 | global memory | 大 N 时多 block 协作 |

## 4. Kernel 实现

```cuda
// top_k_selection.cu —— Top K Selection（bitonic sort + 取后 k）
// 编译命令: nvcc -O3 -arch=sm_120 top_k_selection.cu -o top_k
// 运行:     ./top_k

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// bitonic sort kernel：对一个 block 内的数据排序（升序）
// 每步 compare-swap：比较距离 j 的两元素，按方向交换
__global__ void bitonic_sort_kernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // log2(N) 个阶段，每阶段做 bitonic merge
    for (int k = 2; k <= N; k *= 2) {           // 子序列长度
        for (int j = k / 2; j > 0; j /= 2) {    // 比较距离
            int i = tid ^ j;                     // 配对索引
            if (i > tid && i < N) {
                bool ascending = ((tid & k) == 0);
                if ((ascending && data[tid] > data[i]) ||
                    (!ascending && data[tid] < data[i])) {
                    // 交换（用原子或 warp shuffle；教学版用简单条件）
                    int tmp = data[tid];
                    data[tid] = data[i];
                    data[i] = tmp;
                }
            }
            __syncthreads();
        }
    }
}

// 教学版：用单 block 排序小数组（N ≤ 1024），正式版需多 block 协作
// 注意：上述 __syncthreads() 跨 block 无效，正式版用 cooperative groups 或多 kernel
//       此处简化演示 bitonic sort 的 compare-swap 逻辑

// 更实用的版本：每 thread 处理多元素，block 内 shared memory 排序
#define BLOCK 256

__global__ void bitonic_sort_block(int* data, int N) {
    __shared__ int sdata[2 * BLOCK];
    int tid = threadIdx.x;

    // 加载数据到 shared memory
    if (tid < N) sdata[tid] = data[tid];
    else sdata[tid] = INT_MIN;
    __syncthreads();

    // bitonic sort in shared memory
    for (int k = 2; k <= 2 * BLOCK; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int i = tid ^ j;
            if (i > tid) {
                bool up = ((tid & k) == 0);
                int a = sdata[tid], b = sdata[i];
                if ((up && a > b) || (!up && a < b)) {
                    sdata[tid] = b; sdata[i] = a;
                }
            }
            __syncthreads();
        }
    }
    if (tid < N) data[tid] = sdata[tid];
}

int main() {
    int N = 8, k = 3;
    std::vector<int> h_input = {5, 2, 8, 1, 9, 3, 7, 4};

    int *d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // bitonic sort（升序）
    bitonic_sort_block<<<1, 2*BLOCK>>>(d_data, N);
    cudaDeviceSynchronize();

    // 取后 k 个（最大的 k 个）
    std::vector<int> h_out(N);
    cudaMemcpy(h_out.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted: ");
    for (int x : h_out) printf("%d ", x);
    printf("\nTop %d: ", k);
    for (int i = N - k; i < N; i++) printf("%d ", h_out[i]);
    printf("\n");

    cudaFree(d_data);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `bitonic_sort_block` 填进 `solve`。教学版用单 block shared memory 排序（N ≤ 512），正式版大 N 需多 block 协作（cooperative groups 或多 kernel launch）。bitonic sort 的核心是 `compare-swap` 网络：`tid ^ j` 配对、`(tid & k)==0` 定方向。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_120 top_k_selection.cu -o top_k
ncu --set full ./top_k | rg -i "Memory Throughput|Occupancy"
```

**关键指标**：

| 方法 | 时间复杂度 | 适合场景 |
|------|-----------|---------|
| 完整 bitonic sort | `O(N log²N)` | k 接近 N |
| 堆归约（block 维护 k-堆） | `O(N log k)` | k 远小于 N |
| radix select（桶筛选） | `O(N)` | 通用，k 任意 |

**优化方向**：

1. **k 远小于 N 时用堆归约**：每个 block 维护大小 k 的最小堆，扫描数据弹出堆顶，最后归约各 block 的堆
2. **radix select**：按高位 radix 分桶，递归选包含第 k 大的桶，`O(N)` 复杂度
3. **warp 级归约**：block 内用 `__shfl_down_sync` 做 warp 归约，减少 shared memory 竞争
4. **多 block 协作**：大 N 用 cooperative groups，跨 block bitonic merge

## 6. 复杂度分析

| 维度 | bitonic sort | 堆归约 | radix select |
|------|-------------|--------|-------------|
| 时间 | `O(N log²N)` | `O(N log k)` | `O(N)` |
| 空间 | `O(N)` shared | `O(k)` per block | `O(N)` 桶 |
| 瓶颈 | compute（比较网络） | compute（堆操作） | memory（桶扫描） |
| 适合 | k≈N | k≪N | 通用 |

> 💡 **一句话总结**：Top K Selection 是 benchmark P99 统计的 GPU 加速版——bitonic sort 的 compare-swap 网络并行排序，对应 P99 用排序后取分位数。k≪N 时用堆归约或 radix select 避免 `O(N log²N)` 全排序浪费。

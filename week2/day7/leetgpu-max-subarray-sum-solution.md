# LeetGPU Max Subarray Sum 题解（Week2 Day7 综合验收）

> 本题解与 [Week6 Day2 的 Max Subarray Sum 题解](../../leetgpu/week6/day2/leetgpu-max-subarray-sum-solution.md) 内容相同，Week2 Day7 综合验收日链接指向此处。

## 1. 题目概述

- **标题 / 题号**：Max Subarray Sum（#51，medium）
- **链接**：https://leetgpu.com/challenges/max-subarray-sum
- **难度**：中等
- **标签**：CUDA、滑动窗口、prefix sum、reduction、memory-bound

**题意**：给定长度为 `N` 的 `int32` 数组 `input` 和窗口大小 `window_size`，求所有长度恰好为 `window_size` 的连续子数组的**最大和**。

**约束**：`1 ≤ window_size ≤ N ≤ 10,000,000`。

> 💡 与 [Week2 Day7 综合验收](../../../aiinfra/daily/week2/day7/README.md) 的关联：本题综合了 Week2 的两大主题——Prefix Sum（Day1）+ Reduction（Week1 Day4/Day5）。用 prefix sum 计算窗口和，再用 reduction 求最大值，是一道"两阶段 kernel"的综合手撕题。

## 2. GPU 设计

两阶段：
1. **Prefix Sum**：计算 `pref[i] = input[0] + ... + input[i-1]`（Day1 的 inclusive scan）
2. **Window Sum + Max**：`window_sum[i] = pref[i+w] - pref[i]`，然后 block reduce 求最大值

## 3. Kernel 实现

```cuda
// max_subarray_sum.cu —— Prefix Sum + Window Max
#include <cuda_runtime.h>

__inline__ __device__ int warp_reduce_max(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    return val;
}

// Stage 1: compute prefix sum (simplified, single-block for clarity)
__global__ void prefix_sum_kernel(const int* input, long long* prefix, int N) {
    int tid = threadIdx.x;
    // Simple sequential prefix sum per block (for large N, use multi-block scan)
    __shared__ long long s_prefix[1024];
    s_prefix[tid] = (tid < N) ? (long long)input[tid] : 0LL;
    __syncthreads();

    // Hillis-Steele scan
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        long long val = (tid >= offset) ? s_prefix[tid] + s_prefix[tid - offset] : s_prefix[tid];
        __syncthreads();
        s_prefix[tid] = val;
        __syncthreads();
    }

    if (tid < N)
        prefix[tid] = s_prefix[tid];
}

// Stage 2: window sum + max reduction
__global__ void window_max_kernel(const long long* prefix, int* output, int N, int W) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int local_max = -2147483647;

    // Grid-stride loop: each thread computes one window sum
    for (int i = gid; i <= N - W; i += gridDim.x * blockDim.x) {
        long long sum = prefix[i + W - 1] - (i > 0 ? prefix[i - 1] : 0LL);
        local_max = max(local_max, (int)sum);
    }

    // Warp reduce max
    local_max = warp_reduce_max(local_max);

    // Block reduce max via shared memory
    __shared__ int warp_max[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0)
        warp_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local_max = (lane < num_warps) ? warp_max[lane] : -2147483647;
        local_max = warp_reduce_max(local_max);
        if (lane == 0)
            atomicMax(output, local_max);
    }
}

extern "C" void solve(const int* input, int* output, int N, int window_size) {
    // Stage 1: prefix sum
    long long* d_prefix;
    cudaMalloc(&d_prefix, N * sizeof(long long));
    prefix_sum_kernel<<<1, 1024>>>(input, d_prefix, N);

    // Stage 2: window max
    int init = -2147483647;
    cudaMemcpy(output, &init, sizeof(int), cudaMemcpyHostToDevice);
    int gridSize = min((N + 255) / 256, 1024);
    window_max_kernel<<<gridSize, 256>>>(d_prefix, output, N, window_size);

    cudaFree(d_prefix);
}
```

### 3.1 LeetGPU 提交版本

下面给出适配官方 starter 签名 `solve(input, output, N, window_size)` 的完整提交版本。它分三阶段：先做 block 级 inclusive prefix sum，再用 block 和的 exclusive 前缀把各块拼成全局前缀，最后 grid-stride 求窗口最大和。

```cuda
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <vector>

#define SCAN_BLOCK 1024

__inline__ __device__ int warp_reduce_max(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(0xffffffff, val, offset);
        val = max(val, other);
    }
    return val;
}

__global__ void block_inclusive_scan(const int* input, long long* prefix, long long* block_sums, int N) {
    __shared__ long long sdata[SCAN_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * SCAN_BLOCK + tid;
    sdata[tid] = (gid < N) ? (long long)input[gid] : 0LL;
    __syncthreads();

    for (int offset = 1; offset < SCAN_BLOCK; offset <<= 1) {
        long long tmp = (tid >= offset) ? sdata[tid - offset] : 0LL;
        __syncthreads();
        sdata[tid] += tmp;
        __syncthreads();
    }

    if (gid < N)
        prefix[gid] = sdata[tid];
    if (tid == SCAN_BLOCK - 1)
        block_sums[blockIdx.x] = sdata[tid];
}

__global__ void add_carry(long long* prefix, const long long* carry, int N) {
    int gid = blockIdx.x * SCAN_BLOCK + threadIdx.x;
    if (gid < N)
        prefix[gid] += carry[blockIdx.x];
}

__global__ void window_max_kernel(const long long* prefix, int* output, int N, int W) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * SCAN_BLOCK + tid;
    int local_max = INT_MIN;

    for (int i = gid; i <= N - W; i += gridDim.x * SCAN_BLOCK) {
        long long sum = prefix[i + W - 1] - (i > 0 ? prefix[i - 1] : 0LL);
        local_max = max(local_max, (int)sum);
    }

    local_max = warp_reduce_max(local_max);

    __shared__ int warp_max[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0)
        warp_max[warp_id] = local_max;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (SCAN_BLOCK + 31) / 32;
        local_max = (lane < num_warps) ? warp_max[lane] : INT_MIN;
        local_max = warp_reduce_max(local_max);
        if (lane == 0)
            atomicMax(output, local_max);
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int window_size) {
    if (N <= 0 || window_size <= 0 || window_size > N) return;

    int num_blocks = (N + SCAN_BLOCK - 1) / SCAN_BLOCK;

    long long *d_prefix = nullptr, *d_block_sums = nullptr, *d_carry = nullptr;
    cudaMalloc(&d_prefix, (size_t)N * sizeof(long long));
    cudaMalloc(&d_block_sums, (size_t)num_blocks * sizeof(long long));
    cudaMalloc(&d_carry, (size_t)num_blocks * sizeof(long long));

    // Stage 1: 每 block 做 inclusive prefix sum
    block_inclusive_scan<<<num_blocks, SCAN_BLOCK>>>(input, d_prefix, d_block_sums, N);

    // Stage 2: block 和在 CPU 上做 exclusive prefix，作为 carry 加回
    std::vector<long long> h_sums(num_blocks), h_carry(num_blocks);
    cudaMemcpy(h_sums.data(), d_block_sums, num_blocks * sizeof(long long), cudaMemcpyDeviceToHost);
    long long acc = 0;
    for (int b = 0; b < num_blocks; ++b) {
        h_carry[b] = acc;
        acc += h_sums[b];
    }
    cudaMemcpy(d_carry, h_carry.data(), num_blocks * sizeof(long long), cudaMemcpyHostToDevice);
    add_carry<<<num_blocks, SCAN_BLOCK>>>(d_prefix, d_carry, N);

    // Stage 3: 窗口和 + 最大归约
    int init = INT_MIN;
    cudaMemcpy(output, &init, sizeof(int), cudaMemcpyHostToDevice);

    int win_blocks = std::min((N + SCAN_BLOCK - 1) / SCAN_BLOCK, 1024);
    window_max_kernel<<<win_blocks, SCAN_BLOCK>>>(d_prefix, output, N, window_size);

    cudaFree(d_prefix);
    cudaFree(d_block_sums);
    cudaFree(d_carry);
    cudaDeviceSynchronize();
}
```

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N)` prefix sum + `O(N)` window max |
| 算术强度 | 低 → memory-bound |
| 瓶颈类型 | **memory-bound**：两次 O(N) 扫描 |
| 综合考察 | Prefix Sum（Day1）+ Warp Shuffle Reduce（Week1）+ atomicMax |

> 💡 **一句话总结**：Max Subarray Sum 是 Week2 综合验收的理想题目——融合了 Prefix Sum（Day1）+ Reduction（Week1），考察两阶段 kernel 设计和 warp shuffle 归约，适合限时手撕。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 16 | [Prefix Sum](https://leetgpu.com/challenges/prefix-sum) | 中等 | — | Prefix Sum，本题的核心基础 |
| 47 | [Subarray Sum](https://leetgpu.com/challenges/subarray-sum) | 中等 | — | Subarray Sum，prefix sum 直接应用 |
| 48 | [2D Subarray Sum](https://leetgpu.com/challenges/2d-subarray-sum) | 中等 | — | 2D Subarray Sum，扩展到二维 |
| 72 | [Stream Compaction](https://leetgpu.com/challenges/stream-compaction) | 中等 | — | Stream Compaction，scan 的另一应用 |

> 💡 **选题思路**：prefix sum + Kadane scan + 归约，练习 scan 的综合应用。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

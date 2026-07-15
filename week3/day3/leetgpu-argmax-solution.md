# LeetGPU Argmax 题解

## 1. 题目概述

- **标题 / 题号**：Argmax（#29，medium）
- **链接**：https://leetgpu.com/challenges/argmax
- **难度**：中等
- **标签**：CUDA、归约、Argmax、warp shuffle、`__shfl_down_sync`

**题意**：给定长度为 `N` 的浮点数组 `input`，找到最大值所在的下标。多个相同最大值时返回最小下标。

**约束**：`1 ≤ N ≤ 10,000,000`。

> 💡 与 [Week3 Day3 优化对比实验](../../aiinfra/daily/week3/day3/README.md) 的关联：Argmax 是"带状态追踪的归约"——不仅要找最大值，还要记录其下标。正是 warp 级 vs block 级 reduce 的直接实战。

## 2. GPU 设计

三阶段归约：线程级（grid-stride 扫描）→ warp 级（`__shfl_down_sync`）→ block 级（shared memory）→ 跨 block（atomic 或第二次 kernel）。

核心难点：平局处理——值相同时取较小下标。

## 3. Kernel 实现

```cuda
// argmax.cu —— Argmax with warp shuffle
#include <cuda_runtime.h>

struct ValIdx {
    float val;
    int idx;
};

__device__ ValIdx warp_reduce_argmax(ValIdx v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        // 平局取较小 idx
        if (other_val > v.val || (other_val == v.val && other_idx < v.idx)) {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

__global__ void argmax_kernel(const float* input, int* output, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    ValIdx local = {-1e30f, -1};
    // grid-stride loop
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        if (input[i] > local.val || (input[i] == local.val && i < local.idx)) {
            local.val = input[i];
            local.idx = i;
        }
    }

    // warp reduce
    local = warp_reduce_argmax(local);

    // block reduce via shared memory
    __shared__ ValIdx warp_results[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0)
        warp_results[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local = (lane < num_warps) ? warp_results[lane] : ValIdx{-1e30f, -1};
        local = warp_reduce_argmax(local);
        if (lane == 0) {
            atomicMax(output, local.idx); // 简化：用 atomic（实际需要 atomicCAS 处理平局）
        }
    }
}

extern "C" void solve(const float* input, int* output, int N) {
    int blockSize = 256;
    int gridSize = min((N + blockSize - 1) / blockSize, 1024);
    int init = -1;
    cudaMemcpy(output, &init, sizeof(int), cudaMemcpyHostToDevice);
    argmax_kernel<<<gridSize, blockSize>>>(input, output, N);
}
```

### 3.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本。与上方教学版不同，这里使用两次 kernel（block 内 warp reduce 出局部最优，再一个 block 归约出全局下标）来正确处理平局与跨 block 竞争。

```cuda
#include <cuda_runtime.h>
#include <climits>

struct ValIdx {
    float val;
    int idx;
};

__device__ ValIdx warp_reduce_argmax(ValIdx v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        if (other_val > v.val || (other_val == v.val && other_idx < v.idx)) {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

__global__ void argmax_kernel(const float* input, ValIdx* block_results, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    ValIdx local = {__int_as_float(0xff800000), INT_MAX}; // -inf, 哨兵下标
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        if (input[i] > local.val || (input[i] == local.val && i < local.idx)) {
            local.val = input[i];
            local.idx = i;
        }
    }

    local = warp_reduce_argmax(local);

    __shared__ ValIdx warp_results[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0)
        warp_results[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local = (lane < num_warps) ? warp_results[lane]
                                  : ValIdx{__int_as_float(0xff800000), INT_MAX};
        local = warp_reduce_argmax(local);
        if (lane == 0)
            block_results[blockIdx.x] = local;
    }
}

__global__ void reduce_argmax_kernel(const ValIdx* block_results, int* output, int num_blocks) {
    int tid = threadIdx.x;
    ValIdx local = {__int_as_float(0xff800000), INT_MAX};

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        ValIdx other = block_results[i];
        if (other.val > local.val || (other.val == local.val && other.idx < local.idx)) {
            local = other;
        }
    }

    local = warp_reduce_argmax(local);

    __shared__ ValIdx warp_results[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0)
        warp_results[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local = (lane < num_warps) ? warp_results[lane]
                                  : ValIdx{__int_as_float(0xff800000), INT_MAX};
        local = warp_reduce_argmax(local);
        if (lane == 0)
            *output = local.idx;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, int* output, int N) {
    int blockSize = 256;
    int gridSize = min((N + blockSize - 1) / blockSize, 1024);

    ValIdx* d_block_results;
    cudaMalloc(&d_block_results, gridSize * sizeof(ValIdx));
    argmax_kernel<<<gridSize, blockSize>>>(input, d_block_results, N);
    reduce_argmax_kernel<<<1, 256>>>(d_block_results, output, gridSize);
    cudaFree(d_block_results);
    cudaDeviceSynchronize();
}
```

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N)` + `O(log 32)` warp reduce |
| 瓶颈类型 | memory-bound（读 N 个 float，计算极轻） |
| 关键技巧 | `__shfl_down_sync` 同时 shuffle val 和 idx，平局取较小 idx |

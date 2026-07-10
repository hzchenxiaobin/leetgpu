// prefix_sum_inclusive.cu —— LeetGPU Prefix Sum 提交版（阶段一直接算 inclusive）
// 平台接口：extern "C" void solve(const float* input, float* output, int N)
//
// 与题解 4.7 版（exclusive）的核心区别：
//   - 阶段一：block 内直接 inclusive scan，output[tid] 已经包含 input[tid]
//   - 阶段三：只需把全局偏移加回去，不需要再读 input[tid]
// 这样省掉阶段三对 input 的一次全局内存重读。

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define NUM_WARPS  (BLOCK_SIZE / WARP_SIZE)   // 8

// ============================================================
// warp 内 inclusive scan：__shfl_up_sync，5 步蝶形
// ============================================================
__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val;   // lane i 持有本 warp 内 [0..i] 的前缀和
}

// ============================================================
// block 内 inclusive scan：warp scan + shared 汇总 + 偏移加回
// 返回每线程对应的 inclusive 前缀和；block 总和由 lane (BLOCK_SIZE-1) 写入 *block_sum
// ============================================================
__inline__ __device__ float block_inclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    // ① 每 warp 各自 inclusive scan
    float inclusive = warp_inclusive_scan(val);

    // ② 每 warp 的 lane 31 记录本 warp 总和
    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;
    }
    __syncthreads();

    // ③ 第一个 warp 对 warp_sums[0..NUM_WARPS-1] 做 inclusive scan
    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS) warp_sums[lane] = v;
    }
    __syncthreads();

    // ④ 当前 warp 之前所有 warp 的总和
    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];

    // ⑤ block 内 inclusive = 前面所有 warp 的和 + 本 warp 内 inclusive
    float inclusive_block = warp_offset + inclusive;

    // ⑥ block 总和 = 最后一个线程的 inclusive_block
    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = inclusive_block;
    }
    return inclusive_block;
}

// ============================================================
// block 内 exclusive scan：供阶段二对 block_sums 做 exclusive prefix sum 使用
// 返回每线程对应的 exclusive 前缀和；chunk 总和由 lane (BLOCK_SIZE-1) 写入 *block_sum
// ============================================================
__inline__ __device__ float block_exclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    float inclusive = warp_inclusive_scan(val);

    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;
    }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS) warp_sums[lane] = v;
    }
    __syncthreads();

    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];
    float exclusive = warp_offset + (inclusive - val);

    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = warp_offset + inclusive;
    }
    return exclusive;
}

// ============================================================
// 阶段一：每 block 对自己那段做 inclusive scan，结果存 output，总和写 block_sums
// ============================================================
__global__ void scan_block_kernel(const float* input, float* output,
                                  float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool valid = (tid < N);
    float val = valid ? input[tid] : 0.0f;
    float inclusive = block_inclusive_scan(val, &block_sums[blockIdx.x]);
    if (valid) output[tid] = inclusive;
}

// ============================================================
// 阶段二：对 block_sums[] 做 exclusive prefix sum → block_offsets[]
// 使用 grid-stride 迭代，支持 numBlocks > BLOCK_SIZE 的场景
// ============================================================
__global__ void scan_offsets_kernel(const float* block_sums,
                                    float* block_offsets, int M) {
    __shared__ float s_chunk_total;
    __shared__ float s_running;
    int tid = threadIdx.x;

    if (tid == 0) { s_running = 0.0f; }
    __syncthreads();

    for (int chunk = 0; chunk < M; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < M) ? block_sums[idx] : 0.0f;

        float exclusive = block_exclusive_scan(val, &s_chunk_total);

        if (idx < M) {
            block_offsets[idx] = exclusive + s_running;
        }

        __syncthreads();
        if (tid == 0) s_running += s_chunk_total;
        __syncthreads();
    }
}

// ============================================================
// 阶段三：每元素 = 阶段一的 inclusive + 本 block 全局偏移
// 注意：不需要再读 input[tid]
// ============================================================
__global__ void add_offset_kernel(float* output,
                                  const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N) return;
    output[tid] += block_offsets[blockIdx.x];
}

// ============================================================
// LeetGPU 平台入口
// ============================================================
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* block_sums;
    float* block_offsets;
    cudaMalloc(&block_sums,    numBlocks * sizeof(float));
    cudaMalloc(&block_offsets, numBlocks * sizeof(float));

    scan_block_kernel<<<numBlocks, BLOCK_SIZE>>>(input, output, block_sums, N);
    scan_offsets_kernel<<<1, BLOCK_SIZE>>>(block_sums, block_offsets, numBlocks);
    add_offset_kernel<<<numBlocks, BLOCK_SIZE>>>(output, block_offsets, N);

    cudaDeviceSynchronize();
    cudaFree(block_sums);
    cudaFree(block_offsets);
}

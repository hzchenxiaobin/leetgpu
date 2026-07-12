#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS 8

__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }

    return val;
}

__inline__ __device__ float block_exclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    float inclusive = warp_inclusive_scan(val);

    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;
    }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS) {
            warp_sums[lane] = v;
        }
    }
    __syncthreads();

    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];
    float exclusive = warp_offset + (inclusive - val);

    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = warp_offset + inclusive;
    }

    return exclusive;
}

__global__ void scan_offsets_kernel(const float* block_sums, float* block_offsets, int M) {
    __shared__ float s_chunk_total;
    __shared__ float s_running;
    int tid = threadIdx.x;

    if (tid == 0) {
        s_running = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < M; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < M) ? block_sums[idx] : 0.0f;

        float exclusive = block_exclusive_scan(val, &s_chunk_total);

        if (idx < M) {
            block_offsets[idx] = exclusive + s_running;
        }
        __syncthreads();

        if (tid == 0)
            s_running += s_chunk_total;
        __syncthreads();
    }
}

__global__ void add_offset_kernel(float* output, const float* input, const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N)
        return;
    output[tid] = output[tid] + block_offsets[blockIdx.x] + input[tid];
}

__global__ void scan_block_kernel(const float* input, float* output, float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    bool valid = tid < N;
    float val = valid ? input[tid] : 0.0f;
    float exclusive = block_exclusive_scan(val, &block_sums[blockIdx.x]);
    if (valid) {
        output[tid] = exclusive;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *blockSums, *blockOffsets;

    cudaMalloc(&blockSums, BLOCK_SIZE * sizeof(float));
    cudaMalloc(&blockOffsets, BLOCK_SIZE * sizeof(float));

    // 1.求 block 里面的前缀和
    scan_block_kernel<<<numBlocks, BLOCK_SIZE>>>(input, output, blockSums, N);

    // 2.求 block 间的前缀和
    scan_offsets_kernel<<<1, BLOCK_SIZE>>>(blockSums, blockOffsets, numBlocks);

    // 3.求每个位置的前缀和
    add_offset_kernel<<<numBlocks, BLOCK_SIZE>>>(output, input, blockOffsets, N);
}

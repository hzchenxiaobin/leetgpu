#include <cuda_runtime.h>
constexpr int TILE = 256;
constexpr int THREAD_PER_BLOCK = TILE;
constexpr int MAX_BLOCK_NUM = 1024;
__device__ float g_block_sum[MAX_BLOCK_NUM];
__device__ float warp_pre_sum(float r_val, int lane_id)
{
    unsigned int mask = 0xffff'ffff;
    for (int offset = 1; offset <= 16; offset *= 2) {
        float r_up_val = __shfl_up_sync(mask, r_val, offset);
        if (lane_id >= offset) {
            r_val += r_up_val;
        }
    }
    return r_val;
}
__global__ void intra_block_reduce(const float* input, float* output, int N) {
    __shared__ float smem[TILE];
    int blk_start = blockIdx.x * TILE;
    int offset = blk_start + threadIdx.x;
    int tid = threadIdx.x;
    constexpr int k_warp_size = 32;
    int lane_id = tid % k_warp_size;
    int warp_id = tid / k_warp_size;
    int warp_num = THREAD_PER_BLOCK / k_warp_size;
    if (offset < N) {
        smem[tid] = input[offset];
    }
    __syncthreads();
    float r_val = smem[tid];
    r_val = warp_pre_sum(r_val, lane_id);
    __syncthreads();
    smem[tid] =  r_val;
    __syncthreads();
    for (int wid_offset = 1; wid_offset <= warp_num / 2; wid_offset *= 2) {
        if (warp_id >= wid_offset) {
            float r_up_val = smem[(warp_id-wid_offset)*k_warp_size + k_warp_size -1];
             r_val += r_up_val;
        }
        __syncthreads();
        if (warp_id >= wid_offset) {
            smem[tid] = r_val;
        }
        __syncthreads();
    }
    if (offset < N) {
        output[blk_start + tid] = smem[tid];
    }
    if (tid == 0) {
        g_block_sum[blockIdx.x] = smem[TILE-1];
    }
}

__global__ void inter_block_reduce(const float* input, float* output, int N) {
    int blk_start = blockIdx.x * TILE;
    int offset = blk_start + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem[TILE];
    float val = 0.0f;
    int size = blockIdx.x;
    for (int i = tid; i < size; i += THREAD_PER_BLOCK) {
        val += g_block_sum[i];
    }
    smem[tid] = val;
    __syncthreads();
    for (int i = TILE/2; i >= 1; i /= 2) {
        if (tid < i) {
            smem[tid] += smem[tid + i];
        }
        __syncthreads();
    }
    val = smem[0];
    if (offset < N) {
        output[offset] += val;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    dim3 block(THREAD_PER_BLOCK);
    dim3 grid((N + TILE - 1) / TILE);
    intra_block_reduce<<<grid, block>>>(input, output, N);
    inter_block_reduce<<<grid, block>>>(input, output, N);
}

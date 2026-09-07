// 50-rms-normalization.cu —— RMSNorm：一次 reduce（sum of squares）+ 归一化
// 编译命令: nvcc -O3 -arch=sm_120 50-rms-normalization.cu -o rmsnorm -lineinfo
// 运行:     ./rmsnorm 128 8192

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

// ---- warp 级归约：sum（复用 Day 4 模板）----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 + 广播 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            shared[0] = val; // 广播 slot
    }
    __syncthreads();
    return shared[0];
}

// ---- RMSNorm kernel：一个 block 负责一行，一次 reduce ----
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ y,
                               int M, int D, float eps) {
    __shared__ float shared[NUM_WARPS + 1];

    int r = blockIdx.x;
    if (r >= M)
        return;
    const float* xr = x + r * D;
    float* yr = y + r * D;

    // ---- Pass 1：求 sum of squares（只 reduce 一次，比 LayerNorm 少一次）----
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float v = xr[i];
        local_sq += v * v;
    }
    float row_sq = block_reduce_sum(local_sq, shared);

    // ---- Pass 2：归一化 + affine：y = x * rrms * gamma ----
    float rrms = rsqrtf(row_sq / D + eps);
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = xr[i] * rrms * gamma[i];
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 128;
    int D = (argc > 2) ? atoi(argv[2]) : 8192;
    float eps = 1e-5f;
    size_t bytes = (size_t)M * D * sizeof(float);
    printf("M=%d, D=%d  (%.1f MB)\n", M, D, bytes / 1e6);

    // ---- host ----
    float* hX = (float*)malloc(bytes);
    float* hY = (float*)malloc(bytes);
    float* hG = (float*)malloc(D * sizeof(float));
    float* hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * D; ++i)
        hX[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f; // [-10, 10]
    for (int i = 0; i < D; ++i)
        hG[i] = 1.0f;

    // ---- device ----
    float *dX, *dG, *dY;
    cudaMalloc(&dX, bytes);
    cudaMalloc(&dG, D * sizeof(float));
    cudaMalloc(&dY, bytes);
    cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dG, hG, D * sizeof(float), cudaMemcpyHostToDevice);

    // ---- launch ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    rmsnorm_kernel<<<M, BLOCK_SIZE>>>(dX, dG, dY, M, D, eps);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 2 遍读 x + 1 遍读 gamma + 1 遍写 y
    float bw_gbs = (2.0f * bytes + D * sizeof(float) + bytes) / 1e9 / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证 ----
    cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost);
    float maxDiff = 0.0f;
    for (int r = 0; r < M; ++r) {
        float sq = 0.0f;
        for (int i = 0; i < D; ++i)
            sq += hX[r * D + i] * hX[r * D + i];
        float rrms = 1.0f / sqrtf(sq / D + eps);
        for (int i = 0; i < D; ++i) {
            float ref = hX[r * D + i] * rrms * hG[i];
            maxDiff = fmaxf(maxDiff, fabsf(hY[r * D + i] - ref));
        }
    }
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff < 1e-4f ? "PASS" : "FAIL");

    cudaFree(dX);
    cudaFree(dG);
    cudaFree(dY);
    free(hX);
    free(hY);
    free(hG);
    free(hRef);
    return 0;
}

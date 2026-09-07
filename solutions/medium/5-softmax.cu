// 5-softmax.cu —— Softmax：三遍扫描（max → sum(exp) → normalize），safe softmax
// 编译命令: nvcc -O3 -arch=sm_120 5-softmax.cu -o softmax -lineinfo
// 运行:     ./softmax 128 8192

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                               \
cudaError_t e = (call);                                                                                        \
if (e != cudaSuccess) {                                                                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
    exit(EXIT_FAILURE);                                                                                        \
}                                                                                                              \
} while (0)

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

// ---- warp 级归约：max（把 + 换成 fmaxf，初值 -INFINITY）----
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// ---- block 级归约：sum（warp shuffle + shared 汇总 + 广播）----
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

// ---- block 级归约：max（同结构，初值 -INFINITY）----
__inline__ __device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0)
            shared[0] = val; // 广播 row_max
    }
    __syncthreads();
    return shared[0];
}

// ---- Softmax kernel：一个 block 负责一行，三遍扫描 ----
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int M, int D) {
    __shared__ float shared[NUM_WARPS + 1];

    int r = blockIdx.x;
    if (r >= M)
        return;
    const float* xr = x + r * D;
    float* yr = y + r * D;

    // ---- Pass 1：求 row_max（数值稳定的关键：减掉它后 exp ≤ 1）----
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_max = fmaxf(local_max, xr[i]);
    float row_max = block_reduce_max(local_max, shared);

    // ---- Pass 2：求 row_sum = Σ exp(x - row_max) ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_sum += expf(xr[i] - row_max);
    float row_sum = block_reduce_sum(local_sum, shared);
    float inv_sum = 1.0f / row_sum; // 用乘法替代除法

    // ---- Pass 3：归一化 y = exp(x - row_max) / row_sum ----
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = expf(xr[i] - row_max) * inv_sum;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 128;
    int D = (argc > 2) ? atoi(argv[2]) : 8192;
    size_t bytes = (size_t)M * D * sizeof(float);
    printf("M=%d, D=%d  (%.1f MB)\n", M, D, bytes / 1e6);

    // ---- host ----
    float* hX = (float*)malloc(bytes);
    float* hY = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * D; ++i)
        hX[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f; // [-10, 10]

    // ---- device ----
    float *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, bytes));
    CHECK_CUDA(cudaMalloc(&dY, bytes));
    CHECK_CUDA(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    softmax_kernel<<<M, BLOCK_SIZE>>>(dX, dY, M, D);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 3 遍读 x + 1 遍写 y
    float bw_gbs = (3.0f * bytes + bytes) / 1e9 / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证：CPU 用 double 累加做参考 ----
    CHECK_CUDA(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));
    float maxDiff = 0.0f;
    for (int r = 0; r < M; ++r) {
        float m = hX[r * D];
        for (int i = 1; i < D; ++i)
            m = fmaxf(m, hX[r * D + i]);
        double s = 0.0;
        for (int i = 0; i < D; ++i)
            s += exp((double)hX[r * D + i] - m);
        for (int i = 0; i < D; ++i) {
            float ref = (float)(exp((double)hX[r * D + i] - m) / s);
            maxDiff = fmaxf(maxDiff, fabsf(hY[r * D + i] - ref));
        }
    }
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff < 1e-5f ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    free(hX);
    free(hY);
    return 0;
}

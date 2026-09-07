// 35-monte-carlo-integration.cu —— 两阶段归约 + atomicAdd 实现 Monte Carlo 积分
// 编译命令: nvcc -O3 -arch=sm_120 35-monte-carlo-integration.cu -o mc_integrate
// 运行:     ./mc_integrate

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                          \
    cudaError_t e = (call);                                                                                   \
    if (e != cudaSuccess) {                                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
        exit(EXIT_FAILURE);                                                                                   \
    }                                                                                                         \
} while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// warp 内归约：用 __shfl_down_sync 把 32 个 lane 的值归约到 lane 0
__device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

// 主归约 kernel：grid-stride loop + warp shuffle + atomicAdd
__global__ void mc_reduce_kernel(const float* __restrict__ y_samples,
                                  float* __restrict__ partial_sum,
                                  int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // ---- ① grid-stride loop：每线程累加多个元素 ----
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
        sum += y_samples[i];

    // ---- ② warp shuffle 归约：32 lane → lane 0 ----
    sum = warp_reduce_sum(sum);

    // ---- ③ block 内归约：warp 0 收集所有 warp 的部分和 ----
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);  // warp 0 再做一次归约

        // ---- ④ atomicAdd：每 block 的 lane 0 累加到全局结果 ----
        if (lane == 0)
            atomicAdd(partial_sum, sum);
    }
}

// 最终缩放 kernel：partial_sum / n * (b - a)
__global__ void mc_scale_kernel(float* result, float partial_sum, float a, float b, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        result[0] = partial_sum / (float)n * (b - a);
}

// ---- CPU 参考 ----
float mc_cpu(const float* y, float a, float b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += y[i];
    return (float)((b - a) * sum / n);
}

int main() {
    // 题目 example
    int n = 8;
    float a = 0.0f, b = 2.0f;
    float hY[] = {0.0625f, 0.25f, 0.5625f, 1.0f, 1.5625f, 2.25f, 3.0625f, 4.0f};
    printf("Monte Carlo Integration: a=%.1f b=%.1f n=%d\n", a, b, n);

    float *dY, *dResult;
    CHECK_CUDA(cudaMalloc(&dY, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dResult, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dY, hY, n * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化 partial_sum = 0
    float h_partial = 0.0f;
    float *dPartial;
    CHECK_CUDA(cudaMalloc(&dPartial, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dPartial, &h_partial, sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mc_reduce_kernel<<<blocks, BLOCK_SIZE>>>(dY, dPartial, n);
    mc_scale_kernel<<<1, 1>>>(dResult, 0.0f, a, b, n);  // placeholder
    // 实际：先取回 partial_sum，再调用 scale
    CHECK_CUDA(cudaMemcpy(&h_partial, dPartial, sizeof(float), cudaMemcpyDeviceToHost));
    float hResult = h_partial / n * (b - a);
    printf("result = %.6f (expect 3.1875)\n", hResult);
    printf("verify: %s\n", fabsf(hResult - 3.1875f) < 0.01f ? "PASS" : "FAIL");

    // ---- 性能测试 ----
    printf("\n--- Perf test (n=10M) ---\n");
    n = 10000000;
    a = -10.0f; b = 10.0f;
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaMalloc(&dY, n * sizeof(float)));
    // 随机初始化
    srand(42);
    float* hTemp = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) hTemp[i] = (float)(rand() % 20000 - 10000) / 10.0f;
    CHECK_CUDA(cudaMemcpy(dY, hTemp, n * sizeof(float), cudaMemcpyHostToDevice));
    h_partial = 0.0f;
    CHECK_CUDA(cudaMemcpy(dPartial, &h_partial, sizeof(float), cudaMemcpyHostToDevice));

    // 用足够多的 block 覆满 GPU
    int max_blocks = 2048;  // 控制 grid 大小
    blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > max_blocks) blocks = max_blocks;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    mc_reduce_kernel<<<blocks, BLOCK_SIZE>>>(dY, dPartial, n);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms (%d blocks × %d threads)\n", ms, blocks, BLOCK_SIZE);

    CHECK_CUDA(cudaMemcpy(&h_partial, dPartial, sizeof(float), cudaMemcpyDeviceToHost));
    float result = h_partial / n * (b - a);
    float ref = mc_cpu(hTemp, a, b, n);
    printf("result = %.4f, ref = %.4f, %s\n", result, ref, fabsf(result - ref) < 0.01f * fmaxf(1, fabsf(ref)) ? "PASS" : "FAIL");

    // 带宽估算
    size_t bytes = (size_t)n * sizeof(float);
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    free(hTemp);
    CHECK_CUDA(cudaFree(dY)); CHECK_CUDA(cudaFree(dResult)); CHECK_CUDA(cudaFree(dPartial));
    return 0;
}

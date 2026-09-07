// 27-mean-squared-error.cu —— MSE（融合 grid-stride 平方差累加 + 两级 block 归约 + 缩放）
// 编译命令: nvcc -O3 -arch=sm_120 27-mean-squared-error.cu -o mse
// 运行:     ./mse

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 融合：grid-stride 平方差累加 → warp 归约 → block 归约 → atomicAdd 到 mse
__global__ void mse_kernel(const float* predictions, const float* targets,
                           float* mse, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;
    __shared__ float warp_sums[WARP];

    float sum = 0.0f;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float d = predictions[i] - targets[i];
        sum += d * d;                       // 平方差在寄存器累加，不落 HBM
    }

    sum = warp_reduce(sum);
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (lane == 0)
            atomicAdd(mse, sum);
    }
}

// 单线程：mse[0] /= N
__global__ void scale_kernel(float* mse, int N) {
    mse[0] /= (float)N;
}

int main() {
    int N = 50000000;
    size_t bytes = (size_t)N * sizeof(float);

    std::vector<float> h_p(N), h_t(N);
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_p[i] = (float)(rand() % 10000) / 100.0f;
        h_t[i] = (float)(rand() % 10000) / 100.0f;
    }

    float *d_p, *d_t, *d_mse;
    cudaMalloc(&d_p, bytes);
    cudaMalloc(&d_t, bytes);
    cudaMalloc(&d_mse, sizeof(float));
    cudaMemcpy(d_p, h_p.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_t, h_t.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_mse, 0, sizeof(float));

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = num_sm * 4;
    int threads = BLOCK;
    printf("launch: blocks=%d  threads=%d  (SM=%d, N=%d)\n", blocks, threads, num_sm, N);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    mse_kernel<<<blocks, threads>>>(d_p, d_t, d_mse, N);
    scale_kernel<<<1, 1>>>(d_mse, N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    float gpu_mse;
    cudaMemcpy(&gpu_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 验证（double 累加）
    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = (double)h_p[i] - (double)h_t[i];
        cpu_sum += d * d;
    }
    float cpu_mse = (float)(cpu_sum / N);

    printf("GPU: %.6f, CPU: %.6f, %s\n", gpu_mse, cpu_mse,
           fabsf(gpu_mse - cpu_mse) < 1e-3f ? "PASS" : "FAIL");

    // 带宽估算：读 p + 读 t = 2 * N * 4 字节（融合后无中间写）
    size_t rw_bytes = 2 * bytes;
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective read bandwidth: %.1f GB/s\n", bw_gbs);

    cudaFree(d_p);
    cudaFree(d_t);
    cudaFree(d_mse);
    return 0;
}

// 68-sigmoid.cu —— grid-stride loop + __expf 快速数学实现 Sigmoid
// 编译命令: nvcc -O3 -arch=sm_120 68-sigmoid.cu -o sigmoid
// 运行:     ./sigmoid 50000000

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                       \
    cudaError_t e = (call);                                                                                \
    if (e != cudaSuccess) {                                                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));              \
        exit(EXIT_FAILURE);                                                                                \
    }                                                                                                      \
} while (0)

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float x = X[i];
        // __expf 快速数学：~3 条指令 vs expf 的 ~20+ 条
        Y[i] = 1.0f / (1.0f + __expf(-x));
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 50000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB per vector)\n", N, bytes / 1e6);

    // ---- host 端分配与初始化 ----
    float* hIn = (float*)malloc(bytes);
    float* hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f; // [-100, 100)
    }

    // ---- device 端分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // ---- grid 规模：SM 数 × 4 ----
    int threads = 256;
    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;
    printf("launch: blocks=%d  threads=%d  (SM=%d)\n", blocks, threads, num_sm);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    sigmoid_kernel<<<blocks, threads>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < N; ++i) {
        float ref = 1.0f / (1.0f + expf(-hIn[i]));
        if (fabsf(hOut[i] - ref) > 1e-4f) {
            if (++err <= 5)
                printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], ref);
        }
    }
    printf("verify: %s  (%d / %d mismatch)\n", err ? "FAIL" : "PASS", err, N);

    // ---- 带宽估算：读 X + 写 Y = 2 × bytes ----
    size_t rw_bytes = 2 * bytes;
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hOut);
    return 0;
}

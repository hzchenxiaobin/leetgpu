// 63-interleave.cu —— grid-stride loop 实现数组交错，支持 N 远大于 grid*block
// 编译命令: nvcc -O3 -arch=sm_120 63-interleave.cu -o interleave
// 运行:     ./interleave 25000000

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

__global__ void interleave_kernel(const float* A, const float* B, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        output[2 * i]     = A[i];
        output[2 * i + 1] = B[i];
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 25000000;
    size_t bytes_in  = (size_t)N * sizeof(float);
    size_t bytes_out = (size_t)(2 * N) * sizeof(float);
    printf("N = %d  (in %.1f MB each, out %.1f MB)\n", N, bytes_in / 1e6, bytes_out / 1e6);

    // ---- host 端分配与初始化 ----
    float* hA = (float*)malloc(bytes_in);
    float* hB = (float*)malloc(bytes_in);
    float* hO = (float*)malloc(bytes_out);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hA[i] = (float)(rand() % 10000) / 100.0f;
        hB[i] = (float)(rand() % 10000) / 100.0f;
    }

    // ---- device 端分配与拷贝 ----
    float *dA, *dB, *dO;
    CHECK_CUDA(cudaMalloc(&dA, bytes_in));
    CHECK_CUDA(cudaMalloc(&dB, bytes_in));
    CHECK_CUDA(cudaMalloc(&dO, bytes_out));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes_in, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes_in, cudaMemcpyHostToDevice));

    // ---- 选择 grid 规模：SM 数 × 4，让 grid-stride 发挥作用 ----
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
    interleave_kernel<<<blocks, threads>>>(dA, dB, dO, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hO, dO, bytes_out, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(hO[2 * i] - hA[i]) > 1e-5f || fabsf(hO[2 * i + 1] - hB[i]) > 1e-5f) {
            if (++err <= 5)
                printf("MISMATCH @%d: got (%f,%f), expect (%f,%f)\n",
                       i, hO[2*i], hO[2*i+1], hA[i], hB[i]);
        }
    }
    printf("verify: %s  (%d / %d mismatch)\n", err ? "FAIL" : "PASS", err, N);

    // ---- 带宽估算：读 A + 读 B + 写 output = 2*4*N + 2*4*N = 4N*4 字节 ----
    size_t rw_bytes = 2 * bytes_in + bytes_out; // = 4 * N * 4
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dO));
    free(hA);
    free(hB);
    free(hO);
    return 0;
}

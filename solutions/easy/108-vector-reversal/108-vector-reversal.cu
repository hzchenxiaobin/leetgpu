// 108-vector-reversal.cu —— Vector Reversal（coalesced 逆序映射），含验证与带宽测量
// 编译命令: nvcc -O3 -arch=sm_120 108-vector-reversal.cu -o vector_reversal
// 运行:     ./vector_reversal 10000000

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define CHECK_CUDA(call)                                                                               \
    do {                                                                                               \
        cudaError_t e = (call);                                                                        \
        if (e != cudaSuccess) {                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));      \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

__global__ void reverse_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = input[N - 1 - i];
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    float* hIn = (float*)malloc(bytes);
    float* hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i)
        hIn[i] = (float)(rand() % 10000) / 100.0f;

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    reverse_kernel<<<gridSize, BLOCK_SIZE>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    printf("I/O bandwidth: %.1f GB/s\n", (2.0 * bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    int err = 0;
    for (int i = 0; i < N; ++i) {
        float ref = hIn[N - 1 - i];
        if (fabsf(hOut[i] - ref) > 1e-5f) {
            if (++err <= 5)
                printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], ref);
        }
    }
    printf("verify: %s  (%d / %d mismatch)\n", err ? "FAIL" : "PASS", err, N);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hOut);
    return 0;
}

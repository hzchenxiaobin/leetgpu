// 19-reverse-array.cu —— In-place array reversal
// 编译命令: nvcc -O3 -arch=sm_120 19-reverse-array.cu -o reverse_array
// 运行:     ./reverse_array 1000000

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

// In-place reversal: 每个 thread 交换一对对称元素，只处理前半段
__global__ void reverse_kernel(float* input, int N) {
    int half = N / 2;
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < half; i += stride) {
        float tmp = input[i];
        input[i] = input[N - 1 - i];
        input[N - 1 - i] = tmp;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N=%d (%.1f MB)\n", N, bytes / 1e6);

    // ---- host ----
    float* hIn = (float*)malloc(bytes);
    float* hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i)
        hIn[i] = (float)(rand() % 10000) / 10.0f;
    // CPU 参考
    memcpy(hRef, hIn, bytes);
    for (int i = 0; i < N / 2; ++i) {
        float tmp = hRef[i];
        hRef[i] = hRef[N - 1 - i];
        hRef[N - 1 - i] = tmp;
    }

    // ---- device ----
    float* dIn;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    int half = N / 2;
    int blocks = (half + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = (blocks > 65535) ? 65535 : blocks; // 限制 grid 大小

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    reverse_kernel<<<blocks, BLOCK_SIZE>>>(dIn, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 读写各 N/2 对 = N 元素 × 4B × 2（读+写）= N×8B
    float bw_gbs = ((float)N * 2 * sizeof(float)) / 1e9 / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证 ----
    CHECK_CUDA(cudaMemcpy(hIn, dIn, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < N && err < 5; ++i) {
        if (fabsf(hIn[i] - hRef[i]) > 1e-5f) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, hIn[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dIn));
    free(hIn);
    free(hRef);
    return 0;
}

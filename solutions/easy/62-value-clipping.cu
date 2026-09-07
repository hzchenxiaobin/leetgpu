// 62-value-clipping.cu —— 无分支 clamp with grid-stride loop
// 编译命令: nvcc -O3 -arch=sm_120 62-value-clipping.cu -o clip
// 运行:     ./clip

#include <cstdio>
#include <cstdlib>
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

// 无分支 clamp kernel
__global__ void clip_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             float lo, float hi, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        float y = fmaxf(x, lo);   // 下界：无分支
        y = fminf(y, hi);          // 上界：无分支
        output[i] = y;
    }
}

// ---- CPU 参考 ----
void clip_cpu(const float* input, float* output, float lo, float hi, int N) {
    for (int i = 0; i < N; i++) {
        float x = input[i];
        output[i] = x < lo ? lo : (x > hi ? hi : x);
    }
}

int main() {
    // 题目 example
    int N = 4;
    float lo = 0.0f, hi = 3.5f;
    float hIn[] = {1.5f, -2.0f, 3.0f, 4.5f};
    float hOut[4], hRef[4];
    printf("Value Clipping: lo=%.1f hi=%.1f N=%d\n", lo, hi, N);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, lo, hi, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hOut, dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
    clip_cpu(hIn, hRef, lo, hi, N);

    printf("input  = [%.1f, %.1f, %.1f, %.1f]\n", hIn[0], hIn[1], hIn[2], hIn[3]);
    printf("output = [%.1f, %.1f, %.1f, %.1f]\n", hOut[0], hOut[1], hOut[2], hOut[3]);
    int err = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(hOut[i] - hRef[i]) > 1e-5f) err++;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 ----
    printf("\n--- Perf test (N=100K) ---\n");
    N = 100000;
    lo = -51.24f; hi = 39.51f;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));
    float* hTemp = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) hTemp[i] = (float)(rand() % 200000 - 100000) / 100.0f;
    CHECK_CUDA(cudaMemcpy(dIn, hTemp, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, lo, hi, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 带宽估算：读 4B + 写 4B = 8B/元素
    size_t bytes = (size_t)N * 8;
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    free(hTemp);
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}

// 65-geglu.cu —— 融合 GeGLU kernel with erff
// 编译命令: nvcc -O3 -arch=sm_120 65-geglu.cu -o geglu
// 运行:     ./geglu

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
#define INV_SQRT_2 0.70710678f  // 1/√2，编译期内联

// 融合 GeGLU kernel：split + GELU + multiply 在单线程内完成
__global__ void geglu_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int halfN) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // grid-stride loop
    for (int i = tid; i < halfN; i += stride) {
        float x1 = input[i];              // 前半：直通
        float x2 = input[i + halfN];      // 后半：门控输入
        // GELU(x2) = 0.5 * x2 * (1 + erf(x2 / √2))
        float gelu = 0.5f * x2 * (1.0f + erff(x2 * INV_SQRT_2));
        // 融合乘法
        output[i] = x1 * gelu;
    }
}

// ---- CPU 参考 ----
void geglu_cpu(const float* input, float* output, int N) {
    int halfN = N / 2;
    for (int i = 0; i < halfN; i++) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float gelu = 0.5f * x2 * (1.0f + erff(x2 / 1.41421356f));
        output[i] = x1 * gelu;
    }
}

int main() {
    // 题目 example
    int N = 4;
    float hIn[] = {2.0f, -1.0f, 1.0f, 0.5f};
    float hOut[2], hRef[2];
    printf("GeGLU: N=%d\n", N);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (N / 2) * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice));

    int halfN = N / 2;
    int blocks = (halfN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geglu_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, halfN);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hOut, dOut, halfN * sizeof(float), cudaMemcpyDeviceToHost));
    geglu_cpu(hIn, hRef, N);

    printf("input  = [%.1f, %.1f, %.1f, %.1f]\n", hIn[0], hIn[1], hIn[2], hIn[3]);
    printf("output = [%.7f, %.7f]\n", hOut[0], hOut[1]);
    printf("expect = [1.6826895, -0.3457312]\n");
    int err = 0;
    for (int i = 0; i < halfN; i++)
        if (fabsf(hOut[i] - hRef[i]) > 1e-4f) err++;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 ----
    printf("\n--- Perf test (N=1M) ---\n");
    N = 1000000;
    halfN = N / 2;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, halfN * sizeof(float)));
    float* hTemp = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) hTemp[i] = (float)(rand() % 20000 - 10000) / 100.0f;
    CHECK_CUDA(cudaMemcpy(dIn, hTemp, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    blocks = (halfN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    geglu_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, halfN);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 带宽估算：读 8B/输出元素 + 写 4B/输出元素 = 12B/输出元素 = 6B/input元素
    size_t bytes = (size_t)N * 6;  // N 个 input float，每个有效贡献 6B IO
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    free(hTemp);
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}

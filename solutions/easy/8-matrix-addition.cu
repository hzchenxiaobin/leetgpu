// 8-matrix-addition.cu —— Matrix Addition：1D grid-stride + float4 向量化
// 编译命令: nvcc -O3 -arch=sm_120 8-matrix-addition.cu -o matadd
// 运行:     ./matadd 4096 4096

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

__global__ void matadd_kernel(const float* A, const float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int vec_count = N / 4; // float4 元素数（N 是 4 的倍数时）

    // ---- ① float4 向量化主循环 ----
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    for (int i = tid; i < vec_count; i += stride) {
        float4 a = A4[i]; // 1 条 16B load
        float4 b = B4[i]; // 1 条 16B load
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C4[i] = c; // 1 条 16B store
    }

    // ---- ② 尾部：处理 N%4 个剩余元素（本题 N=4096%4=0，通常不执行）----
    int tail_start = vec_count * 4;
    for (int i = tail_start + tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int num = M * N;
    size_t bytes = (size_t)num * sizeof(float);
    printf("M=%d, N=%d  (%.1f MB per matrix)\n", M, N, bytes / 1e6);

    // ---- host ----
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < num; ++i) {
        hA[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
        hB[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    }

    // ---- device ----
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // ---- launch（grid-stride：block 数不必等于元素数，限制上限即可）----
    int blocks = min((num / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    printf("launch: blocks=%d threads=%d\n", blocks, BLOCK_SIZE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matadd_kernel<<<blocks, BLOCK_SIZE>>>(dA, dB, dC, num);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 带宽 ----
    float bw_gbs = (3.0f * bytes / 1e9) / (ms / 1e3); // 读 A+B + 写 C
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证 ----
    CHECK_CUDA(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < num; ++i) {
        if (fabsf(hC[i] - (hA[i] + hB[i])) > 1e-5f) {
            if (++err <= 5)
                printf("MISMATCH @%d: got %f, expect %f\n", i, hC[i], hA[i] + hB[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    return 0;
}

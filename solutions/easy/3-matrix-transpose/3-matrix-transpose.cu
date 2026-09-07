// 3-matrix-transpose.cu —— Shared memory tiling + bank conflict padding，含验证与带宽测量
// 编译命令: nvcc -O3 -arch=sm_120 3-matrix-transpose.cu -o matrix_transpose
// 运行:     ./matrix_transpose 4096

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32

#define CHECK_CUDA(call)                                                                               \
    do {                                                                                               \
        cudaError_t e = (call);                                                                        \
        if (e != cudaSuccess) {                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));      \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

__global__ void transpose_kernel(const float* src, float* dst, int M, int N) {
    __shared__ float smem[TILE][TILE + 1]; // +1 padding 避免 bank conflict

    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    // 读 src（按 row，coalesced）→ 写 smem
    if (i < M && j < N)
        smem[threadIdx.y][threadIdx.x] = src[i * N + j];
    __syncthreads();

    // 读 smem（按 col，转置）→ 写 dst（按 row，coalesced）
    int j_out = blockIdx.x * TILE + threadIdx.y;
    int i_out = blockIdx.y * TILE + threadIdx.x;
    if (j_out < N && i_out < M)
        dst[j_out * M + i_out] = smem[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : M;
    size_t bytes = (size_t)M * N * sizeof(float);
    printf("M=%d N=%d (%.1f MB)\n", M, N, bytes / 1e6);

    float* hSrc = (float*)malloc(bytes);
    float* hDst = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * N; i++)
        hSrc[i] = (float)(rand() % 1000) / 10.0f;

    float *dSrc, *dDst;
    CHECK_CUDA(cudaMalloc(&dSrc, bytes));
    CHECK_CUDA(cudaMalloc(&dDst, bytes));
    CHECK_CUDA(cudaMemcpy(dSrc, hSrc, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    transpose_kernel<<<grid, block>>>(dSrc, dDst, M, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    printf("I/O bandwidth: %.1f GB/s\n", (2.0 * bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaMemcpy(hDst, dDst, bytes, cudaMemcpyDeviceToHost));

    int fail = 0;
    for (int i = 0; i < M && !fail; i++)
        for (int j = 0; j < N; j++)
            if (fabsf(hDst[j * M + i] - hSrc[i * N + j]) > 1e-5f) {
                printf("FAIL at (%d,%d)\n", i, j);
                fail = 1;
                break;
            }
    printf("%s\n", fail ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dSrc));
    CHECK_CUDA(cudaFree(dDst));
    free(hSrc);
    free(hDst);
    return 0;
}

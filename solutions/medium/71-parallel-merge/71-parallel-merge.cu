// 71-parallel-merge.cu —— Co-rank 二分搜索 + Block 分块并行归并
// 编译命令: nvcc -O3 -arch=sm_80 71-parallel-merge.cu -o parallel_merge

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE 256

// co-rank: 对输出位置 k，返回 A 中应贡献的元素数 i
// 二分搜索 i ∈ [max(0,k-N), min(k,M)]，使 A[0..i-1] ⊕ B[0..k-i-1] = C[0..k-1]
__device__ __forceinline__ int co_rank(int k, const float* A, int M,
                                       const float* B, int N) {
    int i_low = max(0, k - N);
    int i_high = min(k, M);
    while (i_low <= i_high) {
        int i = (i_low + i_high) >> 1;
        int j = k - i;
        if (i > 0 && j < N && A[i - 1] > B[j]) {
            i_high = i - 1;      // A 贡献太多，减小 i
        } else if (j > 0 && i < M && B[j - 1] > A[i]) {
            i_low = i + 1;       // A 贡献太少，增大 i
        } else {
            return i;            // 找到
        }
    }
    return i_low;
}

__global__ void parallel_merge_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N) {
    int tile_start = blockIdx.x * TILE;
    int total = M + N;
    if (tile_start >= total) return;

    int tile_end = min(tile_start + TILE, total);

    // ===== Step 1: 全局 co-rank（定位 tile 边界）=====
    int i_start = co_rank(tile_start, A, M, B, N);
    int j_start = tile_start - i_start;
    int i_end   = co_rank(tile_end, A, M, B, N);
    int j_end   = tile_end - i_end;

    int a_len = i_end - i_start;
    int b_len = j_end - j_start;

    // ===== Step 2: 合并加载到 shared memory =====
    __shared__ float shared_A[TILE + 1];
    __shared__ float shared_B[TILE + 1];

    for (int t = threadIdx.x; t < a_len; t += blockDim.x)
        shared_A[t] = A[i_start + t];
    for (int t = threadIdx.x; t < b_len; t += blockDim.x)
        shared_B[t] = B[j_start + t];
    __syncthreads();

    // ===== Step 3: 每 thread 局部 co-rank + 写一个输出 =====
    int local_k = threadIdx.x;
    if (tile_start + local_k < total) {
        int local_i = co_rank(local_k, shared_A, a_len, shared_B, b_len);
        int local_j = local_k - local_i;
        float a_val = (local_i < a_len) ? shared_A[local_i] : INFINITY;
        float b_val = (local_j < b_len) ? shared_B[local_j] : INFINITY;
        C[tile_start + local_k] = (a_val <= b_val) ? a_val : b_val;
    }
}

int cmpfloat(const void* a, const void* b) {
    float fa = *(const float*)a, fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

// ===== Host 端 =====
int main() {
    // 功能测试: A=[1,3,5,7], B=[2,4,6,8]
    int M = 4, N = 4;
    float h_A[] = {1.0f, 3.0f, 5.0f, 7.0f};
    float h_B[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float h_C[8];

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, (M + N) * sizeof(float));
    cudaMemcpy(d_A, h_A, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (M + N + TILE - 1) / TILE;
    parallel_merge_kernel<<<blocks, TILE>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, (M + N) * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Functional Test ===\n");
    printf("A = [1, 3, 5, 7], B = [2, 4, 6, 8]\n");
    printf("C = [");
    for (int i = 0; i < M + N; i++) printf("%.0f%s", h_C[i], i < M+N-1 ? ", " : "");
    printf("]\n");
    float ref[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int pass = 1;
    for (int i = 0; i < M + N; i++)
        if (h_C[i] != ref[i]) pass = 0;
    printf("%s\n\n", pass ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: M=N=25M =====
    int M2 = 25000000, N2 = 25000000;
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A2, (size_t)M2 * sizeof(float));
    cudaMalloc(&d_B2, (size_t)N2 * sizeof(float));
    cudaMalloc(&d_C2, (size_t)(M2 + N2) * sizeof(float));

    float *hA2 = (float*)malloc((size_t)M2 * sizeof(float));
    float *hB2 = (float*)malloc((size_t)N2 * sizeof(float));
    srand(42);
    for (int i = 0; i < M2; i++) hA2[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
    for (int i = 0; i < N2; i++) hB2[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
    // 排序
    qsort(hA2, M2, sizeof(float), (int(*)(const void*,const void*))cmpfloat);
    qsort(hB2, N2, sizeof(float), (int(*)(const void*,const void*))cmpfloat);

    cudaMemcpy(d_A2, hA2, (size_t)M2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, hB2, (size_t)N2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blocks2 = (M2 + N2 + TILE - 1) / TILE;
    cudaEventRecord(start);
    parallel_merge_kernel<<<blocks2, TILE>>>(d_A2, d_B2, d_C2, M2, N2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("=== Perf Test (M=%d, N=%d) ===\n", M2, N2);
    printf("Blocks = %d, TILE = %d\n", blocks2, TILE);
    printf("Kernel time = %.3f ms\n", ms);
    size_t bytes = ((size_t)M2 + N2 + (M2 + N2)) * sizeof(float);
    printf("Data traffic = %.2f MB (read A+B + write C)\n", bytes / 1e6);
    printf("Effective bandwidth = %.2f GB/s\n", bytes / (ms * 1e6));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(hA2); free(hB2);
    return 0;
}

// 75-sparse-matrix-dense-matrix-multiplication.cu —— SpMM: 稀疏 A × 稠密 B，CSR 遍历 + warp 分摊 K 维
// 编译命令: nvcc -O3 -arch=sm_80 75-sparse-matrix-dense-matrix-multiplication.cu -o spmm

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

// SpMM kernel: 一个 warp 处理 A 的一行
// A 以密集格式传入（含零），kernel 内跳过零元素
__global__ void spmm_kernel(
    const float* __restrict__ A,  // [M, N] dense (含零)
    const float* __restrict__ B,  // [N, K] dense
    float* __restrict__ C,        // [M, K] dense output
    int M, int N, int K)
{
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    if (row >= M) return;

    // 每 thread 负责的 K 列范围
    int k_per_thread = (K + WARP_SIZE - 1) / WARP_SIZE;
    int k_start = lane * k_per_thread;
    int k_end = min(k_start + k_per_thread, K);

    // Register 累加器
    float acc[64];  // 假设 K_per_thread <= 64
    for (int i = 0; i < k_per_thread && (k_start + i) < K; i++)
        acc[i] = 0.0f;

    // 遍历 A 的第 row 行，跳过零元素
    const float* a_row = A + (size_t)row * N;
    for (int col = 0; col < N; col++) {
        float val = a_row[col];
        if (val == 0.0f) continue;  // 跳过零元素

        // C[row, k_start..k_end] += val * B[col, k_start..k_end]
        const float* b_row = B + (size_t)col * K;
        for (int i = 0; i < k_per_thread && (k_start + i) < K; i++) {
            acc[i] += val * b_row[k_start + i];
        }
    }

    // 写回 C
    float* c_row = C + (size_t)row * K;
    for (int i = 0; i < k_per_thread && (k_start + i) < K; i++) {
        c_row[k_start + i] = acc[i];
    }
}

// ===== Host 端 =====
int main() {
    // 功能测试: A(3×4) × B(4×2) = C(3×2)
    int M = 3, N = 4, K = 2;
    float h_A[] = {2, 0, 0, 1,  0, 3, 0, 0,  0, 0, 4, 0};
    float h_B[] = {1, 2,  3, 4,  5, 6,  7, 8};
    float h_C[6] = {0};
    float ref_C[] = {9, 12, 9, 12, 20, 24};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    spmm_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Functional Test ===\n");
    printf("A = [[2,0,0,1], [0,3,0,0], [0,0,4,0]]\n");
    printf("B = [[1,2], [3,4], [5,6], [7,8]]\n");
    printf("C = [");
    for (int i = 0; i < M; i++) {
        printf("[%.0f, %.0f]%s", h_C[i*K], h_C[i*K+1], i < M-1 ? ", " : "");
    }
    printf("]\n");
    int pass = 1;
    for (int i = 0; i < M * K; i++)
        if (fabsf(ref_C[i] - h_C[i]) > 0.001) pass = 0;
    printf("%s\n\n", pass ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: M=4096, N=2048, K=512 =====
    int M2 = 4096, N2 = 2048, K2 = 512;
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A2, (size_t)M2 * N2 * sizeof(float));
    cudaMalloc(&d_B2, (size_t)N2 * K2 * sizeof(float));
    cudaMalloc(&d_C2, (size_t)M2 * K2 * sizeof(float));

    float *hA2 = (float*)malloc((size_t)M2 * N2 * sizeof(float));
    float *hB2 = (float*)malloc((size_t)N2 * K2 * sizeof(float));
    srand(42);
    int nnz = 0;
    for (size_t i = 0; i < (size_t)M2 * N2; i++) {
        hA2[i] = (rand() % 100 < 35) ? (-1.0f + 2.0f * (rand() / (float)RAND_MAX)) : 0.0f;
        if (hA2[i] != 0.0f) nnz++;
    }
    for (size_t i = 0; i < (size_t)N2 * K2; i++)
        hB2[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);

    cudaMemcpy(d_A2, hA2, (size_t)M2 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, hB2, (size_t)N2 * K2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blocks2 = (M2 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaEventRecord(start);
    spmm_kernel<<<blocks2, BLOCK_SIZE>>>(d_A2, d_B2, d_C2, M2, N2, K2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("=== Perf Test (M=%d, N=%d, K=%d) ===\n", M2, N2, K2);
    printf("nnz = %d (%.1f%% sparse)\n", nnz, 100.0 * (1.0 - (double)nnz / (M2 * N2)));
    printf("Kernel time = %.3f ms\n", ms);
    printf("FMA count: dense=%zu, sparse=%zu (saved %.0f%%)\n",
           (size_t)M2 * N2 * K2, (size_t)nnz * K2,
           100.0 * (1.0 - (double)nnz / (M2 * N2)));
    size_t bytes = (size_t)M2 * N2 * 4 + (size_t)N2 * K2 * 4 + (size_t)M2 * K2 * 4;
    printf("HBM traffic (dense A+B+C) = %.2f MB\n", bytes / 1e6);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(hA2); free(hB2);
    return 0;
}

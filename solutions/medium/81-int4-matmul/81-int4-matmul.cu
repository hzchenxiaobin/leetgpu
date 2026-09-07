// 81-int4-matmul.cu —— W4A16 量化矩阵乘：即时反量化 + Tiled GEMM + FP32 累加
// 编译命令: nvcc -O3 -arch=sm_80 81-int4-matmul.cu -o int4_matmul

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#define BM 64
#define BN 64
#define BK 32
#define RM 4
#define RN 4
#define WARP_SIZE 32

// 即时反量化的 tiled GEMM kernel
__global__ void int4_matmul_kernel(
    const __half* __restrict__ x,    // [M, K] FP16
    const uint8_t* __restrict__ w_q, // [N, K/2] UINT8 (packed INT4)
    const __half* __restrict__ scales, // [N, K/gs] FP16
    __half* __restrict__ y,          // [M, N] FP16
    int M, int N, int K, int group_size)
{
    int bm = blockIdx.x * BM;
    int bn = blockIdx.y * BN;

    __shared__ __half s_x[BM][BK];       // 激活 tile
    __shared__ __half s_w[BN][BK];       // 反量化后的权重 tile

    // 每 thread 持有的累加器
    float acc[RM][RN];
    for (int i = 0; i < RM; i++)
        for (int j = 0; j < RN; j++)
            acc[i][j] = 0.0f;

    int n_groups = K / group_size;

    // K 维迭代
    for (int bk = 0; bk < K; bk += BK) {
        // ===== ① 加载 x tile 到 shared =====
        for (int i = threadIdx.x; i < BM; i += blockDim.x) {
            for (int j = 0; j < BK; j++) {
                int m = bm + i;
                int k = bk + j;
                s_x[i][j] = (m < M && k < K) ? x[m * K + k] : __float2half(0.0f);
            }
        }

        // ===== ② 加载 w_q 并即时反量化到 shared =====
        for (int j = threadIdx.x; j < BN; j += blockDim.x) {
            for (int i = 0; i < BK; i++) {
                int n = bn + j;
                int k = bk + i;
                if (n < N && k < K) {
                    // 解包 nibble
                    uint8_t byte = w_q[n * (K / 2) + k / 2];
                    int nibble = (k % 2 == 0) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                    int signed_val = nibble - 8;
                    // 乘 group scale
                    __half scale = scales[n * n_groups + k / group_size];
                    float w = (float)signed_val * __half2float(scale);
                    s_w[j][i] = __float2half(w);
                } else {
                    s_w[j][i] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ===== ③ Register tiling + FP32 累加 =====
        for (int kk = 0; kk < BK; kk++) {
            // 每 thread 负责 RM 行
            float reg_x[RM];
            for (int i = 0; i < RM; i++)
                reg_x[i] = __half2float(s_x[threadIdx.x * RM / blockDim.x * BM / (blockDim.x / WARP_SIZE)][kk]);
            // 简化版：直接遍历
            for (int i = 0; i < RM; i++) {
                int row = (threadIdx.x / (BN / RN)) * RM + i;
                if (row < BM) {
                    float xv = __half2float(s_x[row][kk]);
                    for (int j = 0; j < RN; j++) {
                        int col = (threadIdx.x % (BN / RN)) * RN + j;
                        if (col < BN) {
                            float wv = __half2float(s_w[col][kk]);
                            acc[i][j] += xv * wv;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // ===== ④ 写回 y（FP32 → FP16）=====
    for (int i = 0; i < RM; i++) {
        for (int j = 0; j < RN; j++) {
            int row = bm + (threadIdx.x / (BN / RN)) * RM + i;
            int col = bn + (threadIdx.x % (BN / RN)) * RN + j;
            if (row < M && col < N) {
                y[row * N + col] = __float2half(acc[i][j]);
            }
        }
    }
}

// ===== Host 端 =====
int main() {
    // 功能测试: M=2, N=4, K=4, gs=2
    int M = 2, N = 4, K = 4, gs = 2;
    __half h_x[] = {
        __float2half(1.0f), __float2half(0.0f), __float2half(1.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(1.0f), __float2half(0.0f), __float2half(1.0f)
    };
    uint8_t h_wq[] = {0x99, 0x99, 0xAA, 0xAA, 0x77, 0x77, 0x88, 0x88};
    __half h_scales[8];
    for (int i = 0; i < 8; i++) h_scales[i] = __float2half(0.5f);
    __half h_y[8];

    __half *d_x; uint8_t *d_wq; __half *d_scales, *d_y;
    cudaMalloc(&d_x, M * K * sizeof(__half));
    cudaMalloc(&d_wq, N * (K/2) * sizeof(uint8_t));
    cudaMalloc(&d_scales, N * (K/gs) * sizeof(__half));
    cudaMalloc(&d_y, M * N * sizeof(__half));
    cudaMemcpy(d_x, h_x, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wq, h_wq, N * (K/2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, h_scales, N * (K/gs) * sizeof(__half), cudaMemcpyHostToDevice);

    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(256);
    int4_matmul_kernel<<<grid, block>>>(d_x, d_wq, d_scales, d_y, M, N, K, gs);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, M * N * sizeof(__half), cudaMemcpyHostToHost);

    printf("=== Functional Test ===\n");
    printf("Expected: [1, 2, -1, 0, 1, 2, -1, 0]\n");
    printf("Got:      [");
    for (int i = 0; i < M * N; i++) printf("%.1f%s", __half2float(h_y[i]), i < M*N-1 ? ", " : "");
    printf("]\n");

    // CPU 参考
    float ref[8];
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                uint8_t byte = h_wq[n * (K/2) + k/2];
                int nibble = (k % 2 == 0) ? (byte >> 4) & 0xF : byte & 0xF;
                float w = (float)(nibble - 8) * __half2float(h_scales[n * (K/gs) + k/gs]);
                sum += __half2float(h_x[m * K + k]) * w;
            }
            ref[m * N + n] = sum;
        }
    int pass = 1;
    for (int i = 0; i < M * N; i++)
        if (fabsf(ref[i] - __half2float(h_y[i])) > 0.01) pass = 0;
    printf("%s\n\n", pass ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: M=N=K=4096, gs=128 =====
    int M2 = 4096, N2 = 4096, K2 = 4096, gs2 = 128;
    __half *d_x2; uint8_t *d_wq2; __half *d_s2, *d_y2;
    cudaMalloc(&d_x2, (size_t)M2 * K2 * sizeof(__half));
    cudaMalloc(&d_wq2, (size_t)N2 * (K2/2));
    cudaMalloc(&d_s2, (size_t)N2 * (K2/gs2) * sizeof(__half));
    cudaMalloc(&d_y2, (size_t)M2 * N2 * sizeof(__half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 grid2((M2+BM-1)/BM, (N2+BN-1)/BN);
    cudaEventRecord(start);
    int4_matmul_kernel<<<grid2, block>>>(d_x2, d_wq2, d_s2, d_y2, M2, N2, K2, gs2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("=== Perf Test (M=N=K=%d, gs=%d) ===\n", M2, gs2);
    printf("Kernel time = %.3f ms\n", ms);
    // HBM: read x(M*K*2) + w_q(N*K/2) + scales(small) + write y(M*N*2)
    size_t bytes = (size_t)M2*K2*2 + (size_t)N2*(K2/2) + (size_t)M2*N2*2;
    printf("HBM traffic ≈ %.2f MB (x + w_q + y)\n", bytes / 1e6);
    printf("Effective bandwidth = %.2f GB/s\n", bytes / (ms * 1e6));
    printf("INT4 saves %.2f MB vs FP16 GEMM\n",
           ((size_t)N2 * K2 * 2 - (size_t)N2 * (K2/2)) / 1e6);

    cudaFree(d_x); cudaFree(d_wq); cudaFree(d_scales); cudaFree(d_y);
    cudaFree(d_x2); cudaFree(d_wq2); cudaFree(d_s2); cudaFree(d_y2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}

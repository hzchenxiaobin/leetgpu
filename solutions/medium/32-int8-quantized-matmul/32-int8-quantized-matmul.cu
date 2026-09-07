// 32-int8-quantized-matmul.cu —— INT8 Quantized MatMul（tiled GEMM + INT32 累加 + requantize）
// 编译命令: nvcc -O3 -arch=sm_120 32-int8-quantized-matmul.cu -o int8_matmul
// 运行:     ./int8_matmul

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32

// tiled INT8 GEMM：grid((N+TILE-1)/TILE, (M+TILE-1)/TILE), block(TILE, TILE)
__global__ void int8_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C,
                                   int M, int N, int K,
                                   float scale_A, float scale_B, float scale_C,
                                   int zp_A, int zp_B, int zp_C) {
    int row = blockIdx.y * TILE + threadIdx.y;   // M 维
    int col = blockIdx.x * TILE + threadIdx.x;   // N 维

    __shared__ int8_t sA[TILE][TILE];
    __shared__ int8_t sB[TILE][TILE];

    int32_t acc = 0;

    // 沿 K 方向分 tile 累加
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : (int8_t)0;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : (int8_t)0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            int a = (int)sA[threadIdx.y][k] - zp_A;
            int b = (int)sB[k][threadIdx.x] - zp_B;
            acc += a * b;
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // requantize：与 reference 浮点顺序一致 ((acc * sA) * sB) / sC
        float cf = (float)acc * scale_A * scale_B / scale_C;
        int q = (int)rintf(cf) + zp_C;
        if (q < -128) q = -128;
        if (q > 127)  q = 127;
        C[row * N + col] = (int8_t)q;
    }
}

int main() {
    int M = 64, N = 64, K = 128;
    float sA = 0.1f, sB = 0.2f, sC = 0.05f;
    int zpA = 0, zpB = 0, zpC = 0;

    std::vector<int8_t> h_A(M * K), h_B(K * N), h_C(M * N);
    srand(42);
    for (auto& x : h_A) x = (int8_t)(rand() % 256 - 128);
    for (auto& x : h_B) x = (int8_t)(rand() % 256 - 128);

    int8_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int8_t));
    cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N, cudaMemcpyHostToDevice);

    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    int8_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, sA, sB, sC, zpA, zpB, zpC);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(), d_C, M * N, cudaMemcpyDeviceToHost);

    // CPU 验证：INT32 累加 + rintf
    bool pass = true;
    for (int i = 0; i < M && pass; i++)
        for (int j = 0; j < N && pass; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += ((int)h_A[i * K + k] - zpA) * ((int)h_B[k * N + j] - zpB);
            float cf = (float)acc * sA * sB / sC;
            int q = (int)rintf(cf) + zpC;
            q = q < -128 ? -128 : (q > 127 ? 127 : q);
            if (q != (int)h_C[i * N + j]) {
                printf("Mismatch at (%d,%d): cpu=%d gpu=%d\n", i, j, q, (int)h_C[i * N + j]);
                pass = false;
            }
        }
    printf("M=%d N=%d K=%d, %s\n", M, N, K, pass ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

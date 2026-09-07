// 64-weight-dequantization.cu —— Weight Dequantization（element-wise, memory-bound）
// 编译命令: nvcc -O3 -arch=sm_120 64-weight-dequantization.cu -o dequant -lineinfo
// 运行:     ./dequant 8192 8192 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// ---------- fused kernel ----------
__global__ void weight_dequant_kernel(const float* __restrict__ X, // (M, N)
                                      const float* __restrict__ S, // (s_rows, s_cols)
                                      float* __restrict__ Y,       // (M, N)
                                      int M, int N, int T) {

    int s_cols = (N + T - 1) / T;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid; idx < M * N; idx += stride) {
        int i = idx / N;
        int j = idx % N;
        int sr = i / T;
        int sc = j / T;
        float scale = S[sr * s_cols + sc];
        Y[idx] = X[idx] * scale;
    }
}

// ---------- CPU 参考 ----------
void dequant_cpu(const float* X, const float* S, float* Y, int M, int N, int T) {
    int s_cols = (N + T - 1) / T;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            Y[i * N + j] = X[i * N + j] * S[(i / T) * s_cols + (j / T)];
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 8192;
    int T = (argc > 3) ? atoi(argv[3]) : 128;
    int s_rows = (M + T - 1) / T, s_cols = (N + T - 1) / T;
    printf("M=%d N=%d T=%d  s_rows=%d s_cols=%d\n", M, N, T, s_rows, s_cols);

    size_t xy_bytes = (size_t)M * N * sizeof(float);
    size_t s_bytes = (size_t)s_rows * s_cols * sizeof(float);
    printf("X+Y = %.2f MB (%.2f MB each), S = %.2f KB\n", 2.0 * xy_bytes / 1e6, xy_bytes / 1e6, s_bytes / 1024.0);

    std::vector<float> hX(M * N), hS(s_rows * s_cols), hY(M * N), hRef(M * N);
    srand(42);
    for (auto& x : hX)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hS)
        x = 0.5f + (rand() % 1000) / 1000.f;

    float *dX, *dS, *dY;
    cudaMalloc(&dX, xy_bytes);
    cudaMemcpy(dX, hX.data(), xy_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dS, s_bytes);
    cudaMemcpy(dS, hS.data(), s_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dY, xy_bytes);

    int grid = (M * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // warmup
    weight_dequant_kernel<<<grid, BLOCK_SIZE>>>(dX, dS, dY, M, N, T);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    weight_dequant_kernel<<<grid, BLOCK_SIZE>>>(dX, dS, dY, M, N, T);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(hY.data(), dY, xy_bytes, cudaMemcpyDeviceToHost);
    dequant_cpu(hX.data(), hS.data(), hRef.data(), M, N, T);
    float maxd = 0;
    for (int i = 0; i < M * N; ++i)
        maxd = fmaxf(maxd, fabsf(hY[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-5)\n", maxd, maxd < 1e-5f ? "PASS" : "FAIL");

    // Roofline 分析
    float flops = 1.0f * M * N;                                            // 每元素 1 次乘法
    float bytes = (2.0f * M * N + 1.0f * s_rows * s_cols) * sizeof(float); // 读X+读S+写Y
    float ai = flops * 4 / bytes;                                          // FLOP/Byte
    float achieved_bw = bytes / (ms / 1000) / 1e9;                         // GB/s
    printf("\n[Roofline] FLOPs=%.2fG  Bytes=%.2fMB  AI=%.2f FLOP/Byte\n", flops / 1e9, bytes / 1e6, ai);
    printf("[Roofline] achieved BW = %.1f GB/s (RTX 5090 peak ~1555 GB/s)\n", achieved_bw);
    printf("[Roofline] AI=%.2f ≪ Ridge(12.6) → memory-bound\n", ai);

    cudaFree(dX);
    cudaFree(dS);
    cudaFree(dY);
    return 0;
}

// 84-swiglu-mlp-block.cu —— SwiGLU MLP Block: 3 GEMM + 1 融合 elementwise
// 编译命令: nvcc -O3 -arch=sm_120 84-swiglu-mlp-block.cu -o swiglu_mlp
// 运行:     ./swiglu_mlp

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLOCK 256

// ---- shared memory tiling GEMM: C[M,N] = A[M,K] @ B[K,N] ----
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // ---- 协作加载 A_tile 和 B_tile ----
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        A_tile[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        B_tile[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // ---- 从 shared 读数据做乘加 ----
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ---- 融合 SwiGLU elementwise: hidden = SiLU(gate) * up ----
__global__ void swiglu_fused(const float* gate, const float* up, float* hidden, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float g = gate[i];
        float u = up[i];
        float silu = g / (1.0f + __expf(-g)); // SiLU（register 内）
        hidden[i] = silu * u;                 // 融合乘法（register 内）
    }
}

// ---- 本地自测 ----
int main() {
    // LLaMA-3 8B 缩小版用于快速验证
    int M = 64, d_model = 256, d_ffn = 512;
    size_t x_bytes = (size_t)M * d_model * sizeof(float);
    size_t wg_bytes = (size_t)d_model * d_ffn * sizeof(float);
    size_t wu_bytes = wg_bytes;
    size_t wd_bytes = (size_t)d_ffn * d_model * sizeof(float);
    size_t out_bytes = x_bytes;
    size_t tmp_bytes = (size_t)M * d_ffn * sizeof(float);

    float *h_x = (float*)malloc(x_bytes);
    float *h_wg = (float*)malloc(wg_bytes);
    float *h_wu = (float*)malloc(wu_bytes);
    float *h_wd = (float*)malloc(wd_bytes);
    float *h_out = (float*)malloc(out_bytes);
    float *h_ref = (float*)malloc(out_bytes);
    srand(42);
    for (size_t i = 0; i < M * d_model; ++i) h_x[i] = (rand() % 200 - 100) / 1000.0f;
    for (size_t i = 0; i < d_model * d_ffn; ++i) { h_wg[i] = (rand() % 200 - 100) / 1000.0f; h_wu[i] = (rand() % 200 - 100) / 1000.0f; }
    for (size_t i = 0; i < d_ffn * d_model; ++i) h_wd[i] = (rand() % 200 - 100) / 1000.0f;

    float *d_x, *d_wg, *d_wu, *d_wd, *d_out, *d_gate, *d_up;
    cudaMalloc(&d_x, x_bytes);   cudaMalloc(&d_wg, wg_bytes);
    cudaMalloc(&d_wu, wu_bytes); cudaMalloc(&d_wd, wd_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMalloc(&d_gate, tmp_bytes); cudaMalloc(&d_up, tmp_bytes);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wg, h_wg, wg_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wu, h_wu, wu_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wd, h_wd, wd_bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);

    // ① gate = x @ W_gate
    dim3 grid_gate((d_ffn + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid_gate, threads>>>(d_x, d_wg, d_gate, M, d_ffn, d_model);

    // ② up = x @ W_up
    matmul_tiled<<<grid_gate, threads>>>(d_x, d_wu, d_up, M, d_ffn, d_model);

    // ③ hidden = SiLU(gate) * up（in-place 覆写 d_gate）
    int n_hidden = M * d_ffn;
    swiglu_fused<<<(n_hidden + BLOCK - 1) / BLOCK, BLOCK>>>(d_gate, d_up, d_gate, n_hidden);

    // ④ output = hidden @ W_down
    dim3 grid_down((d_model + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid_down, threads>>>(d_gate, d_wd, d_out, M, d_model, d_ffn);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    // ---- CPU 参考验证 ----
    float *gate = (float*)malloc(tmp_bytes), *up = (float*)malloc(tmp_bytes), *hidden = (float*)malloc(tmp_bytes);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < d_ffn; ++j) {
            float g = 0, u = 0;
            for (int k = 0; k < d_model; ++k) {
                g += h_x[i * d_model + k] * h_wg[k * d_ffn + j];
                u += h_x[i * d_model + k] * h_wu[k * d_ffn + j];
            }
            gate[i * d_ffn + j] = g; up[i * d_ffn + j] = u;
        }
    for (int i = 0; i < n_hidden; ++i)
        hidden[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < d_model; ++j) {
            float s = 0;
            for (int k = 0; k < d_ffn; ++k) s += hidden[i * d_ffn + k] * h_wd[k * d_model + j];
            h_ref[i * d_model + j] = s;
        }

    bool pass = true;
    for (int i = 0; i < M * d_model; ++i)
        if (fabsf(h_out[i] - h_ref[i]) > 1e-3) { pass = false; printf("MISMATCH @%d: got %f expect %f\n", i, h_out[i], h_ref[i]); break; }
    printf("SwiGLU MLP Block M=%d d_model=%d d_ffn=%d: %s\n", M, d_model, d_ffn, pass ? "PASS" : "FAIL");

    // ---- 计时 ----
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_tiled<<<grid_gate, threads>>>(d_x, d_wg, d_gate, M, d_ffn, d_model);
    matmul_tiled<<<grid_gate, threads>>>(d_x, d_wu, d_up, M, d_ffn, d_model);
    swiglu_fused<<<(n_hidden + BLOCK - 1) / BLOCK, BLOCK>>>(d_gate, d_up, d_gate, n_hidden);
    matmul_tiled<<<grid_down, threads>>>(d_gate, d_wd, d_out, M, d_model, d_ffn);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    double gflop = 2.0 * M * d_ffn * d_model * 3 / 1e6;
    printf("Time: %.3f ms,  %.1f GFLOP,  %.1f GFLOP/s\n", ms, gflop, gflop / ms);

    cudaFree(d_x); cudaFree(d_wg); cudaFree(d_wu); cudaFree(d_wd);
    cudaFree(d_out); cudaFree(d_gate); cudaFree(d_up);
    free(h_x); free(h_wg); free(h_wu); free(h_wd); free(h_out); free(h_ref);
    free(gate); free(up); free(hidden);
    return 0;
}

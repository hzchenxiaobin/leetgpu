// 113-fused-qkv-projection.cu —— 融合 QKV 投影：GEMM + layout transform 单 kernel
// 编译: nvcc -O3 -arch=sm_80 113-fused-qkv-projection.cu -o fused_qkv
// 运行: ./fused_qkv

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 16

__global__ void fused_qkv_kernel(
    const float* __restrict__ x,       // [M, D]
    const float* __restrict__ W_qkv,   // [3*D, D]
    float* __restrict__ Q,             // [num_heads, M, head_dim]
    float* __restrict__ K,             // [num_heads, M, head_dim]
    float* __restrict__ V,             // [num_heads, M, head_dim]
    int M, int D, int num_heads, int head_dim)
{
    int row = blockIdx.y * TILE + threadIdx.y;   // M 维
    int col = blockIdx.x * TILE + threadIdx.x;   // 3*D 维

    __shared__ float sA[TILE][TILE];   // x tile: sA[ty][k] = x[row][k]
    __shared__ float sB[TILE][TILE];   // W_qkv tile: sB[k][tx] = W_qkv[col][k]

    float sum = 0.0f;
    int num_tiles = (D + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        // 加载 x tile: sA[ty][tx] = x[row][t*TILE + tx]
        int x_col = t * TILE + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] = (row < M && x_col < D)
            ? x[row * D + x_col] : 0.0f;

        // 加载 W_qkv tile: sB[ty][tx] = W_qkv[col][t*TILE + ty]
        // 注意: W_qkv[col * D + w_k], col 随 threadIdx.x 变化 → stride-D 非合并
        int w_k = t * TILE + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] = (col < 3 * D && w_k < D)
            ? W_qkv[col * D + w_k] : 0.0f;

        __syncthreads();

        // tile 内累加: sum += x[row][k] * W_qkv[col][k]
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < 3 * D) {
        // 融合写回: j → (part, head, hd) → Q/K/V
        int part = col / D;
        int local = col % D;
        int head = local / head_dim;
        int hd = local % head_dim;
        int out_idx = head * M * head_dim + row * head_dim + hd;

        // branchless: 指针数组消除 if-else
        float* out[3] = {Q, K, V};
        out[part][out_idx] = sum;
    }
}

// ---------- 完整测试 harness ----------
void verify(const float* h_x, const float* h_W, float* h_Q, float* h_K, float* h_V,
            int M, int num_heads, int head_dim) {
    int D = num_heads * head_dim;
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < 3 * D; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < D; ++k)
                sum += h_x[m * D + k] * h_W[j * D + k];
            int part = j / D, local = j % D;
            int h = local / head_dim, d = local % head_dim;
            int idx = h * M * head_dim + m * head_dim + d;
            float* arr[3] = {h_Q, h_K, h_V};
            float got = arr[part][idx];
            if (fabsf(got - sum) > 1e-4f) {
                printf("MISMATCH m=%d j=%d expected=%.4f got=%.4f\n", m, j, sum, got);
                return;
            }
        }
    printf("PASS: all %d elements verified\n", M * 3 * D);
}

int main() {
    // 测试 1: 官方 example (M=2, num_heads=2, head_dim=2, D=4)
    {
        int M = 2, num_heads = 2, head_dim = 2, D = num_heads * head_dim;
        float h_x[] = {1,0,0,0, 0,1,0,0};
        float h_W[12*4] = {0};
        // w_q = I
        for (int i = 0; i < 4; i++) h_W[i*4+i] = 1;
        // w_k = swap pairs
        h_W[4*4+0*4+1] = 1; h_W[5*4+0*4+0] = 1; h_W[6*4+0*4+3] = 1; h_W[7*4+0*4+2] = 1;
        // w_v = 2*I
        for (int i = 0; i < 4; i++) h_W[(8+i)*4+i] = 2;

        float *d_x, *d_W, *d_Q, *d_K, *d_V;
        size_t sz_x = M * D * sizeof(float);
        size_t sz_w = 3 * D * D * sizeof(float);
        size_t sz_out = num_heads * M * head_dim * sizeof(float);
        cudaMalloc(&d_x, sz_x);  cudaMalloc(&d_W, sz_w);
        cudaMalloc(&d_Q, sz_out); cudaMalloc(&d_K, sz_out); cudaMalloc(&d_V, sz_out);
        cudaMemcpy(d_x, h_x, sz_x, cudaMemcpyHostToDevice);
        cudaMemcpy(d_W, h_W, sz_w, cudaMemcpyHostToDevice);

        dim3 grid((3*D + TILE-1)/TILE, (M + TILE-1)/TILE);
        dim3 block(TILE, TILE);
        fused_qkv_kernel<<<grid, block>>>(d_x, d_W, d_Q, d_K, d_V, M, D, num_heads, head_dim);
        cudaDeviceSynchronize();

        float h_Q[8], h_K[8], h_V[8];
        cudaMemcpy(h_Q, d_Q, sz_out, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K, d_K, sz_out, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V, d_V, sz_out, cudaMemcpyDeviceToHost);

        printf("=== Test 1: M=2, D=4, heads=2, hd=2 ===\n");
        verify(h_x, h_W, h_Q, h_K, h_V, M, num_heads, head_dim);
        // 打印 Q[0]
        printf("Q[0] = [%.1f, %.1f] [%.1f, %.1f]\n",
               h_Q[0], h_Q[1], h_Q[4], h_Q[5]);

        cudaFree(d_x); cudaFree(d_W); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    }

    // 测试 2: 随机大规模 (LLaMA-2-7B 风格)
    {
        int M = 512, num_heads = 32, head_dim = 128, D = num_heads * head_dim;
        size_t sz_x = M * D * sizeof(float);
        size_t sz_w = 3 * D * D * sizeof(float);
        size_t sz_out = num_heads * M * head_dim * sizeof(float);

        float *h_x = (float*)malloc(sz_x);
        float *h_W = (float*)malloc(sz_w);
        float *h_Q = (float*)malloc(sz_out);
        float *h_K = (float*)malloc(sz_out);
        float *h_V = (float*)malloc(sz_out);
        for (int i = 0; i < M * D; i++) h_x[i] = (float)(rand() % 1000) / 5000.0f - 0.1f;
        for (int i = 0; i < 3 * D * D; i++) h_W[i] = (float)(rand() % 1000) / 50000.0f;

        float *d_x, *d_W, *d_Q, *d_K, *d_V;
        cudaMalloc(&d_x, sz_x);  cudaMalloc(&d_W, sz_w);
        cudaMalloc(&d_Q, sz_out); cudaMalloc(&d_K, sz_out); cudaMalloc(&d_V, sz_out);
        cudaMemcpy(d_x, h_x, sz_x, cudaMemcpyHostToDevice);
        cudaMemcpy(d_W, h_W, sz_w, cudaMemcpyHostToDevice);

        dim3 grid((3*D + TILE-1)/TILE, (M + TILE-1)/TILE);
        dim3 block(TILE, TILE);
        fused_qkv_kernel<<<grid, block>>>(d_x, d_W, d_Q, d_K, d_V, M, D, num_heads, head_dim);
        cudaDeviceSynchronize();

        cudaMemcpy(h_Q, d_Q, sz_out, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_K, d_K, sz_out, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_V, d_V, sz_out, cudaMemcpyDeviceToHost);

        printf("\n=== Test 2: M=%d, D=%d, heads=%d, hd=%d ===\n", M, D, num_heads, head_dim);
        verify(h_x, h_W, h_Q, h_K, h_V, M, num_heads, head_dim);

        cudaFree(d_x); cudaFree(d_W); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
        free(h_x); free(h_W); free(h_Q); free(h_K); free(h_V);
    }

    printf("\nAll tests done.\n");
    return 0;
}

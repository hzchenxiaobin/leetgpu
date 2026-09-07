// 114-gemv.cu —— GEMV：1 block per row，threads 切分 K + block 归约 + float4 向量化
// 编译: nvcc -O3 -arch=sm_80 114-gemv.cu -o gemv
// 运行: ./gemv

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_DIM 256

// block 内归约：warp shuffle + shared 跨 warp 汇总
__device__ __forceinline__ float block_reduce(float val) {
    static __shared__ float shared[32];      // 每 warp 一个 slot（blockDim<=1024 → 最多 32 warp）
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 阶段一：warp 内 5 步蝶形归约
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) shared[wid] = val;        // 每 warp 的和写入 shared
    __syncthreads();

    // 阶段二：warp 0 归约所有 warp 的部分和
    int num_warps = blockDim.x >> 5;
    val = (tid < num_warps) ? shared[tid] : 0.0f;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void gemv_kernel(const float* __restrict__ W,     // [N, K]
                            const float* __restrict__ x,     // [K]
                            const float* __restrict__ bias,  // [N]
                            float* __restrict__ y,           // [N]
                            int N, int K) {
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;

    float sum = 0.0f;
    const int VEC = 4;
    int Kv = K / VEC;                         // 假设 K % 4 == 0（性能测点满足）
    const float4* W4 = reinterpret_cast<const float4*>(W + n * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    // grid-stride 沿向量化 K 维累加
    for (int kv = tid; kv < Kv; kv += BLOCK_DIM) {
        float4 wv = W4[kv];                   // 合并 + 向量化读 W
        float4 xv = x4[kv];                   // x 命中 L2
        sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    // 处理 K % 4 的尾部（性能测点无尾部，保留以保正确性）
    int k_tail_start = Kv * VEC;
    for (int k = k_tail_start + tid; k < K; k += BLOCK_DIM)
        sum += W[n * K + k] * x[k];

    sum = block_reduce(sum);
    if (tid == 0) y[n] = sum + bias[n];
}

// ---------- 完整测试 harness ----------
void gemv_cpu(const float* W, const float* x, const float* bias,
              float* y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float s = bias[n];
        for (int k = 0; k < K; ++k) s += W[n * K + k] * x[k];
        y[n] = s;
    }
}

int main() {
    // 测试 1：官方 example (N=4, K=4)
    {
        int N = 4, K = 4;
        float h_W[]  = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float h_x[]  = {1,0,1,0};
        float h_b[]  = {0,0,0,0};
        float h_y[4], ref[4];

        float *d_W, *d_x, *d_b, *d_y;
        cudaMalloc(&d_W, N*K*sizeof(float));
        cudaMalloc(&d_x, K*sizeof(float));
        cudaMalloc(&d_b, N*sizeof(float));
        cudaMalloc(&d_y, N*sizeof(float));
        cudaMemcpy(d_W, h_W, N*K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

        gemv_kernel<<<N, BLOCK_DIM>>>(d_W, d_x, d_b, d_y, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

        gemv_cpu(h_W, h_x, h_b, ref, N, K);
        printf("=== Test 1: N=%d, K=%d ===\n", N, K);
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            printf("y[%d] = %.1f (ref %.1f)\n", i, h_y[i], ref[i]);
            if (fabsf(h_y[i] - ref[i]) > 1e-4f) ok = false;
        }
        printf("%s\n\n", ok ? "PASS" : "FAIL");

        cudaFree(d_W); cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
    }

    // 测试 2：随机大规模 (N=K=4096，性能测点)
    {
        int N = 4096, K = 4096;
        size_t sz_W = (size_t)N * K * sizeof(float);
        float *h_W = (float*)malloc(sz_W);
        float *h_x = (float*)malloc(K * sizeof(float));
        float *h_b = (float*)malloc(N * sizeof(float));
        float *h_y = (float*)malloc(N * sizeof(float));
        float *ref = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N * K; ++i) h_W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        for (int i = 0; i < K; ++i) h_x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        for (int i = 0; i < N; ++i) h_b[i] = (float)(rand() % 1000) / 1000.0f;

        float *d_W, *d_x, *d_b, *d_y;
        cudaMalloc(&d_W, sz_W);
        cudaMalloc(&d_x, K * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaMemcpy(d_W, h_W, sz_W, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

        gemv_kernel<<<N, BLOCK_DIM>>>(d_W, d_x, d_b, d_y, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        gemv_cpu(h_W, h_x, h_b, ref, N, K);
        printf("=== Test 2: N=%d, K=%d ===\n", N, K);
        double max_err = 0.0;
        for (int i = 0; i < N; ++i)
            max_err = fmax(max_err, fabs((double)h_y[i] - ref[i]));
        printf("max abs err = %.3e  (%s)\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");

        cudaFree(d_W); cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
        free(h_W); free(h_x); free(h_b); free(h_y); free(ref);
    }

    printf("\nAll tests done.\n");
    return 0;
}

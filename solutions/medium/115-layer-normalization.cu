// 115-layer-normalization.cu —— LayerNorm：两阶段块归约（mean → variance）+ 归一化
// 编译: nvcc -O3 -arch=sm_80 115-layer-normalization.cu -o layernorm
// 运行: ./layernorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)   // 8

// ---- warp 级归约：sum（复用 Reduction 模板）----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 + 广播 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;        // 每 warp 的和写入 shared
    __syncthreads();

    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;      // 广播 slot
    }
    __syncthreads();
    return shared[0];
}

// ---- LayerNorm kernel：一个 block 负责一行，两次块归约 ----
__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int M, int D, float eps) {
    __shared__ float shared[NUM_WARPS + 1];

    int r = blockIdx.x;
    if (r >= M) return;
    const float* xr = x + r * D;
    float* yr = y + r * D;

    // ---- Pass 1：求 mean = sum(x) / D ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_sum += xr[i];
    float mean = block_reduce_sum(local_sum, shared) / D;

    // ---- Pass 2：求 variance = sum((x-mean)^2) / D   ← 依赖 mean ----
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float d = xr[i] - mean;
        local_sq += d * d;
    }
    float var  = block_reduce_sum(local_sq, shared) / D;
    float rstd = rsqrtf(var + eps);

    // ---- Pass 3：归一化 + affine ----
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
}

// ---------- 完整测试 harness ----------
void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                   float* y, int M, int D, float eps) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += xr[i];
        float mean = sum / D;
        float sq = 0.0f;
        for (int i = 0; i < D; ++i) { float d = xr[i] - mean; sq += d * d; }
        float rstd = 1.0f / sqrtf(sq / D + eps);
        for (int i = 0; i < D; ++i)
            yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

int main() {
    const float eps = 1e-5f;

    // 测试 1：官方 example (M=1, D=4)
    {
        int M = 1, D = 4;
        float h_x[]  = {1, 2, 3, 4};
        float h_g[]  = {1, 1, 1, 1};
        float h_b[]  = {0, 0, 0, 0};
        float h_y[4], ref[4];

        float *d_x, *d_g, *d_b, *d_y;
        cudaMalloc(&d_x, M * D * sizeof(float));
        cudaMalloc(&d_g, D * sizeof(float));
        cudaMalloc(&d_b, D * sizeof(float));
        cudaMalloc(&d_y, M * D * sizeof(float));
        cudaMemcpy(d_x, h_x, M * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice);

        layernorm_kernel<<<M, BLOCK_SIZE>>>(d_x, d_g, d_b, d_y, M, D, eps);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, M * D * sizeof(float), cudaMemcpyDeviceToHost);

        layernorm_cpu(h_x, h_g, h_b, ref, M, D, eps);
        printf("=== Test 1: M=%d, D=%d ===\n", M, D);
        bool ok = true;
        for (int i = 0; i < D; ++i) {
            printf("y[%d] = %.5f (ref %.5f)\n", i, h_y[i], ref[i]);
            if (fabsf(h_y[i] - ref[i]) > 1e-4f) ok = false;
        }
        printf("%s\n\n", ok ? "PASS" : "FAIL");

        cudaFree(d_x); cudaFree(d_g); cudaFree(d_b); cudaFree(d_y);
    }

    // 测试 2：随机大规模 (M=128, D=8192，性能测点风格)
    {
        int M = 128, D = 8192;
        size_t bytes = (size_t)M * D * sizeof(float);
        float *h_x = (float*)malloc(bytes);
        float *h_g = (float*)malloc(D * sizeof(float));
        float *h_b = (float*)malloc(D * sizeof(float));
        float *h_y = (float*)malloc(bytes);
        float *ref = (float*)malloc(bytes);
        srand(42);
        for (int i = 0; i < M * D; ++i) h_x[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f; // [-10, 10]
        for (int i = 0; i < D; ++i) { h_g[i] = (float)(rand() % 1000) / 1000.0f; h_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f; }

        float *d_x, *d_g, *d_b, *d_y;
        cudaMalloc(&d_x, bytes);
        cudaMalloc(&d_g, D * sizeof(float));
        cudaMalloc(&d_b, D * sizeof(float));
        cudaMalloc(&d_y, bytes);
        cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        layernorm_kernel<<<M, BLOCK_SIZE>>>(d_x, d_g, d_b, d_y, M, D, eps);
        cudaEventRecord(t1);
        cudaDeviceSynchronize();
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
        layernorm_cpu(h_x, h_g, h_b, ref, M, D, eps);

        printf("=== Test 2: M=%d, D=%d ===\n", M, D);
        double max_err = 0.0;
        for (int i = 0; i < M * D; ++i)
            max_err = fmax(max_err, fabs((double)h_y[i] - ref[i]));
        // 3 遍读 x + 1 遍读 gamma + 1 遍读 beta + 1 遍写 y
        double bw_gbs = (3.0 * bytes + D * sizeof(float) + D * sizeof(float) + bytes) / 1e9 / (ms / 1e3);
        printf("kernel time: %.3f ms\n", ms);
        printf("effective bandwidth: %.1f GB/s\n", bw_gbs);
        printf("max abs err = %.3e  (%s)\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");

        cudaFree(d_x); cudaFree(d_g); cudaFree(d_b); cudaFree(d_y);
        free(h_x); free(h_g); free(h_b); free(h_y); free(ref);
    }

    printf("\nAll tests done.\n");
    return 0;
}

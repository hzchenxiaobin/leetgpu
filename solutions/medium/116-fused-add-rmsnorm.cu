// 116-fused-add-rmsnorm.cu —— 融合 Residual Add + RMSNorm：单 kernel，无中间临时张量
// 编译: nvcc -O3 -arch=sm_80 116-fused-add-rmsnorm.cu -o fused_add_rmsnorm
// 运行: ./fused_add_rmsnorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)   // 8

// ---- warp 级归约：sum（复用 Reduction / RMSNorm 模板）----
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

// ---- 融合 kernel：一个 block 负责一行，add 嵌入归约 pass ----
__global__ void fused_add_rmsnorm_kernel(const float* __restrict__ x,
                                         const float* __restrict__ residual,
                                         const float* __restrict__ gamma,
                                         float* __restrict__ y,
                                         int M, int D, float eps) {
    __shared__ float shared[NUM_WARPS + 1];

    int r = blockIdx.x;
    if (r >= M) return;
    const float* xr = x + r * D;
    const float* rr = residual + r * D;
    float* yr = y + r * D;

    // ---- Pass 1：add + sum_sq 融合（h 不落地 HBM）----
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float h = xr[i] + rr[i];        // 残差相加，结果留在 register
        local_sq += h * h;              // 累加平方，无需写 temp
    }
    float mean_sq = block_reduce_sum(local_sq, shared) / D;
    float rrms = rsqrtf(mean_sq + eps);

    // ---- Pass 2：归一化写回（重算 h，避免缓存整行到 shared）----
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float h = xr[i] + rr[i];        // 重算 h（命中 L1/L2 cache）
        yr[i] = h * rrms * gamma[i];
    }
}

// ---------- 完整测试 harness ----------
void fused_add_rmsnorm_cpu(const float* x, const float* residual, const float* gamma,
                           float* y, int M, int D, float eps) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + r * D;
        const float* rr = residual + r * D;
        float* yr = y + r * D;
        float sq = 0.0f;
        for (int i = 0; i < D; ++i) {
            float h = xr[i] + rr[i];
            sq += h * h;
        }
        float rrms = 1.0f / sqrtf(sq / D + eps);
        for (int i = 0; i < D; ++i) {
            float h = xr[i] + rr[i];
            yr[i] = h * rrms * gamma[i];
        }
    }
}

int main() {
    const float eps = 1e-5f;

    // 测试 1：官方 example (M=1, D=4)
    {
        int M = 1, D = 4;
        float h_x[]  = {1, 2, 3, 4};
        float h_r[]  = {3, 2, 1, 0};
        float h_g[]  = {1, 1, 1, 1};
        float h_y[4], ref[4];

        float *d_x, *d_r, *d_g, *d_y;
        cudaMalloc(&d_x, M * D * sizeof(float));
        cudaMalloc(&d_r, M * D * sizeof(float));
        cudaMalloc(&d_g, D * sizeof(float));
        cudaMalloc(&d_y, M * D * sizeof(float));
        cudaMemcpy(d_x, h_x, M * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, h_r, M * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, D * sizeof(float), cudaMemcpyHostToDevice);

        fused_add_rmsnorm_kernel<<<M, BLOCK_SIZE>>>(d_x, d_r, d_g, d_y, M, D, eps);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, M * D * sizeof(float), cudaMemcpyDeviceToHost);

        fused_add_rmsnorm_cpu(h_x, h_r, h_g, ref, M, D, eps);
        printf("=== Test 1: M=%d, D=%d ===\n", M, D);
        bool ok = true;
        for (int i = 0; i < M * D; ++i) {
            printf("y[%d] = %.5f (ref %.5f)\n", i, h_y[i], ref[i]);
            if (fabsf(h_y[i] - ref[i]) > 1e-4f) ok = false;
        }
        printf("%s\n\n", ok ? "PASS" : "FAIL");

        cudaFree(d_x); cudaFree(d_r); cudaFree(d_g); cudaFree(d_y);
    }

    // 测试 2：随机大规模 (M=128, D=8192，性能测点)
    {
        int M = 128, D = 8192;
        size_t sz = (size_t)M * D * sizeof(float);
        float *h_x = (float*)malloc(sz);
        float *h_r = (float*)malloc(sz);
        float *h_g = (float*)malloc(D * sizeof(float));
        float *h_y = (float*)malloc(sz);
        float *ref = (float*)malloc(sz);
        for (int i = 0; i < M * D; ++i) {
            h_x[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
            h_r[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
        }
        for (int i = 0; i < D; ++i) h_g[i] = (float)(rand() % 1000) / 1000.0f;

        float *d_x, *d_r, *d_g, *d_y;
        cudaMalloc(&d_x, sz);
        cudaMalloc(&d_r, sz);
        cudaMalloc(&d_g, D * sizeof(float));
        cudaMalloc(&d_y, sz);
        cudaMemcpy(d_x, h_x, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_r, h_r, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, D * sizeof(float), cudaMemcpyHostToDevice);

        fused_add_rmsnorm_kernel<<<M, BLOCK_SIZE>>>(d_x, d_r, d_g, d_y, M, D, eps);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, sz, cudaMemcpyDeviceToHost);

        fused_add_rmsnorm_cpu(h_x, h_r, h_g, ref, M, D, eps);
        printf("=== Test 2: M=%d, D=%d ===\n", M, D);
        double max_err = 0.0;
        for (int i = 0; i < M * D; ++i)
            max_err = fmax(max_err, fabs((double)h_y[i] - ref[i]));
        printf("max abs err = %.3e  (%s)\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");

        cudaFree(d_x); cudaFree(d_r); cudaFree(d_g); cudaFree(d_y);
        free(h_x); free(h_r); free(h_g); free(h_y); free(ref);
    }

    printf("\nAll tests done.\n");
    return 0;
}

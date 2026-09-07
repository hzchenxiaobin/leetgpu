// 105-group-normalization.cu —— Group Normalization：两遍 scan（mean + var）+ 归一化
// 编译命令: nvcc -O3 -arch=sm_120 105-group-normalization.cu -o group_norm -lineinfo
// 运行:     ./group_norm 8 512 64 64 32

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// ---- warp 级归约：sum ----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 + 广播 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ---- GroupNorm kernel：一个 block 负责一个 (n, g) 对 ----
__global__ void group_norm_kernel(const float* __restrict__ X, const float* __restrict__ gamma,
                                   const float* __restrict__ beta, float* __restrict__ Y,
                                   int N, int C, int H, int W, int G, float eps) {
    __shared__ float shared[NUM_WARPS + 1];

    int n = blockIdx.x / G;
    int g = blockIdx.x % G;
    if (n >= N)
        return;

    int CPG = C / G;            // channels per group
    int SP = H * W;             // spatial size
    int EPG = CPG * SP;         // elements per group
    int base = (n * C + g * CPG) * SP;  // 全局基址

    // ---- Pass 1：求 mean ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < EPG; i += BLOCK_SIZE) {
        float v = X[base + i];
        local_sum += v;
    }
    float sum = block_reduce_sum(local_sum, shared);
    float mean = sum / EPG;

    // ---- Pass 2：求 var ----
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < EPG; i += BLOCK_SIZE) {
        float v = X[base + i];
        float diff = v - mean;
        local_sq += diff * diff;
    }
    float sq_sum = block_reduce_sum(local_sq, shared);
    float var = sq_sum / EPG;
    float inv_std = rsqrtf(var + eps);

    // ---- Pass 3：归一化 + affine：y = (x - mean) * inv_std * gamma[c] + beta[c] ----
    for (int i = threadIdx.x; i < EPG; i += BLOCK_SIZE) {
        int c = g * CPG + i / SP;   // 通道号
        float v = X[base + i];
        Y[base + i] = (v - mean) * inv_std * gamma[c] + beta[c];
    }
}

// ---------- CPU 参考 ----------
void group_norm_cpu(const float* X, const float* gamma, const float* beta, float* Y,
                    int N, int C, int H, int W, int G, float eps) {
    int CPG = C / G, SP = H * W, EPG = CPG * SP;
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < G; ++g) {
            double sum = 0.0;
            for (int c = 0; c < CPG; ++c)
                for (int s = 0; s < SP; ++s)
                    sum += X[(n * C + g * CPG + c) * SP + s];
            float mean = (float)(sum / EPG);
            double sq = 0.0;
            for (int c = 0; c < CPG; ++c)
                for (int s = 0; s < SP; ++s) {
                    float diff = X[(n * C + g * CPG + c) * SP + s] - mean;
                    sq += diff * diff;
                }
            float var = (float)(sq / EPG);
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int c = 0; c < CPG; ++c) {
                int ci = g * CPG + c;
                for (int s = 0; s < SP; ++s) {
                    int idx = (n * C + ci) * SP + s;
                    Y[idx] = (X[idx] - mean) * inv_std * gamma[ci] + beta[ci];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 8;
    int C = (argc > 2) ? atoi(argv[2]) : 512;
    int H = (argc > 3) ? atoi(argv[3]) : 64;
    int W = (argc > 4) ? atoi(argv[4]) : 64;
    int G = (argc > 5) ? atoi(argv[5]) : 32;
    float eps = 1e-5f;
    if (C % G != 0) {
        printf("C=%d 必须能被 G=%d 整除\n", C, G);
        return 1;
    }
    size_t bytes = (size_t)N * C * H * W * sizeof(float);
    printf("N=%d C=%d H=%d W=%d G=%d (%.1f MB)\n", N, C, H, W, G, bytes / 1e6);

    std::vector<float> hX(N * C * H * W), hY(N * C * H * W), hRef(N * C * H * W);
    std::vector<float> hG(C), hB(C);
    srand(42);
    for (auto& x : hX)
        x = ((rand() % 20000) - 10000) / 1000.0f;
    for (int c = 0; c < C; ++c) {
        hG[c] = 0.5f + 0.1f * (rand() % 100) / 100.0f;
        hB[c] = ((rand() % 2000) - 1000) / 1000.0f;
    }

    float *dX, *dG, *dB, *dY;
    cudaMalloc(&dX, bytes);
    cudaMemcpy(dX, hX.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dG, C * sizeof(float));
    cudaMemcpy(dG, hG.data(), C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dB, C * sizeof(float));
    cudaMemcpy(dB, hB.data(), C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dY, bytes);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    // warmup
    group_norm_kernel<<<N * G, BLOCK_SIZE>>>(dX, dG, dB, dY, N, C, H, W, G, eps);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    group_norm_kernel<<<N * G, BLOCK_SIZE>>>(dX, dG, dB, dY, N, C, H, W, G, eps);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 3 遍读 X + 1 遍读 gamma/beta + 1 遍写 Y
    float total_bytes = (3.0f * bytes + 2.0f * C * sizeof(float) + bytes);
    printf("effective bandwidth: %.1f GB/s\n", total_bytes / 1e9 / (ms / 1e3));

    cudaMemcpy(hY.data(), dY, bytes, cudaMemcpyDeviceToHost);
    group_norm_cpu(hX.data(), hG.data(), hB.data(), hRef.data(), N, C, H, W, G, eps);
    float maxDiff = 0.0f;
    for (int i = 0; i < N * C * H * W; ++i)
        maxDiff = fmaxf(maxDiff, fabsf(hY[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-4)\n", maxDiff, maxDiff < 1e-4f ? "PASS" : "FAIL");

    cudaFree(dX);
    cudaFree(dG);
    cudaFree(dB);
    cudaFree(dY);
    return 0;
}

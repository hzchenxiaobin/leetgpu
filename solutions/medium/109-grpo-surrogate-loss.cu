// 109-grpo-surrogate-loss.cu —— GRPO 前向损失：两阶段融合（advantage + fused loss）
// 编译命令: nvcc -O3 -arch=sm_120 109-grpo-surrogate-loss.cu -o grpo -lineinfo
// 运行:     ./grpo            # 默认 B=64,G=16,S=4096
//           ./grpo 1 2 2       # worked example

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

// ---- warp 级归约：sum（复用 Day 4 模板）----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ============================================================
// Kernel 1：组内标准化求 advantages（1 warp / batch，G <= 32）
// ============================================================
__global__ void compute_advantages_kernel(const float* __restrict__ rewards,
                                          float* __restrict__ advantages,
                                          int B, int G) {
    int b = blockIdx.x;
    int tid = threadIdx.x;          // lane id（一个 block 恰好 1 个 warp）
    if (b >= B) return;

    bool valid = (tid < G);
    float r = valid ? rewards[b * G + tid] : 0.0f;

    // Pass 1：求 mean
    float sum = warp_reduce_sum(r);
    float mean = __shfl_sync(0xffffffff, sum, 0) / G;

    // Pass 2：求 population std（unbiased=False）
    float d = valid ? (r - mean) : 0.0f;
    float sum_sq = warp_reduce_sum(d * d);
    float var = __shfl_sync(0xffffffff, sum_sq, 0) / G;
    float std = sqrtf(var > 0.0f ? var : 0.0f);

    if (valid)
        advantages[b * G + tid] = (r - mean) / (std + 1e-8f);
}

// ============================================================
// Kernel 2：全融合 loss（1 block / (b,g) 组，grid-stride 扫 S）
// ============================================================
__global__ void grpo_loss_kernel(const float* __restrict__ log_pi,
                                 const float* __restrict__ log_pi_old,
                                 const float* __restrict__ log_ref,
                                 const float* __restrict__ advantages,
                                 float* __restrict__ output,
                                 float clip_eps, float beta,
                                 int B, int G, int S, float inv_neg_N) {
    __shared__ float shared[NUM_WARPS + 1];
    __shared__ float s_adv;

    int bg = blockIdx.x;
    if (bg >= B * G) return;

    // advantage 对整个组（b,g）恒定，读一次广播给全 block
    if (threadIdx.x == 0) s_adv = advantages[bg];
    __syncthreads();
    float adv = s_adv;

    const float* pi   = log_pi     + bg * S;
    const float* pio  = log_pi_old + bg * S;
    const float* ref  = log_ref    + bg * S;

    float lo = 1.0f - clip_eps;
    float hi = 1.0f + clip_eps;

    // grid-stride 累加 term = surrogate - beta * kl_penalty
    float local = 0.0f;
    for (int s = threadIdx.x; s < S; s += BLOCK_SIZE) {
        float ratio   = __expf(pi[s] - pio[s]);        // 随后 clamp，用 fast exp
        float clipped = fminf(fmaxf(ratio, lo), hi);
        float sur     = fminf(ratio * adv, clipped * adv);
        float kl_diff = ref[s] - pi[s];
        float kl_pen  = expf(kl_diff) - kl_diff - 1.0f; // 大动态范围，用精确 exp
        local += sur - beta * kl_pen;
    }

    float block_sum = block_reduce_sum(local, shared);
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum * inv_neg_N);      // -sum/N
}

// ---- CPU 参考实现（验证用）----
void grpo_loss_cpu(const float* rewards, const float* log_pi, const float* log_pi_old,
                   const float* log_ref, float* output, float clip_eps, float beta,
                   int B, int G, int S) {
    float* adv = (float*)malloc(B * G * sizeof(float));
    for (int b = 0; b < B; ++b) {
        float mean = 0.0f;
        for (int g = 0; g < G; ++g) mean += rewards[b * G + g];
        mean /= G;
        float var = 0.0f;
        for (int g = 0; g < G; ++g) { float d = rewards[b * G + g] - mean; var += d * d; }
        float std = sqrtf(var / G);
        for (int g = 0; g < G; ++g)
            adv[b * G + g] = (rewards[b * G + g] - mean) / (std + 1e-8f);
    }
    double acc = 0.0;
    int N = B * G * S;
    for (int idx = 0; idx < N; ++idx) {
        int bg = idx / S, s = idx - bg * S;
        float ratio   = expf(log_pi[idx] - log_pi_old[idx]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float a = adv[bg];
        float sur = fminf(ratio * a, clipped * a);
        float kl_diff = log_ref[idx] - log_pi[idx];
        float kl_pen = expf(kl_diff) - kl_diff - 1.0f;
        acc += (double)(sur - beta * kl_pen);
    }
    output[0] = -(float)(acc / N);
    free(adv);
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 64;
    int G = (argc > 2) ? atoi(argv[2]) : 16;
    int S = (argc > 3) ? atoi(argv[3]) : 4096;
    float clip_eps = 0.2f, beta = 0.01f;
    int N = B * G * S;
    printf("B=%d G=%d S=%d  N=%d  (%.1f MB per tensor)\n", B, G, S, N, N * 4.0f / 1e6);

    size_t bytes_r = (size_t)B * G * sizeof(float);
    size_t bytes_t = (size_t)N * sizeof(float);

    float *hR, *hPi, *hPiO, *hRef, *hOut;
    hR   = (float*)malloc(bytes_r);
    hPi  = (float*)malloc(bytes_t);
    hPiO = (float*)malloc(bytes_t);
    hRef = (float*)malloc(bytes_t);
    hOut = (float*)malloc(sizeof(float));
    srand(42);
    for (int i = 0; i < B * G; ++i)  hR[i]   = ((float)(rand() % 20000) - 10000.0f) / 1000.0f; // [-10,10]
    for (int i = 0; i < N; ++i) {     hPi[i]  = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;  // [-1,1]
                                     hPiO[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
                                     hRef[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f; }

    float *dR, *dPi, *dPiO, *dRef, *dAdv, *dOut;
    cudaMalloc(&dR, bytes_r);   cudaMemcpy(dR, hR, bytes_r, cudaMemcpyHostToDevice);
    cudaMalloc(&dPi, bytes_t);  cudaMemcpy(dPi, hPi, bytes_t, cudaMemcpyHostToDevice);
    cudaMalloc(&dPiO, bytes_t); cudaMemcpy(dPiO, hPiO, bytes_t, cudaMemcpyHostToDevice);
    cudaMalloc(&dRef, bytes_t); cudaMemcpy(dRef, hRef, bytes_t, cudaMemcpyHostToDevice);
    cudaMalloc(&dAdv, bytes_r);
    cudaMalloc(&dOut, sizeof(float));

    float inv_neg_N = -1.0f / (float)N;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    compute_advantages_kernel<<<B, 32>>>(dR, dAdv, B, G);
    cudaMemset(dOut, 0, sizeof(float));
    grpo_loss_kernel<<<B * G, BLOCK_SIZE>>>(dPi, dPiO, dRef, dAdv, dOut,
                                            clip_eps, beta, B, G, S, inv_neg_N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);

    // 3 遍读 (B,G,S) + 1 遍 rewards + 1 遍 advantages 读写
    float io_bytes = (3.0f * bytes_t + 2.0f * bytes_r) ;
    printf("kernel time: %.3f ms\n", ms);
    printf("effective bandwidth: %.1f GB/s\n", io_bytes / 1e9 / (ms / 1e3));
    printf("gpu output: %.6f\n", hOut[0]);

    // ---- 验证 ----
    float hRefOut[1];
    grpo_loss_cpu(hR, hPi, hPiO, hRef, hRefOut, clip_eps, beta, B, G, S);
    printf("cpu output: %.6f\n", hRefOut[0]);
    float diff = fabsf(hOut[0] - hRefOut[0]);
    float tol  = 1e-4f + 1e-4f * fabsf(hRefOut[0]);
    printf("max diff: %.2e (tol %.2e)  %s\n", diff, tol, diff < tol ? "PASS" : "FAIL");

    cudaFree(dR); cudaFree(dPi); cudaFree(dPiO); cudaFree(dRef); cudaFree(dAdv); cudaFree(dOut);
    free(hR); free(hPi); free(hPiO); free(hRef); free(hOut);
    return 0;
}

// 107-ppo-clipped-surrogate-loss.cu —— PPO 前向损失：单 kernel 全融合
// 编译命令: nvcc -O3 -arch=sm_120 107-ppo-clipped-surrogate-loss.cu -o ppo -lineinfo
// 运行:     ./ppo            # 默认 B=256,S=16384
//           ./ppo 1 4         # worked example

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

// ---- warp 级归约：sum ----
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
// Kernel：全融合 PPO loss（1 block / 行，grid-stride 扫 S）
// ============================================================
__global__ void ppo_loss_kernel(const float* __restrict__ advantages,
                                const float* __restrict__ log_pi,
                                const float* __restrict__ log_pi_old,
                                float* __restrict__ output,
                                float clip_eps, int B, int S, float inv_neg_N) {
    __shared__ float shared[NUM_WARPS];

    int b = blockIdx.x;
    if (b >= B) return;

    const float* adv = advantages   + b * S;
    const float* pi  = log_pi      + b * S;
    const float* pio = log_pi_old  + b * S;

    float lo = 1.0f - clip_eps;
    float hi = 1.0f + clip_eps;

    // grid-stride 累加 surrogate
    float local = 0.0f;
    for (int s = threadIdx.x; s < S; s += BLOCK_SIZE) {
        float ratio   = __expf(pi[s] - pio[s]);           // fast exp，随后 clamp
        float clipped = fminf(fmaxf(ratio, lo), hi);      // 无分支 clip
        float a       = adv[s];
        float sur     = fminf(ratio * a, clipped * a);    // PPO clip surrogate
        local += sur;
    }

    float block_sum = block_reduce_sum(local, shared);
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum * inv_neg_N);          // -sum/(B*S)
}

// ---- CPU 参考实现（验证用）----
void ppo_loss_cpu(const float* advantages, const float* log_pi, const float* log_pi_old,
                  float* output, float clip_eps, int B, int S) {
    double acc = 0.0;
    int N = B * S;
    for (int idx = 0; idx < N; ++idx) {
        float ratio   = expf(log_pi[idx] - log_pi_old[idx]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float a       = advantages[idx];
        float sur     = fminf(ratio * a, clipped * a);
        acc += (double)sur;
    }
    output[0] = -(float)(acc / N);
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 256;
    int S = (argc > 2) ? atoi(argv[2]) : 16384;
    float clip_eps = 0.2f;
    int N = B * S;
    printf("B=%d S=%d  N=%d  (%.1f MB per tensor)\n", B, S, N, N * 4.0f / 1e6);

    size_t bytes = (size_t)N * sizeof(float);

    float *hAdv, *hPi, *hPiO, *hOut;
    hAdv = (float*)malloc(bytes);
    hPi  = (float*)malloc(bytes);
    hPiO = (float*)malloc(bytes);
    hOut = (float*)malloc(sizeof(float));
    srand(42);
    for (int i = 0; i < N; ++i) {
        hAdv[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;  // [-10, 10]
        hPi[i]  = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;    // [-1, 1]
        hPiO[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    }

    float *dAdv, *dPi, *dPiO, *dOut;
    cudaMalloc(&dAdv, bytes);  cudaMemcpy(dAdv, hAdv, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dPi, bytes);   cudaMemcpy(dPi, hPi, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dPiO, bytes);  cudaMemcpy(dPiO, hPiO, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dOut, sizeof(float));

    float inv_neg_N = -1.0f / (float)N;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    cudaMemset(dOut, 0, sizeof(float));
    ppo_loss_kernel<<<B, BLOCK_SIZE>>>(dAdv, dPi, dPiO, dOut, clip_eps, B, S, inv_neg_N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);

    // 3 遍读 (B,S)
    float io_bytes = 3.0f * bytes;
    printf("kernel time: %.3f ms\n", ms);
    printf("effective bandwidth: %.1f GB/s\n", io_bytes / 1e9 / (ms / 1e3));
    printf("gpu output: %.6f\n", hOut[0]);

    // ---- 验证 ----
    float hRefOut[1];
    ppo_loss_cpu(hAdv, hPi, hPiO, hRefOut, clip_eps, B, S);
    printf("cpu output: %.6f\n", hRefOut[0]);
    float diff = fabsf(hOut[0] - hRefOut[0]);
    float tol  = 1e-4f + 1e-4f * fabsf(hRefOut[0]);
    printf("max diff: %.2e (tol %.2e)  %s\n", diff, tol, diff < tol ? "PASS" : "FAIL");

    cudaFree(dAdv); cudaFree(dPi); cudaFree(dPiO); cudaFree(dOut);
    free(hAdv); free(hPi); free(hPiO); free(hOut);
    return 0;
}

// 92-decaying-causal-attention.cu —— Decaying Causal Attention（fused, 增量衰减, 无 softmax）
// 编译命令: nvcc -O3 -arch=sm_120 92-decaying-causal-attention.cu -o decaying_attn -lineinfo
// 运行:     ./decaying_attn 4096 64

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 256
#define MAX_DPT ((D_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE)

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0)
        sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0)
            sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- fused decaying causal attention kernel ----------
// grid = (seq_len,)，每 block 处理 query n 对 key 0..n 的 decaying attention
__global__ void decaying_causal_attention_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ output,
    int seq_len, int d_model, float gamma) {

    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float weight_shm;

    int n = blockIdx.x, tid = threadIdx.x;
    if (n >= seq_len)
        return;
    const float inv_scale = 1.0f / sqrtf((float)d_model);

    // ① 载入 Q[n,:] 到 shared
    for (int t = tid; t < d_model; t += BLOCK_SIZE)
        q_shm[t] = Q[n * d_model + t];
    __syncthreads();

    // 输出累加器（每 thread 处理 MAX_DPT 个 d 维）
    float o_local[MAX_DPT];
    #pragma unroll
    for (int i = 0; i < MAX_DPT; ++i)
        o_local[i] = 0.0f;

    // ② 逆序遍历 key m = n, n-1, ..., 0（causal + 增量衰减）
    float decay = 1.0f;  // gamma^0 for m=n
    for (int m = n; m >= 0; --m) {
        const float* Km = K + m * d_model;
        const float* Vm = V + m * d_model;

        // dot product Q[n] · K[m]
        float part = 0.0f;
        for (int t = tid; t < d_model; t += BLOCK_SIZE)
            part += q_shm[t] * Km[t];
        float s_k = block_reduce_sum(part, red) * inv_scale;

        // 乘性衰减：weight = score × gamma^(n-m) = score × decay
        if (tid == 0)
            weight_shm = s_k * decay;
        __syncthreads();
        float weight = weight_shm;

        // 线性累加（无 softmax！）：o += weight × V[m]
        int oi = 0;
        for (int t = tid; t < d_model; t += BLOCK_SIZE) {
            o_local[oi] += weight * Vm[t];
            ++oi;
        }

        // 增量更新衰减因子：gamma^(n-(m-1)) = gamma^(n-m) × gamma
        decay *= gamma;
        __syncthreads();
    }

    // ③ 写回 output[n,:]
    int oi = 0;
    for (int t = tid; t < d_model; t += BLOCK_SIZE) {
        output[n * d_model + t] = o_local[oi];
        ++oi;
    }
}

// ---------- CPU 参考 ----------
void decaying_attn_cpu(const float* Q, const float* K, const float* V, float* O,
                       int seq_len, int d_model, float gamma) {
    float scale = sqrtf((float)d_model);
    for (int n = 0; n < seq_len; ++n) {
        for (int t = 0; t < d_model; ++t)
            O[n * d_model + t] = 0.0f;
        for (int m = 0; m <= n; ++m) {
            float s = 0.f;
            for (int t = 0; t < d_model; ++t)
                s += Q[n * d_model + t] * K[m * d_model + t];
            s /= scale;
            float decay = powf(gamma, (float)(n - m));
            float weight = s * decay;
            for (int t = 0; t < d_model; ++t)
                O[n * d_model + t] += weight * V[m * d_model + t];
        }
    }
}

int main(int argc, char** argv) {
    int seq_len = (argc > 1) ? atoi(argv[1]) : 4096;
    int d_model = (argc > 2) ? atoi(argv[2]) : 64;
    float gamma = 0.5f;
    if (d_model > D_MAX) {
        printf("要求 d_model <= %d\n", D_MAX);
        return 1;
    }
    printf("seq_len=%d d_model=%d gamma=%.2f\n", seq_len, d_model, gamma);

    size_t bytes = (size_t)seq_len * d_model * sizeof(float);
    std::vector<float> hQ(seq_len * d_model), hK(seq_len * d_model);
    std::vector<float> hV(seq_len * d_model), hO(seq_len * d_model), hRef(seq_len * d_model);
    srand(42);
    for (auto& x : hQ) x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hK) x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hV) x = ((rand() % 2000) - 1000) / 100.f;

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes);  cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, bytes);  cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, bytes);  cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, bytes);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);  cudaEventCreate(&t1);
    // warmup
    decaying_causal_attention_kernel<<<seq_len, BLOCK_SIZE>>>(dQ, dK, dV, dO, seq_len, d_model, gamma);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    decaying_causal_attention_kernel<<<seq_len, BLOCK_SIZE>>>(dQ, dK, dV, dO, seq_len, d_model, gamma);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // causal FLOPs = seq_len*(seq_len+1)/2 * d_model * 2 (QK^T) + same (attn@V)
    float causal_flops = (float)seq_len * (seq_len + 1) / 2 * d_model * 2 * 2;
    printf("causal FLOPs = %.2f G, throughput = %.1f GFLOPS\n",
           causal_flops / 1e9, causal_flops / 1e9 / (ms / 1e3));

    cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost);
    decaying_attn_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), seq_len, d_model, gamma);
    float maxd = 0;
    for (int i = 0; i < seq_len * d_model; ++i)
        maxd = fmaxf(maxd, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-3)\n", maxd, maxd < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dQ);  cudaFree(dK);  cudaFree(dV);  cudaFree(dO);
    return 0;
}

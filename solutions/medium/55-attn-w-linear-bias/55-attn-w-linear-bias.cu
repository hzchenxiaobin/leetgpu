// 55-attn-w-linear-bias.cu —— ALiBi Attention（fused, online softmax, 不物化 S/P）
// 编译命令: nvcc -O3 -arch=sm_120 55-attn-w-linear-bias.cu -o alibi_attn -lineinfo
// 运行:     ./alibi_attn 2048 2048 1024

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 1024
#define MAX_DPT ((D_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE)  // d per thread, 4

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

// ---------- fused ALiBi attention kernel ----------
// grid = (M,)，每 block 处理 query m 对所有 N 个 key 的 attention
__global__ void alibi_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                        const float* __restrict__ V, float* __restrict__ output,
                                        int M, int N, int d, float alpha) {
    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int m = blockIdx.x, tid = threadIdx.x;
    if (m >= M)
        return;
    const float inv_scale = 1.0f / sqrtf((float)d);

    // ① 载入 Q[m,:] 到 shared
    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[m * d + t];
    __syncthreads();

    // 输出累加器（每 thread 处理 MAX_DPT 个 d 维）
    float o_local[MAX_DPT];
    #pragma unroll
    for (int i = 0; i < MAX_DPT; ++i)
        o_local[i] = 0.0f;

    float max_val = -INFINITY, sum_val = 0.0f;

    // ② 遍历所有 key（ALiBi 非因果，全部 N 个 key 都参与）
    for (int n = 0; n < N; ++n) {
        const float* Kn = K + n * d;
        const float* Vn = V + n * d;

        // dot product Q[m] · K[n]
        float part = 0.0f;
        for (int t = tid; t < d; t += BLOCK_SIZE)
            part += q_shm[t] * Kn[t];
        float s_k = block_reduce_sum(part, red) * inv_scale;

        // ★ ALiBi 线性位置偏置：加性
        s_k += alpha * (float)(m - n);

        if (tid == 0)
            s_k_shm = s_k;
        __syncthreads();
        s_k = s_k_shm;

        // online softmax 三公式
        if (tid == 0) {
            float m_new = fmaxf(max_val, s_k);
            float a_old = expf(max_val - m_new);  // 旧输出的缩放因子
            float p = expf(s_k - m_new);           // 新 key 的权重
            float l_new = sum_val * a_old + p;
            alpha_shm = (sum_val * a_old) / l_new;  // 旧输出 / 新 sum
            beta_shm = p / l_new;                    // 新 key / 新 sum
            max_val = m_new;
            sum_val = l_new;
        }
        __syncthreads();

        // 加权累加：o = o * alpha_shm + beta_shm * V[n]
        int oi = 0;
        for (int t = tid; t < d; t += BLOCK_SIZE) {
            o_local[oi] = o_local[oi] * alpha_shm + beta_shm * Vn[t];
            ++oi;
        }
        __syncthreads();
    }

    // ③ 写回 output[m,:]
    int oi = 0;
    for (int t = tid; t < d; t += BLOCK_SIZE) {
        output[m * d + t] = o_local[oi];
        ++oi;
    }
}

// ---------- CPU 参考 ----------
void alibi_attn_cpu(const float* Q, const float* K, const float* V, float* O,
                    int M, int N, int d, float alpha) {
    float scale = sqrtf((float)d);
    std::vector<float> row(N);
    for (int m = 0; m < M; ++m) {
        float mx = -INFINITY;
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int t = 0; t < d; ++t)
                s += Q[m * d + t] * K[n * d + t];
            row[n] = s / scale + alpha * (m - n);
            mx = fmaxf(mx, row[n]);
        }
        float sum = 0.f;
        for (int n = 0; n < N; ++n) {
            row[n] = expf(row[n] - mx);
            sum += row[n];
        }
        for (int t = 0; t < d; ++t) {
            float acc = 0.f;
            for (int n = 0; n < N; ++n)
                acc += row[n] * V[n * d + t];
            O[m * d + t] = acc / sum;
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : 256;
    int d = (argc > 3) ? atoi(argv[3]) : 64;
    float alpha = -0.5f;
    if (d > D_MAX) {
        printf("要求 d <= %d\n", D_MAX);
        return 1;
    }
    printf("M=%d N=%d d=%d alpha=%.2f\n", M, N, d, alpha);

    size_t qk = (size_t)M * d * sizeof(float);
    size_t vk = (size_t)N * d * sizeof(float);
    std::vector<float> hQ(M * d), hK(N * d), hV(N * d), hO(M * d), hRef(M * d);
    srand(42);
    for (auto& x : hQ) x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hK) x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hV) x = ((rand() % 2000) - 1000) / 100.f;

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, qk);   cudaMemcpy(dQ, hQ.data(), qk, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, vk);   cudaMemcpy(dK, hK.data(), vk, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, vk);   cudaMemcpy(dV, hV.data(), vk, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, qk);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);  cudaEventCreate(&t1);
    // warmup
    alibi_attention_kernel<<<M, BLOCK_SIZE>>>(dQ, dK, dV, dO, M, N, d, alpha);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    alibi_attention_kernel<<<M, BLOCK_SIZE>>>(dQ, dK, dV, dO, M, N, d, alpha);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    float flops = (2.0f * M * N * d + 2.0f * M * N * d) / 1e9;  // QK^T + attn@V
    printf("FLOPs = %.2f G, throughput = %.1f GFLOPS\n", flops, flops / (ms / 1e3));

    cudaMemcpy(hO.data(), dO, qk, cudaMemcpyDeviceToHost);
    alibi_attn_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), M, N, d, alpha);
    float maxd = 0;
    for (int i = 0; i < M * d; ++i)
        maxd = fmaxf(maxd, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-3)\n", maxd, maxd < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dQ);  cudaFree(dK);  cudaFree(dV);  cudaFree(dO);
    return 0;
}

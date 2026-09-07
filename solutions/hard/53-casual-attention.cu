// 53-casual-attention.cu —— Causal Self-Attention（fused, online softmax, 不物化 S/P）
// 编译命令: nvcc -O3 -arch=sm_120 53-casual-attention.cu -o causal_attn -lineinfo
// 运行:     ./causal_attn 5000 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 256

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

// ---------- fused causal self-attention kernel ----------
// grid = (M,)，每 block 处理 query i 对 key 0..i 的 causal attention
__global__ void causal_self_attention_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                             const float* __restrict__ V, float* __restrict__ output, int M, int d) {

    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int i = blockIdx.x, tid = threadIdx.x;
    if (i >= M)
        return;
    const float scale = 1.0f / sqrtf((float)d); // 题目用 √d 作 scale

    // ① 载入 Q[i,:] 到 shared
    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[i * d + t];
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float o_local = 0.f;

    // ② 遍历 key 0..i（causal：j ≤ i，天然截断，无需 mask）
    for (int j = 0; j <= i; ++j) {
        const float* Kj = K + j * d;
        const float* Vj = V + j * d;

        float part = 0.f;
        for (int t = tid; t < d; t += BLOCK_SIZE)
            part += q_shm[t] * Kj[t];
        float s_k = block_reduce_sum(part, red) * scale;
        if (tid == 0)
            s_k_shm = s_k;
        __syncthreads();
        s_k = s_k_shm;

        if (tid == 0) {
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new);
            float p = expf(s_k - m_new);
            float l_new = l * alpha + p;
            alpha_shm = (l * alpha) / l_new;
            beta_shm = p / l_new;
            m = m_new;
            l = l_new;
        }
        __syncthreads();

        for (int t = tid; t < d; t += BLOCK_SIZE)
            o_local = o_local * alpha_shm + beta_shm * Vj[t];
        __syncthreads();
    }
    // ③ 写回 output[i,:]
    for (int t = tid; t < d; t += BLOCK_SIZE)
        output[i * d + t] = o_local;
}

// ---------- CPU 参考 ----------
void causal_attn_cpu(const float* Q, const float* K, const float* V, float* O, int M, int d) {
    float scale = sqrtf((float)d);
    std::vector<float> row(M);
    for (int i = 0; i < M; ++i) {
        float mx = -INFINITY;
        for (int j = 0; j <= i; ++j) {
            float s = 0.f;
            for (int t = 0; t < d; ++t)
                s += Q[i * d + t] * K[j * d + t];
            row[j] = s / scale;
            mx = fmaxf(mx, row[j]);
        }
        float sum = 0.f;
        for (int j = 0; j <= i; ++j) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        for (int t = 0; t < d; ++t) {
            float acc = 0.f;
            for (int j = 0; j <= i; ++j)
                acc += row[j] * V[j * d + t];
            O[i * d + t] = acc / sum;
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 5000;
    int d = (argc > 2) ? atoi(argv[2]) : 128;
    if (d > D_MAX) {
        printf("要求 d <= %d\n", D_MAX);
        return 1;
    }
    printf("M=%d d=%d\n", M, d);

    size_t qkv = (size_t)M * d * sizeof(float);
    std::vector<float> hQ(M * d), hK(M * d), hV(M * d), hO(M * d), hRef(M * d);
    srand(42);
    for (auto& x : hQ)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hK)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hV)
        x = ((rand() % 2000) - 1000) / 100.f;

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, qkv);
    cudaMemcpy(dQ, hQ.data(), qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, qkv);
    cudaMemcpy(dK, hK.data(), qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, qkv);
    cudaMemcpy(dV, hV.data(), qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, qkv);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    causal_self_attention_kernel<<<M, BLOCK_SIZE>>>(dQ, dK, dV, dO, M, d);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    causal_self_attention_kernel<<<M, BLOCK_SIZE>>>(dQ, dK, dV, dO, M, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(hO.data(), dO, qkv, cudaMemcpyDeviceToHost);
    causal_attn_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), M, d);
    float maxd = 0;
    for (int i = 0; i < M * d; ++i)
        maxd = fmaxf(maxd, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-3)\n", maxd, maxd < 1e-3f ? "PASS" : "FAIL");

    // 对比标准（非 causal）attention 的计算量
    float causal_flops = (float)M * (M + 1) / 2 * d * 2; // 下三角
    float full_flops = (float)M * M * d * 2;
    printf("causal FLOPs = %.2f G (%.1f%% of full attention %.2f G)\n", causal_flops / 1e9,
           100.0 * causal_flops / full_flops, full_flops / 1e9);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return 0;
}

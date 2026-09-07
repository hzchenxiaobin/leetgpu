// 6-softmax-attention.cu —— naive(物化 S/P) vs fused(online softmax) 对比
// 编译命令: nvcc -O3 -arch=sm_120 6-softmax-attention.cu -o attention -lineinfo
// 运行:     ./attention 1024 64

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 128 // fused 版假设 head_dim <= 128

// ---------- 块归约模板（复用 Day 4）----------
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
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
__inline__ __device__ float block_reduce_max(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0)
        sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : -INFINITY;
        v = warp_reduce_max(v);
        if (lane == 0)
            sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- naive 版：物化 S、P 到 HBM（一个 block 一行 query）----------
__global__ void attention_naive_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                       const float* __restrict__ V, float* __restrict__ S, float* __restrict__ P,
                                       float* __restrict__ O, int N, int d) {
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float row_max_shm, row_sum_shm;
    int i = blockIdx.x, tid = threadIdx.x;
    if (i >= N)
        return;
    const float scale = 1.0f / sqrtf((float)d);

    // ① S[i][k] = Q[i]·K[k]/√d  → 写 HBM
    for (int k = tid; k < N; k += BLOCK_SIZE) {
        float s = 0.f;
        for (int t = 0; t < d; ++t)
            s += Q[i * d + t] * K[k * d + t];
        S[i * N + k] = s * scale;
    }
    __syncthreads();
    // ② 读回 S 求 row max（数值稳定）
    float lm = -INFINITY;
    for (int k = tid; k < N; k += BLOCK_SIZE)
        lm = fmaxf(lm, S[i * N + k]);
    float rmax = block_reduce_max(lm, red);
    if (tid == 0)
        row_max_shm = rmax;
    __syncthreads();
    rmax = row_max_shm;
    // ③ P[i][k] = exp(S[i][k]-rmax) → 写 HBM；同时求 sum
    float ls = 0.f;
    for (int k = tid; k < N; k += BLOCK_SIZE) {
        float p = expf(S[i * N + k] - rmax);
        P[i * N + k] = p;
        ls += p;
    }
    float rsum = block_reduce_sum(ls, red);
    if (tid == 0)
        row_sum_shm = rsum;
    __syncthreads();
    rsum = row_sum_shm;
    // ④ 读回 P、V 算 O[i][t] = Σ_k (P[i][k]/rsum)·V[k][t]
    float inv = 1.0f / rsum;
    for (int t = tid; t < d; t += BLOCK_SIZE) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k)
            acc += P[i * N + k] * V[k * d + t];
        O[i * d + t] = acc * inv;
    }
}

// ---------- fused 版：online softmax，不物化 S/P（一个 block 一行 query）----------
__global__ void attention_fused_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                       const float* __restrict__ V, float* __restrict__ O, int N, int d) {
    __shared__ float q_shm[D_MAX]; // Q[i] 行
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;
    int i = blockIdx.x, tid = threadIdx.x;
    if (i >= N)
        return;

    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[i * d + t];
    __syncthreads();

    float m = -INFINITY, l = 0.f; // running max / sum（thread 0 维护）
    float o_local = 0.f;          // 本 thread 拥有的输出 O[i][tid]
    const float scale = 1.0f / sqrtf((float)d);

    for (int k = 0; k < N; ++k) {
        // ① 点积 s_k = Q[i]·K[k]/√d（每 thread 算自己那维，块归约）
        float part = (tid < d) ? q_shm[tid] * K[k * d + tid] : 0.f;
        float s_k = block_reduce_sum(part, red) * scale;
        if (tid == 0)
            s_k_shm = s_k;
        __syncthreads();
        s_k = s_k_shm;
        // ② online softmax 三公式（thread 0 算，广播 α、β）
        if (tid == 0) {
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new); // 旧状态缩放
            float p = expf(s_k - m_new);   // 新 key 权重
            float l_new = l * alpha + p;
            alpha_shm = (l * alpha) / l_new; // o 的缩放因子
            beta_shm = p / l_new;            // 新 V 的权重
            m = m_new;
            l = l_new;
        }
        __syncthreads();
        // ③ 累加输出：o = o*α + β*v
        if (tid < d)
            o_local = o_local * alpha_shm + beta_shm * V[k * d + tid];
        __syncthreads();
    }
    if (tid < d)
        O[i * d + tid] = o_local;
}

// ---------- CPU 参考实现 ----------
void attention_cpu(const float* Q, const float* K, const float* V, float* O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * sizeof(float));
    float* P = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        float mx = -INFINITY;
        for (int k = 0; k < N; ++k) {
            float s = 0.0f;
            for (int t = 0; t < d; ++t)
                s += Q[i * d + t] * K[k * d + t];
            s *= scale;
            S[k] = s;
            mx = fmaxf(mx, s);
        }
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            P[k] = expf(S[k] - mx);
            sum += P[k];
        }
        for (int t = 0; t < d; ++t) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k)
                acc += P[k] * V[k * d + t];
            O[i * d + t] = acc / sum;
        }
    }
    free(S);
    free(P);
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int d = (argc > 2) ? atoi(argv[2]) : 64;
    if (d > D_MAX) {
        printf("fused 版要求 d <= %d\n", D_MAX);
        return 1;
    }
    size_t qkv = (size_t)N * d * sizeof(float), sp = (size_t)N * N * sizeof(float);
    printf("N=%d d=%d  QKV=%.2f MB  S/P(naive)=%.2f MB each\n", N, d, 3.0 * qkv / 1e6, sp / 1e6);

    float *hQ = (float*)malloc(qkv), *hK = (float*)malloc(qkv), *hV = (float*)malloc(qkv);
    float *hOn = (float*)malloc(qkv), *hOf = (float*)malloc(qkv), *hRef = (float*)malloc(qkv);
    srand(42);
    for (int i = 0; i < N * d; ++i) {
        hQ[i] = ((rand() % 2000) - 1000) / 100.f;
        hK[i] = ((rand() % 2000) - 1000) / 100.f;
        hV[i] = ((rand() % 2000) - 1000) / 100.f;
    }

    float *dQ, *dK, *dV, *dS, *dP, *dOn, *dOf;
    cudaMalloc(&dQ, qkv);
    cudaMemcpy(dQ, hQ, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, qkv);
    cudaMemcpy(dK, hK, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, qkv);
    cudaMemcpy(dV, hV, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dS, sp);
    cudaMalloc(&dP, sp);
    cudaMalloc(&dOn, qkv);
    cudaMalloc(&dOf, qkv);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attention_naive_kernel<<<N, BLOCK_SIZE>>>(dQ, dK, dV, dS, dP, dOn, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms_n = 0;
    cudaEventElapsedTime(&ms_n, t0, t1);
    cudaEventRecord(t0);
    attention_fused_kernel<<<N, BLOCK_SIZE>>>(dQ, dK, dV, dOf, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms_f = 0;
    cudaEventElapsedTime(&ms_f, t0, t1);
    printf("naive: %.3f ms   fused: %.3f ms\n", ms_n, ms_f);

    attention_cpu(hQ, hK, hV, hRef, N, d);
    cudaMemcpy(hOn, dOn, qkv, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOf, dOf, qkv, cudaMemcpyDeviceToHost);
    float dN = 0, dF = 0;
    for (int i = 0; i < N * d; ++i) {
        dN = fmaxf(dN, fabsf(hOn[i] - hRef[i]));
        dF = fmaxf(dF, fabsf(hOf[i] - hRef[i]));
    }
    printf("naive max diff: %.2e (%s)\n", dN, dN < 1e-3f ? "PASS" : "FAIL");
    printf("fused max diff: %.2e (%s)\n", dF, dF < 1e-3f ? "PASS" : "FAIL");

    // 估算 HBM 流量：fused 省掉 S/P 的 4×N² 写读
    float bytes_KV = 2.0f * N * N * d * sizeof(float); // K/V 被 N 个 query 重读
    float bytes_SP = 4.0f * sp;                        // S/P 物化的额外 IO
    printf("est. DRAM: naive=%.2f GB  fused=%.2f GB  (fused 省 S/P=%.2f GB)\n",
           (bytes_KV + bytes_SP + 3.0f * qkv) / 1e9, (bytes_KV + 3.0f * qkv) / 1e9, bytes_SP / 1e9);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dS);
    cudaFree(dP);
    cudaFree(dOn);
    cudaFree(dOf);
    free(hQ);
    free(hK);
    free(hV);
    free(hOn);
    free(hOf);
    free(hRef);
    return 0;
}

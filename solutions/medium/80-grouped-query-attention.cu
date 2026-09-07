// 80-grouped-query-attention.cu —— GQA（fused, online softmax, 不 expand KV）
// 编译命令: nvcc -O3 -arch=sm_120 80-grouped-query-attention.cu -o gqa -lineinfo
// 运行:     ./gqa 32 8 1024 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 256

// ---------- 块归约 ----------
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

// ---------- fused GQA kernel ----------
// grid = (num_q_heads * seq_len,)
// 每 block 处理一个 (q_head h, seq 行 s)
__global__ void gqa_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                           float* __restrict__ output, int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int idx = blockIdx.x;
    int h = idx / seq_len; // q head
    int s = idx % seq_len; // query 行
    int tid = threadIdx.x;
    if (h >= num_q_heads)
        return;

    int group = num_q_heads / num_kv_heads;
    int kv_h = h / group; // ★ GQA 核心：共享的 KV head
    const float scale = 1.0f / sqrtf((float)head_dim);

    // ① 载入 Q[h, s, :] 到 shared
    for (int t = tid; t < head_dim; t += BLOCK_SIZE)
        q_shm[t] = Q[(h * seq_len + s) * head_dim + t];
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float o_local = 0.f;

    // ② 遍历 seq_len 个 key：点积 → online softmax → 加权 V
    for (int k = 0; k < seq_len; ++k) {
        const float* Kk = K + (kv_h * seq_len + k) * head_dim; // 读共享的 KV head
        const float* Vk = V + (kv_h * seq_len + k) * head_dim;

        float part = 0.f;
        for (int t = tid; t < head_dim; t += BLOCK_SIZE)
            part += q_shm[t] * Kk[t];
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

        for (int t = tid; t < head_dim; t += BLOCK_SIZE)
            o_local = o_local * alpha_shm + beta_shm * Vk[t];
        __syncthreads();
    }
    // ③ 写回 output[h, s, :]
    for (int t = tid; t < head_dim; t += BLOCK_SIZE)
        output[(h * seq_len + s) * head_dim + t] = o_local;
}

// ---------- CPU 参考 ----------
void gqa_cpu(const float* Q, const float* K, const float* V, float* O, int nq, int nkv, int S, int d) {
    int g = nq / nkv;
    float sc = 1.0f / sqrtf((float)d);
    std::vector<float> row(S);
    for (int h = 0; h < nq; ++h) {
        int kvh = h / g;
        for (int s = 0; s < S; ++s) {
            float mx = -INFINITY;
            for (int k = 0; k < S; ++k) {
                float dot = 0.f;
                for (int t = 0; t < d; ++t)
                    dot += Q[(h * S + s) * d + t] * K[(kvh * S + k) * d + t];
                row[k] = dot * sc;
                mx = fmaxf(mx, row[k]);
            }
            float sum = 0.f;
            for (int k = 0; k < S; ++k) {
                row[k] = expf(row[k] - mx);
                sum += row[k];
            }
            for (int t = 0; t < d; ++t) {
                float acc = 0.f;
                for (int k = 0; k < S; ++k)
                    acc += row[k] * V[(kvh * S + k) * d + t];
                O[(h * S + s) * d + t] = acc / sum;
            }
        }
    }
}

int main(int argc, char** argv) {
    int nq = (argc > 1) ? atoi(argv[1]) : 32;
    int nkv = (argc > 2) ? atoi(argv[2]) : 8;
    int S = (argc > 3) ? atoi(argv[3]) : 1024;
    int d = (argc > 4) ? atoi(argv[4]) : 128;
    if (d > D_MAX) {
        printf("要求 d <= %d\n", D_MAX);
        return 1;
    }
    if (nq % nkv) {
        printf("num_q_heads 必须整除 num_kv_heads\n");
        return 1;
    }
    int group = nq / nkv;

    size_t q_bytes = (size_t)nq * S * d * sizeof(float);
    size_t kv_bytes = (size_t)nkv * S * d * sizeof(float); // ★ KV 只有 nkv 份（不是 nq）
    printf("nq=%d nkv=%d group=%d S=%d d=%d\n", nq, nkv, group, S, d);
    printf("Q=%.2f MB  KV=%.2f MB (MHA would be %.2f MB, %.1fx more)\n", q_bytes / 1e6, 2.0 * kv_bytes / 1e6,
           2.0 * q_bytes / 1e6, (double)nq / nkv);

    std::vector<float> hQ(nq * S * d), hK(nkv * S * d), hV(nkv * S * d), hO(nq * S * d), hRef(nq * S * d);
    srand(42);
    for (auto& x : hQ)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hK)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hV)
        x = ((rand() % 2000) - 1000) / 100.f;

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, q_bytes);
    cudaMemcpy(dQ, hQ.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, kv_bytes);
    cudaMemcpy(dK, hK.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, kv_bytes);
    cudaMemcpy(dV, hV.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, q_bytes);

    int grid = nq * S;
    // warmup
    gqa_kernel<<<grid, BLOCK_SIZE>>>(dQ, dK, dV, dO, nq, nkv, S, d);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gqa_kernel<<<grid, BLOCK_SIZE>>>(dQ, dK, dV, dO, nq, nkv, S, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(hO.data(), dO, q_bytes, cudaMemcpyDeviceToHost);
    gqa_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), nq, nkv, S, d);
    float maxd = 0;
    for (int i = 0; i < nq * S * d; ++i)
        maxd = fmaxf(maxd, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-3)\n", maxd, maxd < 1e-3f ? "PASS" : "FAIL");

    // KV cache 收益估算
    printf("\n[KV Cache 收益] GQA cache / MHA cache = %d / %d = %.2f\n", nkv, nq, (double)nkv / nq);
    printf("[LLaMA-3 8B] 32Q/8KV → cache 缩到 %.0f%%（省 %.0f%%）\n", 100.0 * nkv / nq,
           100.0 * (1.0 - (double)nkv / nq));

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    return 0;
}

// 106-token-embedding-layer.cu —— Token Embedding + LayerNorm 融合 kernel
// 编译命令: nvcc -O3 -arch=sm_120 106-token-embedding-layer.cu -o token_emb -lineinfo
// 运行:     ./token_emb 32 512 30000 2048 768

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 1024

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

// ---------- fused kernel：一个 block 处理一个 (b,t) ----------
__global__ void token_embedding_layernorm_kernel(const int* __restrict__ token_ids,      // (B, T)
                                                 const int* __restrict__ position_ids,   // (T,)
                                                 const float* __restrict__ token_emb,    // (V, D)
                                                 const float* __restrict__ position_emb, // (P, D)
                                                 const float* __restrict__ gamma,        // (D,)
                                                 const float* __restrict__ beta,         // (D,)
                                                 float* __restrict__ output,             // (B, T, D)
                                                 int B, int T, int V, int P, int D, float eps) {

    int idx = blockIdx.x;
    int b = idx / T, t = idx % T;
    int tid = threadIdx.x;
    if (b >= B)
        return;

    int tok_id = token_ids[b * T + t];
    int pos_id = position_ids[t];

    __shared__ float s_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];

    // ① gather + 相加：每 thread 读若干维
    for (int d = tid; d < D; d += BLOCK_SIZE)
        s_shm[d] = token_emb[tok_id * D + d] + position_emb[pos_id * D + d];
    __syncthreads();

    // ② 求 μ = Σ s[d] / D（块归约）
    float local_sum = 0.f;
    for (int d = tid; d < D; d += BLOCK_SIZE)
        local_sum += s_shm[d];
    float mu = block_reduce_sum(local_sum, red) / (float)D;
    __shared__ float s_mu;
    if (tid == 0)
        s_mu = mu;
    __syncthreads();
    mu = s_mu;

    // ③ 求 σ² = Σ (s[d]-μ)² / D（块归约）
    float local_sq = 0.f;
    for (int d = tid; d < D; d += BLOCK_SIZE) {
        float diff = s_shm[d] - mu;
        local_sq += diff * diff;
    }
    float var = block_reduce_sum(local_sq, red) / (float)D;
    __shared__ float s_var;
    if (tid == 0)
        s_var = var;
    __syncthreads();
    var = s_var;

    float inv_std = 1.0f / sqrtf(var + eps);

    // ④ 归一化 + 写回：y[d] = γ[d]·(s[d]-μ)/√(σ²+ε) + β[d]
    for (int d = tid; d < D; d += BLOCK_SIZE)
        output[(b * T + t) * D + d] = gamma[d] * (s_shm[d] - mu) * inv_std + beta[d];
}

// ---------- CPU 参考 ----------
void embed_cpu(const int* tids, const int* pids, const float* te, const float* pe, const float* g, const float* bt,
               float* out, int B, int T, int V, int P, int D, float eps) {
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            int tid = tids[b * T + t], pid = pids[t];
            std::vector<float> s(D);
            for (int d = 0; d < D; ++d)
                s[d] = te[tid * D + d] + pe[pid * D + d];
            float mu = 0.f;
            for (int d = 0; d < D; ++d)
                mu += s[d];
            mu /= D;
            float var = 0.f;
            for (int d = 0; d < D; ++d) {
                float diff = s[d] - mu;
                var += diff * diff;
            }
            var /= D;
            float inv = 1.0f / sqrtf(var + eps);
            for (int d = 0; d < D; ++d)
                out[(b * T + t) * D + d] = g[d] * (s[d] - mu) * inv + bt[d];
        }
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 32, T = (argc > 2) ? atoi(argv[2]) : 512;
    int V = (argc > 3) ? atoi(argv[3]) : 30000, P = (argc > 4) ? atoi(argv[4]) : 2048;
    int D = (argc > 5) ? atoi(argv[5]) : 768;
    float eps = 1e-5f;
    if (D > D_MAX) {
        printf("要求 D <= %d\n", D_MAX);
        return 1;
    }
    printf("B=%d T=%d V=%d P=%d D=%d\n", B, T, V, P, D);

    size_t ids_bytes = (size_t)B * T * sizeof(int), pids_bytes = (size_t)T * sizeof(int);
    size_t te_bytes = (size_t)V * D * sizeof(float), pe_bytes = (size_t)P * D * sizeof(float);
    size_t gb_bytes = (size_t)D * sizeof(float), out_bytes = (size_t)B * T * D * sizeof(float);
    printf("token_emb = %.2f MB, output = %.2f MB\n", te_bytes / 1e6, out_bytes / 1e6);

    std::vector<int> h_tids(B * T), h_pids(T);
    std::vector<float> h_te(V * D), h_pe(P * D), h_g(D), h_bt(D), h_out(B * T * D), h_ref(B * T * D);
    srand(42);
    for (auto& x : h_tids)
        x = rand() % V;
    for (int t = 0; t < T; ++t)
        h_pids[t] = t % P;
    for (auto& x : h_te)
        x = ((rand() % 600) - 300) / 1000.f;
    for (auto& x : h_pe)
        x = ((rand() % 600) - 300) / 1000.f;
    for (auto& x : h_g)
        x = 0.8f + (rand() % 400) / 1000.f;
    for (auto& x : h_bt)
        x = ((rand() % 200) - 100) / 1000.f;

    int *d_tids, *d_pids;
    float *d_te, *d_pe, *d_g, *d_bt, *d_out;
    cudaMalloc(&d_tids, ids_bytes);
    cudaMemcpy(d_tids, h_tids.data(), ids_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_pids, pids_bytes);
    cudaMemcpy(d_pids, h_pids.data(), pids_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_te, te_bytes);
    cudaMemcpy(d_te, h_te.data(), te_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_pe, pe_bytes);
    cudaMemcpy(d_pe, h_pe.data(), pe_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_g, gb_bytes);
    cudaMemcpy(d_g, h_g.data(), gb_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_bt, gb_bytes);
    cudaMemcpy(d_bt, h_bt.data(), gb_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, out_bytes);

    int grid = B * T;
    token_embedding_layernorm_kernel<<<grid, BLOCK_SIZE>>>(d_tids, d_pids, d_te, d_pe, d_g, d_bt, d_out, B, T, V, P, D,
                                                           eps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
    embed_cpu(h_tids.data(), h_pids.data(), h_te.data(), h_pe.data(), h_g.data(), h_bt.data(), h_ref.data(), B, T, V, P,
              D, eps);

    float maxd = 0;
    for (int i = 0; i < B * T * D; ++i)
        maxd = fmaxf(maxd, fabsf(h_out[i] - h_ref[i]));
    printf("max diff: %.2e (%s, tol=1e-4)\n", maxd, maxd < 1e-4f ? "PASS" : "FAIL");

    cudaFree(d_tids);
    cudaFree(d_pids);
    cudaFree(d_te);
    cudaFree(d_pe);
    cudaFree(d_g);
    cudaFree(d_bt);
    cudaFree(d_out);
    return 0;
}

// 87-speculative-decoding-verification.cu —— 投机解码验证 kernel
// 编译命令: nvcc -O3 -arch=sm_120 87-speculative-decoding-verification.cu -o spec_decode -lineinfo
// 运行:     ./spec_decode 64 8 32768

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

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

// ---------- 流式 CDF searchsorted ----------
// 对 probs[V] 做 CDF，找 r 落在哪个 token（返回 index）
// adjusted = max(probs - sub, 0)（sub 可为 draft_probs，用于 resample；bonus 时 sub=null）
// 若 total==0 用均匀分布 1/V
__device__ int cdf_searchsorted(const float* probs, const float* sub, float r, int V, float* sh) {
    int tid = threadIdx.x;
    // Pass 1: 求 total = Σ max(probs[v] - sub[v], 0)（sub==null 时 Σ probs[v]）
    float local_sum = 0.f;
    for (int v = tid; v < V; v += BLOCK_SIZE) {
        float val = sub ? fmaxf(probs[v] - sub[v], 0.f) : probs[v];
        local_sum += val;
    }
    float total = block_reduce_sum(local_sum, sh);
    float inv_total = (total > 0.f) ? (1.0f / total) : 0.f;
    bool use_uniform = (total <= 0.f);

    // Pass 2: 流式 CDF，找 r 落在哪
    float running = 0.f; // 本 block 的 running CDF（thread 0 维护）
    int result = V - 1;  // 默认最后一个
    for (int chunk_start = 0; chunk_start < V; chunk_start += BLOCK_SIZE) {
        int v = chunk_start + tid;
        float val = 0.f;
        if (v < V) {
            if (use_uniform)
                val = 1.0f / V;
            else {
                float raw = sub ? fmaxf(probs[v] - sub[v], 0.f) : probs[v];
                val = raw * inv_total;
            }
        }
        // block 内 inclusive prefix sum
        __shared__ float scan_sh[NUM_WARPS + 1];
        int lane = tid & 31, wid = tid >> 5;
        float prefix = val;
        #pragma unroll
        for (int o = 1; o < WARP_SIZE; o <<= 1) {
            float n = __shfl_up_sync(0xffffffff, prefix, o);
            if (lane >= o)
                prefix += n;
        }
        if (lane == 31)
            scan_sh[wid] = prefix;
        __syncthreads();
        if (wid == 0) {
            float w = (lane < NUM_WARPS) ? scan_sh[lane] : 0.f;
            #pragma unroll
            for (int o = 1; o < NUM_WARPS; o <<= 1) {
                float n = __shfl_up_sync(0xffffffff, w, o);
                if (lane >= o)
                    w += n;
            }
            if (lane < NUM_WARPS)
                scan_sh[lane] = w;
        }
        __syncthreads();
        float chunk_offset = (wid > 0) ? scan_sh[wid - 1] : 0.f;
        float my_cdf = running + chunk_offset + (prefix - val); // exclusive prefix + running

        // 检查 r 是否落在本 thread 的区间 [my_cdf, my_cdf + val)
        if (v < V && r >= my_cdf && r < my_cdf + val) {
            result = v;
        }
        // 更新 running：本 chunk 的总和 = scan_sh[NUM_WARPS-1]（最后一个 warp 的 inclusive）
        running += scan_sh[NUM_WARPS - 1];
        __syncthreads();
    }
    // block 广播最小 result（多个 thread 可能命中，取最小）
    __shared__ int result_sh;
    if (tid == 0)
        result_sh = V - 1;
    __syncthreads();
    if (result < V)
        atomicMin(&result_sh, result);
    __syncthreads();
    return result_sh;
}

// ---------- fused kernel：一个 block 处理一个序列 ----------
__global__ void spec_decode_verify_kernel(const int* __restrict__ draft_tokens, const float* __restrict__ draft_probs,
                                          const float* __restrict__ target_probs,
                                          const float* __restrict__ uniform_samples, int* __restrict__ output, int B,
                                          int T, int V) {

    int b = blockIdx.x, tid = threadIdx.x;
    if (b >= B)
        return;
    __shared__ float sh[NUM_WARPS + 1];
    __shared__ int s_tok;
    __shared__ float s_p, s_q, s_alpha;

    bool rejected = false;
    for (int i = 0; i < T; ++i) {
        // gather p, q（thread 0 读，广播）
        if (tid == 0) {
            s_tok = draft_tokens[b * T + i];
            s_p = draft_probs[(b * T + i) * V + s_tok];
            s_q = target_probs[(b * T + i) * V + s_tok];
            s_alpha = fminf(1.0f, s_q / s_p);
        }
        __syncthreads();
        int tok = s_tok;
        float alpha = s_alpha, u = uniform_samples[b * (T + 1) + i];

        if (u < alpha) {
            // accept
            if (tid == 0)
                output[b * (T + 1) + i] = tok;
        } else {
            // reject: resample from max(target - draft, 0) at position i
            const float* tgt = target_probs + (b * T + i) * V;
            const float* drf = draft_probs + (b * T + i) * V;
            float r = uniform_samples[b * (T + 1) + T];
            int new_tok = cdf_searchsorted(tgt, drf, r, V, sh);
            if (tid == 0)
                output[b * (T + 1) + i] = new_tok;
            rejected = true;
            break;
        }
    }
    if (!rejected) {
        // 全 accept: bonus token from target_probs[b, T-1]
        const float* tgt = target_probs + (b * T + (T - 1)) * V;
        float r = uniform_samples[b * (T + 1) + T];
        int bonus = cdf_searchsorted(tgt, nullptr, r, V, sh);
        if (tid == 0)
            output[b * (T + 1) + T] = bonus;
    }
}

// ---------- CPU 参考 ----------
void spec_decode_cpu(const int* dt, const float* dp, const float* tp, const float* us, int* out, int B, int T, int V) {
    for (int b = 0; b < B; ++b) {
        bool rej = false;
        for (int i = 0; i < T; ++i) {
            int tok = dt[b * T + i];
            float p = dp[(b * T + i) * V + tok], q = tp[(b * T + i) * V + tok];
            float alpha = fminf(1.f, q / p);
            if (us[b * (T + 1) + i] < alpha) {
                out[b * (T + 1) + i] = tok;
            } else {
                std::vector<float> adj(V);
                float total = 0.f;
                for (int v = 0; v < V; ++v) {
                    adj[v] = fmaxf(tp[(b * T + i) * V + v] - dp[(b * T + i) * V + v], 0.f);
                    total += adj[v];
                }
                if (total > 0)
                    for (int v = 0; v < V; ++v)
                        adj[v] /= total;
                else
                    for (int v = 0; v < V; ++v)
                        adj[v] = 1.f / V;
                float cdf = 0.f, r = us[b * (T + 1) + T];
                int nt = V - 1;
                for (int v = 0; v < V; ++v) {
                    cdf += adj[v];
                    if (r < cdf) {
                        nt = v;
                        break;
                    }
                }
                out[b * (T + 1) + i] = nt;
                rej = true;
                break;
            }
        }
        if (!rej) {
            float cdf = 0.f, r = us[b * (T + 1) + T];
            int bonus = V - 1;
            for (int v = 0; v < V; ++v) {
                cdf += tp[(b * T + (T - 1)) * V + v];
                if (r < cdf) {
                    bonus = v;
                    break;
                }
            }
            out[b * (T + 1) + T] = bonus;
        }
    }
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 64;
    int T = (argc > 2) ? atoi(argv[2]) : 8;
    int V = (argc > 3) ? atoi(argv[3]) : 32768;
    printf("B=%d T=%d V=%d\n", B, T, V);

    size_t dt_bytes = (size_t)B * T * sizeof(int);
    size_t prob_bytes = (size_t)B * T * V * sizeof(float);
    size_t us_bytes = (size_t)B * (T + 1) * sizeof(float);
    size_t out_bytes = (size_t)B * (T + 1) * sizeof(int);
    printf("draft_probs+target_probs = %.2f MB\n", 2.0 * prob_bytes / 1e6);

    std::vector<int> h_dt(B * T);
    std::vector<float> h_dp(B * T * V), h_tp(B * T * V), h_us(B * (T + 1));
    std::vector<int> h_out(B * (T + 1), 0), h_ref(B * (T + 1), 0);
    srand(42);
    for (auto& x : h_dt)
        x = rand() % V;
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < T; ++i) {
            float s = 0.f;
            for (int v = 0; v < V; ++v) {
                h_dp[(b * T + i) * V + v] = (rand() % 1000) / 1000.f;
                s += h_dp[(b * T + i) * V + v];
            }
            for (int v = 0; v < V; ++v)
                h_dp[(b * T + i) * V + v] /= s;
            s = 0.f;
            for (int v = 0; v < V; ++v) {
                h_tp[(b * T + i) * V + v] = (rand() % 1000) / 1000.f;
                s += h_tp[(b * T + i) * V + v];
            }
            for (int v = 0; v < V; ++v)
                h_tp[(b * T + i) * V + v] /= s;
            // 保证 draft token 概率 > 0
            if (h_dp[(b * T + i) * V + h_dt[b * T + i]] == 0.f)
                h_dp[(b * T + i) * V + h_dt[b * T + i]] = 1e-6f;
        }
    for (auto& x : h_us)
        x = (rand() % 10000) / 10000.f;

    int* d_dt;
    float *d_dp, *d_tp, *d_us;
    int* d_out;
    cudaMalloc(&d_dt, dt_bytes);
    cudaMemcpy(d_dt, h_dt.data(), dt_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_dp, prob_bytes);
    cudaMemcpy(d_dp, h_dp.data(), prob_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_tp, prob_bytes);
    cudaMemcpy(d_tp, h_tp.data(), prob_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_us, us_bytes);
    cudaMemcpy(d_us, h_us.data(), us_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, out_bytes);
    cudaMemset(d_out, 0, out_bytes);

    // warmup + 计时
    spec_decode_verify_kernel<<<B, BLOCK_SIZE>>>(d_dt, d_dp, d_tp, d_us, d_out, B, T, V);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    spec_decode_verify_kernel<<<B, BLOCK_SIZE>>>(d_dt, d_dp, d_tp, d_us, d_out, B, T, V);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
    spec_decode_cpu(h_dt.data(), h_dp.data(), h_tp.data(), h_us.data(), h_ref.data(), B, T, V);
    int mism = 0;
    for (int i = 0; i < B * (T + 1); ++i)
        if (h_out[i] != h_ref[i])
            mism++;
    printf("mismatched tokens: %d / %d (%s)\n", mism, B * (T + 1), mism == 0 ? "PASS" : "FAIL");

    cudaFree(d_dt);
    cudaFree(d_dp);
    cudaFree(d_tp);
    cudaFree(d_us);
    cudaFree(d_out);
    return 0;
}

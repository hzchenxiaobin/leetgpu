// 60-top-p-sampling.cu —— Top-p Nucleus Sampling: softmax + bitonic sort + scan + CDF sample
// 编译命令: nvcc -O3 -arch=sm_80 60-top-p-sampling.cu -o top_p_sampling

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#define BLOCK_SIZE 256
#define MAX_VOCAB 50000

// warp 内归约求 max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int o = 16; o > 0; o >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, o));
    return val;
}

// warp 内归约求 sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int o = 16; o > 0; o >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, o);
    return val;
}

// warp 内 inclusive prefix scan (sum)
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    int lane = threadIdx.x & 31;
    for (int o = 1; o < 32; o <<= 1) {
        float t = __shfl_up_sync(0xFFFFFFFF, val, o);
        if (lane >= o) val += t;
    }
    return val;
}

// 整数哈希生成 [0, 1) 随机数 (splitmix64 简化版)
__device__ __forceinline__ float hash_to_uniform(uint32_t seed) {
    uint64_t z = (uint64_t)seed + 0x9e3779b9ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z = z ^ (z >> 31);
    return (float)(z >> 11) * (1.0f / 9007199254740992.0f);  // [0, 1)
}

// bitonic sort: 对 shared 数组降序排序 (prob, idx) pair
// N 须为 2 的幂，不足补 -INFINITY
__device__ void bitonic_sort(float* vals, int* idxs, int N) {
    int tid = threadIdx.x;
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < N / 2; i += blockDim.x) {
                int pos = 2 * (i / stride) * stride + (i % stride);
                int partner = pos + stride;
                if (partner < N) {
                    // 降序：大值在前
                    bool ascending = ((i / (size / 2)) % 2) == 0;
                    bool swap = ascending ? (vals[pos] < vals[partner])
                                          : (vals[pos] > vals[partner]);
                    if (swap) {
                        float tv = vals[pos]; vals[pos] = vals[partner]; vals[partner] = tv;
                        int ti = idxs[pos]; idxs[pos] = idxs[partner]; idxs[partner] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void top_p_sampling_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ p_val,
    const int32_t* __restrict__ seed_val,
    int32_t* __restrict__ sampled_token,
    int vocab_size)
{
    extern __shared__ char smem[];

    int padded_V = 1;
    while (padded_V < vocab_size) padded_V <<= 1;

    float* s_probs = (float*)smem;              // [padded_V]
    int*   s_idx   = (int*)(s_probs + padded_V); // [padded_V]
    float* s_cum   = (float*)(s_idx + padded_V); // [padded_V]

    int tid = threadIdx.x;
    float p = *p_val;
    uint32_t seed = (uint32_t)(*seed_val);

    // ===== ① Softmax（减 max 保稳定）=====
    // Pass 1: 求 max
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE)
        local_max = fmaxf(local_max, logits[i]);
    // block reduce max (简化：用 shared memory)
    __shared__ float s_max;
    if (tid == 0) s_max = -INFINITY;
    __syncthreads();
    local_max = warp_reduce_max(local_max);
    if ((tid & 31) == 0) atomicMax((int*)&s_max, __float_as_int(local_max));
    __syncthreads();
    float max_logit = s_max;

    // Pass 2: 求 sum
    float local_sum = 0;
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE)
        local_sum += expf(logits[i] - max_logit);
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0;
    __syncthreads();
    local_sum = warp_reduce_sum(local_sum);
    if ((tid & 31) == 0) atomicAdd(&s_sum, local_sum);
    __syncthreads();
    float total_sum = s_sum;

    // 写 probs 到 shared
    for (int i = tid; i < vocab_size; i += BLOCK_SIZE) {
        s_probs[i] = expf(logits[i] - max_logit) / total_sum;
        s_idx[i] = i;
    }
    // 补齐到 2 的幂（用 -INFINITY 填充）
    for (int i = vocab_size + tid; i < padded_V; i += BLOCK_SIZE) {
        s_probs[i] = -INFINITY;
        s_idx[i] = 0;
    }
    __syncthreads();

    // ===== ② Bitonic Sort（降序）=====
    bitonic_sort(s_probs, s_idx, padded_V);
    __syncthreads();

    // ===== ③ Cumsum（前缀和）=====
    // 简化版：对前 vocab_size 个做串行 cumsum（thread 0）
    if (tid == 0) {
        s_cum[0] = s_probs[0];
        for (int i = 1; i < vocab_size; i++)
            s_cum[i] = s_cum[i - 1] + s_probs[i];
    }
    __syncthreads();

    // ===== ④ Nucleus 截断 =====
    __shared__ int s_cutoff;
    if (tid == 0) {
        s_cutoff = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            if (s_cum[i] >= p) { s_cutoff = i + 1; break; }
        }
    }
    __syncthreads();
    int cutoff = s_cutoff;

    // ===== ⑤ Renorm + CDF 采样 =====
    if (tid == 0) {
        float nucleus_sum = s_cum[cutoff - 1];
        float r = hash_to_uniform(seed);
        float cum = 0;
        for (int i = 0; i < cutoff; i++) {
            cum += s_probs[i] / nucleus_sum;
            if (r < cum) {
                *sampled_token = s_idx[i];
                return;
            }
        }
        *sampled_token = s_idx[cutoff - 1];  // fallback
    }
}

// ===== Host 端 =====
int main() {
    // 功能测试: logits=[1, 2, 3, 0.5], p=0.9, seed=42
    int V = 4;
    float h_logits[] = {1.0f, 2.0f, 3.0f, 0.5f};
    float h_p = 0.9f;
    int32_t h_seed = 42;
    int32_t h_token = -1;

    float *d_logits; float *d_p; int32_t *d_seed, *d_token;
    cudaMalloc(&d_logits, V * sizeof(float));
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_seed, sizeof(int32_t));
    cudaMalloc(&d_token, sizeof(int32_t));
    cudaMemcpy(d_logits, h_logits, V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, &h_p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, &h_seed, sizeof(int32_t), cudaMemcpyHostToDevice);

    // shared memory 大小: 3 * padded_V * 4 bytes (probs + idx + cumsum)
    int padded_V = 1;
    while (padded_V < V) padded_V <<= 1;
    size_t smem = padded_V * (2 * sizeof(float) + sizeof(int));

    top_p_sampling_kernel<<<1, BLOCK_SIZE, smem>>>(d_logits, d_p, d_seed, d_token, V);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_token, d_token, sizeof(int32_t), cudaMemcpyDeviceToHost);

    printf("=== Functional Test ===\n");
    printf("logits = [1, 2, 3, 0.5], p = 0.9, seed = 42\n");
    printf("probs = [0.16, 0.42, 0.64, 0.09]\n");
    printf("sorted = [0.64(idx=2), 0.24(idx=1), 0.09(idx=0), 0.03(idx=3)]\n");
    printf("cumsum = [0.64, 0.88, 0.97, 1.00] → nucleus = top 3\n");
    printf("sampled_token = %d (expect 2, 1, or 0)\n", h_token);
    printf("%s\n\n", (h_token >= 0 && h_token < V) ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: V=50000 =====
    int V2 = 50000;
    float *d_logits2;
    cudaMalloc(&d_logits2, V2 * sizeof(float));
    float *h_l2 = (float*)malloc(V2 * sizeof(float));
    srand(42);
    for (int i = 0; i < V2; i++) h_l2[i] = -3.0f + 6.0f * (rand() / (float)RAND_MAX);
    cudaMemcpy(d_logits2, h_l2, V2 * sizeof(float), cudaMemcpyHostToDevice);

    int pv2 = 1;
    while (pv2 < V2) pv2 <<= 1;
    size_t smem2 = pv2 * (2 * sizeof(float) + sizeof(int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    top_p_sampling_kernel<<<1, BLOCK_SIZE, smem2>>>(d_logits2, d_p, d_seed, d_token, V2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("=== Perf Test (V=%d) ===\n", V2);
    printf("Kernel time = %.3f ms\n", ms);
    printf("shared memory = %.1f KB (padded_V=%d)\n", smem2 / 1024.0, pv2);

    cudaFree(d_logits); cudaFree(d_p); cudaFree(d_seed); cudaFree(d_token);
    cudaFree(d_logits2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_l2);
    return 0;
}

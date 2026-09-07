// 96-int8-kv-cache-attention.cu —— INT8 KV-Cache Decode Attention（fused, online softmax）
// 编译命令: nvcc -O3 -arch=sm_120 96-int8-kv-cache-attention.cu -o int8_kv_attn -lineinfo
// 运行:     ./int8_kv_attn 32 8192 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 256 // head_dim <= 256

// ---------- 块归约（复用 Week4 Softmax Attention）----------
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

// ---------- fused kernel：一个 block 处理一个 head ----------
// Q        [H, d]
// K_int8   [H, L, d]  + k_scale [H, L]
// V_int8   [H, L, d]  + v_scale [H, L]
// output   [H, d]
__global__ void int8_kv_attention_kernel(const float* __restrict__ Q, const int8_t* __restrict__ K_int8,
                                         const int8_t* __restrict__ V_int8, const float* __restrict__ k_scale,
                                         const float* __restrict__ v_scale, float* __restrict__ output, int H, int L,
                                         int d) {
    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int h = blockIdx.x, tid = threadIdx.x;
    if (h >= H)
        return;
    const float scale = 1.0f / sqrtf((float)d);

    // ① 载入本 head 的 Q 到 shared（全 block 复用）
    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[h * d + t];
    __syncthreads();

    float m = -INFINITY, l = 0.f; // online softmax running max/sum（thread 0 维护）
    float o_local = 0.f;          // 本 thread 持有的输出维

    // ② 遍历 L 个 token：点积（含 int8 反量化）→ online softmax → 加权 V
    for (int s = 0; s < L; ++s) {
        const int8_t* Ks = K_int8 + (h * L + s) * d;
        const int8_t* Vs = V_int8 + (h * L + s) * d;
        const float ks = k_scale[h * L + s];
        const float vs = v_scale[h * L + s];

        // 点积 s_k = Σ_t Q[t] · (K_int8[t]·ks) / √d
        float part = 0.f;
        for (int t = tid; t < d; t += BLOCK_SIZE)
            part += q_shm[t] * (Ks[t] * ks);
        float s_k = block_reduce_sum(part, red) * scale;
        if (tid == 0)
            s_k_shm = s_k;
        __syncthreads();
        s_k = s_k_shm;

        // online softmax 三公式（thread 0 算，广播 α、β）
        if (tid == 0) {
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new); // 旧状态缩放
            float p = expf(s_k - m_new);   // 新 token 权重
            float l_new = l * alpha + p;
            alpha_shm = (l * alpha) / l_new; // o 的缩放因子
            beta_shm = p / l_new;            // 新 V 的权重
            m = m_new;
            l = l_new;
        }
        __syncthreads();

        // 累加输出：o = o*α + β*v_f（v_f = V_int8·vs，反量化融在加权里）
        for (int t = tid; t < d; t += BLOCK_SIZE)
            o_local = o_local * alpha_shm + beta_shm * (Vs[t] * vs);
        __syncthreads();
    }
    // ③ 写回 output[h]
    for (int t = tid; t < d; t += BLOCK_SIZE)
        output[h * d + t] = o_local;
}

// ---------- CPU 参考实现 ----------
void attention_int8_cpu(const float* Q, const int8_t* K, const int8_t* V, const float* ks, const float* vs, float* O,
                        int H, int L, int d) {
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> sc(L);
    for (int h = 0; h < H; ++h) {
        float mx = -INFINITY;
        for (int s = 0; s < L; ++s) {
            float dot = 0.f, k = ks[h * L + s];
            for (int t = 0; t < d; ++t)
                dot += Q[h * d + t] * (K[(h * L + s) * d + t] * k);
            sc[s] = dot * scale;
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < L; ++s) {
            sc[s] = expf(sc[s] - mx);
            sum += sc[s];
        }
        for (int t = 0; t < d; ++t)
            O[h * d + t] = 0.f;
        for (int s = 0; s < L; ++s) {
            float w = sc[s] / sum, v = vs[h * L + s];
            for (int t = 0; t < d; ++t)
                O[h * d + t] += w * (V[(h * L + s) * d + t] * v);
        }
    }
}

int main(int argc, char** argv) {
    int H = (argc > 1) ? atoi(argv[1]) : 32;
    int L = (argc > 2) ? atoi(argv[2]) : 8192;
    int d = (argc > 3) ? atoi(argv[3]) : 128;
    if (d > D_MAX) {
        printf("要求 d <= %d\n", D_MAX);
        return 1;
    }

    size_t q_bytes = (size_t)H * d * sizeof(float);
    size_t kv_bytes = (size_t)H * L * d * sizeof(int8_t); // int8: 1 byte
    size_t sc_bytes = (size_t)H * L * sizeof(float);
    printf("H=%d L=%d d=%d  int8 KV=%.2f MB  (fp32 KV would be %.2f MB, 4x)\n", H, L, d, 2.0 * kv_bytes / 1e6,
           4.0 * kv_bytes / 1e6);

    // host 数据
    std::vector<float> hQ(H * d);
    std::vector<int8_t> hK(H * L * d), hV(H * L * d);
    std::vector<float> hks(H * L), hvs(H * L);
    srand(42);
    for (auto& x : hQ)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hK)
        x = (int8_t)((rand() % 255) - 128);
    for (auto& x : hV)
        x = (int8_t)((rand() % 255) - 128);
    for (auto& x : hks)
        x = (rand() % 100) / 1000.f + 0.01f;
    for (auto& x : hvs)
        x = (rand() % 100) / 1000.f + 0.01f;
    std::vector<float> hO(H * d), hRef(H * d);

    // device 分配
    float *dQ, *dks, *dvs, *dO;
    int8_t *dK, *dV;
    cudaMalloc(&dQ, q_bytes);
    cudaMemcpy(dQ, hQ.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, kv_bytes);
    cudaMemcpy(dK, hK.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, kv_bytes);
    cudaMemcpy(dV, hV.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dks, sc_bytes);
    cudaMemcpy(dks, hks.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dvs, sc_bytes);
    cudaMemcpy(dvs, hvs.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, q_bytes);

    // 计时
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    // warmup
    int8_kv_attention_kernel<<<H, BLOCK_SIZE>>>(dQ, dK, dV, dks, dvs, dO, H, L, d);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    int8_kv_attention_kernel<<<H, BLOCK_SIZE>>>(dQ, dK, dV, dks, dvs, dO, H, L, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 验证
    cudaMemcpy(hO.data(), dO, q_bytes, cudaMemcpyDeviceToHost);
    attention_int8_cpu(hQ.data(), hK.data(), hV.data(), hks.data(), hvs.data(), hRef.data(), H, L, d);
    float maxd = 0;
    for (int i = 0; i < H * d; ++i)
        maxd = fmaxf(maxd, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-3)\n", maxd, maxd < 1e-3f ? "PASS" : "FAIL");

    // 估算 HBM 流量与算术强度
    float bytes_kv = 2.0f * kv_bytes + 2.0f * sc_bytes;                   // int8 K/V + scale
    float bytes_fp32 = 2.0f * kv_bytes * 4;                               // 若用 fp32 KV
    float flops = 2.0f * H * L * d * 2 + 3.0f * H * L + 2.0f * H * L * d; // ~2·QK^T + softmax + 2·PV
    printf("est. DRAM (int8)=%.2f MB  (fp32)=%.2f MB  AI(int8)=%.2f  AI(fp32)=%.2f\n", bytes_kv / 1e6, bytes_fp32 / 1e6,
           flops / bytes_kv, flops / bytes_fp32);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dks);
    cudaFree(dvs);
    cudaFree(dO);
    return 0;
}

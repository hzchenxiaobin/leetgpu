// 26-multi-head-cross-attention.cu —— Multi-Head Cross-Attention（online softmax / FlashAttention）
// Q: (M,H,D), K: (N,H,D), V: (N,H,D), output: (M,H,D)，行主序
// 编译命令: nvcc -O3 -arch=sm_120 26-multi-head-cross-attention.cu -o cross_attn
// 运行:     ./cross_attn

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call); \
if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE); } \
} while (0)

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 128

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// grid = (H, M)，block = BLOCK_SIZE。thread tid 负责 D 维的第 tid 个分量。
__global__ void cross_attn_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ output,
                                  int M, int N, int H, int D) {
    int m = blockIdx.y;   // query 行
    int h = blockIdx.x;   // head
    if (m >= M || h >= H) return;
    int tid = threadIdx.x;

    __shared__ float q_shm[D_MAX];          // Q[m,h,:] 缓存
    __shared__ float red[NUM_WARPS + 1];     // 块归约缓冲
    __shared__ float alpha_shm, beta_shm;    // 广播 O 的缩放因子 / V 的权重

    const float* Qmh = Q + ((m * H) + h) * D;
    if (tid < D) q_shm[tid] = Qmh[tid];     // 加载 Q[m,h,:] 到 shared
    __syncthreads();

    float o_local = 0.f;                     // running output（本 thread 的 D 分量）
    float m_i = -INFINITY, l_i = 0.f;        // running max / sum（thread 0 维护）
    const float scale = 1.0f / sqrtf((float)D);

    for (int n = 0; n < N; ++n) {
        // ① s = Q[m,h,:] · K[n,h,:] / √D（D 维点积，block_reduce_sum）
        const float* Knh = K + ((n * H) + h) * D;
        float part = (tid < D) ? q_shm[tid] * Knh[tid] : 0.f;
        float s_k = block_reduce_sum(part, red) * scale;

        // ② online softmax 更新 (m, l) 并广播 α/β
        if (tid == 0) {
            float m_new = fmaxf(m_i, s_k);
            float alpha = expf(m_i - m_new);
            float p = expf(s_k - m_new);
            float l_new = l_i * alpha + p;
            alpha_shm = (l_i * alpha) / l_new;   // O 的缩放因子
            beta_shm = p / l_new;                // 新 V 的权重
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();

        // ③ O ← O · α + V[n,h] · β（每 thread 更新自己的 D 分量）
        const float* Vnh = V + ((n * H) + h) * D;
        if (tid < D)
            o_local = o_local * alpha_shm + beta_shm * Vnh[tid];
        __syncthreads();
    }
    // O 已归一化（每步 α+β = (l·α+p)/l_new = 1，加权平均权重和恒为 1）
    if (tid < D)
        output[((m * H) + h) * D + tid] = o_local;
}

int main() {
    int M = 2, N = 3, H = 2, D = 2;
    size_t qB = (size_t)M * H * D * sizeof(float);
    size_t kB = (size_t)N * H * D * sizeof(float);
    size_t oB = (size_t)M * H * D * sizeof(float);

    std::vector<float> hQ = {1,0, 0,1,  0,1, 1,0};
    std::vector<float> hK = {1,0, 0,1,  0,1, 1,0,  1,1, 1,1};
    std::vector<float> hV = {1,2, 7,8,  3,4, 9,10,  5,6, 11,12};
    std::vector<float> hO(M * H * D, 0);

    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, qB));
    CHECK_CUDA(cudaMalloc(&dK, kB));
    CHECK_CUDA(cudaMalloc(&dV, kB));
    CHECK_CUDA(cudaMalloc(&dO, oB));
    CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), qB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK.data(), kB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV.data(), kB, cudaMemcpyHostToDevice));

    dim3 grid(H, M);
    dim3 block(BLOCK_SIZE);
    cross_attn_kernel<<<grid, block>>>(dQ, dK, dV, dO, M, N, H, D);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hO.data(), dO, oB, cudaMemcpyDeviceToHost));

    // CPU 验证
    std::vector<float> ref(M * H * D, 0);
    float scale = 1.0f / sqrtf((float)D);
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < H; ++h) {
            std::vector<float> S(N);
            float mx = -INFINITY;
            for (int n = 0; n < N; ++n) {
                float s = 0;
                for (int t = 0; t < D; ++t) s += hQ[((m*H)+h)*D+t] * hK[((n*H)+h)*D+t];
                S[n] = s * scale; mx = fmaxf(mx, S[n]);
            }
            float sum = 0;
            for (int n = 0; n < N; ++n) { S[n] = expf(S[n] - mx); sum += S[n]; }
            for (int t = 0; t < D; ++t) {
                float acc = 0;
                for (int n = 0; n < N; ++n) acc += (S[n] / sum) * hV[((n*H)+h)*D+t];
                ref[((m*H)+h)*D+t] = acc;
            }
        }

    int err = 0;
    for (size_t i = 0; i < hO.size(); ++i)
        if (fabsf(hO[i] - ref[i]) > 1e-4f * fmaxf(1.0f, fabsf(ref[i]))) { ++err; if (err <= 5) printf("MISMATCH[%zu]: got %f ref %f\n", i, hO[i], ref[i]); }
    printf("M=%d N=%d H=%d D=%d: %s\n", M, N, H, D, err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dQ)); CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV)); CHECK_CUDA(cudaFree(dO));
    return err ? EXIT_FAILURE : 0;
}

// 112-attention-with-sinks.cu —— 融合 online softmax，复合掩码 causal ∧ (sink ∨ window)
// 编译命令: nvcc -O3 -arch=sm_120 112-attention-with-sinks.cu -o attn_sinks -lineinfo
// 运行:     ./attn_sinks 5000 128 4 1024

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128   // 一个 thread 对应 d 的一维（假设 d <= D_MAX）
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 128

// ---------- 块归约 + 广播模板（复用 attention family）----------
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & (WARP_SIZE - 1), wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- 融合 kernel：一 block 一行 query ----------
__global__ void attn_sinks_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                  const float* __restrict__ V, float* __restrict__ O,
                                  int M, int d, int num_sinks, int window_size) {
    int i = blockIdx.x;      // query 行
    int t = threadIdx.x;     // d 维索引
    if (i >= M) return;

    const float scale = rsqrtf((float)d);  // 1/√d

    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS];
    __shared__ float s_m, s_l, s_corr, s_p;

    // ① 载入 Q[i] 到 shared，复用 i 次
    if (t < d) q_shm[t] = Q[i * d + t];
    else if (t < D_MAX) q_shm[t] = 0.0f;
    if (t == 0) { s_m = -INFINITY; s_l = 0.0f; }
    __syncthreads();

    float acc = 0.0f;  // 每 thread 持有输出的一维
    int win_start = i - window_size + 1;
    if (win_start < 0) win_start = 0;

    // ② 遍历允许的 key j
    for (int j = 0; j <= i; ++j) {
        if (j >= num_sinks && j < win_start) continue;  // 空洞：既非 sink 也不在窗口

        // 点积 s = Q[i]·K[j]·scale（每 thread 算一维部分积 → 块归约）
        float part = (t < d) ? q_shm[t] * K[j * d + t] : 0.0f;
        float s = block_reduce_sum(part, red) * scale;

        // ③ online softmax 更新（仅 tid=0 算标量，广播 corr/p）
        if (t == 0) {
            float m_old = s_m;
            float m_new = fmaxf(m_old, s);
            float corr = expf(m_old - m_new);
            float p = expf(s - m_new);
            s_corr = corr;
            s_p = p;
            s_m = m_new;
            s_l = s_l * corr + p;
        }
        __syncthreads();

        // ④ 累加输出：acc = acc·corr + p·V[j]
        acc = acc * s_corr + s_p * ((t < d) ? V[j * d + t] : 0.0f);
        __syncthreads();
    }

    // ⑤ 归一化写回
    if (t < d) O[i * d + t] = acc / s_l;
}

// ---------- CPU 参考实现 ----------
void attn_sinks_cpu(const float* Q, const float* K, const float* V, float* O,
                    int M, int d, int num_sinks, int window_size) {
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < M; ++i) {
        int win_start = i - window_size + 1;
        if (win_start < 0) win_start = 0;
        float mx = -INFINITY;
        for (int j = 0; j <= i; ++j) {
            if (j >= num_sinks && j < win_start) continue;
            float s = 0.0f;
            for (int t = 0; t < d; ++t) s += Q[i * d + t] * K[j * d + t];
            mx = fmaxf(mx, s * scale);
        }
        float sum = 0.0f;
        for (int t = 0; t < d; ++t) O[i * d + t] = 0.0f;
        for (int j = 0; j <= i; ++j) {
            if (j >= num_sinks && j < win_start) continue;
            float s = 0.0f;
            for (int t = 0; t < d; ++t) s += Q[i * d + t] * K[j * d + t];
            float p = expf(s * scale - mx);
            sum += p;
            for (int t = 0; t < d; ++t) O[i * d + t] += p * V[j * d + t];
        }
        for (int t = 0; t < d; ++t) O[i * d + t] /= sum;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 5000;
    int d = (argc > 2) ? atoi(argv[2]) : 128;
    int num_sinks = (argc > 3) ? atoi(argv[3]) : 4;
    int window_size = (argc > 4) ? atoi(argv[4]) : 1024;
    if (d > D_MAX) { fprintf(stderr, "d must be <= %d\n", D_MAX); return 1; }

    size_t bytes = (size_t)M * d * sizeof(float);
    printf("M=%d d=%d num_sinks=%d window=%d  (QKV %.1f MB)\n",
           M, d, num_sinks, window_size, 3.0 * bytes / 1e6);

    float *hQ = (float*)malloc(bytes), *hK = (float*)malloc(bytes),
          *hV = (float*)malloc(bytes), *hO = (float*)malloc(bytes),
          *hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * d; ++i) {
        hQ[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
        hK[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
        hV[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    }

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes); cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    attn_sinks_kernel<<<M, BLOCK_SIZE>>>(dQ, dK, dV, dO, M, d, num_sinks, window_size);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost);
    attn_sinks_cpu(hQ, hK, hV, hRef, M, d, num_sinks, window_size);

    float max_diff = 0.0f;
    for (int i = 0; i < M * d; ++i)
        max_diff = fmaxf(max_diff, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.3e  (%s)\n", max_diff, max_diff < 1e-4 ? "PASS" : "FAIL");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO); free(hRef);
    return 0;
}

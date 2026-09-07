// 109-attention.cu —— Scaled Dot-Product Attention（FlashAttention 简化版）
// 编译命令: nvcc -O3 -arch=sm_120 109-attention.cu -o flash_attn -lineinfo
// 运行:     ./flash_attn 256 256 64      # M N d

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                          \
    cudaError_t e = (call);                                                                                   \
    if (e != cudaSuccess) {                                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
        exit(EXIT_FAILURE);                                                                                   \
    }                                                                                                         \
} while (0)

#define MAX_D 128
#define BLOCK_SIZE 128          // 一个 thread 负责一个 d 维度（d ≤ MAX_D）
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)   // 4

// ---- warp 级归约：sum（复用 Softmax #5 模板）----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：sum（warp shuffle + shared 汇总 + 广播给全 block）----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);   // lane = tid % 32
    int warpId = threadIdx.x >> 5;               // warpId = tid / 32
    val = warp_reduce_sum(val);                  // 阶段1：warp 内归约
    if (lane == 0) shared[warpId] = val;         // 阶段1：lane 0 写 shared
    __syncthreads();                             // 屏障：等 4 个 warp 都写完
    if (warpId == 0) {                           // 阶段2：仅 warp 0 执行
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;          // 写广播槽
    }
    __syncthreads();                             // 屏障：等 warp 0 写完
    return shared[0];                            // 阶段3：全 block 读 shared[0]
}

// ---- FlashAttention kernel：一个 block 负责一行 query ----
__global__ void flash_attention_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int M, int N, int d) {
    int i = blockIdx.x;                  // 第 i 行 query
    if (i >= M) return;
    int tid = threadIdx.x;

    __shared__ float sQ[MAX_D];          // Q[i] 整行（常驻）
    __shared__ float sK[MAX_D];          // K[k] 整行（逐 k 滑入）
    __shared__ float sV[MAX_D];          // V[k] 整行
    __shared__ float reduce_shared[NUM_WARPS + 1];

    // ---- 加载 Q[i] 到 shared（整轮循环常驻，只读一次 HBM）----
    if (tid < d) sQ[tid] = Q[i * d + tid];
    __syncthreads();

    float scale = rsqrtf((float)d);

    // ---- running state：m, l 全 block 复制一致；o 每 thread 持有自己的维度 ----
    float m = -INFINITY;                 // running max of scaled score
    float l = 0.0f;                      // running sum of exp(s - m)
    float o = 0.0f;                      // O[tid] 的未归一化累加器

    // ---- 遍历每个 key（S/P 从不物化）----
    for (int k = 0; k < N; ++k) {
        // ① 协作加载 K[k], V[k]（连续 thread 读连续地址 → coalesced）
        if (tid < d) {
            sK[tid] = K[k * d + tid];
            sV[tid] = V[k * d + tid];
        }
        __syncthreads();

        // ② s_k = (Q[i] · K[k]) * scale：每 thread 算一维部分积，块归约求和
        float partial = (tid < d) ? sQ[tid] * sK[tid] : 0.0f;
        float s_k = block_reduce_sum(partial, reduce_shared) * scale;
        // → s_k 已通过 shared[0] 广播给全 block 所有 thread

        // ③ online softmax + P·V 融合更新
        //    所有 thread 拿到相同的 s_k，独立计算标量 m/l（结果天然一致）
        float m_new = fmaxf(m, s_k);
        float alpha = __expf(m - m_new);     // 旧状态缩放因子（m=-∞ 时为 0）
        float p = __expf(s_k - m_new);       // 当前 key 的未归一化权重
        l = l * alpha + p;
        if (tid < d) o = o * alpha + p * sV[tid];   // 每 thread 更新自己的 O 维度
        m = m_new;

        __syncthreads();                     // 确保 sK/sV 被读完，下一轮可覆盖
    }

    // ---- 归一化（除以 l）并写回 O[i] ----
    if (tid < d) O[i * d + tid] = o / l;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 256;
    int N = (argc > 2) ? atoi(argv[2]) : 256;
    int d = (argc > 3) ? atoi(argv[3]) : 64;
    if (d > MAX_D) { fprintf(stderr, "d must be <= %d\n", MAX_D); return 1; }

    size_t bytes_Q  = (size_t)M * d * sizeof(float);
    size_t bytes_KV = (size_t)N * d * sizeof(float);
    printf("M=%d, N=%d, d=%d  (Q %.2f MB, K/V %.2f MB each)\n",
           M, N, d, bytes_Q / 1e6, bytes_KV / 1e6);

    // ---- host ----
    float *hQ = (float*)malloc(bytes_Q);
    float *hK = (float*)malloc(bytes_KV);
    float *hV = (float*)malloc(bytes_KV);
    float *hO = (float*)malloc(bytes_Q);
    srand(42);
    for (size_t i = 0; i < (size_t)M * d; ++i) hQ[i] = ((rand() % 2000) - 1000) / 1000.0f; // [-1,1]
    for (size_t i = 0; i < (size_t)N * d; ++i) {
        hK[i] = ((rand() % 2000) - 1000) / 1000.0f;
        hV[i] = ((rand() % 2000) - 1000) / 1000.0f;
    }

    // ---- device ----
    float *dQ, *dK, *dV, *dO;
    CHECK_CUDA(cudaMalloc(&dQ, bytes_Q));
    CHECK_CUDA(cudaMalloc(&dK, bytes_KV));
    CHECK_CUDA(cudaMalloc(&dV, bytes_KV));
    CHECK_CUDA(cudaMalloc(&dO, bytes_Q));
    CHECK_CUDA(cudaMemcpy(dQ, hQ, bytes_Q, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, hK, bytes_KV, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, hV, bytes_KV, cudaMemcpyHostToDevice));

    // ---- launch ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    flash_attention_kernel<<<M, BLOCK_SIZE>>>(dQ, dK, dV, dO, M, N, d);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 有效 HBM 流量：读 Q + 读 K + 读 V + 写 O（S/P 不计）
    double hbm = (double)(bytes_Q + 2.0 * bytes_KV + bytes_Q);
    printf("effective bandwidth: %.1f GB/s\n", hbm / 1e9 / (ms / 1e3));

    // ---- 验证：CPU naive 三步走（double 累加）做参考 ----
    CHECK_CUDA(cudaMemcpy(hO, dO, bytes_Q, cudaMemcpyDeviceToHost));
    double scale = 1.0 / sqrt((double)d);
    double maxDiff = 0.0;
    float* ref = (float*)malloc((size_t)N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        double smax = -1e300;
        for (int j = 0; j < N; ++j) {
            double dot = 0;
            for (int kk = 0; kk < d; ++kk) dot += (double)hQ[i * d + kk] * hK[j * d + kk];
            ref[j] = (float)(dot * scale);
            if (ref[j] > smax) smax = ref[j];
        }
        double ssum = 0;
        for (int j = 0; j < N; ++j) { ref[j] = (float)exp((double)ref[j] - smax); ssum += ref[j]; }
        for (int kk = 0; kk < d; ++kk) {
            double acc = 0;
            for (int j = 0; j < N; ++j) acc += (ref[j] / ssum) * hV[j * d + kk];
            maxDiff = fmax(maxDiff, fabs((double)hO[i * d + kk] - acc));
        }
    }
    free(ref);
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff < 1e-4 ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
    free(hQ);
    free(hK);
    free(hV);
    free(hO);
    return 0;
}

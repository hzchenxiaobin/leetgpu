// 6-softmax-attention-fa2-tiling.cu —— FlashAttention v2 Forward Kernel（Br×Bc tiling + warp work partitioning）
// 与 ai-infra-notes W5D3 的 6-softmax-attention-fa2-tiling.cu 结构一致，适配单头 M×N 场景
// 编译: nvcc -O3 -arch=sm_120 6-softmax-attention-fa2-tiling.cu -o fa2_tiling -lineinfo
// 运行: ./fa2_tiling 1024 1024 64

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

constexpr int Br = 64;
constexpr int Bc = 64;
constexpr int D = 64;

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
static_assert(Br % WARPS_PER_BLOCK == 0, "Br must be divisible by WARPS_PER_BLOCK");
constexpr int ROWS_PER_WARP = Br / WARPS_PER_BLOCK;

__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q, const float* __restrict__ K,
    const float* __restrict__ V, float* __restrict__ O,
    int M, int N, int d)
{
    __shared__ float s_Q[Br][D];
    __shared__ float s_K[Bc][D];
    __shared__ float s_V[Bc][D];

    int qTileRow = blockIdx.x * Br;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warpId = tid / 32;
    int qRowStart = warpId * ROWS_PER_WARP;

    // 协作加载 Q tile（常驻 shared memory）
    #pragma unroll
    for (int idx = tid; idx < Br * d; idx += THREADS_PER_BLOCK) {
        int r = idx / d;
        int c = idx % d;
        int globalRow = qTileRow + r;
        s_Q[r][c] = (globalRow < M) ? Q[globalRow * d + c] : 0.0f;
    }
    __syncthreads();

    // 每个 warp 维护 ROWS_PER_WARP 个 Q 行的 running 状态
    float m_arr[ROWS_PER_WARP];
    float l_arr[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP][D];

    #pragma unroll
    for (int i = 0; i < ROWS_PER_WARP; i++) {
        m_arr[i] = -1e30f;
        l_arr[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < d; j++)
            acc[i][j] = 0.0f;
    }

    float scale = 1.0f / sqrtf((float)d);

    // 内层循环：遍历 KV tile
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
        // 协作加载 K/V tile
        #pragma unroll
        for (int idx = tid; idx < Bc * d; idx += THREADS_PER_BLOCK) {
            int r = idx / d;
            int c = idx % d;
            int globalRow = kvStart + r;
            s_K[r][c] = (globalRow < N) ? K[globalRow * d + c] : 0.0f;
            s_V[r][c] = (globalRow < N) ? V[globalRow * d + c] : 0.0f;
        }
        __syncthreads();

        // 每个 warp 独立处理自己的 ROWS_PER_WARP 个 Q 行（无需跨 warp 通信）
        #pragma unroll
        for (int localRow = 0; localRow < ROWS_PER_WARP; localRow++) {
            int qi = qRowStart + localRow;
            int globalQi = qTileRow + qi;
            if (qi >= Br || globalQi >= M)
                continue;

            // Step 1: Sij[c] = Qi · Kj[c]^T（每线程算 Bc/32 个点积）
            float Sij[Bc / 32];
            #pragma unroll
            for (int c = lane; c < Bc; c += 32) {
                float dot = 0.0f;
                #pragma unroll
                for (int di = 0; di < d; di++)
                    dot += s_Q[qi][di] * s_K[c][di];
                Sij[c / 32] = dot * scale;
            }

            // Step 2: 局部 max（warpReduceMax）
            float localMax = -1e30f;
            #pragma unroll
            for (int i = 0; i < Bc / 32; i++)
                localMax = fmaxf(localMax, Sij[i]);
            localMax = warpReduceMax(localMax);

            // Step 3: online softmax 缩放旧状态
            float m_prev = m_arr[localRow];
            float m_new = fmaxf(m_prev, localMax);
            float scale_old = expf(m_prev - m_new);
            m_arr[localRow] = m_new;
            l_arr[localRow] *= scale_old;
            #pragma unroll
            for (int di = 0; di < d; di++)
                acc[localRow][di] *= scale_old;

            // Step 4: 处理新块——累加 p 和 p×V
            #pragma unroll
            for (int i = 0; i < Bc / 32; i++) {
                int c = lane + i * 32;
                bool valid = c < Bc && (kvStart + c) < N;
                float s_val = valid ? Sij[i] : -1e30f;
                float p_val = valid ? expf(s_val - m_new) : 0.0f;

                float p_sum = warpReduceSum(p_val);
                if (lane == 0)
                    l_arr[localRow] += p_sum;

                #pragma unroll
                for (int di = 0; di < d; di++) {
                    float contrib = valid ? p_val * s_V[c][di] : 0.0f;
                    float sum_contrib = warpReduceSum(contrib);
                    if (lane == 0)
                        acc[localRow][di] += sum_contrib;
                }
            }

            // 广播 l 和 acc 到 warp 内所有线程
            l_arr[localRow] = __shfl_sync(0xFFFFFFFF, l_arr[localRow], 0);
            #pragma unroll
            for (int di = 0; di < d; di++)
                acc[localRow][di] = __shfl_sync(0xFFFFFFFF, acc[localRow][di], 0);
        }

        __syncthreads();
    }

    // 写回 O（归一化：除以 l）
    #pragma unroll
    for (int localRow = 0; localRow < ROWS_PER_WARP; localRow++) {
        int qi = qRowStart + localRow;
        int globalRow = qTileRow + qi;
        if (qi >= Br || globalRow >= M)
            continue;
        float inv_l = 1.0f / l_arr[localRow];
        #pragma unroll
        for (int di = lane; di < d; di += 32)
            O[globalRow * d + di] = acc[localRow][di] * inv_l;
    }
}

// ---------- CPU 参考实现 ----------
void attention_cpu(const float* Q, const float* K, const float* V, float* O, int M, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * sizeof(float));
    float* P = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < M; ++i) {
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
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int d = (argc > 3) ? atoi(argv[3]) : 64;
    if (d != D) {
        printf("tiling 版当前固定 d=%d（与 W5D3 README 一致），传入 d=%d 不支持\n", D, d);
        return 1;
    }

    size_t q_bytes = (size_t)M * d * sizeof(float);
    size_t kv_bytes = (size_t)N * d * sizeof(float);
    printf("M=%d N=%d d=%d  Q=%.2fMB  K/V=%.2fMB each\n", M, N, d, q_bytes / 1e6, kv_bytes / 1e6);

    float *hQ = (float*)malloc(q_bytes);
    float *hK = (float*)malloc(kv_bytes);
    float *hV = (float*)malloc(kv_bytes);
    float *hO = (float*)malloc(q_bytes);
    float *hRef = (float*)malloc(q_bytes);
    srand(42);
    for (int i = 0; i < M * d; ++i)
        hQ[i] = ((rand() % 2000) - 1000) / 100.0f;
    for (int i = 0; i < N * d; ++i) {
        hK[i] = ((rand() % 2000) - 1000) / 100.0f;
        hV[i] = ((rand() % 2000) - 1000) / 100.0f;
    }

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, q_bytes);
    cudaMemcpy(dQ, hQ, q_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, kv_bytes);
    cudaMemcpy(dK, hK, kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, kv_bytes);
    cudaMemcpy(dV, hV, kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, q_bytes);

    dim3 grid((M + Br - 1) / Br);
    dim3 block(THREADS_PER_BLOCK);

    // warmup
    flash_attention_v2_kernel<<<grid, block>>>(dQ, dK, dV, dO, M, N, d);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    flash_attention_v2_kernel<<<grid, block>>>(dQ, dK, dV, dO, M, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(hO, dO, q_bytes, cudaMemcpyDeviceToHost);
    attention_cpu(hQ, hK, hV, hRef, M, N, d);

    float maxDiff = 0;
    for (int i = 0; i < M * d; ++i)
        maxDiff = fmaxf(maxDiff, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff < 1e-3f ? "PASS" : "FAIL");
    printf("GPU Time: %.3f ms  (grid=%d, block=%d, SRAM=48KB)\n",
           ms, (M + Br - 1) / Br, THREADS_PER_BLOCK);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    free(hQ); free(hK); free(hV); free(hO); free(hRef);
    return 0;
}

// 12-multi-head-attention.cu —— Multi-Head FlashAttention
// 编译命令: nvcc -O3 -arch=sm_120 12-multi-head-attention.cu -o mha
// 运行:     ./mha 2 4 512 64   (B H N d)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define D_MAX 128
#define WARP_SIZE 32
#define D_PER_THREAD (D_MAX / WARP_SIZE) // 4
#define NUM_WARPS 8
#define Br NUM_WARPS                       // 8 Q rows per block
#define Bc 32                              // K/V tile size
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS) // 256

// ---------- warp / block 归约（复用 Day 4 / Day 5）----------
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

// ---------- 标准 MHA：物化 S/P 到 HBM（1 block = 1 行 query）----------
__global__ void mha_standard_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                    const float* __restrict__ V, float* __restrict__ S, float* __restrict__ P,
                                    float* __restrict__ O, int B, int H, int N, int d) {
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float row_max_shm, row_sum_shm;
    __shared__ float q_shm[D_MAX];
    int batch = blockIdx.z, head = blockIdx.y, i = blockIdx.x;
    int tid = threadIdx.x;
    int bhQ = (batch * H + head) * N * d;
    int bhS = (batch * H + head) * N * N;
    float scale = 1.0f / sqrtf((float)d);
    if (i >= N)
        return;

    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[bhQ + i * d + t];
    __syncthreads();
    // ① S[i][k] = Q[i]·K[k]/√d → 写 HBM
    for (int k = tid; k < N; k += BLOCK_SIZE) {
        float s = 0.f;
        for (int t = 0; t < d; ++t)
            s += q_shm[t] * K[bhQ + k * d + t];
        S[bhS + i * N + k] = s * scale;
    }
    __syncthreads();
    // ② row max（数值稳定）
    float lm = -INFINITY;
    for (int k = tid; k < N; k += BLOCK_SIZE)
        lm = fmaxf(lm, S[bhS + i * N + k]);
    float rmax = block_reduce_max(lm, red);
    if (tid == 0)
        row_max_shm = rmax;
    __syncthreads();
    rmax = row_max_shm;
    // ③ P[i][k] = exp(S[i][k]-rmax) → 写 HBM; 求 sum
    float ls = 0.f;
    for (int k = tid; k < N; k += BLOCK_SIZE) {
        float p = expf(S[bhS + i * N + k] - rmax);
        P[bhS + i * N + k] = p;
        ls += p;
    }
    float rsum = block_reduce_sum(ls, red);
    if (tid == 0)
        row_sum_shm = rsum;
    __syncthreads();
    rsum = row_sum_shm;
    // ④ O[i][t] = Σ_k (P[i][k]/rsum)·V[k][t]
    float inv = 1.0f / rsum;
    for (int t = tid; t < d; t += BLOCK_SIZE) {
        float acc = 0.f;
        for (int k = 0; k < N; ++k)
            acc += P[bhS + i * N + k] * V[bhQ + k * d + t];
        O[bhQ + i * d + t] = acc * inv;
    }
}

// ---------- Flash MHA：online softmax + tiling，不物化 S/P ----------
// gridDim = (N/Br, H, B), blockDim = (32, NUM_WARPS)
// 每 warp 处理 1 行 query，32 lane 拆分 d 维
__global__ void mha_flash_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
                                 float* __restrict__ O, int B, int H, int N, int d) {
    __shared__ float s_Q[Br][D_MAX];
    __shared__ float s_K[Bc][D_MAX];
    __shared__ float s_V[Bc][D_MAX];

    int batch = blockIdx.z;
    int head = blockIdx.y;
    int qTileBase = blockIdx.x * Br;
    int lane = threadIdx.x; // 0..31, 拆分 d 维
    int wid = threadIdx.y;  // 0..7, 1 warp = 1 Q row
    int qRow = qTileBase + wid;
    int bhOff = (batch * H + head) * N * d;
    float scale = 1.0f / sqrtf((float)d);

    // ---- 协作加载 Q tile 到 shared memory ----
    for (int idx = wid * WARP_SIZE + lane; idx < Br * d; idx += BLOCK_SIZE) {
        int row = idx / d, col = idx % d;
        int r = qTileBase + row;
        s_Q[row][col] = (r < N) ? Q[bhOff + r * d + col] : 0.f;
    }
    __syncthreads();

    // ---- 每 warp 的 running state（1 行 query）----
    float m = -INFINITY, l = 0.f;
    float o_reg[D_PER_THREAD], q_reg[D_PER_THREAD];
    #pragma unroll
    for (int dd = 0; dd < D_PER_THREAD; dd++) {
        o_reg[dd] = 0.f;
        int d_idx = lane + dd * WARP_SIZE; // coalesced d 映射
        q_reg[dd] = (d_idx < d && qRow < N) ? s_Q[wid][d_idx] : 0.f;
    }

    // ---- 外层循环：遍历 K/V tile（Bc 行/tile）----
    for (int kvStart = 0; kvStart < N; kvStart += Bc) {
        // 协作加载 K/V tile
        for (int idx = wid * WARP_SIZE + lane; idx < Bc * d; idx += BLOCK_SIZE) {
            int row = idx / d, col = idx % d;
            int kvRow = kvStart + row;
            if (kvRow < N) {
                s_K[row][col] = K[bhOff + kvRow * d + col];
                s_V[row][col] = V[bhOff + kvRow * d + col];
            } else {
                s_K[row][col] = 0.f;
                s_V[row][col] = 0.f;
            }
        }
        __syncthreads();

        // 内层循环：逐行处理 K/V（从 shared memory 读，Br 个 warp 复用）
        for (int k = 0; k < Bc && kvStart + k < N; k++) {
            // ① s_k = dot(Q[row], K[k]) / √d（warp 内 partial sum → reduce → broadcast）
            float partial = 0.f;
            #pragma unroll
            for (int dd = 0; dd < D_PER_THREAD; dd++) {
                int d_idx = lane + dd * WARP_SIZE;
                if (d_idx < d)
                    partial += q_reg[dd] * s_K[k][d_idx];
            }
            float s_k = warp_reduce_sum(partial) * scale;
            s_k = __shfl_sync(0xffffffff, s_k, 0); // broadcast to all lanes

            // ② online softmax 三公式
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new);
            float p = expf(s_k - m_new);
            float l_new = l * alpha + p;
            float o_scale = (l * alpha) / l_new; // o 的 rescale 因子
            float v_scale = p / l_new;           // 新 V 的权重

// ③ 累加输出：o = o·o_scale + v_scale·V[k]
            #pragma unroll
            for (int dd = 0; dd < D_PER_THREAD; dd++) {
                int d_idx = lane + dd * WARP_SIZE;
                o_reg[dd] = o_reg[dd] * o_scale;
                if (d_idx < d)
                    o_reg[dd] += v_scale * s_V[k][d_idx];
            }
            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    // ---- 写回输出 ----
    if (qRow < N) {
        #pragma unroll
        for (int dd = 0; dd < D_PER_THREAD; dd++) {
            int d_idx = lane + dd * WARP_SIZE;
            if (d_idx < d)
                O[bhOff + qRow * d + d_idx] = o_reg[dd];
        }
    }
}

// ---------- CPU 参考实现 ----------
void mha_cpu(const float* Q, const float* K, const float* V, float* O, int B, int H, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    float* S = (float*)malloc(N * sizeof(float));
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            int bh = (b * H + h) * N * d;
            for (int i = 0; i < N; ++i) {
                float mx = -INFINITY;
                for (int k = 0; k < N; ++k) {
                    float s = 0.f;
                    for (int t = 0; t < d; ++t)
                        s += Q[bh + i * d + t] * K[bh + k * d + t];
                    S[k] = s * scale;
                    mx = fmaxf(mx, S[k]);
                }
                float sum = 0.f;
                for (int k = 0; k < N; ++k) {
                    S[k] = expf(S[k] - mx);
                    sum += S[k];
                }
                for (int t = 0; t < d; ++t) {
                    float acc = 0.f;
                    for (int k = 0; k < N; ++k)
                        acc += S[k] * V[bh + k * d + t];
                    O[bh + i * d + t] = acc / sum;
                }
            }
        }
    free(S);
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 2;
    int H = (argc > 2) ? atoi(argv[2]) : 4;
    int N = (argc > 3) ? atoi(argv[3]) : 512;
    int d = (argc > 4) ? atoi(argv[4]) : 64;
    if (d > D_MAX) {
        printf("d must be <= %d\n", D_MAX);
        return 1;
    }

    size_t qkv = (size_t)B * H * N * d * sizeof(float);
    size_t sp = (size_t)B * H * N * N * sizeof(float);
    printf("B=%d H=%d N=%d d=%d  QKV=%.2f MB  S/P(std)=%.2f MB each\n", B, H, N, d, 3.0 * qkv / 1e6, sp / 1e6);

    float *hQ = (float*)malloc(qkv), *hK = (float*)malloc(qkv), *hV = (float*)malloc(qkv);
    float *hOs = (float*)malloc(qkv), *hOf = (float*)malloc(qkv), *hRef = (float*)malloc(qkv);
    srand(42);
    for (size_t i = 0; i < (size_t)B * H * N * d; ++i) {
        hQ[i] = ((rand() % 2000) - 1000) / 100.f;
        hK[i] = ((rand() % 2000) - 1000) / 100.f;
        hV[i] = ((rand() % 2000) - 1000) / 100.f;
    }

    float *dQ, *dK, *dV, *dS, *dP, *dOs, *dOf;
    cudaMalloc(&dQ, qkv);
    cudaMemcpy(dQ, hQ, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, qkv);
    cudaMemcpy(dK, hK, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, qkv);
    cudaMemcpy(dV, hV, qkv, cudaMemcpyHostToDevice);
    cudaMalloc(&dS, sp);
    cudaMalloc(&dP, sp);
    cudaMalloc(&dOs, qkv);
    cudaMalloc(&dOf, qkv);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ===== 标准 MHA =====
    dim3 stdGrid(N, H, B), stdBlock(BLOCK_SIZE);
    cudaEventRecord(t0);
    mha_standard_kernel<<<stdGrid, stdBlock>>>(dQ, dK, dV, dS, dP, dOs, B, H, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms_s = 0;
    cudaEventElapsedTime(&ms_s, t0, t1);

    // ===== Flash MHA =====
    dim3 flGrid((N + Br - 1) / Br, H, B), flBlock(WARP_SIZE, NUM_WARPS);
    cudaEventRecord(t0);
    mha_flash_kernel<<<flGrid, flBlock>>>(dQ, dK, dV, dOf, B, H, N, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms_f = 0;
    cudaEventElapsedTime(&ms_f, t0, t1);

    printf("standard: %.3f ms   flash: %.3f ms\n", ms_s, ms_f);

    // ===== CPU 验证 =====
    mha_cpu(hQ, hK, hV, hRef, B, H, N, d);
    cudaMemcpy(hOs, dOs, qkv, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOf, dOf, qkv, cudaMemcpyDeviceToHost);
    float dS_err = 0, dF_err = 0;
    for (size_t i = 0; i < (size_t)B * H * N * d; ++i) {
        dS_err = fmaxf(dS_err, fabsf(hOs[i] - hRef[i]));
        dF_err = fmaxf(dF_err, fabsf(hOf[i] - hRef[i]));
    }
    printf("standard max diff: %.2e (%s)\n", dS_err, dS_err < 1e-3f ? "PASS" : "FAIL");
    printf("flash    max diff: %.2e (%s)\n", dF_err, dF_err < 1e-3f ? "PASS" : "FAIL");

    // ===== DRAM 流量估算 =====
    double bytes_KV = 2.0 * (double)B * H * N * N * d * sizeof(float); // K+V 被所有 query 读
    double bytes_SP = 4.0 * (double)B * H * N * N * sizeof(float);     // S+P 物化写读
    printf("est. DRAM: standard=%.2f GB  flash=%.2f GB  (flash 省 S/P=%.2f GB)\n",
           (bytes_KV + bytes_SP + 3.0 * qkv) / 1e9, (bytes_KV / Br + 3.0 * qkv) / 1e9, bytes_SP / 1e9);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dS);
    cudaFree(dP);
    cudaFree(dOs);
    cudaFree(dOf);
    free(hQ);
    free(hK);
    free(hV);
    free(hOs);
    free(hOf);
    free(hRef);
    return 0;
}

// 56-linear-attention.cu —— Linear Self-Attention (kernel trick, O(Md²))
// 编译命令: nvcc -O3 -arch=sm_80 56-linear-attention.cu -o linear_attention
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define D_MAX 128
#define BLOCK 256

// φ(x) = ELU(x) + 1 = (x>0) ? (x+1) : expf(x)
__inline__ __device__ float phi(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

// block 内求和归约（warp shuffle + 跨 warp 共享内存），要求 blockDim 为 32 的倍数
__inline__ __device__ float block_reduce_sum(float val) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    __shared__ float warp_sum[16];          // 最多 16 个 warp（512 线程）
    if (lane == 0) warp_sum[warp] = val;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    if (warp == 0) {
        val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (tid == 0) warp_sum[0] = val;
    }
    __syncthreads();
    return warp_sum[0];
}

// ① elementwise: φQ = φ(Q), φK = φ(K)  —— grid-stride + coalesced
__global__ void phi_kernel(const float* Q, const float* K,
                           float* phiQ, float* phiK, int n) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= n) return;
    phiQ[idx] = phi(Q[idx]);
    phiK[idx] = phi(K[idx]);
}

// ② S = φK^T @ V，输出 d×d，归约维度 M  —— naive GEMM，一线程一 (i,j)
__global__ void compute_S_kernel(const float* phiK, const float* V,
                                 float* S, int M, int d) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    int total = d * d;
    if (idx >= total) return;
    int i = idx / d;
    int j = idx % d;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m)
        acc += phiK[m * d + i] * V[m * d + j];
    S[i * d + j] = acc;
}

// ② z[i] = Σ_m φK[m][i]  —— 一 block 归约一列，warp shuffle reduction
__global__ void compute_z_kernel(const float* phiK, float* z, int M, int d) {
    int i = blockIdx.x;
    float acc = 0.0f;
    for (int m = threadIdx.x; m < M; m += BLOCK)
        acc += phiK[m * d + i];
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        z[i] = acc;
}

// ③④ fused: output[m] = (φQ[m] @ S) / (φQ[m] @ z)
//    一 block 处理一行 query，d 线程，smem_q 缓存 φQ 行，thread0 归约分母广播
__global__ void compute_output_kernel(const float* phiQ, const float* S,
                                      const float* z, float* output, int M, int d) {
    int m = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem_q[D_MAX];
    if (tid < d) smem_q[tid] = phiQ[m * d + tid];
    __syncthreads();

    // ③ num = φQ[m] @ S 的第 tid 列：Σ_i smem_q[i] * S[i][tid]
    float num = 0.0f;
    if (tid < d)
        for (int i = 0; i < d; ++i)
            num += smem_q[i] * S[i * d + tid];

    // ④ den = φQ[m] @ z = Σ_i smem_q[i] * z[i]，thread 0 串行归约后广播
    __shared__ float smem_den;
    if (tid == 0) {
        float den = 0.0f;
        for (int i = 0; i < d; ++i)
            den += smem_q[i] * z[i];
        smem_den = den;
    }
    __syncthreads();

    if (tid < d)
        output[m * d + tid] = num / smem_den;
}

// ---------- LeetGPU 提交版本：见 §4.1 solve ----------

float phi_cpu(float x) { return (x > 0.0f) ? (x + 1.0f) : expf(x); }

void linear_attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int M, int d) {
    float *phiQ = (float*)malloc((size_t)M * d * sizeof(float));
    float *phiK = (float*)malloc((size_t)M * d * sizeof(float));
    for (int i = 0; i < M * d; ++i) { phiQ[i] = phi_cpu(Q[i]); phiK[i] = phi_cpu(K[i]); }
    float *S = (float*)malloc((size_t)d * d * sizeof(float));
    float *z = (float*)malloc(d * sizeof(float));
    for (int i = 0; i < d; ++i) { z[i] = 0; for (int m = 0; m < M; ++m) z[i] += phiK[m * d + i]; }
    for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
        float s = 0; for (int m = 0; m < M; ++m) s += phiK[m * d + i] * V[m * d + j]; S[i * d + j] = s;
    }
    for (int m = 0; m < M; ++m) {
        float den = 0; for (int i = 0; i < d; ++i) den += phiQ[m * d + i] * z[i];
        for (int j = 0; j < d; ++j) {
            float num = 0; for (int i = 0; i < d; ++i) num += phiQ[m * d + i] * S[i * d + j];
            O[m * d + j] = num / den;
        }
    }
    free(phiQ); free(phiK); free(S); free(z);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2;
    int d = (argc > 2) ? atoi(argv[2]) : 4;
    if (d > D_MAX) { printf("d must be <= %d\n", D_MAX); return 1; }

    size_t md = (size_t)M * d * sizeof(float);
    float *hQ = (float*)malloc(md), *hK = (float*)malloc(md), *hV = (float*)malloc(md);
    float *hO = (float*)malloc(md), *hRef = (float*)malloc(md);

    if (M == 2 && d == 4) {  // 官方 Example 1
        float Q0[] = {1,0,0,0, 0,1,0,0}, K0[] = {1,0,0,0, 0,1,0,0}, V0[] = {1,2,3,4, 5,6,7,8};
        memcpy(hQ, Q0, sizeof(Q0)); memcpy(hK, K0, sizeof(K0)); memcpy(hV, V0, sizeof(V0));
    } else {
        srand(42);
        for (int i = 0; i < M * d; ++i) {
            hQ[i] = ((rand() % 2000) - 1000) / 100.0f;
            hK[i] = ((rand() % 2000) - 1000) / 100.0f;
            hV[i] = ((rand() % 2000) - 1000) / 100.0f;
        }
    }

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, md); cudaMemcpy(dQ, hQ, md, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, md); cudaMemcpy(dK, hK, md, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, md); cudaMemcpy(dV, hV, md, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, md);

    // 直接发射四个 kernel（与 §4.1 solve 内部一致）
    float *dphiQ, *dphiK, *dS, *dz;
    cudaMalloc(&dphiQ, md); cudaMalloc(&dphiK, md);
    cudaMalloc(&dS, (size_t)d * d * sizeof(float));
    cudaMalloc(&dz, d * sizeof(float));
    int n = M * d;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    phi_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dQ, dK, dphiQ, dphiK, n);
    compute_S_kernel<<<(d * d + BLOCK - 1) / BLOCK, BLOCK>>>(dphiK, dV, dS, M, d);
    compute_z_kernel<<<d, BLOCK>>>(dphiK, dz, M, d);
    compute_output_kernel<<<M, d>>>(dphiQ, dS, dz, dO, M, d);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(hO, dO, md, cudaMemcpyDeviceToHost);
    linear_attention_cpu(hQ, hK, hV, hRef, M, d);
    float diff = 0;
    for (int i = 0; i < M * d; ++i) diff = fmaxf(diff, fabsf(hO[i] - hRef[i]));
    printf("M=%d d=%d  time=%.3f ms  max diff=%.2e (%s)\n",
           M, d, ms, diff, diff < 1e-3f ? "PASS" : "FAIL");

    if (M == 2 && d == 4) {
        printf("output:\n");
        for (int m = 0; m < M; ++m) { for (int j = 0; j < d; ++j) printf("%.4f ", hO[m*d+j]); printf("\n"); }
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(dphiQ); cudaFree(dphiK); cudaFree(dS); cudaFree(dz);
    free(hQ); free(hK); free(hV); free(hO); free(hRef);
    return 0;
}

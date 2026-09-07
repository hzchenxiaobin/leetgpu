// 9-1d-convolution.cu —— 1D Convolution (shared memory + halo)
// 编译命令: nvcc -o conv1d 9-1d-convolution.cu -O3 -arch=sm_120
// 运行命令: ./conv1d

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 256

// 卷积核放 constant 内存（小且广播访问）
__constant__ float c_kernel[64];

__global__ void conv1dKernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int N, int K) {
    int half = K / 2;
    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILE;
    int i = block_start + tid;          // 该线程负责的输出下标

    __shared__ float s_x[TILE + 2 * 32]; // halo 最大 half=32，足够 K<=65

    // ---- 阶段 ①：协作加载 tile + halo ----
    // tile 区域：[block_start - half, block_start + TILE + half)
    int s_len = TILE + 2 * half;
    for (int idx = tid; idx < s_len; idx += blockDim.x) {
        int gidx = block_start - half + idx;
        s_x[idx] = (gidx >= 0 && gidx < N) ? x[gidx] : 0.0f;
    }
    __syncthreads();

    // ---- 阶段 ②：计算（从 shared memory 读） ----
    if (i < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; j++) {
            acc += s_x[tid + j] * c_kernel[j];   // tid + j 已含 left halo 偏移
        }
        y[i] = acc;
    }
}

void initArray(float* a, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        a[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > eps) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    return true;
}

int main() {
    int N = 1 << 20;   // 1M
    int K = 5;
    size_t bytes = N * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_kernel = (float*)malloc(K * sizeof(float));
    initArray(h_x, N);
    initArray(h_kernel, K);

    // CPU 参考
    int half = K / 2;
    for (int i = 0; i < N; i++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) {
            int idx = i + j - half;
            acc += (idx >= 0 && idx < N) ? h_x[idx] * h_kernel[j] : 0.0f;
        }
        h_ref[i] = acc;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_kernel, K * sizeof(float));

    int threads = TILE;
    int blocks = (N + TILE - 1) / TILE;

    conv1dKernel<<<blocks, threads>>>(d_x, d_y, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    conv1dKernel<<<blocks, threads>>>(d_x, d_y, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bool ok = checkResult(h_y, h_ref, N, 1e-4f);

    printf("=== 1D Convolution (Shared Memory + Halo) ===\n");
    printf("N=%d, K=%d, TILE=%d\n", N, K, TILE);
    printf("Kernel time: %.3f ms\n", ms);
    float gbytes = (float)(N + N) * sizeof(float) / (ms * 1e6);
    printf("Effective bandwidth: %.1f GB/s\n", gbytes);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref); free(h_kernel);
    return 0;
}

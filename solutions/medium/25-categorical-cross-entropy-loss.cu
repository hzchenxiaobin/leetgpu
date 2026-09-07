// 25-categorical-cross-entropy-loss.cu —— Cross Entropy Loss: 一 block 一行 + 两遍扫描 + warp shuffle
// 编译命令: nvcc -O3 -arch=sm_80 25-categorical-cross-entropy-loss.cu -o categorical_cross_entropy

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP 32
#define BLOCK_SIZE 256
#define MAX_WARPS_PER_BLOCK (BLOCK_SIZE / WARP)

// warp 内树形归约求 max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// warp 内树形归约求 sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// block 内归约求 max（warp shuffle + shared memory 两级）
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < MAX_WARPS_PER_BLOCK) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// block 内归约求 sum（同上模板）
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < MAX_WARPS_PER_BLOCK) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// Cross Entropy Loss kernel: 一个 block 处理一行
__global__ void cross_entropy_kernel(const float* __restrict__ logits,
                                     const int* __restrict__ true_labels,
                                     float* __restrict__ loss,
                                     int N, int C) {
    int j = blockIdx.x;
    if (j >= N) return;

    const float* row = logits + (long)j * C;
    __shared__ float shared[MAX_WARPS_PER_BLOCK];

    // ===== Pass 1: 求行最大值 m =====
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < C; k += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[k]);
    }
    float m = block_reduce_max(local_max, shared);

    // ===== Pass 2: 求 Σ exp(z_k - m) =====
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < C; k += BLOCK_SIZE) {
        local_sum += expf(row[k] - m);
    }
    float s = block_reduce_sum(local_sum, shared);

    // ===== 最终计算: thread 0 算 loss 并 atomicAdd =====
    if (threadIdx.x == 0) {
        float lse = m + logf(s);
        float true_logit = row[true_labels[j]];
        float loss_j = (lse - true_logit) / (float)N;
        atomicAdd(loss, loss_j);
    }
}

// ===== Host 端：分配、launch、验证 =====
int main() {
    // 测试数据: N=2, C=3, true_labels=[1,1]
    int N = 2, C = 3;
    float h_logits[] = {1.0f, 2.0f, 0.5f, 0.1f, 3.0f, 1.5f};
    int h_labels[] = {1, 1};
    float h_loss = 0.0f;

    // CPU 参考计算
    float ref_loss = 0.0f;
    for (int j = 0; j < N; j++) {
        float m = h_logits[j * C];
        for (int k = 1; k < C; k++) m = fmaxf(m, h_logits[j * C + k]);
        float s = 0.0f;
        for (int k = 0; k < C; k++) s += expf(h_logits[j * C + k] - m);
        ref_loss += (m + logf(s)) - h_logits[j * C + h_labels[j]];
    }
    ref_loss /= N;

    // GPU 分配
    float *d_logits, *d_loss;
    int *d_labels;
    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_logits, h_logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, sizeof(float));

    // launch: N 个 block，每个 block 256 threads
    cross_entropy_kernel<<<N, BLOCK_SIZE>>>(d_logits, d_labels, d_loss, N, C);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    printf("CPU ref loss = %.7f\n", ref_loss);
    printf("GPU     loss = %.7f\n", h_loss);
    float diff = fabsf(ref_loss - h_loss);
    printf("diff = %.7e  %s\n", diff, diff < 1e-5 ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: N=10000, C=1000 =====
    int N2 = 10000, C2 = 1000;
    float *d_logits2, *d_loss2;
    int *d_labels2;
    cudaMalloc(&d_logits2, (size_t)N2 * C2 * sizeof(float));
    cudaMalloc(&d_labels2, N2 * sizeof(int));
    cudaMalloc(&d_loss2, sizeof(float));

    // 随机初始化 logits [-10, 10]
    float* h_logits2 = (float*)malloc((size_t)N2 * C2 * sizeof(float));
    int* h_labels2 = (int*)malloc(N2 * sizeof(int));
    srand(42);
    for (size_t i = 0; i < (size_t)N2 * C2; i++)
        h_logits2[i] = -10.0f + 20.0f * (rand() / (float)RAND_MAX);
    for (int i = 0; i < N2; i++)
        h_labels2[i] = rand() % C2;

    cudaMemcpy(d_logits2, h_logits2, (size_t)N2 * C2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels2, h_labels2, N2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_loss2, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cross_entropy_kernel<<<N2, BLOCK_SIZE>>>(d_logits2, d_labels2, d_loss2, N2, C2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float perf_loss;
    cudaMemcpy(&perf_loss, d_loss2, sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nPerf test: N=%d, C=%d\n", N2, C2);
    printf("GPU loss = %.7f\n", perf_loss);
    printf("Kernel time = %.3f ms\n", ms);
    printf("Data read = %.2f MB (2 passes × %d×%d×4B)\n",
           2.0f * N2 * C2 * 4 / 1e6, N2, C2);
    printf("Effective bandwidth = %.2f GB/s\n",
           2.0f * N2 * C2 * 4 / (ms * 1e6));

    // cleanup
    cudaFree(d_logits); cudaFree(d_labels); cudaFree(d_loss);
    cudaFree(d_logits2); cudaFree(d_labels2); cudaFree(d_loss2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_logits2); free(h_labels2);

    return 0;
}

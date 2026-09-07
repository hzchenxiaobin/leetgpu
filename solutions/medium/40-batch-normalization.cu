// 40-batch-normalization.cu —— Batch Normalization Forward (fused: mean+var+normalize in one kernel)
// 编译命令: nvcc -o batchnorm 40-batch-normalization.cu -O3 -arch=sm_120
// 运行命令: ./batchnorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Warp 内归约（求和）
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// Block 内归约：warp reduce → shared memory → 第一个 warp 汇总
__inline__ __device__ float blockReduceSum(float val, float* s_partial, int tid) {
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) s_partial[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (tid < num_warps) ? s_partial[lane] : 0.0f;
        val = warpReduceSum(val);
        if (lane == 0) s_partial[0] = val;
    }
    __syncthreads();
    return s_partial[0];
}

// Fused BatchNorm: 每个 block 处理一个通道 c
// gridDim = (1, C), blockDim = NUM_THREADS
__global__ void batchnormForward(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int N, int C, int H, int W, float eps) {
    int c = blockIdx.y;
    int spatial = N * H * W;        // 每通道元素数
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    __shared__ float s_partial[32]; // warp 部分和
    __shared__ float s_mean;
    __shared__ float s_inv_std;

    // NCHW 布局：通道 c 的元素地址 = n*C*H*W + c*H*W + h*W + w
    // 通道 c 的起始偏移 = c*H*W，步长 = C*H*W（每个 n 跳一个通道块）
    int hw = H * W;
    int chw = C * hw;

    // ---- 阶段 ①：求 mean ----
    float local_sum = 0.0f;
    for (int idx = tid; idx < spatial; idx += num_threads) {
        int n = idx / hw;
        int rem = idx % hw;
        local_sum += x[n * chw + c * hw + rem];
    }
    float mean = blockReduceSum(local_sum, s_partial, tid) / (float)spatial;
    if (tid == 0) s_mean = mean;
    __syncthreads();
    mean = s_mean;

    // ---- 阶段 ②：求 var = E[(x - mean)^2] ----
    float local_sqsum = 0.0f;
    for (int idx = tid; idx < spatial; idx += num_threads) {
        int n = idx / hw;
        int rem = idx % hw;
        float d = x[n * chw + c * hw + rem] - mean;
        local_sqsum += d * d;
    }
    float var = blockReduceSum(local_sqsum, s_partial, tid) / (float)spatial;
    float inv_std = 1.0f / sqrtf(var + eps);
    if (tid == 0) s_inv_std = inv_std;
    __syncthreads();
    inv_std = s_inv_std;

    // ---- 阶段 ③：融合 normalize 写回 ----
    float g = gamma[c];
    float b = beta[c];
    for (int idx = tid; idx < spatial; idx += num_threads) {
        int n = idx / hw;
        int rem = idx % hw;
        int gidx = n * chw + c * hw + rem;
        y[gidx] = g * (x[gidx] - mean) * inv_std + b;
    }
}

void initMatrix(float* mat, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        mat[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
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
    int N = 64, C = 128, H = 32, W = 32;
    int total = N * C * H * W;
    size_t bytes = total * sizeof(float);
    float eps = 1e-5f;

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_gamma = (float*)malloc(C * sizeof(float));
    float *h_beta = (float*)malloc(C * sizeof(float));
    initMatrix(h_x, total);
    initMatrix(h_gamma, C);
    initMatrix(h_beta, C);

    // CPU 参考
    int spatial = N * H * W;
    for (int c = 0; c < C; c++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < H * W; hw++)
                sum += h_x[n * C * H * W + c * H * W + hw];
        float mean = sum / spatial;
        float sqsum = 0.0f;
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < H * W; hw++) {
                float d = h_x[n * C * H * W + c * H * W + hw] - mean;
                sqsum += d * d;
            }
        float var = sqsum / spatial;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int n = 0; n < N; n++)
            for (int hw = 0; hw < H * W; hw++) {
                int idx = n * C * H * W + c * H * W + hw;
                h_ref[idx] = h_gamma[c] * (h_x[idx] - mean) * inv_std + h_beta[c];
            }
    }

    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    dim3 grid(1, C);
    dim3 block(threads);

    // warmup + timing
    batchnormForward<<<grid, block>>>(d_x, d_gamma, d_beta, d_y, N, C, H, W, eps);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    batchnormForward<<<grid, block>>>(d_x, d_gamma, d_beta, d_y, N, C, H, W, eps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bool ok = checkResult(h_y, h_ref, total, 1e-3f);

    printf("=== BatchNorm Forward (Fused) ===\n");
    printf("N=%d C=%d H=%d W=%d, threads/block=%d\n", N, C, H, W, threads);
    printf("Kernel time: %.3f ms\n", ms);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_gamma); cudaFree(d_beta);
    free(h_x); free(h_y); free(h_ref); free(h_gamma); free(h_beta);
    return 0;
}

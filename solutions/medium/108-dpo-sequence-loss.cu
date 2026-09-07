// 108-dpo-sequence-loss.cu —— DPO Sequence Loss: fused element-wise + reduction
// 编译命令: nvcc -O3 -arch=sm_120 108-dpo-sequence-loss.cu -o dpo
// 运行:     ./dpo

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)   // 8

// ---- warp 级归约：__shfl_down_sync 折半累加到 lane 0 ----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 + 广播 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            shared[0] = val;               // 广播 slot
    }
    __syncthreads();
    return shared[0];
}

// ---- 稳定 softplus: log(1+exp(z)) = max(z,0) + log1p(exp(-|z|)) ----
__device__ __forceinline__ float stable_softplus(float z) {
    return fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)));
}

// ---- 优化版：融合 kernel，grid-stride + 两级归约 + atomicAdd ----
__global__ void dpo_loss_kernel(
    const float* __restrict__ chosen_logps,
    const float* __restrict__ rejected_logps,
    const float* __restrict__ chosen_ref_logps,
    const float* __restrict__ rejected_ref_logps,
    float* output, float beta, int B) {

    __shared__ float shared[NUM_WARPS + 1];   // +1 避免 bank conflict

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // ① grid-stride 累加每元素 loss
    float local_sum = 0.0f;
    for (int i = tid; i < B; i += stride) {
        float chosen_margin = chosen_logps[i] - rejected_logps[i];
        float ref_margin    = chosen_ref_logps[i] - rejected_ref_logps[i];
        float logits = beta * (chosen_margin - ref_margin);
        float loss = stable_softplus(-logits);    // -log σ(logits) = softplus(-logits)
        local_sum += loss;
    }

    // ② block 级归约
    float block_sum = block_reduce_sum(local_sum, shared);

    // ③ thread 0 原子加到 output
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum / (float)B);
}

// ---- 朴素版：5 个独立 kernel（对比基准） ----
__global__ void naive_elementwise(const float* a, const float* b, float* out, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) out[i] = a[i] - b[i];
}
__global__ void naive_logits(const float* cm, const float* rm, float* logits, float beta, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) logits[i] = beta * (cm[i] - rm[i]);
}
__global__ void naive_softplus(const float* logits, float* loss, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) loss[i] = stable_softplus(-logits[i]);
}
__global__ void naive_reduce(const float* loss, float* output, int B) {
    __shared__ float shared[NUM_WARPS + 1];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < B; i += BLOCK_SIZE)
        local += loss[i];
    float sum = block_reduce_sum(local, shared);
    if (tid == 0) output[0] = sum / B;
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 65536;
    float beta = (argc > 2) ? (float)atof(argv[2]) : 0.1f;
    size_t bytes = (size_t)B * sizeof(float);
    printf("B = %d, beta = %.2f  (%.2f KB per input)\n", B, beta, bytes / 1e3);

    // 分配 host
    float *hCh = (float*)malloc(bytes), *hRej = (float*)malloc(bytes);
    float *hChRef = (float*)malloc(bytes), *hRejRef = (float*)malloc(bytes);
    float *hOut = (float*)malloc(sizeof(float)), *hRef = (float*)malloc(sizeof(float));

    srand(42);
    for (int i = 0; i < B; ++i) {
        hCh[i]    = (float)((rand() % 2000) - 1000) / 10.0f;
        hRej[i]   = (float)((rand() % 2000) - 1000) / 10.0f;
        hChRef[i] = (float)((rand() % 2000) - 1000) / 10.0f;
        hRejRef[i]= (float)((rand() % 2000) - 1000) / 10.0f;
    }

    // CPU 参考（用 double 累加）
    {
        double sum = 0.0;
        for (int i = 0; i < B; ++i) {
            float cm = hCh[i] - hRej[i];
            float rm = hChRef[i] - hRejRef[i];
            float logits = beta * (cm - rm);
            float z = -logits;
            float loss = fmaxf(z, 0.0f) + log1pf(expf(-fabsf(logits)));
            sum += loss;
        }
        hRef[0] = (float)(sum / B);
    }

    // 分配 device
    float *dCh, *dRej, *dChRef, *dRejRef, *dOut;
    CHECK_CUDA(cudaMalloc(&dCh, bytes));
    CHECK_CUDA(cudaMalloc(&dRej, bytes));
    CHECK_CUDA(cudaMalloc(&dChRef, bytes));
    CHECK_CUDA(cudaMalloc(&dRejRef, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dCh, hCh, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRej, hRej, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dChRef, hChRef, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRejRef, hRejRef, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (B + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (numBlocks > 65536) numBlocks = 65536;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ---- 优化版（fused） ----
    CHECK_CUDA(cudaMemsetAsync(dOut, 0, sizeof(float)));
    cudaEventRecord(t0);
    dpo_loss_kernel<<<numBlocks, BLOCK_SIZE>>>(
        dCh, dRej, dChRef, dRejRef, dOut, beta, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_fused = 0; cudaEventElapsedTime(&ms_fused, t0, t1);
    CHECK_CUDA(cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    double err = fabs((double)hOut[0] - hRef[0]);
    printf("[fused]  time: %.4f ms  output: %.6f  ref: %.6f  err: %.3e  %s\n",
           ms_fused, hOut[0], hRef[0], err,
           err < 1e-4 * (1 + fabs(hRef[0])) ? "PASS" : "FAIL");

    // ---- 朴素版（5 kernel） ----
    float *dCm, *dRm, *dLogits, *dLoss;
    CHECK_CUDA(cudaMalloc(&dCm, bytes));
    CHECK_CUDA(cudaMalloc(&dRm, bytes));
    CHECK_CUDA(cudaMalloc(&dLogits, bytes));
    CHECK_CUDA(cudaMalloc(&dLoss, bytes));

    cudaEventRecord(t0);
    naive_elementwise<<<numBlocks, BLOCK_SIZE>>>(dCh, dRej, dCm, B);
    naive_elementwise<<<numBlocks, BLOCK_SIZE>>>(dChRef, dRejRef, dRm, B);
    naive_logits<<<numBlocks, BLOCK_SIZE>>>(dCm, dRm, dLogits, beta, B);
    naive_softplus<<<numBlocks, BLOCK_SIZE>>>(dLogits, dLoss, B);
    naive_reduce<<<1, BLOCK_SIZE>>>(dLoss, dOut, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);
    printf("[naive]  time: %.4f ms  speedup: %.2fx\n", ms_naive, ms_naive / ms_fused);

    float bw_gbs = (4.0 * bytes / 1e9) / (ms_fused / 1e3);   // 读 4 个数组
    printf("I/O bandwidth (fused): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dCh)); CHECK_CUDA(cudaFree(dRej));
    CHECK_CUDA(cudaFree(dChRef)); CHECK_CUDA(cudaFree(dRejRef));
    CHECK_CUDA(cudaFree(dOut)); CHECK_CUDA(cudaFree(dCm));
    CHECK_CUDA(cudaFree(dRm)); CHECK_CUDA(cudaFree(dLogits));
    CHECK_CUDA(cudaFree(dLoss));
    free(hCh); free(hRej); free(hChRef); free(hRejRef);
    free(hOut); free(hRef);
    return 0;
}

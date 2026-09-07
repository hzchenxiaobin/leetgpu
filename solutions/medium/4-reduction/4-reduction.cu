// 4-reduction.cu —— Warp shuffle 两阶段归约（两阶段共用同一 kernel），含验证与带宽测量
// 编译命令: nvcc -O3 -arch=sm_120 4-reduction.cu -o reduction
// 运行:     ./reduction 10000000

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define CHECK_CUDA(call)                                                                               \
    do {                                                                                               \
        cudaError_t e = (call);                                                                        \
        if (e != cudaSuccess) {                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));      \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

__inline__ __device__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_kernel(const float* input, float* output, int N) {
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // grid-stride：第一阶段覆盖全数组；第二阶段 gridDim.x=1，stride 退化为 BLOCK_SIZE
    float val = 0.0f;
    for (int i = gid; i < N; i += gridDim.x * BLOCK_SIZE)
        val += input[i];

    val = warp_reduce(val);
    if (lane == 0)
        warp_sums[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        val = warp_reduce(val);
        if (lane == 0)
            output[blockIdx.x] = val;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N=%d (%.1f MB)\n", N, bytes / 1e6);

    float* hInput = (float*)malloc(bytes);
    srand(42);
    double cpu_sum = 0.0;
    for (int i = 0; i < N; i++) {
        hInput[i] = (float)(rand() % 1000) / 100.0f;
        cpu_sum += hInput[i];
    }

    float *dInput, *dPartial, *dOutput;
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&dInput, bytes));
    CHECK_CUDA(cudaMalloc(&dPartial, sizeof(float) * gridSize));
    CHECK_CUDA(cudaMalloc(&dOutput, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dInput, hInput, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    reduce_kernel<<<gridSize, BLOCK_SIZE>>>(dInput, dPartial, N);   // 第一阶段：大 grid
    reduce_kernel<<<1, BLOCK_SIZE>>>(dPartial, dOutput, gridSize);  // 第二阶段：单 block
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    float hResult = 0.0f;
    CHECK_CUDA(cudaMemcpy(&hResult, dOutput, sizeof(float), cudaMemcpyDeviceToHost));

    printf("kernel time: %.3f ms\n", ms);
    printf("read bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));
    printf("GPU sum = %.2f\nCPU sum = %.2f\n", hResult, (float)cpu_sum);

    double rel_err = fabs((double)hResult - cpu_sum) / fabs(cpu_sum);
    int pass = (rel_err < 1e-3) || (fabs((double)hResult - cpu_sum) < 1.0);
    printf("%s (rel_err=%.2e)\n", pass ? "PASS" : "FAIL", rel_err);

    CHECK_CUDA(cudaFree(dInput));
    CHECK_CUDA(cudaFree(dPartial));
    CHECK_CUDA(cudaFree(dOutput));
    free(hInput);
    return 0;
}

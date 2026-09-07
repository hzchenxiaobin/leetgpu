// 17-dot-product.cu —— Dot Product（两级归约：block 内 warp shuffle + 跨 block atomicAdd）
// 编译命令: nvcc -O3 -arch=sm_120 17-dot-product.cu -o dot_product
// 运行:     ./dot_product

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

// warp 内树形归约（__shfl_down_sync）
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// block 内归约：每 thread 算一段乘积和 → warp 归约 → block 归约 → atomicAdd
__global__ void dot_product_kernel(const float* a, const float* b, float* result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    __shared__ float warp_sums[WARP];

    // 每 thread 算自己负责元素的乘积和（grid-stride）
    float sum = 0.0f;
    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }

    // warp 内归约
    sum = warp_reduce(sum);
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    // 第一个 warp 归约 warp_sums
    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (lane == 0)
            atomicAdd(result, sum); // 跨 block 归约
    }
}

int main() {
    int N = 1000000;
    size_t bytes = N * sizeof(float);
    std::vector<float> h_a(N), h_b(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = (rand() % 100) / 100.0f;
        h_b[i] = (rand() % 100) / 100.0f;
    }

    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
    float zero = 0.0f;
    cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK - 1) / BLOCK;
    dot_product_kernel<<<blocks, BLOCK>>>(d_a, d_b, d_result, N);
    cudaDeviceSynchronize();

    float gpu_result;
    cudaMemcpy(&gpu_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 验证
    double cpu_result = 0;
    for (int i = 0; i < N; i++)
        cpu_result += (double)h_a[i] * h_b[i];

    printf("GPU: %.4f, CPU: %.4f, %s\n", gpu_result, (float)cpu_result,
           fabs((double)gpu_result - cpu_result) < 1e-3 * fabs(cpu_result) ? "PASS" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    return 0;
}

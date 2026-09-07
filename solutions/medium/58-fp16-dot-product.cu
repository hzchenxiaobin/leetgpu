// 58-fp16-dot-product.cu —— FP16 Dot Product（half→float 转换 + FP32 两级归约 + 最后转 half）
// 编译命令: nvcc -O3 -arch=sm_120 58-fp16-dot-product.cu -o fp16_dot
// 运行:     ./fp16_dot

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define BLOCK 256
#define WARP 32

// warp 内 FP32 树形归约（__shfl_down_sync）
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// block 内归约：每 thread 算一段乘积和 → warp 归约 → block 归约 → atomicAdd 到 FP32 结果
__global__ void fp16_dot_kernel(const half* A, const half* B, float* result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    __shared__ float warp_sums[WARP];

    // 每 thread 算自己负责元素的乘积和（grid-stride），half→float 转换后 FP32 累加
    float sum = 0.0f;
    for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
        float a = __half2float(A[i]);
        float b = __half2float(B[i]);
        sum += a * b;
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
            atomicAdd(result, sum); // 跨 block 归约（FP32）
    }
}

// 单线程把 FP32 总和转成 half 写入 result[0]
__global__ void convert_fp32_to_half(const float* src, half* dst) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        dst[0] = __float2half(src[0]);
}

int main() {
    int N = 1000000;
    size_t bytes_half = N * sizeof(half);
    std::vector<half> h_a(N), h_b(N);
    srand(42);
    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half((rand() % 100) / 100.0f);
        h_b[i] = __float2half((rand() % 100) / 100.0f);
    }

    half *d_a, *d_b, *d_result;
    float *d_partial;
    cudaMalloc(&d_a, bytes_half);
    cudaMalloc(&d_b, bytes_half);
    cudaMalloc(&d_result, sizeof(half));
    cudaMalloc(&d_partial, sizeof(float));
    cudaMemcpy(d_a, h_a.data(), bytes_half, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes_half, cudaMemcpyHostToDevice);
    float zero = 0.0f;
    cudaMemcpy(d_partial, &zero, sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK - 1) / BLOCK;
    fp16_dot_kernel<<<blocks, BLOCK>>>(d_a, d_b, d_partial, N);
    cudaDeviceSynchronize();

    // 直接读 FP32 结果（N=1M 时点积 ~250k，超过 half 范围 65504，转 half 会 inf）
    float gpu_result;
    cudaMemcpy(&gpu_result, d_partial, sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 验证（FP32 累加）
    double cpu_result = 0;
    for (int i = 0; i < N; i++)
        cpu_result += (double)__half2float(h_a[i]) * __half2float(h_b[i]);

    printf("GPU: %.4f, CPU: %.4f, %s\n", gpu_result, (float)cpu_result,
           fabs((double)gpu_result - cpu_result) < 1e-3 * fabs(cpu_result) ? "PASS" : "FAIL");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    cudaFree(d_partial);
    return 0;
}

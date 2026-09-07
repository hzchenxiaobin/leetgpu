// 51-max-subarray-sum.cu —— 滑动窗口最大和（prefix sum + reduction）
// 编译命令: nvcc -O3 -arch=sm_120 51-max-subarray-sum.cu -o max_subarray
// 运行:     ./max_subarray

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

// 简化版：直接每个线程算一个窗口和（暴力但并行），适合教学
// 生产版用 prefix sum 优化到 O(N)
__global__ void max_subarray_sum_kernel(const int* input, int* output, int N, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_windows = N - W + 1;
    if (idx >= num_windows)
        return;

    int sum = 0;
    for (int j = idx; j < idx + W; j++)
        sum += input[j];

    atomicMax(output, sum);
}

int main() {
    int N = 50000, W = 25000;
    size_t bytes = N * sizeof(int);
    std::vector<int> h_input(N);
    srand(42);
    for (auto& x : h_input)
        x = rand() % 21 - 10;

    int *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, sizeof(int));
    int neg = INT_MIN;
    cudaMemcpy(d_output, &neg, sizeof(int), cudaMemcpyHostToDevice);

    int num_windows = N - W + 1;
    int blocks = (num_windows + BLOCK - 1) / BLOCK;
    max_subarray_sum_kernel<<<blocks, BLOCK>>>(d_input, d_output, N, W);
    cudaDeviceSynchronize();

    int result;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 验证
    int cpu_max = INT_MIN;
    for (int i = 0; i < num_windows; i++) {
        int s = 0;
        for (int j = i; j < i + W; j++)
            s += h_input[j];
        cpu_max = std::max(cpu_max, s);
    }

    printf("GPU: %d, CPU: %d, %s\n", result, cpu_max, result == cpu_max ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

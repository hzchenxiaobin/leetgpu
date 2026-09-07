// 66-rgb-to-grayscale.cu —— RGB to Grayscale with grid-stride + float3 vectorized read
// 编译命令: nvcc -O3 -arch=sm_120 66-rgb-to-grayscale.cu -o rgb2gray
// 运行:     ./rgb2gray

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                          \
    cudaError_t e = (call);                                                                                   \
    if (e != cudaSuccess) {                                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
        exit(EXIT_FAILURE);                                                                                   \
    }                                                                                                         \
} while (0)

#define BLOCK_SIZE 256

// RGB to Grayscale: 每线程处理一个像素
__global__ void rgb2gray_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int num_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // grid-stride loop: 覆盖任意 num_pixels
    for (int p = tid; p < num_pixels; p += stride) {
        // float3 向量化读取：一次 128-bit load 读 R,G,B
        float3 rgb = *reinterpret_cast<const float3*>(&input[p * 3]);
        // 加权求和（编译器融合为 FMA）
        output[p] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

// ---- CPU 参考 ----
void rgb2gray_cpu(const float* input, float* output, int width, int height) {
    int n = width * height;
    for (int p = 0; p < n; p++)
        output[p] = 0.299f * input[p*3] + 0.587f * input[p*3+1] + 0.114f * input[p*3+2];
}

int main() {
    // 题目 example
    int width = 2, height = 2;
    int num_pixels = width * height;
    float hIn[] = {255,0,0, 0,255,0, 0,0,255, 128,128,128};
    float hOut[4], hRef[4];
    printf("RGB to Grayscale: %dx%d\n", width, height);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, num_pixels * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, num_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rgb2gray_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num_pixels);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hOut, dOut, num_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    rgb2gray_cpu(hIn, hRef, width, height);

    printf("output = [%.3f, %.3f, %.3f, %.3f]\n", hOut[0], hOut[1], hOut[2], hOut[3]);
    printf("expect = [76.245, 149.685, 29.070, 128.000]\n");
    int err = 0;
    for (int i = 0; i < num_pixels; i++)
        if (fabsf(hOut[i] - hRef[i]) > 1e-4f) err++;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 ----
    printf("\n--- Perf test (2048x2048) ---\n");
    width = 2048; height = 2048;
    num_pixels = width * height;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, (size_t)num_pixels * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (size_t)num_pixels * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    blocks = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rgb2gray_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num_pixels);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 带宽估算：读 12B/像素 + 写 4B/像素 = 16B/像素
    size_t bytes = (size_t)num_pixels * 16;
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}

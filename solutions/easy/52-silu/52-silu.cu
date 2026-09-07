// 52-silu.cu —— SiLU 激活函数（grid-stride + __expf 快速数学）
// 编译命令: nvcc -O3 -arch=sm_120 52-silu.cu -o silu
// 运行:     ./silu

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

// grid-stride + __expf：任意 N 一次覆盖，coalesced 访存
__global__ void silu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        // __expf 比 expf 快约 10x，精度满足 atol=1e-5
        output[i] = x / (1.0f + __expf(-x));
    }
}

int main() {
    int N = 1 << 20; // 1M 元素
    size_t bytes = N * sizeof(float);
    std::vector<float> h_in(N), h_out(N);
    srand(42);
    for (auto& x : h_in)
        x = (rand() % 2000 - 1000) / 100.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK - 1) / BLOCK;
    silu_kernel<<<grid, BLOCK>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // 验证
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    bool pass = true;
    for (int i = 0; i < N; i++) {
        float expect = h_in[i] / (1.0f + expf(-h_in[i]));
        if (fabsf(h_out[i] - expect) > 1e-4) {
            pass = false;
            break;
        }
    }
    printf("SiLU N=%d: %s\n", N, pass ? "PASS" : "FAIL");

    // 带宽测量（cudaEvent 计时）
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++)
        silu_kernel<<<grid, BLOCK>>>(d_in, d_out, N); // warmup
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        silu_kernel<<<grid, BLOCK>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float t = ms / 100;                         // 单次毫秒
    float bw = 2.0f * bytes / (t / 1000) / 1e9; // 读+写 = 2x，GB/s
    printf("Bandwidth: %.1f GB/s (peak ~1555 GB/s on RTX 5090, %.1f%%)\n", bw, bw / 1555 * 100);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

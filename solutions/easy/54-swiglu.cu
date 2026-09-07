// 54-swiglu.cu —— SwiGLU 融合 Kernel（grid-stride + __expf + kernel fusion）
// 编译命令: nvcc -O3 -arch=sm_120 54-swiglu.cu -o swiglu
// 运行:     ./swiglu

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

// 融合 kernel：SiLU(x₁) * x₂ 在一个 kernel 内完成
__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < halfN; i += stride) {
        float x1 = input[i];                    // 前半
        float x2 = input[i + halfN];            // 后半
        float silu = x1 / (1.0f + __expf(-x1)); // SiLU（register 内）
        output[i] = silu * x2;                  // 融合乘法（register 内）
    }
}

int main() {
    int N = 1 << 20; // 1M
    int halfN = N / 2;
    std::vector<float> h_in(N), h_out(halfN);
    srand(42);
    for (auto& x : h_in)
        x = (rand() % 200 - 100) / 50.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, halfN * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int grid = (halfN + BLOCK - 1) / BLOCK;
    swiglu_kernel<<<grid, BLOCK>>>(d_in, d_out, halfN);
    cudaDeviceSynchronize();

    // 验证
    cudaMemcpy(h_out.data(), d_out, halfN * sizeof(float), cudaMemcpyDeviceToHost);
    bool pass = true;
    for (int i = 0; i < halfN; i++) {
        float x1 = h_in[i], x2 = h_in[i + halfN];
        float expect = (x1 / (1.0f + expf(-x1))) * x2;
        if (fabsf(h_out[i] - expect) > 1e-3) {
            pass = false;
            break;
        }
    }
    printf("SwiGLU N=%d: %s\n", N, pass ? "PASS" : "FAIL");

    // 带宽测量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++)
        swiglu_kernel<<<grid, BLOCK>>>(d_in, d_out, halfN);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        swiglu_kernel<<<grid, BLOCK>>>(d_in, d_out, halfN);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float t = ms / 100;
    // 读 2*halfN + 写 halfN = 3*halfN floats
    float bytes = 3.0f * halfN * sizeof(float);
    float bw = bytes / (t / 1000) / 1e9;
    printf("Bandwidth: %.1f GB/s (%.1f%% of 1555 GB/s)\n", bw, bw / 1555 * 100);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

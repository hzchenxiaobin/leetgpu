// 61-rope-embedding.cu —— Rotary Positional Embedding（grid-stride + 2D 映射）
// 编译命令: nvcc -O3 -arch=sm_120 61-rope-embedding.cu -o rope
// 运行:     ./rope

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_X 32
#define BLOCK_Y 8

// grid-stride + 2D 映射：避免除法，coalesced 访存
__global__ void rope_kernel(const float* Q, const float* cos, const float* sin, float* output, int M, int D) {
    int half = D / 2;
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < M; row += gridDim.y * blockDim.y) {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < D; col += gridDim.x * blockDim.x) {
            int idx = row * D + col;
            float q = Q[idx];
            // rotate_half: 前半取后半取反，后半取前半
            float rotated = (col < half) ? -Q[idx + half] : Q[idx - half];
            output[idx] = q * cos[idx] + rotated * sin[idx];
        }
    }
}

int main() {
    int M = 1024, D = 64;
    size_t bytes = (size_t)M * D * sizeof(float);
    std::vector<float> h_Q(M * D), h_cos(M * D), h_sin(M * D), h_out(M * D);
    srand(42);
    for (int i = 0; i < M * D; i++) {
        h_Q[i] = (rand() % 200 - 100) / 10.0f;
        h_cos[i] = cosf(i * 0.01f);
        h_sin[i] = sinf(i * 0.01f);
    }

    float *d_Q, *d_cos, *d_sin, *d_out;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_cos, bytes);
    cudaMalloc(&d_sin, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((D + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
    rope_kernel<<<grid, block>>>(d_Q, d_cos, d_sin, d_out, M, D);
    cudaDeviceSynchronize();

    // 验证
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    int half = D / 2;
    bool pass = true;
    for (int m = 0; m < M && pass; m++)
        for (int d = 0; d < D && pass; d++) {
            int idx = m * D + d;
            float rot = (d < half) ? -h_Q[idx + half] : h_Q[idx - half];
            float expect = h_Q[idx] * h_cos[idx] + rot * h_sin[idx];
            if (fabsf(h_out[idx] - expect) > 1e-4) {
                pass = false;
            }
        }
    printf("RoPE M=%d D=%d: %s\n", M, D, pass ? "PASS" : "FAIL");

    // 带宽测量
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++)
        rope_kernel<<<grid, block>>>(d_Q, d_cos, d_sin, d_out, M, D);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        rope_kernel<<<grid, block>>>(d_Q, d_cos, d_sin, d_out, M, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float t = ms / 100;
    float bw = 4.0f * bytes / (t / 1000) / 1e9; // 读 Q+cos+sin=3D + 写 out=D = 4D
    printf("Bandwidth: %.1f GB/s (%.1f%% of 1555 GB/s peak)\n", bw, bw / 1555 * 100);

    cudaFree(d_Q);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_out);
    return 0;
}

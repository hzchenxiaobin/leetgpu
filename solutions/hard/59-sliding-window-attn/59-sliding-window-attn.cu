// 59-sliding-window-attn.cu —— Sliding Window Self-Attention
// 编译命令: nvcc -O3 -arch=sm_120 59-sliding-window-attn.cu -o swa
// 运行:     ./swa

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void sliding_window_attention_kernel(const float* Q, const float* K, const float* V, float* O, int N, int d,
                                                int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    float scale = 1.0f / sqrtf((float)d);
    int win_start = max(0, i - W + 1);

    // 把当前 Q[i] 读到 register
    extern __shared__ float s_q[]; // 可选：block 内共享当前 tile 的 Q
    // 简化版：直接用 register
    float q[64]; // 假设 d <= 64
    for (int k = 0; k < d; k++)
        q[k] = Q[i * d + k];

    // 第一遍：求窗口内 max score
    float m = -1e30f;
    for (int j = win_start; j <= i; j++) {
        float s = 0.0f;
        for (int k = 0; k < d; k++)
            s += q[k] * K[j * d + k];
        m = fmaxf(m, s * scale);
    }

    // 第二遍：求 softmax 分母
    float l = 0.0f;
    for (int j = win_start; j <= i; j++) {
        float s = 0.0f;
        for (int k = 0; k < d; k++)
            s += q[k] * K[j * d + k];
        l += expf(s * scale - m);
    }

    // 第三遍：加权 V 得到输出
    for (int k = 0; k < d; k++)
        O[i * d + k] = 0.0f;
    for (int j = win_start; j <= i; j++) {
        float s = 0.0f;
        for (int k = 0; k < d; k++)
            s += q[k] * K[j * d + k];
        float p = expf(s * scale - m) / l;
        for (int k = 0; k < d; k++)
            O[i * d + k] += p * V[j * d + k];
    }
}

int main() {
    int N = 4096, d = 64, W = 256;
    size_t bytes = (size_t)N * d * sizeof(float);
    std::vector<float> h_Q(N * d), h_K(N * d), h_V(N * d), h_O(N * d), h_O_CPU(N * d);
    srand(42);
    for (auto& x : h_Q)
        x = (rand() % 200 - 100) / 50.0f;
    for (auto& x : h_K)
        x = (rand() % 200 - 100) / 50.0f;
    for (auto& x : h_V)
        x = (rand() % 200 - 100) / 50.0f;

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK - 1) / BLOCK;
    sliding_window_attention_kernel<<<grid, BLOCK>>>(d_Q, d_K, d_V, d_O, N, d, W);
    cudaMemcpy(h_O.data(), d_O, bytes, cudaMemcpyDeviceToHost);

    // CPU 验证
    float scale = 1.0f / sqrtf((float)d);
    for (int i = 0; i < N; i++) {
        int win_start = std::max(0, i - W + 1);
        float m = -1e30f;
        for (int j = win_start; j <= i; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += h_Q[i * d + k] * h_K[j * d + k];
            m = fmaxf(m, s * scale);
        }
        float l = 0.0f;
        for (int j = win_start; j <= i; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += h_Q[i * d + k] * h_K[j * d + k];
            l += expf(s * scale - m);
        }
        for (int k = 0; k < d; k++)
            h_O_CPU[i * d + k] = 0.0f;
        for (int j = win_start; j <= i; j++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++)
                s += h_Q[i * d + k] * h_K[j * d + k];
            float p = expf(s * scale - m) / l;
            for (int k = 0; k < d; k++)
                h_O_CPU[i * d + k] += p * h_V[j * d + k];
        }
    }

    bool pass = true;
    for (int i = 0; i < N * d; i++)
        if (fabsf(h_O[i] - h_O_CPU[i]) > 1e-3f) {
            pass = false;
            break;
        }
    printf("Sliding Window Attention N=%d d=%d W=%d: %s\n", N, d, W, pass ? "PASS" : "FAIL");

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    return 0;
}

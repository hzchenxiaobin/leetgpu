// 85-lora-linear.cu —— LoRA Linear forward
// 编译命令: nvcc -O3 -arch=sm_120 85-lora-linear.cu -o lora
// 运行:     ./lora

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

__global__ void lora_linear_kernel(const float* W, const float* A, const float* B, const float* x, float* y, int m,
                                   int n, int r) {
    extern __shared__ float sbuf[]; // x[?] + h[r]
    float* s_x = sbuf;
    float* s_h = sbuf + n;

    int tid = threadIdx.x;
    int total = m;

    // 1) 协作加载 x 到 shared memory
    for (int k = tid; k < n; k += blockDim.x)
        s_x[k] = x[k];
    __syncthreads();

    // 2) 并行计算 h[j] = (A[j,:] · s_x) / r
    for (int j = tid; j < r; j += blockDim.x) {
        float s = 0.0f;
        for (int k = 0; k < n; k++)
            s += A[j * n + k] * s_x[k];
        s_h[j] = s / r;
    }
    __syncthreads();

    // 3) 每个 thread 算一个输出元素 y[i]
    for (int i = blockIdx.x * blockDim.x + tid; i < m; i += blockDim.x * gridDim.x) {
        float y0 = 0.0f;
        for (int k = 0; k < n; k++)
            y0 += W[i * n + k] * s_x[k];

        float delta = 0.0f;
        for (int j = 0; j < r; j++)
            delta += B[i * r + j] * s_h[j];

        y[i] = y0 + delta;
    }
}

int main() {
    int m = 1024, n = 512, r = 16;
    size_t Wbytes = (size_t)m * n * sizeof(float);
    size_t Abytes = (size_t)r * n * sizeof(float);
    size_t Bbytes = (size_t)m * r * sizeof(float);
    size_t xbytes = (size_t)n * sizeof(float);
    size_t ybytes = (size_t)m * sizeof(float);

    std::vector<float> h_W(m * n), h_A(r * n), h_B(m * r), h_x(n), h_y(m), h_y_cpu(m);
    srand(42);
    for (auto& v : h_W)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_A)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_B)
        v = (rand() % 200 - 100) / 100.0f;
    for (auto& v : h_x)
        v = (rand() % 200 - 100) / 100.0f;

    float *d_W, *d_A, *d_B, *d_x, *d_y;
    cudaMalloc(&d_W, Wbytes);
    cudaMalloc(&d_A, Abytes);
    cudaMalloc(&d_B, Bbytes);
    cudaMalloc(&d_x, xbytes);
    cudaMalloc(&d_y, ybytes);
    cudaMemcpy(d_W, h_W.data(), Wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A.data(), Abytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), Bbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), xbytes, cudaMemcpyHostToDevice);

    int smem = (n + r) * sizeof(float);
    int grid = std::min(64, (m + BLOCK - 1) / BLOCK);
    lora_linear_kernel<<<grid, BLOCK, smem>>>(d_W, d_A, d_B, d_x, d_y, m, n, r);
    cudaMemcpy(h_y.data(), d_y, ybytes, cudaMemcpyDeviceToHost);

    // CPU reference
    for (int i = 0; i < m; i++) {
        float y0 = 0.0f;
        for (int k = 0; k < n; k++)
            y0 += h_W[i * n + k] * h_x[k];
        std::vector<float> h(r);
        for (int j = 0; j < r; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++)
                s += h_A[j * n + k] * h_x[k];
            h[j] = s / r;
        }
        float delta = 0.0f;
        for (int j = 0; j < r; j++)
            delta += h_B[i * r + j] * h[j];
        h_y_cpu[i] = y0 + delta;
    }

    bool pass = true;
    for (int i = 0; i < m; i++)
        if (fabsf(h_y[i] - h_y_cpu[i]) > 1e-3f) {
            pass = false;
            break;
        }
    printf("LoRA m=%d n=%d r=%d: %s\n", m, n, r, pass ? "PASS" : "FAIL");

    cudaFree(d_W);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

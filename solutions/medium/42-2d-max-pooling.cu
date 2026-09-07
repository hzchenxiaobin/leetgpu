// 42-2d-max-pooling.cu —— 2D Max Pooling（每 thread 一个输出，W_out 维内层合并访存）
// 编译命令: nvcc -O3 -arch=sm_120 42-2d-max-pooling.cu -o max_pool2d
// 运行:     ./max_pool2d

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

// 2D max pooling kernel：每 thread 算一个 output[n,c,oy,ox]
__global__ void max_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N, int C, int H, int W,
                                  int kernel_size, int stride, int padding,
                                  int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // 线性索引 → (n, c, oy, ox)，W_out 在最内层（保证 warp 内 ox 连续 → 合并访存）
    int ox = idx % W_out;
    int oy = (idx / W_out) % H_out;
    int c  = (idx / (W_out * H_out)) % C;
    int n  = idx / (C * H_out * W_out);

    // 窗口左上角对应的输入坐标
    int base_h = oy * stride - padding;
    int base_w = ox * stride - padding;

    // 累加器初值 -∞，保证任何有效输入都能成为首个 max
    float m = -INFINITY;

    int nc_offset = (n * C + c) * H * W;   // 固定 n、c 后的 input 行偏移
    #pragma unroll
    for (int ky = 0; ky < kernel_size; ++ky) {
        int ih = base_h + ky;
        if (ih < 0 || ih >= H) continue;
        int row_offset = nc_offset + ih * W;
        #pragma unroll
        for (int kx = 0; kx < kernel_size; ++kx) {
            int iw = base_w + kx;
            if (iw < 0 || iw >= W) continue;
            float v = input[row_offset + iw];
            if (v > m) m = v;
        }
    }

    output[idx] = m;
}

// ---- CPU 参考 ----
void max_pool2d_cpu(const float* input, float* output,
                    int N, int C, int H, int W,
                    int kernel_size, int stride, int padding) {
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < C; ++c)
            for (int oy = 0; oy < H_out; ++oy)
                for (int ox = 0; ox < W_out; ++ox) {
                    float m = -INFINITY;
                    for (int ky = 0; ky < kernel_size; ++ky)
                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int ih = oy * stride - padding + ky;
                            int iw = ox * stride - padding + kx;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float v = input[((n * C + c) * H + ih) * W + iw];
                                if (v > m) m = v;
                            }
                        }
                    output[((n * C + c) * H_out + oy) * W_out + ox] = m;
                }
}

int main() {
    int N = 4, C = 64, H = 256, W = 256;
    int kernel_size = 3, stride = 2, padding = 1;
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    size_t in_bytes = (size_t)N * C * H * W * sizeof(float);
    size_t out_bytes = (size_t)N * C * H_out * W_out * sizeof(float);
    printf("input: %dx%dx%dx%d  K=%d stride=%d pad=%d  output: %dx%dx%dx%d\n",
           N, C, H, W, kernel_size, stride, padding, N, C, H_out, W_out);

    std::vector<float> h_in(N * C * H * W), h_out(N * C * H_out * W_out), h_ref(N * C * H_out * W_out);
    srand(42);
    for (auto& v : h_in) v = (float)(rand() % 1000) / 10.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, in_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMemcpy(d_in, h_in.data(), in_bytes, cudaMemcpyHostToDevice);

    int total = N * C * H_out * W_out;
    int blocks = (total + BLOCK - 1) / BLOCK;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    max_pool2d_kernel<<<blocks, BLOCK>>>(d_in, d_out, N, C, H, W,
                                         kernel_size, stride, padding, H_out, W_out);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
    max_pool2d_cpu(h_in.data(), h_ref.data(), N, C, H, W, kernel_size, stride, padding);

    int err = 0;
    for (int i = 0; i < total && err < 5; ++i) {
        if (fabsf(h_out[i] - h_ref[i]) > 1e-5f) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, h_out[i], h_ref[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // 带宽估算：读 input（含 L2 复用）+ 写 output
    size_t rw_bytes = ((size_t)N * C * H * W + (size_t)N * C * H_out * W_out) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth (approx): %.1f GB/s\n", bw_gbs);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

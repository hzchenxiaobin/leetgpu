// 90-causal-depthwise-conv1d.cu —— 因果深度卷积（每 thread 一个输出，D 维连续合并访存）
// 编译命令: nvcc -O3 -arch=sm_120 90-causal-depthwise-conv1d.cu -o causal_dwconv1d
// 运行:     ./causal_dwconv1d

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256

// 因果深度卷积 kernel：每 thread 算一个 output[b,l,d]
__global__ void causal_depthwise_conv1d_kernel(const float* __restrict__ x,
                                               const float* __restrict__ weight,
                                               const float* __restrict__ bias,
                                               float* __restrict__ output,
                                               int B, int L, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * L * D;
    if (idx >= total) return;

    // 线性索引 → (b, l, d)，D 在最内层（保证 warp 内 d 连续 → 合并访存）
    int d = idx % D;
    int l = (idx / D) % L;
    int b = idx / (L * D);

    // bias 作为累加器初值
    float acc = bias[d];

    // 因果窗口：li = l - (K-1) + k，li < 0 时跳过（等价 0 填充）
    int base = l - (K - 1);
    int row_offset = b * L * D + d;   // x 的行偏移（固定 b、d，沿 l 变化）
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        int li = base + k;
        if (li >= 0)
            acc += weight[d * K + k] * x[row_offset + li * D];
    }

    output[idx] = acc;
}

// ---- CPU 参考 ----
void causal_depthwise_conv1d_cpu(const float* x, const float* weight, const float* bias,
                                 float* output, int B, int L, int D, int K) {
    for (int b = 0; b < B; ++b)
        for (int l = 0; l < L; ++l)
            for (int d = 0; d < D; ++d) {
                float acc = bias[d];
                for (int k = 0; k < K; ++k) {
                    int li = l - (K - 1) + k;
                    if (li >= 0)
                        acc += weight[d * K + k] * x[b * L * D + li * D + d];
                }
                output[b * L * D + l * D + d] = acc;
            }
}

int main() {
    int B = 8, L = 2048, D = 4096, K = 4;
    size_t x_bytes = (size_t)B * L * D * sizeof(float);
    size_t w_bytes = (size_t)D * K * sizeof(float);
    size_t b_bytes = D * sizeof(float);
    size_t o_bytes = x_bytes;

    std::vector<float> h_x(B * L * D), h_w(D * K), h_bias(D), h_out(o_bytes / sizeof(float)), h_ref(o_bytes / sizeof(float));
    srand(42);
    for (auto& v : h_x) v = (rand() % 100) / 100.0f;
    for (auto& v : h_w) v = (rand() % 100) / 100.0f;
    for (auto& v : h_bias) v = (rand() % 100) / 100.0f;

    float *d_x, *d_w, *d_bias, *d_out;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_bias, b_bytes);
    cudaMalloc(&d_out, o_bytes);
    cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), b_bytes, cudaMemcpyHostToDevice);

    int total = B * L * D;
    int blocks = (total + BLOCK - 1) / BLOCK;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    causal_depthwise_conv1d_kernel<<<blocks, BLOCK>>>(d_x, d_w, d_bias, d_out, B, L, D, K);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(h_out.data(), d_out, o_bytes, cudaMemcpyDeviceToHost);
    causal_depthwise_conv1d_cpu(h_x.data(), h_w.data(), h_bias.data(), h_ref.data(), B, L, D, K);

    int err = 0;
    for (int i = 0; i < total && err < 5; ++i) {
        if (fabsf(h_out[i] - h_ref[i]) > 1e-3f) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, h_out[i], h_ref[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // 带宽估算：读 x（~每个输入被 K 个输出读，但有 L2 cache）+ 写 output
    size_t rw_bytes = ((size_t)B * L * D * K + (size_t)B * L * D) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth (approx): %.1f GB/s\n", bw_gbs);

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_bias);
    cudaFree(d_out);
    return 0;
}

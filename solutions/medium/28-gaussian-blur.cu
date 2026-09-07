// 28-gaussian-blur.cu —— shared memory halo + __constant__ 权重 + 零填充 same 卷积
// 编译命令: nvcc -O3 -arch=sm_120 28-gaussian-blur.cu -o gaussian_blur
// 运行:     ./gaussian_blur 512 512 7

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                          \
    cudaError_t e = (call);                                                                                   \
    if (e != cudaSuccess) {                                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
        exit(EXIT_FAILURE);                                                                                   \
    }                                                                                                         \
} while (0)

#define OT 16       // 输出 tile 边长
#define MAX_KH 64   // 卷积核最大高度（常量内存预留）
#define MAX_KW 64   // 卷积核最大宽度

// 卷积核权重放常量内存：全 thread 读同一地址 → 硬件广播
__constant__ float c_kernel[MAX_KH * MAX_KW];

// shared memory halo + 常数权重 + 零填充 的 same-padding 2D 卷积
__global__ void gaussian_blur_halo(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int H, int W, int KH, int KW) {
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;
    const int tileH = OT + KH - 1;   // 输入 tile 高（含上下 halo）
    const int tileW = OT + KW - 1;   // 输入 tile 宽（含左右 halo）
    extern __shared__ float smem[];

    const int ox0 = blockIdx.x * OT;  // 本 block 输出 tile 左上角 col
    const int oy0 = blockIdx.y * OT;  // 本 block 输出 tile 左上角 row
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * OT + tx;
    const int nTH = OT * OT;

    // ---- ① 协作加载 input tile（含 halo）到 shared memory，越界填 0 ----
    // tile 起点偏移 (-pad_h, -pad_w)：输出 tile (oy0,ox0) 的输入区域从 (oy0-pad_h, ox0-pad_w) 开始
    for (int idx = tid; idx < tileH * tileW; idx += nTH) {
        int sy = idx / tileW;
        int sx = idx % tileW;
        int gx = ox0 - pad_w + sx;   // smem 列 → 全局列（偏移 -pad_w）
        int gy = oy0 - pad_h + sy;   // smem 行 → 全局行（偏移 -pad_h）
        // 越界填 0（零填充语义）
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            smem[sy * tileW + sx] = input[gy * W + gx];
        else
            smem[sy * tileW + sx] = 0.0f;
    }
    __syncthreads();

    // ---- ② 每个线程算一个输出像素：KH×KW 窗口全从 shared 读 ----
    const int ox = ox0 + tx;
    const int oy = oy0 + ty;
    if (ox < W && oy < H) {
        float acc = 0.0f;
        for (int ky = 0; ky < KH; ++ky) {
            const float* srow = &smem[(ty + ky) * tileW + tx];
            const float* krow = &c_kernel[ky * KW];
            for (int kx = 0; kx < KW; ++kx) {
                acc += srow[kx] * krow[kx];
            }
        }
        output[oy * W + ox] = acc;
    }
}

// ---- CPU 参考（same-padding 零填充卷积）----
void gaussian_blur_cpu(const float* input, const float* kernel, float* output,
                       int H, int W, int KH, int KW) {
    int pad_h = KH / 2, pad_w = KW / 2;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            float acc = 0.0f;
            for (int m = 0; m < KH; ++m)
                for (int n = 0; n < KW; ++n) {
                    int gy = i - pad_h + m;
                    int gx = j - pad_w + n;
                    if (gy >= 0 && gy < H && gx >= 0 && gx < W)
                        acc += input[gy * W + gx] * kernel[m * KW + n];
                }
            output[i * W + j] = acc;
        }
}

int main(int argc, char** argv) {
    int H  = (argc > 1) ? atoi(argv[1]) : 512;
    int W  = (argc > 2) ? atoi(argv[2]) : 512;
    int KH = (argc > 3) ? atoi(argv[3]) : 7;
    int KW = KH;  // 本题测试用例均为方核，但也支持矩形核
    if (KH % 2 == 0 || KW % 2 == 0 || KH > MAX_KH || KW > MAX_KW) {
        fprintf(stderr, "KH, KW must be odd and <= %d\n", MAX_KH);
        return 1;
    }
    size_t in_bytes  = (size_t)H * W * sizeof(float);
    size_t out_bytes = (size_t)H * W * sizeof(float);
    size_t ker_bytes = (size_t)KH * KW * sizeof(float);
    printf("input: %dx%d  kernel: %dx%d  output: %dx%d (same)\n", H, W, KH, KW, H, W);

    // ---- host 分配与初始化 ----
    float* hIn  = (float*)malloc(in_bytes);
    float* hKer = (float*)malloc(ker_bytes);
    float* hOut = (float*)malloc(out_bytes);
    float* hRef = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < H * W; ++i)
        hIn[i] = (float)(rand() % 1000) / 100.0f;
    // 生成可分离 Gaussian 核用于测试
    int pad_h = KH / 2;
    float sigma = (float)pad_h / 2.0f + 0.5f;
    float sum = 0.0f;
    for (int m = 0; m < KH; ++m)
        for (int n = 0; n < KW; ++n) {
            int dy = m - pad_h, dx = n - pad_h;
            hKer[m * KW + n] = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            sum += hKer[m * KW + n];
        }
    for (int i = 0; i < KH * KW; ++i)
        hKer[i] /= sum;  // 归一化

    // ---- device 分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));

    // ---- 启动配置 ----
    dim3 threads(OT, OT);
    dim3 blocks((W + OT - 1) / OT, (H + OT - 1) / OT);
    int tileH = OT + KH - 1;
    int tileW = OT + KW - 1;
    size_t smem_bytes = (size_t)tileH * tileW * sizeof(float);
    printf("launch: blocks=(%d,%d)  threads=(%d,%d)  smem=%.1f KB/block\n",
           blocks.x, blocks.y, threads.x, threads.y, smem_bytes / 1024.0);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gaussian_blur_halo<<<blocks, threads, smem_bytes>>>(dIn, dOut, H, W, KH, KW);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    gaussian_blur_cpu(hIn, hKer, hRef, H, W, KH, KW);
    int err = 0;
    for (int i = 0; i < H * W && err < 5; ++i) {
        if (fabsf(hOut[i] - hRef[i]) > 1e-4f) {
            ++err;
            printf("MISMATCH @(%d,%d): got %f, expect %f\n", i / W, i % W, hOut[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽估算 ----
    size_t rw_bytes = ((size_t)H * W + (size_t)H * W) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hKer); free(hOut); free(hRef);
    return 0;
}

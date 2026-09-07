// 10-2d-convolution.cu —— shared memory halo + __constant__ 权重实现 2D valid 卷积
// 编译命令: nvcc -O3 -arch=sm_120 10-2d-convolution.cu -o conv2d
// 运行:     ./conv2d 4096 4096 3

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                               \
cudaError_t e = (call);                                                                                        \
if (e != cudaSuccess) {                                                                                        \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
    exit(EXIT_FAILURE);                                                                                        \
}                                                                                                              \
} while (0)

#define OT 16    // 输出 tile 边长
#define MAX_K 16 // 卷积核最大边长（常量内存预留）

// 卷积核权重放常量内存：全 thread 读同一地址 → 硬件广播，1 cycle
__constant__ float c_kernel[MAX_K * MAX_K];

// shared memory halo + 常数权重 的 2D valid 卷积
__global__ void conv2d_shared_halo(const float* __restrict__ input, float* __restrict__ output, int H, int W, int K) {
    const int P = K / 2;       // 卷积半径
    const int IT = OT + K - 1; // input tile 边长（含 halo）
    // 静态 shared：按最大 K 预留，实际只用 [0..IT-1][0..IT-1]
    __shared__ float smem[OT + MAX_K - 1][OT + MAX_K - 1];

    const int ox0 = blockIdx.x * OT; // 本 block 输出 tile 左上角 col
    const int oy0 = blockIdx.y * OT; // 本 block 输出 tile 左上角 row
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * OT + tx;
    const int nTH = OT * OT;

    // ---- ① 协作加载 input tile（含 halo）到 shared memory ----
    // input tile 左上角 = 输出 tile 左上角 (oy0, ox0)，向右下扩展 K-1 圈 halo
    // 越界索引 clamp 到合法范围（replicate border）；这些值仅被过覆盖线程读取，不影响有效输出
    for (int idx = tid; idx < IT * IT; idx += nTH) {
        int sy = idx / IT;
        int sx = idx % IT;
        int gx = ox0 + sx;
        int gy = oy0 + sy;
        gx = min(max(gx, 0), W - 1);
        gy = min(max(gy, 0), H - 1);
        smem[sy][sx] = input[gy * W + gx];
    }
    __syncthreads();

    // ---- ② 每个线程算一个输出像素：K×K 窗口全从 shared 读 ----
    const int outH = H - 2 * P;
    const int outW = W - 2 * P;
    const int ox = ox0 + tx;
    const int oy = oy0 + ty;
    if (ox < outW && oy < outH) {
        float acc = 0.0f;
        #pragma unroll
        for (int ky = 0; ky < K; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                // 窗口左上角在 smem 的 (ty, tx)，覆盖 smem[ty..ty+K-1][tx..tx+K-1]
                acc += smem[ty + ky][tx + kx] * c_kernel[ky * K + kx];
            }
        }
        output[oy * outW + ox] = acc;
    }
}

// ---- CPU 参考（valid 卷积）----
void conv2d_cpu(const float* input, const float* kernel, float* output, int H, int W, int K) {
    int P = K / 2, outH = H - 2 * P, outW = W - 2 * P;
    for (int oy = 0; oy < outH; ++oy)
        for (int ox = 0; ox < outW; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky)
                for (int kx = 0; kx < K; ++kx)
                    acc += input[(oy + ky) * W + (ox + kx)] * kernel[ky * K + kx];
            output[oy * outW + ox] = acc;
        }
}

int main(int argc, char** argv) {
    int H = (argc > 1) ? atoi(argv[1]) : 4096;
    int W = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 3;
    if (K % 2 == 0 || K > MAX_K) {
        fprintf(stderr, "K must be odd and <= %d\n", MAX_K);
        return 1;
    }
    int P = K / 2;
    int outH = H - 2 * P, outW = W - 2 * P;
    size_t in_bytes = (size_t)H * W * sizeof(float);
    size_t out_bytes = (size_t)outH * outW * sizeof(float);
    size_t ker_bytes = (size_t)K * K * sizeof(float);
    printf("input: %dx%d  kernel: %dx%d  output: %dx%d\n", H, W, K, K, outH, outW);

    // ---- host 分配与初始化 ----
    float* hIn = (float*)malloc(in_bytes);
    float* hKer = (float*)malloc(ker_bytes);
    float* hOut = (float*)malloc(out_bytes);
    float* hRef = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < H * W; ++i)
        hIn[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < K * K; ++i)
        hKer[i] = (float)(rand() % 1000) / 100.0f;

    // ---- device 分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));

    // ---- 启动配置 ----
    dim3 threads(OT, OT);
    dim3 blocks((outW + OT - 1) / OT, (outH + OT - 1) / OT);
    printf("launch: blocks=(%d,%d)  threads=(%d,%d)\n", blocks.x, blocks.y, threads.x, threads.y);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    conv2d_shared_halo<<<blocks, threads>>>(dIn, dOut, H, W, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    conv2d_cpu(hIn, hKer, hRef, H, W, K);
    int err = 0;
    for (int i = 0; i < outH * outW && err < 5; ++i) {
        if (fabsf(hOut[i] - hRef[i]) > 1e-3f) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽估算：读 input(含 halo ~1.27×, K=3) + 写 output ----
    size_t rw_bytes = ((size_t)H * W + (size_t)outH * outW) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hKer);
    free(hOut);
    free(hRef);
    return 0;
}

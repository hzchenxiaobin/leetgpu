// 11-3d-convolution.cu —— 3D shared memory halo + __constant__ 权重实现 valid 3D 卷积
// 编译命令: nvcc -O3 -arch=sm_120 11-3d-convolution.cu -o conv3d
// 运行:     ./conv3d 256 128 128 5

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

#define OT 8         // 输出 tile 边长（3D: OT³=512 threads）
#define MAX_KD 8     // 卷积核最大深度（常量内存预留）
#define MAX_KH 8
#define MAX_KW 8

// 卷积核权重放常量内存：全 thread 读同一地址 → 硬件广播
__constant__ float c_kernel[MAX_KD * MAX_KH * MAX_KW];

// 3D shared memory halo + 常数权重 的 valid 3D 卷积
__global__ void conv3d_shared_halo(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int D, int H, int W,
                                    int KD, int KH, int KW) {
    const int outD = D - KD + 1;
    const int outH = H - KH + 1;
    const int outW = W - KW + 1;

    const int tileD = OT + KD - 1;   // 输入 tile 深度（含 halo）
    const int tileH = OT + KH - 1;   // 输入 tile 高度
    const int tileW = OT + KW - 1;   // 输入 tile 宽度
    const int tileVol = tileD * tileH * tileW;

    extern __shared__ float smem[];

    const int ox0 = blockIdx.x * OT;  // 本 block 输出 tile 左上角 col
    const int oy0 = blockIdx.y * OT;  // row
    const int oz0 = blockIdx.z * OT;  // depth
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tz  = threadIdx.z;
    const int tid = tz * OT * OT + ty * OT + tx;
    const int nTH = OT * OT * OT;

    // ---- ① 协作加载 input tile（含 halo）到 shared memory ----
    // valid 卷积：tile 起点就是输出起点，向右下扩展 K-1 圈 halo
    for (int idx = tid; idx < tileVol; idx += nTH) {
        int sz = idx / (tileH * tileW);
        int sy = (idx / tileW) % tileH;
        int sx = idx % tileW;
        int gx = ox0 + sx;
        int gy = oy0 + sy;
        int gz = oz0 + sz;
        // valid 卷积：越界不读（输出不会引用，但 tile 区域需覆盖到合法输入范围）
        if (gx >= 0 && gx < W && gy >= 0 && gy < H && gz >= 0 && gz < D)
            smem[sz * tileH * tileW + sy * tileW + sx] = input[gz * H * W + gy * W + gx];
        else
            smem[sz * tileH * tileW + sy * tileW + sx] = 0.0f;
    }
    __syncthreads();

    // ---- ② 每个线程算一个输出体素：K³ 窗口全从 shared 读 ----
    const int ox = ox0 + tx;
    const int oy = oy0 + ty;
    const int oz = oz0 + tz;
    if (ox < outW && oy < outH && oz < outD) {
        float acc = 0.0f;
        #pragma unroll
        for (int kd = 0; kd < MAX_KD; kd++) {
            if (kd < KD) {
                #pragma unroll
                for (int kh = 0; kh < MAX_KH; kh++) {
                    if (kh < KH) {
                        #pragma unroll
                        for (int kw = 0; kw < MAX_KW; kw++) {
                            if (kw < KW) {
                                acc += smem[(tz + kd) * tileH * tileW + (ty + kh) * tileW + (tx + kw)]
                                     * c_kernel[kd * KH * KW + kh * KW + kw];
                            }
                        }
                    }
                }
            }
        }
        output[oz * outH * outW + oy * outW + ox] = acc;
    }
}

// ---- CPU 参考 ----
void conv3d_cpu(const float* input, const float* kernel, float* output,
                int D, int H, int W, int KD, int KH, int KW) {
    int outD = D - KD + 1, outH = H - KH + 1, outW = W - KW + 1;
    for (int oi = 0; oi < outD; oi++)
        for (int oj = 0; oj < outH; oj++)
            for (int ok = 0; ok < outW; ok++) {
                float acc = 0.0f;
                for (int kd = 0; kd < KD; kd++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            acc += input[(oi+kd)*H*W + (oj+kh)*W + (ok+kw)]
                                 * kernel[kd*KH*KW + kh*KW + kw];
                output[oi*outH*outW + oj*outW + ok] = acc;
            }
}

int main(int argc, char** argv) {
    int D  = (argc > 1) ? atoi(argv[1]) : 256;
    int H  = (argc > 2) ? atoi(argv[2]) : 128;
    int W  = (argc > 3) ? atoi(argv[3]) : 128;
    int KD = (argc > 4) ? atoi(argv[4]) : 5;
    int KH = KD, KW = KD;
    if (KD > MAX_KD || KH > MAX_KH || KW > MAX_KW) {
        fprintf(stderr, "Kernel dims must be <= %d\n", MAX_KD);
        return 1;
    }
    int outD = D - KD + 1, outH = H - KH + 1, outW = W - KW + 1;
    size_t in_bytes  = (size_t)D * H * W * sizeof(float);
    size_t out_bytes = (size_t)outD * outH * outW * sizeof(float);
    size_t ker_bytes = (size_t)KD * KH * KW * sizeof(float);
    printf("input: %dx%dx%d  kernel: %dx%dx%d  output: %dx%dx%d\n", D, H, W, KD, KH, KW, outD, outH, outW);

    // host 分配与初始化
    float* hIn  = (float*)malloc(in_bytes);
    float* hKer = (float*)malloc(ker_bytes);
    float* hOut = (float*)malloc(out_bytes);
    float* hRef = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < D * H * W; i++) hIn[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < KD * KH * KW; i++) hKer[i] = (float)(rand() % 1000) / 100.0f;

    // device 分配与拷贝
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));

    // 启动配置
    dim3 threads(OT, OT, OT);
    dim3 blocks((outW + OT - 1) / OT, (outH + OT - 1) / OT, (outD + OT - 1) / OT);
    int tileD = OT + KD - 1, tileH = OT + KH - 1, tileW = OT + KW - 1;
    size_t smem_bytes = (size_t)tileD * tileH * tileW * sizeof(float);
    printf("launch: blocks=(%d,%d,%d)  threads=(%d,%d,%d)  smem=%.1f KB/block\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, smem_bytes / 1024.0);

    // 计时
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    conv3d_shared_halo<<<blocks, threads, smem_bytes>>>(dIn, dOut, D, H, W, KD, KH, KW);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 回拷并验证
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    conv3d_cpu(hIn, hKer, hRef, D, H, W, KD, KH, KW);
    int err = 0;
    for (int i = 0; i < outD * outH * outW && err < 5; i++) {
        if (fabsf(hOut[i] - hRef[i]) > 1e-3f * fmaxf(1.0f, fabsf(hRef[i]))) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // 带宽估算
    size_t rw_bytes = ((size_t)D * H * W + (size_t)outD * outH * outW) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hKer); free(hOut); free(hRef);
    return 0;
}

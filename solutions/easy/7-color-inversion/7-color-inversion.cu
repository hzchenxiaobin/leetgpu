// 7-color-inversion.cu —— grid-stride + uchar4 向量化实现颜色反转
// 编译命令: nvcc -O3 -arch=sm_120 7-color-inversion.cu -o invert
// 运行:     ./invert 4096 5120

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                       \
    cudaError_t e = (call);                                                                                \
    if (e != cudaSuccess) {                                                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));              \
        exit(EXIT_FAILURE);                                                                                \
    }                                                                                                      \
} while (0)

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int num_pixels = width * height;
    for (int p = tid; p < num_pixels; p += stride) {
        // uchar4 向量化：一次读 4 字节（整个像素）
        uchar4 pixel = reinterpret_cast<uchar4*>(image)[p];
        pixel.x = 255 - pixel.x; // R
        pixel.y = 255 - pixel.y; // G
        pixel.z = 255 - pixel.z; // B
        // pixel.w (A) 不动
        reinterpret_cast<uchar4*>(image)[p] = pixel; // 一次写 4 字节
    }
}

int main(int argc, char** argv) {
    int width = (argc > 1) ? atoi(argv[1]) : 4096;
    int height = (argc > 2) ? atoi(argv[2]) : 5120;
    size_t num_pixels = (size_t)width * height;
    size_t bytes = num_pixels * 4;
    printf("image: %d x %d  (%zu pixels, %.1f MB)\n", width, height, num_pixels, bytes / 1e6);

    // ---- host 端分配与初始化 ----
    unsigned char* hImg = (unsigned char*)malloc(bytes);
    unsigned char* hRef = (unsigned char*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < bytes; ++i) {
        hImg[i] = (unsigned char)(rand() % 256);
        hRef[i] = hImg[i];
    }

    // ---- device 端分配与拷贝 ----
    unsigned char* dImg;
    CHECK_CUDA(cudaMalloc(&dImg, bytes));
    CHECK_CUDA(cudaMemcpy(dImg, hImg, bytes, cudaMemcpyHostToDevice));

    // ---- grid 规模：SM 数 × 4 ----
    int threads = 256;
    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;
    printf("launch: blocks=%d  threads=%d  (SM=%d)\n", blocks, threads, num_sm);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    invert_kernel<<<blocks, threads>>>(dImg, width, height);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hImg, dImg, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (size_t p = 0; p < num_pixels; ++p) {
        size_t idx = p * 4;
        unsigned char r_exp = (unsigned char)(255 - hRef[idx + 0]);
        unsigned char g_exp = (unsigned char)(255 - hRef[idx + 1]);
        unsigned char b_exp = (unsigned char)(255 - hRef[idx + 2]);
        unsigned char a_exp = hRef[idx + 3];
        if (hImg[idx + 0] != r_exp || hImg[idx + 1] != g_exp ||
            hImg[idx + 2] != b_exp || hImg[idx + 3] != a_exp) {
            if (++err <= 5)
                printf("MISMATCH @pixel %zu: got [%d,%d,%d,%d], expect [%d,%d,%d,%d]\n",
                       p, hImg[idx], hImg[idx + 1], hImg[idx + 2], hImg[idx + 3],
                       r_exp, g_exp, b_exp, a_exp);
        }
    }
    printf("verify: %s  (%d / %zu mismatch)\n", err ? "FAIL" : "PASS", err, num_pixels);

    // ---- 带宽估算：读 image + 写 image = 2 × bytes ----
    size_t rw_bytes = 2 * bytes;
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dImg));
    free(hImg);
    free(hRef);
    return 0;
}

// 24-rainbow-table.cu —— grid-stride loop + R 轮 FNV-1a 串行哈希
// 编译命令: nvcc -O3 -arch=sm_120 24-rainbow-table.cu -o rainbow_table
// 运行:     ./rainbow_table 5000000 10

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                       \
    cudaError_t e = (call);                                                                                \
    if (e != cudaSuccess) {                                                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));              \
        exit(EXIT_FAILURE);                                                                                \
    }                                                                                                      \
} while (0)

// FNV-1a 一轮：unsigned int 乘法自动 mod 2^32，等价于 64 位乘后 & 0xFFFFFFFF
__device__ __forceinline__ unsigned int fnv1a_round(unsigned int x) {
    const unsigned int FNV_PRIME = 16777619u;
    const unsigned int OFFSET_BASIS = 2166136261u;
    unsigned int hash = OFFSET_BASIS;
    #pragma unroll
    for (int b = 0; b < 4; ++b) {
        unsigned int byte = (x >> (8 * b)) & 0xFFu;
        hash = (hash ^ byte) * FNV_PRIME;   // 32 位乘法自然回绕 = 低 32 位
    }
    return hash;
}

__global__ void rainbow_kernel(const int* input, unsigned int* output, int N, int R) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {       // 外层 grid-stride：元素间并行
        unsigned int x = (unsigned int)input[i];   // int32 按位重解释为 uint32
        for (int r = 0; r < R; ++r) {              // 内层串行：R 轮哈希依赖链
            x = fnv1a_round(x);
        }
        output[i] = x;
    }
}

// ---- CPU 参考实现（用 uint64 乘 + 掩码，与平台 reference_impl 等价）----
uint32_t fnv1a_cpu(uint32_t x) {
    uint32_t hash = 2166136261u;
    for (int b = 0; b < 4; ++b) {
        uint32_t byte = (x >> (8 * b)) & 0xFFu;
        hash = (uint32_t)((uint64_t)(hash ^ byte) * 16777619ull & 0xFFFFFFFFull);
    }
    return hash;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5000000;
    int R = (argc > 2) ? atoi(argv[2]) : 10;
    size_t bytes_in = (size_t)N * sizeof(int);
    size_t bytes_out = (size_t)N * sizeof(unsigned int);
    printf("N = %d  R = %d  (%.1f MB in + %.1f MB out)\n", N, R, bytes_in / 1e6, bytes_out / 1e6);

    // ---- host 端分配与初始化 ----
    int* hIn = (int*)malloc(bytes_in);
    unsigned int* hOut = (unsigned int*)malloc(bytes_out);
    srand(42);
    for (int i = 0; i < N; ++i) hIn[i] = (int)((rand() << 16) ^ rand());   // 含负数

    // ---- device 端分配与拷贝 ----
    int* dIn;
    unsigned int* dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes_in));
    CHECK_CUDA(cudaMalloc(&dOut, bytes_out));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes_in, cudaMemcpyHostToDevice));

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
    rainbow_kernel<<<blocks, threads>>>(dIn, dOut, N, R);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes_out, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < N; ++i) {
        unsigned int x = (unsigned int)hIn[i];
        for (int r = 0; r < R; ++r) x = fnv1a_cpu(x);
        if (hOut[i] != x) {
            if (++err <= 5) printf("MISMATCH @%d: got %u, expect %u\n", i, hOut[i], x);
        }
    }
    printf("verify: %s  (%d / %d mismatch)\n", err ? "FAIL" : "PASS", err, N);

    // ---- 带宽与算术强度估算 ----
    size_t rw = bytes_in + bytes_out;                 // 读 input + 写 output
    float bw_gbs = (rw / 1e9) / (ms / 1e3);
    float ops = (float)N * R * 4 * 3.0f;              // 每轮 4 字节 × (XOR+移位+乘) ≈ 3 op
    printf("effective bandwidth: %.1f GB/s   (~%.1f Gop/s integer)\n", bw_gbs, ops / (ms / 1e3) / 1e9);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hOut);
    return 0;
}

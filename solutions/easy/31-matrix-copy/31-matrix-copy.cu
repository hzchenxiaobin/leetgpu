// 31-matrix-copy.cu —— Coalesced Matrix Copy with Bandwidth Measurement
// 编译命令: nvcc -O3 -std=c++14 -arch=sm_120 31-matrix-copy.cu -o matrix_copy
// 运行:     ./matrix_copy 4096 4096

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

#define BLOCK_SIZE 256

// ---- ① 标量 grid-stride 版：每线程搬 1 个 float ----
__global__ void matrix_copy_kernel(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        out[i] = in[i]; // coalesced，但每事务只搬 4B
    }
}

// ---- ② float4 向量化版：每线程搬 4 个 float（128-bit）----
__global__ void matrix_copy_vectorized(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int vec_n = N / 4; // float4 元素数

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);

    // 主循环：4 元素一组，1 条 16B load + 1 条 16B store
    for (int i = tid; i < vec_n; i += stride) {
        out4[i] = in4[i];
    }

    // 尾部：处理 N%4 个剩余元素（本题 N=4096%4=0，通常不执行）
    int tail_start = vec_n * 4;
    for (int i = tail_start + tid; i < N; i += stride) {
        out[i] = in[i];
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 4096;
    int N = (argc > 2) ? atoi(argv[2]) : 4096;
    int num = M * N;
    size_t bytes = (size_t)num * sizeof(float);
    printf("M=%d, N=%d  (%.1f MB per matrix)\n", M, N, bytes / 1e6);

    // ---- host 端分配与初始化 ----
    float* hIn = (float*)malloc(bytes);
    float* hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < num; ++i) {
        hIn[i] = (float)(rand() % 10000) / 100.0f;
    }

    // ---- device 端分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // ---- 选择 grid 规模：SM 数 × 4，让 grid-stride 发挥作用 ----
    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4; // 经验值：填满 SM 又不过度启动
    printf("launch: blocks=%d  threads=%d  (SM=%d)\n", blocks, BLOCK_SIZE, num_sm);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // 计时辅助：跑 kernel/memcpy 并返回毫秒
    auto time_one = [&](auto launcher) -> float {
        cudaEventRecord(t0);
        launcher();
        cudaEventRecord(t1);
        CHECK_CUDA(cudaDeviceSynchronize());
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        return ms;
    };

    float ms_scalar = time_one([&] { matrix_copy_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num); });
    float ms_vec4 = time_one([&] { matrix_copy_vectorized<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num); });
    float ms_memcpy = time_one([&] { CHECK_CUDA(cudaMemcpy(dOut, dIn, bytes, cudaMemcpyDeviceToDevice)); });

    printf("\n--- timing (ms) / bandwidth 2x bytes (GB/s) ---\n");
    printf("scalar      : %.3f ms / %.1f\n", ms_scalar, (2.0f * bytes / 1e9) / (ms_scalar / 1e3));
    printf("float4      : %.3f ms / %.1f\n", ms_vec4, (2.0f * bytes / 1e9) / (ms_vec4 / 1e3));
    printf("cudaMemcpy  : %.3f ms / %.1f\n", ms_memcpy, (2.0f * bytes / 1e9) / (ms_memcpy / 1e3));

    // ---- 回拷并验证（用 float4 版结果）----
    matrix_copy_vectorized<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < num; ++i) {
        if (fabsf(hOut[i] - hIn[i]) > 1e-5f) {
            if (++err <= 5)
                printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], hIn[i]);
        }
    }
    printf("verify: %s  (%d / %d mismatch)\n", err ? "FAIL" : "PASS", err, num);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hOut);
    return 0;
}

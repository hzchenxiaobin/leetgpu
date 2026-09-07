// 15-sorting.cu —— Bitonic sort：补齐 2 的幂 → shared-mem 局部排序 → 全局逐级归并
// 编译命令: nvcc -O3 -arch=sm_120 15-sorting.cu -o sorting
// 运行:     ./sorting [N]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

#define TILE 1024          // shared-mem tile：一个 block 排 1024 个元素（1 元素/线程）
#define THREADS_G 256      // 全局 compare-swap kernel 的每 block 线程数

// ① 局部排序：每 block 在 shared memory 里跑完前 log2(TILE) 级 bitonic 网络
__global__ void bitonic_local_sort(const float* data, float* buf, int N, int P) {
    int base = blockIdx.x * TILE;
    int tid = threadIdx.x;
    __shared__ float sh[TILE];
    // 加载到 shared mem，越界补 +∞（让非 2 的幂 N 也能套同一网络）
    sh[tid] = (base + tid < N) ? data[base + tid] : INFINITY;
    __syncthreads();

    for (int size = 2; size <= TILE; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int partner = tid ^ stride;
            if (tid < partner) {                       // 只让小索引端做交换，避免竞争
                int gi = base + tid;                    // 全局索引（决定升降方向）
                bool asc = ((gi & size) == 0);
                float a = sh[tid], b = sh[partner];
                if ((asc && a > b) || (!asc && a < b)) {
                    sh[tid] = b; sh[partner] = a;
                }
            }
            __syncthreads();
        }
    }
    if (base + tid < P) buf[base + tid] = sh[tid];
}

// ② 全局 compare-swap：处理一个 (size, stride) 子步，跨 block 在 global memory 上交换
__global__ void bitonic_global_step(float* buf, int P, int size, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P) return;
    int partner = i ^ stride;
    if (i < partner) {
        bool asc = ((i & size) == 0);
        float a = buf[i], b = buf[partner];
        if ((asc && a > b) || (!asc && a < b)) {
            buf[i] = b; buf[partner] = a;
        }
    }
}

// LeetGPU 提交接口：原地升序排序 data[0..N)
extern "C" void solve(float* data, int N) {
    if (N <= 1) return;
    int P = 1;
    while (P < N) P <<= 1;                            // 补齐到 2 的幂
    float* buf;
    CHECK_CUDA(cudaMalloc(&buf, (size_t)P * sizeof(float)));

    int numLocalBlocks = (P + TILE - 1) / TILE;
    bitonic_local_sort<<<numLocalBlocks, TILE>>>(data, buf, N, P);

    for (int size = 2 * TILE; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            int blocks = (P + THREADS_G - 1) / THREADS_G;
            bitonic_global_step<<<blocks, THREADS_G>>>(buf, P, size, stride);
        }
    }
    // 前 N 个即排序结果（+∞ 已沉到末尾），写回原数组
    CHECK_CUDA(cudaMemcpy(data, buf, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaFree(buf));
}

// ---------------- 本地自测 ----------------
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.2f MB)\n", N, bytes / 1e6);

    float *hData = (float*)malloc(bytes), *hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hData[i] = (float)((rand() % 200000) - 100000) / 100.0f;
        hRef[i] = hData[i];
    }
    std::sort(hRef, hRef + N);                        // CPU 参考

    float *dData;
    CHECK_CUDA(cudaMalloc(&dData, bytes));
    CHECK_CUDA(cudaMemcpy(dData, hData, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    solve(dData, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    CHECK_CUDA(cudaMemcpy(hData, dData, bytes, cudaMemcpyDeviceToHost));

    // 验证：逐元素比对 + 单调性
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double d = fabs((double)hData[i] - hRef[i]);
        if (d > max_err) max_err = d;
    }
    bool ok = max_err < 1e-4;
    printf("[bitonic] time: %.3f ms  max_err: %.3e  %s\n", ms, max_err, ok ? "PASS" : "FAIL");
    printf("throughput: %.2f M elem/s\n", N / ms / 1000.0);

    CHECK_CUDA(cudaFree(dData));
    free(hData); free(hRef);
    return 0;
}

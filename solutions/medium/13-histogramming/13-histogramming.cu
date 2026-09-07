// 13-histogramming.cu —— shared memory privatization 直方图
// 编译命令: nvcc -O3 -arch=sm_120 13-histogramming.cu -o histogram
// 运行:     ./histogram 50000000 256

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// 朴素版：所有线程 atomicAdd 到 global histogram（剧烈竞争，用于对比基准）
__global__ void histogram_naive(const int* input, int* hist, int N, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&hist[bin], 1);
        }
    }
}

// 优化版：privatization —— 每 block 一份 shared histogram，最后合并到 global
// 用动态 shared memory（extern __shared__）适配任意 num_bins（1..1024）
__global__ void histogram_privatized(const int* input, int* hist, int N, int B) {
    extern __shared__ int s_hist[];        // 大小 = B，启动时传入 B*sizeof(int)

    int tid = threadIdx.x;

    // ① 初始化 shared histogram 为 0（block 内协作清零）
    for (int b = tid; b < B; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // ② grid-stride 读输入，atomicAdd 到 shared（block 内竞争远小于 global）
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&s_hist[bin], 1);    // shared atomic，低延迟
        }
    }
    __syncthreads();

    // ③ 把 shared histogram 合并到 global（每 bin 一次 global atomic）
    for (int b = tid; b < B; b += blockDim.x) {
        int v = s_hist[b];
        if (v > 0) {
            atomicAdd(&hist[b], v);
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 50000000;
    int B = (argc > 2) ? atoi(argv[2]) : 256;
    size_t bytes = (size_t)N * sizeof(int);
    printf("N = %d, B = %d  (%.1f MB input)\n", N, B, bytes / 1e6);

    // ---- host 端 ----
    int* hIn = (int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = rand() % B;               // 值域 [0, B)
    }

    // ---- device 端 ----
    int *dIn, *dHist;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dHist, B * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;                // 经验值，保证 wave 充足但不过载
    printf("blocks = %d, threads/block = %d\n", blocks, BLOCK_SIZE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- 优化版 ----
    CHECK_CUDA(cudaMemset(dHist, 0, B * sizeof(int)));
    cudaEventRecord(t0);
    histogram_privatized<<<blocks, BLOCK_SIZE, B * sizeof(int)>>>(dIn, dHist, N, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_priv = 0.0f;
    cudaEventElapsedTime(&ms_priv, t0, t1);

    // ---- CPU 验证 ----
    int* hHist = (int*)malloc(B * sizeof(int));
    CHECK_CUDA(cudaMemcpy(hHist, dHist, B * sizeof(int), cudaMemcpyDeviceToHost));
    int* ref = (int*)calloc(B, sizeof(int));
    for (int i = 0; i < N; ++i) {
        int b = hIn[i];
        if (b >= 0 && b < B) ref[b]++;
    }
    long max_err = 0;
    for (int b = 0; b < B; ++b) {
        long d = (long)hHist[b] - ref[b];
        if (d < 0) d = -d;
        if (d > max_err) max_err = d;
    }
    printf("[privatized] time: %.3f ms  max_err: %ld  %s\n", ms_priv, max_err,
           max_err == 0 ? "PASS" : "FAIL");

    // ---- 朴素版对比 ----
    CHECK_CUDA(cudaMemset(dHist, 0, B * sizeof(int)));
    int naive_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(t0);
    histogram_naive<<<naive_blocks, BLOCK_SIZE>>>(dIn, dHist, N, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, t0, t1);
    CHECK_CUDA(cudaMemcpy(hHist, dHist, B * sizeof(int), cudaMemcpyDeviceToHost));
    max_err = 0;
    for (int b = 0; b < B; ++b) {
        long d = (long)hHist[b] - ref[b];
        if (d < 0) d = -d;
        if (d > max_err) max_err = d;
    }
    printf("[naive]       time: %.3f ms  max_err: %ld  %s  speedup: %.2fx\n",
           ms_naive, max_err, max_err == 0 ? "PASS" : "FAIL", ms_naive / ms_priv);

    // ---- 带宽估算（只算读 input 的量）----
    float bw_gbs = (bytes / 1e9) / (ms_priv / 1e3);
    printf("read bandwidth (privatized): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dHist));
    free(hIn);
    free(hHist);
    free(ref);
    return 0;
}

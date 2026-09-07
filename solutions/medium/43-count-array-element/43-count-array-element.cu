// 43-count-array-element.cu —— Count Array Element（predicate + 两级归约）
// 编译命令: nvcc -O3 -arch=sm_120 43-count-array-element.cu -o count_array
// 运行:     ./count_array 100000000 501010

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// warp 内树形归约（__shfl_down_sync），对 int 同样适用
__inline__ __device__ int warp_reduce(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 朴素版：命中线程 atomicAdd 到单个全局地址（剧烈竞争，用于对比基准）
__global__ void count_naive(const int* input, int* output, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (input[i] == K)
            atomicAdd(&output[0], 1);
    }
}

// 优化版：predicate + grid-stride 局部计数 + 两级归约，全程零 global atomic
__global__ void count_kernel(const int* input, int* output, int N, int K) {
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // ① grid-stride 读输入，predicate 判定 + 局部计数（寄存器内累加）
    int cnt = 0;
    for (int i = gid; i < N; i += stride) {
        cnt += (input[i] == K);   // 布尔隐式转 0/1，融合判定与求和
    }

    // ② warp 内归约：32 lane 的 cnt 树形归约到 lane 0
    cnt = warp_reduce(cnt);
    if (lane == 0)
        warp_sums[warp_id] = cnt;   // 8 个 warp 的部分和写入 shared
    __syncthreads();

    // ③ warp 间归约：由第一个 warp 把 8 个 warp_sums 再归约一次
    if (warp_id == 0) {
        cnt = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0;
        cnt = warp_reduce(cnt);
        if (lane == 0)
            output[blockIdx.x] = cnt;   // block 的总命中数写入 global
    }
}

// final 归约：聚合所有 block 的部分和到 output[0]
__global__ void final_reduce(int* output, int num_blocks) {
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int val = (tid < num_blocks) ? output[tid] : 0;
    val = warp_reduce(val);
    if (tid % WARP_SIZE == 0)
        warp_sums[tid / WARP_SIZE] = val;
    __syncthreads();

    if (tid < WARP_SIZE) {
        val = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_sums[tid] : 0;
        val = warp_reduce(val);
        if (tid == 0)
            output[0] = val;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 100000000;
    int K = (argc > 2) ? atoi(argv[2]) : 501010;
    size_t bytes = (size_t)N * sizeof(int);
    printf("N = %d, K = %d  (%.1f MB input)\n", N, K, bytes / 1e6);

    // ---- host 端 ----
    int* hIn = (int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i)
        hIn[i] = (rand() % 100000) + 1;   // 值域 [1, 100000]

    // ---- device 端 ----
    int *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(int)));   // 优化版临时复用为部分和缓冲
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 8;   // 经验值，保证 wave 充足但不过载
    if (blocks > (N + BLOCK_SIZE - 1) / BLOCK_SIZE)
        blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("blocks = %d, threads/block = %d\n", blocks, BLOCK_SIZE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- CPU 验证 ----
    int ref = 0;
    for (int i = 0; i < N; ++i)
        ref += (hIn[i] == K);

    // ---- 优化版：两阶段归约 ----
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(int)));
    cudaEventRecord(t0);
    count_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N, K);
    final_reduce<<<1, BLOCK_SIZE>>>(dOut, blocks);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_opt = 0.0f;
    cudaEventElapsedTime(&ms_opt, t0, t1);
    int hOut;
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[reduction]  time: %.3f ms  result: %d  ref: %d  %s\n", ms_opt, hOut, ref,
           hOut == ref ? "PASS" : "FAIL");

    // ---- 朴素版：atomicAdd ----
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(int)));
    int naive_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(t0);
    count_naive<<<naive_blocks, BLOCK_SIZE>>>(dIn, dOut, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, t0, t1);
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[atomic]     time: %.3f ms  result: %d  ref: %d  %s  speedup: %.2fx\n",
           ms_naive, hOut, ref, hOut == ref ? "PASS" : "FAIL", ms_naive / ms_opt);

    // ---- 带宽估算（只算读 input 的量）----
    float bw_gbs = (bytes / 1e9) / (ms_opt / 1e3);
    printf("read bandwidth (reduction): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    return 0;
}

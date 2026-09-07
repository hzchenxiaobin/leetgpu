// 72-stream-compaction.cu —— Stream Compaction（predicate + exclusive scan + scatter）
// 编译命令: nvcc -O3 -arch=sm_120 72-stream-compaction.cu -o stream_compaction
// 运行:     ./stream_compaction

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

// 单个 warp 的 exclusive prefix sum（Hillis-Steele），结果放在各 lane 的寄存器
__device__ __forceinline__ int warp_excl_scan(int val) {
    int orig = val;
    int sum = val;
// exclusive：先减自己再加前缀
    #pragma unroll
    for (int offset = 1; offset < WARP; offset *= 2) {
        int v = __shfl_up_sync(0xffffffff, sum, offset);
        if ((threadIdx.x & (WARP - 1)) >= offset)
            sum += v;
    }
    return sum - orig; // exclusive = inclusive - 自己
}

// block 内 exclusive scan（每个 thread 处理 1 个元素）
__global__ void block_excl_scan_kernel(const int* pred, int* ps, int* block_sums, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    __shared__ int warp_sums[WARP];

    int val = (tid < N) ? pred[tid] : 0;
    int warp_excl = warp_excl_scan(val);

    // 每个 warp 的总和 = 最后一个 lane 的 inclusive
    int warp_total = warp_excl + val;
    if (lane == WARP - 1)
        warp_sums[warp_id] = warp_total;
    __syncthreads();

    // 第一个 warp 扫描 warp_sums
    if (warp_id == 0) {
        int w = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0;
        int w_excl = warp_excl_scan(w);
        if (lane < blockDim.x / WARP)
            warp_sums[lane] = w_excl;
    }
    __syncthreads();

    // 把 warp 前缀加到每个元素上
    int block_excl = warp_excl + warp_sums[warp_id];
    if (tid < N)
        ps[tid] = block_excl;

    // block 总和写到 block_sums
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = block_excl + val;
    }
}

// 第二遍：把前序 block 的和累加到每个 block 的 ps 上
__global__ void add_prev_blocks(int* ps, const int* block_sums_excl, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N && blockIdx.x > 0) {
        ps[tid] += block_sums_excl[blockIdx.x];
    }
}

// scatter：pred[i]==1 时 output[ps[i]] = input[i]
__global__ void scatter_kernel(const int* input, const int* pred, const int* ps, int* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && pred[i] == 1) {
        output[ps[i]] = input[i];
    }
}

// predicate：pred[i] = (input[i] != 0) ? 1 : 0
__global__ void predicate_kernel(const int* input, int* pred, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        pred[i] = (input[i] != 0) ? 1 : 0;
}

int main() {
    int N = 1000000;
    std::vector<int> h_input(N);
    srand(42);
    for (auto& x : h_input)
        x = (rand() % 3 == 0) ? 0 : (rand() % 100); // ~1/3 为 0

    size_t bytes = N * sizeof(int);
    int *d_input, *d_pred, *d_ps, *d_output, *d_block_sums;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_pred, bytes);
    cudaMalloc(&d_ps, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_block_sums, bytes);
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // 1. predicate
    int blocks = (N + BLOCK - 1) / BLOCK;
    predicate_kernel<<<blocks, BLOCK>>>(d_input, d_pred, N);

    // 2. block 内 exclusive scan
    block_excl_scan_kernel<<<blocks, BLOCK>>>(d_pred, d_ps, d_block_sums, N);

    // 3. 对 block_sums 做 exclusive scan（num_blocks 可能 > 1024，CPU 上做最稳妥）
    int num_blocks = blocks;
    int* d_block_sums_excl;
    cudaMalloc(&d_block_sums_excl, num_blocks * sizeof(int));
    int* h_block_sums = (int*)malloc(num_blocks * sizeof(int));
    cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
    int* h_block_sums_excl = (int*)malloc(num_blocks * sizeof(int));
    h_block_sums_excl[0] = 0;
    for (int i = 1; i < num_blocks; i++)
        h_block_sums_excl[i] = h_block_sums_excl[i - 1] + h_block_sums[i - 1];
    cudaMemcpy(d_block_sums_excl, h_block_sums_excl, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    free(h_block_sums);
    free(h_block_sums_excl);

    // 4. 累加前序 block
    add_prev_blocks<<<blocks, BLOCK>>>(d_ps, d_block_sums_excl, N);

    // 5. scatter
    scatter_kernel<<<blocks, BLOCK>>>(d_input, d_pred, d_ps, d_output, N);

    cudaDeviceSynchronize();

    // 取回 count = ps[N-1] + pred[N-1]
    int h_ps_last, h_pred_last;
    cudaMemcpy(&h_ps_last, &d_ps[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pred_last, &d_pred[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    int count = h_ps_last + h_pred_last;

    // CPU 验证
    std::vector<int> cpu_out;
    for (auto x : h_input)
        if (x != 0)
            cpu_out.push_back(x);
    bool pass = ((int)cpu_out.size() == count);

    std::vector<int> h_gpu_out(count);
    cudaMemcpy(h_gpu_out.data(), d_output, count * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count && pass; i++)
        if (h_gpu_out[i] != cpu_out[i])
            pass = false;

    printf("GPU count=%d, CPU count=%d, %s\n", count, (int)cpu_out.size(), pass ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_pred);
    cudaFree(d_ps);
    cudaFree(d_output);
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_excl);
    return 0;
}

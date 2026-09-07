// 47-subarray-sum.cu —— Subarray Sum（grid-stride 累加 + 两级 block 归约 + long long 累加）
// 编译命令: nvcc -O3 -arch=sm_120 47-subarray-sum.cu -o subarray_sum
// 运行:     ./subarray_sum

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

__device__ __forceinline__ long long warp_reduce_ll(long long val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// grid-stride 累加 [S, E] → warp 归约 → block 归约 → atomicAdd 到 scratch(long long)
__global__ void subarray_sum_kernel(const int* input, unsigned long long* scratch,
                                    int S, int range_len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;
    __shared__ long long warp_sums[WARP];

    long long sum = 0;
    int stride = gridDim.x * blockDim.x;
    for (int off = tid; off < range_len; off += stride) {
        sum += (long long)input[S + off];
    }

    sum = warp_reduce_ll(sum);
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0;
        sum = warp_reduce_ll(sum);
        if (lane == 0)
            atomicAdd(scratch, (unsigned long long)sum);
    }
}

// 单线程：把 long long scratch cast 成 int 写入 output[0]
__global__ void cast_to_int(const unsigned long long* scratch, int* output) {
    output[0] = (int)((long long)scratch[0]);
}

int main() {
    int N = 100000000;
    int S = 1000, E = N - 1;          // 区间几乎全长
    int range_len = E - S + 1;
    size_t bytes = (size_t)N * sizeof(int);

    std::vector<int> h_input(N);
    srand(42);
    for (int i = 0; i < N; ++i)
        h_input[i] = (rand() % 2000) - 1000;   // [-1000, 999]

    int* d_input;
    int* d_output;
    unsigned long long* d_scratch;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(int));
    cudaMalloc(&d_scratch, sizeof(unsigned long long));
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_scratch, 0, sizeof(unsigned long long));

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = num_sm * 4;
    int threads = BLOCK;
    printf("launch: blocks=%d  threads=%d  (SM=%d, range_len=%d)\n",
           blocks, threads, num_sm, range_len);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    subarray_sum_kernel<<<blocks, threads>>>(d_input, d_scratch, S, range_len);
    cast_to_int<<<1, 1>>>(d_scratch, d_output);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    int gpu_result;
    cudaMemcpy(&gpu_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 验证（long long 累加再 cast）
    long long cpu_sum = 0;
    for (int i = S; i <= E; ++i)
        cpu_sum += h_input[i];
    int cpu_result = (int)cpu_sum;

    printf("GPU: %d, CPU: %d, %s\n", gpu_result, cpu_result,
           gpu_result == cpu_result ? "PASS" : "FAIL");

    // 带宽估算：只读 range_len 个 int
    size_t rw_bytes = (size_t)range_len * sizeof(int);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective read bandwidth: %.1f GB/s\n", bw_gbs);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scratch);
    return 0;
}

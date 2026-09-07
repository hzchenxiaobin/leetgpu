// 70-segmented-prefix-sum.cu —— Segmented Exclusive Prefix Sum（段内 scan + 段边界归零）
// 编译命令: nvcc -O3 -arch=sm_120 70-segmented-prefix-sum.cu -o segmented_prefix_sum
// 运行:     ./segmented_prefix_sum

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

// 单个 warp 的 exclusive prefix sum（Hillis-Steele）
__device__ __forceinline__ int warp_excl_scan(int val) {
    int orig = val;
    int sum = val;
    #pragma unroll
    for (int offset = 1; offset < WARP; offset *= 2) {
        int v = __shfl_up_sync(0xffffffff, sum, offset);
        if ((threadIdx.x & (WARP - 1)) >= offset)
            sum += v;
    }
    return sum - orig; // exclusive = inclusive - 自己
}

// segmented exclusive scan：段首元素的前缀强制为 0
// is_seg_start[i]==1 表示 i 是段首（新段开始）
__global__ void segmented_excl_scan_kernel(const int* input, int* output, const int* is_seg_start, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    __shared__ int warp_sums[WARP];
    __shared__ int warp_carry[WARP + 1]; // 本 block 内前序 warp 的段和 carry

    int val = (tid < N) ? input[tid] : 0;
    int seg_flag = (tid < N) ? is_seg_start[tid] : 0;

    // 分段 warp inclusive scan：段首不累加前序，非段首累加
    int sum = val;
    int flag = seg_flag;
    #pragma unroll
    for (int offset = 1; offset < WARP; offset *= 2) {
        int v = __shfl_up_sync(0xffffffff, sum, offset);
        int f = __shfl_up_sync(0xffffffff, flag, offset);
        if ((threadIdx.x & (WARP - 1)) >= offset) {
            if (!flag) sum += v;
            flag |= f;
        }
    }
    int warp_excl = sum - val;
    int warp_total = sum;
    if (lane == WARP - 1)
        warp_sums[warp_id] = warp_total;
    __syncthreads();

    // 第一个 warp 累加 warp_sums（block 内跨 warp carry）
    if (warp_id == 0) {
        int w = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0;
        int w_excl = warp_excl_scan(w);
        if (lane < blockDim.x / WARP)
            warp_sums[lane] = w_excl;
    }
    __syncthreads();

    int block_excl = warp_excl + warp_sums[warp_id];

    // 段首元素强制前缀为 0（不 carry 前序段的和）
    if (seg_flag)
        block_excl = 0;

    if (tid < N)
        output[tid] = block_excl;
    // 注：跨 block 的段 carry 需要第二遍 kernel（类似三阶段 scan），
    //     此处教学版假设每段不超过一个 block，正式版需补 block 间段和传递。
}

int main() {
    // 两段：A=[3,1,2], B=[4,2,1]
    std::vector<int> h_input = {3, 1, 2, 4, 2, 1};
    std::vector<int> h_seg = {1, 0, 0, 1, 0, 0}; // 段首标记
    int N = h_input.size();

    int *d_input, *d_output, *d_seg;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMalloc(&d_seg, N * sizeof(int));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seg, h_seg.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    int blocks = (N + BLOCK - 1) / BLOCK;
    segmented_excl_scan_kernel<<<blocks, BLOCK>>>(d_input, d_output, d_seg, N);
    cudaDeviceSynchronize();

    std::vector<int> h_out(N);
    cudaMemcpy(h_out.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 验证
    std::vector<int> cpu_out(N);
    int sum = 0;
    for (int i = 0; i < N; i++) {
        if (h_seg[i])
            sum = 0; // 段首归零
        cpu_out[i] = sum;
        sum += h_input[i];
    }

    bool pass = true;
    for (int i = 0; i < N; i++) {
        printf("out[%d]=%d (cpu=%d) %s\n", i, h_out[i], cpu_out[i], h_out[i] == cpu_out[i] ? "✓" : "✗");
        if (h_out[i] != cpu_out[i])
            pass = false;
    }
    printf("%s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_seg);
    return 0;
}

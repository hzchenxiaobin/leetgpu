// 29-top-k-selection.cu —— Top K Selection（bitonic sort + 取后 k）
// 编译命令: nvcc -O3 -arch=sm_120 29-top-k-selection.cu -o top_k
// 运行:     ./top_k

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
// 每步 compare-swap：比较距离 j 的两元素，按方向交换
__global__ void bitonic_sort_kernel(int* data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    // log2(N) 个阶段，每阶段做 bitonic merge
    for (int k = 2; k <= N; k *= 2) {        // 子序列长度
        for (int j = k / 2; j > 0; j /= 2) { // 比较距离
            int i = tid ^ j;                 // 配对索引
            if (i > tid && i < N) {
                bool ascending = ((tid & k) == 0);
                if ((ascending && data[tid] > data[i]) || (!ascending && data[tid] < data[i])) {
                    // 交换（用原子或 warp shuffle；教学版用简单条件）
                    int tmp = data[tid];
                    data[tid] = data[i];
                    data[i] = tmp;
                }
            }
            __syncthreads();
        }
    }
}

// 教学版：用单 block 排序小数组（N ≤ 1024），正式版需多 block 协作
// 注意：上述 __syncthreads() 跨 block 无效，正式版用 cooperative groups 或多 kernel
//       此处简化演示 bitonic sort 的 compare-swap 逻辑

// 更实用的版本：每 thread 处理多元素，block 内 shared memory 排序
#define BLOCK 256

__global__ void bitonic_sort_block(int* data, int N) {
    __shared__ int sdata[2 * BLOCK];
    int tid = threadIdx.x;

    // 加载数据到 shared memory
    if (tid < N)
        sdata[tid] = data[tid];
    else
        sdata[tid] = INT_MAX;
    __syncthreads();

    // bitonic sort in shared memory
    for (int k = 2; k <= 2 * BLOCK; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int i = tid ^ j;
            if (i > tid) {
                bool up = ((tid & k) == 0);
                int a = sdata[tid], b = sdata[i];
                if ((up && a > b) || (!up && a < b)) {
                    sdata[tid] = b;
                    sdata[i] = a;
                }
            }
            __syncthreads();
        }
    }
    if (tid < N)
        data[tid] = sdata[tid];
}

int main() {
    int N = 8, k = 3;
    std::vector<int> h_input = {5, 2, 8, 1, 9, 3, 7, 4};

    int* d_data;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // bitonic sort（升序）
    bitonic_sort_block<<<1, 2 * BLOCK>>>(d_data, N);
    cudaDeviceSynchronize();

    // 取后 k 个（最大的 k 个）
    std::vector<int> h_out(N);
    cudaMemcpy(h_out.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证：排序正确 + top-k 正确
    std::vector<int> ref = h_input;
    std::sort(ref.begin(), ref.end());
    bool pass = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != ref[i]) pass = false;
    for (int i = 0; i < k; i++)
        if (h_out[N - k + i] != ref[N - k + i]) pass = false;

    printf("Sorted: ");
    for (int x : h_out)
        printf("%d ", x);
    printf("\nTop %d: ", k);
    for (int i = N - k; i < N; i++)
        printf("%d ", h_out[i]);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_data);
    return 0;
}

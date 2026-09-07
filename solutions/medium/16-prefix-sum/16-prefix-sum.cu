// 16-prefix-sum.cu —— 三阶段分块 scan：warp shuffle + block scan + 全局偏移加回
// 编译命令: nvcc -O3 -arch=sm_120 16-prefix-sum.cu -o prefix_sum
// 运行:     ./prefix_sum 16777216

    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val;
}

__inline__ __device__ float block_exclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    float inclusive = warp_inclusive_scan(val);

    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;
    }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS)
            warp_sums[lane] = v;
    }
    __syncthreads();

    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];
    float exclusive = warp_offset + (inclusive - val);

    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = warp_offset + inclusive;
    }
    return exclusive;
}

__global__ void scan_block_kernel(const float* input, float* output, float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool valid = (tid < N);
    float val = valid ? input[tid] : 0.0f;
    float exclusive = block_exclusive_scan(val, &block_sums[blockIdx.x]);
    if (valid)
        output[tid] = exclusive;
}

__global__ void scan_offsets_kernel(const float* block_sums, float* block_offsets, int M) {
    __shared__ float s_chunk_total;
    __shared__ float s_running;
    int tid = threadIdx.x;

    if (tid == 0) {
        s_running = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < M; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < M) ? block_sums[idx] : 0.0f;

        float exclusive = block_exclusive_scan(val, &s_chunk_total);

        if (idx < M) {
            block_offsets[idx] = exclusive + s_running;
        }

        __syncthreads();
        if (tid == 0)
            s_running += s_chunk_total;
        __syncthreads();
    }
}

__global__ void add_offset_kernel(float* output, const float* input, const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N)
        return;
    output[tid] = output[tid] + block_offsets[blockIdx.x] + input[tid];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16777216;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    float* hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;
    }

    float *dIn, *dOut, *dBlockSums, *dBlockOffsets;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&dBlockSums, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBlockOffsets, numBlocks * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    scan_block_kernel<<<numBlocks, BLOCK_SIZE>>>(dIn, dOut, dBlockSums, N);
    scan_offsets_kernel<<<1, BLOCK_SIZE>>>(dBlockSums, dBlockOffsets, numBlocks);
    add_offset_kernel<<<numBlocks, BLOCK_SIZE>>>(dOut, dIn, dBlockOffsets, N);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (three-pass): %.3f ms\n", ms);

    float* hOut = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    double acc = 0.0;
    int fail = 0;
    int checkPts[] = {0, 1, 2, N / 4, N / 2, N - 2, N - 1};
    for (int k = 0; k < 7; ++k) {
        int i = checkPts[k];
        for (int j = (k == 0 ? 0 : checkPts[k - 1] + 1); j <= i; ++j)
            acc += hIn[j];
        if (fabsf(hOut[i] - (float)acc) > 1e-2f * (1.0f + fabsf((float)acc))) {
            printf("FAIL at i=%d: GPU=%f CPU=%f\n", i, hOut[i], (float)acc);
            fail = 1;
            break;
        }
    }
    printf("%s\n", fail ? "FAIL" : "PASS");

    float bw_gbs = (2.0 * bytes / 1e9) / (ms / 1e3);
    printf("I/O bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaFree(dBlockSums));
    CHECK_CUDA(cudaFree(dBlockOffsets));
    free(hIn);
    free(hOut);
    return 0;
}

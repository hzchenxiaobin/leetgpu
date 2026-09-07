// 38-nearest-neighbor.cu —— Tiled nearest neighbor（shared memory 数据复用）
// 编译命令: nvcc -O3 -arch=sm_120 38-nearest-neighbor.cu -o nn
// 运行:     ./nn 10000

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_SIZE  256

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// 朴素版：每 thread 一个 query，遍历全部 N 个 reference（无复用）
__global__ void nn_naive(const float* points, int* indices, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi = points[i*3], yi = points[i*3+1], zi = points[i*3+2];
    float best = FLT_MAX;
    int best_j = -1;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = xi - points[j*3];
        float dy = yi - points[j*3+1];
        float dz = zi - points[j*3+2];
        float d = dx*dx + dy*dy + dz*dz;
        if (d < best) { best = d; best_j = j; }
    }
    indices[i] = best_j;
}

// 优化版：tiled —— reference 点分块载入 shared，256 thread 共享复用
__global__ void nn_tiled(const float* points, int* indices, int N) {
    __shared__ float shared_pts[TILE_SIZE][3];

    int i = blockIdx.x * blockDim.x + threadIdx.x;   // query 点索引
    int tid = threadIdx.x;

    // 每 thread 把自己的 query 点载入寄存器（全程常驻）
    float xi = (i < N) ? points[i*3]   : 0.0f;
    float yi = (i < N) ? points[i*3+1] : 0.0f;
    float zi = (i < N) ? points[i*3+2] : 0.0f;

    float min_dist = FLT_MAX;
    int min_idx = -1;

    // 遍历所有 reference tile
    for (int tile_base = 0; tile_base < N; tile_base += TILE_SIZE) {
        // ① 协作加载：每 thread 载 1 个 reference 点到 shared
        int ref_idx = tile_base + tid;
        if (ref_idx < N) {
            shared_pts[tid][0] = points[ref_idx*3];
            shared_pts[tid][1] = points[ref_idx*3+1];
            shared_pts[tid][2] = points[ref_idx*3+2];
        }
        __syncthreads();   // ② 等待 shared 写入完成

        // ③ 每 thread 用自己 query 对 tile 内 256 点算距离 + 更新 argmin
        if (i < N) {
            int tile_end = min(tile_base + TILE_SIZE, N);
            for (int k = tile_base; k < tile_end; ++k) {
                if (k == i) continue;   // 跳过自身
                float dx = xi - shared_pts[k - tile_base][0];
                float dy = yi - shared_pts[k - tile_base][1];
                float dz = zi - shared_pts[k - tile_base][2];
                float d = dx*dx + dy*dy + dz*dz;
                if (d < min_dist) {
                    min_dist = d;
                    min_idx = k;
                }
            }
        }
        __syncthreads();   // ④ 等待计算完成，再加载下个 tile
    }

    if (i < N)
        indices[i] = min_idx;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000;
    size_t bytes_pts = (size_t)N * 3 * sizeof(float);
    size_t bytes_idx = (size_t)N * sizeof(int);
    printf("N = %d  (points = %.1f KB)\n", N, bytes_pts / 1e3);

    // ---- host ----
    float* hPts = (float*)malloc(bytes_pts);
    srand(42);
    for (int i = 0; i < N * 3; ++i)
        hPts[i] = (rand() % 2000) / 100.0f - 10.0f;   // [-10, 10]

    // ---- device ----
    float* dPts;
    int* dIdx;
    CHECK_CUDA(cudaMalloc(&dPts, bytes_pts));
    CHECK_CUDA(cudaMalloc(&dIdx, bytes_idx));
    CHECK_CUDA(cudaMemcpy(dPts, hPts, bytes_pts, cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- CPU 验证 ----
    int* hRef = (int*)malloc(bytes_idx);
    for (int i = 0; i < N; ++i) {
        float xi = hPts[i*3], yi = hPts[i*3+1], zi = hPts[i*3+2];
        float best = FLT_MAX; int bj = -1;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            float dx = xi - hPts[j*3], dy = yi - hPts[j*3+1], dz = zi - hPts[j*3+2];
            float d = dx*dx + dy*dy + dz*dz;
            if (d < best) { best = d; bj = j; }
        }
        hRef[i] = bj;
    }

    // ---- 朴素版 ----
    cudaEventRecord(t0);
    nn_naive<<<blocks, BLOCK_SIZE>>>(dPts, dIdx, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, t0, t1);

    // ---- tiled 版 ----
    cudaEventRecord(t0);
    nn_tiled<<<blocks, BLOCK_SIZE>>>(dPts, dIdx, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_tiled = 0.0f;
    cudaEventElapsedTime(&ms_tiled, t0, t1);

    // ---- 验证 ----
    int* hIdx = (int*)malloc(bytes_idx);
    CHECK_CUDA(cudaMemcpy(hIdx, dIdx, bytes_idx, cudaMemcpyDeviceToHost));
    int mism = 0;
    for (int i = 0; i < N; ++i)
        if (hIdx[i] != hRef[i]) ++mism;
    printf("[naive] time: %.3f ms\n", ms_naive);
    printf("[tiled ] time: %.3f ms  speedup: %.2fx  mismatch: %d  %s\n",
           ms_tiled, ms_naive / ms_tiled, mism, mism == 0 ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dPts));
    CHECK_CUDA(cudaFree(dIdx));
    free(hPts); free(hIdx); free(hRef);
    return 0;
}

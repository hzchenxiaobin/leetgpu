// 73-all-pairs-shortest-paths.cu —— Floyd-Warshall 全源最短路：朴素 vs shared tiling（min-plus 半环）
// 编译命令: nvcc -O3 -arch=sm_80 73-all-pairs-shortest-paths.cu -o apsp
// 运行:     ./apsp 512

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define INF_C 1e9f   // 内部测试用的"无穷大"（1e9+1e9 不溢出 float）

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// ---- 朴素 FW：每 k 一次 launch，每 thread 一个 (i,j)，直接读 global ----
__global__ void fw_naive_kernel(float* d, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float dik = d[i * N + k];     // 第 k 列
        float dkj = d[k * N + j];     // 第 k 行
        float nd  = dik + dkj;
        float dij = d[i * N + j];
        if (nd < dij) d[i * N + j] = nd;
    }
}

// ---- shared tiling FW：每 block 处理 BM×BN tile，缓存第 k 列/行切片 ----
__global__ void fw_tiled_kernel(float* d, int N, int k) {
    __shared__ float dik_s[BM];   // 第 k 列在本 tile 行范围内的 BM 个值
    __shared__ float dkj_s[BN];   // 第 k 行在本 tile 列范围内的 BN 个值

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,   by = blockIdx.y;
    int i = by * BM + ty;          // 全局行
    int j = bx * BN + tx;          // 全局列

    // 协作载入第 k 列（本 tile 的 BM 个行）：tx 当 loader 索引
    if (tx < BM) {
        int ii = by * BM + tx;
        dik_s[tx] = (ii < N) ? d[ii * N + k] : INF_C;
    }
    // 协作载入第 k 行（本 tile 的 BN 个列）：ty 当 loader 索引
    if (ty < BN) {
        int jj = bx * BN + ty;
        dkj_s[ty] = (jj < N) ? d[k * N + jj] : INF_C;
    }
    __syncthreads();

    if (i < N && j < N) {
        float dij = d[i * N + j];
        float nd  = dik_s[ty] + dkj_s[tx];   // d[i][k] + d[k][j]，均来自 shared
        if (nd < dij) d[i * N + j] = nd;
    }
}

void apsp_naive(float* d_d, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    for (int k = 0; k < N; ++k)
        fw_naive_kernel<<<grid, block>>>(d_d, N, k);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void apsp_tiled(float* d_d, int N) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    for (int k = 0; k < N; ++k)
        fw_tiled_kernel<<<grid, block>>>(d_d, N, k);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- CPU 参考（标准 Floyd-Warshall，与平台 reference_impl 等价） ----
void apsp_cpu(float* d, int N) {
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i) {
            float dik = d[i * N + k];
            for (int j = 0; j < N; ++j) {
                float nd = dik + d[k * N + j];
                if (nd < d[i * N + j]) d[i * N + j] = nd;
            }
        }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    if (N < 1) N = 1;
    printf("N=%d\n", N);

    size_t bf = (size_t)N * N * sizeof(float);
    float* h_in = (float*)malloc(bf);
    srand(42);
    // 随机有向图：对角 0，约 30% 边权 1..100，其余 INF
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j)            h_in[i * N + j] = 0.0f;
            else if (rand() % 100 < 30) h_in[i * N + j] = (float)(rand() % 100 + 1);
            else                  h_in[i * N + j] = INF_C;
        }

    float *d_n, *d_t;
    CHECK_CUDA(cudaMalloc(&d_n, bf));
    CHECK_CUDA(cudaMalloc(&d_t, bf));
    CHECK_CUDA(cudaMemcpy(d_n, h_in, bf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_t, h_in, bf, cudaMemcpyHostToDevice));

    // CPU 参考
    float* h_ref = (float*)malloc(bf);
    memcpy(h_ref, h_in, bf);
    apsp_cpu(h_ref, N);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    apsp_naive(d_n, N);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    cudaEventRecord(t0);
    apsp_tiled(d_t, N);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_tiled = 0; cudaEventElapsedTime(&ms_tiled, t0, t1);

    // 验证
    float* h_n = (float*)malloc(bf);
    float* h_t = (float*)malloc(bf);
    CHECK_CUDA(cudaMemcpy(h_n, d_n, bf, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_t, d_t, bf, cudaMemcpyDeviceToHost));

    float err_n = 0, err_t = 0;
    for (size_t x = 0; x < (size_t)N * N; ++x) {
        err_n = fmaxf(err_n, fabsf(h_n[x] - h_ref[x]));
        err_t = fmaxf(err_t, fabsf(h_t[x] - h_ref[x]));
    }
    printf("[naive] time: %.3f ms  max err: %.2e\n", ms_naive, err_n);
    printf("[tiled] time: %.3f ms  max err: %.2e  speedup: %.2fx\n",
           ms_tiled, err_t, ms_naive / ms_tiled);
    printf("%s\n", (err_n < 1e-3f && err_t < 1e-3f) ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_n)); CHECK_CUDA(cudaFree(d_t));
    free(h_in); free(h_ref); free(h_n); free(h_t);
    return 0;
}

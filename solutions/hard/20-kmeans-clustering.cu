// 20-kmeans-clustering.cu —— Lloyd K-Means：assign(shared 中心) + accum(atomic 归约) + finalize(均值/空簇)
// 编译命令: nvcc -O3 -arch=sm_80 20-kmeans-clustering.cu -o kmeans
// 运行:     ./kmeans 10000 5 30

#include <cstdio>
#include <cstdlib>
#include <cfloat>
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

// ---- assign：每点一 thread，k 个中心载入 shared，argmin（严格 <） ----
__global__ void assign_kernel(const float* dx, const float* dy,
                              const float* cx, const float* cy,
                              int* labels, int N, int k) {
    extern __shared__ float smem[];
    float* sx = smem;          // k 个中心 x
    float* sy = smem + k;      // k 个中心 y
    int tid = threadIdx.x;
    for (int j = tid; j < k; j += blockDim.x) {
        sx[j] = cx[j];
        sy[j] = cy[j];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;
    float px = dx[i], py = dy[i];
    float best = FLT_MAX;
    int best_j = 0;
    for (int j = 0; j < k; ++j) {
        float ddx = px - sx[j];
        float ddy = py - sy[j];
        float d = ddx * ddx + ddy * ddy;
        if (d < best) { best = d; best_j = j; }   // 严格 <：并列取小索引
    }
    labels[i] = best_j;
}

// ---- 朴素 update：每中心一 thread 串行扫全部点（用于对比） ----
__global__ void update_naive(const float* dx, const float* dy, const int* labels,
                             float* fx, float* fy, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    float sx = 0, sy = 0; int c = 0;
    for (int i = 0; i < N; ++i)
        if (labels[i] == j) { sx += dx[i]; sy += dy[i]; ++c; }
    if (c > 0) { fx[j] = sx / c; fy[j] = sy / c; }
}

// ---- accum：每点一 thread，atomicAdd 到 k 个 bin ----
__global__ void accum_kernel(const float* dx, const float* dy, const int* labels,
                             float* sum_x, float* sum_y, int* count, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = labels[i];
    atomicAdd(&sum_x[j], dx[i]);
    atomicAdd(&sum_y[j], dy[i]);
    atomicAdd(&count[j], 1);
}

// ---- finalize：每中心一 thread，count>0 取均值，空簇保留旧值 ----
__global__ void finalize_kernel(float* fx, float* fy,
                                const float* sum_x, const float* sum_y,
                                const int* count, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    int c = count[j];
    if (c > 0) {
        fx[j] = sum_x[j] / c;
        fy[j] = sum_y[j] / c;
    }
    // count==0：空簇，不写，保留上一轮 fx[j]/fy[j]
}

// 朴素版 K-Means（assign + update_naive）
void kmeans_naive(const float* d_dx, const float* d_dy, int* d_labels,
                  const float* d_ix, const float* d_iy,
                  float* d_fx, float* d_fy, int N, int k, int max_iter) {
    CHECK_CUDA(cudaMemcpy(d_fx, d_ix, k * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_fy, d_iy, k * sizeof(float), cudaMemcpyDeviceToDevice));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int kblocks = (k + 31) / 32;
    for (int it = 0; it < max_iter; ++it) {
        assign_kernel<<<blocks, BLOCK_SIZE, 2 * k * sizeof(float)>>>(
            d_dx, d_dy, d_fx, d_fy, d_labels, N, k);
        update_naive<<<kblocks, 32>>>(d_dx, d_dy, d_labels, d_fx, d_fy, N, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 优化版 K-Means（assign + accum + finalize）
void kmeans_opt(const float* d_dx, const float* d_dy, int* d_labels,
                const float* d_ix, const float* d_iy,
                float* d_fx, float* d_fy, int N, int k, int max_iter,
                float* d_sum_x, float* d_sum_y, int* d_count) {
    CHECK_CUDA(cudaMemcpy(d_fx, d_ix, k * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_fy, d_iy, k * sizeof(float), cudaMemcpyDeviceToDevice));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int kblocks = (k + 31) / 32;
    size_t kf = k * sizeof(float), ki = k * sizeof(int);
    for (int it = 0; it < max_iter; ++it) {
        assign_kernel<<<blocks, BLOCK_SIZE, 2 * k * sizeof(float)>>>(
            d_dx, d_dy, d_fx, d_fy, d_labels, N, k);
        CHECK_CUDA(cudaMemsetAsync(d_sum_x, 0, kf));
        CHECK_CUDA(cudaMemsetAsync(d_sum_y, 0, kf));
        CHECK_CUDA(cudaMemsetAsync(d_count, 0, ki));
        accum_kernel<<<blocks, BLOCK_SIZE>>>(
            d_dx, d_dy, d_labels, d_sum_x, d_sum_y, d_count, N);
        finalize_kernel<<<kblocks, 32>>>(d_fx, d_fy, d_sum_x, d_sum_y, d_count, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- CPU 参考实现（与平台 reference_impl 等价） ----
void kmeans_cpu(const float* dx, const float* dy, int* labels,
                const float* ix, const float* iy,
                float* fx, float* fy, int N, int k, int max_iter) {
    for (int j = 0; j < k; ++j) { fx[j] = ix[j]; fy[j] = iy[j]; }
    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < N; ++i) {
            float best = FLT_MAX; int bj = 0;
            for (int j = 0; j < k; ++j) {
                float ddx = dx[i] - fx[j], ddy = dy[i] - fy[j];
                float d = ddx * ddx + ddy * ddy;
                if (d < best) { best = d; bj = j; }
            }
            labels[i] = bj;
        }
        for (int j = 0; j < k; ++j) {
            float sx = 0, sy = 0; int c = 0;
            for (int i = 0; i < N; ++i) if (labels[i] == j) { sx += dx[i]; sy += dy[i]; ++c; }
            if (c > 0) { fx[j] = sx / c; fy[j] = sy / c; }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000;
    int k = (argc > 2) ? atoi(argv[2]) : 5;
    int max_iter = (argc > 3) ? atoi(argv[3]) : 30;
    printf("N=%d  k=%d  max_iter=%d\n", N, k, max_iter);

    // ---- host：构造 k 个明显分离的高斯簇，保证标签无歧义 ----
    size_t bf = (size_t)N * sizeof(float);
    size_t bi = (size_t)N * sizeof(int);
    float* hdx = (float*)malloc(bf);
    float* hdy = (float*)malloc(bf);
    float* hix = (float*)malloc(k * sizeof(float));
    float* hiy = (float*)malloc(k * sizeof(float));
    srand(42);
    float centers[8] = {100, 300, 500, 700, 900, 150, 350, 550};  // x,y 各取前 k
    for (int j = 0; j < k; ++j) { hix[j] = centers[j]; hiy[j] = centers[(j + 4) % 8]; }
    for (int i = 0; i < N; ++i) {
        int c = i % k;
        hdx[i] = hix[c] + (rand() % 200 - 100) / 10.0f;   // 簇中心 ±10
        hdy[i] = hiy[c] + (rand() % 200 - 100) / 10.0f;
    }

    // ---- device ----
    float *d_dx, *d_dy, *d_ix, *d_iy, *d_fx, *d_fy, *d_sum_x, *d_sum_y;
    int *d_labels, *d_count;
    CHECK_CUDA(cudaMalloc(&d_dx, bf));   CHECK_CUDA(cudaMalloc(&d_dy, bf));
    CHECK_CUDA(cudaMalloc(&d_labels, bi));
    CHECK_CUDA(cudaMalloc(&d_ix, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_iy, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fx, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fy, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum_x, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum_y, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_count, k * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_dx, hdx, bf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dy, hdy, bf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ix, hix, k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iy, hiy, k * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ---- CPU 参考 ----
    int* h_ref_lab = (int*)malloc(bi);
    float* h_ref_fx = (float*)malloc(k * sizeof(float));
    float* h_ref_fy = (float*)malloc(k * sizeof(float));
    kmeans_cpu(hdx, hdy, h_ref_lab, hix, hiy, h_ref_fx, h_ref_fy, N, k, max_iter);

    // ---- 朴素版 ----
    cudaEventRecord(t0);
    kmeans_naive(d_dx, d_dy, d_labels, d_ix, d_iy, d_fx, d_fy, N, k, max_iter);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    // ---- 优化版 ----
    cudaEventRecord(t0);
    kmeans_opt(d_dx, d_dy, d_labels, d_ix, d_iy, d_fx, d_fy, N, k, max_iter,
               d_sum_x, d_sum_y, d_count);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_opt = 0; cudaEventElapsedTime(&ms_opt, t0, t1);

    // ---- 验证 ----
    int* h_lab = (int*)malloc(bi);
    float* h_fx = (float*)malloc(k * sizeof(float));
    float* h_fy = (float*)malloc(k * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_lab, d_labels, bi, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fx, d_fx, k * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fy, d_fy, k * sizeof(float), cudaMemcpyDeviceToHost));

    int lab_mism = 0;
    for (int i = 0; i < N; ++i) if (h_lab[i] != h_ref_lab[i]) ++lab_mism;
    float cmax = 0;
    for (int j = 0; j < k; ++j) {
        cmax = fmaxf(cmax, fabsf(h_fx[j] - h_ref_fx[j]));
        cmax = fmaxf(cmax, fabsf(h_fy[j] - h_ref_fy[j]));
    }
    printf("[naive] time: %.3f ms\n", ms_naive);
    printf("[opt  ] time: %.3f ms  speedup: %.2fx\n", ms_opt, ms_naive / ms_opt);
    printf("labels mismatch: %d  centroid max err: %.2e  %s\n",
           lab_mism, cmax, (lab_mism == 0 && cmax < 1e-2f) ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_dx)); CHECK_CUDA(cudaFree(d_dy));
    CHECK_CUDA(cudaFree(d_labels)); CHECK_CUDA(cudaFree(d_ix));
    CHECK_CUDA(cudaFree(d_iy)); CHECK_CUDA(cudaFree(d_fx)); CHECK_CUDA(cudaFree(d_fy));
    CHECK_CUDA(cudaFree(d_sum_x)); CHECK_CUDA(cudaFree(d_sum_y));
    CHECK_CUDA(cudaFree(d_count));
    free(hdx); free(hdy); free(hix); free(hiy);
    free(h_ref_lab); free(h_ref_fx); free(h_ref_fy);
    free(h_lab); free(h_fx); free(h_fy);
    return 0;
}

// 18-sparse-matrix-vector-multiplication.cu —— CSR SpMV（warp-per-row + shuffle 归约）
// 编译命令: nvcc -O3 -arch=sm_120 18-sparse-matrix-vector-multiplication.cu -o spmv
// 运行:     ./spmv 1000 10000 3500000

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// warp 内树形归约（__shfl_down_sync）
__inline__ __device__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- 第 0 步：稠密 → CSR 转换（两遍 kernel）----

// 0a. 统计每行非零元数 → row_count（后面 exclusive scan 成 row_ptr）
__global__ void count_nnz_per_row(const float* A, int* row_count, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int cnt = 0;
        const float* row_ptr_A = A + (size_t)row * N;
        for (int j = 0; j < N; ++j)
            cnt += (row_ptr_A[j] != 0.0f);
        row_count[row] = cnt;
    }
}

// 0b. 并行 exclusive scan（Hillis-Steele，M 较小时单 block 够用）
__global__ void exclusive_scan(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    if (tid < n) temp[tid] = data[tid];
    __syncthreads();
    // Hillis-Steele inclusive scan
    int step = 1;
    for (int offset = 1; offset < n; offset <<= 1) {
        int v = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += v;
        __syncthreads();
    }
    // 转成 exclusive：整体右移一位，首位补 0
    int excl = (tid == 0) ? 0 : temp[tid - 1];
    __syncthreads();
    if (tid < n) data[tid] = excl;
    // 末尾写入总数到 data[n]（若 n < blockDim.x）
    if (tid == 0 && n < blockDim.x) data[n] = temp[n - 1];
}

// 0c. 填充 col_idx / values（每线程一个非零元，原子分配位置）
__global__ void fill_csr(const float* A, const int* row_ptr,
                         int* col_idx, float* values, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int pos = row_ptr[row];   // 本行非零元在 values 中的起始位置
        const float* row_ptr_A = A + (size_t)row * N;
        for (int j = 0; j < N; ++j) {
            float v = row_ptr_A[j];
            if (v != 0.0f) {
                col_idx[pos] = j;
                values[pos] = v;
                ++pos;
            }
        }
    }
}

// ---- 第 1 步：朴素稠密 GEMV（对比基准）----
__global__ void gemv_dense(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        const float* row_ptr_A = A + (size_t)row * N;
        for (int j = 0; j < N; ++j)
            sum += row_ptr_A[j] * x[j];
        y[row] = sum;
    }
}

// ---- 第 2 步：warp-per-row CSR SpMV ----
__global__ void spmv_warp(const int* row_ptr, const int* col_idx,
                          const float* values, const float* x, float* y, int M) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id_global >= M) return;

    int row_start = row_ptr[warp_id_global];
    int row_end   = row_ptr[warp_id_global + 1];

    // ① lane 分割非零元：每 lane 处理 stride=32 的元素
    float sum = 0.0f;
    for (int k = row_start + lane; k < row_end; k += WARP_SIZE) {
        sum += values[k] * x[col_idx[k]];   // 间接 gather x
    }

    // ② warp 内 shuffle 归约到 lane 0
    sum = warp_reduce(sum);

    // ③ lane 0 写回该行结果
    if (lane == 0)
        y[warp_id_global] = sum;
}

// ---- host 主流程 ----
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1000;
    int N = (argc > 2) ? atoi(argv[2]) : 10000;
    int target_nnz = (argc > 3) ? atoi(argv[3]) : 3500000;
    size_t bytes_A = (size_t)M * N * sizeof(float);
    printf("M = %d, N = %d  (A = %.1f MB)\n", M, N, bytes_A / 1e6);

    // ---- host：生成稀疏矩阵 ----
    float* hA = (float*)calloc(M * N, sizeof(float));
    srand(42);
    int placed = 0;
    while (placed < target_nnz) {
        int idx = rand() % (M * N);
        if (hA[idx] == 0.0f) {
            hA[idx] = ((rand() % 2000) / 100.0f) - 10.0f;   // [-10, 10]
            ++placed;
        }
    }
    float* hx = (float*)malloc(N * sizeof(float));
    for (int j = 0; j < N; ++j) hx[j] = (rand() % 1000) / 100.0f;

    // ---- device ----
    float *dA, *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dA, bytes_A));
    CHECK_CUDA(cudaMalloc(&dx, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, M * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));

    int *d_row_ptr, *d_col_idx, *d_row_count;
    float* d_values;
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_row_count, (M + 2) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, target_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, target_nnz * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- CSR 转换 ----
    int blocks_row = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMemset(d_row_count, 0, (M + 2) * sizeof(int)));
    count_nnz_per_row<<<blocks_row, BLOCK_SIZE>>>(dA, d_row_count, M, N);
    // exclusive scan（需要 blockDim.x >= M+1，用 1024 threads 覆盖 M <= 1023）
    int scan_threads = 1024;
    int scan_shared = scan_threads * sizeof(int);
    exclusive_scan<<<1, scan_threads, scan_shared>>>(d_row_count, M + 1);
    CHECK_CUDA(cudaMemcpy(d_row_ptr, d_row_count, (M + 1) * sizeof(int),
                           cudaMemcpyDeviceToDevice));
    fill_csr<<<blocks_row, BLOCK_SIZE>>>(dA, d_row_ptr, d_col_idx, d_values, M, N);

    // ---- CPU 验证 ----
    float* hy_ref = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < M; ++i) {
        float s = 0.0f;
        for (int j = 0; j < N; ++j)
            s += hA[i * N + j] * hx[j];
        hy_ref[i] = s;
    }

    // ---- 朴素稠密 GEMV ----
    cudaEventRecord(t0);
    gemv_dense<<<blocks_row, BLOCK_SIZE>>>(dA, dx, dy, M, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_dense = 0.0f;
    cudaEventElapsedTime(&ms_dense, t0, t1);

    // ---- warp-per-row SpMV ----
    int spmv_blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaEventRecord(t0);
    spmv_warp<<<spmv_blocks, BLOCK_SIZE>>>(d_row_ptr, d_col_idx, d_values, dx, dy, M);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_spmv = 0.0f;
    cudaEventElapsedTime(&ms_spmv, t0, t1);

    // ---- 验证 ----
    float* hy = (float*)malloc(M * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hy, dy, M * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    for (int i = 0; i < M; ++i) {
        float d = fabsf(hy[i] - hy_ref[i]);
        float rel = d / fmaxf(1.0f, fabsf(hy_ref[i]));
        if (d > max_err) max_err = d;
        if (rel > max_rel_err) max_rel_err = rel;
    }
    printf("[dense GEMV] time: %.3f ms\n", ms_dense);
    printf("[CSR SpMV ]  time: %.3f ms  speedup: %.2fx  max_err: %.4e (rel %.4e)  %s\n",
           ms_spmv, ms_dense / ms_spmv, max_err, max_rel_err, max_rel_err < 1e-3 ? "PASS" : "FAIL");

    // 带宽估算（SpMV：读 values+col_idx + gather x）
    float bytes_read = (float)(target_nnz * (sizeof(int) + sizeof(float)) + target_nnz * sizeof(float));
    printf("read bandwidth (SpMV): %.1f GB/s\n", (bytes_read / 1e9) / (ms_spmv / 1e3));

    CHECK_CUDA(cudaFree(dA));  CHECK_CUDA(cudaFree(dx));  CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(d_row_ptr)); CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_values));  CHECK_CUDA(cudaFree(d_row_count));
    free(hA); free(hx); free(hy); free(hy_ref);
    return 0;
}

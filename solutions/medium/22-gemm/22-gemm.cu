// 22-gemm.cu —— FP16 GEMM with WMMA Tensor Cores
// C = alpha * (A @ B) + beta * C,  A: M×K, B: K×N, C: M×N (FP16)
// 编译: nvcc -O3 -arch=sm_120 -lcublas 22-gemm.cu -o gemm
// 运行: ./gemm 1024 1024 1024

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

    #define CHECK_CUBLAS(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t s = (call);                                                                                     \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                                              \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s);                                        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// ---- tiling 参数 ----
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
const int BM = 128, BN = 128, BK = 16;    // BK == WMMA_K
const int WARPS_M = 4, WARPS_N = 2;       // 8 warps / block
const int NUM_WARPS = WARPS_M * WARPS_N;  // 8
const int NUM_THREADS = NUM_WARPS * 32;   // 256
const int WARP_TILE_M = BM / WARPS_M;     // 32
const int WARP_TILE_N = BN / WARPS_N;     // 64
const int FRAGS_M = WARP_TILE_M / WMMA_M; // 2
const int FRAGS_N = WARP_TILE_N / WMMA_N; // 4
const int LOAD_A = BM * BK / NUM_THREADS; // 8 half / thread
const int LOAD_B = BK * BN / NUM_THREADS; // 8 half / thread

// 朴素版：每 thread 算一个 C[i][j]，仅用 CUDA Core，用于对照
__global__ void gemm_naive(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
        }
        float c_init = __half2float(C[i * N + j]);
        C[i * N + j] = __float2half(alpha * sum + beta * c_init);
    }
}

// WMMA Tensor Core GEMM：每 warp 算 FRAGS_M×FRAGS_N 个 16×16 输出
__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N,
                          int K, float alpha, float beta) {
    __shared__ half As[BM][BK];   // A 的 BM×BK 子块
    __shared__ half Bs[BK][BN];   // B 的 BK×BN 子块
    extern __shared__ float Cs[]; // BM×BN fp32 staging（epilogue 暂存累加器）

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;      // 0..3
    const int warp_n = warp_id % WARPS_N;      // 0..1
    const int warp_row = warp_m * WARP_TILE_M; // 本 warp 输出子块在 block tile 内的行起点
    const int warp_col = warp_n * WARP_TILE_N; // 列起点

    // fp32 累加器：FRAGS_M×FRAGS_N 个 16×16 fragment
    using AccFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    AccFrag acc[FRAGS_M][FRAGS_N];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // 沿 K 维滑动 BK=16 的 tile
    using AFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using BFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    for (int bk = 0; bk < K; bk += BK) {
// ---- ① 协作加载 As[BM][BK] ----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS; // 0..2047
            int r = lin / BK, c = lin % BK;
            int ar = by * BM + r, ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
        }
// ---- ② 协作加载 Bs[BK][BN] ----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN, c = lin % BN;
            int br = bk + r, bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
        }
        __syncthreads();

// ---- ③ 每 warp 做 FRAGS_M×FRAGS_N 次 mma（Tensor Core）----
        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                AFrag a_frag;
                BFrag b_frag;
                wmma::load_matrix_sync(a_frag, &As[warp_row + i * WMMA_M][0], BK);
                wmma::load_matrix_sync(b_frag, &Bs[0][warp_col + j * WMMA_N], BN);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads(); // tile 用完才能覆盖
    }

// ---- ④ epilogue：累加器存入 shared staging（fp32）----
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(&Cs[(warp_row + i * WMMA_M) * BN + (warp_col + j * WMMA_N)], acc[i][j], BN,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();

    // ---- ⑤ 写回 C：alpha*acc + beta*C_initial -> half ----
    // 256 threads 覆盖 128×128 = 16384 元素，每 thread 64 个
    const int total = BM * BN;
    #pragma unroll
    for (int i = 0; i < total / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int r = idx / BN, c = idx % BN;
        int gr = by * BM + r, gc = bx * BN + c;
        if (gr < M && gc < N) {
            float acc_val = Cs[idx];
            float c_init = (beta != 0.0f) ? __half2float(C[gr * N + gc]) : 0.0f;
            C[gr * N + gc] = __float2half(alpha * acc_val + beta * c_init);
        }
    }
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int dyn_smem = BM * BN * sizeof(float); // 64 KB staging
    static bool attr_set = false;
    if (!attr_set) {
        // staging 64KB + static 8KB > 默认 48KB，需放开 dynamic shared 上限
        cudaFuncSetAttribute(gemm_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
        attr_set = true;
    }
    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma<<<blocks, threads, dyn_smem>>>(A, B, C, M, N, K, alpha, beta);
}

// ---- 本地自测 / cuBLAS 对比 ----
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    size_t aB = (size_t)M * K * sizeof(half);
    size_t bB = (size_t)K * N * sizeof(half);
    size_t cB = (size_t)M * N * sizeof(half);
    double gflop = 2.0 * M * N * K / 1e9;
    printf("A:%dx%d B:%dx%d C:%dx%d  FLOPs=%.2f GFLOP\n", M, K, K, N, M, N, gflop);

    half *hA = (half*)malloc(aB), *hB = (half*)malloc(bB);
    half *hC = (half*)malloc(cB), *hOut = (half*)malloc(cB), *hRef = (half*)malloc(cB);
    srand(42);
    auto rh = [&]() { return __float2half((float)(rand() % 2000) / 1000.0f - 1.0f); };
    for (int i = 0; i < M * K; ++i)
        hA[i] = rh();
    for (int i = 0; i < K * N; ++i)
        hB[i] = rh();
    for (int i = 0; i < M * N; ++i)
        hC[i] = rh();
    float alpha = 1.0f, beta = 1.0f; // 与性能测试一致

    half *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- WMMA warmup + 计时 ----
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        solve(dA, dB, dC, M, N, K, alpha, beta);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_w = 0.0f;
    cudaEventElapsedTime(&ms_w, t0, t1);
    ms_w /= 10.0f;
    double tf_w = (2.0 * M * N * K / 1e12) / (ms_w / 1e3);
    // 单次干净运行取结果用于验证
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaMemcpy(hOut, dC, cB, cudaMemcpyDeviceToHost));

    // ---- cuBLAS 基线（行主序：C^T = B^T A^T，col-major）----
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F,
                                  K, &beta, dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_c = 0.0f;
    cudaEventElapsedTime(&ms_c, t0, t1);
    ms_c /= 10.0f;
    double tf_c = (2.0 * M * N * K / 1e12) / (ms_c / 1e3);
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaMemcpy(hRef, dC, cB, cudaMemcpyDeviceToHost));

    // ---- 验证（atol=rtol=0.05）----
    int err = 0;
    for (int i = 0; i < M * N && err < 5; ++i) {
        float ref = __half2float(hRef[i]), got = __half2float(hOut[i]);
        if (fabsf(got - ref) > 0.05f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int r = i / N, c = i % N;
            printf("MISMATCH @(%d,%d): got %f ref %f\n", r, c, got, ref);
        }
    }

    printf("\n[WMMA  ] %.3f ms  %.2f TFLOPS\n", ms_w, tf_w);
    printf("[cuBLAS] %.3f ms  %.2f TFLOPS\n", ms_c, tf_c);
    printf("[ratio ] %.1f%% of cuBLAS\n", 100.0 * tf_w / tf_c);
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    free(hOut);
    free(hRef);
    return err ? EXIT_FAILURE : 0;
}

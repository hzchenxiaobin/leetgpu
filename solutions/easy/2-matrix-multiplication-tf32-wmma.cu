// 2-matrix-multiplication-tf32-wmma.cu —— TF32 Tensor Core 矩阵乘法
// C = A × B,  A: M×K, B: K×N, C: M×N (FP32 in/out, TF32 compute)
// 编译: nvcc -O3 -arch=sm_80 2-matrix-multiplication-tf32-wmma.cu -o matmul_tc
// 运行: ./matmul_tc 8192 6144 4096

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// TF32 WMMA 参数
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 8;
const int BM = 128, BN = 128, BK = 16;
const int WARPS_M = 4, WARPS_N = 2;
const int NUM_WARPS = WARPS_M * WARPS_N;
const int NUM_THREADS = NUM_WARPS * 32;
const int WARP_TILE_M = BM / WARPS_M;
const int WARP_TILE_N = BN / WARPS_N;
const int FRAGS_M = WARP_TILE_M / WMMA_M;
const int FRAGS_N = WARP_TILE_N / WMMA_N;
const int LOAD_A = BM * BK / NUM_THREADS;
const int LOAD_B = BK * BN / NUM_THREADS;

__global__ void matmul_tf32_wmma(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    extern __shared__ float Cs[]; // BM×BN fp32 staging

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    using AccFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    AccFrag acc[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i)
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    using AFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_tf32, wmma::row_major>;
    using BFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_tf32, wmma::row_major>;

    int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        // ---- ① 协作加载 As[BM][BK] / Bs[BK][BN]（float，越界补 0）----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK, c = lin % BK;
            int ar = by * BM + r, ac = t * BK + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN, c = lin % BN;
            int br = t * BK + r, bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
        }
        __syncthreads();

        // ---- ② TF32 mma：BK/WMMA_K = 2 个子步，每步 8 个 fragment ----
        #pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            #pragma unroll
            for (int i = 0; i < FRAGS_M; ++i) {
                #pragma unroll
                for (int j = 0; j < FRAGS_N; ++j) {
                    AFrag a_frag;
                    BFrag b_frag;
                    wmma::load_matrix_sync(a_frag, &As[warp_row + i * WMMA_M][kk], BK);
                    wmma::load_matrix_sync(b_frag, &Bs[kk][warp_col + j * WMMA_N], BN);
                    wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // ---- ③ epilogue：累加器存入 shared staging，再写回 global C ----
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(&Cs[(warp_row + i * WMMA_M) * BN + warp_col + j * WMMA_N],
                                    acc[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    const int total = BM * BN;
    #pragma unroll
    for (int i = 0; i < total / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int r = idx / BN, c = idx % BN;
        int gr = by * BM + r, gc = bx * BN + c;
        if (gr < M && gc < N)
            C[gr * N + gc] = Cs[idx];
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 6144;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    size_t aB = (size_t)M * K * sizeof(float);
    size_t bB = (size_t)K * N * sizeof(float);
    size_t cB = (size_t)M * N * sizeof(float);
    printf("A:%dx%d B:%dx%d C:%dx%d  FLOPs=%.2f GFLOP\n", M, K, K, N, M, N, 2.0 * M * N * K / 1e9);

    float *hA = (float*)malloc(aB), *hB = (float*)malloc(bB), *hC = (float*)malloc(cB);
    srand(42);
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand() % 1000) / 100.0f;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    const int dyn_smem = BM * BN * sizeof(float);
    cudaFuncSetAttribute(matmul_tf32_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);

    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    printf("launch: blocks=(%d,%d) threads=%d  BM=%d BN=%d BK=%d WMMA=%dx%dx%d\n",
           blocks.x, blocks.y, NUM_THREADS, BM, BN, BK, WMMA_M, WMMA_N, WMMA_K);

    // warmup
    matmul_tf32_wmma<<<blocks, threads, dyn_smem>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_tf32_wmma<<<blocks, threads, dyn_smem>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    double tflops = (2.0 * M * N * K / 1e12) / (ms / 1e3);
    printf("kernel time: %.3f ms\nperformance: %.2f TFLOPS\n", ms, tflops);

    CHECK_CUDA(cudaMemcpy(hC, dC, cB, cudaMemcpyDeviceToHost));
    int err = 0;
    int checks[] = {0, N - 1, (M / 2) * N + N / 2, (M - 1) * N + N - 1};
    for (int idx : checks) {
        int i = idx / N, j = idx % N;
        float ref = 0.0f;
        for (int k = 0; k < K; ++k)
            ref += hA[i * K + k] * hB[k * N + j];
        if (fabsf(hC[idx] - ref) > 1e-4f * fmaxf(1.0f, fabsf(ref))) {
            if (++err <= 5)
                printf("MISMATCH @(%d,%d): got %f, expect %f, err %.2e\n", i, j, hC[idx], ref,
                       fabsf(hC[idx] - ref));
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}

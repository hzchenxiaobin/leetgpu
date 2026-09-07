// 57-fp16-batched-matmul.cu —— FP16 Batched MatMul with WMMA Tensor Cores
// C[b] = A[b] @ B[b], A: [BATCH, M, K], B: [BATCH, K, N], C: [BATCH, M, N] (FP16)
// FP32 累加 via WMMA accumulator fragment, batch 维用 blockIdx.z
// 编译命令: nvcc -O3 -arch=sm_120 57-fp16-batched-matmul.cu -o fp16_bmm_wmma
// 运行:     ./fp16_bmm_wmma

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call) \
do {                                                                                                          \
    cudaError_t e = (call);                                                                                   \
    if (e != cudaSuccess) {                                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
        exit(EXIT_FAILURE);                                                                                   \
    }                                                                                                         \
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

// WMMA Tensor Core batched GEMM：每 warp 算 FRAGS_M×FRAGS_N 个 16×16 输出
// A: [BATCH, M, K] half, B: [BATCH, K, N] half, C: [BATCH, M, N] half
// grid = (ceil(N/BN), ceil(M/BM), BATCH), blockIdx.z = batch 索引
__global__ void fp16_bmm_wmma_kernel(const half* __restrict__ A,
                                      const half* __restrict__ B,
                                      half* __restrict__ C,
                                      int BATCH, int M, int N, int K) {
    __shared__ half As[BM][BK];   // A 的 BM×BK 子块
    __shared__ half Bs[BK][BN];   // B 的 BK×BN 子块
    extern __shared__ float Cs[]; // BM×BN fp32 staging（epilogue 暂存累加器）

    const int b = blockIdx.z;          // batch 索引
    const int bx = blockIdx.x;         // N 维 block 坐标
    const int by = blockIdx.y;         // M 维 block 坐标
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;      // 0..3
    const int warp_n = warp_id % WARPS_N;      // 0..1
    const int warp_row = warp_m * WARP_TILE_M; // 本 warp 输出子块在 block tile 内的行起点
    const int warp_col = warp_n * WARP_TILE_N; // 列起点

    // batch 基址（每 batch 独立）
    const half* A_b = A + (size_t)b * M * K;
    const half* B_b = B + (size_t)b * K * N;
    half* C_b = C + (size_t)b * M * N;

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

    using AFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using BFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;

    // 沿 K 维滑动 BK=16 的 tile
    for (int bk = 0; bk < K; bk += BK) {
        // ---- ① 协作加载 As[BM][BK] ----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK, c = lin % BK;
            int ar = by * BM + r, ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A_b[ar * K + ac] : __float2half(0.0f);
        }
        // ---- ② 协作加载 Bs[BK][BN] ----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN, c = lin % BN;
            int br = bk + r, bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B_b[br * N + bc] : __float2half(0.0f);
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
            wmma::store_matrix_sync(&Cs[(warp_row + i * WMMA_M) * BN + (warp_col + j * WMMA_N)],
                                    acc[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    // ---- ⑤ 写回 C：fp32 -> half ----
    // 256 threads 覆盖 128×128 = 16384 元素，每 thread 64 个
    const int total = BM * BN;
    #pragma unroll
    for (int i = 0; i < total / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int r = idx / BN, c = idx % BN;
        int gr = by * BM + r, gc = bx * BN + c;
        if (gr < M && gc < N) {
            C_b[gr * N + gc] = __float2half(Cs[idx]);
        }
    }
}

// ---- CPU 参考 ----
void bmm_cpu(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    for (int b = 0; b < BATCH; b++)
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++)
                    acc += __half2float(A[b*M*K + m*K + k])
                         * __half2float(B[b*K*N + k*N + n]);
                C[b*M*N + m*N + n] = __float2half(acc);
            }
}

int main() {
    // 题目 example
    int BATCH = 2, M = 2, K = 3, N = 2;
    printf("FP16 Batched MatMul (WMMA): B=%d M=%d N=%d K=%d\n", BATCH, M, N, K);

    size_t a_size = (size_t)BATCH * M * K;
    size_t b_size = (size_t)BATCH * K * N;
    size_t c_size = (size_t)BATCH * M * N;

    // host 数据
    half hA[] = {__float2half(1),__float2half(2),__float2half(3),
                 __float2half(4),__float2half(5),__float2half(6),
                 __float2half(7),__float2half(8),__float2half(9),
                 __float2half(10),__float2half(11),__float2half(12)};
    half hB[] = {__float2half(1),__float2half(2),
                 __float2half(3),__float2half(4),
                 __float2half(5),__float2half(6),
                 __float2half(6),__float2half(5),
                 __float2half(4),__float2half(3),
                 __float2half(2),__float2half(1)};
    half hC[8], hRef[8];

    // device
    half *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, a_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, b_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, c_size * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(dA, hA, a_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, b_size * sizeof(half), cudaMemcpyHostToDevice));

    // 启动
    const int dyn_smem = BM * BN * sizeof(float); // 64 KB staging
    CHECK_CUDA(cudaFuncSetAttribute(fp16_bmm_wmma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem));

    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    fp16_bmm_wmma_kernel<<<blocks, threads, dyn_smem>>>(dA, dB, dC, BATCH, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 验证
    CHECK_CUDA(cudaMemcpy(hC, dC, c_size * sizeof(half), cudaMemcpyDeviceToHost));
    bmm_cpu(hA, hB, hRef, BATCH, M, N, K);
    int err = 0;
    for (int i = 0; i < (int)c_size && err < 5; i++) {
        float got = __half2float(hC[i]);
        float exp = __half2float(hRef[i]);
        if (fabsf(got - exp) > 0.05f * fmaxf(1.0f, fabsf(exp))) {
            ++err;
            printf("MISMATCH @%d: got %.4f, expect %.4f\n", i, got, exp);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");
    for (int b = 0; b < BATCH; b++) {
        printf("C[%d] = [[%.1f, %.1f], [%.1f, %.1f]]\n", b,
               __half2float(hC[b*4]), __half2float(hC[b*4+1]),
               __half2float(hC[b*4+2]), __half2float(hC[b*4+3]));
    }

    // ---- 性能测试 ----
    printf("\n--- Perf test (B=32, M=N=K=256) ---\n");
    BATCH=32; M=256; N=256; K=256;
    a_size = (size_t)BATCH * M * K;
    b_size = (size_t)BATCH * K * N;
    c_size = (size_t)BATCH * M * N;
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaMalloc(&dA, a_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, b_size * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, c_size * sizeof(half)));

    // 随机初始化
    half* hA2 = (half*)malloc(a_size * sizeof(half));
    half* hB2 = (half*)malloc(b_size * sizeof(half));
    srand(42);
    for (size_t i = 0; i < a_size; i++) hA2[i] = __float2half((float)(rand() % 2000) / 1000.0f - 1.0f);
    for (size_t i = 0; i < b_size; i++) hB2[i] = __float2half((float)(rand() % 2000) / 1000.0f - 1.0f);
    CHECK_CUDA(cudaMemcpy(dA, hA2, a_size * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB2, b_size * sizeof(half), cudaMemcpyHostToDevice));

    dim3 blocks2((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // warmup
    fp16_bmm_wmma_kernel<<<blocks2, threads, dyn_smem>>>(dA, dB, dC, BATCH, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        fp16_bmm_wmma_kernel<<<blocks2, threads, dyn_smem>>>(dA, dB, dC, BATCH, M, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;

    // TFLOPS 估算
    double flops = 2.0 * BATCH * M * N * K;  // 每元素 K 次 mul + K-1 次 add ≈ 2K
    printf("kernel time: %.3f ms\n", ms);
    printf("compute: %.2f GFLOP, %.2f TFLOPS\n", flops / 1e9, flops / 1e9 / (ms / 1e3));

    free(hA2); free(hB2);
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    return 0;
}

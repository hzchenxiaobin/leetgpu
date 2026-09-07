// 2-matrix-multiplication.cu —— register tiling 矩阵乘法
// 编译命令: nvcc -O3 -arch=sm_120 2-matrix-multiplication.cu -o matmul
// 运行:     ./matmul 8192 6144 4096

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

// register tiling 参数：block 负责 64×64 输出，每 thread 算 4×4 = 16 个元素
const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;               // 16
const int BLOCK_N = BN / TN;               // 16
const int NUM_THREADS = BLOCK_M * BLOCK_N; // 256

// register tiling：每 thread 算 TM×TN 个 C 元素
__global__ void matmul_register_tiled(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int N, int K) {
    // shared memory：A 的 BM×BK 子块 + B 的 BK×BN 子块
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;   // K 维（列方向）
    int by = blockIdx.y;   // M 维（行方向）
    int tid = threadIdx.x; // 0..255
    int tx = tid % BLOCK_N; // 0..15，thread 在 block tile 内的列坐标
    int ty = tid / BLOCK_N; // 0..15，thread 在 block tile 内的行坐标

    // 寄存器累加器：TM×TN 个输出，常驻寄存器不落盘
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int LOAD_A = BM * BK / NUM_THREADS; // 4，每 thread 加载 4 个 A 元素
    const int LOAD_B = BK * BN / NUM_THREADS; // 4，每 thread 加载 4 个 B 元素

    // 沿 N 维滑动 BK=16 的 tile
    int num_tiles = (N + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        // ---- ① 协作加载 As[BM][BK] ----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = t * BK + c;
            As[r][c] = (ar < M && ac < N) ? A[ar * N + ac] : 0.0f;
        }
        // ---- ② 协作加载 Bs[BK][BN] ----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = t * BK + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < N && bc < K) ? B[br * K + bc] : 0.0f;
        }
        __syncthreads();

        // ---- ③ register tiling：每 thread 算 TM×TN 个输出 ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads(); // tile 用完才能覆盖
    }

    // ---- ④ 写回 C ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < M && gc < K) {
                C[gr * K + gc] = acc[i][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 6144;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    size_t a_bytes = (size_t)M * N * sizeof(float);
    size_t b_bytes = (size_t)N * K * sizeof(float);
    size_t c_bytes = (size_t)M * K * sizeof(float);
    printf("A: %dx%d, B: %dx%d, C: %dx%d\n", M, N, N, K, M, K);
    printf("FLOPs: %.2f GFLOP\n", 2.0 * M * N * K / 1e9);

    // ---- host ----
    float* hA = (float*)malloc(a_bytes);
    float* hB = (float*)malloc(b_bytes);
    float* hC = (float*)malloc(c_bytes);
    srand(42);
    for (int i = 0; i < M * N; ++i)
        hA[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < N * K; ++i)
        hB[i] = (float)(rand() % 1000) / 100.0f;

    // ---- device ----
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, a_bytes));
    CHECK_CUDA(cudaMalloc(&dB, b_bytes));
    CHECK_CUDA(cudaMalloc(&dC, c_bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, b_bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    dim3 threads(NUM_THREADS);
    dim3 blocks((K + BN - 1) / BN, (M + BM - 1) / BM);
    printf("launch: blocks=(%d,%d) threads=%d  BM=%d BN=%d BK=%d TM=%d TN=%d\n",
           blocks.x, blocks.y, NUM_THREADS, BM, BN, BK, TM, TN);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_register_tiled<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- TFLOPS ----
    double tflops = (2.0 * M * N * K / 1e12) / (ms / 1e3);
    printf("performance: %.2f TFLOPS\n", tflops);

    // ---- 验证（抽检角落 + 随机点）----
    CHECK_CUDA(cudaMemcpy(hC, dC, c_bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    int checks[] = {0, K - 1, (M / 2) * K + K / 2, (M - 1) * K + K - 1};
    for (int idx : checks) {
        int i = idx / K, j = idx % K;
        float ref = 0.0f;
        for (int k = 0; k < N; ++k)
            ref += hA[i * N + k] * hB[k * K + j];
        if (fabsf(hC[idx] - ref) > 1e-3f * fmaxf(1.0f, fabsf(ref))) {
            if (++err <= 5)
                printf("MISMATCH @(%d,%d): got %f, expect %f\n", i, j, hC[idx], ref);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    return 0;
}

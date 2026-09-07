// 22-gemm-cuda-core.cu —— FP16 GEMM，CUDA Core 优化版（无 Tensor Core）
// C = alpha * (A @ B) + beta * C,  A: M×K, B: K×N, C: M×N (FP16)
// 编译: nvcc -O3 -arch=sm_120 22-gemm-cuda-core.cu -o gemm_core
// 运行: ./gemm_core 1024 1024 1024

    #include <cuda_fp16.h>
    #include <cuda_runtime.h>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// CUDA Core 分块参数：block 负责 64×64 输出，每个 thread 算 4×4 = 16 个元素
const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;               // 16
const int BLOCK_N = BN / TN;               // 16
const int NUM_THREADS = BLOCK_M * BLOCK_N; // 256

__global__ void gemm_cuda_core(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M,
                               int N, int K, float alpha, float beta) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int tx = tid % BLOCK_N;
    int ty = tid / BLOCK_N;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int LOAD_A = BM * BK / NUM_THREADS; // 4
    const int LOAD_B = BK * BN / NUM_THREADS; // 4

    // 沿 K 维滑动 BK=16 的 tile
    for (int bk = 0; bk < K; bk += BK) {
// ---- ① 协作加载 As[BM][BK]（half -> float）----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? __half2float(A[ar * K + ac]) : 0.0f;
        }
// ---- ② 协作加载 Bs[BK][BN]（half -> float）----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = bk + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? __half2float(B[br * N + bc]) : 0.0f;
        }
        __syncthreads();

// ---- ③ Register Blocking：每个 thread 算 TM×TN 个输出 ----
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

// ---- ④ epilogue：alpha*acc + beta*C_initial -> half ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < M && gc < N) {
                float c_init = (beta != 0.0f) ? __half2float(C[gr * N + gc]) : 0.0f;
                C[gr * N + gc] = __float2half(alpha * acc[i][j] + beta * c_init);
            }
        }
    }
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_cuda_core<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
}

// ---- CPU 参考 ----
void cpu_gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            float c_init = __half2float(C[i * N + j]);
            C[i * N + j] = __float2half(alpha * sum + beta * c_init);
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    size_t aB = (size_t)M * K * sizeof(half);
    size_t bB = (size_t)K * N * sizeof(half);
    size_t cB = (size_t)M * N * sizeof(half);

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
    float alpha = 1.0f, beta = 1.0f;

    half *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    // GPU
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaMemcpy(hOut, dC, cB, cudaMemcpyDeviceToHost));

    // CPU
    memcpy(hRef, hC, cB);
    cpu_gemm(hA, hB, hRef, M, N, K, alpha, beta);

    int err = 0;
    for (int i = 0; i < M * N && err < 5; ++i) {
        float ref = __half2float(hRef[i]), got = __half2float(hOut[i]);
        if (fabsf(got - ref) > 0.05f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int r = i / N, c = i % N;
            printf("MISMATCH @(%d,%d): got %f ref %f\n", r, c, got, ref);
        }
    }
    printf("CUDA Core GEMM M=%d N=%d K=%d: %s\n", M, N, K, err ? "FAIL" : "PASS");

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

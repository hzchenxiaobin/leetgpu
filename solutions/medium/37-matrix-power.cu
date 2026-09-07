// 37-matrix-power.cu —— 矩阵幂 A^P，tiled GEMM + binary exponentiation
// 编译: nvcc -O3 -arch=sm_120 37-matrix-power.cu -o matrix_power
// 运行: ./matrix_power 512 3

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call) \
do {                                                                                   \
    cudaError_t e = (call);                                                            \
    if (e != cudaSuccess) {                                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                                            \
    }                                                                                  \
} while (0)

// ---- tiling 参数 ----
const int BM = 32, BN = 32, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;               // 8
const int BLOCK_N = BN / TN;               // 8
const int NUM_THREADS = BLOCK_M * BLOCK_N; // 64

// ---- Tiled GEMM: C = A × B (N×N), register blocking ----
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x;
    int tx = tid % BLOCK_N;  // 0..7
    int ty = tid / BLOCK_N;  // 0..7

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int LOAD_A = BM * BK / NUM_THREADS; // 8
    const int LOAD_B = BK * BN / NUM_THREADS; // 8

    for (int bk = 0; bk < N; bk += BK) {
        // ① 协作加载 As[BM][BK]
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK, c = lin % BK;
            int ar = by * BM + r, ac = bk + c;
            As[r][c] = (ar < N && ac < N) ? A[ar * N + ac] : 0.0f;
        }
        // ② 协作加载 Bs[BK][BN]
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN, c = lin % BN;
            int br = bk + r, bc = bx * BN + c;
            Bs[r][c] = (br < N && bc < N) ? B[br * N + bc] : 0.0f;
        }
        __syncthreads();

        // ③ Register Blocking：每 thread 算 TM×TN 个输出
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
        __syncthreads();
    }

    // ④ 写回 C
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < N && gc < N)
                C[gr * N + gc] = acc[i][j];
        }
    }
}

// ---- 单位矩阵 kernel ----
__global__ void identity_kernel(float* mat, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int r = idx / N, c = idx % N;
        mat[idx] = (r == c) ? 1.0f : 0.0f;
    }
}

// ---- 拷贝 kernel ----
__global__ void copy_kernel(float* dst, const float* src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N)
        dst[idx] = src[idx];
}

// ---- matmul 封装 ----
void launch_matmul(const float* dA, const float* dB, float* dC, int N) {
    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (N + BM - 1) / BM);
    matmul_kernel<<<blocks, threads>>>(dA, dB, dC, N);
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const float* input, float* output, int N, int P) {
    if (P == 1) {
        // 直接拷贝 input → output
        int threads = 256;
        int blocks = (N * N + threads - 1) / threads;
        copy_kernel<<<blocks, threads>>>(output, input, N);
        cudaDeviceSynchronize();
        return;
    }

    size_t mat_bytes = (size_t)N * N * sizeof(float);

    // 双缓冲：d_buf[0] 和 d_buf[1] 交替使用
    float* d_buf[2];
    CHECK_CUDA(cudaMalloc(&d_buf[0], mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_buf[1], mat_bytes));

    // result = I（单位阵），放在 d_buf[0]
    int threads = 256;
    int blocks = (N * N + threads - 1) / threads;
    identity_kernel<<<blocks, threads>>>(d_buf[0], N);

    // base = input，拷贝到 d_buf[1]
    copy_kernel<<<blocks, threads>>>(d_buf[1], input, N);

    // 二进制快速幂
    int src = 0;  // result 当前在 d_buf[src]
    int base = 1; // base 当前在 d_buf[base]
    int p = P;
    while (p > 0) {
        if (p & 1) {
            // result = result × base → 写入 d_buf[1 - src]（避免覆盖）
            // 需要 3 个缓冲：result(src), base(base), output(1-src)
            // 但只有 2 个 buf，所以复用：先算到 output，再拷回
            // 优化：用第三个临时缓冲
            float* d_tmp;
            CHECK_CUDA(cudaMalloc(&d_tmp, mat_bytes));
            launch_matmul(d_buf[src], d_buf[base], d_tmp, N);
            CHECK_CUDA(cudaDeviceSynchronize());
            copy_kernel<<<blocks, threads>>>(d_buf[src], d_tmp, N);
            CHECK_CUDA(cudaFree(d_tmp));
        }
        p >>= 1;
        if (p > 0) {
            // base = base × base → 写入临时，再拷回
            float* d_tmp;
            CHECK_CUDA(cudaMalloc(&d_tmp, mat_bytes));
            launch_matmul(d_buf[base], d_buf[base], d_tmp, N);
            CHECK_CUDA(cudaDeviceSynchronize());
            copy_kernel<<<blocks, threads>>>(d_buf[base], d_tmp, N);
            CHECK_CUDA(cudaFree(d_tmp));
        }
    }

    // 拷贝结果到 output
    copy_kernel<<<blocks, threads>>>(output, d_buf[src], N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_buf[0]));
    CHECK_CUDA(cudaFree(d_buf[1]));
}

// ---- CPU 参考 ----
void matrix_power_cpu(const float* A, float* C, int N, int P) {
    for (int i = 0; i < N * N; ++i) C[i] = A[i];
    float* tmp = (float*)malloc(N * N * sizeof(float));
    for (int p = 1; p < P; ++p) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k)
                    sum += C[i * N + k] * A[k * N + j];
                tmp[i * N + j] = sum;
            }
        memcpy(C, tmp, N * N * sizeof(float));
    }
    free(tmp);
}

// ---- 本地自测 ----
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    int P = (argc > 2) ? atoi(argv[2]) : 3;
    size_t bytes = (size_t)N * N * sizeof(float);
    double gflop = (P - 1) * 2.0 * N * N * N / 1e9;
    printf("N=%d P=%d  FLOPs=%.2f GFLOP (naive %d matmuls, binary exp ~%d matmuls)\n",
           N, P, gflop, P - 1, (int)(2 * log2(P + 1)));

    float *hA = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    float *hRef = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N * N; ++i)
        hA[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;

    float *dA, *dOut;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));

    // warmup
    solve(dA, dOut, N, P);

    // 计时
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        solve(dA, dOut, N, P);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= 10.0f;
    double tflops = gflop / (ms / 1e3) / 1e3;

    // 验证
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    matrix_power_cpu(hA, hRef, N, P);

    int err = 0;
    for (int i = 0; i < N * N && err < 5; ++i) {
        float ref = hRef[i], got = hOut[i];
        if (fabsf(got - ref) > 1e-3f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int r = i / N, c = i % N;
            printf("MISMATCH @(%d,%d): got %f ref %f\n", r, c, got, ref);
        }
    }

    printf("\n[tiled GEMM + binary exp] %.3f ms  %.2f TFLOPS\n", ms, tflops);
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dOut));
    free(hA);
    free(hOut);
    free(hRef);
    return err ? EXIT_FAILURE : 0;
}

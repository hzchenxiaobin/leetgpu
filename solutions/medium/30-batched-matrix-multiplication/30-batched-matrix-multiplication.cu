// 30-batched-matrix-multiplication.cu —— Batched Matrix Multiplication（batch 维 + register blocking）
// 编译命令: nvcc -O3 -arch=sm_120 30-batched-matrix-multiplication.cu -o batched_matmul
// 运行:     ./batched_matmul

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

// register blocking 分块参数：block 负责 64×64 输出，每个 thread 算 4×4 = 16 个元素
const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;               // 16
const int BLOCK_N = BN / TN;               // 16
const int NUM_THREADS = BLOCK_M * BLOCK_N; // 256
const int LOAD_A = BM * BK / NUM_THREADS;  // 4
const int LOAD_B = BK * BN / NUM_THREADS;  // 4

// batched matmul：grid((N+BN-1)/BN, (M+BM-1)/BM, batch)
// blockIdx.z = batch index, blockIdx.x/y = 输出 C[b] 的 block tile 位置
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int batch, int M, int N, int K) {
    int b = blockIdx.z;
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tid = threadIdx.x;
    int tx = tid % BLOCK_N;  // 0..15
    int ty = tid / BLOCK_N;  // 0..15

    // batch stride 寻址
    const float* A_b = A + b * M * K;
    const float* B_b = B + b * K * N;
    float* C_b = C + b * M * N;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    // 沿 K 方向分 tile 累加
    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 As[BM][BK]（越界补 0）
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A_b[ar * K + ac] : 0.0f;
        }
        // 协作加载 Bs[BK][BN]（越界补 0）
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = bk + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B_b[br * N + bc] : 0.0f;
        }
        __syncthreads();

        // register blocking：每 thread 算 TM×TN 个输出
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

    // 写回输出
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < M && gc < N)
                C_b[gr * N + gc] = acc[i][j];
        }
    }
}

int main() {
    int batch = 4, M = 64, N = 64, K = 64;
    size_t a_bytes = batch * M * K * sizeof(float);
    size_t b_bytes = batch * K * N * sizeof(float);
    size_t c_bytes = batch * M * N * sizeof(float);

    std::vector<float> h_A(batch * M * K), h_B(batch * K * N), h_C(batch * M * N);
    srand(42);
    for (auto& x : h_A)
        x = (rand() % 100) / 100.0f;
    for (auto& x : h_B)
        x = (rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_bytes);
    cudaMalloc(&d_B, b_bytes);
    cudaMalloc(&d_C, c_bytes);
    cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice);

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, batch);
    dim3 block(NUM_THREADS);
    batched_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, batch, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost);

    // CPU 验证
    bool pass = true;
    for (int b = 0; b < batch && pass; b++)
        for (int i = 0; i < M && pass; i++)
            for (int j = 0; j < N && pass; j++) {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += h_A[b * M * K + i * K + k] * h_B[b * K * N + k * N + j];
                if (fabs(s - h_C[b * M * N + i * N + j]) > 1e-3)
                    pass = false;
            }
    printf("batch=%d M=N=K=%d, %s\n", batch, M, pass ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

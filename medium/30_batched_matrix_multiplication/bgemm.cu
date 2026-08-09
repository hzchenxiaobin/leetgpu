#include <cuda_runtime.h>

const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM, BLOCK_N = BN / TN;
const int NUM_THREADS = BLOCK_M * BLOCK_N;
const int LOAD_A = BM * BK / NUM_THREADS;
const int LOAD_B = BN * BK / NUM_THREADS;

__global__ void bgemm(const float* __restrict__ A, const float* __restrict__ B,
                    float* __restrict__ C, int BATCH, int M, int N, int K) {
    int b = blockIdx.z;
    int tid = threadIdx.x;

    int by = blockIdx.y, bx = blockIdx.x;
    int tx = tid % BLOCK_N, ty = tid / BLOCK_N;

    const float* inputA = A + b * M * K;
    const float* inputB = B + b * K * N;
    float* outputC = C + b * M * N;

    __shared__ float sa[BM][BK];
    __shared__ float sb[BK][BN];

    float acc[TM][TN];
    for(int i = 0; i < TM; ++i) {
        for(int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for(int bk = 0; bk < K; bk+= BK) {
        // 1.将 A 矩阵拷贝到 shared memory
        #pragma unroll
        for(int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int gr = by * BM + r;
            int gc = bk + c;
            sa[r][c] = (gr < M && gc < K) ? inputA[gr * K + gc] : 0.0f;
        }
        
        // 2.将 B 矩阵拷贝到 shared memory
        #pragma unroll
        for(int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int gr = bk + r;
            int gc = bx * BN + c;
            sb[r][c] = (gr < K && gc < N) ? inputB[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        float ra[TM], rb[TN];

        for(int k = 0;k < BK; ++k) {
            //3.将 A 矩阵 拷贝 TM 到寄存器
            #pragma unroll
            for(int i = 0; i < TM; ++i) {
                ra[i] = sa[ty * TM + i][k];
            }

            // 4.将 B 矩阵拷贝 TN 到寄存器
            #pragma unroll
            for(int i = 0; i < TN; ++i) {
                rb[i] = sb[k][tx * TN + i];
            }

            // 5.计算 acc 矩阵
            #pragma unroll
            for(int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += ra[i] * rb[j];
                }
            }
        }
    }

    // 6.将结果拷贝到 C 矩阵中。
    #pragma unroll
    for(int i = 0; i < TM; ++i) {
        #pragma unroll
        for(int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if(gr < M && gc < N)
            outputC[gr * N + gc] = acc[i][j];
        }
    }

}

// A, B, C are device pointers
extern "C" void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, BATCH);
    dim3 block(NUM_THREADS);
    bgemm<<<grid, block>>>(A, B, C, BATCH, M, N, K);
    cudaDeviceSynchronize();
}

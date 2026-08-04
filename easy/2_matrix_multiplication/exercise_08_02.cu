#include <cuda_runtime.h>

const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;
const int BLOCK_N = BN / TN;
const int NUM_THREADS = BLOCK_M * BLOCK_N;

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C,int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int ty = tid / BLOCK_N;
    int tx = tid % BLOCK_N;

    const int LOAD_A = BM * BK / NUM_THREADS;
    const int LOAD_B = BK * BN / NUM_THREADS;

    float acc[TM][TN];
    #pragma unroll
    for(int i = 0; i < TM; ++i) {
        #pragma unroll
        for(int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    int tiles = (K + BK - 1) / BK;
    for(int t = 0; t < tiles; ++t) {
        //1.把数据加载到 shared Memory 中
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = t * BK + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
        }

        #pragma unroll
        for(int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = t * BK + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
        }
        __syncthreads();

        //2.把数据读取到 register 中, 并且计算 acc 矩阵
        #pragma unroll
        for(int k = 0; k < BK; ++k) {
            float a[TM], b[TN];

            // 读取 A 矩阵
            #pragma unroll
            for(int i = 0; i < TM; ++i) {
                a[i] = As[ty * TM + i][k]; 
            }

            // 读取 B 矩阵
            #pragma unroll
            for(int i = 0; i < TN; ++i) {
                b[i] = Bs[k][tx * TN + i];
            }
            
            #pragma unroll
            for(int i = 0;i < TM; ++i) {
                #pragma unroll
                for(int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    //4.将数据拷贝到 gm 中
    #pragma unroll
    for(int i = 0; i < TM; ++i) {
        #pragma unroll
        for(int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if(gr < M && gc < N) {
                C[gr * N + gc] = acc[i][j];
            }
        }
    }

}
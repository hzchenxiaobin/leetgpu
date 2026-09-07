// 33-ordinary-least-squares.cu —— OLS: β = (X^T X)^{-1} X^T y (normal equations + Cholesky)
// 三段流水线: gram_kernel (tiled GEMM) + matvec_kernel + cholesky_solve_kernel
// 编译命令: nvcc -O3 -arch=sm_75 33-ordinary-least-squares.cu -o ols
// 运行:     ./ols

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP  32
#define TILE  16        // gram GEMM 子块边长
#define NMAX  96        // 单 block Cholesky 的 n_features 上限（A 驻 shared，≤48KB）

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
exit(EXIT_FAILURE); } } while (0)

// ---- warp / block 归约（结果落在 warp0 lane0）----
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
__device__ __forceinline__ float block_reduce_sum(float v, float* warp_sums) {
    int lane = threadIdx.x & (WARP - 1);
    int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    v = (lane < nwarps) ? warp_sums[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

// ---- ① A = X^T X（n_feat × n_feat），tiled GEMM：每 block 算一个 TILE×TILE 子块 ----
__global__ void gram_kernel(const float* __restrict__ X, float* __restrict__ A,
                            int n_samples, int n_feat) {
    int tile_i = blockIdx.y, tile_j = blockIdx.x;
    int ii = threadIdx.y, jj = threadIdx.x;          // 块内行列
    int gi = tile_i * TILE + ii, gj = tile_j * TILE + jj;

    __shared__ float Asub[TILE][TILE];               // X[kb..][ gi 列块 ]
    __shared__ float Bsub[TILE][TILE];               // X[kb..][ gj 列块 ]

    float acc = 0.0f;
    for (int kb = 0; kb < n_samples; kb += TILE) {
        int row = kb + threadIdx.y;                  // 复用 ty 作 k 偏移
        int ci  = tile_i * TILE + threadIdx.x;
        int cj  = tile_j * TILE + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] = (row < n_samples && ci < n_feat) ? X[row * n_feat + ci] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (row < n_samples && cj < n_feat) ? X[row * n_feat + cj] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk)            // acc += Σ_kk X[kb+kk][gi]·X[kb+kk][gj]
            acc += Asub[kk][ii] * Bsub[kk][jj];
        __syncthreads();
    }
    if (gi < n_feat && gj < n_feat)
        A[gi * n_feat + gj] = acc;
}

// ---- ② b = X^T y（n_feat），coalesced per-feature ----
__global__ void matvec_kernel(const float* __restrict__ X, const float* __restrict__ y,
                              float* __restrict__ b, int n_samples, int n_feat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_feat) return;
    float sum = 0.0f;
    for (int k = 0; k < n_samples; ++k)              // warp 内 i 连续 → 合并读 X[k][i]
        sum += X[k * n_feat + i] * y[k];
    b[i] = sum;
}

// ---- ③ Cholesky 分解 + 前代/回代（单 block，A 驻 shared，n ≤ NMAX）----
// 共享布局: L(n×n) | z(n) | x(n) | warp_sums(32)
__global__ void cholesky_solve_kernel(float* __restrict__ A, const float* __restrict__ b,
                                      float* __restrict__ beta, int n) {
    extern __shared__ float smem[];
    float* L  = smem;
    float* z  = smem + (size_t)n * n;
    float* x  = z + n;
    float* ws = x + n;
    int tid = threadIdx.x;

    for (int idx = tid; idx < n * n; idx += blockDim.x) L[idx] = A[idx];   // 载入 A
    for (int idx = tid; idx < n; idx += blockDim.x)      z[idx] = b[idx];   // z 暂存右端项
    __syncthreads();

    // —— Cholesky: A = L L^T，in-place 覆盖下三角 ——
    for (int j = 0; j < n; ++j) {
        float ds = 0.0f;                              // 对角元: Σ_{k<j} L[j][k]^2
        for (int k = tid; k < j; k += blockDim.x)
            ds += L[j * n + k] * L[j * n + k];
        ds = block_reduce_sum(ds, ws);
        if (tid == 0) {
            float v = L[j * n + j] - ds;
            if (v < 0.0f) v = 0.0f;
            L[j * n + j] = sqrtf(v);
        }
        __syncthreads();
        float ljj = L[j * n + j];
        for (int i = j + 1 + tid; i < n; i += blockDim.x) {   // 列内并行: 行 i>j
            float s = 0.0f;
            for (int k = 0; k < j; ++k) s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (L[i * n + j] - s) / ljj;
        }
        __syncthreads();
    }

    // —— 前代: L z = b（z 已初值为 b）——
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int k = tid; k < i; k += blockDim.x) s += L[i * n + k] * z[k];
        s = block_reduce_sum(s, ws);
        if (tid == 0) z[i] = (z[i] - s) / L[i * n + i];
        __syncthreads();
    }

    // —— 回代: L^T β = z ——
    for (int i = n - 1; i >= 0; --i) {
        float s = 0.0f;
        for (int k = tid; k < n; k += blockDim.x)
            if (k > i) s += L[k * n + i] * x[k];      // L^T[i][k] = L[k][i]
        s = block_reduce_sum(s, ws);
        if (tid == 0) x[i] = (z[i] - s) / L[i * n + i];
        __syncthreads();
    }

    for (int idx = tid; idx < n; idx += blockDim.x) beta[idx] = x[idx];
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    int n = n_features;
    float *d_A, *d_b;
    cudaMalloc(&d_A, (size_t)n * n * sizeof(float));
    cudaMalloc(&d_b, (size_t)n * sizeof(float));

    dim3 gblock(TILE, TILE);
    dim3 ggrid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    gram_kernel<<<ggrid, gblock>>>(X, d_A, n_samples, n);
    matvec_kernel<<<(n + 255) / 256, 256>>>(X, y, d_b, n_samples, n);

    size_t smem = ((size_t)n * n + 3 * n + 32) * sizeof(float);   // L | z | x | warp_sums
    cholesky_solve_kernel<<<1, BLOCK, smem>>>(d_A, d_b, beta, n);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_b);
}

// ---- CPU 参考（double 累加）----
void ols_cpu(const float* X, const float* y, float* beta, int n_samples, int n_feat) {
    std::vector<double> A(n_feat * n_feat, 0.0), b(n_feat, 0.0);
    for (int i = 0; i < n_feat; ++i)
        for (int j = 0; j < n_feat; ++j) {
            double s = 0.0;
            for (int k = 0; k < n_samples; ++k) s += (double)X[k*n_feat+i] * X[k*n_feat+j];
            A[i*n_feat+j] = s;
        }
    for (int i = 0; i < n_feat; ++i) {
        double s = 0.0;
        for (int k = 0; k < n_samples; ++k) s += (double)X[k*n_feat+i] * y[k];
        b[i] = s;
    }
    for (int j = 0; j < n_feat; ++j) {
        double s = A[j*n_feat+j];
        for (int k = 0; k < j; ++k) s -= A[j*n_feat+k] * A[j*n_feat+k];
        A[j*n_feat+j] = std::sqrt(s);
        for (int i = j+1; i < n_feat; ++i) {
            double s2 = A[i*n_feat+j];
            for (int k = 0; k < j; ++k) s2 -= A[i*n_feat+k] * A[j*n_feat+k];
            A[i*n_feat+j] = s2 / A[j*n_feat+j];
        }
    }
    std::vector<double> z(n_feat);
    for (int i = 0; i < n_feat; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= A[i*n_feat+k] * z[k];
        z[i] = s / A[i*n_feat+i];
    }
    for (int i = n_feat-1; i >= 0; --i) {
        double s = z[i];
        for (int k = i+1; k < n_feat; ++k) s -= A[k*n_feat+i] * beta[k];
        beta[i] = (float)(s / A[i*n_feat+i]);
    }
}

// ---- 本地自测 ----
int main() {
    struct Case { int ns, nf; const float* X; const float* y; const float* ref; };
    float X0[] = {-0.23f,-0.23f,1.52f, 0.77f,-0.47f,1.58f, -0.14f,0.65f,0.5f,
                  -1.91f,-1.72f,0.24f, -0.46f,-0.47f,0.54f};
    float y0[] = {83.01f, 93.4f, 47.33f, -62.22f, 13.06f};
    float ref0[] = {13.97f, 29.12f, 61.05f};
    Case cases[] = {{5, 3, X0, y0, ref0}};

    int allpass = 1;
    for (auto& c : cases) {
        int ns = c.ns, nf = c.nf;
        std::vector<float> hX(c.X, c.X + ns*nf), hy(c.y, c.y + ns), hbeta(nf), href(nf);

        float *dX, *dy, *dbeta;
        CHECK_CUDA(cudaMalloc(&dX, ns*nf*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, ns*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dbeta, nf*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dX, hX.data(), ns*nf*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dy, hy.data(), ns*sizeof(float), cudaMemcpyHostToDevice));

        solve(dX, dy, dbeta, ns, nf);
        CHECK_CUDA(cudaMemcpy(hbeta.data(), dbeta, nf*sizeof(float), cudaMemcpyDeviceToHost));

        ols_cpu(hX.data(), hy.data(), href.data(), ns, nf);

        printf("case ns=%d nf=%d\n", ns, nf);
        int ok = 1;
        for (int i = 0; i < nf; ++i) {
            float r = href[i];
            float tol = 1e-2f * fmaxf(1.0f, fabsf(r));
            int pass = fabsf(hbeta[i] - r) <= tol;
            printf("  beta[%d]: gpu=%.4f cpu=%.4f ref=%.2f %s\n",
                   i, hbeta[i], r, c.ref ? c.ref[i] : 0.0f, pass ? "PASS" : "FAIL");
            ok &= pass;
        }
        allpass &= ok;

        cudaFree(dX); cudaFree(dy); cudaFree(dbeta);
    }

    // 随机规模测试（与 CPU 对比，atol=rtol=1e-2）
    srand(2024);
    for (int t = 0; t < 3; ++t) {
        int ns = 8 + rand() % 24, nf = 1 + rand() % 8; if (nf > ns) std::swap(ns, nf);
        std::vector<float> hX(ns*nf), hy(ns), hbeta(nf), href(nf);
        for (auto& v : hX) v = (float)(rand() % 2000) / 100.0f - 10.0f;
        for (auto& v : hy) v = (float)(rand() % 2000) / 100.0f - 10.0f;
        float *dX, *dy, *dbeta;
        CHECK_CUDA(cudaMalloc(&dX, ns*nf*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, ns*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dbeta, nf*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dX, hX.data(), ns*nf*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dy, hy.data(), ns*sizeof(float), cudaMemcpyHostToDevice));
        solve(dX, dy, dbeta, ns, nf);
        CHECK_CUDA(cudaMemcpy(hbeta.data(), dbeta, nf*sizeof(float), cudaMemcpyDeviceToHost));
        ols_cpu(hX.data(), hy.data(), href.data(), ns, nf);
        int ok = 1;
        for (int i = 0; i < nf; ++i) {
            float tol = 1e-2f * fmaxf(1.0f, fabsf(href[i]));
            if (fabsf(hbeta[i] - href[i]) > tol) ok = 0;
        }
        printf("random ns=%d nf=%d: %s\n", ns, nf, ok ? "PASS" : "FAIL");
        allpass &= ok;
        cudaFree(dX); cudaFree(dy); cudaFree(dbeta);
    }

    printf("\noverall: %s\n", allpass ? "PASS" : "FAIL");
    return allpass ? 0 : 1;
}

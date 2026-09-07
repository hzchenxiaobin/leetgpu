// 34-logistic-regression.cu —— Logistic Regression via Newton-Raphson (IRLS)
// 每次迭代: forward(sigmoid) → gradient → hessian(tiled GEMM) → Cholesky solve → update
// 编译命令: nvcc -O3 -arch=sm_75 34-logistic-regression.cu -o logreg
// 运行:     ./logreg

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP  32
#define TILE  16
#define NMAX  96

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

// 数值稳定 sigmoid: z≥0 → 1/(1+e^{-z}), z<0 → e^z/(1+e^z)
__device__ __forceinline__ float sigmoidf(float z) {
    if (z >= 0.0f) {
        float e = expf(-z);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(z);
        return e / (1.0f + e);
    }
}

// ---- ① forward: z[i]=X[i]·β, p[i]=σ(z), W[i]=max(p·(1-p), 1e-8) ----
__global__ void forward_kernel(const float* __restrict__ X, const float* __restrict__ beta,
                               float* __restrict__ p, float* __restrict__ W,
                               int n_samples, int n_feat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;
    float z = 0.0f;
    for (int j = 0; j < n_feat; ++j)
        z += X[i * n_feat + j] * beta[j];
    float pi = sigmoidf(z);
    p[i] = pi;
    float wi = pi * (1.0f - pi);
    if (wi < 1e-8f) wi = 1e-8f;
    W[i] = wi;
}

// ---- ② gradient: g[j] = Σ_i X[i][j]·(p[i]-y[i]) + l2·β[j] ----
__global__ void gradient_kernel(const float* __restrict__ X, const float* __restrict__ y,
                                const float* __restrict__ p, const float* __restrict__ beta,
                                float* __restrict__ g, float l2,
                                int n_samples, int n_feat) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_feat) return;
    float sum = 0.0f;
    for (int i = 0; i < n_samples; ++i)
        sum += X[i * n_feat + j] * (p[i] - y[i]);
    g[j] = sum + l2 * beta[j];
}

// ---- ③ hessian: H = Xᵀ diag(W) X + l2·I (tiled GEMM, W 行缩放) ----
__global__ void hessian_kernel(const float* __restrict__ X, const float* __restrict__ W,
                               float* __restrict__ H, float l2,
                               int n_samples, int n_feat) {
    int tile_i = blockIdx.y, tile_j = blockIdx.x;
    int ii = threadIdx.y, jj = threadIdx.x;
    int gi = tile_i * TILE + ii, gj = tile_j * TILE + jj;

    __shared__ float Asub[TILE][TILE];               // X[kb..][ gi 列块 ]
    __shared__ float Bsub[TILE][TILE];               // W[kb..]·X[kb..][ gj 列块 ]

    float acc = 0.0f;
    for (int kb = 0; kb < n_samples; kb += TILE) {
        int row = kb + threadIdx.y;
        int ci  = tile_i * TILE + threadIdx.x;
        int cj  = tile_j * TILE + threadIdx.x;
        float w = (row < n_samples) ? W[row] : 0.0f;
        Asub[threadIdx.y][threadIdx.x] = (row < n_samples && ci < n_feat) ? X[row * n_feat + ci] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (row < n_samples && cj < n_feat) ? w * X[row * n_feat + cj] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk)
            acc += Asub[kk][ii] * Bsub[kk][jj];
        __syncthreads();
    }
    if (gi < n_feat && gj < n_feat) {
        if (gi == gj) acc += l2;                      // l2·I 对角项
        H[gi * n_feat + gj] = acc;
    }
}

// ---- ④ Cholesky 分解 + 前代/回代（单 block，H 驻 shared，n ≤ NMAX）----
// 共享布局: L(n×n) | z(n) | x(n) | warp_sums(32)
__global__ void cholesky_solve_kernel(float* __restrict__ H, const float* __restrict__ g,
                                      float* __restrict__ delta, int n) {
    extern __shared__ float smem[];
    float* L  = smem;
    float* z  = smem + (size_t)n * n;
    float* x  = z + n;
    float* ws = x + n;
    int tid = threadIdx.x;

    for (int idx = tid; idx < n * n; idx += blockDim.x) L[idx] = H[idx];   // 载入 H
    for (int idx = tid; idx < n; idx += blockDim.x)      z[idx] = g[idx];   // z 暂存右端项
    __syncthreads();

    // —— Cholesky: H = L Lᵀ，in-place 覆盖下三角 ——
    for (int j = 0; j < n; ++j) {
        float ds = 0.0f;                              // 对角元: Σ_{k<j} L[j][k]²
        for (int k = tid; k < j; k += blockDim.x)
            ds += L[j * n + k] * L[j * n + k];
        ds = block_reduce_sum(ds, ws);
        if (tid == 0) {
            float v = L[j * n + j] - ds;
            if (v < 1e-10f) v = 1e-10f;               // 数值保护
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

    // —— 前代: L z = g（z 已初值为 g）——
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int k = tid; k < i; k += blockDim.x) s += L[i * n + k] * z[k];
        s = block_reduce_sum(s, ws);
        if (tid == 0) z[i] = (z[i] - s) / L[i * n + i];
        __syncthreads();
    }

    // —— 回代: Lᵀ Δ = z ——
    for (int i = n - 1; i >= 0; --i) {
        float s = 0.0f;
        for (int k = tid; k < n; k += blockDim.x)
            if (k > i) s += L[k * n + i] * x[k];      // Lᵀ[i][k] = L[k][i]
        s = block_reduce_sum(s, ws);
        if (tid == 0) x[i] = (z[i] - s) / L[i * n + i];
        __syncthreads();
    }

    for (int idx = tid; idx < n; idx += blockDim.x) delta[idx] = x[idx];
}

// ---- ⑤ update + norm: β -= Δ, 计算 ‖Δ‖ ----
__global__ void update_norm_kernel(float* __restrict__ beta, const float* __restrict__ delta,
                                   float* __restrict__ norm_out, int n) {
    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    float sq = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float d = delta[j];
        beta[j] -= d;
        sq += d * d;
    }
    sq = block_reduce_sum(sq, warp_sums);
    if (tid == 0) *norm_out = sqrtf(sq);
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    int n = n_features;
    float l2 = 1e-6f;
    int max_iter = 1000;
    float tol = 1e-8f;

    float *d_p, *d_W, *d_g, *d_H, *d_delta, *d_norm;
    cudaMalloc(&d_p, n_samples * sizeof(float));
    cudaMalloc(&d_W, n_samples * sizeof(float));
    cudaMalloc(&d_g, n * sizeof(float));
    cudaMalloc(&d_H, (size_t)n * n * sizeof(float));
    cudaMalloc(&d_delta, n * sizeof(float));
    cudaMalloc(&d_norm, sizeof(float));

    cudaMemset(beta, 0, n * sizeof(float));           // β 初始为 0

    for (int iter = 0; iter < max_iter; ++iter) {
        forward_kernel<<<(n_samples + BLOCK - 1) / BLOCK, BLOCK>>>(
            X, beta, d_p, d_W, n_samples, n);
        gradient_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
            X, y, d_p, beta, d_g, l2, n_samples, n);

        dim3 hblock(TILE, TILE);
        dim3 hgrid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        hessian_kernel<<<hgrid, hblock>>>(X, d_W, d_H, l2, n_samples, n);

        size_t smem = ((size_t)n * n + 3 * n + 32) * sizeof(float);
        cholesky_solve_kernel<<<1, BLOCK, smem>>>(d_H, d_g, d_delta, n);

        update_norm_kernel<<<1, BLOCK>>>(beta, d_delta, d_norm, n);
        cudaDeviceSynchronize();

        float norm_val;
        cudaMemcpy(&norm_val, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        if (norm_val < tol) break;
    }

    cudaFree(d_p); cudaFree(d_W); cudaFree(d_g);
    cudaFree(d_H); cudaFree(d_delta); cudaFree(d_norm);
}

// ---- CPU 参考（double 累加，Newton-Raphson / IRLS）----
void logreg_cpu(const float* X, const float* y, float* beta, int n_samples, int n_feat) {
    std::vector<double> b(n_feat, 0.0);
    double l2 = 1e-6, tol = 1e-8;
    for (int iter = 0; iter < 1000; ++iter) {
        std::vector<double> p(n_samples), W(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            double z = 0;
            for (int j = 0; j < n_feat; ++j) z += X[i*n_feat+j] * b[j];
            p[i] = 1.0 / (1.0 + exp(-z));
            W[i] = std::max(p[i] * (1.0 - p[i]), 1e-8);
        }
        std::vector<double> grad(n_feat, 0.0);
        for (int j = 0; j < n_feat; ++j) {
            double s = 0;
            for (int i = 0; i < n_samples; ++i) s += X[i*n_feat+j] * (p[i] - y[i]);
            grad[j] = s + l2 * b[j];
        }
        std::vector<double> H(n_feat*n_feat, 0.0);
        for (int j1 = 0; j1 < n_feat; ++j1)
            for (int j2 = 0; j2 < n_feat; ++j2) {
                double s = 0;
                for (int i = 0; i < n_samples; ++i) s += X[i*n_feat+j1] * W[i] * X[i*n_feat+j2];
                H[j1*n_feat+j2] = s + (j1==j2 ? l2 : 0.0);
            }
        // Cholesky 分解 H = L Lᵀ
        for (int j = 0; j < n_feat; ++j) {
            double s = H[j*n_feat+j];
            for (int k = 0; k < j; ++k) s -= H[j*n_feat+k] * H[j*n_feat+k];
            H[j*n_feat+j] = sqrt(s);
            for (int i = j+1; i < n_feat; ++i) {
                double s2 = H[i*n_feat+j];
                for (int k = 0; k < j; ++k) s2 -= H[i*n_feat+k] * H[j*n_feat+k];
                H[i*n_feat+j] = s2 / H[j*n_feat+j];
            }
        }
        std::vector<double> z(n_feat), delta(n_feat);
        for (int i = 0; i < n_feat; ++i) {
            double s = grad[i];
            for (int k = 0; k < i; ++k) s -= H[i*n_feat+k] * z[k];
            z[i] = s / H[i*n_feat+i];
        }
        for (int i = n_feat-1; i >= 0; --i) {
            double s = z[i];
            for (int k = i+1; k < n_feat; ++k) s -= H[k*n_feat+i] * delta[k];
            delta[i] = s / H[i*n_feat+i];
        }
        for (int j = 0; j < n_feat; ++j) b[j] -= delta[j];
        double norm = 0; for (double d : delta) norm += d*d;
        if (sqrt(norm) < tol) break;
    }
    for (int j = 0; j < n_feat; ++j) beta[j] = (float)b[j];
}

// ---- 本地自测 ----
int main() {
    struct Case { int ns, nf; const float* X; const float* y; const float* ref; };
    float X0[] = {2.f,1.f, 1.f,2.f, 3.f,3.f, 1.5f,2.5f, -1.f,-2.f, -2.f,-1.f, -1.5f,-2.5f, -3.f,-3.f};
    float y0[] = {1.f,1.f,1.f,0.f,0.f,0.f,1.f,0.f};
    float ref0[] = {2.26f, -1.29f};
    Case cases[] = {{8, 2, X0, y0, ref0}};

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

        logreg_cpu(hX.data(), hy.data(), href.data(), ns, nf);

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
    for (int t = 0; t < 5; ++t) {
        int ns = 5 + rand() % 20, nf = 1 + rand() % 6; if (nf > ns) std::swap(ns, nf);
        std::vector<float> hX(ns*nf), hy(ns), hbeta(nf), href(nf);
        for (auto& v : hX) v = (float)(rand() % 2000) / 100.0f - 10.0f;
        for (auto& v : hy) v = (float)(rand() % 2);
        float *dX, *dy, *dbeta;
        CHECK_CUDA(cudaMalloc(&dX, ns*nf*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dy, ns*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&dbeta, nf*sizeof(float)));
        CHECK_CUDA(cudaMemcpy(dX, hX.data(), ns*nf*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dy, hy.data(), ns*sizeof(float), cudaMemcpyHostToDevice));
        solve(dX, dy, dbeta, ns, nf);
        CHECK_CUDA(cudaMemcpy(hbeta.data(), dbeta, nf*sizeof(float), cudaMemcpyDeviceToHost));
        logreg_cpu(hX.data(), hy.data(), href.data(), ns, nf);
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

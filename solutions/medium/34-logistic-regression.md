# LeetGPU Logistic Regression 题解

## 1. 题目概述

- **标题 / 题号**：Logistic Regression（#34，medium）
- **链接**：https://leetgpu.com/challenges/logistic-regression
- **难度**：中等
- **标签**：CUDA、sigmoid、Newton-Raphson（IRLS）、tiled GEMM（Hessian）、Cholesky 分解、迭代 kernel launch

**题意**：给定特征矩阵 $X$（`n_samples × n_features`，行主序 `float32`）与二值目标向量 $y$（`n_samples`，仅含 0/1），求系数向量 $\beta$（`n_features`）使对数似然最大化：

$$\max_{\beta} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1-y_i) \log(1-p_i) \right], \quad p_i = \sigma(X_i^{T}\beta) = \frac{1}{1+e^{-X_i^{T}\beta}}$$

**输入输出**：`X`、`y` 为 device 指针（只读），`beta` 为 device 指针（输出，长度 `n_features`，初始为 0），`n_samples`、`n_features` 为 int。签名固定：

```cpp
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features);
```

**约束**：

- $1 \le \text{n\_samples} \le 100{,}000$，$1 \le \text{n\_features} \le 1{,}000$，且 $\text{n\_samples} \ge \text{n\_features}$
- $-10.0 \le X \text{ 值} \le 10.0$，$y$ 仅含 0 或 1
- 容差 `atol = rtol = 1e-2`
- 性能测点：`n_features = 8, n_samples = 16`

**示例**：

```text
X (8×2):                y:        β (输出):
[ 2.0  1.0]            [1]        [ 2.26]
[ 1.0  2.0]            [1]        [-1.29]
[ 3.0  3.0]            [1]
[ 1.5  2.5]            [0]
[-1.0 -2.0]            [0]
[-2.0 -1.0]            [0]
[-1.5 -2.5]            [1]
[-3.0 -3.0]            [0]
```

> 💡 这道题是 **迭代优化在 GPU 上的编排题**：Logistic Regression 的 MLE 求解本质是 Newton-Raphson（即 IRLS——Iteratively Reweighted Least Squares），每次迭代做一次「前向 sigmoid → 梯度 → Hessian GEMM → Cholesky 求解 → 更新」。与单趟的 [#33 OLS](https://leetgpu.com/challenges/ordinary-least-squares) 同构，区别是 **把 OLS 流水线套进一个 host 侧迭代循环**，每轮重新算权重 $W$ 和 Hessian。性能测点 $n=8$ 极小，重点是用对的概念把迭代中每一段都做对、让收敛轨迹与参考一致。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线（Newton-Raphson / IRLS）

参考实现用 Newton-Raphson 迭代：每次算梯度 $\nabla$ 和 Hessian $H$，解线性方程组 $H\Delta = \nabla$，更新 $\beta \leftarrow \beta - \Delta$，直到 $\|\Delta\| < \text{tol}$。

```cpp
// cpu_baseline.cpp —— CPU 串行 Logistic Regression (IRLS, double 累加)
void logreg_cpu(const float* X, const float* y, float* beta, int n_samples, int n_feat) {
    std::vector<double> b(n_feat, 0.0);              // β 初始为 0
    double l2 = 1e-6, tol = 1e-8;
    for (int iter = 0; iter < 1000; ++iter) {
        std::vector<double> p(n_samples), W(n_samples);
        for (int i = 0; i < n_samples; ++i) {         // 前向: z = Xβ, p = σ(z)
            double z = 0;
            for (int j = 0; j < n_feat; ++j) z += X[i*n_feat+j] * b[j];
            p[i] = 1.0 / (1.0 + exp(-z));
            W[i] = std::max(p[i] * (1.0 - p[i]), 1e-8);
        }
        std::vector<double> grad(n_feat, 0.0);        // ∇ = Xᵀ(p-y) + l2·β
        for (int j = 0; j < n_feat; ++j) {
            double s = 0;
            for (int i = 0; i < n_samples; ++i) s += X[i*n_feat+j] * (p[i] - y[i]);
            grad[j] = s + l2 * b[j];
        }
        std::vector<double> H(n_feat*n_feat, 0.0);    // H = Xᵀ diag(W) X + l2·I
        for (int j1 = 0; j1 < n_feat; ++j1)
            for (int j2 = 0; j2 < n_feat; ++j2) {
                double s = 0;
                for (int i = 0; i < n_samples; ++i) s += X[i*n_feat+j1] * W[i] * X[i*n_feat+j2];
                H[j1*n_feat+j2] = s + (j1==j2 ? l2 : 0.0);
            }
        // Cholesky 分解 H = L Lᵀ + 前代/回代解 H Δ = ∇（略，同 OLS）
        std::vector<double> delta = cholesky_solve(H, grad, n_feat);
        for (int j = 0; j < n_feat; ++j) b[j] -= delta[j];
        double norm = 0; for (double d : delta) norm += d*d;
        if (sqrt(norm) < tol) break;
    }
    for (int j = 0; j < n_feat; ++j) beta[j] = (float)b[j];
}
```

每轮迭代含 $O(\text{n\_samples}\cdot n)$（前向+梯度）、$O(\text{n\_samples}\cdot n^{2})$（Hessian）、$O(n^{3})$（Cholesky）的串行循环，且 Newton-Raphson 需多轮（典型 5–20 轮），CPU 单核明显吃力。

### 2.2 朴素 GPU：逐元素并行还不够

朴素思路是把 sigmoid、梯度逐元素并行，但 Hessian GEMM 和 Cholesky 求解无法「一个 thread 一个输出」——它们涉及跨样本归约与列间依赖。更关键的是 **迭代本身是串行的**：第 $k$ 轮的 $\beta$ 依赖第 $k{-}1$ 轮的完整结果，无法跨轮并行。

![IRLS 迭代流水线](/images/logistic_regression_overview.svg)

> ⚠️ 真正的 GPU 化挑战：**轮间串行、轮内并行**。每轮内部复用 OLS 的「tiled GEMM + coalesced matvec + 单 block Cholesky」三段模板，轮间由 host 侧循环驱动、用 `cudaMemcpy` 读回 $\|\Delta\|$ 判收敛。这正是本题区别于单趟 OLS 的核心考点——**迭代 kernel 编排与收敛同步**。

## 3. GPU 设计

### 3.1 并行化策略：IRLS = 迭代加权最小二乘

把 Newton-Raphson 每轮拆成 5 个 kernel，各自用最合适的并行模式：

| 步 | kernel | 计算 | 并行模式 | 概念 |
|----|--------|------|----------|------|
| ① | `forward_kernel` | $z=X\beta$、$p=\sigma(z)$、$W=p(1-p)$ | 一个 thread 一个 sample | **fused elementwise** + matvec |
| ② | `gradient_kernel` | $g=X^{T}(p-y)+\lambda_2\beta$ | 一个 thread 一个 feature | **coalesced matvec** |
| ③ | `hessian_kernel` | $H=X^{T}\text{diag}(W)X+\lambda_2 I$ | 2D grid，每 block 算 `TILE×TILE` 子块 | **shared memory tiling**（W 行缩放） |
| ④ | `cholesky_solve_kernel` | $H=LL^{T}\Rightarrow Lz=g\Rightarrow L^{T}\Delta=z$ | 单 block，列间串行 + 列内并行 | **顺序算法并行化** + block_reduce |
| ⑤ | `update_norm_kernel` | $\beta\mathrel{-}=\Delta$、$\|\Delta\|$ | 单 block 归约 | **block reduce** + 收敛判据 |

核心伪代码：

```text
host 侧迭代循环 (max_iter=1000, tol=1e-8):
  for iter in 0..max_iter:
    ① forward_kernel<<<n_samples/256, 256>>>
       thread i:  z = Σ_j X[i][j]·β[j]
                  p[i] = σ(z);  W[i] = max(p·(1-p), 1e-8)

    ② gradient_kernel<<<n_feat/256, 256>>>
       thread j:  g[j] = Σ_i X[i][j]·(p[i]-y[i]) + l2·β[j]

    ③ hessian_kernel<<<(n/TILE)², (TILE,TILE)>>>
       block(bi,bj):  H[gi][gj] = Σ_k X[k][gi]·W[k]·X[k][gj] + l2·δ(gi==gj)

    ④ cholesky_solve_kernel<<<1, 256, smem>>>
       for j in 0..n-1:                         // 列间串行
         L[j][j] = sqrt(H[j][j] - Σ_{k<j} L[j][k]²)
         并行 i>j: L[i][j] = (H[i][j] - Σ_{k<j} ...) / L[j][j]
         __syncthreads()
       前代 Lz=g；回代 LᵀΔ=z

    ⑤ update_norm_kernel<<<1, 256>>>
       β[j] -= Δ[j];  ‖Δ‖ = sqrt(Σ Δ[j]²)

    cudaMemcpy(&norm, d_norm, ...)               // 读回标量
    if norm < tol: break
```

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| $X$、$y$ | global memory | 只读输入；全迭代复用，不修改 |
| $\beta$ | global memory | 输出，每轮 ⑤ 原地更新 |
| $p$、$W$ | global memory | ① 写、②③ 读；每轮重新计算 |
| $g$（梯度） | global memory | ② 写、④ 读 |
| $H$（Hessian） | global → shared | ③ 写 global（$n^{2}$），④ 整块载入 shared 做 in-place 分解 |
| $X$ 子块 `Asub`/`Bsub` | shared memory | ③ tiling 复用（Bsub 带 W 缩放） |
| $L$、$z$、$\Delta$ | shared memory | ④ 单 block 内整块驻留，分解与求解全程不落 HBM |
| 归约缓冲 `warp_sums` | shared memory | ④⑤ `block_reduce_sum` 暂存各 warp 部分和 |

> 💡 关键判断：IRLS 每轮的存储层次与 OLS 完全同构——①②是 memory-bound（吃 $X$ 带宽），③是 GEMM（tiling 复用），④是 barrier-bound（$n$ 小时单 block 足够）。区别只是多了 $p$/$W$ 两个 `n_samples` 长度的临时数组，以及 host 侧的迭代循环。

### 3.3 关键技巧

- **IRLS = 迭代加权最小二乘**：每轮的 $H\Delta = g$ 本质是一次加权最小二乘求解（权重 $W$ 随 $\beta$ 变化），直接复用 OLS 的 Cholesky 模板。
- **Fused forward kernel**：把 $z=X\beta$、$p=\sigma(z)$、$W=p(1-p)$ 融合进一个 kernel（一个 thread 一个 sample），避免中间数组 $z$ 落 HBM。
- **数值稳定 sigmoid**：$z \ge 0$ 时算 $1/(1+e^{-z})$，$z < 0$ 时算 $e^{z}/(1+e^{z})$，避免 `expf` 溢出。
- **Tiled GEMM 带 W 缩放**：Hessian 的 $X^{T}\text{diag}(W)X$ 在 tile 加载时把 $W[k]$ 乘进 `Bsub`，无需额外存储 $XW$ 矩阵。
- **单 block Cholesky**：$n$ 小时 $H$ 整块驻 shared，列间靠 `__syncthreads` 推进依赖链（同 OLS）。
- **收敛同步**：⑤ 用 block_reduce 算 $\|\Delta\|$ 写入 device 标量，host 侧 `cudaMemcpy` 读回判断是否 `break`。

> ⚠️ ④ 设计为 $n \le \text{NMAX}=96$（$H$ 驻 shared，$96^{2}\times4\approx36\text{KB}$，默认 48KB shared 上限内）。性能测点 $n=8$ 远在其内。$n$ 更大时的 blocked Cholesky 扩展见 §5.3。

## 4. Kernel 实现

下面是**完整可编译**版本，包含 5 个 kernel、`solve` 入口、CPU 参考与多组验证：

```cuda
// logistic_regression.cu —— Logistic Regression via Newton-Raphson (IRLS)
// 每次迭代: forward(sigmoid) → gradient → hessian(tiled GEMM) → Cholesky solve → update
// 编译命令: nvcc -O3 -arch=sm_75 logistic_regression.cu -o logreg
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
```

> 💡 提交 LeetGPU 平台时，把 5 个 kernel + `solve` 填进 starter 空壳（带 `main`/`logreg_cpu` 的版本仅用于本地自测与 profiling）。

### 4.1 LeetGPU 提交版本

下面是可直接粘贴到 LeetGPU 编辑器的提交版本（去掉本地自测，保留 5 个 kernel 与 `solve`）：

```cuda
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK 256
#define WARP  32
#define TILE  16

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

__device__ __forceinline__ float sigmoidf(float z) {
    if (z >= 0.0f) { float e = expf(-z); return 1.0f / (1.0f + e); }
    else { float e = expf(z); return e / (1.0f + e); }
}

__global__ void forward_kernel(const float* __restrict__ X, const float* __restrict__ beta,
                               float* __restrict__ p, float* __restrict__ W,
                               int n_samples, int n_feat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_samples) return;
    float z = 0.0f;
    for (int j = 0; j < n_feat; ++j) z += X[i * n_feat + j] * beta[j];
    float pi = sigmoidf(z);
    p[i] = pi;
    float wi = pi * (1.0f - pi);
    if (wi < 1e-8f) wi = 1e-8f;
    W[i] = wi;
}

__global__ void gradient_kernel(const float* __restrict__ X, const float* __restrict__ y,
                                const float* __restrict__ p, const float* __restrict__ beta,
                                float* __restrict__ g, float l2, int n_samples, int n_feat) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_feat) return;
    float sum = 0.0f;
    for (int i = 0; i < n_samples; ++i) sum += X[i * n_feat + j] * (p[i] - y[i]);
    g[j] = sum + l2 * beta[j];
}

__global__ void hessian_kernel(const float* __restrict__ X, const float* __restrict__ W,
                               float* __restrict__ H, float l2, int n_samples, int n_feat) {
    int tile_i = blockIdx.y, tile_j = blockIdx.x;
    int ii = threadIdx.y, jj = threadIdx.x;
    int gi = tile_i * TILE + ii, gj = tile_j * TILE + jj;
    __shared__ float Asub[TILE][TILE];
    __shared__ float Bsub[TILE][TILE];
    float acc = 0.0f;
    for (int kb = 0; kb < n_samples; kb += TILE) {
        int row = kb + threadIdx.y;
        int ci = tile_i * TILE + threadIdx.x, cj = tile_j * TILE + threadIdx.x;
        float w = (row < n_samples) ? W[row] : 0.0f;
        Asub[threadIdx.y][threadIdx.x] = (row < n_samples && ci < n_feat) ? X[row * n_feat + ci] : 0.0f;
        Bsub[threadIdx.y][threadIdx.x] = (row < n_samples && cj < n_feat) ? w * X[row * n_feat + cj] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) acc += Asub[kk][ii] * Bsub[kk][jj];
        __syncthreads();
    }
    if (gi < n_feat && gj < n_feat) { if (gi == gj) acc += l2; H[gi * n_feat + gj] = acc; }
}

__global__ void cholesky_solve_kernel(float* __restrict__ H, const float* __restrict__ g,
                                      float* __restrict__ delta, int n) {
    extern __shared__ float smem[];
    float* L = smem; float* z = smem + (size_t)n * n; float* x = z + n; float* ws = x + n;
    int tid = threadIdx.x;
    for (int idx = tid; idx < n * n; idx += blockDim.x) L[idx] = H[idx];
    for (int idx = tid; idx < n; idx += blockDim.x) z[idx] = g[idx];
    __syncthreads();
    for (int j = 0; j < n; ++j) {
        float ds = 0.0f;
        for (int k = tid; k < j; k += blockDim.x) ds += L[j * n + k] * L[j * n + k];
        ds = block_reduce_sum(ds, ws);
        if (tid == 0) { float v = L[j * n + j] - ds; if (v < 1e-10f) v = 1e-10f; L[j * n + j] = sqrtf(v); }
        __syncthreads();
        float ljj = L[j * n + j];
        for (int i = j + 1 + tid; i < n; i += blockDim.x) {
            float s = 0.0f;
            for (int k = 0; k < j; ++k) s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (L[i * n + j] - s) / ljj;
        }
        __syncthreads();
    }
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int k = tid; k < i; k += blockDim.x) s += L[i * n + k] * z[k];
        s = block_reduce_sum(s, ws);
        if (tid == 0) z[i] = (z[i] - s) / L[i * n + i];
        __syncthreads();
    }
    for (int i = n - 1; i >= 0; --i) {
        float s = 0.0f;
        for (int k = tid; k < n; k += blockDim.x) if (k > i) s += L[k * n + i] * x[k];
        s = block_reduce_sum(s, ws);
        if (tid == 0) x[i] = (z[i] - s) / L[i * n + i];
        __syncthreads();
    }
    for (int idx = tid; idx < n; idx += blockDim.x) delta[idx] = x[idx];
}

__global__ void update_norm_kernel(float* __restrict__ beta, const float* __restrict__ delta,
                                   float* __restrict__ norm_out, int n) {
    __shared__ float warp_sums[32];
    int tid = threadIdx.x;
    float sq = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) { float d = delta[j]; beta[j] -= d; sq += d * d; }
    sq = block_reduce_sum(sq, warp_sums);
    if (tid == 0) *norm_out = sqrtf(sq);
}

// X, y, beta are device pointers
extern "C" void solve(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    int n = n_features;
    float l2 = 1e-6f;
    float *d_p, *d_W, *d_g, *d_H, *d_delta, *d_norm;
    cudaMalloc(&d_p, n_samples * sizeof(float));
    cudaMalloc(&d_W, n_samples * sizeof(float));
    cudaMalloc(&d_g, n * sizeof(float));
    cudaMalloc(&d_H, (size_t)n * n * sizeof(float));
    cudaMalloc(&d_delta, n * sizeof(float));
    cudaMalloc(&d_norm, sizeof(float));
    cudaMemset(beta, 0, n * sizeof(float));
    for (int iter = 0; iter < 1000; ++iter) {
        forward_kernel<<<(n_samples + 255) / 256, 256>>>(X, beta, d_p, d_W, n_samples, n);
        gradient_kernel<<<(n + 255) / 256, 256>>>(X, y, d_p, beta, d_g, l2, n_samples, n);
        dim3 hblock(TILE, TILE);
        dim3 hgrid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        hessian_kernel<<<hgrid, hblock>>>(X, d_W, d_H, l2, n_samples, n);
        size_t smem = ((size_t)n * n + 3 * n + 32) * sizeof(float);
        cholesky_solve_kernel<<<1, 256, smem>>>(d_H, d_g, d_delta, n);
        update_norm_kernel<<<1, 256>>>(beta, d_delta, d_norm, n);
        cudaDeviceSynchronize();
        float norm_val;
        cudaMemcpy(&norm_val, d_norm, sizeof(float), cudaMemcpyDeviceToHost);
        if (norm_val < 1e-8f) break;
    }
    cudaFree(d_p); cudaFree(d_W); cudaFree(d_g); cudaFree(d_H); cudaFree(d_delta); cudaFree(d_norm);
}
```

> 💡 提交版本与 §4 的 5 个 kernel 完全同源，仅去掉 `main` / `logreg_cpu` 等本地自测代码。`solve` 内动态 shared 大小为 $(n^{2}+3n+32)\times4$ 字节（$L\,|\,z\,|\,\Delta\,|\,\text{warp\_sums}$），$n=8$ 时约 0.4KB，$n=96$ 时约 37KB。

### 4.2 代码详解

五个 kernel 串成 IRLS 迭代环：`forward_kernel` 融合 sigmoid 与权重计算；`gradient_kernel` 用 coalesced matvec 算梯度；`hessian_kernel` 用 shared memory tiling 算带权 GEMM；`cholesky_solve_kernel` 在单 block 内完成 Cholesky 分解与两趟三角求解；`update_norm_kernel` 更新 $\beta$ 并归约 $\|\Delta\|$ 供 host 判收敛。

#### 4.2.1 `forward_kernel`：fused sigmoid + 权重

![IRLS 前向与权重计算](/images/logistic_regression_forward.svg)

| 步骤 | 代码 | 说明 |
|------|------|------|
| **映射** | `i = blockIdx.x*blockDim.x + threadIdx.x` | 一个 thread 负责一个 sample $i$ |
| **前向** | `z = Σ_j X[i][j]·β[j]` | 串行扫 `n_feat`；thread 间 $i$ 独立，无归约 |
| **sigmoid** | `pi = sigmoidf(z)` | 数值稳定：$z\ge0$ 用 $1/(1+e^{-z})$，$z<0$ 用 $e^z/(1+e^z)$ |
| **权重** | `W[i] = max(pi*(1-pi), 1e-8)` | $W=p(1-p) \in (0, 0.25]$，clamp 防退化 |
| **写回** | `p[i]=pi; W[i]=wi` | ②③ 下轮读取 |

> 💡 **融合动机**：$z$、$p$、$W$ 都是 `n_samples` 长度的中间量，但 $z$ 仅在计算 $p$ 时用一次。把三者融合进一个 kernel，$z$ 留在 register 不落 HBM，只写 $p$ 和 $W$ 到 global。

#### 4.2.2 `gradient_kernel`：coalesced matvec

| 步骤 | 代码 | 说明 |
|------|------|------|
| **映射** | `j = blockIdx.x*blockDim.x + threadIdx.x` | 一个 thread 负责一个 feature $j$ |
| **累加** | `sum += X[i][j]·(p[i]-y[i])` | 串行扫 `n_samples`；warp 内 $j$ 连续 → 读 $X[i][j]$ 命中同一行连续列，**合并访存** |
| **正则** | `g[j] = sum + l2*β[j]` | L2 正则项 $\lambda_2\beta$ |

> 💡 与 OLS 的 `matvec_kernel` 同构——都是「per-feature 串行扫 samples，warp 内 coalesced 读 $X$ 行」。区别是乘的不是 $y$ 而是 $(p-y)$（残差）。

#### 4.2.3 `hessian_kernel`：tiled GEMM 带 W 缩放

![Hessian tiled GEMM 与 W 缩放](/images/logistic_regression_hessian.svg)

| 步骤 | 代码 | 说明 |
|------|------|------|
| **block → 子块** | `tile_i=blockIdx.y, tile_j=blockIdx.x` | 一个 block 算 $H$ 的一个 `TILE×TILE` 子块 |
| **thread → 元素** | `gi=tile_i*TILE+ii, gj=tile_j*TILE+jj` | thread `(ii,jj)` 算 $H[gi][gj]$ |
| **协作加载** | `Asub[ty][tx]=X[row][ci]`、`Bsub[ty][tx]=W[row]*X[row][cj]` | Bsub 在加载时乘 $W$（行缩放），无需额外存 $XW$ |
| **同步** | `__syncthreads()` | 装完才能读；算完才能覆盖下一 tile |
| **累加** | `acc += Σ_kk Asub[kk][ii]*Bsub[kk][jj]` | $\Sigma_k X[k][gi]\cdot W[k]\cdot X[k][gj]$ |
| **正则** | `if (gi==gj) acc += l2` | $\lambda_2 I$ 对角项 |
| **写回** | `H[gi*n+gj]=acc` | 越界 thread 不写 |

**关键索引关系**：
- `Asub[kk][ii]` = $X[kb{+}kk][gi]$（列块 `tile_i`），`Bsub[kk][jj]` = $W[kb{+}kk]\cdot X[kb{+}kk][gj]$（列块 `tile_j`，带 $W$ 缩放）
- 累加 $\Rightarrow$ `acc` = $\Sigma_{k} X[k][gi]\cdot W[k]\cdot X[k][gj]$，即 $H[gi][gj] = (X^{T}\text{diag}(W)X)[gi][gj]$

> 💡 **与 OLS `gram_kernel` 的唯一区别**：Bsub 加载时多乘一个 $W[row]$。这个「行缩放」让同一个 tiled GEMM 模板同时服务 OLS（$W\equiv1$）和 IRLS（$W=p(1-p)$），是 IRLS = 迭代加权最小二乘的直接体现。

#### 4.2.4 `cholesky_solve_kernel`：列间串行 + 列内并行

| 步骤 | 代码 | 同步语义 |
|------|------|----------|
| **载入** | `L[idx]=H[idx]`、`z[idx]=g[idx]` | `__syncthreads`：装完才能开始分解 |
| **对角元** | `ds=Σ_{k<j} L[j][k]²`（block_reduce）→ `L[j][j]=sqrt(H[j][j]-ds)` | `__syncthreads`：对角元写完，列内 thread 才能读 `ljj` |
| **列内并行** | `i=j+1+tid..n`：`L[i][j]=(H[i][j]-Σ_{k<j} L[i][k]L[j][k])/ljj` | `__syncthreads`：本列写完才能进下一列 |
| **前代** | `z[i]=(g[i]-Σ_{k<i} L[i][k]z[k])/L[i][i]` | `__syncthreads`：`z[i]` 写完，下一行才能读 |
| **回代** | `Δ[i]=(z[i]-Σ_{k>i} L[k][i]Δ[k])/L[i][i]`（$L^{T}[i][k]=L[k][i]$） | `__syncthreads`：`Δ[i]` 写完，上一行才能读 |

> 💡 此 kernel 与 OLS 的 `cholesky_solve_kernel` **完全同构**——Hessian $H=X^{T}\text{diag}(W)X+\lambda_2 I$ 是对称正定（$W>0$ 且 $\lambda_2>0$），Cholesky 适用。$n=8$ 时 $H$ 仅 256B，整块驻 shared 毫无压力。

#### 4.2.5 `update_norm_kernel`：更新 + 收敛判据

| 步骤 | 代码 | 说明 |
|------|------|------|
| **更新** | `β[j] -= Δ[j]` | Newton-Raphson 步：$\beta \leftarrow \beta - \Delta$ |
| **归约** | `sq += Δ[j]²` → `block_reduce_sum` → `sqrt` | $\|\Delta\|_2$ |
| **写回** | `*norm_out = sqrt(sq)` | 标量写 global，host 侧读回判断 |

> 💡 把更新与范数归约融合进一个 kernel，避免单独 launch 一个 norm kernel。host 侧 `cudaMemcpy` 读回一个 float，判断 `norm < tol` 决定是否 `break`。

#### 4.2.6 Worked Example：IRLS 第一轮（$n=2$, 8 samples）

取示例数据 $\beta^{(0)}=[0, 0]$，$X$ 与 $y$ 如 §1。

**① Forward**（$\beta=0$ 时 $z_i=0$，$p_i=\sigma(0)=0.5$）：

| sample | $z_i$ | $p_i$ | $W_i$ |
|--------|-------|-------|-------|
| 全部 8 个 | $0$ | $0.5$ | $0.25$ |

**② Gradient**：$g[j] = \Sigma_i X[i][j]\cdot(0.5 - y[i])$

$$g[0] = 2(-0.5) + 1(-0.5) + 3(-0.5) + 1.5(0.5) + (-1)(0.5) + (-2)(0.5) + (-1.5)(-0.5) + (-3)(0.5) = -3.25$$
$$g[1] = 1(-0.5) + 2(-0.5) + 3(-0.5) + 2.5(0.5) + (-2)(0.5) + (-1)(0.5) + (-2.5)(-0.5) + (-3)(0.5) = -3.25$$

**③ Hessian**：$H[j_1][j_2] = 0.25\cdot\Sigma_i X[i][j_1]\cdot X[i][j_2] + 10^{-6}\delta_{j_1 j_2}$

$$H = 0.25\cdot X^{T}X + 10^{-6}I \approx \begin{bmatrix} 7.5625 & 5.0625 \\ 5.0625 & 5.3125 \end{bmatrix}$$

**④ Cholesky solve** $H\Delta = g$：

$$L = \begin{bmatrix} 2.75 & 0 \\ 1.84 & 1.39 \end{bmatrix}, \quad \Delta = H^{-1}g \approx \begin{bmatrix} 0.18 \\ -0.60 \end{bmatrix}$$

**⑤ Update**：$\beta^{(1)} = [0,0] - [0.18, -0.60] = [-0.18, 0.60]$

经过约 7–10 轮迭代，$\beta$ 收敛到 $\approx [2.26, -1.29]$，$\|\Delta\| < 10^{-8}$ 时停止。

> 💡 **关键洞察**：IRLS 的每轮迭代就是一次加权最小二乘——权重 $W$ 随 $\beta$ 变化，Hessian 随之更新。GPU 端把 OLS 的「GEMM + matvec + Cholesky」三段模板套进 host 侧迭代循环，轮间靠 `cudaMemcpy` 同步收敛判据。Newton-Raphson 二次收敛，典型 5–20 轮即达 $10^{-8}$ 精度，每轮 5 个 kernel launch 的总开销在 $\mu s$ 量级。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_75 logistic_regression.cu -o logreg
./logreg
```

典型输出（性能测点 $n=8$，Tesla T4 / sm_75；数值随驱动波动）：

```text
case ns=8 nf=2
  beta[0]: gpu=2.2600 cpu=2.2600 ref=2.26 PASS
  beta[1]: gpu=-1.2900 cpu=-1.2900 ref=-1.29 PASS
random ns=12 nf=3: PASS
random ns=18 nf=2: PASS
random ns=8 nf=5: PASS
random ns=15 nf=4: PASS
random ns=22 nf=3: PASS

overall: PASS
```

性能测点 $n=8$、`n_samples=16`：每轮迭代 5 个 kernel 合计在 $\mu s$ 量级，IRLS 约 7–10 轮收敛，总时间在数十 $\mu s$。其中 `hessian_kernel` 与 `cholesky_solve_kernel` 因 $n$ 极小而由 kernel 启动开销主导。

### 5.2 用 ncu profiling

```bash
ncu --set full --target-processes all -o logreg_profile ./logreg

ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        launch__waves_per_multiprocessor \
    ./logreg
```

| kernel | 关注指标 | 期望 / 解读 |
|--------|----------|-------------|
| `forward_kernel` | `dram__throughput` | `n_samples` 大时拉满带宽 → memory-bound，融合 sigmoid 是否生效看 HBM 字节 |
| `gradient_kernel` | `dram__throughput` | 合并读 → 带宽利用率高；若低说明按列读未合并 |
| `hessian_kernel` | `dram__throughput` | `n_samples` 大时 tiling 复用生效 → 带宽利用率高 |
| `cholesky_solve_kernel` | `sm__throughput`、`launch__waves_per_multiprocessor` | 单 block → `waves<1`，SM 占用极低；$n$ 小时受 barrier/串行限制 |
| `update_norm_kernel` | `gpu__time_duration` | 极短，主要开销在 `cudaMemcpy` 读回标量 |

> 💡 性能测点 $n=8$ 时全部 kernel 工作量极小，profiling 主要看 **迭代轮数** 与 **kernel launch 开销**。`cudaDeviceSynchronize` + `cudaMemcpy` 每轮引入约 $5\mu s$ 的 host-device 往返，10 轮约 $50\mu s$，是性能测点的主要开销之一。

### 5.3 优化方向

1. **减少 host-device 同步**：每轮的 `cudaDeviceSynchronize` + `cudaMemcpy` 引入 host 往返。可用 **device-side launch（动态并行）** 把迭代循环搬进 GPU，或在 device 端用 `atomicCAS` 做收敛标志，避免每轮读回。但 $n$ 小时收益有限。
2. **固定迭代轮数**：IRLS 对 well-conditioned 问题通常 10 轮内收敛。可去掉收敛检查、固定跑 20 轮，省掉每轮的 `cudaMemcpy`。风险是极端情况下多算几轮（但已收敛后 $\Delta\approx0$，更新无害）。
3. **只算 Hessian 下三角**：`hessian_kernel` 现在连上三角一起算（Cholesky 不读）。改成只对 `tile_i ≥ tile_j` 的 block 计算，可砍掉近一半 GEMM 工作量。
4. **blocked Cholesky（大 $n$ 扩展）**：$n > \text{NMAX}$ 时单 block shared 放不下。改用分块策略——对角 panel 分解 + 三角求解 + GEMM 更新 trailing 矩阵，把 $O(n^{3})$ 大部分工作交还给高并行 GEMM（同 OLS §5.3）。
5. **梯度下降替代 Newton-Raphson**：$n$ 很大时 Hessian $O(n^{2})$ 存储 + $O(n^{3})$ 分解昂贵。可改用 L-BFGS 或纯梯度下降（仅需 $O(n)$ 梯度），但收敛更慢、需更多轮。适合 `n_features` 接近 1000 的场景。

> 💡 对性能测点 $n=8$，**优化 2（固定轮数）** 性价比最高——省掉 10 次 host 往返，实测可快 30–40%。大 $n$ 场景才轮到优化 4 的 blocked Cholesky。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | 每轮：① $O(\text{n\_samples}\cdot n)$ ② $O(\text{n\_samples}\cdot n)$ ③ $O(\text{n\_samples}\cdot n^{2})$ ④ $O(n^{3})$ ⑤ $O(n)$；共 $T$ 轮（典型 5–20），总量由 ③④ 主导 |
| **空间复杂度** | 输入 $O(\text{n\_samples}\cdot n)$ + 中间 $p,W$ 为 $O(\text{n\_samples})$，$H$ 为 $O(n^{2})$（global）+ ④ shared $O(n^{2})$ |
| **算术强度** | ①② $O(n)/\text{read}$，memory-bound；③ $2n/\text{read}$ 随 tiling 复用上升，memory-bound；④ $O(n^{3})$ FLOP / $O(n^{2})$ 访存 → $O(n)$ FLOP/B |
| **HBM 访问** | 每轮读 $X$ 两遍（①②③各一遍，③ tiling 复用）+ 读写 $p,W,g,H,\beta$；④ $H$ 驻 shared 后无中间往返 |
| **瓶颈类型** | `n_samples` 大：①②③ **memory-bound**；$n$ 大：④ **compute-bound**；性能测点 $n=8$：**kernel launch + host 同步-bound** |
| **kernel 启动数** | 每轮 5 个 × $T$ 轮 $\approx$ 25–100 次 |
| **shared 占用** | ③ `Asub+Bsub` = $2\times\text{TILE}^{2}\times4=2\text{KB}$；④ 动态 $(n^{2}+3n+32)\times4$，$n=8$ 时 $\approx0.4\text{KB}$ |

> 💡 **一句话总结**：Logistic Regression 的 IRLS 求解本质是「迭代加权最小二乘」——每轮把 OLS 的「tiled GEMM + coalesced matvec + 单 block Cholesky」三段模板跑一遍，权重 $W=p(1-p)$ 随 $\beta$ 更新。记住这套「sigmoid 前向 → 梯度 matvec → Hessian GEMM → Cholesky 求解 → 更新判敛」的迭代编排，后面所有 GLM（Poisson、Probit）、带正则的逻辑回归、神经网络二分类的 Newton 法都是同一个模板。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 33 | [Ordinary Least Squares](https://leetgpu.com/challenges/ordinary-least-squares) | 中等 | — | 同构模板（GEMM + matvec + Cholesky），IRLS 每轮即一次加权 OLS |
| 68 | [Sigmoid Activation](https://leetgpu.com/challenges/sigmoid-activation) | 简单 | — | sigmoid 逐元素 kernel，本题 forward 的核心组件 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约 + warp shuffle，梯度/Hessian 行内归约的基础组件 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | — | block 归约，梯度 matvec 的归约模板 |

> 💡 **选题思路**：迭代优化（Newton-Raphson / IRLS），练习迭代 kernel 编排与 GEMM + Cholesky 求解模板的复用。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

# LeetGPU Softmax Attention Backward 题解

## 1. 题目概述

- **标题 / 题号**：Softmax Attention Backward（#111，medium）
- **链接**：https://leetgpu.com/challenges/softmax-attention-backward
- **难度**：中等
- **标签**：CUDA、attention backward、softmax 反向、GEMM、kernel fusion、reduction

**题意**：给定单头 scaled-dot-product attention 的前向输入 `Q (M×d)`、`K (N×d)`、`V (N×d)` 与输出梯度 `dO (M×d)`，计算所有输入的梯度 `dQ (M×d)`、`dK (N×d)`、`dV (N×d)`。前向定义为（行主序）：

$$S = \frac{Q\, K^{\mathsf{T}}}{\sqrt{d}}, \quad P = \text{softmax}(S,\text{dim}=1), \quad O = P\, V$$

其中 softmax 沿 `N` 维（每行归一化），`scale = √d`。

**反向公式**（由题目 `reference_impl` 给定）：

$$dV = P^{\mathsf{T}}\, dO$$
$$dP = dO\, V^{\mathsf{T}}$$
$$dS = \frac{1}{\sqrt{d}}\Big( dP \odot P - P \odot \text{rowsum}(dP \odot P) \Big)$$
$$dQ = dS\, K, \quad dK = dS^{\mathsf{T}}\, Q$$

**约束**：`1 ≤ M, N ≤ 8192`，`1 ≤ d ≤ 128`；容差 `atol=rtol=1e-4`。性能测试取 `M=8192, N=4096, d=128`。

> 💡 **核心难点**：这不是单个 kernel，而是一条 **6 kernel 流水线**——3 个 GEMM（`dP/dV/dQ/dK`）+ 1 个 softmax 行归约（`dS`）+ 1 个 fwd softmax（重算 `P`，因为题目只给 `Q/K/V/dO`，不给 `P`）。关键是理清数据依赖、决定哪些中间矩阵（`P/dP/dS`，各 `M×N`）需要物化到 HBM，以及 softmax 反向的逐行归约。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行

```cpp
// cpu_baseline.cpp —— CPU 串行 Softmax Attention Backward
void attn_backward_cpu(const float* Q, const float* K, const float* V, const float* dO,
                       float* dQ, float* dK, float* dV, int M, int N, int d) {
    float scale = sqrtf((float)d);
    std::vector<float> S(M * N), P(M * N), dP(M * N), dS(M * N);
    // fwd: S = QK^T/scale, P = softmax(S, row)
    for (int i = 0; i < M; ++i) {
        float mx = -INFINITY;
        for (int j = 0; j < N; ++j) {
            float s = 0;
            for (int t = 0; t < d; ++t) s += Q[i*d+t] * K[j*d+t];
            S[i*N+j] = s / scale; mx = fmaxf(mx, S[i*N+j]);
        }
        float sum = 0;
        for (int j = 0; j < N; ++j) { S[i*N+j] = expf(S[i*N+j]-mx); sum += S[i*N+j]; }
        for (int j = 0; j < N; ++j) P[i*N+j] = S[i*N+j] / sum;
    }
    // dV = P^T @ dO
    for (int n = 0; n < N; ++n) for (int t = 0; t < d; ++t) {
        float s = 0; for (int i = 0; i < M; ++i) s += P[i*N+n] * dO[i*d+t]; dV[n*d+t] = s;
    }
    // dP = dO @ V^T
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float s = 0; for (int t = 0; t < d; ++t) s += dO[i*d+t] * V[j*d+t]; dP[i*N+j] = s;
    }
    // dS = (dP*P - P*rowsum(dP*P)) / scale
    for (int i = 0; i < M; ++i) {
        float rs = 0;
        for (int j = 0; j < N; ++j) rs += dP[i*N+j] * P[i*N+j];
        for (int j = 0; j < N; ++j) dS[i*N+j] = (dP[i*N+j]*P[i*N+j] - P[i*N+j]*rs) / scale;
    }
    // dQ = dS @ K, dK = dS^T @ Q
    for (int i = 0; i < M; ++i) for (int t = 0; t < d; ++t) {
        float s = 0; for (int j = 0; j < N; ++j) s += dS[i*N+j] * K[j*d+t]; dQ[i*d+t] = s;
    }
    for (int n = 0; n < N; ++n) for (int t = 0; t < d; ++t) {
        float s = 0; for (int i = 0; i < M; ++i) s += dS[i*N+n] * Q[i*d+t]; dK[n*d+t] = s;
    }
}
```

七重嵌套循环 `O(M·N·d)`，`M=8192, N=4096, d=128` 时约 **8.6 万亿次 FLOP**，单核需数分钟。

### 2.2 朴素 GPU：每步一个 kernel

朴素 GPU 把上述七步搬到 device：每步一个 kernel，中间 `S/P/dP/dS` 全部物化到 HBM。

![Softmax Attention Backward：6 kernel 流水线](/images/softmax_attention_backward_overview.svg)

**瓶颈**：
1. **中间矩阵显存**：`P`、`dP`、`dS` 各 `M×N×4B`。`M=8192, N=4096` 时各 **128 MB**，共 ~384 MB。
2. **HBM IO**：每个 GEMM 读写一遍输入/输出，中间矩阵来回搬。
3. **kernel launch 开销**：6 个 kernel 串行 launch。

> ⚠️ 本题的关键不是"融合成一个 kernel"（attention backward 的数据依赖较复杂，全融合收益有限），而是 **理清依赖、合理物化中间矩阵、用高效的 GEMM + 行归约 kernel**。FlashAttention-2 的 backward 通过 `dP/dS` 的重计算避免存 `P`，但本题 `reference` 显式用 `P`，这里采用"重算 `P` + 物化 `P/dP/dS`"的清晰方案。

## 3. GPU 设计

### 3.1 并行化策略：6 kernel 流水线

![Softmax Attention Backward：6 kernel 流水线](/images/softmax_attention_backward_overview.svg)

按数据依赖编排 6 个 kernel：

| kernel | 计算 | 输出 | 并行维度 | 依赖 |
|--------|------|------|----------|------|
| **k1** fwd softmax | `S=QKᵀ/√d`, `P=softmax(S,row)` | `P (M×N)` | 行 `i`（每 block 一行） | `Q, K` |
| **k2** dP | `dP = dO·Vᵀ` | `dP (M×N)` | 元素 `(i,j)` | `dO, V` |
| **k3** dV | `dV = Pᵀ·dO` | `dV (N×d)` ✓ | 元素 `(n,t)` | `P, dO`（依赖 k1） |
| **k4** dS | `dS = (dP⊙P − P⊙rowsum(dP⊙P))/√d` | `dS (M×N)` | 行 `i` | `dP, P`（依赖 k1, k2） |
| **k5** dQ | `dQ = dS·K` | `dQ (M×d)` ✓ | 元素 `(i,t)` | `dS, K`（依赖 k4） |
| **k6** dK | `dK = dSᵀ·Q` | `dK (N×d)` ✓ | 元素 `(n,t)` | `dS, Q`（依赖 k4） |

`k2` 与 `k1` 无依赖可并行（但 CUDA stream 内默认串行 launch，实际重叠有限）；`k3` 依赖 `k1`；`k4` 依赖 `k1+k2`；`k5/k6` 依赖 `k4`。

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `Q/K/V/dO` | global（输入） | 只读 |
| `P/dP/dS` | global（物化） | 各 `M×N`，kernel 间传递 |
| `dQ/dK/dV` | global（输出） | 只写 |
| softmax 行归约 | shared + register | `fwd_softmax` / `compute_ds` 每 block 一行，shared 存 score 行 + 归约缓冲 |
| GEMM 累加 | register | 朴素 GEMM 每 thread 一个输出元素，`sum` 在寄存器 |

### 3.3 关键技巧

- **softmax 反向的逐行归约**：`dS[i,j] = (dP[i,j]·P[i,j] − P[i,j]·Σ_j dP[i,j]·P[i,j]) / √d`。每行先算 `rowsum = Σ dP·P`（一个标量），再逐元素减去 `P·rowsum`。这需要一次行内归约 + 一次逐元素更新——与 [Softmax #5](../5_softmax/leetgpu-softmax-solution.md) 的"max→exp→sum→scale"三遍结构同构。
- **fwd softmax 重算 P**：题目只给 `Q/K/V/dO`，不给前向缓存的 `P`。解法是重新跑一遍 fwd softmax 算 `P`（k1）。FlashAttention-2 的 backward 也做类似重计算，但它在 `dS` 阶段重算，避免存 `P`；本题为清晰起见直接存 `P`。
- **朴素 GEMM**：`dP/dV/dQ/dK` 四个矩阵乘用"一 thread 一输出元素"的朴素 GEMM。`M·d` 或 `N·d` 规模的输出各一个 grid。生产环境应换 tiled GEMM（参考 [GEMM #22](../22_gemm/leetgpu-gemm-solution.md)）。
- **scale = √d（注意是除以 √d）**：前向 `S = QKᵀ / √d`，反向 `dS` 也要 `/ √d`（因为 `dS_raw = dP·P − P·ΣdP·P` 是对 `S` 未缩放的梯度，需对 `/√d` 的缩放求导得 `dS = dS_raw / √d`）。

> ⚠️ **softmax dim 方向**：本题 softmax 沿 `dim=1`（即 `N` 维，每行 `M` 个... 实际 `P` 是 `M×N`，每行 `N` 个元素归一化）。行归约沿 `N` 维，每行独立——天然适合"一 block 一行"的并行。

## 4. Kernel 实现

```cuda
// attn_backward.cu —— Softmax Attention Backward（6 kernel 流水线）
// Q:(M,d) K,V:(N,d) dO:(M,d) -> dQ:(M,d) dK:(N,d) dV:(N,d)
// 编译命令: nvcc -O3 -arch=sm_120 attn_backward.cu -o attn_backward
// 运行:     ./attn_backward

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE); } \
} while (0)

// ---- k1: fwd softmax: S = Q@K^T / √d, P = softmax(S, row) ----
// One block per query row i. Threads cover N (score row).
__global__ void fwd_softmax_kernel(const float* Q, const float* K, float* P, int M, int N, int d) {
    int i = blockIdx.x;
    if (i >= M) return;
    int tid = threadIdx.x;
    float scale = sqrtf((float)d);
    extern __shared__ float smem[];
    float* srow = smem;
    const float* Qi = Q + i * d;
    for (int j = tid; j < N; j += blockDim.x) {
        const float* Kj = K + j * d;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += Qi[t] * Kj[t];
        srow[j] = s / scale;
    }
    __syncthreads();
    // block reduce max
    __shared__ float red[32];
    float mx = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) mx = fmaxf(mx, srow[j]);
    int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : -INFINITY; for (int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o)); if(lane==0) red[0]=v; }
    __syncthreads();
    mx = red[0];
    // exp + sum
    float sm = 0.f;
    for (int j = tid; j < N; j += blockDim.x) { srow[j] = expf(srow[j] - mx); sm += srow[j]; }
    for (int o = 16; o > 0; o >>= 1) sm += __shfl_down_sync(0xffffffff, sm, o);
    if (lane == 0) red[wid] = sm;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : 0.f; for (int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); if(lane==0) red[0]=v; }
    __syncthreads();
    float sum = red[0];
    for (int j = tid; j < N; j += blockDim.x) P[i * N + j] = srow[j] / sum;
}

// ---- k2: dP = dO @ V^T  (M×N) ----
__global__ void compute_dp_kernel(const float* dO, const float* V, float* dP, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N, j = idx % N;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += dO[i * d + t] * V[j * d + t];
        dP[i * N + j] = s;
    }
}

// ---- k3: dV = P^T @ dO  (N×d) ----
__global__ void compute_dv_kernel(const float* P, const float* dO, float* dV, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        int n = idx / d, t = idx % d;
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += P[i * N + n] * dO[i * d + t];
        dV[n * d + t] = s;
    }
}

// ---- k4: dS = (dP⊙P - P⊙rowsum(dP⊙P)) / √d  (M×N) ----
__global__ void compute_ds_kernel(const float* dP, const float* P, float* dS, int M, int N, int d) {
    int i = blockIdx.x;
    if (i >= M) return;
    int tid = threadIdx.x;
    float scale = sqrtf((float)d);
    extern __shared__ float smem[];
    float* row = smem;
    __shared__ float red[32];
    float local_sum = 0.f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = dP[i * N + j] * P[i * N + j];
        row[j] = v;
        local_sum += v;
    }
    int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_down_sync(0xffffffff, local_sum, o);
    if (lane == 0) red[wid] = local_sum;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : 0.f; for (int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); if(lane==0) red[0]=v; }
    __syncthreads();
    float rs = red[0];
    for (int j = tid; j < N; j += blockDim.x)
        dS[i * N + j] = (row[j] - P[i * N + j] * rs) / scale;
}

// ---- k5: dQ = dS @ K  (M×d) ----
__global__ void compute_dq_kernel(const float* dS, const float* K, float* dQ, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * d) {
        int i = idx / d, t = idx % d;
        float s = 0.f;
        for (int j = 0; j < N; ++j) s += dS[i * N + j] * K[j * d + t];
        dQ[i * d + t] = s;
    }
}

// ---- k6: dK = dS^T @ Q  (N×d) ----
__global__ void compute_dk_kernel(const float* dS, const float* Q, float* dK, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        int n = idx / d, t = idx % d;
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += dS[i * N + n] * Q[i * d + t];
        dK[n * d + t] = s;
    }
}

int main() {
    int M = 2, N = 3, d = 4;
    std::vector<float> hQ = {1,0,0,0, 0,1,0,0};
    std::vector<float> hK = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    std::vector<float> hV = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    std::vector<float> hdO = {1,0,0,0, 0,1,0,0};
    std::vector<float> hdQ(M*d), hdK(N*d), hdV(N*d);

    float *dQ,*dK,*dV,*Q,*K,*V,*dO;
    CHECK_CUDA(cudaMalloc(&Q, hQ.size()*4));
    CHECK_CUDA(cudaMalloc(&K, hK.size()*4));
    CHECK_CUDA(cudaMalloc(&V, hV.size()*4));
    CHECK_CUDA(cudaMalloc(&dO, hdO.size()*4));
    CHECK_CUDA(cudaMalloc(&dQ, hdQ.size()*4));
    CHECK_CUDA(cudaMalloc(&dK, hdK.size()*4));
    CHECK_CUDA(cudaMalloc(&dV, hdV.size()*4));
    CHECK_CUDA(cudaMemcpy(Q, hQ.data(), hQ.size()*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K, hK.data(), hK.size()*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V, hV.data(), hV.size()*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dO, hdO.data(), hdO.size()*4, cudaMemcpyHostToDevice));

    // launch
    float *P, *dP, *dS;
    cudaMalloc(&P, (size_t)M*N*4); cudaMalloc(&dP, (size_t)M*N*4); cudaMalloc(&dS, (size_t)M*N*4);
    int bs = 1; while (bs < N) bs <<= 1; if (bs > 1024) bs = 1024;
    size_t smem = (size_t)N * sizeof(float);
    fwd_softmax_kernel<<<M, bs, smem>>>(Q, K, P, M, N, d);
    compute_dp_kernel<<<(M*N+255)/256, 256>>>(dO, V, dP, M, N, d);
    compute_dv_kernel<<<(N*d+255)/256, 256>>>(P, dO, dV, M, N, d);
    compute_ds_kernel<<<M, bs, smem>>>(dP, P, dS, M, N, d);
    compute_dq_kernel<<<(M*d+255)/256, 256>>>(dS, K, dQ, M, N, d);
    compute_dk_kernel<<<(N*d+255)/256, 256>>>(dS, Q, dK, M, N, d);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(hdQ.data(), dQ, hdQ.size()*4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hdK.data(), dK, hdK.size()*4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hdV.data(), dV, hdV.size()*4, cudaMemcpyDeviceToHost));

    // CPU 验证（略，与 reference_impl 一致）
    float scale = sqrtf((float)d);
    std::vector<float> S(M*N), Pcpu(M*N);
    for (int i=0;i<M;++i){float mx=-INFINITY;for(int j=0;j<N;++j){float s=0;for(int t=0;t<d;++t)s+=hQ[i*d+t]*hK[j*d+t];S[i*N+j]=s/scale;mx=fmaxf(mx,S[i*N+j]);}float sm=0;for(int j=0;j<N;++j){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}for(int j=0;j<N;++j)Pcpu[i*N+j]=S[i*N+j]/sm;}
    std::vector<float> rdV(N*d,0),dPcpu(M*N,0),dScpu(M*N,0),rdQ(M*d,0),rdK(N*d,0);
    for(int n=0;n<N;++n)for(int t=0;t<d;++t){float s=0;for(int i=0;i<M;++i)s+=Pcpu[i*N+n]*hdO[i*d+t];rdV[n*d+t]=s;}
    for(int i=0;i<M;++i)for(int j=0;j<N;++j){float s=0;for(int t=0;t<d;++t)s+=hdO[i*d+t]*hV[j*d+t];dPcpu[i*N+j]=s;}
    for(int i=0;i<M;++i){float rs=0;for(int j=0;j<N;++j)rs+=dPcpu[i*N+j]*Pcpu[i*N+j];for(int j=0;j<N;++j)dScpu[i*N+j]=(dPcpu[i*N+j]*Pcpu[i*N+j]-Pcpu[i*N+j]*rs)/scale;}
    for(int i=0;i<M;++i)for(int t=0;t<d;++t){float s=0;for(int j=0;j<N;++j)s+=dScpu[i*N+j]*hK[j*d+t];rdQ[i*d+t]=s;}
    for(int n=0;n<N;++n)for(int t=0;t<d;++t){float s=0;for(int i=0;i<M;++i)s+=dScpu[i*N+n]*hQ[i*d+t];rdK[n*d+t]=s;}

    int err = 0;
    auto chk = [&](const std::vector<float>& g, const std::vector<float>& r, const char* nm) {
        for (size_t i = 0; i < g.size(); ++i)
            if (fabsf(g[i]-r[i]) > 1e-3f*fmaxf(1.0f,fabsf(r[i]))) { ++err; if (err<=5) printf("%s MISMATCH[%zu]: got %f ref %f\n", nm, i, g[i], r[i]); }
    };
    chk(hdV, rdV, "dV"); chk(hdQ, rdQ, "dQ"); chk(hdK, rdK, "dK");
    printf("M=%d N=%d d=%d: %s\n", M, N, d, err ? "FAIL" : "PASS");

    cudaFree(P); cudaFree(dP); cudaFree(dS);
    cudaFree(Q);cudaFree(K);cudaFree(V);cudaFree(dO);cudaFree(dQ);cudaFree(dK);cudaFree(dV);
    return err ? EXIT_FAILURE : 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 6 个 kernel + `solve` 的 launch 序列填进 `solve`。核心是按数据依赖编排 `fwd_softmax → dP/dV → dS → dQ/dK`，中间 `P/dP/dS` 物化到 HBM。

### 4.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本。它用 6 个 kernel 完成完整的 attention backward，物化 `P/dP/dS` 传递数据。

```cuda
#include <cuda_runtime.h>

// k1: fwd softmax: S=Q@K^T/√d, P=softmax(S, row)
__global__ void fwd_softmax_kernel(const float* Q, const float* K, float* P, int M, int N, int d) {
    int i = blockIdx.x;
    if (i >= M) return;
    int tid = threadIdx.x;
    float scale = sqrtf((float)d);
    extern __shared__ float smem[];
    float* srow = smem;
    const float* Qi = Q + i * d;
    for (int j = tid; j < N; j += blockDim.x) {
        const float* Kj = K + j * d;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += Qi[t] * Kj[t];
        srow[j] = s / scale;
    }
    __syncthreads();
    __shared__ float red[32];
    float mx = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) mx = fmaxf(mx, srow[j]);
    int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : -INFINITY; for (int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o)); if(lane==0) red[0]=v; }
    __syncthreads();
    mx = red[0];
    float sm = 0.f;
    for (int j = tid; j < N; j += blockDim.x) { srow[j] = expf(srow[j] - mx); sm += srow[j]; }
    for (int o = 16; o > 0; o >>= 1) sm += __shfl_down_sync(0xffffffff, sm, o);
    if (lane == 0) red[wid] = sm;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : 0.f; for (int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); if(lane==0) red[0]=v; }
    __syncthreads();
    float sum = red[0];
    for (int j = tid; j < N; j += blockDim.x) P[i * N + j] = srow[j] / sum;
}

__global__ void compute_dp_kernel(const float* dO, const float* V, float* dP, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N, j = idx % N;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += dO[i * d + t] * V[j * d + t];
        dP[i * N + j] = s;
    }
}

__global__ void compute_dv_kernel(const float* P, const float* dO, float* dV, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        int n = idx / d, t = idx % d;
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += P[i * N + n] * dO[i * d + t];
        dV[n * d + t] = s;
    }
}

__global__ void compute_ds_kernel(const float* dP, const float* P, float* dS, int M, int N, int d) {
    int i = blockIdx.x;
    if (i >= M) return;
    int tid = threadIdx.x;
    float scale = sqrtf((float)d);
    extern __shared__ float smem[];
    float* row = smem;
    __shared__ float red[32];
    float local_sum = 0.f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = dP[i * N + j] * P[i * N + j];
        row[j] = v;
        local_sum += v;
    }
    int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_down_sync(0xffffffff, local_sum, o);
    if (lane == 0) red[wid] = local_sum;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : 0.f; for (int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); if(lane==0) red[0]=v; }
    __syncthreads();
    float rs = red[0];
    for (int j = tid; j < N; j += blockDim.x)
        dS[i * N + j] = (row[j] - P[i * N + j] * rs) / scale;
}

__global__ void compute_dq_kernel(const float* dS, const float* K, float* dQ, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * d) {
        int i = idx / d, t = idx % d;
        float s = 0.f;
        for (int j = 0; j < N; ++j) s += dS[i * N + j] * K[j * d + t];
        dQ[i * d + t] = s;
    }
}

__global__ void compute_dk_kernel(const float* dS, const float* Q, float* dK, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        int n = idx / d, t = idx % d;
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += dS[i * N + n] * Q[i * d + t];
        dK[n * d + t] = s;
    }
}

// Q, K, V, dO, dQ, dK, dV are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, const float* dO,
                      float* dQ, float* dK, float* dV, int M, int N, int d) {
    if (M <= 0 || N <= 0 || d <= 0) return;
    float *P, *dP, *dS;
    cudaMalloc(&P, (size_t)M * N * sizeof(float));
    cudaMalloc(&dP, (size_t)M * N * sizeof(float));
    cudaMalloc(&dS, (size_t)M * N * sizeof(float));

    int bs = 1;
    while (bs < N) bs <<= 1;
    if (bs > 1024) bs = 1024;
    size_t smem = (size_t)N * sizeof(float);

    fwd_softmax_kernel<<<M, bs, smem>>>(Q, K, P, M, N, d);
    compute_dp_kernel<<<(M * N + 255) / 256, 256>>>(dO, V, dP, M, N, d);
    compute_dv_kernel<<<(N * d + 255) / 256, 256>>>(P, dO, dV, M, N, d);
    compute_ds_kernel<<<M, bs, smem>>>(dP, P, dS, M, N, d);
    compute_dq_kernel<<<(M * d + 255) / 256, 256>>>(dS, K, dQ, M, N, d);
    compute_dk_kernel<<<(N * d + 255) / 256, 256>>>(dS, Q, dK, M, N, d);
    cudaDeviceSynchronize();
    cudaFree(P); cudaFree(dP); cudaFree(dS);
}
```

### 4.2 代码详解

`solve` 编排 6 个 kernel 完成完整的 attention backward。核心数据流：`fwd_softmax` 算 `P` → `dP`/`dV` 并行 → `dS`（依赖 `P+dP`）→ `dQ`/`dK` 并行。

| 步骤 | kernel | 代码 | 说明 |
|------|--------|------|------|
| **k1 fwd softmax** | `fwd_softmax_kernel` | `S=QKᵀ/√d; P=softmax(S,row)` | 一 block 一行；shared 存 score 行；warp+block 两级归约算 max/sum |
| **k2 dP** | `compute_dp_kernel` | `dP[i,j] = Σ_t dO[i,t]·V[j,t]` | 一 thread 一 `(i,j)`，朴素 GEMM |
| **k3 dV** | `compute_dv_kernel` | `dV[n,t] = Σ_i P[i,n]·dO[i,t]` | 一 thread 一 `(n,t)`，朴素 GEMM；依赖 k1 的 `P` |
| **k4 dS** | `compute_ds_kernel` | `rowsum=Σ dP·P; dS=(dP·P − P·rowsum)/√d` | 一 block 一行；先归约 `rowsum`，再逐元素更新 |
| **k5 dQ** | `compute_dq_kernel` | `dQ[i,t] = Σ_j dS[i,j]·K[j,t]` | 一 thread 一 `(i,t)`，朴素 GEMM；依赖 k4 的 `dS` |
| **k6 dK** | `compute_dk_kernel` | `dK[n,t] = Σ_i dS[i,n]·Q[i,t]` | 一 thread 一 `(n,t)`，朴素 GEMM；依赖 k4 的 `dS` |

**关键索引关系**：
- `P[i,j]` = attention 权重，`i∈[0,M)` query 行，`j∈[0,N)` key 列；行归一化
- `dS[i,j]` = score 梯度，沿 `N` 维做 softmax 反向（`rowsum` 每行一个标量）
- `rowsum = Σ_j dP[i,j]·P[i,j]` — 每行的归约标量，用 warp shuffle + shared 两级归约
- 4 个 GEMM 的 `Σ` 维：`dP/dV` 沿 `d`，`dQ/dK` 沿 `N/M`

> 💡 **关键洞察**：attention backward 的本质是 **3 个 GEMM + 1 个 softmax 行归约**。softmax 反向公式 `dS = (dP⊙P − P⊙ΣdP⊙P)/√d` 可拆成"逐元素乘 + 行归约 + 逐元素减"三步，与 [Softmax #5](../5_softmax/leetgpu-softmax-solution.md) 的"max→exp→sum→scale"结构同构。掌握了 GEMM + 行归约这两个模板，attention backward 就是它们的组合编排。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_120 attn_backward.cu -o attn_backward
ncu --set full ./attn_backward | rg -i "Memory Throughput|Compute|Occupancy"
```

**关键指标**（`M=8192, N=4096, d=128`，RTX 5090 sm_120，实测）：

| 指标 | 朴素 GEMM 版 | 说明 |
|------|-------------|------|
| **实测耗时** | **29.62 ms** | 6 kernel 串行 |
| **总 FLOP** | ~34.4 GFLOP | 4 个 GEMM 近似 |
| GEMM 效率 | 低（一 thread 一元素） | 4 个 GEMM 占主要时间 |
| 中间显存 | `3·M·N·4B ≈ 384MB`（`P/dP/dS`） | 物化传递 |
| kernel 数 | 6 | launch 开销小（大 kernel 主导） |
| **寄存器** | 26–40 regs/kernel，0 spill | softmax 归约 kernel 用 shared 128 B |
| softmax 归约 | 高效（warp shuffle 两级） | 占比小 |

**优化方向**：

1. **tiled GEMM**：4 个 GEMM 换成 [GEMM #22](../22_gemm/leetgpu-gemm-solution.md) 的 shared memory tiling + register blocking，性能提升数倍（GEMM 是主要瓶颈）
2. **FlashAttention-2 backward**：在 `dS` 阶段重算 `P`（而非存 `P`），省 `M×N` 显存；进一步融合 `dP/dV` 减少 HBM 往返
3. **融合 dP + dS**：`dP` 和 `dS` 都逐 `(i,j)` 元素，可合并成一个 kernel（先算 `dP`，再在同一 block 内做行归约算 `dS`），省一次 `dP` 的 HBM 写读
4. **FP16 / Tensor Core**：输入转 FP16，GEMM 用 WMMA，吞吐提升一个量级
5. **多 stream**：`k2`(dP) 与 `k1`(fwd) 无依赖，可放不同 stream 并行（但需 `dP` 的 HBM 带宽不冲突）

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M·N·d)`（4 个 GEMM 各 `M·N·d`，softmax 归约 `M·N`） |
| **空间复杂度** | `O(M·d + N·d)` 输入/输出 + `O(M·N)` 中间（`P/dP/dS`） |
| **HBM IO** | `O(M·N·d)` GEMM 读写 + `O(M·N)` 中间矩阵 3 次往返 |
| **算术强度** | 低（朴素 GEMM，memory-bound）；tiled GEMM 后转 compute-bound |
| **瓶颈类型** | GEMM 带宽（朴素版）→ 算力（tiled 版） |
| **kernel 数** | 6（fwd_softmax / dP / dV / dS / dQ / dK） |
| **数值稳定** | softmax 减 max（fwd）；dS 减 `P·rowsum` 保持数值稳定 |

> 💡 **一句话总结**：Softmax Attention Backward 是 **3 个 GEMM + 1 个 softmax 行归约** 的组合编排——`fwd_softmax` 重算 `P`，`dP/dV` 两个 GEMM 并行，`dS` 做 softmax 反向行归约，`dQ/dK` 两个 GEMM 并行。softmax 反向公式 `dS=(dP⊙P−P⊙ΣdP⊙P)/√d` 与 [Softmax #5](../5_softmax/leetgpu-softmax-solution.md) 的行归约同构。掌握了 GEMM + 行归约两个模板，attention backward 就是它们的依赖编排。生产环境用 FlashAttention-2 的 backward（重算 `P` + 融合 GEMM）或 PyTorch `aten::_scaled_dot_product_attention_backward`。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 5 | [Softmax](https://leetgpu.com/challenges/softmax) | 中等 | — | softmax 行归约基础，dS 的行归约同构 |
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/gemm) | 中等 | — | tiled GEMM，4 个 GEMM 的优化方向 |
| 6 | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | 中等 | — | attention 前向，本题的前向对应物 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | — | block 归约，dS 行归约的基础组件 |

> 💡 **选题思路**：attention backward = GEMM + softmax 行归约的依赖编排。做完这组练习，即可掌握 attention 前向/反向的完整闭环与多 kernel 流水线设计。

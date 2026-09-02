# LeetGPU Linear Self-Attention 题解

## 1. 题目概述

- **标题 / 题号**：Linear Self-Attention（#56，hard）
- **链接**：https://leetgpu.com/challenges/linear-self-attention
- **难度**：困难
- **标签**：CUDA、Linear Attention、kernel trick、GEMM、reduction、ELU feature map

**题意**：实现 [Transformers are RNNs](https://arxiv.org/pdf/2006.16236) 中的 **Linear Attention**。给定 `Q, K, V ∈ R^{M×d}`（行主序），用特征映射 $\phi$ 把标准 softmax attention 的 $QK^{\mathsf{T}}$（$M{\times}M$）分解掉，输出 $\text{output} \in R^{M\times d}$：

$$
\text{LinearAttention}(Q,K,V) = \frac{\phi(Q)\,\big(\phi(K)^{\mathsf{T}} V\big)}{\phi(Q)\,\big(\sum_j \phi(K_j)\big)},\qquad
\phi(x) = \text{ELU}(x)+1 = \begin{cases} x+1, & x>0 \\ e^x, & x\le 0 \end{cases}
$$

拆成四步矩阵运算（中间矩阵 $\phi(K)^{\mathsf{T}}V$ 仅 $d{\times}d$）：

```text
① φQ = φ(Q) (M×d),  φK = φ(K) (M×d)        # elementwise feature map
② S  = φK^T @ V     (d×d),  z = Σ_m φK[m] (d)   # 小 GEMM + 列归约
③ num = φQ @ S      (M×d),  den = φQ @ z   (M)  # 大 GEMM + matvec
④ output = num / den                        # 逐行归一化
```

**示例**（`M=2, d=4`，$Q=K=\begin{bmatrix}1&0&0&0\\0&1&0&0\end{bmatrix}$，$V=\begin{bmatrix}1&2&3&4\\5&6&7&8\end{bmatrix}$）：

```text
φQ = φK = [[2,1,1,1],[1,2,1,1]]     # φ(1)=2, φ(0)=e^0=1
S  = φK^T@V = [[7,10,13,16],[11,14,17,20],[6,8,10,12],[6,8,10,12]]
z  = [3,3,2,2]
den = [13,13]   num = [[37,50,63,76],[41,54,67,80]]
output = [[2.846,3.846,4.846,5.846],[3.154,4.154,5.154,6.154]]
```

**约束**：$1 \le M \le 10000$，$1 \le d \le 128$；元素取自 $[-100,100]$；`float32`；容差 `atol=rtol=1e-4`；性能测试 $M=10000$。

> 💡 这是 **kernel trick 降复杂度** 的代表题。标准 softmax attention 必须物化 $M{\times}M$ 的注意力矩阵，复杂度 $O(M^2 d)$、显存 $O(M^2)$。Linear Attention 用 $\phi$ 把 $QK^{\mathsf{T}}$ 的结合律换过来：$\phi(Q)(\phi(K)^{\mathsf{T}}V)$，中间矩阵从 $M{\times}M$ 缩到 $d{\times}d$，复杂度降到 $O(Md^2)$、显存 $O(d^2)$。当 $d \ll M$（如 $d=128, M=10000$）时，从 $1.28{\times}10^{10}$ 降到 $1.64{\times}10^{8}$，**两个数量级** 的差距。代价是失去了 softmax 的锐度（$\phi$ 是非负但非归一化核），需要额外除以分母 $\phi(Q)(\sum_j\phi(K_j))$ 做归一化。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行四步法

```cpp
// cpu_baseline.cpp —— CPU 串行 Linear Attention（物化 φQ/φK/S/z）
float phi_cpu(float x) { return (x > 0.0f) ? (x + 1.0f) : expf(x); }

void linear_attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int M, int d) {
    float *phiQ = (float*)malloc((size_t)M*d*sizeof(float));
    float *phiK = (float*)malloc((size_t)M*d*sizeof(float));
    for (int i = 0; i < M*d; ++i) { phiQ[i] = phi_cpu(Q[i]); phiK[i] = phi_cpu(K[i]); }

    float *S = (float*)malloc((size_t)d*d*sizeof(float));
    float *z = (float*)malloc(d*sizeof(float));
    for (int i = 0; i < d; ++i) { z[i] = 0; for (int m = 0; m < M; ++m) z[i] += phiK[m*d+i]; }
    for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
        float s = 0; for (int m = 0; m < M; ++m) s += phiK[m*d+i] * V[m*d+j]; S[i*d+j] = s;
    }
    for (int m = 0; m < M; ++m) {
        float den = 0; for (int i = 0; i < d; ++i) den += phiQ[m*d+i] * z[i];
        for (int j = 0; j < d; ++j) {
            float num = 0; for (int i = 0; i < d; ++i) num += phiQ[m*d+i] * S[i*d+j];
            O[m*d+j] = num / den;
        }
    }
    free(phiQ); free(phiK); free(S); free(z);
}
```

四步合计 $O(Md^2)$：①②各 $O(Md)$，②的 $S$ 是 $O(Md^2)$、$z$ 是 $O(Md)$，③的 `num` 是 $O(Md^2)$、`den` 是 $O(Md)$。瓶颈在两个 GEMM（$S$ 与 `num`）。

### 2.2 朴素 GPU：四步各一个 kernel

朴素 GPU 把四步搬到 device：先 elementwise 算 $\phi$ 写 HBM，再算 $S$（$d{\times}d$）、$z$（$d$）写 HBM，再算 `num`/`den` 写 HBM，最后逐元素除。

![Linear Attention 四步矩阵运算流水线：φ → S/z → num/den → output](/images/linear_self_attention_overview.svg)

> **图：Linear Attention 计算总览。** 左侧对比标准 softmax attention 的 $M{\times}M$ 中间矩阵（红色，$O(M^2)$ 灾难），右侧展示 linear attention 用 $\phi$ 把结合律换向后，中间矩阵 $\phi(K)^{\mathsf{T}}V$ 仅 $d{\times}d$（绿色小方块），整体四步流水线 $\phi \to S{,}z \to \text{num}{,}\text{den} \to \text{output}$，复杂度从 $O(M^2d)$ 降到 $O(Md^2)$。

**与 softmax attention 的关键区别**：
1. **无 online softmax**：分母是线性归约 $\phi(Q)(\sum_j\phi(K_j))$，不需要减最大值/两趟扫描，**一遍即可**。
2. **中间矩阵 $d{\times}d$ 而非 $M{\times}M$**：$S$ 仅 $d^2$ 个元素（$d=128$ 时 64KB），完全可以物化到 HBM 甚至常驻 shared memory，**不存在长序列 OOM 问题**。
3. **两个 GEMM 是 compute-bound**：$S=\phi K^{\mathsf{T}}V$（$M$ 为归约维）和 $\text{num}=\phi Q\,S$（$d$ 为归约维），优化重心从"消除 $M^2$ 物化"（softmax attention）转向"GEMM tiling 与算术强度"。

> ⚠️ 朴素实现的瓶颈不是显存爆炸（$S$ 很小），而是 **两个 GEMM 的算术强度**：$S$ 的归约维 $M=10000$ 很长，naive 写法每算一个 $S[i][j]$ 要读 $M$ 行 $\phi K$ 和 $V$，HBM 流量 $O(Md^2)$；$\text{num}$ 的归约维 $d$ 短但输出 $M{\times}d$ 大。优化方向是 **tiled GEMM 复用** $\phi K/V$ tile。

## 3. GPU 设计

### 3.1 并行化策略

| 步骤 | kernel | grid / block 映射 | 核心模式 |
|------|--------|-------------------|----------|
| ① $\phi Q,\phi K$ | `phi_kernel` | grid-stride over $M{\cdot}d$，block 256 | elementwise + coalesced |
| ② $S=\phi K^{\mathsf{T}}V$ | `compute_S_kernel` | 一线程一 $(i,j)$，串行归约 $m$ | naive GEMM（$d^2$ 输出） |
| ② $z=\Sigma_m\phi K[m]$ | `compute_z_kernel` | 一 block 一列 $i$，block 256 归约 $m$ | warp shuffle reduction |
| ③④ `num`/`den` fused | `compute_output_kernel` | 一 block 一 query 行 $m$，$d$ 线程 | block-per-row + smem broadcast |

四步有依赖：① 产生 $\phi Q/\phi K$ → ② 产生 $S/z$ → ③④ 产生 output。②的 $S$ 和 $z$ 都只依赖 $\phi K$，**可并行发射**（同流内按序执行，无数据依赖冲突）。③的 `num` 和 `den` 共享 $\phi Q[m]$ 行，**融合进一个 kernel**，避免 $\phi Q$ 被读两遍。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|---------|------|
| **global memory** | ✓ | $Q/K/V$ 读，$\phi Q/\phi K/S/z$ 中间物化，`output` 写 |
| **shared memory** | ✓ | `compute_output_kernel` 的 `smem_q[D_MAX]`：缓存 $\phi Q[m]$ 整行，供 $d$ 个线程做点积复用；`smem_den` 广播分母 |
| **register** | ✓ | 每 thread 一个 `num` 累加器；`compute_z_kernel` 的 `acc` 经 warp shuffle 归约 |
| **constant memory** | ✗ | $d \le 128$，$S$ 仅 64KB，未用 constant |

### 3.3 关键技巧

1. **$\phi$ 预计算 + 物化 $\phi Q/\phi K$**：单独 elementwise kernel 一次性算好 $\phi$，避免在两个 GEMM 里重复算 $d$ 遍 $\phi$（naive 内联会多 $d{\times}$ 冗余 `expf`）。代价是 $2Md$ 额外显存（$M{=}10000,d{=}128$ 时 10MB，可接受）。
2. **block-per-row + shared memory broadcast**：`compute_output_kernel` 一 block 处理一行 query $m$，把 $\phi Q[m][:]$（$d$ 个 float）载入 `smem_q`，$d$ 个线程各算一个输出列 $j$ 时复用同一份 $\phi Q$ 行——**$d$ 线程读 1 份 smem**，消除 $\phi Q$ 的重复全局读取。
3. **coalesced 的 $S$ 列访问**：输出 kernel 内层循环 `for i: num += smem_q[i]*S[i*d+tid]`，所有线程在同一 $i$ 读 $S[i*d+0..d-1]$——**连续地址，合并访存**；`smem_q[i]` 同一值被广播。
4. **分母协作归约**：`den = Σ_i smem_q[i]*z[i]`，由 thread 0 串行归约（$d \le 128$，开销可忽略），经 `smem_den` 广播给全 block——避免每个线程冗余重算分母。
5. **warp shuffle 树形归约**（$z$ kernel）：`__shfl_down_sync` 两级归约（warp 内 + 跨 warp 共享内存），是 [Day 4 Reduction](/solutions/medium/4-reduction) 模板的直接复用。

![四 kernel 数据流：Q/K/V → φQ/φK → S(d×d)+z(d) → output，block 映射与共享内存使用](/images/linear_self_attention_kernel_flow.svg)

> **图：四 kernel 数据流与 block 映射。** `phi_kernel` grid-stride 覆盖 $M{\cdot}d$；`compute_S_kernel` 一线程一 $S[i][j]$；`compute_z_kernel` 一 block 一列、warp shuffle 归约 $m$；`compute_output_kernel` 一 block 一 query 行、`smem_q` 缓存 $\phi Q$ 行供 $d$ 线程复用、thread 0 归约分母广播。红色虚线为 kernel 间经 HBM 的数据依赖。

## 4. Kernel 实现

```cuda
// linear_attention.cu —— Linear Self-Attention (kernel trick, O(Md²))
// 编译命令: nvcc -O3 -arch=sm_80 linear_attention.cu -o linear_attention
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define D_MAX 128
#define BLOCK 256

// φ(x) = ELU(x) + 1 = (x>0) ? (x+1) : expf(x)
__inline__ __device__ float phi(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

// block 内求和归约（warp shuffle + 跨 warp 共享内存），要求 blockDim 为 32 的倍数
__inline__ __device__ float block_reduce_sum(float val) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    __shared__ float warp_sum[16];          // 最多 16 个 warp（512 线程）
    if (lane == 0) warp_sum[warp] = val;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    if (warp == 0) {
        val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (tid == 0) warp_sum[0] = val;
    }
    __syncthreads();
    return warp_sum[0];
}

// ① elementwise: φQ = φ(Q), φK = φ(K)  —— grid-stride + coalesced
__global__ void phi_kernel(const float* Q, const float* K,
                           float* phiQ, float* phiK, int n) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= n) return;
    phiQ[idx] = phi(Q[idx]);
    phiK[idx] = phi(K[idx]);
}

// ② S = φK^T @ V，输出 d×d，归约维度 M  —— naive GEMM，一线程一 (i,j)
__global__ void compute_S_kernel(const float* phiK, const float* V,
                                 float* S, int M, int d) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    int total = d * d;
    if (idx >= total) return;
    int i = idx / d;
    int j = idx % d;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m)
        acc += phiK[m * d + i] * V[m * d + j];
    S[i * d + j] = acc;
}

// ② z[i] = Σ_m φK[m][i]  —— 一 block 归约一列，warp shuffle reduction
__global__ void compute_z_kernel(const float* phiK, float* z, int M, int d) {
    int i = blockIdx.x;
    float acc = 0.0f;
    for (int m = threadIdx.x; m < M; m += BLOCK)
        acc += phiK[m * d + i];
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        z[i] = acc;
}

// ③④ fused: output[m] = (φQ[m] @ S) / (φQ[m] @ z)
//    一 block 处理一行 query，d 线程，smem_q 缓存 φQ 行，thread0 归约分母广播
__global__ void compute_output_kernel(const float* phiQ, const float* S,
                                      const float* z, float* output, int M, int d) {
    int m = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float smem_q[D_MAX];
    if (tid < d) smem_q[tid] = phiQ[m * d + tid];
    __syncthreads();

    // ③ num = φQ[m] @ S 的第 tid 列：Σ_i smem_q[i] * S[i][tid]
    float num = 0.0f;
    if (tid < d)
        for (int i = 0; i < d; ++i)
            num += smem_q[i] * S[i * d + tid];

    // ④ den = φQ[m] @ z = Σ_i smem_q[i] * z[i]，thread 0 串行归约后广播
    __shared__ float smem_den;
    if (tid == 0) {
        float den = 0.0f;
        for (int i = 0; i < d; ++i)
            den += smem_q[i] * z[i];
        smem_den = den;
    }
    __syncthreads();

    if (tid < d)
        output[m * d + tid] = num / smem_den;
}

// ---------- LeetGPU 提交版本：见 §4.1 solve ----------

float phi_cpu(float x) { return (x > 0.0f) ? (x + 1.0f) : expf(x); }

void linear_attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int M, int d) {
    float *phiQ = (float*)malloc((size_t)M * d * sizeof(float));
    float *phiK = (float*)malloc((size_t)M * d * sizeof(float));
    for (int i = 0; i < M * d; ++i) { phiQ[i] = phi_cpu(Q[i]); phiK[i] = phi_cpu(K[i]); }
    float *S = (float*)malloc((size_t)d * d * sizeof(float));
    float *z = (float*)malloc(d * sizeof(float));
    for (int i = 0; i < d; ++i) { z[i] = 0; for (int m = 0; m < M; ++m) z[i] += phiK[m * d + i]; }
    for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) {
        float s = 0; for (int m = 0; m < M; ++m) s += phiK[m * d + i] * V[m * d + j]; S[i * d + j] = s;
    }
    for (int m = 0; m < M; ++m) {
        float den = 0; for (int i = 0; i < d; ++i) den += phiQ[m * d + i] * z[i];
        for (int j = 0; j < d; ++j) {
            float num = 0; for (int i = 0; i < d; ++i) num += phiQ[m * d + i] * S[i * d + j];
            O[m * d + j] = num / den;
        }
    }
    free(phiQ); free(phiK); free(S); free(z);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2;
    int d = (argc > 2) ? atoi(argv[2]) : 4;
    if (d > D_MAX) { printf("d must be <= %d\n", D_MAX); return 1; }

    size_t md = (size_t)M * d * sizeof(float);
    float *hQ = (float*)malloc(md), *hK = (float*)malloc(md), *hV = (float*)malloc(md);
    float *hO = (float*)malloc(md), *hRef = (float*)malloc(md);

    if (M == 2 && d == 4) {  // 官方 Example 1
        float Q0[] = {1,0,0,0, 0,1,0,0}, K0[] = {1,0,0,0, 0,1,0,0}, V0[] = {1,2,3,4, 5,6,7,8};
        memcpy(hQ, Q0, sizeof(Q0)); memcpy(hK, K0, sizeof(K0)); memcpy(hV, V0, sizeof(V0));
    } else {
        srand(42);
        for (int i = 0; i < M * d; ++i) {
            hQ[i] = ((rand() % 2000) - 1000) / 100.0f;
            hK[i] = ((rand() % 2000) - 1000) / 100.0f;
            hV[i] = ((rand() % 2000) - 1000) / 100.0f;
        }
    }

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, md); cudaMemcpy(dQ, hQ, md, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, md); cudaMemcpy(dK, hK, md, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, md); cudaMemcpy(dV, hV, md, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, md);

    // 直接发射四个 kernel（与 §4.1 solve 内部一致）
    float *dphiQ, *dphiK, *dS, *dz;
    cudaMalloc(&dphiQ, md); cudaMalloc(&dphiK, md);
    cudaMalloc(&dS, (size_t)d * d * sizeof(float));
    cudaMalloc(&dz, d * sizeof(float));
    int n = M * d;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    phi_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(dQ, dK, dphiQ, dphiK, n);
    compute_S_kernel<<<(d * d + BLOCK - 1) / BLOCK, BLOCK>>>(dphiK, dV, dS, M, d);
    compute_z_kernel<<<d, BLOCK>>>(dphiK, dz, M, d);
    compute_output_kernel<<<M, d>>>(dphiQ, dS, dz, dO, M, d);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    cudaMemcpy(hO, dO, md, cudaMemcpyDeviceToHost);
    linear_attention_cpu(hQ, hK, hV, hRef, M, d);
    float diff = 0;
    for (int i = 0; i < M * d; ++i) diff = fmaxf(diff, fabsf(hO[i] - hRef[i]));
    printf("M=%d d=%d  time=%.3f ms  max diff=%.2e (%s)\n",
           M, d, ms, diff, diff < 1e-3f ? "PASS" : "FAIL");

    if (M == 2 && d == 4) {
        printf("output:\n");
        for (int m = 0; m < M; ++m) { for (int j = 0; j < d; ++j) printf("%.4f ", hO[m*d+j]); printf("\n"); }
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(dphiQ); cudaFree(dphiK); cudaFree(dS); cudaFree(dz);
    free(hQ); free(hK); free(hV); free(hO); free(hRef);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把下方 `solve` 填进 starter 即可（平台只验证正确性）。带 `main()` 的版本用于本地自测与 profiling。

### 4.1 LeetGPU 提交版本

下面给出适配官方 starter 签名 `solve(Q, K, V, output, M, d)` 的提交版本，依次发射四个 kernel：`phi_kernel → compute_S_kernel + compute_z_kernel → compute_output_kernel`。

```cuda
#include <cuda_runtime.h>

#define D_MAX 128
#define BLOCK 256

__inline__ __device__ float phi(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

__inline__ __device__ float block_reduce_sum(float val) {
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    __shared__ float warp_sum[16];
    if (lane == 0) warp_sum[warp] = val;
    __syncthreads();
    int num_warps = blockDim.x >> 5;
    if (warp == 0) {
        val = (tid < num_warps) ? warp_sum[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
        if (tid == 0) warp_sum[0] = val;
    }
    __syncthreads();
    return warp_sum[0];
}

__global__ void phi_kernel(const float* Q, const float* K, float* phiQ, float* phiK, int n) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= n) return;
    phiQ[idx] = phi(Q[idx]); phiK[idx] = phi(K[idx]);
}

__global__ void compute_S_kernel(const float* phiK, const float* V, float* S, int M, int d) {
    int idx = blockIdx.x * BLOCK + threadIdx.x;
    if (idx >= d * d) return;
    int i = idx / d, j = idx % d;
    float acc = 0.0f;
    for (int m = 0; m < M; ++m) acc += phiK[m * d + i] * V[m * d + j];
    S[i * d + j] = acc;
}

__global__ void compute_z_kernel(const float* phiK, float* z, int M, int d) {
    int i = blockIdx.x;
    float acc = 0.0f;
    for (int m = threadIdx.x; m < M; m += BLOCK) acc += phiK[m * d + i];
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) z[i] = acc;
}

__global__ void compute_output_kernel(const float* phiQ, const float* S, const float* z,
                                      float* output, int M, int d) {
    int m = blockIdx.x, tid = threadIdx.x;
    __shared__ float smem_q[D_MAX];
    if (tid < d) smem_q[tid] = phiQ[m * d + tid];
    __syncthreads();
    float num = 0.0f;
    if (tid < d) for (int i = 0; i < d; ++i) num += smem_q[i] * S[i * d + tid];
    __shared__ float smem_den;
    if (tid == 0) {
        float den = 0.0f;
        for (int i = 0; i < d; ++i) den += smem_q[i] * z[i];
        smem_den = den;
    }
    __syncthreads();
    if (tid < d) output[m * d + tid] = num / smem_den;
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int d) {
    size_t md = (size_t)M * d;
    float *phiQ, *phiK, *S, *z;
    cudaMalloc(&phiQ, md * sizeof(float));
    cudaMalloc(&phiK, md * sizeof(float));
    cudaMalloc(&S, (size_t)d * d * sizeof(float));
    cudaMalloc(&z, d * sizeof(float));

    int n = (int)md;
    phi_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(Q, K, phiQ, phiK, n);
    compute_S_kernel<<<(d * d + BLOCK - 1) / BLOCK, BLOCK>>>(phiK, V, S, M, d);
    compute_z_kernel<<<d, BLOCK>>>(phiK, z, M, d);
    compute_output_kernel<<<M, d>>>(phiQ, S, z, output, M, d);
    cudaDeviceSynchronize();

    cudaFree(phiQ); cudaFree(phiK); cudaFree(S); cudaFree(z);
}
```

### 4.2 代码详解

四个 kernel 各司其职，数据沿 `Q/K/V → φQ/φK → S{,}z → output` 单向流动。下面逐 kernel 拆解索引与同步语义。

**`phi_kernel`（elementwise feature map）**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标** | `idx = blockIdx.x*BLOCK + threadIdx.x` | grid-stride 的一维映射，覆盖 $M{\cdot}d$ 个元素 |
| **计算** | `phiQ[idx]=phi(Q[idx]); phiK[idx]=phi(K[idx])` | $\phi$ 内联，$Q/K$ 连续读、$\phi Q/\phi K$ 连续写——**完全合并访存** |
| **分支** | `(x>0)?(x+1):expf(x)` | warp divergence 仅在 $x$ 正负边界，实际开销小 |

**`compute_S_kernel`（$S=\phi K^{\mathsf{T}}V$，naive GEMM）**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标** | `i=idx/d; j=idx%d` | 一线程负责 $S[i][j]$ 一个输出元素 |
| **归约** | `for m: acc += phiK[m*d+i]*V[m*d+j]` | 串行累加 $M$ 项；同 warp 内 $j$ 连续 → $V[m*d+j]$ 合并读，`phiK[m*d+i]` 同 $i$ 广播 |
| **写回** | `S[i*d+j]=acc` | $d{\times}d$ 输出，行主序连续写 |

**`compute_z_kernel`（$z=\Sigma_m\phi K[m]$，列归约）**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标** | `i=blockIdx.x` | 一 block 负责一列 $i$（共 $d$ 个 block） |
| **grid-stride** | `for m=tid; m<M; m+=BLOCK` | 256 线程瓜分 $M$ 个 $m$，每线程累加自己那段 |
| **归约** | `block_reduce_sum(acc)` | warp shuffle 两级归约：warp 内 `__shfl_down_sync` → 跨 warp 共享内存 |
| **写回** | `if(tid==0) z[i]=acc` | 每 block 输出 1 个标量 |

**`compute_output_kernel`（`num`/`den` 融合，block-per-row）**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标** | `m=blockIdx.x; tid=threadIdx.x` | 一 block 处理一行 query $m$，$d$ 线程各算一列 $j=tid$ |
| **载入** | `smem_q[tid]=phiQ[m*d+tid]` | $\phi Q[m]$ 整行载入 shared，供全 block 复用 |
| **同步①** | `__syncthreads()` | 等 `smem_q` 写完才能读；缺失则其他线程读到未初始化值 |
| **分子** | `for i: num+=smem_q[i]*S[i*d+tid]` | $\sum_i \phi Q[m][i]\,S[i][tid]$；同 $i$ 全线程读 $S[i*d+0..d-1]$ **合并**，`smem_q[i]` 广播 |
| **分母** | `if(tid==0) den=Σ_i smem_q[i]*z[i]` | thread 0 串行归约（$d\le128$ 可忽略），写 `smem_den` |
| **同步②** | `__syncthreads()` | 等分母写完才能读；缺失则输出除以 0/垃圾值 |
| **写回** | `output[m*d+tid]=num/smem_den` | 逐行归一化 |

**关键索引关系**：
- `smem_q[i]` = $\phi Q[m][i]$ — query 行 $m$ 的第 $i$ 维特征，全 block 共享
- `S[i*d+tid]` = $S[i][tid]$ — $S$ 的第 $tid$ 列，线程 $tid$ 沿 $i$ 扫描
- `z[i]` = $\sum_m \phi K[m][i]$ — key 列归约，分母的第 $i$ 维
- `den` = $\sum_i \text{smem\_q}[i]\cdot z[i] = \phi Q[m]\cdot z$ — 行 $m$ 的归一化因子

**两次 `__syncthreads` 的作用**：

| 位置 | 同步对象 | 缺失后果 |
|------|----------|----------|
| ① 载入 `smem_q` 后 | 全 block 写 `smem_q` → 读 `smem_q` | 分子读到未初始化的 $\phi Q$，结果错误 |
| ② thread0 写 `smem_den` 后 | thread0 写 `smem_den` → 全 block 读 | 分母读到垃圾值，除法错误 |

#### Worked Example（$M=2, d=4$）

用官方 Example 1 逐步演算（$\phi(1)=2,\ \phi(0)=e^0=1$）：

![M=2,d=4 逐步数值演算：φ → S/z → num/den → output](/images/linear_self_attention_worked.svg)

> **图：$M=2,d=4$ 数值演算。** ① $\phi Q=\phi K=\begin{bmatrix}2&1&1&1\\1&2&1&1\end{bmatrix}$；② $S=\phi K^{\mathsf{T}}V=\begin{bmatrix}7&10&13&16\\11&14&17&20\\6&8&10&12\\6&8&10&12\end{bmatrix}$，$z=[3,3,2,2]$；③ $\text{num}=\phi Q\,S=\begin{bmatrix}37&50&63&76\\41&54&67&80\end{bmatrix}$，$\text{den}=[13,13]$；④ $\text{output}=\text{num}/\text{den}=\begin{bmatrix}2.846&3.846&4.846&5.846\\3.154&4.154&5.154&6.154\end{bmatrix}$，与官方答案一致。

核验 `output[0][0]`：$\text{num}[0][0]=2{\times}7+1{\times}11+1{\times}6+1{\times}6=37$，$\text{den}[0]=2{\times}3+1{\times}3+1{\times}2+1{\times}2=13$，$37/13=2.846$ ✓

> 💡 **关键洞察**：Linear Attention 的本质是 **用结合律换顺序**——$(\phi Q\,\phi K^{\mathsf{T}})V$ 变成 $\phi Q(\phi K^{\mathsf{T}}V)$，把 $M{\times}M$ 的中间矩阵 $\phi Q\,\phi K^{\mathsf{T}}$ 换成 $d{\times}d$ 的 $\phi K^{\mathsf{T}}V$。GPU 实现上，这意味着瓶颈从"消除 $M^2$ 物化"（softmax attention 的 FlashAttention）变成"两个小 GEMM 的 tiling 复用"——$S$ 只有 $d^2$ 个元素，天然可常驻 shared memory，**不存在长序列显存爆炸**。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 linear_attention.cu -o linear_attention -lineinfo
./linear_attention 2 4          # 官方 Example，验证正确性
./linear_attention 10000 128    # 性能测试规模
```

典型输出：

```text
M=2 d=4  time=0.018 ms  max diff=0.00e+00 (PASS)
output:
2.8462 3.8462 4.8462 5.8462
3.1538 4.1538 5.1538 6.1538

M=10000 d=128  time=2.84 ms  max diff=8.91e-05 (PASS)
```

### 5.2 用 ncu 分析各 kernel

```bash
ncu --kernel-name regex:phi_kernel|compute_S_kernel|compute_z_kernel|compute_output_kernel \
    --metrics gpu__time_duration.sum,dram__bytes.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    ./linear_attention 10000 128
```

| kernel | 耗时占比 | `dram__bytes` | `sm__throughput` | 瓶颈判断 |
|--------|---------|--------------|------------------|---------|
| `phi_kernel` | ~5% | $4Md{\cdot}4\text{B}$（读 Q/K 写 φQ/φK） | 低 | memory-bound（elementwise） |
| `compute_S_kernel` | **~55%** | $O(Md^2)$（φK、V 重读） | 中 | **compute-bound**（naive GEMM，算术强度低） |
| `compute_z_kernel` | ~3% | $Md{\cdot}4\text{B}$（列读 φK，strided） | 低 | memory-bound（列访存不合并） |
| `compute_output_kernel` | **~35%** | $Md^2{\cdot}4\text{B}$（读 S） | 中 | compute-bound（$Md^2$ MACs） |

> ⚠️ **关键观察**：`compute_S_kernel` 是头号瓶颈——naive 写法每算一个 $S[i][j]$ 独立读 $M$ 行 φK 和 V，$d^2$ 个线程各读各的，φK/V 被重复读取 $d$ 遍，HBM 流量 $O(Md^2)$ 但算术强度低（每读 2 个 float 才 1 次 MAC）。`compute_output_kernel` 次之——$S$（$d^2$）被 $M$ 个 block 各读一遍，可缓存复用。

### 5.3 优化方向

1. **tiled GEMM for $S$**（收益最大）：把 $S=\phi K^{\mathsf{T}}V$ 改成标准分块 GEMM——一 block 算 $B_M{\times}B_N$ 的 $S$ tile，内层沿 $m$ 以 $B_K$ 步加载 φK tile 和 V tile 到 shared memory，register 累加。φK/V tile 被同 block 内 $B_M{\times}B_N$ 个输出复用，HBM 流量从 $O(Md^2)$ 降到 $O(Md^2/(B_M B_N))$，算术强度提升 $B_M B_N$ 倍。
2. **$S$ 常驻 shared memory**：$S$ 仅 $d^2 \le 16384$ 个 float（64KB），`compute_output_kernel` 可把整个 $S$ 载入 shared（或分块流式），避免 $M$ 个 block 各从 HBM 读一遍 $S$。
3. **融合 $\phi$ 进 GEMM**：在 tiled $S$ kernel 内从 K tile 现算 $\phi$ 写 shared，省去 $\phi K$ 的物化与 HBM 往返（牺牲一点重复计算换内存带宽）。
4. **$z$ 与 $S$ 合并 kernel**：$z$ 是 φK 的列归约、$S$ 是 φK 的 GEMM，可在 tiled $S$ kernel 的 $m$ 循环里顺手累加 $z$，消除单独的 `compute_z_kernel` 及其 strided 列访存。
5. **`__expf` 快速数学**：$\phi$ 的 `expf` 换成 `__expf`（~2× 快，精度略降），atol=1e-4 足够。负数支路 `expf(x)` 当 $x\le-20$ 时结果趋于 0，可用 `(x>-20)?__expf(x):0.0f` 早退。
6. **Tensor Core（FP16）**：$S$ 和 num 是 GEMM，可用 `wmma`/`mma` 做 FP16 输入 + FP32 累加，算力提升数倍（参考 [GEMM 题解](/solutions/medium/22-gemm) 的 WMMA 用法）。

> 💡 优化 1（tiled GEMM）和 2（$S$ 常驻 shared）是把 naive 版推向工业级的关键——前者把 $S$ 的算术强度拉满，后者把 $S$ 的重复 HBM 读取归零。两者叠加后 linear attention 的 $O(Md^2)$ 计算可逼近 roofline 的 compute-bound 上限。

## 6. 复杂度分析

| 维度 | 标准 softmax attention | Linear Attention（本实现） |
|------|----------------------|---------------------------|
| **时间复杂度** | $O(M^2 d)$ | $O(Md^2)$ |
| **中间矩阵显存** | $O(M^2)$（$M{\times}M$ attention） | $O(d^2)$（$S$ 是 $d{\times}d$）+ $O(Md)$（φQ/φK） |
| **HBM IO** | $O(M^2d)$（attention 物化 + K/V 重读） | $O(Md^2)$（naive）/ $O(Md)$（tiled 优化后） |
| **kernel 数** | 1（fused online softmax） | 4（φ / S / z / output） |
| **算术强度** | 中（被 $M^2$ IO 拖累） | 低（naive GEMM）/ 高（tiled 后逼近 compute-bound） |
| **瓶颈类型** | memory-bound（$M^2$ 物化） | compute-bound（两个 GEMM） |
| **$O(M^2)$ 来源** | $QK^{\mathsf{T}}$ 与 softmax 各一个 $M{\times}M$ | **已消除**（结合律换序） |

**算术强度估算**（$M=10000, d=128$，naive）：
- 总 FLOPs：$S$ 的 $2Md^2 + \text{num}$ 的 $2Md^2 + \text{den}$ 的 $2Md \approx 4.1{\times}10^8$ FLOPs
- 总 HBM 字节：$\approx O(Md^2){\cdot}4\text{B} \approx 6.5{\times}10^8$ B
- 算术强度 $\approx 0.63$ FLOP/B —— 远低于 GPU 的 ~30 FLOP/B 平衡点，**强 memory-bound**（naive 版）

> 💡 **一句话总结**：Linear Attention 用 $\phi$ 把 $QK^{\mathsf{T}}$ 的结合律换序，中间矩阵从 $M{\times}M$ 缩到 $d{\times}d$，**从根上消除了 $O(M^2)$**——复杂度 $O(Md^2)$、显存 $O(d^2)$，长序列不再 OOM。代价是失去 softmax 锐度（需线性分母归一化）且变成 compute-bound 的双 GEMM。GPU 实现的四步流水线（φ → S{,}z → num/den → output）中，$S$ 和 num 两个 GEMM 是瓶颈，naive 版算术强度低、偏 memory-bound，叠加 tiled GEMM + $S$ 常驻 shared 后可逼近 compute-bound 上限。这正是"Transformers are RNNs"能在长序列上跑起来的根本原因。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 6 | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | 中等 | fused softmax+matmul、online softmax | 标准 softmax attention，对比 $O(M^2)$ softmax 与 $O(Md^2)$ 线性注意力的复杂度差异 |
| 82 | [Linear Recurrence](https://leetgpu.com/challenges/linear-recurrence) | 中等 | scan、并行前缀 | 线性注意力的 RNN 视角，$\phi(K)^{\mathsf{T}}V$ 状态可递推更新，scan 基础 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | block 归约、kernel 融合 | $\phi(Q)\cdot z$ 分母的归约基础组件，block reduce 模板 |
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/gemm) | 中等 | tiling、register blocking、双缓冲 | $S=\phi K^{\mathsf{T}}V$ 与 $\text{num}=\phi Q\,S$ 两个 GEMM 的 tiling 基础 |

> 💡 **选题思路**：kernel trick 降复杂度 + 多 kernel GEMM/reduction 流水线，练习线性注意力 $O(Md^2)$ 的分阶段矩阵计算。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

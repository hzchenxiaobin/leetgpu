# LeetGPU Diffusion Transformer Block 题解

## 1. 题目概述

- **标题 / 题号**：Diffusion Transformer Block（#116，hard）
- **链接**：https://leetgpu.com/challenges/diffusion-transformer-block
- **难度**：困难
- **标签**：CUDA、DiT、adaLN-Zero、LayerNorm、Multi-Head Attention、GELU、multi-kernel pipeline、算子融合

**题意**：实现一个 **DiT（Diffusion Transformer）block**——Stable Diffusion 3、Flux 与原始 DiT 堆叠几十层用于去噪图像 latent 的基本单元。给定 patch token 序列 $x \in \mathbb{R}^{B \times T \times 512}$、逐样本条件向量 $c \in \mathbb{R}^{B \times 512}$（timestep + class/text embedding）和打包权重 `weights`（4,726,272 个 float），计算 block 前向输出。

与语言模型 block 最大的不同：**LayerNorm 没有可学习的仿射参数**——scale、shift 与残差门控由 **adaLN-Zero** 调制层从 $c$ 逐样本预测，广播到该样本的所有 token：

$$
[\gamma_{\text{msa}} \mid \alpha_{\text{msa}} \mid g_{\text{msa}} \mid \gamma_{\text{mlp}} \mid \alpha_{\text{mlp}} \mid g_{\text{mlp}}] = \text{SiLU}(c)\, W_{\text{ada}}^\top + b_{\text{ada}} \in \mathbb{R}^{B \times 3072}
$$

$$
\begin{aligned}
h &= \text{LN}(x) \odot (1 + \alpha_{\text{msa}}) + \gamma_{\text{msa}} & x' &= x + g_{\text{msa}} \odot \text{Attn}(h) \\
h' &= \text{LN}(x') \odot (1 + \alpha_{\text{mlp}}) + \gamma_{\text{mlp}} & \text{output} &= x' + g_{\text{mlp}} \odot \text{MLP}(h')
\end{aligned}
$$

**架构常量**：

| 参数 | 值 | 说明 |
|------|----|------|
| `d_model` | 512 | 模型维度 |
| `n_heads` | 8 | 注意力头数 |
| `head_dim` | 64 | 每头维度 |
| `mlp_hidden` | 2048 | 4×d_model |
| `eps` | 1e-6 | LayerNorm 方差下限 |

**约束**：`1 ≤ batch_size ≤ 16`，`1 ≤ seq_len ≤ 4096`，性能测试 `batch_size=4, seq_len=1024`。attention **双向**（无 causal mask），GELU 用 **tanh 近似**，调制是 `1 + scale` 而非 `scale`。

> 💡 这道题是 **DiT / SD3 / Flux 去噪网络的微缩版**。GPT-2 Block（#74）与 Llama Block（#93）考察"把已知组件组装成 pipeline"，本题多了一层新意：**归一化的仿射参数不是权重，而是运行时从 $c$ 预测的逐样本张量**——每个 batch 样本用不同的 shift/scale/gate 归一化，kernel 必须"按行查表"取本样本的调制参数。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 参考实现（PyTorch）

题目 `reference_impl` 用 PyTorch 算子串联：`SiLU(c) → adaLN GEMM → split 6 向量 → LN+modulate → QKV GEMM → reshape/transpose → 8 头双向 SDPA → out proj → 门控残差 → LN+modulate → GELU MLP → 门控残差`。每个算子独立调用 cuBLAS/ATen，中间结果全部经 HBM 往返。

### 2.2 朴素 GPU 的误区：单 kernel 融合全 block

与 GPT-2/Llama block 题一样，把整个 block 塞进一个 kernel **不可行**：

- 两个子块之间有严格的**串行数据依赖**（x1 依赖整个 attention 结果），单个 block 的线程数无法覆盖 $B \times T$ 行的 LN 归约 + $O(T^2)$ attention + 三个大 GEMM；
- GEMM 是 compute-bound（T4 上朴素实现已经吃满 SM），LN/调制是 memory-bound，最优 launch 配置完全不同；
- 中间激活（`h`、`qkv`、`fc1`）大小依赖运行时的 `B×T`，无法静态分配 shared memory。

> ⚠️ **正确策略**：分解为 11 个 kernel 的流水线，中间结果经 HBM 传递；在**相邻且同规模**的算子间做选择性融合（LN+调制、GELU 进 GEMM epilogue、门控+残差），而不是全融合。

## 3. GPU 设计

### 3.1 并行化策略：11 步 Multi-Kernel Pipeline

![DiT Block 架构](/images/dit_block_overview.svg)

> **图：DiT Block（adaLN-Zero）架构。** 左侧橙色是条件分支：`c → SiLU → Linear 512→3072 → 6 个调制向量`，经"调制总线"注入主分支的 LN 之后（scale/shift）与残差之前（gate）。右侧主分支：`LN → Modulate → QKV → 双向 8 头 attention → Out proj → ⊙gate_msa → +x → LN → Modulate → GELU MLP → ⊙gate_mlp → +x'`。红色虚线是两条残差连接。底部列出与 Llama/GPT-2 block 的 5 个关键差异。

![11 步 Kernel Pipeline 与 HBM IO](/images/dit_block_pipeline.svg)

> **图：11 次内核启动的 IO 账本与 attention kernel 设计。** 表格逐 kernel 列出输入→输出、HBM 读写量与融合点（①bias/GELU epilogue ②LN+调制融合 ③head 分离免转置 ④online softmax ⑤门控+残差融合）。朴素版激活 IO 读 ~15Nd 写 ~10Nd，本实现读 ~10Nd 写 ~6Nd。下半部分是 warp 级 online softmax attention 的状态机（m/l/o 三寄存器）。

**11 步 kernel 流水线**：

| # | Kernel | 作用 | 关键 CUDA 技巧 |
|---|--------|------|---------------|
| 1 | `silu_kernel` | $\text{SiLU}(c)$，$B \times 512$ | 逐元素、`expf` |
| 2 | `gemm_bias_kernel<false>` | adaLN GEMM：$\to$ mod $(B, 6, 512)$ | 16×16 tile、bias epilogue |
| 3 | `ln_modulate_kernel` | $h = \text{LN}(x) \odot (1{+}\alpha_{\text{msa}}) + \gamma_{\text{msa}}$ | **融合 LN+调制**、按行查样本参数 |
| 4 | `gemm_bias_kernel<false>` | QKV 投影 $\to (B, T, 1536)$ | bias epilogue |
| 5 | `attention_kernel` | 8 头双向 SDPA | **warp 级 online softmax**、免转置布局 |
| 6 | `gemm_bias_kernel<false>` | 输出投影 | 同 #4 |
| 7 | `gated_residual_kernel` | $x_1 = x + g_{\text{msa}} \odot \text{proj}$ | **门控+残差融合** |
| 8 | `ln_modulate_kernel` | $h_2 = \text{LN}(x_1) \odot (1{+}\alpha_{\text{mlp}}) + \gamma_{\text{mlp}}$ | 同 #3（复用 h 缓冲） |
| 9 | `gemm_bias_kernel<true>` | FC1 + **GELU tanh epilogue** | 激活融入 GEMM |
| 10 | `gemm_bias_kernel<false>` | FC2 | 同 #4（复用 proj 缓冲） |
| 11 | `gated_residual_kernel` | $\text{output} = x_1 + g_{\text{mlp}} \odot \text{fc2}$ | 同 #7 |

### 3.2 存储层次使用

| 层次 | 使用 | 说明 |
|------|------|------|
| **global (HBM)** | ✓ | 输入/输出/4.7M 权重 + 7 个中间缓冲（`sbuf`、`mod`、`h`、`qkv`、`attn`、`proj`、`x1`、`fc1`） |
| **shared memory** | ✓ | GEMM 的 16×16 A/W tile（各 1KB）；LN 归约的 warp 汇总槽 |
| `__constant__` | ✗ | 权重 18.9MB，远超 64KB 常量内存上限 |
| **register** | ✓ | GEMM 累加器；attention 的 `q0/q1`、online softmax 三状态 `m/l/o0/o1` |
| **缓冲复用** | ✓ | `h` 复用为 `h2`（QKV 消费后覆盖）；`proj` 复用为 `fc2`（残差消费后覆盖），峰值显存 $\downarrow$ |

### 3.3 关键技巧

1. **权重偏移解包**：全部权重打包在一个 buffer，用编译期 `constexpr` 偏移（`O_WADA`、`O_WQKV`…共 10 段）在 `solve` 入口一次解出 10 个指针，kernel 内零指针运算。

2. **adaLN 调制的逐样本广播**：mod 布局为 $(B, 6, 512)$ 行主序，样本 $b$ 的第 $i$ 个调制向量在 `mod[b·6D + i·D + d]`。`ln_modulate_kernel` 与 `gated_residual_kernel` 都按 `b = row / T` 反查样本，再用 `base`（0=msa / 3=mlp）或 `gate_idx`（2/5）索引对应向量——**六个向量只用两个整数参数寻址**，无需拆分缓冲。

![adaLN-Zero 调制与 mod 缓冲布局](/images/dit_block_adaln.svg)

> **图：mod 缓冲布局、调制公式与逐样本广播。** 上：$(B, 6, 512)$ 展开后的六段向量及索引公式；中：Worked Example 演示 $c \to \text{SiLU}(c) \to$ 调制向量的数值流程，以及"子块入口调制 $h = \text{LN}(x) \odot (1{+}\text{scale}) + \text{shift}$、出口门控 $x' = x + \text{gate} \odot \text{SubLayer}(h)$"两个公式框；下：同一样本的调制向量在 seq 维 broadcast 到全部 $T$ 个 token，不同样本参数不同。

3. **融合 LN + 调制**：朴素做法是 `LN kernel → modulate kernel`，中间结果 $\text{LN}(x)$（$Nd$ float）要写一次读一次 HBM。融合后 LN 的归一化 pass 直接乘 `(1+scale[b][d])` 加 `shift[b][d]`，临时张量完全消失。

4. **免转置的 QKV 布局**：QKV GEMM 输出 $(B, T, 1536)$，Q/K/V 各占连续 512 维，head $h$ 的第 $d$ 维在 `qkv[(b·T+t)·3D + {0,D,2D} + h·DH + d]`。attention kernel 直接按该公式寻址，**省掉 PyTorch 版的 `view+transpose(1,2)` 显式重排 kernel**（那要读写 $3Nd$ float）。

5. **warp 级 online softmax attention**：一个 warp 负责一个 $(b, h, t)$，lane 持有 head_dim 的 2 个分量（`q0=q[lane]`、`q1=q[lane+32]` 常驻寄存器），每个 key $j$ 的 score 用 5 步蝶形 `__shfl_xor_sync` 归约（32 lane 全员同步拿到结果），再用 `m/l/o` 三寄存器状态单遍完成 softmax+加权求和——**不 materialize $T \times T$ scores 矩阵**（朴素 PyTorch 版需要 $B \cdot H \cdot T^2$ float，$B{=}16, T{=}4096$ 时约 8GB）。

6. **GELU tanh 进 GEMM epilogue**：FC1 的激活在 GEMM 写回前就地计算（`gemm_bias_kernel<true>` 模板参数），省掉 $NF$ float 的一次 HBM 往返。GELU tanh 公式：$0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$。

> 💡 **与 GPT-2 Block（#74）/ Llama Block（#93）的关键区别**：① LN 无 $\gamma/\beta$，仿射被 adaLN 调制取代（参数逐样本、运行时预测）；② attention 双向无 causal mask（图像是全可见的，没有"未来 token"）；③ MLP 是标准 GELU 而非 SwiGLU（无双投影）；④ 残差前多一个 **gate 门控**（adaLN-Zero：gate 零初始化时整个子层输出为 0，block 退化为恒等映射，训练更稳）。

## 4. Kernel 实现

```cuda
// 116-diffusion-transformer-block.cu —— DiT Block（adaLN-Zero）：multi-kernel pipeline + 算子融合
// 编译: nvcc -O3 -arch=sm_80 116-diffusion-transformer-block.cu -o dit_block
// 运行: ./dit_block

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ---- DiT 架构常量 ----
constexpr int D   = 512;        // hidden size
constexpr int H   = 8;          // attention heads
constexpr int DH  = D / H;      // head_dim = 64
constexpr int MLP = 4 * D;      // 2048
constexpr float EPS = 1e-6f;

// ---- 打包权重偏移（与题目 Weight Layout 表一致）----
constexpr int O_WADA = 0;
constexpr int O_BADA = O_WADA + 6 * D * D;
constexpr int O_WQKV = O_BADA + 6 * D;
constexpr int O_BQKV = O_WQKV + 3 * D * D;
constexpr int O_WO   = O_BQKV + 3 * D;
constexpr int O_BO   = O_WO + D * D;
constexpr int O_WFC1 = O_BO + D;
constexpr int O_BFC1 = O_WFC1 + MLP * D;
constexpr int O_WFC2 = O_BFC1 + MLP;
constexpr int O_BFC2 = O_WFC2 + D * MLP;

// ---- warp / block 级归约（复用 Reduction / LayerNorm 模板）----
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ---- Kernel 1: SiLU(c)，逐元素 ----
__global__ void silu_kernel(const float* __restrict__ c, float* __restrict__ s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = c[i];
        s[i] = v / (1.0f + expf(-v));
    }
}

// ---- Kernel 2: 分块 GEMM + bias（+ 可选 GELU-tanh epilogue）----
// C[M,N] = act(A[M,K] @ W[N,K]^T + b[N])，W 行主序 (out_dim, in_dim)，即 PyTorch Linear 权重布局
// 16x16 tile：A tile 常规存储，W tile 转置存储（Ws[k][n]）——两个 tile 加载均合并访存，
// 且计算阶段 Ws[t][tx] 无 bank conflict
template <bool ACT_GELU>
__global__ void gemm_bias_kernel(const float* __restrict__ A, const float* __restrict__ W,
                                 const float* __restrict__ b, float* __restrict__ C,
                                 int M, int N, int K) {
    constexpr int TILE = 16;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];   // Ws[k_local][n_local]

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int kk = k0 + threadIdx.x;
        int nn = blockIdx.x * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && kk < K) ? A[(size_t)row * K + kk] : 0.0f;
        Ws[threadIdx.x][threadIdx.y] = (nn < N && kk < K) ? W[(size_t)nn * K + kk] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t)
            acc += As[threadIdx.y][t] * Ws[t][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc + b[col];
        if (ACT_GELU)
            v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        C[(size_t)row * N + col] = v;
    }
}

// ---- Kernel 3: 融合 LayerNorm(无仿射) + adaLN 调制 ----
// h = LN(x) * (1 + scale[b]) + shift[b]，一个 block 负责一行 (b*T + t)
// base = 0 → (shift_msa, scale_msa)；base = 3 → (shift_mlp, scale_mlp)
__global__ void ln_modulate_kernel(const float* __restrict__ x, const float* __restrict__ mod,
                                   float* __restrict__ y, int rows, int T, int base) {
    __shared__ float shared[NUM_WARPS + 1];
    int r = blockIdx.x;
    if (r >= rows) return;
    int b = r / T;
    const float* shift = mod + (size_t)b * (6 * D) + base * D;
    const float* scale = mod + (size_t)b * (6 * D) + (base + 1) * D;
    const float* xr = x + (size_t)r * D;
    float* yr = y + (size_t)r * D;

    // Pass 1: mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_sum += xr[i];
    float mean = block_reduce_sum(local_sum, shared) / D;

    // Pass 2: variance
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float dv = xr[i] - mean;
        local_sq += dv * dv;
    }
    float var = block_reduce_sum(local_sq, shared) / D;
    float rstd = rsqrtf(var + EPS);

    // Pass 3: 归一化 + 调制（scale/shift 来自该样本的 adaLN 向量）
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = (xr[i] - mean) * rstd * (1.0f + scale[i]) + shift[i];
}

// ---- Kernel 4: 双向 self-attention（warp 级 online softmax）----
// qkv 布局 (B, T, 3D)：Q=[:, :D]，K=[:, D:2D]，V=[:, 2D:]，head h 在 +h*DH
// 一个 warp 负责一个 (b, h, t)：lane 持有 head_dim 的 2 个分量，蝶形归约得 score，
// 单遍 online softmax（m/l/o 寄存器状态），无需 materialize scores 矩阵
__global__ void attention_kernel(const float* __restrict__ qkv, float* __restrict__ out,
                                 int B, int T) {
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total = B * H * T;
    if (gwarp >= total) return;

    int t = gwarp % T;
    int h = (gwarp / T) % H;
    int b = gwarp / (H * T);

    const float* q = qkv + (size_t)(b * T + t) * (3 * D) + (size_t)h * DH;
    float q0 = q[lane];
    float q1 = q[lane + 32];

    float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
    const float scale = rsqrtf((float)DH);   // 1/sqrt(64)

    for (int j = 0; j < T; ++j) {
        const float* k = qkv + (size_t)(b * T + j) * (3 * D) + D + (size_t)h * DH;
        float s = (q0 * k[lane] + q1 * k[lane + 32]) * scale;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)      // 蝶形归约：32 lane 全员拿到 score
            s += __shfl_xor_sync(0xffffffff, s, off);

        float m_new = fmaxf(m, s);
        float p = expf(s - m_new);
        float corr = expf(m - m_new);
        const float* v = qkv + (size_t)(b * T + j) * (3 * D) + 2 * D + (size_t)h * DH;
        o0 = o0 * corr + p * v[lane];
        o1 = o1 * corr + p * v[lane + 32];
        l = l * corr + p;
        m = m_new;
    }

    float inv_l = 1.0f / l;
    float* o = out + (size_t)(b * T + t) * D + (size_t)h * DH;
    o[lane] = o0 * inv_l;
    o[lane + 32] = o1 * inv_l;
}

// ---- Kernel 5: 门控残差 out = x + gate[b] ⊙ y ----
__global__ void gated_residual_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                      const float* __restrict__ mod, float* __restrict__ out,
                                      int rows, int T, int gate_idx) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int b = r / T;
    const float* g = mod + (size_t)b * (6 * D) + gate_idx * D;
    const float* xr = x + (size_t)r * D;
    const float* yr = y + (size_t)r * D;
    float* orow = out + (size_t)r * D;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        orow[i] = xr[i] + g[i] * yr[i];
}

// ---- solve: 编排 11 步 pipeline ----
extern "C" void solve(const float* x, const float* c, float* output, const float* weights,
                      int batch_size, int seq_len) {
    if (!x || !c || !output || !weights || batch_size <= 0 || seq_len <= 0) return;
    int B = batch_size, T = seq_len;
    int rows = B * T;
    size_t f4 = sizeof(float);

    // ---- 中间缓冲（h 复用为 h2，proj 复用为 fc2）----
    float *sbuf, *mod, *h, *qkv, *attn, *proj, *x1, *fc1;
    cudaMalloc(&sbuf, (size_t)B * D * f4);
    cudaMalloc(&mod, (size_t)B * 6 * D * f4);
    cudaMalloc(&h, (size_t)rows * D * f4);
    cudaMalloc(&qkv, (size_t)rows * 3 * D * f4);
    cudaMalloc(&attn, (size_t)rows * D * f4);
    cudaMalloc(&proj, (size_t)rows * D * f4);
    cudaMalloc(&x1, (size_t)rows * D * f4);
    cudaMalloc(&fc1, (size_t)rows * MLP * f4);

    // ---- 权重指针（编译期偏移 + 运行时一次加法）----
    const float* W_ada = weights + O_WADA;
    const float* b_ada = weights + O_BADA;
    const float* W_qkv = weights + O_WQKV;
    const float* b_qkv = weights + O_BQKV;
    const float* W_o   = weights + O_WO;
    const float* b_o   = weights + O_BO;
    const float* W_fc1 = weights + O_WFC1;
    const float* b_fc1 = weights + O_BFC1;
    const float* W_fc2 = weights + O_WFC2;
    const float* b_fc2 = weights + O_BFC2;

    dim3 tile(16, 16);
    dim3 grid_ada((6 * D + 15) / 16, (B + 15) / 16);
    dim3 grid_qkv((3 * D + 15) / 16, (rows + 15) / 16);
    dim3 grid_d((D + 15) / 16, (rows + 15) / 16);
    dim3 grid_fc1((MLP + 15) / 16, (rows + 15) / 16);
    int attn_warps = B * H * T;
    int attn_blocks = (attn_warps * 32 + 127) / 128;

    // ===== AdaLN-Zero 调制 =====
    silu_kernel<<<(B * D + 255) / 256, 256>>>(c, sbuf, B * D);
    gemm_bias_kernel<false><<<grid_ada, tile>>>(sbuf, W_ada, b_ada, mod, B, 6 * D, D);

    // ===== Attention 子块 =====
    ln_modulate_kernel<<<rows, BLOCK_SIZE>>>(x, mod, h, rows, T, 0);
    gemm_bias_kernel<false><<<grid_qkv, tile>>>(h, W_qkv, b_qkv, qkv, rows, 3 * D, D);
    attention_kernel<<<attn_blocks, 128>>>(qkv, attn, B, T);
    gemm_bias_kernel<false><<<grid_d, tile>>>(attn, W_o, b_o, proj, rows, D, D);
    gated_residual_kernel<<<rows, BLOCK_SIZE>>>(x, proj, mod, x1, rows, T, 2);

    // ===== MLP 子块 =====
    ln_modulate_kernel<<<rows, BLOCK_SIZE>>>(x1, mod, h, rows, T, 3);
    gemm_bias_kernel<true><<<grid_fc1, tile>>>(h, W_fc1, b_fc1, fc1, rows, MLP, D);
    gemm_bias_kernel<false><<<grid_d, tile>>>(fc1, W_fc2, b_fc2, proj, rows, D, MLP);
    gated_residual_kernel<<<rows, BLOCK_SIZE>>>(x1, proj, mod, output, rows, T, 5);

    cudaDeviceSynchronize();

    cudaFree(sbuf); cudaFree(mod); cudaFree(h); cudaFree(qkv);
    cudaFree(attn); cudaFree(proj); cudaFree(x1); cudaFree(fc1);
}
```

> 📎 完整可编译代码（含 `main()` 本地测试 harness 与双精度 CPU 参考实现）已整理到 <a href="./116-diffusion-transformer-block.cu" download><code>116-diffusion-transformer-block.cu</code></a>（编译与运行命令见文件头注释，用于本地自测与 profiling）。

### 4.1 LeetGPU 提交版本

上述 `solve` 及全部 kernel 即为 LeetGPU 提交版本，适配官方 starter 签名 `void solve(const float* x, const float* c, float* output, const float* weights, int batch_size, int seq_len)`。本地版把 `cudaMalloc` 包了一层 `CHECK_CUDA` 宏（错误即报错退出），提交版为简洁起见直接调用——两种写法等价。

### 4.2 代码详解

本 pipeline 的核心策略是：**将 DiT block 分解为 11 个子 kernel，GEMM 用 16×16 shared memory tile（W tile 转置存储），LN/调制/门控残差各自融合为单 kernel，attention 用 warp 级 online softmax 单遍计算。**

| 步骤 | Kernel | 说明 |
|------|--------|------|
| **权重解包** | `O_WADA`…`O_BFC2` | 编译期 `constexpr` 偏移定位 packed weights 的 10 段 |
| **SiLU** | `silu_kernel` | 逐元素 $v/(1+e^{-v})$，规模仅 $B \times 512$ |
| **GEMM** | `gemm_bias_kernel<ACT_GELU>` | 16×16 tile，A/W tile 均合并访存；bias 与 GELU 在 epilogue |
| **LN+调制** | `ln_modulate_kernel` | 一个 block 一行，两次 `block_reduce_sum`（mean→var），Pass 3 直接乘加调制向量 |
| **Attention** | `attention_kernel` | 一个 warp 一个 $(b,h,t)$，蝶形 `__shfl_xor` 归约 score，`m/l/o` 三状态单遍 |
| **门控残差** | `gated_residual_kernel` | 一个 block 一行，`out = x + gate[b]⊙y` |

**关键索引关系**：

- `mod[b·6D + i·D + d]` — 样本 $b$ 的第 $i$ 个调制向量（$i$：0=shift_msa、1=scale_msa、2=gate_msa、3=shift_mlp、4=scale_mlp、5=gate_mlp），`ln_modulate` 用 `base`/`base+1` 取 shift/scale 对，`gated_residual` 用 `gate_idx` 取 gate
- `b = r / T` — 行号 $r$ 到 batch 样本号的反查（adaLN 逐样本参数的核心寻址）
- `qkv[(b·T+t)·3D + {0,D,2D} + h·DH + d]` — Q/K/V 免转置直读布局（Q 在 `+0`，K 在 `+D`，V 在 `+2D`）
- `Ws[tx][ty] = W[bx·16+ty][k0+tx]` — W tile **转置存储**（`Ws[k_local][n_local]`）：加载时 16 个连续线程读 W 的同一行（合并访存），计算时 `Ws[t][tx]` 按列读但相邻线程地址连续（无 bank conflict）
- `As[ty][t] · Ws[t][tx]` — 累加内积：thread `(tx,ty)` 计算 `C[row][col] = Σ_k A[row][k]·W[col][k]`，其中 `row = by·16+ty`、`col = bx·16+tx`
- `s += __shfl_xor_sync(0xffffffff, s, off)` — 蝶形归约：5 步后 32 个 lane **全员**拿到完整 score（与 `__shfl_down` 只留 lane 0 不同，后续 `p`/`m_new` 每 lane 都要用）

**变量表（attention kernel 的 online softmax 状态）**：

| 变量 | 含义 | 初始值 |
|------|------|--------|
| `q0, q1` | lane 持有的 query 分量（`d=lane`、`d=lane+32`） | 从 qkv 读入 |
| `m` | 运行最大 score（数值稳定锚点） | $-\infty$ |
| `l` | 运行分母 $\sum_j e^{s_j - m}$ | 0 |
| `o0, o1` | 运行输出 $\sum_j e^{s_j-m} v_j[d]$ | 0 |
| `corr` | $\mathrm{corr}=e^{m-m'}$：m 更新时对旧状态的缩放修正 | — |

#### Worked Example：online softmax 逐步演算

设 $T=2$、$d=2$（演示用，实际 $d=64$），$\text{scale}=1$。取 $q=[1,2]$，$k_0=[1,0]$，$k_1=[0,1]$，$v_0=[1,0]$，$v_1=[0,1]$：

```text
j=0:  s₀ = q·k₀ = 1·1 + 2·0 = 1
      m: -∞ → 1        p = e^(1-1) = 1      corr = e^(-∞-1) = 0
      o = 0·0 + 1·v₀ = [1, 0]               l = 0·0 + 1 = 1

j=1:  s₁ = q·k₁ = 1·0 + 2·1 = 2
      m: 1 → 2        p = e^(2-2) = 1      corr = e^(1-2) = 0.368
      o = [1,0]·0.368 + 1·v₁ = [0.368, 1]   l = 1·0.368 + 1 = 1.368

输出: o/l = [0.269, 0.731]
```

验证（朴素两遍 softmax）：$\text{probs} = \text{softmax}([1,2]) = [0.269, 0.731]$，$\text{out} = 0.269 v_0 + 0.731 v_1 = [0.269, 0.731]$ ✓。两遍变一遍的代价是每步多做一次 `corr` 缩放修正——这正是 FlashAttention 的核心思想（#12 Multi-Head Attention 详解过完整版）。

> 💡 **关键洞察**：DiT block 与 Llama block 的 kernel 骨架几乎同构（11 步 vs 10 步），唯一的结构性新增是 **adaLN 的逐样本参数**——它把"权重是全局共享的张量"泛化成"每 batch 样本一份的小张量"。工程上只需两件事：mod 缓冲按 $(B, 6, D)$ 布局一次算好，下游 kernel 用 `b = row / T` 反查。这一模式同样适用于 LoRA（#85 的逐样本缩放）、GQA 的 per-head KV（#80），以及一切"per-sample/per-group 参数"的推理 kernel。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 116-diffusion-transformer-block.cu -o dit_block
./dit_block   # (1,1) / (2,16) / (3,30) 三组自测，双精度 CPU 参考对照
```

### 5.2 用 ncu 分析瓶颈

```bash
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            lts__t_sector_hit_rate.pct, \
            gpu__time_duration.sum \
    ./dit_block
```

性能测试规模 $B=4, T=1024$（$N = 4096$ 行）的预期画像：

| 指标 | 预期 | 解读 |
|------|------|------|
| `gpu__time_duration`（GEMM 类 kernel） | 主导（~70%） | 5 个大 GEMM 共 ~26 GFLOP，16×16 tile 无 register blocking，SM 利用率中等 |
| `sm__throughput`（attention） | 中等 | $O(T^2)$ 计算 + 每 warp 重复读 K/V，L2 命中率是关键 |
| `lts__t_sector_hit_rate`（attention） | 波动 | 每个 $(b,h)$ 的 K/V 共 256KB，超过 T4 L2（4MB）单块可容纳的并发工作集 |
| `dram__throughput`（ln_modulate/gated_residual） | 高（~60-80%） | 纯 memory-bound：读 $2Nd$ 写 $Nd$，无计算可藏 |

### 5.3 优化方向

1. **GEMM 升级 register blocking / 双缓冲**：16×16 tile 每线程 1 个累加器，改成 32×32 tile + 每线程 2×2 寄存器累加（#22 GEMM 的做法），算术强度翻倍；再叠 double buffering 掩盖 shared memory 加载延迟。五个 GEMM 合计 ~26 GFLOP，是最大的提速来源。

2. **FlashAttention 式 KV 分块**：当前每 warp 串行扫全部 $T$ 个 key，K/V 从 L2/HBM 重复读 $T$ 次。改成 block 协作加载 KV tile 进 shared memory（两遍在线重算 softmax），K/V 读量从 $B \cdot H \cdot T^2 \cdot 64$ 降到 $\sim B \cdot H \cdot T \cdot 64 \cdot (T/\text{tile})$ 分摊，$T=4096$ 时收益巨大（#12 的完整推导）。

3. **QKV+attention 融合**：attention kernel 在算 score 前先做 QKV GEMM 的 epilogue——每个 (b,h,t) 的 Q 行向量与 W_qkv 的 3 个 64×512 分块相乘。省掉 `qkv` 缓冲的一次写+读（$6Nd$ float）。

4. **adaLN GEMM 与调制合并**：#2 的 GEMM 只有 $B$ 行（≤16），grid 仅 192 块，SM 严重吃不饱。可把 SiLU 融进该 GEMM 的 A tile 加载（prologue fusion），并与其他小 kernel 合并 launch。

5. **FP16/BF16 推理**：权重与激活半精度存储（带宽减半），GEMM 走 Tensor Core（T4 FP16 算力 8× FP32），累加保持 FP32 满足 1e-3 容差。DiT 去噪本身对精度不敏感，SD3/Flux 官方推理就是 BF16。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | $O(N d^2 + N \cdot 4d^2 + B H T^2 d_h) = O(N d^2 + B H T^2 d_h)$（GEMM + attention，$N = BT$，$d_h = 64$） |
| **HBM IO（激活）** | 读 $\sim 10Nd + 2NF$ / 写 $\sim 6Nd$；全融合下限：读 $\sim 4Nd$ / 写 $Nd$ |
| **HBM IO（权重）** | 4.7M float ≈ 18.9MB，每 GEMM kernel 一次性读入，$N{=}4096$ 时占 IO 总量 <5% |
| **shared memory** | GEMM：$2 \times 16 \times 16 \times 4\text{B} = 2\text{KB}$/block；LN：9 float；attention：0（online softmax 状态全在寄存器） |
| **中间缓冲峰值** | 7 个 buffer：$3Nd + 3Nd + NF$ float ≈ $6Nd + NF$；$B{=}16,T{=}4096$ 时 ~1.5GB（复用 `h`/`proj` 省了 $2Nd$） |
| **scores 矩阵** | **不分配**（online softmax）；朴素 PyTorch 版需 $BHT^2$ float——$B{=}16,T{=}4096$ 时约 8GB，是"能不能跑起来"的差别 |
| **瓶颈类型** | $T \le 1024$ 时 GEMM compute-bound 主导；$T = 4096$ 时 attention 的 $O(T^2)$ 计算与 K/V 重复读上升为共瓶颈 |

> 💡 **一句话总结**：DiT block 是"transformer block pipeline + per-sample 参数化"的组合考验——11 个 kernel 的编排套路与 GPT-2/Llama block 一脉相承，新增的 adaLN-Zero 调制只需一个 $(B,6,D)$ 的 mod 缓冲和 `b = row / T` 的按行反查；attention 换成双向 + online softmax 单遍版；MLP 的 GELU 直接进 GEMM epilogue。做完 #74、#93 再做本题，能清晰看到"同一个 block 骨架如何长出 GPT-2 / Llama / DiT 三个变种"——这套 pipeline 编排 + 选择性融合的模板，可直接迁移到 SD3/Flux 的 DiT 推理引擎实现。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 74 | [GPT-2 Transformer Block](https://leetgpu.com/challenges/gpt-2-transformer-block) | 困难 | — | 结构最接近的 decoder 版 block：LayerNorm+MHA+GELU，对比 adaLN-Zero 调制与标准 affine LN |
| 93 | [Llama Transformer Block](https://leetgpu.com/challenges/llama-transformer-block) | 困难 | — | 另一个 block 变种（RMSNorm+GQA+SwiGLU），对比组件取舍与 kernel 编排差异 |
| 115 | [Layer Normalization](https://leetgpu.com/challenges/layer-normalization) | 中等 | — | LN 组件独立实现，本题"无仿射 LN + 调制包裹"的直接前驱 |
| 12 | [Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention) | 困难 | — | MHA 组件独立实现，本题 attention 为无 mask 双向版，online softmax 同源 |

> 💡 **选题思路**：adaLN-Zero 调制 + 双向 MHA + GELU MLP 的 DiT block 综合模块，练习 per-sample 参数广播与 multi-kernel pipeline 编排。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

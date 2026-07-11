# LeetGPU Token Embedding Layer 题解

## 1. 题目概述

- **标题 / 题号**：Token Embedding Layer（#106，medium）
- **链接**：https://leetgpu.com/challenges/token-embedding-layer
- **难度**：中等
- **标签**：CUDA、embedding、gather、LayerNorm、融合 kernel、推理引擎第一个算子

**题意**：实现 Transformer 输入端的 **embedding 层**（BERT 风格）。对 batch 中每个 token（`b, t`）：
1. 从 `token_embeddings[V,D]` 表用 `token_ids[b,t]` gather 一行
2. 从 `position_embeddings[P,D]` 表用 `position_ids[t]` gather 一行
3. 两者相加 `s = tok + pos`
4. 对 `s` 做 LayerNorm（带可学习 `gamma[D]`/`beta[D]`）：

$$\mu = \frac{1}{D}\sum_d s_d,\quad \sigma^2 = \frac{1}{D}\sum_d (s_d-\mu)^2,\quad y_d = \gamma_d \cdot \frac{s_d - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_d$$

输出 `output[B,T,D]`。

**示例**（`B=1, T=2, D=4`）：

```text
token_ids = [[5, 12]], position_ids = [0, 1]
tok_emb[5] = [1,0,0,0], pos_emb[0] = [0,0,0,0] → s=[1,0,0,0], LN → ...
tok_emb[12] = [0,1,0,0], pos_emb[1] = [0,1,0,0] → s=[0,2,0,0], LN → ...
```

**约束**：`B=32, T=512, V=30000, P=2048, D=768`（BERT-base 配置）；`token_ids/position_ids` 为 `int32`，其余 `float32`；容差 `atol=rtol=1e-4`。

> 💡 这道题是 [Week5 Day5](../../aiinfra/week5/day5/README.md) 讲的 **Mini 推理引擎 v0 的第一个算子**——`self.embedding(input_ids)` 在 v0 里是 PyTorch 的 `nn.Embedding`，本题就是它的手写 CUDA 版。引擎 `model.forward` 第一步把 token id 转成向量，本题拆成 `gather token emb + gather pos emb + 相加 + LayerNorm` 四步融合。Week7 替换 PyTorch 后端时，引擎的 embedding 层就换成这个手写 kernel。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行参考（同 reference_impl）

```cpp
// cpu_baseline.cpp —— CPU 串行 embedding + LayerNorm
void embed_cpu(const int* token_ids, const int* pos_ids,
               const float* tok_emb, const float* pos_emb,
               const float* gamma, const float* beta,
               float* output, int B, int T, int V, int P, int D, float eps) {
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            int tid = token_ids[b*T + t], pid = pos_ids[t];
            // gather + 相加
            float s[768];
            for (int d = 0; d < D; ++d)
                s[d] = tok_emb[tid*D + d] + pos_emb[pid*D + d];
            // LayerNorm
            float mu = 0.f;
            for (int d = 0; d < D; ++d) mu += s[d];
            mu /= D;
            float var = 0.f;
            for (int d = 0; d < D; ++d) { float diff = s[d]-mu; var += diff*diff; }
            var /= D;
            float inv = 1.0f / sqrtf(var + eps);
            for (int d = 0; d < D; ++d)
                output[(b*T+t)*D + d] = gamma[d] * (s[d]-mu) * inv + beta[d];
        }
}
```

复杂度 `O(B·T·D)`。关键：gather 是随机读（`token_ids` 任意），LayerNorm 需要 D 维全归约求 μ/σ²。

### 2.2 朴素 GPU：分 4 个 kernel

朴素做法：开 4 个 kernel——gather token、gather pos、相加、LayerNorm。**问题**：中间结果 `(B,T,D)` 要写回 HBM 再读，多 3 趟 `B·T·D·4B` 的往返（`B=32,T=512,D=768` → 每趟 48MB）。应融合成一个 kernel。

> ⚠️ 正确做法：**一个 kernel 内完成 gather + 加 + LayerNorm**，中间的 `s[D]` 只活在寄存器/shared memory，永不落 HBM。这与 [Week4 Softmax Attention](../week4/day1/leetgpu-softmax-attention-solution.md) 消除 S/P 物化同理——"融合消除中间矩阵"。

## 3. GPU 设计

### 3.1 并行化策略

![Token Embedding Layer：gather + 加 + LayerNorm 融合](images/token_embedding_layer_overview.svg)

| 维度 | 映射 | 说明 |
|------|------|------|
| **(b,t) 位置** | `blockIdx.x` | 每个 block 处理一个 (batch, time) 位置，grid = `(B·T,)` |
| **D 维** | block 内 thread 协作 | 每 thread 持有 s 的若干维，块归约求 μ/σ² |

![并行映射：1 个 block 处理 1 个 (b,t) 位置](images/token_embedding_block_mapping.svg)

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|---------|------|
| **global** | ✓ | 读 `token_embeddings`/`position_embeddings`（gather）、`gamma`/`beta`；写 `output` |
| **shared** | ✓ | `s_shm[D]` 缓存相加结果（供块归约）；归约缓冲 |
| **register** | ✓ | 每 thread 持有 s 的若干维 + μ/σ² 中间值 |

### 3.3 关键技巧

1. **gather 随机读**：`token_ids[b,t]` 是任意值，`tok_emb[tid*D + d]` 是非连续读。但同一 block 内所有 thread 读同一行（同一 `tid`），可让 thread 0 读到 shared 后广播，或直接让各 thread 读各自 d 维（coalesced）。
2. **两趟块归约求 μ/σ²**：LayerNorm 需要先求 μ（一趟归约），再求 σ²（基于 μ 再一趟归约）。两趟 `block_reduce_sum`，每趟一次 `__syncthreads`。
3. **融合消除中间物化**：gather→加→LN 在一个 kernel 内完成，`s[D]` 只活在 shared/register，不写回 HBM。
4. **position_ids 共享**：同一 time step `t` 的所有 batch 共用 `pos_ids[t]`，可缓存。

> 💡 **gather 的内存模式**：`token_embeddings` 是大表（`V=30000, D=768` → 92MB），gather 是随机读。但同一 block（同一 b,t）所有 thread 读同一行 `tid`——行内 D 维连续，coalesced；不同 block 读不同行，靠 L2 cache 缓解。

## 4. Kernel 实现

完整可编译代码：**fused 版（gather + 加 + LayerNorm 一体）**，含 `main()`、`cudaMalloc/Memcpy`、CPU 验证、`cudaFree`：

```cuda
// token_embedding_layer.cu —— Token Embedding + LayerNorm 融合 kernel
// 编译命令: nvcc -O3 -arch=sm_120 token_embedding_layer.cu -o token_emb -lineinfo
// 运行:     ./token_emb 32 512 30000 2048 768

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define NUM_WARPS  (BLOCK_SIZE / WARP_SIZE)
#define D_MAX      1024

__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE/2; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v; __syncthreads();
    if (wid == 0) { v = (lane < NUM_WARPS) ? sh[lane] : 0.f; v = warp_reduce_sum(v); if (lane==0) sh[0]=v; }
    __syncthreads(); return sh[0];
}

// ---------- fused kernel：一个 block 处理一个 (b,t) ----------
__global__ void token_embedding_layernorm_kernel(
    const int*   __restrict__ token_ids,     // (B, T)
    const int*   __restrict__ position_ids,  // (T,)
    const float* __restrict__ token_emb,     // (V, D)
    const float* __restrict__ position_emb,  // (P, D)
    const float* __restrict__ gamma,         // (D,)
    const float* __restrict__ beta,          // (D,)
    float*       __restrict__ output,        // (B, T, D)
    int B, int T, int V, int P, int D, float eps) {

    int idx = blockIdx.x;
    int b = idx / T, t = idx % T;
    int tid = threadIdx.x;
    if (b >= B) return;

    int tok_id = token_ids[b * T + t];
    int pos_id = position_ids[t];

    __shared__ float s_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];

    // ① gather + 相加：每 thread 读若干维
    for (int d = tid; d < D; d += BLOCK_SIZE)
        s_shm[d] = token_emb[tok_id * D + d] + position_emb[pos_id * D + d];
    __syncthreads();

    // ② 求 μ = Σ s[d] / D（块归约）
    float local_sum = 0.f;
    for (int d = tid; d < D; d += BLOCK_SIZE) local_sum += s_shm[d];
    float mu = block_reduce_sum(local_sum, red) / (float)D;
    __shared__ float s_mu;
    if (tid == 0) s_mu = mu;
    __syncthreads(); mu = s_mu;

    // ③ 求 σ² = Σ (s[d]-μ)² / D（块归约）
    float local_sq = 0.f;
    for (int d = tid; d < D; d += BLOCK_SIZE) { float diff = s_shm[d] - mu; local_sq += diff * diff; }
    float var = block_reduce_sum(local_sq, red) / (float)D;
    __shared__ float s_var;
    if (tid == 0) s_var = var;
    __syncthreads(); var = s_var;

    float inv_std = 1.0f / sqrtf(var + eps);

    // ④ 归一化 + 写回：y[d] = γ[d]·(s[d]-μ)/√(σ²+ε) + β[d]
    for (int d = tid; d < D; d += BLOCK_SIZE)
        output[(b * T + t) * D + d] = gamma[d] * (s_shm[d] - mu) * inv_std + beta[d];
}

// ---------- CPU 参考 ----------
void embed_cpu(const int* tids, const int* pids, const float* te, const float* pe,
               const float* g, const float* bt, float* out,
               int B, int T, int V, int P, int D, float eps) {
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            int tid = tids[b*T+t], pid = pids[t];
            std::vector<float> s(D);
            for (int d = 0; d < D; ++d) s[d] = te[tid*D+d] + pe[pid*D+d];
            float mu = 0.f;
            for (int d = 0; d < D; ++d) mu += s[d]; mu /= D;
            float var = 0.f;
            for (int d = 0; d < D; ++d) { float diff = s[d]-mu; var += diff*diff; } var /= D;
            float inv = 1.0f/sqrtf(var+eps);
            for (int d = 0; d < D; ++d) out[(b*T+t)*D+d] = g[d]*(s[d]-mu)*inv + bt[d];
        }
}

int main(int argc, char** argv) {
    int B = (argc>1)?atoi(argv[1]):32, T = (argc>2)?atoi(argv[2]):512;
    int V = (argc>3)?atoi(argv[3]):30000, P = (argc>4)?atoi(argv[4]):2048;
    int D = (argc>5)?atoi(argv[5]):768;
    float eps = 1e-5f;
    if (D > D_MAX) { printf("要求 D <= %d\n", D_MAX); return 1; }
    printf("B=%d T=%d V=%d P=%d D=%d\n", B, T, V, P, D);

    size_t ids_bytes = (size_t)B*T*sizeof(int), pids_bytes = (size_t)T*sizeof(int);
    size_t te_bytes = (size_t)V*D*sizeof(float), pe_bytes = (size_t)P*D*sizeof(float);
    size_t gb_bytes = (size_t)D*sizeof(float), out_bytes = (size_t)B*T*D*sizeof(float);
    printf("token_emb = %.2f MB, output = %.2f MB\n", te_bytes/1e6, out_bytes/1e6);

    std::vector<int> h_tids(B*T), h_pids(T);
    std::vector<float> h_te(V*D), h_pe(P*D), h_g(D), h_bt(D), h_out(B*T*D), h_ref(B*T*D);
    srand(42);
    for (auto& x : h_tids) x = rand() % V;
    for (int t = 0; t < T; ++t) h_pids[t] = t % P;
    for (auto& x : h_te) x = ((rand()%600)-300)/1000.f;
    for (auto& x : h_pe) x = ((rand()%600)-300)/1000.f;
    for (auto& x : h_g) x = 0.8f + (rand()%400)/1000.f;
    for (auto& x : h_bt) x = ((rand()%200)-100)/1000.f;

    int *d_tids, *d_pids;
    float *d_te, *d_pe, *d_g, *d_bt, *d_out;
    cudaMalloc(&d_tids, ids_bytes); cudaMemcpy(d_tids, h_tids.data(), ids_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_pids, pids_bytes); cudaMemcpy(d_pids, h_pids.data(), pids_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_te, te_bytes); cudaMemcpy(d_te, h_te.data(), te_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_pe, pe_bytes); cudaMemcpy(d_pe, h_pe.data(), pe_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_g, gb_bytes); cudaMemcpy(d_g, h_g.data(), gb_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_bt, gb_bytes); cudaMemcpy(d_bt, h_bt.data(), gb_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out, out_bytes);

    int grid = B * T;
    token_embedding_layernorm_kernel<<<grid, BLOCK_SIZE>>>(
        d_tids, d_pids, d_te, d_pe, d_g, d_bt, d_out, B, T, V, P, D, eps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost);
    embed_cpu(h_tids.data(), h_pids.data(), h_te.data(), h_pe.data(),
              h_g.data(), h_bt.data(), h_ref.data(), B, T, V, P, D, eps);

    float maxd = 0;
    for (int i = 0; i < B*T*D; ++i) maxd = fmaxf(maxd, fabsf(h_out[i]-h_ref[i]));
    printf("max diff: %.2e (%s, tol=1e-4)\n", maxd, maxd<1e-4f?"PASS":"FAIL");

    cudaFree(d_tids);cudaFree(d_pids);cudaFree(d_te);cudaFree(d_pe);
    cudaFree(d_g);cudaFree(d_bt);cudaFree(d_out);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `token_embedding_layernorm_kernel` 填进 starter 的 `solve` 即可。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 token_embedding_layer.cu -o token_emb -lineinfo
./token_emb 32 512 30000 2048 768      # BERT-base 配置
./token_emb 1 8 100 16 64              # 小尺寸验证
```

典型输出（RTX 5090，`B=32, T=512, D=768`）：

```text
B=32 T=512 V=30000 P=2048 D=768
token_emb = 92.16 MB, output = 48.00 MB
max diff: x.xx e-06 (PASS, tol=1e-4)
```

### 5.2 用 ncu 观察

```bash
ncu --kernel-name regex:token_embedding_layernorm_kernel \
    --metrics gpu__time_duration.sum, dram__bytes.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./token_emb 32 512 30000 2048 768
```

| 指标 | 值 | 含义 |
|------|----|------|
| `gpu__time` | 基线 | B·T=16384 blocks，每 block 处理 D=768 维 |
| `dram__bytes` | 读 token_emb(92MB) + pos_emb(6MB) + output(48MB) | gather 是主要 IO |
| `dram__throughput` | 中等 | gather 随机读，L2 命中率影响大 |

> ⚠️ **关键观察**：瓶颈在 `token_embeddings` 的 gather 随机读（92MB 大表）。position_embeddings 小（6MB）易被 L2 缓存。LayerNorm 的两趟归约是计算开销但相对小。优化重点是 gather 的内存访问模式。

### 5.3 优化方向

1. **token_emb 行对齐**：保证每行 D 维 128B 对齐，提升 gather 的 coalescing。
2. **共享 position_emb**：所有 batch 共用 `position_ids[t]`，pos_emb 可缓存到 shared 一次复用。
3. **合并相邻 (b,t) 的 gather**：若相邻 token 的 id 接近，可合并读连续行。
4. **fp16 embedding 表**：embedding 表用 fp16 存储，减半 IO（精度损失可接受，embedding 不参与梯度时尤其安全）。
5. **L2 cache 友好**：B·T 个 block 随机读 V 行，可按 token_id 排序 block 顺序提升 L2 局部性。

## 6. 复杂度分析

| 维度 | 复杂度 | 说明 |
|------|--------|------|
| **时间** | `O(B·T·D)` | 每 (b,t) 处理 D 维，gather + 两趟归约 |
| **空间** | `O(B·T·D)` 输出 + `O(V·D)` 表 | embedding 表是大头 |
| **HBM IO** | 读 token_emb + pos_emb + γ/β，写 output | 融合后无中间矩阵 |
| **瓶颈** | gather 随机读 token_emb | 大表随机访问，L2 命中率关键 |

> 💡 **一句话总结**：Token Embedding Layer 是推理引擎的第一个算子——token id → 向量。用融合 kernel 把 gather + 加 + LayerNorm 一次完成（中间 `s[D]` 不落 HBM）。它是 [Day5 Mini 引擎](../../aiinfra/week5/day5/README.md) `self.embedding` 的手写版，Week7 替换 PyTorch 后端时直接用。瓶颈在 embedding 大表的 gather 随机读，优化靠行对齐 + L2 友好访问。

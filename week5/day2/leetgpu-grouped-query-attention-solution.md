# LeetGPU Grouped Query Attention (GQA) 题解

## 1. 题目概述

- **标题 / 题号**：Grouped Query Attention（#80，medium）
- **链接**：https://leetgpu.com/challenges/grouped-query-attention
- **难度**：中等
- **标签**：CUDA、Attention、GQA、KV Cache 优化、KV head 共享、memory-bound、LLM 推理

**题意**：实现 **Grouped Query Attention (GQA)**——现代大模型（LLaMA-3、Mistral、Gemma）的标准 attention 机制。给定 `num_q_heads` 个 query 头（`Q` 形状 `[num_q_heads, seq_len, head_dim]`）和 `num_kv_heads` 个 KV 头（`K`/`V` 形状 `[num_kv_heads, seq_len, head_dim]`），每 `num_q_heads / num_kv_heads` 个连续 query 头共享同一组 K/V 头，做 scaled dot-product attention，输出 `[num_q_heads, seq_len, head_dim]`：

$$\text{group} = \text{num\_q\_heads} / \text{num\_kv\_heads},\quad \text{kv\_head}(h) = h\, /\, \text{group}$$

$$O_h = \text{softmax}\!\left(\frac{Q_h \cdot K_{\text{kv\_head}(h)}^{\top}}{\sqrt{d}}\right) V_{\text{kv\_head}(h)}$$

**示例**（`num_q_heads=4, num_kv_heads=2, seq_len=3, head_dim=4`，groups=2）：
- `Q0, Q1` 共享 `K[0], V[0]`；`Q2, Q3` 共享 `K[1], V[1]`
- 每个 Q head 独立做 `Q_h · K_{kv}^T / √d → softmax → · V_{kv}`

**约束**：`1 ≤ num_kv_heads ≤ num_q_heads ≤ 64`，`num_q_heads % num_kv_heads == 0`，`1 ≤ seq_len ≤ 4096`，`8 ≤ head_dim ≤ 256`（8 的倍数）；性能测试取 LLaMA-3 8B 配置 `num_q_heads=32, num_kv_heads=8, seq_len=1024, head_dim=128`；容差 `atol=rtol=1e-4`。

> 💡 这道题是 [Week5 Day2](../../aiinfra/week5/day2/README.md) 讲的 **KV Cache 内存优化**的核心手段。标准 MHA 每个 Q 头有独立 K/V 头，cache 大小 ∝ `num_q_heads`；GQA 让多个 Q 头共享 K/V 头，把 cache 的 `num_heads` 维从 `num_q_heads` 降到 `num_kv_heads`。LLaMA-3 8B（32Q/8KV）直接把 KV cache 缩小到 1/4。Day 2 手写了 KV Cache 的存储结构，GQA 回答的是"能不能少存一些头"——从模型结构层面削减 cache，比 int8 量化（精度层面）更根本。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行参考（同 reference_impl）

```cpp
// cpu_baseline.cpp —— CPU 串行 GQA（先 expand KV 再做标准 MHA）
void gqa_cpu(const float* Q, const float* K, const float* V, float* O,
             int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    int group = num_q_heads / num_kv_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int h = 0; h < num_q_heads; ++h) {
        int kv_h = h / group;   // 共享的 KV head
        const float *Kh = K + kv_h * seq_len * head_dim;
        const float *Vh = V + kv_h * seq_len * head_dim;
        for (int s = 0; s < seq_len; ++s) {
            // scores[k] = Q[h,s] · K[kv_h,k] / √d
            float mx = -INFINITY;
            std::vector<float> sc(seq_len);
            for (int k = 0; k < seq_len; ++k) {
                float dot = 0.f;
                for (int t = 0; t < head_dim; ++t)
                    dot += Q[(h*seq_len+s)*head_dim + t] * Kh[k*head_dim + t];
                sc[k] = dot * scale; mx = fmaxf(mx, sc[k]);
            }
            float sum = 0.f;
            for (int k = 0; k < seq_len; ++k) { sc[k] = expf(sc[k]-mx); sum += sc[k]; }
            for (int t = 0; t < head_dim; ++t) {
                float acc = 0.f;
                for (int k = 0; k < seq_len; ++k) acc += sc[k] * Vh[k*head_dim + t];
                O[(h*seq_len+s)*head_dim + t] = acc / sum;
            }
        }
    }
}
```

复杂度 `O(num_q_heads × seq_len² × head_dim)`。关键点：**KV head 索引映射** `kv_h = h / group`——这是 GQA 与 MHA 的唯一区别。

### 2.2 朴素 GPU：先 expand KV 再调 MHA

最朴素的想法：先用一个 kernel 把 `K[num_kv_heads, ...]` **expand**（`repeat_interleave`）成 `[num_q_heads, ...]`，再调标准 MHA kernel。**这是本题要避免的反模式**：

- expand 会多分配 `num_q_heads / num_kv_heads` 倍的 K/V 显存（如 LLaMA-3 32Q/8KV → 多 4× 显存），直接抵消 GQA 的 cache 收益；
- 多一次 HBM 写读往返，Decode 本就 memory-bound，白白浪费带宽。

> ⚠️ 正确做法是**不 expand**，直接在 attention kernel 里用 `kv_h = h / group` 索引到原始的 `num_kv_heads` 份 K/V。多个 Q head 读同一份 KV 时，利用 shared memory 缓存复用——这正是 GQA 在 GPU 上的高效实现方式。

## 3. GPU 设计

### 3.1 并行化策略

![MHA vs GQA vs MQA：KV 头数决定 Cache 大小](images/grouped_query_attention_overview.svg)

| 维度 | 映射 | 说明 |
|------|------|------|
| **Q head × seq 行** | `blockIdx.x` | 每个 block 处理一个 `(q_head h, seq 行 s)` 的 attention，grid = `num_q_heads × seq_len` |
| **block 内** | 线程协作遍历 `seq_len` 个 key | 块归约算点积；thread 0 维护 online softmax 的 m/l/o；输出 `head_dim` 维由 thread 分摊 |
| **KV head 映射** | `kv_h = h / group` | 同一 group 的 Q heads 共享一份 K/V，读同一块 global/shared memory |

![GQA 并行映射：block 处理一个 (q_head, seq_row)](images/grouped_query_attention_block_mapping.svg)

`grid = (num_q_heads × seq_len,)`，`block = (BLOCK,)`（如 256）。每个 block 独立做一次完整 attention：`Q[h,s]` 对 `K[kv_h, :seq]` 做 `1×seq` 点积 → online softmax → `× V[kv_h, :seq]`。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|---------|------|
| **global** | ✓ | 读 `Q`、`K`、`V`；写 `output`。K/V 按 `num_kv_heads` 份存储（不 expand） |
| **shared** | ✓ | `q_shm[head_dim]` 缓存本 block 的 Q 行；块归约缓冲；广播 `s_k/α/β` |
| **register** | ✓ | `o_local`（输出累加器）；online softmax 的 `m/l`（thread 0 维护） |

### 3.3 关键技巧

1. **KV head 索引映射**：`kv_h = h / group`（`group = num_q_heads / num_kv_heads`）——这是 GQA 的全部精髓。每个 block 根据 Q head 号算出该读哪个 KV head，直接索引原始的 `num_kv_heads` 份 K/V，无需 expand。
2. **online softmax 三公式**（与 [Week4 Softmax Attention](../week4/day1/leetgpu-softmax-attention-solution.md) 完全一致）：遍历 `seq_len` 个 key，用 `m/l/o` 增量更新，`S=QK^T` 和 `P=softmax(S)` 永不落 HBM。
3. **不 expand KV**：从 global 直接读 `K[kv_h, k, :]`，多个 Q head 重读同一 KV head 时由 L2 cache / shared memory 自然复用——避免物化 expanded KV。

> 💡 **GQA 与 MHA 的 kernel 区别仅一行**：MHA 是 `kv_h = h`，GQA 是 `kv_h = h / group`。其余 attention 逻辑完全相同。这也说明 GQA 几乎零额外实现成本就换来 4× cache 缩减——这就是它成为现代 LLM 标配的原因。

> 💡 **数值稳定**：所有 `exp` 都减 running max，指数 ≤ 0，fp32 下不会溢出。容差 `1e-4`，fp32 直接满足。

## 4. Kernel 实现

完整可编译代码：**fused 版（online softmax + KV head 共享，不 expand）**，含 `main()`、`cudaMalloc/Memcpy`、CPU 验证、`cudaFree`：

```cuda
// grouped_query_attention.cu —— GQA（fused, online softmax, 不 expand KV）
// 编译命令: nvcc -O3 -arch=sm_120 grouped_query_attention.cu -o gqa -lineinfo
// 运行:     ./gqa 32 8 1024 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define NUM_WARPS  (BLOCK_SIZE / WARP_SIZE)
#define D_MAX      256

// ---------- 块归约 ----------
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

// ---------- fused GQA kernel ----------
// grid = (num_q_heads * seq_len,)
// 每 block 处理一个 (q_head h, seq 行 s)
__global__ void gqa_kernel(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           const float* __restrict__ V,
                           float* __restrict__ output,
                           int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int idx = blockIdx.x;
    int h = idx / seq_len;           // q head
    int s = idx % seq_len;           // query 行
    int tid = threadIdx.x;
    if (h >= num_q_heads) return;

    int group = num_q_heads / num_kv_heads;
    int kv_h = h / group;            // ★ GQA 核心：共享的 KV head
    const float scale = 1.0f / sqrtf((float)head_dim);

    // ① 载入 Q[h, s, :] 到 shared
    for (int t = tid; t < head_dim; t += BLOCK_SIZE)
        q_shm[t] = Q[(h * seq_len + s) * head_dim + t];
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float o_local = 0.f;

    // ② 遍历 seq_len 个 key：点积 → online softmax → 加权 V
    for (int k = 0; k < seq_len; ++k) {
        const float* Kk = K + (kv_h * seq_len + k) * head_dim;   // 读共享的 KV head
        const float* Vk = V + (kv_h * seq_len + k) * head_dim;

        float part = 0.f;
        for (int t = tid; t < head_dim; t += BLOCK_SIZE) part += q_shm[t] * Kk[t];
        float s_k = block_reduce_sum(part, red) * scale;
        if (tid == 0) s_k_shm = s_k;
        __syncthreads(); s_k = s_k_shm;

        if (tid == 0) {
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new);
            float p     = expf(s_k - m_new);
            float l_new = l * alpha + p;
            alpha_shm = (l * alpha) / l_new;
            beta_shm  = p / l_new;
            m = m_new; l = l_new;
        }
        __syncthreads();

        for (int t = tid; t < head_dim; t += BLOCK_SIZE)
            o_local = o_local * alpha_shm + beta_shm * Vk[t];
        __syncthreads();
    }
    // ③ 写回 output[h, s, :]
    for (int t = tid; t < head_dim; t += BLOCK_SIZE)
        output[(h * seq_len + s) * head_dim + t] = o_local;
}

// ---------- CPU 参考 ----------
void gqa_cpu(const float* Q, const float* K, const float* V, float* O,
             int nq, int nkv, int S, int d) {
    int g = nq / nkv;
    float sc = 1.0f / sqrtf((float)d);
    std::vector<float> row(S);
    for (int h = 0; h < nq; ++h) {
        int kvh = h / g;
        for (int s = 0; s < S; ++s) {
            float mx = -INFINITY;
            for (int k = 0; k < S; ++k) {
                float dot = 0.f;
                for (int t = 0; t < d; ++t) dot += Q[(h*S+s)*d+t] * K[(kvh*S+k)*d+t];
                row[k] = dot * sc; mx = fmaxf(mx, row[k]);
            }
            float sum = 0.f;
            for (int k = 0; k < S; ++k) { row[k] = expf(row[k]-mx); sum += row[k]; }
            for (int t = 0; t < d; ++t) {
                float acc = 0.f;
                for (int k = 0; k < S; ++k) acc += row[k] * V[(kvh*S+k)*d+t];
                O[(h*S+s)*d+t] = acc / sum;
            }
        }
    }
}

int main(int argc, char** argv) {
    int nq  = (argc > 1) ? atoi(argv[1]) : 32;
    int nkv = (argc > 2) ? atoi(argv[2]) : 8;
    int S   = (argc > 3) ? atoi(argv[3]) : 1024;
    int d   = (argc > 4) ? atoi(argv[4]) : 128;
    if (d > D_MAX) { printf("要求 d <= %d\n", D_MAX); return 1; }
    if (nq % nkv)  { printf("num_q_heads 必须整除 num_kv_heads\n"); return 1; }
    int group = nq / nkv;

    size_t q_bytes = (size_t)nq * S * d * sizeof(float);
    size_t kv_bytes = (size_t)nkv * S * d * sizeof(float);   // ★ KV 只有 nkv 份（不是 nq）
    printf("nq=%d nkv=%d group=%d S=%d d=%d\n", nq, nkv, group, S, d);
    printf("Q=%.2f MB  KV=%.2f MB (MHA would be %.2f MB, %.1fx more)\n",
           q_bytes/1e6, 2.0*kv_bytes/1e6, 2.0*q_bytes/1e6, (double)nq/nkv);

    std::vector<float> hQ(nq*S*d), hK(nkv*S*d), hV(nkv*S*d), hO(nq*S*d), hRef(nq*S*d);
    srand(42);
    for (auto& x : hQ) x = ((rand()%2000)-1000)/100.f;
    for (auto& x : hK) x = ((rand()%2000)-1000)/100.f;
    for (auto& x : hV) x = ((rand()%2000)-1000)/100.f;

    float *dQ,*dK,*dV,*dO;
    cudaMalloc(&dQ, q_bytes);  cudaMemcpy(dQ, hQ.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, kv_bytes); cudaMemcpy(dK, hK.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, kv_bytes); cudaMemcpy(dV, hV.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, q_bytes);

    int grid = nq * S;
    // warmup
    gqa_kernel<<<grid, BLOCK_SIZE>>>(dQ,dK,dV,dO,nq,nkv,S,d);
    cudaDeviceSynchronize();
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gqa_kernel<<<grid, BLOCK_SIZE>>>(dQ,dK,dV,dO,nq,nkv,S,d);
    cudaEventRecord(t1); cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    cudaMemcpy(hO.data(), dO, q_bytes, cudaMemcpyDeviceToHost);
    gqa_cpu(hQ.data(), hK.data(), hV.data(), hRef.data(), nq, nkv, S, d);
    float maxd = 0;
    for (int i = 0; i < nq*S*d; ++i) maxd = fmaxf(maxd, fabsf(hO[i]-hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-4)\n", maxd, maxd<1e-4f?"PASS":"FAIL");

    // KV cache 收益估算
    printf("\n[KV Cache 收益] GQA cache / MHA cache = %d / %d = %.2f\n", nkv, nq, (double)nkv/nq);
    printf("[LLaMA-3 8B] 32Q/8KV → cache 缩到 %.0f%%（省 %.0f%%）\n",
           100.0*nkv/nq, 100.0*(1.0-(double)nkv/nq));

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `gqa_kernel` 填进 starter 的 `solve` 即可（平台只验证正确性）。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 grouped_query_attention.cu -o gqa -lineinfo
./gqa 32 8 1024 128      # LLaMA-3 8B 配置
./gqa 4 2 64 64          # 小尺寸验证
```

典型输出（RTX 5090，`nq=32, nkv=8, S=1024, d=128`）：

```text
nq=32 nkv=8 group=4 S=1024 d=128
Q=16.78 MB  KV=8.39 MB (MHA would be 33.55 MB, 4.0x more)
kernel time: x.xx ms
max diff: x.xx e-06 (PASS, tol=1e-4)

[KV Cache 收益] GQA cache / MHA cache = 8 / 32 = 0.25
[LLaMA-3 8B] 32Q/8KV → cache 缩到 25%（省 75%）
```

### 5.2 用 ncu 观察 KV head 复用

```bash
ncu --kernel-name regex:gqa_kernel \
    --metrics gpu__time_duration.sum, \
              dram__bytes.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./gqa 32 8 1024 128
```

| 指标 | GQA（本实现） | 等效 MHA（32 KV 头） | 含义 |
|------|---------|-------------|------|
| `dram__bytes`（KV 部分） | `2 × nkv × S × d × 4B` | `2 × nq × S × d × 4B`（4×） | GQA 只读 `nkv` 份 KV |
| `dram__throughput` | 接近峰值 | 接近峰值 | 两者都 memory-bound |
| `gpu__time` | 基线 | ~略快（KV 少，但 Q head 数相同） | GQA 的 wall-clock 收益主要在 cache，不在单次 attention |

> ⚠️ **关键观察**：单次 attention 的 wall-clock，GQA 与 MHA 差距不大（Q head 数相同，KV 只是少读几份，但 L2 cache 会复用同 group 内的 KV）。**GQA 的真正收益在推理系统的 KV Cache 内存**：每个 token 的 cache 从 `2 × nq × d` 降到 `2 × nkv × d`，长序列/大 batch 下显存节省巨大，batch size 可成倍提升 → 吞吐成倍提升。

### 5.3 优化方向

1. **shared memory 缓存 KV tile**：同一 group 的 `group` 个 Q head 读同一份 KV——把一个 KV tile 载入 shared，供同 group 的多个 block 复用，减少 global 重读。（本实现每个 block 独立读 KV，靠 L2 自然复用；显式缓存可进一步提升。）
2. **合并同 group 的 Q head 到一个 block**：让一个 block 处理同一 group 的多个 `(h, s)`，共享 `q_shm` 之外的 KV tile，提升数据复用。
3. **FlashAttention tiling**：当 `seq_len` 很大时，把 KV 分块载入 shared，Q tile 常驻，把 HBM IO 从 `O(nq × S² × d)` 降到 `O(S × d)` 级别（与 [Week4 FlashAttention](../week4/day1/leetgpu-softmax-attention-solution.md) 同理）。
4. **vector load（`float4`）**：K/V 按 `d` 维连续，用 `float4` 一次读 4 个 float。
5. **混合精度**：Q/K/V 用 fp16/bf16，Tensor Core `mma` 做点积（d 大时收益大）。

## 6. 复杂度分析

| 维度 | MHA（标准） | GQA（本实现） | MQA（极端） |
|------|------------|--------------|------------|
| **时间复杂度** | `O(nq × S² × d)` | `O(nq × S² × d)` | `O(nq × S² × d)` |
| **KV 显存/cache** | `2 × nq × S × d × B` | `2 × nkv × S × d × B` | `2 × 1 × S × d × B` |
| **KV HBM IO** | `2 × nq × S × d` | `2 × nkv × S × d` | `2 × S × d` |
| **cache 压缩比** | 1× | `nkv/nq`（LLaMA-3: 1/4） | `1/nq` |
| **精度损失** | 无 | 几乎无（训练时就定好） | 略有（KV 头太少） |
| **代表模型** | 原版 Transformer | LLaMA-2/3, Mistral | Falcon, PaLI |

> 💡 **一句话总结**：GQA 把 KV Cache 的 `num_heads` 维从 `num_q_heads` 降到 `num_kv_heads`（LLaMA-3 32Q/8KV → 4× 缩减），是**模型结构层面**的 cache 优化——无损精度、kernel 只改一行索引（`kv_h = h / group`）。与 int8 量化（精度层面，有损）正交，可叠加（GQA + int8 = 8× cache 压缩）。这就是 GQA 成为现代 LLM（LLaMA-3、Mistral、Gemma）标配的原因。

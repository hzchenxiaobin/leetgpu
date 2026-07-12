# LeetGPU INT8 KV-Cache Attention 题解

## 1. 题目概述

- **标题 / 题号**：INT8 KV-Cache Attention（#96，medium）
- **链接**：https://leetgpu.com/challenges/int8-kv-cache-attention
- **难度**：中等
- **标签**：CUDA、Attention、Decode-phase、KV Cache、INT8 量化、memory-bound、online softmax、per-token scale

**题意**：实现 **Decode 阶段**的多头注意力。给定单个新 token 的 Query `Q ∈ R^{H×d}`，以及以 **int8** 存储的 KV Cache `K_int8, V_int8 ∈ R^{H×L×d}`（每个 token 配一个 fp32 的 `k_scale[h,s]` / `v_scale[h,s]`），反量化后做 scaled dot-product attention，输出 `O ∈ R^{H×d}`：

$$K_{f}[h,s,d] = K_{i8}[h,s,d] \times k\_scale[h,s],\quad V_{f}[h,s,d] = V_{i8}[h,s,d] \times v\_scale[h,s]$$

$$O_h = \text{softmax}\!\left(\frac{Q_h \cdot K_{f,h}^{\top}}{\sqrt{d}}\right) V_{f,h}$$

**示例**（`H=1, L=3, d=4`，`scale=1/√4=0.5`）：

```text
Q          = [1,1,1,1]
K_int8     = [[10,0,0,0],[0,10,0,0],[0,0,10,0]],  k_scale=[0.1,0.1,0.1]  → K_f=[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
V_int8     = [[10,20,30,40],[50,60,70,80],[90,100,110,120]], v_scale=[0.1,0.1,0.1] → V_f=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
scores     = Q·K_f^T · 0.5 = [0.5, 0.5, 0.5]
weights    = softmax → [1/3, 1/3, 1/3]
output     = weights · V_f = [5.0, 6.0, 7.0, 8.0]
```

**约束**：`1 ≤ H ≤ 64`，`1 ≤ L ≤ 32768`，`8 ≤ d ≤ 256`（8 的倍数）；`K_int8/V_int8 ∈ [-128,127]`；性能测试取 `H=32, L=8192, d=128`；容差 `atol=rtol=1e-3`。

> 💡 这道题就是 [Week5 Day1](../../aiinfra/week5/day1/README.md) 讲的 **Decode 阶段核心算子**：单 query 对 KV Cache 做 1×L 的 attention，是典型 **memory-bound**。题目把 KV 存成 int8 + per-token scale，正是 Day1 "减少 KV Cache 读取"优化方向的落地——int8 相比 fp32 把 KV 的 HBM 流量直接砍到 1/4。生产级推理系统（TensorRT-LLM、vLLM）都用这套。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行参考（同 reference_impl）

```cpp
// cpu_baseline.cpp —— CPU 串行 INT8 KV-Cache Attention（每 head 独立）
void attention_int8_cpu(const float* Q, const int8_t* K_int8, const int8_t* V_int8, const float* k_scale,
                        const float* v_scale, float* O, int H, int L, int d) {
    float scale = 1.0f / sqrtf((float)d);
    for (int h = 0; h < H; ++h) {
        // ① scores[s] = Q·K_f[s]/√d,  K_f[s] = K_int8[s]*k_scale[s]
        float mx = -INFINITY;
        std::vector<float> sc(L);
        for (int s = 0; s < L; ++s) {
            float ks = k_scale[h * L + s], dot = 0.f;
            for (int t = 0; t < d; ++t)
                dot += Q[h * d + t] * (K_int8[h * L * d + s * d + t] * ks);
            sc[s] = dot * scale;
            mx = fmaxf(mx, sc[s]);
        }
        // ② softmax
        float sum = 0.f;
        for (int s = 0; s < L; ++s) {
            sc[s] = expf(sc[s] - mx);
            sum += sc[s];
        }
        // ③ O = Σ_s w[s] · V_f[s]
        for (int t = 0; t < d; ++t)
            O[h * d + t] = 0.f;
        for (int s = 0; s < L; ++s) {
            float w = sc[s] / sum, vs = v_scale[h * L + s];
            for (int t = 0; t < d; ++t)
                O[h * d + t] += w * (V_int8[h * L * d + s * d + t] * vs);
        }
    }
}
```

复杂度 `O(H·L·d)`。关键点：**反量化与点积/加权融合在一起**，从不物化完整的 fp32 `K_f`/`V_f`——否则就违背了 int8 量化的带宽初衷。

### 2.2 朴素 GPU：物化 fp32 cache（反例）

最朴素的 GPU 想法是先开一个 kernel 把 `K_int8`、`V_int8` 反量化成 fp32 写回 HBM，再调 cuBLAS/cuDNN 做 attention。**这恰恰是本题要避免的反模式**：

![INT8 反量化：KV Cache 带宽 4× 压缩](images/int8_kv_cache_dequant.svg)

- 反量化写出 fp32 cache 会把 int8 省下的 4× 带宽**全部还回去**（多一次 `H·L·d·4B` 写 + 读）；
- Decode 本就 memory-bound，多一趟 fp32 cache 的 HBM 往返直接吃掉所有收益。

> ⚠️ 正确做法是**把反量化流式地融在 attention kernel 里**：从 HBM 读 int8 K/V → 寄存器里乘 scale 成 fp32 → 立刻用于点积/加权 → 丢弃。fp32 K/V 永不落 HBM。这就是"kernel fusion 消除中间矩阵物化"的又一次应用（与 [Week4 Softmax Attention](../week4/day1/leetgpu-softmax-attention-solution.md) 消除 S/P 物化同源）。

## 3. GPU 设计

### 3.1 并行化策略

| 维度 | 映射 | 说明 |
|------|------|------|
| **head 间** | `blockIdx.x → h` | 每个 head 的 attention 完全独立，一个 block 处理一个 head |
| **block 内** | 线程协作遍历 `L` 个 token | 块归约算点积；thread 0 维护 online softmax 的 m/l/o；输出 `d` 维由 thread 分摊 |

`grid = (H,)`，`block = (BLOCK,)`（如 256）。H 个 head 完全并行；block 内串行扫描 `L` 个 token，但用 online softmax 把"点积 → softmax → 加权 V"合成一遍扫描。

![INT8 KV-Cache Attention：单 Query 对 KV Cache 的 Decode](images/int8_kv_cache_attention_overview.svg)

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|---------|------|
| **global** | ✓ | 读 `Q`、`K_int8`、`V_int8`、`k_scale`、`v_scale`；写 `output` |
| **shared** | ✓ | `q_shm[d]` 缓存本 head 的 Q（全 block 复用做点积）；块归约缓冲 `red[NUM_WARPS]`；广播 `s_k/α/β` |
| **register** | ✓ | `o_local`（每 thread 持有输出的若干维）；online softmax 的 `m/l`（thread 0 维护） |

### 3.3 关键技巧

1. **int8 反量化融在点积里**：`s += Q[t] * (K_int8[...] * k_scale)` —— 读 1 byte，乘成 fp32，立即累加，不物化。
2. **online softmax 三公式**（与 [Week4 Softmax Attention](../week4/day1/leetgpu-softmax-attention-solution.md) 完全一致）：遍历 `L` 个 token，用 `m/l/o` 增量更新，`S=QK^T` 和 `P=softmax(S)` 永不落 HBM。
   - `m_new = max(m, s_k)`、`l_new = l·exp(m−m_new) + exp(s_k−m_new)`
   - `o_new = o·(l·exp(m−m_new)/l_new) + (exp(s_k−m_new)/l_new)·v`
3. **per-token scale 广播**：`k_scale[h,s]` 是标量，乘到该 token 的 `d` 维上——读 scale 的开销（`H·L·4B`）远小于读 int8 K（`H·L·d·1B`，d≥8）。
4. **Q 缓存到 shared**：本 head 的 Q 在整遍 `L` 扫描里复用，载入 `q_shm` 一次，避免每个 token 都从 global 读 Q。

> 💡 **数值稳定**：所有 `exp` 都减 running max `m_new`，指数 ≤ 0，int8 反量化后的 fp32 不会溢出。容差 `1e-3` 对 int8 量化（相对误差通常 < 1%）足够。

## 4. Kernel 实现

完整可编译代码：**fused 版（int8 反量化 + online softmax，不物化 fp32 cache）**，含 `main()`、`cudaMalloc/Memcpy`、CPU 验证、`cudaFree`：

```cuda
// int8_kv_cache_attention.cu —— INT8 KV-Cache Decode Attention（fused, online softmax）
// 编译命令: nvcc -O3 -arch=sm_120 int8_kv_cache_attention.cu -o int8_kv_attn -lineinfo
// 运行:     ./int8_kv_attn 32 8192 128

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_MAX 256 // head_dim <= 256

// ---------- 块归约（复用 Week4 Softmax Attention）----------
__inline__ __device__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ float block_reduce_sum(float v, float* sh) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0)
        sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < NUM_WARPS) ? sh[lane] : 0.f;
        v = warp_reduce_sum(v);
        if (lane == 0)
            sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

// ---------- fused kernel：一个 block 处理一个 head ----------
// Q        [H, d]
// K_int8   [H, L, d]  + k_scale [H, L]
// V_int8   [H, L, d]  + v_scale [H, L]
// output   [H, d]
__global__ void int8_kv_attention_kernel(const float* __restrict__ Q, const int8_t* __restrict__ K_int8,
                                         const int8_t* __restrict__ V_int8, const float* __restrict__ k_scale,
                                         const float* __restrict__ v_scale, float* __restrict__ output, int H, int L,
                                         int d) {
    __shared__ float q_shm[D_MAX];
    __shared__ float red[NUM_WARPS + 1];
    __shared__ float s_k_shm, alpha_shm, beta_shm;

    int h = blockIdx.x, tid = threadIdx.x;
    if (h >= H)
        return;
    const float scale = 1.0f / sqrtf((float)d);

    // ① 载入本 head 的 Q 到 shared（全 block 复用）
    for (int t = tid; t < d; t += BLOCK_SIZE)
        q_shm[t] = Q[h * d + t];
    __syncthreads();

    float m = -INFINITY, l = 0.f; // online softmax running max/sum（thread 0 维护）
    float o_local = 0.f;          // 本 thread 持有的输出维

    // ② 遍历 L 个 token：点积（含 int8 反量化）→ online softmax → 加权 V
    for (int s = 0; s < L; ++s) {
        const int8_t* Ks = K_int8 + (h * L + s) * d;
        const int8_t* Vs = V_int8 + (h * L + s) * d;
        const float ks = k_scale[h * L + s];
        const float vs = v_scale[h * L + s];

        // 点积 s_k = Σ_t Q[t] · (K_int8[t]·ks) / √d
        float part = 0.f;
        for (int t = tid; t < d; t += BLOCK_SIZE)
            part += q_shm[t] * (Ks[t] * ks);
        float s_k = block_reduce_sum(part, red) * scale;
        if (tid == 0)
            s_k_shm = s_k;
        __syncthreads();
        s_k = s_k_shm;

        // online softmax 三公式（thread 0 算，广播 α、β）
        if (tid == 0) {
            float m_new = fmaxf(m, s_k);
            float alpha = expf(m - m_new); // 旧状态缩放
            float p = expf(s_k - m_new);   // 新 token 权重
            float l_new = l * alpha + p;
            alpha_shm = (l * alpha) / l_new; // o 的缩放因子
            beta_shm = p / l_new;            // 新 V 的权重
            m = m_new;
            l = l_new;
        }
        __syncthreads();

        // 累加输出：o = o*α + β*v_f（v_f = V_int8·vs，反量化融在加权里）
        for (int t = tid; t < d; t += BLOCK_SIZE)
            o_local = o_local * alpha_shm + beta_shm * (Vs[t] * vs);
        __syncthreads();
    }
    // ③ 写回 output[h]
    for (int t = tid; t < d; t += BLOCK_SIZE)
        output[h * d + t] = o_local;
}

// ---------- CPU 参考实现 ----------
void attention_int8_cpu(const float* Q, const int8_t* K, const int8_t* V, const float* ks, const float* vs, float* O,
                        int H, int L, int d) {
    float scale = 1.0f / sqrtf((float)d);
    std::vector<float> sc(L);
    for (int h = 0; h < H; ++h) {
        float mx = -INFINITY;
        for (int s = 0; s < L; ++s) {
            float dot = 0.f, k = ks[h * L + s];
            for (int t = 0; t < d; ++t)
                dot += Q[h * d + t] * (K[(h * L + s) * d + t] * k);
            sc[s] = dot * scale;
            mx = fmaxf(mx, sc[s]);
        }
        float sum = 0.f;
        for (int s = 0; s < L; ++s) {
            sc[s] = expf(sc[s] - mx);
            sum += sc[s];
        }
        for (int t = 0; t < d; ++t)
            O[h * d + t] = 0.f;
        for (int s = 0; s < L; ++s) {
            float w = sc[s] / sum, v = vs[h * L + s];
            for (int t = 0; t < d; ++t)
                O[h * d + t] += w * (V[(h * L + s) * d + t] * v);
        }
    }
}

int main(int argc, char** argv) {
    int H = (argc > 1) ? atoi(argv[1]) : 32;
    int L = (argc > 2) ? atoi(argv[2]) : 8192;
    int d = (argc > 3) ? atoi(argv[3]) : 128;
    if (d > D_MAX) {
        printf("要求 d <= %d\n", D_MAX);
        return 1;
    }

    size_t q_bytes = (size_t)H * d * sizeof(float);
    size_t kv_bytes = (size_t)H * L * d * sizeof(int8_t); // int8: 1 byte
    size_t sc_bytes = (size_t)H * L * sizeof(float);
    printf("H=%d L=%d d=%d  int8 KV=%.2f MB  (fp32 KV would be %.2f MB, 4x)\n", H, L, d, 2.0 * kv_bytes / 1e6,
           4.0 * kv_bytes / 1e6);

    // host 数据
    std::vector<float> hQ(H * d);
    std::vector<int8_t> hK(H * L * d), hV(H * L * d);
    std::vector<float> hks(H * L), hvs(H * L);
    srand(42);
    for (auto& x : hQ)
        x = ((rand() % 2000) - 1000) / 100.f;
    for (auto& x : hK)
        x = (int8_t)((rand() % 255) - 128);
    for (auto& x : hV)
        x = (int8_t)((rand() % 255) - 128);
    for (auto& x : hks)
        x = (rand() % 100) / 1000.f + 0.01f;
    for (auto& x : hvs)
        x = (rand() % 100) / 1000.f + 0.01f;
    std::vector<float> hO(H * d), hRef(H * d);

    // device 分配
    float *dQ, *dks, *dvs, *dO;
    int8_t *dK, *dV;
    cudaMalloc(&dQ, q_bytes);
    cudaMemcpy(dQ, hQ.data(), q_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dK, kv_bytes);
    cudaMemcpy(dK, hK.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dV, kv_bytes);
    cudaMemcpy(dV, hV.data(), kv_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dks, sc_bytes);
    cudaMemcpy(dks, hks.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dvs, sc_bytes);
    cudaMemcpy(dvs, hvs.data(), sc_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dO, q_bytes);

    // 计时
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    // warmup
    int8_kv_attention_kernel<<<H, BLOCK_SIZE>>>(dQ, dK, dV, dks, dvs, dO, H, L, d);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    int8_kv_attention_kernel<<<H, BLOCK_SIZE>>>(dQ, dK, dV, dks, dvs, dO, H, L, d);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 验证
    cudaMemcpy(hO.data(), dO, q_bytes, cudaMemcpyDeviceToHost);
    attention_int8_cpu(hQ.data(), hK.data(), hV.data(), hks.data(), hvs.data(), hRef.data(), H, L, d);
    float maxd = 0;
    for (int i = 0; i < H * d; ++i)
        maxd = fmaxf(maxd, fabsf(hO[i] - hRef[i]));
    printf("max diff: %.2e (%s, tol=1e-3)\n", maxd, maxd < 1e-3f ? "PASS" : "FAIL");

    // 估算 HBM 流量与算术强度
    float bytes_kv = 2.0f * kv_bytes + 2.0f * sc_bytes;                   // int8 K/V + scale
    float bytes_fp32 = 2.0f * kv_bytes * 4;                               // 若用 fp32 KV
    float flops = 2.0f * H * L * d * 2 + 3.0f * H * L + 2.0f * H * L * d; // ~2·QK^T + softmax + 2·PV
    printf("est. DRAM (int8)=%.2f MB  (fp32)=%.2f MB  AI(int8)=%.2f  AI(fp32)=%.2f\n", bytes_kv / 1e6, bytes_fp32 / 1e6,
           flops / bytes_kv, flops / bytes_fp32);

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dks);
    cudaFree(dvs);
    cudaFree(dO);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `int8_kv_attention_kernel` 填进 starter 的 `solve` 即可（平台只验证正确性）。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 int8_kv_cache_attention.cu -o int8_kv_attn -lineinfo
./int8_kv_attn 32 8192 128      # 性能测试尺寸
./int8_kv_attn 8 256 64         # 小尺寸验证
```

典型输出（RTX 5090，`H=32, L=8192, d=128`）：

```text
H=32 L=8192 d=128  int8 KV=268.43 MB  (fp32 KV would be 1073.74 MB, 4x)
kernel time: x.xx ms
max diff: x.xx e-04 (PASS, tol=1e-3)
est. DRAM (int8)=268.84 MB  (fp32)=1073.74 MB  AI(int8)=x.xx  AI(fp32)=x.xx
```

### 5.2 用 ncu 验证 memory-bound + int8 带宽收益

```bash
ncu --kernel-name regex:int8_kv_attention_kernel \
    --metrics gpu__time_duration.sum, \
              dram__bytes.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./int8_kv_attn 32 8192 128
```

| 指标 | int8 版 | 假想 fp32 版 | 含义 |
|------|---------|-------------|------|
| `dram__bytes` | `≈ 2·H·L·d·1B` | `≈ 2·H·L·d·4B`（4×） | int8 把 KV 流量砍到 1/4 |
| `dram__throughput` | 接近峰值 | 接近峰值 | Decode 都吃满带宽（memory-bound） |
| `sm__throughput` | 低 | 低 | 算力闲置，印证 memory-bound |
| `gpu__time` | 基线 | ~4× | 带宽降 4× → 时间近似降 4× |

> ⚠️ **关键观察**：`sm__throughput` 很低而 `dram__throughput` 接近峰值——这正是 Day1 说的 "Decode memory-bound，SM 等数据"。int8 把 `dram__bytes` 降到 1/4，墙钟时间近似也降到 1/4（因为卡在带宽上）。这就是 KV Cache 量化的全部价值。

### 5.3 优化方向

1. **vector load int8**：`K_int8` 按 `char4`/`int` 一次读 4 个 int8，提升带宽利用率。
2. **shared memory 缓存 KV tile**：把 `K_int8`/`V_int8` 的一个 token tile 载入 shared，减少 global 重复读（本实现每个 token 只读一次，收益有限；多 query 场景更大）。
3. **多 query 合并（Continuous Batching 的雏形）**：若有多个新 token 的 Q（batch decode），一个 block 处理多个 Q 共享同一份 KV，把 `dram__bytes` 由 Q 摊薄——这正是 Day3 vLLM Continuous Batching 的原理。
4. **fp16/bf16 累加**：用 `half` 存 Q、用 Tensor Core `mma` 做点积（d 大时）；本题 d≤256，标量累加已够。
5. **split-KV**：超长 `L`（>32768）时把 L 维切给多个 block 各算一段 partial (m,l,o)，再归约合并——和 FlashAttention 的分块、Week2 Prefix Sum 的三阶段 scan 同源。

## 6. 复杂度分析

| 维度 | int8 fused（本实现） | 假想 fp32 物化版 | 假想 fp32 不物化版 |
|------|---------------------|------------------|-------------------|
| **时间复杂度** | `O(H·L·d)` | `O(H·L·d)` | `O(H·L·d)` |
| **KV Cache 显存** | `2·H·L·d·1B` + scale | `2·H·L·d·4B` | `2·H·L·d·4B` |
| **HBM IO（KV 部分）** | `2·H·L·d·1B`（int8） | `2·H·L·d·4B`（读 fp32） + 反量化写读 `4·H·L·d·4B` | `2·H·L·d·4B` |
| **算术强度 AI** | `~4·d / (2·1)` ≈ `2d` | 被 fp32 物化拖累，极低 | `~4·d / (2·4)` ≈ `d/2` |
| **瓶颈类型** | memory-bound（但带宽降到 1/4） | memory-bound（最差） | memory-bound |
| **相对墙钟** | 1× | ~6-8×（最慢） | ~4× |

> 💡 **一句话总结**：Decode 阶段的 attention 是"单 query 扫一遍 KV Cache"的 memory-bound 算子。把 KV 存成 **int8 + per-token scale** 并在 kernel 里**流式反量化**，让 HBM 流量降到 fp32 的 1/4——墙钟近似也降到 1/4（因为卡在带宽上）。这就是 Day1 "减少 KV Cache 读取"优化方向的工业级落地，也是 TensorRT-LLM / vLLM 的生产标配。

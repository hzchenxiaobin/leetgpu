# LeetGPU GEMV 题解

## 1. 题目概述

- **标题 / 题号**：GEMV（Matrix-Vector Multiplication，#114，medium）
- **链接**：https://leetgpu.com/challenges/gemv
- **难度**：中等
- **标签**：CUDA、GEMV、memory-bound、roofline、合并访存、block 归约、vectorized load

**题意**：给定权重矩阵 $W$（`N×K`，row-major）、输入向量 $x$（`K`）和偏置 $bias$（`N`），计算 $y = W x + bias$。这正是 GEMM 在 $M=1$ 时的特例——**单 token 自回归解码**中一个 `Linear` 层的核心计算。

$$
y[n] = \sum_{k=0}^{K-1} W[n][k] \cdot x[k] + bias[n], \qquad n = 0, \dots, N-1
$$

**示例**（`N=4, K=4`）：

```text
W = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9,10,11,12],
     [13,14,15,16]]
x    = [1, 0, 1, 0]
bias = [0, 0, 0, 0]

y[0] = 1·1 + 2·0 + 3·1 + 4·0 = 4
y[1] = 5·1 + 6·0 + 7·1 + 8·0 = 12
y[2] = 9·1 + 10·0 + 11·1 + 12·0 = 20
y[3] = 13·1 + 14·0 + 15·1 + 16·0 = 28
→ y = [4, 12, 20, 28]
```

**约束**：`1 ≤ N, K ≤ 16384`；性能测点 `N=4096, K=4096`（LLaMA-2-7B 隐藏维风格；MLP up-projection 的 `11008×4096` 同理）；容差 `atol = rtol = 1e-4`。

> 💡 本题是 [Simple Inference (#41)](./leetgpu-simple-inference-solution.html) 的**极致反面**：#41 用大 `batch_size` 把 GEMM 推到 compute-bound，而 GEMV 是 `batch_size=1` 的极端——**算术强度只有 0.5 FLOP/byte，深度 memory-bound**。LLM 自回归解码（逐 token 生成）阶段几乎每个 `Linear` 都是 GEMV，这正是解码阶段被带宽卡住的根因，也是 Continuous Batching / speculative decoding 等系统优化的动机。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 GEMV
void gemv_cpu(const float* W, const float* x, const float* bias,
              float* y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float sum = bias[n];
        for (int k = 0; k < K; ++k)
            sum += W[n * K + k] * x[k];
        y[n] = sum;
    }
}
```

双重循环 $O(N \cdot K)$。性能测点下约 **3350 万次乘加**，单核需数毫秒，但完全无法利用并行性。

### 2.2 朴素 GPU：一个 thread 一个输出行，串行 K 循环

![GEMV 朴素 vs 优化](/images/gemv_overview.svg)

```cuda
// gemv_naive.cu —— 朴素 GEMV：1 thread / row，串行 K，W 跨行非合并读取
__global__ void gemv_naive(const float* W, const float* x, const float* bias,
                           float* y, int N, int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // 一个 thread 负责一行
    if (n >= N) return;
    float sum = bias[n];
    for (int k = 0; k < K; ++k)
        sum += W[n * K + k] * x[k];   // W 跨行 stride=K 读取
    y[n] = sum;
}
```

**瓶颈**：

1. **W 的跨行非合并访存**：相邻 thread（`n, n+1`）在同一 `k` 读取 `W[n*K+k]` 与 `W[(n+1)*K+k]`，地址相距 `K` 个 float（性能测点下 `K=4096` → 16 KB stride）。一个 128 B cache line 只命中 1 个有用 float，**有效带宽利用率约 1/32**。
2. **K 维串行**：每个 thread 串行走完整 `K`，并行度只取决于 `N`；`N` 小时 SM 大量空闲。
3. **x 被 N 个 thread 重复读**：好在 `x[k]` 对所有 thread 是同一地址，硬件广播 + L2 缓存能扛住，不是主要矛盾。

> ⚠️ 朴素版真正的杀手是 **W 的非合并读取**——它把宝贵的 HBM 带宽浪费在搬运无用数据上。优化版的核心就是把「沿 N 并行、K 串行」翻转为「沿 K 并行（同一行内 thread 连续读 W）、再归约」。

## 3. GPU 设计

### 3.1 并行化策略：1 block per row，threads 切分 K 维 + 归约

![GEMV block 归约结构](/images/gemv_block_reduce.svg)

核心思想：**每个 block 负责一个输出元素 $y[n]$，block 内 `BLOCK_DIM` 个 thread 协作切分长度为 $K$ 的点积**。

- **grid**：`(N, 1, 1)`，第 `n` 个 block 算 `y[n]`。
- **block**：`BLOCK_DIM=256` 个 thread，沿 K 维 grid-stride：thread `tid` 处理 `k = tid, tid+256, tid+512, ...`。
- **合并访存**：同一 block（固定 `n`）内，相邻 `tid` 读 `W[n*K + tid]` → 地址连续 → **完全合并**，一个 cache line 命中 32 个 float。
- **x 的复用**：`x[k]` 被所有 `N` 个 block 重用，但 `x` 仅 `K·4` 字节（性能测点 16 KB），完全驻留 L2，跨 block 命中 → 等效「免费广播」。
- **归约**：每个 thread 持有一个 partial sum，经 **warp shuffle `__shfl_down_sync`** + shared memory 跨 warp 汇总，得到 `y[n] = sum + bias[n]`。

### 3.2 存储层次使用

| 数据 | 存储 | 访问模式 | 说明 |
|------|------|----------|------|
| `W` | global memory | 合并读 | `W[n*K+k]`，沿 k 连续，每个元素恰被读 1 次（带宽下界） |
| `x` | global → L2 缓存 | 广播 | `x[k]` 被 N 个 block 重用，16 KB 驻留 L2，等效无 HBM 流量 |
| `bias` / `y` | global memory | 标量 | 每 block 读 1 个 bias、写 1 个 y，可忽略 |
| partial sum | registers | 每 thread 1 个 `float` | 累加中间结果 |
| 跨 warp 汇总 | shared memory | `shared[32]` | 每 warp 1 个 slot，供 warp 0 做最终归约 |

### 3.3 关键技巧

- **沿 K 并行 + block 归约**：把朴素版「N 并行 / K 串行」翻转为「K 并行 / N 分块」，让 W 的读取从跨行 stride 变为行内连续 → 合并访存，带宽利用率从 ~1/32 提升到 ~1。
- **`float4` 向量化加载**：每个 thread 一次读 4 个 float（16 B 对齐），减少 load 指令数、提升单位指令搬运的字节数，进一步逼近带宽峰值。
- **warp shuffle 归约**：warp 内用 `__shfl_down_sync` 做树形归约（无 shared memory、无 bank conflict），仅最后跨 warp 用 `shared[32]` 汇总。
- **x 驻留 L2**：不显式开 shared memory 缓存 `x`——`x` 极小且被全体 block 共享，L2 自然广播比 per-block shared 更高效（shared 无法跨 block 共享）。

## 4. Kernel 实现

### 4.1 LeetGPU 提交版本

```cuda
// gemv.cu —— GEMV：1 block per row，threads 切分 K + block 归约 + float4 向量化
// 编译: nvcc -O3 -arch=sm_80 gemv.cu -o gemv
// 运行: ./gemv

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_DIM 256

// block 内归约：warp shuffle + shared 跨 warp 汇总
__device__ __forceinline__ float block_reduce(float val) {
    static __shared__ float shared[32];      // 每 warp 一个 slot（blockDim<=1024 → 最多 32 warp）
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid  = tid >> 5;

    // 阶段一：warp 内 5 步蝶形归约
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) shared[wid] = val;        // 每 warp 的和写入 shared
    __syncthreads();

    // 阶段二：warp 0 归约所有 warp 的部分和
    int num_warps = blockDim.x >> 5;
    val = (tid < num_warps) ? shared[tid] : 0.0f;
    if (wid == 0) {
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void gemv_kernel(const float* __restrict__ W,     // [N, K]
                            const float* __restrict__ x,     // [K]
                            const float* __restrict__ bias,  // [N]
                            float* __restrict__ y,           // [N]
                            int N, int K) {
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;

    float sum = 0.0f;
    const int VEC = 4;
    int Kv = K / VEC;                         // 假设 K % 4 == 0（性能测点满足）
    const float4* W4 = reinterpret_cast<const float4*>(W + n * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);

    // grid-stride 沿向量化 K 维累加
    for (int kv = tid; kv < Kv; kv += BLOCK_DIM) {
        float4 wv = W4[kv];                   // 合并 + 向量化读 W
        float4 xv = x4[kv];                   // x 命中 L2
        sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
    }
    // 处理 K % 4 的尾部（性能测点无尾部，保留以保正确性）
    int k_tail_start = Kv * VEC;
    for (int k = k_tail_start + tid; k < K; k += BLOCK_DIM)
        sum += W[n * K + k] * x[k];

    sum = block_reduce(sum);
    if (tid == 0) y[n] = sum + bias[n];
}

// ---------- 完整测试 harness ----------
void gemv_cpu(const float* W, const float* x, const float* bias,
              float* y, int N, int K) {
    for (int n = 0; n < N; ++n) {
        float s = bias[n];
        for (int k = 0; k < K; ++k) s += W[n * K + k] * x[k];
        y[n] = s;
    }
}

int main() {
    // 测试 1：官方 example (N=4, K=4)
    {
        int N = 4, K = 4;
        float h_W[]  = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float h_x[]  = {1,0,1,0};
        float h_b[]  = {0,0,0,0};
        float h_y[4], ref[4];

        float *d_W, *d_x, *d_b, *d_y;
        cudaMalloc(&d_W, N*K*sizeof(float));
        cudaMalloc(&d_x, K*sizeof(float));
        cudaMalloc(&d_b, N*sizeof(float));
        cudaMalloc(&d_y, N*sizeof(float));
        cudaMemcpy(d_W, h_W, N*K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, K*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

        gemv_kernel<<<N, BLOCK_DIM>>>(d_W, d_x, d_b, d_y, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

        gemv_cpu(h_W, h_x, h_b, ref, N, K);
        printf("=== Test 1: N=%d, K=%d ===\n", N, K);
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            printf("y[%d] = %.1f (ref %.1f)\n", i, h_y[i], ref[i]);
            if (fabsf(h_y[i] - ref[i]) > 1e-4f) ok = false;
        }
        printf("%s\n\n", ok ? "PASS" : "FAIL");

        cudaFree(d_W); cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
    }

    // 测试 2：随机大规模 (N=K=4096，性能测点)
    {
        int N = 4096, K = 4096;
        size_t sz_W = (size_t)N * K * sizeof(float);
        float *h_W = (float*)malloc(sz_W);
        float *h_x = (float*)malloc(K * sizeof(float));
        float *h_b = (float*)malloc(N * sizeof(float));
        float *h_y = (float*)malloc(N * sizeof(float));
        float *ref = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N * K; ++i) h_W[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        for (int i = 0; i < K; ++i) h_x[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        for (int i = 0; i < N; ++i) h_b[i] = (float)(rand() % 1000) / 1000.0f;

        float *d_W, *d_x, *d_b, *d_y;
        cudaMalloc(&d_W, sz_W);
        cudaMalloc(&d_x, K * sizeof(float));
        cudaMalloc(&d_b, N * sizeof(float));
        cudaMalloc(&d_y, N * sizeof(float));
        cudaMemcpy(d_W, h_W, sz_W, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x, K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

        gemv_kernel<<<N, BLOCK_DIM>>>(d_W, d_x, d_b, d_y, N, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

        gemv_cpu(h_W, h_x, h_b, ref, N, K);
        printf("=== Test 2: N=%d, K=%d ===\n", N, K);
        double max_err = 0.0;
        for (int i = 0; i < N; ++i)
            max_err = fmax(max_err, fabs((double)h_y[i] - ref[i]));
        printf("max abs err = %.3e  (%s)\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");

        cudaFree(d_W); cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
        free(h_W); free(h_x); free(h_b); free(h_y); free(ref);
    }

    printf("\nAll tests done.\n");
    return 0;
}
```

### 4.2 代码详解

`gemv_kernel` 采用 **"1 block per row + K 维切分 + block 归约"** 结构：每个 block 固定一行 `n`，256 个 thread 沿 K 维 grid-stride 各算一段点积，最后两阶段归约得到 `y[n]`。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **block 映射** | `n = blockIdx.x` | 1 个 block 负责一个输出元素 `y[n]`，N 个 block 并行 |
| **K 维切分** | `for (kv = tid; kv < Kv; kv += BLOCK_DIM)` | thread 沿向量化 K 做 grid-stride，相邻 tid → 相邻地址 → 合并 |
| **向量化读 W** | `float4 wv = W4[kv]` | 一次读 4 个 float，需 `K%4==0`；地址 16 B 对齐 |
| **x 读** | `float4 xv = x4[kv]` | `x` 被 N 个 block 重用，命中 L2 广播，无 HBM 流量 |
| **点积累加** | `sum += wv.x*xv.x + ...` | 每 thread 持有 register 内 partial sum |
| **warp 归约** | `__shfl_down_sync(..., off)` | warp 内树形归约，lane 0 得 warp 和；无 shared/bank conflict |
| **跨 warp 汇总** | `shared[wid] = val` → warp 0 再归约 | `shared[32]` 容纳最多 32 个 warp（block≤1024） |
| **写回** | `y[n] = sum + bias[n]` | 仅 tid 0 写 1 个输出，bias 在最后加（避免每 thread 都加再归约） |

**关键索引关系**：

| 变量 | 公式 | 含义 |
|------|------|------|
| `n` | `blockIdx.x` | 输出行索引（0 ≤ n < N） |
| `tid` | `threadIdx.x` | block 内线程号（0 ≤ tid < 256） |
| `kv` | `tid, tid+256, ...` | 向量化 K 维索引（grid-stride） |
| `W[n][k]` 地址 | `n*K + k` | 固定 n、k 连续 → 合并访存的关键 |
| `num_warps` | `blockDim.x >> 5` | block 内 warp 数（256 → 8） |

**Worked Example**（`N=4, K=4, BLOCK_DIM=4`，即每 block 4 thread 各算 1 个 k）：

![GEMV worked example](/images/gemv_worked.svg)

以 block `n=0` 为例（`W[0]=[1,2,3,4]`, `x=[1,0,1,0]`）：

```text
thread 0: W[0][0]*x[0] = 1·1 = 1
thread 1: W[0][1]*x[1] = 2·0 = 0
thread 2: W[0][2]*x[2] = 3·1 = 3
thread 3: W[0][3]*x[3] = 4·0 = 0
                                    block reduce → 1+0+3+0 = 4
                                    y[0] = 4 + bias[0] = 4 ✓
```

四个 block 并行得到 `y = [4, 12, 20, 28]`。

> 💡 **关键洞察**：GEMV 的本质瓶颈是 **W 的读取**——每个 `W[n][k]` 恰好被用一次，`N·K·4` 字节是不可压缩的带宽下界。优化的全部意义在于「**用合并 + 向量化把这个下界读得尽可能快**」，而非减少读取量。朴素版的问题不是读得多，而是读得「散」——跨行 stride 让带宽浪费在无用 cache line 上。把并行轴从 N 翻转到 K，W 的访问立刻从 stride 变连续，带宽利用率从 ~1/32 跃升到 ~1。

## 5. 性能分析与优化

```bash
# 编译
nvcc -O3 -arch=sm_80 gemv.cu -o gemv

# ncu profiling（性能测点 N=K=4096）
ncu --set full \
    --kernel-name gemv_kernel \
    --launch-skip 1 --launch-count 1 \
    ./gemv 2>&1 | \
    grep -iE "Memory Throughput|Compute|Occupancy|dram__bytes|dram__throughput|sm__throughput|Achieved"
```

**算术强度与 roofline**（性能测点 `N=K=4096`）：

```text
FLOPs       = 2·N·K        = 2·4096·4096 ≈ 3.36e7 FLOP
HBM 下界    = N·K·4 (读 W) + K·4 (读 x) + N·4 (读 bias) + N·4 (写 y)
            ≈ 64 MB + 16 KB + 16 KB + 16 KB ≈ 64 MB
算术强度    = FLOPs / bytes ≈ 3.36e7 / 6.4e7 ≈ 0.5 FLOP/byte   ← 深度 memory-bound
```

| 指标 | 朴素版（N 并行 / K 串行） | 优化版（K 并行 + 归约） |
|------|--------------------------|------------------------|
| W 访问模式 | 跨行 stride=K，非合并 | 行内连续，完全合并 |
| 有效带宽利用率 | ~1/32（cache line 仅 1 float 有用） | ~1.0（float4 + 合并） |
| 算术强度 | 0.5 FLOP/byte | 0.5 FLOP/byte（不变，W 仍读一次） |
| 瓶颈 | 非合并读 W → 带宽被浪费 | HBM 带宽本身（已逼近峰值） |
| kernel launch | 1 次 | 1 次 |

> ⚠️ 优化版**没有改变算术强度**（W 仍读一次），但把有效带宽从峰值的 ~3% 提到 ~80%+。这印证了 memory-bound kernel 优化的第一原则：**先确保合并访存与向量化，再谈别的**。

**与 M=large GEMM 的对比**（解释 batching 为何提升吞吐）：

| 场景 | M | W 复用次数 | 算术强度 | 瓶颈 |
|------|---|-----------|----------|------|
| GEMV（单 token） | 1 | 1 | 0.5 FLOP/byte | memory-bound |
| GEMM（batch=64） | 64 | 64 | 32 FLOP/byte | compute-bound |

W 在 GEMM 中被 M 个输入行复用 → 算术强度随 M 线性增长 → 从带宽受限翻转为算力受限。这正是 [Simple Inference (#41)](./leetgpu-simple-inference-solution.html) 中 `batch_size` 从 1 到 1000 吞吐提升 ~60 倍的根因。

**优化方向**：

1. **ROWS_PER_BLOCK**：每个 block 算多行（如 4 行），`x` 的 L2 命中转化为 shared memory 内显式复用，减少 `x` 的重复 cache miss。但 `x` 本就极小，收益有限。
2. **tensor core（wmma）的 M=1 陷阱**：Tensor Core 要求 M≥8（如 16×8×16），M=1 无法直接用 → 需把多个独立 GEMV 沿 M 维拼成 `M≥8` 的 GEMM，即 Continuous Batching 的 kernel 层动机。
3. **权重量化（INT8/INT4）**：把 W 压到 1 字节甚至 4 bit，带宽下界降 4×–8×，是解码阶段最有效的加速（参考 [INT8 Quantized MatMul (#32)](./leetgpu-int8-quantized-matmul-solution.html)、[INT4 Weight-Only (#81)](./leetgpu-int4-matmul-solution.html)）。
4. **KV/权重常驻 L2 提示**：对推理场景用 `cudaStreamSetAttribute` 设置 L2 持久化窗口，把热点 W 钉在 L2。

## 6. 复杂度分析

| 维度 | 朴素版 | 优化版 |
|------|--------|--------|
| **时间** | $O(N \cdot K)$ | $O(N \cdot K / \text{带宽})$，常数更小（合并 + 向量化） |
| **空间** | $O(N \cdot K)$ 输入 + $O(N)$ 输出 | 同左，额外 $O(\text{BLOCK\_DIM}/32)$ shared |
| **HBM 流量** | $N \cdot K \cdot 4$（理论下界，但非合并致有效带宽 ~3%） | $N \cdot K \cdot 4$（逼近带宽峰值） |
| **算术强度** | 0.5 FLOP/byte | 0.5 FLOP/byte |
| **瓶颈** | memory-bound（非合并读 W） | memory-bound（HBM 带宽本身，已接近 roofline） |

> 💡 **一句话总结**：GEMV 是 LLM 解码阶段的微缩骨架——$M=1$ 让算术强度跌到 0.5 FLOP/byte，**深度 memory-bound**。优化不靠减少计算，而靠「沿 K 并行 + 合并访存 + float4 向量化」把 W 的必读流量推到带宽峰值。真正的质变来自系统层：Continuous Batching 把多个 $M=1$ GEMV 拼成 $M \gg 1$ 的 GEMM，让算术强度随 batch 线性增长，从带宽受限翻转为算力受限——这与权重量化（降带宽下界）一起，是解码加速的两条主线。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 2 | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) | 简单 | tiled matmul、register tiling | GEMV 是 $M=1$ 的 GEMM，对比 compute-bound tiled GEMM 与 memory-bound GEMV 的优化差异 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | block 归约、warp shuffle | GEMV 每个输出元素即一个 dot product，block 归约是本题的核心组件 |
| 41 | [Simple Inference](https://leetgpu.com/challenges/simple-inference) | 简单 | PyTorch Linear、batch size | 对比 $M=1$（GEMV，memory-bound）与大 batch GEMM（compute-bound），解释 batching 为何提升吞吐 |
| 18 | [Sparse Matrix-Vector Multiplication](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication) | 中等 | CSR、稀疏、warp 归约 | GEMV 的稀疏变体（SpMV），对比稠密 vs 稀疏的访存模式与归约策略 |

> 💡 **选题思路**：memory-bound GEMV + 合并访存 + block 归约，练习 LLM 解码阶段的带宽受限 kernel 优化。做完这组练习，即可掌握 GEMM 类 kernel 在 $M=1$ 极端下的优化范式与 batching/量化等系统层加速手段的迁移应用。

# LeetGPU Layer Normalization 题解

## 1. 题目概述

- **标题 / 题号**：Layer Normalization（#115，medium）
- **链接**：https://leetgpu.com/challenges/layer-normalization
- **难度**：中等
- **标签**：CUDA、normalization、reduction、LayerNorm、memory-bound、warp shuffle、mean-centering

**题意**：给定 `M` 行 `D` 列的 `float32` 矩阵 `x`（行主序）、可学习权重 `gamma ∈ R^D` 和偏置 `beta ∈ R^D`，对**每一行独立**做 Layer Normalization——先减均值、再除标准差，最后做仿射变换：

$$
\mu = \frac{1}{D}\sum_{j=0}^{D-1} x_j, \qquad
\sigma^2 = \frac{1}{D}\sum_{j=0}^{D-1}(x_j - \mu)^2, \qquad
y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_i + \beta_i
$$

**示例**（单行 `D=4`，`eps=1e-5`，`gamma=1`，`beta=0`）：

```text
输入：      [1.0, 2.0, 3.0, 4.0]
μ          = (1+2+3+4)/4 = 2.5
deviation  = [-1.5, -0.5, 0.5, 1.5]
σ²         = (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25
rstd       = 1/√(1.25 + 1e-5) ≈ 0.89442
output     = [-1.34164, -0.44721, 0.44721, 1.34164]   // (x-μ)·rstd·γ + β
```

**约束**：

- `1 ≤ M × D ≤ 1,000,000`（总元素数）
- 元素范围 `[-10.0, 10.0]`
- 容差 `atol = rtol = 1e-4`
- 性能测试取较大 `M×D`（如 `M=128, D=8192`）

> 💡 LayerNorm 是 **Transformer（GPT/BERT）** 的标配归一化层，沿 **feature 维度**（最后一维 `D`）归一化，与 BatchNorm 沿 batch 维度归一化形成对照。它和 [RMS Normalization (#50)](./leetgpu-rms-normalization-solution.html) 是「归约 + 归一化」组合的两种形态：LayerNorm **两次归约**（先 mean，再 variance，且 variance 依赖 mean），RMSNorm 省掉 mean-centering 只需 **一次归约**。本题是练习「**串行两次块归约** + mean-centering 归一化」的经典模板，也是理解 Llama 为何用 RMSNorm 替代 LayerNorm 的性能参照。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 LayerNorm
void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                   float* y, int M, int D, float eps) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        // ① 求 mean
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += xr[i];
        float mean = sum / D;
        // ② 求 variance（依赖 mean）
        float sq = 0.0f;
        for (int i = 0; i < D; ++i) {
            float d = xr[i] - mean;
            sq += d * d;
        }
        float var = sq / D;
        // ③ 归一化 + affine
        float rstd = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < D; ++i)
            yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
    }
}
```

每行三遍扫描 `O(D)`，总计 `O(M×D)`。`M=128, D=8192` 时单核约 0.7-1.2ms。

### 2.2 朴素 GPU：每 thread 算一个元素（错误示范）

```cuda
// 错误示范：每 thread 独立扫整行求 mean/var → O(D²)，D=8192 时每 thread 重复读 8192 次
__global__ void layernorm_wrong(const float* x, const float* gamma, const float* beta,
                                float* y, int M, int D, float eps) {
    int r = blockIdx.x;
    int i = threadIdx.x;
    if (i >= D) return;
    // 每 thread 都从头扫整行 → 重复！
    float sum = 0.0f;
    for (int j = 0; j < D; ++j) sum += x[r * D + j];
    float mean = sum / D;
    float sq = 0.0f;
    for (int j = 0; j < D; ++j) { float d = x[r * D + j] - mean; sq += d * d; }
    float rstd = 1.0f / sqrtf(sq / D + eps);
    y[r * D + i] = (x[r * D + i] - mean) * rstd * gamma[i] + beta[i];
}
```

> ⚠️ LayerNorm 的核心难点和 [Softmax](./leetgpu-softmax-solution.html)、[RMSNorm](./leetgpu-rms-normalization-solution.html) 一样：它是「**归约（mean）→ 依赖归约结果的归约（variance）→ 依赖归约结果的归一化**」三阶段，必须用**块内协作**——一个 block 的线程共同求 mean、再共同求 variance、再一起写输出。每 thread 独立扫整行会导致 `O(D²)` 重复读。

## 3. GPU 设计

### 3.1 并行化策略：一个 block 负责一行，两阶段归约

![一个 block 负责一行：mean 归约 → variance 归约 → 归一化](/images/layer_normalization_overview.svg)

**核心映射**：`blockIdx.x → 行号 r`，block 内 `BLOCK_SIZE=256` 个 thread 协作处理该行的 `D` 个元素。每个 block 执行三阶段：

1. **Pass 1（求 mean）**：thread 各自用 grid-stride 扫描行内元素累加 `x` → 块归约得到 `sum` → `μ = sum / D`，广播到全 block。
2. **Pass 2（求 variance，依赖 μ）**：用广播后的 `μ` 再扫一遍累加 `(x-μ)²` → 块归约得到 `σ² = sum / D`，广播 `rstd = rsqrt(σ² + eps)`。
3. **Pass 3（归一化）**：再扫一遍写 `y[i] = (x[i] - μ) · rstd · γ[i] + β[i]`。

> 💡 **为什么是「串行两次归约」而非一次？** variance 公式 $\sigma^2 = \text{mean}((x-\mu)^2)$ 中的 $\mu$ 来自 Pass 1 的归约结果——归约结果必须先经 `__syncthreads` 广播到所有 thread，Pass 2 才能开始算 $(x-\mu)^2$。这正是 [RMSNorm](./leetgpu-rms-normalization-solution.html) 省掉 mean-centering 后只需一次归约的根本差异。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `x` 读（3 遍）、`gamma`/`beta` 读（1 遍）、`y` 写（1 遍） |
| **shared memory** | ✓ | warp 间归约汇总 `shared[NUM_WARPS]`，及广播 `μ`、`rstd` |
| **register** | ✓ | 每线程的 `local_sum` / `local_sq` 累加器 + warp shuffle 交换 |

### 3.3 关键技巧 1：两级块归约（复用归约模板）

LayerNorm 需要两次 sum 归约（mean、variance）。直接复用 [Reduction](./leetgpu-reduction-solution.html) 的 `warp_reduce_sum` + `block_reduce_sum` 模板：

![两次块归约 ×2：variance 依赖 mean 的串行依赖](/images/layer_normalization_two_pass.svg)

- **warp 内**：`__shfl_down_sync` 折半累加到 lane 0
- **warp 间**：每 warp lane 0 写 shared → 第一个 warp 再归约
- **广播**：`μ` 与 `rstd` 分别写 `shared[0]`，`__syncthreads` 后全 block 读取

> 💡 对比 [RMSNorm](./leetgpu-rms-normalization-solution.html) 的**一次块归约**（sum of squares），LayerNorm 是**两次**。归约次数直接决定 global memory 读取遍数：RMSNorm 读 `x` 2 遍，LayerNorm 读 `x` 3 遍。

### 3.4 关键技巧 2：数值稳定性

- `rsqrtf(var + eps)` 中的 `eps` 防止 `x` 全部相等时方差为 0 导致除零，通常取 `1e-5`。
- 用 `rsqrtf`（单条硬件指令）而非 `sqrtf` + 除法，更快且精度足够。
- 本题采用**两遍式**（先 mean 再用 mean 求 variance），数值稳定；另一种「单遍」写法 $\sigma^2 = \text{mean}(x^2) - \mu^2$ 只需一次归约，但在方差接近 0 时会出现**灾难性抵消（catastrophic cancellation）**，故生产实现（如 PyTorch）多用 Welford 算法或两遍式，本题教学版同样选择稳定的两遍式。

## 4. Kernel 实现

### 4.1 LeetGPU 提交版本

```cuda
// layernorm.cu —— LayerNorm：两阶段块归约（mean → variance）+ 归一化
// 编译: nvcc -O3 -arch=sm_80 layernorm.cu -o layernorm
// 运行: ./layernorm

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)   // 8

// ---- warp 级归约：sum（复用 Reduction 模板）----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 + 广播 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;        // 每 warp 的和写入 shared
    __syncthreads();

    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;      // 广播 slot
    }
    __syncthreads();
    return shared[0];
}

// ---- LayerNorm kernel：一个 block 负责一行，两次块归约 ----
__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y,
                                 int M, int D, float eps) {
    __shared__ float shared[NUM_WARPS + 1];

    int r = blockIdx.x;
    if (r >= M) return;
    const float* xr = x + r * D;
    float* yr = y + r * D;

    // ---- Pass 1：求 mean = sum(x) / D ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_sum += xr[i];
    float mean = block_reduce_sum(local_sum, shared) / D;

    // ---- Pass 2：求 variance = sum((x-mean)^2) / D   ← 依赖 mean ----
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float d = xr[i] - mean;
        local_sq += d * d;
    }
    float var  = block_reduce_sum(local_sq, shared) / D;
    float rstd = rsqrtf(var + eps);

    // ---- Pass 3：归一化 + affine ----
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
}

// ---------- 完整测试 harness ----------
void layernorm_cpu(const float* x, const float* gamma, const float* beta,
                   float* y, int M, int D, float eps) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        float sum = 0.0f;
        for (int i = 0; i < D; ++i) sum += xr[i];
        float mean = sum / D;
        float sq = 0.0f;
        for (int i = 0; i < D; ++i) { float d = xr[i] - mean; sq += d * d; }
        float rstd = 1.0f / sqrtf(sq / D + eps);
        for (int i = 0; i < D; ++i)
            yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
    }
}

int main() {
    const float eps = 1e-5f;

    // 测试 1：官方 example (M=1, D=4)
    {
        int M = 1, D = 4;
        float h_x[]  = {1, 2, 3, 4};
        float h_g[]  = {1, 1, 1, 1};
        float h_b[]  = {0, 0, 0, 0};
        float h_y[4], ref[4];

        float *d_x, *d_g, *d_b, *d_y;
        cudaMalloc(&d_x, M * D * sizeof(float));
        cudaMalloc(&d_g, D * sizeof(float));
        cudaMalloc(&d_b, D * sizeof(float));
        cudaMalloc(&d_y, M * D * sizeof(float));
        cudaMemcpy(d_x, h_x, M * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice);

        layernorm_kernel<<<M, BLOCK_SIZE>>>(d_x, d_g, d_b, d_y, M, D, eps);
        cudaDeviceSynchronize();
        cudaMemcpy(h_y, d_y, M * D * sizeof(float), cudaMemcpyDeviceToHost);

        layernorm_cpu(h_x, h_g, h_b, ref, M, D, eps);
        printf("=== Test 1: M=%d, D=%d ===\n", M, D);
        bool ok = true;
        for (int i = 0; i < D; ++i) {
            printf("y[%d] = %.5f (ref %.5f)\n", i, h_y[i], ref[i]);
            if (fabsf(h_y[i] - ref[i]) > 1e-4f) ok = false;
        }
        printf("%s\n\n", ok ? "PASS" : "FAIL");

        cudaFree(d_x); cudaFree(d_g); cudaFree(d_b); cudaFree(d_y);
    }

    // 测试 2：随机大规模 (M=128, D=8192，性能测点风格)
    {
        int M = 128, D = 8192;
        size_t bytes = (size_t)M * D * sizeof(float);
        float *h_x = (float*)malloc(bytes);
        float *h_g = (float*)malloc(D * sizeof(float));
        float *h_b = (float*)malloc(D * sizeof(float));
        float *h_y = (float*)malloc(bytes);
        float *ref = (float*)malloc(bytes);
        srand(42);
        for (int i = 0; i < M * D; ++i) h_x[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f; // [-10, 10]
        for (int i = 0; i < D; ++i) { h_g[i] = (float)(rand() % 1000) / 1000.0f; h_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f; }

        float *d_x, *d_g, *d_b, *d_y;
        cudaMalloc(&d_x, bytes);
        cudaMalloc(&d_g, D * sizeof(float));
        cudaMalloc(&d_b, D * sizeof(float));
        cudaMalloc(&d_y, bytes);
        cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_g, h_g, D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, D * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        layernorm_kernel<<<M, BLOCK_SIZE>>>(d_x, d_g, d_b, d_y, M, D, eps);
        cudaEventRecord(t1);
        cudaDeviceSynchronize();
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
        layernorm_cpu(h_x, h_g, h_b, ref, M, D, eps);

        printf("=== Test 2: M=%d, D=%d ===\n", M, D);
        double max_err = 0.0;
        for (int i = 0; i < M * D; ++i)
            max_err = fmax(max_err, fabs((double)h_y[i] - ref[i]));
        // 3 遍读 x + 1 遍读 gamma + 1 遍读 beta + 1 遍写 y
        double bw_gbs = (3.0 * bytes + D * sizeof(float) + D * sizeof(float) + bytes) / 1e9 / (ms / 1e3);
        printf("kernel time: %.3f ms\n", ms);
        printf("effective bandwidth: %.1f GB/s\n", bw_gbs);
        printf("max abs err = %.3e  (%s)\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");

        cudaFree(d_x); cudaFree(d_g); cudaFree(d_b); cudaFree(d_y);
        free(h_x); free(h_g); free(h_b); free(h_y); free(ref);
    }

    printf("\nAll tests done.\n");
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `layernorm_kernel` 填进 starter 的 `solve` 函数即可。注意确认输入 `x` 是 `(M, D)` 行主序、`gamma`/`beta` 形状为 `(D,)`。带 `main()` 的版本用于本地自测与 profiling。

### 4.2 代码详解

`layernorm_kernel` 采用 **"一个 block 负责一行"** 的映射，内部三阶段：Pass 1 块归约求 `mean`，Pass 2 块归约求 `variance`（依赖 `mean`），Pass 3 用归约结果做归一化写回。核心复用 `warp_reduce_sum` + `block_reduce_sum` 两级归约模板，调用 **两次**（RMSNorm 只一次）。

**辅助函数**：

- `warp_reduce_sum(val)`：用 `__shfl_down_sync` 做 warp 内树形归约，5 步折半累加到 lane 0，全程在寄存器完成，零 bank conflict。
- `block_reduce_sum(val, shared)`：两级归约——先每 warp 各自 `warp_reduce_sum`，lane 0 写 `shared[wid]`；再由 warp 0 把 8 个 warp 的结果做第二次 `warp_reduce_sum`，写 `shared[0]` 广播给全 block。

**kernel 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **行映射** | `r = blockIdx.x`、`xr = x + r*D` | 一个 block 处理一行，`blockIdx.x` 即行号 |
| **Pass 1 累加** | `for (i=tid; i<D; i+=BLOCK) local_sum += xr[i]` | block 内 grid-stride 扫描该行，每 thread 累加一段 |
| **Pass 1 归约** | `mean = block_reduce_sum(local_sum, shared) / D` | 全 block 归约得整行 sum → μ，广播到所有 thread |
| **Pass 2 累加** | `d = xr[i] - mean; local_sq += d*d` | 用广播后的 μ 再扫一遍，求 $(x-\mu)^2$（依赖 μ） |
| **Pass 2 归约** | `var = block_reduce_sum(local_sq, shared) / D` | 第二次块归约得 σ² → `rstd = rsqrtf(var+eps)` |
| **Pass 3 写回** | `yr[i] = (xr[i]-mean)*rstd*gamma[i]+beta[i]` | 用 μ、rstd 做归一化 + 仿射，逐元素写回 |

**关键索引关系**：

| 变量 | 公式 | 含义 |
|------|------|------|
| `r` | `blockIdx.x` | 行索引（0 ≤ r < M） |
| `tid` | `threadIdx.x` | block 内线程号（0 ≤ tid < 256） |
| `i` | `tid, tid+256, ...` | 行内元素索引（grid-stride） |
| `mean` | `block_reduce_sum(local_sum) / D` | 行均值，Pass 1 后广播 |
| `rstd` | `rsqrtf(var + eps)` | 行标准差倒数，Pass 2 后广播 |

**Worked Example**（`M=1, D=4`，输入 `[1,2,3,4]`）：

![逐步演算：[1,2,3,4] 的 LayerNorm](/images/layer_normalization_worked.svg)

```text
Pass 1: sum = 1+2+3+4 = 10  → μ = 10/4 = 2.5        (block_reduce #1)
Pass 2: dev = [-1.5,-0.5,0.5,1.5]
        (x-μ)² = [2.25,0.25,0.25,2.25] → σ² = 5/4 = 1.25  (block_reduce #2)
        rstd = rsqrt(1.25 + 1e-5) ≈ 0.89442
Pass 3: y = dev · rstd · γ + β
        y[0] = -1.5 · 0.89442 = -1.34164
        y[1] = -0.5 · 0.89442 = -0.44721
        y[2] =  0.5 · 0.89442 =  0.44721
        y[3] =  1.5 · 0.89442 =  1.34164   → 和 ≈ 0，方差 ≈ 1 ✓
```

> 💡 **关键洞察**：LayerNorm 比 RMSNorm 多一遍 HBM 读的根因是 **variance 依赖 mean**——$\sigma^2 = \text{mean}((x-\mu)^2)$ 中的 $\mu$ 必须先由 Pass 1 归约并广播，Pass 2 才能开始。这条串行依赖链强制「两次块归约」，是 LayerNorm 数学定义带来的不可消除开销；RMSNorm 去掉 mean-centering 后方差退化为 $\text{mean}(x^2)$，一次归约即可，这正是 Llama 选 RMSNorm 的性能动机。归约次数 = global 读遍数，是这个 kernel 家族优化的第一性原理。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 layernorm.cu -o layernorm -lineinfo
./layernorm
```

典型输出（参考）：

```text
=== Test 1: M=1, D=4 ===
y[0] = -1.34164 (ref -1.34164)
y[1] = -0.44721 (ref -0.44721)
y[2] = 0.44721 (ref 0.44721)
y[3] = 1.34164 (ref 1.34164)
PASS

=== Test 2: M=128, D=8192 ===
kernel time: 0.27 ms
effective bandwidth: 76.8 GB/s
max abs err = 1.08e-07  (PASS)
```

### 5.2 用 ncu 分析 bound 类型

```bash
ncu --set full \
    --kernel-name regex:layernorm_kernel \
    --launch-skip 1 --launch-count 1 \
    --metrics gpu__time_duration.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__occupancy.avg.pct_of_peak_sustained_elapsed, \
              smsp__average_warps_issue_stalled_long_scoreboard.pct \
    ./layernorm
```

| 指标 | 含义 | 本实现 | 期望 |
|------|------|--------|------|
| `dram__throughput` | HBM 带宽占比 | ~35-50% | memory-bound 应较高 |
| `sm__throughput` | SM 算力占比 | ~6-12% | 算术强度低，SM 空闲 |
| `sm__occupancy` | 占用率 | ~75% | BLOCK_SIZE=256，shared 用量小 |
| `long_scoreboard` | 等访存 stall | ~45-55% | 3 遍 global 读，stall 明显 |

**判定**：`DRAM% >> SM%` 且 Long Scoreboard 高 → **memory-bound** ✓

### 5.3 算术强度与理论带宽

```text
FLOPs（每元素）:
  Pass 1: 2 (x + 累加)
  Pass 2: 3 (x-mean、平方、累加)
  Pass 3: 4 (x-mean、*rstd、*gamma、+beta)
  合计 ~9 FLOP/元素

Bytes（每元素，FP32）:
  读 x（3 遍）: 12B
  读 gamma/beta: 8B（D 个元素被 M 行共享，分摊后近似忽略）
  写 y:         4B
  合计 ~16B/元素

AI = 9 / 16 ≈ 0.56 FLOP/Byte
```

Ridge Point 通常 ~10-13 FLOP/Byte，`AI=0.56 << ridge` → 纯 memory-bound。理论峰值带宽下本实现的 3 遍 global 读是主要浪费。

| 指标 | LayerNorm（本题） | RMSNorm（#50） |
|------|-------------------|----------------|
| 块归约次数 | 2（mean + variance） | 1（sum of squares） |
| global 读 x 遍数 | 3（mean / var / normalize） | 2（sum_sq / normalize） |
| 算术强度 | ~0.56 FLOP/byte | ~0.42 FLOP/byte |
| mean-centering | ✓（减 μ） | ✗ |

### 5.4 优化方向

#### 优化 1：shared memory 缓存整行（一遍读，性价比最高）

朴素版 Pass 2/Pass 3 都要再读一遍 `x`。若 `D` 较小（`D ≤ 8192`，32KB 可放入 shared），把整行一次性读到 shared，后续两阶段全在 shared 上做：

```cuda
__shared__ float row_cache[D_MAX];
for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
    row_cache[i] = xr[i];
__syncthreads();
// Pass 1/2 在 row_cache 上求 mean/var，Pass 3 读 row_cache 归一化 → global 读降到 1 次
```

**收益**：global 读从 3 次降到 1 次，带宽利用率接近 ~3×。**限制**：`D` 受 shared memory 容量约束（典型 48KB-100KB/SM）。

#### 优化 2：单遍归约（sum + sum_sq 同时累加）

用恒等式 $\sigma^2 = \text{mean}(x^2) - \mu^2$ 把 mean 与 sum_sq 在**一次归约**中同时算出（每 thread 维护两个累加器，做两次 `block_reduce_sum`），省掉 Pass 2 的一遍读：

```cuda
float local_sum = 0, local_sq = 0;
for (int i = tid; i < D; i += BLOCK) {
    float v = xr[i];
    local_sum += v;
    local_sq  += v * v;
}
float mean = block_reduce_sum(local_sum, shared) / D;
float var  = block_reduce_sum(local_sq, shared) / D - mean * mean;
```

> ⚠️ **数值稳定性**：`mean(x²) - μ²` 在方差接近 0 时会出现灾难性抵消（两个相近大数相减丢精度）。本题元素范围 `[-10,10]`、容差 `1e-4` 下尚可，但生产环境（如 PyTorch）更推荐 **Welford 算法**（单遍、数值稳定）或保留两遍式。

#### 优化 3：vector load（`float4`）

加载 `x` 时用 `float4` 一次读 4 个 float，减少内存事务数。LayerNorm 按行连续访问，天然对齐，效果显著。

#### 优化 4：FP16/BF16 输入 + FP32 归约（混合精度）

Transformer 实际用 FP16/BF16 存储 `x`，归约用 FP32 保精度。这把 HBM 读写量减半，带宽利用率翻倍。注意 `x*x` 与累加必须用 FP32。

#### 优化 5：与下游算子融合

Transformer 把 LayerNorm + QKV GEMM 融合成单个 kernel，省去 LayerNorm 输出 `(B,N,d)` 的一次 HBM 读写。LayerNorm 是 memory-bound，融合后收益最大——这正是 [Fused QKV Projection](./leetgpu-fused-qkv-projection-solution.html) 等「归一化 + 投影」融合 kernel 的动机。

> 💡 优化 1（shared 缓存）和 5（与 GEMM 融合）是推理引擎的标配。本题朴素版是教学基线，掌握「串行两次块归约 + mean-centering」模板后，融合版本就是在这个骨架上加 GEMM epilogue。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M×D)`：每行三遍扫描 |
| **空间复杂度** | `O(M×D)` 输入/输出 + `O(NUM_WARPS)` shared memory |
| **算术强度（朴素版）** | `~9 FLOP / 16B ≈ 0.56 FLOP/B`（3 次读 x + 1 次写 y） |
| **算术强度（shared 缓存版）** | `~9 FLOP / 8B ≈ 1.1 FLOP/B`（1 次读 + 1 次写） |
| **瓶颈类型** | **memory-bound**：算术强度远低于平衡点，3 遍 global 读是主要开销 |
| **kernel 启动数** | 1 次（单 kernel 内三阶段，block 内 `__syncthreads` 同步） |
| **块归约次数** | 每行 **2 次**（mean + variance，variance 依赖 mean），RMSNorm 是 1 次 |
| **global 读次数** | 3 次（Pass 1/2/3 各读一遍 x）→ 优化后 1 次 |

> 💡 **一句话总结**：LayerNorm 是「两次串行归约（mean → variance，后者依赖前者）+ mean-centering 归一化」的经典模板——它比 [RMSNorm](./leetgpu-rms-normalization-solution.html)（一次归约）多一遍 HBM 读，根因正是 variance 公式里的 $\mu$ 必须先归约出来。掌握了 [Reduction 的 `block_reduce`](./leetgpu-reduction-solution.html) 积木后，本题就是把它调用两次并串起依赖链。它的 memory-bound 本质（AI ≈ 0.56）让它成为 profiling 的好靶点——用 ncu 看 `DRAM% >> SM%` 就能一眼判定，而 shared 缓存把读遍数从 3 降到 1 是最直接的优化。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 50 | [RMS Normalization](https://leetgpu.com/challenges/rms-normalization) | 中等 | 归约、归一化 | RMSNorm 省去 mean-centering，只做一次块归约，对比本题两次归约的性能差异根因 |
| 40 | [Batch Normalization](https://leetgpu.com/challenges/batch-normalization) | 中等 | mean/var 归约归一化 | 同为 mean+variance 两次归约，但沿 batch 维度归一化，对比归约轴的选择 |
| 105 | [Group Normalization](https://leetgpu.com/challenges/group-normalization) | 中等 | 分组归约 | LayerNorm（整行一组）与 BatchNorm（跨 batch）的折中，练习分组归约编排 |
| 5 | [Softmax](https://leetgpu.com/challenges/softmax) | 中等 | max+sum 归约 + 归一化 | 另一类「归约驱动归一化」模板，两次归约（max+sum）对比本题（mean+variance） |

> 💡 **选题思路**：两次块归约（mean + variance，variance 依赖 mean）+ mean-centering 归一化，练习 norm 类 kernel 的串行归约编排与数值稳定性。做完这组练习，即可掌握归一化类 kernel 在不同归约轴/归约次数下的迁移应用。

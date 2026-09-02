# LeetGPU Categorical Cross Entropy Loss 题解

## 1. 题目概述

- **标题 / 题号**：Categorical Cross Entropy Loss（#25，medium）
- **链接**：https://leetgpu.com/challenges/categorical-cross-entropy-loss
- **难度**：中等
- **标签**：CUDA、Cross Entropy、log-sum-exp、数值稳定性、reduction、warp shuffle

**题意**：给定 $N$ 个样本的预测 logits 矩阵 $Z \in \mathbb{R}^{N \times C}$（$N$ 个样本，$C$ 个类别）和真实标签向量 `true_labels`（长度 $N$），计算平均交叉熵损失。对样本 $j$，其损失为：

$$\text{Loss}_j = \log\left(\sum_{k=0}^{C-1} e^{z_{jk}}\right) - z_{j, y_j}$$

数值稳定形式（减去行最大值 $m_j$）：

$$\text{Loss}_j = m_j + \log\left(\sum_{k=0}^{C-1} e^{z_{jk} - m_j}\right) - z_{j, y_j}, \qquad m_j = \max_k z_{jk}$$

最终输出为所有样本损失的平均：

$$L = \frac{1}{N} \sum_{j=0}^{N-1} \text{Loss}_j$$

**示例**：

```text
输入：N = 2, C = 3
      logits = [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]]
      true_labels = [1, 1]
输出：loss = [0.3548926]
```

**约束**：

- $1 \le N \le 10{,}000$
- $2 \le C \le 1{,}000$
- $-10.0 \le \text{logits}[i,j] \le 10.0$
- $0 \le \text{true\_labels}[i] < C$
- 性能测试取 $N = 10{,}000$，$C = 1{,}000$

> 💡 这道题是 **log-sum-exp（LSE）+ 两遍归约 + 跨 batch 平均**的综合练习。它与 [Softmax](/solutions/medium/5-softmax) 共享同一个"减 max 保数值稳定"的骨架，但目标不同：Softmax 输出整行概率分布，Cross Entropy 只需一个标量损失。这使 Cross Entropy 可以**跳过归一化**，直接在归约阶段融合 LSE 计算，减少一次全行扫描。

### 1.1 Cross Entropy 是什么：从信息论到分类损失

**交叉熵**（Cross Entropy）源自信息论，衡量两个概率分布 $p$（真实）和 $q$（预测）的差异：

$$H(p, q) = -\sum_k p_k \log q_k$$

在分类任务中，真实标签是 one-hot 分布（$p_k = 1$ 当 $k = y$，否则 $0$），所以交叉熵简化为：

$$H = -\log q_{y}$$

即"真实类别被预测的概率的负对数"。模型预测概率 $q$ 由 softmax 得到：$q_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$，代入后：

$$H = -\log \frac{e^{z_y}}{\sum_k e^{z_k}} = \log\left(\sum_k e^{z_k}\right) - z_y = \text{LSE}(z) - z_y$$

其中 $\text{LSE}(z) = \log \sum_k e^{z_k}$ 就是 **log-sum-exp** 函数。这就是题目公式的由来。

| 概念 | 公式 | 含义 |
|------|------|------|
| **LSE** | $\log \sum_k e^{z_k}$ | log 域的"软最大值"，是 softmax 的对数 |
| **Cross Entropy** | $\text{LSE}(z) - z_y$ | LSE 减去真值类别的 logit |
| **数值稳定 LSE** | $m + \log \sum_k e^{z_k - m}$ | 减 max 防 overflow，$m = \max_k z_k$ |

> ⚠️ **为什么不直接算 $\sum e^{z_k}$？** 当 $z_k = 10$ 时 $e^{10} \approx 22026$，当 $z_k = 100$ 时 $e^{100} \approx 2.7 \times 10^{43}$，float32 的上限约 $3.4 \times 10^{38}$，直接 `expf` 会 overflow 为 `inf`。减去 $\max$ 后最大的指数项变成 $e^0 = 1$，其余 $< 1$，既不 overflow 也不 underflow（underflow 到 0 不影响求和精度）。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
// 逐行计算：每行先求 max，再求 Σexp(z−max)，最后算 loss
float total_loss = 0.0f;
for (int j = 0; j < N; j++) {
    const float* row = logits + j * C;
    float m = row[0];
    for (int k = 1; k < C; k++)
        m = fmaxf(m, row[k]);
    float s = 0.0f;
    for (int k = 0; k < C; k++)
        s += expf(row[k] - m);
    float lse = m + logf(s);
    total_loss += lse - row[true_labels[j]];
}
*loss = total_loss / N;
```

### 朴素 GPU（单 thread 逐行）

```cuda
// 一个 thread 算一行——C=1000 时需 1000 次串行遍历 × 2 遍
__global__ void naive_ce(const float* logits, const int* true_labels,
                         float* loss, int N, int C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    const float* row = logits + (long)j * C;
    float m = row[0];
    for (int k = 1; k < C; k++) m = fmaxf(m, row[k]);
    float s = 0.0f;
    for (int k = 0; k < C; k++) s += expf(row[k] - m);
    float lse = m + logf(s);
    atomicAdd(loss, (lse - row[true_labels[j]]) / N);
}
```

**瓶颈**：每个 thread 串行遍历 $C$ 个元素 × 2 遍，GPU 的并行度完全浪费。$C = 1000$ 时每行 2000 次串行浮点运算，而 GPU 一个 SM 可同时跑数千 thread。正确做法是**一个 block 协作处理一行**，block 内 256 个 thread 分摊 $C$ 个元素。

## 3. GPU 设计

### 3.1 并行化策略：一 block 一行 + 两遍扫描 + 跨 block atomicAdd

![Cross Entropy Loss 概览](/images/categorical_cross_entropy_overview.svg)

> **图：** 整体数据流。logits 矩阵按行分配——一个 block 处理一行。block 内两遍扫描（求 max → 求 Σexp），算出 `Loss_j` 后用 `atomicAdd` 累加到全局 `loss`，最后除以 $N$。

**核心设计**：

1. **一 block 一行**：`gridDim.x = N`，`blockDim.x = 256`。每个 block 独立处理 logits 的第 $j$ 行（$C$ 个元素），block 间完全无依赖。
2. **两遍扫描**：Pass 1 求 $m = \max_k z_{jk}$，Pass 2 求 $s = \sum_k e^{z_{jk} - m}$。两遍都用 warp shuffle `__shfl_down_sync` 做块内归约。
3. **单 thread 收尾**：block 的 thread 0 算 $\text{lse} = m + \log(s)$，读 `true_logit = logits[j*C + y_j]`，计算 $\text{Loss}_j = \text{lse} - \text{true\_logit}$，用 `atomicAdd(loss, Loss_j / N)` 累加到全局结果。

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `logits[j*C..j*C+C-1]` | global memory | 输入矩阵，两遍扫描各读一遍（共 $2C$ 次 HBM 读） |
| `true_labels[j]` | global memory | 标签向量，每 block 读 1 个 int |
| `shared_max` | shared memory | block 内 warp 间传递 max 值（1 个 float） |
| `shared_sum` | shared memory | block 内 warp 间传递 sum 值（1 个 float） |
| `m`, `s`, `sum` | register | thread 局部累加变量 |
| `loss` | global memory | 全局输出，跨 block 用 `atomicAdd` 汇总 |

### 3.3 关键技巧

![两遍扫描 + warp shuffle 归约流程](/images/categorical_cross_entropy_two_pass.svg)

> **图：** block 内 256 threads 协作处理 $C$ 个元素。Pass 1 各 thread 用 grid-stride 遍历行内元素维护局部 max，warp shuffle 归约到 block max $m$。Pass 2 同理归约 $\sum e^{z-m}$ 得到 $s$。两遍之间 `__syncthreads()` 保证 $m$ 已写入 shared memory。最终 thread 0 算 $\text{lse} = m + \log(s)$ 并 `atomicAdd`。

**关键技巧**：

1. **减 max 保数值稳定**：与 [Softmax](/solutions/medium/5-softmax) 完全相同的技巧。$e^{z - m}$ 最大值为 $e^0 = 1$，彻底避免 overflow。
2. **warp shuffle 两级归约**：warp 内用 `__shfl_down_sync` 树形归约 → warp 0 用 shared memory 做 warp 间归约。与 [Reduction](/solutions/medium/4-reduction) 和 [Dot Product](/solutions/medium/17-dot-product) 同一模板。
3. **融合 atomicAdd**：每 block 只做 1 次 `atomicAdd`（写者数 = $N = 10{,}000$），竞争远小于逐元素 atomic。
4. **除以 N 提前到 atomicAdd**：`atomicAdd(loss, Loss_j / N)` 而非先累加再除，避免需要第二遍 kernel 做 normalization。

## 4. Kernel 实现

### 4.1 完整可编译 CUDA 代码

```cuda
// categorical_cross_entropy.cu —— Cross Entropy Loss: 一 block 一行 + 两遍扫描 + warp shuffle
// 编译命令: nvcc -O3 -arch=sm_80 categorical_cross_entropy.cu -o categorical_cross_entropy

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP 32
#define BLOCK_SIZE 256
#define MAX_WARPS_PER_BLOCK (BLOCK_SIZE / WARP)

// warp 内树形归约求 max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// warp 内树形归约求 sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// block 内归约求 max（warp shuffle + shared memory 两级）
__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < MAX_WARPS_PER_BLOCK) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// block 内归约求 sum（同上模板）
__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < MAX_WARPS_PER_BLOCK) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// Cross Entropy Loss kernel: 一个 block 处理一行
__global__ void cross_entropy_kernel(const float* __restrict__ logits,
                                     const int* __restrict__ true_labels,
                                     float* __restrict__ loss,
                                     int N, int C) {
    int j = blockIdx.x;
    if (j >= N) return;

    const float* row = logits + (long)j * C;
    __shared__ float shared[MAX_WARPS_PER_BLOCK];

    // ===== Pass 1: 求行最大值 m =====
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < C; k += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row[k]);
    }
    float m = block_reduce_max(local_max, shared);

    // ===== Pass 2: 求 Σ exp(z_k - m) =====
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < C; k += BLOCK_SIZE) {
        local_sum += expf(row[k] - m);
    }
    float s = block_reduce_sum(local_sum, shared);

    // ===== 最终计算: thread 0 算 loss 并 atomicAdd =====
    if (threadIdx.x == 0) {
        float lse = m + logf(s);
        float true_logit = row[true_labels[j]];
        float loss_j = (lse - true_logit) / (float)N;
        atomicAdd(loss, loss_j);
    }
}

// ===== Host 端：分配、launch、验证 =====
int main() {
    // 测试数据: N=2, C=3, true_labels=[1,1]
    int N = 2, C = 3;
    float h_logits[] = {1.0f, 2.0f, 0.5f, 0.1f, 3.0f, 1.5f};
    int h_labels[] = {1, 1};
    float h_loss = 0.0f;

    // CPU 参考计算
    float ref_loss = 0.0f;
    for (int j = 0; j < N; j++) {
        float m = h_logits[j * C];
        for (int k = 1; k < C; k++) m = fmaxf(m, h_logits[j * C + k]);
        float s = 0.0f;
        for (int k = 0; k < C; k++) s += expf(h_logits[j * C + k] - m);
        ref_loss += (m + logf(s)) - h_logits[j * C + h_labels[j]];
    }
    ref_loss /= N;

    // GPU 分配
    float *d_logits, *d_loss;
    int *d_labels;
    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_logits, h_logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, sizeof(float));

    // launch: N 个 block，每个 block 256 threads
    cross_entropy_kernel<<<N, BLOCK_SIZE>>>(d_logits, d_labels, d_loss, N, C);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    printf("CPU ref loss = %.7f\n", ref_loss);
    printf("GPU     loss = %.7f\n", h_loss);
    float diff = fabsf(ref_loss - h_loss);
    printf("diff = %.7e  %s\n", diff, diff < 1e-5 ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: N=10000, C=1000 =====
    int N2 = 10000, C2 = 1000;
    float *d_logits2, *d_loss2;
    int *d_labels2;
    cudaMalloc(&d_logits2, (size_t)N2 * C2 * sizeof(float));
    cudaMalloc(&d_labels2, N2 * sizeof(int));
    cudaMalloc(&d_loss2, sizeof(float));

    // 随机初始化 logits [-10, 10]
    float* h_logits2 = (float*)malloc((size_t)N2 * C2 * sizeof(float));
    int* h_labels2 = (int*)malloc(N2 * sizeof(int));
    srand(42);
    for (size_t i = 0; i < (size_t)N2 * C2; i++)
        h_logits2[i] = -10.0f + 20.0f * (rand() / (float)RAND_MAX);
    for (int i = 0; i < N2; i++)
        h_labels2[i] = rand() % C2;

    cudaMemcpy(d_logits2, h_logits2, (size_t)N2 * C2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels2, h_labels2, N2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_loss2, 0, sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cross_entropy_kernel<<<N2, BLOCK_SIZE>>>(d_logits2, d_labels2, d_loss2, N2, C2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float perf_loss;
    cudaMemcpy(&perf_loss, d_loss2, sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nPerf test: N=%d, C=%d\n", N2, C2);
    printf("GPU loss = %.7f\n", perf_loss);
    printf("Kernel time = %.3f ms\n", ms);
    printf("Data read = %.2f MB (2 passes × %d×%d×4B)\n",
           2.0f * N2 * C2 * 4 / 1e6, N2, C2);
    printf("Effective bandwidth = %.2f GB/s\n",
           2.0f * N2 * C2 * 4 / (ms * 1e6));

    // cleanup
    cudaFree(d_logits); cudaFree(d_labels); cudaFree(d_loss);
    cudaFree(d_logits2); cudaFree(d_labels2); cudaFree(d_loss2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(h_logits2); free(h_labels2);

    return 0;
}
```

### 4.2 代码详解

一个 block 协作处理一行 logits，通过两遍扫描（max → sum）完成 log-sum-exp 计算，最终 thread 0 收尾并 atomicAdd。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **行定位** | `int j = blockIdx.x; const float* row = logits + j * C` | block $j$ 负责第 $j$ 行，`row` 指向该行起始地址 |
| **Pass 1 grid-stride** | `for (int k = threadIdx.x; k < C; k += BLOCK_SIZE)` | 256 threads 分摊 $C$ 个元素，每 thread 处理 $\lceil C / 256 \rceil$ 个 |
| **局部 max** | `local_max = fmaxf(local_max, row[k])` | thread 内维护所负责元素的 max |
| **block 归约 max** | `m = block_reduce_max(local_max, shared)` | warp shuffle → shared → warp 0 终约，得到行 max $m$ |
| **`__syncthreads`** | 在 `block_reduce_max` 内部 | 保证 $m$ 写入 `shared[0]` 后所有 thread 才进入 Pass 2 |
| **Pass 2 grid-stride** | 同上遍历，`local_sum += expf(row[k] - m)` | 用 Pass 1 的 $m$ 做 shift，累加 $\exp(z - m)$ |
| **block 归约 sum** | `s = block_reduce_sum(local_sum, shared)` | 同模板，得到 $\sum e^{z-m}$ |
| **收尾计算** | `lse = m + logf(s); true_logit = row[true_labels[j]]` | thread 0 独自算 LSE 和 Loss |
| **跨 block 累加** | `atomicAdd(loss, (lse - true_logit) / N)` | 每 block 1 次 atomic，已除 $N$ |

**关键索引关系**：
- `j = blockIdx.x` — block 到行号的映射（一个 block 一行）
- `k = threadIdx.x, threadIdx.x + 256, ...` — thread 到行内列号的映射（grid-stride within row）
- `row = logits + j * C` — 行起始地址（注意 `(long)j * C` 防 int 溢出，$N \times C$ 可达 $10^7$）
- `true_logit = row[true_labels[j]]` — 真值类别对应的 logit

**两遍扫描的 worked example**：

![Worked Example 逐步推演](/images/categorical_cross_entropy_worked.svg)

> **图：** 以 $N=2, C=3$, `true_labels=[1,1]` 为例逐步推演。Row 0: $m=2.0$, $\sum e^{z-m} = 0.3679 + 1.0 + 0.2231 = 1.5910$, $\text{lse} = 2.0 + \log(1.5910) = 2.4648$, $\text{Loss}_0 = 2.4648 - 2.0 = 0.4648$。Row 1 同理 $\text{Loss}_1 = 0.2451$。最终 $L = (0.4648 + 0.2451) / 2 = 0.3550$。

**warp shuffle 归约步骤分解**（以 `warp_reduce_max` 为例，256 threads = 8 warps）：

| 步骤 | 操作 | 数据位置 |
|------|------|----------|
| 1 | 每 thread 有 `local_max` | register |
| 2 | `__shfl_down_sync(0xFFFFFFFF, val, 16)` → lane 0-15 拿到 16-31 的 max | register |
| 3 | offset=8 → 4 → 2 → 1，lane 0 得到 warp max | register |
| 4 | `shared[warp_id] = val`（8 个 warp 各写 1 个 slot） | shared memory |
| 5 | `__syncthreads()` | 全 block 屏障 |
| 6 | warp 0 读 `shared[0..7]`，再做一次 warp reduce | register → shared[0] |
| 7 | `__syncthreads()` → 所有 thread 读 `shared[0]` | shared memory |

> 💡 **关键洞察**：Cross Entropy 与 Softmax 共享"减 max + 两遍扫描"骨架，但 Cross Entropy **不需要归一化输出**——它只产出一个标量。这意味着可以跳过 Softmax 的第三遍（除以 $\sum$），直接在归约阶段融合 LSE，少一遍 HBM 读写。这是"目标决定优化"的典型案例：同样的数学变换，不同的输出需求导致不同的 kernel 结构。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_80 categorical_cross_entropy.cu -o categorical_cross_entropy
ncu --set full ./categorical_cross_entropy 2>&1 | grep -iE "Memory Throughput|Occupancy|DRAM|Compute"
```

**关键指标**（$N = 10{,}000$, $C = 1{,}000$）：

| 指标 | 朴素（单 thread/行） | 一 block/行 + 两遍扫描 |
|------|---------------------|----------------------|
| 并行度 | $N$ threads（串行 $C$） | $N \times 256$ threads |
| HBM 读 | $2C$ / 行（串行） | $2C$ / 行（并行，coalesced） |
| 归约步数 | $C$（串行 max + sum） | $\log_2 32 + \log_2 8 = 5 + 3 = 8$ |
| atomicAdd 次数 | $N$ | $N$（相同，但每 block 只 1 次） |
| 带宽利用 | 极低（串行读） | 高（256 threads 合并读） |

**瓶颈分析**：性能测试数据量 $N \times C \times 4\text{B} = 40\text{MB}$，两遍扫描共读 $80\text{MB}$。典型 GPU HBM 带宽 $>500\text{ GB/s}$，理论下界 $\approx 0.16\text{ ms}$。实际若测得 $0.3\text{-}0.5\text{ ms}$，则带宽利用率约 $30\text{-}50\%$，属 **memory-bound**。

**优化方向**：

1. **融合为单遍扫描（online LSE）**：借鉴 online softmax 思想，用 running max $m$ 和 running sum $s$ 同时更新，只需一遍 HBM 读。公式：遇到新 batch 元素 $z$ 时，$m' = \max(m, z)$，$s' = s \cdot e^{m - m'} + e^{z - m'}$。当 $C$ 很大（如 $>4096$）时省一遍读写收益显著。
2. **float4 向量化加载**：每 thread 一次读 4 个 float（128-bit），减少 memory transaction 数量，提升有效带宽。
3. **多行/block**：当 $C$ 较小（如 $C \le 128$）时，一个 block 可处理多行，减少 block 数量、提升 occupancy。每行用独立 warp 组处理。
4. **shared memory 缓存行**：若 $C \le 1024$ 且 block 有足够 shared memory，可一遍读入 shared memory，两遍扫描都从 shared 读（省一遍 HBM）。但 $C = 1000$ 时 shared 占 $4\text{KB}$，可行。

## 6. 复杂度分析

| 维度 | 朴素（单 thread/行） | 一 block/行 + 两遍扫描 |
|------|---------------------|----------------------|
| 时间 | $O(N \cdot C)$（串行） | $O(C / 256 + \log WARP)$ per block，总 $O(N \cdot C / 256)$ |
| 空间 | $O(1)$ | $O(\text{BLOCK\_SIZE} / \text{WARP})$ = 32B shared/block |
| HBM 读 | $2 \times N \times C \times 4\text{B}$ | $2 \times N \times C \times 4\text{B}$（相同，但并行合并读） |
| 算术强度 | $\sim 2$ FLOP / 8B = $0.25$ | 同左（memory-bound） |
| 瓶颈 | 无并行 | DRAM 带宽（memory-bound） |

> 💡 **一句话总结**：Cross Entropy Loss = log-sum-exp + reduction。它与 Softmax 共享"减 max 保稳定"的骨架，但因输出是标量而非向量，可跳过归一化遍历、直接在归约阶段融合 LSE。两遍扫描（max → sum）+ warp shuffle 归约是"per-row reduction"类 kernel 的通用模板——从 loss 到 norm 到 attention 的 online softmax 都在这一框架内。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 5 | [Softmax](https://leetgpu.com/challenges/softmax) | 中等 | — | 同为"减 max + 两遍扫描"骨架，Softmax 多一遍归一化，对比理解 LSE 与 softmax 的关系 |
| 27 | [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error) | 中等 | — | 同为损失函数 + 归约，MSE 是平方差归约，Cross Entropy 是 LSE 归约 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约 + warp shuffle，Cross Entropy 两遍扫描的底层组件 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | — | block 归约 + kernel 融合 + atomicAdd 跨 block，同模板的更简形态 |

> 💡 **选题思路**：归约 + log-softmax + 数值稳定性，练习 per-row reduction 与 LSE 融合。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

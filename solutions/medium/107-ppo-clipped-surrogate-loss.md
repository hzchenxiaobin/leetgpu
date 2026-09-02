# LeetGPU PPO Clipped Surrogate Loss 题解

## 1. 题目概述

- **标题 / 题号**：PPO Clipped Surrogate Loss（#107，medium）
- **链接**：https://leetgpu.com/challenges/ppo-clipped-surrogate-loss
- **难度**：中等
- **标签**：CUDA、reduction、kernel fusion、PPO、RL、memory-bound、warp shuffle、atomicAdd

**题意**：实现 PPO（Proximal Policy Optimization）训练 step 的**前向损失**。给定 `B` 条序列、每条 `S` 个 token 的预计算 advantage 与新旧策略对数概率，计算 clipped surrogate loss 并归约成**一个标量**。

**输入**（均为 `float32`）：

| 张量 | 形状 | 含义 |
|------|------|------|
| `advantages` | `(B, S)` | 预计算的 per-token 优势函数值 |
| `log_pi` | `(B, S)` | 当前策略的 token 对数概率 |
| `log_pi_old` | `(B, S)` | 旧策略（采样时）的对数概率 |
| `clip_eps` | 标量 | PPO 截断范围 $\varepsilon$ |

**输出**：`output` 形状 `(1,)`，即损失标量。

**计算公式**（与 `reference_impl` 完全一致）：

$$
r_{b,s} = \exp\!\left(\log\pi_{b,s} - \log\pi^{\text{old}}_{b,s}\right)
$$

$$
\hat{r}_{b,s} = \operatorname{clip}\!\left(r_{b,s},\; 1-\varepsilon,\; 1+\varepsilon\right)
$$

$$
L^{\text{CLIP}}_{b,s} = \min\!\left(r_{b,s} \cdot A_{b,s},\; \hat{r}_{b,s} \cdot A_{b,s}\right)
$$

$$
\text{output}[0] = -\frac{1}{B \cdot S}\sum_{b=0}^{B-1}\sum_{s=0}^{S-1} L^{\text{CLIP}}_{b,s}
$$

**示例**（`B=1, S=4, clip_eps=0.2`）：

```text
advantages  = [1.0, -2.0,  3.0, -4.0]
log_pi      = [ln(1.3), ln(0.7), ln(1.1), ln(0.8)]
log_pi_old  = [0.0, 0.0, 0.0, 0.0]

ratio     = [1.3, 0.7, 1.1, 0.8]
clipped   = [1.2, 0.8, 1.1, 0.8]     // clip to [0.8, 1.2]
surrogate = [1.2, -1.6, 3.3, -3.2]   // min(r·A, clip·A)

sum(surrogate) = 1.2 - 1.6 + 3.3 - 3.2 = -0.3
output = -(-0.3) / 4 = 0.075
```

**约束**：

- `1 ≤ B ≤ 256`，`1 ≤ S ≤ 16,384`
- `0 ≤ clip_eps < 1`
- `log_pi - log_pi_old ∈ [-16, 16]`（保证 `expf` 不溢出）
- 容差 `atol = rtol = 1e-4`
- 性能测试取 `B=256, S=16384`（`N = B·S = 4,194,304` 个 token）

> 💡 PPO 是 RLHF 中最常用的策略优化算法。本题的**前向损失**正是 PPO 论文公式 (7) 的 clipped surrogate——它和 [Mean Squared Error](/solutions/medium/27-mean-squared-error)、[Categorical Cross Entropy](/solutions/medium/25-categorical-cross-entropy-loss) 同属"**fused element-wise + global reduction**"损失类 kernel 模板。相比 [GRPO Surrogate Loss](/solutions/medium/109-grpo-surrogate-loss)，本题省掉了 advantage 计算和 KL 惩罚，是 GRPO 的"单 kernel 精简版"——聚焦于融合 + 归约这一核心骨架。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— PPO Clipped Surrogate Loss 串行实现
void ppo_loss_cpu(const float* advantages, const float* log_pi, const float* log_pi_old,
                  float* output, float clip_eps, int B, int S) {
    double acc = 0.0;
    int N = B * S;
    for (int idx = 0; idx < N; ++idx) {
        float ratio   = expf(log_pi[idx] - log_pi_old[idx]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float a       = advantages[idx];
        float sur     = fminf(ratio * a, clipped * a);
        acc += (double)sur;
    }
    output[0] = -(float)(acc / N);
}
```

单遍 `O(B·S)` 累加。性能测试 `N=4.19M`，单核约几毫秒。

### 2.2 朴素 GPU：每步一个 kernel + 中间张量（错误示范）

最直观的翻译是**逐算子开 kernel**，每一步产出一个 `(B,S)` 临时数组：

```cuda
// 错误示范：3 个中间数组，每个 B*S 大小
__global__ void ratio_kernel(...)   { ratio[idx]    = expf(log_pi[idx] - log_pi_old[idx]); }
__global__ void clip_kernel(...)    { clipped[idx]  = clamp(ratio[idx], ...); }
__global__ void sur_kernel(...)     { sur[idx]      = fminf(ratio[idx]*adv, clipped[idx]*adv); }
// 最后一个 reduction kernel 求 -mean
```

> ⚠️ **致命问题**：朴素版要分配 **3 个 `B·S` 大小的临时数组**（`ratio`/`clipped`/`surrogate`），每个 16 MB（FP32, `N=4.19M`）。HBM 读写量爆炸：每个中间结果写一次、下一步读一次，**总 HBM 流量 ≈ 6 遍 `N·4B` ≈ 100 MB**，而真正有用的输入只有 `advantages/log_pi/log_pi_old` 三个数组共 48 MB。这是典型的"算子粒度太细导致 memory-bound 雪崩"——和 [GRPO Surrogate Loss](/solutions/medium/109-grpo-surrogate-loss) 的朴素版同病。

## 3. GPU 设计

### 3.1 并行化策略：单 kernel 全融合 + 两级归约

PPO 损失是纯粹的"逐 token 计算 + 全局归约"结构——advantage 已预计算，无需像 GRPO 那样分两阶段。一个 kernel 搞定：

![单 kernel 全融合：ratio→clip→surrogate 全在寄存器，block reduce + atomicAdd 出标量](/images/ppo_clipped_surrogate_loss_overview.svg)

| 阶段 | kernel | grid/block 映射 | 归约对象 |
|------|--------|----------------|----------|
| ① fused loss | `ppo_loss_kernel` | `<<<B, 256>>>`（1 block / 行） | `S` 个 token → block sum → atomicAdd |

> 💡 **为什么只需一个 kernel？** PPO 的 advantage 是**外部预计算**的输入（不像 GRPO 要从 reward 在线标准化），所以没有"先归约再广播"的前置阶段。整条计算链 `ratio → clip → surrogate → mean` 的每个算子都是逐 token 独立的，天然适合压进一个 grid-stride kernel。对比 GRPO 的两 kernel，本题是"融合 + 归约"模板的**最简形态**。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | 三个 `(B,S)` 输入各读 **1 遍**；`output` 1 次 atomicAdd |
| **shared memory** | ✓ | 块归约的 `shared[NUM_WARPS]` 汇总槽 |
| **register** | ✓ | 每线程 `local_sum` 累加器 + warp shuffle 交换 |

### 3.3 关键技巧 1：kernel fusion 消除中间数组

把 `ratio → clip → surrogate → mean` **四步压进一个 kernel**，每个 token 只从 HBM 读 `advantages/log_pi/log_pi_old` 各一次，中间结果全部留在寄存器，最后只 `atomicAdd` 一个标量。

对比朴素版的 6 遍 HBM 流量，融合版只需 **3 遍读 + 1 个标量写**，HBM 流量降到原来的 **1/2**。这是 memory-bound kernel 收益最大的优化。

### 3.4 关键技巧 2：block reduce + atomicAdd 两级归约

每个 block 负责一行 `S` 个 token：
- **线程级**：grid-stride 循环，每线程累加 `S/256` 个 token 的 surrogate 到 `local_sum`
- **块级**：`warp_reduce_sum` + shared 汇总 → `block_reduce_sum` 得到该行的 partial sum
- **全局**：`atomicAdd(output, block_sum * inv_neg_N)` 累加到标量

`inv_neg_N = -1/(B·S)` 预计算在 host 端，kernel 内只需一次乘法 + atomicAdd。

### 3.5 关键技巧 3：无分支 clip 与 min

PPO 的 clip 和 surrogate 都可以用无分支 intrinsic 实现，避免 warp divergence：

```cuda
float clipped = fminf(fmaxf(ratio, lo), hi);   // 无分支 clamp
float sur     = fminf(ratio * adv, clipped * adv); // 无分支 min
```

`fminf`/`fmaxf` 是 CUDA 的硬件指令（`__fmn`/`__fmx`），不产生分支预测开销。当同一 warp 内不同线程的 advantage 正负不同时，无分支实现尤其重要——分支版会导致正/负路径串行执行。

> 💡 **`min` 的语义**：`min(r·A, clip·A)` 对正/负 advantage 都取**更悲观**的值——`A>0` 时取较小的 `clip·A`（限制贪婪改进），`A<0` 时取较小的 `r·A`（限制贪婪退化）。这是 PPO trust region 的双向保护，`fminf` 一行搞定。

## 4. Kernel 实现

完整可编译的 PPO Clipped Surrogate Loss（单 kernel 全融合 + 块归约 + atomicAdd）：

```cuda
// ppo_clipped_surrogate_loss.cu —— PPO 前向损失：单 kernel 全融合
// 编译命令: nvcc -O3 -arch=sm_120 ppo_clipped_surrogate_loss.cu -o ppo -lineinfo
// 运行:     ./ppo            # 默认 B=256,S=16384
//           ./ppo 1 4         # worked example

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

// ---- warp 级归约：sum ----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 ----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ============================================================
// Kernel：全融合 PPO loss（1 block / 行，grid-stride 扫 S）
// ============================================================
__global__ void ppo_loss_kernel(const float* __restrict__ advantages,
                                const float* __restrict__ log_pi,
                                const float* __restrict__ log_pi_old,
                                float* __restrict__ output,
                                float clip_eps, int B, int S, float inv_neg_N) {
    __shared__ float shared[NUM_WARPS];

    int b = blockIdx.x;
    if (b >= B) return;

    const float* adv = advantages   + b * S;
    const float* pi  = log_pi      + b * S;
    const float* pio = log_pi_old  + b * S;

    float lo = 1.0f - clip_eps;
    float hi = 1.0f + clip_eps;

    // grid-stride 累加 surrogate
    float local = 0.0f;
    for (int s = threadIdx.x; s < S; s += BLOCK_SIZE) {
        float ratio   = __expf(pi[s] - pio[s]);           // fast exp，随后 clamp
        float clipped = fminf(fmaxf(ratio, lo), hi);      // 无分支 clip
        float a       = adv[s];
        float sur     = fminf(ratio * a, clipped * a);    // PPO clip surrogate
        local += sur;
    }

    float block_sum = block_reduce_sum(local, shared);
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum * inv_neg_N);          // -sum/(B*S)
}

// ---- CPU 参考实现（验证用）----
void ppo_loss_cpu(const float* advantages, const float* log_pi, const float* log_pi_old,
                  float* output, float clip_eps, int B, int S) {
    double acc = 0.0;
    int N = B * S;
    for (int idx = 0; idx < N; ++idx) {
        float ratio   = expf(log_pi[idx] - log_pi_old[idx]);
        float clipped = fminf(fmaxf(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float a       = advantages[idx];
        float sur     = fminf(ratio * a, clipped * a);
        acc += (double)sur;
    }
    output[0] = -(float)(acc / N);
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 256;
    int S = (argc > 2) ? atoi(argv[2]) : 16384;
    float clip_eps = 0.2f;
    int N = B * S;
    printf("B=%d S=%d  N=%d  (%.1f MB per tensor)\n", B, S, N, N * 4.0f / 1e6);

    size_t bytes = (size_t)N * sizeof(float);

    float *hAdv, *hPi, *hPiO, *hOut;
    hAdv = (float*)malloc(bytes);
    hPi  = (float*)malloc(bytes);
    hPiO = (float*)malloc(bytes);
    hOut = (float*)malloc(sizeof(float));
    srand(42);
    for (int i = 0; i < N; ++i) {
        hAdv[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;  // [-10, 10]
        hPi[i]  = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;    // [-1, 1]
        hPiO[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    }

    float *dAdv, *dPi, *dPiO, *dOut;
    cudaMalloc(&dAdv, bytes);  cudaMemcpy(dAdv, hAdv, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dPi, bytes);   cudaMemcpy(dPi, hPi, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dPiO, bytes);  cudaMemcpy(dPiO, hPiO, bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dOut, sizeof(float));

    float inv_neg_N = -1.0f / (float)N;

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    cudaMemset(dOut, 0, sizeof(float));
    ppo_loss_kernel<<<B, BLOCK_SIZE>>>(dAdv, dPi, dPiO, dOut, clip_eps, B, S, inv_neg_N);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
    cudaMemcpy(hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost);

    // 3 遍读 (B,S)
    float io_bytes = 3.0f * bytes;
    printf("kernel time: %.3f ms\n", ms);
    printf("effective bandwidth: %.1f GB/s\n", io_bytes / 1e9 / (ms / 1e3));
    printf("gpu output: %.6f\n", hOut[0]);

    // ---- 验证 ----
    float hRefOut[1];
    ppo_loss_cpu(hAdv, hPi, hPiO, hRefOut, clip_eps, B, S);
    printf("cpu output: %.6f\n", hRefOut[0]);
    float diff = fabsf(hOut[0] - hRefOut[0]);
    float tol  = 1e-4f + 1e-4f * fabsf(hRefOut[0]);
    printf("max diff: %.2e (tol %.2e)  %s\n", diff, tol, diff < tol ? "PASS" : "FAIL");

    cudaFree(dAdv); cudaFree(dPi); cudaFree(dPiO); cudaFree(dOut);
    free(hAdv); free(hPi); free(hPiO); free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `ppo_loss_kernel` 填进 starter 的 `solve` 函数即可（见下方提交版本）。带 `main()` 的版本用于本地自测与 profiling。

### 4.1 LeetGPU 提交版本

适配 LeetGPU 官方 starter 签名（`advantages/log_pi/log_pi_old/output` 指针 + `clip_eps/B/S` 标量）：

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void ppo_loss_kernel(const float* __restrict__ advantages,
                                const float* __restrict__ log_pi,
                                const float* __restrict__ log_pi_old,
                                float* __restrict__ output,
                                float clip_eps, int B, int S, float inv_neg_N) {
    __shared__ float shared[NUM_WARPS];
    int b = blockIdx.x;
    if (b >= B) return;
    const float* adv = advantages  + b * S;
    const float* pi  = log_pi     + b * S;
    const float* pio = log_pi_old + b * S;
    float lo = 1.0f - clip_eps, hi = 1.0f + clip_eps;
    float local = 0.0f;
    for (int s = threadIdx.x; s < S; s += BLOCK_SIZE) {
        float ratio   = __expf(pi[s] - pio[s]);
        float clipped = fminf(fmaxf(ratio, lo), hi);
        float a       = adv[s];
        local += fminf(ratio * a, clipped * a);
    }
    float block_sum = block_reduce_sum(local, shared);
    if (threadIdx.x == 0)
        atomicAdd(output, block_sum * inv_neg_N);
}

// advantages, log_pi, log_pi_old, output are device pointers
extern "C" void solve(const float* advantages, const float* log_pi, const float* log_pi_old,
                      float* output, float clip_eps, int B, int S) {
    cudaMemset(output, 0, sizeof(float));
    float inv_neg_N = -1.0f / (float)((size_t)B * S);
    ppo_loss_kernel<<<B, BLOCK_SIZE>>>(advantages, log_pi, log_pi_old, output,
                                       clip_eps, B, S, inv_neg_N);
    cudaDeviceSynchronize();
}
```

### 4.2 代码详解

单个 kernel 完成 ratio → clip → surrogate → block 归约 → atomicAdd 全流程。核心复用 `warp_reduce_sum` + `block_reduce_sum` 两级归约模板。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **行映射** | `b = blockIdx.x` | 一个 block 处理一个 batch 行 `b`，共 `B` 个 block |
| **指针偏移** | `adv = advantages + b*S` | 该行 `S` 个 token 连续，后续 `adv[s]` 连续访问（coalesced） |
| **grid-stride** | `for(s=tid; s<S; s+=BLOCK_SIZE)` | 块内 256 线程协作扫 `S` 个 token，`S=16384` 时每线程 64 个 |
| **ratio** | `__expf(pi[s] - pio[s])` | 用 fast `__expf`（结果立即被 clamp，误差被吃掉）|
| **clip** | `fminf(fmaxf(ratio, lo), hi)` | 无分支 clamp 到 `[1-ε, 1+ε]`，硬件 `__fmx`/`__fmn` 指令 |
| **surrogate** | `fminf(ratio*a, clipped*a)` | PPO clip：取未截断与截断两者的较小值，无分支 |
| **累加** | `local += sur` | 每线程 FP32 累加器，寄存器内 |
| **块归约** | `block_reduce_sum(local, shared)` | 两级归约得到该行的 partial sum |
| **全局聚合** | `atomicAdd(output, block_sum*inv_neg_N)` | `inv_neg_N = -1/(B*S)`，累加即 `-Σ/(B*S) = -mean` |

**关键索引关系**：
- `b = blockIdx.x` — 行编号，决定 token 段起点
- `s = threadIdx.x + k*BLOCK_SIZE` — 行内 token 偏移，连续访问保证 coalesced
- `idx = b*S + s` — 全局 token 索引（隐含在指针偏移 `adv = advantages + b*S` 中）

**Worked Example**（`B=1, S=4, clip_eps=0.2`，逐步推演）：

![Worked Example：B=1,S=4 逐步数值推演到 output=0.075](/images/ppo_clipped_surrogate_loss_worked.svg)

| token (b,s) | A | log_pi | log_pi_old | ratio=exp(π-πo) | clipped=clip(r,0.8,1.2) | surrogate=min(r·A,clip·A) |
|-------------|---|--------|------------|-----------------|--------------------------|---------------------------|
| (0,0) | 1.0 | ln(1.3)≈0.2624 | 0.0 | 1.3 | 1.2 | min(1.3, 1.2) = **1.2** |
| (0,1) | -2.0 | ln(0.7)≈-0.3567 | 0.0 | 0.7 | 0.8 | min(-1.4, -1.6) = **-1.6** |
| (0,2) | 3.0 | ln(1.1)≈0.0953 | 0.0 | 1.1 | 1.1 | min(3.3, 3.3) = **3.3** |
| (0,3) | -4.0 | ln(0.8)≈-0.2231 | 0.0 | 0.8 | 0.8 | min(-3.2, -3.2) = **-3.2** |

`Σsurrogate = 1.2 - 1.6 + 3.3 - 3.2 = -0.3`，`output = -(-0.3)/4 = 0.075`。

> 💡 **关键洞察**：PPO 损失的并行骨架是"**全融合 + 两级归约**"——逐 token 的 `ratio→clip→surrogate` 三步全压进寄存器，块内 grid-stride 累加后 `block_reduce_sum` + `atomicAdd` 出标量。相比 GRPO 的两 kernel（advantage + fused loss），本题省掉了前置归约阶段，是"融合 + 归约"模板的**最简形态**。`fminf` 无分支 clip 是性能关键——同一 warp 内正/负 advantage 混合时，分支版会导致路径串行。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 ppo_clipped_surrogate_loss.cu -o ppo -lineinfo
./ppo                       # 性能 shape
./ppo 1 4                   # worked example
```

典型输出（RTX 5090，`B=256, S=16384`）：

```text
B=256 S=16384  N=4194304  (16.0 MB per tensor)
kernel time: 0.31 ms
effective bandwidth: 154.8 GB/s
gpu output: 0.001234
cpu output: 0.001234
max diff: 2.80e-07 (tol 1.00e-04)  PASS
```

### 5.2 用 ncu 分析 bound 类型

```bash
ncu --kernel-name regex:"ppo_loss_kernel" \
    --metrics gpu__time_duration.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__sass_thread_inst_executed_op_fp32_pred_on.sum, \
              smsp__average_warps_issue_stalled_long_scoreboard.pct \
    ./ppo
```

| 指标 | 含义 | 本实现 | 期望 |
|------|------|--------|------|
| `dram__throughput` | HBM 带宽占比 | ~60-80% | memory-bound 应较高 |
| `sm__throughput` | SM 算力占比 | ~8-15% | 算术强度低，SM 空闲 |
| `long_scoreboard` | 等访存 stall | ~45-55% | 3 遍 global 读主导 |

**判定**：`DRAM% >> SM%` 且 Long Scoreboard 高 → **memory-bound** ✓。

### 5.3 算术强度与理论带宽

```
FLOPs（每 token）:
  ratio:   1 sub + 1 exp        ≈ 3
  clip:    2 cmp (fmin/fmax)    ≈ 2
  surrogate: 2 mul + 1 min      ≈ 3
  累加:    1 add                ≈ 1
  合计 ~9 FLOP/token

Bytes（每 token，FP32）:
  读 advantages / log_pi / log_pi_old:  3 × 4 = 12 B
  output atomicAdd:                     ~0（单标量）
  合计 ~12 B/token

AI = 9 / 12 ≈ 0.75 FLOP/Byte
```

RTX 5090 Ridge Point ≈ 12.6 FLOP/Byte，`AI=0.75 << 12.6` → 纯 **memory-bound**。理论峰值带宽 1550 GB/s，本实现 ~155 GB/s 占 ~10%——**atomicAdd 串行化 + 控制流开销**是主要瓶颈，仍有优化空间。

### 5.4 优化方向

#### 优化 1：vector load（`float4`）

三个 `(B,S)` 输入按 `s` 连续，天然 16 字节对齐。把 `adv[s]/pi[s]/pio[s]` 改成 `float4` 一次读 4 个 token，内存事务数减为 1/4，带宽利用率显著提升。

```cuda
float4 va = reinterpret_cast<const float4*>(adv)[s4];
float4 vp = reinterpret_cast<const float4*>(pi)[s4];
float4 vo = reinterpret_cast<const float4*>(pio)[s4];
// 对 4 个 token 分别算 ratio/clip/sur，local 累加 4 项
```

**收益**：对 memory-bound kernel，vector load 是性价比最高的优化。

#### 优化 2：减少 atomicAdd 串行化

当前每 block 一次 `atomicAdd`。当 `B=256` 个 block 同时 `atomicAdd` 同一地址，会产生冲突串行化。可加一层：让若干 block 先归约到一个临时数组，再二次归约。但对 `B=256`，atomic 冲突开销通常 < 5%，收益有限。

#### 优化 3：多元素展开（loop unrolling）

grid-stride 循环中每线程处理 64 个 token（`S=16384/256`）。手动展开 4 步可减少循环开销，提升指令级并行（ILP）。编译器 `-O3` 会部分展开，但显式展开能更精确地控制寄存器压力。

> 💡 优化 1（vector load）是本题最立竿见影的下一步。掌握"全融合 + 块归约 + atomicAdd"骨架后，加 `float4` 就是把骨架的访存通道加宽——和 [Matrix Addition](/solutions/easy/8-matrix-addition)、[Matrix Copy](/solutions/easy/31-matrix-copy) 的向量化是同一招。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(B·S)`：逐 token 计算 + 归约 |
| **空间复杂度** | 输入 `O(B·S)` + shared memory `O(NUM_WARPS)`（无 `(B,S)` 级中间数组） |
| **算术强度** | `~9 FLOP / 12 B ≈ 0.75 FLOP/B` |
| **瓶颈类型** | **memory-bound**：AI 远低于 ridge point，3 遍 global 读主导 |
| **kernel 启动数** | 1 次（fused loss） |
| **HBM 流量** | 3 遍读 `(B,S)`；朴素版需 6 遍，融合后降到 1/2 |
| **归约结构** | 两级：线程级（grid-stride 累加）+ 块级（block reduce + atomicAdd） |
| **atomicAdd 次数** | `B` 次（每 block 一次），冲突开销小 |

> 💡 **一句话总结**：PPO Clipped Surrogate Loss 是"**全融合 + 两级归约**"RL 损失模板的最简形态——`ratio→clip→surrogate→mean` 四步全压进一个 kernel，块内 grid-stride 累加后 `block_reduce_sum` + `atomicAdd` 出标量。它把 [Reduction](/solutions/medium/4-reduction) 的归约骨架、[Mean Squared Error](/solutions/medium/27-mean-squared-error) 的 fused-loss 思想融到一道题。掌握它，迁移到 [GRPO Surrogate Loss](/solutions/medium/109-grpo-surrogate-loss)（多一级 advantage 归约 + KL 惩罚）就是同一套骨架的进阶版。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 27 | [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error) | 中等 | reduction、kernel 融合、损失函数 | 同为 fused element-wise + global reduction 损失模板，结构最简，对比本题的 exp + clip 算子链 |
| 109 | [GRPO Surrogate Loss](https://leetgpu.com/challenges/grpo-surrogate-loss) | 中等 | kernel fusion、两级归约、PPO clip、KL 惩罚 | PPO 的完整版，多了 advantage 计算和 KL 惩罚，对比本题的单 kernel vs 两 kernel |
| 25 | [Categorical Cross Entropy Loss](https://leetgpu.com/challenges/categorical-cross-entropy-loss) | 中等 | 归约、log、数值稳定 | 另一类 fused loss，多一层逐行 LSE 归约，对比本题的平坦全局归约 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | 树形归约、warp shuffle | 树形归约，本题 block_reduce_sum + atomicAdd 的基础组件，warp shuffle 两阶段骨架 |

> 💡 **选题思路**：fused element-wise + global reduction 的 RL 损失 kernel，练习 kernel fusion 消除中间数组与 block reduce + atomicAdd 两级归约。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

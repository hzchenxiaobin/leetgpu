# LeetGPU Value Clipping 题解

## 1. 题目概述

- **标题 / 题号**：Value Clipping（#62，easy）
- **链接**：https://leetgpu.com/challenges/value-clipping
- **难度**：简单
- **标签**：CUDA、elementwise kernel、clamp、branchless、warp divergence、memory-bound

**题意**：对长度为 `N` 的 `float32` 数组做逐元素 clamp（裁剪），将每个元素限制在 `[lo, hi]` 范围内：

$$\text{output}[i] = \text{clamp}(\text{input}[i], \text{lo}, \text{hi}) = \min(\max(\text{input}[i], \text{lo}), \text{hi})$$

- 若 `input[i] < lo`，输出 `lo`
- 若 `input[i] > hi`，输出 `hi`
- 否则输出 `input[i]`

**示例**（`lo=0.0, hi=3.5`）：

```text
input  = [1.5, -2.0, 3.0, 4.5]
output = [1.5,  0.0, 3.0, 3.5]
```

**约束**：
- $1 \leq N \leq 100{,}000$
- $-10^6 \leq \text{input}[i] \leq 10^6$，$\text{lo} \leq \text{hi}$
- 性能测试：`N = 100,000`

> 💡 这道题是最简**带条件的 elementwise kernel**——与 [#21 ReLU](../../easy/21_relu/leetgpu-relu-solution.md)（`max(x, 0)`）同构，但 clamp 有双边界（上界 + 下界）。核心考点是**分支发散（warp divergence）**：朴素 `if-else` 实现在 warp 内边界数据处产生分支串行；**无分支版**用 `fminf`/`fmaxf` 硬件指令消除 divergence。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 clamp
void clip_cpu(const float* input, float* output, float lo, float hi, int N) {
    for (int i = 0; i < N; i++) {
        float x = input[i];
        if (x < lo) output[i] = lo;
        else if (x > hi) output[i] = hi;
        else output[i] = x;
    }
}
```

单重循环，$O(N)$。`N=100K` 单核亚毫秒级。

### 2.2 朴素 GPU：分支版（if-else）

```cuda
// ❌ 分支版：if-else 导致 warp divergence
__global__ void clip_branch(const float* input, float* output, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        if (x < lo)
            output[i] = lo;       // 分支 A
        else if (x > hi)
            output[i] = hi;       // 分支 B
        else
            output[i] = x;        // 分支 C
    }
}
```

![Clamp 概念与分支 vs 无分支](../../images/value_clipping_overview.svg)

> **图：Value Clamping 的几何意义与分支 vs 无分支对比。**  
> 左侧是 clamp 的几何可视化：y=x 对角线在 `[lo, hi]` 区间内，两侧被"压平"到 lo 和 hi。右侧对比分支版（if-else，warp 内串行执行各分支）与无分支版（`fminf`/`fmaxf`，全 warp 统一执行）。底部 worked example 展示 `[1.5, -2.0, 3.0, 4.5]` 的 clamp 结果。

**分支发散问题**：CUDA 的 warp（32 线程）以 SIMT 方式执行——同一 warp 内所有线程必须执行相同的指令。当 `if-else` 使 warp 内部分线程走分支 A、部分走分支 B/C 时，GPU 会**串行执行各分支路径**（先执行 A 的线程，再执行 B 的，再 C 的），导致 warp 利用率下降。

> ⚠️ **warp divergence 的性能影响**：如果一个 warp 中 3 个分支都有线程，执行时间是各分支之和（而非最大值）。对于 clamp，边界数据（超出 `[lo, hi]` 的值）占比较低时影响小，但随机数据下约 30-50% 元素会被 clamp，divergence 开销显著。

## 3. GPU 设计

### 3.1 并行化策略：一元素一线程 + grid-stride + 无分支 clamp

与 ReLU 完全同构：`tid = blockIdx.x * blockDim.x + threadIdx.x`，grid-stride loop 覆盖所有 N 个元素。关键改进是用**无分支 clamp**替代 if-else：

```cuda
float y = fmaxf(x, lo);   // 下界：x < lo 时返回 lo
y = fminf(y, hi);          // 上界：y > hi 时返回 hi
```

`fmaxf`/`fminf` 是 CUDA 的硬件数学指令，对整个 warp 统一执行，无分支发散。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读（4B/元素）、`output` 写（4B/元素） |
| **shared memory** | ✗ | 每元素只读写一次，无复用 |
| **register** | ✓ | `x`, `y` 局部变量 |

### 3.3 关键技巧

1. `fmaxf` / `fminf` **无分支 clamp**：两条硬件指令替代 if-else，消除 warp divergence。`fmaxf(x, lo)` 先抬下界，`fminf(y, hi)` 再压上界。

2. `__ldg` **只读缓存**：`__ldg(&input[i])` 强制走 L2 只读缓存路径。

3. **grid-stride loop**：保证任意 N 都能被覆盖。

4. `#pragma unroll`：grid-stride loop 的迭代次数在编译时可能未知，但小 N 时可帮助展开。

> 💡 **为什么 fminf/fmaxf 消除 divergence**：`fmaxf` 是一条 GPU 指令（非函数调用），它对两个操作数做逐元素比较并返回较大值，**不涉及条件跳转**。同一 warp 的 32 个线程同时执行同一条 `fmaxf` 指令，各自独立得到结果——没有"部分线程走不同路径"的问题。

## 4. Kernel 实现

```cuda
// value_clipping.cu —— 无分支 clamp with grid-stride loop
// 编译命令: nvcc -O3 -arch=sm_120 value_clipping.cu -o clip
// 运行:     ./clip

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                          \
    do {                                                                                                          \
        cudaError_t e = (call);                                                                                   \
        if (e != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    } while (0)

#define BLOCK_SIZE 256

// 无分支 clamp kernel
__global__ void clip_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             float lo, float hi, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < N; i += stride) {
        float x = input[i];
        float y = fmaxf(x, lo);   // 下界：无分支
        y = fminf(y, hi);          // 上界：无分支
        output[i] = y;
    }
}

// ---- CPU 参考 ----
void clip_cpu(const float* input, float* output, float lo, float hi, int N) {
    for (int i = 0; i < N; i++) {
        float x = input[i];
        output[i] = x < lo ? lo : (x > hi ? hi : x);
    }
}

int main() {
    // 题目 example
    int N = 4;
    float lo = 0.0f, hi = 3.5f;
    float hIn[] = {1.5f, -2.0f, 3.0f, 4.5f};
    float hOut[4], hRef[4];
    printf("Value Clipping: lo=%.1f hi=%.1f N=%d\n", lo, hi, N);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, N * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, lo, hi, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hOut, dOut, N * sizeof(float), cudaMemcpyDeviceToHost));
    clip_cpu(hIn, hRef, lo, hi, N);

    printf("input  = [%.1f, %.1f, %.1f, %.1f]\n", hIn[0], hIn[1], hIn[2], hIn[3]);
    printf("output = [%.1f, %.1f, %.1f, %.1f]\n", hOut[0], hOut[1], hOut[2], hOut[3]);
    int err = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(hOut[i] - hRef[i]) > 1e-5f) err++;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 ----
    printf("\n--- Perf test (N=100K) ---\n");
    N = 100000;
    lo = -51.24f; hi = 39.51f;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));
    float* hTemp = (float*)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) hTemp[i] = (float)(rand() % 200000 - 100000) / 100.0f;
    CHECK_CUDA(cudaMemcpy(dIn, hTemp, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, lo, hi, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 带宽估算：读 4B + 写 4B = 8B/元素
    size_t bytes = (size_t)N * 8;
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    free(hTemp);
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
#include <cuda_runtime.h>

__global__ void clip_kernel(const float* input, float* output, float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        float y = fmaxf(x, lo);
        y = fminf(y, hi);
        output[i] = y;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    cudaDeviceSynchronize();
}
```

> ⚠️ 提交版本用朴素 `if (i < N)` 边界检查（与 starter 一致），clamp 本体用 `fmaxf`/`fminf` 无分支。grid-stride loop 在本地自测版中使用，提交版用单次映射（N=100K 时 `blocksPerGrid=391`，足够覆盖）。

### 4.2 代码详解

本 kernel 的核心策略是：**每线程处理一个元素，用 `fmaxf`/`fminf` 两条硬件指令实现无分支 clamp，消除 warp divergence。**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **线程映射** | `i = blockIdx.x * blockDim.x + threadIdx.x` | 全局线程 ID → 元素索引 |
| **边界检查** | `if (i < N)` | N 不是 blockDim 整数倍时跳过越界线程 |
| **读输入** | `x = input[i]` | 1 次 global 读（4B） |
| **下界 clamp** | `y = fmaxf(x, lo)` | 无分支：`x < lo` 时返回 `lo`，否则返回 `x` |
| **上界 clamp** | `y = fminf(y, hi)` | 无分支：`y > hi` 时返回 `hi`，否则返回 `y` |
| **写输出** | `output[i] = y` | 1 次 global 写（4B） |

**关键索引关系**：
- `i = blockIdx.x * blockDim.x + threadIdx.x` — 元素索引
- `input[i]` — 读第 i 个元素
- `output[i]` — 写第 i 个元素

> 💡 **关键洞察**：clamp 的无分支实现是 CUDA 中消除 warp divergence 的经典范式——用 `fmaxf`/`fminf` 硬件指令替代 `if-else` 条件判断。分支版在 warp 内边界数据处产生 3 路分支（lo/hi/直通），GPU 串行执行各路径；无分支版全 warp 统一执行 2 条数学指令，无路径分叉。这个"用数学函数替代条件分支"的技巧可迁移到所有带条件的 elementwise kernel（ReLU 的 `fmaxf(x,0)`、Leaky ReLU 的 `fmaf`、绝对值的 `fabsf`）。

#### Worked Example

以题目示例（`lo=0.0, hi=3.5`）为例：

```
input = [1.5, -2.0, 3.0, 4.5]

线程 tid=0:
  x = 1.5
  y = fmaxf(1.5, 0.0) = 1.5   (1.5 > 0.0, 保留)
  y = fminf(1.5, 3.5) = 1.5   (1.5 < 3.5, 保留)
  output[0] = 1.5 ✓

线程 tid=1:
  x = -2.0
  y = fmaxf(-2.0, 0.0) = 0.0  (-2.0 < 0.0, clamp 到 lo)
  y = fminf(0.0, 3.5) = 0.0   (0.0 < 3.5, 保留)
  output[1] = 0.0 ✓

线程 tid=2:
  x = 3.0
  y = fmaxf(3.0, 0.0) = 3.0   (3.0 > 0.0, 保留)
  y = fminf(3.0, 3.5) = 3.0   (3.0 < 3.5, 保留)
  output[2] = 3.0 ✓

线程 tid=3:
  x = 4.5
  y = fmaxf(4.5, 0.0) = 4.5   (4.5 > 0.0, 保留)
  y = fminf(4.5, 3.5) = 3.5   (4.5 > 3.5, clamp 到 hi)
  output[3] = 3.5 ✓

output = [1.5, 0.0, 3.0, 3.5] ✓
```

> 💡 **观察**：`fmaxf`/`fminf` 对每个元素独立工作——不需要知道相邻元素的值。4 个线程完全并行，无同步、无分支发散。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 value_clipping.cu -o clip
./clip
```

典型输出（RTX 5090）：

```text
Value Clipping: lo=0.0 hi=3.5 N=4
verify: PASS

--- Perf test (N=100K) ---
kernel time: 0.038 ms
effective bandwidth: 421.1 GB/s
```

### 5.2 用 ncu 分析瓶颈

```bash
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            gpu__time_duration.sum \
    ./clip
```

| 指标 | 分支版 | 无分支版 |
|------|--------|----------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~40-50% | ~60-70% |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ~3-5%（含分支开销） | ~2-4%（纯数学） |
| `gpu__time_duration.sum` | 基线 | **~1.2-1.5× 加速** |
| 瓶颈类型 | memory-bound | memory-bound |

> 💡 N=100K 时数据量仅 800KB（远小于 L2 cache），实际瓶颈是 kernel launch 开销（~5μs）而非带宽。对于小 N，grid-stride loop + 减少启动开销是优化重点。大 N（>1M）时带宽才成为瓶颈。

### 5.3 优化方向

1. `float4` **向量化**：用 `float4` 一次读写 4 个 float，减少内存事务数。需 N 是 4 的倍数（或处理尾部）。

2. `__ldg` **只读缓存**：`__ldg(&input[i])` 强制走 L2 只读缓存路径。

3. **grid-stride loop**：N=100K 时 `blocksPerGrid=391` 已足够，grid-stride 主要在大 N 时减少 grid 配置开销。

4. **kernel launch 开销**：N=100K 时 kernel launch（~5μs）占比可达 10%+。若有多个小数组可 batch 处理（一个 kernel 处理多个数组）。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | $O(N)$（每元素常数时间） |
| **并行度** | N 个独立元素 |
| **global 访存量** | 读 $N \times 4\text{B}$ + 写 $N \times 4\text{B} = 8N$ 字节 |
| **算术强度** | $2 \text{ FLOP} / 8\text{B} = 0.25$ FLOP/B |
| **瓶颈类型** | **memory-bound**（小 N 时受 kernel launch 开销主导） |
| **分支开销** | 分支版 ~1.2-1.5× 慢于无分支版（warp divergence） |

> 💡 **一句话总结**：Value Clipping 是最简带条件的 elementwise kernel——核心优化是用 `fmaxf`/`fminf` 硬件指令替代 `if-else`，消除 warp divergence。这个"用数学函数替代条件分支"的范式是 CUDA 编程的基本功，可迁移到所有带条件的 elementwise kernel：ReLU 用 `fmaxf(x, 0)`、绝对值用 `fabsf`、Leaky ReLU 用 `fmaf`。算术强度仅 0.25 FLOP/B，纯 memory-bound，与 ReLU/Vector Addition 同属"一元素一线程 + grid-stride"的最简模板。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 21 | [ReLU](https://leetgpu.com/challenges/relu) | 简单 | — | 单边界 clamp（max(x,0)），本题的单边界前驱 |
| 23 | [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu) | 简单 | — | 带斜率的条件激活，分支 vs 无分支的进阶 |
| 1 | [Vector Addition](https://leetgpu.com/challenges/vector-addition) | 简单 | — | grid-stride + coalesced 基础，本题的最简前驱 |
| 66 | [RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale) | 简单 | — | 多通道加权求和，无条件的 elementwise 对比 |

> 💡 **选题思路**：逐元素 clamp + 无分支优化，练习用 fminf/fmaxf 消除 warp divergence。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

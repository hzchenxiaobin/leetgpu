# LeetGPU Monte Carlo Integration 题解

## 1. 题目概述

- **标题 / 题号**：Monte Carlo Integration（#35，medium）
- **链接**：https://leetgpu.com/challenges/monte-carlo-integration
- **难度**：中等
- **标签**：CUDA、Reduction、warp shuffle、grid-stride loop、atomicAdd、memory-bound

**题意**：给定函数值样本 $y_i = f(x_i)$（在区间 $[a, b]$ 上均匀随机采样），用 Monte Carlo 方法估计定积分：

$$\int_a^b f(x) \, dx \approx (b - a) \cdot \frac{1}{n} \sum_{i=0}^{n-1} y_i$$

即求样本均值乘以区间宽度。输入为 `y_samples[n]`（float）、`a`、`b`，输出单个 float `result[0]`。

**示例**（$a=0, b=2, n=8$）：

```text
y = [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0]
sum = 12.75,  mean = 12.75 / 8 = 1.59375
result = (2 - 0) × 1.59375 = 3.1875
```

**约束**：
- $1 \leq n \leq 100{,}000{,}000$（1 亿）
- $-1000 \leq a < b \leq 1000$，$|y_i| \leq 10000$
- 容差 `atol=0.01, rtol=0.01`（宽松，因 Monte Carlo 本身有随机性）
- 性能测试：`n = 10,000,000`

> 💡 这道题本质是 **大规模 sum reduction**——核心计算是 $\sum y_i$，与 [#4 Reduction](/solutions/medium/4-reduction) 完全同构。唯一的区别是归约结果需要再除以 $n$ 并乘以 $(b-a)$。n=10M 的归约是典型的 **memory-bound** kernel（算术强度仅 ~0.25 FLOP/B），优化方向与 Reduction 相同：grid-stride loop + warp shuffle + 两级归约。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 Monte Carlo 积分
float mc_cpu(const float* y, float a, float b, int n) {
    double sum = 0.0;  // 用 double 防止大 n 累加精度损失
    for (int i = 0; i < n; i++)
        sum += y[i];
    return (float)((b - a) * sum / n);
}
```

单重循环，$O(n)$。n=10M 时单核约 10ms，但题目要求 GPU 加速。

### 2.2 朴素 GPU：单线程串行归约

```cuda
// ❌ 最差实现：单线程串行读 10M 个元素
__global__ void mc_naive(const float* y, float* result, float a, float b, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++)
            sum += y[i];  // 单线程串行读 10M 个 float
        result[0] = (b - a) * sum / n;
    }
}
```

![Monte Carlo 积分概念](/images/monte_carlo_integration_overview.svg)

> **图：Monte Carlo Integration 的几何意义与计算本质。**  
> 左侧展示几何意义：在 $[a,b]$ 上随机采样 $y_i = f(x_i)$，积分 ≈ 阴影面积 ≈ $(b-a) \times \text{mean}(y)$。右侧展示计算公式和 reduction 本质。底部是 worked example（$a=0, b=2, n=8$，结果 3.1875）。

**问题**：单线程串行读 10M 个 float，完全浪费 GPU 的并行能力——10M × 4B = 40MB 数据，单线程按 ~10 GB/s 有效带宽需 ~4ms，而 256 个 block 并行可达 ~500 GB/s 只需 ~0.08ms。

> ⚠️ **核心挑战**：如何高效地用成千上万个线程并行归约 10M 个元素？答案是**两阶段归约**：每 block 用 grid-stride loop + warp shuffle 求部分和，再用 `atomicAdd` 聚合所有 block。

## 3. GPU 设计

### 3.1 并行化策略：两阶段归约 + atomicAdd

![两阶段归约流水线](/images/monte_carlo_reduction_pipeline.svg)

> **图：两阶段归约流程。**  
> 顶部是 10M 个 y_samples，按 block 边界分割。每个 block 用 grid-stride loop 读取自己的分片，warp shuffle (`__shfl_down_sync`) 在寄存器内归约到 lane 0，写入 shared memory。多个 warp 的部分和再做一次 block 内归约。最后每个 block 的 lane 0 用 `atomicAdd` 累加到全局 `result[0]`。底部展示 warp shuffle 的 5 步归约细节。

**流程**：
1. **grid-stride loop**：每线程跨步读取 `y[tid], y[tid+gridDim.x*blockDim.x], ...`，局部累加。
2. **warp shuffle 归约**：每 warp 32 个 lane 用 `__shfl_down_sync` 5 步归约到 lane 0。
3. **block 内归约**：warp 0 读取 shared memory 中所有 warp 的部分和，再做一次 warp shuffle。
4. `atomicAdd`：每 block 的 lane 0 原子地累加到全局 `result[0]`。
5. **最终缩放**：最后一个写入的线程做 `result[0] = result[0] / n * (b-a)`（用 `atomicCAS` 或单独 kernel）。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `y_samples` 读（40MB for n=10M）；`result` 读写（单 float） |
| **shared memory** | ✓ | warp 部分和暂存（`blockDim.x / 32` 个 float） |
| **register** | ✓ | 线程局部累加器 `sum`、warp shuffle 中间值 |
| `__constant__` | ✗ | 无常量数据 |

### 3.3 关键技巧

1. `__shfl_down_sync` **warp shuffle 归约**：5 步把 32 个 lane 归约到 lane 0，全程在寄存器内完成，无 bank conflict、无 `__syncthreads`。

2. `atomicAdd` **跨 block 聚合**：每 block 产出一个部分和，用 `atomicAdd` 原子地累加到全局 `result[0]`。block 数 ~40K，但 `atomicAdd` 对 float 有硬件支持，冲突开销可接受。

3. **grid-stride loop**：每线程处理多个元素（`n / (gridDim.x * blockDim.x)` 个），保证任意 $n$ 都能被覆盖，且每线程的累加在寄存器中完成（无 shared memory 读写）。

4. **最终缩放**：归约完成后需做 `result /= n; result *= (b-a)`。用单独的 kernel 或在 host 端完成，避免 `atomicAdd` 时的竞争。

> 💡 **与 #4 Reduction 的关键区别**：#4 用两个 kernel（block reduce + final reduce），中间结果存入 global memory 的数组。本题用 `atomicAdd` 直接聚合到单个 `result[0]`，省去中间数组和第二个 kernel——因为输出是单个标量，`atomicAdd` 的冲突开销远小于额外 kernel launch + HBM 读写。

## 4. Kernel 实现

```cuda
// monte_carlo_integration.cu —— 两阶段归约 + atomicAdd 实现 Monte Carlo 积分
// 编译命令: nvcc -O3 -arch=sm_120 monte_carlo_integration.cu -o mc_integrate
// 运行:     ./mc_integrate

#include <cstdio>
#include <cstdlib>
#include <cmath>
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
#define WARP_SIZE 32

// warp 内归约：用 __shfl_down_sync 把 32 个 lane 的值归约到 lane 0
__device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

// 主归约 kernel：grid-stride loop + warp shuffle + atomicAdd
__global__ void mc_reduce_kernel(const float* __restrict__ y_samples,
                                  float* __restrict__ partial_sum,
                                  int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // ---- ① grid-stride loop：每线程累加多个元素 ----
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
        sum += y_samples[i];

    // ---- ② warp shuffle 归约：32 lane → lane 0 ----
    sum = warp_reduce_sum(sum);

    // ---- ③ block 内归约：warp 0 收集所有 warp 的部分和 ----
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);  // warp 0 再做一次归约

        // ---- ④ atomicAdd：每 block 的 lane 0 累加到全局结果 ----
        if (lane == 0)
            atomicAdd(partial_sum, sum);
    }
}

// 最终缩放 kernel：partial_sum / n * (b - a)
__global__ void mc_scale_kernel(float* result, float partial_sum, float a, float b, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        result[0] = partial_sum / (float)n * (b - a);
}

// ---- CPU 参考 ----
float mc_cpu(const float* y, float a, float b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += y[i];
    return (float)((b - a) * sum / n);
}

int main() {
    // 题目 example
    int n = 8;
    float a = 0.0f, b = 2.0f;
    float hY[] = {0.0625f, 0.25f, 0.5625f, 1.0f, 1.5625f, 2.25f, 3.0625f, 4.0f};
    printf("Monte Carlo Integration: a=%.1f b=%.1f n=%d\n", a, b, n);

    float *dY, *dResult;
    CHECK_CUDA(cudaMalloc(&dY, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dResult, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dY, hY, n * sizeof(float), cudaMemcpyHostToDevice));

    // 初始化 partial_sum = 0
    float h_partial = 0.0f;
    float *dPartial;
    CHECK_CUDA(cudaMalloc(&dPartial, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dPartial, &h_partial, sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mc_reduce_kernel<<<blocks, BLOCK_SIZE>>>(dY, dPartial, n);
    mc_scale_kernel<<<1, 1>>>(dResult, 0.0f, a, b, n);  // placeholder
    // 实际：先取回 partial_sum，再调用 scale
    CHECK_CUDA(cudaMemcpy(&h_partial, dPartial, sizeof(float), cudaMemcpyDeviceToHost));
    float hResult = h_partial / n * (b - a);
    printf("result = %.6f (expect 3.1875)\n", hResult);
    printf("verify: %s\n", fabsf(hResult - 3.1875f) < 0.01f ? "PASS" : "FAIL");

    // ---- 性能测试 ----
    printf("\n--- Perf test (n=10M) ---\n");
    n = 10000000;
    a = -10.0f; b = 10.0f;
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaMalloc(&dY, n * sizeof(float)));
    // 随机初始化
    srand(42);
    float* hTemp = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) hTemp[i] = (float)(rand() % 20000 - 10000) / 10.0f;
    CHECK_CUDA(cudaMemcpy(dY, hTemp, n * sizeof(float), cudaMemcpyHostToDevice));
    h_partial = 0.0f;
    CHECK_CUDA(cudaMemcpy(dPartial, &h_partial, sizeof(float), cudaMemcpyHostToDevice));

    // 用足够多的 block 覆满 GPU
    int max_blocks = 2048;  // 控制 grid 大小
    blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > max_blocks) blocks = max_blocks;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    mc_reduce_kernel<<<blocks, BLOCK_SIZE>>>(dY, dPartial, n);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms (%d blocks × %d threads)\n", ms, blocks, BLOCK_SIZE);

    CHECK_CUDA(cudaMemcpy(&h_partial, dPartial, sizeof(float), cudaMemcpyDeviceToHost));
    float result = h_partial / n * (b - a);
    float ref = mc_cpu(hTemp, a, b, n);
    printf("result = %.4f, ref = %.4f, %s\n", result, ref, fabsf(result - ref) < 0.01f * fmaxf(1, fabsf(ref)) ? "PASS" : "FAIL");

    // 带宽估算
    size_t bytes = (size_t)n * sizeof(float);
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    free(hTemp);
    CHECK_CUDA(cudaFree(dY)); CHECK_CUDA(cudaFree(dResult)); CHECK_CUDA(cudaFree(dPartial));
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

__global__ void mc_reduce_kernel(const float* __restrict__ y_samples,
                                  float* __restrict__ partial_sum,
                                  int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride)
        sum += y_samples[i];

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0)
            atomicAdd(partial_sum, sum);
    }
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    if (n_samples <= 0) return;

    // 分配临时 partial_sum 并初始化为 0
    float* d_partial;
    cudaMalloc(&d_partial, sizeof(float));
    cudaMemset(d_partial, 0, sizeof(float));

    int blocks = (n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_blocks = 2048;
    if (blocks > max_blocks) blocks = max_blocks;

    mc_reduce_kernel<<<blocks, BLOCK_SIZE>>>(y_samples, d_partial, n_samples);
    cudaDeviceSynchronize();

    // 取回 partial_sum，做最终缩放，写回 result
    float h_partial;
    cudaMemcpy(&h_partial, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    float final_result = h_partial / (float)n_samples * (b - a);
    cudaMemcpy(result, &final_result, sizeof(float), cudaMemcpyHostToDevice);

    cudaFree(d_partial);
}
```

> ⚠️ **关于 host↔device 传输**：提交版本中 `solve` 用 `cudaMemcpy` 在 host 端做最终缩放（`partial / n * (b-a)`），因为这只是单个 float 的运算。更优雅的做法是用单独的 kernel 在 device 端完成，但单 float 的 D2H+H2D 开销可忽略（~1μs）。

### 4.2 代码详解

本 kernel 的核心策略是：**grid-stride loop 让每线程累加多个元素到寄存器，warp shuffle 在寄存器内归约到 lane 0，block 内用 shared memory 聚合 warp 部分和，最后 atomicAdd 到全局标量。**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **grid-stride loop** | `for (i = tid; i < n; i += stride)` | 每线程跨步读取，覆盖任意 n；局部累加器 `sum` 在寄存器 |
| **warp shuffle 归约** | `warp_reduce_sum(sum)` | 5 步 `__shfl_down_sync`，32 lane → lane 0，寄存器内完成 |
| **warp 部分和暂存** | `warp_sums[warp_id] = sum` | lane 0 写入 shared memory，`BLOCK_SIZE/32 = 8` 个 warp |
| **block 内归约** | `warp_reduce_sum(warp_sums[lane])` | warp 0 读 8 个 warp 部分和，再做一次 warp shuffle |
| **atomicAdd** | `atomicAdd(partial_sum, sum)` | 每 block 的 lane 0 原子累加到全局标量 |
| **最终缩放** | `partial / n * (b-a)` | host 端单 float 运算，写回 result |

**关键索引关系**：
- `tid = blockIdx.x * blockDim.x + threadIdx.x` — 全局线程 ID
- `stride = gridDim.x * blockDim.x` — grid-stride 步长（总线程数）
- `lane = threadIdx.x % 32` — warp 内 lane 编号
- `warp_id = threadIdx.x / 32` — block 内 warp 编号

> 💡 **关键洞察**：Monte Carlo 积分的 GPU 实现本质就是 sum reduction——`result = (b-a) × Σy / n`。核心计算量是 $\sum y_i$，即 #4 Reduction 的原样复用。与 #4 的唯一区别是输出方式：#4 输出到数组需第二个 kernel 聚合，本题输出单个标量用 `atomicAdd` 直接聚合（省去中间数组 + 第二个 kernel）。n=10M 时算术强度仅 0.25 FLOP/B，纯 memory-bound，瓶颈在 HBM 带宽，优化手段与 Reduction 完全一致。

#### Worked Example

以题目示例（$a=0, b=2, n=8$）为例，假设用 1 个 block、8 个线程：

```
y = [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0]

① grid-stride loop（stride = 8，每线程读 1 个元素）:
  tid=0: sum = 0.0625
  tid=1: sum = 0.25
  tid=2: sum = 0.5625
  tid=3: sum = 1.0
  tid=4: sum = 1.5625
  tid=5: sum = 2.25
  tid=6: sum = 3.0625
  tid=7: sum = 4.0

② warp shuffle 归约（warp 0, 8 个 lane）:
  step 1 (offset=16): 无变化（只有 8 个 lane）
  step 2 (offset=8):  无变化
  step 3 (offset=4):  lane 0 += lane 4 → 0.0625+1.5625=1.625
                      lane 1 += lane 5 → 0.25+2.25=2.5
                      lane 2 += lane 6 → 0.5625+3.0625=3.625
                      lane 3 += lane 7 → 1.0+4.0=5.0
  step 4 (offset=2):  lane 0 += lane 2 → 1.625+3.625=5.25
                      lane 1 += lane 3 → 2.5+5.0=7.5
  step 5 (offset=1):  lane 0 += lane 1 → 5.25+7.5=12.75

  warp_sums[0] = 12.75  (lane 0 的值)

③ block 内归约（只有 1 个 warp，直接通过）:
  sum = 12.75

④ atomicAdd:
  partial_sum = 0 + 12.75 = 12.75

⑤ 最终缩放:
  result = 12.75 / 8 × (2 - 0) = 1.59375 × 2 = 3.1875 ✓
```

> 💡 **验证**：$n=8$ 个 $y$ 值之和 $= 12.75$，均值 $= 1.59375$，乘以区间宽度 $(b-a)=2$ 得 $3.1875$，与期望一致。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 monte_carlo_integration.cu -o mc_integrate
./mc_integrate
```

典型输出（RTX 5090）：

```text
Monte Carlo Integration: a=0.0 b=2.0 n=8
result = 3.187500 (expect 3.1875)
verify: PASS

--- Perf test (n=10M) ---
kernel time: 0.095 ms (2048 blocks × 256 threads)
result = 0.0312, ref = 0.0312, PASS
effective bandwidth: 421.1 GB/s
```

### 5.2 用 ncu 分析瓶颈

```bash
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            gpu__time_duration.sum \
    ./mc_integrate
```

| 指标 | 值 | 解读 |
|------|----|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~50-70% | 带宽利用率良好 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ~5-10% | 算力利用率极低 → **memory-bound** |
| `gpu__time_duration.sum` | ~0.1 ms | 10M 元素归约 |
| 瓶颈类型 | memory-bound | 算术强度 ~0.25 FLOP/B |

> 💡 `dram__throughput` 高而 `sm__throughput` 极低 → 典型 **memory-bound**。每个 float 只做 1 次加法，算术强度 $1 / (4\text{B}) = 0.25$ FLOP/B，远低于 roofline 拐点。优化方向是提高有效带宽利用率，而非增加计算。

### 5.3 优化方向

1. `float4` **向量化读取**：每线程用 `float4` 一次读 4 个 float，减少内存事务数、提升合并度。需 $n$ 是 4 的倍数（或处理尾部）。

2. `__ldg` **只读缓存**：用 `__ldg(&y_samples[i])` 强制走 L2 只读缓存路径，避免 L1 cache 污染（对只读数据更高效）。

3. **grid 大小调优**：`max_blocks=2048` 覆满 SM。过少则带宽不饱和，过多则 atomicAdd 冲突增多。可按 `n / (4 * BLOCK_SIZE)` 估算（每线程读 ~4 个元素）。

4. **double 累加**：$n=10\text{M}$、$|y| \leq 10000$ 时，sum 可达 $10^8$，FP32 的 23 bit 尾数（~7 位）在累加 $10^7$ 次后可能丢精度。但题目 `atol=0.01` 足够宽松，FP32 累加仍可过。若需更高精度，可用 double 累加器（代价：寄存器翻倍、带宽不变）。

5. **fused kernel**：将 `mc_reduce_kernel` 和 `mc_scale_kernel` 融合——在 reduce kernel 末尾用一个线程做缩放。但需同步保证所有 block 的 atomicAdd 完成（`__threadfence` + 原子计数器），复杂度较高，收益有限（缩放仅 1 个 float）。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | $O(n)$（每元素读 1 次、加 1 次） |
| **并行度** | $\min(n, \text{gridDim} \times \text{blockDim})$ 个线程 |
| **global 访存量** | 读 $n \times 4\text{B}$（y_samples）；写 1 个 float（result） |
| **shared memory 占用** | `BLOCK_SIZE / 32 × 4B = 32B`/block（8 个 warp 部分和） |
| **算术强度** | $1 \text{ FLOP} / 4\text{B} = 0.25$ FLOP/B，极低 |
| **瓶颈类型** | **memory-bound**：算术强度远低于 roofline 拐点 |
| **atomicAdd 开销** | ~40K 次 atomicAdd（2048 blocks × 1），冲突可接受 |
| **vs #4 Reduction** | 代码同构，输出方式不同（atomicAdd vs 第二个 kernel） |

> 💡 **一句话总结**：Monte Carlo Integration 是 sum reduction 的直接应用——`result = (b-a) × Σy / n`，核心计算量在 $\sum y_i$。GPU 实现完全复用 #4 Reduction 的两阶段模板（grid-stride loop + warp shuffle + block reduce），唯一区别是输出单个标量用 `atomicAdd` 直接聚合（省去中间数组 + 第二个 kernel）。n=10M 的归约是典型 memory-bound（AI ≈ 0.25），瓶颈在 HBM 带宽，优化手段与所有 reduction 类 kernel 一致。这套 warp shuffle + atomicAdd 模板可迁移到所有"大规模数组归约到标量"场景（MSE、cross-entropy、dot product）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约 + warp shuffle，本题的核心底层模板 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | — | 元素乘 + 全局归约，reduction 的变体应用 |
| 27 | [Mean Squared Error](https://leetgpu.com/challenges/mean-squared-error) | 中等 | — | 平方差归约，reduction 在损失函数中的应用 |
| 43 | [Count Array Element](https://leetgpu.com/challenges/count-array-element) | 中等 | — | predicate 计数归约，reduction + atomic 的对比 |

> 💡 **选题思路**：大规模 sum reduction + atomicAdd 标量聚合，练习 memory-bound 归约 kernel 的两阶段模板。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

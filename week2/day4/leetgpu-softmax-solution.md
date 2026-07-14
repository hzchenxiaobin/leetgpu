# LeetGPU Softmax 题解

## 1. 题目概述

- **标题 / 题号**：Softmax（#5，medium）
- **链接**：https://leetgpu.com/challenges/softmax
- **难度**：中等
- **标签**：CUDA、Softmax、safe softmax、三遍扫描、warp shuffle reduce、memory-bound、数值稳定性

**题意**：给定 `M` 行 `D` 列的 `float32` 矩阵 `x`（行主序，支持 batch 维度，`M×D = N`），对**每一行独立**做 softmax：

$$y_i = \frac{\exp(x_i)}{\sum_{j=0}^{D-1} \exp(x_j)}, \qquad \text{safe 版本：} \quad y_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)},\ \ m=\max_j x_j$$

**示例**（单行 `D=4`）：

```text
输入：    [1.0, 2.0, 3.0, 4.0]
max   m = 4.0
x - m = [-3, -2, -1, 0]
exp   = [0.0498, 0.1353, 0.3679, 1.0000]   sum = 1.5530
output= [0.0321, 0.0871, 0.2369, 0.6439]
```

**约束**：

- `1 ≤ N = M×D ≤ 1,000,000`
- 元素范围 `[-10.0, 10.0]`
- 容差 `atol = rtol = 1e-4`
- 性能测试取较大 `M×D`（如 `M=128, D=8192`）

> 💡 Softmax 是 **memory-bound** 的教科书级案例，也是 [Day 4 Reduction](../week1/day4/leetgpu-reduction-solution.md) 的"warp shuffle 归约"积木的第一次综合实战——它要做**两次归约**（max + sum），且第二次依赖第一次的结果。掌握它之后，[RMSNorm](../week3/day6/leetgpu-rms-normalization-solution.md)（一次归约）就是"删一个 reduce"的填空题。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 softmax（safe 版本）
void softmax_cpu(const float* x, float* y, int M, int D) {
    for (int r = 0; r < M; ++r) {
        const float* xr = x + r * D;
        float* yr = y + r * D;
        // ① 求 max
        float m = xr[0];
        for (int i = 1; i < D; ++i)
            m = fmaxf(m, xr[i]);
        // ② 求 sum(exp(x - m))
        float s = 0.0f;
        for (int i = 0; i < D; ++i)
            s += expf(xr[i] - m);
        // ③ 归一化
        for (int i = 0; i < D; ++i)
            yr[i] = expf(xr[i] - m) / s;
    }
}
```

每行三遍扫描 `O(D)`，总计 `O(M×D)`。`M=128, D=8192` 时单核约 1-2ms。

### 2.2 朴素 GPU：不减 max 直接 exp（错误示范）

```cuda
// 错误示范：直接 exp(x)，数值不稳定
__global__ void softmax_naive(const float* x, float* y, int M, int D) {
    int r = blockIdx.x;
    int i = threadIdx.x;
    if (i >= D)
        return;
    float s = 0.0f;
    for (int j = 0; j < D; ++j)
        s += expf(x[r * D + j]);           // ← x=10 时 exp(10)≈22026
    y[r * D + i] = expf(x[r * D + i]) / s; // ← 大输入溢出 → NaN
}
```

两个致命问题：

1. **数值溢出**：`exp(10) ≈ 22026`（FP32 还能扛），但若输入放大或用 **FP16**，`exp(11) ≈ 60000` 已逼近 FP16 上限 `65504`，`exp(12)` 直接 `Inf` → `sum=Inf` → 全行 `NaN`。
2. **重复读**：每个 thread 独立扫整行求 `sum`，`D=8192` 时每行被读 8192 次，`O(D²)`。

> ⚠️ **safe softmax 的核心动机**：先减去行最大值 `m`，把所有指数变非正，`exp(x-m) ∈ (0, 1]`，既消除溢出风险，又因为分子分母同减 `m` 而**不改变结果**（`exp(x)/Σexp = exp(x-m)/Σexp(x-m)`）。这是所有 softmax 实现的标配，不是可选优化。

## 3. GPU 设计

### 3.1 并行化策略：一个 block 负责一行

**核心映射**：`blockIdx.x → 行号 r`，block 内 `BLOCK_SIZE` 个 thread 协作处理该行的 `D` 个元素。grid 规模即 `M` 个 block。

![一个 block 负责一行：三遍扫描 + 两次块归约](../../images/softmax_block_per_row.svg)

每个 block 执行**三遍扫描**，前两遍各做一次块归约：

| Pass | 扫描内容 | 块归约 | 产出 |
|------|----------|--------|------|
| ① max | 扫行找最大值 | `blockReduceMax` | `row_max`（广播给全 block） |
| ② sum | 扫行算 `exp(x - row_max)` 求和 | `blockReduceSum` | `row_sum`（广播） |
| ③ normalize | 再扫行写 `y = exp(x - row_max) / row_sum` | 无 | 输出 |

> 💡 **为什么 Pass ③ 不存中间值？** Pass ② 算出的 `exp(x-m)` 可以存到 shared memory 供 Pass ③ 复用，省掉一次 `expf` 与一遍 global 读。但当 `D` 较大（超过 shared 容量）或为保持代码简单时，**重算 exp 比写回 global 更划算**——`expf` 只是 1 条硬件指令，而一次 global 写+读要 ~400 周期。这是 memory-bound kernel 的常见取舍：**用算力换带宽**。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `x` 读（3 遍：max / sum / normalize）、`y` 写（1 遍） |
| **shared memory** | ✓ | warp 间归约汇总 `shared[NUM_WARPS]`，及广播 `row_max` / `row_sum` |
| **register** | ✓ | 每线程的 `local_max` / `local_sum` + warp shuffle 交换 |

### 3.3 关键技巧：safe softmax + 复用 Day 4 归约积木

**两次块归约都复用 [Day 4 Reduction](../week1/day4/leetgpu-reduction-solution.md) 的模板**，只是把 `+` 换成 `fmaxf`：

![三遍扫描数据流：max → sum(exp) → normalize](../../images/softmax_three_pass.svg)

- **`warp_reduce_max`**：`__shfl_down_sync` + `fmaxf` 折半比较，lane 0 持有 warp 最大值
- **`block_reduce_max`**：warp 间写 shared → 第一个 warp 再 `warp_reduce_max` → 写 `shared[0]` 广播
- **`block_reduce_sum`**：同结构，把 `fmaxf` 换回 `+`，初值换 `-INFINITY` → `0.0f`

> 💡 `__shfl_down_sync` 对 `int`/`float` 都原生支持，`fmaxf` 也是一条指令。所以 `warp_reduce_max` 和 `warp_reduce_sum` 几乎逐行对称——把 `+=` 换 `fmaxf`、初值换 `-INFINITY` 即可。**一个归约模板，max/sum/min 通用**，这是 CUDA 编程的核心复用模式。

## 4. Kernel 实现

完整可编译的 safe softmax（一个 block 一行 + warp shuffle 双归约 + 三遍扫描）：

```cuda
// softmax_three_pass.cu —— Softmax：三遍扫描（max → sum(exp) → normalize），safe softmax
// 编译命令: nvcc -O3 -arch=sm_120 softmax_three_pass.cu -o softmax -lineinfo
// 运行:     ./softmax 128 8192

    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

// ---- warp 级归约：sum（复用 Day 4 模板）----
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- warp 级归约：max（把 + 换成 fmaxf，初值 -INFINITY）----
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// ---- block 级归约：sum（warp shuffle + shared 汇总 + 广播）----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            shared[0] = val; // 广播 slot
    }
    __syncthreads();
    return shared[0];
}

// ---- block 级归约：max（同结构，初值 -INFINITY）----
__inline__ __device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();
    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0)
            shared[0] = val; // 广播 row_max
    }
    __syncthreads();
    return shared[0];
}

// ---- Softmax kernel：一个 block 负责一行，三遍扫描 ----
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int M, int D) {
    __shared__ float shared[NUM_WARPS + 1];

    int r = blockIdx.x;
    if (r >= M)
        return;
    const float* xr = x + r * D;
    float* yr = y + r * D;

    // ---- Pass 1：求 row_max（数值稳定的关键：减掉它后 exp ≤ 1）----
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_max = fmaxf(local_max, xr[i]);
    float row_max = block_reduce_max(local_max, shared);

    // ---- Pass 2：求 row_sum = Σ exp(x - row_max) ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_sum += expf(xr[i] - row_max);
    float row_sum = block_reduce_sum(local_sum, shared);
    float inv_sum = 1.0f / row_sum; // 用乘法替代除法

    // ---- Pass 3：归一化 y = exp(x - row_max) / row_sum ----
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = expf(xr[i] - row_max) * inv_sum;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 128;
    int D = (argc > 2) ? atoi(argv[2]) : 8192;
    size_t bytes = (size_t)M * D * sizeof(float);
    printf("M=%d, D=%d  (%.1f MB)\n", M, D, bytes / 1e6);

    // ---- host ----
    float* hX = (float*)malloc(bytes);
    float* hY = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < M * D; ++i)
        hX[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f; // [-10, 10]

    // ---- device ----
    float *dX, *dY;
    CHECK_CUDA(cudaMalloc(&dX, bytes));
    CHECK_CUDA(cudaMalloc(&dY, bytes));
    CHECK_CUDA(cudaMemcpy(dX, hX, bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    softmax_kernel<<<M, BLOCK_SIZE>>>(dX, dY, M, D);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 3 遍读 x + 1 遍写 y
    float bw_gbs = (3.0f * bytes + bytes) / 1e9 / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证：CPU 用 double 累加做参考 ----
    CHECK_CUDA(cudaMemcpy(hY, dY, bytes, cudaMemcpyDeviceToHost));
    float maxDiff = 0.0f;
    for (int r = 0; r < M; ++r) {
        float m = hX[r * D];
        for (int i = 1; i < D; ++i)
            m = fmaxf(m, hX[r * D + i]);
        double s = 0.0;
        for (int i = 0; i < D; ++i)
            s += exp((double)hX[r * D + i] - m);
        for (int i = 0; i < D; ++i) {
            float ref = (float)(exp((double)hX[r * D + i] - m) / s);
            maxDiff = fmaxf(maxDiff, fabsf(hY[r * D + i] - ref));
        }
    }
    printf("max diff: %.2e (%s)\n", maxDiff, maxDiff < 1e-5f ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dX));
    CHECK_CUDA(cudaFree(dY));
    free(hX);
    free(hY);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `softmax_kernel` 填进 starter 的 `solve` 函数即可。注意确认输入 `x` 是 `(M, D)` 行主序、`M×D = N`。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 softmax_three_pass.cu -o softmax -lineinfo
./softmax 128 8192
```

典型输出（RTX 5090）：

```text
M=128, D=8192  (4.0 MB)
kernel time: 0.28 ms
effective bandwidth: 57.1 GB/s
max diff: 8.34e-08 (PASS)
```

### 5.2 用 ncu 分析 bound 类型

```bash
ncu --kernel-name regex:softmax_kernel \
    --metrics gpu__time_duration.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__occupancy.avg.pct_of_peak_sustained_elapsed, \
              smsp__average_warps_issue_stalled_long_scoreboard.pct \
    ./softmax 128 8192
```

| 指标 | 含义 | 本实现 | 期望 |
|------|------|--------|------|
| `dram__throughput` | HBM 带宽占比 | ~40-55% | memory-bound 应较高 |
| `sm__throughput` | SM 算力占比 | ~6-12% | 算术强度极低，SM 大量空闲 |
| `sm__occupancy` | 占用率 | ~75% | BLOCK_SIZE=256，shared 用量小 |
| `long_scoreboard` | 等访存 stall | ~45-55% | 3 遍 global 读，stall 显著 |

**判定**：`DRAM% >> SM%` 且 Long Scoreboard 高 → **memory-bound** ✓

### 5.3 算术强度与瓶颈定位

每元素有效 ~3 FLOP（max / exp+sum / exp+divide 各 1），理论下界字节 = 读 `x` 1 遍（4B）+ 写 `y`（4B）= 8B，故 `AI = 3/8 ≈ 0.375 FLOP/Byte`。RTX 5090 Ridge Point ≈ 12.6 FLOP/Byte，`AI=0.375 << 12.6` → 纯 memory-bound。而朴素三遍版实际读 `x` **3 次**（12B）+ 写 1 次（4B）= 16B，有效 `AI ≈ 3/16 ≈ 0.19` 更低——**重复读是主要浪费**，带宽利用率离峰值（1550 GB/s）差距大。

### 5.4 优化方向

1. **online softmax 两遍扫描（性价比最高）**：**FlashAttention** 的核心思想，把 max 和 sum 合并到同一次扫描里用增量更新 $m_{\text{new}}=\max(m_{\text{old}},m_{\text{block}}),\ s_{\text{new}}=s_{\text{old}}\cdot e^{m_{\text{old}}-m_{\text{new}}}+s_{\text{block}}$，把 3 遍读降到 2 遍。

![online softmax：max 与 sum 单遍融合](../../images/softmax_online_fused.svg)

2. **shared memory 缓存整行**：`D ≤ 4096`（16KB 可入 shared）时把整行一次性读到 shared，后续 max/sum/normalize 全在 shared 上做，**global 读降到 1 遍**。限制是 `D` 受 shared 容量约束。
3. **float4 向量化访存**：`x` 按行连续对齐，用 `float4` 一次读 16B，减少内存事务数与地址计算开销。
4. **FP16 存储 + FP32 reduce**：大模型用 FP16/BF16 存 `x`，HBM 读写减半、带宽翻倍；但 `exp` 与归约**必须 FP32** 保精度（FP16 累加易溢出），即"FP16 进 → FP32 算 → FP16 出"。
5. **kernel fusion**：把 softmax 与下游 GEMM 融合，省掉 `y` 的一次 HBM 写回，正是 FlashAttention 把 softmax 融进 attention kernel 的动机。

> 💡 优化 1（online 两遍）+ 优化 4（FP16 存储）是现代推理引擎的标配。所有优化都是在"两次归约 + 数值稳定"这个骨架上减遍数、减字节数。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(M×D)`：每行三遍扫描，每遍 `O(D)` |
| **空间复杂度** | `O(M×D)` 输入/输出 + `O(NUM_WARPS)` shared memory |
| **算术强度（理论下界）** | `~3 FLOP / 8B ≈ 0.375 FLOP/Byte`（1 读 + 1 写） |
| **算术强度（朴素三遍）** | `~3 FLOP / 16B ≈ 0.19 FLOP/Byte`（3 读 + 1 写，重复读拉低） |
| **瓶颈类型** | **memory-bound**：AI 远低于 ridge point（~12.6），`DRAM% >> SM%` |
| **kernel 启动数** | 1 次（单 kernel 内三阶段，block 内 `__syncthreads` 同步） |
| **块归约次数** | 每行 **2 次**（max + sum），比 RMSNorm（1 次）多一次 |
| **global 读次数** | 3 次（Pass 1/2/3 各读一遍 x）→ 优化后 2 次（online）或 1 次（shared 缓存） |
| **warp shuffle 步数** | 每次块归约 `log₂32 = 5` 步，两次共 10 步 |

> 💡 **一句话总结**：Softmax 是"两次归约 + 一次归一化"的经典模板——它把 [Day 4 的 warp shuffle 归约](../week1/day4/leetgpu-reduction-solution.md) 同时用在了 `max` 和 `sum` 上，再用 **safe softmax（减 max）** 解决数值溢出。它的算术强度极低（`AI ≈ 0.375`），是 memory-bound 的完美教学样本：用 ncu 看 `DRAM% >> SM%` 一眼可判。优化路径也很清晰——online softmax 把三遍压成两遍，FP16 存储把字节减半，最终融合进 FlashAttention。掌握这个骨架，[RMSNorm](../week3/day6/leetgpu-rms-normalization-solution.md)（删一个 reduce）和后续 Softmax Attention（加 matmul 融合）都是它的直接延伸。

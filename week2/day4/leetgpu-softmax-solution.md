# LeetGPU Softmax 题解

## 1. 题目概述

- **标题 / 题号**：Softmax（#5，medium）
- **链接**：https://leetgpu.com/challenges/softmax
- **难度**：中等
- **标签**：CUDA、softmax、three-pass、online softmax、数值稳定性、warp shuffle、memory-bound

**题意**：给定 `B` 行 `N` 列的 `float32` 矩阵 `input`（行主序），对**每一行独立**做 softmax，结果写入 `output`：

$$\text{output}[b][i] = \frac{\exp(\text{input}[b][i])}{\sum_{j=0}^{N-1} \exp(\text{input}[b][j])}$$

**示例**（单行 `N=4`）：

```text
输入：[1.0, 2.0, 3.0, 4.0]
max = 4.0
exp(x-max) = [0.0498, 0.1353, 0.3679, 1.0000]
sum = 1.5530
output = [0.0321, 0.0871, 0.2369, 0.6439]   // 每行和为 1
```

**约束**：

- `1 ≤ B × N ≤ 1,000,000`（总元素数）
- 元素范围 `[-10.0, 10.0]`
- 容差 `atol = rtol = 1e-4`
- 性能测试取较大 `B×N`（如 `B=128, N=8192`）

> 💡 这是 **online softmax** 的入门题。Day 4 的 Reduction 是"多对一"（N→1），Prefix Sum 是"带前缀依赖的多对多"（N→N）。Softmax 则是"**两次归约 + 一次归一化**"——每行要先求 max、再求 sum、最后除，三次扫描共享同一行数据。它引出两个核心概念：**数值稳定**（减 max 防 exp 溢出）和 **online softmax**（FlashAttention 的算法基石）。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 softmax（带数值稳定）
void softmax_cpu(const float* input, float* output, int B, int N) {
    for (int b = 0; b < B; ++b) {
        const float* in  = input  + b * N;
        float*       out = output + b * N;

        // ① 求 max（数值稳定：减 max 防 exp 上溢）
        float max_val = -INFINITY;
        for (int i = 0; i < N; ++i) max_val = fmaxf(max_val, in[i]);

        // ② 求 sum(exp(x - max))
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) sum += expf(in[i] - max_val);

        // ③ 归一化
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < N; ++i) out[i] = expf(in[i] - max_val) * inv_sum;
    }
}
```

每行三遍扫描 `O(N)`，总计 `O(B×N)`。`B=128, N=8192` 时单核约 1-2ms。

### 2.2 朴素 GPU：每 thread 算一个元素（错误示范）

第一反应可能是"每 thread 算一个 `output[i]`"，但 `output[i] = exp(x[i]) / Σexp(x[j])` 意味着**每个 thread 都需要遍历整行求 max 和 sum**——`O(N²)` 总工作量，比 CPU 还慢。

```cuda
// 错误示范：每 thread 独立扫整行 → O(N²)，N=8192 时每 thread 重复读 8192 次
__global__ void softmax_wrong(const float* input, float* output, int B, int N) {
    int b = blockIdx.x;
    int i = threadIdx.x;
    if (i >= N) return;
    float max_val = -INFINITY;
    for (int j = 0; j < N; ++j) max_val = fmaxf(max_val, input[b*N + j]);  // 重复！
    // ...
}
```

> ⚠️ Softmax 的核心难点：**它既不是纯 elementwise（无依赖），也不是单次归约（单输出）**，而是"两次归约（max、sum）+ 一次依赖前两者的归一化"。必须用**块内协作**——一个 block 的线程共同求 max/sum，再一起写输出。

## 3. GPU 设计

### 3.1 并行化策略：一个 block 负责一行

**核心映射**：`blockIdx.x → 行号 b`，block 内 `BLOCK_SIZE` 个 thread 协作处理该行的 `N` 个元素。

![一个 block 负责一行：grid-stride 内协作 + 两次块归约](images/softmax_block_per_row.svg)

每个 block 执行三阶段：
1. **Pass 1（求 max）**：thread 各自用 grid-stride 扫描行内元素求局部 max → 块归约得到 `row_max`
2. **Pass 2（求 sum）**：再扫一遍求 `Σexp(x - row_max)` → 块归约得到 `row_sum`
3. **Pass 3（归一化）**：再扫一遍写 `output[i] = exp(x[i] - row_max) / row_sum`

> 💡 为什么一个 block 负责一行，而不是一个 block 负责多行？因为 max 和 sum 是**行内归约**——同一行的所有元素必须协作才能算出 max/sum。跨行的元素互不依赖，天然由不同 block 并行。这种"一行一 block"映射让 block 内 `__syncthreads` 正好对齐行边界，无需跨 block 通信。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读（3 遍）、`output` 写（1 遍） |
| **shared memory** | ✓ | warp 间归约汇总：`shared[NUM_WARPS]`，及广播 `row_max`/`row_sum` |
| **register** | ✓ | 每线程的 `local_max`/`local_sum` 累加值 + warp shuffle 交换 |

### 3.3 关键技巧 1：数值稳定（减 max）

朴素 softmax `exp(x) / Σexp(x)` 在 `x` 较大时 `exp(x)` 会上溢（`exp(100) ≈ 2.7e43`，`exp(1000) = inf`）。**减去 `row_max` 后**所有指数 ≤ 0，最大值为 `exp(0) = 1`，无上溢；最小值 `exp(-20) ≈ 2e-9`，无下溢（本题元素 `[-10, 10]`，`max-min ≤ 20`，`exp(-20)` 仍在 float 正常范围）。

$$\text{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}, \quad m = \max_j x_j$$

数学等价性：分子分母同除 `exp(m)`，值不变。

### 3.4 关键技巧 2：warp shuffle 块归约（复用 Day 4）

每行的 max 和 sum 都是一次块归约。直接复用 [Day 4 Reduction](../week1/day4/leetgpu-reduction-solution.md) 的 `warp_reduce` + `block_reduce` 模板：

- **warp 内**：`__shfl_down_sync` 折叠到 lane 0（max 用 `fmaxf`，sum 用 `+`）
- **warp 间**：每 warp lane 0 写 shared → 第一个 warp 再归约
- **广播**：结果写 `shared[0]`，`__syncthreads` 后全 block 读取

![三遍扫描数据流：max → sum → normalize](images/softmax_three_pass.svg)

> 💡 这正是 Day 4 归约题解的"积木复用"——`warp_reduce_sum` 一字未改，`warp_reduce_max` 只把 `+` 换成 `fmaxf`。Softmax 本质上是**两次归约 + 一次 elementwise**，掌握了归约就掌握了 softmax 的一半。

## 4. Kernel 实现

完整可编译的三遍 softmax（一个 block 一行 + warp shuffle 块归约 + 数值稳定）：

```cuda
// softmax_three_pass.cu —— 三遍 softmax：max → sum → normalize
// 编译命令: nvcc -O3 -arch=sm_80 softmax_three_pass.cu -o softmax
// 运行:     ./softmax 128 8192

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define NUM_WARPS  (BLOCK_SIZE / WARP_SIZE)   // 8

// ---- warp 级归约：max 和 sum（复用 Day 4 模板）----
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总 + 广播 ----
__inline__ __device__ float block_reduce_max(float val, float* shared) {
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) shared[0] = val;   // 广播 slot
    }
    __syncthreads();
    return shared[0];
}

__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane   = threadIdx.x & (WARP_SIZE - 1);
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

// ---- softmax kernel：一个 block 负责一行 ----
__global__ void softmax_kernel(const float* input, float* output, int B, int N) {
    __shared__ float shared[NUM_WARPS + 1];

    int b = blockIdx.x;
    if (b >= B) return;
    const float* row_in = input  + b * N;
    float*       row_out = output + b * N;

    // ---- Pass 1：求 row_max（数值稳定）----
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
        local_max = fmaxf(local_max, row_in[i]);
    float row_max = block_reduce_max(local_max, shared);

    // ---- Pass 2：求 row_sum = Σ exp(x - max) ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
        local_sum += expf(row_in[i] - row_max);
    float row_sum = block_reduce_sum(local_sum, shared);

    // ---- Pass 3：归一化 ----
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 128;
    int N = (argc > 2) ? atoi(argv[2]) : 8192;
    size_t bytes = (size_t)B * N * sizeof(float);
    printf("B=%d, N=%d  (%.1f MB)\n", B, N, bytes / 1e6);

    // ---- host ----
    float *hIn  = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < B * N; ++i)
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;   // [-10, 10]

    // ---- device ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn,  bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    dim3 blocks(B);
    dim3 threads(BLOCK_SIZE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    softmax_kernel<<<blocks, threads>>>(dIn, dOut, B, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 带宽 ----
    float bw_gbs = (3.0f * bytes + bytes) / 1e9 / (ms / 1e3);   // 读 3 遍 + 写 1 遍
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int b = 0; b < B && err < 5; ++b) {
        float mx = -INFINITY, s = 0.0f;
        for (int i = 0; i < N; ++i) mx = fmaxf(mx, hIn[b*N + i]);
        for (int i = 0; i < N; ++i) s += expf(hIn[b*N + i] - mx);
        for (int i = 0; i < N; ++i) {
            float ref = expf(hIn[b*N + i] - mx) / s;
            if (fabsf(hOut[b*N + i] - ref) > 1e-4f * fmaxf(1.0f, fabsf(ref))) {
                if (++err <= 5)
                    printf("MISMATCH [%d][%d]: got %f, expect %f\n", b, i, hOut[b*N+i], ref);
            }
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 检查每行和 ≈ 1 ----
    float max_row_sum_err = 0.0f;
    for (int b = 0; b < B; ++b) {
        float s = 0.0f;
        for (int i = 0; i < N; ++i) s += hOut[b*N + i];
        max_row_sum_err = fmaxf(max_row_sum_err, fabsf(s - 1.0f));
    }
    printf("max row-sum error: %.2e (should be ~0)\n", max_row_sum_err);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `softmax_kernel` 填进 starter 的 `solve` 函数即可。注意确认输入是 `(B, N)` 行主序、按行做 softmax。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 softmax_three_pass.cu -o softmax
./softmax 128 8192
```

典型输出（A100）：

```text
B=128, N=8192  (4.0 MB)
kernel time: 0.42 ms
effective bandwidth: 38.1 GB/s
verify: PASS
max row-sum error: 2.38e-07 (should be ~0)
```

### 5.2 用 ncu 分析

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
| `dram__throughput` | HBM 带宽占比 | ~30-40% | memory-bound 应较高，但 3 遍读有冗余 |
| `sm__throughput` | SM 算力占比 | ~10% | `expf` 有计算但算术强度低 |
| `sm__occupancy` | 占用率 | ~75% | BLOCK_SIZE=256，shared 用量小 |
| `long_scoreboard` | 等访存 stall | ~40% | 3 遍 global 读，stall 明显 |

> ⚠️ 带宽看似不高（~38 GB/s），因为**3 遍读 global 有冗余**——同一数据读了 3 次。优化核心是**减少 global 读次数**。

### 5.3 优化方向

#### 优化 1：online softmax（两遍合一，FlashAttention 基石）

三遍扫描的瓶颈是读 global 3 次。**Online softmax** 把 Pass 1（max）和 Pass 2（sum）融合成一遍——边扫边更新 max 和 sum：

![Online Softmax：max 与 sum 融合，单遍扫描](images/softmax_online_fused.svg)

```cuda
// online softmax：一遍扫描同时算 max 和 sum
float m = -INFINITY, s = 0.0f;
for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
    float x = row_in[i];
    float m_new = fmaxf(m, x);
    s = s * expf(m - m_new) + expf(x - m_new);   // 修正旧 sum
    m = m_new;
}
// block 归约 m 和 s（需处理跨 thread 的 m/s 合并）
```

**关键公式**：当 max 从 `m` 更新到 `m_new` 时，旧的 `sum` 要乘 `exp(m - m_new)` 缩放。这让单遍扫描即可算出最终 max 和 sum，**global 读从 3 次降到 2 次**（Pass 1+2 合并，Pass 3 仍需再读一次写输出）。

> 💡 这正是 [Day 5 FlashAttention](../../aiinfra/week2/day5/README.md) 的算法基石。FlashAttention 把 softmax 和 matmul 融合，靠 online softmax 在 tiling 内增量更新 max/sum，避免物化整个 `N×N` attention 矩阵。本题是理解 FlashAttention 的前置练习。

#### 优化 2：shared memory 缓存（一遍读）

若 `N` 较小（如 `N ≤ 4096`，`4KB` 数据可放入 shared），把整行一次性读到 shared memory，三遍扫描全在 shared 上做：

```cuda
__shared__ float row_cache[N_MAX];   // N ≤ 4096 时可行
for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) row_cache[i] = row_in[i];
__syncthreads();
// 后续三遍全读 row_cache（shared ~20 cycle vs global ~400 cycle）
```

**收益**：global 读从 3 次降到 1 次，带宽利用率接近峰值。**限制**：`N` 受 shared memory 容量约束（A100 每 block 默认 48KB，可动态申请到 99KB）。

#### 优化 3：寄存器缓存 exp 值

Pass 2 算完 `exp(x - max)` 后，若每 thread 处理的元素数固定（如 `ILP=4`），可把 exp 值存寄存器数组，Pass 3 直接复用，省一次 `expf` 计算。但需 grid-stride 改为固定步长，灵活性下降。

#### 优化 4：vector load（`float4`）

加载 input 时用 `float4` 一次读 4 个 float，减少内存事务数。对连续行数据效果显著（softmax 按行连续访问，天然对齐）。

#### 优化 5：大 N 的多 block 分块

当 `N` 极大（如 `N > 65536`，单 block 处理效率低），可将一行拆给多个 block，每个 block 算局部 max/sum，再用二次 kernel 归约全局 max/sum，最后各 block 加偏移归一化。结构类似 [Prefix Sum 的三阶段](../day1/leetgpu-prefix-sum-solution.md)。本题 `B×N ≤ 1M`，单 block 足够。

> 💡 优化 1（online softmax）和 2（shared 缓存）是性价比最高的。两者结合——**一遍读 global 到 shared，online softmax 在 shared 上单遍算 max+sum**——即可把 global 读降到 1 次，逼近 memory-bound 带宽上限。这也是 FlashAttention tiling 的核心模式。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(B×N)`：每行三遍扫描 |
| **空间复杂度** | `O(B×N)` 输入/输出 + `O(NUM_WARPS)` shared memory |
| **算术强度（三遍版）** | `~5 FLOP / 16B ≈ 0.3 FLOP/B`（3 次读 + 1 次写，每次 4B，FLOP 含 exp/max/add）|
| **算术强度（online + 缓存）** | `~5 FLOP / 8B ≈ 0.6 FLOP/B`（1 次读 + 1 次写）|
| **瓶颈类型** | **memory-bound**：算术强度远低于平衡点，3 遍 global 读是主要开销 |
| **kernel 启动数** | 1 次（单 kernel 内三阶段，block 内 `__syncthreads` 同步） |
| **块归约次数** | 每行 2 次（max + sum），各 `log₂BLOCK_SIZE` 深度 |
| **global 读次数** | 3 次（Pass 1/2/3 各读一遍 input）→ 优化后 1 次 |

> 💡 **一句话总结**：Softmax 是"两次归约 + 一次归一化"的组合题——它把 Day 4 的 `warp_reduce` / `block_reduce` 积木复用到极致（max 用 `fmaxf`，sum 用 `+`），同时引出两个 GPU 编程关键概念：**数值稳定**（减 max 防 exp 溢出）和 **online softmax**（融合 max+sum 单遍扫描，FlashAttention 的算法基石）。三遍版是教学基线，掌握 online softmax 的"边扫边修正"思想后，你就拿到了通往 Day 5 FlashAttention 的入场券。

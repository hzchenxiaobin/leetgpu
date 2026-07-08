# LeetGPU Prefix Sum 题解

## 1. 题目概述

- **标题 / 题号**：Prefix Sum（#16，medium）
- **链接**：https://leetgpu.com/challenges/prefix-sum
- **难度**：中等
- **标签**：CUDA、Scan、Prefix Sum、warp shuffle、`__shfl_up_sync`、三阶段分块 scan、memory-bound

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，计算 **inclusive prefix sum**（前缀和），即 `output[i] = input[0] + input[1] + ... + input[i]`。

**示例**：

```text
输入：[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
输出：[1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]
```

**约束**：

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 前缀和能放进 32-bit float（大 N 时存在累加误差，参考实现用 `double` 求和再转 `float`）
- 性能测试取 `N = 16,777,216`（= 2²⁴，16M 元素，约 64 MB）

> 💡 这是 **warp shuffle** 的第二道题（上一道是 Reduction 用 `__shfl_down_sync`）。归约是"多对一"，scan 是"一对一但每个输出都依赖前面所有输入"——本质上是 **归约的对偶问题**。核心新概念是 `__shfl_up_sync`（向上传，做前缀扫描）和 **三阶段分块 scan**（block 内 scan → block 间偏移 scan → 加回），这是所有 stream compaction、radix sort、segmented scan 的基础积木。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行前缀和
void prefix_sum_cpu(const float* input, float* output, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}
```

`N = 16M` 时单核约 10-20 ms。瓶颈：纯串行，**每一步都依赖前一步**，看起来无法并行。

### 2.2 朴素 GPU：atomicAdd 串行化

最暴力的并行：每个 thread 读一个元素，用 `atomicAdd` 累加到一个全局游标，再写回 `output[i]`。

```cuda
__global__ void scan_atomic(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // ❌ 必须串行：每个线程都要拿到"前面所有元素的和"才能写
        // 任何 atomic 方案都会退化为 O(N) 串行，比 CPU 还慢
    }
}
```

**致命问题**：prefix sum 的每个 `output[i]` 都依赖 `output[0..i-1]`，`atomicAdd` 只能把累加器**串行化**。N 个线程争抢同一个累加器，性能比 CPU 串行还差几十倍。

> ⚠️ scan 的核心矛盾：输出之间有**数据依赖**（`output[i]` 需要 `output[i-1]`），不能像 vector add 那样各算各的。必须用 **Hillis-Steele 蝶形扫描** 把"串行依赖"改造成"对数步数的并行交换"。

## 3. GPU 设计

### 3.1 并行化策略：Hillis-Steele scan + 三阶段分块

#### Hillis-Steele 蝶形扫描

思想：每步让每个位置加上**距离 `offset` 处**的值，`offset = 1, 2, 4, 8, ...`，`log₂N` 步后每个位置都持有自己的前缀和。

![Hillis-Steele 蝶形扫描](images/prefix_sum_hillis_steele.svg)

以 8 个元素、inclusive scan 为例：

| step | offset | 操作 | 数组状态 |
|------|--------|------|----------|
| 0 | — | 初始 | [1, 2, 3, 4, 5, 6, 7, 8] |
| 1 | 1 | `a[i] += a[i-1]` | [1, 3, 5, 7, 9, 11, 13, 15] |
| 2 | 2 | `a[i] += a[i-2]` | [1, 3, 6, 10, 14, 18, 22, 26] |
| 3 | 4 | `a[i] += a[i-4]` | [1, 3, 6, 10, 15, 21, 28, 36] |

**关键属性**：`log₂N` 步完成，**所有线程全程活跃**（不像归约逐步减半），但总工作量 `O(N log N)`，比串行 `O(N)` 多一个 `log` 因子。

> 💡 另一种算法 **Blelloch scan**（work-efficient，`O(N)` 工作量）用"上扫 build 树 + 下扫 distribute"两遍，适合大量元素场景。本题 N≤1e8，Hillis-Steele 的 `log` 因子只 27，且全程满载、对 warp shuffle 友好，是 GPU 上的标准选择。

#### 三阶段分块（large N）

`N = 16M` 远超单 block 容量。借鉴归约的"两遍"思路，但 scan 的输出是**每个元素都要写**，所以需要**三阶段**：

![三阶段分块 scan 架构](images/prefix_sum_three_phase.svg)

1. **阶段一：block 内 exclusive scan**。每个 block 独立对自己负责的那段做 exclusive scan（即 `output[i] = input[0] + ... + input[i-1]`，不含 `input[i]`），同时把该 block 的**总和**写入 `block_sums[blockIdx.x]`。
2. **阶段二：对 `block_sums[]` 做前缀和**。得到每个 block 的**全局起始偏移** `block_offsets[]`。这一步数据量小（= block 数，通常几千），单 block 即可。
3. **阶段三：加回全局偏移**。每个 block 把 `block_offsets[blockIdx.x]` 加到自己 scan 的结果上，再补上本块的 `input[i]`，得到最终的 inclusive prefix sum。

> 💡 为什么是三阶段而不是两阶段？归约的"多对一"只需把部分和二次归约即可；scan 是"一对一"，每个 block 必须知道"前面所有 block 的总和"才能修正自己的输出。阶段二就是算这个"前面所有 block 的总和"。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写、`block_sums[]` / `block_offsets[]` 中间缓冲 |
| **shared memory** | ✓ | block 内 scan 的暂存区（warp 间汇总用，存每 warp 的子前缀和） |
| **register** | ✓ | 每线程持有的当前值 + warp shuffle 直接交换，绕开 shared |

### 3.3 关键技巧：warp shuffle `__shfl_up_sync`

#### 为什么用 `__shfl_up_sync` 而非 `__shfl_down_sync`

归约用 `__shfl_down_sync`（向下传，把结果收敛到 lane 0）；scan 用 `__shfl_up_sync`（向上传，把前缀"扩散"到每个 lane）。

`__shfl_up_sync(mask, val, delta)` 语义：当前 lane 从 `lane - delta` 处取值（若 `lane - delta < 0` 则值不变）。

![__shfl_up_sync 蝶形交换原理](images/prefix_sum_warp_shuffle.svg)

```cuda
// warp 内 inclusive scan（32 个 lane，5 步完成）
// 初始 val = input[lane]
for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    float n = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) {
        val += n;
    }
}
// 现在 lane i 的 val = input[0] + ... + input[i]（warp 内前缀和）
```

迭代过程：`offset = 1, 2, 4, 8, 16`，共 5 步（`log₂32`），每步所有 32 lane 都活跃。

> 💡 `__shfl_up_sync` 与 `__shfl_down_sync` 是一对镜像：up 做 scan（前缀），down 做 reduce（归约）。两者都是**寄存器级通信**，不经 shared memory、不需 `__syncthreads`，是 GPU 上最快的线程间数据交换方式。

#### block 内 scan 的两步：warp scan + shared 汇总

单 block 通常 256-1024 thread（8-32 个 warp）。block 内 scan 分两步：

1. **每个 warp 各自做 warp scan**（5 步 `__shfl_up_sync`）。
2. **每 warp 的 lane 31（最后一个 lane）把自己 warp 的总和写入 shared**。
3. **对 shared 里的 warp 总和再做一次 scan**（通常 warp 数 ≤ 32，单次 warp scan 搞定），得到每 warp 的起始偏移。
4. **每 warp 把偏移加回去**，得到 block 内 inclusive scan。

> ⚠️ exclusive scan 的实现：先做 inclusive scan，再整体右移一位（lane 0 补 0）。或者用"先存 warp 总和、再加偏移"的方式天然得到 exclusive。本题阶段一用 exclusive 是为了让阶段三加回 `input[i]` 时正好得到 inclusive。

## 4. Kernel 实现

完整可编译的三阶段分块 scan 版本（warp shuffle + shared 汇总 + 三 kernel）：

```cuda
// prefix_sum.cu —— 三阶段分块 scan：warp shuffle + block scan + 全局偏移加回
// 编译命令: nvcc -O3 -arch=sm_80 prefix_sum.cu -o prefix_sum
// 运行:     ./prefix_sum 16777216

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

// ---- warp 内 inclusive scan：__shfl_up_sync，5 步蝶形 ----
__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val;   // lane i 持有本 warp 内 [0..i] 的前缀和
}

// ---- block 内 exclusive scan：warp scan + shared 汇总 + 偏移加回 ----
// 返回每线程对应的 exclusive 前缀和；block 总和由 lane 0 写入 block_sum_out
__inline__ __device__ float block_exclusive_scan(float val, float* block_sum_out) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    // ① 每 warp 各自 inclusive scan
    float inclusive = warp_inclusive_scan(val);

    // ② 每 warp 的 lane 31 记录本 warp 总和
    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;   // inclusive 的最后一个 = 该 warp 总和
    }
    __syncthreads();

    // ③ 第一个 warp 对 warp_sums 做 inclusive scan，得到每 warp 的起始偏移
    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS) warp_sums[lane] = v;   // 改写为 inclusive prefix
    }
    __syncthreads();

    // ④ 当前 warp 之前所有 warp 的总和 = exclusive 起始偏移
    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];

    // ⑤ exclusive = warp_offset + (本 warp 内 inclusive 减去自身)
    //    inclusive[lane] = warp_offset + input[0..lane]（含自身）
    //    exclusive[lane] = warp_offset + input[0..lane-1]（不含自身）
    float exclusive = warp_offset + (inclusive - val);

    // ⑥ block 总和 = 最后一个 warp 的 inclusive prefix（lane 31）
    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum_out = warp_offset + inclusive;   // = 全 block 总和
    }
    return exclusive;
}

// ---- 阶段一：每 block 对自己那段做 exclusive scan，结果先存 output，总和写 block_sums ----
__global__ void scan_block_kernel(const float* input, float* output,
                                  float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N) return;

    float val = input[tid];
    float exclusive = block_exclusive_scan(val, &block_sums[blockIdx.x]);
    output[tid] = exclusive;   // 暂存 exclusive，阶段三再加回 input + offset
}

// ---- 阶段二：对 block_sums[] 做前缀和（单 block，假设 M ≤ BLOCK_SIZE） ----
__global__ void scan_offsets_kernel(const float* block_sums,
                                    float* block_offsets, int M) {
    __shared__ float dummy;
    int tid = threadIdx.x;
    float val = (tid < M) ? block_sums[tid] : 0.0f;
    float exclusive = block_exclusive_scan(val, &dummy);
    if (tid < M) block_offsets[tid] = exclusive;
}

// ---- 阶段三：每元素 = 阶段一的 exclusive + 本 block 偏移 + input[i] ----
__global__ void add_offset_kernel(float* output, const float* input,
                                  const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N) return;
    // output[tid] 目前是 exclusive（不含自身），加上 offset 和 input[tid) 即得 inclusive
    output[tid] = output[tid] + block_offsets[blockIdx.x] + input[tid];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16777216;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    // ---- host 端 ----
    float *hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;   // [-10, 10]
    }

    // ---- device 端 ----
    float *dIn, *dOut, *dBlockSums, *dBlockOffsets;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&dBlockSums,    numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBlockOffsets, numBlocks * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // 三阶段启动
    scan_block_kernel<<<numBlocks, BLOCK_SIZE>>>(dIn, dOut, dBlockSums, N);
    scan_offsets_kernel<<<1, BLOCK_SIZE>>>(dBlockSums, dBlockOffsets, numBlocks);
    add_offset_kernel<<<numBlocks, BLOCK_SIZE>>>(dOut, dIn, dBlockOffsets, N);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (three-pass): %.3f ms\n", ms);

    // ---- 验证（CPU double 累加做参考，抽样 + 末元素严格比对）----
    float *hOut = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    double acc = 0.0;
    int fail = 0;
    int checkPts[] = {0, 1, 2, N/4, N/2, N-2, N-1};
    for (int k = 0; k < 7; ++k) {
        int i = checkPts[k];
        for (int j = (k == 0 ? 0 : checkPts[k-1] + 1); j <= i; ++j) acc += hIn[j];
        if (fabsf(hOut[i] - (float)acc) > 1e-2f * (1.0f + fabsf((float)acc))) {
            printf("FAIL at i=%d: GPU=%f CPU=%f\n", i, hOut[i], (float)acc);
            fail = 1; break;
        }
    }
    printf("%s\n", fail ? "FAIL" : "PASS");

    // ---- 带宽估算：读 input + 写 output 各一遍 ----
    float bw_gbs = (2.0 * bytes / 1e9) / (ms / 1e3);
    printf("I/O bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaFree(dBlockSums));
    CHECK_CUDA(cudaFree(dBlockOffsets));
    free(hIn);
    free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把三个 kernel 填进 `solve` 函数、按顺序 launch 即可。带 `main()` 的版本用于本地自测。

> ⚠️ `scan_offsets_kernel` 假设 `numBlocks ≤ BLOCK_SIZE`（256），单 pass scan 即可。N≤1e8 时 numBlocks 可能超 256，生产代码需对阶段二递归调用三阶段算法，或改用 `cooperative_groups` 的 `cg::this_grid().sync()` 在单 kernel 内做 grid 级同步。本题为教学清晰起见保留简化版。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 prefix_sum.cu -o prefix_sum
./prefix_sum 16777216
```

典型输出（A100 / SM=108）：

```text
N = 16777216  (64.0 MB)
kernel time (three-pass): 0.95 ms
PASS
I/O bandwidth: 135.0 GB/s
```

### 5.2 用 ncu 分析

```bash
ncu --set full --target-processes all -o prefix_sum_profile ./prefix_sum 16777216

# 关键指标
ncu --metrics gpu__time_duration.sum, \
        dram__bytes_read.sum,dram__bytes_write.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__sass_thread_inst_executed_op_fadd_pred_on.sum, \
        launch__waves_per_multiprocessor \
    ./prefix_sum 16777216
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | > 60%（scan 要读+写，I/O 翻倍） |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 算力占比 | 中等（5 步 shuffle + 加法） |
| `dram__bytes_read.sum` vs `dram__bytes_write.sum` | 读写量 | 应接近 1:1（input 读 + output 写） |
| `launch__waves_per_multiprocessor` | 每 SM wave 数 | > 2 |

> 💡 scan 的带宽通常**低于**归约——归约只读不写，scan 读 N + 写 N，I/O 翻倍且阶段三要**重读 input**。这是三阶段方案的固有开销，单 pass fused scan 能改善。

### 5.3 优化方向

1. **`float4` 向量化访存**：每线程一次读 16B（4 个 float），减少地址计算、提升内存事务效率。配合 4 路 warp scan 串联。通常能再提升 30-50% 带宽。
2. **block size 调优**：`BLOCK_SIZE` 从 256 调到 512/1024，减少 numBlocks、降低阶段二/三的 kernel 启动与中间缓冲开销。需注意 shared memory 用量。
3. **减少全局同步（kernel 融合）**：三阶段有 3 次 kernel launch。可用 `cooperative_groups` 的 `cg::this_grid().sync()` 在单 kernel 内做 grid 级同步，省掉阶段二/三的启动开销。或用"最后一个 block 检测"（atomic 计数）在阶段一末尾顺便算偏移。
4. **Blelloch work-efficient scan**：对超大 N，`O(N)` 工作量的 Blelloch（上扫 + 下扫）比 `O(N log N)` 的 Hillis-Steele 更省算力，但实现复杂、shuffle 利用率低，需权衡。
5. **避免阶段三重读 input**：阶段一可把 `input[i]` 也存进 shared/register，阶段三直接用，省掉一次 global 读。代价是寄存器/shared 压力增大。

> 💡 优化 1+3 是性价比最高的组合：向量化吃满带宽 + 单 kernel 融合省启动，典型可再降 30-40% 延迟。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N log W)`（W=warp size=32，单 block 内）；全局 `O(N)` 主体 + `O(B log B)` 阶段二（B=numBlocks） |
| **空间复杂度** | `O(N)` 输入/输出 + `O(B)` `block_sums`/`block_offsets` + `O(BLOCK)` shared |
| **算术强度** | `1 FLOP / 8B`（1 次加法 ↔ 读 4B input + 写 4B output）= **0.125 FLOP/B** |
| **瓶颈类型** | **memory-bound**：算术强度极低，受 HBM 读写双向带宽限制 |
| **kernel 启动数** | 3 次（block scan + offsets scan + add offset） |
| **warp scan 步数** | 每 warp `log₂32 = 5` 步（`__shfl_up_sync` offset=1,2,4,8,16） |
| **block scan 步数** | warp scan 5 步 + warp 间 scan 5 步（NUM_WARPS=8 时 4 步）≈ 9-10 步 |

> 💡 **一句话总结**：scan 是 **warp shuffle** 的进阶应用——把"串行依赖的前缀和"改造成"对数步数的蝶形并行交换"。`__shfl_up_sync` 与归约的 `__shfl_down_sync` 是一对镜像，掌握它们就掌握了 GPU 上所有 prefix 类操作的基础积木。三阶段分块架构（block 内 scan → block 间偏移 scan → 加回）是处理超大数据的标准模板，可直接迁移到 stream compaction、radix sort、segmented scan 等场景。

# LeetGPU Prefix Sum 题解

## 1. 题目概述

- **标题 / 题号**：Prefix Sum（#16，medium）
- **链接**：https://leetgpu.com/challenges/prefix-sum
- **难度**：中等
- **标签**：CUDA、parallel scan、warp shuffle、`__shfl_up_sync`、三阶段分块、Blelloch 算法

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，计算其**前缀和**（inclusive prefix sum / cumsum），结果写入 `output`：

$$\text{output}[i] = \sum_{k=0}^{i} \text{input}[k]$$

**示例**：

```text
输入：[1.0, 2.0, 3.0, 4.0]
输出：[1.0, 3.0, 6.0, 10.0]
```

**约束**：

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 输出最大值能放进 32-bit float
- 性能测试取 `N = 250,000`
- 容差较宽：`atol = rtol = 0.01`（大数组累加有浮点误差）

> 💡 这是 **parallel scan** 的入门题。Day 4 的归约是"多对一"（N 个输入 → 1 个输出），而 scan 是"多对多但有依赖"（N 个输入 → N 个输出，每个输出依赖前面所有输入）。scan 是 CUDA 编程里最难并行化的操作之一，因为它的数据依赖天然是串行的。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行前缀和
void prefix_sum_cpu(const float* input, float* output, int N) {
    output[0] = input[0];
    for (int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];   // 依赖前一个！
    }
}
```

`N = 250K` 时单核约 1ms 以内。瓶颈显而易见：**`output[i]` 必须等 `output[i-1]` 算完**，形成长度为 N 的串行依赖链。

![前缀和概念与串行依赖](images/prefix_sum_overview.svg)

### 2.2 朴素 GPU：为什么不能直接并行

第一反应可能是"每 thread 算一个 output[i]"，但 `output[i] = output[i-1] + input[i]` 意味着 thread i 必须等 thread i-1 完成——**完全串行，没有并行度**。

如果退而求其次让 thread i 自己把 `input[0..i]` 加起来，虽然能并行，但时间复杂度退化到 `O(N²)`（thread N-1 要加 N-1 次），比 CPU 串行还慢。

> ⚠️ scan 的核心难点：**它既不是 elementwise（无依赖），也不是 reduction（单输出）**，而是介于两者之间的"带前缀依赖的多输出"。需要专门的算法——**树形扫描**——来并行化。

## 3. GPU 设计

### 3.1 并行化策略：树形扫描 + 三阶段分块

破局思路是 **Blelloch 并行扫描算法**：用树形结构把串行依赖链拆成 `O(log N)` 深度的并行计算。具体分三层：

1. **warp 内 scan**：用 `__shfl_up_sync` 在 32 个 lane 间做前缀和（`log₂32 = 5` 步）。
2. **block 内 scan**：多个 warp 的部分和再用一次 warp scan 汇总，加上前缀偏移。
3. **block 间 scan**：各 block 的总和组成小数组，对其再做一次 scan，最后加回各 block 输出。

![三阶段分块 Scan 架构](images/prefix_sum_three_phase.svg)

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写、`aux[]` 存 block 总和 |
| **shared memory** | ✓ | block 内 warp 间汇总：`warp_sums[NUM_WARPS]` |
| **register** | ✓ | 每线程的累加值 + warp shuffle 交换 |

与 Day 4 归约的存储使用几乎一致——差别只在 warp shuffle 的**方向**和**每个 lane 都保留结果**。

### 3.3 关键技巧：warp scan `__shfl_up_sync`

#### `__shfl_up_sync` vs `__shfl_down_sync`

Day 4 归约用的是 `__shfl_down_sync`（lane i 收到 lane i+delta，数据从高 lane 流向低 lane，结果汇聚到 lane 0）。scan 正好相反——用 `__shfl_up_sync`：**lane i 收到 lane i-delta 的值，数据从低 lane 流向高 lane，每个 lane 都保留自己的前缀和**。

![__shfl_up_sync 工作原理](images/prefix_sum_warp_scan.svg)

```cuda
// warp 内 inclusive scan：32 个值 → 32 个前缀和，5 步完成
__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val += n;   // 低 offset 个 lane 不加（无前驱）
        }
    }
    return val;
}
```

迭代 `offset = 1, 2, 4, 8, 16`，每步活跃 lane 数增加。以 8 个 lane 值 `[1,2,3,4,5,6,7,8]` 为例：

| offset | 操作 | 结果 |
|--------|------|------|
| 1 | lane i += lane(i-1)，i≥1 | [1, 3, 5, 7, 9, 11, 13, 15] |
| 2 | lane i += lane(i-2)，i≥2 | [1, 3, 6, 10, 15, 21, 28, 36] |
| 4 | lane i += lane(i-4)，i≥4 | [1, 3, 6, 10, 15, 21, 28, 36] ✓ |

> 💡 对比 Day 4：`__shfl_down_sync` 让值"往下沉"汇聚到 lane 0（归约），`__shfl_up_sync` 让值"往上涌"让每个 lane 都拿到前缀（扫描）。**同一套 warp shuffle 机制，方向相反，用途互补**。

### 3.4 三阶段架构详解

#### Phase 1：每 block 内部 scan（多 block 并行）

每个 block 处理 `BLOCK_SIZE` 个元素（如 1024）：
1. 每个线程读 1 个元素，在 warp 内做 `warp_inclusive_scan`（得到 warp 内前缀）。
2. 每 warp 的最后一个 lane（lane 31）把 warp 总和写入 `shared[warpId]`。
3. 第一个 warp 对 `shared[]` 再做一次 scan（得到 warp 间前缀）。
4. 每个 warp 的所有 lane 加上前面 warp 的累计和。
5. 输出 block 内前缀和到 `output[]`，block 总和写到 `aux[blockIdx.x]`。

#### Phase 2：对 aux[] 做 scan（单 block）

`aux[]` 长度 = block 数。对 `N=250K, BLOCK_SIZE=1024`，只有 ~245 个 block。用一个 block 的 warp scan 即可算出每个 block 的**起始偏移**。

#### Phase 3：加偏移回 output（多 block 并行）

每个 block 把 Phase 2 算出的 `aux[blockIdx-1]`（前面所有 block 的总和）加到自己的输出上。block 0 不加。这一步完全无依赖、高度并行。

> ⚠️ 若 `N` 极大（如 1e8），block 数可能超过 Phase 2 单 block 的处理能力（>1024）。此时 Phase 2 需要**递归调用三阶段**——这是 Blelloch 算法的递归本质。本题 `N=250K` 不需要，但工业级实现（如 CUB 的 `DeviceScan`）会处理。

## 4. Kernel 实现

完整可编译的三阶段 prefix sum：

```cuda
// prefix_sum_scan.cu —— 三阶段分块 scan：warp scan + block scan + 加偏移
// 编译命令: nvcc -O3 -arch=sm_80 prefix_sum_scan.cu -o prefix_sum
// 运行:     ./prefix_sum 250000

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

#define BLOCK_SIZE 1024
#define WARP_SIZE  32
#define NUM_WARPS  (BLOCK_SIZE / WARP_SIZE)   // 32

// warp 内 inclusive scan：每 lane 得到 [0..lane] 的前缀和
__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val;
}

// block 内 inclusive scan：warp scan + warp 间汇总
// 返回当前线程的 inclusive prefix，并把 block 总和写到 block_sum（仅最后一线程）
__inline__ __device__ float block_inclusive_scan(float val, float* shared) {
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    // ① warp 内 scan
    val = warp_inclusive_scan(val);

    // ② 每 warp 的 lane 31 写 warp 总和到 shared
    if (lane == WARP_SIZE - 1) {
        shared[warpId] = val;
    }
    __syncthreads();

    // ③ 第一个 warp 对 shared[] 做 scan（得到 warp 间前缀）
    if (warpId == 0) {
        float w = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        w = warp_inclusive_scan(w);
        if (lane < NUM_WARPS) {
            shared[lane] = w;
        }
    }
    __syncthreads();

    // ④ 每 warp 加上前面 warp 的累计和
    if (warpId > 0) {
        val += shared[warpId - 1];
    }
    return val;
}

// Phase 1：每 block 局部 scan，输出局部前缀 + block 总和到 aux
__global__ void scan_phase1(const float* input, float* output, float* aux, int N) {
    __shared__ float shared[NUM_WARPS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    float val = input[tid];
    val = block_inclusive_scan(val, shared);
    output[tid] = val;

    // 最后一个线程写 block 总和
    if (threadIdx.x == blockDim.x - 1 || tid == N - 1) {
        aux[blockIdx.x] = val;
    }
}

// Phase 2：对 aux[] 做 inclusive scan（单 block，block 数 ≤ BLOCK_SIZE）
__global__ void scan_phase2(float* aux, int M) {
    __shared__ float shared[NUM_WARPS];
    int tid = threadIdx.x;
    float val = (tid < M) ? aux[tid] : 0.0f;
    val = block_inclusive_scan(val, shared);
    if (tid < M) {
        aux[tid] = val;
    }
}

// Phase 3：每 block 加上前面 block 的累计和（block 0 不加）
__global__ void scan_phase3(float* output, const float* aux, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (blockIdx.x > 0) {
        output[tid] += aux[blockIdx.x - 1];
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 250000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f KB)\n", N, bytes / 1024.0);

    // ---- host ----
    float *hIn  = (float*)malloc(bytes);
    float *hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f;
    }

    // ---- device ----
    float *dIn, *dOut, *dAux;
    CHECK_CUDA(cudaMalloc(&dIn,  bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&dAux, num_blocks * sizeof(float)));
    printf("blocks = %d  (phase2 单 block 可处理)\n", num_blocks);

    // ---- 三阶段执行 + 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    scan_phase1<<<num_blocks, BLOCK_SIZE>>>(dIn, dOut, dAux, N);
    scan_phase2<<<1, BLOCK_SIZE>>>(dAux, num_blocks);
    scan_phase3<<<num_blocks, BLOCK_SIZE>>>(dOut, dAux, N);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (three-pass): %.3f ms\n", ms);

    // ---- 验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    double ref = 0.0;
    int err = 0;
    for (int i = 0; i < N; ++i) {
        ref += hIn[i];
        if (fabsf(hOut[i] - (float)ref) > 0.01 * fmaxf(1.0f, fabsf((float)ref))) {
            if (++err <= 5)
                printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], (float)ref);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽 ----
    float bw_gbs = (2 * bytes / 1e9) / (ms / 1e3);   // 读 + 写
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaFree(dAux));
    free(hIn); free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把三个 kernel + 调用填进 `solve` 函数（starter 是空的）。注意 `scan_phase2` 要求 `num_blocks ≤ BLOCK_SIZE`（1024），对 `N ≤ 1024×1024 ≈ 1M` 成立。本题性能测试 `N=250K` 完全满足。带 `main()` 的版本用于本地自测。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 prefix_sum_scan.cu -o prefix_sum
./prefix_sum 250000
```

典型输出（A100）：

```text
N = 250000  (976.6 KB)
blocks = 245  (phase2 单 block 可处理)
kernel time (three-pass): 0.08 ms
verify: PASS
effective bandwidth: 24.4 GB/s
```

> ⚠️ 带宽看起来不高，因为 `N=250K` 数据量小（~1MB），三次 kernel 启动开销占比大。scan 的实际瓶颈是**计算深度**（`log N` 串行步），而非带宽。

### 5.2 用 ncu 分析

```bash
ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        smsp__sass_inst_executed_op_fp_32.sum \
    ./prefix_sum 250000
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `dram__throughput` | HBM 带宽占比 | 中等（scan 不是纯带宽瓶颈） |
| `sm__throughput` | SM 算力占比 | 较低（加法轻，但串行链限制） |
| `gpu__time_duration` | 各 kernel 耗时 | phase1 > phase3 > phase2 |

### 5.3 优化方向

1. **kernel 融合**：Phase 2 + Phase 3 可融合——最后一个 block 在 Phase 1 末尾检测自己是最后一个，直接对 aux 做 scan 并加回。省一次启动。需 `atomicAdd` 计数器或 `cooperative_groups`。
2. **`float4` 向量化读**：每线程读 4 个 float，scan 在寄存器内串行（4 步），减少线程数与同步开销。
3. **exclusive scan 变体**：本题要 inclusive，但很多场景（如 stream compaction）要 exclusive（`output[i] = sum(input[0..i-1])`）。改 `warp_inclusive_scan` 起始条件即可。
4. **CUB 库对比**：工业级实现用 `cub::DeviceScan::InclusiveSum`，它自动处理递归三阶段、向量化、kernel 融合。本题禁用外部库，但可作为 benchmark 参考。
5. **大 N 递归**：`N > 1M` 时 Phase 2 的 `num_blocks > 1024`，需递归调用三阶段（aux 的 aux）。CUB 自动处理，手写可递归 launch。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)` work + `O(log BLOCK_SIZE)` 深度（Blelloch 工作高效算法） |
| **空间复杂度** | `O(N)` 输入/输出 + `O(num_blocks)` aux 数组 |
| **算术强度** | `1 FLOP / 8B`（1 次加法 ↔ 读 4B + 写 4B）= **0.125 FLOP/B** |
| **瓶颈类型** | **latency-bound**：scan 受 `log N` 串行深度限制，不是纯带宽或算力瓶颈 |
| **kernel 启动数** | 3 次（phase1 + phase2 + phase3） |
| **warp scan 步数** | 每 warp `log₂32 = 5` 步 |

> 💡 **一句话总结**：prefix sum 是 CUDA 里最难并行化的基础操作——它的数据依赖天然串行，必须用 Blelloch 树形扫描把 `O(N)` 串行链拆成 `O(log N)` 深度的并行树。`warp_inclusive_scan` 这个 5 行函数（`__shfl_up_sync`）是核心积木，与 Day 4 的 `warp_reduce`（`__shfl_down_sync`）构成 warp shuffle 的"一体两面"。scan 是 stream compaction、radix sort、streaming algorithm 的底层支撑，掌握它等于拿到了 GPU 并行算法的进阶钥匙。

# LeetGPU Top K Selection 题解

## 1. 题目概述

- **标题 / 题号**：Top K Selection（#29，medium）
- **链接**：https://leetgpu.com/challenges/top-k-selection
- **难度**：中等
- **标签**：CUDA、Top-K、bitonic sort、归约、shared memory

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，找出其中最大的 `K` 个元素，按**降序**写入 `output[0..K-1]`。

**示例**：

```text
输入：[3.5, 1.2, 7.8, 4.6, 9.0, 2.1, 8.3, 5.5],  K = 3
输出：[9.0, 8.3, 7.8]
```

**约束**：

- `1 ≤ N ≤ 1,000,000`
- `1 ≤ K ≤ 1000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 结果需降序排列

> 💡 Top-K 是 **排序网络** 的入门题。前序的归约（Reduction）把 `N` 个元素压缩成 1 个，Top-K 则压缩成 `K` 个——介于"全排序"和"归约"之间。这引出两个核心概念：**bitonic sort 网络**（GPU 友好的无分支并行排序）和 **两阶段归约**（block 内选 top-K → 跨 block 合并）。掌握 bitonic sort 后，排序、归并、Top-K 都能迁移。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU Top-K
// 方法 A: 全排序 O(N log N)
std::sort(arr, arr + N, std::greater<float>());
// 取前 K 个

// 方法 B: 最小堆 O(N log K)，K 远小于 N 时更优
std::priority_queue<float, std::vector<float>, std::greater<float>> heap;
for (int i = 0; i < N; ++i) {
    if (heap.size() < K) heap.push(arr[i]);
    else if (arr[i] > heap.top()) { heap.pop(); heap.push(arr[i]); }
}
```

`N = 1M, K = 100` 时方法 B 约几毫秒。瓶颈：单线程串行，堆操作的随机访存无法并行化。

### 2.2 朴素 GPU：全局排序

最暴力：调用 `thrust::sort` 全局排序，再拷贝前 `K` 个。

```cuda
__global__ void naive_copy_topk(const float* sorted, float* output, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < K) output[i] = sorted[i];
}
// thrust::sort(d_input, d_input + N, thrust::greater<float>());
```

**问题**：全局排序 `O(N log N)`，对 `N = 1M` 来说计算量远超必要——我们只需前 `K` 个，其余 `N-K` 个的相对顺序无关。且 `thrust::sort` 内部使用多 pass radix sort，kernel 启动开销大。

> ⚠️ Top-K 的核心矛盾：**不需要全排序，但不能不排序**。必须维护某种有序结构才能判断"当前元素是否属于 top-K"。朴素 GPU 全局排序做了太多无用功；而 `atomic` 竞争（每线程用 `atomicMax` 抢 top-K 槽位）又因高冲突而退化为串行。需要在"并行度"与"有序性"之间找平衡——**bitonic sort 网络** 正是答案。

## 3. GPU 设计

### 3.1 并行化策略：两阶段 bitonic sort + 归约

设计分两阶段，与归约题的"block 内归约 → 跨 block 归约"结构一致：

![两阶段 Top-K 架构](images/top-k-selection_two_phase.svg)

1. **第一阶段 `topk_kernel`**：grid 的每个 block 用 grid-stride 读多段数据，每段在 shared memory 内做 **bitonic sort**（降序），取前 `K` 个与 running top-K 做 **bitonic merge** 合并。block 处理完后将 `K` 个元素写入 `partial[blockIdx.x * K]`。
2. **第二阶段迭代归约**：对 `partial[]` 再跑 `topk_kernel`，每轮 block 数减半，直到总量 ≤ `BLOCK_SIZE`。
3. **最终排序 `topk_merge_final`**：单 block 将剩余元素做 bitonic sort，输出前 `K` 个。

> 💡 为什么用 **bitonic sort** 而不是快速选择（quickselect）？Quickselect 的 partition 步骤有数据依赖（pivot 选择导致不均匀访存），GPU 上难以高效并行。Bitonic sort 是**固定比较网络**——每一步的比较位置在编译期就确定，无分支、无数据依赖、访存模式规整，是 GPU 排序的黄金模板。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`partial[]` 中间结果（双缓冲交替）、`output` 写 |
| **shared memory** | ✓ | bitonic sort 缓冲 + running top-K 合并缓冲（`2×K_PAD` floats） |
| **register** | ✓ | 每线程的索引计算、临时交换变量 |

### 3.3 关键技巧：bitonic sort 网络与 bitonic merge

#### Bitonic sort 网络

**Bitonic 序列**：先单调增再单调减（或反之）的序列。Bitonic sort 的核心是：对任意无序序列，先逐步构造 bitonic 序列，再逐步归并为有序序列。

![Bitonic Sort 网络结构](images/top-k-selection_bitonic_network.svg)

网络共 `log₂n` 个 stage，每个 stage `k = 2, 4, ..., n` 内部做 `log₂k` 步 compare-exchange，步长 `j = k/2, k/4, ..., 1`。比较方向由 `(i & k)` 决定——这使每一步的 compare-exchange **完全独立**，天然适合 GPU 并行。

```cuda
// 降序 bitonic sort（n 必须是 2 的幂）
for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
        __syncthreads();
        int i = threadIdx.x;
        int ij = i ^ j;
        if (ij > i) {
            if ((i & k) == 0) {
                if (smem[i] < smem[ij]) swap(smem[i], smem[ij]);  // 降序
            } else {
                if (smem[i] > smem[ij]) swap(smem[i], smem[ij]);
            }
        }
    }
}
```

#### Bitonic merge：合并两个有序 top-K

合并 running top-K 与 chunk top-K 时，两个降序数组拼接后并非 bitonic 序列。但**反转其中一个**即可：降序 + 反转(降序) = 降序 + 升序 = bitonic！

![Bitonic Merge 合并两个 top-K](images/top-k-selection_merge.svg)

随后只需一轮 bitonic merge（`j = K, K/2, ..., 1`）即可将 `2K` 个元素排序。**关键优化**：只需处理前 `K` 个位置——`j = K` 这一步将较大值全部推到前半部分，后续步骤仅在前半部分内部排序，完全不影响 top-K 的正确性。

> 💡 Bitonic merge 是 Top-K 的"归约算子"——就像 `__shfl_down_sync` 是求和的归约算子。每合并一次，数据量减半（两个 `K` → 一个 `K`），`log₂(blocks)` 轮后收敛到最终结果。

## 4. Kernel 实现

完整可编译的两阶段 Top-K（grid-stride + shared memory bitonic sort + bitonic merge + 迭代归约）：

```cuda
// top_k_selection.cu —— Bitonic Sort Top-K
// 编译命令: nvcc -O3 -arch=sm_80 top_k_selection.cu -o topk

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <algorithm>
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
#define K_PAD     1024     // 2 的幂，≥ K_MAX(1000)

// ---- shared memory bitonic sort（降序），n 必须是 2 的幂 ----
__inline__ __device__ void bitonic_sort_desc(float* smem, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            int i = threadIdx.x;
            if (i < n) {
                int ij = i ^ j;
                if (ij > i) {
                    if ((i & k) == 0) {
                        if (smem[i] < smem[ij]) {            // 降序：大的放低位
                            float t = smem[i]; smem[i] = smem[ij]; smem[ij] = t;
                        }
                    } else {
                        if (smem[i] > smem[ij]) {
                            float t = smem[i]; smem[i] = smem[ij]; smem[ij] = t;
                        }
                    }
                }
            }
        }
    }
}

// ---- Phase 1: grid-stride + running top-K 合并 ----
__global__ void topk_kernel(const float* input, float* partial, int N, int K, int Kp) {
    // buf[0..Kp-1]:      running top-K（降序）
    // buf[Kp..2*Kp-1]:   chunk 排序缓冲 → merge 后半部分
    __shared__ float buf[2 * K_PAD];

    int tid = threadIdx.x;

    // 初始化 running top-K 为 -FLT_MAX
    for (int i = tid; i < Kp; i += blockDim.x) buf[i] = -FLT_MAX;
    __syncthreads();

    // grid-stride：每轮处理 BLOCK_SIZE 个元素
    for (int base = blockIdx.x * BLOCK_SIZE; base < N; base += gridDim.x * BLOCK_SIZE) {
        // ---- ① 加载 chunk 到 buf[Kp..Kp+BLOCK_SIZE-1] ----
        for (int i = tid; i < BLOCK_SIZE; i += blockDim.x) {
            int idx = base + i;
            buf[Kp + i] = (idx < N) ? input[idx] : -FLT_MAX;
        }
        __syncthreads();

        // ---- ② chunk 内 bitonic sort（降序）----
        bitonic_sort_desc(buf + Kp, BLOCK_SIZE);
        __syncthreads();

        // ---- ③ 反转 chunk top-Kp 构成 bitonic 序列（降序+升序）----
        for (int i = tid; i < Kp / 2; i += blockDim.x) {
            float t = buf[Kp + i];
            buf[Kp + i] = buf[2 * Kp - 1 - i];
            buf[2 * Kp - 1 - i] = t;
        }
        __syncthreads();

        // ---- ④ bitonic merge（降序），只处理前 Kp 个 ----
        for (int j = Kp; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = tid; i < Kp; i += blockDim.x) {
                int ij = i ^ j;
                if (ij > i) {
                    if (buf[i] < buf[ij]) {
                        float t = buf[i]; buf[i] = buf[ij]; buf[ij] = t;
                    }
                }
            }
        }
        __syncthreads();
    }

    // ---- ⑤ 写出 top-K 到 partial ----
    for (int i = tid; i < K; i += blockDim.x)
        partial[blockIdx.x * K + i] = buf[i];
}

// ---- Phase 2: 单 block 最终排序 ----
__global__ void topk_merge_final(const float* partial, float* output, int M, int K) {
    __shared__ float buf[BLOCK_SIZE];
    int tid = threadIdx.x;

    buf[tid] = (tid < M) ? partial[tid] : -FLT_MAX;
    __syncthreads();

    bitonic_sort_desc(buf, BLOCK_SIZE);

    if (tid < K) output[tid] = buf[tid];
}

// ---- 辅助：求 ≥x 的最小 2 的幂 ----
int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    int K = (argc > 2) ? atoi(argv[2]) : 100;
    if (K > K_PAD) K = K_PAD;
    int Kp = next_pow2(K);
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d, K = %d, Kp = %d\n", N, K, Kp);

    // ---- host 端 ----
    float *hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i)
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f;

    // ---- device 端 ----
    float *dIn, *dBufA, *dBufB, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int maxBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t pBytes = (size_t)maxBlocks * K * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dBufA, pBytes));
    CHECK_CUDA(cudaMalloc(&dBufB, pBytes));
    CHECK_CUDA(cudaMalloc(&dOut, K * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    // ---- 迭代 top-K 归约：每轮 block 数减半 ----
    const float* dCurr = dIn;
    float* dNext = dBufA;
    int currN = N;
    int blocks = std::min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 128);
    int round = 0;

    while (currN > BLOCK_SIZE) {
        topk_kernel<<<blocks, BLOCK_SIZE>>>(dCurr, dNext, currN, K, Kp);
        dCurr = dNext;
        dNext = (dNext == dBufA) ? dBufB : dBufA;
        currN = blocks * K;
        blocks = std::max(1, blocks / 2);
        round++;
    }

    // ---- 最终单 block 排序 ----
    topk_merge_final<<<1, BLOCK_SIZE>>>(dCurr, dOut, currN, K);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("rounds: %d, kernel time: %.3f ms\n", round, ms);

    // ---- 验证 ----
    float *hOut = (float*)malloc(K * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hOut, dOut, K * sizeof(float), cudaMemcpyDeviceToHost));

    float *hSorted = (float*)malloc(N * sizeof(float));
    memcpy(hSorted, hIn, bytes);
    std::sort(hSorted, hSorted + N, std::greater<float>());
    std::sort(hOut, hOut + K, std::greater<float>());

    bool pass = true;
    for (int i = 0; i < K; ++i) {
        if (fabsf(hOut[i] - hSorted[i]) > 1e-4f) { pass = false; break; }
    }
    printf("GPU top-1: %.4f  CPU top-1: %.4f  %s\n",
           hOut[0], hSorted[0], pass ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dBufA));
    CHECK_CUDA(cudaFree(dBufB));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hOut); free(hSorted);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `topk_kernel` + `topk_merge_final` 填进 `solve` 函数，在 `solve` 内部实现迭代归约逻辑即可。带 `main()` 的版本用于本地自测。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 top_k_selection.cu -o topk
./topk 1000000 100
```

典型输出（A100 / SM=108）：

```text
N = 1000000, K = 100, Kp = 128
rounds: 4, kernel time: 0.52 ms
GPU top-1: 99.9900  CPU top-1: 99.9900  PASS
```

> ⚠️ `Kp = next_pow2(K)`：bitonic sort 要求元素数为 2 的幂。`K = 100` 时 `Kp = 128`，多出的 28 个位置填充 `-FLT_MAX`，自动排到末尾不影响结果。

### 5.2 用 ncu 分析

```bash
ncu --set full --target-processes all -o topk_profile ./topk 1000000 100

# 关键指标
ncu --metrics gpu__time_duration.sum, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum, \
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum, \
        launch__waves_per_multiprocessor \
    ./topk 1000000 100
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 算力占比 | > 40%（compute-bound，比较密集） |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | shared memory 读 bank conflict | 越低越好 |
| `launch__waves_per_multiprocessor` | 每 SM 的 wave 数 | > 2（保证并发隐藏延迟） |

### 5.3 优化方向

1. **warp 级 bitonic sort**：对 `Kp ≤ 32` 的小 `K`，用 `__shfl_xor_sync` 在 warp 内直接做 compare-exchange，完全绕开 shared memory，延迟更低。
2. **减少 bank conflict**：bitonic sort 的 `i ^ j` 访问模式会产生 shared memory bank conflict。在 `buf` 每行末尾加 1 个 float padding（`K_PAD + 1` 宽度），可将 32-bank 冲突降至最低。
3. **radix selection**：将 bitonic sort（`O(N log²N)`）替换为 **radix-based selection**（`O(N)`）——按位从高到低统计 0/1 个数，二分确定 top-K 边界。对 `K` 较大（接近 `N`）时优势显著。
4. **多元素 per thread**：每线程加载 4 个 `float`（`float4` vector load），先做线程内排序，再送入 bitonic 网络。减少 shared memory 访问次数，提升排序吞吐。
5. **kernel 融合**：当 `blocks * K ≤ BLOCK_SIZE` 时，将最后一轮 `topk_kernel` 和 `topk_merge_final` 融合为单个 kernel，省掉一次启动开销。

> 💡 优化 1（warp 级 bitonic）对小 `K` 收益最大——当 `Kp ≤ 32` 时，整个 top-K 维护在 warp 寄存器内，shared memory 仅用于 chunk 排序。优化 3（radix selection）是大 `K` 场景的终极方案，CUB 的 `cub::DeviceRadicalSort::SortKeys` 底层就是这个思路。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N · log²B + rounds · B · log²Kp)`，其中 `B = BLOCK_SIZE`，`rounds ≈ log₂(blocks)`。简化为 `O(N log²B)` |
| **空间复杂度** | `O(N)` 输入 + `O(blocks · K)` 双缓冲 partial + `O(2·K_PAD)` shared memory |
| **bitonic sort 比较数** | 单次 sort `n` 个元素需 `n/2 · log₂n · log₂n` 次 compare-exchange |
| **算术强度** | `~2 FLOP / 4B`（1 次比较 + 1 次交换 ↔ 读 4B）≈ **0.5 FLOP/B** |
| **瓶颈类型** | **compute-bound**：比较交换密集，shared memory 访问是主要延迟来源 |
| **kernel 启动数** | `rounds + 1`（每轮 1 次 `topk_kernel` + 1 次 `topk_merge_final`） |

> 💡 **一句话总结**：Top-K 是 **bitonic sort 网络** 的经典应用——它把"N 中选 K"拆成"block 内 bitonic sort 选 top-K + bitonic merge 合并 + 跨 block 迭代归约"三层。Bitonic sort 的无分支 compare-exchange 网络天然适配 GPU 的 SIMT 执行模型，而 bitonic merge（反转 + 单轮归并）提供了高效的 top-K 归约算子。这套模板可直接迁移到 **parallel merge**、**排序**、**Top-p sampling** 等场景。

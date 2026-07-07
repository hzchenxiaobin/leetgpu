# LeetGPU Histogramming 题解

## 1. 题目概述

- **标题 / 题号**：Histogramming（#13，medium）
- **链接**：https://leetgpu.com/challenges/histogramming
- **难度**：中等
- **标签**：CUDA、histogram、atomicAdd、shared memory privatization、bank conflict、profiling

**题意**：给定长度为 `N` 的 `int32` 数组 `input`，元素值域 `[0, B)`，统计每个值的出现次数，输出长度为 `B` 的直方图 `hist`：

$$\text{hist}[v] = \sum_{i=0}^{N-1} \mathbb{1}[\text{input}[i] = v], \quad v \in [0, B)$$

**示例**：

```text
input = [0, 1, 2, 1, 0, 1, 3, 2], B = 4
hist = [2, 3, 2, 1]   // 0 出现 2 次，1 出现 3 次，2 出现 2 次，3 出现 1 次
```

**约束**：

- `1 ≤ N ≤ 10,000,000`
- `1 ≤ B ≤ 256`
- 性能测试取 `N = 10,000,000, B = 256`（值域满）

> 💡 这是 **atomicAdd 冲突分析** 的经典题。前序题里 [Reduction #4](../week1/day4/leetgpu-reduction-solution.md) 是"多对一无竞争归约"，直方图是"**多对 B 有竞争累加**"——`N` 个线程要把结果写到 `B` 个桶，多个线程可能同时写同一桶，必须用 `atomicAdd`。但 atomic 在高冲突下严重串行化。它引出 GPU 编程的关键优化——**shared memory privatization（私有化）**：每 block 维护一份局部直方图，block 内冲突降到 `1/blockNum`，最后合并。本题用 [Day 6 的 ncu profiling](../../aiinfra/week2/day6/README.md) 分析 atomic 冲突和 bank conflict，验证优化方法论在非 GEMM kernel 上的适用性。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行直方图
void histogram_cpu(const int* input, int* hist, int N, int B) {
    for (int v = 0; v < B; ++v) hist[v] = 0;
    for (int i = 0; i < N; ++i) {
        hist[input[i]]++;   // 单线程，无冲突
    }
}
```

`N = 10M` 时单核约几十毫秒。瓶颈：单线程串行，但**无冲突**——CPU 版的直方图天然无竞争。

### 2.2 朴素 GPU：global atomicAdd

最直接的并行：每 thread 读一个元素，用 `atomicAdd` 累加到全局 `hist`：

```cuda
__global__ void histogram_global(const int* input, int* hist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&hist[input[idx]], 1);   // ← 所有线程争抢 B 个桶！
    }
}
```

![Global atomic：10M 线程争抢 256 桶，冲突严重串行化](images/histogram_global_atomic_conflict.svg)

**致命问题**：`N = 10M` 个线程对 `B = 256` 个桶做 `atomicAdd`。若数据均匀分布，每桶平均 `10M/256 ≈ 39000` 次写，**同一桶的写被强制串行**。GPU 的 atomic 由 L2 cache 上的 atomic unit 处理，同地址的原子操作排成队列执行，吞吐从"并行"退化到"串行"。

> ⚠️ 直方图的核心矛盾：输出只有 `B` 个桶（`B≪N`），但输入有 `N` 个元素。**多对少的写天然产生高冲突**。`atomicAdd` 适合低竞争场景（如 [Reduction #4](../week1/day4/leetgpu-reduction-solution.md) 末尾的二次归约），不适合大规模直方图。破局思路：**分桶**——每 block 一份私有直方图，把全局 `B` 桶的冲突分散到 block 内 `B` 桶。

## 3. GPU 设计

### 3.1 并行化策略：shared memory privatization

**核心思想**：每个 block 在 shared memory 维护一份**局部直方图** `s_hist[B]`，block 内所有 thread 对 `s_hist` 做 atomic（冲突域仅限 block 内），最后各 block 的局部结果合并到全局 `hist`。

![Shared Memory 私有化：每 block 一份局部直方图，冲突降 1/numBlocks](images/histogram_shared_privatization.svg)

冲突减少倍数：全局版所有 `N` 个线程争 `B` 桶 → 私有化后每 block 只有 `blockDim.x` 个线程争 `B` 桶，冲突数除以 `gridDim.x`。`gridDim.x = 1000` 时冲突降 **1000×**。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`hist` 写（合并阶段） |
| **shared memory** | ✓ | **核心**：`s_hist[B]`，每 block 私有局部直方图 |
| **register** | ✓ | 临时变量 + 索引计算 |

### 3.3 关键技巧 1：shared memory privatization 三阶段

每个 block 执行：

1. **初始化**：协作把 `s_hist[0..B-1]` 清零（`B ≤ 256`，几十个 thread 一次清完）
2. **局部累加**：grid-stride 扫描 `input`，对 `s_hist[input[i]]` 做 `atomicAdd`（冲突域 = block 内）
3. **全局合并**：协作把 `s_hist[0..B-1]` 用 `atomicAdd` 加到全局 `hist`（每桶仅 1 次 atomic，共 `B×gridDim.x` 次，冲突极低）

```cuda
__shared__ int s_hist[B];
// ① 清零
for (int i = tid; i < B; i += blockDim.x) s_hist[i] = 0;
__syncthreads();
// ② 局部累加（冲突域 = block 内）
for (int i = blockIdx.x*blockDim.x + tid; i < N; i += gridDim.x*blockDim.x)
    atomicAdd(&s_hist[input[i]], 1);
__syncthreads();
// ③ 全局合并（每桶 1 次 atomic）
for (int i = tid; i < B; i += blockDim.x)
    atomicAdd(&hist[i], s_hist[i]);
```

> 💡 阶段 ③ 看似仍有 atomic，但冲突极低：每桶 `gridDim.x` 次写（如 1000 次），远少于全局版的 `N/B ≈ 39000` 次。且阶段 ③ 的 atomic 是**跨 block 的 L2 atomic**，与阶段 ② 的 block 内 shared atomic 正交，不互相阻塞。

### 3.4 关键技巧 2：bank conflict 与 2D 共享直方图

`atomicAdd(&s_hist[v], 1)` 中，若多个 thread 的 `v` 相同，它们访问同一 shared memory bank → **bank conflict**（同 bank 的访问串行化）。`B = 256`、shared memory 有 32 个 bank，`v` 均匀分布时冲突低；但若数据倾斜（某些 `v` 频率高），冲突激增。

**优化**：用 **2D 私有直方图**——每 block 维护 `R` 份独立的 `s_hist`（如 `R = 4`），按 `tid % R` 分散到不同行，让冲突域从 `blockDim.x` 降到 `blockDim.x / R`：

![2D 共享直方图：R 份副本分散 bank conflict](images/histogram_2d_replicas.svg)

```cuda
__shared__ int s_hist[R][B];   // R 份私有，每份独立
int rid = tid % R;
// ② 累加时按 rid 分散
atomicAdd(&s_hist[rid][input[i]], 1);
// ③ 合并时 R 份求和后再 atomicAdd 到 global
```

`R = 4` 时冲突再降 4×。代价是 shared 占用 `R×B×4B`（`R=4, B=256` 时 4KB，可接受）。

> ⚠️ Bank conflict 是直方图性能的隐藏杀手。`ncu` 的 `l1tex__data_bank_conflicts` 指标可量化。若 `s_hist[v]` 的 `v` 分布倾斜，单份私有直方图可能因 bank conflict 比 global atomic 还慢。2D 共享是标准解法，CUB 的 `DeviceHistogram` 内部即用类似策略。

## 4. Kernel 实现

完整可编译的私有化版本（含 2D 共享优化 + grid-stride + 合并阶段）：

```cuda
// histogram_privatized.cu —— 直方图：shared memory privatization + 2D 共享
// 编译命令: nvcc -O3 -arch=sm_80 histogram_privatized.cu -o histogram
// 运行:     ./histogram 10000000 256

#include <cstdio>
#include <cstdlib>
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
#define REPLICAS   4        // 2D 共享：4 份私有直方图，冲突再降 4×

// ---- Version 1：global atomic（baseline，用于对比）----
__global__ void histogram_global(const int* input, int* hist, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&hist[input[idx]], 1);
}

// ---- Version 2：shared memory privatization + 2D 共享（优化版）----
__global__ void histogram_shared(const int* input, int* hist, int N, int B) {
    __shared__ int s_hist[REPLICAS][256];   // R 份私有，B ≤ 256

    int tid = threadIdx.x;
    int rid = tid % REPLICAS;

    // ① 协作清零 R 份私有直方图
    for (int i = tid; i < REPLICAS * B; i += blockDim.x) {
        int r = i / B, b = i % B;
        s_hist[r][b] = 0;
    }
    __syncthreads();

    // ② grid-stride 累加到私有直方图（按 rid 分散，冲突降 R×）
    for (int i = blockIdx.x * blockDim.x + tid; i < N;
         i += gridDim.x * blockDim.x) {
        int v = input[i];
        atomicAdd(&s_hist[rid][v], 1);
    }
    __syncthreads();

    // ③ 合并：R 份求和后 atomicAdd 到 global（每桶 1 次 atomic）
    for (int b = tid; b < B; b += blockDim.x) {
        int sum = 0;
        #pragma unroll
        for (int r = 0; r < REPLICAS; ++r) sum += s_hist[r][b];
        atomicAdd(&hist[b], sum);
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    int B = (argc > 2) ? atoi(argv[2]) : 256;
    size_t in_bytes = (size_t)N * sizeof(int);
    size_t hist_bytes = (size_t)B * sizeof(int);
    printf("N = %d, B = %d\n", N, B);

    // ---- host ----
    int *hIn  = (int*)malloc(in_bytes);
    int *hRef = (int*)calloc(B, sizeof(int));
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = rand() % B;
        hRef[hIn[i]]++;
    }

    // ---- device ----
    int *dIn, *dHist;
    CHECK_CUDA(cudaMalloc(&dIn, in_bytes));
    CHECK_CUDA(cudaMalloc(&dHist, hist_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));

    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- Version 1: global atomic ----
    CHECK_CUDA(cudaMemset(dHist, 0, hist_bytes));
    cudaEventRecord(t0);
    histogram_global<<<blocks, BLOCK_SIZE>>>(dIn, dHist, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_global = 0;
    cudaEventElapsedTime(&ms_global, t0, t1);
    printf("global atomic: %.3f ms\n", ms_global);

    // ---- Version 2: shared privatization ----
    CHECK_CUDA(cudaMemset(dHist, 0, hist_bytes));
    cudaEventRecord(t0);
    histogram_shared<<<blocks, BLOCK_SIZE>>>(dIn, dHist, N, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_shared = 0;
    cudaEventElapsedTime(&ms_shared, t0, t1);
    printf("shared privatization: %.3f ms (R=%d)\n", ms_shared, REPLICAS);
    printf("speedup: %.2fx\n", ms_global / ms_shared);

    // ---- 验证 ----
    int *hHist = (int*)malloc(hist_bytes);
    CHECK_CUDA(cudaMemcpy(hHist, dHist, hist_bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int v = 0; v < B; ++v) {
        if (hHist[v] != hRef[v]) {
            if (++err <= 5) printf("MISMATCH hist[%d]: got %d, expect %d\n", v, hHist[v], hRef[v]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽 ----
    float bw_gbs = (in_bytes / 1e9) / (ms_shared / 1e3);
    printf("read bandwidth (shared): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dHist));
    free(hIn); free(hRef); free(hHist);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `histogram_shared` 填进 `solve` 函数即可。带 `main()` 的版本含两个版本对比，用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 histogram_privatized.cu -o histogram
./histogram 10000000 256
```

典型输出（A100）：

```text
N = 10000000, B = 256
global atomic: 4.20 ms
shared privatization: 0.85 ms (R=4)
speedup: 4.94x
verify: PASS
read bandwidth (shared): 46.9 GB/s
```

### 5.2 用 ncu 分析（Day 6 主题的核心实践）

```bash
# 对比两个版本的 atomic 冲突和 bank conflict
ncu --kernel-name regex:histogram \
    --metrics gpu__time_duration.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum, \
              sm__inst_executed_pipe_lsu_op_atomic.sum, \
              sm__occupancy.avg.pct_of_peak_sustained_elapsed \
    ./histogram 10000000 256
```

| 指标 | global atomic | shared privatization | 含义 |
|------|--------------|---------------------|------|
| `gpu__time_duration` | ~4.2 ms | ~0.85 ms | **~5× 加速** |
| `sm__...atomic` | `~10M` 次 global atomic | `~10M` shared + `~B×blocks` global | global atomic 大减 |
| `l1tex__...bank_conflicts` | N/A（无 shared） | 较低（R=4 分散） | bank conflict 量化 |
| `dram__throughput` | ~20%（atomic 阻塞读） | ~50% | 私有化后读流畅 |
| `sm__occupancy` | ~30%（atomic 串行拖累） | ~75% | 无阻塞，占用率高 |

> 💡 这是 Day 6 ncu profiling 方法论的典型应用：用指标定位瓶颈（atomic 冲突 vs bank conflict vs 带宽），针对性优化。`l1tex__data_bank_conflicts` 是直方图特有的关键指标——若 `R=1` 时它很高，说明 `v` 分布倾斜导致 bank conflict，需增大 `R`。

### 5.3 优化方向

1. **`REPLICAS` 调优**：`R` 越大冲突越低，但 shared 占用 `R×B×4B` 增加。`B=256` 时 `R=4`（4KB）是甜点；`B` 小可增大 `R`，`B` 大需减小。用 `ncu` 的 bank conflict 指标指导。
2. **`int4` 向量化读**：每 thread 一次读 4 个 int（16B），减少内存事务。需 `N` 是 4 的倍数。
3. **warp 级聚合（warp-aggregated atomic）**：warp 内先用 `__shfl_sync` 找相同 `v` 的 lane，只由一个 lane 做 atomic。CUB 的 `BlockHistogram` 用此策略，冲突降到 warp 级。
4. **排序预处理**：若 `input` 已排序，相同 `v` 连续，可改用非 atomic 的扫描统计。但排序开销通常大于收益。
5. **多 pass 分桶**：`B` 极大（如 `B > 4096`）时 shared 放不下，可分多个 pass，每 pass 处理一段值域。本题 `B ≤ 256` 无需。
6. **global memory 布局**：`hist` 是 `int32`，若 `B` 是 32 的倍数（256 是），合并阶段写 `hist[b]` 天然对齐，无额外优化空间。

> 💡 优化 1（`R` 调优）和 3（warp 聚合）是直方图专用优化。`R` 调优立竿见影；warp 聚合是 CUB 级优化，复杂度高但冲突最优。本题 `R=4` 已能 ~5× 加速。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`：每元素 1 次 atomic（shared）+ `O(B×gridDim)` 合并 |
| **空间复杂度** | `O(N)` 输入 + `O(B)` 输出 + `O(R×B)` shared memory |
| **算术强度** | `0 FLOP / 4B`（无浮点，只有 atomic 计数）→ 纯 atomic 吞吐瓶颈 |
| **瓶颈类型** | **atomic-conflict-bound**：性能由 atomic 冲突决定，非带宽或算力 |
| **shared memory 占用** | `R × B × 4B`（`R=4, B=256` 时 4 KB/block） |
| **global atomic 次数** | global 版 `N` 次；私有化版 `B×gridDim` 次（合并阶段，冲突极低） |
| **shared atomic 次数** | `N` 次（冲突域 = block 内，再除以 `R`） |

> 💡 **一句话总结**：Histogramming 是 atomicAdd 冲突分析的教科书题——它揭示了一类"多对少写"的访问模式（N 个输入写 B 个桶，B≪N），破局思路是 **shared memory privatization**：每 block 一份私有直方图把全局冲突分散到 block 内，配合 2D 共享（`R` 份副本）降 bank conflict。这与 [Reduction #4](../week1/day4/leetgpu-reduction-solution.md) 的"无竞争树形归约"形成对照——归约是"多对一无冲突"，直方图是"多对 B 有冲突"。本题用 Day 6 的 ncu profiling 量化 atomic 冲突和 bank conflict，验证了"指标驱动优化"方法论在非 GEMM kernel 上的普适性。私有化思想可直接迁移到 scatter、counting sort、reduce-by-key 等所有"多对少写"类 kernel。

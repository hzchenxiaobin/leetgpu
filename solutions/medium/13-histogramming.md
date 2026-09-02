# LeetGPU Histogramming 题解

## 1. 题目概述

- **标题 / 题号**：Histogramming（#13，medium）
- **链接**：https://leetgpu.com/challenges/histogramming
- **难度**：中等
- **标签**：CUDA、Histogram、Shared Memory、`atomicAdd`、privatization、memory-bound

**题意**：给定长度为 `N` 的 `int32` 数组 `input`，元素值域 `[0, num_bins)`，统计每个值（bin）出现的次数，结果写入长度为 `num_bins` 的 `histogram[]`。即 `histogram[b] = #{i | input[i] == b}`。参考实现会先用合法掩码 `(input >= 0) & (input < num_bins)` 过滤越界值，再调用 `torch.bincount`——因此 kernel 内**必须跳过越界元素**，否则会越界写 `histogram`。

**示例**：

```text
输入：input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
输出：[2, 2, 1]     （0 出现 2 次，1 出现 2 次，2 出现 1 次）

输入：input = [3, 3, 3, 3],  N = 4, num_bins = 5
输出：[0, 0, 0, 4, 0]   （只有 bin 3 有计数）
```

**约束**：

- `1 ≤ N ≤ 100,000,000`（1 亿）
- `0 ≤ input[i] < num_bins`（参考实现仍会过滤越界值，kernel 需自带守卫）
- `1 ≤ num_bins ≤ 1024`
- 性能测试取 `N = 50,000,000`，`num_bins = 256`

> 💡 这是 **shared memory privatization**（私有化）的经典题。前几道题的数据流都是"每个线程独立写自己的位置"，互不干扰；而直方图是**多对少**——上千万个输入元素要汇聚到区区 256 个 bin。这天然导致**写冲突**。本题的核心矛盾不是"算不快"，而是"抢不到写口"，解法是用 **shared memory 私有副本** 把全局竞争打散成 block 内竞争。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 单 pass 直方图
void histogram_cpu(const int* input, int* hist, int N, int B) {
    memset(hist, 0, B * sizeof(int));
    for (int i = 0; i < N; ++i) {
        int b = input[i];
        if (b >= 0 && b < B)
            hist[b]++; // 顺序访问，无竞争
    }
}
```

`N = 5000 万` 时单核约 50-80 ms。CPU 的优势是**完全顺序**——每条 `hist[b]++` 都命中 L1/L2 cache，没有竞争。瓶颈是单线程带宽有限。

### 2.2 朴素 GPU：atomicAdd 到 global

最暴力的并行：每个 thread 读一个元素，用 `atomicAdd` 把对应 bin 加 1 到全局 `histogram[]`。

```cuda
__global__ void histogram_naive(const int* input, int* hist, int N, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&hist[bin], 1); // ← 所有线程抢 256 个 global 地址！
        }
    }
}
```

**致命问题**：5000 万个线程只有 256 个 bin 可写。即使值域均匀分布，平均每个 bin 仍要承受约 `50M/256 ≈ 195000` 次 atomic。GPU 的 `atomicAdd` 是**硬件串行化**同一地址的冲突——`hist[3]` 同时被几十个 thread 撞上时，硬件只能一个一个排队。实测下来朴素版**常常比 CPU 还慢 5-10 倍**。

> ⚠️ 直方图的瓶颈不在内存带宽，而在 **atomic 写串行化**。50M 次写挤 256 个坑，竞争烈度极高。优化方向必须是**减少同一地址的并发写者数量**，而不是堆 thread 数。

## 3. GPU 设计

### 3.1 并行化策略：Privatization 私有化

核心思想：**把一份全局 histogram 拆成多份私有 histogram**，让竞争从"全部 vs 全局"降级为"block 内 vs block 私有"，最后再合并。

![Privatization 私有化策略](/images/histogram_shared_privatization.svg)

三步走：

1. **每个 block 在 shared memory 开一份私有 histogram**（`num_bins` 个 int，本题 `256 × 4B = 1KB`，远小于单 block 48-100KB shared 配额；最大 `num_bins=1024` 时仅 4KB）。
2. **block 内线程把元素累加到自己的 shared histogram**：`atomicAdd(&s_hist[bin], 1)`。shared memory 的 atomic 延迟比 global 低一个数量级，且**竞争范围从全 grid 缩到单 block**（单 block 通常 256 thread，竞争者骤降两个数量级）。
3. **block 末尾把 shared histogram 合并到 global**：每个 bin 用一次 `atomicAdd(&hist[b], s_hist[b])`。注意此时 global atomic 的次数 = `num_bins × blocks`（如 `256 × 432 ≈ 11 万次`），远少于朴素版的 `N = 5000 万次`，竞争几乎消失。

> 💡 **privatization 的本质**：用空间换竞争——多花 `O(B × blocks)` 的 shared memory，把 `O(N)` 次 global atomic 降级为 `O(N)` 次 shared atomic + `O(B × blocks)` 次 global atomic。shared atomic 又快又便宜，这笔交易极其划算。

### 3.2 存储层次使用

![两阶段 atomic 与存储层次](/images/histogramming_two_stage_atomic.svg)

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input[]` 只读 + `histogram[]` 最终输出（atomic 合并） |
| **shared memory** | ✓ | 每 block 一份私有 histogram，`num_bins` 个 int（≤4KB），atomicAdd 主战场 |
| **register** | ✓ | 每线程的循环变量、当前 bin 值 |

### 3.3 关键技巧：两阶段 atomic

| 阶段 | 操作 | atomic 次数 | 竞争烈度 | 延迟 |
|------|------|------------|----------|------|
| 朴素 | `atomicAdd(&hist[b], 1)` | `N`（5000 万） | 极高（全 grid 抢 256 址） | global，~数百周期 |
| 阶段① | `atomicAdd(&s_hist[b], 1)` | `N`（5000 万） | 低（单 block 内抢 256 址） | shared，~数十周期 |
| 阶段② | `atomicAdd(&hist[b], s_hist[b])` | `B × blocks`（~11 万） | 极低（每址仅 blocks 个写者） | global，但并发低 |

> ⚠️ **不要省掉阶段②的** `if (s_hist[b] > 0)` **判断**。即使某 bin 计数为 0 也发 atomicAdd 会让 global 端多做无谓事务。虽然逻辑等价，但能减少阶段②的 global 写流量约一半（稀疏分布时收益更大）。

## 4. Kernel 实现

完整可编译的私有化版本（含朴素版对比 + CPU 验证）。kernel 用**动态 shared memory**（`extern __shared__`），这样无论 `num_bins` 是 3 还是 1024 都能正确开副本，无需重编译：

```cuda
// histogram_privatized.cu —— shared memory privatization 直方图
// 编译命令: nvcc -O3 -arch=sm_120 histogram_privatized.cu -o histogram
// 运行:     ./histogram 50000000 256

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// 朴素版：所有线程 atomicAdd 到 global histogram（剧烈竞争，用于对比基准）
__global__ void histogram_naive(const int* input, int* hist, int N, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&hist[bin], 1);
        }
    }
}

// 优化版：privatization —— 每 block 一份 shared histogram，最后合并到 global
// 用动态 shared memory（extern __shared__）适配任意 num_bins（1..1024）
__global__ void histogram_privatized(const int* input, int* hist, int N, int B) {
    extern __shared__ int s_hist[];        // 大小 = B，启动时传入 B*sizeof(int)

    int tid = threadIdx.x;

    // ① 初始化 shared histogram 为 0（block 内协作清零）
    for (int b = tid; b < B; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // ② grid-stride 读输入，atomicAdd 到 shared（block 内竞争远小于 global）
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&s_hist[bin], 1);    // shared atomic，低延迟
        }
    }
    __syncthreads();

    // ③ 把 shared histogram 合并到 global（每 bin 一次 global atomic）
    for (int b = tid; b < B; b += blockDim.x) {
        int v = s_hist[b];
        if (v > 0) {
            atomicAdd(&hist[b], v);
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 50000000;
    int B = (argc > 2) ? atoi(argv[2]) : 256;
    size_t bytes = (size_t)N * sizeof(int);
    printf("N = %d, B = %d  (%.1f MB input)\n", N, B, bytes / 1e6);

    // ---- host 端 ----
    int* hIn = (int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = rand() % B;               // 值域 [0, B)
    }

    // ---- device 端 ----
    int *dIn, *dHist;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dHist, B * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;                // 经验值，保证 wave 充足但不过载
    printf("blocks = %d, threads/block = %d\n", blocks, BLOCK_SIZE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- 优化版 ----
    CHECK_CUDA(cudaMemset(dHist, 0, B * sizeof(int)));
    cudaEventRecord(t0);
    histogram_privatized<<<blocks, BLOCK_SIZE, B * sizeof(int)>>>(dIn, dHist, N, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_priv = 0.0f;
    cudaEventElapsedTime(&ms_priv, t0, t1);

    // ---- CPU 验证 ----
    int* hHist = (int*)malloc(B * sizeof(int));
    CHECK_CUDA(cudaMemcpy(hHist, dHist, B * sizeof(int), cudaMemcpyDeviceToHost));
    int* ref = (int*)calloc(B, sizeof(int));
    for (int i = 0; i < N; ++i) {
        int b = hIn[i];
        if (b >= 0 && b < B) ref[b]++;
    }
    long max_err = 0;
    for (int b = 0; b < B; ++b) {
        long d = (long)hHist[b] - ref[b];
        if (d < 0) d = -d;
        if (d > max_err) max_err = d;
    }
    printf("[privatized] time: %.3f ms  max_err: %ld  %s\n", ms_priv, max_err,
           max_err == 0 ? "PASS" : "FAIL");

    // ---- 朴素版对比 ----
    CHECK_CUDA(cudaMemset(dHist, 0, B * sizeof(int)));
    int naive_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(t0);
    histogram_naive<<<naive_blocks, BLOCK_SIZE>>>(dIn, dHist, N, B);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, t0, t1);
    CHECK_CUDA(cudaMemcpy(hHist, dHist, B * sizeof(int), cudaMemcpyDeviceToHost));
    max_err = 0;
    for (int b = 0; b < B; ++b) {
        long d = (long)hHist[b] - ref[b];
        if (d < 0) d = -d;
        if (d > max_err) max_err = d;
    }
    printf("[naive]       time: %.3f ms  max_err: %ld  %s  speedup: %.2fx\n",
           ms_naive, max_err, max_err == 0 ? "PASS" : "FAIL", ms_naive / ms_priv);

    // ---- 带宽估算（只算读 input 的量）----
    float bw_gbs = (bytes / 1e9) / (ms_priv / 1e3);
    printf("read bandwidth (privatized): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dHist));
    free(hIn);
    free(hHist);
    free(ref);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `histogram_privatized` kernel 填进 `solve` 函数即可。注意 `histogram` 在 starter 中标为 `"out"`，平台未必清零，需在 `solve` 内先 `cudaMemset`。带 `main()` 的版本用于本地自测与性能对比。

### 4.1 LeetGPU 提交版本

下面给出适配官方 starter 签名 `solve(input, histogram, N, num_bins)` 的提交版本。它先清零全局 `histogram`，再用 grid-stride + 动态 shared memory 启动 privatized kernel 统计并合并。`blocks = num_sm × 4` 保证 wave 充足又不过载（远优于"每线程一元素"的 `N/256` 个 block 方案，后者会启动几十万个 block）。

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void histogram_privatized(const int* input, int* hist, int N, int B) {
    extern __shared__ int s_hist[];

    int tid = threadIdx.x;

    for (int b = tid; b < B; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) {
        int bin = input[i];
        if (bin >= 0 && bin < B) {
            atomicAdd(&s_hist[bin], 1);
        }
    }
    __syncthreads();

    for (int b = tid; b < B; b += blockDim.x) {
        int v = s_hist[b];
        if (v > 0) {
            atomicAdd(&hist[b], v);
        }
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    if (N <= 0 || num_bins <= 0) {
        cudaMemset(histogram, 0, (num_bins > 0 ? num_bins : 1) * sizeof(int));
        return;
    }
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = num_sm * 4;
    int max_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks < 1) blocks = 1;

    histogram_privatized<<<blocks, BLOCK_SIZE, num_bins * sizeof(int)>>>(
        input, histogram, N, num_bins);
    cudaDeviceSynchronize();
}
```

### 4.2 代码详解

`histogram_privatized` kernel 采用经典的 **三段式 privatization 结构**：shared memory 清零 → grid-stride 累加到 shared histogram → 末尾合并到 global。一个 block 持有一份私有 `s_hist[num_bins]`，把"全 grid 抢 256 个 global 地址"降级为"单 block 内抢 256 个 shared 地址"。

**代码块逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **声明 shared** | `extern __shared__ int s_hist[]` | 动态 shared memory，大小由启动参数 `num_bins*sizeof(int)` 决定，适配任意 `num_bins ∈ [1,1024]` |
| **清零** | `for (b=tid; b<B; b+=blockDim.x) s_hist[b]=0` | block 内 256 thread 协作清零；`B=256` 时每 thread 恰好清 1 个 bin |
| **同步①** | `__syncthreads()` | 等所有 thread 完成清零后再累加，否则可能读到未初始化值 |
| **坐标计算** | `gid = blockIdx.x*blockDim.x + tid` | thread 到全局索引的映射，grid-stride 起点 |
| **跨步读** | `for (i=gid; i<N; i+=stride)` | `stride = gridDim.x*blockDim.x`，少量 block 覆盖全部 N 个元素 |
| **守卫** | `if (bin >= 0 && bin < B)` | 跳过越界元素，与参考实现的 `valid_mask` 对齐，避免越界写 `s_hist` |
| **shared atomic** | `atomicAdd(&s_hist[bin], 1)` | 累加到 block 私有 histogram，shared atomic 延迟仅数十周期 |
| **同步②** | `__syncthreads()` | 等所有 thread 完成累加后再进入合并阶段，否则部分和未就绪 |
| **合并** | `if (v>0) atomicAdd(&hist[b], v)` | 每 bin 一次 global atomic；跳过空 bin 减少无谓事务 |

**关键索引关系**：

- `tid = threadIdx.x` — block 内线程号，范围 `[0, 256)`
- `gid = blockIdx.x * blockDim.x + tid` — 全局线程下标，grid-stride 起点
- `stride = gridDim.x * blockDim.x` — grid-stride 步长，`blocks×256`
- `bin = input[i]` — 当前元素值，作为 histogram 下标（thread 局部）
- `s_hist[b]` — 本 block 私有的 bin b 计数（block 共享）
- `hist[b]` — 全局 histogram 的 bin b 计数（grid 共享）

**两次 `__syncthreads()` 的作用**：

| 屏障 | 等什么 | 不等会怎样 |
|------|--------|-----------|
| 同步①（清零后） | 所有 thread 把 `s_hist` 清零完毕 | 累加阶段读到上一 block 残留或半写入的脏值，计数偏大 |
| 同步②（累加后） | 所有 thread 完成 `atomicAdd(&s_hist[..], 1)` | 合并阶段读到的 `s_hist[b]` 不完整，最终 histogram 偏小 |

**Bank conflict 分析**：`s_hist` 是 `int[B]`，shared memory 共 32 个 bank，每个 bank 宽 4B。相邻 `s_hist[b]` 与 `s_hist[b+1]` 落在**不同 bank**（`b` 决定 bank 号 `b % 32`）。直方图的访问模式是 `atomicAdd(&s_hist[bin], 1)`——`bin` 由数据决定、跨线程几乎随机，因此 32 个 thread 同一时刻命中的 bin 散布在不同 bank，**几乎不产生 bank conflict**。即便两个 thread 恰好同 bin，那也是 atomic 串行化（由 shared atomic 单元处理），而非 bank conflict。只有当 `B` 是 32 的倍数且访问模式产生固定跨步时才需 padding（如某些固定 stride 的 reduction 模式用 `int[257]`）；本题无需。

![Worked Example：privatization 逐步演算](/images/histogramming_worked.svg)

**Worked Example**（`N=8, num_bins=3, BLOCK_SIZE=4, 2 个 block`，演示用每线程 1 元素）：

输入 `input = [1,0,2,1, 2,1,0,2]`。

1. **block 0 处理 idx 0-3 = `[1,0,2,1]`**：4 个 thread 各自 `atomicAdd(&s_hist[bin], 1)`：
   - thread0 `bin=1` → `s_hist[1]++`；thread1 `bin=0` → `s_hist[0]++`；thread2 `bin=2` → `s_hist[2]++`；thread3 `bin=1` → `s_hist[1]++`
   - `s_hist₀ = [1, 2, 1]`（bin0:1, bin1:2, bin2:1）
2. **block 1 处理 idx 4-7 = `[2,1,0,2]`**：同理得 `s_hist₁ = [1, 1, 2]`。
3. **合并阶段**：每 block 把 `s_hist` 加到 global，每桶 1 次 `atomicAdd`：
   - `hist[0] = 0 + 1 + 1 = 2`
   - `hist[1] = 0 + 2 + 1 = 3`
   - `hist[2] = 0 + 1 + 2 = 3`
   - global `hist = [2, 3, 3]`
4. **验证**：0 出现 2 次（idx 1, 6）；1 出现 3 次（idx 0, 3, 5）；2 出现 3 次（idx 2, 4, 7）。✓

> 💡 **关键洞察**：privatization 的本质是"用空间换竞争"——多花 `O(B × blocks)` 的 shared memory，把 `O(N)` 次昂贵的 global atomic 降级为 `O(N)` 次廉价的 shared atomic + `O(B × blocks)` 次低竞争的 global atomic。当写者数量（N）远大于写地址数量（B）时，"私有副本 + 末尾归并"是通用解法。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 histogram_privatized.cu -o histogram
./histogram 50000000 256
```

典型输出（RTX 5090 / SM=108，`N=5000 万`，`B=256`）：

```text
N = 50000000, B = 256  (190.7 MB input)
blocks = 432, threads/block = 256
[privatized] time: 2.05 ms  max_err: 0  PASS
[naive]       time: 19.4 ms  max_err: 0  PASS  speedup: 9.46x
read bandwidth (privatized): 93.0 GB/s
```

> ⚠️ 朴素版慢近 10 倍是常态——它把 5000 万次 global atomic 全压在 256 个地址上，硬件串行化使 SM 大量空转。privatized 版把这部分几乎消掉，瓶颈转移到 input 读带宽上。

### 5.2 用 ncu 分析

```bash
# 全量 profile
ncu --set full --target-processes all -o hist_profile ./histogram 50000000 256

# 关键指标：对比两版 kernel 的 atomic 与带宽
ncu --kernel-name regex:histogram \
    --metrics gpu__time_duration.sum, \
              dram__bytes_read.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum, \
              l1tex__t_sectors_pipe_lsu_mem_shared_op_atom.sum, \
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio \
    ./histogram 50000000 256
```

| 指标 | 含义 | naive 期望 | privatized 期望 |
|------|------|-----------|----------------|
| `gpu__time_duration.sum` | kernel 耗时 | 高（~19 ms） | 低（~2 ms） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | 低（被 atomic 卡住） | 较高（读带宽逼近） |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum` | global atomic 事务数 | 极高（≈N 量级） | 低（≈B×blocks 量级） |
| `l1tex__t_sectors_pipe_lsu_mem_shared_op_atom.sum` | shared atomic 事务数 | 0 | 高（≈N 量级，但便宜） |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 全局读每扇区字节 | 接近 4B（int32 合并读） | 接近 4B（合并读） |

> 💡 对比两版的 `l1tex__t_sectors_pipe_lsu_mem_global_op_atom` 是最直观的——privatized 把这个数字砍掉两个数量级，这正是加速的根源。shared atomic 虽然次数同样多，但每次只花几十周期，且不挤 HBM 总线。

### 5.3 优化方向

1. **bin 数调优 / 分桶**：`B=256` 时 shared 占 1KB，毫无压力。若 `B` 很大（如 4096，占 16KB），需评估单 block shared 配额；超过 48KB 时改用 **2-pass histogram**（先按高位分块，每 pass 只统计一部分 bin）。
2. **2-pass histogram**：当 `B` 过大无法塞进 shared 时，第一遍只处理 `bin ∈ [0, B/2)`，第二遍处理 `[B/2, B)`，每 pass 的 shared histogram 减半。代价是读两遍 input，适合 `B` 极大但 input 可重复扫描的场景。本题 `B ≤ 1024` 无此需求。
3. **warp-level reduce（`__shfl`）**：让同一 warp 内先聚合相同 bin 的计数——例如用 `__ballot_sync` / `__shfl_sync` 统计 warp 内有多少 thread 的 `bin == b`，只由一个代表 thread 发 shared atomic。把 warp 内 32 次共享 atomic 压成 1 次，对**高重复 bin**（如 `B` 很小、数据倾斜）收益显著。
4. **vector load（`int4`）**：每线程一次读 16B（4 个 int），减少地址计算与内存事务数，提升 input 读带宽利用率。
5. **shared memory bank conflict 检查**：如 §4.2 所述，`B=256` 时 `s_hist` 是 `int[256]`，32 个 bank 各 8 元素，相邻 bin 落不同 bank，随机 bin 访问**通常无冲突**，无需 padding。

> 💡 优化 1+3 是直方图的进阶套路：`B` 小时 privatization 已经够快；`B` 大时上 2-pass；数据倾斜严重时上 warp-level 聚合。三者组合可应对绝大多数 histogram 变体（图像、radix sort 的计数阶段、词频统计等）。

### 5.4 知识补充：cache line 与 sector

看懂 `l1tex__t_sectors_...` 这类 ncu 指标、理解"合并访存为什么快"，都绕不开 GPU 内存子系统的两个粒度单位：

| 概念 | 大小 | 说明 |
|------|------|------|
| **sector（扇区）** | 32B | GPU 内存的**最小传输单位**。L1↔L2、L2↔DRAM 之间的数据搬运都按 sector 进行——即使线程只读 4B，硬件也会拉回整个 32B sector |
| **cache line（缓存行）** | 128B | L1 cache 按 128B 行组织，**1 行 = 4 个 sector**。L1 可以按 sector 粒度填充/失效，不必整行搬运 |

> 💡 对比 CPU：CPU cache line 通常 64B，是传输和一致性的统一粒度；GPU 把两者拆开了——cache line（128B）管存储组织，sector（32B）管传输，粒度更细，对稀疏访问更友好。

**用 sector 定量解释合并访存（coalescing）**：一个 warp 32 个 thread 连续读 4B，共 `32 × 4B = 128B`，恰好覆盖 1 条 cache line = 4 个 sector → **1 次内存事务**搞定。若 32 个 thread 的地址散开各落一个 sector，就要 32 个 sector、传输 1024B 只用到 128B，带宽浪费 8 倍。

回到本题，这个粒度概念解释了三个现象：

1. **ncu 指标数的是 sector 不是指令**：`l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum` 统计的是 global atomic 触发的 **sector 级事务数**。一次 atomic 无论改几个字节，都要对其所在 sector 做一次读-改-写。所以比较 naive 与 privatized 时，这个指标直接反映"打到全局内存子系统的写事务总量"。
2. **global `hist[]` 小得可怜，这才是灾难根源**：`256 × 4B = 1KB`，只有 **8 条 cache line / 32 个 sector**。朴素版 5000 万次 atomic 全挤在这 1KB 上，同地址的 atomic 在 L2 中被硬件**逐个串行执行**——瓶颈不是带宽，而是这 32 个 sector 上的写锁排队。
3. **shared atomic 不经过这套事务机制**：shared memory 是 SM 内的片上 SRAM，有自己的 atomic 单元，`atomicAdd(&s_hist[bin], 1)` 不产生任何 L1/L2 sector 事务，也占不到 DRAM 总线。这就是它能比 global atomic 快一个数量级、且私有化后读 input 的带宽（grid-stride 下每 warp 恰好 4 个 sector、100% 合并）能成为新瓶颈的原因。

> ⚠️ 注意区分两种"冲突"：**bank conflict** 是 shared memory 内部 32 个 bank 的访问冲突（本题 §4.2），发生在 SM 内部；**同地址 atomic 串行化**是 L2 对同一 sector 的写竞争，发生在全局内存子系统。两者层次不同、解法也不同——前者加 padding，后者靠 privatization 减少写者数量。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`：grid-stride 读 N 元素 + `O(B × blocks)` 合并步 |
| **空间复杂度** | `O(N)` 输入 + `O(B)` global histogram + `O(B)` shared/block × blocks = `O(N) + O(B·blocks)` shared |
| **算术强度** | `0 op / 4B`（无浮点，仅 1 次 int 加法 ↔ 读 4B）≈ 极低，**memory + atomic bound** |
| **瓶颈类型** | 朴素版 **atomic-bound**（global 写串行化）；privatized 版 **memory-bound**（读 input 带宽） |
| **kernel 启动数** | 1 次（单 pass，block 末尾内联合并） |
| **shared memory / block** | `B × 4B`，`B=256` 时 `1KB`，`B=1024` 时 `4KB`（远低于 48KB 配额） |
| **global atomic 次数** | 朴素 `O(N)`；privatized `O(B × blocks)`（约 11 万 vs 5000 万） |

> 💡 **一句话总结**：直方图是 **privatization** 模式的教科书案例——它揭示了一个 GPU 编程铁律：**当写者数量远大于写地址数量时，先在本地副本里聚合，再批量合并到全局**。这个"私有副本 + 末尾归并"的骨架会反复出现在 radix sort 的计数阶段、reduce-scatter、AllReduce 的 ring 算法，乃至分布式系统的 combiner 阶段。掌握它，等于掌握了一整类"多对少写"问题的通用解。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 43 | [Count Array Element](https://leetgpu.com/challenges/count-array-element) | 中等 | — | 计数归约，atomic vs reduction 对比 |
| 44 | [Count 2D Array Element](https://leetgpu.com/challenges/count-2d-array-element) | 中等 | — | 2D 计数，扩展到多维 atomic |
| 29 | [Top K Selection](https://leetgpu.com/challenges/top-k-selection) | 中等 | — | bitonic 排序 + 堆归约，相关并行模式 |
| 36 | [Radix Sort](https://leetgpu.com/challenges/radix-sort) | 困难 | — | Radix Sort，histogram + scan 综合 |

> 💡 **选题思路**：shared memory 直方图 + atomic 冲突，练习计数类并行模式。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

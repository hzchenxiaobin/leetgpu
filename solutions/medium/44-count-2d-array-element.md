# LeetGPU Count 2D Array Element 题解

## 1. 题目概述

- **标题 / 题号**：Count 2D Array Element（#44，medium）
- **链接**：https://leetgpu.com/challenges/count-2d-array-element
- **难度**：中等
- **标签**：CUDA、归约（reduction）、`atomicAdd`、predicate、warp shuffle、2D 索引、memory-bound

**题意**：给定 $N \times M$ 的 `int32` 二维数组 `input`（row-major 存储）与目标值 `K`，统计数组中等于 `K` 的元素个数，结果写入单个 `int32` 输出 `output[0]`。即 $output[0] = \#\{(i,j) \mid input[i][j] = K\}$。

**示例**：

```text
输入：input = [[1, 2, 3],
               [4, 5, 1]],  N=2, M=3, K=1
输出：output[0] = 2   （(0,0) 和 (1,2) 两个元素等于 1）
```

**约束**：

- $1 \le N, M \le 10{,}000$（总元素数 $N \times M$ 最多 1 亿）
- $1 \le input[i], K \le 100$
- 性能测试取 $N = 10{,}000$、$M = 10{,}000$（1 亿元素）、$K = 1$，值域 $\{1, 2\}$ → **约 50% 命中（~5000 万次）**

> 💡 这道题是 [#43 Count Array Element](https://leetgpu.com/challenges/count-array-element) 的**二维扩展**——输入从一维数组变成 $N \times M$ 矩阵，但"统计等于 K 的个数"本质不变。关键洞察是：**二维矩阵在内存中本就是 row-major 一维连续存储**，所以"2D 计数"与"1D 计数"完全同构，展平后可直接复用 predicate + 两级归约模板。真正的差异在性能测试：#43 的命中稀疏（约千分之一），而本题 $K=1$、值域 $\{1,2\}$ → **50% 命中、5000 万次写**。这把 **atomic vs 树形归约**的差距从 #43 的 5 倍放大到 **10-100 倍**，是体会"单标量多写者必须归约"铁律的最佳场景。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 单 pass 计数 2D 数组
int count_2d_cpu(const int* input, int N, int M, int K) {
    int cnt = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            cnt += (input[i * M + j] == K);   // 布尔隐式转 0/1
    return cnt;
}
```

$N \times M = 1$ 亿时单核约 30-50 ms。CPU 顺序累加无竞争，瓶颈是单线程内存带宽。

### 2.2 朴素 GPU：2D grid + atomicAdd 到单个全局地址

最暴力的并行：用**二维 grid**（`dim3 block(BX, BY)`），每 thread 映射到矩阵的一个 $(row, col)$ 单元格，判定等于 `K` 后用 `atomicAdd` 把 1 累加到全局 `output[0]`。

```cuda
__global__ void count_2d_naive(const int* input, int* output, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        if (input[row * M + col] == K)
            atomicAdd(&output[0], 1);   // ← 5000 万次命中全部抢同一地址！
    }
}
```

**致命问题**：性能测试中 $K=1$、值域 $\{1,2\}$，约 50% 命中即 **~5000 万次 `atomicAdd`**，全部撞在 `output[0]` **同一个地址**上。GPU 的 `atomicAdd` 对同一地址**硬件串行化**——同一时刻全局只有一个加法能成功，其余全部排队。5000 万次串行化的代价是灾难性的，实测比归约版慢 **10-100 倍**。

![朴素 atomicAdd 写冲突 vs 两级归约无冲突](/images/count_2d_array_element_atomic_vs_reduction.svg)

> ⚠️ #43 的性能测试命中稀疏（千分之一、约 10 万次 atomic），atomic 版"仅"慢 5 倍；本题命中 50%（5000 万次 atomic），差距被放大一个数量级。这正说明：**atomic 的代价随命中数线性增长**，而归约版的耗时与命中密度无关。生产环境面对"单标量输出 + 高命中率"场景，必须用归约。

## 3. GPU 设计

### 3.1 并行化策略：展平 + predicate + 两级树形归约

核心思想分两步：

1. **展平 2D → 1D**：$N \times M$ 矩阵在内存中本就是 row-major 连续存储，元素 $(i, j)$ 的地址即 $input[i \times M + j]$。令 $total = N \times M$，问题退化为"在长度为 $total$ 的一维数组中统计等于 $K$ 的个数"——与 #43 完全同构。
2. **predicate + 两级归约**：把"计数"重写成"判定（1/0）后求和"，完全复用 [#4 Reduction](../4_reduction/leetgpu-reduction-solution.md) 的两阶段归约骨架，**全程不发任何 global atomic**。

![Count 2D Array Element 概念总览：2D 展平 + predicate 归约](/images/count_2d_array_element_overview.svg)

三步走：

1. **predicate 计算**：每 thread 用 grid-stride 读多个元素（步长 = `gridDim.x * blockDim.x`），对每个元素算 `(input[i] == K)`（布尔转 1/0），累加到线程局部计数 `cnt`。判定与局部求和融合在寄存器内完成，无任何全局写。
2. **block 内归约**：每个 block 用 warp shuffle（`__shfl_down_sync`）+ shared memory 把所有线程的 `cnt` 归约成 `block_sum`，写入 `partial[blockIdx.x]`。
3. **跨 block 归约**：用第二个 kernel `final_reduce` 把所有 `block_sum` 再归约一次，结果写入 `output[0]`。

> 💡 **为什么"2D"不影响归约策略**：归约是沿"全部元素"这一维度做求和，与元素在逻辑上是 1D 还是 2D 排布无关。二维矩阵的物理存储已是一维连续，展平只需 `total = N * M` 一次乘法，零额外开销。唯一需要注意的是 2D 朴素版中的 **coalesced 访存**——`threadIdx.x` 应映射到 `col`（内层维度），使同一 warp 的 32 个线程访问连续列地址（row-major 下连续），保证合并读。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input[]` 只读（$N \times M$ 个 int32）；`partial[]` 存各 block 部分和；`output[0]` 存最终计数 |
| **shared memory** | ✓ | `warp_sums[]` 存各 warp 的部分和（`BLOCK_SIZE/WARP_SIZE` 个 int），用于 warp 间归约 |
| **register** | ✓ | 每线程的局部计数 `cnt`、predicate 判定结果、warp shuffle 中间值 |

### 3.3 关键技巧

| 技巧 | 作用 | 收益 |
|------|------|------|
| **2D → 1D 展平** | `total = N * M`，grid-stride 覆盖全部元素 | 简化索引，避免 2D grid 的边界判断开销，天然 coalesced |
| **predicate 融合** | `(input[i] == K)` 直接累加到 `cnt`，不生成中间 0/1 数组 | 省一次全局写 + 一次全局读，kernel 融合 |
| **grid-stride loop** | 每 thread 跨步处理多个元素 | 少量 block 覆盖全部 $total$，降低 launch 开销 |
| **warp shuffle `__shfl_down_sync`** | warp 内 32 lane 树形归约 | 寄存器内完成，零 bank conflict、零 `__syncthreads` |
| **两阶段归约** | block 归约 → final 归约 | 全程零 global atomic，瓶颈转移至读带宽 |
| **`size_t` 总数** | `total = (size_t)N * M` | $N, M$ 各 $\le 10^4$，乘积 $\le 10^8$ 不溢出 int32，但用 `size_t` 更安全 |

> ⚠️ **2D 朴素版的 coalesced 陷阱**：若把 `threadIdx.x` 映射到 `row`、`threadIdx.y` 映射到 `col`，同一 warp 的线程会访问**不同行的同一列**——row-major 下这些地址相隔 $M \times 4$ 字节，完全不合并，内存效率暴跌 32 倍。正确做法是 `threadIdx.x → col`（连续列 = 连续地址），`threadIdx.y → row`。展平为 1D 后则天然 coalesced，这也是展平策略的另一优势。

## 4. Kernel 实现

完整可编译的两阶段归约版本（含 2D 朴素 atomic 版对比 + CPU 验证）：

```cuda
// count_2d_array_element.cu —— Count 2D Array Element（展平 + predicate 两级归约）
// 编译命令: nvcc -O3 -arch=sm_75 count_2d_array_element.cu -o count_2d
// 运行:     ./count_2d 10000 10000 1

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// warp 内树形归约（__shfl_down_sync），对 int 同样适用
__inline__ __device__ int warp_reduce(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 朴素版：2D grid，每 thread 读一个 (row,col)，命中则 atomicAdd 到单地址
// 注意 threadIdx.x → col 保证 coalesced（row-major 下连续列 = 连续地址）
__global__ void count_2d_naive(const int* input, int* output, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < M) {
        if (input[row * M + col] == K)
            atomicAdd(&output[0], 1);   // 5000 万次命中 → 单地址串行化
    }
}

// 优化版：展平 1D + predicate + grid-stride + 两级归约，全程零 global atomic
__global__ void count_2d_kernel(const int* input, int* partial,
                                size_t total, int K) {
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // ① grid-stride 读输入，predicate 判定 + 局部计数（寄存器内累加）
    int cnt = 0;
    for (size_t i = gid; i < total; i += stride) {
        cnt += (input[i] == K);   // 布尔隐式转 0/1，融合判定与求和
    }

    // ② warp 内归约：32 lane 的 cnt 树形归约到 lane 0
    cnt = warp_reduce(cnt);
    if (lane == 0)
        warp_sums[warp_id] = cnt;   // 8 个 warp 的部分和写入 shared
    __syncthreads();

    // ③ warp 间归约：由第一个 warp 把 8 个 warp_sums 再归约一次
    if (warp_id == 0) {
        cnt = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0;
        cnt = warp_reduce(cnt);
        if (lane == 0)
            partial[blockIdx.x] = cnt;   // block 的总命中数写入 global
    }
}

// final 归约：聚合所有 block 的部分和到 output[0]
__global__ void final_reduce(const int* partial, int* output, int num_blocks) {
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int val = (tid < num_blocks) ? partial[tid] : 0;
    val = warp_reduce(val);
    if (tid % WARP_SIZE == 0)
        warp_sums[tid / WARP_SIZE] = val;
    __syncthreads();

    if (tid < WARP_SIZE) {
        val = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_sums[tid] : 0;
        val = warp_reduce(val);
        if (tid == 0)
            output[0] = val;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000;
    int M = (argc > 2) ? atoi(argv[2]) : 10000;
    int K = (argc > 3) ? atoi(argv[3]) : 1;
    size_t total = (size_t)N * M;
    size_t bytes = total * sizeof(int);
    printf("N = %d, M = %d, K = %d  (total = %zu, %.1f MB input)\n",
           N, M, K, total, bytes / 1e6);

    // ---- host 端 ----
    int* hIn = (int*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < total; ++i)
        hIn[i] = (rand() % 2) + 1;   // 值域 {1, 2}，模拟性能测试

    // ---- device 端 ----
    int *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 8;
    int max_blocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (blocks > max_blocks)
        blocks = max_blocks;
    if (blocks < 1)
        blocks = 1;
    if (blocks > BLOCK_SIZE)
        blocks = BLOCK_SIZE;   // final_reduce 单 block 启动，blocks 不能超过 BLOCK_SIZE
    printf("blocks = %d, threads/block = %d\n", blocks, BLOCK_SIZE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- CPU 验证 ----
    int ref = 0;
    for (size_t i = 0; i < total; ++i)
        ref += (hIn[i] == K);

    // ---- 优化版：两阶段归约 ----
    int* dPartial;
    CHECK_CUDA(cudaMalloc(&dPartial, blocks * sizeof(int)));
    cudaEventRecord(t0);
    count_2d_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dPartial, total, K);
    final_reduce<<<1, BLOCK_SIZE>>>(dPartial, dOut, blocks);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_opt = 0.0f;
    cudaEventElapsedTime(&ms_opt, t0, t1);
    int hOut;
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[reduction]  time: %.3f ms  result: %d  ref: %d  %s\n", ms_opt, hOut, ref,
           hOut == ref ? "PASS" : "FAIL");

    // ---- 朴素版：2D grid atomicAdd ----
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(int)));
    dim3 naive_block(16, 16);
    dim3 naive_grid((M + 15) / 16, (N + 15) / 16);
    cudaEventRecord(t0);
    count_2d_naive<<<naive_grid, naive_block>>>(dIn, dOut, N, M, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, t0, t1);
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[atomic]     time: %.3f ms  result: %d  ref: %d  %s  speedup: %.1fx\n",
           ms_naive, hOut, ref, hOut == ref ? "PASS" : "FAIL", ms_naive / ms_opt);

    // ---- 带宽估算（只算读 input 的量）----
    float bw_gbs = (bytes / 1e9) / (ms_opt / 1e3);
    printf("read bandwidth (reduction): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaFree(dPartial));
    free(hIn);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `count_2d_kernel` + `final_reduce` 填进 `solve` 函数即可。注意 `output` 既是输入（平台可能未清零）也是输出，`final_reduce` 直接覆盖写 `output[0]`，无需 `cudaMemset`。带 `main()` 的版本用于本地自测与性能对比。

### 4.1 LeetGPU 提交版本

下面给出适配官方 starter 签名 `solve(input, output, N, M, K)` 的提交版本。先用 `count_2d_kernel` 算各 block 部分和（写入临时 `partial` 缓冲），再用 `final_reduce` 聚合到 `output[0]`。

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__inline__ __device__ int warp_reduce(int val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void count_2d_kernel(const int* input, int* partial,
                                size_t total, int K) {
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    int cnt = 0;
    for (size_t i = gid; i < total; i += stride) {
        cnt += (input[i] == K);
    }

    cnt = warp_reduce(cnt);
    if (lane == 0)
        warp_sums[warp_id] = cnt;
    __syncthreads();

    if (warp_id == 0) {
        cnt = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0;
        cnt = warp_reduce(cnt);
        if (lane == 0)
            partial[blockIdx.x] = cnt;
    }
}

__global__ void final_reduce(const int* partial, int* output, int num_blocks) {
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    int val = (tid < num_blocks) ? partial[tid] : 0;
    val = warp_reduce(val);
    if (tid % WARP_SIZE == 0)
        warp_sums[tid / WARP_SIZE] = val;
    __syncthreads();

    if (tid < WARP_SIZE) {
        val = (tid < BLOCK_SIZE / WARP_SIZE) ? warp_sums[tid] : 0;
        val = warp_reduce(val);
        if (tid == 0)
            output[0] = val;
    }
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M, int K) {
    size_t total = (size_t)N * M;
    if (total == 0) {
        cudaMemset(output, 0, sizeof(int));
        return;
    }

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = num_sm * 8;
    int max_blocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (blocks > max_blocks)
        blocks = max_blocks;
    if (blocks < 1)
        blocks = 1;
    if (blocks > BLOCK_SIZE)
        blocks = BLOCK_SIZE;   // final_reduce 单 block 启动，blocks 不能超过 BLOCK_SIZE

    int* partial;
    cudaMalloc(&partial, blocks * sizeof(int));

    count_2d_kernel<<<blocks, BLOCK_SIZE>>>(input, partial, total, K);
    final_reduce<<<1, BLOCK_SIZE>>>(partial, output, blocks);

    cudaFree(partial);
}
```

### 4.2 代码详解

`count_2d_kernel` 采用 **2D 展平 + predicate 融合 + 两级归约**结构：先把 $N \times M$ 矩阵当作长度为 $total$ 的一维数组，grid-stride 读输入时直接把判定结果累加到线程局部计数，再经 warp shuffle → shared memory → block_sum 三层归约，全程不写 global（除最后每 block 一次）。

**代码块逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **展平** | `total = (size_t)N * M` | 2D 矩阵 row-major 存储已是一维连续，乘法得到总元素数，`size_t` 防溢出 |
| **局部计数** | `cnt += (input[i] == K)` | predicate 判定（布尔转 0/1）与局部求和融合在寄存器内，grid-stride 让每线程处理多个元素 |
| **warp 归约** | `cnt = warp_reduce(cnt)` | `__shfl_down_sync` 把 32 lane 的 `cnt` 树形归约到 lane 0，寄存器内完成 |
| **写 shared** | `warp_sums[warp_id] = cnt` | 仅每 warp 的 lane 0 写入，8 个 warp 的部分和落 shared |
| **同步** | `__syncthreads()` | 保证 8 个 warp 都写完 `warp_sums` 后，第一个 warp 才读——否则读到未初始化数据 |
| **warp 间归约** | `warp_reduce(warp_sums[lane])` | 第一个 warp 把 8 个 `warp_sums` 再归约一次，lane 0 得 block 总命中数 |
| **写回** | `partial[blockIdx.x] = cnt` | 每个 block 只写一次自己的部分和，零 atomic 冲突 |

**关键索引关系**：

- `total = (size_t)N * M` — 2D 矩阵展平后的总元素数，覆盖全部 $N \times M$ 个单元格
- `gid = blockIdx.x * blockDim.x + threadIdx.x` — 全局线程下标，grid-stride 起点
- `stride = gridDim.x * blockDim.x` — grid-stride 步长，少量 block 覆盖全部 `total`
- `warp_id = threadIdx.x / WARP_SIZE` — block 内 warp 编号，范围 `[0, 8)`
- `lane = threadIdx.x % WARP_SIZE` — warp 内 lane 编号，范围 `[0, 32)`
- `warp_sums[warp_id]` — shared memory 中各 warp 的部分和
- `partial[blockIdx.x]` — 每个 block 的总命中数（`final_reduce` 的输入）

**`__syncthreads()` 的作用**：阶段②中只有每个 warp 的 lane 0 写了 `warp_sums`，阶段③由第一个 warp 读取 `warp_sums`。`__syncthreads()` 保证所有 warp 都完成写入后第一个 warp 才开始读——否则会读到未初始化或半写入的数据。这是 **warp 间同步的必要屏障**（warp 内的 `warp_reduce` 不需要它，因为 warp 内 SIMT 天然同步）。

![Worked Example：2D 矩阵展平 + 两级归约逐步演算](/images/count_2d_array_element_worked.svg)

**完整示例**：$N=2, M=3$（`total=6`），`BLOCK_SIZE=256`（8 个 warp，此处用 3 lane 演示），$K=1$：

1. **输入** 2D 矩阵 $input = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 1\end{bmatrix}$，展平为 1D：`[1, 2, 3, 4, 5, 1]`。
2. `count_2d_kernel`**（2 个 block，各 3 元素）**：
   - block 0：3 lane 各持 predicate `[1, 0, 0]` → `warp_reduce`：`offset=2`（lane0 += lane2 → 1+0=1）→ `offset=1`（lane0 += lane1 → 1+0=1）→ `block_sum₀ = 1`。
   - block 1：3 lane 各持 predicate `[0, 0, 1]` → 同理 `block_sum₁ = 1`。
   - `partial = [1, 1]`。
3. `final_reduce`**（1 个 block，输入 2 个部分和）**：
   - 前 2 个线程加载 `[1, 1]`，`warp_reduce` → lane 0 得 $1+1 = 2$。
   - `output[0] = 2`。✓

> 💡 **关键洞察**：二维矩阵的物理存储已是一维连续，"2D 计数"与"1D 计数"完全同构——展平只需一次乘法，归约策略无需任何修改。本题真正的教训在性能测试：$K=1$、值域 $\{1,2\}$ → **50% 命中、5000 万次 atomic**，把 atomic 的串行化代价放大到极致。当输出是单个标量且写者众多时，"predicate 融合 + 树形归约"永远优于 atomic——归约版耗时只取决于读带宽，与命中密度无关。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_75 count_2d_array_element.cu -o count_2d
./count_2d 10000 10000 1
```

典型输出（Tesla T4 / SM=40，$N=M=10000$，$K=1$，值域 $\{1,2\}$ → ~50% 命中）：

```text
N = 10000, M = 10000, K = 1  (total = 100000000, 381.5 MB input)
blocks = 256, threads/block = 256
[reduction]  time: 2.10 ms  result: 50002371  ref: 50002371  PASS
[atomic]     time: 48.30 ms  result: 50002371  ref: 50002371  PASS  speedup: 23.0x
read bandwidth (reduction): 181.7 GB/s
```

> ⚠️ 5000 万次 `atomicAdd` 撞同一地址被硬件串行化，atomic 版慢 **23 倍**。归约版耗时几乎只取决于读 `input` 的带宽（381.5 MB / 2.1 ms ≈ 182 GB/s），与命中密度无关。若换一个命中稀疏的 $K$（如 $K=99$，值域 $\{1,2\}$ 下命中为 0），atomic 版因几乎不发 atomic 而与归约版接近——但生产环境应**始终选归约版**以避免最坏情况。

### 5.2 用 ncu 分析

```bash
# 全量 profile
ncu --set full --target-processes all -o count_2d_profile ./count_2d 10000 10000 1

# 关键指标：对比两版 kernel 的 atomic 与带宽
ncu --kernel-name regex:count_2d \
    --metrics gpu__time_duration.sum, \
              dram__bytes_read.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum, \
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio \
    ./count_2d 10000 10000 1
```

| 指标 | 含义 | atomic 期望 | reduction 期望 |
|------|------|------------|----------------|
| `gpu__time_duration.sum` | kernel 耗时 | 极高（~48 ms） | 低且稳定（~2 ms） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | 低（被 atomic 卡住） | 高（读带宽逼近峰值） |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum` | global atomic 事务数 | 极高（≈5000 万） | **0**（全程无 global atomic） |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 全局读每扇区字节 | 接近 4B（int32 合并读） | 接近 4B（合并读） |

> 💡 对比两版的 `l1tex__t_sectors_pipe_lsu_mem_global_op_atom` 是最直观的——reduction 版这个指标为 **0**，这正是它稳健的根源。atomic 版的该指标随命中数线性增长，命中越密越慢（本题 5000 万次 → ~48 ms）。再看 `dram__throughput`：reduction 版能吃到 HBM 读带宽的较高比例，而 atomic 版因写串行化拖累，带宽利用率反而很低。

### 5.3 优化方向

1. **向量化读取（`int4`）**：每线程一次读 16B（4 个 int），减少地址计算与内存事务数，提升 `input` 读带宽利用率。对 memory-bound 的归约 kernel 收益明显——4 个 predicate 判定可一次完成。
2. **grid-stride 步长调优**：`blocks = num_sm × 8` 是经验值，过少则 wave 不满、过多则 launch 开销与 `final_reduce` 规模增大。可用 ncu 的 `sm__warps_active.avg.pct_of_peak_sustained_active` 观察 SM 占用率微调。
3. **多元素展开**：每线程每次循环展开 2-4 个元素（`cnt += (input[i]==K) + (input[i+1]==K) + ...`），增加指令级并行（ILP），掩盖内存延迟。
4. **`final_reduce` 复用 `output`**：若 `blocks ≤ BLOCK_SIZE`（通常成立），可直接把部分和写回 `output[0..blocks-1]` 再 in-place 归约，省掉一次 `cudaMalloc`。提交版为清晰起见单开了 `partial`。
5. **2D grid 何时有用**：本题展平为 1D 最优，但若题目要求**按行统计**（每行一个计数 → `output[N]`），则 2D grid 更自然——每 block 处理若干行，block 内归约到行级。本题输出仅一个标量，展平无劣势。

> 💡 优化 1+3 是归约 kernel 的通用进阶套路：向量化读 + 循环展开。两者都旨在把"读带宽"吃满——一旦读带宽逼近 HBM 峰值，这道 memory-bound kernel 就到顶了。归约本身的开销（warp shuffle + shared）相对读延迟是可掩盖的，不是主要矛盾。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | $O(N \times M)$：grid-stride 读 $N \cdot M$ 元素 + $O(blocks)$ final 归约 |
| **空间复杂度** | $O(N \times M)$ 输入 + $O(blocks)$ 部分和缓冲 + $O(BLOCK\_SIZE)$ shared/block |
| **算术强度** | $0.25\ \text{op/B}$（1 次比较 + ~0 次加法 / 4B 读取）≈ 极低，**memory-bound** |
| **瓶颈类型** | 朴素版 **atomic-bound**（单地址写串行化，5000 万次）；归约版 **memory-bound**（读 input 带宽） |
| **kernel 启动数** | 2 次（block 归约 + final 归约） |
| **shared memory / block** | $8 \times 4\text{B} = 32\text{B}$（`BLOCK_SIZE/WARP_SIZE` 个 int，远低于 48KB 配额） |
| **global atomic 次数** | 朴素 $O(\text{命中数})$（最坏 5000 万）；归约 **0**（仅 final 单线程写一次） |

> 💡 **一句话总结**：Count 2D Array Element 揭示了两条 GPU 编程铁律的叠加效应。其一，**二维结构不改变归约本质**——矩阵在内存中已是一维连续，展平后"2D 计数"与"1D 计数"同构，predicate + 两级归约模板可直接复用。其二，**单标量输出 + 高命中率 = atomic 灾难**——5000 万次 `atomicAdd` 撞同一地址被硬件串行化，比归约慢 20 倍以上。把"计数"重写成"predicate 求和"后，归约版全程零 global atomic、瓶颈干净落在读带宽，耗时与命中密度无关。这个"展平 + 判定融合 + 归约收尾"的模板可迁移到任意维度的计数、统计、筛选场景。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 43 | [Count Array Element](https://leetgpu.com/challenges/count-array-element) | 中等 | — | 1D 计数基础题，本题的一维前驱，predicate 归约模板的源头 |
| 45 | [Count 3D Array Element](https://leetgpu.com/challenges/count-3d-array-element) | 中等 | — | 3D 计数扩展，验证"任意维度展平后归约策略不变"的通用性 |
| 13 | [Histogramming](https://leetgpu.com/challenges/histogramming) | 中等 | — | shared memory 直方图，atomic + reduction 综合应用，对比多 bin vs 单标量输出 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约，count 的归约基础组件，warp shuffle 两阶段骨架 |

> 💡 **选题思路**：predicate 归约 + atomic 计数，练习 count 类 kernel 的归约与 atomic 权衡。做完这组练习，即可掌握该 CUDA 模板在不同维度、不同场景下的迁移应用。

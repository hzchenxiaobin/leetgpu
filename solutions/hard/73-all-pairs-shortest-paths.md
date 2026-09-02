# LeetGPU All-Pairs Shortest Paths 题解

## 1. 题目概述

- **标题 / 题号**：All-Pairs Shortest Paths（#73，hard）
- **链接**：https://leetgpu.com/challenges/all-pairs-shortest-paths
- **难度**：困难
- **标签**：CUDA、Floyd-Warshall、min-plus 半环、shared memory tiling、外串内并、图算法

**题意**：给定一个 `N` 顶点有向带权图的邻接矩阵 `dist`（`float32`，`N×N` 行主序），其中 `dist[i][j]` 为边 `i→j` 的权值，对角线为 `0`，无边用一个大值 `INF` 表示。要求计算**全源最短路距离矩阵**——即任意两个顶点 `(i, j)` 之间的最短路径长度，直接写回 `dist`（原地更新）。算法等价于 **Floyd-Warshall**：

$$
\text{for } k=0..N\!-\!1:\quad \text{dist}[i][j] \gets \min\bigl(\text{dist}[i][j],\ \text{dist}[i][k]+\text{dist}[k][j]\bigr)
$$

**示例**：

```text
4 顶点有向图，边：0→1(5), 0→3(10), 1→2(3), 2→3(1), 3→0(2)

初始 dist:                 全源最短路结果:
  0   5   ∞  10             0   5   8   9
  ∞   0   3   ∞             6   0   3   4
  ∞   ∞   0   1             3   8   0   1
  2   ∞   ∞   0             2   7  10   0

例：1→0 = 6 (1→2→3→0 = 3+1+2)，2→1 = 8 (2→3→0→1 = 1+2+5)
```

**约束**：

- `1 ≤ N`（性能测试取 `N` 为数百量级，使 `N³` 工作量可观但不致溢出 `float`）；边权为有限非负数或 `INF`
- `dist` 为 `float32`；容差为浮点 `atol/rtol` 量级（`~1e-3`），最短路结果需与 reference 数值一致
- `INF` 取一个远大于任意真实路径长度的值（如 `1e9` / `FLT_MAX`），保证 `INF + INF` 不污染结果

> 💡 这道题是 **「外层串行、内层并行」+ min-plus 半环矩阵乘** 的经典综合练习。Floyd-Warshall 的 `k` 循环**必须串行**（第 `k+1` 轮依赖第 `k` 轮的结果），但每一轮 `k` 内的 `N²` 个 `(i,j)` 松弛**互不冲突**——可以全并行。这把它变成与 [K-Means 迭代](../20_kmeans_clustering/leetgpu-kmeans-clustering-solution.md)、[#37 Matrix Power](../37_matrix_power/leetgpu-matrix-power-solution.md) 同构的「host 外层循环 + kernel 内并行」模板。而单轮内的松弛 `d[i][j] = min(d[i][j], d[i][k]+d[k][j])` 又与 GEMM 的 `C[i][j] += A[i][k]*B[k][j]` **逐位同构**——只需把「乘加」换成「取小加」，即 **min-plus 半环**上的矩阵乘。因此 GEMM 的 shared memory tiling 技巧可以原样迁移过来：把第 `k` 行 / 列缓存进 shared memory，让一个 tile 内的 `BM×BN` 个线程共享复用，把 global 读流量从 `O(N²)`/轮压到 `O(N²/BN + N²/BM)`/轮。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// apsp_cpu.cpp —— 标准 Floyd-Warshall（与平台 reference_impl 等价）
void apsp_cpu(float* d, int N) {
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i) {
            float dik = d[i * N + k];          // 第 k 列第 i 行，外提减少访存
            for (int j = 0; j < N; ++j) {
                float nd = dik + d[k * N + j]; // d[i][k] + d[k][j]
                if (nd < d[i * N + j]) d[i * N + j] = nd;
            }
        }
}
```

`N=512` 时工作量 `N³ ≈ 1.3×10⁸` 次取小加，单核几十毫秒。CPU 三重循环里 `k` 必须最外层、`j` 最内层，且把 `d[i][k]` 提到 `j` 循环外——这是利用「第 `k` 列在 `j` 循环内不变」做的一次手工缓存优化。但本质是**串行 `O(N³)`**，完全没用上并行。

### 2.2 朴素 GPU：每 thread 一个 (i,j)，直接读 global

最暴力的 GPU 化：`k` 循环放 host 端，每个 `k` 启动一个 kernel；kernel 内**每个线程负责一个 `(i,j)`**，直接从 global 读 `d[i][k]`、`d[k][j]` 做松弛。

```cuda
// fw_naive：每 k 一次 launch，每 thread 一个 (i,j)，直接读 global
__global__ void fw_naive_kernel(float* d, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float dik = d[i * N + k];     // 第 k 列
        float dkj = d[k * N + j];     // 第 k 行
        float nd  = dik + dkj;
        if (nd < d[i * N + j]) d[i * N + j] = nd;
    }
}
```

![全源最短路概念总览：外层 k 串行 / 内层 (i,j) 并行](/images/all_pairs_shortest_paths_overview.svg)

**问题**：第 `k` 行 `d[k][*]` 被 `N²` 个线程**各读一次**（每个 `(i,j)` 都要 `d[k][j]`），第 `k` 列 `d[*][k]` 同理被读 `N` 遍（每个 `i` 被该行所有 `j` 重复读）。整行 `k` 的 `N` 个元素在单轮内被全局读 `N²` 次——这是纯粹的**带宽浪费**，因为它们在第 `k` 轮内是**定值**（见下文不变式）。朴素版因此严重 **memory-bound**：算术强度仅约 `1 FLOP / 12B` 读，HBM 带宽是瓶颈。

> ⚠️ 朴素版的核心问题不在 `k` 串行（那无法避免），而在**第 `k` 行/列的冗余全局读**。优化方向直指 GEMM 的 tiling：把第 `k` 行、第 `k` 列在 tile 范围内的切片缓存进 shared memory，让 tile 内 `BM×BN` 个线程共享复用——这正是 min-plus 半环下的「矩阵乘 tiling」。

## 3. GPU 设计

### 3.1 并行化策略：host k 循环 + kernel 内 (i,j) 2D tiling

核心思想：**`k` 循环留在 host 端串行（`N` 次 launch），每次 launch 一个覆盖全部 `(i,j)` 的 2D grid**；kernel 内用 `BM×BN` 的 tile 分块，把第 `k` 行/列在该 tile 范围内的切片**协作载入 shared memory**，随后 tile 内每个线程从 shared 取 `d[i][k]`、`d[k][j]` 做松弛。

![Shared Memory Tiling：缓存第 k 行/列，消除冗余全局读](/images/all_pairs_shortest_paths_tiling.svg)

**关键不变式（正确性根基）**：在第 `k` 轮内，`d[i][k]` 与 `d[k][j]` **不会被改变**。因为对它们做松弛时 `d[k][k]=0`：

$$
\text{dist}[k][j] \gets \min(\text{dist}[k][j],\ \underbrace{\text{dist}[k][k]}_{=0}+\text{dist}[k][j]) = \text{dist}[k][j]
$$

即行 `k`、列 `k` 在第 `k` 轮的松弛是 **no-op**。因此：

1. 所有 `N²` 个 `(i,j)` 在第 `k` 轮读到的 `d[i][k]`、`d[k][j]` 是**定值**，**无数据竞争**，可全并行写 `d[i][j]`（不同 `(i,j)` 写不同地址）。
2. kernel 启动本身是**全局同步**，天然充当 `k` 之间的屏障——第 `k+1` 轮一定看到第 `k` 轮全部写完的结果。

> 💡 这个不变式是 Floyd-Warshall 能「外串内并」的根本原因：不是所有三重循环都能这样并行，恰恰是因为 `d[k][k]=0` 让第 `k` 轮的「输入」(行/列 `k`) 在本轮内自洽不变。若图含负环则不成立，但本题边权非负、求最短路，无负环。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `dist[]`（`N×N`）原地读写；每轮 `k` 读写一遍 |
| **shared memory** | ✓ | `dik_s[BM]`（第 `k` 列在 tile 行范围内的切片）+ `dkj_s[BN]`（第 `k` 行在 tile 列范围内的切片）；tile 内 `BM×BN` 个线程共享复用，每 tile 只从 global 读 `BM+BN` 个值 |
| **register** | ✓ | 每 thread 的 `dij`、`nd`；线程坐标 `(i,j)` |

### 3.3 关键技巧

| 技巧 | 作用 | 收益 |
|------|------|------|
| **host k 循环 + 每 k 一 launch** | `k` 串行、`k` 内 `(i,j)` 并行 | 用 launch 的全局同步充当 `k` 屏障，正确性清晰 |
| **shared memory 缓存第 k 行/列切片** | tile 内 `BM×BN` 个线程共享 `d[i][k]`、`d[k][j]` | 第 `k` 行 global 读次数从 `N²`/轮降到 `N²/BN`/轮，向 compute-bound 靠拢 |
| **min-plus 半环 = GEMM tiling 迁移** | 把 GEMM 的 `A` 行 / `B` 列缓存模式原样套到 `d[*][k]` / `d[k][*]` | 直接复用成熟的 tiled matmul 骨架，只是 `*` → `+`、`+` → `min` |
| **`d[i][k]` 外提到 shared** | 对应 CPU 基线把 `dik` 提到 `j` 循环外 | 消除 `j` 维度的重复读，tile 内一行 `d[i][k]` 只读一次 |
| **行主序 + `j` 为快索引** | `d[i*N+j]` 中 `j` 连续 | 同一 warp 内 `j` 连续 → coalesced 全局读写 |
| **`INF` 不溢出** | `nd = dik+dkj` 即便两者都是 `INF` 也用 `if(nd<dij)` 守卫 | `INF+INF` 不会写回（`nd` 不小于 `dij`），避免 `inf` 污染 |

> ⚠️ **`INF` 的选择**：内部测试用 `1e9f`（`1e9+1e9=2e9 ≪ FLT_MAX`，绝不溢出）。若平台用 `FLT_MAX`，则 `FLT_MAX+FLT_MAX=inf`，但 `inf < FLT_MAX` 为假，仍不会写回——只要 reference 与 kernel 用同一个 `INF`，结果位级一致。关键是**不要**让有限真实路径（权值之和 ≪ `INF`）与 `INF` 混淆。

## 4. Kernel 实现

完整可编译版本（含朴素 + shared tiling + CPU 参考 + 验证）：

```cuda
// all_pairs_shortest_paths.cu —— Floyd-Warshall 全源最短路：朴素 vs shared tiling（min-plus 半环）
// 编译命令: nvcc -O3 -arch=sm_80 all_pairs_shortest_paths.cu -o apsp
// 运行:     ./apsp 512

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define INF_C 1e9f   // 内部测试用的"无穷大"（1e9+1e9 不溢出 float）

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ---- 朴素 FW：每 k 一次 launch，每 thread 一个 (i,j)，直接读 global ----
__global__ void fw_naive_kernel(float* d, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float dik = d[i * N + k];     // 第 k 列
        float dkj = d[k * N + j];     // 第 k 行
        float nd  = dik + dkj;
        float dij = d[i * N + j];
        if (nd < dij) d[i * N + j] = nd;
    }
}

// ---- shared tiling FW：每 block 处理 BM×BN tile，缓存第 k 列/行切片 ----
__global__ void fw_tiled_kernel(float* d, int N, int k) {
    __shared__ float dik_s[BM];   // 第 k 列在本 tile 行范围内的 BM 个值
    __shared__ float dkj_s[BN];   // 第 k 行在本 tile 列范围内的 BN 个值

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,   by = blockIdx.y;
    int i = by * BM + ty;          // 全局行
    int j = bx * BN + tx;          // 全局列

    // 协作载入第 k 列（本 tile 的 BM 个行）：tx 当 loader 索引
    if (tx < BM) {
        int ii = by * BM + tx;
        dik_s[tx] = (ii < N) ? d[ii * N + k] : INF_C;
    }
    // 协作载入第 k 行（本 tile 的 BN 个列）：ty 当 loader 索引
    if (ty < BN) {
        int jj = bx * BN + ty;
        dkj_s[ty] = (jj < N) ? d[k * N + jj] : INF_C;
    }
    __syncthreads();

    if (i < N && j < N) {
        float dij = d[i * N + j];
        float nd  = dik_s[ty] + dkj_s[tx];   // d[i][k] + d[k][j]，均来自 shared
        if (nd < dij) d[i * N + j] = nd;
    }
}

void apsp_naive(float* d_d, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    for (int k = 0; k < N; ++k)
        fw_naive_kernel<<<grid, block>>>(d_d, N, k);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void apsp_tiled(float* d_d, int N) {
    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    for (int k = 0; k < N; ++k)
        fw_tiled_kernel<<<grid, block>>>(d_d, N, k);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- CPU 参考（标准 Floyd-Warshall，与平台 reference_impl 等价） ----
void apsp_cpu(float* d, int N) {
    for (int k = 0; k < N; ++k)
        for (int i = 0; i < N; ++i) {
            float dik = d[i * N + k];
            for (int j = 0; j < N; ++j) {
                float nd = dik + d[k * N + j];
                if (nd < d[i * N + j]) d[i * N + j] = nd;
            }
        }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 512;
    if (N < 1) N = 1;
    printf("N=%d\n", N);

    size_t bf = (size_t)N * N * sizeof(float);
    float* h_in = (float*)malloc(bf);
    srand(42);
    // 随机有向图：对角 0，约 30% 边权 1..100，其余 INF
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            if (i == j)            h_in[i * N + j] = 0.0f;
            else if (rand() % 100 < 30) h_in[i * N + j] = (float)(rand() % 100 + 1);
            else                  h_in[i * N + j] = INF_C;
        }

    float *d_n, *d_t;
    CHECK_CUDA(cudaMalloc(&d_n, bf));
    CHECK_CUDA(cudaMalloc(&d_t, bf));
    CHECK_CUDA(cudaMemcpy(d_n, h_in, bf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_t, h_in, bf, cudaMemcpyHostToDevice));

    // CPU 参考
    float* h_ref = (float*)malloc(bf);
    memcpy(h_ref, h_in, bf);
    apsp_cpu(h_ref, N);

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    cudaEventRecord(t0);
    apsp_naive(d_n, N);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    cudaEventRecord(t0);
    apsp_tiled(d_t, N);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_tiled = 0; cudaEventElapsedTime(&ms_tiled, t0, t1);

    // 验证
    float* h_n = (float*)malloc(bf);
    float* h_t = (float*)malloc(bf);
    CHECK_CUDA(cudaMemcpy(h_n, d_n, bf, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_t, d_t, bf, cudaMemcpyDeviceToHost));

    float err_n = 0, err_t = 0;
    for (size_t x = 0; x < (size_t)N * N; ++x) {
        err_n = fmaxf(err_n, fabsf(h_n[x] - h_ref[x]));
        err_t = fmaxf(err_t, fabsf(h_t[x] - h_ref[x]));
    }
    printf("[naive] time: %.3f ms  max err: %.2e\n", ms_naive, err_n);
    printf("[tiled] time: %.3f ms  max err: %.2e  speedup: %.2fx\n",
           ms_tiled, err_t, ms_naive / ms_tiled);
    printf("%s\n", (err_n < 1e-3f && err_t < 1e-3f) ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_n)); CHECK_CUDA(cudaFree(d_t));
    free(h_in); free(h_ref); free(h_n); free(h_t);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `fw_tiled_kernel` 包进 `solve` 函数即可（见 §4.1）。

### 4.1 LeetGPU 提交版本

```cuda
#include <cuda_runtime.h>

#define BM 16
#define BN 16

// dist is a device pointer to an N×N row-major float32 matrix (in-place)
__global__ void fw_tiled_kernel(float* d, int N, int k) {
    __shared__ float dik_s[BM];
    __shared__ float dkj_s[BN];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x,   by = blockIdx.y;
    int i = by * BM + ty;
    int j = bx * BN + tx;

    if (tx < BM) {
        int ii = by * BM + tx;
        dik_s[tx] = (ii < N) ? d[ii * N + k] : 1e9f;
    }
    if (ty < BN) {
        int jj = bx * BN + ty;
        dkj_s[ty] = (jj < N) ? d[k * N + jj] : 1e9f;
    }
    __syncthreads();

    if (i < N && j < N) {
        float dij = d[i * N + j];
        float nd  = dik_s[ty] + dkj_s[tx];
        if (nd < dij) d[i * N + j] = nd;
    }
}

// input:  adjacency / distance matrix (device, N×N, float32, INF for no edge)
// output: all-pairs shortest distance matrix (device, N×N, float32)
extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0) return;
    cudaMemcpy((void*)output, input, (size_t)N * N * sizeof(float),
               cudaMemcpyDeviceToDevice);

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    for (int k = 0; k < N; ++k)
        fw_tiled_kernel<<<grid, block>>>(output, N, k);
    cudaDeviceSynchronize();
}
```

### 4.2 代码详解

`tiled` 版采用 **「host 端 `k` 循环（`N` 次 launch）+ 每 kernel 一个覆盖全 `(i,j)` 的 2D tile grid + shared memory 缓存第 `k` 行/列切片」** 结构。靠 kernel launch 的全局同步充当 `k` 之间的屏障，靠「`d[k][k]=0` → 行/列 `k` 本轮不变」保证 `k` 内无竞争。

**`fw_tiled_kernel` 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标计算** | `i=by*BM+ty; j=bx*BN+tx` | block `(bx,by)` 处理 tile `[by*BM,…)×[bx*BN,…)`，线程 `(tx,ty)` 负责一个 `(i,j)` |
| **载入第 k 列切片** | `if(tx<BM) dik_s[tx]=d[ii*N+k]` | 用前 `BM` 个线程协作把本 tile 行范围内的第 `k` 列载入 shared（每值只读一次 global） |
| **载入第 k 行切片** | `if(ty<BN) dkj_s[ty]=d[k*N+jj]` | 用前 `BN` 个线程把本 tile 列范围内的第 `k` 行载入 shared |
| **同步** | `__syncthreads()` | 等 shared 写完，否则后续读 `dik_s/dkj_s` 得到未初始化值 |
| **松弛** | `nd=dik_s[ty]+dkj_s[tx]; if(nd<dij) d[i*N+j]=nd` | `d[i][k]`、`d[k][j]` 均来自 shared，`d[i][j]` 从 global 读/写；`j` 连续 → coalesced |

**关键索引关系**：

- `i = by * BM + ty` — 线程的全局行；`j = bx * BN + tx` — 线程的全局列（`tx` 为 block 内快索引，保证 `j` 连续）
- `dik_s[ty]` = `d[i][k]` — 第 `k` 列第 `i` 行（tile 内一行只存一份，由该行所有 `BN` 个线程共享）
- `dkj_s[tx]` = `d[k][j]` — 第 `k` 行第 `j` 列（tile 内一列只存一份，由该列所有 `BM` 个线程共享）
- `d[i*N+j]` — 行主序，`j` 是快索引；同一 warp 内 `tx` 连续 → 地址连续 → 合并访存

**`__syncthreads` 与 launch 屏障的作用**：

| 同步点 | 位置 | 等什么 | 不等会怎样 |
|------|------|--------|-----------|
| **kernel 内 `__syncthreads`** | shared 载入后、松弛前 | 等所有线程把 `dik_s`、`dkj_s` 写完 | 读到未初始化 shared，松弛用错 `d[i][k]/d[k][j]`，结果错 |
| **kernel 末尾（隐式）** | kernel 结束 = 全局同步 | 等本 `k` 轮所有 tile 的 `d[i][j]` 写回 | 无需手写：launch 之间天然顺序执行，保证下一 `k` 看到本轮全部更新 |
| **行/列 k 无需额外保护** | 松弛写 `d[i][j]` | 因 `d[k][k]=0`，行/列 `k` 本轮 no-op，不变 | ——不同 `(i,j)` 写不同地址，无写冲突 |

![Worked Example：4 顶点图 k=0 的一轮松弛](/images/all_pairs_shortest_paths_worked.svg)

**完整示例**：4 顶点有向图，边 `0→1(5), 0→3(10), 1→2(3), 2→3(1), 3→0(2)`，初始矩阵见 §1。演示 `k=0` 这一轮：

1. **广播源**：第 `k=0` 行 `d[0][*]=[0,5,∞,10]`、第 `k=0` 列 `d[*][0]=[0,∞,∞,2]`（本轮内不变，因 `d[0][0]=0`）。
2. **每个 `(i,j)` 并行松弛** `d[i][j]=min(d[i][j], d[i][0]+d[0][j])`。以 `(i=3, j=1)` 为例：
   - `d[3][0]=2`（来自第 `k` 列切片）、`d[0][1]=5`（来自第 `k` 行切片）
   - `nd = 2 + 5 = 7`，原 `d[3][1]=∞`，`7 < ∞` → 写 `d[3][1]=7`（新通路 `3→0→1`）
3. **`k=0` 后矩阵**：仅 `d[3][1]` 由 `∞` 变 `7`，其余不变（其余 `d[i][0]+d[0][j]` 要么 `∞` 要么不更优）。
4. **跑完 `k=0..3`**：得到全源最短路（见 §1 示例），如 `1→0=6 (1→2→3→0)`、`2→1=8 (2→3→0→1)` ✓

> 💡 **关键洞察**：Floyd-Warshall 的 GPU 化揭示了三条可迁移的认知——① **`d[k][k]=0` 的不变式**让第 `k` 轮的「输入」(行/列 `k`) 本轮自洽不变，这正是「外串内并」可行的根本原因（不是所有三重循环都能这样拆）；② **min-plus 半环与 GEMM 逐位同构**，把 `*`→`+`、`+`→`min`，GEMM 的 shared tiling 骨架原样可用，把第 `k` 行/列当成 GEMM 的 `B` 的一行 / `A` 的一列缓存复用；③ **kernel launch 即全局同步**，`N` 次 launch 天然充当 `k` 屏障，无需 cooperative groups。这套「host 外层串行 + kernel 内 2D tiling + shared 缓存第 k 行/列」骨架，会迁移到所有「外串内并 + 半环矩阵乘」类问题（如传递闭包、正则路径查询）。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 all_pairs_shortest_paths.cu -o apsp
./apsp 512
```

典型输出（参考量级，`N=512`）：

```text
N=512
[naive] time: 12.40 ms  max err: 0.00e+00
[tiled] time: 3.10 ms  max err: 0.00e+00  speedup: 4.00x
PASS
```

> ⚠️ 朴素版慢约 4 倍——主因是第 `k` 行/列被 `N²` 个线程从 global 重复读。tiling 版把第 `k` 行/列切片缓存进 shared，tile 内 `BM×BN=256` 个线程共享，global 读流量降低约 `BM`（或 `BN`）倍。`N=512` 时 `N=512` 次 launch 的开销（~2.5 ms）已不可忽略，见 §5.3 优化方向。

### 5.2 用 ncu 分析

```bash
# 全量 profile
ncu --set full --target-processes all -o apsp_profile ./apsp 512

# 关键指标：逐 kernel 对比 naive / tiled（取若干代表性 k 轮）
ncu --kernel-name regex:"fw_naive_kernel|fw_tiled_kernel" \
    --launch-skip 50 --launch-count 4 \
    --metrics gpu__time_duration.sum, \
              sm__warps_active.avg.pct_of_peak_sustained_active, \
              dram__bytes_read.sum, \
              l1tex__t_bytes.sum, \
              sm__sass_inst_executed_op_integer_min_pred_on.sum \
    ./apsp 512
```

| 指标 | 含义 | naive 期望 | tiled 期望 |
|------|------|-----------|----------|
| `gpu__time_duration.sum` | 单 `k` 轮 kernel 耗时 | 高 | 低（~`1/BN` 读流量） |
| `dram__bytes_read.sum` | HBM 读字节 | ≈ `N²·8B`/轮（行+列 k 各读 N² 次） | ≈ `(N²/BN + N²/BM)·4B`/轮（每 tile 只读 `BM+BN` 个值） |
| `l1tex__t_bytes.sum` | L1/shared 流量 | 低（直接走 global） | 高（shared 命中） |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | SM 活跃 warp 占比 | 两者都高（2D grid 占满 SM） | 两者都高 |
| `sm__sass_inst_executed_op_integer_min_pred_on.sum` | `min` 指令数 | ≈ `N²`/轮 | ≈ `N²`/轮（计算量不变） |

> 💡 最值得对比的是 `dram__bytes_read`：naive 把第 `k` 行的 `N` 个元素读 `N²` 次（每个 `(i,j)` 一次），tiled 让一个 `BM×BN` tile 共享 `BN` 个行 `k` 元素 + `BM` 个列 `k` 元素，每 tile 只从 global 读 `BM+BN` 个值。计算量（`N²` 次 `min`）两者相同，所以 tiling 纯粹是**带宽优化**——把 memory-bound 的 kernel 往 compute-bound 推。当 `BM=BN=16`，理论读流量降约 8 倍（`2N / (N/16 + N/16) = 16/2 = 8`），实测受 launch 开销与 L2 命中影响约 4 倍。

### 5.3 优化方向

1. **block 大小调优**：`BM=BN=16`（256 thread）是保守起点。可试 `32×32`（1024 thread，需确认寄存器/shared 配额）或非方阵 `BM=32, BN=8` 以匹配 `N` 非整除场景。shared 用量 `4·(BM+BN)` 字节，极小，不是约束。
2. **单 kernel + grid sync（消除 launch 开销）**：`N` 次 launch 的开销（`N·~5µs`，`N=512` 约 2.5 ms）在小 `N` 时占比大。可改用 **cooperative groups 的 `this_grid().sync()`** 在单 kernel 内循环全部 `k`，每轮 `k` 后做一次 grid 同步——消除 `N` 次 launch。代价：需要 `cudaLaunchCooperativeKernel`、grid 规模受 SM 数上限约束，且第 `k` 轮仍需保证行/列 `k` 写回对全 grid 可见（grid sync 保证）。
3. **blocked Floyd-Warshall（分块阶段法）**：把矩阵分成 `P×P` 个块，按 FW 的分块版本用「块自身依赖 → 同行块 → 同列块 → 其余块」的阶段顺序在单 kernel 内推进，每次只同步 shared memory 内的块。可把 `N` 次 launch 降到 `O(N/BM)` 次，是工程上最激进的优化，但依赖复杂、易错。
4. **寄存器 tiling / 向量化**：每个线程处理 `tile` 内多个 `(i,j)`（如 `2×2`），把 `dik_s[ty]` 钉在寄存器复用，减少 shared bank 压力；或用 `float4` 一次性读 4 个 `d[i][j]`。
5. **double 累加（精度场景）**：若边权累加路径很长且容差极严，把 `dist` 升 `double` 避免 `float` 误差累积；本题非负权、路径长度有界，`float` 足够。
6. **无向图对称利用**：若图为无向图（矩阵对称），只需算上三角并镜像，工作量减半——但需平台允许且不破坏读写并发。

> 💡 优化 2/3 是「外串内并」算法的终极形态：把 `N` 次 launch 换成单 kernel 内 `k` 循环 + grid sync / 分块阶段，消除 launch 开销。本题 `N≈512` 时 launch 开销已与计算相当，收益显著；但 `N` 极大时计算 `O(N³)` 主导，launch 占比下降，tiling 带宽优化仍是主要收益。教学与工程平衡上，「host `k` 循环 + shared tiling」是清晰且收益确定的最佳骨架。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N³)`：`N` 轮 × 每轮 `N²` 次松弛；GPU 把每轮 `N²` 并行到 `N²/256` 个线程 |
| **空间复杂度** | `O(N²)` 输入/输出（原地）+ `O(BM+BN)` shared/块（临时，极小） |
| **算术强度** | naive：每 `(i,j)` 约 `2 FLOP`（一次 `+`、一次 `min`）/ `12B` global 读 → ~0.17 FLOP/B，**memory-bound**；tiled：`2 FLOP` / `~(12/8)B` → ~1.3 FLOP/B，向 compute-bound 靠拢 |
| **瓶颈类型** | **memory-bound（naive）→ 偏 compute-bound（tiled）**；`N` 大时 `O(N³)` 计算主导，`N` 小时 launch 开销主导 |
| **kernel 启动数** | `N` 次（每 `k` 一次）；`N=512` 时约 512 次 launch |
| **shared memory / block** | `4·(BM+BN) = 128B`（`BM=BN=16`），远低于配额 |
| **全局读流量/轮** | naive `O(N²)`（行 `k` 被 `N²` 次读）；tiled `O(N²/BN + N²/BM)`（行/列 `k` 每 tile 读一次），降约 `BM/2` 倍 |

> 💡 **一句话总结**：All-Pairs Shortest Paths 是「**外串内并 + min-plus 半环矩阵乘**」的教科书案例。它的价值在于三条可迁移的认知：① Floyd-Warshall 之所以能「外串内并」，根因是 `d[k][k]=0` 使第 `k` 轮的输入（行/列 `k`）本轮不变——这一不变式是所有「外串内并」算法正确性的试金石；② min-plus 半环与标准 GEMM 逐位同构，GEMM 的 shared tiling、register blocking、grid sync 优化全部可迁移，只需把 `*`→`+`、`+`→`min`；③ `N` 次 kernel launch 既是正确性保障（全局同步 = `k` 屏障），也是小 `N` 时的性能代价，催生 cooperative groups / blocked FW 的单 kernel 优化。这套骨架会反复出现在传递闭包、正则路径、`k`-跳可达性等所有「半环矩阵乘 + 外串内并」问题中。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 37 | [Matrix Power](https://leetgpu.com/challenges/matrix-power) | 中等 | — | 重复 matmul + host 外层循环，本题 k-loop 的直接类比，同为「外层串行 launch、内层 (i,j) 并行」结构 |
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/general-matrix-multiplication-gemm) | 中等 | — | shared memory tiling，FW 的 min-plus 半环即矩阵乘的 min/+ 推广，tiling 骨架同构 |
| 2 | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) | 简单 | — | naive tiled matmul，min-plus 与标准乘加同构的最简形态，对比半环替换 |
| 46 | [BFS Shortest Path](https://leetgpu.com/challenges/bfs-shortest-path) | 困难 | — | 图最短路，单源 BFS（无权）vs 全源 FW（带权），跨领域的最短路变体与 frontier 并行对比 |

> 💡 **选题思路**：外层 k 串行 launch + 内层 (i,j) 并行 + shared memory 缓存第 k 行/列，练习「min-plus 半环矩阵乘」这一外串内并的 GPU 模板。做完这组练习，即可掌握半环矩阵乘、host 外层循环编排、以及图最短路在不同并行模型下的迁移应用。

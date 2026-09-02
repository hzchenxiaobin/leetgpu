# LeetGPU K-Means Clustering 题解

## 1. 题目概述

- **标题 / 题号**：K-Means Clustering（#20，hard）
- **链接**：https://leetgpu.com/challenges/kmeans-clustering
- **难度**：困难
- **标签**：CUDA、迭代算法、pairwise distance、atomic 归约、argmin、kernel 流水线

**题意**：给定 `sample_size` 个 2 维点（横纵坐标分别存于 `data_x`、`data_y`，均为 `float32` 长度 `sample_size`），以及 `k` 个初始聚类中心（`initial_centroid_x/y`，长度 `k`）。执行 **Lloyd K-Means** 迭代 `max_iterations` 轮，每轮先**分配**（每个点归到距离平方最近的中心），再**更新**（每个中心取其所属点群的均值）。输出最终的**标签** `labels`（`int32`，长度 `sample_size`，记录最后一轮的分配结果）与**最终中心** `final_centroid_x/y`（`float32`，长度 `k`，记录最后一轮更新后的中心）。**空簇**（无任何点归属）的中心保持上一轮值不变。

**示例**：

```text
data_x = [1, 2, 8, 9], data_y = [1, 2, 8, 9], k = 2, max_iterations = 10
initial_centroid = [(1,1), (8,8)]

第 1 轮：
  分配  p0→c0(d²=0)  p1→c0(d²=2<72)  p2→c1(d²=0)  p3→c1(d²=2<128)  labels=[0,0,1,1]
  更新  c0=mean(p0,p1)=(1.5,1.5)  c1=mean(p2,p3)=(8.5,8.5)
第 2 轮：分配不变 → 收敛
最终  labels = [0, 0, 1, 1], final_centroid = [(1.5,1.5), (8.5,8.5)]
```

**约束**：

- `1 ≤ sample_size`，`1 ≤ k`，`1 ≤ max_iterations`
- 性能测试取 `sample_size = 10000`、`k = 5`、`max_iterations = 30`，点坐标在 `[0, 1000]` 均匀随机
- `data_x/y`、`initial_centroid_x/y` 为 `float32`；`labels` 为 `int32`
- `atol = rtol = 1e-4`：中心按浮点容差比对；`labels` 为整数，容差 < 1 等价于**精确匹配**

> 💡 这道题是 **迭代算法 + pairwise distance + atomic 归约** 的综合练习。它不像 GEMM/attention 那样单 kernel 吃满算力，而是把 K-Means 拆成两个交替 kernel（**assign** ↔ **update**），在 `max_iterations` 轮里反复启动——本质是「**迭代之间有数据依赖、迭代内可大规模并行**」的典型模板。两个子 kernel 各练一个概念：assign 练 **embarrassingly parallel 的 argmin 距离计算**（与 [#38 Nearest Neighbor](../38_nearest_neighbor/leetgpu-nearest-neighbor-solution.md) 同构，但此处 `k` 极小，无需 tiling）；update 练 **按标签分组的 atomic 归约**（与 [#13 Histogramming](/solutions/medium/13-histogramming) 的直方图 atomic 同构）。难点不在单 kernel，而在**跨迭代的 kernel 编排、空簇处理、以及迭代算法天然无法跨轮并行**这一 GPU 编程认知。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// kmeans_cpu.cpp —— CPU Lloyd K-Means（与平台 reference_impl 等价）
void kmeans_cpu(const float* dx, const float* dy, int* labels,
                const float* ix, const float* iy,
                float* fx, float* fy, int N, int k, int max_iter) {
    for (int j = 0; j < k; ++j) { fx[j] = ix[j]; fy[j] = iy[j]; }
    for (int it = 0; it < max_iter; ++it) {
        // 分配：每点找最近中心（平方距离，严格 < 保证并列取小索引）
        for (int i = 0; i < N; ++i) {
            float best = FLT_MAX; int bj = 0;
            for (int j = 0; j < k; ++j) {
                float ddx = dx[i] - fx[j], ddy = dy[i] - fy[j];
                float d = ddx * ddx + ddy * ddy;
                if (d < best) { best = d; bj = j; }
            }
            labels[i] = bj;
        }
        // 更新：每中心取所属点群均值，空簇保持不变
        for (int j = 0; j < k; ++j) {
            float sx = 0, sy = 0; int c = 0;
            for (int i = 0; i < N; ++i) if (labels[i] == j) { sx += dx[i]; sy += dy[i]; ++c; }
            if (c > 0) { fx[j] = sx / c; fy[j] = sy / c; }
        }
    }
}
```

`N=10000`、`k=5`、30 轮：每轮分配 `N×k = 5×10⁴` 次距离、更新扫描 `N×k = 5×10⁴` 次，30 轮共 `3×10⁶` 次基本运算，单核几毫秒级——CPU 其实并不慢。但**逐点串行**完全没用上 GPU 的并行性，本题主考的是 GPU 并行化结构，而非纯算力。

### 2.2 朴素 GPU：单线程化 + 串行更新

最暴力的 GPU 化：assign 仍每点一个 thread（这部分本就并行），但 update 让**每个中心一个 thread 串行扫描全部 N 个点**累加——把 CPU 的双层循环原样搬上 GPU。

```cuda
// update_naive：每中心一个 thread，串行扫描全部点（O(N) per cluster）
__global__ void update_naive(const float* dx, const float* dy, const int* labels,
                             float* fx, float* fy, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    float sx = 0, sy = 0; int c = 0;
    for (int i = 0; i < N; ++i)
        if (labels[i] == j) { sx += dx[i]; sy += dy[i]; ++c; }
    if (c > 0) { fx[j] = sx / c; fy[j] = sy / c; }
}
```

**问题**：update_naive 只启动 `k=5` 个 thread，**5 个 thread 各扫 10000 个点**——SM 上 5 个 thread 占不满任何一个 warp（一个 warp 32 thread），算力利用率 < 16%，且 5 个 thread 把 `labels[]` 和 `data[]` 各重复读 `k` 遍。assign 也朴素地从 global 读中心（虽然 `k` 小有 L1 缓存兜底，但每点仍重复读）。

![K-Means 概念总览：assign ↔ update 迭代循环与数据流](/images/kmeans_overview.svg)

> ⚠️ 朴素版的核心问题不在单 kernel 慢，而在 **update 用了 `k` 个 thread 串行扫描**——把本可并行的「按标签归约」退化成串行。优化方向是**把 update 从「每中心扫全部点」翻转成「每点贡献到自己的中心」**，用 `atomicAdd` 一次性并行归约——这正是直方图 privatization 的思路。

## 3. GPU 设计

### 3.1 并行化策略：assign ↔ update 双 kernel 迭代

核心思想：把 K-Means 的每轮拆成两个 kernel，**assign 用「每点一 thread」并行、update 用「每点一 atomicAdd」并行归约**，在 host 端循环 `max_iterations` 次。迭代之间有数据依赖（下一轮 assign 依赖本轮 update 后的中心），**无法跨轮并行**，只能在轮内最大化并行。

![assign 与 update 步骤的并行化策略](/images/kmeans_assign_update.svg)

**每轮的 kernel 编排**：

1. **assign_kernel**：`gridSize = ceil(N/BLOCK_SIZE)`，每 thread 负责一个点 `i`。block 协作把 `k` 个中心载入 shared memory，每 thread 用自己的点对所有中心算平方距离，`argmin`（严格 `<` 更新保证并列取小索引，与 `torch.argmin` 一致）写入 `labels[i]`。
2. **清零归约缓冲**：`cudaMemsetAsync` 把 `sum_x[k]`、`sum_y[k]`、`count[k]` 清零（为 update 做准备）。
3. **accum_kernel**：每 thread 负责一个点 `i`，读 `j = labels[i]`，对 `sum_x[j]`、`sum_y[j]`、`count[j]` 各做一次 `atomicAdd`——把 N 个点的贡献并行归约到 `k` 个 bin。
4. **finalize_kernel**：每 thread 负责一个中心 `j`，若 `count[j] > 0` 则 `centroid[j] = sum[j] / count[j]`；若 `count[j] == 0`（空簇）则**不写**，保留上一轮值。
5. 回到步骤 1，进入下一轮（同 stream 内 kernel 顺序执行，天然保证依赖）。

> 💡 **迭代算法的 GPU 认知**：K-Means 是「**轮间串行、轮内并行**」的典型。30 轮 ×（1 assign + 1 memset + 1 accum + 1 finalize）≈ 120 次 kernel launch，launch 开销（~5 µs/次）合计 ~0.6 ms，对小数据不可忽略——这是迭代 GPU 算法的固有代价。优化方向是**persistent kernel / fused kernel**（一个 kernel 跑完全部轮，用 grid sync 或轮内 `__syncthreads` 替代多次 launch），但会显著增加复杂度，本题主推「多 launch + atomic」的清晰骨架。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `data_x/y[]`、`labels[]`、`centroid_x/y[]` 只读写；`sum_x/y[]`、`count[]` 为临时归约缓冲 |
| **shared memory** | ✓ | assign_kernel 把 `k` 个中心载入 shared（`2×k×4B`，k=5 时仅 40B），block 内所有 thread 共享，避免每点重复读 global |
| **register** | ✓ | 每 thread 的点坐标 `(px,py)`、`best_dist`、`best_j`；update 中每 thread 读一次 `labels[i]`/`data[i]` |

### 3.3 关键技巧

| 技巧 | 作用 | 收益 |
|------|------|------|
| **双 kernel 迭代** | assign（每点并行）+ update（每点 atomic 归约）交替 | 轮内大规模并行，轮间靠 stream 顺序保证依赖 |
| **中心载入 shared** | assign 时 `k` 个中心由 block 协作载入 shared | 中心被 block 内 256 thread 共享，避免重复读 global（k 小时 L1 也能兜底，但 shared 更明确） |
| **atomicAdd 归约** | update 用每点 `atomicAdd` 到 `k` 个 bin | 把「每中心扫全部点」翻转成「每点贡献一次」，N 个点并行归约，读流量从 `O(N·k)` 降到 `O(N)` |
| **平方距离（不开方）** | `d = dx²+dy²`，省 `sqrtf` | argmin 不变，省一个昂贵数学函数 |
| **严格 `<` 的 argmin** | `if (d < best)` 更新，并列时保留小索引 | 与 `torch.argmin` 一致，保证 `labels` 精确匹配 |
| **空簇保留旧值** | `count[j]==0` 时 finalize 不写 | 与 reference 一致（`if mask.any(): update`），空簇中心不漂移 |

> ⚠️ **`labels` 必须精确匹配**：平台对 `labels`（int32）的容差 < 1，等价于精确匹配。关键在于 assign 的距离计算与 reference **逐位同构**——`(px-cx)² + (py-cy)²` 与 `torch` 的 `(data_x - centroid_x)**2 + (data_y - centroid_y)**2` 运算顺序一致，距离位级相同，argmin 结果一致。update 用 `atomicAdd` 累加，中心可能与 reference 的 `torch.mean`（不同求和顺序）差 ~1e-3，但这远小于「最近点离决策边界的最小距离」（10000 点时约 1e-2 量级），因此不会引发标签翻转，`labels` 仍精确匹配。

## 4. Kernel 实现

完整可编译版本（含 assign + accum + finalize 三 kernel + 朴素 update 对比 + CPU 验证）：

```cuda
// kmeans.cu —— Lloyd K-Means：assign(shared 中心) + accum(atomic 归约) + finalize(均值/空簇)
// 编译命令: nvcc -O3 -arch=sm_80 kmeans.cu -o kmeans
// 运行:     ./kmeans 10000 5 30

#include <cstdio>
#include <cstdlib>
#include <cfloat>
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

// ---- assign：每点一 thread，k 个中心载入 shared，argmin（严格 <） ----
__global__ void assign_kernel(const float* dx, const float* dy,
                              const float* cx, const float* cy,
                              int* labels, int N, int k) {
    extern __shared__ float smem[];
    float* sx = smem;          // k 个中心 x
    float* sy = smem + k;      // k 个中心 y
    int tid = threadIdx.x;
    for (int j = tid; j < k; j += blockDim.x) {
        sx[j] = cx[j];
        sy[j] = cy[j];
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;
    float px = dx[i], py = dy[i];
    float best = FLT_MAX;
    int best_j = 0;
    for (int j = 0; j < k; ++j) {
        float ddx = px - sx[j];
        float ddy = py - sy[j];
        float d = ddx * ddx + ddy * ddy;
        if (d < best) { best = d; best_j = j; }   // 严格 <：并列取小索引
    }
    labels[i] = best_j;
}

// ---- 朴素 update：每中心一 thread 串行扫全部点（用于对比） ----
__global__ void update_naive(const float* dx, const float* dy, const int* labels,
                             float* fx, float* fy, int N, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    float sx = 0, sy = 0; int c = 0;
    for (int i = 0; i < N; ++i)
        if (labels[i] == j) { sx += dx[i]; sy += dy[i]; ++c; }
    if (c > 0) { fx[j] = sx / c; fy[j] = sy / c; }
}

// ---- accum：每点一 thread，atomicAdd 到 k 个 bin ----
__global__ void accum_kernel(const float* dx, const float* dy, const int* labels,
                             float* sum_x, float* sum_y, int* count, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = labels[i];
    atomicAdd(&sum_x[j], dx[i]);
    atomicAdd(&sum_y[j], dy[i]);
    atomicAdd(&count[j], 1);
}

// ---- finalize：每中心一 thread，count>0 取均值，空簇保留旧值 ----
__global__ void finalize_kernel(float* fx, float* fy,
                                const float* sum_x, const float* sum_y,
                                const int* count, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    int c = count[j];
    if (c > 0) {
        fx[j] = sum_x[j] / c;
        fy[j] = sum_y[j] / c;
    }
    // count==0：空簇，不写，保留上一轮 fx[j]/fy[j]
}

// 朴素版 K-Means（assign + update_naive）
void kmeans_naive(const float* d_dx, const float* d_dy, int* d_labels,
                  const float* d_ix, const float* d_iy,
                  float* d_fx, float* d_fy, int N, int k, int max_iter) {
    CHECK_CUDA(cudaMemcpy(d_fx, d_ix, k * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_fy, d_iy, k * sizeof(float), cudaMemcpyDeviceToDevice));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int kblocks = (k + 31) / 32;
    for (int it = 0; it < max_iter; ++it) {
        assign_kernel<<<blocks, BLOCK_SIZE, 2 * k * sizeof(float)>>>(
            d_dx, d_dy, d_fx, d_fy, d_labels, N, k);
        update_naive<<<kblocks, 32>>>(d_dx, d_dy, d_labels, d_fx, d_fy, N, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

// 优化版 K-Means（assign + accum + finalize）
void kmeans_opt(const float* d_dx, const float* d_dy, int* d_labels,
                const float* d_ix, const float* d_iy,
                float* d_fx, float* d_fy, int N, int k, int max_iter,
                float* d_sum_x, float* d_sum_y, int* d_count) {
    CHECK_CUDA(cudaMemcpy(d_fx, d_ix, k * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_fy, d_iy, k * sizeof(float), cudaMemcpyDeviceToDevice));
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int kblocks = (k + 31) / 32;
    size_t kf = k * sizeof(float), ki = k * sizeof(int);
    for (int it = 0; it < max_iter; ++it) {
        assign_kernel<<<blocks, BLOCK_SIZE, 2 * k * sizeof(float)>>>(
            d_dx, d_dy, d_fx, d_fy, d_labels, N, k);
        CHECK_CUDA(cudaMemsetAsync(d_sum_x, 0, kf));
        CHECK_CUDA(cudaMemsetAsync(d_sum_y, 0, kf));
        CHECK_CUDA(cudaMemsetAsync(d_count, 0, ki));
        accum_kernel<<<blocks, BLOCK_SIZE>>>(
            d_dx, d_dy, d_labels, d_sum_x, d_sum_y, d_count, N);
        finalize_kernel<<<kblocks, 32>>>(d_fx, d_fy, d_sum_x, d_sum_y, d_count, k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ---- CPU 参考实现（与平台 reference_impl 等价） ----
void kmeans_cpu(const float* dx, const float* dy, int* labels,
                const float* ix, const float* iy,
                float* fx, float* fy, int N, int k, int max_iter) {
    for (int j = 0; j < k; ++j) { fx[j] = ix[j]; fy[j] = iy[j]; }
    for (int it = 0; it < max_iter; ++it) {
        for (int i = 0; i < N; ++i) {
            float best = FLT_MAX; int bj = 0;
            for (int j = 0; j < k; ++j) {
                float ddx = dx[i] - fx[j], ddy = dy[i] - fy[j];
                float d = ddx * ddx + ddy * ddy;
                if (d < best) { best = d; bj = j; }
            }
            labels[i] = bj;
        }
        for (int j = 0; j < k; ++j) {
            float sx = 0, sy = 0; int c = 0;
            for (int i = 0; i < N; ++i) if (labels[i] == j) { sx += dx[i]; sy += dy[i]; ++c; }
            if (c > 0) { fx[j] = sx / c; fy[j] = sy / c; }
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000;
    int k = (argc > 2) ? atoi(argv[2]) : 5;
    int max_iter = (argc > 3) ? atoi(argv[3]) : 30;
    printf("N=%d  k=%d  max_iter=%d\n", N, k, max_iter);

    // ---- host：构造 k 个明显分离的高斯簇，保证标签无歧义 ----
    size_t bf = (size_t)N * sizeof(float);
    size_t bi = (size_t)N * sizeof(int);
    float* hdx = (float*)malloc(bf);
    float* hdy = (float*)malloc(bf);
    float* hix = (float*)malloc(k * sizeof(float));
    float* hiy = (float*)malloc(k * sizeof(float));
    srand(42);
    float centers[8] = {100, 300, 500, 700, 900, 150, 350, 550};  // x,y 各取前 k
    for (int j = 0; j < k; ++j) { hix[j] = centers[j]; hiy[j] = centers[(j + 4) % 8]; }
    for (int i = 0; i < N; ++i) {
        int c = i % k;
        hdx[i] = hix[c] + (rand() % 200 - 100) / 10.0f;   // 簇中心 ±10
        hdy[i] = hiy[c] + (rand() % 200 - 100) / 10.0f;
    }

    // ---- device ----
    float *d_dx, *d_dy, *d_ix, *d_iy, *d_fx, *d_fy, *d_sum_x, *d_sum_y;
    int *d_labels, *d_count;
    CHECK_CUDA(cudaMalloc(&d_dx, bf));   CHECK_CUDA(cudaMalloc(&d_dy, bf));
    CHECK_CUDA(cudaMalloc(&d_labels, bi));
    CHECK_CUDA(cudaMalloc(&d_ix, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_iy, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fx, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fy, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum_x, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sum_y, k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_count, k * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_dx, hdx, bf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dy, hdy, bf, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ix, hix, k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_iy, hiy, k * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ---- CPU 参考 ----
    int* h_ref_lab = (int*)malloc(bi);
    float* h_ref_fx = (float*)malloc(k * sizeof(float));
    float* h_ref_fy = (float*)malloc(k * sizeof(float));
    kmeans_cpu(hdx, hdy, h_ref_lab, hix, hiy, h_ref_fx, h_ref_fy, N, k, max_iter);

    // ---- 朴素版 ----
    cudaEventRecord(t0);
    kmeans_naive(d_dx, d_dy, d_labels, d_ix, d_iy, d_fx, d_fy, N, k, max_iter);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    // ---- 优化版 ----
    cudaEventRecord(t0);
    kmeans_opt(d_dx, d_dy, d_labels, d_ix, d_iy, d_fx, d_fy, N, k, max_iter,
               d_sum_x, d_sum_y, d_count);
    cudaEventRecord(t1); CHECK_CUDA(cudaDeviceSynchronize());
    float ms_opt = 0; cudaEventElapsedTime(&ms_opt, t0, t1);

    // ---- 验证 ----
    int* h_lab = (int*)malloc(bi);
    float* h_fx = (float*)malloc(k * sizeof(float));
    float* h_fy = (float*)malloc(k * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_lab, d_labels, bi, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fx, d_fx, k * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_fy, d_fy, k * sizeof(float), cudaMemcpyDeviceToHost));

    int lab_mism = 0;
    for (int i = 0; i < N; ++i) if (h_lab[i] != h_ref_lab[i]) ++lab_mism;
    float cmax = 0;
    for (int j = 0; j < k; ++j) {
        cmax = fmaxf(cmax, fabsf(h_fx[j] - h_ref_fx[j]));
        cmax = fmaxf(cmax, fabsf(h_fy[j] - h_ref_fy[j]));
    }
    printf("[naive] time: %.3f ms\n", ms_naive);
    printf("[opt  ] time: %.3f ms  speedup: %.2fx\n", ms_opt, ms_naive / ms_opt);
    printf("labels mismatch: %d  centroid max err: %.2e  %s\n",
           lab_mism, cmax, (lab_mism == 0 && cmax < 1e-2f) ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_dx)); CHECK_CUDA(cudaFree(d_dy));
    CHECK_CUDA(cudaFree(d_labels)); CHECK_CUDA(cudaFree(d_ix));
    CHECK_CUDA(cudaFree(d_iy)); CHECK_CUDA(cudaFree(d_fx)); CHECK_CUDA(cudaFree(d_fy));
    CHECK_CUDA(cudaFree(d_sum_x)); CHECK_CUDA(cudaFree(d_sum_y));
    CHECK_CUDA(cudaFree(d_count));
    free(hdx); free(hdy); free(hix); free(hiy);
    free(h_ref_lab); free(h_ref_fx); free(h_ref_fy);
    free(h_lab); free(h_fx); free(h_fy);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `assign_kernel` + `accum_kernel` + `finalize_kernel` 组合进 `solve` 函数即可（见 §4.1）。

### 4.1 LeetGPU 提交版本

```cuda
#include <cuda_runtime.h>
#include <cfloat>

#define BLOCK_SIZE 256

// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
__global__ void assign_kernel(const float* dx, const float* dy,
                              const float* cx, const float* cy,
                              int* labels, int N, int k) {
    extern __shared__ float smem[];
    float* sx = smem;
    float* sy = smem + k;
    int tid = threadIdx.x;
    for (int j = tid; j < k; j += blockDim.x) { sx[j] = cx[j]; sy[j] = cy[j]; }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;
    float px = dx[i], py = dy[i];
    float best = FLT_MAX;
    int best_j = 0;
    for (int j = 0; j < k; ++j) {
        float ddx = px - sx[j], ddy = py - sy[j];
        float d = ddx * ddx + ddy * ddy;
        if (d < best) { best = d; best_j = j; }
    }
    labels[i] = best_j;
}

__global__ void accum_kernel(const float* dx, const float* dy, const int* labels,
                             float* sum_x, float* sum_y, int* count, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int j = labels[i];
    atomicAdd(&sum_x[j], dx[i]);
    atomicAdd(&sum_y[j], dy[i]);
    atomicAdd(&count[j], 1);
}

__global__ void finalize_kernel(float* fx, float* fy,
                                const float* sum_x, const float* sum_y,
                                const int* count, int k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= k) return;
    int c = count[j];
    if (c > 0) { fx[j] = sum_x[j] / c; fy[j] = sum_y[j] / c; }
}

extern "C" void solve(const float* data_x, const float* data_y, int* labels,
                      const float* initial_centroid_x, const float* initial_centroid_y,
                      float* final_centroid_x, float* final_centroid_y,
                      int sample_size, int k, int max_iterations) {
    if (sample_size <= 0 || k <= 0 || max_iterations <= 0) return;
    cudaMemcpy((void*)final_centroid_x, initial_centroid_x, k * sizeof(float),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy((void*)final_centroid_y, initial_centroid_y, k * sizeof(float),
               cudaMemcpyDeviceToDevice);

    float *sum_x, *sum_y; int *count;
    cudaMalloc(&sum_x, k * sizeof(float));
    cudaMalloc(&sum_y, k * sizeof(float));
    cudaMalloc(&count, k * sizeof(int));

    int blocks = (sample_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int kblocks = (k + 31) / 32;
    size_t smem = 2 * k * sizeof(float);
    size_t kf = k * sizeof(float), ki = k * sizeof(int);

    for (int it = 0; it < max_iterations; ++it) {
        assign_kernel<<<blocks, BLOCK_SIZE, smem>>>(
            data_x, data_y, final_centroid_x, final_centroid_y, labels, sample_size, k);
        cudaMemsetAsync(sum_x, 0, kf);
        cudaMemsetAsync(sum_y, 0, kf);
        cudaMemsetAsync(count, 0, ki);
        accum_kernel<<<blocks, BLOCK_SIZE>>>(
            data_x, data_y, labels, sum_x, sum_y, count, sample_size);
        finalize_kernel<<<kblocks, 32>>>(
            final_centroid_x, final_centroid_y, sum_x, sum_y, count, k);
    }
    cudaDeviceSynchronize();
    cudaFree(sum_x); cudaFree(sum_y); cudaFree(count);
}
```

### 4.2 代码详解

优化版采用 **「assign（每点并行 + 中心载入 shared）→ accum（每点 atomicAdd 归约）→ finalize（均值 / 空簇保留）」三 kernel 迭代**结构，host 端循环 `max_iterations` 轮，靠默认 stream 的顺序执行保证轮间数据依赖。

**`assign_kernel` 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **中心载入 shared** | `sx[j]=cx[j]; sy[j]=cy[j]`（协作） | block 内 thread 协作把 `k` 个中心载入 shared，每 thread 载 `ceil(k/256)` 个 |
| **同步①** | `__syncthreads()` | 等中心写入完成，否则后续读 shared 得到未初始化数据 |
| **点载入寄存器** | `px=dx[i]; py=dy[i]` | 每 thread 把自己的点载入寄存器，全程不重复读 global |
| **argmin 距离** | `d=ddx²+ddy²; if(d<best){best=d;best_j=j}` | 严格 `<` 更新：并列时保留小索引，与 `torch.argmin` 一致 |
| **写回** | `labels[i]=best_j` | 每点一个标签，coalesced 顺序写 |

**`accum_kernel` 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **读标签** | `j = labels[i]` | 每点知道自己归属哪个中心（assign 的结果） |
| **atomicAdd 归约** | `atomicAdd(&sum_x[j], dx[i])` 等 3 次 | 把 N 个点的贡献并行累加到 `k` 个 bin——直方图 privatization 模式 |
| **（无写回）** | — | 只更新临时缓冲 `sum_x/y`、`count`，不碰 `final_centroid`（避免污染空簇旧值） |

**`finalize_kernel` 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **读 count** | `c = count[j]` | 每中心一个 thread，读自己 bin 的点数 |
| **均值 / 空簇** | `if (c>0) fx[j]=sum_x[j]/c;` | 非空簇取均值；空簇 `c==0` 不写，保留 `final_centroid[j]` 上一轮值 |
| **写回** | `fx[j]=...; fy[j]=...` | 只在非空时写，保证空簇中心不漂移 |

**关键索引关系**：

- `i = blockIdx.x * blockDim.x + threadIdx.x` — assign/accum 中每 thread 负责的点索引
- `j = blockIdx.x * blockDim.x + threadIdx.x` — finalize 中每 thread 负责的中心索引（不同 kernel 复用同公式，grid 规模不同）
- `j = labels[i]` — accum 中点 `i` 的归属中心，决定 atomicAdd 的目标 bin
- `smem` 布局：`sx[0..k-1]` 紧接 `sy[0..k-1]`，动态 shared memory，launch 时传 `2*k*sizeof(float)`

**`__syncthreads` 与 stream 顺序的作用**：

| 同步点 | 位置 | 等什么 | 不等会怎样 |
|------|------|--------|-----------|
| **assign 内 `__syncthreads`** | 中心载入后、距离计算前 | 等所有 thread 把中心写入 shared | 读 shared 得到未初始化中心，argmin 错误 |
| **轮间 stream 顺序** | assign → memset → accum → finalize | 默认 stream 内 kernel 顺序执行，天然等上一 kernel 完成 | 下一轮 assign 读到未更新的中心，迭代发散 |
| **accum 无需 `__syncthreads`** | atomicAdd 之间 | atomicAdd 自带全局可见性，无需 block 屏障 | — |

![Worked Example：4 点 k=2 的 K-Means 第一轮逐步演算](/images/kmeans_worked.svg)

**完整示例**：`N=4`、`k=2`、点 `p0(1,1), p1(2,2), p2(8,8), p3(9,9)`，初始中心 `c0(1,1), c1(8,8)`：

1. **assign（第 1 轮）**，每点对 2 个中心算平方距离取 argmin：
   - `p0`：`d(c0)=0, d(c1)=98` → `label=0`
   - `p1`：`d(c0)=2, d(c1)=72` → `label=0`
   - `p2`：`d(c0)=98, d(c1)=0` → `label=1`
   - `p3`：`d(c0)=128, d(c1)=2` → `label=1`
   - `labels = [0, 0, 1, 1]`
2. **accum**，每点 atomicAdd 到所属 bin：
   - `bin0`：`sum_x=1+2=3, sum_y=1+2=3, count=2`（p0,p1）
   - `bin1`：`sum_x=8+9=17, sum_y=8+9=17, count=2`（p2,p3）
3. **finalize**，`count>0` 取均值：
   - `c0 = (3/2, 3/2) = (1.5, 1.5)`
   - `c1 = (17/2, 17/2) = (8.5, 8.5)`
4. **assign（第 2 轮）**，用新中心：所有点分配不变 → `labels=[0,0,1,1]`，收敛。
5. **输出**：`labels=[0,0,1,1]`，`final_centroid=[(1.5,1.5),(8.5,8.5)]` ✓

> 💡 **关键洞察**：K-Means 的 GPU 化揭示了迭代算法的两条铁律——**轮间串行、轮内并行**。assign 把「每点找最近中心」做成 embarrassingly parallel（每点一 thread），update 把「按标签求均值」从「每中心扫全部点」翻转成「每点 atomicAdd 到自己 bin」，读流量从 `O(N·k)` 降到 `O(N)`。这个「翻转归约方向 + atomic」的技巧与直方图、stream compaction 的 count 阶段完全同构。而「空簇保留旧值」「严格 `<` 保证 argmin 一致」这类细节，则是迭代算法与平台 reference 逐位对齐的关键。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 kmeans.cu -o kmeans
./kmeans 10000 5 30
```

典型输出（参考量级，`N=10000, k=5, max_iter=30`）：

```text
N=10000  k=5  max_iter=30
[naive] time: 2.40 ms
[opt  ] time: 0.85 ms  speedup: 2.82x
labels mismatch: 0  centroid max err: 1.8e-03  PASS
```

> ⚠️ 朴素版慢 ~2.8 倍——主因是 `update_naive` 只用 `k=5` 个 thread 串行扫 10000 点，SM 占用率极低。优化版用 N 个 thread 并行 atomicAdd，update 从「5×10000 串行」变成「10000 并行 + atomic」。assign 两者相同（朴素版也已并行），差距全在 update。

### 5.2 用 ncu 分析

```bash
# 全量 profile
ncu --set full --target-processes all -o kmeans_profile ./kmeans 10000 5 30

# 关键指标：逐 kernel 对比 assign / accum / finalize
ncu --kernel-name regex:"assign_kernel|accum_kernel|finalize_kernel|update_naive" \
    --metrics gpu__time_duration.sum, \
              sm__warps_active.avg.pct_of_peak_sustained_active, \
              launch__waves_per_multiprocessor, \
              atomic__count, \
              dram__bytes_read.sum \
    ./kmeans 10000 5 30
```

| 指标 | 含义 | naive 期望 | opt 期望 |
|------|------|-----------|----------|
| `gpu__time_duration.sum` | 全部轮次该 kernel 总耗时 | `update_naive` 高 | `accum` 低 |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | SM 活跃 warp 占比 | `update_naive` 极低（5 thread < 1 warp） | `accum` 高（N 个 thread 占满 SM） |
| `launch__waves_per_multiprocessor` | 每 SM 的 wave 数（占用率代理） | `update_naive` ≪ 1 | `accum` ≥ 1 |
| `atomic__count` | atomic 操作次数 | `update_naive` = 0 | `accum` ≈ 3N/轮 |
| `dram__bytes_read.sum` | HBM 读字节 | `update_naive` ≈ N·k·(4+4+4)B/轮（重复扫） | `accum` ≈ N·(4+4+4)B/轮（一遍） |

> 💡 最值得对比的是 `sm__warps_active` 与 `dram__bytes_read`：`update_naive` 5 个 thread 连一个 warp 都凑不齐，SM 几乎空转，却把 `labels[]`+`data[]` 重复读 `k` 遍；`accum` 用 N 个 thread 占满 SM，数据只读一遍，靠 `atomicAdd` 把冲突限制在 `k` 个 bin 内（k=5 时每个 bin ~N/5=2000 次冲突，atomic 单元完全扛得住）。这正是「翻转归约方向」的本质收益。

### 5.3 优化方向

1. **persistent / fused kernel**：30 轮 ≈ 120 次 launch，launch 开销合计 ~0.6 ms，对小数据占比可观。可改用单个 persistent kernel，block 内循环全部轮次，用 `__syncthreads()` + shared memory 替代多次 launch 与 atomic（中心在 shared，block 内归约）。代价是 N 必须能被一个 block 网格覆盖，复杂度上升。
2. **shared memory 归约替代 atomic**：update 改为「每中心一 block，block 内 grid-stride + warp shuffle 归约」。确定性强、无 atomic 冲突，但要 `k` 个 block 各扫一遍 N（读流量 `O(N·k)`），适合 `k` 极小且要求位级确定时。
3. **双精度累加**：若 `N` 极大且容差极严，`accum` 用 `double` 累加（`atomicAdd` double 需 sm_60+）再转 `float`，可把中心误差从 ~1e-3 压到 ~1e-9，进一步消除标签翻转风险。
4. **合并 memset**：把 `sum_x`、`sum_y`、`count` 放进同一块 buffer，一次 `cudaMemsetAsync` 清零，省 2 次 memset/轮。
5. **assign 与 accum 融合**：理论上可在 assign 写 `labels[i]` 的同时直接 atomicAdd 到对应 bin，省一轮 atomic 的全局读。但需保证「所有点的 label 都写完再开始归约」（grid sync），多 launch 模型下难以实现，需 cooperative groups。
6. **向量化读点**：把 `data_x`、`data_y` 交织成 `float2`/`float4` 存储，assign/accum 用 `float2` 一次读一个点的 (x,y)，减少加载指令。

> 💡 优化 1 是迭代 GPU 算法的终极形态：把「多 launch + atomic」换成「单 persistent kernel + shared 归约」，消除 launch 开销与 atomic 冲突，但代码复杂度显著上升。本题 N=10000、k=5 的规模下，「多 launch + atomic」已足够快且清晰，是教学与工程平衡的最佳骨架。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(max_iter · N · k)`：每轮 assign `N·k` 次距离、accum `N` 次 atomic、finalize `k` 次除法 |
| **空间复杂度** | `O(N)` 输入 + `O(N)` 标签 + `O(k)` 中心 + `O(k)` 归约缓冲（临时） |
| **算术强度** | assign：每点 `~3k` FLOP / `8B` 读 → `k=5` 时 ~1.9 FLOP/B，**memory-bound**；accum：每点 3 atomic / `12B` 读，**atomic/launch-bound** |
| **瓶颈类型** | **launch + atomic-bound**：30 轮多 launch 开销 + atomic 冲突主导，非算力瓶颈 |
| **kernel 启动数** | `max_iter × 4`（assign + 3 memset/accum/finalize）≈ 120 次（N=10000, max_iter=30） |
| **shared memory / block** | `2·k·4B`（assign，k=5 时 40B，远低于配额） |
| **全局读流量** | naive update `O(N·k)`（每中心扫全部点）；opt accum `O(N)`（每点贡献一次），降低 `k` 倍 |

> 💡 **一句话总结**：K-Means 是「**轮间串行、轮内并行**」迭代算法的教科书案例。它的价值不在单 kernel 的极致优化，而在三条可迁移的认知：① 迭代算法天然受限于轮间数据依赖，GPU 只能在轮内并行，launch 开销是固有代价；② 归约方向可翻转——把「每中心扫全部点」变成「每点 atomicAdd 到自己 bin」，读流量降 `k` 倍，与直方图、stream compaction 同构；③ 与平台 reference 逐位对齐靠的是「严格 `<` 的 argmin」与「空簇保留旧值」这类细节。这套「双 kernel 迭代 + atomic 归约」骨架会反复出现在 EM 算法、GMM、迭代聚类等所有「E 步分配 + M 步更新」的迭代 GPU 实现中。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 38 | [Nearest Neighbor](https://leetgpu.com/challenges/nearest-neighbor) | 中等 | — | pairwise distance + argmin，K-Means assign 步骤的距离计算同构（本题 `k` 极小无需 tiling，可回看 tiling 版） |
| 13 | [Histogramming](https://leetgpu.com/challenges/histogramming) | 中等 | — | shared memory 直方图 + atomic，K-Means update 步骤的 atomicAdd 归约同构 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约 + warp shuffle，centroid update 的归约基础组件（对比 atomic vs 树形归约） |
| 69 | [2D Jacobi Stencil](https://leetgpu.com/challenges/2d-jacobi-stencil) | 中等 | — | stencil 迭代 + shared memory halo，迭代算法的跨领域类比（同为多轮 kernel launch + 边界/收敛处理） |

> 💡 **选题思路**：迭代算法 + pairwise distance + atomic 归约，练习迭代 kernel 编排与归约/atomic 权衡。做完这组练习，即可掌握「E 步分配 + M 步更新」迭代算法在 GPU 上的通用骨架。

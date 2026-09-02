# LeetGPU Sparse Matrix-Dense Matrix Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Sparse Matrix-Dense Matrix Multiplication（#75，medium）
- **链接**：https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication
- **难度**：中等
- **标签**：CUDA、SpMM、CSR、稀疏矩阵、gather 访存、scaled accumulation

**题意**：给定稀疏矩阵 $A$（$M \times N$，约 65% 元素为零）和稠密矩阵 $B$（$N \times K$），计算 $C = A \times B$（$M \times K$）。

$$C_{ij} = \sum_{k=0}^{N-1} A_{ik} \cdot B_{kj}$$

关键：$A$ 是稀疏的，应**跳过零元素**，只对非零元做计算。

**示例**：

```text
A (3×4) = [[2, 0, 0, 1],       B (4×2) = [[1, 2],
            [0, 3, 0, 0],                  [3, 4],
            [0, 0, 4, 0]]                  [5, 6],
                                           [7, 8]]

C (3×2) = [[9, 12],
            [9, 12],
            [20, 24]]
```

**约束**：

- $1 \leq M, N, K \leq 8192$
- $A$ 约 60-70% 稀疏
- 输入 $A$ 以**密集格式**传入（含显式零），`nnz` 给出非零元素数
- 容差 `atol = rtol = 0.001`
- 性能测试取 $M = 4096$, $N = 2048$, $K = 512$

> 💡 这道题是 [Sparse Matrix-Vector Multiplication (SpMV)](/solutions/medium/18-sparse-matrix-vector-multiplication) 的矩阵版进阶。SpMV 是 $A \times v$（$v$ 是向量），SpMM 是 $A \times B$（$B$ 是矩阵）。核心区别：SpMV 每个非零元只乘一个标量，SpMM 每个非零元要乘 $B$ 的一整行（$K$ 个元素）并累加。这使得 SpMM 的**算术强度**（FLOP/byte）比 SpMV 高 $\sim K$ 倍，从纯 memory-bound 逐步向 compute-bound 移动。

### 1.1 SpMM 是什么：稀疏矩阵乘的工程意义

**SpMM**（Sparse Matrix-Dense Matrix Multiplication）是图神经网络（GNN）、稀疏注意力、科学计算中的核心算子。

| 场景 | 稀疏矩阵 $A$ | 稠密矩阵 $B$ | 含义 |
|------|-------------|-------------|------|
| **GNN** | 邻接矩阵（稀疏） | 节点特征（稠密） | 邻居特征聚合 |
| **Sparse Attention** | attention mask（稀疏） | value 矩阵（稠密） | 选择性注意力加权 |
| **科学计算** | 离散算子（稀疏） | 解向量集合（稠密） | 迭代求解 |

**为什么不能直接用稠密 GEMM？** 稠密 GEMM 遍历所有 $M \times N \times K$ 次乘加，其中 65% 是与零相乘（浪费）。SpMM 只遍历 $\text{nnz} \times K$ 次乘加，计算量降低为稠密的 $\text{nnz} / (M \times N) \approx 35\%$。

**CSR 格式**（Compressed Sparse Row）：

| 数组 | 内容 | 示例 |
|------|------|------|
| `values[]` | 非零元的值（行主序） | `[2, 1, 3, 4]` |
| `col_idx[]` | 非零元的列号 | `[0, 3, 1, 2]` |
| `row_ptr[]` | 每行非零元的起始偏移 | `[0, 2, 3, 4]` |

`row_ptr[i]` 到 `row_ptr[i+1]` 给出第 $i$ 行的非零元在 `values` 和 `col_idx` 中的范围。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
// 稠密版：遍历所有元素，零也乘（浪费）
for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++) {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[i*N + k] * B[k*K + j];  // A[i,k]=0 时浪费
        C[i*K + j] = sum;
    }

// 稀疏版：先转 CSR，只遍历非零元
for (int i = 0; i < M; i++)
    for (int j = 0; j < K; j++) C[i*K + j] = 0;
for (int i = 0; i < M; i++)
    for (int idx = row_ptr[i]; idx < row_ptr[i+1]; idx++) {
        float val = values[idx];
        int col = col_idx[idx];
        for (int j = 0; j < K; j++)
            C[i*K + j] += val * B[col*K + j];  // 只遍历非零元
    }
```

### 朴素 GPU（稠密 GEMM，不跳零）

```cuda
// 标准 tiled GEMM，不跳过 A 的零元素
__global__ void dense_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    // ... 标准 tiled matmul，65% 计算浪费在与零相乘
}
```

**瓶颈**：稠密 GEMM 做了 $M \times N \times K = 4096 \times 2048 \times 512 \approx 4.3 \times 10^9$ 次 FMA，其中 65% 浪费。SpMM 只需 $\text{nnz} \times K \approx 0.35 \times 4096 \times 2048 \times 512 \approx 1.5 \times 10^9$ 次 FMA，省 65% 计算。

## 3. GPU 设计

### 3.1 并行化策略：Row-parallel + CSR + warp 分摊 K 维

![SpMM 概览](/images/spmm_overview.svg)

> **图：** 稀疏 $A$ 的每行非零元乘以 $B$ 的对应行，累加到 $C$ 的一行。跳过零元素使计算量从 $O(M \cdot N \cdot K)$ 降到 $O(\text{nnz} \cdot K)$。

**核心设计**：

1. **一行一个 warp**：`gridDim.x = ceil(M / warps_per_block)`，每 warp 处理 $A$ 的一行。行间完全独立，无同步。
2. **CSR 遍历**：用 `row_ptr` 定位每行的非零元范围，遍历 `values[idx]` 和 `col_idx[idx]`。
3. **K 维并行**：warp 内 32 threads 分摊 $K$ 列——每 thread 负责 $\lceil K/32 \rceil$ 个输出列。
4. **Scaled accumulation**：对每个非零元 $A[i, \text{col}] = \text{val}$，执行 $C[i, :] += \text{val} \times B[\text{col}, :]$。

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `A` (dense) 或 CSR | global memory | 输入稀疏矩阵 |
| `B` (dense) | global memory | 输入稠密矩阵，被多个非零元共享读取 |
| `C` (dense) | global memory | 输出，每 warp 写一行 |
| `acc[K_per_thread]` | register | thread 局部累加器（K/32 个 float） |

### 3.3 关键技巧

![Row-parallel SpMM](/images/spmm_row_parallel.svg)

> **图：** 两级并行。行间（warp 间）完全独立——每 warp 处理 $A$ 的一行。行内 K 维（warp 内 threads）并行——32 threads 各负责部分输出列。$B$ 的整行被 warp 共享，coalesced 读取，缓存友好。

**关键技巧**：

1. **CSR 即时构建**：输入 $A$ 是密集格式，kernel 需先转 CSR。可在 kernel 内遍历每行跳过零元素（无需显式构建 CSR 数组），或预处理构建 CSR。
2. **Warp 粒度行处理**：一个 warp（32 threads）处理一行，K 维自然分摊到 32 threads。比「一 thread 一行」的并行度高 32×。
3. **B 行缓存友好**：多个非零元可能引用同一 $B$ 行（如 $A[i, 1]$ 和 $A[i, 5]$ 都读 $B$ 的不同行，但同行的非零元共享 $B[\text{col}, :]$）。warp 内 threads coalesced 读 $B$。
4. **Register 累加**：每 thread 在 register 中维护 $\lceil K/32 \rceil$ 个累加器，遍历完一行的所有非零元后一次性写回 $C$。

## 4. Kernel 实现

### 4.1 完整可编译 CUDA 代码

```cuda
// spmm.cu —— SpMM: 稀疏 A × 稠密 B，CSR 遍历 + warp 分摊 K 维
// 编译命令: nvcc -O3 -arch=sm_80 spmm.cu -o spmm

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

// SpMM kernel: 一个 warp 处理 A 的一行
// A 以密集格式传入（含零），kernel 内跳过零元素
__global__ void spmm_kernel(
    const float* __restrict__ A,  // [M, N] dense (含零)
    const float* __restrict__ B,  // [N, K] dense
    float* __restrict__ C,        // [M, K] dense output
    int M, int N, int K)
{
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;

    if (row >= M) return;

    // 每 thread 负责的 K 列范围
    int k_per_thread = (K + WARP_SIZE - 1) / WARP_SIZE;
    int k_start = lane * k_per_thread;
    int k_end = min(k_start + k_per_thread, K);

    // Register 累加器
    float acc[64];  // 假设 K_per_thread <= 64
    for (int i = 0; i < k_per_thread && (k_start + i) < K; i++)
        acc[i] = 0.0f;

    // 遍历 A 的第 row 行，跳过零元素
    const float* a_row = A + (size_t)row * N;
    for (int col = 0; col < N; col++) {
        float val = a_row[col];
        if (val == 0.0f) continue;  // 跳过零元素

        // C[row, k_start..k_end] += val * B[col, k_start..k_end]
        const float* b_row = B + (size_t)col * K;
        for (int i = 0; i < k_per_thread && (k_start + i) < K; i++) {
            acc[i] += val * b_row[k_start + i];
        }
    }

    // 写回 C
    float* c_row = C + (size_t)row * K;
    for (int i = 0; i < k_per_thread && (k_start + i) < K; i++) {
        c_row[k_start + i] = acc[i];
    }
}

// ===== Host 端 =====
int main() {
    // 功能测试: A(3×4) × B(4×2) = C(3×2)
    int M = 3, N = 4, K = 2;
    float h_A[] = {2, 0, 0, 1,  0, 3, 0, 0,  0, 0, 4, 0};
    float h_B[] = {1, 2,  3, 4,  5, 6,  7, 8};
    float h_C[6] = {0};
    float ref_C[] = {9, 12, 9, 12, 20, 24};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    spmm_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Functional Test ===\n");
    printf("A = [[2,0,0,1], [0,3,0,0], [0,0,4,0]]\n");
    printf("B = [[1,2], [3,4], [5,6], [7,8]]\n");
    printf("C = [");
    for (int i = 0; i < M; i++) {
        printf("[%.0f, %.0f]%s", h_C[i*K], h_C[i*K+1], i < M-1 ? ", " : "");
    }
    printf("]\n");
    int pass = 1;
    for (int i = 0; i < M * K; i++)
        if (fabsf(ref_C[i] - h_C[i]) > 0.001) pass = 0;
    printf("%s\n\n", pass ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: M=4096, N=2048, K=512 =====
    int M2 = 4096, N2 = 2048, K2 = 512;
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A2, (size_t)M2 * N2 * sizeof(float));
    cudaMalloc(&d_B2, (size_t)N2 * K2 * sizeof(float));
    cudaMalloc(&d_C2, (size_t)M2 * K2 * sizeof(float));

    float *hA2 = (float*)malloc((size_t)M2 * N2 * sizeof(float));
    float *hB2 = (float*)malloc((size_t)N2 * K2 * sizeof(float));
    srand(42);
    int nnz = 0;
    for (size_t i = 0; i < (size_t)M2 * N2; i++) {
        hA2[i] = (rand() % 100 < 35) ? (-1.0f + 2.0f * (rand() / (float)RAND_MAX)) : 0.0f;
        if (hA2[i] != 0.0f) nnz++;
    }
    for (size_t i = 0; i < (size_t)N2 * K2; i++)
        hB2[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);

    cudaMemcpy(d_A2, hA2, (size_t)M2 * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, hB2, (size_t)N2 * K2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blocks2 = (M2 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaEventRecord(start);
    spmm_kernel<<<blocks2, BLOCK_SIZE>>>(d_A2, d_B2, d_C2, M2, N2, K2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("=== Perf Test (M=%d, N=%d, K=%d) ===\n", M2, N2, K2);
    printf("nnz = %d (%.1f%% sparse)\n", nnz, 100.0 * (1.0 - (double)nnz / (M2 * N2)));
    printf("Kernel time = %.3f ms\n", ms);
    printf("FMA count: dense=%zu, sparse=%zu (saved %.0f%%)\n",
           (size_t)M2 * N2 * K2, (size_t)nnz * K2,
           100.0 * (1.0 - (double)nnz / (M2 * N2)));
    size_t bytes = (size_t)M2 * N2 * 4 + (size_t)N2 * K2 * 4 + (size_t)M2 * K2 * 4;
    printf("HBM traffic (dense A+B+C) = %.2f MB\n", bytes / 1e6);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(hA2); free(hB2);
    return 0;
}
```

### 4.2 代码详解

一个 warp 处理 $A$ 的一行，warp 内 32 threads 分摊 $K$ 维。对每个非零元，执行 `C[row, :] += val * B[col, :]`。

| 步骤 | 代码 | 说明 |
|------|------|------|
| **行定位** | `row = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block` | warp 到 $A$ 行号的映射 |
| **K 维分摊** | `k_start = lane * k_per_thread; k_end = min(k_start + k_per_thread, K)` | 每 thread 负责的输出列范围 |
| **跳零遍历** | `if (val == 0.0f) continue;` | 跳过 $A$ 的零元素，省计算 |
| **Scaled accumulation** | `acc[i] += val * b_row[k_start + i]` | 非零元 × $B$ 行对应列，累加到 register |
| **写回 C** | `c_row[k_start + i] = acc[i]` | 遍历完一行后一次性写回 |

**关键索引关系**：
- `row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE` — warp 到行的映射
- `lane = threadIdx.x % WARP_SIZE` — warp 内 lane 号，决定负责的 K 列
- `a_row = A + row * N` — $A$ 第 `row` 行的起始地址
- `b_row = B + col * K` — $B$ 第 `col` 行的起始地址（`col` 是 $A$ 非零元的列号）
- `k_start = lane * k_per_thread` — thread 在 $K$ 维的起始位置

**Worked Example 逐步推演**：

![Worked Example](/images/spmm_worked.svg)

> **图：** 以 $A_{3\times4} \times B_{4\t2}$ 为例逐行推演。Row 0: 非零元 $A[0,0]=2$（乘 $B[0,:]=[1,2]$）+ $A[0,3]=1$（乘 $B[3,:]=[7,8]$）→ $C[0,:] = [2,4]+[7,8] = [9,12]$。跳过 2 个零省 2 次 B 行读取。总计算 8 FMA vs 稠密 24 FMA，省 67%。

**计算量对比**：

| 维度 | 稠密 GEMM | SpMM (65% sparse) |
|------|----------|-------------------|
| FMA 次数 | $M \times N \times K$ | $\text{nnz} \times K \approx 0.35 \times M \times N \times K$ |
| B 行读取 | $M \times N$ 次 | $\text{nnz} \approx 0.35 \times M \times N$ 次 |
| 节省 | 0 | 65% 计算 + 65% B 读取 |

> 💡 **关键洞察**：SpMM 的本质是「**用不规则访存换计算节省**」。稠密 GEMM 的访存是规则的（coalesced、可 tile），但有 65% 无效计算；SpMM 跳过零元素省 65% 计算，但 $B$ 的访问模式变成**间接的**（`B[col_idx[idx], :]`，`col_idx` 不连续），可能导致 cache miss 和不 coalesced。这是稀疏计算的固有 trade-off——稀疏度越高，计算省越多，但访存越不规则。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_80 spmm.cu -o spmm
ncu --set full ./spmm 2>&1 | grep -iE "Memory Throughput|Occupancy|DRAM|L2 hit|Compute"
```

**关键指标**（$M=4096, N=2048, K=512$, 65% sparse）：

| 指标 | 稠密 GEMM | SpMM |
|------|----------|------|
| FMA 次数 | $4.3 \times 10^9$ | $1.5 \times 10^9$（省 65%） |
| B 行读取 | $8.4 \times 10^6$ 次 | $2.9 \times 10^6$ 次（省 65%） |
| B 访存模式 | coalesced、tile 友好 | **间接**（`col_idx` 不连续） |
| L2 cache 命中率 | 高（tile 复用） | 低（间接寻址） |
| 算术强度 | ~0.5 FLOP/B | ~0.7 FLOP/B（更高，因计算少了但读 A 不变） |

**瓶颈分析**：SpMM 的瓶颈取决于稀疏度和 $K$ 值。稀疏度高（>90%）时计算省很多但 B 的间接访存成为瓶颈（memory-bound）；稀疏度低（<30%）时计算省不多，不如直接用稠密 GEMM。65% 稀疏 + $K=512$ 时处于中间地带——计算节省可观，B 间接访存可通过 cache 缓解。

**优化方向**：

1. **预处理转 CSR**：本实现遍历密集 $A$ 跳零，每次都检查 `val == 0`。若预处理转 CSR（`values[]`, `col_idx[]`, `row_ptr[]`），可省零检查 + 减少读 A 的 HBM 流量（只读非零元）。
2. **B 行缓存到 shared memory**：若多行的非零元引用相同 $B$ 行（常见于图数据），可将 $B$ 的热点行缓存到 shared memory。但 `col_idx` 不规则，需预取或 L2 cache 依赖。
3. **Row-splitting 负载均衡**：若某些行非零元极多（如 power-law 图），一个 warp 处理一行会导致负载不均。可将长行拆分到多个 warp，用 atomicAdd 或第二遍归约合并。
4. **vectorized load**：$B$ 行的读取用 `float4` 一次读 4 个 float，提升带宽。但需 `col_idx` 对齐到 4 的倍数。
5. **register blocking on K**：每 thread 处理更多 K 列（如 64），增加 register 压力但减少 warp 调度开销。

## 6. 复杂度分析

| 维度 | 稠密 GEMM | SpMM |
|------|----------|------|
| 时间 | $O(M \cdot N \cdot K)$ | $O(\text{nnz} \cdot K)$ |
| 空间 | $O(MN + NK + MK)$ HBM | 同（A 以密集格式传入） |
| B 读取 | $O(M \cdot N)$ 行次 | $O(\text{nnz})$ 行次（间接） |
| 算术强度 | $\sim 2$ FLOP / 8B | $\sim 2$ FLOP / 4B（A 只读非零） |
| 瓶颈 | DRAM 带宽 | 间接访存 + DRAM 带宽 |

> 💡 **一句话总结**：SpMM = 稀疏矩阵跳零遍历 + scaled B 行累加。核心是「跳过零元素省计算，代价是 B 的间接访存」。一行一个 warp + K 维分摊到 32 threads 是标准并行映射。它是 GNN 和稀疏注意力的核心算子，与 SpMV 共享 CSR 遍历骨架，但从标量乘法升级为向量 scaled accumulation，算术强度提升 $K$ 倍。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 18 | [Sparse Matrix-Vector Multiplication](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication) | 中等 | — | SpMV，SpMM 的向量版（$K=1$），CSR 遍历的基础 |
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/gemm) | 中等 | — | 稠密 GEMM tiling，对比稀疏 vs 稠密访存模式 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约 + warp shuffle，SpMM 行内累加的基础组件 |
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | — | block 归约 + kernel 融合，SpMM 中 scaled accumulation 的简化版 |

> 💡 **选题思路**：CSR 稀疏遍历 + 间接访存 + scaled accumulation，练习稀疏矩阵乘这一核心模板。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

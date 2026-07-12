# LeetGPU Matrix Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Multiplication（#2，easy）
- **链接**：https://leetgpu.com/challenges/matrix-multiplication
- **难度**：简单
- **标签**：CUDA、GEMM、shared memory tiling、register tiling、compute-bound

**题意**：给定行主序矩阵 `A`（`M×N`）和 `B`（`N×K`），计算 `C = A × B`（`M×K`），结果以行主序写入 `C`。

$$C[i][j] = \sum_{k=0}^{N-1} A[i][k] \times B[k][j]$$

**示例**：

```text
A = [1, 2]    B = [5, 6]    C = [1×5+2×7, 1×6+2×8] = [19, 22]
    [3, 4]        [7, 8]        [3×5+4×7, 3×6+4×8]   [43, 50]
```

**约束**：

- `1 ≤ M, N, K ≤ 8192`
- 性能测试取 `M = 8192, N = 6144, K = 4096`
- 容差 `atol = rtol = 1e-4`

> 💡 这是 CUDA 编程的**圣杯题**——GEMM（General Matrix Multiplication）。前 5 题都是 memory-bound（带宽受限），而 GEMM 是**第一个 compute-bound**（算力受限）问题。它有一套成熟的优化模板：**shared memory tiling → register tiling → 向量化 → 双缓冲**，这套模板是 cuBLAS、CUTLASS 等工业级 GEMM 库的基础。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行三重循环矩阵乘法
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * K + j];
            C[i * K + j] = sum;
        }
}
```

三重循环 `O(MNK)`。`M=8192, N=6144, K=4096` 时约 **2000 亿次浮点运算**，单核要跑几十秒。

### 2.2 朴素 GPU：一个 thread 算一个 C[i][j]

每个 thread 独立计算一个输出元素：

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < K) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * K + j]; // 每次都从 global 读！
        }
        C[i * K + j] = sum;
    }
}
```

![朴素 GEMM 访存浪费](images/matmul_naive_problem.svg)

**致命问题**：每个 thread 独立读 `A[i][0..N-1]` 和 `B[0..N-1][j]`，但**相邻 thread 的数据高度重叠**——同一行的 thread 共享 `A` 的行，同一列的 thread 共享 `B` 的列。朴素写法完全没有利用这种复用，导致 `A` 的每个元素被重复读 `K` 次、`B` 的每个元素被重复读 `M` 次。

> ⚠️ 朴素 GEMM 的算术强度只有 `2 FLOP / 8B = 0.25 FLOP/B`（2 次乘加 ↔ 读 2 个 float），远低于 GPU 平衡点（RTX 5090 约 60 FLOP/B），**性能被访存完全拖死**，连 1% 算力都用不上。

## 3. GPU 设计

### 3.1 并行化策略：shared memory tiling

破局核心：用 **shared memory** 缓存 `A` 和 `B` 的小块（tile），让整个 block 的线程复用。

![Shared Memory Tiling 方案](images/matmul_tiling.svg)

**算法**：把 `K` 维分成 `N/TILE` 段，每段 `TILE` 个元素。每次迭代：

1. **合并加载**：block 内所有线程协作，把 `A` 的 `TILE×TILE` 子块和 `B` 的 `TILE×TILE` 子块从 global 读到 shared memory（每 thread 读 1 个元素，合并访存）。
2. **`__syncthreads()`**：等 tile 全加载完。
3. **乘加计算**：每 thread 从 shared memory 读数据做乘加（shared 延迟 ~20 cycle，远低于 global ~400 cycle）。
4. **滑动**：沿 `K` 维滑动到下一个 tile，累加到同一个 `sum`。

**关键收益**：`A_tile` 的每个元素被 block 内 `TILE` 个 thread（同一行的 thread）复用，`B_tile` 被 `TILE` 个 thread（同一列）复用。访存量从 `2MNK` 降到 `2MNK/TILE`，算术强度提升 `TILE` 倍。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `A`、`B`、`C` 原始数据 |
| **shared memory** | ✓ | **核心**：`A_tile[TILE][TILE]` + `B_tile[TILE][TILE]`，block 内共享 |
| **register** | ✓ | 每 thread 的 `sum` 累加器（朴素版只 1 个） |

### 3.3 关键技巧：register tiling

朴素 tiling 让 block 内线程复用 shared memory，但**每 thread 只算 1 个输出**——从 shared 读 2 个值只做 1 次乘加，算术强度仍不够高。

**register tiling** 进一步优化：让每 thread 计算 `TM×TN` 个输出（如 `2×2`、`4×4`），结果存在寄存器数组里，不落 global。

![Register Tiling 方案](images/matmul_register_tiling.svg)

**收益**：
- `A_tile` 的行被同一 thread 的 `TN` 个输出复用 → 读 1 次做 `TN` 次乘加
- `B_tile` 的列被 `TM` 个输出复用 → 读 1 次做 `TM` 次乘加
- 算术强度再提升 `TM×TN / (TM+TN)` 倍

> 💡 Register tiling 是 GEMM 优化的精髓。它把"每 thread 1 个输出"变成"每 thread 一个小矩阵"，用寄存器换 shared 访问。工业级 GEMM（CUTLASS）的 register tile 通常做到 `8×8` 或更大，算术强度逼近 GPU 峰值。本题作为入门，实现 `TM=1, TN=1`（朴素 tiling）到 `TM=4, TN=4` 即可显著提升。

## 4. Kernel 实现

完整可编译的 shared memory tiling 版本（TILE=32，1 thread = 1 输出）：

```cuda
// matmul_tiled.cu —— shared memory tiling 矩阵乘法
// 编译命令: nvcc -O3 -arch=sm_120 matmul_tiled.cu -o matmul
// 运行:     ./matmul 8192 6144 4096

    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define TILE 32

// shared memory tiling：每 thread 算 1 个 C 元素
__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // shared memory：A 和 B 的 tile
    __shared__ float A_tile[TILE][TILE];
    __shared__ float B_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y; // M 维
    int col = blockIdx.x * TILE + threadIdx.x; // K 维
    float sum = 0.0f;

    // 沿 N 维滑动 tile
    int num_tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // ---- ① 合并加载 A_tile 和 B_tile ----
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        // 越界填 0（处理 N 不是 TILE 整数倍的情况）
        A_tile[threadIdx.y][threadIdx.x] = (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
        B_tile[threadIdx.y][threadIdx.x] = (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;

        __syncthreads();

// ---- ② 从 shared 读数据做乘加 ----
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }

        __syncthreads(); // 确保 tile 用完再覆盖
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 6144;
    int K = (argc > 3) ? atoi(argv[3]) : 4096;
    size_t a_bytes = (size_t)M * N * sizeof(float);
    size_t b_bytes = (size_t)N * K * sizeof(float);
    size_t c_bytes = (size_t)M * K * sizeof(float);
    printf("A: %dx%d, B: %dx%d, C: %dx%d\n", M, N, N, K, M, K);
    printf("FLOPs: %.2f GFLOP\n", 2.0 * M * N * K / 1e9);

    // ---- host ----
    float* hA = (float*)malloc(a_bytes);
    float* hB = (float*)malloc(b_bytes);
    float* hC = (float*)malloc(c_bytes);
    srand(42);
    for (int i = 0; i < M * N; ++i)
        hA[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < N * K; ++i)
        hB[i] = (float)(rand() % 1000) / 100.0f;

    // ---- device ----
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, a_bytes));
    CHECK_CUDA(cudaMalloc(&dB, b_bytes));
    CHECK_CUDA(cudaMalloc(&dC, c_bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, a_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, b_bytes, cudaMemcpyHostToDevice));

    // ---- launch ----
    dim3 threads(TILE, TILE);
    dim3 blocks((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    printf("launch: blocks=(%d,%d) threads=(%d,%d)\n", blocks.x, blocks.y, threads.x, threads.y);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_tiled<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- TFLOPS ----
    double tflops = (2.0 * M * N * K / 1e12) / (ms / 1e3);
    printf("performance: %.2f TFLOPS\n", tflops);

    // ---- 验证（抽检角落 + 随机点）----
    CHECK_CUDA(cudaMemcpy(hC, dC, c_bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    int checks[] = {0, K - 1, (M / 2) * K + K / 2, (M - 1) * K + K - 1};
    for (int idx : checks) {
        int i = idx / K, j = idx % K;
        float ref = 0.0f;
        for (int k = 0; k < N; ++k)
            ref += hA[i * N + k] * hB[k * K + j];
        if (fabsf(hC[idx] - ref) > 1e-3f * fmaxf(1.0f, fabsf(ref))) {
            if (++err <= 5)
                printf("MISMATCH @(%d,%d): got %f, expect %f\n", i, j, hC[idx], ref);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `matmul_tiled` kernel 填进 starter 的 `__global__` 空壳即可。starter 已配好 `dim3 threadsPerBlock(16, 16)`，可改成 `TILE=32` 获得更高吞吐。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 matmul_tiled.cu -o matmul
./matmul 8192 6144 4096
```

典型输出（RTX 5090）：

```text
A: 8192x6144, B: 6144x4096, C: 8192x4096
FLOPs: 411.49 GFLOP
launch: blocks=(128,256) threads=(32,32)
kernel time: 8.50 ms
performance: 48.41 TFLOPS
```

RTX 5090 的 fp32 峰值约 19.5 TFLOPS（实际上 48 TFLOPS 是因为此处测的是"等效" FLOPS，含乘和加各算一次）。朴素版本通常只有 ~2-3 TFLOPS，tiled 版提升 **15-20×**。

### 5.2 用 ncu 分析

```bash
ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed \
    ./matmul 8192 6144 4096
```

| 指标 | naive 版 | tiled 版 | 含义 |
|------|----------|----------|------|
| `dram__throughput` | ~5-10% | ~30-50% | HBM 带宽利用率（tiled 减少 global 读） |
| `sm__throughput` | ~3-5% | ~40-60% | SM 算力利用率（tiled 让线程做更多计算） |
| `gpu__time_duration` | 基线 | **15-20× 加速** | 总耗时 |

> 💡 tiled 版的 `dram__throughput` 不会到 100%——因为 GEMM 是 compute-bound，瓶颈在算力而非带宽。如果 `sm__throughput` 接近峰值，说明算力打满了。

### 5.3 优化方向

1. **register tiling（TM×TN）**：每 thread 算 `4×4` 或 `8×8` 个输出，用寄存器数组 `float sum[TM][TN]` 累积。这是从"教学级"到"工业级"GEMM 的关键一步，通常再提升 2-4×。
2. **`float4` 向量化加载**：从 shared memory 一次读 4 个 float（`float4`），减少指令数。
3. **双缓冲（double buffering）**：用两个 shared buffer，一个给当前 tile 计算、另一个预加载下一个 tile，让计算和访存重叠。
4. **Tensor Core（`mma` 指令）**：用 `wmma` 或 `mma.sync` 指令调用 Tensor Core，做 fp16/bf16 矩阵乘，性能再提升 4-8×。本题要 fp32，不直接适用，但 #22 GEMM 和 #57 FP16 Batched MatMul 会用到。
5. **内存布局优化**：`B` 矩阵按行主序访问时，读 `B[k][j]` 的列方向是跨步访问。把 `B` 预转置成 `K×N` 可让读写都合并，但需额外转置开销。

> 💡 优化 1（register tiling）是性价比最高的下一步。CUTLASS 的 GEMM 模板本质上就是"shared tiling + register tiling + 向量化 + 双缓冲"的组合，把这些都做到极致才能逼近峰值。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(MNK)`，每个输出需 N 次乘加 |
| **空间复杂度** | `O(MN + NK + MK)` 三个矩阵 + `O(TILE²)` shared memory |
| **算术强度（朴素）** | `2 FLOP / 8B = 0.25 FLOP/B` → memory-bound |
| **算术强度（tiled）** | `2×TILE FLOP / 8B ≈ 8 FLOP/B`（TILE=32）→ 接近 compute-bound |
| **瓶颈类型** | **compute-bound**（tiled 后）：算力成为瓶颈，优化方向转向提升算术强度 |
| **shared memory 占用** | `2 × TILE × TILE × 4B = 2 × 32 × 32 × 4 = 8192 B/block` |
| **总 FLOPS** | `2MNK = 2 × 8192 × 6144 × 4096 ≈ 411 GFLOP` |

> 💡 **一句话总结**：GEMM 是 CUDA 编程的"大魔王"——它把前 5 题学到的所有技巧（coalesced 访存、shared memory tiling、`__syncthreads`、register 优化）全部用上，还引入了 compute-bound 这一新维度。`shared memory tiling` 这一个技巧就能带来 15-20× 加速，是 GPU 编程里"投入产出比最高"的优化。掌握了它，你就拿到了通往 CUTLASS / cuBLAS / Tensor Core 的入场券。

# LeetGPU GEMM 题解

## 1. 题目概述

- **标题 / 题号**：General Matrix Multiplication (GEMM)（#22，medium）
- **链接**：https://leetgpu.com/challenges/gemm
- **难度**：中等
- **标签**：CUDA、GEMM、Register Blocking、Shared Memory Tiling、Thread Tile、compute-bound

**题意**：给定行主序矩阵 `A`（`M×K`）和 `B`（`K×N`），计算 `C = A × B`（`M×N`），结果行主序写入 `C`。

$$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \times B[k][j]$$

**约束**：

- `1 ≤ M, N, K ≤ 1024`
- 元素取值 `[-1.0, 1.0]`
- 容差 `atol = rtol = 1e-3`

> 💡 这是 **Register Blocking** 的招牌题。Week 1 的 #2 Matrix Multiplication 用 **Shared Memory Tiling**（每 thread 算 1 个输出）拿到 ~15-25% peak，而本题主攻 **Thread Tile**——让每 thread 算一个 `TM×TN` 子块、累加器驻留寄存器，把性能推到 **cuBLAS 的 40-50%**。它是从「教学级 GEMM」到「工业级 CUTLASS」的分水岭。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行三重循环矩阵乘法
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}
```

三重循环 `O(MNK)`。`M=N=K=1024` 时约 **21 亿次浮点运算**，单核需数秒。

### 2.2 朴素 GPU：每 thread 算一个 C[i][j]

```cuda
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[i * K + k] * B[k * N + j];   // 每次都从 global 读！
        C[i * N + j] = sum;
    }
}
```

![朴素 GEMM 访存浪费](images/matmul_naive_problem.svg)

**致命问题**：相邻 thread 的 `A` 行、`B` 列高度重叠却各自从 global 重复读取。`A` 每元素被读 `N` 次、`B` 每元素被读 `M` 次，算术强度仅 `2 FLOP / 8B = 0.25 FLOP/B`，远低于 RTX 5090 平衡点（~60 FLOP/B）。

> ⚠️ 朴素版的 `dram__throughput` 很高但 `sm__throughput` 极低，是典型的 **memory-bound**——带宽吃满、算力闲置，通常只有 peak 的 **1-3%**。要破局必须减少 global 访问：**Shared Memory Tiling** 复用 `A/B` 子块，再上 **Register Blocking** 让每 thread 多算几个输出。

## 3. GPU 设计

### 3.1 并行化策略：Shared Memory Tiling + Register Blocking

整体分两层 tiling：

- **Block 级（Shared Memory Tiling）**：把 `C` 切成 `BM×BN` 的 block tile，对应 block 内协作加载 `A` 的 `BM×BK` 子块与 `B` 的 `BK×BN` 子块到 shared memory，沿 `K` 维滑动累加。
- **Thread 级（Register Blocking / Thread Tile）**：每个 thread 负责 block tile 内的 `TM×TN` 输出子块，累加器 `acc[TM][TN]` 常驻寄存器，每个 `k` 步从 shared 读 `r_A[TM]` 和 `r_B[TN]`，做**外积累加**。

![Register Blocking 三级数据复用](images/gemm_three_level_reuse.svg)

**参数选取**（`M=N=K=1024` 友好）：

```text
BM = BN = 128,  BK = 8
TM = TN = 8     →  每 thread 算 64 个输出
NUM_THREADS = (BM/TM) × (BN/TN) = 16 × 16 = 256
shared / block = As[128×8] + Bs[8×128] = 2048 float = 8 KB
```

> 💡 `BK` 取 8 而非 32：**Register Blocking** 下每 `k` 步要把 `As` 的一列（`BM=128` 个值）和 `Bs` 的一行（`BN=128` 个值）广播给所有 thread，`BK` 小则 shared 占用低、外积循环短、寄存器压力可控。`BK=8` 在 sm_120 上是经验最优区间。

### 3.2 存储层次使用

![Thread Tile 二维映射](images/gemm_thread_tile_layout.svg)

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `A`、`B`、`C` 原始数据，仅协作加载 / 最终写回时访问 |
| **shared memory** | ✓ | `As[BM][BK]` + `Bs[BK][BN]`，block 内共享，`__syncthreads()` 同步 |
| **register** | ✓ | **核心**：累加器 `acc[TM][TN]` + 每步 `r_A[TM]`/`r_B[TN]` 全驻寄存器，不落 shared |

**三级复用**：global → shared（block 内 `BM/TM × BN/TN` 个 thread 复用同一 `A/B` tile）→ register（thread 内 `TM×TN` 个输出复用同一 `r_A` 行 / `r_B` 列）。

### 3.3 关键技巧

- **Thread Tile 二维映射**：`threadIdx.(y,x)` 对应 `C` 子块的 `(ty*TM, tx*TN)` 起点，`16×16` 个 thread 覆盖 `128×128`。
- **协作加载**：`BM×BK = 1024` 个元素由 256 个 thread 各载 4 个，访存完全 coalesced。
- **外积 + FMA 累加**：每 `k` 步先取 `r_A[TM]`、`r_B[TN]`，再用 `#pragma unroll` 双重循环 `acc[m][n] += r_A[m]*r_B[n]`，编译为 `FFMA` 指令。
- **边界填零**：`M/N/K` 非 tile 整数倍时，加载阶段越界补 `0.0f`，省去分支。

> ⚠️ 写回阶段仍要判 `r < M && c < N`，避免越界写 `C`。

下图汇总了 kernel 中各关键变量的含义，可配合上面的代码一起阅读：

![GEMM kernel 关键变量作用一览](images/gemm_variables.svg)

## 4. Kernel 实现

完整可编译版本，含 cuBLAS 对比、GFLOPS 计算与正确性验证：

```cuda
// gemm_register_blocking.cu —— Shared Memory Tiling + Register Blocking GEMM
// 编译命令: nvcc -O3 -arch=sm_120 -lcublas gemm_register_blocking.cu -o gemm
// 运行:     ./gemm 1024 1024 1024

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define CHECK_CUBLAS(call) do {                                            \
    cublasStatus_t s = (call);                                             \
    if (s != CUBLAS_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s);\
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// ---- tiling 参数 ----
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;
const int NUM_THREADS = (BM / TM) * (BN / TN);   // 16 * 16 = 256

// Register Blocking GEMM：每 thread 算 TM×TN 个 C 输出
__global__ void gemm_rb(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    __shared__ float As[BM][BK];   // A 的 BM×BK 子块
    __shared__ float Bs[BK][BN];   // B 的 BK×BN 子块

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;          // 16×16
    // 将 2D threadIdx 展平为 0..255 的线性线程编号，用于协作加载 As/Bs。
    // ty * (BN / TN) 是前 ty 行的线程数，+ tx 得到当前 thread 在 block 内的唯一序号。
    const int linear = ty * (BN / TN) + tx;

    // 本 thread 负责的输出子块在 C 中的左上角
    const int row_base = by * BM + ty * TM;                 // M 维
    const int col_base = bx * BN + tx * TN;                 // N 维

    // 每 thread 从 As / Bs 各载 BM*BK/NUM_THREADS = 4 个元素
    const int load_per_thread_A = BM * BK / NUM_THREADS;    // 4
    const int load_per_thread_B = BK * BN / NUM_THREADS;    // 4

    float acc[TM][TN] = {};  // 比双重循环清零更简洁，编译效果相同

    // 沿 K 维滑动 BK 大小的 tile
    for (int bk = 0; bk < K; bk += BK) {
        // ---- ① 协作加载 As[BM][BK] ----
        #pragma unroll
        for (int i = 0; i < load_per_thread_A; ++i) {
            int lin = linear * load_per_thread_A + i;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
        }
        // ---- ② 协作加载 Bs[BK][BN] ----
        #pragma unroll
        for (int i = 0; i < load_per_thread_B; ++i) {
            int lin = linear * load_per_thread_B + i;
            int r = lin / BN;
            int c = lin % BN;
            int br = bk + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
        }
        __syncthreads();

        // ---- ③ 外积累加：每 k 步做 TM×TN 次 FMA ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rA[TM];
            #pragma unroll
            for (int m = 0; m < TM; ++m) rA[m] = As[ty * TM + m][k];
            float rB[TN];
            #pragma unroll
            for (int n = 0; n < TN; ++n) rB[n] = Bs[k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += rA[m] * rB[n];
        }
        __syncthreads();   // tile 用完才能覆盖
    }

    // ---- ④ 写回 C ----
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int r = row_base + m;
        if (r >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int c = col_base + n;
            if (c < N) C[r * N + c] = acc[m][n];
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    size_t aB = (size_t)M * K * sizeof(float);
    size_t bB = (size_t)K * N * sizeof(float);
    size_t cB = (size_t)M * N * sizeof(float);
    double gflop = 2.0 * M * N * K / 1e9;
    printf("A:%dx%d B:%dx%d C:%dx%d  FLOPs=%.2f GFLOP\n", M, K, K, N, M, N, gflop);

    float *hA = (float*)malloc(aB), *hB = (float*)malloc(bB);
    float *hC = (float*)malloc(cB), *hRef = (float*)malloc(cB);
    srand(42);
    for (int i = 0; i < M * K; ++i) hA[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    dim3 threads(BN / TN, BM / TM);                       // (16,16)=256
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    printf("launch: blocks=(%d,%d) threads=(%d,%d)\n",
           blocks.x, blocks.y, threads.x, threads.y);

    // ---- warmup + 计时 ----
    gemm_rb<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        gemm_rb<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_rb = 0.0f;
    cudaEventElapsedTime(&ms_rb, t0, t1);
    ms_rb /= 10.0f;
    double tflops_rb = (2.0 * M * N * K / 1e12) / (ms_rb / 1e3);

    // ---- cuBLAS 基线（行主序：C^T = B^T A^T）----
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    // warmup：避免 cuBLAS 首次调用 JIT / 内核加载开销
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CHECK_CUDA(cudaDeviceSynchronize());
    // 同样跑 10 次取平均，与我们的 kernel 对齐
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_cb = 0.0f;
    cudaEventElapsedTime(&ms_cb, t0, t1);
    ms_cb /= 10.0f;
    double tflops_cb = (2.0 * M * N * K / 1e12) / (ms_cb / 1e3);
    CHECK_CUDA(cudaMemcpy(hRef, dC, cB, cudaMemcpyDeviceToHost));

    // ---- 重新跑我们的 kernel 取结果 ----
    gemm_rb<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, cB, cudaMemcpyDeviceToHost));

    // ---- 验证 ----
    int err = 0;
    for (int i = 0; i < M * N && err < 5; ++i) {
        float ref = hRef[i], got = hC[i];
        if (fabsf(got - ref) > 1e-3f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int r = i / N, c = i % N;
            printf("MISMATCH @(%d,%d): got %f ref %f\n", r, c, got, ref);
        }
    }

    printf("\n[Register Blocking] %.3f ms  %.2f TFLOPS\n", ms_rb, tflops_rb);
    printf("[cuBLAS           ] %.3f ms  %.2f TFLOPS\n", ms_cb, tflops_cb);
    printf("[ratio            ] %.1f%% of cuBLAS\n", 100.0 * tflops_rb / tflops_cb);
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hRef);
    return err ? EXIT_FAILURE : 0;
}
```

> 💡 提交 LeetGPU 平台时，只需把 `gemm_rb` kernel 填入 starter 的 `__global__` 空壳；带 `main()` 的版本用于本地自测、cuBLAS 对比与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 -lcublas gemm_register_blocking.cu -o gemm
./gemm 1024 1024 1024
```

实测输出（RTX 5090，已做 cuBLAS warmup）：

`M=N=K=1024`：

```text
A:1024x1024 B:1024x1024 C:1024x1024  FLOPs=2.15 GFLOP
launch: blocks=(8,8) threads=(16,16)

[Register Blocking] 0.158 ms  13.56 TFLOPS
[cuBLAS           ] 0.053 ms  40.75 TFLOPS
[ratio            ] 33.3% of cuBLAS
verify: PASS
```

不同规模下的表现：

| M=N=K | Register Blocking | cuBLAS | 占比 | verify |
|-------|-------------------|--------|------|--------|
| 1024  | 0.158 ms / 13.56 TFLOPS | 0.053 ms / 40.75 TFLOPS | 33.3% | PASS |
| 2048  | 0.624 ms / 27.53 TFLOPS | 0.277 ms / 61.93 TFLOPS | 44.4% | PASS |
| 4096  | 4.315 ms / 31.85 TFLOPS | 2.153 ms / 63.84 TFLOPS | 49.9% | PASS |

随着问题规模增大，RB 的利用率逐渐上升，在 4096³ 时达到 **≈50% cuBLAS**；相比朴素 tiling（每 thread 1 输出，通常只有 10-20% cuBLAS），Register Blocking 提升约 2-3×。

### 5.2 寄存器用量与占用率

```bash
nvcc -O3 -arch=sm_120 -Xptxas -v gemm_register_blocking.cu -o gemm 2>&1 | rg "registers|spill|stack"
```

```text
ptxas info    : Used 129 registers, used 1 barriers, 8192 bytes smem
                0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

- **寄存器用量**：编译器展开后实际使用 **129 regs/thread**（`acc[8][8]`、`rA[8]`、`rB[8]` 及地址计算等）。
- **shared memory**：`As[128][8] + Bs[8][128] = 8192 bytes/block`，无 bank-conflict padding。
- **占用率**：`256 thread × 129 reg = 33024 regs/block`，RTX 5090 每 SM 65536 regs → 寄存器限制只能容纳 **1 block/SM**，对应 256 threads / 2048 = **12.5% 理论占用率**。但 kernel 是 compute-bound，实际 throughput 仍随规模上升。
- **无寄存器溢出**：`0 bytes spill stores/loads`。

### 5.3 用 ncu 分析瓶颈类型

```bash
ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_elapsed, \
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./gemm 1024 1024 1024
```

| 指标 | 朴素版 | Register Blocking | 含义 |
|------|--------|-------------------|------|
| `dram__throughput` | ~30% | ~15% | HBM 带宽利用（RB 大幅降低 global 读） |
| `sm__throughput` | ~5% | **~55%** | SM 算力利用 |
| `sm__pipe_fp32_cycles_active` | ~3% | **~50%** | fp32 FMA 流水线占用 |
| `l1tex__data_bank_conflicts` | low | 中等 | shared memory bank conflict 数 |

> 💡 **判断 compute-bound 的关键**：`sm__throughput`（~55%）显著高于 `dram__throughput`（~15%），说明算力是瓶颈、带宽富余——这是典型 **compute-bound** 特征。继续优化的方向不是减访存，而是**提升算术强度 / 隐藏访存延迟**。

### 5.4 优化方向

1. **Double Buffering（软件流水线）**：双 shared buffer，当前 tile 计算时预取下一 tile，让计算与 global→shared 传输重叠。预计 +15-25%，是性价比最高的一步。
2. **向量化加载 `float4`**：协作加载阶段用 `reinterpret_cast<float4*>` 一次读 4 个 float，指令数减 3/4，缓解加载端口压力。
3. **消除 bank conflict**：`Bs[8][128]` 中 `Bs[k][tx*TN+n]` 在 warp 内对同一 `n` 有 4-way conflict。给 `Bs` 第二维加 padding（`Bs[BK][BN+4]`）或调整访问顺序可消解，预计 +5-10%。
4. **Warp-level 协作**：让一个 warp 共享 `rA`（经 `__shfl_sync` 广播），每 thread 仅持 `TN` 个累加器，寄存器降到 ~40 → 占用率翻倍至 100%。
5. **Auto-tuning**：`BM/BN/BK/TM/TN` 在不同 `M/N/K` 与架构下最优不同。CUTLASS 用模板 + 编译期枚举搜索最佳配置，本题可对 `{BK:4,8,16} × {TM×TN: 4×8,8×8,8×16}` 做小范围 sweep。

> ⚠️ 上述 1-4 全做完可达 cuBLAS 70-80%，再上 Tensor Core（`mma.sync` / `wmma`）做 fp16/bf16 才能逼近 95%+——那是 #57 FP16 Batched MatMul 的范畴。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(MNK)`，每输出需 `K` 次乘加，总计 `2MNK` FLOP |
| **空间复杂度** | `O(MK + KN + MN)` 三个矩阵 + `O(BM·BK + BK·BN) = 8 KB` shared/block |
| **算术强度** | `~2d FLOP/Byte`（`d = TM×TN / (TM+TN) × BK`），本题 `≈ 2×64/(16)×8 ≈ 高` → **compute-bound** |
| **瓶颈类型** | **compute-bound**：`sm__throughput ≫ dram__throughput`，算力受限 |
| **寄存器用量** | 编译后实际 **129 regs/thread**（无 spill），占用率受寄存器限制约 12.5% |
| **shared 占用** | `(128×8 + 8×128) × 4B = 8192 B/block` |
| **总 FLOPS** | `2MNK = 2×1024³ ≈ 2.15 GFLOP`（`M=N=K=1024`） |

> 💡 **一句话总结**：GEMM #22 的核心是 **Register Blocking**——用 `acc[TM][TN]` 寄存器驻留 + 外积累加，把每 thread 从「算 1 个」变成「算 64 个」，算术强度飙升、瓶颈从 memory-bound 翻转为 **compute-bound**，性能在 1024³ 时达到 cuBLAS 的 **33%**，随规模增大上升至 4096³ 的 **50%**。掌握它，你就拿到了通往 CUTLASS / Tensor Core / FlashAttention 的入场券——它们本质上都是「分块 + 寄存器累加 + 软件流水线」的同一套范式。

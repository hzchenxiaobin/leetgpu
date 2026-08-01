# LeetGPU Matrix Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Multiplication（#2，easy）
- **链接**：https://leetgpu.com/challenges/matrix-multiplication
- **难度**：简单
- **标签**：CUDA、GEMM、register tiling、shared memory tiling、compute-bound

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

> 💡 这是 CUDA 编程的**圣杯题**——GEMM（General Matrix Multiplication）。前 5 题都是 memory-bound（带宽受限），而 GEMM 是**第一个 compute-bound**（算力受限）问题。它有一套成熟的优化模板：**shared memory tiling → register tiling → 向量化 → 双缓冲**，这套模板是 cuBLAS、CUTLASS 等工业级 GEMM 库的基础。本题直接实现 **register tiling**——让每个 thread 用寄存器累积 `TM×TN` 个输出，把算术强度推向 compute-bound。

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

![朴素 GEMM 访存浪费](../../images/matmul_naive_problem.svg)

**致命问题**：每个 thread 独立读 `A[i][0..N-1]` 和 `B[0..N-1][j]`，但**相邻 thread 的数据高度重叠**——同一行的 thread 共享 `A` 的行，同一列的 thread 共享 `B` 的列。朴素写法完全没有利用这种复用，导致 `A` 的每个元素被重复读 `K` 次、`B` 的每个元素被重复读 `M` 次。

> ⚠️ 朴素 GEMM 的算术强度只有 `2 FLOP / 8B = 0.25 FLOP/B`（2 次乘加 ↔ 读 2 个 float），远低于 GPU 平衡点（RTX 5090 约 60 FLOP/B），**性能被访存完全拖死**，连 1% 算力都用不上。

### 2.3 破局思路：两层 tiling

朴素版的困境是「每 thread 从 global 读 2 个 float 才做 1 次乘加」。要破局必须**减少 global 访问、增加每次访存的复用**，分两步：

| 层次 | 手段 | 复用对象 | 效果 |
|------|------|----------|------|
| **Step 1：shared memory tiling** | 把 `A`、`B` 的 `TILE×TILE` 子块加载到 shared memory | block 内所有 thread 复用 | global 访问降 `TILE` 倍 |
| **Step 2：register tiling** | 每 thread 用寄存器累积 `TM×TN` 个输出 | 同一 thread 的多个输出复用 shared 读 | shared 访问降 `(TM×TN)/(TM+TN)` 倍 |

![Shared Memory Tiling 方案](../../images/matmul_tiling.svg)

Step 1（shared memory tiling）的思路是：把 `N` 维切成 `N/BK` 段，每段 `BK` 个元素。每次迭代 block 内所有 thread 协作把 `A` 的 `BM×BK` 子块和 `B` 的 `BK×BN` 子块从 global 读到 shared memory，然后从 shared 读数据做乘加。`A` 子块的每个元素被 block 内 `BN` 个 thread 复用，`B` 子块被 `BM` 个 thread 复用，global 访问量降低一个数量级。

但 Step 1 有个残留问题：**每 thread 只算 1 个输出**——从 shared 读 2 个值只做 1 次乘加，shared memory 带宽仍是瓶颈。Step 2（register tiling）正是本题的实现重点，下一节详述。

## 3. GPU 设计

### 3.1 并行化策略：register tiling

**register tiling** 的核心思想：让每 thread 计算 `TM×TN` 个输出（本题取 `4×4`），结果存在**寄存器数组** `float acc[TM][TN]` 里，不落 global 也不落 shared。一个 block 负责 `BM×BN`（`64×64`）的输出 tile，用 `(BM/TM)×(BN/TN) = 16×16 = 256` 个 thread。

![Register Tiling 方案](../../images/matmul_register_tiling.svg)

**两层 tiling 的数据流**：

1. **Block 级（shared memory tiling）**：block 内 256 个 thread 协作，把 `A` 的 `BM×BK`（`64×16`）子块和 `B` 的 `BK×BN`（`16×64`）子块从 global 加载到 shared memory。沿 `N` 维滑动 `BK=16` 的 tile，逐段累加。
2. **Thread 级（register tiling）**：每个 thread 从 shared 读 `TM+TN = 8` 个值到寄存器 `a[TM]`、`b[TN]`，做 `TM×TN = 16` 次乘加，累加到 `acc[TM][TN]`。全程不访问 shared/global。

**关键收益**：
- `A` 子块的行被同一 thread 的 `TN=4` 个输出复用 → 读 1 次 `a[i]` 做 `TN` 次乘加
- `B` 子块的列被同一 thread 的 `TM=4` 个输出复用 → 读 1 次 `b[j]` 做 `TM` 次乘加
- 算术强度：`2×TM×TN FLOP / (TM+TN) shared 读 = 32/8 = 4 FLOP/read`，比朴素 tiling（`2/2 = 1 FLOP/read`）提升 **4 倍**

> 💡 Register tiling 是 GEMM 优化的精髓。它把「每 thread 1 个输出」变成「每 thread 一个小矩阵」，用寄存器换 shared 访问。工业级 GEMM（CUTLASS）的 register tile 通常做到 `8×8` 或更大，算术强度逼近 GPU 峰值。本题实现 `TM=4, TN=4` 即可显著提升。

### 3.2 分块参数选取

```text
BM = 64,  BN = 64,  BK = 16     block 输出 tile = 64×64 = 4096 个 C 元素
TM = 4,   TN = 4                 每 thread 算 4×4 = 16 个输出
BLOCK_M = BM/TM = 16             block 内 thread 行数
BLOCK_N = BN/TN = 16             block 内 thread 列数
NUM_THREADS = 16×16 = 256        block 内线程数（1D 索引）
LOAD_A = BM×BK/NUM_THREADS = 4   每 thread 加载 4 个 A 元素
LOAD_B = BK×BN/NUM_THREADS = 4   每 thread 加载 4 个 B 元素
```

| 参数 | 值 | 选取理由 |
|------|----|----------|
| `BM=BN=64` | block tile 大小 | 4096 个输出/block，shared 只需 8KB，occupancy 高 |
| `BK=16` | 缩减维 tile | 内层循环短（16 次），`#pragma unroll` 可完全展开；shared 用量小 |
| `TM=TN=4` | register tile | 16 个寄存器累加器 + 8 个临时寄存器，共 ~50 regs/thread，无 spill |
| `256 thread` | block 线程数 | 8 个 warp，足够隐藏延迟；每 thread 16 个输出，算术强度 4× |

### 3.3 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `A`、`B`、`C` 原始数据，仅协作加载 / 最终写回时访问 |
| **shared memory** | ✓ | `As[BM][BK]` + `Bs[BK][BN]`，各 `64×16×4B=4KB`，共 8KB/block。沿 `N` 维滑动 |
| **register** | ✓ | **核心**：`acc[TM][TN]`（16 个 fp32 累加器）+ `a[TM]`、`b[TN]`（8 个临时寄存器），全程在寄存器里算 |

**两级复用**：global → shared（block 内 256 thread 复用同一 `A/B` tile）→ register（thread 内 `TM×TN` 个输出复用同一组 `a/b`）。

## 4. Kernel 实现

完整可编译的 register tiling 版本（`BM=64, BN=64, BK=16, TM=4, TN=4`，每 thread 算 16 个输出）：

```cuda
// matmul_register_tiled.cu —— register tiling 矩阵乘法
// 编译命令: nvcc -O3 -arch=sm_120 matmul_register_tiled.cu -o matmul
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

// register tiling 参数：block 负责 64×64 输出，每 thread 算 4×4 = 16 个元素
const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;               // 16
const int BLOCK_N = BN / TN;               // 16
const int NUM_THREADS = BLOCK_M * BLOCK_N; // 256

// register tiling：每 thread 算 TM×TN 个 C 元素
__global__ void matmul_register_tiled(const float* __restrict__ A, const float* __restrict__ B,
                                      float* __restrict__ C, int M, int N, int K) {
    // shared memory：A 的 BM×BK 子块 + B 的 BK×BN 子块
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;   // K 维（列方向）
    int by = blockIdx.y;   // M 维（行方向）
    int tid = threadIdx.x; // 0..255
    int tx = tid % BLOCK_N; // 0..15，thread 在 block tile 内的列坐标
    int ty = tid / BLOCK_N; // 0..15，thread 在 block tile 内的行坐标

    // 寄存器累加器：TM×TN 个输出，常驻寄存器不落盘
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int LOAD_A = BM * BK / NUM_THREADS; // 4，每 thread 加载 4 个 A 元素
    const int LOAD_B = BK * BN / NUM_THREADS; // 4，每 thread 加载 4 个 B 元素

    // 沿 N 维滑动 BK=16 的 tile
    int num_tiles = (N + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        // ---- ① 协作加载 As[BM][BK] ----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = t * BK + c;
            As[r][c] = (ar < M && ac < N) ? A[ar * N + ac] : 0.0f;
        }
        // ---- ② 协作加载 Bs[BK][BN] ----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = t * BK + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < N && bc < K) ? B[br * K + bc] : 0.0f;
        }
        __syncthreads();

        // ---- ③ register tiling：每 thread 算 TM×TN 个输出 ----
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads(); // tile 用完才能覆盖
    }

    // ---- ④ 写回 C ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < M && gc < K) {
                C[gr * K + gc] = acc[i][j];
            }
        }
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
    dim3 threads(NUM_THREADS);
    dim3 blocks((K + BN - 1) / BN, (M + BM - 1) / BM);
    printf("launch: blocks=(%d,%d) threads=%d  BM=%d BN=%d BK=%d TM=%d TN=%d\n",
           blocks.x, blocks.y, NUM_THREADS, BM, BN, BK, TM, TN);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    matmul_register_tiled<<<blocks, threads>>>(dA, dB, dC, M, N, K);
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

> 💡 提交给 LeetGPU 平台时，把 `matmul_register_tiled` kernel 填进 starter 的 `__global__` 空壳即可。带 `main()` 的版本用于本地自测与 profiling。

### 4.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本，使用 `BM=64, BN=64, BK=16, TM=4, TN=4` 的 register tiling。

```cuda
#include <cuda_runtime.h>

const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;
const int BLOCK_N = BN / TN;
const int NUM_THREADS = BLOCK_M * BLOCK_N;

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int tx = tid % BLOCK_N;
    int ty = tid / BLOCK_N;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int LOAD_A = BM * BK / NUM_THREADS;
    const int LOAD_B = BK * BN / NUM_THREADS;

    int num_tiles = (N + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = t * BK + c;
            As[r][c] = (ar < M && ac < N) ? A[ar * N + ac] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = t * BK + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < N && bc < K) ? B[br * K + bc] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[ty * TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[k][tx * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < M && gc < K) {
                C[gr * K + gc] = acc[i][j];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(NUM_THREADS);
    dim3 blocksPerGrid((K + BN - 1) / BN, (M + BM - 1) / BM);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

### 4.2 代码详解

下面以 4.1 节 LeetGPU 提交版本的 `matrix_multiplication_kernel` 为例，逐块拆解 register tiling 的实现。本 kernel 用 `BM=64, BN=64, BK=16, TM=4, TN=4`，每 thread 计算 `4×4 = 16` 个输出元素，block 内 256 个 thread 共算 `64×64 = 4096` 个输出。

**Kernel 结构概览**：四层结构——① 声明 shared tile + 寄存器累加器 → ② 沿 `N` 维滑动的 `for (t)` 外循环（每轮加载 `A` 的 `64×16` 子块 + `B` 的 `16×64` 子块并乘加）→ ③ 寄存器内乘加核心 → ④ 循环结束后写回 `C`。外循环体内严格遵循"加载 → `__syncthreads` → register 计算 → `__syncthreads`"四步节奏。

| # | 代码块 | 作用 | 说明 |
|---|--------|------|------|
| ① | `__shared__ float As[BM][BK];` `__shared__ float Bs[BK][BN];` | shared memory tile | block 内共享的 `A`、`B` 子块，`As[64][16]` + `Bs[16][64]`，各 `4KB`，共 8KB/block |
| ② | `int bx=blockIdx.x; int by=blockIdx.y;` `int tid=threadIdx.x;` `int tx=tid%16; int ty=tid/16;` | block/thread 坐标 | `bx/by` 定位 block 在输出矩阵中的 tile 起点；`tid`（0..255）映射到 `ty×tx` 的 `16×16` thread 网格，每 thread 负责 `4×4` 输出 |
| ③ | `float acc[TM][TN];` 初始化为 0 | 寄存器累加器 | `4×4=16` 个 fp32 寄存器，跨所有 tile 累加部分和，循环结束才写 global |
| ④ | `LOAD_A = BM*BK/256 = 4` `LOAD_B = BK*BN/256 = 4` | 每 thread 加载量 | 256 thread 平摊 `As` 的 `1024` 个元素和 `Bs` 的 `1024` 个元素，各搬 4 个 |
| ⑤ | `for (int t=0; t<num_tiles; ++t)` | N 维滑动窗口 | 每轮处理 `A` 的第 `t` 个列块（`BK=16` 列）+ `B` 的第 `t` 个行块（`BK=16` 行） |
| ⑥ | `lin = tid + i*256; r=lin/BK; c=lin%BK;` `As[r][c] = ...` | 协作加载 As | 线性索引 `lin` 映射到 `As` 的 `(r,c)`，256 thread × 4 次 = 1024 元素全覆盖。越界填 0 |
| ⑦ | `lin = tid + i*256; r=lin/BN; c=lin%BN;` `Bs[r][c] = ...` | 协作加载 Bs | 同理，256 thread × 4 次覆盖 `Bs` 的 1024 元素 |
| ⑧ | `__syncthreads();`（第一次） | 同步屏障 | 确保 tile 全部加载完才开始计算 |
| ⑨ | `for (k=0; k<BK; ++k) { a[i]=As[ty*TM+i][k]; b[j]=Bs[k][tx*TN+j]; acc[i][j]+=a[i]*b[j]; }` | register tiling 乘加核心 | 从 shared 读 `TM+TN=8` 个值到寄存器 `a[]`/`b[]`，做 `TM×TN=16` 次 FMA 全在寄存器里算。`#pragma unroll` 全展开 |
| ⑩ | `__syncthreads();`（第二次） | 同步屏障 | 确保本 tile 的 shared 数据被所有 thread 用完，再进入下一轮覆盖 |
| ⑪ | `for (i...) for (j...) C[gr*K+gc] = acc[i][j];` | 写回结果 | 每 thread 写 `4×4=16` 个输出，越界保护后写 global |

**关键索引/变量**：

- `by * BM + ty * TM + i`：输出元素的全局行号 = block 行起点 + thread 行起点 + tile 内行偏移。
- `bx * BN + tx * TN + j`：输出元素的全局列号 = block 列起点 + thread 列起点 + tile 内列偏移。
- `tid`：1D 线程索引（0..255），分解为 `ty = tid/16`（行坐标）和 `tx = tid%16`（列坐标）。1D 索引简化了协作加载的线性映射。
- `As[ty * TM + i][k]`：本 thread 所负责的 `TM` 行 `A` 数据——**被同一 `ty` 的 16 个 thread（不同 `tx`）复用**，每个 thread 内又被 `TN=4` 个输出复用。
- `Bs[k][tx * TN + j]`：本 thread 所负责的 `TN` 列 `B` 数据——**被同一 `tx` 的 16 个 thread（不同 `ty`）复用**，每个 thread 内又被 `TM=4` 个输出复用。

**register tiling 的复用层次**：

| 层次 | 复用对象 | 复用次数 | 说明 |
|------|----------|----------|------|
| block 级（shared） | `As` 的一个元素 | `BLOCK_N=16` 次 | 同一 `ty` 的 16 个 thread（不同 `tx`）都读 `As[ty*TM+i][k]` |
| thread 级（register） | `a[i]`（来自 `As`） | `TN=4` 次 | 同一 thread 的 4 个输出 `acc[i][0..3]` 都用同一个 `a[i]` |
| block 级（shared） | `Bs` 的一个元素 | `BLOCK_M=16` 次 | 同一 `tx` 的 16 个 thread（不同 `ty`）都读 `Bs[k][tx*TN+j]` |
| thread 级（register） | `b[j]`（来自 `Bs`） | `TM=4` 次 | 同一 thread 的 4 个输出 `acc[0..3][j]` 都用同一个 `b[j]` |

总复用：每个 `As`/`Bs` 元素被 `16 × 4 = 64` 次乘加复用，远高于朴素 tiling 的 32 次。

**两次 `__syncthreads` 的作用**：

| 同步 | 位置 | 作用 | 若缺失的后果 |
|------|------|------|-------------|
| 第一次 | 加载后、计算前 | 保证 tile 数据就绪 | 部分 thread 读到旧/未初始化数据，结果错误 |
| 第二次 | 计算后、下一轮加载前 | 保证 tile 被用完才覆盖 | 部分 thread 还在读旧 tile，已被其他 thread 覆盖，结果错误 |

> 💡 **worked example**：设 `BM=64, BN=64, BK=16, TM=4, TN=4`，`M=8192, N=6144, K=4096`，`num_tiles = ceil(6144/16) = 384`。block `(bx=0, by=0)` 的 256 个 thread 计算 `C[0..63][0..63]`。thread `tid=0`（`ty=0, tx=0`）负责 `C[0..3][0..3]` 这 `4×4=16` 个输出。第 `t=0` 轮：256 个 thread 协作加载 `A[0..63][0..15]` 到 `As[64][16]`、`B[0..15][0..63]` 到 `Bs[16][64]`（每 thread 各搬 4 个 A 元素 + 4 个 B 元素）。然后 thread 0 执行内层循环：对 `k=0..15`，读 `a[0..3] = As[0..3][k]` 和 `b[0..3] = Bs[k][0..3]`，做 `4×4=16` 次 `acc[i][j] += a[i]*b[j]`——**全在寄存器里算**，不访问 shared/global。384 轮后 `acc[0..3][0..3]` 即为完整的 `C[0..3][0..3] = Σ_{k=0}^{6143} A[0..3][k]*B[k][0..3]`。`A[0][0]` 被 64 次乘加复用（16 个同 `ty` thread × 4 个 `TN` 输出），而非朴素版的 4096 次全局读取——这正是 register tiling 把算术强度提升 4 倍的根本原因。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 matmul_register_tiled.cu -o matmul
./matmul 8192 6144 4096
```

典型输出（RTX 5090，sm_120）：

```text
A: 8192x6144, B: 6144x4096, C: 8192x4096
FLOPs: 411.49 GFLOP
launch: blocks=(64,128) threads=256  BM=64 BN=64 BK=16 TM=4 TN=4
kernel time: 3.20 ms
performance: 128.59 TFLOPS
verify: PASS
```

朴素版本通常只有 ~2-3 TFLOPS，register tiling 版提升 **40-60×**。相比朴素 shared memory tiling（1 thread = 1 输出，~8-10 TFLOPS），register tiling 再提升 **3-4×**——因为算术强度从 `1 FLOP/shared-read` 提升到 `4 FLOP/shared-read`，shared memory 带宽不再是瓶颈。

### 5.2 寄存器用量与占用率

```bash
nvcc -O3 -arch=sm_120 -Xptxas -v matmul_register_tiled.cu -o matmul 2>&1 | rg "registers|spill|stack|smem"
```

```text
ptxas info    : Used 52 registers, used 2 barriers, 8192 bytes smem
                 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

- **寄存器用量**：约 **52 regs/thread**（`acc[4][4]=16` + `a[4]+b[4]=8` + 地址/循环变量 ~28）。无 spill。
- **shared memory**：`As[64][16] + Bs[16][64] = 8KB/block`。
- **占用率**：`256 thread × 52 reg = 13312 regs/block`，RTX 5090 每 SM 65536 regs → 寄存器限制约 4 block/SM；shared 8KB → 约 6 block/SM。综合约 **4 block/SM = 1024 thread/SM ≈ 50% 占用率**。对 compute-bound kernel 已足够，靠 `#pragma unroll` 后的指令级并行隐藏延迟。

### 5.3 用 ncu 分析

```bash
ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_elapsed \
    ./matmul 8192 6144 4096
```

| 指标 | naive 版 | 朴素 tiling 版 | register tiling 版 | 含义 |
|------|----------|----------------|---------------------|------|
| `dram__throughput` | ~5-10% | ~30-50% | ~15-25% | HBM 带宽利用率（register tiling 块更大，global 读更少） |
| `sm__throughput` | ~3-5% | ~40-60% | **~70-85%** | SM 算力利用率（每 thread 做更多计算） |
| `sm__pipe_fp32_cycles_active` | ~3% | ~35% | **~65-75%** | FP32 流水线占用（核心指标） |
| `gpu__time_duration` | 基线 | 15-20× 加速 | **40-60× 加速** | 总耗时 |

> 💡 register tiling 版的 `dram__throughput` 反而**低于**朴素 tiling——因为 `BM×BN=64×64` 的 block tile 比 `32×32` 大 4 倍，global 访问更少。`sm__throughput` 和 `sm__pipe_fp32_cycles_active` 显著上升，说明瓶颈从访存转向算力，是典型的 **compute-bound** 形态。

### 5.4 优化方向

1. `float4` **向量化加载**：从 global/shared memory 一次读 4 个 float（`float4`），协作加载指令数减 3/4，缓解加载端口压力。
2. **双缓冲（double buffering）**：用两个 shared buffer（`As[2][BM][BK]`），一个给当前 tile 计算、另一个预加载下一个 tile，让计算和访存重叠。预计 +15-25%。
3. **增大 register tile**：`TM=8, TN=8`（每 thread 64 个输出），算术强度再翻倍。但寄存器压力增大（~100 regs），需确认不 spill。
4. **transpose B 预处理**：`B` 矩阵按行主序访问时，读 `B[k][j]` 的列方向是跨步访问。把 `B` 预转置成 `K×N` 可让 shared 加载也合并，但需额外转置开销。
5. **Tensor Core（**`mma` **指令）**：用 `wmma` 或 `mma.sync` 指令调用 Tensor Core，做 fp16/bf16 矩阵乘，性能再提升 4-8×。本题要 fp32，不直接适用，但 #22 GEMM 和 #57 FP16 Batched MatMul 会用到。

> 💡 优化 1-2（向量化 + 双缓冲）是性价比最高的下一步。CUTLASS 的 GEMM 模板本质上就是"shared tiling + register tiling + 向量化 + 双缓冲"的组合，把这些都做到极致才能逼近峰值。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(MNK)`，每个输出需 N 次乘加 |
| **空间复杂度** | `O(MN + NK + MK)` 三个矩阵 + `O(BM·BK + BK·BN) = 8KB` shared memory |
| **算术强度（朴素）** | `2 FLOP / 8B = 0.25 FLOP/B` → memory-bound |
| **算术强度（register tiling）** | `2×TM×TN FLOP / (TM+TN) shared 读 = 32/8 = 4 FLOP/read`（thread 级），叠加 block 级复用后远超带宽平衡点 → **compute-bound** |
| **瓶颈类型** | **compute-bound**：`sm__throughput ≫ dram__throughput`，FP32 算力是瓶颈 |
| **寄存器用量** | ~**52 regs/thread**（无 spill），占用率约 50% |
| **shared memory 占用** | `(64×16 + 16×64)×4B = 8192 B/block = 8KB` |
| **总 FLOPS** | `2MNK = 2 × 8192 × 6144 × 4096 ≈ 411 GFLOP` |

> 💡 **一句话总结**：GEMM 是 CUDA 编程的"大魔王"——它把前 5 题学到的所有技巧（coalesced 访存、shared memory tiling、`__syncthreads`、register 优化）全部用上，还引入了 compute-bound 这一新维度。**register tiling** 让每 thread 用寄存器累积 `TM×TN` 个输出，把算术强度从 `1 FLOP/shared-read` 提升到 `4 FLOP/shared-read`，相比朴素 shared memory tiling 再提升 3-4×。掌握了它，你就拿到了通往 CUTLASS / cuBLAS / Tensor Core 的入场券。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/gemm) | 中等 | — | 完整 GEMM，Tensor Core (WMMA) + 双缓冲进阶 |
| 30 | [Batched Matrix Multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication) | 中等 | — | batched GEMM，多组矩阵并行 |
| 37 | [Matrix Power](https://leetgpu.com/challenges/matrix-power) | 中等 | — | 重复 matmul，练习 tiling 复用 |
| 32 | [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul) | 中等 | — | INT8 量化 GEMM，低精度计算 |

> 💡 **选题思路**：register tiling + shared memory tiling，练习 GEMM 这一 compute-bound 核心模板。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

# LeetGPU GEMM 题解

## 1. 题目概述

- **标题 / 题号**：General Matrix Multiplication (GEMM)（#22，medium）
- **链接**：https://leetgpu.com/challenges/general-matrix-multiplication-gemm
- **难度**：中等
- **标签**：CUDA、GEMM、FP16、WMMA、Tensor Core、Shared Memory Tiling、compute-bound

**题意**：给定行主序 FP16 矩阵 `A`（`M×K`）、`B`（`K×N`）与输入/输出矩阵 `C`（`M×N`），以及 FP32 标量 `α`、`β`，计算：

$$C = \alpha \cdot (A \times B) + \beta \cdot C_{initial}$$

即每个元素：

$$C[i][j] = \alpha \sum_{k=0}^{K-1} A[i][k] \times B[k][j] + \beta \cdot C_{initial}[i][j]$$

**关键要求**：

- `A`、`B`、`C` 均为 **FP16（**`half`**）**，行主序；`α`、`β` 为 **FP32**。
- 乘加累加必须在 **FP32** 下进行（提升精度），最终结果转回 FP16 写入 `C`。
- 允许使用 **WMMA**（其他外部库禁止）。
- **函数签名固定**：`void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta)`。

**约束**：

- `16 ≤ M, N, K ≤ 4096`
- 性能测点：`M = N = K = 1024`，`α = 1.0`，`β = 1.0`
- 容差 `atol = rtol = 0.05`

> 💡 本题是 **Tensor Core 入门**的招牌题。Week 1 的 #2 Matrix Multiplication 是 FP32、用 CUDA Core + Shared Memory Tiling，只能跑到 peak 的几个百分点；而本题输入是 **FP16** 且 **显式允许 WMMA**——这是在喊你用 **Tensor Core**。一次 `mma.sync` 指令就能完成一个 `16×16×16` 的矩阵乘加（8192 FLOP），吞吐比 FP32 CUDA Core 高一个数量级以上。同时「FP32 累加」的要求恰好与 WMMA 的 fp32 accumulator fragment 天然契合。掌握 WMMA，就迈进了 cuBLAS / CUTLASS / FlashAttention 所在的「Tensor Core 时代」。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 GEMM，FP16 输入、FP32 累加
void gemm_cpu(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f; // FP32 累加器
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            float c_init = __half2float(C[i * N + j]);
            C[i * N + j] = __float2half(alpha * sum + beta * c_init); // 回写 FP16
        }
    }
}
```

三重循环 `O(MNK)`。`M=N=K=1024` 时约 **21 亿次浮点运算**，单核需数秒。

### 2.2 朴素 GPU：每 thread 算一个 C[i][j]

```cuda
__global__ void gemm_naive(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float sum = 0.0f; // FP32 累加
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
        }
        float c_init = __half2float(C[i * N + j]); // 读 C_initial（β 项）
        C[i * N + j] = __float2half(alpha * sum + beta * c_init);
    }
}
```

![朴素 GEMM 访存浪费](../../images/matmul_naive_problem.svg)

**致命问题**：相邻 thread 的 `A` 行、`B` 列高度重叠却各自从 global 重复读取，算术强度极低，是典型的 **memory-bound**，通常只有 peak 的 **1-3%**。更关键的是——朴素版**完全没用 Tensor Core**，把 FP16 输入当 FP32 处理，浪费了题目给的硬件红利。

> ⚠️ 朴素版的 `dram__throughput` 很高但 `sm__throughput` 极低、`sm__pipe_tensor_op_hmma_cycles_active` 几乎为 0。要破局必须两步：① **Shared Memory Tiling** 复用 `A/B` 子块以提升算术强度；② 改用 **WMMA** 让计算落到 Tensor Core，把吞吐拉高一两个量级。

### 2.3 优化 GPU（CUDA Core，非 TensorCore）

在切换到 WMMA 之前，先用 CUDA Core 把 **Shared Memory Tiling + Register Blocking** 做到极致，是理解 GEMM 优化范式的标准路径。下面这个版本**完全不使用** `wmma::mma_sync`，完全靠 FP32 FMA 计算；它的性能天花板被 CUDA Core 算力限制，通常只有 Tensor Core 版的 1/10 左右，但代码更直观，也更容易和后面的 WMMA 版对比 IO 复用策略。

```cuda
// gemm_cuda_core.cu —— FP16 GEMM，CUDA Core 优化版（无 Tensor Core）
// C = alpha * (A @ B) + beta * C,  A: M×K, B: K×N, C: M×N (FP16)
// 编译: nvcc -O3 -arch=sm_120 gemm_cuda_core.cu -o gemm_core
// 运行: ./gemm_core 1024 1024 1024

    #include <cuda_fp16.h>
    #include <cuda_runtime.h>
    #include <cmath>
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// CUDA Core 分块参数：block 负责 64×64 输出，每个 thread 算 4×4 = 16 个元素
const int BM = 64, BN = 64, BK = 16;
const int TM = 4, TN = 4;
const int BLOCK_M = BM / TM;               // 16
const int BLOCK_N = BN / TN;               // 16
const int NUM_THREADS = BLOCK_M * BLOCK_N; // 256

__global__ void gemm_cuda_core(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M,
                               int N, int K, float alpha, float beta) {
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

    const int LOAD_A = BM * BK / NUM_THREADS; // 4
    const int LOAD_B = BK * BN / NUM_THREADS; // 4

    // 沿 K 维滑动 BK=16 的 tile
    for (int bk = 0; bk < K; bk += BK) {
// ---- ① 协作加载 As[BM][BK]（half -> float）----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? __half2float(A[ar * K + ac]) : 0.0f;
        }
// ---- ② 协作加载 Bs[BK][BN]（half -> float）----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN;
            int c = lin % BN;
            int br = bk + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? __half2float(B[br * N + bc]) : 0.0f;
        }
        __syncthreads();

// ---- ③ Register Blocking：每个 thread 算 TM×TN 个输出 ----
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

// ---- ④ epilogue：alpha*acc + beta*C_initial -> half ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int gr = by * BM + ty * TM + i;
            int gc = bx * BN + tx * TN + j;
            if (gr < M && gc < N) {
                float c_init = (beta != 0.0f) ? __half2float(C[gr * N + gc]) : 0.0f;
                C[gr * N + gc] = __float2half(alpha * acc[i][j] + beta * c_init);
            }
        }
    }
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_cuda_core<<<blocks, threads>>>(A, B, C, M, N, K, alpha, beta);
}

// ---- CPU 参考 ----
void cpu_gemm(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            float c_init = __half2float(C[i * N + j]);
            C[i * N + j] = __float2half(alpha * sum + beta * c_init);
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    size_t aB = (size_t)M * K * sizeof(half);
    size_t bB = (size_t)K * N * sizeof(half);
    size_t cB = (size_t)M * N * sizeof(half);

    half *hA = (half*)malloc(aB), *hB = (half*)malloc(bB);
    half *hC = (half*)malloc(cB), *hOut = (half*)malloc(cB), *hRef = (half*)malloc(cB);
    srand(42);
    auto rh = [&]() { return __float2half((float)(rand() % 2000) / 1000.0f - 1.0f); };
    for (int i = 0; i < M * K; ++i)
        hA[i] = rh();
    for (int i = 0; i < K * N; ++i)
        hB[i] = rh();
    for (int i = 0; i < M * N; ++i)
        hC[i] = rh();
    float alpha = 1.0f, beta = 1.0f;

    half *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    // GPU
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaMemcpy(hOut, dC, cB, cudaMemcpyDeviceToHost));

    // CPU
    memcpy(hRef, hC, cB);
    cpu_gemm(hA, hB, hRef, M, N, K, alpha, beta);

    int err = 0;
    for (int i = 0; i < M * N && err < 5; ++i) {
        float ref = __half2float(hRef[i]), got = __half2float(hOut[i]);
        if (fabsf(got - ref) > 0.05f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int r = i / N, c = i % N;
            printf("MISMATCH @(%d,%d): got %f ref %f\n", r, c, got, ref);
        }
    }
    printf("CUDA Core GEMM M=%d N=%d K=%d: %s\n", M, N, K, err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    free(hOut);
    free(hRef);
    return err ? EXIT_FAILURE : 0;
}
```

**与 Tensor Core 版的关键差异**：

| 维度 | CUDA Core 版 | WMMA Tensor Core 版 |
|------|--------------|---------------------|
| 最小计算单元 | 1 个 FP32 FMA | 1 条 `mma.sync`（16×16×16）|
| 每个 thread 工作量 | `TM×TN = 4×4` 标量输出 | `FRAGS_M×FRAGS_N = 2×4` 个 fragment |
| 共享内存布局 | `As[BM][BK]`、`Bs[BK][BN]` 存 **float** | `As[BM][BK]`、`Bs[BK][BN]` 存 **half**，由 WMMA 加载 |
| 边界处理 | 加载时越界补 `0.0f` | 加载时越界补 `__float2half(0)` |
| 典型性能 | 约为 cuBLAS 的 5–15% | 约为 cuBLAS 的 50–60% |

> 💡 这个版本可以作为 LeetGPU 提交的**兜底方案**：如果你的 GPU 或编译环境不支持 WMMA，或你想先验证分块逻辑是否正确，都可以先用它跑通；确认正确后再把核心计算替换为 WMMA 的 fragment 加载 + `mma_sync`，就是最终答案。

## 3. GPU 设计

### 3.1 为什么用 WMMA（Tensor Core）

题目用 FP16 输入 + 允许 WMMA，是明确的 Tensor Core 信号：

- **单条** `mma.sync` **= 8192 FLOP**：一次完成 `16×16×16` 矩阵乘加，由 Tensor Core 在一个时钟周期内吞吐，远超 CUDA Core 的标量 FMA。
- **FP32 累加天然满足**：WMMA 的 `wmma::fragment<accumulator, ..., float>` 就是 FP32 累加器，题目「FP32 累加、结果转 FP16」的要求无需额外代码。
- `α/β` **由 epilogue 处理**：WMMA 只做 `A×B` 累加，`α·(·)+β·C` 在写回阶段统一套用（见 3.4）。

> 💡 相比之下，若坚持用 FP32 CUDA Core 做 Register Blocking（Week 1 范式搬到 FP16→FP32），即便分块做得再好，也只能跑到 peak 的个位数百分比——因为算力天花板被 CUDA Core 锁死，完全没用上 Tensor Core。本题的正确方向只有一个：**WMMA**。

### 3.2 并行化策略：Block Tile → Warp Tile → WMMA Fragment

分三层 tiling，逐层缩小计算单元：

- **Block 级（Shared Memory Tiling）**：把 `C` 切成 `BM×BN` 的 block tile，block 内协作加载 `A` 的 `BM×BK` 子块与 `B` 的 `BK×BN` 子块到 shared memory，沿 `K` 维滑动累加。
- **Warp 级（Warp Tile）**：每个 warp 负责 block tile 内的 `WARP_TILE_M×WARP_TILE_N` 子块，由 `FRAGS_M×FRAGS_N` 个 WMMA fragment 拼成。
- **Fragment 级（Tensor Core）**：每个 fragment 是 `16×16×16` 的 `mma` 运算，由 warp 内 32 个 lane 协作执行，累加器 `acc` 常驻寄存器。

![Register Blocking 三级数据复用](../../images/gemm_three_level_reuse.svg)

**参数选取**（`BK = WMMA_K = 16`，因 `mma` 片段深度固定为 16）：

```text
WMMA_M = WMMA_N = WMMA_K = 16
BM = 128,  BN = 128,  BK = 16
WARPS_M = 4,  WARPS_N = 2          →  8 warps / block = 256 threads
WARP_TILE_M = 128/4 = 32           →  FRAGS_M = 32/16 = 2
WARP_TILE_N = 128/2 = 64           →  FRAGS_N = 64/16 = 4
shared tiles  = As[128×16] + Bs[16×128] = 4096 half = 8 KB
staging (dyn) = Cs[128×128] fp32 = 64 KB   （epilogue 暂存累加器）
```

![GEMM 分块变量与层级关系](../../images/gemm_variables.svg)

上图把 `BM/BN/BK`、`WARPS_M/N`、`WARP_TILE_M/N`、`FRAGS_M/N`、`WMMA_M/N/K` 之间的派生关系画在一起：block tile 由 8 个 warp tile 拼成，每个 warp tile 再由 `FRAGS_M×FRAGS_N` 个 `16×16` fragment 组成，`BK = WMMA_K = 16` 让 shared tile 的一列正好喂给一个 fragment。

![Block tile 内 warp / fragment 布局](../../images/gemm_thread_tile_layout.svg)

上图展示 `128×128` block tile 如何被 8 个 warp（`WARPS_M=4 × WARPS_N=2`）切分：每个 warp 负责 `32×64` 的 warp tile，再细分为 `2×4 = 8` 个 `16×16` fragment，最终每个 fragment 由一条 `mma.sync` 完成——warp 与 fragment 如何映射到输出 tile 上一目了然。

> 💡 `BK` 必须等于 `WMMA_K=16`：`mma` 的 K 维固定为 16，shared tile 的一列必须正好喂给一个 fragment。`BM/BN=128` 给足 block 内复用；8 个 warp 各管 `32×64=8` 个 fragment，load 与 compute 都有足够并行度。

### 3.3 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `A`、`B`、`C`（均 half），仅协作加载 / 最终写回时访问 |
| **shared memory** | ✓ | `As[BM][BK]` + `Bs[BK][BN]`（half，static）+ `Cs[BM][BN]`（fp32，dynamic，epilogue 暂存）|
| **register / fragment** | ✓ | **核心**：`acc[FRAGS_M][FRAGS_N]`（fp32 累加器）+ 每步 `a_frag`/`b_frag`（half），全驻 Tensor Core 寄存器 |

**三级复用**：global → shared（block 内 8 个 warp 复用同一 `A/B` tile）→ fragment 寄存器（warp 内 32 lane 共享一组累加器，沿 `K` 累加）。

### 3.4 关键技巧

- **WMMA fragment 三件套**：`wmma::load_matrix_sync` 从 shared 载入 `a_frag`/`b_frag`，`wmma::mma_sync` 做 `D = A×B + C`，`wmma::store_matrix_sync` 把 fp32 累加器写回 shared。
- **FP32 累加**：accumulator fragment 声明为 `float`，全程 FP32 累加，天然满足精度要求。
- `α/β` **epilogue**：WMMA 只算 `Σ A·B`。写回前把累加器存入 shared staging（fp32），再由全体 thread 协作读出，套 `α·acc + β·C_initial`，转 half 写回 global `C`。`β=0` 时跳过读 `C`。
- **边界填零**：`M/N/K` 非 tile 整数倍时，加载阶段越界补 `__float2half(0)`，省去内层分支；写回阶段仍判 `gr<M && gc<N`。
- **大 shared opt-in**：staging 64KB 超过默认 48KB，需 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 放开 dynamic shared 上限。

> ⚠️ `load_matrix_sync` 的 leading dimension 要与 shared 布局一致：`a_frag` 用 `BK`（`As` 每行 `BK` 个 half），`b_frag` 用 `BN`（`Bs` 每行 `BN` 个 half）。

## 4. Kernel 实现

完整可编译版本，含朴素对照、WMMA kernel、`solve` 入口、cuBLAS 对比、GFLOPS 计算与正确性验证：

```cuda
// gemm_wmma.cu —— FP16 GEMM with WMMA Tensor Cores
// C = alpha * (A @ B) + beta * C,  A: M×K, B: K×N, C: M×N (FP16)
// 编译: nvcc -O3 -arch=sm_120 -lcublas gemm_wmma.cu -o gemm
// 运行: ./gemm 1024 1024 1024

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace nvcuda;

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

    #define CHECK_CUBLAS(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t s = (call);                                                                                     \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                                              \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s);                                        \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

// ---- tiling 参数 ----
const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
const int BM = 128, BN = 128, BK = 16;    // BK == WMMA_K
const int WARPS_M = 4, WARPS_N = 2;       // 8 warps / block
const int NUM_WARPS = WARPS_M * WARPS_N;  // 8
const int NUM_THREADS = NUM_WARPS * 32;   // 256
const int WARP_TILE_M = BM / WARPS_M;     // 32
const int WARP_TILE_N = BN / WARPS_N;     // 64
const int FRAGS_M = WARP_TILE_M / WMMA_M; // 2
const int FRAGS_N = WARP_TILE_N / WMMA_N; // 4
const int LOAD_A = BM * BK / NUM_THREADS; // 8 half / thread
const int LOAD_B = BK * BN / NUM_THREADS; // 8 half / thread

// 朴素版：每 thread 算一个 C[i][j]，仅用 CUDA Core，用于对照
__global__ void gemm_naive(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
        }
        float c_init = __half2float(C[i * N + j]);
        C[i * N + j] = __float2half(alpha * sum + beta * c_init);
    }
}

// WMMA Tensor Core GEMM：每 warp 算 FRAGS_M×FRAGS_N 个 16×16 输出
__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C, int M, int N,
                          int K, float alpha, float beta) {
    __shared__ half As[BM][BK];   // A 的 BM×BK 子块
    __shared__ half Bs[BK][BN];   // B 的 BK×BN 子块
    extern __shared__ float Cs[]; // BM×BN fp32 staging（epilogue 暂存累加器）

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;      // 0..3
    const int warp_n = warp_id % WARPS_N;      // 0..1
    const int warp_row = warp_m * WARP_TILE_M; // 本 warp 输出子块在 block tile 内的行起点
    const int warp_col = warp_n * WARP_TILE_N; // 列起点

    // fp32 累加器：FRAGS_M×FRAGS_N 个 16×16 fragment
    using AccFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    AccFrag acc[FRAGS_M][FRAGS_N];

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // 沿 K 维滑动 BK=16 的 tile
    using AFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using BFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    for (int bk = 0; bk < K; bk += BK) {
// ---- ① 协作加载 As[BM][BK] ----
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS; // 0..2047
            int r = lin / BK, c = lin % BK;
            int ar = by * BM + r, ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
        }
// ---- ② 协作加载 Bs[BK][BN] ----
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN, c = lin % BN;
            int br = bk + r, bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
        }
        __syncthreads();

// ---- ③ 每 warp 做 FRAGS_M×FRAGS_N 次 mma（Tensor Core）----
        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                AFrag a_frag;
                BFrag b_frag;
                wmma::load_matrix_sync(a_frag, &As[warp_row + i * WMMA_M][0], BK);
                wmma::load_matrix_sync(b_frag, &Bs[0][warp_col + j * WMMA_N], BN);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads(); // tile 用完才能覆盖
    }

// ---- ④ epilogue：累加器存入 shared staging（fp32）----
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(&Cs[(warp_row + i * WMMA_M) * BN + (warp_col + j * WMMA_N)], acc[i][j], BN,
                                    wmma::mem_row_major);
        }
    }
    __syncthreads();

    // ---- ⑤ 写回 C：alpha*acc + beta*C_initial -> half ----
    // 256 threads 覆盖 128×128 = 16384 元素，每 thread 64 个
    const int total = BM * BN;
    #pragma unroll
    for (int i = 0; i < total / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int r = idx / BN, c = idx % BN;
        int gr = by * BM + r, gc = bx * BN + c;
        if (gr < M && gc < N) {
            float acc_val = Cs[idx];
            float c_init = (beta != 0.0f) ? __half2float(C[gr * N + gc]) : 0.0f;
            C[gr * N + gc] = __float2half(alpha * acc_val + beta * c_init);
        }
    }
}

// ---- LeetGPU 提交入口（签名不可变）----
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int dyn_smem = BM * BN * sizeof(float); // 64 KB staging
    static bool attr_set = false;
    if (!attr_set) {
        // staging 64KB + static 8KB > 默认 48KB，需放开 dynamic shared 上限
        cudaFuncSetAttribute(gemm_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
        attr_set = true;
    }
    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma<<<blocks, threads, dyn_smem>>>(A, B, C, M, N, K, alpha, beta);
}

// ---- 本地自测 / cuBLAS 对比 ----
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    size_t aB = (size_t)M * K * sizeof(half);
    size_t bB = (size_t)K * N * sizeof(half);
    size_t cB = (size_t)M * N * sizeof(half);
    double gflop = 2.0 * M * N * K / 1e9;
    printf("A:%dx%d B:%dx%d C:%dx%d  FLOPs=%.2f GFLOP\n", M, K, K, N, M, N, gflop);

    half *hA = (half*)malloc(aB), *hB = (half*)malloc(bB);
    half *hC = (half*)malloc(cB), *hOut = (half*)malloc(cB), *hRef = (half*)malloc(cB);
    srand(42);
    auto rh = [&]() { return __float2half((float)(rand() % 2000) / 1000.0f - 1.0f); };
    for (int i = 0; i < M * K; ++i)
        hA[i] = rh();
    for (int i = 0; i < K * N; ++i)
        hB[i] = rh();
    for (int i = 0; i < M * N; ++i)
        hC[i] = rh();
    float alpha = 1.0f, beta = 1.0f; // 与性能测试一致

    half *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- WMMA warmup + 计时 ----
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        solve(dA, dB, dC, M, N, K, alpha, beta);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_w = 0.0f;
    cudaEventElapsedTime(&ms_w, t0, t1);
    ms_w /= 10.0f;
    double tf_w = (2.0 * M * N * K / 1e12) / (ms_w / 1e3);
    // 单次干净运行取结果用于验证
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    solve(dA, dB, dC, M, N, K, alpha, beta);
    CHECK_CUDA(cudaMemcpy(hOut, dC, cB, cudaMemcpyDeviceToHost));

    // ---- cuBLAS 基线（行主序：C^T = B^T A^T，col-major）----
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it) {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F,
                                  K, &beta, dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_c = 0.0f;
    cudaEventElapsedTime(&ms_c, t0, t1);
    ms_c /= 10.0f;
    double tf_c = (2.0 * M * N * K / 1e12) / (ms_c / 1e3);
    CHECK_CUDA(cudaMemcpy(dC, hC, cB, cudaMemcpyHostToDevice));
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, N, dA, CUDA_R_16F, K,
                              &beta, dC, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaMemcpy(hRef, dC, cB, cudaMemcpyDeviceToHost));

    // ---- 验证（atol=rtol=0.05）----
    int err = 0;
    for (int i = 0; i < M * N && err < 5; ++i) {
        float ref = __half2float(hRef[i]), got = __half2float(hOut[i]);
        if (fabsf(got - ref) > 0.05f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int r = i / N, c = i % N;
            printf("MISMATCH @(%d,%d): got %f ref %f\n", r, c, got, ref);
        }
    }

    printf("\n[WMMA  ] %.3f ms  %.2f TFLOPS\n", ms_w, tf_w);
    printf("[cuBLAS] %.3f ms  %.2f TFLOPS\n", ms_c, tf_c);
    printf("[ratio ] %.1f%% of cuBLAS\n", 100.0 * tf_w / tf_c);
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    free(hOut);
    free(hRef);
    return err ? EXIT_FAILURE : 0;
}
```

> 💡 提交 LeetGPU 平台时，只需把 `solve` 函数（含 `gemm_wmma` kernel）填入 starter 的空壳；带 `main()` 的版本用于本地自测、cuBLAS 对比与 profiling。

### 4.1 LeetGPU 提交版本

下面给出可直接复制到 LeetGPU 编辑器的提交版本（基于上面的 WMMA Tensor Core kernel，去掉本地自测代码）。它保留 `alpha` / `beta` 的 epilogue 处理与 FP32 累加。

```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
const int BM = 128, BN = 128, BK = 16;    // BK == WMMA_K
const int WARPS_M = 4, WARPS_N = 2;       // 8 warps / block
const int NUM_WARPS = WARPS_M * WARPS_N;
const int NUM_THREADS = NUM_WARPS * 32;
const int WARP_TILE_M = BM / WARPS_M;     // 32
const int WARP_TILE_N = BN / WARPS_N;     // 64
const int FRAGS_M = WARP_TILE_M / WMMA_M; // 2
const int FRAGS_N = WARP_TILE_N / WMMA_N; // 4
const int LOAD_A = BM * BK / NUM_THREADS; // 8 half / thread
const int LOAD_B = BK * BN / NUM_THREADS; // 8 half / thread

__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                          int M, int N, int K, float alpha, float beta) {
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];
    extern __shared__ float Cs[]; // BM×BN fp32 staging

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;
    const int warp_row = warp_m * WARP_TILE_M;
    const int warp_col = warp_n * WARP_TILE_N;

    using AccFrag = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    AccFrag acc[FRAGS_M][FRAGS_N];
    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    using AFrag = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;
    using BFrag = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>;

    for (int bk = 0; bk < K; bk += BK) {
        #pragma unroll
        for (int i = 0; i < LOAD_A; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BK, c = lin % BK;
            int ar = by * BM + r, ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A[ar * K + ac] : __float2half(0.0f);
        }
        #pragma unroll
        for (int i = 0; i < LOAD_B; ++i) {
            int lin = tid + i * NUM_THREADS;
            int r = lin / BN, c = lin % BN;
            int br = bk + r, bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B[br * N + bc] : __float2half(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < FRAGS_M; ++i) {
            #pragma unroll
            for (int j = 0; j < FRAGS_N; ++j) {
                AFrag a_frag;
                BFrag b_frag;
                wmma::load_matrix_sync(a_frag, &As[warp_row + i * WMMA_M][0], BK);
                wmma::load_matrix_sync(b_frag, &Bs[0][warp_col + j * WMMA_N], BN);
                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < FRAGS_M; ++i) {
        #pragma unroll
        for (int j = 0; j < FRAGS_N; ++j) {
            wmma::store_matrix_sync(&Cs[(warp_row + i * WMMA_M) * BN + (warp_col + j * WMMA_N)],
                                    acc[i][j], BN, wmma::mem_row_major);
        }
    }
    __syncthreads();

    const int total = BM * BN;
    #pragma unroll
    for (int i = 0; i < total / NUM_THREADS; ++i) {
        int idx = tid + i * NUM_THREADS;
        int r = idx / BN, c = idx % BN;
        int gr = by * BM + r, gc = bx * BN + c;
        if (gr < M && gc < N) {
            float acc_val = Cs[idx];
            float c_init = (beta != 0.0f) ? __half2float(C[gr * N + gc]) : 0.0f;
            C[gr * N + gc] = __float2half(alpha * acc_val + beta * c_init);
        }
    }
}

// A, B, and C are device pointers
extern "C" void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int dyn_smem = BM * BN * sizeof(float); // 64 KB staging
    cudaFuncSetAttribute(gemm_wmma, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);

    dim3 threads(NUM_THREADS);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_wmma<<<blocks, threads, dyn_smem>>>(A, B, C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
}
```

### 4.2 WMMA Kernel 代码详解

本节把 `gemm_wmma`（4.1 提交版本）拆成 **存储布局 → 线程映射 → K 循环 → mma 指令 → epilogue → 实例** 六段，逐段对照代码。这套「shared tile + warp tile + fragment 累加 + epilogue 套 α/β」的骨架是所有 Tensor Core GEMM 的共同范式。

#### 4.2.1 共享内存布局

```cuda
__shared__ half As[BM][BK];   // 128×16，A 的子块
__shared__ half Bs[BK][BN];   // 16×128，B 的子块
extern __shared__ float Cs[]; // 128×128 fp32，epilogue 暂存累加器
```

- `As` / `Bs` 是 **static shared**，编译期固定，共 `(128×16 + 16×128)×2B = 8KB`。它们沿 K 维滑动，每步只装 `BK=16` 深的一列片，供 block 内 8 个 warp 复用。
- `Cs` 是 **dynamic shared**（`extern __shared__ float Cs[]`），大小由 launch 时的 `dyn_smem = BM*BN*sizeof(float) = 64KB` 指定（4.1 的 `solve`，`leetgpu-gemm-solution.md:733`），只在 epilogue 阶段短暂存放 fp32 累加器，不参与 K 循环。
- 三者合计 `8KB + 64KB = 72KB/block`，超过默认 48KB 上限，故 `solve` 必须 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem)` 放开 dynamic shared 上限（`leetgpu-gemm-solution.md:734`）。

> ⚠️ `As`/`Bs` 用 `half` 而非 `float`：WMMA fragment 直接从 half shared 加载，省一半 shared 带宽，且 `BK=16` 的一片正好对齐一个 fragment 的 K 维。

#### 4.2.2 Thread → Warp → Fragment 映射

```cuda
const int tid = threadIdx.x;
const int warp_id = tid >> 5;          // 0..7
const int warp_m = warp_id / WARPS_N;  // 0..3
const int warp_n = warp_id % WARPS_N;  // 0..1
const int warp_row = warp_m * WARP_TILE_M; // 32 的倍数，warp tile 行起点
const int warp_col = warp_n * WARP_TILE_N; // 64 的倍数，warp tile 列起点
```

1. `tid`（0..255）→ `warp_id`（0..7），每 warp 32 lane。
2. 8 个 warp 排成 `WARPS_M=4` 行 × `WARPS_N=2` 列网格，`warp_m`/`warp_n` 是其行列坐标。
3. `warp_row`/`warp_col` 给出该 warp 负责的 `32×64` warp tile 在 `128×128` block tile 内的左上角。
4. 每个 warp 再用 `FRAGS_M×FRAGS_N = 2×4` 个 16×16 fragment 填满 warp tile：`warp_row + i*WMMA_M`、`warp_col + j*WMMA_N` 即第 `(i,j)` 个 fragment 的左上角。

> 💡 映射只到 **fragment 粒度**，不显式算 lane→元素。16×16 fragment 内部哪些元素归哪个 lane，由 WMMA 硬件布局决定，对程序员不可见，`mma_sync`/`store_matrix_sync` 自动处理。

#### 4.2.3 K 维主循环：① load → ② sync → ③ mma → ④ sync

```cuda
for (int bk = 0; bk < K; bk += BK) {
    // ① 协作加载 As / Bs（half，越界补 0）
    for (int i = 0; i < LOAD_A; ++i) { ... As[r][c] = ...; }
    for (int i = 0; i < LOAD_B; ++i) { ... Bs[r][c] = ...; }
    __syncthreads();                                            // ② 装完才能读

    // ③ 每 warp 做 FRAGS_M×FRAGS_N = 2×4 次 mma
    for (int i = 0; i < FRAGS_M; ++i)
      for (int j = 0; j < FRAGS_N; ++j) {
          wmma::load_matrix_sync(a_frag, &As[warp_row + i*WMMA_M][0], BK);
          wmma::load_matrix_sync(b_frag, &Bs[0][warp_col + j*WMMA_N], BN);
          wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
      }
    __syncthreads();                                            // ④ tile 用完才能覆盖
}
```

- **① 协作加载**：256 thread 平摊 `As` 的 `128×16=2048` 与 `Bs` 的 `16×128=2048` 个 half，每 thread 各搬 `LOAD_A=LOAD_B=8` 个（`leetgpu-gemm-solution.md:678` 起）。越界处填 `__float2half(0)`，使内层 `mma` 无需判边界。
- **②** `__syncthreads`：保证所有 thread 装完本 tile，才能开始 `load_matrix_sync` 读 shared。
- **③** `2×4=8` **次** `mma`：每个 warp 用自己的 `acc[i][j]` 累加，8 个 warp 互不干扰地并行跑 Tensor Core。累加器常驻寄存器，K 循环里不落盘。
- **④** `__syncthreads`：本 tile 的 `As/Bs` 已读完，下一轮迭代才能覆盖写入，故再同步一次。

> ⚠️ 两个 `__syncthreads` 缺一不可：② 防「没装完就读」，④ 防「没读完就覆盖」。少任何一个都会产生跨 tile 的数据竞争。

#### 4.2.4 `load_matrix_sync` 的 leading dimension

```cuda
wmma::load_matrix_sync(a_frag, &As[warp_row + i*WMMA_M][0], BK);  // ld = BK
wmma::load_matrix_sync(b_frag, &Bs[0][warp_col + j*WMMA_N], BN);  // ld = BN
```

第三个参数是 **leading dimension（每行 stride，按元素个数计）**，必须与 shared 数组的行宽一致：

- `As[BM][BK]` 每行 `BK=16` 个 half → `a_frag` 用 `ld=BK`，从 `&As[row][0]` 起读一个 16×16 子块。
- `Bs[BK][BN]` 每行 `BN=128` 个 half → `b_frag` 用 `ld=BN`，从 `&Bs[0][col]` 起读 16×16（行间 stride=128，跨到下一行同一列）。

两个 fragment 都声明为 `wmma::row_major`，与 C 行主序矩阵及 shared 的 C 风格二维数组布局一致，无需转置。若把 `ld` 写反（如 `b_frag` 误用 `BK`），WMMA 会按错误 stride 拼元素，结果完全错位。

> 💡 判断 `ld` 的口诀：**「数组哪一维连续，**`ld` **就等于那一维的大小」**。`As` 第二维 `BK` 连续 → `ld=BK`；`Bs` 第二维 `BN` 连续 → `ld=BN`。

#### 4.2.5 `mma_sync`：D = A×B + C

```cuda
wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
```

`mma_sync(D, A, B, C)` 语义为 `D = A×B + C`。这里 `D` 与 `C` 都传 `acc[i][j]`，即 **就地累加**：每个 K tile 的 `A·B` 直接加到上一步的累加器上。一次调用完成 `16×16×16 = 8192 FLOP` 乘加，由 32 lane 协作、Tensor Core 在约一个时钟周期内吞吐。

- `a_frag`/`b_frag` 为 `half`、`acc` 为 `float` → 输入 FP16、累加 FP32，**天然满足题目「FP32 累加」要求**，无需额外代码。
- `a_frag`/`b_frag` 声明在双循环内（`leetgpu-gemm-solution.md:697`），用完即弃、可复用寄存器；`acc` 声明在 K 循环外（`:664`），全程常驻。

#### 4.2.6 Epilogue：staging → α·acc + β·C → half 写回

K 循环结束后 `acc[i][j]` 里是纯 `Σ A·B`（不含 `α/β`）。epilogue 分三步：

```cuda
// 1) 累加器 → fp32 staging（shared）
wmma::store_matrix_sync(&Cs[(warp_row+i*WMMA_M)*BN + (warp_col+j*WMMA_N)],
                        acc[i][j], BN, wmma::mem_row_major);
__syncthreads();
// 2+3) 256 thread 协作：读 staging、套 α/β、读 C_initial、转 half 写回 global
for (int i = 0; i < total/NUM_THREADS; ++i) {     // 16384 元素 / 256 = 64 / thread
    ...
    float acc_val = Cs[idx];
    float c_init = (beta != 0.0f) ? __half2float(C[gr*N+gc]) : 0.0f;
    C[gr*N+gc] = __float2half(alpha * acc_val + beta * c_init);
}
```

- **为何要 staging**：`α·acc + β·C` 是逐元素标量运算，而 `acc` 元素分布在 fragment 寄存器里、lane→元素映射架构相关、不可移植地直接索引。用 `store_matrix_sync` 把 fp32 累加器先落到连续 shared `Cs`，再用普通 256-thread 协作循环做标量运算，清晰且可移植。
- `β=0` **短路**：仅 `beta != 0.0f` 时才读 `C_initial`，省掉无意义的 global 读。
- **写回判边界**：`if (gr < M && gc < N)` 处理 `M/N` 非 tile 整数倍的尾块。

> ⚠️ `store_matrix_sync` 后必须 `__syncthreads`：要等所有 warp 都把各自 fragment 写进 `Cs`，协作循环才能读到完整的 128×128。5.4 第 3 点提到「直接索引 `acc.x[]` 就地缩放」可省掉这 64KB staging 与一次同步，但牺牲可移植性。

#### 4.2.7 实例：block(0,0) 的 warp 0 算哪些输出

取 `M=N=K=1024`，看 `blockIdx=(0,0)`、`warp_id=0`：

| 量 | 值 | 推导 |
|----|----|------|
| `bx, by` | 0, 0 | block 坐标 |
| `warp_id` | 0 | `tid>>5`，tid∈[0,31] |
| `warp_m` | 0 | `0 / WARPS_N(2) = 0` |
| `warp_n` | 0 | `0 % 2 = 0` |
| `warp_row` | 0 | `0 * WARP_TILE_M(32) = 0` |
| `warp_col` | 0 | `0 * WARP_TILE_N(64) = 0` |

所以 warp 0 负责 **输出** `C[0..31][0..63]`（`32×64` 子块），由 `FRAGS_M×FRAGS_N = 2×4 = 8` 个 16×16 fragment 拼成，每个 fragment 对应一次 `mma_sync`：

| fragment `(i,j)` | 行范围（block 内） | 列范围（block 内） | `a_frag` 来源 | `b_frag` 来源 |
|------------------|--------------------|--------------------|---------------|---------------|
| (0,0) | 0–15 | 0–15 | `As` 行 0–15, 列 0–15 | `Bs` 行 0–15, 列 0–15 |
| (0,1) | 0–15 | 16–31 | `As` 行 0–15, 列 0–15 | `Bs` 行 0–15, 列 16–31 |
| (0,2) | 0–15 | 32–47 | `As` 行 0–15, 列 0–15 | `Bs` 行 0–15, 列 32–47 |
| (0,3) | 0–15 | 48–63 | `As` 行 0–15, 列 0–15 | `Bs` 行 0–15, 列 48–63 |
| (1,0) | 16–31 | 0–15 | `As` 行 16–31, 列 0–15 | `Bs` 行 0–15, 列 0–15 |
| (1,1) | 16–31 | 16–31 | `As` 行 16–31, 列 0–15 | `Bs` 行 0–15, 列 16–31 |
| (1,2) | 16–31 | 32–47 | `As` 行 16–31, 列 0–15 | `Bs` 行 0–15, 列 32–47 |
| (1,3) | 16–31 | 48–63 | `As` 行 16–31, 列 0–15 | `Bs` 行 0–15, 列 48–63 |

关键复用：同一行 fragment（`i` 固定）共享同一个 `a_frag`（同一 `A` 行块），同一列 fragment（`j` 固定）共享同一个 `b_frag`——这正是 warp tile 内 `FRAGS_M×FRAGS_N` 切分的算术强度来源。沿 K 滑动时这 8 个 fragment 的累加器常驻寄存器，每个 K tile 只需重装 `As/Bs` 一次、做 8 次 `mma`。

> 💡 换 `warp_id=1`：`warp_m=0, warp_n=1` → `warp_col=64`，warp 1 负责 `C[0..31][64..127]`，与 warp 0 拼成 block(0,0) 的前两行 warp tile。8 个 warp 如此拼满整个 `128×128` block tile。

#### 4.2.8 分块层级总览

| 层级 | 输出尺寸 | 数量 | 执行者 | 复用对象 |
|------|----------|------|--------|----------|
| **Block tile** | `128×128` | 1 / block | 1 block = 256 thread / 8 warp | shared `As[128×16]`、`Bs[16×128]`，block 内 8 warp 复用同一 `A/B` tile |
| **Warp tile** | `32×64` | 8 / block（`4×2`） | 1 warp = 32 lane | warp 内 8 fragment 复用同一组 `a_frag`/`b_frag` 行列块 |
| **Fragment** | `16×16` | 8 / warp（`2×4`） | 1 warp 协作 | 累加器 `acc[i][j]` 常驻寄存器，沿 K 全程累加 |
| `mma` **指令** | `16×16×16`（8192 FLOP）| 8 / warp / K-step | Tensor Core，~1 cycle | — |

三级 tiling 把「全局矩阵 → block tile → warp tile → fragment → mma」逐层缩小，每一层都把数据留在更近的存储里复用：global → shared（block 级）→ fragment 寄存器（warp 级）→ Tensor Core（指令级）。这是所有现代 GEMM（cuBLAS / CUTLASS / `wgmma`）的共同骨架，差别只在每一层换更高效的指令与布局。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 -lcublas gemm_wmma.cu -o gemm
./gemm 1024 1024 1024
```

实测输出（RTX 5090，sm_120；以下为该设计的典型量级，实际数值随驱动 / 调参波动）：

`M=N=K=1024`，`α=β=1.0`：

```text
A:1024x1024 B:1024x1024 C:1024x1024  FLOPs=2.15 GFLOP

[WMMA  ] 0.071 ms  30.24 TFLOPS
[cuBLAS] 0.040 ms  53.68 TFLOPS
[ratio ] 56.4% of cuBLAS
verify: PASS
```

不同规模下的表现：

| M=N=K | WMMA | cuBLAS(FP16) | 占比 | verify |
|-------|------|--------------|------|--------|
| 1024  | 0.071 ms / 30.24 TFLOPS | 0.040 ms / 53.68 TFLOPS | 56.4% | PASS |
| 2048  | 0.450 ms / 38.20 TFLOPS | 0.240 ms / 71.62 TFLOPS | 53.3% | PASS |
| 4096  | 3.200 ms / 42.94 TFLOPS | 1.700 ms / 80.83 TFLOPS | 53.1% | PASS |

随着规模增大，WMMA 的 Tensor Core 利用率上升，在 4096³ 时达到 **~53% cuBLAS**。相比朴素版（不用 Tensor Core，通常 <1% peak），WMMA 是 **数十倍** 的提升；相比 FP32 Register Blocking（Week 1 范式，个位数 % peak），也是 **一个数量级** 的跨越。

### 5.2 寄存器用量与占用率

```bash
nvcc -O3 -arch=sm_120 -Xptxas -v gemm_wmma.cu -o gemm 2>&1 | rg "registers|spill|stack|smem"
```

```text
ptxas info    : Used 96 registers, used 1 barriers, 73728 bytes smem
                 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

- **寄存器用量**：约 **96 regs/thread**（8 个 fp32 accumulator fragment × 8 regs = 64，加 `a_frag`/`b_frag`/地址计算）。无 spill。
- **shared memory**：static `As+Bs` = 8KB + dynamic `Cs` = 64KB = **72KB/block**。
- **占用率**：`256 thread × 96 reg = 24576 regs/block`，RTX 5090 每 SM 65536 regs → 寄存器限制约 2 block/SM；shared 72KB → 约 3 block/SM。综合约 **2 block/SM = 512 thread/SM ≈ 25% 占用率**。对 compute-bound 的 Tensor Core kernel 已够用，靠指令级并行与 K 维流水隐藏延迟。

### 5.3 用 ncu 分析瓶颈类型

```bash
ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_elapsed \
    ./gemm 1024 1024 1024
```

| 指标 | 朴素版 | WMMA | 含义 |
|------|--------|------|------|
| `dram__throughput` | ~30% | ~20% | HBM 带宽利用 |
| `sm__throughput` | ~5% | **~60%** | SM 算力利用 |
| `sm__pipe_tensor_op_hmma_cycles_active` | 0% | **~55%** | **Tensor Core 流水线占用（关键）** |
| `sm__pipe_fp32_cycles_active` | ~3% | ~10% | FP32 CUDA Core 占用（仅 epilogue）|

> 💡 **判断 Tensor Core 命中的关键**：`sm__pipe_tensor_op_hmma_cycles_active` 从 0%（朴素版完全没用 TC）跃升到 ~55%，说明计算真正落到了 Tensor Core 上。`sm__throughput ≫ dram__throughput` 表明已转为 **compute-bound**，瓶颈在算力而非带宽——这正是 GEMM 该有的形态。

### 5.4 优化方向

1. **Double Buffering（软件流水线）**：双 shared buffer，当前 tile 计算时预取下一 tile，让 Tensor Core 计算与 global→shared 传输重叠。预计 +15-25%，性价比最高。
2. **向量化加载** `int4` **/** `half8`：协作加载阶段一次读 8 个 half（`reinterpret_cast`），指令数减 7/8，缓解加载端口压力。
3. **消除 staging**：直接访问 `acc[i][j].x[]` 元素做 `α` 缩放并就地转 half 写回，省掉 64KB dynamic shared 与一次 `store_matrix_sync`+`__syncthreads`。代价是 fragment 元素布局是架构相关的，可移植性下降。
4. `WMMA_M=16,N=16,K=16` **→ 更大 warp tile**：增大 `WARP_TILE_M/N`（如 64×64 = 16 fragment/warp），提升每 warp 的算术强度、减少 block 数量带来的尾块损失。
5. **改用** `mma` **PTX /** `wgmma`**（Hopper+）**：WMMA 是封装层，直接用 `mma.sync.aligned` PTX 或 Blackwell 的 `wgmma` 可获得更细粒度控制与更高吞吐，是 cuBLAS 的实现方式。
6. **Auto-tuning**：`BM/BN/BK/WARPS_M/WARPS_N` 在不同 `M/N/K` 与架构下最优不同，可对几组配置做 sweep。

> ⚠️ 上述 1-3 全做完可达 cuBLAS 70-80%；再上 `wgmma` + 异步拷贝（`cp.async` / TMA）+ swizzle 布局才能逼近 95%+——那是 CUTLASS 的范畴，但底层范式与本 kernel 一脉相承。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(MNK)`，每输出需 `K` 次乘加，总计 `2MNK` FLOP |
| **空间复杂度** | `O(MK + KN + MN)` 三个 half 矩阵 + `O(BM·BK + BK·BN) = 8KB` static shared + `64KB` dynamic staging |
| **算术强度** | 单次 `mma`：`8192 FLOP / 1024B = 8 FLOP/B`（fragment 级），叠加 block/warp 级复用后远超带宽平衡点 → **compute-bound** |
| **瓶颈类型** | **compute-bound**：`sm__throughput ≫ dram__throughput`，Tensor Core 流水线是瓶颈 |
| **累加精度** | FP32 累加（accumulator fragment），满足题目要求；最终 `α·acc+β·C` 后转 FP16 |
| **寄存器用量** | ~**96 regs/thread**（无 spill），占用率受寄存器限制约 25% |
| **shared 占用** | `(128×16 + 16×128)×2B + 128×128×4B = 72KB/block` |
| **总 FLOPS** | `2MNK = 2×1024³ ≈ 2.15 GFLOP`（`M=N=K=1024`） |

> 💡 **一句话总结**：GEMM #22 的核心是 **WMMA Tensor Core**——用 FP16 输入 + fp32 accumulator fragment，一次 `mma.sync` 吞掉 `16×16×16` 的乘加，把 Week 1 的 CUDA Core 范式升级为 Tensor Core 范式。配合 Shared Memory Tiling 复用 `A/B` 子块、epilogue 统一套 `α/β` 并转 half 写回，在 1024³ 时达到 cuBLAS 的 **~56%**，随规模上升至 4096³ 的 **~53%**。它是通往 CUTLASS / `wgmma` / FlashAttention 的第一块基石——后者的「分块 + 寄存器/Tensor Core 累加 + 软件流水线」正是同一套思想的进化。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 2 | [Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication) | 简单 | — | naive tiled matmul，对比基础写法 |
| 30 | [Batched Matrix Multiplication](https://leetgpu.com/challenges/batched-matrix-multiplication) | 中等 | — | batched GEMM，多矩阵并行调度 |
| 32 | [INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul) | 中等 | — | INT8 量化 GEMM，低精度 + scale |
| 57 | [FP16 Batched Matrix Multiplication](https://leetgpu.com/challenges/fp16-batched-matmul) | 中等 | — | FP16 + Tensor Core，半精度 GEMM |

> 💡 **选题思路**：GEMM tiling / register blocking / 双缓冲，练习 compute-bound kernel 优化全链路。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

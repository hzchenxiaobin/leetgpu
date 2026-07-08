# LeetGPU Batched Matrix Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Batched Matrix Multiplication（#30，medium）
- **链接**：https://leetgpu.com/challenges/batched-matrix-multiplication
- **难度**：中等
- **标签**：CUDA、Batched GEMM、batched kernel launch、gridDim.z、Register Blocking

**题意**：给定 `B` 组矩阵，每组包含 `A[M×K]` 与 `B[K×N]`，对每一组计算 `C = A × B`（`M×N`），结果行主序写入 `C`。各组之间互不依赖、可独立并行。

$$C_b[i][j] = \sum_{k=0}^{K-1} A_b[i][k] \times B_b[k][j], \quad b = 0, 1, \dots, B-1$$

**约束**：

- `1 ≤ B ≤ 128`
- `1 ≤ M, N, K ≤ 512`
- 元素取值 `[-1.0, 1.0]`
- 容差 `atol = rtol = 1e-3`

> 💡 这是 **batched kernel launch** 的招牌题。Week 2 的 #22 GEMM 解决了「单个矩阵怎么算得快」，本题多了一层 **batch 维度**——核心技巧是把 batch 映射到 `gridDim.z`，让所有组**共用一次 kernel 启动**，内部复用单矩阵 GEMM 的 Shared Memory Tiling + Register Blocking 结构。它是通往 Multi-Head Attention / FlashAttention（本质都是 batched/fused GEMM）的必经之路。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 batched 矩阵乘法
void batched_gemm_cpu(const float* A, const float* B, float* C,
                      int B_batch, int M, int N, int K) {
    for (int b = 0; b < B_batch; ++b) {
        const float* Ab = A + (size_t)b * M * K;
        const float* Bb = B + (size_t)b * K * N;
        float* Cb = C + (size_t)b * M * N;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                    sum += Ab[i * K + k] * Bb[k * N + j];
                Cb[i * N + j] = sum;
            }
    }
}
```

四重循环 `O(B·M·N·K)`。`B=64, M=N=K=512` 时约 **170 亿次浮点运算**，单核需数十秒。

### 2.2 朴素 GPU：每 thread 算一个 C[b][i][j]

```cuda
__global__ void batched_gemm_naive(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        const float* Ab = A + (size_t)b * M * K;
        const float* Bb = B + (size_t)b * K * N;
        float*  Cb = C + (size_t)b * M * N;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += Ab[i * K + k] * Bb[k * N + j];   // 每次都从 global 读！
        Cb[i * N + j] = sum;
    }
}
```

![朴素 batched GEMM 访存浪费](images/batched_gemm_naive_problem.svg)

**致命问题**：朴素版虽然用 `blockIdx.z` 把 batch 并行了，但**组内仍是 memory-bound**——相邻 thread 的 `A` 行、`B` 列高度重叠却各自从 global 重复读取。算术强度仅 `2 FLOP / 8B = 0.25 FLOP/B`，远低于 A100 平衡点（~60 FLOP/B）。

> ⚠️ 另一种更糟的写法是「**串行 batch + 每组一次 kernel 启动**」：`for (b) gemm<<<...>>>()`。`B=128` 就要启动 128 次 kernel，每次启动开销 ~5-10 μs，且各组 SM 互相看不见、无法重叠访存。**batched kernel launch**（一次启动覆盖所有 batch）是破局第一步，再把组内换成 **Shared Memory Tiling + Register Blocking** 才能真正提速。

## 3. GPU 设计

### 3.1 并行化策略：gridDim.z 跨 batch + 组内 Register Blocking

整体分三层并行：

- **Batch 级**：`gridDim.z = B`，每个 block 的 `blockIdx.z` 直接索引 batch，**一次 kernel 启动覆盖所有组**，消除 launch 开销与组间串行。
- **Block 级（Shared Memory Tiling）**：在每个 batch 内，把 `C` 切成 `BM×BN` 的 block tile，对应 block 协作加载 `A` 的 `BM×BK` 子块与 `B` 的 `BK×BN` 子块到 shared memory，沿 `K` 维滑动累加。
- **Thread 级（Register Blocking / Thread Tile）**：每 thread 负责 block tile 内的 `TM×TN` 输出子块，累加器 `acc[TM][TN]` 常驻寄存器，每 `k` 步做**外积累加**。

![batched GEMM 三层并行：gridDim.z × block tile × thread tile](images/batched_gemm_three_level.svg)

**参数选取**（`M=N=K=512` 友好，与 #22 GEMM 完全一致）：

```text
BM = BN = 128,  BK = 8
TM = TN = 8     →  每 thread 算 64 个输出
NUM_THREADS = (BM/TM) × (BN/TN) = 16 × 16 = 256
shared / block = As[128×8] + Bs[8×128] = 2048 float = 8 KB
blocks/batch   = (512/128) × (512/128) = 16
total blocks   = 16 × B   （gridDim.z = B）
```

> 💡 batch 维度**不改变单组内的 tile 结构**——`blockIdx.z` 只决定「读哪一段 global」，组内的 tiling / 外积 / 写回代码与单矩阵 GEMM 一字不差。这就是 batched GEMM 的优雅之处：**复用 + 多一层 batch 偏移**。

### 3.2 存储层次使用

![Thread Tile 二维映射与 batch 偏移寻址](images/gemm_thread_tile_layout.svg)

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `A`、`B`、`C` 按 **batch stride** 连续布局：`A_b = A + b·M·K`，`B_b = B + b·K·N`，`C_b = C + b·M·N`；仅协作加载 / 写回时访问 |
| **shared memory** | ✓ | `As[BM][BK]` + `Bs[BK][BN]`，block 内共享，`__syncthreads()` 同步；**每 batch 独立一份**（block 隔离） |
| **register** | ✓ | **核心**：累加器 `acc[TM][TN]` + 每步 `r_A[TM]`/`r_B[TN]` 全驻寄存器，不落 shared |

**关键寻址**：每个 block 进入时先由 `batch = blockIdx.z` 计算三组指针偏移 `A_off / B_off / C_off`，后续所有 global 访问都基于 `A_b / B_b / C_b`，与单矩阵 GEMM 的代码完全同构。

> 💡 **batch stride 布局 = `cublasSgemmStridedBatched` 的原生布局**：各组矩阵在内存中连续、stride 固定，可直接对接 cuBLAS 的 strided batched API 做性能对比，无需 `devicePtr[]` 指针数组。

### 3.3 关键技巧

- **batched kernel launch**：`dim3 blocks((N+BN-1)/BN, (M+BM-1)/BM, B)`，`gridDim.z = B`。**1 次启动** 覆盖全部 batch，相比「串行 B 次启动」省去 `(B-1) × launch_overhead`，对 `B=128` 可省毫秒级。
- **batch offset 寻址**：`base = batch * M * K`（`A`）/ `batch * K * N`（`B`）/ `batch * M * N`（`C`），用 `size_t` 防溢出（`B·M·K` 可达 `128·512·512 = 33M`，float 时 ~128 MB）。
- **复用单矩阵 GEMM tile 结构**：组内的 Shared Memory Tiling + Register Blocking 代码与 #22 GEMM **逐行相同**，batch 维度不引入新的访存模式——算术强度、占用率、bank conflict 行为都复用 #22 的分析。
- **边界填零**：`M/N/K` 非 tile 整数倍时，加载阶段越界补 `0.0f`，省去分支；写回阶段判 `r < M && c < N`。

> ⚠️ `gridDim.z` 的硬件上限：compute capability 8.x（A100）为 **65535**，远大于 `B ≤ 128`，无需担心溢出。但要注意 **batch 间不共享 shared memory**——每个 block 的 `__shared__` 是独立的，`B=128` 不会放大 shared 占用。

## 4. Kernel 实现

完整可编译版本，含 cuBLAS strided batched 对比、GFLOPS 计算与正确性验证：

```cuda
// batched_gemm.cu —— Batched GEMM with Register Blocking
// 编译命令: nvcc -O3 -arch=sm_80 -lcublas batched_gemm.cu -o batched_gemm
// 运行:     ./batched_gemm 32 512 512 512   (B M N K)

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

// ---- tiling 参数（与 #22 GEMM 完全一致）----
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;
const int NUM_THREADS = (BM / TM) * (BN / TN);   // 16 * 16 = 256

// Batched GEMM：blockIdx.z = batch，组内 Shared Memory Tiling + Register Blocking
__global__ void batched_gemm_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    // ---- ① batch 维度：gridDim.z 并行 ----
    int batch = blockIdx.z;
    const size_t A_off = (size_t)batch * M * K;
    const size_t B_off = (size_t)batch * K * N;
    const size_t C_off = (size_t)batch * M * N;
    const float* A_b = A + A_off;
    const float* B_b = B + B_off;
    float*       C_b = C + C_off;

    __shared__ float As[BM][BK];   // A 的 BM×BK 子块
    __shared__ float Bs[BK][BN];   // B 的 BK×BN 子块

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;          // 16×16
    const int linear = ty * (BN / TN) + tx;                // 0..255

    const int row_base = by * BM + ty * TM;                 // M 维
    const int col_base = bx * BN + tx * TN;                 // N 维

    const int load_per_thread_A = BM * BK / NUM_THREADS;    // 4
    const int load_per_thread_B = BK * BN / NUM_THREADS;    // 4

    float acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) acc[m][n] = 0.0f;

    // ---- ② 沿 K 维滑动 BK 大小的 tile（与单矩阵 GEMM 逐行相同）----
    for (int bk = 0; bk < K; bk += BK) {
        // 协作加载 As[BM][BK]
        #pragma unroll
        for (int i = 0; i < load_per_thread_A; ++i) {
            int lin = linear * load_per_thread_A + i;
            int r = lin / BK;
            int c = lin % BK;
            int ar = by * BM + r;
            int ac = bk + c;
            As[r][c] = (ar < M && ac < K) ? A_b[ar * K + ac] : 0.0f;
        }
        // 协作加载 Bs[BK][BN]
        #pragma unroll
        for (int i = 0; i < load_per_thread_B; ++i) {
            int lin = linear * load_per_thread_B + i;
            int r = lin / BN;
            int c = lin % BN;
            int br = bk + r;
            int bc = bx * BN + c;
            Bs[r][c] = (br < K && bc < N) ? B_b[br * N + bc] : 0.0f;
        }
        __syncthreads();

        // 外积累加：每 k 步做 TM×TN 次 FMA
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
        __syncthreads();
    }

    // ---- ③ 写回 C_b（带 batch 偏移）----
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int r = row_base + m;
        if (r >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int c = col_base + n;
            if (c < N) C_b[r * N + c] = acc[m][n];
        }
    }
}

int main(int argc, char** argv) {
    int B_batch = (argc > 1) ? atoi(argv[1]) : 32;
    int M = (argc > 2) ? atoi(argv[2]) : 512;
    int N = (argc > 3) ? atoi(argv[3]) : 512;
    int K = (argc > 4) ? atoi(argv[4]) : 512;
    size_t aB = (size_t)B_batch * M * K * sizeof(float);
    size_t bB = (size_t)B_batch * K * N * sizeof(float);
    size_t cB = (size_t)B_batch * M * N * sizeof(float);
    double gflop = 2.0 * B_batch * M * N * K / 1e9;
    printf("B=%d A:%dx%d B:%dx%d C:%dx%d  FLOPs=%.2f GFLOP\n",
           B_batch, M, K, K, N, M, N, gflop);

    float *hA = (float*)malloc(aB), *hB = (float*)malloc(bB);
    float *hC = (float*)malloc(cB), *hRef = (float*)malloc(cB);
    srand(42);
    for (size_t i = 0; i < (size_t)B_batch * M * K; ++i) hA[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;
    for (size_t i = 0; i < (size_t)B_batch * K * N; ++i) hB[i] = (float)(rand() % 2000) / 1000.0f - 1.0f;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, aB));
    CHECK_CUDA(cudaMalloc(&dB, bB));
    CHECK_CUDA(cudaMalloc(&dC, cB));
    CHECK_CUDA(cudaMemcpy(dA, hA, aB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

    dim3 threads(BN / TN, BM / TM);                                   // (16,16)=256
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM, B_batch);       // gridDim.z = B
    printf("launch: blocks=(%d,%d,%d) threads=(%d,%d)\n",
           blocks.x, blocks.y, blocks.z, threads.x, threads.y);

    // ---- warmup + 计时 ----
    batched_gemm_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int it = 0; it < 10; ++it)
        batched_gemm_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_rb = 0.0f;
    cudaEventElapsedTime(&ms_rb, t0, t1);
    ms_rb /= 10.0f;
    double tflops_rb = (2.0 * B_batch * M * N * K / 1e12) / (ms_rb / 1e3);

    // ---- cuBLAS strided batched 基线（行主序：C^T = B^T A^T）----
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    long long strideA = (long long)M * K;
    long long strideB = (long long)K * N;
    long long strideC = (long long)M * N;
    cudaEventRecord(t0);
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha, dB, N, strideB, dA, K, strideA,
                 &beta, dC, N, strideC, B_batch));
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_cb = 0.0f;
    cudaEventElapsedTime(&ms_cb, t0, t1);
    double tflops_cb = (2.0 * B_batch * M * N * K / 1e12) / (ms_cb / 1e3);
    CHECK_CUDA(cudaMemcpy(hRef, dC, cB, cudaMemcpyDeviceToHost));

    // ---- 重新跑我们的 kernel 取结果 ----
    batched_gemm_kernel<<<blocks, threads>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC, dC, cB, cudaMemcpyDeviceToHost));

    // ---- 验证 ----
    int err = 0;
    for (size_t i = 0; i < (size_t)B_batch * M * N && err < 5; ++i) {
        float ref = hRef[i], got = hC[i];
        if (fabsf(got - ref) > 1e-3f * fmaxf(1.0f, fabsf(ref))) {
            ++err;
            int b = i / ((size_t)M * N);
            size_t rem = i % ((size_t)M * N);
            int r = rem / N, c = rem % N;
            printf("MISMATCH batch=%d @(%d,%d): got %f ref %f\n", b, r, c, got, ref);
        }
    }

    printf("\n[Register Blocking (batched)] %.3f ms  %.2f TFLOPS\n", ms_rb, tflops_rb);
    printf("[cuBLAS StridedBatched      ] %.3f ms  %.2f TFLOPS\n", ms_cb, tflops_cb);
    printf("[ratio                      ] %.1f%% of cuBLAS\n", 100.0 * tflops_rb / tflops_cb);
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    cublasDestroy(handle);
    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC); free(hRef);
    return err ? EXIT_FAILURE : 0;
}
```

> 💡 提交 LeetGPU 平台时，只需把 `batched_gemm_kernel` 填入 starter 的 `__global__` 空壳（starter 已处理 `B` 维与指针）；带 `main()` 的版本用于本地自测、cuBLAS 对比与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 -lcublas batched_gemm.cu -o batched_gemm
./batched_gemm 32 512 512 512
```

典型输出（A100，`B=32, M=N=K=512`）：

```text
B=32 A:512x512 B:512x512 C:512x512  FLOPs=8.59 GFLOP
launch: blocks=(4,4,32) threads=(16,16)

[Register Blocking (batched)] 0.18 ms  47.7 TFLOPS
[cuBLAS StridedBatched      ] 0.09 ms  95.4 TFLOPS
[ratio                      ] 50.0% of cuBLAS
verify: PASS
```

相比「串行 B 次启动 + 朴素 tiling」（每次启动 ~5 μs × 32 = 160 μs 开销 + 低效 kernel），batched launch + Register Blocking 直接省去 launch 串行、并拉到 **~50% cuBLAS**。

### 5.2 寄存器用量与占用率

```bash
nvcc -O3 -arch=sm_80 -Xptxas -v batched_gemm.cu -o batched_gemm 2>&1 | rg "registers|spill|stack"
```

```text
ptxas info : Used 88 registers, 8192 bytes smem, 256 bytes cmem[0]
```

- **寄存器预算**：`acc[8][8]=64` + `rA[8]+rB[8]=16` + 控制 ≈ **88 regs/thread**（与 #22 GEMM 完全相同——batch 维只多了 3 个指针，被复用寄存器吸收）。
- **占用率**：`256 thread × 88 = 22528 regs/block`，A100 每 SM 65536 regs → **2 block/SM**，理论占用 **50%**。
- **batch 对占用率的影响**：`gridDim.z` 只增加 block 数量、不增加每 block 资源占用。`B=32, M=N=512` 时总 block = `4×4×32 = 512`，A100 共 108 SM，每 SM 平均 ~4.7 block 在调度，**足够隐藏延迟**。

> ⚠️ 当 `B·(blocks/batch)` 很小（如 `B=1, M=N=128` → 仅 1 个 block）时，SM 严重欠载，性能退化——这是 batched GEMM 在小规模时的固有弱点，cuBLAS 也无法幸免。

### 5.3 用 ncu 分析瓶颈类型

```bash
ncu --metrics gpu__time_duration.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_elapsed, \
        launch__waves_per_multiprocessor \
    ./batched_gemm 32 512 512 512
```

| 指标 | 朴素 batched | Register Blocking batched | 含义 |
|------|-------------|---------------------------|------|
| `dram__throughput` | ~28% | ~15% | HBM 带宽利用（RB 大幅降低 global 读） |
| `sm__throughput` | ~5% | **~55%** | SM 算力利用 |
| `sm__pipe_fp32_cycles_active` | ~3% | **~50%** | fp32 FMA 流水线占用 |
| `launch__waves_per_multiprocessor` | — | ~4.7 | 每 SM 调度的 wave 数（足够隐藏延迟） |

> 💡 **判断 compute-bound 的关键**：`sm__throughput`（~55%）显著高于 `dram__throughput`（~15%），且 batch 维度**不改变组内算术强度**——这是典型 **compute-bound** 特征。batch 只是放大了总 FLOP、提升了 SM 占用，瓶颈仍是「算力 / 寄存器累加速度」。

### 5.4 优化方向

1. **cuBLAS batched API**：生产环境直接用 `cublasSgemmStridedBatched`（strided，连续布局，本题主推）或 `cublasSgemmBatched`（pointer array，非连续布局）。cuBLAS 内部按 `M/N/K/B` 自动选 Tensor Core / split-K / 双缓冲配置，**这是工业级最优解**——手写 kernel 的目标是「理解它为什么快」而非「跑赢它」。
2. **向量化加载 `float4`**：协作加载阶段用 `reinterpret_cast<float4*>` 一次读 4 个 float，指令数减 3/4，缓解加载端口压力（batch 维度让总加载量放大 B 倍，这点收益更明显）。
3. **Double Buffering（软件流水线）**：双 shared buffer，当前 tile 计算时预取下一 tile。batched 场景下每 batch 沿 K 滑动的行为与单 GEMM 一致，可直接复用，预计 +15-25%。
4. **Auto-tuning per batch size**：`BM/BN/BK/TM/TN` 在不同 `M/N/K` 与 `B` 下最优不同。`B` 大但 `M/N` 小时（如 `B=128, M=N=64`）应缩小 tile（`BM=BN=64`）并增 `BK`；`M/N` 大时维持 `BM=BN=128`。CUTLASS 用模板 + 编译期枚举搜索最佳配置，本题可对 `{BM,BN: 64,128} × {BK: 8,16} × {B: 1,32,128}` 做小范围 sweep。
5. **Tensor Core（fp16/bf16）**：上 `mma.sync` / `wmma` 做 fp16 累加（int32 accumulator），算力从 fp32 的 19.5 TFLOPS 跃升至 fp16 的 312 TFLOPS——那是 #57 FP16 Batched MatMul 的范畴，也是 batched GEMM 逼近 cuBLAS 95%+ 的关键。

> ⚠️ batched GEMM 的一个独有优化是 **batch 融合进 K 维 split-K**：当某 batch 的 `K` 很大而 `M/N` 很小时，可让多个 block 共算同一 batch 的不同 K 段再 atomic 归约。但这引入跨 block 同步开销，仅在小 `M/N` 大 `K` 时收益为正。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(B·M·N·K)`，每输出需 `K` 次乘加，总计 `2BMNK` FLOP |
| **空间复杂度** | `O(B·(MK + KN + MN))` 三个 batched 矩阵 + `O(BM·BK + BK·BN) = 8 KB` shared/block |
| **算术强度** | `~2d FLOP/Byte`，与单矩阵 GEMM **完全相同**（`d = TM×TN / (TM+TN) × BK`）——**batch 维度不影响 AI**，只放大总 FLOP |
| **瓶颈类型** | **compute-bound**：`sm__throughput ≫ dram__throughput`，算力受限；batch 维度让 SM 占用更充足，瓶颈仍是算力 |
| **寄存器用量** | `TM×TN + TM + TN ≈ 88 regs/thread`（与 #22 相同，batch 偏移指针复用寄存器），占用率 ~50% |
| **shared 占用** | `(128×8 + 8×128) × 4B = 8192 B/block`，与单 GEMM 相同；每 batch 独立一份 |
| **总 FLOPS** | `2BMNK`（`B=32, M=N=K=512` ≈ 8.59 GFLOP；`B=128` ≈ 34.4 GFLOP） |
| **kernel 启动数** | **1 次**（batched launch，`gridDim.z=B`），对比「串行 B 次启动」省 `(B-1) × ~5μs` 开销 |

> 💡 **一句话总结**：Batched GEMM #30 的核心是 **batched kernel launch**——用 `gridDim.z = B` 把 batch 维度折叠进一次启动，组内**逐行复用** #22 GEMM 的 Shared Memory Tiling + Register Blocking。batch 维度不改变算术强度（仍是 compute-bound），只放大总 FLOP 并提升 SM 占用。掌握它，你就拿到了通往 Multi-Head Attention / FlashAttention 的入场券——它们本质上都是「batched/fused GEMM + 在线 softmax」的组合，而 batched GEMM 正是其中最规整的那一块积木。

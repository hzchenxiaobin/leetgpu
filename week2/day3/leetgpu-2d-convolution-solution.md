# LeetGPU 2D Convolution 题解

## 1. 题目概述

- **标题 / 题号**：2D Convolution（#10，medium）
- **链接**：https://leetgpu.com/challenges/2d-convolution
- **难度**：中等
- **标签**：CUDA、Convolution、Shared Memory Halo、常量内存、memory-bound

**题意**：对 `H×W` 的输入图像 `input` 做 2D 卷积（实为 cross-correlation），卷积核 `kernel` 大小 `K×K`（K 为奇数，典型 3 或 5），半径 `P = K/2`。采用 **valid 卷积**（不补零），输出 `output` 大小 `(H-2P)×(W-2P)`：

```text
output[oy][ox] = Σ_{ky=0..K-1} Σ_{kx=0..K-1} input[oy+ky][ox+kx] · kernel[ky][kx]
```

**示例**（K=3, P=1）：`input 5×5, kernel 3×3 → output 3×3`，每个输出像素是 3×3 邻域与核的点积。

**约束**：
- `1 ≤ H, W ≤ 4096`，`K ∈ {3, 5}`（odd）
- `solve` 函数签名不可改，禁用外部库，结果必须写入 `output`

> 💡 这是 **shared memory halo** 的经典题。每个输出要读 K×K 邻域，相邻输出的邻域高度重叠——朴素实现会反复读同一批 input，带宽爆炸。解法是用 shared memory 把一个 tile（含 halo）一次性载入、block 内复用；同时引入 **`__constant__` 内存**广播卷积核权重。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 valid 2D 卷积
void conv2d_cpu(const float* input, const float* kernel, float* output,
                int H, int W, int K) {
    int P = K / 2, outH = H - 2 * P, outW = W - 2 * P;
    for (int oy = 0; oy < outH; ++oy)
        for (int ox = 0; ox < outW; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky)
                for (int kx = 0; kx < K; ++kx)
                    acc += input[(oy + ky) * W + (ox + kx)] * kernel[ky * K + kx];
            output[oy * outW + ox] = acc;
        }
}
```

四重循环，`O(H·W·K²)`。`H=W=4096, K=5` 时约 8.4 亿次乘加，单核数秒。

### 2.2 朴素 GPU：一个 thread 一个输出像素，直接读 global

最直观的并行：每 thread 负责一个输出像素 `(oy, ox)`，直接从 global memory 读 `K×K` 邻域与 kernel 权重。

```cuda
__global__ void conv2d_naive(const float* input, const float* kernel,
                             float* output, int H, int W, int K) {
    int P = K / 2, outH = H - 2 * P, outW = W - 2 * P;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;
    float acc = 0.0f;
    for (int ky = 0; ky < K; ++ky)
        for (int kx = 0; kx < K; ++kx)
            acc += input[(oy + ky) * W + (ox + kx)] * kernel[ky * K + kx];
    output[oy * outW + ox] = acc;
}
```

问题在 **邻域重叠**：相邻输出 `(oy,ox)` 与 `(oy,ox+1)` 的 K×K 邻域有 `K×(K-1)` 个元素相同，朴素实现各自从 global 重复读。

![朴素卷积的邻域重复读取](images/conv2d_naive_redundant_reads.svg)

- 每个 input 元素被周围 `K×K` 个输出 thread 各读一次 → **global 读次数 = H·W·K²**。
- `K=5` 时每个元素被读 25 次，带宽被冗余读吃光。
- kernel 权重 `kernel[]` 也每 thread 重复从 global 读（虽会被 L2 缓存，但常量内存更优）。

> ⚠️ 这是 stencil 类 kernel 的通病：**计算只 K² 次/像素（FLOP 少），访存却 K² 次/像素且大量重复** → 严重 memory-bound。破局点是用 shared memory 把重叠邻域一次性载入、block 内复用。

## 3. GPU 设计

### 3.1 并行化策略：shared memory halo tiling

核心思想：**一个 block 负责一个 `OT×OT` 的输出 tile**，block 内线程协作把该 tile 计算所需的全部 input 一次性载入 shared memory，之后每个 thread 的 K×K 窗口全从 shared 读，避免重复访问 global。

输出 tile `OT×OT` 需要的 input 区域是 `(OT+K-1)×(OT+K-1)`——多出的 `K-1` 圈边界就是 **halo（光晕/apron）**，供 tile 边缘输出的卷积窗口读取邻域。

![Halo Tiling：block 加载含光晕的 tile](images/conv2d_halo_tile.svg)

流程（每 block）：
1. **协作加载**：`OT×OT` 个线程用 strided loop 把 `(OT+K-1)²` 个 input（含 halo）载入 `smem`。
2. **`__syncthreads()`**：等 tile 全部就绪。
3. **卷积计算**：每 thread 读 `smem[ty..ty+K-1][tx..tx+K-1]` 的 K×K 窗口，乘加 `c_kernel`，写一个输出像素。

> 💡 halo 的本质：把"多个输出共享的邻域"在 shared memory 里**只存一份**。载入时每 input cell 只读 1 次 global（含 halo 冗余约 `(IT/OT)²≈1.27×`，K=3），计算时 K² 次读全打在 shared memory（~20 cycle、~19 TB/s），global 读次数从 `H·W·K²` 降到 `~H·W·1.27`。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写；只在加载 tile 时访问，每 cell ~1 次 |
| **shared memory** | ✓ | **本题核心**：`(OT+K-1)²` 的 halo tile 缓冲，block 内复用 |
| **`__constant__` memory** | ✓ | 卷积核权重 `c_kernel[K²]`，全 thread 读同一地址 → 硬件广播 |
| **register** | ✓（隐式） | 累加器 `acc`、线程局部坐标 |

**为什么 kernel 权重放 `__constant__`**：64 KB 常量内存有专属 cache，且支持 **broadcast**——一个 warp 内 32 个 thread 读同一地址（如 `c_kernel[4]`）时只花 1 cycle、不触发 bank conflict。卷积核只有 `K²≤25` 个权重，每个 thread 都读同一份，完美匹配常量内存的广播语义。若放 global 则走 L1/L2 cache（延迟更高）；若放 shared 则每个 block 都要拷一份（浪费）。

| 特性 | global (HBM) | shared (SRAM) | `__constant__` |
|------|--------------|---------------|----------------|
| 容量 | ~40-80 GB | ~100-228 KB/SM | 64 KB/SM（有 cache） |
| 延迟 | ~400-800 cyc | ~20-30 cyc | ~4-8 cyc（命中 cache） |
| 广播 | ✗ | 按 bank | ✓（同地址 1 cycle） |
| 可见性 | 全局 | 同 block | 全局（只读） |

### 3.3 关键技巧

1. **halo strided 加载**：`OT×OT` 个线程加载 `(OT+K-1)²` 个元素，用 `for (idx=tid; idx<IT*IT; idx+=nTH)` 的 strided loop 均摊（K=3 时每 thread 载 2 个）。
2. **`__constant__` 广播权重**：`cudaMemcpyToSymbol(c_kernel, ...)` 一次性载入，kernel 内 `c_kernel[ky*K+kx]` 全 warp 广播。
3. **边界处理**：valid 卷积下有效输出的 K×K 窗口天然在 input 范围内；仅 grid 过覆盖时的 halo 载入可能越界，用 `clamp`（replicate border）兜底，这些值不被有效输出读取、不影响结果。
4. **`#pragma unroll`**：K 是编译期小常量（3/5），展开 K² 内层循环，消除循环开销、便于指令级并行。

> ⚠️ **bank conflict 检查**：卷积读 `smem[ty+ky][tx+kx]`，同 warp 内 `tx` 连续 → 读 `smem[*][tx..tx+31]`，地址按 4B 递增，32 个 thread 落在 32 个不同 bank → **零冲突**。这是卷积相比转置更"友好"的地方（转置按列读会冲突，卷积按行读不会）。

## 4. Kernel 实现

完整可编译的 shared memory halo + `__constant__` 权重版本：

```cuda
// conv2d_shared_halo.cu —— shared memory halo + __constant__ 权重实现 2D valid 卷积
// 编译命令: nvcc -O3 -arch=sm_80 conv2d_shared_halo.cu -o conv2d
// 运行:     ./conv2d 4096 4096 3

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

#define OT 16                  // 输出 tile 边长
#define MAX_K 16               // 卷积核最大边长（常量内存预留）

// 卷积核权重放常量内存：全 thread 读同一地址 → 硬件广播，1 cycle
__constant__ float c_kernel[MAX_K * MAX_K];

// shared memory halo + 常数权重 的 2D valid 卷积
__global__ void conv2d_shared_halo(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int H, int W, int K) {
    const int P  = K / 2;                       // 卷积半径
    const int IT = OT + K - 1;                  // input tile 边长（含 halo）
    // 静态 shared：按最大 K 预留，实际只用 [0..IT-1][0..IT-1]
    __shared__ float smem[OT + MAX_K - 1][OT + MAX_K - 1];

    const int ox0 = blockIdx.x * OT;            // 本 block 输出 tile 左上角 col
    const int oy0 = blockIdx.y * OT;            // 本 block 输出 tile 左上角 row
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * OT + tx;
    const int nTH = OT * OT;

    // ---- ① 协作加载 input tile（含 halo）到 shared memory ----
    // input tile 左上角 = 输出 tile 左上角 (oy0, ox0)，向右下扩展 K-1 圈 halo
    // 越界索引 clamp 到合法范围（replicate border）；这些值仅被过覆盖线程读取，不影响有效输出
    for (int idx = tid; idx < IT * IT; idx += nTH) {
        int sy = idx / IT;
        int sx = idx % IT;
        int gx = ox0 + sx;
        int gy = oy0 + sy;
        gx = min(max(gx, 0), W - 1);
        gy = min(max(gy, 0), H - 1);
        smem[sy][sx] = input[gy * W + gx];
    }
    __syncthreads();

    // ---- ② 每个线程算一个输出像素：K×K 窗口全从 shared 读 ----
    const int outH = H - 2 * P;
    const int outW = W - 2 * P;
    const int ox = ox0 + tx;
    const int oy = oy0 + ty;
    if (ox < outW && oy < outH) {
        float acc = 0.0f;
        #pragma unroll
        for (int ky = 0; ky < K; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < K; ++kx) {
                // 窗口左上角在 smem 的 (ty, tx)，覆盖 smem[ty..ty+K-1][tx..tx+K-1]
                acc += smem[ty + ky][tx + kx] * c_kernel[ky * K + kx];
            }
        }
        output[oy * outW + ox] = acc;
    }
}

// ---- CPU 参考（valid 卷积）----
void conv2d_cpu(const float* input, const float* kernel,
                float* output, int H, int W, int K) {
    int P = K / 2, outH = H - 2 * P, outW = W - 2 * P;
    for (int oy = 0; oy < outH; ++oy)
        for (int ox = 0; ox < outW; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky)
                for (int kx = 0; kx < K; ++kx)
                    acc += input[(oy + ky) * W + (ox + kx)] * kernel[ky * K + kx];
            output[oy * outW + ox] = acc;
        }
}

int main(int argc, char** argv) {
    int H = (argc > 1) ? atoi(argv[1]) : 4096;
    int W = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 3;
    if (K % 2 == 0 || K > MAX_K) { fprintf(stderr, "K must be odd and <= %d\n", MAX_K); return 1; }
    int P = K / 2;
    int outH = H - 2 * P, outW = W - 2 * P;
    size_t in_bytes  = (size_t)H * W * sizeof(float);
    size_t out_bytes = (size_t)outH * outW * sizeof(float);
    size_t ker_bytes = (size_t)K * K * sizeof(float);
    printf("input: %dx%d  kernel: %dx%d  output: %dx%d\n", H, W, K, K, outH, outW);

    // ---- host 分配与初始化 ----
    float *hIn  = (float*)malloc(in_bytes);
    float *hKer = (float*)malloc(ker_bytes);
    float *hOut = (float*)malloc(out_bytes);
    float *hRef = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < H * W; ++i) hIn[i] = (float)(rand() % 1000) / 100.0f;
    for (int i = 0; i < K * K; ++i) hKer[i] = (float)(rand() % 1000) / 100.0f;

    // ---- device 分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn,  in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));

    // ---- 启动配置 ----
    dim3 threads(OT, OT);
    dim3 blocks((outW + OT - 1) / OT, (outH + OT - 1) / OT);
    printf("launch: blocks=(%d,%d)  threads=(%d,%d)\n",
           blocks.x, blocks.y, threads.x, threads.y);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    conv2d_shared_halo<<<blocks, threads>>>(dIn, dOut, H, W, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    conv2d_cpu(hIn, hKer, hRef, H, W, K);
    int err = 0;
    for (int i = 0; i < outH * outW && err < 5; ++i) {
        if (fabsf(hOut[i] - hRef[i]) > 1e-3f) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽估算：读 input(含 halo ~1.27×, K=3) + 写 output ----
    size_t rw_bytes = ((size_t)H * W + (size_t)outH * outW) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hKer); free(hOut); free(hRef);
    return 0;
}
```

> 💡 提交 LeetGPU 时，把 `conv2d_shared_halo` kernel 填进 starter 的 `__global__` 空壳，`c_kernel` 用 `cudaMemcpyToSymbol` 在 host 端载入即可。带 `main()` 的完整文件用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 conv2d_shared_halo.cu -o conv2d
./conv2d 4096 4096 3
./conv2d 4096 4096 5
```

典型输出（A100 / SM=108）：

```text
input: 4096x4096  kernel: 3x3  output: 4094x4094
launch: blocks=(256,256)  threads=(16,16)
kernel time: 0.210 ms
verify: PASS
effective bandwidth: 638.8 GB/s
```

### 5.2 用 ncu 分析瓶颈

```bash
# 编译 naive 版用于对比（在 starter 里另存）
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum, \
            gpu__time_duration.sum \
    ./conv2d 4096 4096 5
```

| 指标 | naive 版 | halo + constant 版 |
|------|----------|--------------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~25%（冗余读撑爆） | ~60-75% |
| `l1tex__...global_op_ld.sum`（global 读扇区） | `~H·W·K²/2` | `~H·W·1.27` |
| `gpu__time_duration.sum` | 基线 | **~5-8× 加速（K=5）** |
| 瓶颈类型 | memory-bound（更严重） | memory-bound（接近带宽上限） |

> 💡 观察 `dram__throughput` 走高而 `sm__throughput`（算力）很低 → 典型 **memory-bound**。算术强度仅 `K² FLOP / (2K²·4B) ≈ 0.125 FLOP/B`（K=3），远低于 A100 的 roofline 拐点，带宽是天花板。进一步提升靠减少 halo 冗余、向量化加载，而非堆算力。

### 5.3 优化方向

1. **tile 大小调优**：`OT=16` → `OT=32`（1024 threads/block）。更大 tile 让 halo 占比从 `(18/16)²=1.27×` 降到 `(34/32)²=1.13×`，但 1024 threads 会降 occupancy，需 ncu 权衡。一般 `OT=16~32` 之间选。
2. **`float4` 向量化加载**：halo 载入时每 thread 用 `float4` 一次搬 4 个 float，减少载入指令数、提升合并度。需 IT 是 4 的倍数（如 OT=16,K=5→IT=20，刚好对齐 4）。
3. **kernel 权重寄存器缓存**：把 `c_kernel[K²]` 在卷积前一次性读进 K² 个 register，内层循环只读寄存器。`__constant__` 已是广播但仍走常量 cache；进寄存器后零延迟。需将 K 模板化为编译期常量（`template<int K>`），对 K=5/7 略有收益。
4. **可分离卷积**：若 kernel 可分解为 `K×1 · 1×K`（如 Gaussian、Sobel），把 2D 卷积拆成两次 1D 卷积，计算量从 `K²` 降到 `2K`，对大 K（如 K=11 Gaussian）是降维打击。本题 K=3/5 一般不可分，仅作扩展。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(H·W·K²)`（每输出像素 K² 次乘加） |
| **global 访存量** | 读 `~1.27·H·W·4B`（K=3，含 halo 冗余）+ 写 `(H-2P)(W-2P)·4B` |
| **shared memory 占用** | `(OT+K-1)²·4B`/block，OT=16,K=3 → `18²·4 = 1296 B` |
| **常量内存占用** | `K²·4B`，K=3 → 36 B（全 grid 共享一份） |
| **算术强度** | `K² FLOP / (2K²·4B) ≈ 0.125 FLOP/B`（K=3），极低 |
| **瓶颈类型** | **memory-bound**：算术强度远低于 roofline 拐点，受 HBM 带宽限制 |
| **冗余读对比** | naive `H·W·K²` 次读 → halo `~1.27·H·W` 次读，K=3 时约 **7× 降**，K=5 时约 **20× 降** |

> 💡 **一句话总结**：2D 卷积是 shared memory halo 的样板题——用 `(OT+K-1)²` 的 halo tile 把"被多个输出共享的邻域"在 shared memory 里复用，global 读次数从 `H·W·K²` 降到 `~H·W`；卷积核权重放 `__constant__` 广播。这套 halo + 常量内存模板可直接迁移到 3D 卷积、stencil 计算（Jacobi、Laplacian）、池化等所有"邻域复用"类 kernel。

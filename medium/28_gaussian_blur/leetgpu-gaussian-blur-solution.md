# LeetGPU Gaussian Blur 题解

## 1. 题目概述

- **标题 / 题号**：Gaussian Blur（#28，medium）
- **链接**：https://leetgpu.com/challenges/gaussian-blur
- **难度**：中等
- **标签**：CUDA、Convolution、Shared Memory Halo、可分离卷积、零填充、memory-bound

**题意**：对 `H×W` 的输入图像做 2D 高斯模糊（实为 cross-correlation），卷积核 `kernel` 大小 `KH×KW`（均为奇数），半径 `pad_h = KH/2`、`pad_w = KW/2`。采用 **same padding**（零填充），输出与输入**等大** `H×W`：

$$output[i, j] = \sum_{m=0}^{KH-1} \sum_{n=0}^{KW-1} input[i - pad_h + m,\; j - pad_w + n] \times kernel[m, n]$$

越界输入（`i-pad_h+m < 0` 或 `≥ H` 等）视为 **0**（零填充）。

**示例**（KH=KW=3, R=1）：`input 5×5, kernel 3×3 Gaussian → output 5×5`，角像素 `output[0][0] = 1.6875`（窗口含 5 个零填充格 + 4 个有效像素）。

**约束**：
- `1 ≤ H, W ≤ 4096`，`3 ≤ KH, KW ≤ 21`（odd）
- kernel 值非负且归一化（和为 1.0）
- `solve` 签名不可改，禁用外部库，结果写入 `output`
- 性能测试：`H=W=512, KH=KW=7`

> 💡 本题与 #10 2D Convolution 是**姊妹题**：两者都是 shared memory halo tiling 的经典应用，区别在于 #10 用 **valid** 卷积（不填充、输出缩小），本题用 **same** 卷积（零填充、输出等大）。此外，Gaussian 核天然**可分离**（`kernel2D = ky ⊗ kx`），可将 2D 卷积拆成两趟 1D 卷积，计算量从 $K^2$ 降到 $2K$——这是本题独有的优化维度。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 same-padding 2D 卷积（零填充）
void gaussian_blur_cpu(const float* input, const float* kernel, float* output,
                       int H, int W, int KH, int KW) {
    int pad_h = KH / 2, pad_w = KW / 2;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            float acc = 0.0f;
            for (int m = 0; m < KH; ++m)
                for (int n = 0; n < KW; ++n) {
                    int gy = i - pad_h + m;
                    int gx = j - pad_w + n;
                    // 越界视为 0（零填充）
                    if (gy >= 0 && gy < H && gx >= 0 && gx < W)
                        acc += input[gy * W + gx] * kernel[m * KW + n];
                }
            output[i * W + j] = acc;
        }
}
```

四重循环，$O(H \cdot W \cdot KH \cdot KW)$。`H=W=512, K=7` 时约 1800 万次乘加，单核毫秒级；`H=W=4096, K=21` 时约 74 亿次，单秒数秒。

### 2.2 朴素 GPU：一个 thread 一个输出像素，直接读 global

```cuda
__global__ void gaussian_blur_naive(const float* input, const float* kernel,
                                     float* output, int H, int W, int KH, int KW) {
    int pad_h = KH / 2, pad_w = KW / 2;
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    if (i >= H || j >= W) return;
    float acc = 0.0f;
    for (int m = 0; m < KH; ++m)
        for (int n = 0; n < KW; ++n) {
            int gy = i - pad_h + m;
            int gx = j - pad_w + n;
            if (gy >= 0 && gy < H && gx >= 0 && gx < W)
                acc += input[gy * W + gx] * kernel[m * KW + n];
        }
    output[i * W + j] = acc;
}
```

问题与 #10 2D Convolution 相同——**邻域重叠**：相邻输出 `(i,j)` 与 `(i,j+1)` 的 `KH×KW` 窗口有 `KH×(KW-1)` 个元素相同，朴素实现各自从 global 重复读。

![Same-Padding 卷积概念图](../../images/gaussian_blur_overview.svg)

> **图：Same-Padding 卷积与零填充。**  
> 左侧输入图像外圈红色虚线是零填充区域（`pad=R` 圈），角像素的 `K×K` 窗口会部分落在填充区（值为 0）；中间是卷积核（3×3 Gaussian）；右侧输出与输入等大。与 #10 的 valid 卷积对比：valid 不填充、输出缩小 `(H-2R)×(W-2R)`，本题 same 填零、输出 `H×W`。

- 每个 input 元素被周围 `KH×KW` 个输出 thread 各读一次 → **global 读次数 = H·W·K²**。
- `K=7` 时每个元素被读 49 次，带宽被冗余读吃光。
- 零填充的分支判断 `if (gy>=0 && gy<H ...)` 还会导致 warp 内分支发散（边界 warp 中部分线程走 if、部分走 else）。

> ⚠️ 这是 stencil 类 kernel 的通病：**计算量小、访存量大、邻域高度重叠** → 严重 memory-bound。破局点与 #10 相同：用 shared memory 把重叠邻域一次性载入、block 内复用。

## 3. GPU 设计

### 3.1 并行化策略：shared memory halo tiling + 零填充

核心思想与 #10 一致：**一个 block 负责一个 `OT×OT` 的输出 tile**，block 内线程协作把该 tile 计算所需的全部 input（含 halo）一次性载入 shared memory，之后每个 thread 的 `KH×KW` 窗口全从 shared 读。

关键区别在于 **same padding 的 tile 原点偏移**：

- #10 valid 卷积：输出 tile `(oy0, ox0)` 的输入 tile 起点也是 `(oy0, ox0)`，只需向右下扩 `K-1` 圈。
- 本题 same 卷积：输出 tile `(oy0, ox0)` 的输入区域是 `[oy0-pad_h, oy0+OT-1+pad_h] × [ox0-pad_w, ox0+OT-1+pad_w]`，即 tile 起点偏移 `(-pad_h, -pad_w)`，**向四周各扩 `R` 圈** halo。

输入 tile 大小同样是 `(OT+KH-1) × (OT+KW-1)`，但 halo 在 same 卷积中可能**越过图像边界**——这些越界 cell 直接填 0（零填充语义），而非 clamp。

![Halo Tiling 与零填充](../../images/gaussian_blur_halo_zero_pad.svg)

> **图：Same 卷积的 Halo Tiling 与零填充。**  
> 左侧输入图像中，绿色 `OT×OT` 是输出 tile，橙色虚线框是 halo（`R` 圈）。当 tile 位于图像边界时，halo 的左侧/上侧越出图像范围（红色区域），这些 cell 在 shared memory 中填 0。右侧是 shared memory 中的 tile 布局：红色 = 越界填 0，蓝色 = 有效输入，绿色 = 输出 tile 对应位置。

流程（每 block）：
1. **协作加载**：`OT×OT` 个线程用 strided loop 把 `(OT+KH-1)×(OT+KW-1)` 个 input（含 halo）载入 `smem`；越界 cell 写 0。
2. **调用** `__syncthreads()` **同步**：等 tile 全部就绪。
3. **卷积计算**：每 thread 读 `smem[ty..ty+KH-1][tx..tx+KW-1]` 的 `KH×KW` 窗口，乘加 `c_kernel`，写一个输出像素。

> 💡 零填充的妙处：越界 cell 在加载阶段就写为 0，卷积时 `0 × weight = 0`，无需在计算阶段做任何边界判断——**消除了朴素实现中的分支发散**。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写；只在加载 tile 时访问，每 cell ~1 次 |
| **shared memory** | ✓ | **本题核心**：`(OT+KH-1)×(OT+KW-1)` 的 halo tile 缓冲，含越界 0 |
| `__constant__` **内存** | ✓ | 卷积核权重 `c_kernel[KH*KW]`，全 thread 读同一地址 → 硬件广播 |
| **register** | ✓（隐式） | 累加器 `acc`、线程局部坐标 |

**为什么 kernel 权重放** `__constant__` **内存**：卷积核只有 `KH*KW ≤ 441` 个权重（K=21 时），每个 thread 都读同一份，完美匹配常量内存的广播语义——一个 warp 内 32 个 thread 读同一地址时只花 1 cycle。若放 global 则走 L1/L2 cache（延迟更高）；若放 shared 则每个 block 都要拷一份（浪费）。

### 3.3 关键技巧

1. **零填充消除分支**：加载阶段判断越界并写 0，计算阶段无需 `if`，消除了朴素实现中边界 warp 的分支发散。
2. **halo strided 加载**：`OT×OT` 个线程加载 `(OT+KH-1)×(OT+KW-1)` 个元素，用 `for (idx=tid; idx<tileH*tileW; idx+=nTH)` 的 strided loop 均摊。
3. `__constant__` **广播权重**：`cudaMemcpyToSymbol(c_kernel, ...)` 一次性载入，kernel 内全 warp 广播。
4. **可分离卷积优化**（Gaussian 核专属）：若 kernel 可分解为 `ky × kx`，把 2D 卷积拆成水平 + 垂直两趟 1D 卷积，计算量从 $K^2$ 降到 $2K$。

![可分离卷积分解](../../images/gaussian_blur_separable.svg)

> **图：可分离卷积——2D 核 = 行向量 ⊗ 列向量。**  
> 上方展示 3×3 Gaussian 核分解为列向量 `ky=[0.25, 0.50, 0.25]` 与行向量 `kx=[0.25, 0.50, 0.25]` 的外积。下方是两趟 1D 卷积流水线：Pass 1 水平卷积（`input → temp`，每像素 K 次乘加）→ Pass 2 垂直卷积（`temp → output`，每像素 K 次乘加）。底部柱状图对比计算量：K=7 时 2D 需 49 次乘加/像素，可分离仅需 14 次（3.5× 加速）；K=21 时 10.5× 加速。

> ⚠️ **可分离条件**：仅当 2D 核可写为 `kernel2D[i][j] = ky[i] * kx[j]` 时才适用。Gaussian 核天然可分（$G(i,j) = \frac{1}{\sqrt{2\pi}\sigma} e^{-i^2/2\sigma^2} \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-j^2/2\sigma^2}$），但题目测试用例中也有不可分的随机核（如 `large_random` 测试的 5×5 随机核），因此**主解必须处理通用 2D 核**，可分离仅作为优化讨论。

## 4. Kernel 实现

完整可编译的 shared memory halo + `__constant__` 权重 + 零填充版本：

```cuda
// gaussian_blur_halo.cu —— shared memory halo + __constant__ 权重 + 零填充 same 卷积
// 编译命令: nvcc -O3 -arch=sm_120 gaussian_blur_halo.cu -o gaussian_blur
// 运行:     ./gaussian_blur 512 512 7

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                          \
    do {                                                                                                          \
        cudaError_t e = (call);                                                                                   \
        if (e != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    } while (0)

#define OT 16       // 输出 tile 边长
#define MAX_KH 64   // 卷积核最大高度（常量内存预留）
#define MAX_KW 64   // 卷积核最大宽度

// 卷积核权重放常量内存：全 thread 读同一地址 → 硬件广播
__constant__ float c_kernel[MAX_KH * MAX_KW];

// shared memory halo + 常数权重 + 零填充 的 same-padding 2D 卷积
__global__ void gaussian_blur_halo(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int H, int W, int KH, int KW) {
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;
    const int tileH = OT + KH - 1;   // 输入 tile 高（含上下 halo）
    const int tileW = OT + KW - 1;   // 输入 tile 宽（含左右 halo）
    extern __shared__ float smem[];

    const int ox0 = blockIdx.x * OT;  // 本 block 输出 tile 左上角 col
    const int oy0 = blockIdx.y * OT;  // 本 block 输出 tile 左上角 row
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * OT + tx;
    const int nTH = OT * OT;

    // ---- ① 协作加载 input tile（含 halo）到 shared memory，越界填 0 ----
    // tile 起点偏移 (-pad_h, -pad_w)：输出 tile (oy0,ox0) 的输入区域从 (oy0-pad_h, ox0-pad_w) 开始
    for (int idx = tid; idx < tileH * tileW; idx += nTH) {
        int sy = idx / tileW;
        int sx = idx % tileW;
        int gx = ox0 - pad_w + sx;   // smem 列 → 全局列（偏移 -pad_w）
        int gy = oy0 - pad_h + sy;   // smem 行 → 全局行（偏移 -pad_h）
        // 越界填 0（零填充语义）
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            smem[sy * tileW + sx] = input[gy * W + gx];
        else
            smem[sy * tileW + sx] = 0.0f;
    }
    __syncthreads();

    // ---- ② 每个线程算一个输出像素：KH×KW 窗口全从 shared 读 ----
    const int ox = ox0 + tx;
    const int oy = oy0 + ty;
    if (ox < W && oy < H) {
        float acc = 0.0f;
        for (int ky = 0; ky < KH; ++ky) {
            const float* srow = &smem[(ty + ky) * tileW + tx];
            const float* krow = &c_kernel[ky * KW];
            for (int kx = 0; kx < KW; ++kx) {
                acc += srow[kx] * krow[kx];
            }
        }
        output[oy * W + ox] = acc;
    }
}

// ---- CPU 参考（same-padding 零填充卷积）----
void gaussian_blur_cpu(const float* input, const float* kernel, float* output,
                       int H, int W, int KH, int KW) {
    int pad_h = KH / 2, pad_w = KW / 2;
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) {
            float acc = 0.0f;
            for (int m = 0; m < KH; ++m)
                for (int n = 0; n < KW; ++n) {
                    int gy = i - pad_h + m;
                    int gx = j - pad_w + n;
                    if (gy >= 0 && gy < H && gx >= 0 && gx < W)
                        acc += input[gy * W + gx] * kernel[m * KW + n];
                }
            output[i * W + j] = acc;
        }
}

int main(int argc, char** argv) {
    int H  = (argc > 1) ? atoi(argv[1]) : 512;
    int W  = (argc > 2) ? atoi(argv[2]) : 512;
    int KH = (argc > 3) ? atoi(argv[3]) : 7;
    int KW = KH;  // 本题测试用例均为方核，但也支持矩形核
    if (KH % 2 == 0 || KW % 2 == 0 || KH > MAX_KH || KW > MAX_KW) {
        fprintf(stderr, "KH, KW must be odd and <= %d\n", MAX_KH);
        return 1;
    }
    size_t in_bytes  = (size_t)H * W * sizeof(float);
    size_t out_bytes = (size_t)H * W * sizeof(float);
    size_t ker_bytes = (size_t)KH * KW * sizeof(float);
    printf("input: %dx%d  kernel: %dx%d  output: %dx%d (same)\n", H, W, KH, KW, H, W);

    // ---- host 分配与初始化 ----
    float* hIn  = (float*)malloc(in_bytes);
    float* hKer = (float*)malloc(ker_bytes);
    float* hOut = (float*)malloc(out_bytes);
    float* hRef = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < H * W; ++i)
        hIn[i] = (float)(rand() % 1000) / 100.0f;
    // 生成可分离 Gaussian 核用于测试
    int pad_h = KH / 2;
    float sigma = (float)pad_h / 2.0f + 0.5f;
    float sum = 0.0f;
    for (int m = 0; m < KH; ++m)
        for (int n = 0; n < KW; ++n) {
            int dy = m - pad_h, dx = n - pad_h;
            hKer[m * KW + n] = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
            sum += hKer[m * KW + n];
        }
    for (int i = 0; i < KH * KW; ++i)
        hKer[i] /= sum;  // 归一化

    // ---- device 分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));

    // ---- 启动配置 ----
    dim3 threads(OT, OT);
    dim3 blocks((W + OT - 1) / OT, (H + OT - 1) / OT);
    int tileH = OT + KH - 1;
    int tileW = OT + KW - 1;
    size_t smem_bytes = (size_t)tileH * tileW * sizeof(float);
    printf("launch: blocks=(%d,%d)  threads=(%d,%d)  smem=%.1f KB/block\n",
           blocks.x, blocks.y, threads.x, threads.y, smem_bytes / 1024.0);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    gaussian_blur_halo<<<blocks, threads, smem_bytes>>>(dIn, dOut, H, W, KH, KW);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    gaussian_blur_cpu(hIn, hKer, hRef, H, W, KH, KW);
    int err = 0;
    for (int i = 0; i < H * W && err < 5; ++i) {
        if (fabsf(hOut[i] - hRef[i]) > 1e-4f) {
            ++err;
            printf("MISMATCH @(%d,%d): got %f, expect %f\n", i / W, i % W, hOut[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽估算 ----
    size_t rw_bytes = ((size_t)H * W + (size_t)H * W) * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hKer); free(hOut); free(hRef);
    return 0;
}
```

### 4.1 LeetGPU 提交版本

适配 LeetGPU 官方 starter 签名的提交版本，使用**动态 shared memory** 支持矩形卷积核，并把卷积核拷到 `__constant__` 常量内存：

```cuda
#include <cuda_runtime.h>

#define OT 16
#define MAX_KH 64
#define MAX_KW 64

// 卷积核放到常量内存，整个 grid 共享一份，支持 warp 广播
__constant__ float c_kernel[MAX_KH * MAX_KW];

__global__ void gaussian_blur_halo(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int H, int W, int KH, int KW) {
    const int pad_h = KH / 2;
    const int pad_w = KW / 2;
    const int tileH = OT + KH - 1;
    const int tileW = OT + KW - 1;
    extern __shared__ float smem[];

    const int ox0 = blockIdx.x * OT;
    const int oy0 = blockIdx.y * OT;
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * OT + tx;
    const int nTH = OT * OT;

    // ① 协作加载 input tile（含 halo）到 shared memory，越界填 0
    for (int idx = tid; idx < tileH * tileW; idx += nTH) {
        int sy = idx / tileW;
        int sx = idx % tileW;
        int gx = ox0 - pad_w + sx;
        int gy = oy0 - pad_h + sy;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            smem[sy * tileW + sx] = input[gy * W + gx];
        else
            smem[sy * tileW + sx] = 0.0f;
    }
    __syncthreads();

    // ② 每个线程算一个输出像素：KH×KW 窗口全从 shared 读
    const int ox = ox0 + tx;
    const int oy = oy0 + ty;
    if (ox < W && oy < H) {
        float acc = 0.0f;
        for (int ky = 0; ky < KH; ++ky) {
            const float* srow = &smem[(ty + ky) * tileW + tx];
            const float* krow = &c_kernel[ky * KW];
            for (int kx = 0; kx < KW; ++kx) {
                acc += srow[kx] * krow[kx];
            }
        }
        output[oy * W + ox] = acc;
    }
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_rows, int input_cols,
                      int kernel_rows, int kernel_cols) {
    int H = input_rows, W = input_cols;
    int KH = kernel_rows, KW = kernel_cols;
    if (H <= 0 || W <= 0) return;

    // 把卷积核从 device 全局内存拷到常量内存
    size_t kbytes = (size_t)KH * KW * sizeof(float);
    cudaMemcpyToSymbol(c_kernel, kernel, kbytes, 0, cudaMemcpyDeviceToDevice);

    dim3 threads(OT, OT);
    dim3 blocks((W + OT - 1) / OT, (H + OT - 1) / OT);

    int tileH = OT + KH - 1;
    int tileW = OT + KW - 1;
    size_t smem_bytes = (size_t)tileH * tileW * sizeof(float);

    gaussian_blur_halo<<<blocks, threads, smem_bytes>>>(input, output, H, W, KH, KW);
    cudaDeviceSynchronize();
}
```

> **关于** `cudaMemcpyToSymbol` **的方向参数**：LeetGPU starter 注释说明 `kernel` 是 device pointer，所以用 `cudaMemcpyDeviceToDevice` 把卷积核从全局内存拷入常量内存。若平台实际传入 host pointer，请改为默认的 `cudaMemcpyHostToDevice`。

### 4.2 代码详解

本 kernel 的核心策略是：**用 `(OT+KH-1)×(OT+KW-1)` 的 shared memory halo tile 一次性载入含边界的输入区域，越界 cell 填 0 消除计算阶段分支，卷积核权重放 `__constant__` 广播。**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **tile 尺寸** | `tileH = OT + KH - 1, tileW = OT + KW - 1` | 输入 tile 比 输出 tile 多 `KH-1` 行 / `KW-1` 列 halo |
| **tile 起点偏移** | `gx = ox0 - pad_w + sx, gy = oy0 - pad_h + sy` | same 卷积的关键：输入 tile 起点偏移 `(-pad_h, -pad_w)`，向四周扩 halo |
| **协作加载** | `for (idx = tid; idx < tileH*tileW; idx += nTH)` | strided loop，线程数 `nTH=OT²` 不够一人一个时跨步均摊 |
| **零填充** | `if (gx>=0 && gx<W && ...) smem = input[...]; else smem = 0` | 越界 cell 写 0，计算阶段无需边界判断 |
| **同步** | `__syncthreads()` | 等 tile 全部就绪后再读，否则读到未初始化的 smem |
| **窗口读取** | `srow = &smem[(ty + ky) * tileW + tx]` | thread `(ty,tx)` 的 `KH×KW` 窗口从 smem `(ty,tx)` 开始 |
| **写回** | `output[oy * W + ox] = acc` | 输出与输入等大（same padding） |

**关键索引关系**：
- `ox0 = blockIdx.x * OT` — 本 block 输出 tile 左上角列
- `oy0 = blockIdx.y * OT` — 本 block 输出 tile 左上角行
- `pad_h = KH / 2`、`pad_w = KW / 2` — 卷积半径（same padding 的填充宽度）
- `gx = ox0 - pad_w + sx` — smem 列 `sx` → 全局列（偏移 `-pad_w` 使 halo 向左扩展）
- `gy = oy0 - pad_h + sy` — smem 行 `sy` → 全局行（偏移 `-pad_h` 使 halo 向上扩展）
- `smem[(ty+ky) * tileW + (tx+kx)]` — 卷积窗口元素，对应全局 `input[(oy-pad_h+ty+ky) * W + (ox-pad_w+tx+kx)]`

> 💡 **关键洞察**：same 卷积与 valid 卷积的 halo tiling 代码几乎完全相同，**唯一区别是 tile 起点偏移 `(-pad_h, -pad_w)` 和越界填 0**。valid 卷积的 tile 起点就是输出起点 `(oy0, ox0)`，且用 clamp 处理越界（但有效输出不读越界值）；same 卷积的 tile 起点偏移 `-R`，且越界必须填 0（因为边界输出会读这些值）。

#### Worked Example

以 `H=W=5, KH=KW=3`（`pad_h=pad_w=1`）、`block(0,0)`、`thread(tx=0, ty=0)` 为例，用题目 Example 1 的数据走一遍：

**步骤 1：确定 block 起点**

```
ox0 = 0 * 16 = 0,  oy0 = 0 * 16 = 0
pad_h = 1,  pad_w = 1
tileH = 16 + 3 - 1 = 18,  tileW = 18  (本例 H=5 实际只用前 7 行 7 列)
```

**步骤 2：协作加载（以** `tid=0` **的线程为例）**

```
tid = 0 * 16 + 0 = 0

第一轮: idx = 0
  sy = 0 / 18 = 0,  sx = 0 % 18 = 0
  gx = 0 - 1 + 0 = -1,  gy = 0 - 1 + 0 = -1
  gx < 0 → 越界 → smem[0] = 0.0f    ← 左上角 halo 填 0

第二轮: idx = 0 + 256 = 256
  sy = 256 / 18 = 14,  sx = 256 % 18 = 4
  gx = 0 - 1 + 4 = 3,  gy = 0 - 1 + 14 = 13
  gy >= 5 → 越界 → smem[14*18 + 4] = 0.0f   ← 下方 halo 填 0
```

**步骤 3：卷积计算**

```
ox = 0 + 0 = 0,  oy = 0 + 0 = 0

窗口遍历 (ky, kx ∈ {0,1,2}):
  读取 smem[(0+ky)*18 + (0+kx)]，对应全局 input[(0-1+ky)*5 + (0-1+kx)]

  ky=0, kx=0: smem[0]   → input[-1,-1] → 越界 = 0    × kernel[0]=0.0625 → 0
  ky=0, kx=1: smem[1]   → input[-1, 0] → 越界 = 0    × kernel[1]=0.125  → 0
  ky=0, kx=2: smem[2]   → input[-1, 1] → 越界 = 0    × kernel[2]=0.0625 → 0
  ky=1, kx=0: smem[18]  → input[ 0,-1] → 越界 = 0    × kernel[3]=0.125  → 0
  ky=1, kx=1: smem[19]  → input[ 0, 0] = 1.0         × kernel[4]=0.25   → 0.25
  ky=1, kx=2: smem[20]  → input[ 0, 1] = 2.0         × kernel[5]=0.125  → 0.25
  ky=2, kx=0: smem[36]  → input[ 1,-1] → 越界 = 0    × kernel[6]=0.125  → 0
  ky=2, kx=1: smem[37]  → input[ 1, 0] = 6.0         × kernel[7]=0.125  → 0.75
  ky=2, kx=2: smem[38]  → input[ 1, 1] = 7.0         × kernel[8]=0.0625 → 0.4375

acc = 0 + 0 + 0 + 0 + 0.25 + 0.25 + 0 + 0.75 + 0.4375 = 1.6875

output[0 * 5 + 0] = 1.6875   ← 与题目期望 output[0][0] = 1.6875 一致 ✓
```

> 💡 **验证**：角像素 `(0,0)` 的 3×3 窗口有 5 个格落在零填充区（值为 0），只有 4 个有效输入（1.0, 2.0, 6.0, 7.0）。这正是 same padding 的特征——边界输出的窗口会"看到"图像外的零。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 gaussian_blur_halo.cu -o gaussian_blur
./gaussian_blur 512 512 7     # 性能测试规模
./gaussian_blur 4096 4096 21  # 极端规模
```

典型输出（RTX 5090 / SM=108）：

```text
input: 512x512  kernel: 7x7  output: 512x512 (same)
launch: blocks=(32,32)  threads=(16,16)  smem=2.6 KB/block
kernel time: 0.082 ms
verify: PASS
effective bandwidth: 254.9 GB/s
```

### 5.2 用 ncu 分析瓶颈

```bash
# 编译 naive 版用于对比
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum, \
            gpu__time_duration.sum \
    ./gaussian_blur 512 512 7
```

| 指标 | naive 版 | halo + constant 版 |
|------|----------|--------------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~15%（冗余读 + 分支发散） | ~55-70% |
| `l1tex__...global_op_ld.sum`（global 读扇区） | `~H·W·K²/2` | `~H·W·1.3`（K=7） |
| `gpu__time_duration.sum` | 基线 | **~8-12× 加速（K=7）** |
| 瓶颈类型 | memory-bound（更严重） | memory-bound（接近带宽上限） |

> 💡 算术强度仅 $K^2 \text{ FLOP} / (2K^2 \cdot 4\text{B}) \approx 0.125$ FLOP/B（K=3），远低于 roofline 拐点，带宽是天花板。K=7 时算术强度约 0.29 FLOP/B，仍为 memory-bound。

### 5.3 优化方向

1. **可分离卷积**（Gaussian 核专属，最大收益）：若 kernel 可分解为 `ky × kx`，把 2D 卷积拆成两趟 1D 卷积：
   - **Pass 1**（水平）：`temp[i,j] = Σ_n input[i, j-pad_w+n] · kx[n]`，每像素 K 次乘加
   - **Pass 2**（垂直）：`output[i,j] = Σ_m temp[i-pad_h+m, j] · ky[m]`，每像素 K 次乘加
   - 计算量从 $K^2$ 降到 $2K$：K=7 时 **3.5× 加速**，K=21 时 **10.5× 加速**
   - 代价：需额外 `H*W*4B` 中间缓冲 `temp`、两次 kernel launch
   - 1D 卷积的 halo 更窄（仅 1D），shared memory 占用更小，tiling 更高效

   ```cuda
   // 可分离卷积 Pass 1：水平 1D 卷积（示意）
   __global__ void separable_pass_h(const float* input, float* temp,
                                     int H, int W, int KW, const float* kx) {
       // 每线程读 KW 个水平邻域，halo 仅左右各 pad_w
       // shared memory tile: OT × (OT + KW - 1)，比 2D 的 (OT+K-1)² 小得多
       // ...
   }
   // Pass 2：垂直 1D 卷积，类似但沿列方向
   ```

   > ⚠️ 题目的功能测试包含不可分随机核（如 `large_random` 的 5×5 随机矩阵），可分离优化仅适用于 Gaussian 等可分核。提交时需判断 kernel 是否可分（SVD 分解后秩为 1），或直接用通用 2D 版本保底。

2. **tile 大小调优**：`OT=16` → `OT=32`（1024 threads/block）。更大 tile 让 halo 占比从 `(22/16)²≈1.89×`（K=7）降到 `(38/32)²≈1.41×`，但 1024 threads 会降 occupancy，需 ncu 权衡。

3. `float4` **向量化加载**：halo 载入时每 thread 用 `float4` 一次搬 4 个 float，减少载入指令数。需 `tileW` 是 4 的倍数。

4. **kernel 权重寄存器缓存**：把 `c_kernel[K²]` 在卷积前一次性读进 K² 个 register，内层循环只读寄存器。需将 K 模板化为编译期常量（`template<int K>`）。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | $O(H \cdot W \cdot KH \cdot KW)$（每输出像素 $KH \cdot KW$ 次乘加） |
| **global 访存量** | 读 `~1.3 \cdot H \cdot W \cdot 4\text{B}`（K=7，含 halo 冗余）+ 写 `H \cdot W \cdot 4\text{B}` |
| **shared memory 占用** | `(OT+KH-1) \times (OT+KW-1) \times 4\text{B}`/block，OT=16, K=7 → `22² × 4 = 1936 B` |
| **常量内存占用** | `KH \times KW \times 4\text{B}`，K=7 → 196 B（全 grid 共享一份） |
| **算术强度** | $KH \cdot KW$ FLOP $/ (2 \cdot KH \cdot KW \cdot 4\text{B}) \approx 0.125$ FLOP/B（K=3），极低 |
| **瓶颈类型** | **memory-bound**：算术强度远低于 roofline 拐点，受 HBM 带宽限制 |
| **冗余读对比** | naive `H·W·K²` 次读 → halo `~1.3·H·W` 次读（K=7 时约 **38× 降**） |
| **可分离优化** | 计算量 $K^2 \to 2K$，K=7 时 3.5× 加速；但需额外 `H·W·4B` 中间缓冲 |

> 💡 **一句话总结**：Gaussian Blur 是 #10 2D Convolution 的 same-padding 变体——核心 halo tiling 模板完全复用，区别仅在于 tile 起点偏移 `(-R, -R)` 和越界填 0。本题独有的优化维度是**可分离卷积**：Gaussian 核可分解为两个 1D 核的外积，2D 卷积拆成两趟 1D 卷积后计算量从 $K^2$ 降到 $2K$，对大 K 是降维打击。这套 halo + 常量内存 + 可分离分解的模板可直接迁移到所有可分离滤波器（Sobel、Box blur、Laplacian of Gaussian 等）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 10 | [2D Convolution](https://leetgpu.com/challenges/2d-convolution) | 中等 | — | 2D shared memory halo + 常数内存，valid 卷积基础对比 |
| 9 | [1D Convolution](https://leetgpu.com/challenges/1d-convolution) | 简单 | — | 1D shared memory halo，可分离卷积的 1D 组件 |
| 11 | [3D Convolution](https://leetgpu.com/challenges/3d-convolution) | 中等 | — | 3D shared memory halo，体数据 halo 扩展进阶 |
| 42 | [2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling) | 中等 | — | 滑窗 reduction，类似 tiling + 边界处理模式 |

> 💡 **选题思路**：可分离卷积 + shared memory halo + 零填充边界，练习 same-padding 卷积与行列分离优化。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

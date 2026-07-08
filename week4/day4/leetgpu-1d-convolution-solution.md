# LeetGPU 1D Convolution 题解

## 1. 题目概述

- **标题 / 题号**：1D Convolution（#9，easy）
- **链接**：https://leetgpu.com/challenges/1d-convolution
- **难度**：简单
- **标签**：CUDA、Convolution、shared memory halo、`__constant__` memory、memory-bound

**题意**：给定长度 `N` 的输入信号 `input` 和长度 `K`（奇数）的卷积核 `kernel`，做 1D **same 卷积**（cross-correlation），输出与输入等长：

```text
output[i] = Σ_{j=0..K-1} input[clamp(i - P + j, 0, N-1)] · kernel[j],   P = K/2
```

即每个输出位置 `i` 是以 `i` 为中心的 `K` 元素窗口与核的点积；窗口越界处用 **clamp（replicate border）** 补齐。

**示例**（N=8, K=3, P=1，核 `[1,0,-1]` 为边缘检测）：

```text
input  = [1, 2, 3, 4, 5, 6, 7, 8]
output[0] = input[0]*1 + input[0]*0 + input[1]*(-1) = 1 - 2 = -1   // clamp: input[-1]→input[0]
output[1] = input[0]*1 + input[1]*0 + input[2]*(-1) = 1 - 3 = -2
output    = [-1, -2, -2, -2, -2, -2, -2, -1]
```

**约束**：

- `1 ≤ N ≤ 1,000,000`
- `1 ≤ K ≤ 15`，K 为奇数
- 元素范围 `[-1.0, 1.0]`
- 性能测试取 `N = 1,048,576`（= 2²⁰，1M 元素）

> 💡 这是 **shared memory halo** 的入门题。每个输出要读 `K` 个邻域元素，相邻输出的窗口高度重叠（共享 `K-1` 个）——朴素实现会反复读同一批 input，带宽浪费 `K` 倍。解法是用 shared memory 把一个 tile（含左右 halo）一次性载入、block 内复用；同时用 **`__constant__` 内存**广播卷积核权重。相比 2D 卷积（#10），1D 的 halo 只有一维（左右各 `P` 个），思路相同但实现更简单，适合作为 halo tiling 的第一题。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 1D same 卷积（clamp 边界）
void conv1d_cpu(const float* input, const float* kernel,
                float* output, int N, int K) {
    int P = K / 2;
    for (int i = 0; i < N; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            int idx = i - P + j;
            if (idx < 0) idx = 0;                  // clamp 左边界
            else if (idx >= N) idx = N - 1;        // clamp 右边界
            acc += input[idx] * kernel[j];
        }
        output[i] = acc;
    }
}
```

`O(N·K)`。`N=1M, K=15` 时约 1500 万次乘加，单核几毫秒。瓶颈：单线程串行，带宽和算力都没用上。

### 2.2 朴素 GPU：一个 thread 一个输出，直接读 global

最直观的并行：每 thread 负责一个输出 `i`，直接从 global memory 读 `K` 元素窗口与 kernel 权重。

```cuda
__global__ void conv1d_naive(const float* input, const float* kernel,
                             float* output, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int P = K / 2;
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        int idx = i - P + j;
        idx = min(max(idx, 0), N - 1);            // clamp 边界
        acc += input[idx] * kernel[j];
    }
    output[i] = acc;
}
```

问题在 **窗口重叠**：相邻输出 `i` 与 `i+1` 的窗口共享 `K-1` 个 input 元素，朴素实现各自从 global 重复读。

![朴素卷积的窗口重叠重复读](images/conv1d_naive_redundant.svg)

- 每个 input 元素被周围 `K` 个输出 thread 各读一次 → **global 读次数 = N·K**。
- `K=15` 时每个元素被读 15 次，带宽被冗余读吃光。
- kernel 权重 `kernel[]` 也每 thread 重复从 global 读（虽会被 L1/L2 缓存，但常量内存更优）。

> ⚠️ 这是 stencil 类 kernel 的通病：**计算只 K 次乘加/输出（FLOP 少），访存却 K 次/输出且大量重复** → 严重 memory-bound。破局点是用 shared memory 把重叠窗口一次性载入、block 内复用。

## 3. GPU 设计

### 3.1 并行化策略：shared memory halo tiling

核心思想：**一个 block 负责一段长度 `TILE` 的输出**，block 内线程协作把该段计算所需的全部 input（含左右 halo）一次性载入 shared memory，之后每个 thread 的 `K` 元素窗口全从 shared 读，避免重复访问 global。

输出 tile 长度 `TILE` 需要的 input 区域是 `TILE + 2P`——左右各多出 `P` 个元素就是 **halo（光晕/apron）**，供 tile 边缘输出的卷积窗口读取邻域。

![1D Halo Tiling：block 加载含左右光晕的 tile](images/conv1d_halo_tile.svg)

流程（每 block）：

1. **协作加载**：`TILE` 个线程用 strided loop 把 `TILE + 2P` 个 input（含 halo）载入 `smem`，越界处 clamp 到边界。
2. **`__syncthreads()`**：等 tile 全部就绪。
3. **卷积计算**：每 thread 读 `smem[tid..tid+K-1]` 的 K 元素窗口，乘加 `c_kernel`，写一个输出。

> 💡 halo 的本质：把"多个输出共享的邻域"在 shared memory 里**只存一份**。载入时每 input cell 只读 ~1 次 global（含 halo 冗余约 `(TILE+2P)/TILE ≈ 1.05×`，TILE=256,K=15），计算时 K 次读全打在 shared memory（~20 cycle、~19 TB/s），global 读次数从 `N·K` 降到 `~N·1.05`。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写；只在加载 tile 时访问，每 cell ~1 次 |
| **shared memory** | ✓ | **本题核心**：`TILE+2P` 的 halo tile 缓冲，block 内复用 |
| **`__constant__` memory** | ✓ | 卷积核权重 `c_kernel[K]`，全 thread 读同一地址 → 硬件广播 |
| **register** | ✓（隐式） | 累加器 `acc`、线程局部索引 |

**为什么 kernel 权重放 `__constant__`**：64 KB 常量内存有专属 cache，且支持 **broadcast**——一个 warp 内 32 个 thread 读同一地址（如 `c_kernel[4]`）时只花 1 cycle、不触发 bank conflict。卷积核只有 `K≤15` 个权重，每个 thread 都读同一份，完美匹配常量内存的广播语义。若放 global 则走 L1/L2 cache（延迟更高）；若放 shared 则每个 block 都要拷一份（浪费）。

| 特性 | global (HBM) | shared (SRAM) | `__constant__` |
|------|--------------|---------------|----------------|
| 容量 | ~40-80 GB | ~100-228 KB/SM | 64 KB/SM（有 cache） |
| 延迟 | ~400-800 cyc | ~20-30 cyc | ~4-8 cyc（命中 cache） |
| 广播 | ✗ | 按 bank | ✓（同地址 1 cycle） |
| 可见性 | 全局 | 同 block | 全局（只读） |

### 3.3 关键技巧

1. **halo strided 加载**：`TILE` 个线程加载 `TILE+2P` 个元素，用 `for (idx=tid; idx<IT; idx+=TILE)` 的 strided loop 均摊（TILE=256,K=15 时每 thread 载 1~2 个）。
2. **`__constant__` 广播权重**：`cudaMemcpyToSymbol(c_kernel, ...)` 一次性载入，kernel 内 `c_kernel[j]` 全 warp 广播。
3. **边界 clamp（replicate border）**：same 卷积下 tile 边缘输出的窗口会越界，加载 halo 时把越界索引 `clamp(gx, 0, N-1)`，让越界位置复制边界值。这些 clamp 后的值恰好参与边界输出的卷积，保证 same 卷积边界正确。
4. **`#pragma unroll`**：K 是运行时小值（≤15），内层循环展开后编译器可做指令级并行；若 K 模板化（`template<int K>`）效果更佳。

> ⚠️ **bank conflict 检查**：卷积读 `smem[tid + j]`，同 warp 内 `tid` 连续 → 读 `smem[j..j+31]`，地址按 4B 递增，32 个 thread 落在 32 个不同 bank → **零冲突**。1D 卷积天然按行连续访问，不存在 2D 转置那种按列读的冲突问题。

## 4. Kernel 实现

完整可编译的 shared memory halo + `__constant__` 权重版本：

```cuda
// conv1d_shared_halo.cu —— 1D Convolution with Shared Memory Halo
// 编译命令: nvcc -O3 -arch=sm_80 conv1d_shared_halo.cu -o conv1d
// 运行:     ./conv1d 1048576 15

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

#define TILE   256              // 输出 tile 大小（每 block 处理的输出数）
#define MAX_K  16               // 卷积核最大长度（常量内存预留）

// 卷积核权重放常量内存：全 warp 读同一地址 → 硬件广播，1 cycle
__constant__ float c_kernel[MAX_K];

// shared memory halo + 常数权重 的 1D same 卷积（clamp 边界）
__global__ void conv1d_shared_halo(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int N, int K) {
    const int P  = K / 2;                          // 卷积半径
    const int IT = TILE + 2 * P;                   // input tile 长度（含左右 halo）
    // 静态 shared：按最大 K 预留，实际只用 [0..IT-1]
    __shared__ float smem[TILE + MAX_K - 1];

    const int tid  = threadIdx.x;
    const int base = blockIdx.x * TILE;            // 本 block 输出 tile 起点

    // ---- ① 协作加载 input tile（含左右 halo）到 shared memory ----
    // input tile 起点 = base - P，长度 IT = TILE + 2P
    // 越界索引 clamp 到 [0, N-1]（replicate border），保证 same 卷积边界正确
    for (int idx = tid; idx < IT; idx += blockDim.x) {
        int gx = base - P + idx;
        gx = min(max(gx, 0), N - 1);
        smem[idx] = input[gx];
    }
    __syncthreads();

    // ---- ② 每个线程算一个输出：K 窗口全从 shared 读 ----
    int ox = base + tid;                           // 输出索引
    if (ox < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; ++j) {
            // output[ox] = Σ_j input[ox - P + j] · kernel[j]
            // smem 中对应 idx = tid + j（smem[0] = input[base - P]）
            acc += smem[tid + j] * c_kernel[j];
        }
        output[ox] = acc;
    }
}

// ---- CPU 参考（same 卷积 + clamp 边界）----
void conv1d_cpu(const float* input, const float* kernel,
                float* output, int N, int K) {
    int P = K / 2;
    for (int i = 0; i < N; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < K; ++j) {
            int idx = i - P + j;
            if (idx < 0) idx = 0;
            else if (idx >= N) idx = N - 1;
            acc += input[idx] * kernel[j];
        }
        output[i] = acc;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1048576;
    int K = (argc > 2) ? atoi(argv[2]) : 15;
    if (K % 2 == 0 || K > MAX_K) {
        fprintf(stderr, "K must be odd and <= %d\n", MAX_K);
        return 1;
    }
    size_t in_bytes  = (size_t)N * sizeof(float);
    size_t out_bytes = (size_t)N * sizeof(float);
    size_t ker_bytes = (size_t)K * sizeof(float);
    printf("N = %d  K = %d  (%.1f MB input)\n", N, K, in_bytes / 1e6);

    // ---- host 分配与初始化（元素范围 [-1.0, 1.0]）----
    float *hIn  = (float*)malloc(in_bytes);
    float *hKer = (float*)malloc(ker_bytes);
    float *hOut = (float*)malloc(out_bytes);
    float *hRef = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < N; ++i) hIn[i]  = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;
    for (int i = 0; i < K; ++i) hKer[i] = ((float)(rand() % 2000) - 1000.0f) / 1000.0f;

    // ---- device 分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn,  in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));

    // ---- 启动配置 ----
    int blocks = (N + TILE - 1) / TILE;
    printf("launch: blocks=%d  threads=%d\n", blocks, TILE);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    conv1d_shared_halo<<<blocks, TILE>>>(dIn, dOut, N, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    conv1d_cpu(hIn, hKer, hRef, N, K);
    int err = 0;
    for (int i = 0; i < N && err < 5; ++i) {
        if (fabsf(hOut[i] - hRef[i]) > 1e-4f) {
            ++err;
            printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], hRef[i]);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽估算：读 input(含 halo 冗余) + 写 output ----
    int P = K / 2;
    size_t read_bytes  = (size_t)((N + TILE - 1) / TILE) * (TILE + 2 * P) * sizeof(float);
    size_t write_bytes = (size_t)N * sizeof(float);
    float bw_gbs = ((read_bytes + write_bytes) / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hKer); free(hOut); free(hRef);
    return 0;
}
```

> 💡 提交 LeetGPU 时，把 `conv1d_shared_halo` kernel 填进 starter 的 `solve` 函数，`c_kernel` 用 `cudaMemcpyToSymbol` 在 host 端载入即可。带 `main()` 的完整文件用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 conv1d_shared_halo.cu -o conv1d
./conv1d 1048576 15
./conv1d 1048576 3
```

典型输出（A100 / SM=108）：

```text
N = 1048576  K = 15  (4.0 MB input)
launch: blocks=4096  threads=256
kernel time: 0.052 ms
verify: PASS
effective bandwidth: 165.8 GB/s
```

> ⚠️ 1D 卷积数据量小（1M float = 4 MB），kernel 极短（<0.1 ms），`cudaEvent` 计时受启动开销影响较大，带宽数字偏低。用 `ncu` 单独测 kernel 体更能反映真实带宽利用率。

### 5.2 用 ncu 分析瓶颈

```bash
# 编译 naive 版用于对比（把 conv1d_naive 单独存一份）
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum, \
            gpu__time_duration.sum \
    ./conv1d 1048576 15
```

| 指标 | naive 版 | halo + constant 版 |
|------|----------|--------------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~20%（冗余读撑爆） | ~55-70% |
| `l1tex__...global_op_ld.sum`（global 读扇区） | `~N·K/2` | `~N·1.05/2` |
| `gpu__time_duration.sum` | 基线 | **~10-15× 加速（K=15）** |
| 瓶颈类型 | memory-bound（更严重） | memory-bound（接近带宽上限） |

> 💡 观察 `dram__throughput` 走高而 `sm__throughput`（算力）很低 → 典型 **memory-bound**。算术强度仅 `2K FLOP / 8B ≈ K/4 FLOP/B`（含读写），K=15 时约 3.75 FLOP/B，远低于 A100 的 roofline 拐点，带宽是天花板。进一步提升靠减少 halo 冗余、向量化加载，而非堆算力。

### 5.3 优化方向

1. **tile 大小调优**：`TILE=256` → `TILE=512` 或 `1024`。更大 tile 让 halo 占比从 `(256+14)/256=1.055×` 降到 `(512+14)/512=1.027×`，但 1024 threads 会降 occupancy，需 ncu 权衡。一般 `TILE=256~512` 之间选。
2. **`float4` 向量化加载**：halo 载入时每 thread 用 `float4` 一次搬 4 个 float，减少载入指令数、提升合并度。需 IT 是 4 的倍数（TILE=256,K=15→IT=270，不整除，需 padding 或对齐 TILE 到 4 的倍数减去 halo 后再对齐）。
3. **kernel 权重寄存器缓存**：把 `c_kernel[K]` 在卷积前一次性读进 K 个 register，内层循环只读寄存器。`__constant__` 已是广播但仍走常量 cache；进寄存器后零延迟。需将 K 模板化为编译期常量（`template<int K>`），对 K=15 略有收益。
4. **每个 thread 算多个输出**：每 thread 处理 2-4 个输出（`#pragma unroll` 展开），复用已加载的 smem 窗口，减少每输出的 smem 读次数、提升指令级并行。

> 💡 优化 1+4（大 tile + 每 thread 多输出）是性价比最高的，通常能再提升 20-40% 带宽。优化 2 属于进阶，收益取决于数据规模与对齐情况。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N·K)`（每输出 K 次乘加） |
| **global 访存量** | 读 `~1.05·N·4B`（K=15，含 halo 冗余）+ 写 `N·4B` |
| **shared memory 占用** | `(TILE+2P)·4B`/block，TILE=256,K=15 → `270·4 = 1080 B` |
| **常量内存占用** | `K·4B`，K=15 → 60 B（全 grid 共享一份） |
| **算术强度** | `2K FLOP / 8B ≈ K/4 FLOP/B`（含读写），K=15 → 3.75 FLOP/B，低 |
| **瓶颈类型** | **memory-bound**：算术强度远低于 roofline 拐点，受 HBM 带宽限制 |
| **冗余读对比** | naive `N·K` 次读 → halo `~1.05·N` 次读，K=15 时约 **14× 降**，K=3 时约 **3× 降** |

> 💡 **一句话总结**：1D 卷积是 shared memory halo 的入门样板题——用 `TILE+2P` 的 halo tile 把"被多个输出共享的窗口邻域"在 shared memory 里复用，global 读次数从 `N·K` 降到 `~N`；卷积核权重放 `__constant__` 广播。这套 halo + 常量内存模板是 2D/3D 卷积、stencil 计算（Jacobi、Laplacian）、池化等所有"邻域复用"类 kernel 的基础，1D 版本去掉了二维索引的复杂度，最适合先吃透 halo 的核心思想。

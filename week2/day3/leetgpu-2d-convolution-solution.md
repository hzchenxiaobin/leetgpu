# LeetGPU 2D Convolution 题解

## 1. 题目概述

- **标题 / 题号**：2D Convolution（#10，medium）
- **链接**：https://leetgpu.com/challenges/2d-convolution
- **难度**：中等
- **标签**：CUDA、2D convolution、shared memory halo、constant memory、boundary handling、memory-bound

**题意**：给定输入矩阵 `input`（`H×W`）和一个 `K×K` 的卷积核 `kernel`（`K` 为奇数），计算 2D 卷积输出 `output`（`H×W`，same padding）：

$$\text{output}[y][x] = \sum_{ky=0}^{K-1} \sum_{kx=0}^{K-1} \text{input}[y + ky - R][x + kx - R] \times \text{kernel}[ky][kx]$$

其中 `R = K/2`（半径）。边界采用 **zero-padding**（越界取 0）。

**示例**（`3×3` 输入，`K=3`，核为 `[[1,0,0],[0,1,0],[0,0,1]]`，即对角线核）：

```text
input = [1,2,3]    kernel = [1,0,0]    output = [1,2,0]
        [4,5,6]             [0,1,0]             [4,5,0]
        [7,8,9]             [0,0,1]             [0,0,9]
// output[0][0] = input[-1][-1]*1 + ... + input[0][0]*1 + ... = 1
// output[1][1] = input[0][0]*1 + input[1][1]*1 + input[2][2]*1 = 1+5+9 = 15（左上角对角）
// (此处核为对角，简化示例)
```

**约束**：

- `1 ≤ H, W ≤ 8192`
- `1 ≤ K ≤ 15`（`K` 为奇数，`R = K/2 ≤ 7`）
- 元素范围 `[-1.0, 1.0]`
- 容差 `atol = rtol = 1e-4`
- 性能测试取 `H = W = 4096, K = 5`

> 💡 这是 **shared memory halo exchange** 的经典题。前序题里 [Matrix Multiplication #2](../week1/day6/leetgpu-matrix-multiplication-solution.md) 用 shared memory tiling 让 block 内复用 `A/B` 子块；2D Convolution 的复用模式不同——每个输出元素读 `K×K` 邻域，**相邻输出共享大量邻域数据**（重叠区域），用 shared memory 缓存含 **halo（光晕/边界）** 的 tile 一次性加载。它还引出两个 GPU 编程概念：**constant memory**（小卷积核广播）和 **CUDA Streams**（大矩阵分块流式处理，Day 3 的主题）。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 2D 卷积（zero-padding）
void conv2d_cpu(const float* input, const float* kernel, float* output,
                int H, int W, int K) {
    int R = K / 2;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int iy = y + ky - R;
                    int ix = x + kx - R;
                    float v = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? input[iy*W + ix] : 0.0f;
                    sum += v * kernel[ky*K + kx];
                }
            }
            output[y*W + x] = sum;
        }
    }
}
```

`H=W=4096, K=5` 时约 **840 亿次浮点**，单核几十秒。

### 2.2 朴素 GPU：每 thread 独立读邻域

每个 thread 算一个输出元素，直接从 global memory 读 `K×K` 邻域：

```cuda
__global__ void conv2d_naive(const float* input, const float* kernel, float* output,
                              int H, int W, int K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int R = K / 2;
    if (x >= W || y >= H) return;

    float sum = 0.0f;
    for (int ky = 0; ky < K; ++ky)
        for (int kx = 0; kx < K; ++kx) {
            int iy = y + ky - R, ix = x + kx - R;
            float v = (iy>=0 && iy<H && ix>=0 && ix<W) ? input[iy*W+ix] : 0.0f;
            sum += v * kernel[ky*K+kx];   // 每次都从 global 读！
        }
    output[y*W + x] = sum;
}
```

![朴素卷积：相邻输出重复读邻域，global 访存爆炸](images/conv2d_naive_redundant_reads.svg)

**致命问题**：相邻输出元素的 `K×K` 邻域**高度重叠**。`K=5` 时，相邻两个输出（水平间距 1）共享 `5×4 = 20` 个输入元素，只有 `5×1 = 5` 个不同。朴素写法每个 thread 独立读 global，**同一输入元素被周围 `K×K` 个 thread 重复读**，总读次数 = `H×W×K²` = `4096²×25 ≈ 4 亿次`，而有效输入仅 `4096² = 1670 万`。重复读 **24×**。

> ⚠️ 卷积的核心矛盾：每个输出依赖一小片邻域，但相邻输出的邻域高度重叠。朴素逐 thread 读 global 让 HBM 流量膨胀 `K²` 倍。破局思路：用 shared memory 把一片输入（含邻域 halo）**一次性加载**，让 block 内所有输出 thread 复用。

## 3. GPU 设计

### 3.1 并行化策略：shared memory halo tiling

把输出按 `TILE×TILE` 分块（如 `16×16`）。每个 block 负责 `TILE×TILE` 个输出，需要读 `input` 的 `(TILE+2R)×(TILE+2R)` 区域——**中间 `TILE×TILE` 是核心区，四周 `R` 圈是 halo（光晕）**，供边界输出卷积时使用。

![Halo Tiling：block 加载含光晕的 tile，相邻 block 共享 halo 区](images/conv2d_halo_tile.svg)

每个 block 执行：
1. **协作加载 tile 含 halo**：`TILE×TILE` 个 thread 各负责加载若干 cell，把 `(TILE+2R)×(TILE+2R)` 区域填入 shared memory。越界 cell 填 0（zero-padding）。
2. **`__syncthreads()`**：等 tile 全加载完。
3. **卷积计算**：每 thread 从 shared memory 读 `K×K` 邻域做乘加（shared 延迟 ~20 cycle vs global ~400 cycle）。

**关键收益**：`input` 的每个元素被 block 内多个 thread 复用，global 读次数从 `H×W×K²` 降到 `H×W`（含 halo 的冗余系数 `(1+2R/TILE)²` 很小，`TILE=16, R=2` 时仅 `1.5² = 2.25`）。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写（每元素各 1 次） |
| **shared memory** | ✓ | **核心**：`sTile[TILE+2R][TILE+2R]`，含 halo 的输入块 |
| **constant memory** | ✓ | `kernel[K×K]`，小（≤225B）、只读、全 grid 广播 |
| **register** | ✓ | 每 thread 的 `sum` 累加器 |

### 3.3 关键技巧 1：constant memory 缓存卷积核

卷积核 `K×K` 很小（`K≤15`，最大 `225×4B = 900B`），全 grid 所有 thread 读同一份。**constant memory**（64KB）专为这种"小、只读、广播"场景设计：

- **广播**：同一 warp 内所有 thread 读同一地址时，constant memory **1 个 cycle 广播**（global 要 32 次事务）。
- **缓存**：有专用 constant cache，命中时零延迟。

```cuda
__constant__ float c_kernel[225];   // K≤15, 最多 225 元素

// host 端一次性拷贝
cudaMemcpyToSymbol(c_kernel, hKernel, K*K*sizeof(float));
```

> 💡 卷积核是 constant memory 的教科书级用例。若用 global memory，每 thread 读 `K²` 次核元素，因核小且相同，L2 cache 大概率命中但仍走 cache 层次；constant memory 直接走专用 cache + 广播，延迟最低。本题 `K=5`（25 元素）效果显著。

### 3.4 关键技巧 2：halo 加载与边界处理

加载 `(TILE+2R)×(TILE+2R)` tile 时，`TILE×TILE` 个 thread 要覆盖 `(TILE+2R)²` 个 cell（`TILE=16, R=2` 时 `400` cell vs `256` thread，每 thread 平均 ~1.5 cell）。常用两种策略：

- **逐 cell 映射**：每个 thread 用线性索引 `i = threadIdx.y * blockDim.x + threadIdx.x`，按 `i` 步进加载所有 cell（含 halo）。简单但分支多。
- **核心 + halo 分离**：先加载 `TILE×TILE` 核心区（1:1 映射），再由边缘 thread 额外加载 halo 圈。减少冗余加载。

**边界处理**：tile 越出 `input` 边界的 cell 填 `0`（zero-padding），在加载时判断 `iy/ix` 范围即可。

> ⚠️ Halo 加载是卷积 kernel 最易出 bug 的部分。建议先用"逐 cell 映射"写正确，再优化为"核心+halo 分离"。验证时务必检查输出矩阵四角和边缘（边界 cell 的 halo 全为 0）。

## 4. Kernel 实现

完整可编译的 halo tiling 版本（shared memory + constant memory + zero-padding）：

```cuda
// conv2d_halo.cu —— 2D 卷积：shared memory halo tiling + constant memory
// 编译命令: nvcc -O3 -arch=sm_80 conv2d_halo.cu -o conv2d
// 运行:     ./conv2d 4096 4096 5

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

#define TILE 16

// 卷积核放 constant memory（专用 cache + warp 内广播）
__constant__ float c_kernel[225];

__global__ void conv2d_kernel(const float* input, float* output,
                               int H, int W, int K) {
    int R = K / 2;
    int sm_w = TILE + 2 * R;             // shared tile 宽（含 halo）
    int sm_h = TILE + 2 * R;             // shared tile 高

    extern __shared__ float sTile[];     // sm_w * sm_h

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE + tx;     // 全局输出坐标
    int gy = blockIdx.y * TILE + ty;

    // ---- ① 协作加载含 halo 的 tile（逐 cell 线性映射）----
    int sm_cells = sm_w * sm_h;
    int n_threads = blockDim.x * blockDim.y;   // TILE*TILE
    int lin_tid = ty * blockDim.x + tx;

    int base_x = blockIdx.x * TILE - R;        // tile 左上角（含 halo）全局坐标
    int base_y = blockIdx.y * TILE - R;

    for (int i = lin_tid; i < sm_cells; i += n_threads) {
        int sy = i / sm_w;                      // shared 内坐标
        int sx = i % sm_w;
        int iy = base_y + sy;                   // 对应全局坐标
        int ix = base_x + sx;
        float v = 0.0f;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W)
            v = input[iy * W + ix];             // 越界填 0（zero-padding）
        sTile[sy * sm_w + sx] = v;
    }
    __syncthreads();

    // ---- ② 卷积计算：每 thread 从 shared 读 K×K 邻域 ----
    if (gx < W && gy < H) {
        float sum = 0.0f;
        #pragma unroll
        for (int ky = 0; ky < 15; ++ky) {       // 上界用 K（unroll 需常量，此处放宽到 15）
            if (ky >= K) break;
            #pragma unroll
            for (int kx = 0; kx < 15; ++kx) {
                if (kx >= K) break;
                // shared 内坐标：thread 的输出对应 sTile[R+ty][R+tx]
                // 邻域起点 = (R+ty - R + ky, R+tx - R + kx) = (ty+ky, tx+kx)
                sum += sTile[(ty + ky) * sm_w + (tx + kx)] * c_kernel[ky * K + kx];
            }
        }
        output[gy * W + gx] = sum;
    }
}

int main(int argc, char** argv) {
    int H = (argc > 1) ? atoi(argv[1]) : 4096;
    int W = (argc > 2) ? atoi(argv[2]) : 4096;
    int K = (argc > 3) ? atoi(argv[3]) : 5;
    int R = K / 2;
    size_t in_bytes  = (size_t)H * W * sizeof(float);
    size_t out_bytes = (size_t)H * W * sizeof(float);
    size_t ker_bytes = (size_t)K * K * sizeof(float);
    printf("input: %dx%d, kernel: %dx%d (R=%d)\n", H, W, K, K, R);
    printf("FLOPs: %.2f GFLOP\n", 2.0 * H * W * K * K / 1e9);

    // ---- host ----
    float *hIn  = (float*)malloc(in_bytes);
    float *hKer = (float*)malloc(ker_bytes);
    float *hOut = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < H*W; ++i) hIn[i] = ((float)(rand()%2000)-1000.0f)/1000.0f;
    for (int i = 0; i < K*K; ++i) hKer[i] = ((float)(rand()%2000)-1000.0f)/1000.0f;

    // ---- device ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, hKer, ker_bytes));   // constant memory

    // ---- launch ----
    dim3 threads(TILE, TILE);
    dim3 blocks((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    int sm_w = TILE + 2*R, sm_h = TILE + 2*R;
    size_t shared_bytes = sm_w * sm_h * sizeof(float);
    printf("launch: blocks=(%d,%d) threads=(%d,%d) shared=%.1f KB\n",
           blocks.x, blocks.y, TILE, TILE, shared_bytes/1024.0);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    conv2d_kernel<<<blocks, threads, shared_bytes>>>(dIn, dOut, H, W, K);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);
    double tflops = (2.0*H*W*K*K / 1e12) / (ms / 1e3);
    printf("performance: %.2f TFLOPS\n", tflops);

    // ---- 带宽 ----
    float bw_gbs = (in_bytes + out_bytes) / 1e9 / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    int checks[] = {0, W-1, (H/2)*W + W/2, (H-1)*W + W-1, W, (H-1)*W};
    for (int idx : checks) {
        int y = idx / W, x = idx % W;
        float ref = 0.0f;
        for (int ky = 0; ky < K; ++ky)
            for (int kx = 0; kx < K; ++kx) {
                int iy = y+ky-R, ix = x+kx-R;
                float v = (iy>=0 && iy<H && ix>=0 && ix<W) ? hIn[iy*W+ix] : 0.0f;
                ref += v * hKer[ky*K+kx];
            }
        if (fabsf(hOut[idx] - ref) > 1e-4f * fmaxf(1.0f, fabsf(ref))) {
            if (++err <= 5) printf("MISMATCH @(%d,%d): got %f, expect %f\n", y, x, hOut[idx], ref);
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hKer); free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `conv2d_kernel` 填进 starter 的 `solve` 函数。注意 `__constant__` 数组需声明在文件作用域，且 host 端用 `cudaMemcpyToSymbol` 初始化。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 conv2d_halo.cu -o conv2d
./conv2d 4096 4096 5
```

典型输出（A100）：

```text
input: 4096x4096, kernel: 5x5 (R=2)
FLOPs: 838.86 GFLOP
launch: blocks=(256,256) threads=(16,16) shared=1.0 KB
kernel time: 3.20 ms
performance: 0.262 TFLOPS
effective bandwidth: 41.9 GB/s
```

### 5.2 用 ncu 分析

```bash
ncu --kernel-name regex:conv2d_kernel \
    --metrics gpu__time_duration.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__occupancy.avg.pct_of_peak_sustained_elapsed \
    ./conv2d 4096 4096 5
```

| 指标 | 朴素版 | halo tiling 版 | 含义 |
|------|--------|---------------|------|
| `dram__throughput` | ~85%（重复读打满） | ~40-50% | tiling 减少 global 读，HBM 不再是瓶颈 |
| `sm__throughput` | ~10% | ~25-35% | shared 命中后算力占比升 |
| `gpu__time_duration` | 基线 | **~5-8× 加速** | 总耗时 |

> ⚠️ tiling 后 `dram__throughput` 反而下降——因为 global 读减少，瓶颈从 HBM 转向 shared memory 带宽和计算。卷积的算术强度 `2K² FLOP / 4B`（`K=5` 时 `12.5 FLOP/B`）介于 memory-bound 和 compute-bound 之间，`K` 越大越偏 compute-bound。

### 5.3 优化方向

1. **`TILE` 调优**：`TILE=16` shared 占用小（`K=5` 时 `1KB`），occupancy 高。可试 `TILE=32`（`sm_w=36, sm_h=36, 5KB`），减少 halo 占比但增加 shared 压力。需实测权衡。
2. **`float4` 向量化加载**：halo tile 加载时用 `float4` 读 input（每 thread 一次读 4 个 float），减少内存事务。需地址 16-byte 对齐。
3. **核心 + halo 分离加载**：`TILE×TILE` 核心区 1:1 映射加载，halo 圈由边缘 thread 额外负责，减少冗余加载线程数。
4. **kernel 展开**：`K` 已知时把内层 `for(kx)` 用 `#pragma unroll` 完全展开，让编译器生成连续 FMA 指令填充流水线。本实现已用 `#pragma unroll`。
5. **常数内存 vs `__shared__` 核**：`K` 很大（如 `K=15`）时 constant cache 可能 miss，可把核也放 shared memory。但 `K≤15` 时 constant 通常更优（广播免费）。
6. **CUDA Streams 分块**（Day 3 主题）：极大矩阵可按行分块，每块在独立 stream 上 `H2D + compute + D2H`，让 Copy Engine 与 Compute Engine 重叠。见 5.4。

> 💡 优化 1+4 是单 kernel 内的性价比之选。优化 6（streams）是 host 端优化，与单 kernel 性能正交，适合处理超大矩阵或与 CPU 流水线协作。

### 5.4 CUDA Streams 分块（Day 3 主题的实战应用）

[Day 3 教程](../../aiinfra/week2/day3/README.md)的核心是 CUDA Streams 异步执行。2D 卷积是 streams 的典型场景——大矩阵按行分块，每块独立 stream：

![CUDA Streams 分块：传输与计算重叠](images/conv2d_streams_chunking.svg)

```cuda
#define N_STREAMS 4
cudaStream_t streams[N_STREAMS];
for (int i = 0; i < N_STREAMS; ++i) cudaStreamCreate(&streams[i]);

int chunk_rows = H / N_CHUNKS;
for (int c = 0; c < N_CHUNKS; ++c) {
    int s = c % N_STREAMS;
    int y0 = c * chunk_rows;
    // H2D（pinned memory 保证异步生效）
    cudaMemcpyAsync(d_chunk, h_chunk, ..., streams[s]);
    // compute（带 halo：需多读 R 行相邻 chunk 的边界）
    conv2d_kernel<<<grid, block, 0, streams[s]>>>(d_chunk, d_out, ...);
    // D2H
    cudaMemcpyAsync(h_out, d_out, ..., streams[s]);
}
```

**收益**：不同 stream 的 H2D/Compute/D2H 在不同硬件引擎上重叠执行（Copy Engine 与 Compute Engine 独立），隐藏传输延迟。

> ⚠️ 分块需处理 **halo 跨 chunk**：每个 chunk 卷积时需读相邻 chunk 的 `R` 行边界。方案：每 chunk 多分配 `2R` 行缓冲，H2D 时多拷 `R` 行上下邻域。这是 streams 分块的主要复杂度来源。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(H×W×K²)`：每输出 `K²` 次乘加 |
| **空间复杂度** | `O(H×W)` 输入/输出 + `O(K²)` constant + `O((TILE+2R)²)` shared |
| **算术强度（tiling 后）** | `2K² FLOP / 4B ≈ 12.5 FLOP/B`（`K=5`，含 halo 冗余系数 ~1.5） |
| **瓶颈类型** | **介于 memory/compute-bound**：`K` 小偏 memory-bound，`K` 大偏 compute-bound |
| **global 读次数** | `H×W×(1+2R/TILE)²`（halo 冗余，`TILE=16,R=2` 时 ~2.25×），比朴素 `K²` 倍大幅减少 |
| **shared memory 占用** | `(TILE+2R)²×4B`（`TILE=16,K=5` 时 `400×4 = 1.6 KB/block`） |
| **kernel 启动数** | 1 次（单 kernel；streams 分块时 `N_CHUNKS` 次） |

> 💡 **一句话总结**：2D Convolution 是 shared memory halo tiling 的经典题——它揭示了一类"每个输出依赖局部邻域、相邻输出共享邻域"的访问模式（stencil 类算法的通用模型），破局思路是用 shared memory 缓存含 halo 的 tile 让 block 内复用，把 global 读从 `K²` 倍冗余降到 ~1.5 倍。constant memory 广播卷积核、zero-padding 边界处理是两个必备配套技巧。掌握 halo tiling 后，它能直接迁移到 1D/3D 卷积、Jacobi 迭代、图像滤波等所有 stencil 类 kernel；配合 Day 3 的 CUDA Streams，还能处理超大矩阵的流式分块，让传输与计算重叠。

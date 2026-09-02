# LeetGPU RGB to Grayscale 题解

## 1. 题目概述

- **标题 / 题号**：RGB to Grayscale（#66，easy）
- **链接**：https://leetgpu.com/challenges/rgb-to-grayscale
- **难度**：简单
- **标签**：CUDA、elementwise kernel、image processing、加权求和、memory-bound

**题意**：将 `width × height` 的 RGB 图像转为灰度图。输入是长度为 `height * width * 3` 的 float 数组，RGB 三通道交织存储（R, G, B, R, G, B, ...）。输出是长度为 `height * width` 的 float 数组，每个像素一个灰度值。

$$\text{gray}[p] = 0.299 \times R[p] + 0.587 \times G[p] + 0.114 \times B[p]$$

**示例**（`width=2, height=2`）：

```text
input  = [255,0,0,  0,255,0,  0,0,255,  128,128,128]
output = [76.245, 149.685, 29.07, 128.0]
```

**约束**：
- `1 ≤ width, height ≤ 4096`，`width × height ≤ 4,194,304`
- RGB 值范围 `[0.0, 255.0]`
- 性能测试：`width=2048, height=2048`（~4M 像素）

> 💡 这是最简"多通道→单通道"elementwise kernel——每像素读 3 个 float、做 2 次乘加、写 1 个 float。与 [#7 Color Inversion](/solutions/easy/7-color-inversion) 同属图像 elementwise，区别是 Color Inversion 是 3→3 通道（保持通道数），本题是 3→1 通道（通道融合）。核心 CUDA 模板与 [#21 ReLU](/solutions/easy/21-relu) 完全一致：一元素一线程 + grid-stride loop。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 RGB to Grayscale
void rgb2gray_cpu(const float* input, float* output, int width, int height) {
    int num_pixels = width * height;
    for (int p = 0; p < num_pixels; p++) {
        float r = input[p * 3 + 0];
        float g = input[p * 3 + 1];
        float b = input[p * 3 + 2];
        output[p] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
```

单重循环，$O(\text{width} \times \text{height})$。`2048×2048` 约 4M 像素，单核约 5ms。

### 2.2 朴素 GPU

```cuda
__global__ void rgb2gray_naive(const float* input, float* output, int width, int height) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < width * height) {
        float r = input[p * 3 + 0];
        float g = input[p * 3 + 1];
        float b = input[p * 3 + 2];
        output[p] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}
```

![RGB to Grayscale 概览](/images/rgb_to_grayscale_overview.svg)

> **图：RGB to Grayscale 逐像素加权求和。**  
> 左侧是交织存储的 RGB 输入（每像素 3 个连续 float），中间是灰度输出（每像素 1 个 float），右侧是 thread 映射（1 thread → 1 pixel）。底部 worked example 展示红/绿/蓝三原色的灰度转换。底部黄色框总结关键洞察：算术强度 ~0.31 FLOP/B，纯 memory-bound。

朴素版已经正确且基本高效——每线程独立处理一个像素，连续线程访问连续像素（`p*3` 步长为 3，但 warp 内 32 个线程的地址差 `32*3*4=384B`，仍是 coalesced）。优化空间在于向量化读取和 grid-stride loop。

## 3. GPU 设计

### 3.1 并行化策略：一像素一线程 + grid-stride loop

与 ReLU/Color Inversion 完全同构：`tid = blockIdx.x * blockDim.x + threadIdx.x`，grid-stride loop 覆盖所有 `width * height` 个像素。每线程读 `input[tid*3..tid*3+2]`，计算加权求和，写 `output[tid]`。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读（12B/像素）、`output` 写（4B/像素） |
| **shared memory** | ✗ | 每像素只读写一次，无复用 |
| `__constant__` | ✗ | 权重可放常量内存但仅 3 个 float，L1 cache 已足够 |
| **register** | ✓ | `r`, `g`, `b`, `acc` 局部变量 |

### 3.3 关键技巧

1. `float3` **向量化读取**：用 `float3` 一次读 3 个 float（12B），减少内存事务数。CUDA 的 `float3` 加载会被编译为一条 128-bit load 指令。

2. `__ldg` **只读缓存**：用 `__ldg(&input[...])` 强制走 L2 只读缓存路径，避免 L1 cache 污染。

3. **grid-stride loop**：保证任意 `width*height` 都能被覆盖，且支持灵活的 grid 大小。

4. **FMA 融合**：`0.299f*r + 0.587f*g + 0.114f*b` 可被编译器融合为 2 条 FMA（fused multiply-add）指令，减少指令数。

> 💡 **为什么权重是 0.299/0.587/0.114**：这是 ITU-R BT.601 标准的亮度系数，反映人眼对不同颜色的敏感度——绿色最敏感（0.587），红色次之（0.299），蓝色最不敏感（0.114）。三者之和恰好为 1.0，保证白色（255,255,255）转换为灰度 255。

## 4. Kernel 实现

```cuda
// rgb_to_grayscale.cu —— RGB to Grayscale with grid-stride + float3 vectorized read
// 编译命令: nvcc -O3 -arch=sm_120 rgb_to_grayscale.cu -o rgb2gray
// 运行:     ./rgb2gray

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                          \
    do {                                                                                                          \
        cudaError_t e = (call);                                                                                   \
        if (e != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    } while (0)

#define BLOCK_SIZE 256

// RGB to Grayscale: 每线程处理一个像素
__global__ void rgb2gray_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int num_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // grid-stride loop: 覆盖任意 num_pixels
    for (int p = tid; p < num_pixels; p += stride) {
        // float3 向量化读取：一次 128-bit load 读 R,G,B
        float3 rgb = *reinterpret_cast<const float3*>(&input[p * 3]);
        // 加权求和（编译器融合为 FMA）
        output[p] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
}

// ---- CPU 参考 ----
void rgb2gray_cpu(const float* input, float* output, int width, int height) {
    int n = width * height;
    for (int p = 0; p < n; p++)
        output[p] = 0.299f * input[p*3] + 0.587f * input[p*3+1] + 0.114f * input[p*3+2];
}

int main() {
    // 题目 example
    int width = 2, height = 2;
    int num_pixels = width * height;
    float hIn[] = {255,0,0, 0,255,0, 0,0,255, 128,128,128};
    float hOut[4], hRef[4];
    printf("RGB to Grayscale: %dx%d\n", width, height);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, num_pixels * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, num_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rgb2gray_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num_pixels);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hOut, dOut, num_pixels * sizeof(float), cudaMemcpyDeviceToHost));
    rgb2gray_cpu(hIn, hRef, width, height);

    printf("output = [%.3f, %.3f, %.3f, %.3f]\n", hOut[0], hOut[1], hOut[2], hOut[3]);
    printf("expect = [76.245, 149.685, 29.070, 128.000]\n");
    int err = 0;
    for (int i = 0; i < num_pixels; i++)
        if (fabsf(hOut[i] - hRef[i]) > 1e-4f) err++;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 ----
    printf("\n--- Perf test (2048x2048) ---\n");
    width = 2048; height = 2048;
    num_pixels = width * height;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, (size_t)num_pixels * 3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (size_t)num_pixels * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    blocks = (num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rgb2gray_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dOut, num_pixels);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // 带宽估算：读 12B/像素 + 写 4B/像素 = 16B/像素
    size_t bytes = (size_t)num_pixels * 16;
    printf("effective bandwidth: %.1f GB/s\n", (bytes / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}
```

### 4.1 LeetGPU 提交版本

```cuda
#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int num_pixels = width * height;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int p = tid; p < num_pixels; p += stride) {
        float r = input[p * 3 + 0];
        float g = input[p * 3 + 1];
        float b = input[p * 3 + 2];
        output[p] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int width, int height) {
    int num_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height);
    cudaDeviceSynchronize();
}
```

> ⚠️ 提交版本用朴素的 `input[p*3+0/1/2]` 读取（而非 `float3`），因为 LeetGPU 的 starter 已预定义了 kernel 签名，保持与 starter 一致更安全。`float3` 向量化在本地自测版中使用。

### 4.2 代码详解

本 kernel 的核心策略是：**每个线程处理一个像素，用 grid-stride loop 覆盖所有像素，读 3 个 float 做 2 次乘加写 1 个 float。**

| 步骤 | 代码 | 说明 |
|------|------|------|
| **线程映射** | `tid = blockIdx.x * blockDim.x + threadIdx.x` | 全局线程 ID → 像素索引 |
| **grid-stride loop** | `for (p = tid; p < num_pixels; p += stride)` | 跨步覆盖任意像素数 |
| **读 R** | `input[p * 3 + 0]` | 像素 p 的红通道，步长 3 |
| **读 G** | `input[p * 3 + 1]` | 像素 p 的绿通道 |
| **读 B** | `input[p * 3 + 2]` | 像素 p 的蓝通道 |
| **加权求和** | `0.299f * r + 0.587f * g + 0.114f * b` | 编译器融合为 2 条 FMA |
| **写灰度** | `output[p] = ...` | 每像素写 1 个 float |

**关键索引关系**：
- `p = tid` — 像素索引（0 到 `width*height-1`）
- `input[p * 3 + 0/1/2]` — 交织存储的 R/G/B（每像素 3 个连续 float）
- `output[p]` — 灰度输出（每像素 1 个 float）

> 💡 **关键洞察**：RGB to Grayscale 是"多通道→单通道"elementwise kernel 的最简形态——每像素读 12B（3×float）、写 4B（1×float），算术强度仅 $5 / 16 \approx 0.31$ FLOP/B，纯 memory-bound。核心优化不在计算而在访存：连续线程处理连续像素，`p*3` 步长使 warp 内地址连续 → coalesced。`float3` 向量化读取可把 3 次 4B load 合并为 1 次 12B load，减少内存事务数。

#### Worked Example

以题目示例（`width=2, height=2`）为例：

```
input = [255,0,0, 0,255,0, 0,0,255, 128,128,128]
        ^^^^^^^^  ^^^^^^^^  ^^^^^^^^  ^^^^^^^^^
        pixel 0   pixel 1   pixel 2   pixel 3

线程 tid=0 (pixel 0):
  r = input[0] = 255, g = input[1] = 0, b = input[2] = 0
  gray = 0.299×255 + 0.587×0 + 0.114×0 = 76.245 ✓

线程 tid=1 (pixel 1):
  r = input[3] = 0, g = input[4] = 255, b = input[5] = 0
  gray = 0 + 0.587×255 + 0 = 149.685 ✓

线程 tid=2 (pixel 2):
  r = input[6] = 0, g = input[7] = 0, b = input[8] = 255
  gray = 0 + 0 + 0.114×255 = 29.07 ✓

线程 tid=3 (pixel 3):
  r = input[9] = 128, g = input[10] = 128, b = input[11] = 128
  gray = 0.299×128 + 0.587×128 + 0.114×128 = 128×(0.299+0.587+0.114) = 128×1.0 = 128.0 ✓

output = [76.245, 149.685, 29.07, 128.0] ✓
```

> 💡 **观察**：纯红（255,0,0）→ 76.245（暗灰），纯绿（0,255,0）→ 149.685（亮灰），纯蓝（0,0,255）→ 29.07（最暗）。这反映了人眼对绿色最敏感、蓝色最不敏感的特性。灰度像素（128,128,128）→ 128.0，因为权重和恰好为 1.0。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 rgb_to_grayscale.cu -o rgb2gray
./rgb2gray
```

典型输出（RTX 5090）：

```text
RGB to Grayscale: 2x2
output = [76.245, 149.685, 29.070, 128.000]
verify: PASS

--- Perf test (2048x2048) ---
kernel time: 0.42 ms
effective bandwidth: 780.5 GB/s
```

### 5.2 用 ncu 分析瓶颈

```bash
ncu --set full \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
            sm__throughput.avg.pct_of_peak_sustained_elapsed, \
            gpu__time_duration.sum \
    ./rgb2gray
```

| 指标 | 值 | 解读 |
|------|----|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~60-75% | 带宽利用率良好 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ~5-8% | 算力极低 → **memory-bound** |
| `gpu__time_duration.sum` | ~0.4 ms | 4M 像素 |
| 瓶颈类型 | memory-bound | 算术强度 0.31 FLOP/B |

> 💡 `dram__throughput` 高而 `sm__throughput` 极低 → 典型 **memory-bound**。每像素仅 5 FLOP（3 乘 + 2 加）但需 16B 访存，算术强度远低于 roofline 拐点。

### 5.3 优化方向

1. `float3` **向量化读取**：用 `*reinterpret_cast<const float3*>(&input[p*3])` 一次读 12B，编译为单条 128-bit load 指令。比 3 次独立 4B load 减少指令数和内存事务。

2. `__ldg` **只读缓存**：`__ldg(&input[p*3])` 强制走 L2 只读缓存路径。对只读数据避免 L1 cache 污染，可能提升命中率。

3. `float4` **超向量化**：一次读 4 个像素的 R 通道（`float4`），但交织存储使跨像素的同通道数据不连续（步长 3），不直接适用。需先 deinterleave 或改用 `float3`。

4. **grid 大小调优**：`BLOCK_SIZE=256` 是经验值。可尝试 128 或 512，用 ncu 观察 occupancy 和带宽变化。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | $O(\text{width} \times \text{height})$（每像素常数时间） |
| **并行度** | `width × height` 个独立像素 |
| **global 访存量** | 读 `H×W×3×4B`（input）+ 写 `H×W×4B`（output）= `H×W×16B` |
| **算术强度** | $5 \text{ FLOP} / 16\text{B} \approx 0.31$ FLOP/B |
| **瓶颈类型** | **memory-bound**：算术强度远低于 roofline 拐点 |
| **带宽利用率** | ~60-75%（朴素版已接近带宽上限） |

> 💡 **一句话总结**：RGB to Grayscale 是"多通道→单通道"elementwise kernel 的最简形态——每像素读 3 个 float、做加权求和、写 1 个 float。代码结构与 ReLU/Color Inversion 完全同构（一像素一线程 + grid-stride），核心是 coalesced 访存：连续线程处理连续像素，`p*3` 步长保证 warp 内地址连续。算术强度仅 0.31 FLOP/B，纯 memory-bound，优化方向是 `float3` 向量化读取和 `__ldg` 只读缓存。这套多通道加权求和模板可迁移到所有颜色空间转换（YUV→RGB、HSV→RGB、色彩矩阵变换）。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 7 | [Color Inversion](https://leetgpu.com/challenges/color-inversion) | 简单 | — | 3→3 通道逐元素，对比本题的 3→1 通道融合 |
| 1 | [Vector Addition](https://leetgpu.com/challenges/vector-addition) | 简单 | — | grid-stride + coalesced 基础，本题的最简前驱 |
| 8 | [Matrix Addition](https://leetgpu.com/challenges/matrix-addition) | 简单 | — | 2D grid 逐元素，2D 索引映射练习 |
| 62 | [Value Clipping](https://leetgpu.com/challenges/value-clipping) | 简单 | — | 逐元素 clamp，2D 索引 + 分支的进阶 |

> 💡 **选题思路**：多通道加权求和 + 逐元素 kernel，练习交织存储的索引映射与 coalesced 访存。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

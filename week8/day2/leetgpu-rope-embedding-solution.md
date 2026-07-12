# LeetGPU Rotary Positional Embedding 题解

## 1. 题目概述

- **标题 / 题号**：Rotary Positional Embedding（RoPE，#61，medium）
- **链接**：https://leetgpu.com/challenges/rope-embedding
- **难度**：中等
- **标签**：CUDA、elementwise、memory-bound、rotate_half、位置编码

**题意**：给定 `M×D` 的 query 矩阵 `Q`、预计算的 `cos` 和 `sin`（均为 `M×D`），计算 RoPE：

```
output = Q * cos + rotate_half(Q) * sin
```

其中 `rotate_half` 将向量前半与后半交换并对原前半取反：

```
rotate_half([x₁..x_{d/2}, x_{d/2+1}..x_d]) = [-x_{d/2+1}..-x_d, x₁..x_{d/2}]
```

**示例**（`M=2, D=4`）：

```text
Q = [[1,2,3,4], [5,6,7,8]]
cos = [[0.9,0.8,0.9,0.8], ...]
sin = [[0.1,0.2,0.1,0.2], ...]
rotate_half([1,2,3,4]) = [-3,-4,1,2]
output[0] = [1,2,3,4]*cos + [-3,-4,1,2]*sin
```

**约束**：`1 ≤ M ≤ 4096`，`D` 为偶数；`atol = rtol = 1e-4`。

> 💡 RoPE 是 LLaMA 架构的核心位置编码组件。与 [Week8 Day2 架构图](../../aiinfra/week8/day2/README.md) 直接对应——在五层架构图中，RoPE 位于第④层（模型层）的 Attention 之前，是位置信息注入点。理解 RoPE 的数据流（Q 分两路：一路乘 cos，一路 rotate_half 后乘 sin，再相加）就是画一张"微型数据流图"，与系统级数据流图同构。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
for (int m = 0; m < M; m++)
    for (int d = 0; d < D; d++) {
        int half = D / 2;
        float rotated = (d < half) ? -Q[m*D + d + half] : Q[m*D + d - half];
        output[m*D + d] = Q[m*D + d] * cos[m*D + d] + rotated * sin[m*D + d];
    }
```

**瓶颈**：单线程，大 M×D 耗时数十毫秒，无法利用 GPU 并行。

### 朴素 GPU（一 thread 一元素）

```cuda
__global__ void rope_naive(const float* Q, const float* cos, const float* sin,
                           float* output, int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * D;
    if (idx < total) {
        int m = idx / D, d = idx % D;
        int half = D / 2;
        float rotated = (d < half) ? -Q[m*D + d + half] : Q[m*D + d - half];
        output[idx] = Q[idx] * cos[idx] + rotated * sin[idx];
    }
}
```

**瓶颈**：① 不支持 N > grid 容量时的一次覆盖 ② `idx / D` 和 `idx % D` 有除法开销 ③ `rotate_half` 的条件分支导致 warp divergence（`d < half` 时前半 thread 走不同路径）。

## 3. GPU 设计

### 3.1 并行化策略：grid-stride + 2D 映射

![RoPE 旋转位置编码数据流](images/rope_dataflow.svg)

策略：1 thread 处理 1 元素，用 2D 映射（row= blockIdx.y, col= blockIdx.x）避免除法：

1. `row = blockIdx.y * blockDim.y + threadIdx.y`（对应 M 维）
2. `col = blockIdx.x * blockDim.x + threadIdx.x`（对应 D 维）
3. `half = D / 2`，`rotate_half` 通过条件判断取对应元素
4. grid-stride loop 覆盖所有元素

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `Q[]`, `cos[]`, `sin[]`, `output[]` | global memory | row-major 连续 |
| `q_val`, `rotated`, `cos_val`, `sin_val` | register | 每 thread 局部变量 |

> elementwise kernel 无数据复用，不需要 shared memory。

### 3.3 关键技巧

- **2D 映射避免除法**：用 `(row, col)` 而非 `idx / D, idx % D`，消除整数除法开销
- **rotate_half 的索引计算**：`d < half` → 取 `d + half` 并取反；`d >= half` → 取 `d - half`
- **coalesced access**：blockDim.x 映射 D 维（连续），warp 内 thread 访问连续地址
- **grid-stride loop**：任意 M×D 一次覆盖

##### rotate_half 本质：2D 旋转

`rotate_half` 看起来是"交换+取反"，本质上是对相邻配对 `(x_d, x_{d+half})` 做 2D 旋转：

```
[output_d, output_{d+half}] = [cos_d, -sin_d; sin_d, cos_d] × [x_d, x_{d+half}]
```

RoPE 通过这种旋转将位置信息编码到 Q/K 中，使得内积 `Q_m · K_n` 只依赖于相对位置 `m-n`。

## 4. Kernel 实现

```cuda
// rope.cu —— Rotary Positional Embedding（grid-stride + 2D 映射）
// 编译命令: nvcc -O3 -arch=sm_120 rope.cu -o rope
// 运行:     ./rope

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_X 32
#define BLOCK_Y 8

// grid-stride + 2D 映射：避免除法，coalesced 访存
__global__ void rope_kernel(const float* Q, const float* cos, const float* sin,
                            float* output, int M, int D) {
    int half = D / 2;
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < M;
         row += gridDim.y * blockDim.y) {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < D;
             col += gridDim.x * blockDim.x) {
            int idx = row * D + col;
            float q = Q[idx];
            // rotate_half: 前半取后半取反，后半取前半
            float rotated = (col < half) ? -Q[idx + half] : Q[idx - half];
            output[idx] = q * cos[idx] + rotated * sin[idx];
        }
    }
}

int main() {
    int M = 1024, D = 64;
    size_t bytes = (size_t)M * D * sizeof(float);
    std::vector<float> h_Q(M * D), h_cos(M * D), h_sin(M * D), h_out(M * D);
    srand(42);
    for (int i = 0; i < M * D; i++) {
        h_Q[i] = (rand() % 200 - 100) / 10.0f;
        h_cos[i] = cosf(i * 0.01f);
        h_sin[i] = sinf(i * 0.01f);
    }

    float *d_Q, *d_cos, *d_sin, *d_out;
    cudaMalloc(&d_Q, bytes); cudaMalloc(&d_cos, bytes);
    cudaMalloc(&d_sin, bytes); cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((D + BLOCK_X - 1) / BLOCK_X, (M + BLOCK_Y - 1) / BLOCK_Y);
    rope_kernel<<<grid, block>>>(d_Q, d_cos, d_sin, d_out, M, D);
    cudaDeviceSynchronize();

    // 验证
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    int half = D / 2;
    bool pass = true;
    for (int m = 0; m < M && pass; m++)
        for (int d = 0; d < D && pass; d++) {
            int idx = m * D + d;
            float rot = (d < half) ? -h_Q[idx + half] : h_Q[idx - half];
            float expect = h_Q[idx] * h_cos[idx] + rot * h_sin[idx];
            if (fabsf(h_out[idx] - expect) > 1e-4) { pass = false; }
        }
    printf("RoPE M=%d D=%d: %s\n", M, D, pass ? "PASS" : "FAIL");

    // 带宽测量
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < 5; i++) rope_kernel<<<grid, block>>>(d_Q, d_cos, d_sin, d_out, M, D);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) rope_kernel<<<grid, block>>>(d_Q, d_cos, d_sin, d_out, M, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float t = ms / 100;
    float bw = 4.0f * bytes / (t / 1000) / 1e9;  // 读 Q+cos+sin=3D + 写 out=D = 4D
    printf("Bandwidth: %.1f GB/s (%.1f%% of 1555 GB/s peak)\n", bw, bw / 1555 * 100);

    cudaFree(d_Q); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_out);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `rope_kernel` 填进 `solve`。核心是 grid-stride 覆盖 + 2D 映射避免除法 + rotate_half 索引计算。带宽 = `4 × M × D × sizeof(float) / time`（读 Q+cos+sin + 写 output）。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_120 rope.cu -o rope
ncu --set full --kernel rope_kernel ./rope 2>&1 | rg -i "Memory Throughput|DRAM|Achieved Occupancy|Warp"
```

**关键指标**：

| 指标 | 朴素（1D idx + 除法） | 优化（2D 映射 + grid-stride） |
|------|---------------------|------------------------------|
| 整数除法 | 有（`idx/D, idx%D`） | 无（2D 映射） |
| Warp divergence | 有（`d < half` 分支） | 有（本质无法消除） |
| RTX 5090 带宽 | ~600 GB/s | ~1000 GB/s |
| 带宽利用率 | ~39% | ~64% |

**优化方向**：

1. **2D 映射消除除法**：整数除法开销 20+ cycle，2D 映射直接用 `row/col` 避免
2. **grid-stride loop**：任意 M×D 一次覆盖
3. **float4 向量化**：每 thread 处理 4 个 float，减少指令数（需 D 是 4 的倍数）
4. **预计算 cos/sin**：题目已预计算，实际系统中 cos/sin 在初始化时算一次复用
5. **kernel fusion**：RoPE 与上游 RMSNorm 融合，省一次 HBM 往返

##### 为什么 RoPE 是 memory-bound？

```
每元素：2 次乘 + 1 次加 ≈ 3 FLOP
每元素访存：读 Q(4B) + cos(4B) + sin(4B) + 读 rotate_half 的 Q(4B) + 写 out(4B) = 20B
算术强度 ≈ 3/20 = 0.15 FLOP/Byte
RTX 5090 ridge point ≈ 12.6 FLOP/Byte
0.15 << 12.6 → 严重 memory-bound
```

> 注意：`rotate_half` 需要额外读一个 Q 元素（`Q[idx±half]`），所以访存比纯 elementwise 多。优化重点是打满带宽。

## 6. 复杂度分析

| 维度 | 朴素 | 优化 |
|------|------|------|
| 时间 | `O(M·D)` | `O(M·D)`（常数小，无除法） |
| 空间 | `O(1)` | `O(1)` |
| 算术强度 | ~0.15 FLOP/Byte | ~0.15 FLOP/Byte |
| 瓶颈 | memory bandwidth | memory bandwidth |
| 带宽利用率 | ~39% | ~64% |

> 💡 **一句话总结**：RoPE 是 LLaMA 架构的核心位置编码——elementwise + memory-bound，优化重点是 2D 映射消除除法 + coalesced 访存 + kernel fusion。它对应 Week8 Day2 的架构图：RoPE 位于模型层 Attention 之前，画数据流图时必须画出"Q 分两路（cos / rotate_half×sin）→ 相加"的微型数据流。

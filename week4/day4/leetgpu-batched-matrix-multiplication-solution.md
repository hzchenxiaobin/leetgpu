# LeetGPU Batched Matrix Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Batched Matrix Multiplication（#30，medium）
- **链接**：https://leetgpu.com/challenges/batched-matrix-multiplication
- **难度**：中等
- **标签**：CUDA、batched GEMM、tiled matmul、register blocking

**题意**：给定一个 batch 的矩阵对 `(A[i], B[i])`，对每个 batch 元素独立做矩阵乘法 `C[i] = A[i] × B[i]`。`A[i]` 是 `M×K`，`B[i]` 是 `K×N`，`C[i]` 是 `M×N`。所有 batch 元素共享相同的 `M, N, K` 形状。

**示例**：

```text
batch=2, M=2, K=2, N=2
A[0] = [[1,2],[3,4]]  B[0] = [[5,6],[7,8]]  → C[0] = [[19,22],[43,50]]
A[1] = [[1,0],[0,1]]  B[1] = [[1,2],[3,4]]  → C[1] = [[1,2],[3,4]]
```

**约束**：`1 ≤ batch ≤ 256`，`1 ≤ M, N, K ≤ 1024`；性能测试取大 batch。

> 💡 这道题的 **batched GEMM** 与 [Week6 Day5](../../aiinfra/week6/day5/README.md) Mini Engine v1 的多请求并发 forward 同构——v1 每轮把多个请求拼 batch 送 model forward，其中 attention/FFN 的核心计算就是 batched GEMM（batch=请求数）。batched matmul 的"每个 batch 独立计算、共享 kernel launch"正是 v1 Scheduler"每轮选多请求组 batch、一次 forward"的底层映射。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
// 对每个 batch 顺序做矩阵乘法，O(batch × M × N × K)
for (int b = 0; b < batch; b++)
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++)
                sum += A[b][i][k] * B[b][k][j];
            C[b][i][j] = sum;
        }
```

### 朴素 GPU（一 thread 一输出元素）

```cuda
// 每个 thread 算 C[b][i][j] 一个元素
__global__ void naive_batched_matmul(const float* A, const float* B, float* C, int batch, int M, int N, int K) {
    int b = blockIdx.z;
    int j = blockIdx.x * blockDim.x + threadIdx.x; // N 维
    int i = blockIdx.y * blockDim.y + threadIdx.y; // M 维
    if (b >= batch || i >= M || j >= N)
        return;
    float sum = 0;
    for (int k = 0; k < K; k++)
        sum += A[b * M * K + i * K + k] * B[b * K * N + k * N + j];
    C[b * M * N + i * N + j] = sum;
}
```

**瓶颈**：每个 thread 重复读 A 的行和 B 的列，global memory 访问冗余严重，无 tiling。

## 3. GPU 设计

### 3.1 并行化策略：batch 维 + tiled matmul

![Batched Matmul：batch 维 + 输出 tile 维并行](images/batched_matmul_overview.svg)

三维并行：
1. **batch 维**（`blockIdx.z`）：每个 batch 元素独立，一个 block 处理一个 batch 的一个 tile
2. **输出 tile 维**（`blockIdx.x/y`）：tiled matmul，把 `C[b]` 切成 `TILE×TILE` 的块
3. **K 维累加**：沿 K 方向遍历 tile，shared memory 缓存 A/B 的 tile

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `A[b]`, `B[b]` | global memory（stride 索引） | batch 维 stride 寻址 |
| A/B tile | shared memory | `TILE×TILE`，block 内共享 |
| C tile 累加器 | registers | 每个 thread 持有部分和 |

### 3.3 关键技巧

- **`blockIdx.z` 索引 batch**：grid 第三维天然映射 batch 维，各 batch 独立
- **stride 寻址**：`A[b][i][k] = A_flat[b * M * K + i * K + k]`，batch 间 stride = `M*K`
- **tiled matmul**：沿 K 方向分块，shared memory 缓存，减少 global 读取（同 Week1 Day6 的 tiling）

## 4. Kernel 实现

```cuda
// batched_matmul.cu —— Batched Matrix Multiplication（batch 维 + tiled matmul）
// 编译命令: nvcc -O3 -arch=sm_120 batched_matmul.cu -o batched_matmul
// 运行:     ./batched_matmul

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define TILE 16

// batched matmul：grid((N+TILE-1)/TILE, (M+TILE-1)/TILE, batch)
// blockIdx.z = batch index, blockIdx.x/y = 输出 C[b] 的 tile 位置
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int batch, int M, int N, int K) {
    int b = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y; // M 维
    int col = blockIdx.x * TILE + threadIdx.x; // N 维
    if (b >= batch || row >= M || col >= N)
        return;

    // batch stride 寻址
    const float* A_b = A + b * M * K;
    const float* B_b = B + b * K * N;
    float* C_b = C + b * M * N;

    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    float sum = 0.0f;
    // 沿 K 方向分 tile 累加
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // 加载 A/B tile 到 shared memory（越界补 0）
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A_b[row * K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B_b[b_row * N + col] : 0.0f;
        __syncthreads();

// tile 内累加
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    C_b[row * N + col] = sum;
}

int main() {
    int batch = 4, M = 64, N = 64, K = 64;
    size_t a_bytes = batch * M * K * sizeof(float);
    size_t b_bytes = batch * K * N * sizeof(float);
    size_t c_bytes = batch * M * N * sizeof(float);

    std::vector<float> h_A(batch * M * K), h_B(batch * K * N), h_C(batch * M * N);
    srand(42);
    for (auto& x : h_A)
        x = (rand() % 100) / 100.0f;
    for (auto& x : h_B)
        x = (rand() % 100) / 100.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_bytes);
    cudaMalloc(&d_B, b_bytes);
    cudaMalloc(&d_C, c_bytes);
    cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice);

    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
    dim3 block(TILE, TILE);
    batched_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, batch, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost);

    // CPU 验证
    bool pass = true;
    for (int b = 0; b < batch && pass; b++)
        for (int i = 0; i < M && pass; i++)
            for (int j = 0; j < N && pass; j++) {
                float s = 0;
                for (int k = 0; k < K; k++)
                    s += h_A[b * M * K + i * K + k] * h_B[b * K * N + k * N + j];
                if (fabs(s - h_C[b * M * N + i * N + j]) > 1e-3)
                    pass = false;
            }
    printf("batch=%d M=N=K=%d, %s\n", batch, M, pass ? "PASS" : "FAIL");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `batched_matmul_kernel` 填进 `solve`。核心是 `blockIdx.z` 索引 batch + stride 寻址 `A + b*M*K`。tiled 部分同 Week1 Day6 的 matmul tiling，沿 K 方向分块用 shared memory 缓存。

## 5. 性能分析与优化

```bash
nvcc -O3 -arch=sm_120 batched_matmul.cu -o batched_matmul
ncu --set full ./batched_matmul | rg -i "Memory Throughput|Compute|Occupancy"
```

**关键指标**：

| 指标 | 朴素（无 tiling） | tiled + batched |
|------|-----------------|-----------------|
| global 读取冗余 | 高（每元素重复读行列） | 低（shared mem 缓存） |
| batch launch | 可选（每 batch 一 kernel） | 单次 launch（blockIdx.z） |
| 算术强度 | 低 | 高（tile 内复用） |

**优化方向**：

1. **register blocking**：每 thread 持有多个 C 元素，提升算术强度（同 Week2 Day2 GEMM）
2. **vectorized load**：`float4` 一次读 4 个 float，提升带宽利用率
3. **大 TILE**：`TILE=32` 减少边界开销（但占更多 shared memory）
4. **cuBLASLt batched**：生产环境用 `cublasLtMatmul` 的 batched 接口，已极致优化

## 6. 复杂度分析

| 维度 | 朴素 | tiled + batched |
|------|------|-----------------|
| 时间 | `O(batch×M×N×K)` | `O(batch×M×N×K)`（常数更小） |
| 空间 | `O(1)` 额外 | `O(TILE²)` shared memory/block |
| 算术强度 | ~0.5（memory-bound） | ~2-4（接近 compute-bound） |
| 瓶颈 | global 带宽 | 算力（大 K 时） |

> 💡 **一句话总结**：Batched Matmul 是 Mini Engine v1 多请求 forward 的底层映射——`blockIdx.z` 索引 batch = v1 每轮 batch 个请求，stride 寻址 = 各请求独立 KV Cache，tiled 累加 = 共享 kernel 代码复用。生产环境用 cuBLASLt batched 接口。

# LeetGPU Convolution 题解

## 1. 题目概述

- **标题 / 题号**：Convolution
- **链接**：https://leetgpu.com/challenges/convolution
- **难度**：中等
- **标签**：CUDA、Convolution、CUDA Streams、Halo Exchange、Shared Memory

给定输入矩阵 `input` 和一个 `K×K` 的卷积核 `kernel`，计算 2D 卷积输出。要求用多 Stream 分块并行处理大矩阵。

约束：`1 ≤ M, N ≤ 8192`，`1 ≤ K ≤ 15`（K 为奇数），元素范围 `[-1.0, 1.0]`。

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 基线

```cpp
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
        float sum = 0.0f;
        for (int ki = 0; ki < K; ++ki)
            for (int kj = 0; kj < K; ++kj)
                sum += input[(i+ki-K/2)*N + (j+kj-K/2)] * kernel[ki*K+kj];
        output[i*N+j] = sum;
    }
```

### 朴素 GPU 方法（单 Stream）

每个线程计算一个输出像素，直接访问全局内存。无 halo exchange，无多 Stream，性能受限于全局内存延迟。

## 3. GPU 设计

### 3.1 并行化策略：多 Stream 分块

大矩阵按行分块，每块在独立 Stream 上处理：

```
Stream 1: [H2D chunk1] → [Conv chunk1] → [D2H chunk1]
Stream 2:        [H2D chunk2] → [Conv chunk2] → [D2H chunk2]
Stream 3:               [H2D chunk3] → [Conv chunk3] → [D2H chunk3]
```

- H2D 与 Compute 重叠（Copy Engine + Compute Engine 独立）
- 需要 **pinned memory** 保证异步传输生效

### 3.2 Shared Memory Halo Exchange

每个 block 加载 `BLOCK_SIZE + K-1` 的 tile（含 halo 区域）到 Shared Memory，避免卷积边界重复访问全局内存。

### 3.3 存储层次使用

| 层次 | 用途 |
|------|------|
| Pinned Host Memory | H2D/D2H 异步传输 |
| Global Memory | chunk 数据 |
| Shared Memory | tile + halo |
| Register | 累加器 |

## 4. Kernel 实现

```cuda
// convolution.cu —— 多 Stream 分块 2D 卷积
// 编译命令: nvcc -o convolution convolution.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK_SIZE 16

__global__ void conv2d(const float* input, const float* kernel, float* output,
                       int width, int height, int K) {
    int radius = K / 2;
    __shared__ float tile[BLOCK_SIZE + 14][BLOCK_SIZE + 14];  // K<=15

    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * BLOCK_SIZE + tx;
    int gy = blockIdx.y * BLOCK_SIZE + ty;

    // 加载含 halo 的 tile
    int sx = tx, sy = ty;
    while (sx < BLOCK_SIZE + 2*radius && sy < BLOCK_SIZE + 2*radius) {
        int ix = blockIdx.x * BLOCK_SIZE + sx - radius;
        int iy = blockIdx.y * BLOCK_SIZE + sy - radius;
        ix = max(0, min(ix, width - 1));
        iy = max(0, min(iy, height - 1));
        tile[sy][sx] = input[iy * width + ix];
        sx += blockDim.x; sy += blockDim.y;
    }
    __syncthreads();

    if (gx < width && gy < height) {
        float sum = 0.0f;
        for (int ky = 0; ky < K; ky++)
            for (int kx = 0; kx < K; kx++)
                sum += tile[ty + ky][tx + kx] * kernel[ky * K + kx];
        output[gy * width + gx] = sum;
    }
}

int main() {
    int M = 4096, N = 4096, K = 5;
    int nStreams = 4;
    int chunkRows = M / nStreams;

    // Pinned memory (异步传输必需)
    float *h_in, *h_out;
    cudaMallocHost(&h_in, M * N * sizeof(float));
    cudaMallocHost(&h_out, M * N * sizeof(float));

    float *d_in, *d_out, *d_kernel;
    cudaMalloc(&d_in, M * N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));

    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    for (int i = 0; i < nStreams; i++) {
        int offset = i * chunkRows;
        size_t bytes = chunkRows * N * sizeof(float);

        cudaMemcpyAsync(d_in + offset * N, h_in + offset * N, bytes,
                        cudaMemcpyHostToDevice, streams[i]);

        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (chunkRows + BLOCK_SIZE - 1) / BLOCK_SIZE);
        conv2d<<<grid, block, 0, streams[i]>>>(
            d_in + offset * N, d_kernel, d_out + offset * N, N, chunkRows, K);

        cudaMemcpyAsync(h_out + offset * N, d_out + offset * N, bytes,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();
    // (省略验证代码)

    for (int i = 0; i < nStreams; i++) cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

## 5. 性能分析与优化

### nsys 观察 Stream 重叠

```bash
nsys profile -o conv_timeline ./convolution
```

在 nsys timeline 中观察 H2D/Compute/D2H 是否真正重叠。

### 优化方向

1. **Pinned Memory**：`cudaMallocHost`，否则 async 退化为 sync
2. **cudaStreamNonBlocking**：避免 Default Stream 隐式同步
3. **Halo exchange 优化**：用 Shared Memory 缓存边界

## 6. 复杂度分析

- **时间复杂度**：`O(M×N×K²)`。
- **空间复杂度**：`O(M×N)` + `O((BLOCK_SIZE+K)²)` Shared Memory。
- **算术强度**：`K² FLOP / (4+4) Bytes`，K=5 时 AI≈3.1，**compute-bound 偏向**。

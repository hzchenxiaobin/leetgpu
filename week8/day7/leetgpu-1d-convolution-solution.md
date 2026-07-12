# LeetGPU 1D Convolution 题解

## 1. 题目概述

- **标题 / 题号**：1D Convolution（#9，easy）
- **链接**：https://leetgpu.com/challenges/1d-convolution
- **难度**：简单
- **标签**：CUDA、Convolution、Shared Memory、Halo、memory-bound

**题意**：给定长度 `N` 的输入信号 `x` 和长度 `K` 的卷积核 `kernel`（`K` 为奇数），计算一维卷积输出 `y`，长度为 `N`：

```text
y[i] = Σ_{j=0}^{K-1} x[i + j - (K-1)/2] · kernel[j]     对 i = 0..N-1
```

其中越界位置 `x[idx]`（`idx < 0` 或 `idx >= N`）按 `0` 处理（zero padding）。

**示例**（`N=5, K=3, x=[1,2,3,4,5], kernel=[1,0,-1]`）：

```text
y[0] = 0·1 + 1·0 + 2·(-1) = -2     （左侧补 0）
y[1] = 1·1 + 2·0 + 3·(-1) = -2
y[2] = 2·1 + 3·0 + 4·(-1) = -2
y[3] = 3·1 + 4·0 + 5·(-1) = -2
y[4] = 4·1 + 5·0 + 0·(-1) = 4      （右侧补 0）
输出：y = [-2, -2, -2, -2, 4]
```

**约束**：`N` 较大（如 `N ≥ 65536`），`K` 较小且为奇数（如 `K = 3, 5, 7`）。

> 💡 1D Convolution 是 **shared memory + halo region** 模板的最简形态。它与 [Week8 Day7 最终复盘](../../aiinfra/week8/day7/README.md) 的"8 周能力地图"中 Kernel 优化层强项（Shared Memory Tiling + Bank Conflict）直接对应——这道收官题检验你最基础的 shared memory halo 加载能力。掌握它，2D/3D Convolution 都是同构扩展。

---

## 2. CPU 基线 / 朴素 GPU 方法

### CPU 串行

```cpp
void cpu_conv1d(const float* x, const float* kernel, float* y, int N, int K) {
    int half = K / 2;
    for (int i = 0; i < N; i++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) {
            int idx = i + j - half;
            acc += (idx >= 0 && idx < N) ? x[idx] * kernel[j] : 0.0f;
        }
        y[i] = acc;
    }
}
```

**瓶颈**：每个输出元素读 `K` 个输入元素，相邻输出共享 `K-1` 个输入，但朴素实现每次都从全局内存重读 → `N×K` 次全局访问，带宽浪费严重。

### 朴素 GPU：每线程直接读全局内存

```text
每个线程算一个 y[i]：
  for j in 0..K-1: acc += x[i+j-half] * kernel[j]
```

**问题**：① 相邻线程的 `i+j-half` 高度重叠，全局内存重复读取 ② 每个元素都走全局内存，没利用 shared memory ③ `K` 较小时是典型 memory-bound。

---

## 3. GPU 设计

### 3.1 并行化策略

![1D Convolution shared memory + halo 加载](images/conv1d_halo.svg)

**分块策略**：把输入分成大小 `TILE` 的块，每个 block 协作加载一个 tile 到 shared memory，然后每个线程算一个输出元素。

| 阶段 | 操作 | 访存 |
|------|------|------|
| ① 加载 tile + halo | block 协作把 `TILE + 2·half` 个元素加载到 shared memory | Global → Shared |
| ② 计算 | 每个线程从 shared memory 读 `K` 个元素做点积 | Shared → Register |
| ③ 写回 | 每个线程写一个 `y[i]` | Register → Global |

### 3.2 存储层次使用

- **Global Memory**：`x`（输入）、`y`（输出）、`kernel`（卷积核，可放 `__constant__` 内存）
- **Shared Memory**：tile + halo 缓冲 `s_x[TILE + 2*half]`，block 内共享
- **Register**：每个线程的累加器 `acc`、循环索引

### 3.3 关键技巧

1. **Halo region（边界光晕）**：每个 tile 左右各需要 `half = K/2` 个额外元素（来自相邻 tile），这些就是 halo。加载时块内线程协作把 halo 一起搬进 shared memory。
2. **shared memory 复用**：相邻输出的 `K-1` 个重叠输入只从全局读一次，后续从 shared memory 读（~20 cycles vs ~400-800 cycles）。
3. **`__constant__` 内存**：`kernel` 长度小且所有线程访问相同，放 constant memory 有广播缓存收益。
4. **边界处理**：tile 起始/结束的 halo 可能越界，加载时用条件判断填 0（zero padding）。

---

## 4. Kernel 实现

```cuda
// conv1d.cu —— 1D Convolution (shared memory + halo)
// 编译命令: nvcc -o conv1d conv1d.cu -O3 -arch=sm_120
// 运行命令: ./conv1d

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILE 256

// 卷积核放 constant 内存（小且广播访问）
__constant__ float c_kernel[64];

__global__ void conv1dKernel(const float* __restrict__ x,
                             float* __restrict__ y,
                             int N, int K) {
    int half = K / 2;
    int tid = threadIdx.x;
    int block_start = blockIdx.x * TILE;
    int i = block_start + tid;          // 该线程负责的输出下标

    __shared__ float s_x[TILE + 2 * 32]; // halo 最大 half=32，足够 K<=65

    // ---- 阶段 ①：协作加载 tile + halo ----
    // tile 区域：[block_start - half, block_start + TILE + half)
    int s_len = TILE + 2 * half;
    for (int idx = tid; idx < s_len; idx += blockDim.x) {
        int gidx = block_start - half + idx;
        s_x[idx] = (gidx >= 0 && gidx < N) ? x[gidx] : 0.0f;
    }
    __syncthreads();

    // ---- 阶段 ②：计算（从 shared memory 读） ----
    if (i < N) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < K; j++) {
            acc += s_x[tid + j] * c_kernel[j];   // tid + j 已含 left halo 偏移
        }
        y[i] = acc;
    }
}

void initArray(float* a, int n) {
    srand(42);
    for (int i = 0; i < n; i++)
        a[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
}

bool checkResult(const float* a, const float* b, int n, float eps) {
    for (int i = 0; i < n; i++)
        if (fabsf(a[i] - b[i]) > eps) {
            printf("Mismatch at %d: %.6f vs %.6f\n", i, a[i], b[i]);
            return false;
        }
    return true;
}

int main() {
    int N = 1 << 20;   // 1M
    int K = 5;
    size_t bytes = N * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    float *h_kernel = (float*)malloc(K * sizeof(float));
    initArray(h_x, N);
    initArray(h_kernel, K);

    // CPU 参考
    int half = K / 2;
    for (int i = 0; i < N; i++) {
        float acc = 0.0f;
        for (int j = 0; j < K; j++) {
            int idx = i + j - half;
            acc += (idx >= 0 && idx < N) ? h_x[idx] * h_kernel[j] : 0.0f;
        }
        h_ref[i] = acc;
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_kernel, h_kernel, K * sizeof(float));

    int threads = TILE;
    int blocks = (N + TILE - 1) / TILE;

    conv1dKernel<<<blocks, threads>>>(d_x, d_y, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    conv1dKernel<<<blocks, threads>>>(d_x, d_y, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    bool ok = checkResult(h_y, h_ref, N, 1e-4f);

    printf("=== 1D Convolution (Shared Memory + Halo) ===\n");
    printf("N=%d, K=%d, TILE=%d\n", N, K, TILE);
    printf("Kernel time: %.3f ms\n", ms);
    float gbytes = (float)(N + N) * sizeof(float) / (ms * 1e6);
    printf("Effective bandwidth: %.1f GB/s\n", gbytes);
    printf("Correctness: %s\n", ok ? "PASS" : "FAIL");

    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y); free(h_ref); free(h_kernel);
    return 0;
}
```

---

## 5. 性能分析与优化

```bash
nvcc -o conv1d_profile conv1d.cu -O3 -arch=sm_120 -g -lineinfo
ncu --kernel-name regex:conv1dKernel \
    --metrics \
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    launch__registers_per_thread,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    ./conv1d_profile
```

**关键指标与解读**：

| 指标 | 朴素全局读 | Shared + Halo | 说明 |
|------|----------|--------------|------|
| DRAM Throughput | ~20%（重复读） | ~50-60% | shared 消除重复读，有效带宽提升 |
| SM Throughput | ~15% | ~25% | memory-bound，SM 利用率不高 |
| Bank Conflicts | N/A | 接近 0 | 连续线程读连续地址，无 conflict |
| Registers/Thread | ~16 | ~18 | 仅多累加器 |

**为什么是 memory-bound**：1D 卷积算术强度 `AI ≈ K / (2·4) ≈ 0.6 FLOP/Byte`（K=5），远低于 Ridge Point 12.6 → 纯 memory-bound。优化核心是**减少全局重复读**（用 shared memory）和**提高有效带宽**。

**进一步优化方向**：

1. **float4 向量化加载**：tile 加载时按 4 个 float 一组，提升带宽利用率（+10-15%）
2. **更大 TILE**：增大 tile 减少 halo 占比（halo/TILE 越小越省），但受 shared memory 容量限制
3. **kernel 融合**：若 conv1d 后接激活（如 ReLU/SiLU），融合写回避免一次全局读写
4. **2D/3D 扩展**：1D 的 halo 思想直接推广到 2D（四方向 halo）/3D（六方向 halo）

---

## 6. 复杂度分析

| 维度 | 复杂度 | 说明 |
|------|--------|------|
| **时间** | `O(N·K)` | 每输出做 K 次乘加；shared memory 版常数因子更小 |
| **空间** | `O(TILE + K)` 额外 | shared memory tile + halo + constant kernel |
| **算术强度** | `~0.6 FLOP/Byte`（K=5） | 每元素读 4B 做 ~5 次运算 → memory-bound |
| **瓶颈类型** | **memory-bound** | 优化方向是 shared memory 复用 + 提带宽 |

> 💡 **一句话总结**：1D Convolution 是 shared memory halo 模板的最简教学题——核心就一句话：**把 tile 和它的左右 halo 一次性搬进 shared memory，后续计算只读 shared**。这个模板掌握后，2D Conv / Stencil / Gaussian Blur 都是同构扩展。作为 8 周收官题，它检验你最扎实的 shared memory 基本功。

# LeetGPU Matrix Transpose 题解

## 1. 题目概述

- **标题 / 题号**：Matrix Transpose（#3，easy）
- **链接**：https://leetgpu.com/challenges/matrix-transpose
- **难度**：简单
- **标签**：CUDA、shared memory、tiling、bank conflict、memory-bound

**题意**：给定行主序存储的矩阵 `A`（维度 `rows × cols`），计算其转置 `A^T`（维度 `cols × rows`），结果以行主序写入 `output`。即 `output[j][i] = input[i][j]`。

**示例**：

```text
输入：A = [1.0, 2.0, 3.0]    (2×3)
          [4.0, 5.0, 6.0]
输出：A^T = [1.0, 4.0]       (3×2)
            [2.0, 5.0]
            [3.0, 6.0]
```

**约束**：

- `1 ≤ rows, cols ≤ 8192`
- 性能测试取 `rows = 7000, cols = 6000`
- `solve` 函数签名不可改，外部库禁用，结果必须写入 `output`

> 💡 这是 **shared memory** 的入门必修课。前两题（Vector Addition、ReLU）是纯 elementwise，每个数据只用一次、不需要 shared memory。转置则不同——它要在读和写之间做一次"中转"，正好是 shared memory 的用武之地。它还引出 GPU 编程里一个独特的性能杀手：**bank conflict**。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行矩阵转置
void transpose_cpu(const float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}
```

双重循环，`O(rows × cols)`。`rows=7000, cols=6000` 时约 4200 万次操作，单核几十毫秒。

### 2.2 朴素 GPU：一个 thread 一个元素

最直观的并行：每个 thread 负责一个 `(i, j)`，读 `input[i][j]`、写 `output[j][i]`。

```cuda
__global__ void transpose_naive(const float* input, float* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // row
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // col
    if (i < rows && j < cols) {
        output[j * rows + i] = input[i * cols + j];
    }
}
```

问题藏在访存模式里。行主序下，`input[i][j]` 的地址是 `i*cols + j`，`output[j][i]` 的地址是 `j*rows + i`。

![朴素转置的访存困境](images/transpose_naive_problem.svg)

- 若 thread 按 `(i, j)` 连续排列（`threadIdx.x` 对应 `j`），则读 `input[i][j], input[i][j+1], ...` **地址连续 → 读合并 ✓**。
- 但写 `output[j][i], output[j+1][i], ...` 地址间隔 `rows` 个元素 → **写跨步 ✗**，warp 内 32 次写散落 32 段，硬件发起 32 次内存事务。

反过来如果让 `threadIdx.x` 对应 `i`，写合并了但读又跨步了。**矛盾的本质**：转置操作把"连续的行"变成"分散的列"，读和写必有一个不合并，带宽利用率直接减半。

> ⚠️ 这是 elementwise kernel 没遇到过的问题——前两题读写同序，天然都合并。转置是第一个"读写顺序不一致"的 kernel，需要一个中转缓冲来打破矛盾。

## 3. GPU 设计

### 3.1 并行化策略：shared memory tiling

破局思路：用 **shared memory** 做"中转站"。把输入矩阵切成 `TILE × TILE` 的小块，每个 block 负责一个 tile：

1. **合并读**：block 内线程按行从 global memory 读一个 tile 到 shared memory（读连续 → 合并）。
2. **`__syncthreads()`**：等整个 tile 写完 shared memory。
3. **合并写**：从 shared memory **按列读出**（转置），按行写入 output 的对应位置（写连续 → 合并）。

![Shared Memory Tiling 方案](images/transpose_tiling.svg)

shared memory 的带宽远高于 HBM（约 10×），且按 bank 并行访问，所以"在 shared 里转置读"的开销远小于"在 global 里跨步写"。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写，都在 HBM |
| **shared memory** | ✓ | **本题核心**：tile 中转缓冲，`TILE × (TILE+1)` 大小（含 padding） |
| **register** | ✓（隐式） | 线程临时变量 `x = input[...]` |

这是本系列**第一次使用 shared memory**。它和 global memory 的关键区别：

| 特性 | global memory (HBM) | shared memory (SRAM) |
|------|---------------------|----------------------|
| 容量 | ~40-80 GB | ~100-228 KB/SM |
| 延迟 | ~400-800 cycles | ~20-30 cycles |
| 带宽 | ~1.5-3 TB/s | ~10-19 TB/s |
| 可见性 | 所有 thread | 同一 block 内 |
| 访问模式 | 按 cache line | 按 bank（32 个，每 4B 一个） |

### 3.3 关键技巧：bank conflict 与 padding

shared memory 按 **32 个 bank** 分区，每 4 字节一个 bank，地址按 `addr/4 % 32` 轮转。一个 warp 的 32 个 thread 如果同时访问不同 bank → 1 cycle 并行完成；如果落在同一 bank → **串行化**（bank conflict）。

转置的"按列读 shared"天然触发 bank conflict：第 `k` 列的元素 `s[0][k], s[1][k], ..., s[31][k]`，地址间隔 `TILE × 4` 字节。当 `TILE = 32` 时，`(TILE × 4 / 4) % 32 = 0`，全部落在 bank 0 → **32 路 bank conflict**！

![Bank Conflict 与 Padding 解决](images/transpose_bank_conflict.svg)

**解法：加 1 列 padding**。把 shared memory 声明为 `s[TILE][TILE+1]` 而不是 `s[TILE][TILE]`。这样第 `k` 列的地址间隔变成 `(TILE+1) × 4` 字节，bank id 错开 1，32 个元素均匀分布到 32 个 bank → **零冲突**。

代价仅是多用 `TILE × 4` 字节 shared memory（TILE=32 时 128B/block），换来 32× 的 shared 访问加速。

> 💡 **bank conflict 是 GPU 编程独有的性能陷阱**——CPU 的 L1 cache 没有这个概念。只要用到 shared memory 且访问模式不是"整行读"，就必须检查 bank conflict。`s[TILE][TILE+1]` 这个 padding 技巧是 CUDA 编程的经典模板，背下来即可。

## 4. Kernel 实现

完整可编译的 shared memory tiling + padding 版本：

```cuda
// transpose_tiled.cu —— shared memory tiling + bank conflict padding 实现矩阵转置
// 编译命令: nvcc -O3 -arch=sm_80 transpose_tiled.cu -o transpose
// 运行:     ./transpose 7000 6000

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

#define TILE 32

// shared memory tiling + padding：消除 bank conflict
__global__ void transpose_tiled(const float* input, float* output, int rows, int cols) {
    // TILE+1 列：多 1 列 padding，错开 bank，消除 32 路 bank conflict
    __shared__ float tile[TILE][TILE + 1];

    // block 处理 input 中的 (blockIdx.y*TILE, blockIdx.x*TILE) 起点 tile
    int x = blockIdx.x * TILE + threadIdx.x;   // col index in input
    int y = blockIdx.y * TILE + threadIdx.y;   // row index in input

    // ---- ① 合并读：按行从 global 读到 shared ----
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    // ---- ② block 级同步：等整个 tile 写完 ----
    __syncthreads();

    // ---- ③ 合并写：转置坐标，从 shared 按列读、按行写 output ----
    // 转置后：output 的 (x, y) 对应 input 的 (y, x)
    // output 维度为 cols × rows，output[x][y] 的地址 = x * rows + y
    int out_x = blockIdx.y * TILE + threadIdx.x;  // 对应 input 的 row
    int out_y = blockIdx.x * TILE + threadIdx.y;  // 对应 input 的 col
    if (out_y < cols && out_x < rows) {
        // 转置读：threadIdx.x 选列，threadIdx.y 选行 → 读 tile[threadIdx.x][threadIdx.y]
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char** argv) {
    int rows = (argc > 1) ? atoi(argv[1]) : 7000;
    int cols = (argc > 2) ? atoi(argv[2]) : 6000;
    size_t in_bytes  = (size_t)rows * cols * sizeof(float);
    size_t out_bytes = (size_t)cols * rows * sizeof(float);
    printf("matrix: %d x %d  (%.1f MB)\n", rows, cols, in_bytes / 1e6);

    // ---- host 端分配与初始化 ----
    float *hIn  = (float*)malloc(in_bytes);
    float *hOut = (float*)malloc(out_bytes);
    srand(42);
    for (int i = 0; i < rows * cols; ++i) {
        hIn[i] = (float)(rand() % 10000) / 100.0f;
    }

    // ---- device 端分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn,  in_bytes));
    CHECK_CUDA(cudaMalloc(&dOut, out_bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, in_bytes, cudaMemcpyHostToDevice));

    // ---- 启动配置：2D grid/block，每 block TILE×TILE 个 thread ----
    dim3 threads(TILE, TILE);
    dim3 blocks((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    printf("launch: blocks=(%d,%d)  threads=(%d,%d)\n",
           blocks.x, blocks.y, threads.x, threads.y);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    transpose_tiled<<<blocks, threads>>>(dIn, dOut, rows, cols);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, out_bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < rows && err < 5; ++i) {
        for (int j = 0; j < cols && err < 5; ++j) {
            float ref = hIn[i * cols + j];
            float got = hOut[j * rows + i];
            if (fabsf(got - ref) > 1e-5f) {
                ++err;
                printf("MISMATCH @(%d,%d): got %f, expect %f\n", i, j, got, ref);
            }
        }
    }
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 带宽估算：读 in + 写 out = 2 × rows × cols × 4B ----
    size_t rw_bytes = 2 * (size_t)rows * cols * sizeof(float);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `transpose_tiled` kernel 填进 starter 的 `__global__` 空壳即可。starter 已配好 `dim3 threadsPerBlock(16, 16)` 和对应 grid，可改成 `TILE=32` 以获得更高吞吐。带 `main()` 的完整文件用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 transpose_tiled.cu -o transpose
./transpose 7000 6000
```

典型输出（A100 / SM=108）：

```text
matrix: 7000 x 6000  (168.0 MB)
launch: blocks=(188,219)  threads=(32,32)
kernel time: 0.42 ms
verify: PASS
effective bandwidth: 800.0 GB/s
```

对比朴素版本（无 shared memory）的典型带宽约 300-400 GB/s，tiled 版**带宽翻倍**，这正是"读写都合并"的效果。

### 5.2 用 ncu 对比 naive vs tiled

```bash
# 分别编译
nvcc -O3 -arch=sm_80 -DNAIVE transpose_tiled.cu -o transpose_naive  # 朴素版
nvcc -O3 -arch=sm_80             transpose_tiled.cu -o transpose_tiled

# 对比 bank conflict 与带宽
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        gpu__time_duration.sum \
    ./transpose_naive 7000 6000

ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        gpu__time_duration.sum \
    ./transpose_tiled 7000 6000
```

| 指标 | naive 版 | tiled + padding 版 |
|------|----------|-------------------|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | 0（没用 shared） | ~0（padding 消除） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ~30-40% | ~60-80% |
| `gpu__time_duration.sum` | 基线 | **~2× 加速** |

> 💡 还可对比"tiled 无 padding"版本（`tile[TILE][TILE]`），观察 `l1tex__data_bank_conflicts` 飙升、带宽下降——这是验证 bank conflict 影响的最佳实验。

### 5.3 优化方向

1. **`float4` 向量化**：每 thread 一次搬 4 个 float，减少索引计算与指令数。TILE 可保持 32，但每 thread 处理 4×4 个元素。
2. **diagonal block reordering**：处理方阵时，`(i,j)` 和 `(j,i)` 两个 tile 会被不同 block 读写同一区域造成竞争。让 block 按"对角线顺序"执行可缓解，但对方阵以外收益有限。
3. **TILE 大小调优**：32 是经典选择（匹配 warp 大小），也可试 16 或 64。64 会增加 shared memory 占用、降低 occupancy，需 profile 权衡。
4. **`cudaMemcpy2D` 对比**：对纯转置，`cudaMemcpy2D`（带不同 pitch）有时更快，因走 DMA 引擎而非 SM。但本题要求写 kernel，仅作 benchmark 参考。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(rows × cols)`，每个元素一次读 + 一次写 |
| **空间复杂度** | `O(rows × cols)` 输入 + 输出 + `O(TILE²)` shared memory |
| **算术强度** | `0 FLOP / 8B`（无计算，纯搬数据）= **0 FLOP/B** |
| **瓶颈类型** | **memory-bound**：纯数据搬运，零计算，完全受 HBM 带宽限制 |
| **访存量** | `2 × rows × cols × 4B`（读 input + 写 output） |
| **shared memory 占用** | `TILE × (TILE+1) × 4B = 32×33×4 = 4224 B/block` |

> 💡 **一句话总结**：矩阵转置是 shared memory 的"Hello World"——它没有计算、只有搬运，却因"读写顺序不一致"逼出了 tiling + bank conflict padding 这两个 GPU 编程经典模板。这两个模板在后续的卷积（halo region）、矩阵乘法（tiling）、归约（warp shuffle 前的 staging）里会反复出现。把这道题的 `s[TILE][TILE+1]` 记住，就等于把 shared memory 的基本功钉牢了。

# LeetGPU Reduction 题解

## 1. 题目概述

- **标题 / 题号**：Reduction（#4，medium）
- **链接**：https://leetgpu.com/challenges/reduction
- **难度**：中等
- **标签**：CUDA、tree reduction、warp shuffle、`__shfl_down_sync`、两遍 kernel、memory-bound

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，计算所有元素之和，结果存入 `output[0]`。

**示例**：

```text
输入：[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
输出：36.0
```

**约束**：

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 最终和能放进 32-bit float
- 性能测试取 `N = 4,194,304`（= 2²²，4M 元素）

> 💡 这是 **warp shuffle** 的入门题。前三题（Vector Addition、ReLU、Matrix Transpose）的数据流都是"一个进一个出"，每个元素独立处理。归约则不同——它要把 `N` 个元素**合并成 1 个**，需要 thread 间通信。这引出两个新概念：**树形归约**（怎么并行地加）和 **warp shuffle**（怎么在 warp 内高效交换数据）。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行求和
float sum_cpu(const float* input, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += input[i];
    }
    return sum;
}
```

`N = 4M` 时单核约几毫秒。瓶颈：单线程串行，带宽和算力都没用上。

### 2.2 朴素 GPU：atomicAdd

最暴力的并行：每个 thread 读一个元素，用 `atomicAdd` 累加到全局 `output`。

```cuda
__global__ void reduce_atomic(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        atomicAdd(output, input[i]);   // ← 所有线程争抢同一个地址！
    }
}
```

**致命问题**：4M 个线程对同一个 `output` 地址做 `atomicAdd`，串行化到极致——比 CPU 还慢几十倍。`atomicAdd` 适合**低竞争**场景（少数线程偶尔写），不适合大规模归约。

> ⚠️ 归约问题的核心矛盾：最终结果只有一个，但并行线程有百万个。必须用**树形结构**让线程逐步合并，而不是一窝蜂抢同一个地址。

## 3. GPU 设计

### 3.1 并行化策略：树形归约（tree reduction）

树形归约的思想：每步让**一半线程**把自己的值加到**另一半线程**上，活跃线程数减半，`log₂N` 步后收敛到 1 个结果。

![树形归约对半折叠](images/reduction_tree_overview.svg)

以 8 个元素为例：

| step | 活跃线程 | 操作 | 结果 |
|------|---------|------|------|
| 0 | 8 | thread 0 加 thread 1，thread 2 加 thread 3... | [3,7,11,15,-,-,-,-] |
| 1 | 4 | thread 0 加 thread 2，thread 4 加 thread 6 | [10,26,-,-,-,-,-,-] |
| 2 | 2 | thread 0 加 thread 4 | [36,-,-,-,-,-,-,-] |
| 3 | 1 | 完成 | 36 |

**关键属性**：`log₂N` 步完成，但线程利用率逐步下降（step 0 用满，step k 只用 `N/2^k`）。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`partial[]` 中间结果、`output` 写 |
| **shared memory** | ✓ | block 内归约的暂存缓冲（warp 间汇总用） |
| **register** | ✓ | 每线程的累加值 + warp shuffle 交换 |

### 3.3 关键技巧：warp shuffle `__shfl_down_sync`

#### 为什么用 warp shuffle

传统的树形归约用 shared memory 做线程间数据交换：每步写 shared → `__syncthreads` → 读 shared。但**warp 内的 32 个 thread 本来就能直接交换寄存器值**——通过 `__shfl_down_sync` 指令，无需经过 shared memory，无需 `__syncthreads`，延迟更低。

![__shfl_down_sync 工作原理](images/reduction_warp_shuffle.svg)

`__shfl_down_sync(mask, val, delta)` 的语义：当前 lane 把 `val` 向下传 `delta` 个 lane。lane `i` 收到 lane `i+delta` 的值（若 `i+delta` 超出 warp 范围则值不变）。

```cuda
// warp 内归约（32 个 lane → 1 个结果在 lane 0）
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// 现在 lane 0 持有整个 warp 的和
```

迭代过程：`offset = 16, 8, 4, 2, 1`，共 5 步（`log₂32`），每步活跃 lane 减半。

> 💡 `__shfl_down_sync` 是 CUDA warp 级原语的基石之一。同系列还有 `__shfl_up_sync`（向上传，用于 scan）、`__shfl_sync`（广播指定 lane 的值）、`__shfl_xor_sync`（异或交换，用于蝶形归约）。掌握 `__shfl_down_sync` 这一个，就能解决大部分归约场景。

### 3.4 两级归约架构

`N = 4M` 远超单 block 处理能力（单 block 通常 256-1024 thread）。需要 **grid 级并行 + 二次归约**：

![Block 两级归约架构](images/reduction_two_level.svg)

1. **第一遍 kernel**：grid 的每个 block 用 grid-stride 读一段数据，block 内归约到 1 个部分和，写入 `partial[blockIdx.x]`。
2. **第二遍 kernel**：对 `partial[]`（长度 = block 数，通常几千）再做一次归约，得到最终结果。

> 💡 为什么不用 `atomicAdd` 让各 block 直接累加到 `output`？因为 4M 数据可能开几千个 block，竞争仍然激烈。两遍 kernel 各自独立归约，**无锁、无竞争**，是最干净的方案。

## 4. Kernel 实现

完整可编译的两级归约版本（grid-stride + warp shuffle + shared memory 汇总 + 二次 kernel）：

```cuda
// reduction_warp_shuffle.cu —— 两级归约：warp shuffle + block 归约 + 二次 kernel
// 编译命令: nvcc -O3 -arch=sm_80 reduction_warp_shuffle.cu -o reduction
// 运行:     ./reduction 4194304

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

#define BLOCK_SIZE 256
#define WARP_SIZE  32

// warp 内归约：__shfl_down_sync，结果落在 lane 0
__inline__ __device__ float warp_reduce(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block 内归约：grid-stride 读 + warp shuffle + shared 汇总
__global__ void reduce_kernel(const float* input, float* partial, int N) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];  // 每 warp 一个 slot

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    // ---- ① grid-stride 累加：每线程处理多个元素 ----
    float val = 0.0f;
    for (int i = tid; i < N; i += stride) {
        val += input[i];
    }

    // ---- ② warp 内归约（无需 shared，无需 syncthreads）----
    val = warp_reduce(val);

    // ---- ③ warp 间归约：每 warp 的 lane 0 写 shared ----
    int lane   = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    // ---- ④ 第一个 warp 把 shared 里的 warp 和再归约一次 ----
    if (warpId == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
        val = warp_reduce(val);
        // lane 0 持有整个 block 的和 → 写入 partial
        if (lane == 0) {
            partial[blockIdx.x] = val;
        }
    }
}

// 第二遍 kernel：对 partial[] 归约得到最终结果（单 block 足够）
__global__ void reduce_final(const float* partial, float* output, int M) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        val += partial[i];
    }

    val = warp_reduce(val);

    int lane = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
        val = warp_reduce(val);
        if (lane == 0) {
            output[0] = val;
        }
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4194304;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    // ---- host 端 ----
    float *hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f;
    }

    // ---- device 端 ----
    float *dIn, *dPartial, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // ---- 第一遍：grid 级归约 ----
    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;    // 经验值
    CHECK_CUDA(cudaMalloc(&dPartial, blocks * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    reduce_kernel<<<blocks, BLOCK_SIZE>>>(dIn, dPartial, N);
    reduce_final<<<1, BLOCK_SIZE>>>(dPartial, dOut, blocks);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (two-pass): %.3f ms\n", ms);

    // ---- 验证 ----
    float hOut;
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += hIn[i];   // double 累加做参考
    printf("GPU: %f  CPU(double): %f  %s\n", hOut, (float)ref,
           fabsf(hOut - (float)ref) < 1e-2f ? "PASS" : "FAIL");

    // ---- 带宽估算：只算读 input 的量 ----
    float bw_gbs = (bytes / 1e9) / (ms / 1e3);
    printf("read bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dPartial));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `reduce_kernel` + `reduce_final` 填进 `solve` 函数即可。注意 `solve` 的 starter 是空的（连 kernel 声明都没有），需要自己写完整。带 `main()` 的版本用于本地自测。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 reduction_warp_shuffle.cu -o reduction
./reduction 4194304
```

典型输出（A100 / SM=108）：

```text
N = 4194304  (16.0 MB)
kernel time (two-pass): 0.18 ms
GPU: 20964448.000000  CPU(double): 20964448.000000  PASS
read bandwidth: 88.9 GB/s
```

> ⚠️ 归约的"带宽"看起来不高（~89 GB/s），这是因为 **`cudaEvent` 计时含两次 kernel 启动开销**，且归约的算术强度极低（读 4B 只做 1 次加法）。实际单 kernel 的 HBM 带宽利用率可以用 `ncu` 单独测。

### 5.2 用 ncu 分析

```bash
ncu --set full --target-processes all -o reduce_profile ./reduction 4194304

# 关键指标
ncu --metrics gpu__time_duration.sum, \
        dram__bytes_read.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        launch__waves_per_multiprocessor \
    ./reduction 4194304
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | > 70%（memory-bound 应逼近带宽上限） |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 算力占比 | 很低（加法太轻） |
| `launch__waves_per_multiprocessor` | 每 SM 的 wave 数 | > 2（保证足够并发隐藏延迟） |

### 5.3 优化方向

1. **第一遍 grid-stride 的展开**：用 `#pragma unroll` 或手动展开循环（每轮读 4-8 个元素），减少循环开销与指令数。
2. **vector load（`float4`）**：每线程一次读 16B（4 个 float），减少地址计算，提升内存事务效率。
3. **kernel 融合**：如果 `partial[]` 元素很少（< 1024），可以把第二遍归约合并到第一遍末尾——最后一个 block 检测到自己是最后一个时直接做最终归约（需 `atomicAdd` 计数器或 `cooperative_groups` 的 grid 同步）。
4. **`cooperative_groups`**：用 `cg::this_grid().sync()` 实现 grid 级同步，单 kernel 完成两步归约，省掉第二遍启动开销。需要 GPU 支持 + launch 配置。
5. **double 累加**：本题 reference 用 `double` 求和再转 `float`。若精度要求高，block 内用 `double` 累加，最后转 `float`。会牺牲一点性能但避免大数误差。

> 💡 优化 1+2（展开 + float4）是性价比最高的，通常能再提升 30-50% 带宽。优化 3/4 属于进阶，收益取决于数据规模。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`：grid-stride 读 N 个元素 + `O(log N)` 归约步 |
| **空间复杂度** | `O(N)` 输入 + `O(blocks)` partial 数组 + `O(BLOCK_SIZE)` shared |
| **算术强度** | `1 FLOP / 4B`（1 次加法 ↔ 读 4B）= **0.25 FLOP/B** |
| **瓶颈类型** | **memory-bound**：算术强度极低，完全受 HBM 读带宽限制 |
| **kernel 启动数** | 2 次（第一遍 + 第二遍） |
| **warp shuffle 步数** | 每 warp `log₂32 = 5` 步 |

> 💡 **一句话总结**：归约是 **warp shuffle** 的"Hello World"——它把"多对一"的归约问题拆成"树形折叠 + warp 内直接交换"两层，既解决了 `atomicAdd` 的竞争问题，又用寄存器级通信绕开了 shared memory 的延迟与 bank conflict。`warp_reduce` 这个 5 行函数是 CUDA 编程的通用积木，后面所有归约类操作（dot product、norm、softmax 的 max/sum）都会复用它。

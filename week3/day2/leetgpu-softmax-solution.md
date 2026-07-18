# LeetGPU Softmax 题解（Week3 Day2）

> 本题解与 [Week2 Day4 的 Softmax 题解](../../leetgpu/week2/day4/leetgpu-softmax-solution.md) 内容相同，Week3 Day2 的教程链接指向此处。

## 1. 题目概述

- **标题 / 题号**：Softmax（#17，medium）
- **链接**：https://leetgpu.com/challenges/softmax
- **难度**：中等
- **标签**：CUDA、Softmax、Profiling、Memory-bound、Three-pass

**题意**：给定长度为 `N` 的浮点数组（支持 batch 多行），计算 softmax：`output[i] = exp(input[i]) / Σ exp(input[j])`。

**约束**：`1 ≤ N ≤ 1,000,000`，支持 batch 维度。

> 💡 与 [Week3 Day2 手写 Softmax + LayerNorm Kernel](../../../aiinfra/daily/week3/day2/README.md) 的关联：本题就是今天 row-wise Softmax 的直接实战。核心是 safe softmax（减 max）+ block 内两级归约（`blockReduceMax` + `blockReduceSum`）。

## 2. GPU 设计

一行一个 block（`gridDim.x = M`），block 内 256 线程协作：
1. Pass 1：grid-stride 扫描求 `row_max`，`blockReduceMax` 归约并广播
2. Pass 2：扫描求 `row_sum = Σ exp(x - max)`，`blockReduceSum` 归约并广播
3. Pass 3：归一化写出 `y[i] = exp(x[i] - max) / sum`

## 3. Kernel 实现

```cuda
// softmax.cu —— LeetGPU Softmax 提交版（三遍扫描 safe softmax）
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    // Pass 1: find max (numerical stability)
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, in_row[i]);
    // block reduce max via shared memory
    __shared__ float s_max;
    if (threadIdx.x == 0)
        s_max = -1e30f;
    __syncthreads();
    atomicMax((int*)&s_max, __float_as_int(max_val));
    __syncthreads();
    max_val = s_max;

    // Pass 2: exp + sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }
    __shared__ float s_sum;
    if (threadIdx.x == 0)
        s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, sum);
    __syncthreads();
    sum = s_sum;

    // Pass 3: normalize
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        out_row[i] /= sum;
}

extern "C" void solve(const float* input, float* output, int M, int N) {
    softmax_kernel<<<M, 256>>>(input, output, M, N);
}
```

### 3.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名的提交版本。这里改用 warp shuffle 完成 block 内 `max` 与 `sum` 两级归约，避免对浮点位做 `atomicMax`，可同时兼容正/负输入。

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < (BLOCK_SIZE >> 5)) ? shared[lane] : -1e30f;
        val = warp_reduce_max(val);
        if (lane == 0)
            shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0)
        shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < (BLOCK_SIZE >> 5)) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0)
            shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    if (row >= M)
        return;

    const float* in_row = input + row * N;
    float* out_row = output + row * N;

    __shared__ float shared[BLOCK_SIZE >> 5];

    // Pass 1: 求行内 max（数值稳定性）
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
        max_val = fmaxf(max_val, in_row[i]);
    max_val = block_reduce_max(max_val, shared);

    // Pass 2: 求 exp 之和
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE) {
        float e = expf(in_row[i] - max_val);
        out_row[i] = e;
        sum += e;
    }
    sum = block_reduce_sum(sum, shared);

    // Pass 3: 归一化
    for (int i = threadIdx.x; i < N; i += BLOCK_SIZE)
        out_row[i] /= sum;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int M, int N) {
    softmax_kernel<<<M, BLOCK_SIZE>>>(input, output, M, N);
    cudaDeviceSynchronize();
}
```

### 3.2 归约积木代码详解

上面的 kernel 核心是**两次块归约**（`block_reduce_max` + `block_reduce_sum`），它们都由相同的两块积木组成：`warp_reduce_*`（warp 内 shuffle 归约）和 `block_reduce_*`（warp 间 shared memory 汇总 + 广播）。

#### Warp 级归约：`__shfl_down_sync`

![Warp Shuffle 归约详解](../../images/softmax_warp_shuffle.svg)

> **图：**`__shfl_down_sync` **逐步归约。** 左侧 `warp_reduce_sum` 以 8 个 lane 为例（实际 32 个），offset 从 4→2→1 折半，每步 lane[i] 与 lane[i+offset] 运算，最终 lane 0 持有全局 sum。右侧 `warp_reduce_max` 结构完全对称，只把 `+=` 换成 `fmaxf`。

```cuda
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

- `offset` **折半**：从 16→8→4→2→1，5 步完成 32→1 归约（`log₂32 = 5`），每步数据量减半
- `__shfl_down_sync(mask, val, offset)`：lane i 收到 lane (i+offset) 的 `val`；超出 warp 边界的 lane 值不变
- `#pragma unroll`：编译器展开 5 次迭代，零循环开销，便于指令级并行
- **结果只在 lane 0**：归约后其余 lane 的值是中间结果，需用 shared memory 广播给全 block
- `warp_reduce_max` **完全对称**：只把 `+=` → `fmaxf`，shuffle 机制不变

> 💡 Shuffle 在寄存器间直接传递数据，不经过 shared memory，零 bank conflict、零内存延迟，是 GPU 上最快的归约方式。

#### Block 级归约：warp 间 shared memory 汇总 + 广播

![Block 归约详解](../../images/softmax_block_reduce.svg)

> **图：Block 归约三阶段流程。** 阶段 1：8 个 warp 各自做 warp_reduce，lane 0 写入 `shared[warpId]`；阶段 2：`__syncthreads` 后，warp 0 读 `shared[0..7]` 再做一次 warp_reduce，结果写入 `shared[0]`；阶段 3：第二次 `__syncthreads` 后，全 block 从 `shared[0]` 读取最终结果。

```cuda
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);   // lane = threadIdx.x % 32
    int warpId = threadIdx.x >> 5;               // warpId = threadIdx.x / 32
    val = warp_reduce_sum(val);                  // 阶段 1a：warp 内归约
    if (lane == 0)
        shared[warpId] = val;                    // 阶段 1b：lane 0 写 shared
    __syncthreads();                             // 屏障：等 8 个 warp 都写完
    if (warpId == 0) {                           // 阶段 2a：仅 warp 0 执行
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;  // 读 8 个 warp 结果
        val = warp_reduce_sum(val);              // 阶段 2b：对 8 个值再归约
        if (lane == 0)
            shared[0] = val;                     // 阶段 2c：写入广播槽
    }
    __syncthreads();                             // 屏障：等 warp 0 写完
    return shared[0];                            // 阶段 3：全 block 读 shared[0]
}
```

| 步骤 | 作用 |
|------|------|
| **阶段 1a：warp 归约** | 每 warp 内 32 个值归约为 1 个，结果在 lane 0 |
| **阶段 1b：写 shared** | 8 个 warp 的 lane 0 各写一个 slot → `shared[0..7]` 填满 |
| **屏障 1** | `__syncthreads()` 确保 8 个 warp 都写完后 warp 0 才读 |
| **阶段 2a：warp 0 读取** | warp 0 的 lane 0~7 读 `shared[0..7]`，lane 8~31 填默认值 |
| **阶段 2b：最终归约** | 对 8 个 warp 结果再做 warp_reduce（3 步：offset 4→2→1） |
| **阶段 2c：写广播槽** | lane 0 写最终结果到 `shared[0]` |
| **屏障 2** | `__syncthreads()` 确保 warp 0 写完后全 block 才读 |
| **阶段 3：广播** | 全 block 256 个 thread 同时读 `shared[0]`，获得最终结果 |

> 💡 `block_reduce_max` 与 `block_reduce_sum` 结构完全相同，只把 `warp_reduce_sum` → `warp_reduce_max`，默认值 `0.0f` → `-INFINITY`（max 归约的幺元是负无穷）。

## 4. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(M×N)`，三趟扫描 |
| 算术强度 | `~3 FLOP / 8B` → memory-bound |
| 瓶颈类型 | **memory-bound**：`DRAM% >> SM%` |

> 💡 完整版题解（含 online 两遍扫描优化、Roofline 分析）见 [Week2 Day4 Softmax 题解](../../leetgpu/week2/day4/leetgpu-softmax-solution.md)。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 50 | [RMS Normalization](https://leetgpu.com/challenges/rms-normalization) | 中等 | — | RMS Norm，归约 + 归一化变体 |
| 6 | [Softmax Attention](https://leetgpu.com/challenges/softmax-attention) | 中等 | — | fused softmax+matmul，数值稳定进阶 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约，softmax 的基础组件 |
| 40 | [Batch Normalization](https://leetgpu.com/challenges/batch-normalization) | 中等 | — | Batch Norm，mean/var 归约归一化 |

> 💡 **选题思路**：三遍 kernel + 数值稳定，练习归约与归一化的融合。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

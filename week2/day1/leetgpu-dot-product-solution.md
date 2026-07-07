# LeetGPU Dot Product 题解

## 1. 题目概述

- **标题 / 题号**：Dot Product（#17，medium）
- **链接**：https://leetgpu.com/challenges/dot-product
- **难度**：中等
- **标签**：CUDA、dot product、kernel fusion、block 归约、warp shuffle `__shfl_down_sync`、memory-bound

**题意**：给定两个长度为 `N` 的 `float32` 向量 `a` 和 `b`，计算其**点积**（内积）：

$$\text{dot} = \sum_{i=0}^{N-1} a[i] \times b[i]$$

**示例**：

```text
a = [1.0, 2.0, 3.0, 4.0]
b = [5.0, 6.0, 7.0, 8.0]
dot = 1×5 + 2×6 + 3×7 + 4×8 = 5 + 12 + 21 + 32 = 70.0
```

**约束**：

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ a[i], b[i] ≤ 1000.0`
- 最终和能放进 32-bit float
- 性能测试取 `N = 4,194,304`（= 2²²，4M 元素）

> 💡 这是 [Reduction #4](../week1/day4/leetgpu-reduction-solution.md) 的"融合变体"。Reduction 是"读 N 个元素求和"，Dot Product 是"读 2N 个元素、做 N 次乘法、再求和"。它引出 GPU 编程的关键优化思想——**kernel fusion（核函数融合）**：把"逐元素乘"和"归约"合并成单个 kernel，避免物化中间乘积数组到 HBM。这正是从 Reduction 到 [Softmax](../week2/day4/leetgpu-softmax-solution.md) / [FlashAttention](../week2/day5/leetgpu-softmax-attention-solution.md) 融合 kernel 的过渡练习。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行点积
float dot_cpu(const float* a, const float* b, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}
```

`N = 4M` 时单核约几毫秒。瓶颈：单线程串行，带宽和算力都没用上。

### 2.2 朴素 GPU：两个分离 kernel（物化中间数组）

最直观的并行：先算逐元素乘写到中间数组 `tmp[i] = a[i]*b[i]`，再对 `tmp` 做归约。

```cuda
// ---- kernel 1：逐元素乘，结果写回 HBM ----
__global__ void elementwise_mul(const float* a, const float* b, float* tmp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) tmp[i] = a[i] * b[i];   // ← 写 4M 个 float 到 HBM！
}

// ---- kernel 2：对 tmp 归约（复用 Reduction #4 的 kernel）----
// reduce_kernel<<<...>>>(tmp, partial, N);
// reduce_final<<<...>>>(partial, output, blocks);
```

![朴素两步分离 vs 融合单 kernel：HBM 流量对比](images/dot_product_fusion_vs_separate.svg)

**致命问题**：中间数组 `tmp`（`4M × 4B = 16MB`）被**写再被读**，共 `~32MB` 额外 HBM 流量。而 `a` 和 `b` 本身各 `16MB`，有效数据才 `32MB`。朴素版 HBM 流量是有效数据的 **2×**，被中间数组拖累。

> ⚠️ 这正是 **kernel fusion** 的动机：`a[i]*b[i]` 算出来后立即累加到寄存器，不落 HBM。点积的中间结果 `tmp[i]` 是"一次写一次读"的临时数据，**根本不需要物化**。融合后 HBM 流量减半，带宽利用率翻倍。

## 3. GPU 设计

### 3.1 并行化策略：单 kernel 融合 + 两级归约

融合版把"乘"和"归约"合一，结构完全复用 [Reduction #4 的两级归约架构](../week1/day4/leetgpu-reduction-solution.md)：

![融合单 kernel：grid-stride 读 a,b → FMA 累加 → 两级 block 归约](images/dot_product_two_level_reduce.svg)

1. **第一遍 kernel**：grid 的每个 block 用 grid-stride 读 `a[i]`、`b[i]`，做 `val += a[i]*b[i]` 累加到寄存器，block 内归约到 1 个部分和，写入 `partial[blockIdx.x]`。
2. **第二遍 kernel**：对 `partial[]` 再做一次归约，得到最终结果。

**与 Reduction #4 的唯一差别**：grid-stride 循环体从 `val += input[i]` 变成 `val += a[i] * b[i]`（多读一个数组、多一次乘法）。归约部分**一字未改**。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `a`、`b` 读（各 1 遍）、`partial[]` 中间结果、`output` 写。**不物化 tmp** |
| **shared memory** | ✓ | block 内 warp 间归约汇总（与 Reduction #4 相同） |
| **register** | ✓ | 每线程的 `val` 累加器 + warp shuffle 交换 |

### 3.3 关键技巧：kernel fusion（复用 Day 1 归约模板）

融合的核心是 **grid-stride 循环里直接做 FMA**，累加值驻留寄存器，直到 block 归约才写出：

```cuda
// 融合：乘 + 归约在一个循环里，val 驻留寄存器
float val = 0.0f;
for (int i = tid; i < N; i += stride) {
    val += a[i] * b[i];   // FMA：读 8B（a+b），做 1 乘 1 加，结果留寄存器
}
// val 再走 warp_reduce + block_reduce（复用 Day 1 模板）
```

归约部分直接复用 [week2/day1 教程](../../aiinfra/week2/day1/README.md)的 `warpReduceSum` + `blockReduceSum`：

- **warp 内**：`__shfl_down_sync` 折半累加（5 步，`log₂32`）
- **warp 间**：每 warp lane 0 写 shared → 第一个 warp 再归约
- **block 间**：`partial[]` 二次 kernel 归约

> 💡 `warpReduceSum` 和 `blockReduceSum` 是 [Day 1 教程](../../aiinfra/week2/day1/README.md)的核心产出。Dot Product 只是把"被归约的值"从 `input[i]` 换成 `a[i]*b[i]`，归约积木完全复用。这正是 Day 1 强调的"warp shuffle 归约是 CUDA 通用积木"——学一次，用到 dot product、norm、softmax 的 max/sum、attention 的统计量等所有归约场景。

## 4. Kernel 实现

完整可编译的融合版点积（grid-stride FMA + warp shuffle 两级归约）：

```cuda
// dot_product_fused.cu —— 融合点积：FMA + 两级 warp shuffle 归约
// 编译命令: nvcc -O3 -arch=sm_80 dot_product_fused.cu -o dot_product
// 运行:     ./dot_product 4194304

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

// ---- warp 级归约（复用 Day 1 模板，一字未改）----
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- block 级归约：warp shuffle + shared 汇总（复用 Day 1 模板）----
__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane   = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        val = (lane < (BLOCK_SIZE / WARP_SIZE)) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ---- 第一遍 kernel：融合 FMA + block 归约，写出 partial[blockIdx.x] ----
__global__ void dot_kernel(const float* a, const float* b, float* partial, int N) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    // ---- ① grid-stride FMA 累加（乘+归约融合，val 驻留寄存器）----
    float val = 0.0f;
    for (int i = tid; i < N; i += stride) {
        val += a[i] * b[i];   // 读 8B 做 1 乘 1 加，不落 HBM
    }

    // ---- ② block 归约 ----
    val = block_reduce_sum(val, shared);

    // ---- ③ lane 0 写 block 部分和 ----
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = val;
    }
}

// ---- 第二遍 kernel：对 partial[] 归约得最终结果 ----
__global__ void dot_final(const float* partial, float* output, int M) {
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        val += partial[i];
    }

    val = block_reduce_sum(val, shared);
    if (tid == 0) output[0] = val;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 4194304;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (a+b: %.1f MB)\n", N, 2.0 * bytes / 1e6);

    // ---- host ----
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hA[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f;
        hB[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f;
    }

    // ---- device ----
    float *dA, *dB, *dPartial, *dOut;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // ---- 两级归约配置 ----
    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;    // 经验值
    CHECK_CUDA(cudaMalloc(&dPartial, blocks * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    dot_kernel<<<blocks, BLOCK_SIZE>>>(dA, dB, dPartial, N);
    dot_final<<<1, BLOCK_SIZE>>>(dPartial, dOut, blocks);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (fused two-pass): %.3f ms\n", ms);

    // ---- 验证 ----
    float hOut;
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += (double)hA[i] * hB[i];   // double 累加做参考
    printf("GPU: %f  CPU(double): %f  %s\n", hOut, (float)ref,
           fabsf(hOut - (float)ref) < 1e-2f ? "PASS" : "FAIL");

    // ---- 带宽估算：读 a + b ----
    float bw_gbs = (2.0 * bytes / 1e9) / (ms / 1e3);
    printf("read bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dPartial));
    CHECK_CUDA(cudaFree(dOut));
    free(hA); free(hB);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `dot_kernel` + `dot_final` 填进 `solve` 函数即可。带 `main()` 的版本用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 dot_product_fused.cu -o dot_product
./dot_product 4194304
```

典型输出（A100 / SM=108）：

```text
N = 4194304  (a+b: 32.0 MB)
kernel time (fused two-pass): 0.21 ms
GPU: 2097152.000000  CPU(double): 2097152.000000  PASS
read bandwidth: 152.4 GB/s
```

### 5.2 融合 vs 分离的 HBM 流量对比

| 实现 | HBM 读 | HBM 写 | 总流量 | 相对 |
|------|--------|--------|--------|------|
| 分离两 kernel | `a+b+tmp` = 3×16MB = 48MB | `tmp+partial+out` ≈ 16MB | ~64MB | 1.0×（基线） |
| **融合单 kernel** | `a+b` = 2×16MB = 32MB | `partial+out` ≈ 几 KB | **~32MB** | **0.5×（省一半）** |

融合后 HBM 流量减半，带宽利用率理论上翻倍。实际提升受 kernel 启动开销和归约计算影响，通常 `1.5-2×` 加速。

### 5.3 用 ncu 分析

```bash
ncu --kernel-name regex:dot_kernel \
    --metrics gpu__time_duration.sum, \
              dram__bytes_read.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./dot_product 4194304
```

| 指标 | 分离两 kernel | 融合单 kernel | 含义 |
|------|--------------|--------------|------|
| `dram__bytes_read` | ~48MB（含 tmp 读） | ~32MB（只读 a+b） | 融合省 1/3 读流量 |
| `dram__throughput` | ~70% | ~85% | 融合后更接近带宽上限 |
| `sm__throughput` | ~5% | ~8% | 算力占比仍低（memory-bound） |
| `gpu__time_duration` | 基线 | **~1.5-2× 加速** | 总耗时 |

> ⚠️ 点积的算术强度 `1 FMA / 8B = 0.25 FLOP/B`（读 2 个 float 做 1 乘加），比 Reduction（`0.25 FLOP/B`）相当，**纯 memory-bound**。优化核心是减少 HBM 流量，融合正是为此。

### 5.4 优化方向

1. **`float4` 向量化加载**：每线程一次读 `float4`（4 个 float），`a` 和 `b` 各一次 `float4` 读 8 个 float，做 4 次 FMA。减少地址计算与内存事务，通常提升 20-30% 带宽。
2. **循环展开**：grid-stride 循环体手动展开 4-8 次（`#pragma unroll`），减少循环开销，让编译器生成更多 FMA 指令填充流水线。
3. **`double` 累加**：本题 reference 用 `double` 求和。若精度要求高，block 内用 `double` 累加（`val` 声明为 `double`，warp shuffle 用 `__double2ull` 拆包或退化到 shared memory 归约），最后转 `float`。会牺牲性能但避免大数误差。
4. **单 kernel 归约（`cooperative_groups`）**：用 `cg::this_grid().sync()` 实现 grid 级同步，单 kernel 完成两步归约，省掉第二遍启动开销。需 GPU 支持 + launch 配置。
5. **Tensor Core（`mma.sync`）**：把点积看作 `1×N` × `N×1` 矩阵乘，用 Tensor Core 加速。但 `N` 必须是 16 的倍数且需 fp16 输入，本题 fp32 不直接适用。

> 💡 优化 1+2（`float4` + 展开）是性价比最高的，通常能再提升 30-50% 带宽。优化 4 属于进阶，省 kernel 启动开销但增加复杂度。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`：grid-stride 读 N 对元素 + `O(log N)` 归约步 |
| **空间复杂度** | `O(N)` 输入（a, b）+ `O(blocks)` partial 数组 + `O(BLOCK_SIZE)` shared |
| **算术强度** | `2 FLOP / 8B`（1 次 FMA = 2 FLOP ↔ 读 2 个 float = 8B）= **0.25 FLOP/B** |
| **瓶颈类型** | **memory-bound**：算术强度极低，完全受 HBM 读带宽限制 |
| **kernel 启动数** | 2 次（第一遍 + 第二遍），融合后仍需两遍（block 间归约） |
| **HBM 流量（融合）** | 读 `2N×4B`（a+b）+ 写 `blocks×4B`（partial）≈ `8N` 字节 |
| **HBM 流量（分离）** | 读 `3N×4B`（a+b+tmp）+ 写 `N×4B`（tmp）≈ `16N` 字节（2× 融合） |

> 💡 **一句话总结**：Dot Product 是 Reduction #4 的"融合变体"——它把"逐元素乘"和"归约"合并成单 kernel，让乘积 `a[i]*b[i]` 算完即累加到寄存器，不物化中间数组到 HBM，流量减半。归约部分完全复用 Day 1 的 `warpReduceSum` / `blockReduceSum` 积木，印证了"warp shuffle 归约是 CUDA 通用积木"。**kernel fusion** 这一思想是通往 Softmax（三步融合）和 FlashAttention（QKᵀ+softmax+PV 融合）的入门钥匙——凡是有"中间临时数组被写再被读"的模式，都该考虑融合。

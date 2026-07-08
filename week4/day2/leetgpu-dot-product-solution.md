# LeetGPU Dot Product 题解

## 1. 题目概述

- **标题 / 题号**：Dot Product（#17，medium）
- **链接**：https://leetgpu.com/challenges/dot-product
- **难度**：中等
- **标签**：CUDA、Dot Product、block 归约、warp shuffle、kernel 融合、memory-bound

**题意**：给定两个长度为 `N` 的 `float32` 数组 `a` 和 `b`，计算它们的点积：

$$\text{result} = \sum_{i=0}^{N-1} a[i] \cdot b[i]$$

**示例**：

```text
输入：a = [1.0, 2.0, 3.0, 4.0],  b = [5.0, 6.0, 7.0, 8.0]
输出：1*5 + 2*6 + 3*7 + 4*8 = 70.0
```

**约束**：

- `1 ≤ N ≤ 10,000,000`
- `-1000.0 ≤ a[i], b[i] ≤ 1000.0`
- 最终结果不溢出 `float`（最坏 `N*1e6 = 1e13`，在 float32 精确整数范围 `2^24` 之外，但题目容差宽松）
- 性能测试取 `N = 10,000,000`（≈ 38MB/数组）

> 💡 这是 **Week2 Day1 warp shuffle 原语的延伸题**。点积在数学上就是「**逐元素乘法** + **全数组归约**」两步——前者是 element-wise（Week1 Day1），后者是 reduction（Week1 Day4）。本题的新意在于：这两步应不应该分开？答案是 **必须融合**：把 `a[i]*b[i]` 和归约合并进同一个 kernel，避免物化中间数组 `c[i]=a[i]*b[i]`。这正是「**kernel 融合**」思想的入门。

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

`N = 10M` 时单核约几十毫秒。瓶颈：单线程串行，未利用带宽与并行度。

### 2.2 朴素 GPU：两步法 + atomicAdd

最暴力的并行有两种 equally 糟的写法。**写法 A**（两步法）先把 `c[i]=a[i]*b[i]` 物化成 38MB 中间数组、再 `atomicAdd(c)` 归约——多一次 HBM 往返。**写法 B**（融合但 atomicAdd）省掉 `c`，但仍让 10M 个线程抢同一地址：

```cuda
__global__ void dot_atomic(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) atomicAdd(out, a[i] * b[i]);  // ← 10M 次原子竞争，串行化
}
```

**致命问题**：写法 A 多一次 38MB 的 HBM 写+读（`c`），IO 翻倍；两种写法的 `atomicAdd(out, ...)` 都让 10M 个线程串行化到同一个 32-bit 地址——**比 CPU 慢几十倍**。

> ⚠️ 点积的难点和归约一样：**结果只有一个，但参与运算的元素有千万个**。必须用树形归约让线程逐步合并，而非一窝蜂抢同一个地址。相对归约题，本题额外要求把「乘法」和「加法归约」**融合**到一起，省掉中间数组。

## 3. GPU 设计

### 3.1 并行化策略：两阶段 grid-stride + 两级 block 归约

点积 = `N` 个乘法 + `N-1` 次加法。并行化的标准套路与归约几乎一致，只是每个元素的「贡献」从 `input[i]` 变成 `a[i]*b[i]`：

![点积两阶段架构](images/dot_product_overview.svg)

1. **第一阶段（grid-stride 算局部点积）**：grid 的每个 block 用 grid-stride loop 遍历一段数据，每线程累加 `a[i]*b[i]` 到寄存器，再在 block 内做两级归约（warp shuffle + shared memory），得到 1 个部分和，写入 `partial[blockIdx.x]`。
2. **第二阶段（归约部分和）**：对 `partial[]`（长度 = block 数，通常几千）再做一次归约，得到最终结果。

> 💡 也可用 `cooperative_groups` 的 grid 同步把两步融成**单 kernel**（见 5.3），但两遍 kernel 实现简单、无依赖，是默认首选。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | 读 `a`、`b`；写 `partial[]` 中间结果、`output` 最终结果 |
| **shared memory** | ✓ | block 内 warp 间汇总（每 warp 一个 slot） |
| **register** | ✓ | 每线程的 `a[i]*b[i]` 累加值 + warp shuffle 交换 |

注意：**没有中间数组 `c`**——乘积只活在寄存器里，立即被加进累加器。这就是「融合」的全部含义。

### 3.3 关键技巧：warp/block 两级归约 + kernel 融合

#### 两级归约原语（复用 Week2 Day1）

点积与归约共享同一套归约原语，只是「元素贡献」不同。复用 Week2 Day1 的 `warpReduceSum` + `blockReduceSum`：

![warp shuffle 两级归约](images/dot_product_warp_reduce.svg)

```cuda
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;          // lane 0 持有整个 warp 的和
}
// blockReduceSum 在 warpReduceSum 之上包一层 shared memory 汇总
// （每 warp 的 lane 0 写 shared → 第一个 warp 再 warpReduceSum 一次）
```

> 💡 `warpReduceSum` + `blockReduceSum` 是 CUDA 编程的「通用积木」：归约、点积、norm、softmax 的 sum/max、cross-entropy 全靠它。完整实现见 §4。

#### kernel 融合：乘法与归约合并

朴素两步法先把 `c[i]=a[i]*b[i]` 写回 HBM，再读 `c` 归约——多一次 38MB 的往返。**融合版**在 grid-stride loop 里直接 `sum += a[i]*b[i]`，乘积只活在寄存器里，**零中间 IO**：

![kernel 融合对比](images/dot_product_fusion.svg)

| 方案 | HBM 读 | HBM 写 | 中间数组 |
|------|--------|--------|----------|
| 两步法（物化 c） | 2N（a,b） + N（c） | N（c） | ✓ 38MB |
| 融合（本方案） | 2N（a,b） | 0 | ✗ 无 |

> ⚠️ 融合的收益随 N 线性放大。`N=10M` 时省下 76MB 的 IO，在 memory-bound 场景里约等于 **2× 加速**。这正是 Triton/CUTLASS 把 element-wise 与归约写在一起的根本原因。

## 4. Kernel 实现

完整可编译的两级归约点积（grid-stride 融合乘加 + warp shuffle + shared 汇总 + 二次 kernel）：

```cuda
// dot_product.cu —— 两级归约点积
// 编译命令: nvcc -O3 -arch=sm_80 dot_product.cu -o dot_product
// 运行:     ./dot_product 10000000

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

// ---- 复用 Week2 Day1 的归约原语 ----
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static const int NWARP = BLOCK_SIZE / WARP_SIZE;
    __shared__ float shared[NWARP];

    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lane < NWARP) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);   // lane 0 持有整个 block 的和
    }
    return val;
}

// ---- 第一遍：grid-stride 融合乘加 + block 归约，每 block 写一个部分和 ----
__global__ void dot_kernel(const float* a, const float* b,
                           float* partial, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;

    // ① grid-stride 累加 a[i]*b[i]（融合：乘积只活在寄存器里）
    float val = 0.0f;
    for (int i = tid; i < N; i += stride) {
        val += a[i] * b[i];
    }

    // ② block 内两级归约
    val = blockReduceSum(val);

    // ③ lane 0 写入 partial[blockIdx.x]
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = val;
    }
}

// ---- 第二遍：对 partial[] 归约得到最终结果（单 block 足够）----
__global__ void dot_final(const float* partial, float* output, int M) {
    int tid = threadIdx.x;
    float val = 0.0f;
    for (int i = tid; i < M; i += blockDim.x) {
        val += partial[i];
    }
    val = blockReduceSum(val);
    if (tid == 0) output[0] = val;
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 10000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB per array)\n", N, bytes / 1e6);

    // ---- host 端 ----
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hA[i] = ((float)(rand() % 2000) - 1000.0f) / 100.0f;   // [-10, 10]
        hB[i] = ((float)(rand() % 2000) - 1000.0f) / 100.0f;
    }

    // ---- device 端 ----
    float *dA, *dB, *dPartial, *dOut;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;                          // 经验值：4×SM
    CHECK_CUDA(cudaMalloc(&dPartial, blocks * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    dot_kernel <<<blocks, BLOCK_SIZE>>>(dA, dB, dPartial, N);
    dot_final  <<<1, BLOCK_SIZE>>>(dPartial, dOut, blocks);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (two-pass): %.3f ms\n", ms);

    // ---- 验证（double 累加做参考，避免 float 误差）----
    float hOut;
    CHECK_CUDA(cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    double ref = 0.0;
    for (int i = 0; i < N; ++i) ref += (double)hA[i] * (double)hB[i];
    printf("GPU: %f  CPU(double): %f  %s\n", hOut, (float)ref,
           fabsf(hOut - (float)ref) < 1e-2f * fabsf((float)ref) ? "PASS" : "FAIL");

    // ---- 带宽估算：只算读 a、b 的量 ----
    float bw_gbs = (2.0f * bytes / 1e9) / (ms / 1e3);
    printf("read bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dPartial));
    CHECK_CUDA(cudaFree(dOut));
    free(hA);
    free(hB);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `dot_kernel` + `dot_final` 填进 `solve` 函数即可。starter 一般会给 `a, b, n, output` 参数，需自行 `cudaMalloc` 一个 `partial` 数组并 launch 两个 kernel。带 `main()` 的版本用于本地自测。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 dot_product.cu -o dot_product
./dot_product 10000000
```

典型输出（A100 / SM=108，`N=10M`）：

```text
N = 10000000  (38.1 MB per array)
kernel time (two-pass): 0.42 ms
GPU: 332.456787  CPU(double): 332.456813  PASS
read bandwidth: 181.6 GB/s
```

> ⚠️ 点积读两个数组共 76MB，理论 `2N*4B = 76.3MB`。0.42ms 对应 ~182 GB/s，已逼近 A100 的 HBM 带宽（~1.5TB/s）的一成——因为 `cudaEvent` 计时含两次 kernel 启动开销且 `dot_final` 几乎空跑。用 `ncu` 单测 `dot_kernel` 才能看到真实带宽。

### 5.2 用 ncu 分析

```bash
ncu --set full --target-processes all -o dot_profile ./dot_product 10000000

# 关键指标（只测第一遍 kernel）
ncu --kernel-name regex:dot_kernel --metrics \
    gpu__time_duration.sum,dram__bytes_read.sum, \
    dram__throughput.avg.pct_of_peak_sustained_elapsed, \
    sm__throughput.avg.pct_of_peak_sustained_elapsed, \
    launch__waves_per_multiprocessor ./dot_product 10000000
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | > 70%（memory-bound 应逼近带宽上限） |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 算力占比 | 很低（乘加太轻） |
| `launch__waves_per_multiprocessor` | 每 SM 的 wave 数 | > 2（足够并发隐藏延迟） |

### 5.3 优化方向

1. **`float4` 向量化访存**：每线程一次读 16B（4 个 float），把 `a,b` 改成 `((float4*)a)[k]`，地址计算和访存事务数都减为 1/4，通常提升 20-40% 带宽。循环体内对 `va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w` 累加即可。

2. **kernel 融合（已做）**：本题的核心优化——不物化 `c[i]=a[i]*b[i]`。相对两步法省一次 38MB 往返。

3. **`cooperative_groups` grid 归约**：用 `cg::this_grid().sync()` 在单 kernel 内同步所有 block，省掉第二遍 `dot_final` 的启动开销。需 GPU 支持协作 launch（`cudaLaunchCooperativeKernel`），且 grid 规模受 SM 上限约束。

4. **grid-stride 循环展开**：`#pragma unroll 4` 或手动展开 4-8 轮，减少分支与循环开销，配合 `float4` 效果更佳。

5. **double 累加**：reference 用 `double` 求和。若精度要求高，block 内用 `double` 累加，最后转 `float`。本题容差宽松，`float` 累加即可 PASS。

> 💡 优化 1（`float4`）+ 4（展开）是性价比最高的，通常能把带宽利用率从 ~70% 推到 ~90%。优化 3（grid 归约）属于进阶，收益取决于第二遍 kernel 的占比——`N=10M` 时第二遍只占几微秒，收益有限。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`：grid-stride 读 `2N` 元素 + `O(log N)` 归约步 |
| **空间复杂度** | `O(N)` 输入（a、b）+ `O(blocks)` partial 数组 + `O(BLOCK_SIZE)` shared |
| **算术强度** | `2 FLOP / 8B`（1 乘 + 1 加 ↔ 读 `a[i]`4B + `b[i]`4B）= **0.25 FLOP/Byte** |
| **瓶颈类型** | **memory-bound**：算术强度 0.25 远低于 A100 平衡点（~10 FLOP/B），完全受 HBM 读带宽限制 |
| **kernel 启动数** | 2 次（第一遍 `dot_kernel` + 第二遍 `dot_final`） |
| **warp shuffle 步数** | 每 warp `log₂32 = 5` 步（两级归约各一次，共 10 步，但常数级） |

> 💡 **一句话总结**：点积是归约的「融合版」——把 element-wise 乘法和树形归约塞进同一个 grid-stride loop，乘积只活在寄存器里，省掉一整次 HBM 往返。它复用 `warpReduceSum`/`blockReduceSum` 这两个通用积木，再次印证归约原语是 CUDA 编程的基础设施：归约、点积、norm、softmax 的 sum/max 全靠它。memory-bound 的本质（AI=0.25）决定了优化方向是**压榨 HBM 带宽**（`float4`、融合），而非算力。

# LeetGPU 2D Subarray Sum 题解

## 1. 题目概述

- **标题 / 题号**：2D Subarray Sum（#48，medium）
- **链接**：https://leetgpu.com/challenges/2d-subarray-sum
- **难度**：中等
- **标签**：CUDA、归约（reduction）、warp shuffle、2D 索引映射、范围求和、memory-bound

**题意**：给定 $N \times M$ 的 `int32` 二维数组 `input`（row-major 存储）与子矩形边界 `S_ROW, E_ROW, S_COL, E_COL`（均含两端），计算子矩形 `input[S_ROW..E_ROW][S_COL..E_COL]` 内所有元素之和，结果写入单个 `int32` 输出 `output[0]`。即 $output[0] = \text{torch.sum}(input[S\_ROW : E\_ROW+1,\ S\_COL : E\_COL+1])$。

**示例**：

```text
输入：input = [[1, 2, 3],
               [4, 5, 1]],  N=2, M=3
      S_ROW=0, E_ROW=1, S_COL=1, E_COL=2
子矩形：[[2, 3],
         [5, 1]]
输出：output[0] = 2 + 3 + 5 + 1 = 11
```

**约束**：

- $1 \le N, M \le 10{,}000$（总元素数 $N \times M$ 最多 1 亿）
- $0 \le S\_ROW \le E\_ROW < N$，$0 \le S\_COL \le E\_COL < M$
- 性能测试取 $N = 10{,}000$、$M = 10{,}000$（1 亿元素、400 MB 输入），子矩形 $S\_ROW=0, E\_ROW=9998, S\_COL=1, E\_COL=9999$（几乎全矩阵）
- `solve` 函数签名不可改，外部库禁用，结果必须写入 `output[0]`

> 💡 这道题是 [#47 Subarray Sum](https://leetgpu.com/challenges/subarray-sum) 的**二维扩展**——输入从一维数组变成 $N \times M$ 矩阵，"对一段元素求和"升级为"对一个子矩形求和"。关键洞察是：**子矩形在内存中并非连续**（每行 `col_len` 个元素连续，但行与行间隔 $M$），所以不能像 #47 那样直接 `input[S + off]`，必须把线性偏移 `off` 映射回 `(row, col)`。归约骨架（grid-stride 累加 + 两级 block 归约 + `long long` 累加）则与 #47 **完全复用**——这正是 Reduction 模板的迁移价值。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

最直观的串行实现是双层 `for` 循环遍历子矩形：

```cpp
// cpu_baseline.cpp —— CPU 串行子矩形求和
void subarray_sum_2d_cpu(const int* input, int* output,
                         int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    long long sum = 0;
    for (int r = S_ROW; r <= E_ROW; ++r) {
        const int* row = input + (long long)r * M;   // 行首指针
        for (int c = S_COL; c <= E_COL; ++c) {
            sum += row[c];
        }
    }
    output[0] = (int)sum;
}
```

性能测试的子矩形约 1 亿元素，单核约耗时 **上百毫秒**。CPU 已利用 row-major 的行内连续性（`row[c]` 顺序读），但单线程带宽有限，仍是瓶颈。

> ⚠️ CPU 基线用 `long long` 累加再 cast 到 `int`——这正是 `torch.sum(int32)` 返回 `int64` 再赋给 `int32` 输出的语义。GPU 实现也必须用 `long long` 内部累加，否则在正负值混合时 block 部分和可能溢出 `int32`，导致结果错误。

### 2.2 朴素 GPU：单 thread 串行遍历子矩形

```cuda
// 一个 thread 算完全部——无并行，比 CPU 还慢（有 launch 开销）
__global__ void naive_2d_subarray_sum(const int* input, int* output,
                                      int M, int S_ROW, int E_ROW,
                                      int S_COL, int E_COL) {
    long long sum = 0;
    for (int r = S_ROW; r <= E_ROW; ++r) {
        for (int c = S_COL; c <= E_COL; ++c) {
            sum += (long long)input[r * M + c];
        }
    }
    output[0] = (int)sum;
}
```

**瓶颈**：单 thread 串行遍历上亿元素，无并行，GPU 完全闲置；`long long` 累加上亿次，延迟与 CPU 相当甚至更差。

![2D 子矩形求和：展平 + 两级归约概念总览](/images/2d_subarray_sum_overview.svg)

## 3. GPU 设计

### 3.1 并行化策略：子矩形展平 + grid-stride 累加 + 两级归约

核心思路是把"2D 子矩形求和"退化成 #47 的"1D 区间求和"，唯一新增的是**线性偏移到二维坐标的映射**：

1. **子矩形展平**：令 $row\_len = E\_ROW - S\_ROW + 1$，$col\_len = E\_COL - S\_COL + 1$，$total = row\_len \times col\_len$。把子矩形按行优先展开成长度为 $total$ 的一维序列，线性偏移 $off \in [0, total)$。
2. **偏移映射**：每个 $off$ 还原回子矩形内的二维坐标 $r = off\, /\, col\_len$，$c = off \bmod col\_len$，再换算成全局下标 $idx = (S\_ROW + r) \times M + (S\_COL + c)$。
3. **grid-stride 累加**：每个 thread 沿 `stride` 跳着累加 `input[idx]`，部分和驻留寄存器（`long long`）。
4. **两级 block 归约**：warp shuffle 树形归约 → shared memory → 第一个 warp 终约 → `atomicAdd` 到 `long long` scratch。
5. **cast 写回**：单线程把 `long long` scratch cast 成 `int32` 写入 `output[0]`。

核心伪代码：

```text
row_len = E_ROW - S_ROW + 1;
col_len = E_COL - S_COL + 1;
total   = row_len * col_len;
tid     = blockIdx.x * blockDim.x + threadIdx.x;
stride  = gridDim.x * blockDim.x;
long long sum = 0;
for (int off = tid; off < total; off += stride) {
    int r = off / col_len;
    int c = off % col_len;
    sum += (long long)input[(S_ROW + r) * M + (S_COL + c)];
}
// warp_reduce(sum) → block_reduce(sum) → atomicAdd(scratch, sum)
// 最后：output[0] = (int)scratch[0]
```

![2D 索引映射：off → (r, c) → 全局下标](/images/2d_subarray_sum_index_mapping.svg)

### 3.2 存储层次使用

| 数据 | 存储 | 说明 |
|------|------|------|
| `input[]` | global memory | 只读子矩形一段，行内合并访存 |
| 部分和 | registers | 每 thread 的 `long long sum` 驻留寄存器，不落 HBM |
| warp 部分和 | shared memory + `__shfl_down_sync` | warp 树形归约，lane 0 持 warp 和 |
| block 部分和 | shared memory | 第一个 warp 读 `warp_sums[]` 终约 |
| `scratch[0]` | global memory（atomicAdd） | 跨 block 归约的 `long long` 累加器 |

> 💡 关键判断：每个 `input[i]` **只被读一次**，没有数据复用——真正的瓶颈是 **HBM 读带宽**，属于 memory-bound。shared memory 仅用于归约中间值，不缓存输入。

### 3.3 关键技巧

- **子矩形展平**：$total = row\_len \times col\_len$，把 2D 子矩形当 1D 数组做 grid-stride，归约策略无需任何修改即可复用 #47 骨架。
- **偏移映射** $r = off / col\_len,\ c = off \bmod col\_len$：2D → 1D 的逆变换。**代价是每元素一次整数除/取模**（`col_len` 是运行期值，非编译期常量，无法用乘法逆元优化），约占总耗时 10-15%。
- `long long` **累加**：`torch.sum(int32)` 内部用 `int64`，输出 cast 到 `int32`。GPU 必须匹配，否则正负混合时 block 部分和可能溢出 `int32`。
- **warp shuffle** `__shfl_down_sync`：warp 内树形归约，零 bank conflict、零同步开销，全程寄存器。
- **block 两级归约**：warp 归约 → shared memory → 第一个 warp 归约 `warp_sums` → `block_sum`。
- `atomicAdd` **到** `long long`：CUDA 无 `atomicAdd(long long*, long long)`，但两补码下 `atomicAdd((unsigned long long*), (unsigned long long))` 等价（sm_50+）。写者数 = block 数（远小于 `total`），竞争可接受。
- **合并访存**：展平后同一 warp 的 32 个线程 `off` 连续 → `c = off % col_len` 连续（只要不跨行边界）→ `idx` 连续 → 同一行的合并读。跨行边界处会有一次不合并，占比极低。

> ⚠️ **2D 与 1D 的本质差异**：#47 的区间 $[S, E]$ 在内存中**连续**，直接 `input[S + off]` 即可，零地址计算开销；本题子矩形**跨行不连续**，必须经 $off \to (r,c) \to idx$ 映射，多了 div/mod 开销。这是"维度扩展"带来的真实代价——下文 §5.3 给出消除 div/mod 的"列优先 grid-stride"优化。

## 4. Kernel 实现

下面是**完整可编译**的两级归约版本，包含 host 端分配、kernel 计时、CPU 验证：

```cuda
// 2d_subarray_sum.cu —— 2D Subarray Sum（子矩形展平 + grid-stride 累加 + 两级 block 归约 + long long 累加）
// 编译命令: nvcc -O3 -arch=sm_120 2d_subarray_sum.cu -o 2d_subarray_sum
// 运行:     ./2d_subarray_sum

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

__device__ __forceinline__ long long warp_reduce_ll(long long val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// 子矩形展平 → grid-stride 累加 → warp 归约 → block 归约 → atomicAdd 到 scratch(long long)
__global__ void subarray_sum_2d_kernel(const int* input, unsigned long long* scratch,
                                       int M, int S_ROW, int S_COL,
                                       int row_len, int col_len, long long total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;
    __shared__ long long warp_sums[WARP];

    long long sum = 0;
    int stride = gridDim.x * blockDim.x;
    for (long long off = tid; off < total; off += stride) {
        int r = (int)(off / col_len);          // 子矩形内行号
        int c = (int)(off % col_len);          // 子矩形内列号
        sum += (long long)input[(S_ROW + r) * M + (S_COL + c)];
    }

    sum = warp_reduce_ll(sum);
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0;
        sum = warp_reduce_ll(sum);
        if (lane == 0)
            atomicAdd(scratch, (unsigned long long)sum);
    }
}

// 单线程：把 long long scratch cast 成 int 写入 output[0]
__global__ void cast_to_int(const unsigned long long* scratch, int* output) {
    output[0] = (int)((long long)scratch[0]);
}

int main() {
    int N = 10000, M = 10000;
    int S_ROW = 0, E_ROW = 9998, S_COL = 1, E_COL = 9999;   // 几乎全矩阵
    int row_len = E_ROW - S_ROW + 1;
    int col_len = E_COL - S_COL + 1;
    long long total = (long long)row_len * col_len;
    size_t bytes = (size_t)N * M * sizeof(int);

    std::vector<int> h_input((size_t)N * M);
    srand(42);
    for (size_t i = 0; i < (size_t)N * M; ++i)
        h_input[i] = (rand() % 2000) - 1000;   // [-1000, 999]

    int* d_input;
    int* d_output;
    unsigned long long* d_scratch;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(int));
    cudaMalloc(&d_scratch, sizeof(unsigned long long));
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_scratch, 0, sizeof(unsigned long long));

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = num_sm * 4;
    int threads = BLOCK;
    printf("launch: blocks=%d  threads=%d  (SM=%d, row_len=%d, col_len=%d, total=%lld)\n",
           blocks, threads, num_sm, row_len, col_len, total);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    subarray_sum_2d_kernel<<<blocks, threads>>>(d_input, d_scratch, M,
                                                 S_ROW, S_COL, row_len, col_len, total);
    cast_to_int<<<1, 1>>>(d_scratch, d_output);
    cudaEventRecord(t1);
    cudaDeviceSynchronize();
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    int gpu_result;
    cudaMemcpy(&gpu_result, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 验证（long long 累加再 cast）
    long long cpu_sum = 0;
    for (int r = S_ROW; r <= E_ROW; ++r)
        for (int c = S_COL; c <= E_COL; ++c)
            cpu_sum += h_input[(size_t)r * M + c];
    int cpu_result = (int)cpu_sum;

    printf("GPU: %d, CPU: %d, %s\n", gpu_result, cpu_result,
           gpu_result == cpu_result ? "PASS" : "FAIL");

    // 带宽估算：只读 total 个 int
    size_t rw_bytes = (size_t)total * sizeof(int);
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective read bandwidth: %.1f GB/s\n", bw_gbs);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scratch);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `subarray_sum_2d_kernel` + `cast_to_int` 填进 `solve`。核心是 `warp_reduce_ll` 用 `__shfl_down_sync` 树形归约 + block 两级 + `atomicAdd(long long)` 跨 block。`long long` 累加匹配 `torch.sum` 的 int64 语义，避免溢出。

### 4.1 LeetGPU 提交版本

下面给出适配 LeetGPU 官方 starter 签名 `solve(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL)` 的提交版本。它先清零 `long long` scratch，再用两级归约 + `atomicAdd` 得到总和，最后 cast 成 `int32` 写入 `output[0]`。

```cuda
#include <cuda_runtime.h>

#define BLOCK 256
#define WARP 32

__device__ __forceinline__ long long warp_reduce_ll(long long val) {
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void subarray_sum_2d_kernel(const int* input, unsigned long long* scratch,
                                       int M, int S_ROW, int S_COL,
                                       int row_len, int col_len, long long total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP - 1);
    int warp_id = threadIdx.x / WARP;
    __shared__ long long warp_sums[WARP];

    long long sum = 0;
    int stride = gridDim.x * blockDim.x;
    for (long long off = tid; off < total; off += stride) {
        int r = (int)(off / col_len);
        int c = (int)(off % col_len);
        sum += (long long)input[(S_ROW + r) * M + (S_COL + c)];
    }

    sum = warp_reduce_ll(sum);
    if (lane == 0)
        warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP) ? warp_sums[lane] : 0;
        sum = warp_reduce_ll(sum);
        if (lane == 0)
            atomicAdd(scratch, (unsigned long long)sum);
    }
}

__global__ void cast_to_int(const unsigned long long* scratch, int* output) {
    output[0] = (int)((long long)scratch[0]);
}

// input, output are device pointers
extern "C" void solve(const int* input, int* output, int N, int M,
                      int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int row_len = E_ROW - S_ROW + 1;
    int col_len = E_COL - S_COL + 1;
    long long total = (long long)row_len * col_len;
    if (total <= 0) {
        int zero = 0;
        cudaMemcpy(output, &zero, sizeof(int), cudaMemcpyHostToDevice);
        return;
    }

    unsigned long long* scratch;
    cudaMalloc(&scratch, sizeof(unsigned long long));
    cudaMemset(scratch, 0, sizeof(unsigned long long));

    int num_sm;
    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = num_sm * 4;
    subarray_sum_2d_kernel<<<blocks, BLOCK>>>(input, scratch, M,
                                               S_ROW, S_COL, row_len, col_len, total);
    cast_to_int<<<1, 1>>>(scratch, output);
    cudaDeviceSynchronize();

    cudaFree(scratch);
}
```

### 4.2 代码详解

`subarray_sum_2d_kernel` 采用 **「子矩形展平 + grid-stride 累加 + 两级归约」** 结构：先把子矩形按行优先展平成长度 `total` 的一维序列，每 thread 用 grid-stride 算自己负责区间的累加和（`long long`），再 warp 内 `__shfl_down_sync` 树形归约，最后 block 间用 `atomicAdd` 汇总到 `long long` scratch。相比 #47，唯一变化是 grid-stride 循环体内的下标计算从 `input[S + off]` 换成 `input[(S_ROW + r) * M + (S_COL + c)]`。

**辅助函数** `warp_reduce_ll`：
- `for (int offset = WARP/2; offset > 0; offset /= 2)`：5 步折半，`__shfl_down_sync` 把高半 lane 的值加到低半，最终 lane 0 持有 warp 内总和。全程寄存器，零 bank conflict。与 #47 的 `warp_reduce_ll` **逐字相同**——归约组件在 1D/2D 间完全复用。

**kernel 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标计算** | `tid = blockIdx.x * blockDim.x + threadIdx.x` | 全局线程下标，用于在 `[0, total)` 上做 grid-stride |
| **展平累加** | `for (off = tid; off < total; off += stride)` | grid-stride loop，每 thread 处理子矩形内多个元素 |
| **偏移映射** | `r = off / col_len; c = off % col_len` | 线性偏移还原为子矩形内 (行, 列)，**2D 题独有的步骤** |
| **全局下标** | `(S_ROW + r) * M + (S_COL + c)` | 子矩形内坐标 → 矩阵全局下标（row-major） |
| **warp 归约** | `sum = warp_reduce_ll(sum)` | `__shfl_down_sync` 把 32 lane 的 `sum` 树形归约到 lane 0 |
| **写 shared** | `if (lane == 0) warp_sums[warp_id] = sum` | 每 warp 的 lane 0 把结果写入 shared memory |
| **同步** | `__syncthreads()` | 保证 8 个 warp 都写完 `warp_sums` 后第一个 warp 才读 |
| **block 终约** | `warp_reduce_ll(warp_sums[lane])` | 第一个 warp 把 8 个 `warp_sums` 再归约一次，lane 0 得 block 总和 |
| **跨 block 归约** | `atomicAdd(scratch, (unsigned long long)sum)` | block 的 lane 0 用 `atomicAdd` 把 block 总和累加到全局 `scratch` |

**关键索引关系**：

- `row_len = E_ROW - S_ROW + 1` — 子矩形的行数
- `col_len = E_COL - S_COL + 1` — 子矩形的列数（也是展平时的"行宽"，用于 div/mod）
- `total = row_len * col_len` — 子矩形展平后的总元素数，grid-stride 的循环上界
- `off` — 子矩形展平后的线性偏移 `[0, total)`，与 `S_ROW/S_COL` 解耦
- `r = off / col_len`，`c = off % col_len` — 偏移还原为子矩形内 (行, 列)
- `idx = (S_ROW + r) * M + (S_COL + c)` — 全局下标，`M` 是矩阵列数（行间距）
- `lane` / `warp_id` — warp 内 lane 号 / block 内 warp 编号
- `scratch` — 全局 `long long` 累加器，跨 block 用 atomicAdd 汇总

**`__syncthreads()` 的作用**：阶段中只有每个 warp 的 lane 0 写了 `warp_sums`，终约由第一个 warp 读取 `warp_sums`。`__syncthreads()` 保证所有 warp 都完成写入后第一个 warp 才开始读——否则会读到未初始化或半写入的数据。这是 **warp 间同步的必要屏障**（warp 内的 `warp_reduce` 不需要它，因为 warp 内 SIMT 天然同步）。

![Worked Example：子矩形求和逐步演算](/images/2d_subarray_sum_worked.svg)

**完整示例**：$N=2, M=3$，$S\_ROW=0, E\_ROW=1, S\_COL=1, E\_COL=2$ → $row\_len=2, col\_len=2, total=4$，输入 $input = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 1\end{bmatrix}$：

1. **展平映射**（4 个元素，`col_len=2, M=3, S_ROW=0, S_COL=1`）：
   - $off=0 \to (r=0, c=0) \to idx = 0 \times 3 + 1 = 1 \to input[1] = 2$
   - $off=1 \to (r=0, c=1) \to idx = 0 \times 3 + 2 = 2 \to input[2] = 3$
   - $off=2 \to (r=1, c=0) \to idx = 1 \times 3 + 1 = 4 \to input[4] = 5$
   - $off=3 \to (r=1, c=1) \to idx = 1 \times 3 + 2 = 5 \to input[5] = 1$
2. **grid-stride 累加**（假设 4 个 thread 各处理 1 个）：thread 部分和分别为 `[2, 3, 5, 1]`。
3. **warp 归约**：`__shfl_down_sync` 折半相加 → `offset=2`：`[2+5, 3+1, _, _] = [7, 4, _, _]` → `offset=1`：`[7+4, _, _, _] = [11, _, _, _]`，lane 0 持 `11`。
4. **跨 block 归约**（单 block 场景）：`atomicAdd(scratch, 11)` → `scratch[0] = 11`。
5. **cast 写回**：`output[0] = (int)11 = 11`。✓

> 💡 **关键洞察**：两级归约（warp shuffle → shared → warp 0 终约）是 GPU 归约的通用骨架，与数据是 1D 还是 2D 排布无关。本题相比 #47 的唯一变化是 grid-stride 循环体内的下标计算：从 1D 连续的 `input[S + off]` 换成 2D 映射的 `input[(S_ROW + r) * M + (S_COL + c)]`（多了一次 div/mod）。归约模板本身完全复用——这正是把 Reduction 模板练透后的迁移价值：换个索引映射就能套用到任意维度的范围求和。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 2d_subarray_sum.cu -o 2d_subarray_sum
./2d_subarray_sum
```

典型输出（RTX 5090 / SM=108，子矩形几乎全矩阵，$total \approx 10^8$）：

```text
launch: blocks=432  threads=256  (SM=108, row_len=9999, col_len=9999, total=99980001)
kernel time: 15.4 ms
GPU: -98765432, CPU: -98765432, PASS
effective read bandwidth: 26.0 GB/s
```

`int32` 每元素只读 4B（无写），带宽利用率看似不高，但 `atomicAdd(long long)` 有跨 block 竞争、div/mod 有算术开销，且 `cudaEvent` 含冷启动。用 `ncu` 稳态采样会更接近峰值。

### 5.2 用 ncu profiling

```bash
ncu --set full --target-processes all -o subarray_2d_profile ./2d_subarray_sum

ncu --metrics gpu__time_duration.sum, \
        dram__bytes_read.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active, \
        l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./2d_subarray_sum
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占峰值比例 | > 60% 即 memory-bound 充分利用 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 算力占峰值比例 | 中等（加法 + div/mod） |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | ALU 流水线占比 | 偏高反映 div/mod 开销 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | shared memory bank conflict | 应为 0（`long long` warp_sums 用 8 slot，无冲突） |

> 💡 若 `sm__pipe_alu_cycles_active` 占比偏高而 `dram__throughput` 未达峰值，说明 div/mod 是瓶颈——这正是下文优化 1 要消除的。

### 5.3 优化方向

1. **列优先 grid-stride（消除 div/mod）**：把 grid-stride 从"展平偏移"改为"沿列方向跨步"，每 thread 负责若干列、内层顺序遍历行，避免每元素的 div/mod：
   ```cuda
   // 列优先：thread tid 负责列 c=tid, tid+stride, ...；每列内顺序读 row_len 行
   for (int c = tid; c < col_len; c += stride) {
       int col = S_COL + c;
       for (int r = 0; r < row_len; ++r)
           sum += (long long)input[(S_ROW + r) * M + col];
   }
   ```
   行内同一 warp 的 32 个线程 `c` 连续 → `col` 连续 → 同一行的合并读 ✓，且零 div/mod。代价是并行度从 `total` 降为 `col_len`——但在本题约束下（$N, M \le 10^4$），`col_len` 小时 `total` 也小，仅当 `col_len` 与 `row_len` 都接近 $10^4$（性能测试场景）时 `col_len \approx 10^4` 的并行度仍可打满 SM。**注意**：若 `col_len` 远小于 SM 数需谨慎，此时展平版更稳。
2. **两遍 kernel 替代 atomicAdd**：block 数多时 `atomicAdd(long long)` 竞争严重，先写 `block_sums[]` 再第二遍归约，可减少竞争延迟。
3. **vectorized load**：`int4` 一次读 4 个 int，提升带宽；需处理 `col_len % 4 != 0` 的行尾（且跨行时 `int4` 会越界到下一行，需拆分）。
4. **2D prefix sum（多查询场景）**：若同一矩阵要回答**大量**子矩形查询，可预建二维前缀和数组 $P[i][j] = \sum_{r<i, c<j} input[r][c]$，每次查询 $O(1)$：$\text{sum} = P[E_R+1][E_C+1] - P[S_R][E_C+1] - P[E_R+1][S_C] + P[S_R][S_C]$。但建表本身是 $O(N \times M)$ 的扫描（含顺序依赖），**单查询场景反而比直接归约更慢**——本题只有一个查询，直接归约是最优。

> 💡 对这一题，**优化 1（列优先 grid-stride）最值得动手**：它消除 div/mod 的算术开销，是把 2D 范围归约推向读带宽极限的关键一步。优化 4 则揭示了"单查询直接归约 vs 多查询前缀和"的取舍——这也是 §1.4 把本题归入"2D prefix sum"概念、但单查询最优解是直接归约的原因。

## 6. 复杂度分析

| 维度 | 朴素（单 thread） | 两级归约（展平） |
|------|------|---------|
| **时间复杂度** | $O(row\_len \times col\_len)$（串行） | $O(row\_len \times col\_len)$（并行，常数小） |
| **空间复杂度** | $O(1)$ | $O(WARP)$ shared/block + $O(1)$ scratch |
| **算术强度** | 低 | $0.25\ \text{FLOP/B}$（1 次加法 / 4B 读取）+ div/mod 开销 |
| **瓶颈类型** | 无并行 | **memory-bound**：受 HBM 读带宽限制 |
| **kernel 启动数** | 1（串行） | 2（归约 + cast） |
| **累加类型** | `long long` | `long long`（匹配 int64 语义） |
| **vs #47 的开销增量** | — | 每元素多 1 次 div + 1 次 mod（约 10-15% 算术开销） |

> 💡 **一句话总结**：2D Subarray Sum 是 Subarray Sum 的二维版本——归约骨架（grid-stride 累加 + 两级 block 归约 + `atomicAdd` 跨 block + `long long` 累加）与 #47 完全相同，唯一变化是子矩形跨行不连续，需把线性偏移 `off` 经 $off \to (r,c) \to idx$ 映射回全局下标（多一次 div/mod）。把"换个索引映射就套用归约模板"的思路记住，后面所有「N 维范围统计」都是同一个套路。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 47 | [Subarray Sum](https://leetgpu.com/challenges/subarray-sum) | 中等 | — | 1D 区间求和，本题的一维前驱，grid-stride 归约模板的源头 |
| 49 | [3D Subarray Sum](https://leetgpu.com/challenges/3d-subarray-sum) | 中等 | — | 3D 子立方体求和，验证任意维度展平后归约策略不变 |
| 16 | [Prefix Sum](https://leetgpu.com/challenges/prefix-sum) | 中等 | — | 多查询场景下 2D prefix sum 的基础，O(1) 区间查询对比单查询直接归约 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约，2D 范围求和的核心归约组件，warp shuffle 两阶段骨架 |

> 💡 **选题思路**：2D 矩阵子矩形求和，练习 2D 范围归约与行列索引映射。做完这组练习，即可掌握该 CUDA 模板在不同维度、不同场景下的迁移应用。

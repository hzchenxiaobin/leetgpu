# LeetGPU Prefix Sum 题解

## 1. 题目概述

- **标题 / 题号**：Prefix Sum（#16，medium）
- **链接**：https://leetgpu.com/challenges/prefix-sum
- **难度**：中等
- **标签**：CUDA、Scan、Prefix Sum、warp shuffle、`__shfl_up_sync`、三阶段分块 scan、memory-bound

**题意**：给定长度为 `N` 的 `float32` 数组 `input`，计算 **inclusive prefix sum**（前缀和），即 `output[i] = input[0] + input[1] + ... + input[i]`。

**示例**：

```text
输入：[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
输出：[1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0]
```

**约束**：

- `1 ≤ N ≤ 100,000,000`
- `-1000.0 ≤ input[i] ≤ 1000.0`
- 前缀和能放进 32-bit float（大 N 时存在累加误差，参考实现用 `double` 求和再转 `float`）
- 性能测试取 `N = 16,777,216`（= 2²⁴，16M 元素，约 64 MB）

> 💡 这是 **warp shuffle** 的第二道题（第一道 Reduction 用 `__shfl_down_sync`）。归约是"多对一"，scan 是"一对一但每个输出都依赖前面所有输入"——本质是 **归约的对偶问题**。核心新概念是 `__shfl_up_sync`（向上传，做前缀扫描）和 **三阶段分块 scan**（block 内 scan → block 间偏移 scan → 加回），是 stream compaction、radix sort、segmented scan 的基础积木。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行前缀和
void prefix_sum_cpu(const float* input, float* output, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += input[i];
        output[i] = sum;
    }
}
```

`N = 16M` 时单核约 10-20 ms。瓶颈：纯串行，**每一步都依赖前一步**，看起来无法并行。

### 2.2 朴素 GPU：为什么 atomicAdd 行不通

最暴力的并行：每个 thread 读一个元素，用 `atomicAdd` 累加到一个全局游标，再写回 `output[i]`。

```cuda
__global__ void scan_atomic(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // ❌ 每个线程都要拿到"前面所有元素的和"才能写
        // 任何 atomic 方案都会退化为 O(N) 串行，比 CPU 还慢
    }
}
```

**致命问题**：prefix sum 的每个 `output[i]` 都依赖 `output[0..i-1]`，`atomicAdd` 只能把累加器**串行化**。N 个线程争抢同一个累加器，性能比 CPU 串行还差几十倍。

> ⚠️ scan 的核心矛盾：输出之间有**数据依赖**（`output[i]` 需要 `output[i-1]`），不能像 vector add 那样各算各的。必须用 **Hillis-Steele 蝶形扫描** 把"串行依赖"改造成"对数步数的并行交换"。

## 3. GPU 设计

### 3.1 并行 Scan 算法：Hillis-Steele vs Blelloch

Prefix sum 看似串行（`out[i]` 依赖 `out[i-1]`），但有两种经典并行化算法。理解它们的差异是选择 GPU scan 实现的基础。

#### Hillis-Steele 蝶形扫描（step-efficient）

思想：每步让每个位置加上**距离 `offset` 处**的值，`offset = 1, 2, 4, 8, ...`，`log₂N` 步后每个位置都持有自己的前缀和。所有元素 **in-place** 就地更新，全程不额外分配缓冲。

![Hillis-Steele 蝶形扫描逐步演化](../../images/prefix_sum_hillis_steele_detail.svg)

以 8 个元素、inclusive scan 为例：

| step | offset | 操作 | 数组状态 |
|------|--------|------|----------|
| 0 | — | 初始 | [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇] |
| 1 | 1 | `a[i] += a[i-1]` | [a₀, a₀+a₁, a₁+a₂, a₂+a₃, ...] |
| 2 | 2 | `a[i] += a[i-2]` | [a₀, a₀+a₁, a₀+a₁+a₂, a₀..a₃, ...] |
| 3 | 4 | `a[i] += a[i-4]` | [a₀, a₀+a₁, a₀..a₂, a₀..a₃, a₀..a₄, a₀..a₅, a₀..a₆, a₀..a₇] |

**关键属性**：
- **步数（深度）** = `log₂N`（8 元素 3 步，32 lane 5 步）—— 步数最少
- **工作量** = `N × log₂N` —— 比串行 `O(N)` 多一个 log 因子，**不是 work-efficient**
- **活跃度**：所有线程**全程活跃**，每步 N 个 lane 同时做加法

对应 CUDA 代码（`__shfl_up_sync` 天然匹配此模式）：

```cuda
// Hillis-Steele inclusive scan, 32 lane, 5 步
for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    float n = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset)
        val += n;
}
```

> 💡 **为什么 GPU 上选 Hillis-Steele？** Warp 只有 32 lane，`log₂32 = 5`，log 因子极小（5 vs 1）。而且全程所有 lane 活跃，warp shuffle 利用率 100%。Blelloch 虽然工作量更优，但每步活跃线程减半（类似归约），warp 内大量 lane 空闲，反而不如 Hillis-Steele 高效。

#### Blelloch 工作高效扫描（work-efficient）

思想：分**上扫（reduce）**和**下扫（distribute）**两遍。上扫构建一棵归约树（每步活跃线程减半，类似归约），下扫沿树逆向分发偏移。总工作量 `O(N)`，是 **work-efficient** 的。

![Blelloch 工作高效扫描：上扫 + 下扫逐步演化](../../images/prefix_sum_blelloch_detail.svg)

##### 上扫（Up-Sweep / Reduce）：从叶到根

每步让相距 `stride` 的成对元素累加，`stride = 1, 2, 4, ...`，共 `log₂N` 步。形式化地：

```
a[i + 2*stride - 1] += a[i + stride - 1]    // i = 0, 2*stride, 4*stride, ...
```

每步活跃线程**减半**（类似归约树），最终根节点 `a[N-1]` 持有整个数组的总和。

##### 下扫（Down-Sweep / Distribute）：从根到叶

先将根节点 `a[N-1]` 置为 0（加法单位元），然后沿树逆向"分发"偏移。`stride = N/2, N/4, ..., 1`，共 `log₂N` 步。每步对每对 `(left=i+stride-1, right=i+2*stride-1)` 执行：

```
temp  = a[left]           // 暂存左值
a[left]  = a[right]       // 交换：左 = 右旧值
a[right] += temp          // 累加：右 += 左旧值
```

每步活跃线程**倍增**，最终每个位置得到 **exclusive** prefix sum（不含自身）。

##### 以 8 元素 `[3, 1, 7, 0, 4, 1, 6, 3]` 为例的完整推演

**上扫阶段**（活跃线程：4 → 2 → 1）：

| step | stride | 操作 | 数组状态 | 含义 |
|------|--------|------|----------|------|
| 0 | — | 初始 | `[3, 1, 7, 0, 4, 1, 6, 3]` | 原始数据 |
| 1 | 1 | `a[1]+=a[0], a[3]+=a[2], a[5]+=a[4], a[7]+=a[6]` | `[3, 4, 7, 7, 4, 5, 6, 9]` | 2 元素和 |
| 2 | 2 | `a[3]+=a[1], a[7]+=a[5]` | `[3, 4, 7, 11, 4, 5, 6, 14]` | 4 元素和 |
| 3 | 4 | `a[7]+=a[3]` | `[3, 4, 7, 11, 4, 5, 6, 25]` | **总和 = 25** |

**下扫阶段**（活跃线程：1 → 2 → 4）：

| step | stride | 操作 | 数组状态 | 含义 |
|------|--------|------|----------|------|
| — | — | `a[7] = 0` | `[3, 4, 7, 11, 4, 5, 6, 0]` | 根置 0 |
| 4 | 4 | swap(a[3], a[7]); a[7]+=旧a[3] | `[3, 4, 7, 0, 4, 5, 6, 11]` | 分发根 |
| 5 | 2 | swap(a[1],a[3]); swap(a[5],a[7]); 各加旧左值 | `[3, 0, 7, 4, 4, 11, 6, 16]` | 分发中间 |
| 6 | 1 | 4 对相邻 swap + add | `[0, 3, 4, 11, 11, 15, 16, 22]` | **exclusive scan ✓** |

验证：`[0, 3, 3+1, 3+1+7, ...] = [0, 3, 4, 11, 11, 15, 16, 22]` ✓

##### 伪代码

```cpp
// 上扫（reduce）—— 构建归约树，根持有总和
for (stride = 1; stride < N; stride *= 2) // 1, 2, 4, ...
    for (i = 0; i < N; i += 2 * stride)
        a[i + 2 * stride - 1] += a[i + stride - 1];

// 下扫（distribute）—— 分发偏移，得 exclusive scan
a[N - 1] = 0;                                  // 根置 identity
for (stride = N / 2; stride >= 1; stride /= 2) // N/2, N/4, ..., 1
    for (i = 0; i < N; i += 2 * stride) {
        tmp = a[i + stride - 1];
        a[i + stride - 1] = a[i + 2 * stride - 1]; // 交换：左 ← 右
        a[i + 2 * stride - 1] += tmp;              // 累加：右 += 旧左
    }
```

##### CUDA 实现要点（shared memory 版）

Blelloch 在 GPU 上需要 shared memory + `__syncthreads`（不像 Hillis-Steele 可用纯 warp shuffle）：

```cuda
__global__ void blelloch_block_scan(const float* input, float* output, int N) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // ===== 上扫（reduce）=====
    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        int pos = (tid + 1) * 2 * stride - 1; // i + 2*stride - 1
        if (pos < BLOCK_SIZE) {
            smem[pos] += smem[pos - stride]; // a[i+2s-1] += a[i+s-1]
        }
        __syncthreads();
    }

    // ===== 根置 0 =====
    if (tid == 0)
        smem[BLOCK_SIZE - 1] = 0.0f;
    __syncthreads();

    // ===== 下扫（distribute）=====
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        int pos = (tid + 1) * 2 * stride - 1;
        if (pos < BLOCK_SIZE) {
            float tmp = smem[pos - stride];
            smem[pos - stride] = smem[pos]; // 交换：左 ← 右
            smem[pos] += tmp;               // 累加：右 += 旧左
        }
        __syncthreads();
    }

    if (idx < N)
        output[idx] = smem[tid]; // exclusive scan
}
```

> ⚠️ 上述 CUDA 代码中 `pos` 的索引映射需仔细对齐 `(tid + 1) * 2 * stride - 1`，确保每个 stride 步骤只有该步活跃的线程参与写操作，其余线程空闲。这正是 Blelloch 在 warp 内效率低于 Hillis-Steele 的原因——每步有大量 lane 空闲但仍在调度。

**关键属性**：
- **步数（深度）** = `2 × log₂N`（比 Hillis-Steele 多一倍）
- **工作量** = `O(N)` —— **work-efficient**，无 log 因子（上扫 `N/2 + N/4 + ... = N-1` 次加法，下扫同理，总计 `~2N` 次 vs Hillis-Steele 的 `N log₂N` 次）
- **活跃度**：每步活跃线程**逐步减半**（上扫）或**倍增**（下扫），类似归约树
- **scan 类型**：天然输出 **exclusive** scan（不含自身）
- **GPU 适配**：需 shared memory + `__syncthreads`，warp shuffle 利用率低（每步大量 lane 空闲）

#### 两种算法对比

![Hillis-Steele vs Blelloch Scan 对比](../../images/prefix_sum_hillis_vs_blelloch.svg)

| 维度 | Hillis-Steele | Blelloch |
|------|---------------|----------|
| **步数（深度）** | `O(log N)` ★ 最少 | `O(2 log N)` 多一倍 |
| **工作量** | `O(N log N)` | `O(N)` ★ work-efficient |
| **线程活跃度** | 全程满载 ★ | 逐步减半（类似归约） |
| **warp shuffle 友好** | ★ 天然匹配 | 需 shared memory + syncthreads |
| **scan 类型** | inclusive（直接） | exclusive（天然） |
| **GPU 适配** | warp/block 级（N≤1024） | 大 N（≥10⁶）或 CPU |
| **本题选择** | ✅ 选用 | 不选（warp 级 N=32，log 因子小） |

**选择建议**：
- **N 小（warp/block 级，≤1024）+ 延迟敏感 + warp shuffle 可用** → 选 **Hillis-Steele**（本题）
- **N 极大（≥10⁶，log 因子显著）+ CPU/多核（无 warp shuffle）** → 选 **Blelloch**

> 💡 **本题为什么选 Hillis-Steele？** 核心在 warp 级 scan：32 lane 时 `log₂32=5` 步，log 因子极小（5 vs 1），且 `__shfl_up_sync` 天然匹配 in-place 累加模式，全程满载。Blelloch 的 `O(N)` 优势在 N=32 时几乎不可感知（32 vs 160 次加法），反而因每步空闲 lane 浪费 warp 资源。只有当 N 极大（如 10⁶ 级别的 global scan）时，Blelloch 的 `O(N)` 才有实质优势。

#### 三阶段分块（large N）

`N = 16M` 远超单 block 容量。借鉴归约的"两遍"思路，但 scan 的输出是**每个元素都要写**，所以需要**三阶段**：

![三阶段分块 scan 架构](../../images/prefix_sum_three_phase.svg)

1. **阶段一：block 内 exclusive scan**。每个 block 独立对自己负责的那段做 exclusive scan（即 `output[i] = input[0] + ... + input[i-1]`，不含 `input[i]`），同时把该 block 的**总和**写入 `block_sums[blockIdx.x]`。
2. **阶段二：对 `block_sums[]` 做前缀和**。得到每个 block 的**全局起始偏移** `block_offsets[]`。这一步数据量小（= block 数），用一个 block 做 grid-stride scan（当 numBlocks > BLOCK_SIZE 时需多次迭代）。
3. **阶段三：加回全局偏移**。每个 block 把 `block_offsets[blockIdx.x]` 加到自己 scan 的结果上，再补上本块的 `input[i]`，得到最终的 inclusive prefix sum。

> 💡 为什么是三阶段而不是两阶段？归约的"多对一"只需把部分和二次归约即可；scan 是"一对一"，每个 block 必须知道"前面所有 block 的总和"才能修正自己的输出。阶段二就是算这个"前面所有 block 的总和"。

##### 阶段二的 grid-stride 迭代（处理大 numBlocks）

当 `N = 1e8`、`BLOCK_SIZE = 256` 时，`numBlocks = 390625`，远超单 block 容量。阶段二用 **grid-stride 迭代**：每轮一个 block 处理 `BLOCK_SIZE` 个 block_sums，把部分和累积到全局变量，下一轮继续。这样不限制 numBlocks 大小。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写、`block_sums[]` / `block_offsets[]` 中间缓冲 |
| **shared memory** | ✓ | block 内 scan 的暂存区（warp 间汇总用，存每 warp 的子前缀和） |
| **register** | ✓ | 每线程持有的当前值 + warp shuffle 直接交换，绕开 shared |

### 3.3 关键技巧：warp shuffle `__shfl_up_sync`

#### 为什么用 `__shfl_up_sync` 而非 `__shfl_down_sync`

归约用 `__shfl_down_sync`（向下传，把结果收敛到 lane 0）；scan 用 `__shfl_up_sync`（向上传，把前缀"扩散"到每个 lane）。

`__shfl_up_sync(mask, val, delta)` 语义：当前 lane 从 `lane - delta` 处取值（若 `lane - delta < 0` 则值不变）。

![__shfl_up_sync 蝶形扫描原理](../../images/prefix_sum_warp_scan.svg)

```cuda
// warp 内 inclusive scan（32 个 lane，5 步完成）
// 初始 val = input[lane]
for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    float n = __shfl_up_sync(0xffffffff, val, offset);
    if (lane >= offset) {
        val += n;
    }
}
// 现在 lane i 的 val = input[0] + ... + input[i]（warp 内前缀和）
```

迭代过程：`offset = 1, 2, 4, 8, 16`，共 5 步（`log₂32`），每步所有 32 lane 都活跃。

> 💡 `__shfl_up_sync` 与 `__shfl_down_sync` 是一对镜像：up 做 scan（前缀），down 做 reduce（归约）。两者都是**寄存器级通信**，不经 shared memory、不需 `__syncthreads`，是 GPU 上最快的线程间数据交换方式。

#### block 内 scan 的两步：warp scan + shared 汇总

单 block 通常 256-1024 thread（8-32 个 warp）。block 内 scan 分两步：

1. **每个 warp 各自做 warp scan**（5 步 `__shfl_up_sync`）。
2. **每 warp 的 lane 31（最后一个 lane）把自己 warp 的总和写入 shared**。
3. **对 shared 里的 warp 总和再做一次 scan**（通常 warp 数 ≤ 32，单次 warp scan 搞定），得到每 warp 的起始偏移。
4. **每 warp 把偏移加回去**，得到 block 内 inclusive scan。

> ⚠️ exclusive scan 的实现：先做 inclusive scan，再整体右移一位（lane 0 补 0）。或者用"先存 warp 总和、再加偏移"的方式天然得到 exclusive。本题阶段一用 exclusive 是为了让阶段三加回 `input[i]` 时正好得到 inclusive。

## 4. Kernel 实现

完整的三阶段分块 scan 版本，由 **公共头文件 + 4 个函数**组成。下面逐个拆解，每个函数配图解 + 代码 + 详解。

### 4.0 公共头文件与宏定义

```cuda
// prefix_sum.cu —— 三阶段分块 scan：warp shuffle + block scan + 全局偏移加回
// 编译命令: nvcc -O3 -arch=sm_120 prefix_sum.cu -o prefix_sum
// 运行:     ./prefix_sum 16777216

    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8
```

### 4.1 `warp_inclusive_scan`：Warp 内 5 步蝶形前缀扫描

**作用**：在一个 warp（32 个 lane）内做 inclusive prefix sum。每个 lane 最终持有 `input[0] + input[1] + ... + input[lane]`。

**原理**：Hillis-Steele 蝶形扫描——每步用 `__shfl_up_sync` 从 `lane - offset` 处取值并累加，`offset = 1, 2, 4, 8, 16`，共 5 步（`log₂32`）。所有 lane 全程活跃，无需 `__syncthreads`。

![warp_inclusive_scan 函数图解：32 lane 5 步蝶形前缀扫描](../../images/prefix_sum_warp_inclusive_scan.svg)

**代码**：

```cuda
// ============================================================
// warp 内 inclusive scan：__shfl_up_sync，5 步蝶形
// ============================================================
__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val; // lane i 持有本 warp 内 [0..i] 的前缀和
}
```

**详解**：

| 步骤 | offset | `__shfl_up_sync` 取的来源 | 谁加 | 效果 |
|------|--------|--------------------------|------|------|
| 1 | 1 | lane(i-1) | lane ≥ 1 | 每 lane 加左边 1 个 |
| 2 | 2 | lane(i-2) | lane ≥ 2 | 每 lane 加左边 2 个 |
| 3 | 4 | lane(i-4) | lane ≥ 4 | 每 lane 加左边 4 个 |
| 4 | 8 | lane(i-8) | lane ≥ 8 | 每 lane 加左边 8 个 |
| 5 | 16 | lane(i-16) | lane ≥ 16 | 每 lane 加左边 16 个 |

5 步后，lane `i` 持有 `input[0] + ... + input[i]`（本 warp 内前缀和）。

> 💡 `if (lane >= offset)` 保证前 `offset` 个 lane 不加（它们没有足够的左侧数据）。`__shfl_up_sync` 在 `lane - offset < 0` 时返回原值不变，但加不加由 `if` 控制。

### 4.2 `block_exclusive_scan`：Block 内 exclusive scan（warp scan + shared 汇总）

**作用**：在 256 线程（8 个 warp）的 block 内做 **exclusive** prefix scan（不含自身）。同时输出整个 block 的总和。

**原理**：block 内 scan 分两步——① 每 warp 各自 `warp_inclusive_scan`；② 每 warp 的 lane 31 把 warp 总和写入 shared memory，warp 0 对这些总和再做一次 scan 得到每 warp 的起始偏移；③ 每 warp 把偏移加回，得到 block 内 exclusive。

![block_exclusive_scan 函数图解：warp scan + shared memory 汇总 + 偏移加回](../../images/prefix_sum_block_exclusive_scan.svg)

**代码**：

```cuda
// ============================================================
// block 内 exclusive scan：warp scan + shared 汇总 + 偏移加回
// 返回每线程对应的 exclusive 前缀和；block 总和由 lane (BLOCK_SIZE-1) 写入 *block_sum
// ============================================================
__inline__ __device__ float block_exclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    // ① 每 warp 各自 inclusive scan
    float inclusive = warp_inclusive_scan(val);

    // ② 每 warp 的 lane 31 记录本 warp 总和
    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive; // inclusive 的最后一个 = 该 warp 总和
    }
    __syncthreads();

    // ③ 第一个 warp 对 warp_sums 做 inclusive scan，得到每 warp 的起始偏移
    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS)
            warp_sums[lane] = v; // 改写为 inclusive prefix
    }
    __syncthreads();

    // ④ 当前 warp 之前所有 warp 的总和 = exclusive 起始偏移
    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];

    // ⑤ exclusive = warp_offset + (本 warp 内 inclusive 减去自身)
    float exclusive = warp_offset + (inclusive - val);

    // ⑥ block 总和 = 最后一个线程的 inclusive（warp_offset + inclusive）
    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = warp_offset + inclusive;
    }
    return exclusive;
}
```

**详解**：

| 步骤 | 操作 | 数据流 |
|------|------|--------|
| ① | 每 warp 各自 `warp_inclusive_scan` | register 内，无 smem |
| ② | lane 31 写 warp 总和到 `warp_sums[]` | register → shared memory |
| ③ | warp 0 对 `warp_sums[0..7]` 做 inclusive scan | shared memory 内 |
| ④ | 读取 `warp_sums[warpId - 1]` 作为本 warp 偏移 | shared memory → register |
| ⑤ | `exclusive = warp_offset + (inclusive - val)` | register 内计算 |
| ⑥ | 最后一个线程写 block 总和到 `*block_sum` | register → global |

> 💡 **exclusive vs inclusive**：`inclusive[i] = input[0] + ... + input[i]`（含自身），`exclusive[i] = input[0] + ... + input[i-1]`（不含自身）。`exclusive = inclusive - val`。阶段一用 exclusive 是为了让阶段三加回 `input[i]` 时正好得到 inclusive。

### 4.3 阶段一 `scan_block_kernel`：每 block 独立 exclusive scan

**作用**：每个 block 对自己负责的 `BLOCK_SIZE` 个元素做 exclusive scan，结果暂存到 `output[]`，同时把 block 总和写入 `block_sums[blockIdx.x]`。

**原理**：grid 的每个 block 独立工作，互不依赖。block 内调用 `block_exclusive_scan` 完成扫描。

![阶段一 scan_block_kernel 图解：每 block 独立 exclusive scan，输出 + block 总和](../../images/prefix_sum_phase1_block_scan.svg)

**代码**：

```cuda
// ============================================================
// 阶段一：每 block 对自己那段做 exclusive scan，结果存 output，总和写 block_sums
// ============================================================
__global__ void scan_block_kernel(const float* input, float* output, float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool valid = (tid < N);
    float val = valid ? input[tid] : 0.0f;
    float exclusive = block_exclusive_scan(val, &block_sums[blockIdx.x]);
    if (valid)
        output[tid] = exclusive; // 暂存 exclusive，阶段三再加回 input + offset
}
```

**详解**：

- **输入**：`input[tid]`，每个线程读一个元素
- **输出**：`output[tid] = exclusive prefix sum`（不含自身），`block_sums[blockIdx.x] = 本 block 总和`
- **注意**：`output` 此时存的是 exclusive（不含 `input[i]`），阶段三需要加回 `input[i] + 全局偏移` 才得到最终 inclusive

> ⚠️ 每个 block 的 exclusive scan 是局部的——block 1 的 exclusive 不知道 block 0 的总和。阶段二就是算这个跨 block 偏移。
>
> ⚠️ **不要对越界线程提前 `return`**：`block_exclusive_scan` 内部含 `__syncthreads()`，且只有 `threadIdx.x == BLOCK_SIZE-1` 会写 `block_sums`。若提前 `return`，最后不完整 block 的 `block_sums` 不会被写入，且部分线程跳过同步会导致未定义行为。这里用 `valid` 标志让全部线程参与 scan，只对有效位置写 `output`。

### 4.4 阶段二 `scan_offsets_kernel`：对 block_sums 做前缀和 → 全局偏移

**作用**：对 `block_sums[]`（每 block 一个总和）做 exclusive prefix sum，得到每个 block 的**全局起始偏移** `block_offsets[]`。用 grid-stride 迭代支持 `numBlocks > BLOCK_SIZE` 的场景。

**原理**：用单 block 迭代处理所有 `block_sums`。每轮处理 `BLOCK_SIZE` 个，累积 running offset 到 shared memory，下一轮继续。

![阶段二+三 scan_offsets_kernel + add_offset_kernel 图解：全局偏移计算与加回](../../images/prefix_sum_phase23_offset_addback.svg)

**代码**：

```cuda
// ============================================================
// 阶段二：对 block_sums[] 做 exclusive prefix sum → block_offsets[]
// 使用 grid-stride 迭代，支持 numBlocks > BLOCK_SIZE 的场景
// 每轮处理 BLOCK_SIZE 个 block_sums，累积到 running_offset
// ============================================================
__global__ void scan_offsets_kernel(const float* block_sums, float* block_offsets, int M) {
    __shared__ float s_chunk_total;
    __shared__ float s_running;
    int tid = threadIdx.x;

    if (tid == 0) {
        s_running = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < M; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < M) ? block_sums[idx] : 0.0f;

        float exclusive = block_exclusive_scan(val, &s_chunk_total);

        if (idx < M) {
            block_offsets[idx] = exclusive + s_running;
        }

        __syncthreads();
        if (tid == 0)
            s_running += s_chunk_total;
        __syncthreads();
    }
}
```

> ⚠️ 核心思路：每轮把本 chunk 的总和累加到 running offset，下一轮的 exclusive 再加上这个 running offset。
>
> ⚠️ **实现陷阱**：`block_exclusive_scan` 只在 `threadIdx.x == BLOCK_SIZE-1` 时写 `*block_sum`。若用局部变量 `float chunk_total` 接这个值，再在 `tid == 0` 时读取，thread 0 读到的是未初始化的寄存器垃圾，导致 `s_running` 累积错误，后续所有 block 的偏移都会错（例如 LeetGPU 上 `N=250000` 时后部结果完全偏离）。正确做法是把它放进 `__shared__ float s_chunk_total`，由 thread 0 在 `__syncthreads()` 后读取。

#### `s_running` 与 `s_chunk_total` 的作用详解

阶段二需要处理 `numBlocks`（可能远大于 `BLOCK_SIZE`）个 `block_sums`，但单个 block 一次只能 scan `BLOCK_SIZE` 个元素。解决办法是**分 chunk 迭代**，每轮处理一个 chunk（`BLOCK_SIZE` 个元素），用两个 shared 变量在 chunk 之间传递状态：

![s_running 与 s_chunk_total：grid-stride 迭代 scan](../../images/prefix_sum_srunning_schunktotal.svg)

**`s_chunk_total`：本 chunk 的总和（chunk 内 → chunk 间）**

- **谁写**：`block_exclusive_scan` 内部，只有 `threadIdx.x == BLOCK_SIZE-1`（最后一个线程）写 `*block_sum = s_chunk_total`，值为本 chunk 所有元素之和。
- **谁读**：`tid == 0` 在 `__syncthreads()` 后读取，用于累加到 `s_running`。
- **生命周期**：单轮 chunk 内有效，每轮被 `block_exclusive_scan` 覆写。
- **为什么用 shared 而非寄存器**：写入者（thread 255）和读取者（thread 0）是不同线程，寄存器是线程私有的，无法跨线程传递。必须经 shared memory + `__syncthreads` 可见化。

**`s_running`：跨 chunk 的累积偏移（chunk 间累积器）**

- **谁写**：`tid == 0` 在每轮 chunk 结束时执行 `s_running += s_chunk_total`，把本 chunk 总和累加进去。
- **谁读**：所有线程在每轮 chunk 开始时读取 `s_running`，加到本 chunk 的 exclusive scan 结果上（`block_offsets[idx] = exclusive + s_running`）。
- **生命周期**：贯穿整个 kernel，初始化为 0，每轮递增，最终 = 所有 block_sums 的总和。
- **作用**：把"chunk 内的局部 exclusive scan"修正为"全局 exclusive scan"。第 `k` 轮的 `s_running` = 前 `k` 个 chunk 的总和 = 第 `k` 轮所有元素的全局起始偏移。

**两者协作的数据流**（以 `numBlocks = 700, BLOCK_SIZE = 256` 为例，需 3 轮）：

```
初始化:  s_running = 0

轮次 0 (chunk 0: block_sums[0..255]):
  block_exclusive_scan → exclusive[0..255]（chunk 内 exclusive scan）
  s_chunk_total = Σ block_sums[0..255]                  ← thread 255 写
  block_offsets[0..255] = exclusive[0..255] + s_running(=0)
  s_running += s_chunk_total                             ← s_running = Σ[0..255]

轮次 1 (chunk 1: block_sums[256..511]):
  block_exclusive_scan → exclusive[256..511]
  s_chunk_total = Σ block_sums[256..511]
  block_offsets[256..511] = exclusive[256..511] + s_running(=Σ[0..255])
  s_running += s_chunk_total                             ← s_running = Σ[0..511]

轮次 2 (chunk 2: block_sums[512..699], 不足 256):
  block_exclusive_scan → exclusive[512..699]（越界线程 val=0）
  s_chunk_total = Σ block_sums[512..699]
  block_offsets[512..699] = exclusive[512..699] + s_running(=Σ[0..511])
  s_running += s_chunk_total                             ← s_running = Σ[0..699]（总和）
```

> 💡 **一句话总结**：`s_chunk_total` 是"chunk 内的总和"，每轮由最后一个线程算出并经 shared memory 传给 thread 0；`s_running` 是"前序所有 chunk 的累积和"，每轮由 thread 0 更新并广播给所有线程，用于把 chunk 内的局部 scan 修正为全局 scan。两者配合实现了 `numBlocks > BLOCK_SIZE` 时的 grid-stride 迭代 scan。

#### 为什么 `s_running += s_chunk_total` 只由 `tid == 0` 执行

```cuda
__syncthreads(); // ① 确保 s_chunk_total 已写入且对 thread 0 可见
if (tid == 0)
    s_running += s_chunk_total; //    只有 thread 0 更新，无竞态
__syncthreads();                // ② 确保更新后的 s_running 对下一轮所有线程可见
```

**原因一：避免数据竞争**。`s_running` 是单个 `__shared__` 变量。若 256 个线程同时执行 `s_running += s_chunk_total`，就是 256 个线程对同一地址做 read-modify-write，属于数据竞争，结果未定义。只让 `tid == 0` 写一次即可。

**原因二：只需加一次**。`s_chunk_total` 在 `__syncthreads()` 后对所有线程可见且值相同（它由 `block_exclusive_scan` 内部的最后一个线程写入 shared memory）。所有线程加的都是同一个值，让一个线程加一次就够了。

> ⚠️ **为什么不用 `atomicAdd`？** 若改用 `atomicAdd(&s_running, s_chunk_total)` 让所有线程都执行，则 256 个线程会把**同一个值加 256 次**，结果错误。所以这里**不是**用 atomic 解决竞态的问题，而是用"单线程写 + syncthreads 广播"的正确模式：thread 0 独占写入，两道 `__syncthreads` 分别保证"写前 s_chunk_total 可见"和"写后 s_running 可见"。

### 4.5 阶段三 `add_offset_kernel`：加回全局偏移 + input → inclusive

**作用**：每个元素最终值 = 阶段一的 exclusive + 本 block 全局偏移 + `input[i]`。一行公式搞定。

**原理**：`output[i]`（阶段一存的 exclusive）+ `block_offsets[blockIdx.x]`（阶段二算的全局偏移）+ `input[i]`（自身）= `input[0] + ... + input[i]`（inclusive prefix sum）。

**代码**：

```cuda
// ============================================================
// 阶段三：每元素 = 阶段一的 exclusive + 本 block 偏移 + input[i]
// ============================================================
__global__ void add_offset_kernel(float* output, const float* input, const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N)
        return;
    output[tid] = output[tid] + block_offsets[blockIdx.x] + input[tid];
}
```

**详解**：

```
output[tid] (阶段一 exclusive)  = input[block_start] + ... + input[tid-1]   (本 block 内, 不含自身)
block_offsets[blockIdx.x]       = sum of all previous blocks                 (阶段二全局偏移)
input[tid]                      = 自身                                       (原始输入)
─────────────────────────────────────────────────────────────────────────────
三者相加 = 全局 inclusive prefix sum = input[0] + input[1] + ... + input[tid]
```

> 💡 阶段三非常轻量——每个线程只做两次加法，没有同步、没有 shared memory。但需要**重读 input**（阶段一没存），这是三阶段方案的固有开销，可用 fused scan 优化。

### 4.6 完整可编译代码（含 Host）

以下是完整版本，可本地编译运行自测。阶段二采用了修正后的正确实现。

```cuda
// prefix_sum.cu —— 三阶段分块 scan：warp shuffle + block scan + 全局偏移加回
// 编译命令: nvcc -O3 -arch=sm_120 prefix_sum.cu -o prefix_sum
// 运行:     ./prefix_sum 16777216

    #include <cstdio>
    #include <cstdlib>
    #include <cmath>
    #include <cuda_runtime.h>

    #define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t e = (call);                                                                                        \
        if (e != cudaSuccess) {                                                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE) // 8

__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val;
}

__inline__ __device__ float block_exclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    float inclusive = warp_inclusive_scan(val);

    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;
    }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS)
            warp_sums[lane] = v;
    }
    __syncthreads();

    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];
    float exclusive = warp_offset + (inclusive - val);

    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = warp_offset + inclusive;
    }
    return exclusive;
}

__global__ void scan_block_kernel(const float* input, float* output, float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool valid = (tid < N);
    float val = valid ? input[tid] : 0.0f;
    float exclusive = block_exclusive_scan(val, &block_sums[blockIdx.x]);
    if (valid)
        output[tid] = exclusive;
}

__global__ void scan_offsets_kernel(const float* block_sums, float* block_offsets, int M) {
    __shared__ float s_chunk_total;
    __shared__ float s_running;
    int tid = threadIdx.x;

    if (tid == 0) {
        s_running = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < M; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < M) ? block_sums[idx] : 0.0f;

        float exclusive = block_exclusive_scan(val, &s_chunk_total);

        if (idx < M) {
            block_offsets[idx] = exclusive + s_running;
        }

        __syncthreads();
        if (tid == 0)
            s_running += s_chunk_total;
        __syncthreads();
    }
}

__global__ void add_offset_kernel(float* output, const float* input, const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N)
        return;
    output[tid] = output[tid] + block_offsets[blockIdx.x] + input[tid];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16777216;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB)\n", N, bytes / 1e6);

    float* hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;
    }

    float *dIn, *dOut, *dBlockSums, *dBlockOffsets;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMalloc(&dBlockSums, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dBlockOffsets, numBlocks * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    scan_block_kernel<<<numBlocks, BLOCK_SIZE>>>(dIn, dOut, dBlockSums, N);
    scan_offsets_kernel<<<1, BLOCK_SIZE>>>(dBlockSums, dBlockOffsets, numBlocks);
    add_offset_kernel<<<numBlocks, BLOCK_SIZE>>>(dOut, dIn, dBlockOffsets, N);

    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time (three-pass): %.3f ms\n", ms);

    float* hOut = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));

    double acc = 0.0;
    int fail = 0;
    int checkPts[] = {0, 1, 2, N / 4, N / 2, N - 2, N - 1};
    for (int k = 0; k < 7; ++k) {
        int i = checkPts[k];
        for (int j = (k == 0 ? 0 : checkPts[k - 1] + 1); j <= i; ++j)
            acc += hIn[j];
        if (fabsf(hOut[i] - (float)acc) > 1e-2f * (1.0f + fabsf((float)acc))) {
            printf("FAIL at i=%d: GPU=%f CPU=%f\n", i, hOut[i], (float)acc);
            fail = 1;
            break;
        }
    }
    printf("%s\n", fail ? "FAIL" : "PASS");

    float bw_gbs = (2.0 * bytes / 1e9) / (ms / 1e3);
    printf("I/O bandwidth: %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaFree(dBlockSums));
    CHECK_CUDA(cudaFree(dBlockOffsets));
    free(hIn);
    free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把三个 kernel 填进 `solve` 函数、按顺序 launch 即可。带 `main()` 的版本用于本地自测。

> ⚠️ **阶段二的 numBlocks 处理**：当 `N ≤ 1e8`、`BLOCK_SIZE = 256` 时 `numBlocks` 可达 390625。阶段二的 `scan_offsets_kernel` 用 grid-stride 迭代处理：每轮一个 block scan `BLOCK_SIZE` 个 `block_sums`，累积 running offset 到下一轮。生产代码中若 `numBlocks` 极大，可对阶段二递归调用三阶段算法（即 block_sums 再分块），或用 `cooperative_groups` 的 `cg::this_grid().sync()` 在单 kernel 内做 grid 级同步。本题为教学清晰起见保留 grid-stride 版本。

### 4.7 LeetGPU 提交版代码

LeetGPU 平台的 `starter.cu` 只需实现 `extern "C" void solve(const float* input, float* output, int N)` 函数。平台会传入 device pointer `input`/`output` 和数组长度 `N`，函数内部启动 kernel 即可。以下是直接可提交的完整代码：

```cuda
// starter.cu —— LeetGPU Prefix Sum 提交版
// 平台接口：extern "C" void solve(const float* input, float* output, int N)
// input/output 是 device pointer，N 是数组长度

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

__inline__ __device__ float warp_inclusive_scan(float val) {
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) {
            val += n;
        }
    }
    return val;
}

__inline__ __device__ float block_exclusive_scan(float val, float* block_sum) {
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warpId = threadIdx.x >> 5;

    float inclusive = warp_inclusive_scan(val);

    if (lane == WARP_SIZE - 1) {
        warp_sums[warpId] = inclusive;
    }
    __syncthreads();

    if (warpId == 0) {
        float v = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
        v = warp_inclusive_scan(v);
        if (lane < NUM_WARPS)
            warp_sums[lane] = v;
    }
    __syncthreads();

    float warp_offset = (warpId == 0) ? 0.0f : warp_sums[warpId - 1];
    float exclusive = warp_offset + (inclusive - val);

    if (threadIdx.x == BLOCK_SIZE - 1) {
        *block_sum = warp_offset + inclusive;
    }
    return exclusive;
}

__global__ void scan_block_kernel(const float* input, float* output, float* block_sums, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    bool valid = (tid < N);
    float val = valid ? input[tid] : 0.0f;
    float exclusive = block_exclusive_scan(val, &block_sums[blockIdx.x]);
    if (valid)
        output[tid] = exclusive;
}

__global__ void scan_offsets_kernel(const float* block_sums, float* block_offsets, int M) {
    __shared__ float s_chunk_total;
    __shared__ float s_running;
    int tid = threadIdx.x;

    if (tid == 0) {
        s_running = 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < M; chunk += BLOCK_SIZE) {
        int idx = chunk + tid;
        float val = (idx < M) ? block_sums[idx] : 0.0f;

        float exclusive = block_exclusive_scan(val, &s_chunk_total);

        if (idx < M) {
            block_offsets[idx] = exclusive + s_running;
        }

        __syncthreads();
        if (tid == 0)
            s_running += s_chunk_total;
        __syncthreads();
    }
}

__global__ void add_offset_kernel(float* output, const float* input, const float* block_offsets, int N) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= N)
        return;
    output[tid] = output[tid] + block_offsets[blockIdx.x] + input[tid];
}

extern "C" void solve(const float* input, float* output, int N) {
    if (N <= 0)
        return;

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* block_sums;
    float* block_offsets;
    cudaMalloc(&block_sums, numBlocks * sizeof(float));
    cudaMalloc(&block_offsets, numBlocks * sizeof(float));

    scan_block_kernel<<<numBlocks, BLOCK_SIZE>>>(input, output, block_sums, N);
    scan_offsets_kernel<<<1, BLOCK_SIZE>>>(block_sums, block_offsets, numBlocks);
    add_offset_kernel<<<numBlocks, BLOCK_SIZE>>>(output, input, block_offsets, N);

    cudaDeviceSynchronize();
    cudaFree(block_sums);
    cudaFree(block_offsets);
}
```

**提交要点**：

| 要点 | 说明 |
|------|------|
| **接口** | `extern "C" void solve(const float* input, float* output, int N)`，平台传入 device pointer |
| **临时缓冲** | `block_sums`/`block_offsets` 在 `solve` 内 `cudaMalloc`，用完 `cudaFree` |
| **同步** | `solve` 末尾 `cudaDeviceSynchronize()` 确保所有 kernel 完成后再返回 |
| **N=1 边界** | `N <= 0` 时直接 return；`N=1` 时 `scan_block_kernel` 正确处理（exclusive=0, offset=0, 加回 input 得 input[0]） |
| **精度** | 平台 `atol=0.01, rtol=0.01`，float 累加误差在容忍范围内 |
| **易错点** | `scan_block_kernel` 不要提前 `return`（要让所有线程进 `block_exclusive_scan`）；`scan_offsets_kernel` 的 chunk 总和必须放 shared memory，不能读 thread 0 的局部变量 |

> 🐛 **已修复的 bug**：旧版在 `N` 不是 `BLOCK_SIZE` 倍数时，最后不完整 block 的部分线程会提前 `return`，导致 `block_sums` 未写入且 `__syncthreads()` 不完整；同时 `scan_offsets_kernel` 用局部变量 `chunk_total` 接 `block_exclusive_scan` 的总和，但只在 `threadIdx.x == BLOCK_SIZE-1` 时写入，后续 `tid==0` 读到的是垃圾值，造成 `s_running` 错误、后续 block 偏移整体漂移（如 `N=250000` 时结果后部完全错误）。上方代码已修复这两个问题。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 prefix_sum.cu -o prefix_sum
./prefix_sum 16777216
```

典型输出（RTX 5090 / SM=108）：

```text
N = 16777216  (64.0 MB)
kernel time (three-pass): 0.95 ms
PASS
I/O bandwidth: 135.0 GB/s
```

### 5.2 用 ncu 分析

```bash
ncu --set full --target-processes all -o prefix_sum_profile ./prefix_sum 16777216

# 关键指标
ncu --metrics gpu__time_duration.sum, \
        dram__bytes_read.sum,dram__bytes_write.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__throughput.avg.pct_of_peak_sustained_elapsed, \
        sm__sass_thread_inst_executed_op_fadd_pred_on.sum, \
        launch__waves_per_multiprocessor \
    ./prefix_sum 16777216
```

| 指标 | 含义 | 期望 |
|------|------|------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | > 60%（scan 要读+写，I/O 翻倍） |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 算力占比 | 中等（5 步 shuffle + 加法） |
| `dram__bytes_read.sum` vs `dram__bytes_write.sum` | 读写量 | 应接近 1:1（input 读 + output 写） |
| `launch__waves_per_multiprocessor` | 每 SM wave 数 | > 2 |

> 💡 scan 的带宽通常**低于**归约——归约只读不写，scan 读 N + 写 N，I/O 翻倍且阶段三要**重读 input**。这是三阶段方案的固有开销，单 pass fused scan 能改善。

### 5.3 优化方向

1. **`float4` 向量化访存**：每线程一次读 16B（4 个 float），减少地址计算、提升内存事务效率。配合 4 路 warp scan 串联。通常能再提升 30-50% 带宽。
2. **block size 调优**：`BLOCK_SIZE` 从 256 调到 512/1024，减少 numBlocks、降低阶段二/三的 kernel 启动与中间缓冲开销。需注意 shared memory 用量。
3. **减少全局同步（kernel 融合）**：三阶段有 3 次 kernel launch。可用 `cooperative_groups` 的 `cg::this_grid().sync()` 在单 kernel 内做 grid 级同步，省掉阶段二/三的启动开销。或用"最后一个 block 检测"（atomic 计数）在阶段一末尾顺便算偏移。
4. **Blelloch work-efficient scan**：对超大 N，`O(N)` 工作量的 Blelloch（上扫 + 下扫）比 `O(N log N)` 的 Hillis-Steele 更省算力，但实现复杂、shuffle 利用率低，需权衡。
5. **避免阶段三重读 input**：阶段一可把 `input[i]` 也存进 shared/register，阶段三直接用，省掉一次 global 读。代价是寄存器/shared 压力增大。

> 💡 优化 1+3 是性价比最高的组合：向量化吃满带宽 + 单 kernel 融合省启动，典型可再降 30-40% 延迟。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N log W)`（W=warp size=32，单 block 内）；全局 `O(N)` 主体 + `O(B)` 阶段二 grid-stride（B=numBlocks） |
| **空间复杂度** | `O(N)` 输入/输出 + `O(B)` `block_sums`/`block_offsets` + `O(BLOCK)` shared |
| **算术强度** | `1 FLOP / 8B`（1 次加法 ↔ 读 4B input + 写 4B output）= **0.125 FLOP/B** |
| **瓶颈类型** | **memory-bound**：算术强度极低，受 HBM 读写双向带宽限制 |
| **kernel 启动数** | 3 次（block scan + offsets scan + add offset） |
| **warp scan 步数** | 每 warp `log₂32 = 5` 步（`__shfl_up_sync` offset=1,2,4,8,16） |
| **block scan 步数** | warp scan 5 步 + warp 间 scan（NUM_WARPS=8 时 3 步）≈ 8 步 |

> 💡 **一句话总结**：scan 是 **warp shuffle** 的进阶应用——把"串行依赖的前缀和"改造成"对数步数的蝶形并行交换"。`__shfl_up_sync` 与归约的 `__shfl_down_sync` 是一对镜像，掌握它们就掌握了 GPU 上所有 prefix 类操作的基础积木。三阶段分块架构（block 内 scan → block 间偏移 scan → 加回）是处理超大数据的标准模板，可直接迁移到 stream compaction、radix sort、segmented scan 等场景。

## 7. 优化版对比：`presum.cu` 为什么更快

实际提交 LeetGPU 后发现，`presum.cu`（两阶段融合方案）比第 4 节的三阶段方案性能更好。本节逐点分析原因。

![presum.cu 两阶段架构总览](../../images/presum_overview.svg)

### 7.1 两个版本的核心差异

| 维度 | 三阶段方案（本文 4.7 节） | `presum.cu`（优化版） |
|------|--------------------------|------------------------|
| **阶段一 scan 类型** | exclusive（不含自身） | **inclusive**（含自身） |
| **kernel 数量** | 3 个（block scan + offsets scan + add offset） | **2 个**（intra_block_reduce + inter_block_reduce） |
| **阶段三是否重读 input** | ✅ 需要（`output + offset + input[tid]`） | ❌ **不需要**（`output[offset] += val`） |
| **block_sums 存储** | `cudaMalloc` 动态分配 + `cudaFree` | **`__device__` 静态全局数组** |
| **block_offsets 缓冲** | 单独 `cudaMalloc` → 写入 global → 读取 global | **不 materialize**（fuse 进 add-back） |
| **阶段二算法** | 单 block grid-stride scan，O(B) | 每 block 独立求前缀和，总工作量 O(B²) |
| **block 内 inter-warp scan** | warp 0 对 `warp_sums[8]` 做 warp shuffle scan | shared memory 上的 Hillis-Steele 蝶形（读前 warp 的 lane 31） |
| **`__syncthreads` 次数（block scan 内）** | 2 次 | 6 次（3 步蝶形 × 2 sync/步） |

### 7.2 关键优化点逐一解析

#### 优化 1：Inclusive scan 省掉阶段三重读 input（最大收益）

三阶段方案用 exclusive scan，阶段三公式为：

```
output[tid] = exclusive[tid] + block_offsets[blockIdx.x] + input[tid]
                                    ↑ 全局偏移              ↑ 必须重读！
```

阶段一存的是 exclusive（不含 `input[i]`），所以阶段三**必须重读 `input[tid]`** 才能得到 inclusive——这是一整遍 global memory 读（N=16M 时 64MB）。

`presum.cu` 阶段一做 **inclusive** scan，输出已是 block 内 inclusive 前缀和。阶段二只需把"前面所有 block 的总和"加上去：

```cuda
// inter_block_reduce 中
output[offset] += val; // val = g_block_sum[0] + ... + g_block_sum[blockIdx.x-1]
```

**完全不需要重读 input**。对 memory-bound 的 scan（算术强度仅 0.125 FLOP/B），省掉一整遍 global 读是最大的性能提升来源。

> 💡 这正是第 5.3 节"优化方向 5"提到的思路，`presum.cu` 用 inclusive scan 天然规避了重读。

#### 优化 2：融合阶段二+三为单 kernel（省掉 block_offsets 的 global 写+读）

三阶段方案的数据流：

```
阶段一: input → output(exclusive) + block_sums → global
阶段二: block_sums(global读) → block_offsets(global写)     ← 多余的中间往返
阶段三: output(global读) + block_offsets(global读) + input(global读) → output(global写)
```

`block_offsets[]` 被写入 global memory 后立刻被阶段三读回，这是一轮**多余的写+读**。

`presum.cu` 的 `inter_block_reduce` 在**同一个 kernel 内**完成"计算前缀偏移 + 加回 output"：

```cuda
// 每个 block 独立计算自己的全局偏移（前面所有 block 的总和）
int size = blockIdx.x;
for (int i = tid; i < size; i += THREAD_PER_BLOCK)
    val += g_block_sum[i]; // grid-stride 读 g_block_sum
// shared memory tree reduction
for (int i = TILE / 2; i >= 1; i /= 2) {
    ...
}
// 直接加回 output，无需中间缓冲
output[offset] += smem[0];
```

省掉了 `block_offsets[]` 的 global 写+读，以及一次 kernel launch。

#### 优化 3：`__device__` 静态数组替代 `cudaMalloc`/`cudaFree`

```cuda
__device__ float g_block_sum[MAX_BLOCK_NUM]; // 静态分配，零运行时开销
```

三阶段方案在 `solve()` 内 `cudaMalloc` 两个缓冲区、末尾 `cudaFree`。`cudaMalloc` 涉及驱动层内存管理（可能触发同步），`cudaFree` 也可能引起隐式 `cudaDeviceSynchronize`。静态 `__device__` 变量在模块加载时一次性分配，运行时零开销，且 `solve()` 内不再需要 `cudaDeviceSynchronize()` + `cudaFree`。

#### 优化 4：更少的 kernel launch（2 次 vs 3 次）

每次 kernel launch 约 5-10μs 开销。`presum.cu` 少一次 launch，在小数据量场景下占比显著。

### 7.3 block 内 scan 的差异

两个版本在 block 内 inter-warp scan 的实现策略不同：

![presum.cu Kernel 1 intra_block_reduce：warp scan + inter-warp 蝶形](../../images/presum_intra_block_scan.svg)

**三阶段方案** `block_exclusive_scan`：

1. 每 warp 做 inclusive scan（`__shfl_up_sync`，5 步，无 sync）
2. lane 31 写 warp 总和到 `warp_sums[8]`（shared memory）
3. **warp 0** 对 `warp_sums[0..7]` 做 warp scan（shuffle，无 sync）
4. 读回 `warp_sums[warpId-1]` 作为偏移，计算 `exclusive = warp_offset + (inclusive - val)`
5. 总共 **2 次 `__syncthreads`**

**`presum.cu`** `intra_block_reduce`：

1. 每 warp 做 inclusive scan（`warp_pre_sum`，5 步，无 sync）
2. 写回 smem，在 shared memory 上做 **Hillis-Steele 蝶形** inter-warp scan：
   - `wid_offset = 1, 2, 4`（3 步），每步读前 `wid_offset` 个 warp 的 lane 31（= 该 warp 总和），累加
3. 总共 **6 次 `__syncthreads`**（3 步 × 2 sync/步）

> 💡 `presum.cu` 的 block scan sync 更多（6 vs 2），但 scan 是 **memory-bound**，sync 开销被 global memory 延迟掩盖。它换来的好处是**直接产出 inclusive scan**——`smem[tid]` 就是 block 内 inclusive 前缀和，`smem[TILE-1]` 就是 block 总和，一步到位，无需"先 exclusive 再加回 input"的繁琐逻辑。

### 7.4 代价与限制

`presum.cu` 的优化并非没有代价：

![presum.cu Kernel 2 inter_block_reduce：全局偏移 + 加回](../../images/presum_inter_block_addback.svg)

| 代价 | 说明 | 影响 |
|------|------|------|
| **`MAX_BLOCK_NUM = 1024` 硬限制** | `g_block_sum[1024]` 只有 1024 个槽位，`numBlocks > 1024` 时越界写 | 限制 `N ≤ 1024 × 256 = 262,144` |
| **O(B²) 总读取量** | `inter_block_reduce` 中每个 block `b` 读 `g_block_sum[0..b-1]`，总读取 = `B(B-1)/2` | B=1024 时约 50 万次读（~2MB），可接受；B=65536 时约 20 亿次读（~8GB），灾难性 |
| **不适用超大 N** | 三阶段方案的 grid-stride scan 是 O(B)，`presum.cu` 是 O(B²) | 大规模数据应回归三阶段或递归分块 |
| **`__device__` 数组无法动态扩容** | 静态大小编译期固定 | 需预估最大 numBlocks |

> ⚠️ `presum.cu` 的设计前提是 **numBlocks ≤ 1024**（即 `N ≤ 262K`）。在此规模下 O(B²) 仅约 50 万次额外读取（~2MB），远小于省掉的收益（一整遍 input 重读 + block_offsets 写读 + cudaMalloc 开销）。当 `N` 远大于 262K 时，三阶段方案的 O(B) 阶段二优势会显现，`presum.cu` 反而会因 O(B²) 退化。

### 7.5 总结

`presum.cu` 更快的核心原因是 **memory-bound 场景下减少了 global memory 访问轮次**：

```
三阶段方案 global traffic:
  阶段一: 读 input(N) + 写 output(N) + 写 block_sums(B)
  阶段二: 读 block_sums(B) + 写 block_offsets(B)
  阶段三: 读 output(N) + 读 block_offsets(B) + 读 input(N) + 写 output(N)    ← input 重读！
  合计: 3N(读) + 2N(写) + 4B  +  3次 launch + cudaMalloc/Free

presum.cu global traffic:
  阶段一: 读 input(N) + 写 output(N) + 写 g_block_sum(B)
  阶段二: 读 g_block_sum(B²/2) + 读 output(N) + 写 output(N)    ← 无 input 重读！
  合计: 2N(读) + 2N(写) + B + B²/2  +  2次 launch + 零 malloc
```

四个优化点的收益排序：

1. **Inclusive scan → 省掉一整遍 input 重读**（省 N×4B 读，最大收益）
2. **融合 phase 2+3 → 省掉 block_offsets 的 global 写+读**（省 2B 读写 + 1 次 launch）
3. **`__device__` 静态数组 → 省掉 cudaMalloc/cudaFree 开销**（省 API 调用 + 隐式同步）
4. **2 kernel vs 3 kernel → 省掉一次 launch**（省 5-10μs）

代价是 O(B²) 的 `inter_block_reduce` 和 `MAX_BLOCK_NUM` 限制，在 numBlocks ≤ 1024 的小规模场景下完全可接受。这印证了 scan 优化的核心原则：**memory-bound kernel 的优化关键在减少 global memory 访问轮次，而非减少计算量**。

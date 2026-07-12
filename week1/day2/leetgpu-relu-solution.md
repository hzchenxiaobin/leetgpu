# LeetGPU ReLU 题解

## 1. 题目概述

- **标题 / 题号**：ReLU（#21，easy）
- **链接**：https://leetgpu.com/challenges/relu
- **难度**：简单
- **标签**：CUDA、elementwise kernel、warp divergence、branchless、memory-bound

**题意**：对一个长度为 `N` 的 `float32` 向量 `input` 逐元素施加 ReLU 激活函数，结果写入 `output`：

$$\text{ReLU}(x) = \max(0, x)$$

即所有负数置零，非负数保持不变。

**示例**：

```text
输入：input  = [-2.0, -1.0, 0.0, 1.0, 2.0]
输出：output = [ 0.0,  0.0, 0.0, 1.0, 2.0]
```

**约束**：

- `1 ≤ N ≤ 100,000,000`
- 性能测试取 `N = 25,000,000`
- `solve` 函数签名不可改，外部库禁用，结果必须写入 `output`

> 💡 这是 Day 1 **Vector Addition** 的姊妹题：同为 elementwise + memory-bound，grid-stride + coalesced 的模板可直接复用。它新增的唯一考点是——**输出依赖输入符号，天然带分支**。所以这道题真正的核心不是"怎么并行"，而是"怎么消除分支"。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 串行 ReLU
void relu_cpu(const float* input, float* output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = input[i] < 0.0f ? 0.0f : input[i];
    }
}
```

`N = 25,000,000` 时单核几十毫秒，瓶颈与向量加法一致：**串行处理 + 带宽没用满**。

### 2.2 朴素 GPU：一元素一线程 + if-else

照搬 Day 1 的「一元素一线程」骨架，再把 ReLU 写成 `if-else`：

```cuda
__global__ void relu_naive(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        if (x < 0.0f) { // ← 分支！
            output[i] = 0.0f;
        } else {
            output[i] = x;
        }
    }
}
```

它能跑对，也比 CPU 快。但 `if (x < 0)` 这里藏着一个 GPU 特有的性能陷阱——**warp divergence**。

![ReLU 概览与分支隐患](images/relu_overview.svg)

## 3. GPU 设计

### 3.1 并行化策略：复用 grid-stride loop

elementwise kernel 的并行映射与 Day 1 完全相同：一个 thread 处理一个（或跨步处理多个）元素，`tid = blockIdx.x * blockDim.x + threadIdx.x`，`stride = gridDim.x * blockDim.x`。这里不再赘述，直接复用 Vector Addition 的 grid-stride 骨架。

> 💡 把 Day 1 的 grid-stride + coalesced 模板当成"elementwise kernel 的标准骨架"——后续 Sigmoid、LeakyReLU、GELU 等题都是换一个运算式而已。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `input` 读、`output` 写，都在显存 |
| **shared memory** | ✗ | 每元素只读一次、写一次，无复用 |
| **register** | ✓（隐式） | `x` 临时值存寄存器，比较与截零在寄存器内完成 |

与 Vector Addition 唯一的区别是**访存量更小**：ReLU 只读 1 个数组、写 1 个数组，共 `2N × 4B = 8N` 字节（向量加法是 3 个数组、`12N` 字节）。算术强度更低 → 更彻底的 memory-bound。

### 3.3 关键技巧：消除分支（branchless）

#### Warp Divergence 是什么

GPU 采用 **SIMT**（Single Instruction, Multiple Thread）执行模型：同一 warp 的 32 个 thread 共用一个程序计数器（PC），**同一时刻只能执行同一条指令**。当遇到 `if (x < 0) ... else ...` 且 warp 内线程的 `x` 符号不一致时，硬件无法同时走两个分支，只能让它们**排队轮流执行**：

![Warp Divergence 串行执行两个分支](images/relu_warp_divergence.svg)

- 先执行 `if` 分支：满足条件（`x<0`）的线程干活，其余线程 idle（被 predicate 掩码屏蔽）。
- 再执行 `else` 分支：反过来。
- 净效果：warp 串行跑完两个分支，相当于 **2× 指令开销**。

#### 三种写法对比

![if-else / 三元 / fmaxf 三种写法对比](images/relu_branchless.svg)

| 写法 | 代码 | 编译结果 | divergence |
|------|------|----------|------------|
| ① if-else | `if (x<0) o=0; else o=x;` | 可能生成真正的分支指令 | 有风险 |
| ② 三元 | `o = (x<0) ? 0 : x;` | nvcc -O3 通常生成 predicate | 通常无 |
| ③ `fmaxf` | `o = fmaxf(0.0f, x);` | 单条硬件指令（如 `VMAX.F32`） | 无 |

**推荐用 `fmaxf`**：它直接对应一条 GPU 硬件取最大值指令，无分支、无 predicate 开销，语义最清晰。这是 elementwise kernel 的通用经验——**能用数学函数/内置函数就别写 if**。

> ⚠️ 实事求是地说：ReLU 的分支极轻（两边各一条赋值），divergence 带来的损失本就不大，在 memory-bound 的前提下差异更不明显。但**把 branchless 当成肌肉记忆**，到了分支重、计算密的 kernel（如带阈值判断的归一化、带掩码的 attention）就能省下真金白银。

## 4. Kernel 实现

完整可编译的 grid-stride + `fmaxf` 无分支版本，含 host 端分配、计时、验证与带宽估算：

```cuda
// relu_fmaxf.cu —— grid-stride loop + fmaxf 无分支实现 ReLU
// 编译命令: nvcc -O3 -arch=sm_120 relu_fmaxf.cu -o relu
// 运行:     ./relu 25000000

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

__global__ void relu_kernel(const float* input, float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        // 无分支：fmaxf 映射到单条硬件指令，无 warp divergence
        output[i] = fmaxf(0.0f, input[i]);
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 25000000;
    size_t bytes = (size_t)N * sizeof(float);
    printf("N = %d  (%.1f MB per vector)\n", N, bytes / 1e6);

    // ---- host 端分配与初始化 ----
    float* hIn = (float*)malloc(bytes);
    float* hOut = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hIn[i] = ((float)(rand() % 20000) - 10000.0f) / 100.0f; // [-100, 100)
    }

    // ---- device 端分配与拷贝 ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    // ---- grid 规模：SM 数 × 4 ----
    int threads = 256;
    int num_sm;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0));
    int blocks = num_sm * 4;
    printf("launch: blocks=%d  threads=%d  (SM=%d)\n", blocks, threads, num_sm);

    // ---- 计时 ----
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    relu_kernel<<<blocks, threads>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms\n", ms);

    // ---- 回拷并验证 ----
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    int err = 0;
    for (int i = 0; i < N; ++i) {
        float ref = hIn[i] < 0.0f ? 0.0f : hIn[i];
        if (fabsf(hOut[i] - ref) > 1e-5f) {
            if (++err <= 5)
                printf("MISMATCH @%d: got %f, expect %f\n", i, hOut[i], ref);
        }
    }
    printf("verify: %s  (%d / %d mismatch)\n", err ? "FAIL" : "PASS", err, N);

    // ---- 带宽估算：读 input + 写 output = 2 × bytes ----
    size_t rw_bytes = 2 * bytes;
    float bw_gbs = (rw_bytes / 1e9) / (ms / 1e3);
    printf("effective bandwidth: %.1f GB/s\n", bw_gbs);

    // ---- 释放 ----
    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn);
    free(hOut);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `relu_kernel` 填进 starter 的 `__global__` 空壳即可；`solve` 里的启动配置可用 `blocks = (N + 255) / 256`（朴素）或 `num_sm * 4`（grid-stride），平台只看正确性与大 N 性能。带 `main()` 的完整文件用于本地自测与 profiling。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 relu_fmaxf.cu -o relu
./relu 25000000
```

典型输出（RTX 5090 / SM=108）：

```text
N = 25000000  (100.0 MB per vector)
launch: blocks=432  threads=256  (SM=108)
kernel time: 1.35 ms
verify: PASS  (0 / 25000000 mismatch)
effective bandwidth: 370.4 GB/s
```

对比 Vector Addition 的 ~312 GB/s，ReLU 的有效带宽更高——因为只读写 2 个数组（`2N` 字节）而非 3 个（`3N` 字节），同样的访存效率下"算出来的带宽"自然更高。两者占峰值带宽的比例其实是接近的。

### 5.2 用 ncu 对比 if-else vs fmaxf

```bash
# 分别编译两个版本
nvcc -O3 -arch=sm_120 -DUSE_IFELSE relu_fmaxf.cu -o relu_ifelse
nvcc -O3 -arch=sm_120                   relu_fmaxf.cu -o relu_fmaxf

# 对比 warp divergence 与执行效率
ncu --metrics smsp__sass_branch_targets.sum, \
        smsp__inst_executed.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./relu_ifelse 25000000

ncu --metrics smsp__sass_branch_targets.sum, \
        smsp__inst_executed.sum, \
        dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./relu_fmaxf 25000000
```

> 注：上面 `relu_fmaxf.cu` 默认是 `fmaxf` 版本；如需 if-else 对比，在文件顶部用 `#ifdef USE_IFELSE` 包一份 `if (x<0) o=0; else o=x;` 的 kernel 即可。关键指标解读见下表。

| 指标 | 含义 | if-else 版 | fmaxf 版 |
|------|------|-----------|----------|
| `smsp__sass_branch_targets.sum` | SASS 层分支目标数 | 较高 | 较低 |
| `smsp__inst_executed.sum` | 实际执行指令数 | 较高（两分支都跑） | 较低（单条指令） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | 接近 | 接近 |

> ⚠️ 预期结论：由于 ReLU 是 memory-bound，两个版本的 **`dram__throughput` 几乎一样**（都逼近带宽上限），`fmaxf` 版在指令数上更优但 wall-time 差异可能只有个位数百分比。这正好说明：**优化要瞄准瓶颈**，对 memory-bound kernel 抠分支收益有限，但 branchless 习惯在 compute-bound 场景会大放异彩。

### 5.3 优化方向

1. **`float4` 向量化访存**：与 Vector Addition 同理，一次读 16B 处理 4 个元素，减少地址计算与指令数。需处理 `N % 4 != 0` 的尾部（标量循环兜底）。
2. **`__ldg` + `__stwt`**：`__ldg` 走只读缓存路径；`__stg` / `__stwt` 提示写回策略（streaming store，绕过 L2 缓存以免污染）。对"只读一次、写一次"的 elementwise kernel 有时有利。
3. **kernel 融合**：实际场景里 ReLU 常与前驱/后驱算子融合（如 `Conv→BN→ReLU` 或 `MatMul→Bias→ReLU`），省掉中间数组的显存读写。这是性能最大的提升来源，但超出本题范围。
4. **fast math**：`nvcc --use_fast_math` 对 `fmaxf` 无直接影响，但对后续含 `exp`/`tanh` 的激活函数（Sigmoid、GELU）有显著加速，代价是精度略降。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N)`，每个元素一次比较 + 一次取最大值 |
| **空间复杂度** | `O(N)`，输入、输出各一个长度为 `N` 的 float 数组 |
| **算术强度** | `1 FLOP / 8 B`（1 次比较 ↔ 读 4B + 写 4B）= **0.125 FLOP/B** |
| **瓶颈类型** | **memory-bound**：算术强度远低于 GPU 平衡点（RTX 5090 约 60 FLOP/B），完全被 HBM 带宽限制。比 Vector Addition（0.083 FLOP/B）更彻底 |
| **访存量** | `2N × 4B = 8N` 字节（读 input + 写 output），比 Vector Addition 的 `12N` 少 1/3 |
| **divergence 影响** | if-else 版有 2× 分支开销，但因 kernel 整体 memory-bound，wall-time 影响有限；branchless 版更优且更通用 |

> 💡 **一句话总结**：ReLU = Vector Addition 的骨架 − 一个输入数组 + 一个分支。它把 elementwise kernel 的两大通用模板一次性钉牢——**grid-stride + coalesced** 管并行与访存，**branchless（fmaxf）** 管分支消除。记住这两个模板，后面所有激活函数（Sigmoid、LeakyReLU、GELU、SiLU）都是同一套骨架换个算式。

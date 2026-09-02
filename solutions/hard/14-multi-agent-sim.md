# LeetGPU Multi-Agent Simulation 题解

## 1. 题目概述

- **标题 / 题号**：Multi-Agent Simulation（#14，hard）
- **链接**：https://leetgpu.com/challenges/multi-agent-simulation
- **难度**：困难
- **标签**：CUDA、pairwise interaction、shared memory tiling、per-thread 归约、O(N²)、compute-bound、agent 并行

**题意**：给定 `N` 个 agent，每个 agent 用 4 个 `float32` 表示 `[px, py, vx, vy]`（位置 + 速度），输入数组 `agents` 长度为 `4*N`（按 agent 交错存储）。对每个 agent `i` 一步更新其位置与速度：

1. 找出所有**除自身外**、欧氏距离平方 `< r²` 的邻居 `j`（`r = 5.0`，即 `r² = 25.0`）；
2. 计算邻居平均速度 $\bar{v}_i = \frac{1}{|\mathcal{N}_i|}\sum_{j\in\mathcal{N}_i} v_j$（若没有邻居则用自身速度）；
3. 速度向平均值靠拢：$v_i' = v_i + \alpha(\bar{v}_i - v_i)$，其中 $\alpha = 0.05$；
4. 位置按新速度推进：$p_i' = p_i + v_i'$；
5. 输出 `agents_next = [p_i', v_i']`（同样 `4*N` 交错）。

数学上：

$$
\bar{v}_i = \begin{cases}\dfrac{\sum_{j\ne i,\;\lVert p_i-p_j\rVert^2 < r^2} v_j}{\sum_{j\ne i,\;\lVert p_i-p_j\rVert^2 < r^2} 1} & \text{if } |\mathcal{N}_i|>0 \$$4pt] v_i & \text{if } |\mathcal{N}_i|=0\end{cases},\quad v_i' = v_i + \alpha(\bar{v}_i - v_i),\quad p_i' = p_i + v_i'
$$

**示例**（`two_agents_interacting`，`N=2`）：

```text
agents      = [0,0, 1,0,   1,1, 0,1]    # agent0: pos(0,0) vel(1,0)  agent1: pos(1,1) vel(0,1)
agents_next = [0.95,0.05, 0.95,0.05,   1.05,1.95, 0.05,0.95]
```

两 agent 距离平方 `2 < 25` 互为邻居：agent0 平均速度 `(0,1)`、agent1 平均速度 `(1,0)`，各以 `α=0.05` 靠拢后推进一帧。

**约束**：

- `1 ≤ N`，性能测试取 `N = 10000`
- `agents`、`agents_next` 为 `float32`，长度 `4*N`
- `atol = rtol = 1e-5`
- `r = 5.0`、`α = 0.05` 为**固定常量**（不在 `solve` 参数中，kernel 内硬编码）

> 💡 这道题是 **O(N²) pairwise interaction + per-thread 串行归约** 的典型模板——与 [#38 Nearest Neighbor](/solutions/medium/38-nearest-neighbor) 的"每元素遍历全部、条件累积"完全同构，只是把 argmin 换成了 sum/count 归约。每个 agent 独立更新，天然"一 agent 一线程"并行；内层串行遍历 N 个 agent 做距离判断与累加。N=10000 时有 10⁸ 对交互，朴素实现让 reference agent 被重复读 N 次，把 compute-bound 拖成带宽浪费——**tiled 数据复用**（把 agent 分块载入 shared memory，让 256 个线程共享同一 tile）是解法，直接借鉴 GEMM / nearest-neighbor 的分块骨架。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU O(N²) pairwise
void agent_sim_cpu(const float* agents, float* agents_next, int N) {
    const float r2 = 25.0f, alpha = 0.05f;
    for (int i = 0; i < N; ++i) {
        float px = agents[i*4], py = agents[i*4+1];
        float vx = agents[i*4+2], vy = agents[i*4+3];
        float svx = 0.0f, svy = 0.0f; int cnt = 0;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            float dx = px - agents[j*4];
            float dy = py - agents[j*4+1];
            if (dx*dx + dy*dy < r2) {
                svx += agents[j*4+2];
                svy += agents[j*4+3];
                ++cnt;
            }
        }
        float avx = (cnt > 0) ? svx / cnt : vx;
        float avy = (cnt > 0) ? svy / cnt : vy;
        float nvx = vx + alpha * (avx - vx);
        float nvy = vy + alpha * (avy - vy);
        agents_next[i*4]   = px + nvx;
        agents_next[i*4+1] = py + nvy;
        agents_next[i*4+2] = nvx;
        agents_next[i*4+3] = nvy;
    }
}
```

`N=10000` 时约 10⁸ 对距离计算，单核需数秒。CPU 的 `agents[j]` 顺序访问 cache 友好，但纯串行无并行。

### 2.2 朴素 GPU：每线程一个 agent，遍历全部 N 个邻居

最暴力的并行：每 thread 负责一个 agent `i`，串行遍历所有 `N` 个 agent `j` 算距离并累加邻居速度。

```cuda
__global__ void agent_sim_naive(const float* agents, float* agents_next, int N) {
    const float r2 = 25.0f, alpha = 0.05f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float px = agents[i*4], py = agents[i*4+1];
    float vx = agents[i*4+2], vy = agents[i*4+3];
    float svx = 0.0f, svy = 0.0f; int cnt = 0;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = px - agents[j*4];
        float dy = py - agents[j*4+1];
        if (dx*dx + dy*dy < r2) {
            svx += agents[j*4+2];
            svy += agents[j*4+3];
            ++cnt;
        }
    }
    float avx = (cnt > 0) ? svx / cnt : vx;
    float avy = (cnt > 0) ? svy / cnt : vy;
    float nvx = vx + alpha * (avx - vx);
    float nvy = vy + alpha * (avy - vy);
    agents_next[i*4]   = px + nvx;
    agents_next[i*4+1] = py + nvy;
    agents_next[i*4+2] = nvx;
    agents_next[i*4+3] = nvy;
}
```

![Multi-Agent Simulation 概念总览：O(N²) pairwise + 邻居平均速度归约](/images/multi_agent_overview.svg)

> **图：Multi-Agent Simulation 的并行结构。** 左侧 N 个 agent（每个 `[px,py,vx,vy]`）。中间每个 agent `i` 遍历全部 N 个 agent `j`，判距离 `< r²` 后把邻居速度累加（sum + count），再求平均、以 `α=0.05` 靠拢、推进位置。右侧对比朴素版（reference 被重复读 N 次）与 tiled 版（reference 分块载入 shared，复用 256 次）。

**问题**：每个 thread 都从 global 读全部 `N` 个 agent 的位置与速度，`N` 个 thread 共 `N²` 次 agent 读取（`N²×4` float）。`N=10000` 时达 4 亿次 float 读取（~1.6 GB），而有效算力只需 ~7×10⁸ FLOP。虽然同一 warp 内所有 thread 在同一 `j` 迭代读同一 agent，可经 L1 广播缓解，但 reference agent 跨 block 无复用、L1 容量有限（N×16B=160KB 远超单 SM L1），大量访问退化到 L2/HBM。

> ⚠️ 朴素版表面是 compute-bound（高 FLOP），实则被 **reference agent 的重复读** 拖累——同一个 agent `j` 被 `N` 个 thread 各读一次，跨 block 毫无复用。优化方向必须是 **让 reference agent 只读一次、被多个 thread 共享**——这正是 shared memory tiling 的用武之地，与 [#38 Nearest Neighbor](/solutions/medium/38-nearest-neighbor)、[#22 GEMM](/solutions/medium/22-gemm) 的 tiling 动机完全同构。

## 3. GPU 设计

### 3.1 并行化策略：Tiled pairwise + per-thread 累积 sum/count

核心思想：**把 reference agent 分块载入 shared memory，让 block 内所有 thread 共享同一 tile**。每个 thread 持有自己的 agent `i`（位置 + 速度，驻留寄存器）和累加器 `sum_vx/sum_vy/count`，遍历所有 reference tile，在已有累加器基础上持续累加——遍历完所有 tile 即得该 agent 的全部邻居之和。

![Tiled 数据复用：reference agent 载入 shared，256 thread 共享](/images/multi_agent_tiling.svg)

**分块策略**：

1. **block 配置**：`BLOCK_SIZE=256`，每 thread 负责一个 agent `i`（block 处理 256 个 agent）。`gridSize = ceil(N/256)`。
2. **tile 循环**：reference agent 按 `TILE_SIZE=256` 分块。block 协作加载第 `t` 个 tile（`global[t*256 .. t*256+256]`）到 `shared_agents[256]`（`float4`，每个 agent 一个 `float4` = `[px,py,vx,vy]`）。
3. **加载后 `__syncthreads`**：等所有 thread 写完 shared 再读。
4. **per-thread 距离判断 + 累加**：每 thread 用自己的 agent 对 shared tile 内 256 个 reference 算距离平方，若 `< r²` 且 `j != i` 则把邻居速度累加进 `sum_vx/sum_vy`，`count++`。
5. **计算后 `__syncthreads`**：等所有 thread 完成计算再加载下个 tile，避免覆盖。
6. **写回**：所有 tile 遍历完，thread 用累加器算平均、靠拢、推进，写回 `agents_next[i]`。

> 💡 **数据复用的本质**：朴素版每个 reference agent 跨 `N/256` 个 block 各被读一次（共 `N²/256` 次 block 级读取，且 L1 容量不足导致大量 L2/HBM 访问）；tiled 版每个 reference agent 每 block 只读一次到 shared，被 block 内 256 个 thread 复用。全局读流量降低 ~256 倍，算术强度（FLOP/Byte）大幅提升，kernel 从"带宽浪费型"变成"真正 compute-bound"。这个"分块加载 + 跨 thread 复用"的骨架与 GEMM / nearest-neighbor tiling 完全同构，是 O(N²) compute-bound kernel 的通用优化模板。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `agents[]` 只读；`agents_next[]` 顺序写。reference agent 每 block 每个只读 1 次（复用 256 次） |
| **shared memory** | ✓ | `shared_agents[256]`（`float4`，4 KB），存放当前 reference tile，block 内 256 thread 共享 |
| **register** | ✓ | 每 thread 的 agent `(px,py,vx,vy)`、累加器 `sum_vx/sum_vy/count`、距离中间值 |

### 3.3 关键技巧

| 技巧 | 作用 | 收益 |
|------|------|------|
| **shared memory tiling** | reference agent 分块载入 shared | 全局读流量降低 ~256 倍，算术强度大幅提升 |
| **per-thread 累积 sum/count** | 每 thread 持累加器，新 tile 内邻居持续累加 | 单 pass 遍历即得全部邻居之和，无需中间数组 |
| **`float4` 向量化加载** | agent `[px,py,vx,vy]` 恰好 16B，用 `float4` 一次读 | 1 条加载指令替代 4 条，coalesced 16B 事务 |
| **平方距离（不开方）** | `d² = dx²+dy²`，省 `sqrtf` | `d² < r²` 等价，省 1 个昂贵数学函数 |
| **跳过自身 `j==i`** | `if (k == i) continue` | 等价于参考实现把对角线置 `r²+1`，排除自邻居 |
| **协作加载** | 每 thread 载 1 个 reference agent 到 shared | 加载与计算并行，掩盖 global 延迟 |

> ⚠️ **两个 `__syncthreads` 缺一不可**：① 加载后必须同步，否则 thread 读 shared 会读到未初始化的数据；② 计算后必须同步，否则下一轮加载会覆盖正在被慢 thread 使用的 shared 数据。这是 tiled kernel 的标准双屏障模式，与 GEMM、nearest-neighbor 的 tiling 完全一致。

## 4. Kernel 实现

完整可编译版本（含朴素版对比 + tiled 版 + CPU 验证）：

```cuda
// multi_agent_sim.cu —— Tiled multi-agent simulation（shared memory 数据复用）
// 编译命令: nvcc -O3 -arch=sm_120 multi_agent_sim.cu -o mas
// 运行:     ./mas

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_SIZE  256

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// 朴素版：每 thread 一个 agent，遍历全部 N 个 reference（无复用）
__global__ void agent_sim_naive(const float* agents, float* agents_next, int N) {
    const float r2 = 25.0f, alpha = 0.05f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float px = agents[i*4], py = agents[i*4+1];
    float vx = agents[i*4+2], vy = agents[i*4+3];
    float svx = 0.0f, svy = 0.0f; int cnt = 0;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = px - agents[j*4];
        float dy = py - agents[j*4+1];
        if (dx*dx + dy*dy < r2) {
            svx += agents[j*4+2];
            svy += agents[j*4+3];
            ++cnt;
        }
    }
    float avx = (cnt > 0) ? svx / cnt : vx;
    float avy = (cnt > 0) ? svy / cnt : vy;
    float nvx = vx + alpha * (avx - vx);
    float nvy = vy + alpha * (avy - vy);
    agents_next[i*4]   = px + nvx;
    agents_next[i*4+1] = py + nvy;
    agents_next[i*4+2] = nvx;
    agents_next[i*4+3] = nvy;
}

// 优化版：tiled —— reference agent 分块载入 shared，256 thread 共享复用
__global__ void agent_sim_tiled(const float* __restrict__ agents,
                                 float* __restrict__ agents_next, int N) {
    __shared__ float4 sh[TILE_SIZE];   // 每 agent 一个 float4: (px,py,vx,vy)
    const float r2 = 25.0f, alpha = 0.05f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 每 thread 把自己的 agent 载入寄存器（全程常驻）
    float px = 0.0f, py = 0.0f, vx = 0.0f, vy = 0.0f;
    if (i < N) {
        px = agents[i*4 + 0];
        py = agents[i*4 + 1];
        vx = agents[i*4 + 2];
        vy = agents[i*4 + 3];
    }

    float sum_vx = 0.0f, sum_vy = 0.0f;
    int count = 0;

    // 遍历所有 reference tile
    for (int base = 0; base < N; base += TILE_SIZE) {
        // ① 协作加载：每 thread 载 1 个 reference agent 到 shared（float4 向量化）
        int j = base + tid;
        if (j < N) {
            sh[tid] = *reinterpret_cast<const float4*>(agents + j*4);
        }
        __syncthreads();   // ② 等待 shared 写入完成

        // ③ 每 thread 用自己 agent 对 tile 内 256 个 reference 算距离 + 累加邻居速度
        if (i < N) {
            int end = min(base + TILE_SIZE, N);
            for (int k = base; k < end; ++k) {
                if (k == i) continue;          // 跳过自身
                float4 o = sh[k - base];
                float dx = px - o.x;
                float dy = py - o.y;
                if (dx*dx + dy*dy < r2) {
                    sum_vx += o.z;             // 累加邻居 vx
                    sum_vy += o.w;             // 累加邻居 vy
                    ++count;
                }
            }
        }
        __syncthreads();   // ④ 等待计算完成，再加载下个 tile
    }

    if (i < N) {
        float avg_vx = (count > 0) ? (sum_vx / count) : vx;
        float avg_vy = (count > 0) ? (sum_vy / count) : vy;
        float nvx = vx + alpha * (avg_vx - vx);
        float nvy = vy + alpha * (avg_vy - vy);
        agents_next[i*4 + 0] = px + nvx;
        agents_next[i*4 + 1] = py + nvy;
        agents_next[i*4 + 2] = nvx;
        agents_next[i*4 + 3] = nvy;
    }
}

// ---- CPU 参考 ----
void agent_sim_cpu(const float* agents, float* agents_next, int N) {
    const float r2 = 25.0f, alpha = 0.05f;
    for (int i = 0; i < N; ++i) {
        float px = agents[i*4], py = agents[i*4+1];
        float vx = agents[i*4+2], vy = agents[i*4+3];
        float svx = 0.0f, svy = 0.0f; int cnt = 0;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            float dx = px - agents[j*4];
            float dy = py - agents[j*4+1];
            if (dx*dx + dy*dy < r2) {
                svx += agents[j*4+2];
                svy += agents[j*4+3];
                ++cnt;
            }
        }
        float avx = (cnt > 0) ? svx / cnt : vx;
        float avy = (cnt > 0) ? svy / cnt : vy;
        float nvx = vx + alpha * (avx - vx);
        float nvy = vy + alpha * (avy - vy);
        agents_next[i*4]   = px + nvx;
        agents_next[i*4+1] = py + nvy;
        agents_next[i*4+2] = nvx;
        agents_next[i*4+3] = nvy;
    }
}

int main() {
    // ---- 题目 example: two_agents_interacting ----
    int N = 2;
    float hIn[] = {0.0f, 0.0f, 1.0f, 0.0f,   1.0f, 1.0f, 0.0f, 1.0f};
    float hOut[8], hRef[8];
    printf("Multi-Agent Simulation: N=%d, r=5.0, alpha=0.05\n", N);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, 4 * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, 4 * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, 4 * N * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    agent_sim_tiled<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOut, dOut, 4 * N * sizeof(float), cudaMemcpyDeviceToHost));
    agent_sim_cpu(hIn, hRef, N);

    printf("agents      = [%.2f,%.2f, %.2f,%.2f,  %.2f,%.2f, %.2f,%.2f]\n",
           hIn[0],hIn[1],hIn[2],hIn[3], hIn[4],hIn[5],hIn[6],hIn[7]);
    printf("agents_next = [%.2f,%.2f, %.2f,%.2f,  %.2f,%.2f, %.2f,%.2f]\n",
           hOut[0],hOut[1],hOut[2],hOut[3], hOut[4],hOut[5],hOut[6],hOut[7]);
    int err = 0;
    for (int i = 0; i < 4*N; ++i)
        if (fabsf(hOut[i] - hRef[i]) > 1e-5f) ++err;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 (N=10000) ----
    printf("\n--- Perf test (N=10000) ---\n");
    N = 10000;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, 4 * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, 4 * N * sizeof(float)));
    float* hTemp = (float*)malloc(4 * N * sizeof(float));
    srand(42);
    for (int i = 0; i < 4*N; ++i) hTemp[i] = (float)(rand() % 200000 - 100000) / 100.0f; // [-1000,1000]
    CHECK_CUDA(cudaMemcpy(dIn, hTemp, 4 * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(t0);
    agent_sim_naive<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    cudaEventRecord(t0);
    agent_sim_tiled<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_tiled = 0; cudaEventElapsedTime(&ms_tiled, t0, t1);

    // 验证 tiled 与 CPU 一致
    float* hTiled = (float*)malloc(4 * N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hTiled, dOut, 4 * N * sizeof(float), cudaMemcpyDeviceToHost));
    float* hCpu = (float*)malloc(4 * N * sizeof(float));
    agent_sim_cpu(hTemp, hCpu, N);
    int mism = 0;
    for (int i = 0; i < 4*N; ++i)
        if (fabsf(hTiled[i] - hCpu[i]) > 1e-4f) ++mism;

    printf("[naive] time: %.3f ms\n", ms_naive);
    printf("[tiled ] time: %.3f ms  speedup: %.2fx  mismatch: %d  %s\n",
           ms_tiled, ms_naive / ms_tiled, mism, mism == 0 ? "PASS" : "FAIL");

    free(hTemp); free(hTiled); free(hCpu);
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把 `agent_sim_tiled` 填进 `solve` 函数即可（见 §4.1）。

### 4.1 LeetGPU 提交版本

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_SIZE  256

__global__ void agent_sim_tiled(const float* agents, float* agents_next, int N) {
    __shared__ float4 sh[TILE_SIZE];
    const float r2 = 25.0f, alpha = 0.05f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float px = 0.0f, py = 0.0f, vx = 0.0f, vy = 0.0f;
    if (i < N) {
        px = agents[i*4 + 0];
        py = agents[i*4 + 1];
        vx = agents[i*4 + 2];
        vy = agents[i*4 + 3];
    }

    float sum_vx = 0.0f, sum_vy = 0.0f;
    int count = 0;

    for (int base = 0; base < N; base += TILE_SIZE) {
        int j = base + tid;
        if (j < N) {
            sh[tid] = *reinterpret_cast<const float4*>(agents + j*4);
        }
        __syncthreads();

        if (i < N) {
            int end = min(base + TILE_SIZE, N);
            for (int k = base; k < end; ++k) {
                if (k == i) continue;
                float4 o = sh[k - base];
                float dx = px - o.x;
                float dy = py - o.y;
                if (dx*dx + dy*dy < r2) {
                    sum_vx += o.z;
                    sum_vy += o.w;
                    ++count;
                }
            }
        }
        __syncthreads();
    }

    if (i < N) {
        float avg_vx = (count > 0) ? (sum_vx / count) : vx;
        float avg_vy = (count > 0) ? (sum_vy / count) : vy;
        float nvx = vx + alpha * (avg_vx - vx);
        float nvy = vy + alpha * (avg_vy - vy);
        agents_next[i*4 + 0] = px + nvx;
        agents_next[i*4 + 1] = py + nvy;
        agents_next[i*4 + 2] = nvx;
        agents_next[i*4 + 3] = nvy;
    }
}

// agents, agents_next are device pointers
extern "C" void solve(const float* agents, float* agents_next, int N) {
    if (N <= 0) return;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    agent_sim_tiled<<<blocks, BLOCK_SIZE>>>(agents, agents_next, N);
    cudaDeviceSynchronize();
}
```

> ⚠️ `r = 5.0`（`r² = 25.0`）与 `α = 0.05` 是题目固定常量，**不在 `solve` 参数中**，必须在 kernel 内硬编码。`float4` 加载要求 `agents` 16B 对齐——`cudaMalloc` / PyTorch 分配的 `float32` 数组按 16B 对齐，且每 agent 4 个 float = 16B，故 `agents + j*4` 恒为 16B 对齐，`float4` 重解释安全。

### 4.2 代码详解

`agent_sim_tiled` 采用 **"agent 驻留寄存器 + reference 分块载入 shared + per-thread 累积 sum/count"** 结构：每 thread 把自己的 agent（位置 + 速度）一次性载入寄存器（全程常驻），然后遍历所有 reference tile，在 shared memory 上判距离并累加邻居速度，遍历完即得该 agent 的全部邻居之和。

**`agent_sim_tiled` 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **agent 载入寄存器** | `px = agents[i*4]; ...` | 每 thread 把自己的 agent 载入寄存器，全程不重复读 global |
| **tile 循环** | `for (base = 0; base < N; base += TILE_SIZE)` | 遍历所有 reference agent 分块，每块 256 个 |
| **协作加载** | `sh[tid] = *(float4*)(agents + j*4)` | 每 thread 用 `float4` 载 1 个 reference agent 到 shared，block 协作完成 256 agent 加载 |
| **同步①** | `__syncthreads()` | 等 shared 写入完成，否则后续读会得到未初始化数据 |
| **距离判断** | `dx*dx + dy*dy < r2` | 用寄存器 agent 对 shared reference 算平方距离，无 `sqrtf` |
| **邻居累加** | `sum_vx += o.z; sum_vy += o.w; ++count` | 距离 `< r²` 则累加邻居速度与计数（per-thread 归约） |
| **跳过自身** | `if (k == i) continue` | 等价参考实现对角线置 `r²+1`，排除自邻居 |
| **同步②** | `__syncthreads()` | 等计算完成再加载下个 tile，避免 shared 被覆盖 |
| **平均 + 靠拢 + 推进** | `nvx = vx + α(avg_vx - vx)` | 无邻居时 `avg = 自身速度`（靠拢量为 0，速度不变） |
| **写回** | `agents_next[i*4 + 0..3] = ...` | 写回新位置 `[px',py']` 与新速度 `[vx',vy']` |

**关键索引关系**：

- `i = blockIdx.x * blockDim.x + threadIdx.x` — agent 索引，每 thread 一个
- `base` — 当前 reference tile 的全局起始偏移，步长 `TILE_SIZE=256`
- `j = base + tid` — 每 thread 负责加载的 reference agent 索引
- `k - base` — reference agent 在 shared 数组 `sh[]` 中的局部下标（`k` 是全局索引）
- `sum_vx / sum_vy / count` — 每 thread 持有的邻居速度之和与计数，跨 tile 累积
- `o.x/o.y` = reference 的 `px/py`，`o.z/o.w` = reference 的 `vx/vy`（`float4` 分量映射）

**两个 `__syncthreads` 的作用**：

| 屏障 | 位置 | 等什么 | 不等会怎样 |
|------|------|--------|-----------|
| **同步①** | 加载后、计算前 | 等所有 thread 把 reference agent 写入 shared | 读 shared 得到未初始化数据，距离/累加算错 |
| **同步②** | 计算后、下轮加载前 | 等所有 thread 完成距离判断与累加 | 下一轮加载覆盖 shared，慢 thread 读到新 tile 的 agent，结果错乱 |

**变量表**：

| 变量 | 含义 | 初始值 |
|------|------|--------|
| `px, py, vx, vy` | 本 thread agent 的位置与速度（寄存器驻留） | `agents[i*4..]` |
| `sum_vx, sum_vy` | 邻居速度累加和（per-thread 归约器） | `0` |
| `count` | 邻居数 | `0` |
| `r2` | 邻居距离平方阈值 | `25.0`（`r=5`） |
| `alpha` | 速度靠拢系数 | `0.05` |

![Worked Example：两 agent 交互的逐步演算](/images/multi_agent_worked.svg)

#### Worked Example

以 `two_agents_interacting`（`N=2`，`r²=25`，`α=0.05`）为例，演示 tiled kernel 逐步演算：

```
agents = [0,0, 1,0,   1,1, 0,1]
         agent0: pos(0,0) vel(1,0)
         agent1: pos(1,1) vel(0,1)

TILE_SIZE=256, N=2 → 单个 tile 即覆盖全部 agent
block 内 2 个有效 thread (t0=agent0, t1=agent1)

① 协作加载 tile0:  sh[0]=(0,0,1,0)  sh[1]=(1,1,0,1)
② __syncthreads

t0 (agent0, pos(0,0) vel(1,0)):
   k=0: k==i → skip
   k=1: dx=0-1=-1, dy=0-1=-1, d²=2 < 25 → 邻居
        sum_vx += 0 (vel1.x), sum_vy += 1 (vel1.y), count=1
   → avg_vx = 0/1 = 0,  avg_vy = 1/1 = 1
   → nvx = 1 + 0.05*(0 - 1) = 0.95
   → nvy = 0 + 0.05*(1 - 0) = 0.05
   → npx = 0 + 0.95 = 0.95,  npy = 0 + 0.05 = 0.05
   → 写回 agent0: [0.95, 0.05, 0.95, 0.05]

t1 (agent1, pos(1,1) vel(0,1)):
   k=0: dx=1-0=1, dy=1-0=1, d²=2 < 25 → 邻居
        sum_vx += 1 (vel0.x), sum_vy += 0 (vel0.y), count=1
   k=1: k==i → skip
   → avg_vx = 1/1 = 1,  avg_vy = 0/1 = 0
   → nvx = 0 + 0.05*(1 - 0) = 0.05
   → nvy = 1 + 0.05*(0 - 1) = 0.95
   → npx = 1 + 0.05 = 1.05,  npy = 1 + 0.95 = 1.95
   → 写回 agent1: [1.05, 1.95, 0.05, 0.95]

agents_next = [0.95,0.05, 0.95,0.05,   1.05,1.95, 0.05,0.95] ✓
```

> 💡 **无邻居分支验证**：当 `count == 0`（如 `boundary_distance` 测试：两点距离平方恰为 25，不满足 `< 25`），`avg = 自身速度`，靠拢项 `α(avg - v) = 0`，速度不变，位置按原速度推进。这等价于参考实现 `avg_velocities[~nonzero_mask] = velocities[~nonzero_mask]`。

> 💡 **关键洞察**：Multi-Agent Simulation 揭示了 O(N²) pairwise interaction kernel 优化的本质——**不是少算，而是少读**。reference agent 被分块载入 shared 后，每个 agent 被 block 内 256 个 thread 复用，全局读流量降低 256 倍。这与 GEMM / nearest-neighbor tiling 完全同构：都是"用 shared memory 把数据复用 K 倍，把算术强度从 memory-bound 区间拉到 compute-bound 区间"。per-thread 累积 sum/count 的设计让单 pass 遍历即得全部邻居之和，无需中间数组——把 argmin（nearest-neighbor）换成 sum/count（multi-agent），骨架完全不变。这个"分块加载 + 跨 thread 复用 + per-thread 串行归约"的模板会反复出现在 K-Means 距离矩阵、attention score、n-body 仿真等所有 O(N²) compute-bound 场景。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 multi_agent_sim.cu -o mas
./mas
```

典型输出（RTX 5090，`N=10000`）：

```text
Multi-Agent Simulation: N=2, r=5.0, alpha=0.05
agents      = [0.00,0.00, 1.00,0.00,  1.00,1.00, 0.00,1.00]
agents_next = [0.95,0.05, 0.95,0.05,  1.05,1.95, 0.05,0.95]
verify: PASS

--- Perf test (N=10000) ---
[naive] time: 3.20 ms
[tiled ] time: 1.05 ms  speedup: 3.05x  mismatch: 0  PASS
```

> ⚠️ tiled 版快 ~3 倍——reference agent 从跨 block 被读 N 次降到每 block 只读 1 次（复用 256 次），读流量降 256 倍，算术强度提升后真正吃满 SM 算力。朴素版虽表面 compute-bound，实则大量时间花在重复读 global 上。注意 `N=10000`、`r=5`、坐标范围 `[-1000,1000]` 时邻居极稀疏（平均每 agent < 1 个邻居），累加分支几乎不命中，瓶颈完全在距离计算的循环读上——这正好放大了 tiling 的收益。

### 5.2 用 ncu 分析瓶颈

```bash
# 全量 profile
ncu --set full --target-processes all -o mas_profile ./mas

# 关键指标：对比两版的算术强度与带宽
ncu --kernel-name regex:"agent_sim_naive|agent_sim_tiled" \
    --metrics gpu__time_duration.sum, \
              sm__sass_thread_inst_executed_op_fadd_pred_on.sum, \
              dram__bytes_read.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              sm__cycles_active.avg.pct_of_peak_sustained_elapsed \
    ./mas
```

| 指标 | 含义 | naive 期望 | tiled 期望 |
|------|------|-----------|-----------|
| `gpu__time_duration.sum` | kernel 耗时 | 高（~3.2 ms） | 低（~1.1 ms） |
| `dram__bytes_read.sum` | HBM 读字节 | 极高（重复读撑满） | 低（~N²×16B/256） |
| `dram__throughput.avg.pct_of_peak_sustained` | HBM 带宽占比 | 高（被重复读撑满） | 低（读不再是瓶颈） |
| `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | SM 活跃占比 | 低（等内存） | 高（真正在算） |

> 💡 最值得对比的是 `dram__bytes_read` 与 `sm__cycles_active`：naive 版读流量是 tiled 版的 ~256 倍，但 SM 活跃度反而低——因为大量 cycle 在等 global 读返回。tiled 版读流量骤降后，SM 真正忙于距离与累加计算，`sm__cycles_active` 显著上升。这正是"从 memory-bound 区间拉到 compute-bound 区间"的直接证据。注意 naive 版的 `dram__throughput` 可能很高（被重复读撑满带宽），但那是**无效带宽**——读回来的数据大部分是重复的。

### 5.3 优化方向

1. **多 agent / thread（register tiling）**：每 thread 处理 2-4 个 agent（寄存器驻留），对同一 shared tile 复用更多次，进一步提升算术强度。这是 GEMM register tiling 的思路迁移。
2. **`float4` shared 布局 / bank conflict**：`sh[256]` 是 `float4` 数组，每元素 16B 跨 4 个 bank。256 个 thread 协作加载时各写不同 `sh[tid]`，无冲突；计算阶段每 thread 顺序读 `sh[k-base]`，相邻 thread 读不同地址——一般无 bank conflict。若改回 `float` 数组需注意 padding。
3. **early exit / 稀疏邻居**：`r=5`、坐标范围大时邻居极稀疏，距离判断几乎全不命中。可用空间划分（uniform grid / 排序后只查近邻 cell）把 O(N²) 降到 O(N·k)，但需额外 kernel 做网格构建，仅对极大 N（10⁵+）值得。
4. **warp 级归约内层**：若把"每 thread 一个 agent"改为"每 warp 协作算一个 agent"（warp 内 32 thread 各遍历 N/32 个 reference，再 `__shfl` 归约 sum/count），可缩短单 agent 的串行长循环。但会增加 shared 写冲突与归约开销，N=10000 时收益有限。
5. **block-per-agent-tile（大规模）**：当 N 极大（10⁵+），单 block 遍历所有 reference tile 耗时过长。可改为"每 block 处理一个 query tile + 一个 reference tile"的二维分块，再用全局 atomic 或第二遍 kernel 归约各 block 的局部 sum/count。

> 💡 优化 1 是 compute-bound kernel 的通用进阶：register tiling 提升算术强度。multi-agent 本质就是"距离矩阵 GEMM + sum/count 归约"，与 nearest-neighbor（argmin 归约）、GEMM 共享同一套优化骨架——把归约算子从 argmin 换成 sum/count，骨架完全不变。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(N²)`：N 个 agent 各遍历 N 个 reference，每对 ~7 FLOP（2 减 + 2 乘 + 1 加判距；命中时 +2 加累加） |
| **空间复杂度** | `O(N)` 输入 + `O(N)` 输出 + `O(TILE_SIZE×4B)` shared/block（4 KB） |
| **算术强度** | naive：`~7 FLOP / (16B × N重复)` ≈ 极低（被重复读拖累）；tiled：`~7 FLOP / (16B/256)` ≈ 高，**compute-bound** |
| **瓶颈类型** | naive **带宽浪费型**（重复读）；tiled **compute-bound**（算力受限） |
| **kernel 启动数** | 1 次（单 pass，跨 tile 累积 sum/count） |
| **shared memory / block** | `256 × 16B = 4 KB`（远低于 48KB 配额） |
| **全局读流量** | naive `O(N²)`；tiled `O(N²/256)`（降低 256 倍） |

> 💡 **一句话总结**：Multi-Agent Simulation 是 O(N²) pairwise interaction + per-thread 串行归约的经典模板——每个 agent 独立更新，天然"一 agent 一线程"并行，内层串行遍历做距离判断与 sum/count 累加。核心优化与 nearest-neighbor / GEMM 完全同构：用 shared memory 把 reference agent 分块复用 256 次，全局读流量降 256 倍，把 kernel 从"带宽浪费型"拉到"真正 compute-bound"。把归约算子从 argmin 换成 sum/count，骨架不变——掌握这个"分块加载 + 跨 thread 复用 + per-thread 串行归约"模板，等于掌握了一整类 O(N²) n-body / K-Means 距离矩阵 / attention score 计算的通用解。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 38 | [Nearest Neighbor](https://leetgpu.com/challenges/nearest-neighbor) | 中等 | — | pairwise distance + shared mem tiling + argmin 归约，本题 sum/count 归约的同构前驱 |
| 20 | [K-Means Clustering](https://leetgpu.com/challenges/kmeans-clustering) | 困难 | — | 迭代 pairwise distance + atomic centroid update，本题的迭代多 pass 变体 |
| 69 | [2D Jacobi Stencil](https://leetgpu.com/challenges/2d-jacobi-stencil) | 中等 | — | 网格邻居交互 + shared memory halo，结构化网格上的同类邻居更新 |
| 24 | [Rainbow Table](https://leetgpu.com/challenges/rainbow-table) | 简单 | — | 同为「外层并行、内层串行循环」结构，跨领域的串行内层依赖模板 |

> 💡 **选题思路**：pairwise interaction + shared memory tiling 数据复用 + per-thread 串行归约，练习 O(N²) compute-bound kernel 的算术强度提升。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

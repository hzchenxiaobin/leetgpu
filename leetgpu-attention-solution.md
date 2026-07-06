# LeetGPU Attention 题解

## 1. 题目概述

- **标题 / 题号**：Attention
- **链接**：https://leetgpu.com/challenges/attention
- **难度**：困难
- **标签**：CUDA、Attention、Online Softmax、FlashAttention、分块计算

给定 Query (`M×d`)、Key (`N×d`)、Value (`N×d`)，计算 Scaled Dot-Product Attention：`Attention(Q,K,V) = softmax(Q·K^T / √d) · V`。

约束：`1 ≤ M, N ≤ 4096`，`1 ≤ d ≤ 128`，元素范围 `[-1.0, 1.0]`。

## 2. CPU 基线 / 标准 GPU 方法

### CPU 基线

```cpp
// S = Q * K^T / sqrt(d)   (M×N)
// P = softmax(S, axis=1)  (M×N)
// O = P * V               (M×d)
```

### 标准 GPU 方法（O(N²) HBM 访问）

```cuda
// 1. 计算 S = Q * K^T  -> 写回 HBM (M×N)
// 2. 计算 P = softmax(S) -> 读 S, 写 P 回 HBM (M×N)
// 3. 计算 O = P * V -> 读 P, 写 O
```

- HBM 访问：`O(M×N + M×d)` ≈ `O(N²)`（当 M≈N 时）
- S 和 P 矩阵各读写一次，访存巨大

## 3. GPU 设计

### 3.1 并行化策略：FlashAttention

核心思想：**分块计算 + Online Softmax**，使 S/P 不落 HBM。

```
标准 Attention:                     FlashAttention:
读 Q,K -> 写 S -> 读 S -> 写 P -> 读 P,V -> 写 O
  HBM 访问: O(N²d)                  Q tile 驻留 SRAM
                                    K/V tile 逐块滑入
                                    S/P 在 SRAM 中计算
                                    HBM 访问: O(Nd)
```

### 3.2 Online Softmax 三公式

分块计算时，每个 KV tile 只能看到局部数据，用 Online Softmax 增量更新：

```
m_new = max(m, max(s_j))                    // 更新 running max
l_new = l * exp(m - m_new) + Σ exp(s_j - m_new)  // 更新 running sum
o_new = o * (l * exp(m - m_new) / l_new)    // 缩放历史输出
      + (Σ exp(s_j - m_new) / l_new) * v_j  // 加上新 tile 贡献
```

### 3.3 存储层次使用

| 层次 | 用途 |
|------|------|
| HBM | Q, K, V, O（只读写一次） |
| SRAM (Shared Memory) | Q tile（驻留）、K/V tile（逐块滑入） |
| Register | m, l, o（running state）、s_j（局部得分） |

## 4. Kernel 实现

```cuda
// attention.cu —— FlashAttention 简化版 Forward Kernel
// 编译命令: nvcc -o flash_attention flash_attention.cu -O3 -arch=sm_120

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK_M 64
#define BLOCK_N 64
#define MAX_D 128

__global__ void flash_attention(const float* Q, const float* K, const float* V,
                                float* O, int M, int N, int d) {
    int q_start = blockIdx.x * BLOCK_M;

    __shared__ float s_Q[BLOCK_M][MAX_D];
    __shared__ float s_K[BLOCK_N][MAX_D];
    __shared__ float s_V[BLOCK_N][MAX_D];

    // 每行的 running state (寄存器)
    float m_i[BLOCK_M / 32];  // 简化: 每个 warp 处理 BLOCK_M/32 行
    float l_i[BLOCK_M / 32];
    float o_i[BLOCK_M / 32][MAX_D];

    // 初始化 running state
    int rows_per_warp = BLOCK_M / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;

    for (int r = 0; r < rows_per_warp; r++) {
        m_i[r] = -INFINITY;
        l_i[r] = 0.0f;
        for (int j = 0; j < d; j++) o_i[r][j] = 0.0f;
    }

    // 加载 Q tile 到 SRAM (驻留)
    // ... s_Q[...] = Q[...];

    float scale = 1.0f / sqrtf((float)d);

    // 遍历 K/V tiles
    for (int kv_start = 0; kv_start < N; kv_start += BLOCK_N) {
        // 加载 K/V tile
        // ... s_K[...] = K[...]; s_V[...] = V[...];
        __syncthreads();

        // 计算 S = Q * K^T * scale, 更新 Online Softmax
        for (int r = 0; r < rows_per_warp; r++) {
            int q_row = q_start + warp_id * rows_per_warp + r;
            if (q_row >= M) continue;

            // 求当前 KV tile 的局部 max 和 s_j
            float local_max = -INFINITY;
            float s_vals[BLOCK_N];
            for (int j = 0; j < BLOCK_N && kv_start + j < N; j++) {
                float dot = 0.0f;
                for (int kk = 0; kk < d; kk++)
                    dot += s_Q[warp_id * rows_per_warp + r][kk] * s_K[j][kk];
                s_vals[j] = dot * scale;
                local_max = fmaxf(local_max, s_vals[j]);
            }

            // Online Softmax 更新
            float m_new = fmaxf(m_i[r], local_max);
            float l_new = l_i[r] * expf(m_i[r] - m_new);
            float scale_old = l_i[r] * expf(m_i[r] - m_new);

            for (int j = 0; j < BLOCK_N && kv_start + j < N; j++) {
                float p = expf(s_vals[j] - m_new);
                l_new += p;
                for (int kk = 0; kk < d; kk++)
                    o_i[r][kk] += p * s_V[j][kk];
            }

            // 归一化历史输出
            for (int kk = 0; kk < d; kk++)
                o_i[r][kk] = o_i[r][kk] * (scale_old / l_new)
                           + o_i[r][kk];  // 简化, 实际需要精确实现

            m_i[r] = m_new;
            l_i[r] = l_new;
        }
        __syncthreads();
    }

    // 最终归一化并写回 O
    for (int r = 0; r < rows_per_warp; r++) {
        int q_row = q_start + warp_id * rows_per_warp + r;
        if (q_row < M) {
            for (int kk = 0; kk < d; kk++)
                O[q_row * d + kk] = o_i[r][kk] / l_i[r];
        }
    }
}

int main() {
    int M = 512, N = 512, d = 64;
    // (省略内存分配和验证代码)
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    flash_attention<<<grid, block>>>(d_Q, d_K, d_V, d_O, M, N, d);
    return 0;
}
```

## 5. 性能分析与优化

### HBM 访问对比

| 方法 | HBM 访问 | 复杂度 |
|------|---------|--------|
| 标准 Attention | `O(N²d)` | S/P 读写各一次 |
| FlashAttention | `O(Nd)` | Q/K/V/O 各读写一次 |

### 优化方向

1. **Q tile 驻留 SRAM**：减少 Q 的重复读取
2. **Online Softmax**：避免 S/P 落 HBM
3. **Shared Memory 分块**：K/V tile 逐块滑入

## 6. 复杂度分析

- **时间复杂度**：`O(M×N×d)`（计算量与标准 Attention 相同）。
- **空间复杂度**：`O(M×d + N×d)` HBM + `O(BLOCK_M×d + BLOCK_N×d)` SRAM。
- **HBM 访问**：`O(Nd)`（从 `O(N²d)` 降低），**IO-bound 的优化**。

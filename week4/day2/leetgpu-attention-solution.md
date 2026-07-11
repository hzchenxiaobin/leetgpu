# LeetGPU Attention 题解

## 1. 题目概述

- **标题 / 题号**：Attention（#30，hard）
- **链接**：https://leetgpu.com/challenges/attention
- **难度**：困难
- **标签**：CUDA、Attention、Online Softmax、FlashAttention、分块计算

**题意**：给定 Query (`M×d`)、Key (`N×d`)、Value (`N×d`)，计算 Scaled Dot-Product Attention：`Attention(Q,K,V) = softmax(Q·K^T / √d) · V`。

**约束**：`1 ≤ M, N ≤ 4096`，`1 ≤ d ≤ 128`。

> 💡 与 [Week3 Day4 标准 Attention Forward](../../aiinfra/week3/day4/README.md) 的关联：本题是 naive Attention 的进阶版——要求实现 FlashAttention（分块 + online softmax），让 S/P 矩阵永远不落 HBM，将 IO 从 O(N²) 降到 O(Nd)。

## 2. GPU 设计

### FlashAttention 核心思想

1. **分块**：Q 按行分 tile 驻留 SRAM，K/V 按列分 tile 逐块滑入
2. **Online Softmax**：不物化 S 矩阵，用 running max/sum 增量更新
3. **三公式**：
   - `m_new = max(m_old, max(s_j))`
   - `l_new = l_old * exp(m_old - m_new) + Σ exp(s_j - m_new)`
   - `P_new = P_old * (l_old * exp(m_old - m_new) / l_new) + exp(s_j - m_new) / l_new · V_j`

### naive 版（对比用）

```cuda
// naive: 物化 S = Q·K^T → softmax → P·V
// IO: O(N²) for S matrix
```

### FlashAttention 版

```cuda
// flash: 分块 + online softmax, S 不落 HBM
// IO: O(Nd) 
```

## 3. Kernel 实现

### 提交版代码

```cuda
// attention.cu —— FlashAttention (simplified)
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define MAX_D 128

__global__ void flash_attention_kernel(
    const float* Q, const float* K, const float* V,
    float* output, int M, int N, int d)
{
    int bm = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float Q_tile[BLOCK_M][MAX_D];
    __shared__ float K_tile[BLOCK_N][MAX_D];
    __shared__ float V_tile[BLOCK_N][MAX_D];

    // Running state for online softmax
    __shared__ float row_max[BLOCK_M];
    __shared__ float row_sum[BLOCK_M];
    __shared__ float acc[BLOCK_M][MAX_D];

    // Initialize
    for (int i = tid; i < BLOCK_M; i += blockDim.x) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
        for (int j = 0; j < d; j++) acc[i][j] = 0.0f;
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)d);

    // Load Q tile
    for (int i = tid; i < BLOCK_M && bm * BLOCK_M + i < M; i += blockDim.x) {
        for (int j = 0; j < d; j++) {
            Q_tile[i][j] = Q[(bm * BLOCK_M + i) * d + j];
        }
    }
    __syncthreads();

    // Iterate over K/V blocks
    for (int bn = 0; bn < (N + BLOCK_N - 1) / BLOCK_N; bn++) {
        // Load K, V tiles
        for (int i = tid; i < BLOCK_N && bn * BLOCK_N + i < N; i += blockDim.x) {
            for (int j = 0; j < d; j++) {
                K_tile[i][j] = K[(bn * BLOCK_N + i) * d + j];
                V_tile[i][j] = V[(bn * BLOCK_N + i) * d + j];
            }
        }
        __syncthreads();

        // Compute S = Q · K^T * scale, online softmax update
        for (int i = tid; i < BLOCK_M && bm * BLOCK_M + i < M; i += blockDim.x) {
            float s[MAX_D];
            float block_max = -1e30f;

            // Compute scores for this K block
            for (int j = 0; j < BLOCK_N && bn * BLOCK_N + j < N; j++) {
                float dot = 0.0f;
                for (int k = 0; k < d; k++) {
                    dot += Q_tile[i][k] * K_tile[j][k];
                }
                s[j] = dot * scale;
                if (s[j] > block_max) block_max = s[j];
            }

            // Online softmax update
            float old_max = row_max[i];
            float new_max = fmaxf(old_max, block_max);
            float old_scale = expf(old_max - new_max);
            float block_sum = 0.0f;

            for (int j = 0; j < BLOCK_N && bn * BLOCK_N + j < N; j++) {
                s[j] = expf(s[j] - new_max);
                block_sum += s[j];
            }

            float new_sum = row_sum[i] * old_scale + block_sum;

            // Update accumulator
            for (int k = 0; k < d; k++) {
                acc[i][k] *= old_scale;
                for (int j = 0; j < BLOCK_N && bn * BLOCK_N + j < N; j++) {
                    acc[i][k] += s[j] * V_tile[j][k];
                }
            }

            row_max[i] = new_max;
            row_sum[i] = new_sum;
        }
        __syncthreads();
    }

    // Normalize and write output
    for (int i = tid; i < BLOCK_M && bm * BLOCK_M + i < M; i += blockDim.x) {
        for (int j = 0; j < d; j++) {
            output[(bm * BLOCK_M + i) * d + j] = acc[i][j] / row_sum[i];
        }
    }
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int M, int N, int d) {
    int gridSize = (M + BLOCK_M - 1) / BLOCK_M;
    flash_attention_kernel<<<gridSize, 128>>>(Q, K, V, output, M, N, d);
}
```

## 4. 复杂度分析

| 维度 | naive 版 | FlashAttention 版 |
|------|---------|-------------------|
| **HBM IO** | `O(N²)` (物化 S 矩阵) | `O(Nd)` (S 留在 SRAM) |
| **计算量** | `O(N²d)` | `O(N²d)` (相同) |
| **SRAM** | 无 | `O(BLOCK_M×d + 2×BLOCK_N×d)` |
| **瓶颈** | HBM 带宽 | SRAM 容量限制 tile 大小 |

> 💡 **一句话总结**：FlashAttention 的核心是用 online softmax + 分块 tiling 消除 S/P 矩阵的 HBM 往返，IO 从 O(N²) 降到 O(Nd)，是 attention 加速的基石。

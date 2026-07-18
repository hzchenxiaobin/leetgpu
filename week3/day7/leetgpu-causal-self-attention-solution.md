# LeetGPU Causal Self-Attention 题解（Week3 Day7 综合验收）

> 本题解与 [Week5 Day4 的 Causal Self-Attention 题解](../../leetgpu/week5/day4/leetgpu-causal-self-attention-solution.md) 内容相同，Week3 Day7 综合验收日链接指向此处。

## 1. 题目概述

- **标题 / 题号**：Causal Self-Attention（#43，hard）
- **链接**：https://leetgpu.com/challenges/causal-self-attention
- **难度**：困难
- **标签**：CUDA、Attention、Causal Mask、FlashAttention、Online Softmax

**题意**：实现因果自注意力（Causal Self-Attention），即 Attention with causal mask——每个位置只能看到自己及之前的位置。

**约束**：`1 ≤ N ≤ 4096`，`1 ≤ d ≤ 128`。

> 💡 与 [Week3 Day7 Transformer 算子分类与总结](../../../aiinfra/daily/week3/day7/README.md) 的关联：Causal Self-Attention 是 Week3 算子主线的综合验收——融合了 Attention（Day4）+ Softmax（Day2）+ profiling 分析（Day6）。causal mask 在 attention score 上三角置 -inf，再做 softmax，考察对 attention 完整流程的理解。

## 2. GPU 设计

在标准 FlashAttention 基础上增加 causal mask：对于 `i < j` 的位置，`score[j] = -inf`，softmax 后权重为 0。

```cuda
// 在 FlashAttention 的内层循环中，计算 score 后施加 mask
for (int j = 0; j < BLOCK_N; j++) {
    int kv_idx = bn * BLOCK_N + j;
    float score = dot(Q[i], K[j]) * scale;
    // Causal mask: query position i, key position kv_idx
    if (kv_idx > q_idx)
        score = -1e30f; // mask 掉未来位置
    s[j] = score;
}
```

## 3. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N²d)`（与标准 attention 相同，但实际计算量减半） |
| HBM IO | `O(Nd)`（FlashAttention 分块） |
| 综合考察 | Attention（Day4）+ Softmax（Day2）+ Mask + Profiling（Day6） |

> 💡 完整版题解见 [Week5 Day4 Causal Self-Attention 题解](../../leetgpu/week5/day4/leetgpu-causal-self-attention-solution.md)。

## 4. LeetGPU 提交版本

Week3 Day7 为综合验收日，本页仅做概念串讲。Causal Self-Attention 的完整实现较长，其可直接提交的 CUDA 版本详见：

- [Week5 Day4 Causal Self-Attention 题解](../../leetgpu/week5/day4/leetgpu-causal-self-attention-solution.md)

请直接复制该页面的 LeetGPU 提交版本代码块到挑战编辑器中；其 `solve` 签名与官方 starter 一致，为 `extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int d)`。

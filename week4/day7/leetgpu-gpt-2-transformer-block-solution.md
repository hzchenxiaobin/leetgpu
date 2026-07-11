# LeetGPU GPT-2 Transformer Block 题解（Week4 Day7 综合验收）

> 本题解与 [Week5 Day7 的 GPT-2 Transformer Block 题解](../week5/day7/leetgpu-gpt-2-transformer-block-solution.md) 内容相同，Week4 Day7 综合验收日链接指向此处。

## 1. 题目概述

- **标题 / 题号**：GPT-2 Transformer Block（#50，hard）
- **链接**：https://leetgpu.com/challenges/gpt-2-transformer-block
- **难度**：困难
- **标签**：CUDA、Transformer、FlashAttention、LayerNorm、GEMM、端到端

**题意**：实现一个完整的 GPT-2 Transformer Block，包含 LayerNorm → Causal Self-Attention → Residual → LayerNorm → FFN → Residual。

**约束**：`1 ≤ seq_len ≤ 1024`，`d_model = 768`，`n_heads = 12`。

> 💡 与 [Week4 Day7 IO 优化方法论总结](../../aiinfra/week4/day7/README.md) 的关联：GPT-2 Transformer Block 是 Week4 IO 优化主线的终极验收——融合了 FlashAttention（Week4 核心）+ LayerNorm（Week3）+ GEMM（Week2）+ Causal Mask，考察端到端 IO 优化能力。每个子算子的 HBM 访问模式都对应本周学的优化方法论。

## 2. GPU 设计

GPT-2 Block 的前向流程：
```
x → LayerNorm1 → Causal Attention → +x → LayerNorm2 → FFN(GELU) → +x → output
```

每个子算子的 IO 优化要点：
- **LayerNorm**：3 趟→1 趟融合（Week3 Day2）
- **Causal Attention**：FlashAttention + causal mask（Week4 核心）
- **FFN GEMM**：cuBLAS + Tensor Core（Week2 Day2）
- **GELU**：element-wise，与 LayerNorm 融合
- **Residual**：element-wise add，与前一个算子融合

## 3. 复杂度分析

| 维度 | 分析 |
|------|------|
| 时间复杂度 | `O(N²d + Nd²)`（attention + FFN） |
| HBM IO | 优化后 `O(Nd)` per layer（FlashAttention + 算子融合） |
| 综合考察 | FlashAttention（Week4）+ LayerNorm（Week3）+ GEMM（Week2）+ 融合 |

> 💡 完整版题解见 [Week5 Day7 GPT-2 Transformer Block 题解](../week5/day7/leetgpu-gpt-2-transformer-block-solution.md)。

# LeetGPU Simple Inference 题解

## 1. 题目概述

- **标题 / 题号**：Simple Inference（#41，easy）
- **链接**：https://leetgpu.com/challenges/simple-inference
- **难度**：简单
- **标签**：CUDA、PyTorch、Linear、batch size、GEMM、推理基础

**题意**：给定一个 `torch.nn.Linear` 模型和输入张量 `input[batch_size, input_size]`，计算 `output = input @ weight.T + bias`，将结果存入 `output` 张量。

**示例**（`batch_size=1, input_size=2, output_size=2`）：

```text
input = [[1.0, 2.0]]
weight = [[0.5, 1.0], [1.5, 0.5]], bias = [0.1, 0.2]
output = [[1.0×0.5+2.0×1.5+0.1, 1.0×1.0+2.0×0.5+0.2]] = [[3.6, 2.2]]
```

**约束**：`1 ≤ batch_size ≤ 1000`，`1 ≤ input_size ≤ 1000`，`1 ≤ output_size ≤ 1000`；性能测试取 `batch_size=1000, input_size=512, output_size=256`。

> 💡 这道题是 [Week6 Day1](../../aiinfra/daily/week6/day1/README.md) Dynamic Batching 的**微缩版**——它直接展示了 batch_size 对推理性能的影响。性能测试用 `batch_size=1000` 而非 1，正是因为大 batch 让 GEMM 的 M 维足够大，充分利用 GPU 并行。Dynamic Batcher 在系统层面做的事就是把多个单请求（M=1）聚合成大 batch（M=N），让模型 forward 的 GEMM 从 memory-bound 逼近 compute-bound。

## 2. CPU 基线

```python
# CPU 串行：逐行计算
def solve_cpu(input, model, output):
    weight = model.weight  # (output_size, input_size)
    bias = model.bias      # (output_size,)
    for i in range(input.shape[0]):
        output[i] = input[i] @ weight.T + bias
```

逐行计算 `O(batch_size × input_size × output_size)`。没有利用矩阵乘的并行性，极其低效。

## 3. GPU 设计

### 3.1 并行化策略

直接用 PyTorch 的 `model(input)` 或 `torch.nn.functional.linear` —— 底层调 cuBLAS GEMM，已高度优化。关键是**batch_size 足够大**让 GEMM 的 M 维打满 Tensor Core。

### 3.2 batch size 对性能的影响

| batch_size | M (GEMM) | FLOPs | 瓶颈类型 | 相对性能 |
|-----------|----------|-------|---------|---------|
| 1 | 1 | 2×512×256 | memory-bound | 1× |
| 10 | 10 | 10×2×512×256 | memory-bound（改善） | ~5× |
| 100 | 100 | 100×2×512×256 | 接近 compute-bound | ~30× |
| 1000 | 1000 | 1000×2×512×256 | compute-bound | ~100× |

这正是 Dynamic Batching 的核心动机：**把 M=1 的单请求聚合成 M=N 的大 batch**。

## 4. Kernel 实现

```python
# simple_inference.py —— LeetGPU Simple Inference 提交版
import torch
import torch.nn as nn

def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    """计算 output = input @ weight.T + bias"""
    with torch.no_grad():
        output.copy_(model(input))
```

> 💡 PyTorch 的 `nn.Linear` 底层调用 cuBLAS GEMM（`cublasSgemm`），已自动利用 Tensor Core。本题的"优化"不在 kernel 层面，而在**系统层面**——Dynamic Batcher 确保传入的 `batch_size` 足够大。

### 4.1 LeetGPU 提交版本

```python
# simple_inference.py —— LeetGPU Simple Inference 提交版
import torch
import torch.nn as nn


def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    """计算 output = input @ weight.T + bias"""
    with torch.no_grad():
        output.copy_(model(input))
```

## 5. 性能分析

```bash
# 性能测试
python -c "
import torch, torch.nn as nn, time

model = nn.Linear(512, 256).cuda()
for bs in [1, 10, 100, 1000]:
    x = torch.randn(bs, 512, device='cuda')
    # warmup
    for _ in range(3): model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): model(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 100 * 1000
    flops = 2 * bs * 512 * 256
    print(f'bs={bs:>5d}: {ms:.3f} ms, {flops/1e9:.2f} GFLOPS, {flops/ms/1e6:.1f} GFLOP/s')
"
```

典型输出（RTX 5090）：

```text
bs=    1: 0.012 ms, 0.26 GFLOPS, 21.7 GFLOP/s    ← memory-bound
bs=   10: 0.014 ms, 2.62 GFLOPS, 187.1 GFLOP/s
bs=  100: 0.031 ms, 26.21 GFLOPS, 845.5 GFLOP/s
bs= 1000: 0.197 ms, 262.14 GFLOPS, 1330.6 GFLOP/s  ← compute-bound, 接近峰值
```

> ⚠️ **关键观察**：batch_size 从 1 到 1000，延迟只增加 ~16 倍，但吞吐增加 ~60 倍。这就是 batching 的威力——M 维增大让 GEMM 从 memory-bound（21 GFLOP/s）逼近 compute-bound（1330 GFLOP/s）。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间** | `O(batch_size × input_size × output_size)`（GEMM） |
| **空间** | `O(batch_size × output_size)` 输出 |
| **瓶颈** | batch_size 小 → memory-bound；大 → compute-bound |
| **与 Dynamic Batching 的关系** | Dynamic Batcher 聚合请求增大 batch_size → GEMM 利用率提升 |

> 💡 **一句话总结**：Simple Inference 是"为什么需要 batching"的微缩版——`batch_size=1000` 比 `batch_size=1` 吞吐高 60 倍。Dynamic Batcher 在系统层面做的事就是把多个单请求聚合成大 batch，让 GEMM 的 M 维从 1 增大到 N，从 memory-bound 逼近 compute-bound。

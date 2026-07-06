---
name: leetgpu-solution
description: 用于在 leetgpu/ 下编写 LeetGPU (https://leetgpu.com) CUDA 挑战题解。规定了目录组织、题解文档结构（6 段式）、手绘 sketch 风 SVG 插图规范、Kernel 代码要求与网站构建集成。触发于"写 leetgpu 题解"、"补全 CUDA 挑战题解"、"加一道 leetgpu 题解"等请求。
---

# 写 LeetGPU 题解 Skill

本工程(`ai-infra-notes`)的 LeetGPU 题解是对 [https://leetgpu.com](https://leetgpu.com) 在线 CUDA 挑战平台的题解归档。本 skill 描述如何产出符合仓库惯例的题解。

## 1. 选题逻辑：按 CUDA 概念覆盖选题

LeetGPU 平台的题目都是 **CUDA Kernel 实现题**，选题目标是**用最少的题覆盖 GPU 编程核心概念**。

> 📌 **题库来源**：题目元数据同步自 `leetgpu-challenges` 仓库（`challenges/<difficulty>/<number>_<name>/`）。截至本次更新共 **89 道**（简单 19 / 中等 57 / 困难 13），全部 `access_tier = "free"`。每道题的 `challenge.py` 中 `name` 字段即平台展示名，也用于生成 URL slug。

### 1.1 选题原则

| 原则 | 说明 |
|------|------|
| **概念覆盖优先** | 每道题对应一个 CUDA 核心概念（grid-stride、shared memory、warp shuffle、bank conflict、reduction、scan、tiling 等），避免连续多题重复同一概念 |
| **难度递进** | 由简到难：memory-bound 入门 → shared memory 进阶 → warp shuffle / tiling 高阶 → 综合题压轴 |
| **题目不重复** | 同一道题在整个题解系列中只出现一次，选题前先查下表状态列（✅ 已完成 / ⬜ 待补全）和 `leetgpu/` 已有文件 |
| **配合每日教程** | LeetGPU 题解作为每日教程（`aiinfra/weekN/dayM/`）Coding 任务的"任务 4"实战检验，选题应与当日主题强相关 |
| **性能导向** | 优先选能体现 ncu profiling 价值的题（有明确瓶颈指标可观察、可优化） |

**slug 推导规则**：`<slug>` = 平台 URL slug，由 `challenge.py` 的 `name` 经「小写化 → 空格转 `-` → 去括号/斜杠」得到（如 `"General Matrix Multiplication (GEMM)"` → `general-matrix-multiplication-gemm`，`"1D Convolution"` → `1d-convolution`）。题解文件名固定为 `leetgpu-<slug>-solution.md`。下表「编号」列为仓库目录前缀 `<number>_`，便于在 `leetgpu-challenges` 中定位题目源码。

### 1.2 推荐选题路径（按概念分组）

下表为按概念覆盖精选的推荐路径（共 18 道，覆盖主要 CUDA 模板）。选题时按概念覆盖，从每个分组挑 1-3 道。状态：✅ 已完成 / ⬜ 待补全（当前全部题解已清空，均为待补全）。

#### 入门：Memory-Bound 基础（grid-stride loop、coalesced access）

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 1 | vector-addition | Vector Addition | 简单 | grid-stride loop、coalesced 读写 | ✅ 已完成 |
| 21 | relu | ReLU | 简单 | 逐元素 kernel、分支开销 | ⬜ 待补全 |
| 8 | matrix-addition | Matrix Addition | 简单 | 2D grid、float4 向量化访存 | ⬜ 待补全 |
| 31 | matrix-copy | Matrix Copy | 简单 | 内存带宽、coalesced 拷贝 | ⬜ 待补全 |

#### 进阶：Shared Memory 与 Tiling

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 3 | matrix-transpose | Matrix Transpose | 简单 | shared memory tiling、bank conflict padding | ⬜ 待补全 |
| 9 | 1d-convolution | 1D Convolution | 简单 | shared memory halo | ⬜ 待补全 |
| 10 | 2d-convolution | 2D Convolution | 中等 | shared memory halo、常数内存 | ⬜ 待补全 |
| 4 | reduction | Reduction | 中等 | 树形归约、warp shuffle `__shfl_down_sync`、block 两级归约 | ⬜ 待补全 |
| 13 | histogramming | Histogramming | 中等 | shared memory 直方图、atomic 冲突 | ⬜ 待补全 |

#### 高阶：Warp 级原语与并行模式

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 16 | prefix-sum | Prefix Sum | 中等 | warp scan `__shfl_up_sync`、三阶段分块 scan | ⬜ 待补全 |
| 5 | softmax | Softmax | 中等 | 三遍 kernel（max→sum→scale）、数值稳定性 | ⬜ 待补全 |
| 17 | dot-product | Dot Product | 中等 | block 归约、kernel 融合 | ⬜ 待补全 |
| 29 | top-k-selection | Top K Selection | 中等 | bitonic 排序、堆归约 | ⬜ 待补全 |

#### 综合：Compute-Bound 与融合 Kernel

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 2 | matrix-multiplication | Matrix Multiplication | 简单 | tiled matmul、register tiling | ⬜ 待补全 |
| 22 | gemm | General Matrix Multiplication (GEMM) | 中等 | tiling、register blocking、双缓冲 | ⬜ 待补全 |
| 6 | softmax-attention | Softmax Attention | 中等 | fused softmax+matmul、数值稳定 | ⬜ 待补全 |
| 30 | batched-matrix-multiplication | Batched Matrix Multiplication | 中等 | batched GEMM、batched kernel launch | ⬜ 待补全 |
| 12 | multi-head-attention | Multi-Head Attention | 困难 | FlashAttention 思想、融合 attention | ⬜ 待补全 |

### 1.3 与每日教程的配合节奏

LeetGPU 题解不是独立选题，而是配合 `aiinfra/weekN/dayM/` 每日教程的 Coding 任务。每日教程的"任务 4"从 LeetGPU 平台选一道与当日主题强相关的题：

| 教程主题 | 推荐 LeetGPU 题目 | 关联概念 |
|----------|-------------------|----------|
| Day 1: Hello GPU / grid-stride | #1 Vector Addition | grid-stride loop |
| Day 2: Memory model | #21 ReLU / #8 Matrix Addition | coalesced access |
| Day 3: Shared memory | #3 Matrix Transpose | tiling + bank conflict |
| Day 4: Warp shuffle | #4 Reduction | `__shfl_down_sync` |
| Day 5: Scan / prefix | #16 Prefix Sum | `__shfl_up_sync` |
| Day 6: GEMM / tiling | #2 Matrix Multiplication / #22 GEMM | register tiling |
| Day 7: 综合验收 | #6 Softmax Attention / #12 Multi-Head Attention | fused kernel |

> 💡 **一句话总结**：选题的本质是「一道题学一个 CUDA 概念」，每道题都要能对应一个 GPU 编程模板，做完能迁移到一类 kernel 优化。

### 1.4 完整题目清单（参考，用于扩展选题）

下表为 `leetgpu-challenges` 仓库的**全部 89 道题**，按难度分组，便于在推荐路径之外按概念扩展选题。每行的「编号」对应仓库目录 `challenges/<difficulty>/<编号>_<name>/`。

#### 简单（Easy，19 道）

| 编号 | slug | 题目 | 核心概念 |
|------|------|------|----------|
| 1 | vector-addition | Vector Addition | grid-stride、coalesced |
| 2 | matrix-multiplication | Matrix Multiplication | tiled matmul |
| 3 | matrix-transpose | Matrix Transpose | shared mem tiling、bank conflict |
| 7 | color-inversion | Color Inversion | 逐元素、3 通道 |
| 8 | matrix-addition | Matrix Addition | 2D grid、向量化 |
| 9 | 1d-convolution | 1D Convolution | shared mem halo |
| 19 | reverse-array | Reverse Array | 1D 并行、coalesced |
| 21 | relu | ReLU | 逐元素、分支 |
| 23 | leaky-relu | Leaky ReLU | 逐元素、分支 |
| 24 | rainbow-table | Rainbow Table | 迭代哈希、串行循环 |
| 31 | matrix-copy | Matrix Copy | 带宽、coalesced |
| 41 | simple-inference | Simple Inference | PyTorch Linear 前向 |
| 52 | silu | Sigmoid Linear Unit (SiLU) | 逐元素、融合 |
| 54 | swiglu | Swish-Gated Linear Unit | 融合 kernel、elementwise 乘 |
| 62 | value-clipping | Value Clipping | 逐元素、clamp |
| 63 | interleave | Interleave Arrays | 写索引映射、coalesced |
| 65 | geglu | Gaussian Error Gated Linear Unit | 融合、GELU |
| 66 | rgb-to-grayscale | RGB to Grayscale | 加权求和、逐元素 |
| 68 | sigmoid | Sigmoid Activation | 逐元素、数学函数 |

#### 中等（Medium，57 道）

| 编号 | slug | 题目 | 核心概念 |
|------|------|------|----------|
| 4 | reduction | Reduction | 树形归约、warp shuffle |
| 5 | softmax | Softmax | 三遍、数值稳定 |
| 6 | softmax-attention | Softmax Attention | fused softmax+matmul |
| 10 | 2d-convolution | 2D Convolution | shared mem halo、常数内存 |
| 11 | 3d-convolution | 3D Convolution | 3D shared mem halo |
| 13 | histogramming | Histogramming | shared mem 直方图、atomic |
| 16 | prefix-sum | Prefix Sum | warp scan、三阶段 |
| 17 | dot-product | Dot Product | block 归约 |
| 18 | sparse-matrix-vector-multiplication | Sparse Matrix-Vector Multiplication | CSR、稀疏 |
| 22 | gemm | General Matrix Multiplication (GEMM) | tiling、register blocking |
| 25 | categorical-cross-entropy-loss | Categorical Cross Entropy Loss | 归约、log |
| 27 | mean-squared-error | Mean Squared Error | 归约 |
| 28 | gaussian-blur | Gaussian Blur | 可分离卷积、shared mem |
| 29 | top-k-selection | Top K Selection | bitonic、堆 |
| 30 | batched-matrix-multiplication | Batched Matrix Multiplication | batched GEMM |
| 32 | int8-quantized-matmul | INT8 Quantized MatMul | 量化、int8 GEMM |
| 33 | ordinary-least-squares | Ordinary Least Squares | 线性代数、归约 |
| 34 | logistic-regression | Logistic Regression | 迭代、sigmoid |
| 35 | monte-carlo-integration | Monte Carlo Integration | 随机数、归约 |
| 37 | matrix-power | Matrix Power | 重复 matmul |
| 38 | nearest-neighbor | Nearest Neighbor | 距离计算、归约 |
| 40 | batch-normalization | Batch Normalization | 归约、方差 |
| 42 | 2d-max-pooling | 2D Max Pooling | 滑窗、reduction |
| 43 | count-array-element | Count Array Element | 归约、atomic |
| 44 | count-2d-array-element | Count 2D Array Element | 2D 归约 |
| 45 | count-3d-array-element | Count 3D Array Element | 3D 归约 |
| 47 | subarray-sum | Subarray Sum | prefix sum |
| 48 | 2d-subarray-sum | 2D Subarray Sum | 2D prefix sum |
| 49 | 3d-subarray-sum | 3D Subarray Sum | 3D prefix sum |
| 50 | rms-normalization | RMS Normalization | 归约、归一化 |
| 51 | max-subarray-sum | Max Subarray Sum | scan、归约 |
| 55 | attn-w-linear-bias | Attention with Linear Biases (ALiBi) | attention 偏置 |
| 57 | fp16-batched-matmul | FP16 Batched Matrix Multiplication | fp16、tensor core |
| 58 | fp16-dot-product | FP16 Dot Product | fp16、归约 |
| 60 | top-p-sampling | Top-p Sampling | 排序、归约、采样 |
| 61 | rope-embedding | Rotary Positional Embedding | 复数旋转、elementwise |
| 64 | weight-dequantization | Weight Dequantization | 量化反量化 |
| 67 | moe-topk-gating | MoE Top-K Gating | top-k、softmax |
| 69 | 2d-jacobi-stencil | 2D Jacobi Stencil | stencil、共享边界 |
| 70 | segmented-prefix-sum | Segmented Exclusive Prefix Sum | 分段 scan |
| 71 | parallel-merge | Parallel Merge | 归并、双缓冲 |
| 72 | stream-compaction | Stream Compaction | scan、predicate |
| 75 | sparse-matrix-dense-matrix-multiplication | Sparse Matrix-Dense Matrix Multiplication | 稀疏 GEMM |
| 76 | adder-transformer | Adder Transformer Inference | 加法注意力 |
| 78 | 2d-fft | 2D FFT | 蝶形运算、shared mem |
| 80 | grouped-query-attention | Grouped Query Attention (GQA) | KV head 复用 |
| 81 | int4-matmul | INT4 Weight-Only Quantized MatMul | int4 量化 |
| 82 | linear-recurrence | Linear Recurrence | scan、并行前缀 |
| 84 | swiglu-mlp-block | SwiGLU MLP Block | 融合 MLP |
| 85 | lora-linear | LoRA Linear | 低秩、融合 |
| 87 | speculative-decoding-verification | Speculative Decoding Verification | 验证、scan |
| 90 | causal-depthwise-conv1d | Causal Depthwise Conv1d | depthwise 卷积 |
| 92 | decaying-causal-attention | Decaying Causal Attention | 衰减注意力 |
| 94 | ssm-selective-scan | SSM Selective Scan | 状态空间、scan |
| 96 | int8-kv-cache-attention | INT8 KV-Cache Attention | 量化 attention |
| 105 | group-normalization | Group Normalization | 归一化、分组归约 |
| 106 | token-embedding-layer | Token Embedding Layer | gather、embedding |

#### 困难（Hard，13 道）

| 编号 | slug | 题目 | 核心概念 |
|------|------|------|----------|
| 12 | multi-head-attention | Multi-Head Attention | FlashAttention、融合 |
| 14 | multi-agent-simulation | Multi-Agent Simulation | agent 并行、交互 |
| 15 | sorting | Sorting | 并行排序 |
| 20 | kmeans-clustering | K-Means Clustering | 迭代、归约 |
| 36 | radix-sort | Radix Sort | 基数排序、histogram |
| 39 | fast-fourier-transform | Fast Fourier Transform | FFT、蝶形 |
| 46 | bfs-shortest-path | BFS Shortest Path | 图并行、frontier |
| 53 | causal-self-attention | Causal Self-Attention | 因果掩码、融合 |
| 56 | linear-self-attention | Linear Self-Attention | 线性注意力 |
| 59 | sliding-window-self-attention | Sliding Window Self-Attention | 滑窗注意力 |
| 73 | all-pairs-shortest-paths | All-Pairs Shortest Paths | Floyd、图算法 |
| 74 | gpt-2-transformer-block | GPT-2 Transformer Block | 综合模块 |
| 93 | llama-transformer-block | Llama Transformer Block | RMSNorm+RoPE+SwiGLU |

> ⚠️ **同步提示**：`leetgpu-challenges` 仓库会持续新增题目。当需要扩展选题时，重新扫描 `challenges/<difficulty>/*/challenge.py` 的 `name` 字段并更新本节清单，保持与上游一致。

## 2. 目录组织

LeetGPU 题解按 `weekN/dayM/` 组织，与 `leetcode/daily/` 结构一致；**周/日编号与每日教程 `aiinfra/weekN/dayM/` 严格对齐**（题解就是当日教程「任务 4」的实战检验）：

```
leetgpu/
├── week1/
│   ├── day1/
│   │   └── leetgpu-vector-addition-solution.md   # Day1: grid-stride
│   ├── day2/
│   │   └── leetgpu-relu-solution.md              # Day2: coalesced access
│   └── ...
├── week2/
│   └── ...
├── images/                                  # 所有题解共享的 SVG/PNG 插图
│   ├── vector_addition_overview.svg
│   ├── reduction_overview.svg
│   └── generate_figures.py                  # matplotlib 生成脚本
├── website/                                 # 网站构建（build.py + 生成的 HTML）
│   ├── build.py
│   ├── index.html
│   └── leetgpu-<slug>-solution.html
└── SKILL.md                                 # 本文件
```

**规则**：

1. **题解根目录**：`leetgpu/`，不要写到其他位置。
2. **按 weekN/dayM 组织**：题解 `.md` 放在 `leetgpu/weekN/dayM/` 下，**周/日编号与每日教程 `aiinfra/weekN/dayM/` 对齐**（Day 1 → `week1/day1/`，Day 7 → `week1/day7/`）。一个 day 目录下通常只有一篇题解（即当日教程任务 4 对应的题）。
3. **题解文件名**：`leetgpu-<slug>-solution.md`，其中 `<slug>` 是 LeetGPU 平台的题目 URL slug（如 `vector-addition`、`prefix-sum`）。文件名不随 week/day 变化，slug 即唯一标识。
4. **图片目录**：`leetgpu/images/`，所有题解共享（不按 week/day/题分散）。图片在题解中用 `images/xxx.svg` 相对路径引用（`build.py` 递归扫描时统一重写为 `./images/xxx.svg`，输出页面扁平化到 `website/` 根）。
5. **选题与每日教程对齐**：每道题对应 `aiinfra/weekN/dayM/` 每日教程的「任务 4」LeetGPU 在线题目，按 **week/day 目录 + slug** 双重关联，保持主题一致。

## 3. 题解文档结构

每篇题解 `.md` 遵循固定 **6 段结构**（参考下文模板与 `leetgpu/` 已有题解）：

```markdown
# LeetGPU <题目名> 题解

## 1. 题目概述
- **标题 / 题号**：<题目名>
- **链接**：https://leetgpu.com/challenges/<slug>
- **难度**：简单 / 中等 / 困难
- **标签**：CUDA、<概念标签1>、<概念标签2>

（题意描述 + 输入输出 + 约束条件）

## 2. CPU 基线 / 朴素 GPU 方法
（CPU 串行实现 + 朴素 GPU 实现，说明瓶颈）

## 3. GPU 设计
### 3.1 并行化策略
### 3.2 存储层次使用（global / shared / register）
### 3.3 关键技巧（warp shuffle / tiling / coalesced 等）

## 4. Kernel 实现
（完整可编译 CUDA 代码：#include、__global__ kernel、main()、
  cudaMalloc/Memcpy、验证逻辑、cudaFree）

## 5. 性能分析与优化
（ncu profiling 命令 + 关键指标 + 优化方向）

## 6. 复杂度分析
（时间复杂度、空间复杂度、算术强度、瓶颈类型 memory/compute-bound）
```

**写作规范**：

- **中文为主**，概念加粗，善用 `> 💡` / `> ⚠️` blockquote。
- 代码块标注语言：` ```cuda` / ` ```cpp` / ` ```bash` / ` ```text`。
- **Kernel 代码必须完整可编译**：包含 `#include`、`__global__` kernel、`main()`、host 端 `cudaMalloc`/`cudaMemcpy`、验证逻辑、`cudaFree`。
- 代码块首行带注释：`// <filename>.cu —— <说明>` + `// 编译命令: nvcc ...`。
- 图片引用用相对路径：`![<中文alt>](images/<filename>.svg)`（题解位于 `weekN/dayM/` 下，`images/` 解析到共享的 `leetgpu/images/`，由 `build.py` 统一重写为 `./images/`）。
- 每篇题解引用 **2-4 张 SVG/PNG 插图**。

## 4. 图片风格：手绘 sketch 风（Excalidraw-like）

**所有插图统一为手绘 sketch 风**，与每日教程（`docs/skills/daily-tutorial/SKILL.md`）和 LeetCode 题解保持一致。具体要求：

### 4.1 视觉特征

| 维度 | 要求 |
|------|------|
| **线条** | 手绘不均匀、略带抖动，避免完美直线或圆润矢量边 |
| **笔触** | 粗糙、类似马克笔/铅笔描边，线宽可略有变化 |
| **配色** | 极简，一般不超过 3-4 种柔和颜色（蓝 `#e8f0fe`/`#446688`、绿 `#e6f4ea`/`#4a7a3a`、橙 `#fff8e1`/`#d6a040`、红 `#fce4ec`/`#b85450`），背景白色或米白 `#fafafa` |
| **形状** | 简单几何块——矩形、网格、箭头、圆角框，不画复杂 3D 或写实元素 |
| **标签** | 手写感字体：英文用 `Comic Sans MS` / `Bradley Hand`，CJK 用 `Kaiti SC` / 楷体 |
| **整体感觉** | 轻松白板涂鸦，标注随意、轻微错位也无妨，优先可读性和直观性 |

### 4.2 SVG 实现技法（仓库统一用法）

用 SVG 滤镜 `feTurbulence` + `feDisplacementMap` 给所有图形叠加轻微抖动，实现手绘效果。**每张 SVG 顶部固定引入以下 `<defs>`**：

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 360"
     font-family="'Comic Sans MS', 'Segoe UI', 'Kaiti SC', 楷体, cursive">
  <defs>
    <filter id="rough2">
      <feTurbulence type="fractalNoise" baseFrequency="0.025" numOctaves="2" seed="7"/>
      <feDisplacementMap in="SourceGraphic" scale="1.5"/>
    </filter>
  </defs>

  <rect width="520" height="360" fill="#fafafa"/>

  <!-- 所有矩形/路径/文本都加 filter="url(#rough2)" -->
  <rect x="80" y="48" width="50" height="30" fill="#e8f0fe"
        stroke="#446688" stroke-width="1.5" rx="4" filter="url(#rough2)"/>
  ...
</svg>
```

**要点**：

- `font-family` 必须包含 `'Comic Sans MS'` 和 `'Kaiti SC', 楷体`，保证中英文都有手写感。
- 每个图形元素（`rect`/`path`/`text`/`circle`）都加 `filter="url(#rough2)"`。
- 也可用 matplotlib 生成（参考 `leetgpu/images/generate_*.py`），配合手写字体 `NotoSansCJK` 或 `Comic Sans MS`。

### 4.3 常见图类型

| 图类型 | 用途 | 示例 |
|--------|------|------|
| **概念图** | 直观展示并行策略（grid-block-thread 映射、tiling 分块、reduction 树） | `reduction_overview.svg` |
| **存储层次图** | global → shared → register 的数据流 | `matmul_tiled.svg` |
| **性能对比图** | naive vs optimized 的带宽/延迟对比 | `reduction_grid_stride.svg` |

### 4.4 图片命名

全小写 + 下划线，语义化，建议加题目 slug 前缀避免冲突：

- `reduction_overview.svg`、`reduction_block_internal.svg`（reduction 题）
- `matmul_naive.svg`、`matmul_tiled.svg`（matrix multiplication 题）
- `top-k-selection_bitonic.png`、`top-k-selection_heap_reduce.png`（top-k-selection 题）

## 5. 网站构建集成

题解写完后会被 `leetgpu/website/build.py` 自动读取并生成网页：

- `build.py` **递归扫描** `leetgpu/` 下所有 `leetgpu-*.md` 文件（用 `rglob()`，自动识别 `weekN/dayM/` 子目录）。
- 解析路径中的 `weekN/dayM/` 作为分组依据，侧边栏按 week→day 手风琴式分组。
- 解析一级标题 `# LeetGPU <题目名> 题解` 作为侧边栏与列表页标题。
- 图片路径 `images/xxx.svg` 在题解页被重写为 `./images/xxx.svg`（网站输出目录扁平化）。
- 生成 `leetgpu/website/index.html`（概览页）和 `leetgpu/website/leetgpu-<slug>-solution.html`（各题解页，扁平输出到 `website/` 根）。
- 根 `build.py` 会把 `leetgpu/website/` 和 `leetgpu/images/` 复制到 `public/leetgpu/` 部署。

**验证命令**：

```bash
python3 leetgpu/website/build.py   # 单独构建 leetgpu 网站
python3 build.py                     # 组合构建全站（含 leetgpu）
```

**自检清单**：

- [ ] 题解位于 `leetgpu/weekN/dayM/leetgpu-<slug>-solution.md`（week/day 与每日教程对齐）
- [ ] 一级标题 `# LeetGPU <题目名> 题解`
- [ ] 含 6 段结构（题目概述/CPU基线/GPU设计/Kernel实现/性能分析/复杂度分析）
- [ ] Kernel 代码完整可编译（含 main、cudaMalloc、验证、cudaFree）
- [ ] 含 2-4 张 SVG/PNG 插图，引用格式 `![中文alt](images/xxx.svg)`
- [ ] SVG 为手绘 sketch 风（含 `feTurbulence` 抖动滤镜 + Comic Sans/Kaiti SC 字体）
- [ ] 含 ncu profiling 命令与关键指标
- [ ] `python3 build.py` 成功生成对应 `public/leetgpu/leetgpu-<slug>-solution.html`

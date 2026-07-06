---
name: leetgpu-solution
description: 用于在 leetgpu/ 下编写 LeetGPU (https://leetgpu.com) CUDA 挑战题解。规定了目录组织、题解文档结构（6 段式）、手绘 sketch 风 SVG 插图规范、Kernel 代码要求与网站构建集成。触发于"写 leetgpu 题解"、"补全 CUDA 挑战题解"、"加一道 leetgpu 题解"等请求。
---

# 写 LeetGPU 题解 Skill

本工程(`ai-infra-notes`)的 LeetGPU 题解是对 [https://leetgpu.com](https://leetgpu.com) 在线 CUDA 挑战平台的题解归档。本 skill 描述如何产出符合仓库惯例的题解。

## 1. 选题逻辑：按 CUDA 概念覆盖选题

LeetGPU 平台的题目都是 **CUDA Kernel 实现题**，选题目标是**用最少的题覆盖 GPU 编程核心概念**。

### 1.1 选题原则

| 原则 | 说明 |
|------|------|
| **概念覆盖优先** | 每道题对应一个 CUDA 核心概念（grid-stride、shared memory、warp shuffle、bank conflict、reduction、scan、tiling 等），避免连续多题重复同一概念 |
| **难度递进** | 由简到难：memory-bound 入门 → shared memory 进阶 → warp shuffle / tiling 高阶 → 综合题压轴 |
| **题目不重复** | 同一道题在整个题解系列中只出现一次，选题前先查下表状态列（✅ 已完成 / ⬜ 待补全）和 `leetgpu/` 已有文件 |
| **配合每日教程** | LeetGPU 题解作为每日教程（`aiinfra/weekN/dayM/`）Coding 任务的"任务 4"实战检验，选题应与当日主题强相关 |
| **性能导向** | 优先选能体现 ncu profiling 价值的题（有明确瓶颈指标可观察、可优化） |

### 1.2 LeetGPU 挑战题库（按概念分组）

下表为各概念分组的题目，选题时按概念覆盖。状态：✅ 已完成 / ⬜ 待补全（选题已定，题解待写）。

#### 入门：Memory-Bound 基础（grid-stride loop、coalesced access）

| slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|----------|------|
| vector-add | Vector Add | 简单 | grid-stride loop、coalesced 读写 | ⬜ 待补全 |
| relu | ReLU | 简单 | 逐元素 kernel、分支开销 | ⬜ 待补全 |
| matrix-addition | Matrix Addition | 简单 | 2D grid、float4 向量化访存 | ⬜ 待补全 |

#### 进阶：Shared Memory 与 Tiling

| slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|----------|------|
| matrix-transpose | Matrix Transpose | 中等 | shared memory tiling、bank conflict padding | ⬜ 待补全 |
| reduction | Reduction | 中等 | 树形归约、warp shuffle `__shfl_down_sync`、block 两级归约 | ⬜ 待补全 |
| histogram | Histogram | 中等 | shared memory 直方图、atomic 冲突 | ⬜ 待补全 |
| convolution | Convolution | 中等 | shared memory halo、常数内存 | ⬜ 待补全 |

#### 高阶：Warp 级原语与并行模式

| slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|----------|------|
| prefix-sum | Prefix Sum | 中等 | warp scan `__shfl_up_sync`、三阶段分块 scan | ⬜ 待补全 |
| softmax | Softmax | 中等 | 三遍 kernel（max→sum→scale）、数值稳定性 | ⬜ 待补全 |
| argmax | Argmax | 中等 | warp 归约带索引、tie-breaking | ⬜ 待补全 |

#### 综合：Compute-Bound 与融合 Kernel

| slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|----------|------|
| gemm | GEMM | 困难 | tiling、register blocking、双缓冲 | ⬜ 待补全 |
| matrix-multiplication | Matrix Multiplication | 困难 | tiled matmul、bank conflict 分析、register tiling | ⬜ 待补全 |
| attention | Attention | 困难 | fused softmax+matmul、FlashAttention 思想 | ⬜ 待补全 |

### 1.3 与每日教程的配合节奏

LeetGPU 题解不是独立选题，而是配合 `aiinfra/weekN/dayM/` 每日教程的 Coding 任务。每日教程的"任务 4"从 LeetGPU 平台选一道与当日主题强相关的题：

| 教程主题 | 推荐 LeetGPU 题目 | 关联概念 |
|----------|-------------------|----------|
| Day 1: Hello GPU / grid-stride | vector-add | grid-stride loop |
| Day 2: Memory model | relu / matrix-addition | coalesced access |
| Day 3: Shared memory | matrix-transpose | tiling + bank conflict |
| Day 4: Warp shuffle | reduction | `__shfl_down_sync` |
| Day 5: Scan / prefix | prefix-sum | `__shfl_up_sync` |
| Day 6: GEMM / tiling | matrix-multiplication | register tiling |
| Day 7: 综合验收 | attention | fused kernel |

> 💡 **一句话总结**：选题的本质是「一道题学一个 CUDA 概念」，每道题都要能对应一个 GPU 编程模板，做完能迁移到一类 kernel 优化。

## 2. 目录组织

所有 LeetGPU 题解**扁平存放在 `leetgpu/` 根目录下**（不按周/日建子目录），与 `leetcode/daily/` 的 `weekN/dayM/` 结构不同：

```
leetgpu/
├── leetgpu-vector-add-solution.md          # 题解主体（文件名 = leetgpu-<slug>-solution.md）
├── leetgpu-relu-solution.md
├── leetgpu-prefix-sum-solution.md
├── ...                                     # 其余题解均直接放在根目录
├── images/                                  # 所有题解共享的 SVG/PNG 插图
│   ├── reduction_overview.svg
│   ├── matmul_naive.svg
│   └── generate_figures.py                  # matplotlib 生成脚本
├── website/                                 # 网站构建（build.py + 生成的 HTML）
│   ├── build.py
│   ├── index.html
│   └── leetgpu-<slug>-solution.html
└── SKILL.md                                 # 本文件
```

**规则**：

1. **题解根目录**：`leetgpu/`，不要写到其他位置。
2. **扁平存放**：所有题解 `.md` 直接放在 `leetgpu/` 根目录下，**不**建 `weekN/dayM/` 子目录（与 `leetcode/daily/` 不同）。
3. **题解文件名**：`leetgpu-<slug>-solution.md`，其中 `<slug>` 是 LeetGPU 平台的题目 URL slug（如 `vector-add`、`prefix-sum`）。
4. **图片目录**：`leetgpu/images/`，所有题解共享（不按题分散）。图片在题解中用 `images/xxx.svg` 相对路径引用（从 `leetgpu/` 根出发，与 `build.py` 的路径重写逻辑一致）。
5. **选题与每日教程对齐**：每道题对应 `aiinfra/weekN/dayM/` 每日教程的「任务 4」LeetGPU 在线题目，按 **slug** 关联（不按目录），保持主题一致。

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
- 图片引用用相对路径：`![<中文alt>](images/<filename>.svg)`（从 `leetgpu/` 根出发，`images/` 解析到 `leetgpu/images/`，与 `build.py` 路径重写一致）。
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
- `argmax_warp_shuffle.png`、`argmax_tie_breaking.png`（argmax 题）

## 5. 网站构建集成

题解写完后会被 `leetgpu/website/build.py` 自动读取并生成网页：

- `build.py` **扁平扫描** `leetgpu/` 根目录下所有 `leetgpu-*.md` 文件（非递归，用 `iterdir()`）。
- 解析一级标题 `# LeetGPU <题目名> 题解` 作为侧边栏与列表页标题。
- 图片路径 `images/xxx.svg` 在题解页被重写为 `./images/xxx.svg`（网站输出目录扁平化）。
- 生成 `leetgpu/website/index.html`（概览页）和 `leetgpu/website/leetgpu-<slug>-solution.html`（各题解页）。
- 根 `build.py` 会把 `leetgpu/website/` 和 `leetgpu/images/` 复制到 `public/leetgpu/` 部署。

**验证命令**：

```bash
python3 leetgpu/website/build.py   # 单独构建 leetgpu 网站
python3 build.py                     # 组合构建全站（含 leetgpu）
```

**自检清单**：

- [ ] 题解位于 `leetgpu/leetgpu-<slug>-solution.md`（扁平存放，无 weekN/dayM 子目录）
- [ ] 一级标题 `# LeetGPU <题目名> 题解`
- [ ] 含 6 段结构（题目概述/CPU基线/GPU设计/Kernel实现/性能分析/复杂度分析）
- [ ] Kernel 代码完整可编译（含 main、cudaMalloc、验证、cudaFree）
- [ ] 含 2-4 张 SVG/PNG 插图，引用格式 `![中文alt](images/xxx.svg)`
- [ ] SVG 为手绘 sketch 风（含 `feTurbulence` 抖动滤镜 + Comic Sans/Kaiti SC 字体）
- [ ] 含 ncu profiling 命令与关键指标
- [ ] `python3 build.py` 成功生成对应 `public/leetgpu/leetgpu-<slug>-solution.html`

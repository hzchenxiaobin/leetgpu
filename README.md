# LeetGPU 题解

> 62 道 [LeetGPU](https://leetgpu.com/) CUDA 挑战题解 —— 每道含完整可编译 kernel + ncu profiling + 手绘 sketch 风 SVG 图解，按 CUDA 概念覆盖选题、按周/日归档。

📚 **在线网站**：https://hzchenxiaobin.github.io/leetgpu/

本仓库是 [ai-infra-notes](https://github.com/hzchenxiaobin/ai-infra-notes) 8 周 AI Infra 学习路线中 LeetGPU 题解的独立归档，按周/日与每日教程对齐，作为 Coding 任务的实战检验。

## 选题逻辑

LeetGPU 平台的题目都是 **CUDA Kernel 实现题**，选题目标是**用最少的题覆盖 GPU 编程核心概念**：

- **概念覆盖优先**：每道题对应一个 CUDA 核心概念（grid-stride、shared memory、warp shuffle、bank conflict、reduction、scan、tiling 等），避免连续多题重复
- **难度递进**：memory-bound 入门 → shared memory 进阶 → warp shuffle / tiling 高阶 → 综合题压轴
- **配合每日教程**：题解作为每日教程 Coding 任务的实战检验，选题与当日主题强相关
- **性能导向**：优先选能体现 ncu profiling 价值的题（有明确瓶颈指标可观察、可优化）

详细写作规范见 [SKILL.md](SKILL.md)。

## 题解列表

共 **62 道**（简单 / 中等 / 困难），覆盖 Vector Addition、GEMM、Softmax、Attention、Prefix Sum、PagedAttention、GQA、Speculative Decoding、GPT-2 Block、FlashAttention 等。

### Week 1 · GPU 执行本质 + Profiling

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [Vector Addition](week1/day1/leetgpu-vector-addition-solution.md) | 简单 | grid-stride loop、coalesced access、memory-bound |
| 2 | [ReLU](week1/day2/leetgpu-relu-solution.md) | 简单 | elementwise、warp divergence、branchless |
| 3 | [Matrix Addition](week1/day3/leetgpu-matrix-addition-solution.md) | 简单 | float4 向量化、Roofline |
| 4 | [Matrix Transpose](week1/day4/leetgpu-matrix-transpose-solution.md) | 中等 | shared memory tiling、bank conflict padding |
| 5 | [Subarray Sum](week1/day5/leetgpu-subarray-sum-solution.md) | 中等 | reduction、warp shuffle、block 归约 |
| 6 | [Matrix Multiplication](week1/day6/leetgpu-matrix-multiplication-solution.md) | 简单 | GEMM、shared memory tiling、register tiling |
| 7 | [Leaky ReLU](week1/day7/leetgpu-leaky-relu-solution.md) | 简单 | elementwise、branchless、activation |

### Week 2 · GEMM & 算子优化

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [Prefix Sum](week2/day1/leetgpu-prefix-sum-solution.md) | 中等 | scan、warp shuffle `__shfl_up_sync`、三阶段分块 |
| 2 | [GEMM](week2/day2/leetgpu-gemm-solution.md) | 中等 | FP16、WMMA、Tensor Core、shared memory tiling |
| 3 | [2D Convolution](week2/day3/leetgpu-2d-convolution-solution.md) | 中等 | shared memory halo、常量内存 |
| 4 | [Softmax](week2/day4/leetgpu-softmax-solution.md) | 中等 | safe softmax、三遍扫描、数值稳定性 |
| 5 | [Softmax Attention](week2/day5/leetgpu-softmax-attention-solution.md) | 中等 | fused softmax+matmul、online softmax |
| 6 | [Histogramming](week2/day6/leetgpu-histogramming-solution.md) | 中等 | shared memory 直方图、`atomicAdd`、privatization |
| 7 | [Mean Squared Error](week2/day7/leetgpu-mean-squared-error-solution.md) | 中等 | reduction、kernel 融合、损失函数 |

### Week 3 · Transformer 执行本质

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [Causal Depthwise Conv1d](week3/day1/leetgpu-causal-depthwise-conv1d-solution.md) | 中等 | causal、depthwise、边界处理 |
| 2 | [Group Normalization](week3/day2/leetgpu-group-normalization-solution.md) | 中等 | normalization、reduction、GroupNorm |
| 3 | [Argmax](week3/day3/leetgpu-argmax-solution.md) | 中等 | 归约、argmax、`__shfl_down_sync` |
| 4 | [Attention](week3/day4/leetgpu-attention-solution.md) | 困难 | online softmax、FlashAttention、分块计算 |
| 5 | [2D Max Pooling](week3/day5/leetgpu-2d-max-pooling-solution.md) | 中等 | pooling、滑窗 reduction、padding 边界 |
| 6 | [RMS Normalization](week3/day6/leetgpu-rms-normalization-solution.md) | 中等 | RMSNorm、warp shuffle、Llama |
| 7 | [Attention with Linear Biases (ALiBi)](week3/day7/leetgpu-attn-w-linear-bias-solution.md) | 中等 | ALiBi、positional bias、online softmax |

### Week 4 · FlashAttention 深挖

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [Decaying Causal Attention](week4/day1/leetgpu-decaying-causal-attention-solution.md) | 中等 | causal mask、exponential decay、增量计算 |
| 2 | [Adder Transformer Inference](week4/day2/leetgpu-adder-transformer-solution.md) | 中等 | 多 kernel 流水线、autoregressive 推理、RoPE |
| 3 | [Dot Product](week4/day3/leetgpu-dot-product-solution.md) | 中等 | reduction、warp shuffle、kernel 融合 |
| 4 | [Batched Matrix Multiplication](week4/day4/leetgpu-batched-matrix-multiplication-solution.md) | 中等 | batched GEMM、tiled matmul、register blocking |
| 5 | [Matrix Copy](week4/day5/leetgpu-matrix-copy-solution.md) | 简单 | 内存带宽、coalesced access、float4 向量化 |
| 6 | [Multi-Head Attention](week4/day6/leetgpu-multi-head-attention-solution.md) | 困难 | MHA、FlashAttention、融合 attention |
| 7 | [GPT-2 Transformer Block](week4/day7/leetgpu-gpt-2-transformer-block-solution.md) | 困难 | Transformer、FlashAttention、LayerNorm、GEMM 端到端 |

### Week 5 · 推理系统与 KV Cache

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [INT8 KV-Cache Attention](week5/day1/leetgpu-int8-kv-cache-attention-solution.md) | 中等 | decode-phase、KV Cache、INT8 量化、per-token scale |
| 2 | [Grouped Query Attention (GQA)](week5/day2/leetgpu-grouped-query-attention-solution.md) | 中等 | GQA、KV head 共享、LLM 推理 |
| 3 | [Speculative Decoding Verification](week5/day3/leetgpu-speculative-decoding-verification-solution.md) | 中等 | 投机解码、accept/reject 采样、CDF 查找 |
| 4 | [Causal Self-Attention](week5/day4/leetgpu-causal-self-attention-solution.md) | 困难 | causal mask、online softmax、LLM prefill、PagedAttention 对偶 |
| 5 | [Token Embedding Layer](week5/day5/leetgpu-token-embedding-layer-solution.md) | 中等 | embedding、gather、LayerNorm、融合 kernel |
| 6 | [Weight Dequantization](week5/day6/leetgpu-weight-dequantization-solution.md) | 中等 | element-wise、分块 scale、量化推理 |
| 7 | [Simple Inference](week5/day7/leetgpu-simple-inference-solution.md) | 简单 | PyTorch、Linear、batch size、GEMM |

### Week 6 · Batching & 调度

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [MoE Top-K Gating](week6/day1/leetgpu-moe-topk-gating-solution.md) | 中等 | top-k 选择、并行归约、softmax、MoE 路由 |
| 2 | [Max Subarray Sum](week6/day2/leetgpu-max-subarray-sum-solution.md) | 中等 | 滑动窗口、prefix sum、reduction |
| 3 | [Stream Compaction](week6/day3/leetgpu-stream-compaction-solution.md) | 中等 | scan、predicate、stream compaction |
| 4 | [Segmented Prefix Sum](week6/day4/leetgpu-segmented-prefix-sum-solution.md) | 中等 | segmented scan、warp shuffle |
| 5 | [INT8 Quantized MatMul](week6/day5/leetgpu-int8-quantized-matmul-solution.md) | 中等 | INT8 量化、tiled GEMM、INT32 累加、requantize |
| 6 | [Top K Selection](week6/day6/leetgpu-top-k-selection-solution.md) | 中等 | bitonic sort、堆归约、selection |
| 7 | [FP16 Dot Product](week6/day7/leetgpu-fp16-dot-product-solution.md) | 中等 | half 精度、warp shuffle、FP32 累加 |

### Week 7 · 系统整合

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [Color Inversion](week7/day1/leetgpu-color-inversion-solution.md) | 简单 | elementwise、image processing、`uchar4` 向量化 |
| 2 | [Reverse Array](week7/day2/leetgpu-reverse-array-solution.md) | 简单 | in-place swap、1D 并行、coalesced access |
| 2 | [Vector Reversal](week7/day2/leetgpu-vector-reversal-solution.md) | 简单 | 索引映射、coalesced access |
| 3 | [Scalar Multiply](week7/day3/leetgpu-scalar-multiply-solution.md) | 简单 | element-wise、attention scaling |
| 4 | [Interleave Arrays](week7/day4/leetgpu-interleave-solution.md) | 简单 | grid-stride loop、索引映射 |
| 5 | [Element Reversal](week7/day5/leetgpu-element-reversal-solution.md) | 简单 | element-wise、结果验证 |
| 6 | [Reduction](week7/day6/leetgpu-reduction-solution.md) | 中等 | warp shuffle、`__shfl_down_sync` |
| 7 | [Sigmoid](week7/day7/leetgpu-sigmoid-solution.md) | 简单 | elementwise、fast math `__expf`、activation |

### Week 8 · 项目打磨 + 面试准备

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [SiLU](week8/day1/leetgpu-silu-solution.md) | 简单 | elementwise、grid-stride、`__expf` 快速数学 |
| 2 | [Rotary Positional Embedding](week8/day2/leetgpu-rope-embedding-solution.md) | 中等 | elementwise、rotate_half、位置编码 |
| 3 | [SwiGLU MLP Block](week8/day3/leetgpu-swiglu-mlp-block-solution.md) | 中等 | SwiGLU、MLP、GEMM、kernel fusion、LLaMA |
| 3 | [SwiGLU](week8/day3/leetgpu-swiglu-solution.md) | 简单 | elementwise、kernel fusion、SiLU |
| 4 | [Sliding Window Self-Attention](week8/day4/leetgpu-sliding-window-self-attention-solution.md) | 中等/困难 | sliding window、kernel fusion |
| 5 | [LoRA Linear](week8/day5/leetgpu-lora-linear-solution.md) | 中等 | Low-Rank Adaptation、参数高效微调 |
| 6 | [Batch Normalization](week8/day6/leetgpu-batch-normalization-solution.md) | 中等 | normalization、reduction、数值稳定性 |
| 7 | [1D Convolution](week8/day7/leetgpu-1d-convolution-solution.md) | 简单 | convolution、shared memory、halo |

### Week 9 · 综合扩展

| Day | 题目 | 难度 | 核心概念 |
|-----|------|------|----------|
| 1 | [Count Array Element](week9/day1/leetgpu-count-array-element-solution.md) | 中等 | reduction、`atomicAdd`、predicate、warp shuffle |
| 2 | [Sparse Matrix-Vector Multiplication](week9/day2/leetgpu-sparse-matrix-vector-multiplication-solution.md) | 中等 | CSR、SpMV、warp shuffle、间接访存（gather） |
| 3 | [Nearest Neighbor](week9/day3/leetgpu-nearest-neighbor-solution.md) | 中等 | pairwise distance、shared memory tiling、argmin 归约 |
| 4 | [2D Jacobi Stencil](week9/day4/leetgpu-2d-jacobi-stencil-solution.md) | 中等 | stencil 计算、shared memory halo、Jacobi 迭代 |

## 题解结构

每篇题解 `.md` 遵循固定 **6 段结构**：

```
# LeetGPU <题目名> 题解
## 1. 题目概述      ← 题意 / 输入输出 / 约束
## 2. CPU 基线 / 朴素 GPU 方法  ← 串行实现 + 朴素 kernel，说明瓶颈
## 3. GPU 设计       ← 并行化策略 / 存储层次 / 关键技巧
## 4. Kernel 实现    ← 完整可编译 CUDA 代码（含 nvcc 命令 + 验证逻辑）
## 5. 性能分析与优化  ← ncu profiling 命令 + 关键指标 + 优化方向
## 6. 复杂度分析      ← 时间/空间复杂度、算术强度、瓶颈类型
```

- **Kernel 代码必须完整可编译**：含 `#include`、`__global__` kernel、`main()`、`cudaMalloc`/`cudaMemcpy`、验证逻辑、`cudaFree`
- **数学公式**：行内 `$...$`、块级 `$$...$$`，由 KaTeX 渲染
- **插图**：统一手绘 sketch 风 SVG（Excalidraw-like，`feTurbulence` 抖动滤镜），存放于 `images/`，每篇引用 2-4 张

## 仓库结构

```
leetgpu/
├── week1~week9/        # 题解，按周/日归档
│   └── dayM/
│       └── leetgpu-<slug>-solution.md
├── images/             # 手绘 sketch 风 SVG 插图（118 张）
├── SKILL.md            # 写 LeetGPU 题解的 Skill 规范
├── build.py            # 网站构建入口
├── build/              # 构建系统（common.py + leetgpu.py）
├── static/             # 网站静态资源（css + js）
└── .github/workflows/  # GitHub Pages 自动部署
```

## 在线网站

每次推送到 `main` 分支自动构建并部署到 GitHub Pages：

> https://hzchenxiaobin.github.io/leetgpu/

站点特性：侧边栏周手风琴导航、随机选题按钮、KaTeX 数学渲染、Prism 代码高亮（含 CUDA 语法扩展）。

## 本地预览

```bash
python3 build.py      # 生成 public/
```

然后在浏览器打开 `public/index.html`，或用任意静态服务器托管 `public/` 目录。

## 关联仓库

| 仓库 | 说明 |
|------|------|
| [ai-infra-notes](https://github.com/hzchenxiaobin/ai-infra-notes) | 8 周 AI Infra 学习路线主仓库（每日教程 + Profiling + Mini 引擎） |
| [LeetCode 题解](https://hzchenxiaobin.github.io/leetcode/) | 配套 LeetCode 面试题解，按周/日与教程对齐 |
| [LeetGPU](https://leetgpu.com/) | 在线 CUDA 挑战平台（题库来源） |

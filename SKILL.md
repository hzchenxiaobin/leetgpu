---
name: leetgpu-solution
description: 用于在 leetgpu/ 下编写 LeetGPU (https://leetgpu.com) CUDA 挑战题解。规定了目录组织、题解文档结构（6 段式）、手绘 sketch 风 SVG 插图规范、Kernel 代码要求与网站构建集成。触发于"写 leetgpu 题解"、"补全 CUDA 挑战题解"、"加一道 leetgpu 题解"等请求。
---

# 写 LeetGPU 题解 Skill

本工程(`ai-infra-notes`)的 LeetGPU 题解是对 [https://leetgpu.com](https://leetgpu.com) 在线 CUDA 挑战平台的题解归档。本 skill 描述如何产出符合仓库惯例的题解。

## 1. 选题逻辑：按 CUDA 概念覆盖选题

LeetGPU 平台的题目都是 **CUDA Kernel 实现题**，选题目标是**用最少的题覆盖 GPU 编程核心概念**。

> 📌 **题库来源**：题目元数据同步自 `leetgpu-challenges` 仓库（`challenges/<difficulty>/<number>_<name>/`）。截至本次更新共 **89 道**（简单 19 / 中等 57 / 困难 13），全部 `access_tier = "free"`。每道题的 `challenge.py` 中 `name` 字段即平台展示名，也用于生成 URL slug。
>
> 📁 **本地参考仓库**：`/mnt/workspace/code/github/leetgpu-challenges`

### 1.1 选题原则

| 原则 | 说明 |
|------|------|
| **概念覆盖优先** | 每道题对应一个 CUDA 核心概念（grid-stride、shared memory、warp shuffle、bank conflict、reduction、scan、tiling 等），避免连续多题重复同一概念 |
| **难度递进** | 由简到难：memory-bound 入门 → shared memory 进阶 → warp shuffle / tiling 高阶 → 综合题压轴 |
| **题目不重复** | 同一道题在整个题解系列中只出现一次，选题前先查下表状态列（✅ 已完成 / ⬜ 待补全）和 `leetgpu/` 已有文件 |
| **配合每日教程** | LeetGPU 题解作为每日教程（`aiinfra/daily/weekN/dayM/`）Coding 任务的"任务 4"实战检验，选题应与当日主题强相关 |
| **性能导向** | 优先选能体现 ncu profiling 价值的题（有明确瓶颈指标可观察、可优化） |

**slug 推导规则**：`<slug>` = 平台 URL slug，由 `challenge.py` 的 `name` 经「小写化 → 空格转 `-` → 去括号/斜杠」得到（如 `"General Matrix Multiplication (GEMM)"` → `general-matrix-multiplication-gemm`，`"1D Convolution"` → `1d-convolution`）。题解文件名固定为 `leetgpu-<slug>-solution.md`。下表「编号」列为仓库目录前缀 `<number>_`，便于在 `leetgpu-challenges` 中定位题目源码。

### 1.2 推荐选题路径（按概念分组）

下表为按概念覆盖精选的推荐路径（共 18 道，覆盖主要 CUDA 模板）。选题时按概念覆盖，从每个分组挑 1-3 道。状态：✅ 已完成 / ⬜ 待补全（当前全部题解已清空，均为待补全）。

#### 入门：Memory-Bound 基础（grid-stride loop、coalesced access）

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 1 | vector-addition | Vector Addition | 简单 | grid-stride loop、coalesced 读写 | ✅ 已完成 |
| 21 | relu | ReLU | 简单 | 逐元素 kernel、分支开销 | ✅ 已完成 |
| 8 | matrix-addition | Matrix Addition | 简单 | 2D grid、float4 向量化访存 | ✅ 已完成 |
| 31 | matrix-copy | Matrix Copy | 简单 | 内存带宽、coalesced 拷贝 | ✅ 已完成 |

#### 进阶：Shared Memory 与 Tiling

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 3 | matrix-transpose | Matrix Transpose | 简单 | shared memory tiling、bank conflict padding | ✅ 已完成 |
| 9 | 1d-convolution | 1D Convolution | 简单 | shared memory halo | ✅ 已完成 |
| 10 | 2d-convolution | 2D Convolution | 中等 | shared memory halo、常数内存 | ✅ 已完成 |
| 4 | reduction | Reduction | 中等 | 树形归约、warp shuffle `__shfl_down_sync`、block 两级归约 | ✅ 已完成 |
| 13 | histogramming | Histogramming | 中等 | shared memory 直方图、atomic 冲突 | ✅ 已完成 |

#### 高阶：Warp 级原语与并行模式

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 16 | prefix-sum | Prefix Sum | 中等 | warp scan `__shfl_up_sync`、三阶段分块 scan | ✅ 已完成 |
| 5 | softmax | Softmax | 中等 | 三遍 kernel（max→sum→scale）、数值稳定性 | ✅ 已完成 |
| 17 | dot-product | Dot Product | 中等 | block 归约、kernel 融合 | ✅ 已完成 |
| 29 | top-k-selection | Top K Selection | 中等 | bitonic 排序、堆归约 | ✅ 已完成 |

#### 综合：Compute-Bound 与融合 Kernel

| 编号 | slug | 题目 | 难度 | 核心概念 | 状态 |
|------|------|------|------|----------|------|
| 2 | matrix-multiplication | Matrix Multiplication | 简单 | tiled matmul、register tiling | ✅ 已完成 |
| 22 | gemm | General Matrix Multiplication (GEMM) | 中等 | tiling、register blocking、双缓冲 | ✅ 已完成 |
| 6 | softmax-attention | Softmax Attention | 中等 | fused softmax+matmul、数值稳定 | ✅ 已完成 |
| 30 | batched-matrix-multiplication | Batched Matrix Multiplication | 中等 | batched GEMM、batched kernel launch | ✅ 已完成 |
| 12 | multi-head-attention | Multi-Head Attention | 困难 | FlashAttention 思想、融合 attention | ✅ 已完成 |

### 1.3 与每日教程的配合节奏

LeetGPU 题解不是独立选题，而是配合 `aiinfra/daily/weekN/dayM/` 每日教程的 Coding 任务。每日教程的"任务 4"从 LeetGPU 平台选一道与当日主题强相关的题：

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
| 19 | reverse-array | Reverse Array | 1D 并行、in-place swap、coalesced（✅ 已完成） |
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
| 70 | segmented-prefix-sum | Segmented Prefix Sum | 分段 scan、段边界处理 |
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

### 1.5 LeetGPU 题型与知识地图

下面按**GPU / ML 知识领域**重新组织全部 89 道题，便于按概念系统学习和选题。每个领域给出「核心知识点」和「对应题目」。

#### A. 基础并行模式（Element-wise / Memory-bound）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| grid-stride loop | 每个线程处理多个元素，保证任意长度张量覆盖 | #1 Vector Addition |
| coalesced global memory | 线程连续访问连续地址，最大化内存带宽 | #1 Vector Addition, #31 Matrix Copy, #63 Interleave Arrays |
| 2D grid 映射 | row/col 索引映射到矩阵元素 | #8 Matrix Addition, #3 Matrix Transpose |
| 逐元素激活函数 | ReLU / Leaky ReLU / SiLU / SwiGLU / GELU / Sigmoid | #21 ReLU, #23 Leaky ReLU, #52 SiLU, #54 SwiGLU, #65 GeGLU, #68 Sigmoid |
| 颜色/图像变换 | 多通道加权、反色、灰度 | #7 Color Inversion, #66 RGB to Grayscale |
| 向量/数组重排 | reverse、interleave、value clipping | #19 Reverse Array, #63 Interleave Arrays, #62 Value Clipping |
| 简单 PyTorch 算子封装 | 调用 `torch.nn.functional` 实现 Linear | #41 Simple Inference |

#### B. 卷积与池化（Convolution & Pooling）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| 1D shared-memory halo | 一维卷积边界填充、tile 加载 | #9 1D Convolution |
| 2D shared-memory halo | 二维卷积、常数内存存储 kernel | #10 2D Convolution |
| 3D shared-memory halo | 三维卷积、体数据 | #11 3D Convolution |
| 可分离卷积 / 高斯模糊 | 行/列分离卷积核 | #28 Gaussian Blur |
| 滑窗最大值 | 2D max pooling、reduction in window | #42 2D Max Pooling |
| Causal / Depthwise Conv1d | 因果卷积、depthwise 分组 | #90 Causal Depthwise Conv1d |

#### C. 归约与扫描（Reduction & Scan）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| 树形归约 | block 内两两相加、warp shuffle `__shfl_down_sync` | #4 Reduction |
| warp 级 scan | `__shfl_up_sync`、inclusive/exclusive prefix sum | #16 Prefix Sum |
| 分段扫描 | segment flag、段边界处理 | #70 Segmented Exclusive Prefix Sum |
| 子数组和 / 最大子数组 | prefix sum、Kadane-like scan | #47 Subarray Sum, #48 2D Subarray Sum, #49 3D Subarray Sum, #51 Max Subarray Sum |
| stream compaction | predicate + scan 得到输出位置 | #72 Stream Compaction |
| dot product | 元素乘 + 全局归约 | #17 Dot Product |
| 计数 / 直方图 | atomic、shared memory histogram | #13 Histogramming, #43 Count Array Element, #44 Count 2D, #45 Count 3D |

#### D. 矩阵乘法与 GEMM（GEMM & Matmul）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| naive matmul | 全局内存直接累加 | #2 Matrix Multiplication |
| shared memory tiling | A/B tile 加载到 shared mem、register blocking | #22 General Matrix Multiplication (GEMM) |
| batched GEMM | 多组小矩阵并行 | #30 Batched Matrix Multiplication |
| FP16 / Tensor Core | half precision、wmma / mma.sync | #57 FP16 Batched Matrix Multiplication, #58 FP16 Dot Product |
| INT8 量化 GEMM | 量化矩阵乘、scale | #32 INT8 Quantized MatMul |
| INT4 weight-only | 4-bit 权重量化、反量化 | #81 INT4 Weight-Only Quantized MatMul |
| 稀疏矩阵乘 | CSR / sparse-dense | #18 Sparse Matrix-Vector Multiplication, #75 Sparse Matrix-Dense Matrix Multiplication |

#### E. 注意力机制（Attention）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| softmax 数值稳定 | online softmax、三趟 kernel | #5 Softmax |
| fused softmax+matmul | attention score / softmax / weighted sum | #6 Softmax Attention |
| multi-head attention | Q/K/V split、head 并行 | #12 Multi-Head Attention |
| causal mask | 下三角掩码、避免未来信息 | #53 Causal Self-Attention |
| linear attention | kernel trick 降低复杂度 | #56 Linear Self-Attention |
| sliding window attention | 局部注意力窗口 | #59 Sliding Window Self-Attention |
| ALiBi | 线性偏置注意力 | #55 Attention with Linear Biases |
| grouped query attention | KV head 共享 | #80 Grouped Query Attention |
| decaying causal attention | 衰减因子 | #92 Decaying Causal Attention |
| int8 KV-cache attention | 量化 KV cache | #96 INT8 KV-Cache Attention |

#### F. 归一化与嵌入（Normalization & Embedding）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| batch normalization | mean / variance、归一化 | #40 Batch Normalization |
| RMS normalization | root mean square norm | #50 RMS Normalization |
| group normalization | 分组 mean/var | #105 Group Normalization |
| RoPE | 旋转位置编码 | #61 Rotary Positional Embedding |
| token embedding | gather / lookup table | #106 Token Embedding Layer |

#### G. Transformer 组件与推理优化（Transformer Blocks & Inference）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| MLP block / SwiGLU | gate + up + down 投影融合 | #84 SwiGLU MLP Block |
| MoE top-k gating | router + top-k expert | #67 MoE Top-K Gating |
| LoRA | 低秩适配、融合低秩矩阵 | #85 LoRA Linear |
| speculative decoding | draft token 验证 | #87 Speculative Decoding Verification |
| GPT-2 block | LN + attn + MLP 综合 | #74 GPT-2 Transformer Block |
| Llama block | RMSNorm + RoPE + SwiGLU + GQA | #93 Llama Transformer Block |
| Adder Transformer | 加法注意力替代 | #76 Adder Transformer Inference |

#### H. 量化与低精度（Quantization）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| INT8 量化 MatMul | symmetric/asymmetric、scale | #32 INT8 Quantized MatMul |
| INT4 weight-only | 4-bit 权重、packing | #81 INT4 Weight-Only Quantized MatMul |
| weight dequantization | 反量化到 fp16/fp32 | #64 Weight Dequantization |
| INT8 KV cache | KV cache 量化 attention | #96 INT8 KV-Cache Attention |
| FP16 运算 | half precision、tensor core | #57 FP16 Batched MatMul, #58 FP16 Dot Product |

#### I. 采样、排序与搜索（Sampling, Sorting & Search）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| top-k selection | bitonic sort、heap reduce | #29 Top K Selection |
| top-p sampling | 排序 + 累积概率 + 采样 | #60 Top-p Sampling |
| parallel sort | 比较排序网络 | #15 Sorting |
| radix sort | 按位 histogram + scan | #36 Radix Sort |
| parallel merge | 二路归并、rank | #71 Parallel Merge |

#### J. 高级算法与数学（Advanced Algorithms & Math）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| FFT / 2D FFT | 蝶形运算、位逆序 | #39 Fast Fourier Transform, #78 2D FFT |
| K-Means | 迭代聚类、归约 | #20 K-Means Clustering |
| 图遍历 | BFS、frontier | #46 BFS Shortest Path, #73 All-Pairs Shortest Paths |
| Monte Carlo | 随机采样、归约 | #35 Monte Carlo Integration |
| 线性回归 / OLS | 正规方程、矩阵运算 | #33 Ordinary Least Squares |
| Logistic Regression | sigmoid + gradient | #34 Logistic Regression |
| nearest neighbor | 距离矩阵、归约 | #38 Nearest Neighbor |
| matrix power | 重复 square | #37 Matrix Power |
| 线性递推 / SSM | scan、selective scan | #82 Linear Recurrence, #94 SSM Selective Scan |
| stencil | Jacobi 迭代、边界交换 | #69 2D Jacobi Stencil |

#### K. 损失函数与基础 ML（Losses & Basic ML）

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| cross entropy loss | 归约 + log softmax | #25 Categorical Cross Entropy Loss |
| MSE | 平方差、归约 | #27 Mean Squared Error |

#### L. 其他综合与模拟

| 知识点 | 说明 | 对应题目 |
|--------|------|----------|
| rainbow table | 迭代哈希、串行循环 | #24 Rainbow Table |
| multi-agent simulation | agent 并行交互 | #14 Multi-Agent Simulation |

---

**学习路径建议**：

1. **入门**：A → B（1D conv）→ D（naive matmul）
2. **进阶**：C（reduction、prefix sum）→ D（GEMM tiling）→ E（softmax → attention）
3. **高阶**：E（causal / linear / sliding window）→ G（transformer block）→ H（quantization）→ J（FFT / graph / SSM）

### 1.6 同类练习题推荐映射

每篇题解末尾的「## 同类练习题」章节从下表取材，**为每道题推荐 4 道考查相同 CUDA 概念的练习题**，附直达链接与一句话关联说明。本表是推荐映射的**唯一权威来源**——新增/修改题解时，直接从下表对应行复制 4 条推荐到题解的「同类练习题」表格中，避免各题解推荐不一致。

> 📌 **使用规则**：
> 1. 题解的「同类练习题」章节内容**必须与下表完全一致**（题目编号、关联说明、主线总结），不得自行增删。
> 2. 同 slug 的重复题解（stub 或副本）也用**同一组推荐**，保持同步。
> 3. 下表的「领域」分组与 §1.5 知识地图（A–L）对齐，便于按概念查找。
> 4. 题目编号对应 §1.4 完整题目清单；链接格式固定为 `https://leetgpu.com/challenges/<slug>`。

#### A. 基础并行模式（Element-wise / Memory-bound）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `vector-addition` | #21 ReLU（同为逐元素 kernel，多了分支判断，练习 coalesced 读写）· #31 Matrix Copy（纯拷贝，专注带宽优化与 float4 向量化）· #68 Sigmoid（数学函数逐元素，练习 fused kernel 思想）· #63 Interleave（写索引映射练习，coalesced 写回） | memory-bound 逐元素 kernel，练习 grid-stride loop 与合并访存 |
| `relu` | #23 Leaky ReLU（带负斜率分支，对比无分支优化）· #52 SiLU（融合 sigmoid+mul，练习 fused kernel）· #68 Sigmoid（纯数学函数逐元素，练习 exp 实现）· #65 GeGLU（GELU 激活，更复杂的逐元素融合） | 逐元素激活函数 family，练习分支/无分支 kernel 与合并访存 |
| `matrix-addition` | #31 Matrix Copy（纯矩阵拷贝，专注 coalesced 带宽优化）· #1 Vector Addition（1D 向量加法，grid-stride 基础）· #8 Matrix Addition（同题，可对比不同 tile 写法）· #62 Value Clipping（逐元素 clamp，练习 2D 索引） | 2D grid 映射 + 合并访存，练习矩阵级 elementwise kernel |
| `matrix-copy` | #1 Vector Addition（grid-stride + coalesced 基础）· #8 Matrix Addition（2D coalesced）· #3 Matrix Transpose（非连续访存对比）· #63 Interleave（写索引映射练习） | 纯带宽优化 + coalesced 拷贝，练习 memory-bound kernel 的极限优化 |
| `scalar-multiply` | #1 Vector Addition（grid-stride + coalesced 基础）· #21 ReLU（逐元素 + 分支）· #8 Matrix Addition（2D 逐元素）· #62 Value Clipping（逐元素 clamp） | 标量 × 向量逐元素，练习最简 elementwise kernel |
| `reverse-array` | #63 Interleave（写索引映射 + coalesced）· #1 Vector Addition（1D grid-stride 基础）· #31 Matrix Copy（coalesced 带宽优化）· #62 Value Clipping（逐元素 + 索引） | 1D 并行 in-place swap + coalesced，练习数据重排类 kernel |
| `vector-reversal` | #19 Reverse Array（同类型基础题）· #63 Interleave（索引重排练习）· #62 Value Clipping（逐元素索引）· #31 Matrix Copy（coalesced 带宽优化） | 1D 向量反转 + coalesced，练习 in-place swap 与索引映射 |
| `element-reversal` | #19 Reverse Array（同类型基础题）· #63 Interleave（索引重排练习）· #62 Value Clipping（逐元素索引）· #31 Matrix Copy（coalesced 带宽优化） | 逐元素反转 + 索引映射，练习 elementwise 重排 |
| `leaky-relu` | #21 ReLU（最简激活函数对比，无负斜率）· #52 SiLU（融合激活函数，练习 __expf）· #68 Sigmoid（数学函数逐元素，练习 exp 实现）· #65 GeGLU（GELU 门控变体，更复杂激活） | 逐元素激活函数 family，练习分支/无分支 kernel 与合并访存 |
| `sigmoid` | #21 ReLU（最简激活函数对比）· #52 SiLU（融合 sigmoid+mul，练习 fused kernel）· #23 Leaky ReLU（分支激活对比）· #54 SwiGLU（SwiGLU 使用 sigmoid 组件） | 逐元素数学函数，练习 __expf 快速数学与合并访存 |
| `color-inversion` | #1 Vector Addition（grid-stride + coalesced 基础）· #21 ReLU（逐元素 kernel，分支开销）· #66 RGB to Grayscale（多通道加权求和，类似逐元素）· #8 Matrix Addition（2D grid 逐元素） | 逐元素图像变换，练习多通道 coalesced 访存与索引映射 |
| `interleave` | #1 Vector Addition（grid-stride + coalesced 基础）· #31 Matrix Copy（纯拷贝带宽优化）· #19 Reverse Array（1D 并行 in-place swap）· #62 Value Clipping（逐元素 clamp） | 写索引映射练习，coalesced 写回与数据重排 |

#### B. 卷积与池化（Convolution & Pooling）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `1d-convolution` | #10 2D Convolution（halo 扩展到二维）· #11 3D Convolution（体数据 halo）· #90 Causal Depthwise Conv1d（因果卷积变体）· #28 Gaussian Blur（可分离卷积） | 1D shared memory halo，练习卷积边界填充与 tile 加载 |
| `2d-convolution` | #9 1D Convolution（halo 基础入门）· #11 3D Convolution（体数据 halo 扩展）· #28 Gaussian Blur（可分离卷积，行列分离优化）· #42 2D Max Pooling（滑窗 reduction，类似 tiling 模式） | shared memory halo + 常数内存，练习卷积类 kernel 的边界处理与 tiling |
| `causal-depthwise-conv1d` | #9 1D Convolution（1D 卷积基础，halo 填充入门）· #10 2D Convolution（2D shared memory halo + 常数内存）· #11 3D Convolution（3D 体数据 halo 扩展）· #28 Gaussian Blur（可分离卷积，行列分离优化） | 因果卷积 + depthwise 分组，练习卷积边界处理与通道独立并行 |
| `2d-max-pooling` | #10 2D Convolution（2D shared memory halo + tiling）· #9 1D Convolution（1D 卷积，halo 基础）· #28 Gaussian Blur（可分离卷积，滑窗模式）· #90 Causal Depthwise Conv1d（因果卷积变体） | 滑窗 reduction，练习 2D 索引映射与 padding 边界处理 |

#### C. 归约与扫描（Reduction & Scan）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `reduction` | #17 Dot Product（元素乘 + 全局归约，归约的直接应用）· #43 Count Array Element（计数归约 + atomic，对比归约与 atomic）· #27 MSE（平方差归约，归约在损失函数中的应用）· #51 Max Subarray Sum（scan + 归约的综合练习） | 树形归约 + warp shuffle，练习并行归约这一核心模板 |
| `prefix-sum` | #70 Segmented Prefix Sum（分段 scan，段边界处理进阶）· #72 Stream Compaction（predicate + scan 得到输出位置）· #47 Subarray Sum（prefix sum 直接应用求子和）· #82 Linear Recurrence（线性递推，scan 的数学扩展） | warp scan + 三阶段分块 scan，练习并行前缀扫描这一核心模板 |
| `segmented-prefix-sum` | #16 Prefix Sum（分段 scan 的基础）· #72 Stream Compaction（scan 的另一应用）· #82 Linear Recurrence（scan 的数学扩展）· #94 SSM Selective Scan（分段 scan 的前沿应用） | 分段 scan + 段边界处理，练习 prefix sum 的高阶变体 |
| `stream-compaction` | #16 Prefix Sum（stream compaction 的基础）· #70 Segmented Prefix Sum（分段 scan 进阶）· #43 Count Array Element（predicate 计数）· #87 Speculative Decoding Verification（compaction 的推理应用） | predicate + scan 得到输出位置，练习 scan 的筛选应用 |
| `dot-product` | #4 Reduction（树形归约，dot product 的基础组件）· #58 FP16 Dot Product（半精度归约）· #27 MSE（平方差归约的变体）· #17 Dot Product（同题，可对比不同归约写法） | 元素乘 + block 归约，练习融合 kernel 与归约 |
| `max-subarray-sum` | #16 Prefix Sum（本题的核心基础）· #47 Subarray Sum（prefix sum 直接应用）· #48 2D Subarray Sum（扩展到二维）· #72 Stream Compaction（scan 的另一应用） | prefix sum + Kadane scan + 归约，练习 scan 的综合应用 |
| `histogramming` | #43 Count Array Element（计数归约，atomic vs reduction 对比）· #44 Count 2D Array Element（2D 计数，扩展到多维 atomic）· #29 Top K Selection（bitonic 排序 + 堆归约，相关并行模式）· #36 Radix Sort（Radix Sort，histogram + scan 综合） | shared memory 直方图 + atomic 冲突，练习计数类并行模式 |
| `count-array-element` | #4 Reduction（树形归约，count 的归约基础组件）· #44 Count 2D Array Element（2D 计数，扩展到多维 atomic）· #13 Histogramming（shared memory 直方图，atomic + reduction 综合应用）· #27 Mean Squared Error（平方差归约，归约在损失函数中的应用） | predicate 归约 + atomic 计数，练习 count 类 kernel 的归约与 atomic 权衡 |
| `argmax` | #4 Reduction（树形归约，argmax 的基础组件）· #29 Top K Selection（排序归约进阶）· #5 Softmax（先求 max 再归一化）· #17 Dot Product（block 归约练习） | 归约变体（求最大值索引），练习比较归约与 warp shuffle |
| `top-k-selection` | #60 Top-p Sampling（排序 + 累积概率 + 采样）· #15 Sorting（通用并行排序）· #36 Radix Sort（按位 histogram + scan 排序）· #71 Parallel Merge（归并排序网络） | bitonic 排序 + 堆归约，练习并行排序与选择 |
| `subarray-sum` | #16 Prefix Sum（prefix sum 直接应用求子和）· #4 Reduction（树形归约基础组件）· #48 2D Subarray Sum（扩展到二维前缀和）· #51 Max Subarray Sum（scan + 归约综合练习） | prefix sum 直接应用，练习范围归约与 block reduce |
| `mean-squared-error` | #4 Reduction（树形归约，MSE 的基础组件）· #17 Dot Product（block 归约，类似模式）· #25 Categorical Cross Entropy Loss（归约 + log，损失函数变体）· #58 FP16 Dot Product（半精度归约，低精度变体） | 归约在损失函数中的应用，练习 fused kernel + block reduce |
| `fp16-dot-product` | #4 Reduction（树形归约基础组件）· #17 Dot Product（FP32 版 dot product 对比）· #57 FP16 Batched Matrix Multiplication（FP16 + Tensor Core，半精度 GEMM）· #27 Mean Squared Error（归约在损失函数中的应用） | 半精度归约，练习 __half 类型转换与 FP32 累加精度保证 |

#### D. 矩阵乘法与 GEMM（GEMM & Matmul）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `matrix-multiplication` | #22 GEMM（完整 GEMM，register blocking + 双缓冲进阶）· #30 Batched Matrix Multiplication（batched GEMM，多组矩阵并行）· #37 Matrix Power（重复 matmul，练习 tiling 复用）· #32 INT8 Quantized MatMul（INT8 量化 GEMM，低精度计算） | tiled matmul + register tiling，练习 GEMM 这一 compute-bound 核心模板 |
| `gemm` | #2 Matrix Multiplication（naive tiled matmul，对比基础写法）· #30 Batched Matrix Multiplication（batched GEMM，多矩阵并行调度）· #32 INT8 Quantized MatMul（INT8 量化 GEMM，低精度 + scale）· #57 FP16 Batched MatMul（FP16 + Tensor Core，半精度 GEMM） | GEMM tiling / register blocking / 双缓冲，练习 compute-bound kernel 优化全链路 |
| `batched-matrix-multiplication` | #22 GEMM（完整 GEMM，register blocking 基础）· #57 FP16 Batched MatMul（半精度 + Tensor Core）· #32 INT8 Quantized MatMul（低精度 batch）· #37 Matrix Power（重复 matmul 调度） | batched GEMM + 多组矩阵并行调度，练习 batch 维度的 kernel 设计 |
| `matrix-transpose` | #31 Matrix Copy（纯拷贝带宽优化，对比转置的访存模式）· #10 2D Convolution（2D shared memory halo + tiling）· #2 Matrix Multiplication（tiled matmul，同样用 shared mem 分块）· #63 Interleave（写索引重排，coalesced 练习） | shared memory tiling + bank conflict padding，练习矩阵数据重排类 kernel |
| `int8-quantized-matmul` | #22 GEMM（GEMM tiling 基础）· #30 Batched Matrix Multiplication（batched GEMM）· #81 INT4 Weight-Only Quantized MatMul（4-bit 量化进阶）· #64 Weight Dequantization（反量化基础操作） | INT8 量化 GEMM，练习低精度计算与 requantize 流程 |
| `sparse-matrix-vector-multiplication` | #17 Dot Product（warp shuffle 归约，SpMV 行内归约的基础组件）· #75 Sparse Matrix-Dense Matrix Multiplication（稀疏 GEMM，SpMV 的矩阵版进阶）· #22 General Matrix Multiplication (GEMM)（稠密 GEMM tiling，对比稀疏 vs 稠密访存模式）· #4 Reduction（树形归约，SpMV 行内归约的基础组件） | CSR 稀疏格式 + warp shuffle 行内归约，练习不规则访存与稀疏矩阵乘模板 |

#### E. 注意力机制（Attention）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `softmax-attention` | #12 Multi-Head Attention（FlashAttention 思想）· #53 Causal Self-Attention（因果掩码，下三角掩码）· #17 Dot Product（attention 的基础组件）· #5 Softmax（attention 的基础组件） | fused softmax+matmul + 数值稳定，练习 attention score 计算全流程 |
| `attention` | #6 Softmax Attention（本题的基础版本）· #12 Multi-Head Attention（head 并行进阶）· #53 Causal Self-Attention（因果掩码）· #59 Sliding Window Self-Attention（局部窗口） | attention score + softmax + weighted sum，练习 fused attention 全流程 |
| `causal-self-attention` | #59 Sliding Window（另一种局部 attention 窗口）· #80 GQA（KV head 复用的 attention 变体）· #12 Multi-Head Attention（head 并行）· #6 Softmax Attention（无 mask 基础版） | 因果掩码 + fused attention，练习 mask 对 attention 的影响 |
| `multi-head-attention` | #6 Softmax Attention（单 head 基础版）· #80 GQA（KV head 共享变体）· #53 Causal Self-Attention（因果掩码）· #74 GPT-2 Block（attention 的综合应用） | FlashAttention 思想 + head 并行，练习融合 attention 的高阶优化 |
| `sliding-window-self-attention` | #53 Causal Self-Attention（因果掩码变体）· #80 GQA（KV head 共享变体）· #6 Softmax Attention（无窗口基础版）· #92 Decaying Causal Attention（衰减因子变体） | 局部注意力窗口 + fused attention，练习窗口 mask 对 attention 的影响 |
| `grouped-query-attention` | #12 Multi-Head Attention（MHA 基础版）· #53 Causal Self-Attention（mask 变体）· #96 INT8 KV-Cache Attention（量化 + KV cache）· #59 Sliding Window（另一种 attention 变体） | KV head 共享 + attention，练习 GQA 的分组调度 |
| `int8-kv-cache-attention` | #80 GQA（KV head 复用的 attention 基础）· #64 Weight Dequantization（反量化基础）· #53 Causal Self-Attention（attention 基础）· #32 INT8 Quantized MatMul（INT8 计算基础） | 量化 KV cache + attention，练习低精度推理与 attention 的结合 |
| `attn-w-linear-bias` | #6 Softmax Attention（fused softmax+matmul 基础版）· #53 Causal Self-Attention（因果掩码变体）· #12 Multi-Head Attention（head 并行进阶）· #59 Sliding Window Self-Attention（滑窗注意力变体） | 线性偏置注意力，练习 attention + positional bias 的融合 |
| `decaying-causal-attention` | #53 Causal Self-Attention（因果掩码基础版）· #59 Sliding Window Self-Attention（滑窗注意力变体）· #6 Softmax Attention（无掩码基础版）· #80 Grouped Query Attention (GQA)（KV head 共享变体） | 衰减因子 + 因果掩码，练习 attention mask 变体与增量衰减计算 |

#### F. 归一化与嵌入（Normalization & Embedding）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `softmax` | #50 RMS Normalization（RMS Norm，归约 + 归一化变体）· #6 Softmax Attention（fused softmax+matmul，数值稳定进阶）· #4 Reduction（树形归约，softmax 的基础组件）· #40 Batch Normalization（Batch Norm，mean/var 归约归一化） | 三遍 kernel + 数值稳定，练习归约与归一化的融合 |
| `rms-normalization` | #40 Batch Normalization（mean/var 归约归一化）· #105 Group Normalization（分组归约）· #5 Softmax（max+sum 归约 + 归一化）· #50 RMS Normalization（同题对比不同实现） | 归约 + 归一化（root mean square），练习 norm 类 kernel |
| `batch-normalization` | #50 RMS Normalization（归约 + 归一化变体）· #105 Group Normalization（分组归约）· #4 Reduction（mean/var 归约的基础组件）· #5 Softmax（max + sum 归约归一化） | mean/var 归约 + 归一化，练习统计归约类 norm kernel |
| `group-normalization` | #40 Batch Normalization（mean/var 归约归一化，跨 batch 维度）· #50 RMS Normalization（RMS Norm，归约 + 归一化变体）· #5 Softmax（max+sum 归约 + 归一化）· #4 Reduction（树形归约，norm 的基础组件） | 分组归约归一化，练习两遍 scan + shared memory reduction |
| `token-embedding-layer` | #61 RoPE（位置嵌入的另一种实现）· #41 Simple Inference（embedding 的推理应用）· #64 Weight Dequantization（查表式反量化）· #106 Token Embedding Layer（同题，可对比不同实现） | gather / lookup table，练习嵌入查表类 kernel |
| `rope-embedding` | #106 Token Embedding（嵌入查表基础）· #54 SwiGLU（融合 elementwise 进阶）· #52 SiLU（fused elementwise）· #50 RMS Normalization（归约 + elementwise） | 复数旋转 + elementwise，练习位置编码的并行实现 |

#### G. Transformer 组件与推理优化（Transformer Blocks & Inference）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `gpt-2-transformer-block` | #12 Multi-Head Attention（block 的核心组件）· #50 RMS Norm（归一化组件）· #54 SwiGLU（激活/MLP 组件）· #85 LoRA Linear（低秩线性层变体） | LN + Attention + MLP 综合模块，练习多 kernel 流水线与模块融合 |
| `swiglu` | #52 SiLU（SwiGLU 的激活组件）· #21 ReLU（最简激活对比）· #65 GeGLU（GELU 门控变体）· #84 SwiGLU MLP Block（SwiGLU 的完整 MLP 应用） | 融合激活 + 门控乘法，练习 fused MLP 组件 kernel |
| `swiglu-mlp-block` | #54 SwiGLU（SwiGLU 激活组件，本 block 的核心 elementwise）· #22 GEMM（GEMM tiling，3 个 matmul 的基础组件）· #74 GPT-2 Transformer Block（更大的 transformer block 综合）· #52 SiLU（SiLU 激活，SwiGLU 的子组件） | 融合 MLP block，SwiGLU 的完整应用 |
| `silu` | #21 ReLU（最简激活函数对比）· #68 Sigmoid（silu 的组件）· #54 SwiGLU（融合激活 + 门控进阶）· #23 Leaky ReLU（分支激活对比） | 融合 sigmoid + mul 逐元素，练习 fused activation kernel |
| `lora-linear` | #41 Simple Inference（基础推理管线）· #64 Weight Dequantization（低精度推理基础）· #84 SwiGLU MLP Block（融合 MLP 模块）· #2 Matrix Multiplication（低秩 matmul 基础） | 低秩适配 + 融合低秩矩阵，练习推理优化中的低秩计算 |
| `simple-inference` | #85 LoRA Linear（低秩适配的推理变体）· #106 Token Embedding（推理管线组件）· #74 GPT-2 Block（完整推理模块）· #2 Matrix Multiplication（Linear 的 CUDA 实现） | PyTorch Linear 前向封装，练习推理管线的最简形态 |
| `speculative-decoding-verification` | #29 Top-K Selection（排序归约基础）· #60 Top-p Sampling（排序 + 采样）· #72 Stream Compaction（scan + predicate）· #16 Prefix Sum（验证的 scan 基础） | draft token 验证 + scan，练习推理优化中的并行验证 |
| `adder-transformer` | #74 GPT-2 Transformer Block（完整 transformer block 综合应用）· #12 Multi-Head Attention（标准 MHA，对比加法注意力）· #6 Softmax Attention（softmax attention 基础版）· #85 LoRA Linear（低秩线性层变体） | 加法注意力替代 softmax，练习多 kernel 推理流水线 |
| `moe-topk-gating` | #29 Top K Selection（bitonic 排序 + 堆归约基础）· #60 Top-p Sampling（排序 + 累积概率 + 采样）· #5 Softmax（softmax，top-k 后的归一化）· #84 SwiGLU MLP Block（MoE 中的 MLP 组件） | top-k 选择 + softmax，练习排序归约与 MoE 路由 |

#### H. 量化与低精度（Quantization）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `weight-dequantization` | #32 INT8 Quantized MatMul（量化计算的应用）· #81 INT4 Weight-Only（4-bit 打包反量化）· #96 INT8 KV-Cache（量化 attention 应用）· #85 LoRA Linear（低秩 + 量化推理） | 量化反量化到 fp16/fp32，练习低精度推理的基础操作 |

#### J. 高级算法与数学（Advanced Algorithms & Math）

| 题解 (slug) | 推荐练习（编号 · 关联） | 选材主线 |
|-------------|------------------------|----------|
| `nearest-neighbor` | #22 General Matrix Multiplication (GEMM)（GEMM tiling，nearest neighbor 的分块复用同构）· #20 K-Means Clustering（K-Means 距离矩阵，pairwise distance 的迭代应用）· #4 Reduction（树形归约，argmin 更新的归约基础组件）· #33 Ordinary Least Squares（线性代数 + 归约，距离/矩阵计算的另一变体） | pairwise distance + shared memory tiling 数据复用，练习 compute-bound kernel 的算术强度提升 |
| `2d-jacobi-stencil` | #10 2D Convolution（2D shared memory halo + tiling，stencil 的加权变体）· #9 1D Convolution（1D shared memory halo，stencil 的一维基础）· #42 2D Max Pooling（滑窗 reduction，类似的 tiling + 边界处理模式）· #11 3D Convolution（3D shared memory halo，stencil 扩展到体数据） | stencil 计算 + shared memory halo 边界复用，练习网格类 kernel 的邻居冗余读消除 |

> 💡 **选材原则**：每道题的 4 条推荐遵循「**1 道同类型基础题 + 1 道进阶变体 + 1 道综合应用 + 1 道跨领域延伸**」的结构，确保从基础到进阶的渐进练习路径。推荐题优先选 §1.2 推荐路径中已完成的题（可回看自己题解），其次选 §1.4 清单中相关概念题。

## 2. 目录组织

LeetGPU 题解按 `weekN/dayM/` 组织，与 `leetcode/daily/` 结构一致；`weekN/dayM/` 主要用于本地归类与侧边栏分组，**不强求与每日教程** `aiinfra/daily/weekN/dayM/` **严格一一对应**。题解系列可以独立扩展，例如继续新增 `week9/week10/` 等，只要题解文件名中的 slug 唯一即可：

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
2. **按 weekN/dayM 组织**：题解 `.md` 放在 `leetgpu/weekN/dayM/` 下，**周/日编号与每日教程** `aiinfra/daily/weekN/dayM/` **对齐**（Day 1 → `week1/day1/`，Day 7 → `week1/day7/`）。一个 day 目录下通常只有一篇题解（即当日教程任务 4 对应的题）。
3. **题解文件名**：`leetgpu-<slug>-solution.md`，其中 `<slug>` 是 LeetGPU 平台的题目 URL slug（如 `vector-addition`、`prefix-sum`）。文件名不随 week/day 变化，slug 即唯一标识。
4. **图片目录**：`leetgpu/images/`，所有题解共享（不按 week/day/题分散）。图片在题解中用 `../../images/xxx.svg` 相对路径引用（题解位于 `weekN/dayM/` 下，`../../` 回到 `leetgpu/`；`build.py` 递归扫描时统一重写为 `./images/xxx.svg`，输出页面扁平化到 `website/` 根）。
5. **选题与每日教程对齐（推荐但非强制）**：题解尽量与 `aiinfra/daily/weekN/dayM/` 每日教程的「任务 4」LeetGPU 在线题目主题保持一致，便于读者按周学习；但 LeetGPU 题解可以独立扩展，不强制一一对应，新增题解可放入 `week9/week10/` 等目录，slug 为唯一标识。

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
- 图片引用用相对路径：`![<中文alt>](../../images/<filename>.svg)`（题解位于 `weekN/dayM/` 下，`../../images/` 解析到共享的 `leetgpu/images/`，由 `build.py` 统一重写为 `./images/`）。
- 每篇题解引用 **2-4 张 SVG/PNG 插图**，并配 `### 4.2 代码详解` 子节（详见 §5）。

### 数学公式

- 行内公式用 `$...$`，块级公式用 `$$...$$`
- **禁止**用反引号 `` `...` `` 包裹数学公式，否则会被渲染为等宽代码，KaTeX 不会识别
- 公式内函数/运算符使用 LaTeX 命令：`\exp`、`\log`、`\sum`、`\max`、`\frac`、`\sqrt`，避免直接写 `exp`、`log`、`Σ`、`√`


## 4. 图片风格：手绘 sketch 风（Excalidraw-like）

**所有插图统一为手绘 sketch 风**，与每日教程（`aiinfra/daily-tutorial/SKILL.md`）和 LeetCode 题解保持一致。具体要求：

- **禁止 ASCII 图片**：所有示意图、流程图、架构图一律用 SVG，不要在 Markdown 中嵌入 ASCII 字符画（如用 `+---+`、`|   |` 拼成的表格或流程图）

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

用 SVG 滤镜 `feTurbulence` + `feDisplacementMap` 给所有图形叠加轻微抖动，实现手绘效果。**每张 SVG 顶部固定引入以下** `<defs>`：

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

## 5. 代码详解与 SVG 图解规范

每篇题解除了 6 段基础结构外，还应包含 **代码详解** 子节和足够的 **SVG 图解**。这是题解质量的核心区分点——有图有详解的题解能让读者从"看懂代码"升级到"理解为什么这样写"。

### 5.1 代码详解子节

在 **§4 Kernel 实现** 的代码块之后（通常在 `### 4.1 LeetGPU 提交版本` 和 `## 5. 性能分析` 之间），添加 `### 4.2 代码详解` 子节。

#### 结构模板

```markdown
### 4.2 代码详解

<1-2 句话概括 kernel 的核心策略>

| 步骤 | 代码 | 说明 |
|------|------|------|
| **坐标计算** | `i = blockIdx.x * blockDim.x + threadIdx.x` | thread 到全局索引的映射 |
| **加载/计算** | `... = input[...]` | 核心访存或计算逻辑 |
| **同步** | `__syncthreads()` | 屏障的作用与缺失后果 |
| **写回** | `output[...] = ...` | 结果写回 |

**关键索引关系**：
- `<变量>` = `<公式>` — <含义>
- ...

> 💡 **关键洞察**：<一句话点出 kernel 设计的本质洞察>
```

#### 详解内容要求

| 维度 | 要求 |
|------|------|
| **逐行覆盖** | kernel 中每一段关键代码都要在表格或列表中有对应解释 |
| **索引计算** | 必须解释 `threadIdx` → `blockIdx` → 全局坐标的映射链 |
| **同步语义** | 每次 `__syncthreads()` 要说明"等什么"和"不等会怎样" |
| **Worked Example** | 对复杂 kernel（convolution、attention、scan 等）给出具体数值的逐步演算 |
| **变量表** | 对多变量 kernel 给出"变量-含义-初始值"对照表 |
| **关键洞察** | 用 `> 💡` blockquote 点出 kernel 设计的核心洞察（1-2 句） |

#### 不同复杂度的详解深度

| Kernel 类型 | 详解深度 | 示例文件 |
|-------------|----------|----------|
| **简单 element-wise**（vector-add、relu、scalar-multiply） | 3-5 行表格 + 索引公式 | `week1/day1` |
| **Shared memory tiling**（transpose、matmul、convolution） | 完整索引映射表 + worked example + bank conflict 分析 | `week2/day3` |
| **归约类**（reduction、softmax、dot-product） | warp shuffle 步骤分解 + block reduce 两阶段流程图 | `week2/day4` |
| **融合 kernel**（flash attention、online softmax） | 三公式逐步数值演算 + k 循环数据流图 + `__syncthreads` 作用表 | `week2/day5` |
| **多 kernel 流水线**（GPT-2 block、stream compaction） | kernel 链调用顺序表 + 每 kernel 一句话 + HBM IO 表 | `week4/day7` |

### 5.2 SVG 图解规范

#### 每篇题解的 SVG 数量要求

| 题解类型 | SVG 数量 | 说明 |
|----------|----------|------|
| 简单题（element-wise） | 1-2 张 | 至少 1 张概念图（数据流或索引映射） |
| 中等题（shared memory / reduction） | 2-3 张 | 概念图 + 存储层次图或索引详解图 |
| 复杂题（attention / scan / 多 kernel） | 3-5 张 | 概念图 + 数据流图 + 逐步演算图 + 性能对比图 |

> ⚠️ **最低要求**：每篇含 CUDA kernel 的题解**至少 1 张 SVG**。纯 stub（deferral 到其他文件）和 PyTorch 题解（无 CUDA kernel）无需 SVG。

#### SVG 内容类型

| 类型 | 用途 | 何时使用 | 命名模式 |
|------|------|----------|----------|
| **概念总览图** | 展示 kernel 整体数据流和并行策略 | 每篇必选 | `<slug>_overview.svg` |
| **索引计算图** | 逐步展示坐标映射（thread→shared→global） | tiling / halo 类必选 | `<slug>_index_calculation.svg` |
| **逐步演算图** | 具体数值的 step-by-step 推演 | attention / online softmax 类必选 | `<slug>_worked.svg` |
| **数据流图** | 多阶段 kernel 的 pipeline 流程 | 多 kernel 或三遍扫描类必选 | `<slug>_dataflow.svg` |
| **性能对比图** | naive vs optimized 的指标对比 | 有明确优化对比时可选 | `<slug>_roofline.svg` |
| **block 映射图** | grid/block 到数据的映射关系 | 多 head / batched 类可选 | `<slug>_block_mapping.svg` |

#### SVG 引用路径

题解位于 `leetgpu/weekN/dayM/`，SVG 位于 `leetgpu/images/`，因此引用路径为：

```markdown
![<中文描述>](../../images/<filename>.svg)
```

`build.py` 会自动将 `../../images/` 重写为 `./images/`（网站输出扁平化）。

#### SVG 创建要点

1. **手绘 sketch 风**：使用 `feTurbulence` + `feDisplacementMap` 滤镜（详见 §4.2）
2. **配色一致**：蓝（输入）、绿（输出）、橙（shared/中间）、红（关键操作/警告）
3. **中文标注**：标题、图例、公式说明用中文；变量名和代码用 monospace 英文
4. **具体数值**：worked example 类 SVG 必须用具体数字（如 `N=3, d=2, scale=0.707`），不能只有抽象符号
5. **viewBox**：使用 `viewBox="0 0 W H"` 而非固定 width/height，保证响应式缩放

### 5.3 重复 slug 文件同步

LeetGPU 平台的同一道题可能在多个 week/day 出现（如 `softmax-attention` 在 `week2/day5` 和 `week4/day1` 都有）。由于 `build.py` 按 slug 扁平输出 HTML，**后构建的文件会覆盖先构建的**。

**同步规则**：

1. **主文件**：内容最完整的版本作为主文件（通常是首次出现的 week/day）
2. **副本文件**：用 `cp` 从主文件同步，保持内容完全一致
3. **stub 文件**：如果某 week/day 的题解只是指向其他文件的 deferral stub（如 `> 本题解与 ... 内容相同`），则**不需要同步**——stub 只保留标题和指引链接
4. **同步检查**：修改主文件后，用 `diff` 检查所有同 slug 文件是否需要同步

```bash
# 检查同 slug 文件是否一致
diff leetgpu/week2/day5/leetgpu-softmax-attention-solution.md \
     leetgpu/week4/day1/leetgpu-softmax-attention-solution.md
# 如不一致，同步
cp leetgpu/week2/day5/leetgpu-softmax-attention-solution.md \
   leetgpu/week4/day1/leetgpu-softmax-attention-solution.md
```

### 5.4 完成度检查清单

为题解补充 SVG + 代码详解后，用以下清单自检：

- [ ] 至少 1 张 SVG 引用（`![...](../../images/...svg)`）
- [ ] SVG 文件存在于 `leetgpu/images/`
- [ ] `### 4.2 代码详解` 子节存在（或等效的详解标题）
- [ ] 详解覆盖 kernel 的关键代码段（索引计算、访存模式、同步屏障）
- [ ] 复杂 kernel 有 worked example（具体数值逐步推演）
- [ ] `> 💡 关键洞察` blockquote 存在
- [ ] 同 slug 的重复文件已同步（`diff` 无差异）
- [ ] `## 同类练习题` 章节存在，且内容与 §1.6 推荐映射完全一致（4 条推荐 + 选材主线）
- [ ] `python3 build.py` 构建成功
- [ ] 生成的 HTML 中 SVG 路径正确（`./images/...svg`）

## 6. 网站构建集成

题解写完后会被 `build/leetgpu.py` 自动读取并生成网页：

- `build.py` **递归扫描** `leetgpu/` 下所有 `leetgpu-*.md` 文件（用 `rglob()`，自动识别 `weekN/dayM/` 子目录）。
- 解析路径中的 `weekN/dayM/` 作为分组依据，侧边栏按 week→day 手风琴式分组。
- 解析一级标题 `# LeetGPU <题目名> 题解` 作为侧边栏与列表页标题。
- 图片路径 `images/xxx.svg` 在题解页被重写为 `./images/xxx.svg`（网站输出目录扁平化）。
- 生成 `public/leetgpu/index.html`（概览页）和 `public/leetgpu/leetgpu-<slug>-solution.html`（各题解页，扁平输出）。
- `leetgpu/images/` 自动复制到 `public/leetgpu/images/` 部署。

**验证命令**：

```bash
python3 build.py                     # 组合构建全站（含 leetgpu）
```

**自检清单**：

- [ ] 题解位于 `leetgpu/weekN/dayM/leetgpu-<slug>-solution.md`（week/day 用于本地归类，slug 唯一）
- [ ] 一级标题 `# LeetGPU <题目名> 题解`
- [ ] 含 6 段结构（题目概述/CPU基线/GPU设计/Kernel实现/性能分析/复杂度分析）
- [ ] Kernel 代码完整可编译（含 main、cudaMalloc、验证、cudaFree）
- [ ] 含 2-4 张 SVG/PNG 插图，引用格式 `![中文alt](../../images/xxx.svg)`
- [ ] 含 `### 4.2 代码详解` 子节（逐行解释 + 索引表 + 关键洞察）
- [ ] SVG 为手绘 sketch 风（含 `feTurbulence` 抖动滤镜 + Comic Sans/Kaiti SC 字体）
- [ ] 复杂 kernel 有 worked example（具体数值逐步推演）
- [ ] 同 slug 重复文件已同步（`diff` 无差异）
- [ ] 含 `## 同类练习题` 章节，内容与 §1.6 推荐映射一致
- [ ] 含 ncu profiling 命令与关键指标
- [ ] `python3 build.py` 成功生成对应 `public/leetgpu/leetgpu-<slug>-solution.html`
- [ ] `git push origin` 推送题解（commit + push 到远程）

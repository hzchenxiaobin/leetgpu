# LeetGPU 题解

> 90 道 [LeetGPU](https://leetgpu.com/) CUDA 挑战题解 —— 每道含完整可编译 kernel + ncu profiling + 手绘 sketch 风 SVG 图解，按 CUDA 概念覆盖选题、按难度归档。

📚 **在线网站**：https://hzchenxiaobin.github.io/leetgpu/

本仓库是 [ai-infra-notes](https://github.com/hzchenxiaobin/ai-infra-notes) 8 周 AI Infra 学习路线中 LeetGPU 题解的独立归档，题解存放于 `solutions/<difficulty>/` 目录，按 **easy / medium / hard** 三档难度归类，编号对齐 [leetgpu-challenges](https://github.com/sayaklahiri/leetgpu-challenges) 的 `challenges/<difficulty>/<number>_<name>/`，作为 Coding 任务的实战检验。

## 选题逻辑

LeetGPU 平台的题目都是 **CUDA Kernel 实现题**，选题目标是**用最少的题覆盖 GPU 编程核心概念**：

- **概念覆盖优先**：每道题对应一个 CUDA 核心概念（grid-stride、shared memory、warp shuffle、bank conflict、reduction、scan、tiling 等），避免连续多题重复
- **难度递进**：memory-bound 入门 → shared memory 进阶 → warp shuffle / tiling 高阶 → 综合题压轴
- **配合每日教程**：题解作为每日教程 Coding 任务的实战检验，选题与当日主题强相关
- **性能导向**：优先选能体现 ncu profiling 价值的题（有明确瓶颈指标可观察、可优化）

详细写作规范见 [SKILL.md](SKILL.md)。

## 题解列表

共 **105 道**（简单 22 / 中等 68 / 困难 15），覆盖 Vector Addition、GEMM、Softmax、Attention、Prefix Sum、PagedAttention、GQA、Speculative Decoding、GPT-2 Block、FlashAttention、Linear Attention、K-Means、Bitonic Sort、Radix Sort、Multi-Agent Simulation、BFS、Floyd-Warshall、GRPO、StreamingLLM 等。

### Easy · 简单（22 道）

| # | 题目 | 核心概念 |
|---|------|----------|
| 1 | [Vector Addition](solutions/easy/1-vector-add.md) | grid-stride loop、coalesced access、memory-bound |
| 2 | [Matrix Multiplication](solutions/easy/2-matrix-multiplication.md) | GEMM、shared memory tiling、register tiling |
| 3 | [Matrix Transpose](solutions/easy/3-matrix-transpose.md) | shared memory tiling、bank conflict padding |
| 7 | [Color Inversion](solutions/easy/7-color-inversion.md) | elementwise、image processing、`uchar4` 向量化 |
| 8 | [Matrix Addition](solutions/easy/8-matrix-addition.md) | float4 向量化、Roofline |
| 9 | [1D Convolution](solutions/easy/9-1d-convolution.md) | convolution、shared memory、halo |
| 19 | [Reverse Array](solutions/easy/19-reverse-array.md) | in-place swap、1D 并行、coalesced access |
| 21 | [ReLU](solutions/easy/21-relu.md) | elementwise、warp divergence、branchless |
| 23 | [Leaky ReLU](solutions/easy/23-leaky-relu.md) | elementwise、branchless、activation |
| 24 | [Rainbow Table](solutions/easy/24-rainbow-table.md) | elementwise、grid-stride、串行哈希循环、整数回绕 |
| 31 | [Matrix Copy](solutions/easy/31-matrix-copy.md) | 内存带宽、coalesced access、float4 向量化 |
| 41 | [Simple Inference](solutions/easy/41-simple-inference.md) | PyTorch、Linear、batch size、GEMM |
| 52 | [SiLU](solutions/easy/52-silu.md) | elementwise、grid-stride、`__expf` 快速数学 |
| 54 | [SwiGLU](solutions/easy/54-swiglu.md) | elementwise、kernel fusion、SiLU |
| 62 | [Value Clipping](solutions/easy/62-value-clipping.md) | clamp、fminf/fmaxf 无分支、warp divergence |
| 63 | [Interleave Arrays](solutions/easy/63-interleave.md) | grid-stride loop、索引映射 |
| 65 | [Gaussian Error Gated Linear Unit](solutions/easy/65-geglu.md) | GELU、kernel fusion、erf、门控激活 |
| 66 | [RGB to Grayscale](solutions/easy/66-rgb-to-grayscale.md) | 多通道加权求和、交织存储、coalesced |
| 68 | [Sigmoid](solutions/easy/68-sigmoid.md) | elementwise、fast math `__expf`、activation |
| 108 | [Vector Reversal](solutions/easy/108-vector-reversal.md) | 索引映射、coalesced access |
| 110 | [Scalar Multiply](solutions/easy/110-scalar-multiply.md) | element-wise、attention scaling |
| 111 | [Element Reversal](solutions/easy/111-element-reversal.md) | element-wise、结果验证 |

### Medium · 中等（68 道）

| # | 题目 | 核心概念 |
|---|------|----------|
| 4 | [Reduction](solutions/medium/4-reduction.md) | warp shuffle、`__shfl_down_sync` |
| 5 | [Softmax](solutions/medium/5-softmax.md) | safe softmax、三遍扫描、数值稳定性 |
| 6 | [Softmax Attention](solutions/medium/6-softmax-attention.md) | fused softmax+matmul、online softmax |
| 10 | [2D Convolution](solutions/medium/10-2d-convolution.md) | shared memory halo、常量内存 |
| 11 | [3D Convolution](solutions/medium/11-3d-convolution.md) | 3D halo tiling、立方级 shared memory、`__constant__` |
| 13 | [Histogramming](solutions/medium/13-histogramming.md) | shared memory 直方图、`atomicAdd`、privatization |
| 16 | [Prefix Sum](solutions/medium/16-prefix-sum.md) | scan、warp shuffle `__shfl_up_sync`、三阶段分块 |
| 17 | [Dot Product](solutions/medium/17-dot-product.md) | reduction、warp shuffle、kernel 融合 |
| 18 | [Sparse Matrix-Vector Multiplication](solutions/medium/18-sparse-matrix-vector-multiplication.md) | CSR、SpMV、warp shuffle、间接访存（gather） |
| 22 | [GEMM](solutions/medium/22-gemm.md) | FP16、WMMA、Tensor Core、shared memory tiling |
| 25 | [Categorical Cross Entropy Loss](solutions/medium/25-categorical-cross-entropy-loss.md) | Cross Entropy、log-sum-exp、数值稳定性、reduction、warp shuffle |
| 27 | [Mean Squared Error](solutions/medium/27-mean-squared-error.md) | reduction、kernel 融合、损失函数 |
| 28 | [Gaussian Blur](solutions/medium/28-gaussian-blur.md) | same-padding 卷积、shared memory halo、可分离卷积、零填充 |
| 29 | [Top K Selection](solutions/medium/29-top-k-selection.md) | bitonic sort、堆归约、selection |
| 30 | [Batched Matrix Multiplication](solutions/medium/30-batched-matrix-multiplication.md) | batched GEMM、tiled matmul、register blocking |
| 32 | [INT8 Quantized MatMul](solutions/medium/32-int8-quantized-matmul.md) | INT8 量化、tiled GEMM、INT32 累加、requantize |
| 33 | [Ordinary Least Squares](solutions/medium/33-ordinary-least-squares.md) | 线性代数、GEMM（XᵀX）、归约、Cholesky 分解、三角求解 |
| 34 | [Logistic Regression](solutions/medium/34-logistic-regression.md) | sigmoid、Newton-Raphson（IRLS）、tiled GEMM（Hessian）、Cholesky 分解、迭代 kernel launch |
| 35 | [Monte Carlo Integration](solutions/medium/35-monte-carlo-integration.md) | sum reduction、warp shuffle、atomicAdd、memory-bound |
| 37 | [Matrix Power](solutions/medium/37-matrix-power.md) | GEMM、shared memory tiling、register blocking、binary exponentiation、compute-bound |
| 38 | [Nearest Neighbor](solutions/medium/38-nearest-neighbor.md) | pairwise distance、shared memory tiling、argmin 归约 |
| 40 | [Batch Normalization](solutions/medium/40-batch-normalization.md) | normalization、reduction、数值稳定性 |
| 42 | [2D Max Pooling](solutions/medium/42-2d-max-pooling.md) | pooling、滑窗 reduction、padding 边界 |
| 43 | [Count Array Element](solutions/medium/43-count-array-element.md) | reduction、`atomicAdd`、predicate、warp shuffle |
| 44 | [Count 2D Array Element](solutions/medium/44-count-2d-array-element.md) | 2D 展平、predicate 归约、`atomicAdd`、warp shuffle |
| 45 | [Count 3D Array Element](solutions/medium/45-count-3d-array-element.md) | 3D 展平、predicate 归约、`size_t` 防溢出、warp shuffle |
| 47 | [Subarray Sum](solutions/medium/47-subarray-sum.md) | reduction、warp shuffle、block 归约 |
| 48 | [2D Subarray Sum](solutions/medium/48-2d-subarray-sum.md) | 2D 索引映射、子矩形展平、reduction、warp shuffle |
| 49 | [3D Subarray Sum](solutions/medium/49-3d-subarray-sum.md) | 3D 索引映射、子立方体展平、reduction、warp shuffle |
| 50 | [RMS Normalization](solutions/medium/50-rms-normalization.md) | RMSNorm、warp shuffle、Llama |
| 51 | [Max Subarray Sum](solutions/medium/51-max-subarray-sum.md) | 滑动窗口、prefix sum、reduction |
| 55 | [Attention with Linear Biases (ALiBi)](solutions/medium/55-attn-w-linear-bias.md) | ALiBi、positional bias、online softmax |
| 57 | [FP16 Batched Matrix Multiplication](solutions/medium/57-fp16-batched-matmul.md) | FP16 存储、FP32 累加、batched GEMM、Tensor Core |
| 58 | [FP16 Dot Product](solutions/medium/58-fp16-dot-product.md) | half 精度、warp shuffle、FP32 累加 |
| 60 | [Top-p Sampling](solutions/medium/60-top-p-sampling.md) | top-p sampling、nucleus sampling、softmax、bitonic sort、cumsum、CDF 采样、LLM 推理 |
| 61 | [Rotary Positional Embedding](solutions/medium/61-rope-embedding.md) | elementwise、rotate_half、位置编码 |
| 64 | [Weight Dequantization](solutions/medium/64-weight-dequantization.md) | element-wise、分块 scale、量化推理 |
| 67 | [MoE Top-K Gating](solutions/medium/67-moe-topk-gating.md) | top-k 选择、并行归约、softmax、MoE 路由 |
| 69 | [2D Jacobi Stencil](solutions/medium/69-jacobi-stencil-2d.md) | stencil 计算、shared memory halo、Jacobi 迭代 |
| 70 | [Segmented Prefix Sum](solutions/medium/70-segmented-prefix-sum.md) | segmented scan、warp shuffle |
| 71 | [Parallel Merge](solutions/medium/71-parallel-merge.md) | parallel merge、co-rank、binary search、merge path |
| 72 | [Stream Compaction](solutions/medium/72-stream-compaction.md) | scan、predicate、stream compaction |
| 75 | [Sparse Matrix-Dense Matrix Multiplication](solutions/medium/75-sparse-matrix-dense-matrix-multiplication.md) | SpMM、CSR、稀疏矩阵、gather 访存、scaled accumulation |
| 76 | [Adder Transformer Inference](solutions/medium/76-adder-transformer.md) | 多 kernel 流水线、autoregressive 推理、RoPE |
| 78 | [2D FFT](solutions/medium/78-2d-fft.md) | DFT、FFT、行-列分解（row-column decomposition）、shared memory、twiddle factor、compute-bound |
| 80 | [Grouped Query Attention (GQA)](solutions/medium/80-grouped-query-attention.md) | GQA、KV head 共享、LLM 推理 |
| 81 | [INT4 Weight-Only Quantized MatMul](solutions/medium/81-int4-matmul.md) | INT4 量化、W4A16、nibble 打包、group-wise dequant、FP16、LLM 推理 |
| 82 | [Linear Recurrence](solutions/medium/82-linear-recurrence.md) | associative scan、linear recurrence、warp shuffle、State Space Model |
| 84 | [SwiGLU MLP Block](solutions/medium/84-swiglu-mlp-block.md) | SwiGLU、MLP、GEMM、kernel fusion、LLaMA |
| 85 | [LoRA Linear](solutions/medium/85-lora-linear.md) | Low-Rank Adaptation、参数高效微调 |
| 87 | [Speculative Decoding Verification](solutions/medium/87-speculative-decoding-verification.md) | 投机解码、accept/reject 采样、CDF 查找 |
| 90 | [Causal Depthwise Conv1d](solutions/medium/90-causal-depthwise-conv1d.md) | causal、depthwise、边界处理 |
| 92 | [Decaying Causal Attention](solutions/medium/92-decaying-causal-attention.md) | causal mask、exponential decay、增量计算 |
| 94 | [SSM Selective Scan](solutions/medium/94-ssm-selective-scan.md) | Mamba、sequential recurrence、register state、thread-per-channel |
| 96 | [INT8 KV-Cache Attention](solutions/medium/96-int8-kv-cache-attention.md) | decode-phase、KV Cache、INT8 量化、per-token scale |
| 105 | [Group Normalization](solutions/medium/105-group-normalization.md) | normalization、reduction、GroupNorm |
| 106 | [Token Embedding Layer](solutions/medium/106-token-embedding-layer.md) | embedding、gather、LayerNorm、融合 kernel |
| 107 | [Argmax](solutions/medium/107-argmax.md) | 归约、argmax、`__shfl_down_sync` |
| 107 | [PPO Clipped Surrogate Loss](solutions/medium/107-ppo-clipped-surrogate-loss.md) | reduction、kernel fusion、PPO、RL、memory-bound、warp shuffle、atomicAdd |
| 108 | [DPO Sequence Loss](solutions/medium/108-dpo-sequence-loss.md) | Reduction、Kernel Fusion、Loss Function、Numerical Stability、softplus、memory-bound |
| 109 | [GRPO Surrogate Loss](solutions/medium/109-grpo-surrogate-loss.md) | GRPO、kernel fusion、两级归约、PPO clip、KL 惩罚 |
| 110 | [Parallel Reverse Scan (GAE)](solutions/medium/110-gae-reverse-scan.md) | Scan、Reverse Scan、Linear Recurrence、warp shuffle `__shfl_down_sync`、GAE、RL、memory-bound |
| 111 | [Softmax Attention Backward](solutions/medium/111-softmax-attention-backward.md) | attention backward、softmax 反向、GEMM、kernel fusion、reduction |
| 112 | [Attention with Sinks](solutions/medium/112-attention-with-sinks.md) | StreamingLLM、sink token、sliding window、复合掩码、online softmax |
| 113 | [Fused QKV Projection](solutions/medium/113-fused-qkv-projection.md) | kernel fusion、GEMM epilogue、reshape 融合 |
| 114 | [GEMV](solutions/medium/114-gemv.md) | memory-bound、合并访存、block 归约、float4 |
| 115 | [Layer Normalization](solutions/medium/115-layer-normalization.md) | 两次串行归约、mean-centering、数值稳定 |
| 116 | [Fused Add and RMSNorm](solutions/medium/116-fused-add-rmsnorm.md) | kernel fusion、RMSNorm、residual、memory-bound |

### Hard · 困难（15 道）

| # | 题目 | 核心概念 |
|---|------|----------|
| 12 | [Multi-Head Attention](solutions/hard/12-multi-head-attention.md) | MHA、FlashAttention、融合 attention |
| 14 | [Multi-Agent Simulation](solutions/hard/14-multi-agent-sim.md) | O(N²) pairwise interaction、shared memory tiling、per-thread 串行归约、float4 向量化 |
| 15 | [Sorting](solutions/hard/15-sorting.md) | bitonic sort、排序网络、compare-swap、shared memory 局部排序 |
| 20 | [K-Means Clustering](solutions/hard/20-kmeans-clustering.md) | 迭代算法、assign↔update 双 kernel、atomicAdd 归约、空簇处理 |
| 26 | [Multi-Head Cross-Attention](solutions/hard/26-multi-head-cross-attention.md) | Cross-Attention、FlashAttention、online softmax、融合 attention、batched kernel launch |
| 36 | [Radix Sort](solutions/hard/36-radix-sort.md) | 分布式排序、按位 histogram、exclusive prefix sum、stable scatter、warp shuffle scan |
| 39 | [Fast Fourier Transform](solutions/hard/39-fast-fourier-transform.md) | FFT、radix-2、Cooley-Tukey、蝶形运算（butterfly）、位反转（bit-reversal）、shared memory、twiddle factor、compute-bound |
| 46 | [BFS Shortest Path](solutions/hard/46-bfs-shortest-path.md) | level-synchronous BFS、pull-based 扩散、frontier 并行、atomicCAS |
| 53 | [Causal Self-Attention](solutions/hard/53-casual-attention.md) | causal mask、online softmax、LLM prefill、PagedAttention 对偶 |
| 56 | [Linear Self-Attention](solutions/hard/56-linear-attention.md) | linear attention、kernel trick、ELU feature map、GEMM+reduction 流水线 |
| 59 | [Sliding Window Self-Attention](solutions/hard/59-sliding-window-attn.md) | sliding window、kernel fusion |
| 73 | [All-Pairs Shortest Paths](solutions/hard/73-all-pairs-shortest-paths.md) | Floyd-Warshall、min-plus 半环矩阵乘、外串内并、shared memory 缓存第 k 行/列 |
| 74 | [GPT-2 Transformer Block](solutions/hard/74-gpt2-block.md) | Transformer、FlashAttention、LayerNorm、GEMM 端到端 |
| 93 | [Llama Transformer Block](solutions/hard/93-llama-transformer-block.md) | RMSNorm+RoPE+GQA+SwiGLU、multi-kernel pipeline、算子融合 |
| 109 | [Attention](solutions/hard/109-attention.md) | online softmax、FlashAttention、分块计算 |

> 编号对齐 `leetgpu-challenges` 仓库的 `challenges/<difficulty>/<编号>_<name>/`。其中 `#107 Argmax`、`#108 Vector Reversal`、`#109 Attention`、`#110 Scalar Multiply`、`#111 Element Reversal` 暂未收录进 `leetgpu-challenges`，编号为本仓库顺延分配。

## leetgpu-challenges 题目完成情况

下表对照 [leetgpu-challenges](https://github.com/sayaklahiri/leetgpu-challenges) 仓库 `challenges/<difficulty>/<编号>_<name>/` 的 **全部 96 道题**，标注本仓库题解完成情况：✅ 已完成 96 道 / ⬜ 未完成 0 道。已完成题目链接到本仓库题解，未完成题目链接到 LeetGPU 在线题目。


### Easy · 简单（19/19）

| # | 题目 | 状态 | 题解 / 链接 |
|---|------|:----:|------------|
| 1 | Vector Addition | ✅ | [题解](solutions/easy/1-vector-add.md) |
| 2 | Matrix Multiplication | ✅ | [题解](solutions/easy/2-matrix-multiplication.md) |
| 3 | Matrix Transpose | ✅ | [题解](solutions/easy/3-matrix-transpose.md) |
| 7 | Color Inversion | ✅ | [题解](solutions/easy/7-color-inversion.md) |
| 8 | Matrix Addition | ✅ | [题解](solutions/easy/8-matrix-addition.md) |
| 9 | 1D Convolution | ✅ | [题解](solutions/easy/9-1d-convolution.md) |
| 19 | Reverse Array | ✅ | [题解](solutions/easy/19-reverse-array.md) |
| 21 | ReLU | ✅ | [题解](solutions/easy/21-relu.md) |
| 23 | Leaky ReLU | ✅ | [题解](solutions/easy/23-leaky-relu.md) |
| 24 | Rainbow Table | ✅ | [题解](solutions/easy/24-rainbow-table.md) |
| 31 | Matrix Copy | ✅ | [题解](solutions/easy/31-matrix-copy.md) |
| 41 | Simple Inference | ✅ | [题解](solutions/easy/41-simple-inference.md) |
| 52 | Sigmoid Linear Unit | ✅ | [题解](solutions/easy/52-silu.md) |
| 54 | Swish-Gated Linear Unit | ✅ | [题解](solutions/easy/54-swiglu.md) |
| 62 | Value Clipping | ✅ | [题解](solutions/easy/62-value-clipping.md) |
| 63 | Interleave Arrays | ✅ | [题解](solutions/easy/63-interleave.md) |
| 65 | Gaussian Error Gated Linear Unit | ✅ | [题解](solutions/easy/65-geglu.md) |
| 66 | RGB to Grayscale | ✅ | [题解](solutions/easy/66-rgb-to-grayscale.md) |
| 68 | Sigmoid Activation | ✅ | [题解](solutions/easy/68-sigmoid.md) |

### Medium · 中等（64/64）

| # | 题目 | 状态 | 题解 / 链接 |
|---|------|:----:|------------|
| 4 | Reduction | ✅ | [题解](solutions/medium/4-reduction.md) |
| 5 | Softmax | ✅ | [题解](solutions/medium/5-softmax.md) |
| 6 | Softmax Attention | ✅ | [题解](solutions/medium/6-softmax-attention.md) |
| 10 | 2D Convolution | ✅ | [题解](solutions/medium/10-2d-convolution.md) |
| 11 | 3D Convolution | ✅ | [题解](solutions/medium/11-3d-convolution.md) |
| 13 | Histogramming | ✅ | [题解](solutions/medium/13-histogramming.md) |
| 16 | Prefix Sum | ✅ | [题解](solutions/medium/16-prefix-sum.md) |
| 17 | Dot Product | ✅ | [题解](solutions/medium/17-dot-product.md) |
| 18 | Sparse Matrix-Vector Multiplication | ✅ | [题解](solutions/medium/18-sparse-matrix-vector-multiplication.md) |
| 22 | General Matrix Multiplication (GEMM) | ✅ | [题解](solutions/medium/22-gemm.md) |
| 25 | Categorical Cross Entropy Loss | ✅ | [题解](solutions/medium/25-categorical-cross-entropy-loss.md) |
| 27 | Mean Squared Error | ✅ | [题解](solutions/medium/27-mean-squared-error.md) |
| 28 | Gaussian Blur | ✅ | [题解](solutions/medium/28-gaussian-blur.md) |
| 29 | Top K Selection | ✅ | [题解](solutions/medium/29-top-k-selection.md) |
| 30 | Batched Matrix Multiplication | ✅ | [题解](solutions/medium/30-batched-matrix-multiplication.md) |
| 32 | INT8 Quantized MatMul | ✅ | [题解](solutions/medium/32-int8-quantized-matmul.md) |
| 33 | Ordinary Least Squares | ✅ | [题解](solutions/medium/33-ordinary-least-squares.md) |
| 34 | Logistic Regression | ✅ | [题解](solutions/medium/34-logistic-regression.md) |
| 35 | Monte Carlo Integration | ✅ | [题解](solutions/medium/35-monte-carlo-integration.md) |
| 37 | Matrix Power | ✅ | [题解](solutions/medium/37-matrix-power.md) |
| 38 | Nearest Neighbor | ✅ | [题解](solutions/medium/38-nearest-neighbor.md) |
| 40 | Batch Normalization | ✅ | [题解](solutions/medium/40-batch-normalization.md) |
| 42 | 2D Max Pooling | ✅ | [题解](solutions/medium/42-2d-max-pooling.md) |
| 43 | Count Array Element | ✅ | [题解](solutions/medium/43-count-array-element.md) |
| 44 | Count 2D Array Element | ✅ | [题解](solutions/medium/44-count-2d-array-element.md) |
| 45 | Count 3D Array Element | ✅ | [题解](solutions/medium/45-count-3d-array-element.md) |
| 47 | Subarray Sum | ✅ | [题解](solutions/medium/47-subarray-sum.md) |
| 48 | 2D Subarray Sum | ✅ | [题解](solutions/medium/48-2d-subarray-sum.md) |
| 49 | 3D Subarray Sum | ✅ | [题解](solutions/medium/49-3d-subarray-sum.md) |
| 50 | RMS Normalization | ✅ | [题解](solutions/medium/50-rms-normalization.md) |
| 51 | Max Subarray Sum | ✅ | [题解](solutions/medium/51-max-subarray-sum.md) |
| 55 | Attention with Linear Biases | ✅ | [题解](solutions/medium/55-attn-w-linear-bias.md) |
| 57 | FP16 Batched Matrix Multiplication | ✅ | [题解](solutions/medium/57-fp16-batched-matmul.md) |
| 58 | FP16 Dot Product | ✅ | [题解](solutions/medium/58-fp16-dot-product.md) |
| 60 | Top-p Sampling | ✅ | [题解](solutions/medium/60-top-p-sampling.md) |
| 61 | Rotary Positional Embedding | ✅ | [题解](solutions/medium/61-rope-embedding.md) |
| 64 | Weight Dequantization | ✅ | [题解](solutions/medium/64-weight-dequantization.md) |
| 67 | MoE Top-K Gating | ✅ | [题解](solutions/medium/67-moe-topk-gating.md) |
| 69 | 2D Jacobi Stencil | ✅ | [题解](solutions/medium/69-jacobi-stencil-2d.md) |
| 70 | Segmented Exclusive Prefix Sum | ✅ | [题解](solutions/medium/70-segmented-prefix-sum.md) |
| 71 | Parallel Merge | ✅ | [题解](solutions/medium/71-parallel-merge.md) |
| 72 | Stream Compaction | ✅ | [题解](solutions/medium/72-stream-compaction.md) |
| 75 | Sparse Matrix-Dense Matrix Multiplication | ✅ | [题解](solutions/medium/75-sparse-matrix-dense-matrix-multiplication.md) |
| 76 | Adder Transformer Inference | ✅ | [题解](solutions/medium/76-adder-transformer.md) |
| 78 | 2D FFT | ✅ | [题解](solutions/medium/78-2d-fft.md) |
| 80 | Grouped Query Attention | ✅ | [题解](solutions/medium/80-grouped-query-attention.md) |
| 81 | INT4 Weight-Only Quantized MatMul | ✅ | [题解](solutions/medium/81-int4-matmul.md) |
| 82 | Linear Recurrence | ✅ | [题解](solutions/medium/82-linear-recurrence.md) |
| 84 | SwiGLU MLP Block | ✅ | [题解](solutions/medium/84-swiglu-mlp-block.md) |
| 85 | LoRA Linear | ✅ | [题解](solutions/medium/85-lora-linear.md) |
| 87 | Speculative Decoding Verification | ✅ | [题解](solutions/medium/87-speculative-decoding-verification.md) |
| 90 | Causal Depthwise Conv1d | ✅ | [题解](solutions/medium/90-causal-depthwise-conv1d.md) |
| 92 | Decaying Causal Attention | ✅ | [题解](solutions/medium/92-decaying-causal-attention.md) |
| 94 | SSM Selective Scan | ✅ | [题解](solutions/medium/94-ssm-selective-scan.md) |
| 96 | INT8 KV-Cache Attention | ✅ | [题解](solutions/medium/96-int8-kv-cache-attention.md) |
| 105 | Group Normalization | ✅ | [题解](solutions/medium/105-group-normalization.md) |
| 106 | Token Embedding Layer | ✅ | [题解](solutions/medium/106-token-embedding-layer.md) |
| 109 | GRPO Surrogate Loss | ✅ | [题解](solutions/medium/109-grpo-surrogate-loss.md) |
| 110 | Parallel Reverse Scan (GAE) | ✅ | [题解](solutions/medium/110-gae-reverse-scan.md) |
| 112 | Attention with Sinks | ✅ | [题解](solutions/medium/112-attention-with-sinks.md) |
| 113 | Fused QKV Projection | ✅ | [题解](solutions/medium/113-fused-qkv-projection.md) |
| 114 | GEMV | ✅ | [题解](solutions/medium/114-gemv.md) |
| 115 | Layer Normalization | ✅ | [题解](solutions/medium/115-layer-normalization.md) |
| 116 | Fused Add and RMSNorm | ✅ | [题解](solutions/medium/116-fused-add-rmsnorm.md) |

### Hard · 困难（13/13）

| # | 题目 | 状态 | 题解 / 链接 |
|---|------|:----:|------------|
| 12 | Multi-Head Attention | ✅ | [题解](solutions/hard/12-multi-head-attention.md) |
| 14 | Multi-Agent Simulation | ✅ | [题解](solutions/hard/14-multi-agent-sim.md) |
| 15 | Sorting | ✅ | [题解](solutions/hard/15-sorting.md) |
| 20 | K-Means Clustering | ✅ | [题解](solutions/hard/20-kmeans-clustering.md) |
| 36 | Radix Sort | ✅ | [题解](solutions/hard/36-radix-sort.md) |
| 39 | Fast Fourier Transform | ✅ | [题解](solutions/hard/39-fast-fourier-transform.md) |
| 46 | BFS Shortest Path | ✅ | [题解](solutions/hard/46-bfs-shortest-path.md) |
| 53 | Causal Self-Attention | ✅ | [题解](solutions/hard/53-casual-attention.md) |
| 56 | Linear Self-Attention | ✅ | [题解](solutions/hard/56-linear-attention.md) |
| 59 | Sliding Window Self-Attention | ✅ | [题解](solutions/hard/59-sliding-window-attn.md) |
| 73 | All-Pairs Shortest Paths | ✅ | [题解](solutions/hard/73-all-pairs-shortest-paths.md) |
| 74 | GPT-2 Transformer Block | ✅ | [题解](solutions/hard/74-gpt2-block.md) |
| 93 | Llama Transformer Block | ✅ | [题解](solutions/hard/93-llama-transformer-block.md) |

### 补充题解（未收录在 leetgpu-challenges）

以下 5 道题暂未收录进 `leetgpu-challenges` 仓库，编号为本仓库顺延分配（与下表 challenges 编号无对应关系）：

| # | 难度 | 题目 | 题解 |
|---|------|------|------|
| 107 | medium | [Argmax](https://leetgpu.com/challenges/argmax) | [题解](solutions/medium/107-argmax.md) |
| 108 | easy | [Vector Reversal](https://leetgpu.com/challenges/vector-reversal) | [题解](solutions/easy/108-vector-reversal.md) |
| 109 | hard | [Attention](https://leetgpu.com/challenges/attention) | [题解](solutions/hard/109-attention.md) |
| 110 | easy | [Scalar Multiply](https://leetgpu.com/challenges/scalar-multiply) | [题解](solutions/easy/110-scalar-multiply.md) |
| 111 | easy | [Element Reversal](https://leetgpu.com/challenges/element-reversal) | [题解](solutions/easy/111-element-reversal.md) |

> ⚠️ **编号冲突待修正**：本仓库此前顺延分配的 `hard #109 Attention` 与 `easy #110 Scalar Multiply`，与 `leetgpu-challenges` 新增的 `medium #109 GRPO Surrogate Loss`、`medium #110 GAE Reverse Scan` 编号冲突。上述 5 道补充题解（#107–#111）的编号需重新分配以避免与 challenges 实际编号重叠。

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
├── solutions/           # 题解，<难度>/<编号>-<name>.md（easy 22 / medium 68 / hard 15）
├── images/              # 手绘 sketch 风 SVG 插图（234 张）
├── index.md 等          # 站点页面（easy.md / medium.md / hard.md）
├── .vitepress/          # VitePress 站点配置与主题
├── SKILL.md             # 写 LeetGPU 题解的 Skill 规范
└── .github/workflows/   # GitHub Pages 自动部署
```

> 题解文件名 `<编号>-<name>.md` 的编号对齐 `leetgpu-challenges` 仓库的 `challenges/<difficulty>/<编号>_<name>/`，便于题解与原题一一对照。

## 在线网站

每次推送到 `main` 分支自动构建并部署到 GitHub Pages：

> https://hzchenxiaobin.github.io/leetgpu/

站点特性：难度导航、上一题/下一题导航、本地搜索、KaTeX 数学渲染、代码高亮（CUDA 语法）、图片点击放大、暗色模式。

## 本地预览

```bash
npm install
npm run dev           # 本地开发预览（http://localhost:5173）
npm run build         # 构建静态站点到 dist/
```

## 关联仓库

| 仓库 | 说明 |
|------|------|
| [ai-infra-notes](https://github.com/hzchenxiaobin/ai-infra-notes) | 8 周 AI Infra 学习路线主仓库（每日教程 + Profiling + Mini 引擎） |
| [LeetCode 题解](https://hzchenxiaobin.github.io/leetcode/) | 配套 LeetCode 面试题解，按周/日与教程对齐 |
| [LeetGPU](https://leetgpu.com/) | 在线 CUDA 挑战平台（题库来源） |

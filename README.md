# LeetGPU 题解

> 62 道 [LeetGPU](https://leetgpu.com/) CUDA 挑战题解 —— 每道含完整可编译 kernel + ncu profiling + 手绘 sketch 风 SVG 图解，按 CUDA 概念覆盖选题、按难度归档。

📚 **在线网站**：https://hzchenxiaobin.github.io/leetgpu/

本仓库是 [ai-infra-notes](https://github.com/hzchenxiaobin/ai-infra-notes) 8 周 AI Infra 学习路线中 LeetGPU 题解的独立归档，目录组织参考 [leetgpu-challenges](https://github.com/sayaklahiri/leetgpu-challenges) 的 `challenges/<difficulty>/<number>_<name>/` 形式，按 **easy / medium / hard** 三档难度归类，作为 Coding 任务的实战检验。

## 选题逻辑

LeetGPU 平台的题目都是 **CUDA Kernel 实现题**，选题目标是**用最少的题覆盖 GPU 编程核心概念**：

- **概念覆盖优先**：每道题对应一个 CUDA 核心概念（grid-stride、shared memory、warp shuffle、bank conflict、reduction、scan、tiling 等），避免连续多题重复
- **难度递进**：memory-bound 入门 → shared memory 进阶 → warp shuffle / tiling 高阶 → 综合题压轴
- **配合每日教程**：题解作为每日教程 Coding 任务的实战检验，选题与当日主题强相关
- **性能导向**：优先选能体现 ncu profiling 价值的题（有明确瓶颈指标可观察、可优化）

详细写作规范见 [SKILL.md](SKILL.md)。

## 题解列表

共 **63 道**（简单 18 / 中等 40 / 困难 5），覆盖 Vector Addition、GEMM、Softmax、Attention、Prefix Sum、PagedAttention、GQA、Speculative Decoding、GPT-2 Block、FlashAttention 等。

### Easy · 简单（18 道）

| # | 题目 | 核心概念 |
|---|------|----------|
| 1 | [Vector Addition](easy/1_vector_add/leetgpu-vector-addition-solution.md) | grid-stride loop、coalesced access、memory-bound |
| 2 | [Matrix Multiplication](easy/2_matrix_multiplication/leetgpu-matrix-multiplication-solution.md) | GEMM、shared memory tiling、register tiling |
| 3 | [Matrix Transpose](easy/3_matrix_transpose/leetgpu-matrix-transpose-solution.md) | shared memory tiling、bank conflict padding |
| 7 | [Color Inversion](easy/7_color_inversion/leetgpu-color-inversion-solution.md) | elementwise、image processing、`uchar4` 向量化 |
| 8 | [Matrix Addition](easy/8_matrix_addition/leetgpu-matrix-addition-solution.md) | float4 向量化、Roofline |
| 9 | [1D Convolution](easy/9_1d_convolution/leetgpu-1d-convolution-solution.md) | convolution、shared memory、halo |
| 19 | [Reverse Array](easy/19_reverse_array/leetgpu-reverse-array-solution.md) | in-place swap、1D 并行、coalesced access |
| 21 | [ReLU](easy/21_relu/leetgpu-relu-solution.md) | elementwise、warp divergence、branchless |
| 23 | [Leaky ReLU](easy/23_leaky_relu/leetgpu-leaky-relu-solution.md) | elementwise、branchless、activation |
| 31 | [Matrix Copy](easy/31_matrix_copy/leetgpu-matrix-copy-solution.md) | 内存带宽、coalesced access、float4 向量化 |
| 41 | [Simple Inference](easy/41_simple_inference/leetgpu-simple-inference-solution.md) | PyTorch、Linear、batch size、GEMM |
| 52 | [SiLU](easy/52_silu/leetgpu-silu-solution.md) | elementwise、grid-stride、`__expf` 快速数学 |
| 54 | [SwiGLU](easy/54_swiglu/leetgpu-swiglu-solution.md) | elementwise、kernel fusion、SiLU |
| 63 | [Interleave Arrays](easy/63_interleave/leetgpu-interleave-solution.md) | grid-stride loop、索引映射 |
| 68 | [Sigmoid](easy/68_sigmoid/leetgpu-sigmoid-solution.md) | elementwise、fast math `__expf`、activation |
| 108 | [Vector Reversal](easy/108_vector_reversal/leetgpu-vector-reversal-solution.md) | 索引映射、coalesced access |
| 110 | [Scalar Multiply](easy/110_scalar_multiply/leetgpu-scalar-multiply-solution.md) | element-wise、attention scaling |
| 111 | [Element Reversal](easy/111_element_reversal/leetgpu-element-reversal-solution.md) | element-wise、结果验证 |

### Medium · 中等（40 道）

| # | 题目 | 核心概念 |
|---|------|----------|
| 4 | [Reduction](medium/4_reduction/leetgpu-reduction-solution.md) | warp shuffle、`__shfl_down_sync` |
| 5 | [Softmax](medium/5_softmax/leetgpu-softmax-solution.md) | safe softmax、三遍扫描、数值稳定性 |
| 6 | [Softmax Attention](medium/6_softmax_attention/leetgpu-softmax-attention-solution.md) | fused softmax+matmul、online softmax |
| 10 | [2D Convolution](medium/10_2d_convolution/leetgpu-2d-convolution-solution.md) | shared memory halo、常量内存 |
| 13 | [Histogramming](medium/13_histogramming/leetgpu-histogramming-solution.md) | shared memory 直方图、`atomicAdd`、privatization |
| 16 | [Prefix Sum](medium/16_prefix_sum/leetgpu-prefix-sum-solution.md) | scan、warp shuffle `__shfl_up_sync`、三阶段分块 |
| 17 | [Dot Product](medium/17_dot_product/leetgpu-dot-product-solution.md) | reduction、warp shuffle、kernel 融合 |
| 18 | [Sparse Matrix-Vector Multiplication](medium/18_sparse_matrix_vector_multiplication/leetgpu-sparse-matrix-vector-multiplication-solution.md) | CSR、SpMV、warp shuffle、间接访存（gather） |
| 22 | [GEMM](medium/22_gemm/leetgpu-gemm-solution.md) | FP16、WMMA、Tensor Core、shared memory tiling |
| 27 | [Mean Squared Error](medium/27_mean_squared_error/leetgpu-mean-squared-error-solution.md) | reduction、kernel 融合、损失函数 |
| 28 | [Gaussian Blur](medium/28_gaussian_blur/leetgpu-gaussian-blur-solution.md) | same-padding 卷积、shared memory halo、可分离卷积、零填充 |
| 29 | [Top K Selection](medium/29_top_k_selection/leetgpu-top-k-selection-solution.md) | bitonic sort、堆归约、selection |
| 30 | [Batched Matrix Multiplication](medium/30_batched_matrix_multiplication/leetgpu-batched-matrix-multiplication-solution.md) | batched GEMM、tiled matmul、register blocking |
| 32 | [INT8 Quantized MatMul](medium/32_int8_quantized_matmul/leetgpu-int8-quantized-matmul-solution.md) | INT8 量化、tiled GEMM、INT32 累加、requantize |
| 38 | [Nearest Neighbor](medium/38_nearest_neighbor/leetgpu-nearest-neighbor-solution.md) | pairwise distance、shared memory tiling、argmin 归约 |
| 40 | [Batch Normalization](medium/40_batch_normalization/leetgpu-batch-normalization-solution.md) | normalization、reduction、数值稳定性 |
| 42 | [2D Max Pooling](medium/42_2d_max_pooling/leetgpu-2d-max-pooling-solution.md) | pooling、滑窗 reduction、padding 边界 |
| 43 | [Count Array Element](medium/43_count_array_element/leetgpu-count-array-element-solution.md) | reduction、`atomicAdd`、predicate、warp shuffle |
| 47 | [Subarray Sum](medium/47_subarray_sum/leetgpu-subarray-sum-solution.md) | reduction、warp shuffle、block 归约 |
| 50 | [RMS Normalization](medium/50_rms_normalization/leetgpu-rms-normalization-solution.md) | RMSNorm、warp shuffle、Llama |
| 51 | [Max Subarray Sum](medium/51_max_subarray_sum/leetgpu-max-subarray-sum-solution.md) | 滑动窗口、prefix sum、reduction |
| 55 | [Attention with Linear Biases (ALiBi)](medium/55_attn_w_linear_bias/leetgpu-attn-w-linear-bias-solution.md) | ALiBi、positional bias、online softmax |
| 58 | [FP16 Dot Product](medium/58_fp16_dot_product/leetgpu-fp16-dot-product-solution.md) | half 精度、warp shuffle、FP32 累加 |
| 61 | [Rotary Positional Embedding](medium/61_rope_embedding/leetgpu-rope-embedding-solution.md) | elementwise、rotate_half、位置编码 |
| 64 | [Weight Dequantization](medium/64_weight_dequantization/leetgpu-weight-dequantization-solution.md) | element-wise、分块 scale、量化推理 |
| 67 | [MoE Top-K Gating](medium/67_moe_topk_gating/leetgpu-moe-topk-gating-solution.md) | top-k 选择、并行归约、softmax、MoE 路由 |
| 69 | [2D Jacobi Stencil](medium/69_jacobi_stencil_2d/leetgpu-2d-jacobi-stencil-solution.md) | stencil 计算、shared memory halo、Jacobi 迭代 |
| 70 | [Segmented Prefix Sum](medium/70_segmented_prefix_sum/leetgpu-segmented-prefix-sum-solution.md) | segmented scan、warp shuffle |
| 72 | [Stream Compaction](medium/72_stream_compaction/leetgpu-stream-compaction-solution.md) | scan、predicate、stream compaction |
| 76 | [Adder Transformer Inference](medium/76_adder_transformer/leetgpu-adder-transformer-solution.md) | 多 kernel 流水线、autoregressive 推理、RoPE |
| 80 | [Grouped Query Attention (GQA)](medium/80_grouped_query_attention/leetgpu-grouped-query-attention-solution.md) | GQA、KV head 共享、LLM 推理 |
| 84 | [SwiGLU MLP Block](medium/84_swiglu_mlp_block/leetgpu-swiglu-mlp-block-solution.md) | SwiGLU、MLP、GEMM、kernel fusion、LLaMA |
| 85 | [LoRA Linear](medium/85_lora_linear/leetgpu-lora-linear-solution.md) | Low-Rank Adaptation、参数高效微调 |
| 87 | [Speculative Decoding Verification](medium/87_speculative_decoding_verification/leetgpu-speculative-decoding-verification-solution.md) | 投机解码、accept/reject 采样、CDF 查找 |
| 90 | [Causal Depthwise Conv1d](medium/90_causal_depthwise_conv1d/leetgpu-causal-depthwise-conv1d-solution.md) | causal、depthwise、边界处理 |
| 92 | [Decaying Causal Attention](medium/92_decaying_causal_attention/leetgpu-decaying-causal-attention-solution.md) | causal mask、exponential decay、增量计算 |
| 96 | [INT8 KV-Cache Attention](medium/96_int8_kv_cache_attention/leetgpu-int8-kv-cache-attention-solution.md) | decode-phase、KV Cache、INT8 量化、per-token scale |
| 105 | [Group Normalization](medium/105_group_normalization/leetgpu-group-normalization-solution.md) | normalization、reduction、GroupNorm |
| 106 | [Token Embedding Layer](medium/106_token_embedding_layer/leetgpu-token-embedding-layer-solution.md) | embedding、gather、LayerNorm、融合 kernel |
| 107 | [Argmax](medium/107_argmax/leetgpu-argmax-solution.md) | 归约、argmax、`__shfl_down_sync` |

### Hard · 困难（5 道）

| # | 题目 | 核心概念 |
|---|------|----------|
| 12 | [Multi-Head Attention](hard/12_multi_head_attention/leetgpu-multi-head-attention-solution.md) | MHA、FlashAttention、融合 attention |
| 53 | [Causal Self-Attention](hard/53_casual_attention/leetgpu-causal-self-attention-solution.md) | causal mask、online softmax、LLM prefill、PagedAttention 对偶 |
| 59 | [Sliding Window Self-Attention](hard/59_sliding_window_attn/leetgpu-sliding-window-self-attention-solution.md) | sliding window、kernel fusion |
| 74 | [GPT-2 Transformer Block](hard/74_gpt2_block/leetgpu-gpt-2-transformer-block-solution.md) | Transformer、FlashAttention、LayerNorm、GEMM 端到端 |
| 109 | [Attention](hard/109_attention/leetgpu-attention-solution.md) | online softmax、FlashAttention、分块计算 |

> 编号与目录名对应 `leetgpu-challenges` 仓库的 `challenges/<difficulty>/<编号>_<name>/`。其中 `#107 Argmax`、`#108 Vector Reversal`、`#109 Attention`、`#110 Scalar Multiply`、`#111 Element Reversal` 暂未收录进 `leetgpu-challenges`，编号为本仓库顺延分配。

## leetgpu-challenges 题目完成情况

下表对照 [leetgpu-challenges](https://github.com/sayaklahiri/leetgpu-challenges) 仓库 `challenges/<difficulty>/<编号>_<name>/` 的 **全部 91 道题**，标注本仓库题解完成情况：✅ 已完成 57 道 / ⬜ 未完成 34 道。已完成题目链接到本仓库题解，未完成题目链接到 LeetGPU 在线题目。


### Easy · 简单（15/19）

| # | 题目 | 状态 | 题解 / 链接 |
|---|------|:----:|------------|
| 1 | Vector Addition | ✅ | [题解](easy/1_vector_add/leetgpu-vector-addition-solution.md) |
| 2 | Matrix Multiplication | ✅ | [题解](easy/2_matrix_multiplication/leetgpu-matrix-multiplication-solution.md) |
| 3 | Matrix Transpose | ✅ | [题解](easy/3_matrix_transpose/leetgpu-matrix-transpose-solution.md) |
| 7 | Color Inversion | ✅ | [题解](easy/7_color_inversion/leetgpu-color-inversion-solution.md) |
| 8 | Matrix Addition | ✅ | [题解](easy/8_matrix_addition/leetgpu-matrix-addition-solution.md) |
| 9 | 1D Convolution | ✅ | [题解](easy/9_1d_convolution/leetgpu-1d-convolution-solution.md) |
| 19 | Reverse Array | ✅ | [题解](easy/19_reverse_array/leetgpu-reverse-array-solution.md) |
| 21 | ReLU | ✅ | [题解](easy/21_relu/leetgpu-relu-solution.md) |
| 23 | Leaky ReLU | ✅ | [题解](easy/23_leaky_relu/leetgpu-leaky-relu-solution.md) |
| 24 | Rainbow Table | ⬜ | [题目](https://leetgpu.com/challenges/rainbow-table) |
| 31 | Matrix Copy | ✅ | [题解](easy/31_matrix_copy/leetgpu-matrix-copy-solution.md) |
| 41 | Simple Inference | ✅ | [题解](easy/41_simple_inference/leetgpu-simple-inference-solution.md) |
| 52 | Sigmoid Linear Unit | ✅ | [题解](easy/52_silu/leetgpu-silu-solution.md) |
| 54 | Swish-Gated Linear Unit | ✅ | [题解](easy/54_swiglu/leetgpu-swiglu-solution.md) |
| 62 | Value Clipping | ⬜ | [题目](https://leetgpu.com/challenges/value-clipping) |
| 63 | Interleave Arrays | ✅ | [题解](easy/63_interleave/leetgpu-interleave-solution.md) |
| 65 | Gaussian Error Gated Linear Unit | ⬜ | [题目](https://leetgpu.com/challenges/geglu) |
| 66 | RGB to Grayscale | ⬜ | [题目](https://leetgpu.com/challenges/rgb-to-grayscale) |
| 68 | Sigmoid Activation | ✅ | [题解](easy/68_sigmoid/leetgpu-sigmoid-solution.md) |

### Medium · 中等（45/59）

| # | 题目 | 状态 | 题解 / 链接 |
|---|------|:----:|------------|
| 4 | Reduction | ✅ | [题解](medium/4_reduction/leetgpu-reduction-solution.md) |
| 5 | Softmax | ✅ | [题解](medium/5_softmax/leetgpu-softmax-solution.md) |
| 6 | Softmax Attention | ✅ | [题解](medium/6_softmax_attention/leetgpu-softmax-attention-solution.md) |
| 10 | 2D Convolution | ✅ | [题解](medium/10_2d_convolution/leetgpu-2d-convolution-solution.md) |
| 11 | 3D Convolution | ⬜ | [题目](https://leetgpu.com/challenges/3d-convolution) |
| 13 | Histogramming | ✅ | [题解](medium/13_histogramming/leetgpu-histogramming-solution.md) |
| 16 | Prefix Sum | ✅ | [题解](medium/16_prefix_sum/leetgpu-prefix-sum-solution.md) |
| 17 | Dot Product | ✅ | [题解](medium/17_dot_product/leetgpu-dot-product-solution.md) |
| 18 | Sparse Matrix-Vector Multiplication | ✅ | [题解](medium/18_sparse_matrix_vector_multiplication/leetgpu-sparse-matrix-vector-multiplication-solution.md) |
| 22 | General Matrix Multiplication (GEMM) | ✅ | [题解](medium/22_gemm/leetgpu-gemm-solution.md) |
| 25 | Categorical Cross Entropy Loss | ✅ | [题解](medium/25_categorical_cross_entropy_loss/leetgpu-categorical-cross-entropy-loss-solution.md) |
| 27 | Mean Squared Error | ✅ | [题解](medium/27_mean_squared_error/leetgpu-mean-squared-error-solution.md) |
| 28 | Gaussian Blur | ✅ | [题解](medium/28_gaussian_blur/leetgpu-gaussian-blur-solution.md) |
| 29 | Top K Selection | ✅ | [题解](medium/29_top_k_selection/leetgpu-top-k-selection-solution.md) |
| 30 | Batched Matrix Multiplication | ✅ | [题解](medium/30_batched_matrix_multiplication/leetgpu-batched-matrix-multiplication-solution.md) |
| 32 | INT8 Quantized MatMul | ✅ | [题解](medium/32_int8_quantized_matmul/leetgpu-int8-quantized-matmul-solution.md) |
| 33 | Ordinary Least Squares | ⬜ | [题目](https://leetgpu.com/challenges/ordinary-least-squares) |
| 34 | Logistic Regression | ⬜ | [题目](https://leetgpu.com/challenges/logistic-regression) |
| 35 | Monte Carlo Integration | ⬜ | [题目](https://leetgpu.com/challenges/monte-carlo-integration) |
| 37 | Matrix Power | ⬜ | [题目](https://leetgpu.com/challenges/matrix-power) |
| 38 | Nearest Neighbor | ✅ | [题解](medium/38_nearest_neighbor/leetgpu-nearest-neighbor-solution.md) |
| 40 | Batch Normalization | ✅ | [题解](medium/40_batch_normalization/leetgpu-batch-normalization-solution.md) |
| 42 | 2D Max Pooling | ✅ | [题解](medium/42_2d_max_pooling/leetgpu-2d-max-pooling-solution.md) |
| 43 | Count Array Element | ✅ | [题解](medium/43_count_array_element/leetgpu-count-array-element-solution.md) |
| 44 | Count 2D Array Element | ⬜ | [题目](https://leetgpu.com/challenges/count-2d-array-element) |
| 45 | Count 3D Array Element | ⬜ | [题目](https://leetgpu.com/challenges/count-3d-array-element) |
| 47 | Subarray Sum | ✅ | [题解](medium/47_subarray_sum/leetgpu-subarray-sum-solution.md) |
| 48 | 2D Subarray Sum | ⬜ | [题目](https://leetgpu.com/challenges/2d-subarray-sum) |
| 49 | 3D Subarray Sum | ⬜ | [题目](https://leetgpu.com/challenges/3d-subarray-sum) |
| 50 | RMS Normalization | ✅ | [题解](medium/50_rms_normalization/leetgpu-rms-normalization-solution.md) |
| 51 | Max Subarray Sum | ✅ | [题解](medium/51_max_subarray_sum/leetgpu-max-subarray-sum-solution.md) |
| 55 | Attention with Linear Biases | ✅ | [题解](medium/55_attn_w_linear_bias/leetgpu-attn-w-linear-bias-solution.md) |
| 57 | FP16 Batched Matrix Multiplication | ⬜ | [题目](https://leetgpu.com/challenges/fp16-batched-matmul) |
| 58 | FP16 Dot Product | ✅ | [题解](medium/58_fp16_dot_product/leetgpu-fp16-dot-product-solution.md) |
| 60 | Top-p Sampling | ✅ | [题解](medium/60_top_p_sampling/leetgpu-top-p-sampling-solution.md) |
| 61 | Rotary Positional Embedding | ✅ | [题解](medium/61_rope_embedding/leetgpu-rope-embedding-solution.md) |
| 64 | Weight Dequantization | ✅ | [题解](medium/64_weight_dequantization/leetgpu-weight-dequantization-solution.md) |
| 67 | MoE Top-K Gating | ✅ | [题解](medium/67_moe_topk_gating/leetgpu-moe-topk-gating-solution.md) |
| 69 | 2D Jacobi Stencil | ✅ | [题解](medium/69_jacobi_stencil_2d/leetgpu-2d-jacobi-stencil-solution.md) |
| 70 | Segmented Exclusive Prefix Sum | ✅ | [题解](medium/70_segmented_prefix_sum/leetgpu-segmented-prefix-sum-solution.md) |
| 71 | Parallel Merge | ✅ | [题解](medium/71_parallel_merge/leetgpu-parallel-merge-solution.md) |
| 72 | Stream Compaction | ✅ | [题解](medium/72_stream_compaction/leetgpu-stream-compaction-solution.md) |
| 75 | Sparse Matrix-Dense Matrix Multiplication | ✅ | [题解](medium/75_sparse_matrix_dense_matrix_multiplication/leetgpu-sparse-matrix-dense-matrix-multiplication-solution.md) |
| 76 | Adder Transformer Inference | ✅ | [题解](medium/76_adder_transformer/leetgpu-adder-transformer-solution.md) |
| 78 | 2D FFT | ⬜ | [题目](https://leetgpu.com/challenges/2d-fft) |
| 80 | Grouped Query Attention | ✅ | [题解](medium/80_grouped_query_attention/leetgpu-grouped-query-attention-solution.md) |
| 81 | INT4 Weight-Only Quantized MatMul | ✅ | [题解](medium/81_int4_matmul/leetgpu-int4-matmul-solution.md) |
| 82 | Linear Recurrence | ✅ | [题解](medium/82_linear_recurrence/leetgpu-linear-recurrence-solution.md) |
| 84 | SwiGLU MLP Block | ✅ | [题解](medium/84_swiglu_mlp_block/leetgpu-swiglu-mlp-block-solution.md) |
| 85 | LoRA Linear | ✅ | [题解](medium/85_lora_linear/leetgpu-lora-linear-solution.md) |
| 87 | Speculative Decoding Verification | ✅ | [题解](medium/87_speculative_decoding_verification/leetgpu-speculative-decoding-verification-solution.md) |
| 90 | Causal Depthwise Conv1d | ✅ | [题解](medium/90_causal_depthwise_conv1d/leetgpu-causal-depthwise-conv1d-solution.md) |
| 92 | Decaying Causal Attention | ✅ | [题解](medium/92_decaying_causal_attention/leetgpu-decaying-causal-attention-solution.md) |
| 94 | SSM Selective Scan | ⬜ | [题目](https://leetgpu.com/challenges/ssm-selective-scan) |
| 96 | INT8 KV-Cache Attention | ✅ | [题解](medium/96_int8_kv_cache_attention/leetgpu-int8-kv-cache-attention-solution.md) |
| 105 | Group Normalization | ✅ | [题解](medium/105_group_normalization/leetgpu-group-normalization-solution.md) |
| 106 | Token Embedding Layer | ✅ | [题解](medium/106_token_embedding_layer/leetgpu-token-embedding-layer-solution.md) |
| 109 | GRPO Surrogate Loss | ⬜ | [题目](https://leetgpu.com/challenges/grpo-surrogate-loss) |
| 110 | Parallel Reverse Scan (GAE) | ⬜ | [题目](https://leetgpu.com/challenges/gae-reverse-scan) |

### Hard · 困难（4/13）

| # | 题目 | 状态 | 题解 / 链接 |
|---|------|:----:|------------|
| 12 | Multi-Head Attention | ✅ | [题解](hard/12_multi_head_attention/leetgpu-multi-head-attention-solution.md) |
| 14 | Multi-Agent Simulation | ⬜ | [题目](https://leetgpu.com/challenges/multi-agent-simulation) |
| 15 | Sorting | ⬜ | [题目](https://leetgpu.com/challenges/sorting) |
| 20 | K-Means Clustering | ⬜ | [题目](https://leetgpu.com/challenges/kmeans-clustering) |
| 36 | Radix Sort | ⬜ | [题目](https://leetgpu.com/challenges/radix-sort) |
| 39 | Fast Fourier Transform | ⬜ | [题目](https://leetgpu.com/challenges/fast-fourier-transform) |
| 46 | BFS Shortest Path | ⬜ | [题目](https://leetgpu.com/challenges/bfs-shortest-path) |
| 53 | Causal Self-Attention | ✅ | [题解](hard/53_casual_attention/leetgpu-causal-self-attention-solution.md) |
| 56 | Linear Self-Attention | ⬜ | [题目](https://leetgpu.com/challenges/linear-self-attention) |
| 59 | Sliding Window Self-Attention | ✅ | [题解](hard/59_sliding_window_attn/leetgpu-sliding-window-self-attention-solution.md) |
| 73 | All-Pairs Shortest Paths | ⬜ | [题目](https://leetgpu.com/challenges/all-pairs-shortest-paths) |
| 74 | GPT-2 Transformer Block | ✅ | [题解](hard/74_gpt2_block/leetgpu-gpt-2-transformer-block-solution.md) |
| 93 | Llama Transformer Block | ⬜ | [题目](https://leetgpu.com/challenges/llama-transformer-block) |

### 补充题解（未收录在 leetgpu-challenges）

以下 5 道题暂未收录进 `leetgpu-challenges` 仓库，编号为本仓库顺延分配（与下表 challenges 编号无对应关系）：

| # | 难度 | 题目 | 题解 |
|---|------|------|------|
| 107 | medium | [Argmax](https://leetgpu.com/challenges/argmax) | [题解](medium/107_argmax/leetgpu-argmax-solution.md) |
| 108 | easy | [Vector Reversal](https://leetgpu.com/challenges/vector-reversal) | [题解](easy/108_vector_reversal/leetgpu-vector-reversal-solution.md) |
| 109 | hard | [Attention](https://leetgpu.com/challenges/attention) | [题解](hard/109_attention/leetgpu-attention-solution.md) |
| 110 | easy | [Scalar Multiply](https://leetgpu.com/challenges/scalar-multiply) | [题解](easy/110_scalar_multiply/leetgpu-scalar-multiply-solution.md) |
| 111 | easy | [Element Reversal](https://leetgpu.com/challenges/element-reversal) | [题解](easy/111_element_reversal/leetgpu-element-reversal-solution.md) |

> ⚠️ **编号冲突待修正**：本仓库此前顺延分配的 `hard/109_attention` 与 `easy/110_scalar_multiply`，与 `leetgpu-challenges` 新增的 `medium/109_grpo_surrogate_loss`、`medium/110_gae_reverse_scan` 编号冲突。上述 5 道补充题解（#107–#111）的编号需重新分配以避免与 challenges 实际编号重叠。

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
├── easy/                # 简单题解，<编号>_<name>/leetgpu-<slug>-solution.md
├── medium/              # 中等题解
├── hard/                # 困难题解
│   └── 74_gpt2_block/
│       └── leetgpu-gpt-2-transformer-block-solution.md
├── images/             # 手绘 sketch 风 SVG 插图（118 张）
├── SKILL.md            # 写 LeetGPU 题解的 Skill 规范
├── build.py            # 网站构建入口
├── build/              # 构建系统（common.py + leetgpu.py）
├── static/             # 网站静态资源（css + js）
└── .github/workflows/  # GitHub Pages 自动部署
```

> 目录命名 `<编号>_<name>/` 对齐 `leetgpu-challenges` 仓库的 `challenges/<difficulty>/<编号>_<name>/`，便于题解与原题一一对照。

## 在线网站

每次推送到 `main` 分支自动构建并部署到 GitHub Pages：

> https://hzchenxiaobin.github.io/leetgpu/

站点特性：侧边栏难度手风琴导航、随机选题按钮、KaTeX 数学渲染、Prism 代码高亮（含 CUDA 语法扩展）。

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

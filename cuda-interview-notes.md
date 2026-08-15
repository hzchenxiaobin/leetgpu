# CUDA 手撕题专题：AI Infra 面经总结

> **来源**：知乎、牛客网等平台的 AI Infra 公开面经（链接见文末参考资料），检索整理时间 2026-07
> **适用对象**：准备 AI Infra / 推理引擎 / 高性能计算方向岗位、需要手撕 CUDA kernel 的求职者
> **说明**：知乎页面有反爬保护，部分内容基于搜索摘要整理；小红书正文需登录，内容来自转载与面经汇总。细节请点原文链接核对
> **相关专题**：[AI Infra 面经与面试题整理](https://hzchenxiaobin.github.io/ai-infra-notes/interview/)（面试形式与八股）、[Week 1 CUDA 基础](https://hzchenxiaobin.github.io/ai-infra-notes/week1/)（CUDA 入门教程）

---

## 一、考察形式

- 面试一般**不提供 CUDA 运行环境**，也不要求完整可运行代码，通常只写 kernel 函数 + `block_size` / `grid_size` + launch 调用（[牛客：CUDA算子手撕与面试](https://www.nowcoder.com/discuss/697901950464954368)）
- 不局限于 CUDA，Triton / CuTe 也可以写，但直接写 CUDA 是加分项；推荐去 LeetGPU 刷题练习（[知乎：AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231)）
- 少数公司要求**结果与 CPU 版本对齐**（如某大模型公司的 softmax 3-pass 写法，[牛客：模型部署/推理优化社招面经](https://www.nowcoder.com/discuss/599177965083054080)）
- 形式多为共享屏幕、纯文本编辑器现场写（[牛客：百度 AI Infra 一面](https://www.nowcoder.com/discuss/875003802187792384)）

## 二、高频题（几乎必考）

### 1. Softmax —— 出现频率最高

- 要点：减最大值防溢出（safe softmax）、warp shuffle 归约
- 一维数组和 M×N 矩阵**按行 softmax** 都要会
- 变体：快手考过 "M×K 在 K 方向做 Softmax2D，要求避免爆精度"（[知乎：2025 春招实习面经汇总](https://zhuanlan.zhihu.com/p/1896206045161952147)）
- 进阶：online softmax（FlashAttention 的分块递推形式）

### 2. Reduce（sum / max）

优化链路经常被追问，要能说清每一步的收益：

1. naive：`atomicAdd` 全局归约（线程串行化，性能差）
2. shared memory 折半归约（需 `__syncthreads()`）
3. warp shuffle（`__shfl_down_sync` / `__shfl_xor_sync`，warp 内免同步）
4. 加 float4 向量化访存

### 3. LayerNorm / RMSNorm

- 本质是"每行求均值方差 + 归约"，是 reduce 的直接延伸
- 26 秋招面经："手写 RMSNorm CUDA Kernel"（[知乎：AI infra 26秋招面经](https://zhuanlan.zhihu.com/p/2017740483217081305)）
- 变形考法：要求用 SIMD 向量指令（vadd/vsub/vmul/vdiv）写 LayerNorm，不提供 sqrt，需自己牛顿迭代（社招面经）

## 三、中频题

### 1. SGEMM（矩阵乘）

- 层级：naive → block tile（shared memory）→ thread tile（寄存器分块）
- 常见 follow-up：**Split-K**、float4 向量化、双缓冲
- 面试官能一眼看出你是背的还是理解的，背熟 block tile 的 index 会被快速跳到下一题（知乎 AI infra 面试经验贴）
- 美团北斗考过"GEMM base 版本 + 讲优化方法"（[美团北斗 AI Infra 校招面经](http://ningzhengsheng.cn/2026/04/16/%E9%9D%A2%E8%AF%95%E5%AE%9D%E5%85%B8/AI%20Infra/AIInfra%E9%9D%A2%E7%BB%8F/%E7%BE%8E%E5%9B%A2_%E5%8C%97%E6%96%97_AI_Infra_%E6%A0%A1%E6%8B%9B/)）

### 2. 矩阵转置 transpose

- 考点：全局内存合并访存（读写不能同时合并时优先合并写入）、shared memory 中转、padding 解决 bank conflict

### 3. GEMV（矩阵乘向量）

- 一个 warp 负责一行，可拓展到"二维矩阵按行归约"这类变形题

### 4. FlashAttention / online softmax

- 推理岗越来越常考，至少能手写 online softmax 的分块递推

### 5. Scan（前缀和）

- 知乎面经中标注出现两次

### 6. Top-K

- 堆 / 部分排序；变形：Top-P 采样、MoE Top-K 路由

### 7. Histogram

- shared memory 私有化（privatization）+ 原子操作合并，考察 atomic 冲突优化

## 四、低频但出现过

| 题目 | 说明 |
|------|------|
| elementwise（vector add / relu / sigmoid） | 百度一面考过 vector add；追问 float4 向量化（注意是 **grid 除 4** 而不是 block 除 4，否则降低 occupancy） |
| avg pooling、bbox IoU | CUDA 实现（CV 部署岗） |
| NMS、conv2d、双线性插值 | 不好用 CUDA 写，要求 C++ 实现 |
| dot product | reduce 的直接应用 |
| 量化 / 反量化 kernel | 推理优化岗 |
| RoPE | 大模型算子岗 |

（参考：[GitHub：CUDA-Learn-Note](https://github.com/hypertseng/CUDA-Learn-Note) 的大模型手撕 CUDA 清单）

## 五、LeetGPU 题目对照

对照 [leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) 题目目录（编号即 LeetGPU 题目编号），本专题各题在 LeetGPU 上的对应关系如下。刷题时可直接对照[本站题解列表](https://hzchenxiaobin.github.io/leetgpu/)。

### 高频题

| 本专题题目 | LeetGPU 对应题 |
|------------|----------------|
| Softmax | [#5 Softmax](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-solution.html)（medium） |
| online softmax | 无独立题，最接近 [#6 Softmax Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-attention-solution.html) |
| Reduce（sum/max） | [#4 Reduction](https://hzchenxiaobin.github.io/leetgpu/leetgpu-reduction-solution.html)（medium，求和归约） |
| LayerNorm | [#115 Layer Normalization](https://hzchenxiaobin.github.io/leetgpu/leetgpu-layer-normalization-solution.html)（medium）；同类 [#40 Batch Normalization](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batch-normalization-solution.html)、[#105 Group Normalization](https://hzchenxiaobin.github.io/leetgpu/leetgpu-group-normalization-solution.html) |
| RMSNorm | [#50 RMS Normalization](https://hzchenxiaobin.github.io/leetgpu/leetgpu-rms-normalization-solution.html)（medium）、[#116 Fused Add RMSNorm](https://hzchenxiaobin.github.io/leetgpu/leetgpu-fused-add-rmsnorm-solution.html)（融合残差加 + RMSNorm） |

### 中频题

| 本专题题目 | LeetGPU 对应题 |
|------------|----------------|
| SGEMM | [#2 Matrix Multiplication](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-multiplication-solution.html)（easy，含 TF32 Tensor Core 版）、[#22 GEMM](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gemm-solution.html)（medium，带 alpha/beta）、[#30 Batched MatMul](https://hzchenxiaobin.github.io/leetgpu/leetgpu-batched-matrix-multiplication-solution.html)、[#57 FP16 Batched MatMul](https://hzchenxiaobin.github.io/leetgpu/leetgpu-fp16-batched-matmul-solution.html)；量化路径 [#32 INT8 Quantized MatMul](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int8-quantized-matmul-solution.html)、[#81 INT4 MatMul](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int4-matmul-solution.html)；Split-K 无直接对应 |
| 矩阵转置 | [#3 Matrix Transpose](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-transpose-solution.html)（easy） |
| GEMV | [#114 GEMV](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gemv-solution.html)（medium）；同类 [#17 Dot Product](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dot-product-solution.html)、[#18 Sparse Matrix-Vector Multiplication](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sparse-matrix-vector-multiplication-solution.html)（SpMV）、[#75 Sparse Matrix-Dense Matrix Multiplication](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sparse-matrix-dense-matrix-multiplication-solution.html) |
| FlashAttention / attention | [#6 Softmax Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-attention-solution.html)、[#109 Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-attention-solution.html)（hard）、[#53 Causal Self-Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-causal-self-attention-solution.html)（hard）、[#12 Multi-Head Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-multi-head-attention-solution.html)（hard）、[#26 Multi-Head Cross Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-multi-head-cross-attention-solution.html)、[#80 Grouped Query Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-grouped-query-attention-solution.html)、[#59 Sliding Window Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sliding-window-self-attention-solution.html)、[#56 Linear Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-linear-self-attention-solution.html)、[#112 Attention with Sinks](https://hzchenxiaobin.github.io/leetgpu/leetgpu-attention-with-sinks-solution.html)、[#111 Softmax Attention Backward](https://hzchenxiaobin.github.io/leetgpu/leetgpu-softmax-attention-backward-solution.html)（反向传播） |
| Scan（前缀和） | [#16 Prefix Sum](https://hzchenxiaobin.github.io/leetgpu/leetgpu-prefix-sum-solution.html)（medium）、[#70 Segmented Prefix Sum](https://hzchenxiaobin.github.io/leetgpu/leetgpu-segmented-prefix-sum-solution.html) |
| Top-K | [#29 Top-K Selection](https://hzchenxiaobin.github.io/leetgpu/leetgpu-top-k-selection-solution.html)（medium）、[#60 Top-P Sampling](https://hzchenxiaobin.github.io/leetgpu/leetgpu-top-p-sampling-solution.html)、[#67 MoE Top-K Gating](https://hzchenxiaobin.github.io/leetgpu/leetgpu-moe-topk-gating-solution.html) |
| Histogram | [#13 Histogramming](https://hzchenxiaobin.github.io/leetgpu/leetgpu-histogramming-solution.html)（medium） |

### 低频题

| 本专题题目 | LeetGPU 对应题 |
|------------|----------------|
| vector add | [#1 Vector Addition](https://hzchenxiaobin.github.io/leetgpu/leetgpu-vector-addition-solution.html)（easy） |
| relu / sigmoid | [#21 ReLU](https://hzchenxiaobin.github.io/leetgpu/leetgpu-relu-solution.html)、[#23 Leaky ReLU](https://hzchenxiaobin.github.io/leetgpu/leetgpu-leaky-relu-solution.html)、[#68 Sigmoid](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sigmoid-solution.html)；同类还有 [#52 SiLU](https://hzchenxiaobin.github.io/leetgpu/leetgpu-silu-solution.html)、[#54 SwiGLU](https://hzchenxiaobin.github.io/leetgpu/leetgpu-swiglu-solution.html)、[#65 GeGLU](https://hzchenxiaobin.github.io/leetgpu/leetgpu-geglu-solution.html) |
| avg pooling | 无 avg pooling 题；只有 [#42 2D Max Pooling](https://hzchenxiaobin.github.io/leetgpu/leetgpu-2d-max-pooling-solution.html) |
| bbox IoU | **无对应题** |
| NMS | **无对应题** |
| conv2d | [#10 2D Convolution](https://hzchenxiaobin.github.io/leetgpu/leetgpu-2d-convolution-solution.html)（medium）；另有 [#9 1D Convolution](https://hzchenxiaobin.github.io/leetgpu/leetgpu-1d-convolution-solution.html)、[#11 3D Convolution](https://hzchenxiaobin.github.io/leetgpu/leetgpu-3d-convolution-solution.html) |
| 双线性插值 | **无对应题**（图像类仅有 [#28 Gaussian Blur](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gaussian-blur-solution.html)、[#66 RGB to Grayscale](https://hzchenxiaobin.github.io/leetgpu/leetgpu-rgb-to-grayscale-solution.html)） |
| dot product | [#17 Dot Product](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dot-product-solution.html)、[#58 FP16 Dot Product](https://hzchenxiaobin.github.io/leetgpu/leetgpu-fp16-dot-product-solution.html) |
| 量化 / 反量化 kernel | [#64 Weight Dequantization](https://hzchenxiaobin.github.io/leetgpu/leetgpu-weight-dequantization-solution.html)、[#32 INT8 Quantized MatMul](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int8-quantized-matmul-solution.html)、[#81 INT4 MatMul](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int4-matmul-solution.html)、[#96 INT8 KV-Cache Attention](https://hzchenxiaobin.github.io/leetgpu/leetgpu-int8-kv-cache-attention-solution.html) |
| RoPE | [#61 RoPE Embedding](https://hzchenxiaobin.github.io/leetgpu/leetgpu-rope-embedding-solution.html)（medium）；另有 [#55 Attention with Linear Bias](https://hzchenxiaobin.github.io/leetgpu/leetgpu-attn-w-linear-bias-solution.html)（ALiBi） |
| Argmax | [#107 Argmax](https://hzchenxiaobin.github.io/leetgpu/leetgpu-argmax-solution.html)（medium） |
| 排序 / 选择 | [#15 Sorting](https://hzchenxiaobin.github.io/leetgpu/leetgpu-sorting-solution.html)（hard）、[#36 Radix Sort](https://hzchenxiaobin.github.io/leetgpu/leetgpu-radix-sort-solution.html)、[#71 Parallel Merge](https://hzchenxiaobin.github.io/leetgpu/leetgpu-parallel-merge-solution.html)、[#72 Stream Compaction](https://hzchenxiaobin.github.io/leetgpu/leetgpu-stream-compaction-solution.html)（filter） |
| 损失函数 | [#25 Categorical Cross Entropy](https://hzchenxiaobin.github.io/leetgpu/leetgpu-categorical-cross-entropy-loss-solution.html)、[#27 Mean Squared Error](https://hzchenxiaobin.github.io/leetgpu/leetgpu-mean-squared-error-solution.html) |

### 大模型推理与训练方向

以下题目在 LeetGPU 上已有题解，覆盖当前大模型推理/训练岗的高频考点，刷题时建议按方向归类练习。

| 方向 | LeetGPU 对应题 |
|------|----------------|
| Transformer block | [#74 GPT-2 Block](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gpt-2-transformer-block-solution.html)（hard）、[#93 Llama Transformer Block](https://hzchenxiaobin.github.io/leetgpu/leetgpu-llama-transformer-block-solution.html)（hard）、[#76 Adder Transformer](https://hzchenxiaobin.github.io/leetgpu/leetgpu-adder-transformer-solution.html) |
| 算子融合 | [#113 Fused QKV Projection](https://hzchenxiaobin.github.io/leetgpu/leetgpu-fused-qkv-projection-solution.html)、[#116 Fused Add RMSNorm](https://hzchenxiaobin.github.io/leetgpu/leetgpu-fused-add-rmsnorm-solution.html)、[#84 SwiGLU MLP Block](https://hzchenxiaobin.github.io/leetgpu/leetgpu-swiglu-mlp-block-solution.html)、[#85 LoRA Linear](https://hzchenxiaobin.github.io/leetgpu/leetgpu-lora-linear-solution.html) |
| SSM / Mamba | [#94 SSM Selective Scan](https://hzchenxiaobin.github.io/leetgpu/leetgpu-ssm-selective-scan-solution.html)、[#82 Linear Recurrence](https://hzchenxiaobin.github.io/leetgpu/leetgpu-linear-recurrence-solution.html) |
| RLHF / RL 损失 | [#107 PPO Clipped Surrogate Loss](https://hzchenxiaobin.github.io/leetgpu/leetgpu-ppo-clipped-surrogate-loss-solution.html)、[#108 DPO Sequence Loss](https://hzchenxiaobin.github.io/leetgpu/leetgpu-dpo-sequence-loss-solution.html)、[#109 GRPO Surrogate Loss](https://hzchenxiaobin.github.io/leetgpu/leetgpu-grpo-surrogate-loss-solution.html)、[#110 GAE Reverse Scan](https://hzchenxiaobin.github.io/leetgpu/leetgpu-gae-reverse-scan-solution.html) |
| 推测解码 | [#87 Speculative Decoding Verification](https://hzchenxiaobin.github.io/leetgpu/leetgpu-speculative-decoding-verification-solution.html) |
| Embedding | [#106 Token Embedding Layer](https://hzchenxiaobin.github.io/leetgpu/leetgpu-token-embedding-layer-solution.html) |
| 因果卷积 | [#90 Causal Depthwise Conv1d](https://hzchenxiaobin.github.io/leetgpu/leetgpu-causal-depthwise-conv1d-solution.html) |
| FFT | [#39 Fast Fourier Transform](https://hzchenxiaobin.github.io/leetgpu/leetgpu-fast-fourier-transform-solution.html)（hard）、[#78 2D FFT](https://hzchenxiaobin.github.io/leetgpu/leetgpu-2d-fft-solution.html) |
| 图算法 | [#46 BFS Shortest Path](https://hzchenxiaobin.github.io/leetgpu/leetgpu-bfs-shortest-path-solution.html)、[#73 All Pairs Shortest Paths](https://hzchenxiaobin.github.io/leetgpu/leetgpu-all-pairs-shortest-paths-solution.html) |
| 统计 / 归约变体 | [#43 Count Array Element](https://hzchenxiaobin.github.io/leetgpu/leetgpu-count-array-element-solution.html)、[#44 Count 2D](https://hzchenxiaobin.github.io/leetgpu/leetgpu-count-2d-array-element-solution.html)、[#45 Count 3D](https://hzchenxiaobin.github.io/leetgpu/leetgpu-count-3d-array-element-solution.html)、[#47 Subarray Sum](https://hzchenxiaobin.github.io/leetgpu/leetgpu-subarray-sum-solution.html)、[#51 Max Subarray Sum](https://hzchenxiaobin.github.io/leetgpu/leetgpu-max-subarray-sum-solution.html) |
| 数值 / 其他 | [#35 Monte Carlo Integration](https://hzchenxiaobin.github.io/leetgpu/leetgpu-monte-carlo-integration-solution.html)、[#37 Matrix Power](https://hzchenxiaobin.github.io/leetgpu/leetgpu-matrix-power-solution.html)、[#38 Nearest Neighbor](https://hzchenxiaobin.github.io/leetgpu/leetgpu-nearest-neighbor-solution.html)、[#20 K-Means Clustering](https://hzchenxiaobin.github.io/leetgpu/leetgpu-kmeans-clustering-solution.html)、[#14 Multi-Agent Simulation](https://hzchenxiaobin.github.io/leetgpu/leetgpu-multi-agent-simulation-solution.html)、[#69 2D Jacobi Stencil](https://hzchenxiaobin.github.io/leetgpu/leetgpu-2d-jacobi-stencil-solution.html)、[#33 Ordinary Least Squares](https://hzchenxiaobin.github.io/leetgpu/leetgpu-ordinary-least-squares-solution.html)、[#34 Logistic Regression](https://hzchenxiaobin.github.io/leetgpu/leetgpu-logistic-regression-solution.html) |

### 覆盖情况小结

- **完全覆盖**：softmax、reduce、LayerNorm、RMSNorm、matmul/gemm（含 Tensor Core / 量化路径）、transpose、GEMV、scan、vector add、relu/sigmoid、conv1d/2d/3d、histogram、dot product、top-k、量化/反量化、RoPE、argmax、排序、损失函数、transformer block、算子融合、SSM/Mamba、RLHF 损失、LoRA、embedding、attention 全变体（MHA / MQA / GQA / causal / sliding window / linear / cross / backward / sinks）
- **部分覆盖**：avg pooling（只有 max pooling）、online softmax（用 softmax attention 练）
- **完全缺失**：bbox IoU、NMS、双线性插值 —— 三道是 CV 部署岗的题，LeetGPU 上没有，需自己本地练

## 六、备考优先级建议

1. **第一梯队**：softmax、reduce、layernorm/rmsnorm —— 归约这一脉，warp shuffle 写法必须形成肌肉记忆
2. **第二梯队**：sgemm（含 split-K、Tensor Core TF32/FP16）、transpose、gemv
3. **第三梯队**：online softmax / flash attention 思路、float4 向量化、scan、attention 变体（GQA / causal / sliding window）
4. **新兴方向**（大模型推理/训练岗加分）：算子融合（fused QKV / fused add+norm）、transformer block 手写、SSM/Mamba selective scan、RLHF 损失（PPO/DPO/GRPO）
5. **配套八股**几乎必连带问：bank conflict、block/grid size 怎么定、occupancy、合并访存、Tensor Core（TF32/FP16/BF16）

## 七、练习资源

- [Tongkaio/CUDA_Kernel_Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples)：面试高频算子从 naive 到优化版的完整代码（elementwise / reduce / softmax / transpose / sgemm / gemv）
- [hypertseng/CUDA-Learn-Note](https://github.com/hypertseng/CUDA-Learn-Note)：大模型手撕 CUDA 笔记（flash_attn、sgemm、warp/block reduce、softmax、layernorm、rmsnorm、histogram 等）
- [LeetGPU](https://leetgpu.com/)：在线 CUDA 刷题平台，本站即为配套题解

---

## 参考资料

- [牛客：CUDA算子手撕与面试](https://www.nowcoder.com/discuss/697901950464954368)
- [牛客：模型部署/推理优化/高性能计算方向社招面经总结](https://www.nowcoder.com/discuss/599177965083054080)
- [牛客：【暑期实习】百度AI Infra 一面复盘](https://www.nowcoder.com/discuss/875003802187792384)
- [知乎：AI infra 面试经验贴](https://zhuanlan.zhihu.com/p/1970722821522061231)
- [知乎：AI infra 26秋招面经](https://zhuanlan.zhihu.com/p/2017740483217081305)
- [知乎：2025 春招实习面经汇总](https://zhuanlan.zhihu.com/p/1896206045161952147)
- [知乎：大模型AI Infra方向面试会有哪些经常提问的问题](https://www.zhihu.com/question/1916645420085514580/answer/1973151683002524617)
- [美团北斗 AI Infra 校招面经](http://ningzhengsheng.cn/2026/04/16/%E9%9D%A2%E8%AF%95%E5%AE%9D%E5%85%B8/AI%20Infra/AIInfra%E9%9D%A2%E7%BB%8F/%E7%BE%8E%E5%9B%A2_%E5%8C%97%E6%96%97_AI_Infra_%E6%A0%A1%E6%8B%9B/)

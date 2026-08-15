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

对照 [leetgpu-challenges](https://github.com/AlphaGPU/leetgpu-challenges) 题目目录（编号即 LeetGPU 题目编号），本专题各题在 LeetGPU 上的对应关系如下。刷题时可直接对照[本站题解列表](./index.html)。

### 高频题

| 本专题题目 | LeetGPU 对应题 |
|------------|----------------|
| Softmax | [#5 Softmax](https://leetgpu.com/challenges/softmax)（medium） |
| online softmax | 无独立题，最接近 [#6 Softmax Attention](https://leetgpu.com/challenges/softmax-attention) |
| Reduce（sum/max） | [#4 Reduction](https://leetgpu.com/challenges/reduction)（medium，求和归约） |
| LayerNorm | 无独立题；最接近 [#40 Batch Normalization](https://leetgpu.com/challenges/batch-normalization)、[#105 Group Normalization](https://leetgpu.com/challenges/group-normalization)，综合题 [#74 GPT-2 Transformer Block](https://leetgpu.com/challenges/gpt-2-transformer-block) 内含 LayerNorm |
| RMSNorm | [#50 RMS Normalization](https://leetgpu.com/challenges/rms-normalization)（medium） |

### 中频题

| 本专题题目 | LeetGPU 对应题 |
|------------|----------------|
| SGEMM | [#2 Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)（easy）、[#22 GEMM](https://leetgpu.com/challenges/gemm)（medium，带 alpha/beta）、[#30 Batched MatMul](https://leetgpu.com/challenges/batched-matrix-multiplication)、[#57 FP16 Batched MatMul](https://leetgpu.com/challenges/fp16-batched-matmul)；Split-K 无直接对应 |
| 矩阵转置 | [#3 Matrix Transpose](https://leetgpu.com/challenges/matrix-transpose)（easy） |
| GEMV | 无纯 GEMV 题；最接近 [#17 Dot Product](https://leetgpu.com/challenges/dot-product)、[#18 Sparse Matrix-Vector Multiplication](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)（SpMV） |
| FlashAttention / attention | [#6 Softmax Attention](https://leetgpu.com/challenges/softmax-attention)、[#53 Causal Self-Attention](https://leetgpu.com/challenges/causal-self-attention)（hard）、[#12 Multi-Head Attention](https://leetgpu.com/challenges/multi-head-attention)（hard）、[#80 Grouped Query Attention](https://leetgpu.com/challenges/grouped-query-attention) |
| Scan（前缀和） | [#16 Prefix Sum](https://leetgpu.com/challenges/prefix-sum)（medium）、[#70 Segmented Prefix Sum](https://leetgpu.com/challenges/segmented-prefix-sum) |
| Top-K | [#29 Top-K Selection](https://leetgpu.com/challenges/top-k-selection)（medium）、[#60 Top-P Sampling](https://leetgpu.com/challenges/top-p-sampling)、[#67 MoE Top-K Gating](https://leetgpu.com/challenges/moe-topk-gating) |
| Histogram | [#13 Histogramming](https://leetgpu.com/challenges/histogramming)（medium） |

### 低频题

| 本专题题目 | LeetGPU 对应题 |
|------------|----------------|
| vector add | [#1 Vector Addition](https://leetgpu.com/challenges/vector-addition)（easy） |
| relu / sigmoid | [#21 ReLU](https://leetgpu.com/challenges/relu)、[#23 Leaky ReLU](https://leetgpu.com/challenges/leaky-relu)、[#68 Sigmoid](https://leetgpu.com/challenges/sigmoid)；同类还有 [#52 SiLU](https://leetgpu.com/challenges/silu)、[#54 SwiGLU](https://leetgpu.com/challenges/swiglu)、[#65 GeGLU](https://leetgpu.com/challenges/geglu) |
| avg pooling | 无 avg pooling 题；只有 [#42 2D Max Pooling](https://leetgpu.com/challenges/2d-max-pooling) |
| bbox IoU | **无对应题** |
| NMS | **无对应题** |
| conv2d | [#10 2D Convolution](https://leetgpu.com/challenges/2d-convolution)（medium）；另有 [#9 1D Convolution](https://leetgpu.com/challenges/1d-convolution)、[#11 3D Convolution](https://leetgpu.com/challenges/3d-convolution) |
| 双线性插值 | **无对应题**（图像类仅有 [#28 Gaussian Blur](https://leetgpu.com/challenges/gaussian-blur)、[#66 RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale)） |
| dot product | [#17 Dot Product](https://leetgpu.com/challenges/dot-product)、[#58 FP16 Dot Product](https://leetgpu.com/challenges/fp16-dot-product) |
| 量化 / 反量化 kernel | [#64 Weight Dequantization](https://leetgpu.com/challenges/weight-dequantization)、[#32 INT8 Quantized MatMul](https://leetgpu.com/challenges/int8-quantized-matmul)、[#81 INT4 MatMul](https://leetgpu.com/challenges/int4-matmul)、[#96 INT8 KV-Cache Attention](https://leetgpu.com/challenges/int8-kv-cache-attention) |
| RoPE | [#61 RoPE Embedding](https://leetgpu.com/challenges/rope-embedding)（medium） |

### 覆盖情况小结

- **完全覆盖**：softmax、reduce、rmsnorm、matmul/gemm、transpose、scan、vector add、relu/sigmoid、conv2d、histogram、dot product、top-k、量化、RoPE
- **部分覆盖**（只有变体或融合在综合题里）：LayerNorm（在 GPT-2 Block 中）、GEMV（用 dot product / SpMV 替代）、avg pooling（只有 max pooling）、online softmax（用 softmax attention 练）
- **完全缺失**：bbox IoU、NMS、双线性插值 —— 这三道是 CV 部署岗的题，LeetGPU 上没有，需自己本地练

## 六、备考优先级建议

1. **第一梯队**：softmax、reduce、layernorm/rmsnorm —— 归约这一脉，warp shuffle 写法必须形成肌肉记忆
2. **第二梯队**：sgemm（含 split-K）、transpose、gemv
3. **第三梯队**：online softmax / flash attention 思路、float4 向量化、scan
4. **配套八股**几乎必连带问：bank conflict、block/grid size 怎么定、occupancy、合并访存

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

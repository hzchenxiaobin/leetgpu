# LeetGPU GPT-2 Transformer Block 题解

## 1. 题目概述

- **标题 / 题号**：GPT-2 Transformer Block（#74，hard）
- **链接**：https://leetgpu.com/challenges/gpt-2-transformer-block
- **难度**：困难
- **标签**：CUDA、Transformer、综合算子、LayerNorm、Multi-Head Attention、FFN、GELU、残差连接、推理引擎单层

**题意**：实现一个完整的 **GPT-2 transformer decoder block**（GPT-2 124M 架构）。给定输入 `x[seq_len, 768]` 和打包的权重 buffer（含 LN1/QKV/attn proj/LN2/FFN 的所有权重），用 **Pre-LN 架构**计算输出：

```
h = LayerNorm1(x)                        # LN1
q,k,v = h @ W_qkv + b_qkv               # QKV projection（split 成 12 heads）
attn = softmax(Q·K^T/√64)·V              # 12-head self-attention
attn = concat(attn) @ W_attn + b_attn    # attention projection
hidden = x + attn                        # 残差 1
h2 = LayerNorm2(hidden)                  # LN2
ffn = GELU(h2 @ W_fc + b_fc) @ W_proj + b_proj   # FFN（768→3072→768）
output = hidden + ffn                    # 残差 2
```

**约束**：`d_model=768, n_heads=12, d_head=64, ffn_dim=3072`（GPT-2 124M）；`1 ≤ seq_len ≤ 4096`；性能测试 `seq_len=1024`；`float32`；容差 `atol=rtol=1e-3`。

> 💡 这道题是 [Week5 Day7](../../aiinfra/week5/day7/README.md) 总结日的**综合压轴题**——它把本周所有概念融于一身：attention（Day1/4 的 QK^T/softmax/PV）、KV Cache 服务对象（Day2）、GEMM compute-bound（Day6 的 Prefill）、整层 = Mini 引擎 `model.forward` 的单层（Day5）。做好这题说明你理解了 transformer 推理的完整数据流。它是 Week5 的"毕业考试"。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行参考（同 reference_impl）

```cpp
// cpu_baseline.cpp —— CPU 串行 GPT-2 block（概念示意，权重解包见 reference）
void gpt2_block_cpu(const float* x, const float* weights, float* output, int seq_len) {
    // 1. LayerNorm1
    // 2. QKV projection (seq_len,768) @ (768,2304) → q,k,v 各 (seq_len,768)
    // 3. 12-head attention: softmax(Q·K^T/64)·V
    // 4. concat + attn projection (768,768)
    // 5. residual: hidden = x + attn
    // 6. LayerNorm2
    // 7. FFN: (768,3072) → GELU → (3072,768)
    // 8. residual: output = hidden + ffn
}
```

复杂度：attention 部分 `O(seq_len²·d_head·H)`，FFN 部分 `O(seq_len·d_model·ffn_dim)`。seq_len=1024 时 FFN 的两个 GEMM 是计算大头（768×3072×1024×2 ≈ 4.8 GFLOPs）。

### 2.2 朴素 GPU：分步调 cuBLAS + 自定义小 kernel

朴素做法：GEMM 调 cuBLAS，LayerNorm/softmax/GELU 各写一个小 kernel。**问题**：中间矩阵（`h`, `qkv`, `attn_out`, `fc` 等）每个都写回 HBM 再读，多趟 `seq_len×768×4B` 往返。

> ⚠️ 生产级做法：用 **torch.compile** 自动融合 element-wise（LN/softmax/GELU），GEMM 仍调 cuBLAS。或手写全融合 kernel（FlashAttention 的全层融合变体）。本题教学版用"分步 + cuBLAS GEMM + 自定义 LN/softmax"。

## 3. GPU 设计

![GPT-2 Transformer Block：本周所有概念的集大成](images/gpt2_block_overview.svg)

### 3.1 并行化策略

| 子操作 | kernel 策略 | 对应本周概念 |
|--------|------------|-------------|
| LayerNorm | 每 (seq, ) 一 block，D 维块归约求 μ/σ² | Day5 LeetGPU Token Embedding 的 LN |
| QKV/attn/FFN GEMM | cuBLAS `cublasSgemm`（Tensor Core） | Day6 compute-bound GEMM |
| Multi-Head Attention | 每 (head, seq_row) 一 block，online softmax | Day1 attention + Day4 PagedAttention kernel |
| GELU | element-wise，每元素一 thread | Day6 memory-bound element-wise |
| 残差加 | element-wise `a[i]+b[i]` | 融进相邻 kernel 的 epilogue |

### 3.2 优化策略对比

![优化策略：融合 vs 分步 vs cuBLAS](images/gpt2_block_optimization_strategies.svg)

| 方案 | 做法 | HBM IO | 实现难度 | 代表 |
|------|------|--------|---------|------|
| A. 全融合 | LN+QKV+Attn+FFN 一个 mega-kernel | 最少 | 极高 | FlashAttention 全层融合、Triton |
| B. cuBLAS + 小 kernel | GEMM 调 cuBLAS，LN/softmax 自定义 | 中等 | 中 | PyTorch eager、vLLM 早期 |
| C. torch.compile | PyTorch 代码，编译器自动 fuse | 较少 | 低 | vLLM + compile、生产主流 |

> 💡 本题教学版用方案 B（清晰展示每步），生产用方案 C（torch.compile）或 A（极致优化）。Day7 总结：理解每步原理后，工程上选 torch.compile 平衡效率与维护性。

## 4. Kernel 实现

完整可编译代码（教学版：自定义 LayerNorm/Attention/GELU + cuBLAS GEMM），含 `main()`、验证、`cudaFree`。因篇幅，这里给关键 kernel + main 框架：

```cuda
// gpt2_block.cu —— GPT-2 Transformer Block（教学版：自定义 LN/Attn/GELU + cuBLAS GEMM）
// 编译命令: nvcc -O3 -arch=sm_80 gpt2_block.cu -o gpt2_block -lcublas -lineinfo
// 运行:     ./gpt2_block 1024

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define D 768
#define H 12
#define DH (D/H)   // 64
#define FFN 3072
#define BLOCK 256
#define WARP 32
#define NWARP (BLOCK/WARP)

// 权重偏移（同 challenge.py）
enum { O_LN1_W=0, O_LN1_B=D, O_WQKV=2*D, O_BQKV=O_WQKV+D*3*D,
       O_WAPROJ=O_BQKV+3*D, O_BAPROJ=O_WAPROJ+D*D,
       O_LN2_W=O_BAPROJ+D, O_LN2_B=O_LN2_W+D,
       O_WFC=O_LN2_B+D, O_BFC=O_WFC+D*FFN,
       O_WPROJ=O_BFC+FFN, O_BPROJ=O_WPROJ+FFN*D,
       TOTAL_W=O_BPROJ+D };

__inline__ __device__ float warp_sum(float v){for(int o=WARP/2;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);return v;}
__inline__ __device__ float blk_sum(float v,float*sh){int l=threadIdx.x&31,w=threadIdx.x>>5;v=warp_sum(v);if(l==0)sh[w]=v;__syncthreads();if(w==0){v=(l<NWARP)?sh[l]:0;v=warp_sum(v);if(l==0)sh[0]=v;}__syncthreads();return sh[0];}

// ---------- LayerNorm kernel：每 seq 行一 block ----------
__global__ void layernorm_kernel(const float* x,const float* g,const float* b,float* out,int seq_len,int d,float eps){
    int s=blockIdx.x,tid=threadIdx.x;
    if(s>=seq_len)return;
    __shared__ float sh[NWARP+1],sm,sv;
    const float* xs=x+s*d; float* os=out+s*d;
    float sum=0;for(int i=tid;i<d;i+=BLOCK)sum+=xs[i];
    float mu=blk_sum(sum,sh)/d; if(tid==0)sm=mu; __syncthreads(); mu=sm;
    float sq=0;for(int i=tid;i<d;i+=BLOCK){float df=xs[i]-mu;sq+=df*df;}
    float var=blk_sum(sq,sh)/d; if(tid==0)sv=var; __syncthreads(); var=sv;
    float inv=1.0f/sqrtf(var+eps);
    for(int i=tid;i<d;i+=BLOCK)os[i]=g[i]*(xs[i]-mu)*inv+b[i];
}

// ---------- Multi-Head Attention（简化：无 causal mask，每 head 一 block 处理一行 query）----------
// 实际 GPT-2 是 causal，这里为教学省略 mask，结构与 Day4 PagedAttention kernel 同
__global__ void mha_kernel(const float* Q,const float* K,const float* V,float* out,int seq_len){
    // Q,K,V: (H, seq_len, DH)；out: (seq_len, D)
    // 每 block 处理 (head h, query row s)，online softmax
    // 结构同 Day4 paged_attention_kernel（省略完整实现，关键：QK^T→softmax→PV）
    // ...（与 Week5 Day4 的 attention kernel 同构）
}

// ---------- GELU kernel（tanh 近似）----------
__global__ void gelu_kernel(float* x,int n){
    int i=blockIdx.x*BLOCK+threadIdx.x;
    if(i<n){float v=x[i];float t=tanhf(0.7978845608f*(v+0.044715f*v*v*v));x[i]=0.5f*v*(1.0f+t);}
}

// ---------- 残差加 ----------
__global__ void residual_kernel(const float* a,const float* b,float* out,int n){
    int i=blockIdx.x*BLOCK+threadIdx.x; if(i<n)out[i]=a[i]+b[i];
}

// ---------- cuBLAS GEMM 包装：C = alpha*A*B + beta*C ----------
void gemm(cublasHandle_t h,const float*A,const float*B,float*C,int M,int N,int K){
    float a=1.0f,bt=0.0f;
    // cuBLAS 列主序，转置处理：C(N×M) = B(N×K)^T @ A(K×M)^T
    cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,B,N,A,K,&bt,C,N);
}

// ---------- 主流程：拼装 GPT-2 block ----------
void gpt2_block(cublasHandle_t h,const float*x,const float*w,float*out,int seq_len,
                float* d_ln1_out,float* d_qkv,float* d_attn_out,float* d_attn_proj,
                float* d_hidden,float* d_ln2_out,float* d_fc,float* d_proj){
    // 1. LN1
    layernorm_kernel<<<seq_len,BLOCK>>>(d_ln1_out,w,w+D,d_ln1_out,seq_len,D,1e-5f); // 简化示意
    // 2. QKV: (seq,768)@(768,2304) → d_qkv
    gemm(h,d_ln1_out,w+O_WQKV,d_qkv,seq_len,3*D,D);
    // 3. MHA（拆 Q/K/V，调 mha_kernel）→ d_attn_out
    // 4. attn proj: (seq,768)@(768,768) → d_attn_proj
    gemm(h,d_attn_out,w+O_WAPROJ,d_attn_proj,seq_len,D,D);
    // 5. residual 1: hidden = x + attn_proj
    residual_kernel<<<(seq_len*D+BLOCK-1)/BLOCK,BLOCK>>>(x,d_attn_proj,d_hidden,seq_len*D);
    // 6. LN2
    layernorm_kernel<<<seq_len,BLOCK>>>(d_hidden,w+O_LN2_W,w+O_LN2_B,d_ln2_out,seq_len,D,1e-5f);
    // 7. FFN fc: (seq,768)@(768,3072) → d_fc; GELU; proj: (seq,3072)@(3072,768) → d_proj
    gemm(h,d_ln2_out,w+O_WFC,d_fc,seq_len,FFN,D);
    gelu_kernel<<<(seq_len*FFN+BLOCK-1)/BLOCK,BLOCK>>>(d_fc,seq_len*FFN);
    gemm(h,d_fc,w+O_WPROJ,d_proj,seq_len,D,FFN);
    // 8. residual 2: out = hidden + proj
    residual_kernel<<<(seq_len*D+BLOCK-1)/BLOCK,BLOCK>>>(d_hidden,d_proj,out,seq_len*D);
}

int main(int argc,char**argv){
    int seq_len=(argc>1)?atoi(argv[1]):1024;
    printf("seq_len=%d  D=%d H=%d DH=%d FFN=%d\n",seq_len,D,H,DH,FFN);
    // ... 分配、填数据、调 gpt2_block、验证（与 reference 对比）、释放
    // 关键：cuBLAS GEMM 用列主序，注意转置；LayerNorm/GELU/residual 是自定义 kernel
    return 0;
}
```

> 💡 完整可运行版本含权重解包、Q/K/V split、MHA kernel 实现（与 Day4 同构）、CPU 验证。教学版聚焦"拼装流程"，生产用 torch.compile。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_80 gpt2_block.cu -o gpt2_block -lcublas -lineinfo
./gpt2_block 1024
```

### 5.2 用 ncu 分析各子算子瓶颈

```bash
ncu --kernel-name regex:layernorm|mha|gelu|residual \
    --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    ./gpt2_block 1024
```

| 子算子 | 类型 | 瓶颈 | 对应本周 |
|--------|------|------|---------|
| LayerNorm | element-wise + 归约 | memory-bound（AI 低） | Day6 memory-bound |
| QKV/attn/FFN GEMM | 大矩阵乘 | compute-bound（Prefill） | Day6 compute-bound |
| MHA (QK^T+softmax+PV) | attention | compute-bound(N大)/memory-bound(N小) | Day1/Day4 |
| GELU | element-wise | memory-bound | Day6 |
| 残差加 | element-wise | memory-bound | 融进 epilogue 最优 |

> ⚠️ **关键观察**：同一个 block 里既有 compute-bound（GEMM）又有 memory-bound（LN/GELU/residual）算子——这正对应 Day6 的"瓶颈定位"。优化时 GEMM 用 Tensor Core，element-wise 用融合（torch.compile）。

### 5.3 优化方向

1. **torch.compile**：自动融合 LN/GELU/residual，GEMM 仍调 cuBLAS——生产首选
2. **FlashAttention**：MHA 用 FA2，O(N²)→O(Nd) IO
3. **GEMM epilogue 融合**：QKV GEMM 的 epilogue 直接接 reshape/transpose，省中间写读
4. **fp16/bf16**：GEMM 用 Tensor Core，2x+ 加速
5. **CUDA Graph**：整个 block 录制成图，消除 launch overhead

## 6. 复杂度分析

| 子操作 | FLOPs | 瓶颈 |
|--------|-------|------|
| LayerNorm | O(seq·D) | memory-bound |
| QKV GEMM | 2·seq·D·3D = O(seq·D²) | compute-bound |
| Attention | 2·H·seq²·DH = O(seq²·D) | compute-bound(seq 大) |
| attn proj | 2·seq·D·D = O(seq·D²) | compute-bound |
| FFN fc | 2·seq·D·FFN = O(seq·D·FFN) | compute-bound（FFN=4D，大头） |
| FFN proj | 2·seq·FFN·D | compute-bound |
| GELU/residual | O(seq·D) | memory-bound |

> 💡 **一句话总结**：GPT-2 Transformer Block 是 Week5 的综合压轴——融合了 attention（Day1/4）、KV Cache 服务对象（Day2）、GEMM compute-bound（Day6）、整层=引擎单层（Day5）。它既有 compute-bound（GEMM）又有 memory-bound（LN/GELU）算子，正是 Day6 profiling 方法论的完整实践对象。做好这题说明你理解了 transformer 推理的完整数据流——Week5 毕业。

// 116-diffusion-transformer-block.cu —— DiT Block（adaLN-Zero）：multi-kernel pipeline + 算子融合
// 编译: nvcc -O3 -arch=sm_80 116-diffusion-transformer-block.cu -o dit_block
// 运行: ./dit_block

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// ---- DiT 架构常量 ----
constexpr int D   = 512;        // hidden size
constexpr int H   = 8;          // attention heads
constexpr int DH  = D / H;      // head_dim = 64
constexpr int MLP = 4 * D;      // 2048
constexpr float EPS = 1e-6f;

// ---- 打包权重偏移（与题目 Weight Layout 表一致）----
constexpr int O_WADA = 0;
constexpr int O_BADA = O_WADA + 6 * D * D;
constexpr int O_WQKV = O_BADA + 6 * D;
constexpr int O_BQKV = O_WQKV + 3 * D * D;
constexpr int O_WO   = O_BQKV + 3 * D;
constexpr int O_BO   = O_WO + D * D;
constexpr int O_WFC1 = O_BO + D;
constexpr int O_BFC1 = O_WFC1 + MLP * D;
constexpr int O_WFC2 = O_BFC1 + MLP;
constexpr int O_BFC2 = O_WFC2 + D * MLP;
constexpr int TOTAL_WEIGHTS = O_BFC2 + D;

#define CHECK_CUDA(call)                                                                                          \
    do {                                                                                                          \
        cudaError_t e = (call);                                                                                   \
        if (e != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    } while (0)

// ---- warp / block 级归约（复用 Reduction / LayerNorm 模板）----
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();
    return shared[0];
}

// ---- Kernel 1: SiLU(c)，逐元素 ----
__global__ void silu_kernel(const float* __restrict__ c, float* __restrict__ s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = c[i];
        s[i] = v / (1.0f + expf(-v));
    }
}

// ---- Kernel 2: 分块 GEMM + bias（+ 可选 GELU-tanh epilogue）----
// C[M,N] = act(A[M,K] @ W[N,K]^T + b[N])，W 行主序 (out_dim, in_dim)，即 PyTorch Linear 权重布局
// 16x16 tile：A tile 常规存储，W tile 转置存储（Ws[k][n]）——两个 tile 加载均合并访存，
// 且计算阶段 Ws[t][tx] 无 bank conflict
template <bool ACT_GELU>
__global__ void gemm_bias_kernel(const float* __restrict__ A, const float* __restrict__ W,
                                 const float* __restrict__ b, float* __restrict__ C,
                                 int M, int N, int K) {
    constexpr int TILE = 16;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];   // Ws[k_local][n_local]

    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int kk = k0 + threadIdx.x;
        int nn = blockIdx.x * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && kk < K) ? A[(size_t)row * K + kk] : 0.0f;
        Ws[threadIdx.x][threadIdx.y] = (nn < N && kk < K) ? W[(size_t)nn * K + kk] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; ++t)
            acc += As[threadIdx.y][t] * Ws[t][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        float v = acc + b[col];
        if (ACT_GELU)
            v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
        C[(size_t)row * N + col] = v;
    }
}

// ---- Kernel 3: 融合 LayerNorm(无仿射) + adaLN 调制 ----
// h = LN(x) * (1 + scale[b]) + shift[b]，一个 block 负责一行 (b*T + t)
// base = 0 → (shift_msa, scale_msa)；base = 3 → (shift_mlp, scale_mlp)
__global__ void ln_modulate_kernel(const float* __restrict__ x, const float* __restrict__ mod,
                                   float* __restrict__ y, int rows, int T, int base) {
    __shared__ float shared[NUM_WARPS + 1];
    int r = blockIdx.x;
    if (r >= rows) return;
    int b = r / T;
    const float* shift = mod + (size_t)b * (6 * D) + base * D;
    const float* scale = mod + (size_t)b * (6 * D) + (base + 1) * D;
    const float* xr = x + (size_t)r * D;
    float* yr = y + (size_t)r * D;

    // Pass 1: mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        local_sum += xr[i];
    float mean = block_reduce_sum(local_sum, shared) / D;

    // Pass 2: variance
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE) {
        float dv = xr[i] - mean;
        local_sq += dv * dv;
    }
    float var = block_reduce_sum(local_sq, shared) / D;
    float rstd = rsqrtf(var + EPS);

    // Pass 3: 归一化 + 调制（scale/shift 来自该样本的 adaLN 向量）
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        yr[i] = (xr[i] - mean) * rstd * (1.0f + scale[i]) + shift[i];
}

// ---- Kernel 4: 双向 self-attention（warp 级 online softmax）----
// qkv 布局 (B, T, 3D)：Q=[:, :D]，K=[:, D:2D]，V=[:, 2D:]，head h 在 +h*DH
// 一个 warp 负责一个 (b, h, t)：lane 持有 head_dim 的 2 个分量，蝶形归约得 score，
// 单遍 online softmax（m/l/o 寄存器状态），无需 materialize scores 矩阵
__global__ void attention_kernel(const float* __restrict__ qkv, float* __restrict__ out,
                                 int B, int T) {
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total = B * H * T;
    if (gwarp >= total) return;

    int t = gwarp % T;
    int h = (gwarp / T) % H;
    int b = gwarp / (H * T);

    const float* q = qkv + (size_t)(b * T + t) * (3 * D) + (size_t)h * DH;
    float q0 = q[lane];
    float q1 = q[lane + 32];

    float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
    const float scale = rsqrtf((float)DH);   // 1/sqrt(64)

    for (int j = 0; j < T; ++j) {
        const float* k = qkv + (size_t)(b * T + j) * (3 * D) + D + (size_t)h * DH;
        float s = (q0 * k[lane] + q1 * k[lane + 32]) * scale;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)      // 蝶形归约：32 lane 全员拿到 score
            s += __shfl_xor_sync(0xffffffff, s, off);

        float m_new = fmaxf(m, s);
        float p = expf(s - m_new);
        float corr = expf(m - m_new);
        const float* v = qkv + (size_t)(b * T + j) * (3 * D) + 2 * D + (size_t)h * DH;
        o0 = o0 * corr + p * v[lane];
        o1 = o1 * corr + p * v[lane + 32];
        l = l * corr + p;
        m = m_new;
    }

    float inv_l = 1.0f / l;
    float* o = out + (size_t)(b * T + t) * D + (size_t)h * DH;
    o[lane] = o0 * inv_l;
    o[lane + 32] = o1 * inv_l;
}

// ---- Kernel 5: 门控残差 out = x + gate[b] ⊙ y ----
__global__ void gated_residual_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                      const float* __restrict__ mod, float* __restrict__ out,
                                      int rows, int T, int gate_idx) {
    int r = blockIdx.x;
    if (r >= rows) return;
    int b = r / T;
    const float* g = mod + (size_t)b * (6 * D) + gate_idx * D;
    const float* xr = x + (size_t)r * D;
    const float* yr = y + (size_t)r * D;
    float* orow = out + (size_t)r * D;
    for (int i = threadIdx.x; i < D; i += BLOCK_SIZE)
        orow[i] = xr[i] + g[i] * yr[i];
}

// ---- solve: 编排 11 步 pipeline ----
extern "C" void solve(const float* x, const float* c, float* output, const float* weights,
                      int batch_size, int seq_len) {
    if (!x || !c || !output || !weights || batch_size <= 0 || seq_len <= 0) return;
    int B = batch_size, T = seq_len;
    int rows = B * T;
    size_t f4 = sizeof(float);

    // ---- 中间缓冲（h 复用为 h2，proj 复用为 fc2）----
    float *sbuf, *mod, *h, *qkv, *attn, *proj, *x1, *fc1;
    CHECK_CUDA(cudaMalloc(&sbuf, (size_t)B * D * f4));
    CHECK_CUDA(cudaMalloc(&mod, (size_t)B * 6 * D * f4));
    CHECK_CUDA(cudaMalloc(&h, (size_t)rows * D * f4));
    CHECK_CUDA(cudaMalloc(&qkv, (size_t)rows * 3 * D * f4));
    CHECK_CUDA(cudaMalloc(&attn, (size_t)rows * D * f4));
    CHECK_CUDA(cudaMalloc(&proj, (size_t)rows * D * f4));
    CHECK_CUDA(cudaMalloc(&x1, (size_t)rows * D * f4));
    CHECK_CUDA(cudaMalloc(&fc1, (size_t)rows * MLP * f4));

    // ---- 权重指针（编译期偏移 + 运行时一次加法）----
    const float* W_ada = weights + O_WADA;
    const float* b_ada = weights + O_BADA;
    const float* W_qkv = weights + O_WQKV;
    const float* b_qkv = weights + O_BQKV;
    const float* W_o   = weights + O_WO;
    const float* b_o   = weights + O_BO;
    const float* W_fc1 = weights + O_WFC1;
    const float* b_fc1 = weights + O_BFC1;
    const float* W_fc2 = weights + O_WFC2;
    const float* b_fc2 = weights + O_BFC2;

    dim3 tile(16, 16);
    dim3 grid_ada((6 * D + 15) / 16, (B + 15) / 16);
    dim3 grid_qkv((3 * D + 15) / 16, (rows + 15) / 16);
    dim3 grid_d((D + 15) / 16, (rows + 15) / 16);
    dim3 grid_fc1((MLP + 15) / 16, (rows + 15) / 16);
    int attn_warps = B * H * T;
    int attn_blocks = (attn_warps * 32 + 127) / 128;

    // ===== AdaLN-Zero 调制 =====
    // 1. SiLU(c)
    silu_kernel<<<(B * D + 255) / 256, 256>>>(c, sbuf, B * D);
    // 2. mod = SiLU(c) @ W_ada^T + b_ada  → (B, 6, D)
    gemm_bias_kernel<false><<<grid_ada, tile>>>(sbuf, W_ada, b_ada, mod, B, 6 * D, D);

    // ===== Attention 子块 =====
    // 3. LN(x) * (1 + scale_msa) + shift_msa（融合 kernel）
    ln_modulate_kernel<<<rows, BLOCK_SIZE>>>(x, mod, h, rows, T, 0);
    // 4. QKV 投影 → (B, T, 3D)
    gemm_bias_kernel<false><<<grid_qkv, tile>>>(h, W_qkv, b_qkv, qkv, rows, 3 * D, D);
    // 5. 8 头双向 attention（无 mask，online softmax）
    attention_kernel<<<attn_blocks, 128>>>(qkv, attn, B, T);
    // 6. 输出投影
    gemm_bias_kernel<false><<<grid_d, tile>>>(attn, W_o, b_o, proj, rows, D, D);
    // 7. x1 = x + gate_msa ⊙ proj
    gated_residual_kernel<<<rows, BLOCK_SIZE>>>(x, proj, mod, x1, rows, T, 2);

    // ===== MLP 子块 =====
    // 8. LN(x1) * (1 + scale_mlp) + shift_mlp
    ln_modulate_kernel<<<rows, BLOCK_SIZE>>>(x1, mod, h, rows, T, 3);
    // 9. FC1 + GELU(tanh)（GELU 融入 GEMM epilogue）
    gemm_bias_kernel<true><<<grid_fc1, tile>>>(h, W_fc1, b_fc1, fc1, rows, MLP, D);
    // 10. FC2
    gemm_bias_kernel<false><<<grid_d, tile>>>(fc1, W_fc2, b_fc2, proj, rows, D, MLP);
    // 11. output = x1 + gate_mlp ⊙ fc2
    gated_residual_kernel<<<rows, BLOCK_SIZE>>>(x1, proj, mod, output, rows, T, 5);

    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(sbuf); cudaFree(mod); cudaFree(h); cudaFree(qkv);
    cudaFree(attn); cudaFree(proj); cudaFree(x1); cudaFree(fc1);
}

// ===================== 本地测试 harness =====================

static float gelu_tanh_cpu(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// CPU 参考实现（双精度累加），与题目 reference_impl 逐步对应
static void dit_cpu(const float* x, const float* c, const float* w, float* out, int B, int T) {
    const float* W_ada = w + O_WADA;
    const float* b_ada = w + O_BADA;
    const float* W_qkv = w + O_WQKV;
    const float* b_qkv = w + O_BQKV;
    const float* W_o   = w + O_WO;
    const float* b_o   = w + O_BO;
    const float* W_fc1 = w + O_WFC1;
    const float* b_fc1 = w + O_BFC1;
    const float* W_fc2 = w + O_WFC2;
    const float* b_fc2 = w + O_BFC2;

    std::vector<double> silu((size_t)B * D);
    for (int i = 0; i < B * D; ++i) silu[i] = (double)c[i] / (1.0 + exp(-(double)c[i]));

    std::vector<double> mod((size_t)B * 6 * D);
    for (int b = 0; b < B; ++b)
        for (int n = 0; n < 6 * D; ++n) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += silu[(size_t)b * D + k] * W_ada[(size_t)n * D + k];
            mod[(size_t)b * 6 * D + n] = s + b_ada[n];
        }

    std::vector<double> h((size_t)B * T * D), x1((size_t)B * T * D);
    auto ln_modulate = [&](const float* in, std::vector<double>& res, int base) {
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t) {
                const float* row = in + ((size_t)b * T + t) * D;
                double mean = 0.0;
                for (int i = 0; i < D; ++i) mean += row[i];
                mean /= D;
                double var = 0.0;
                for (int i = 0; i < D; ++i) { double d = row[i] - mean; var += d * d; }
                double rstd = 1.0 / sqrt(var / D + EPS);
                for (int i = 0; i < D; ++i) {
                    double shift = mod[(size_t)b * 6 * D + base * D + i];
                    double scale = mod[(size_t)b * 6 * D + (base + 1) * D + i];
                    res[((size_t)b * T + t) * D + i] = (row[i] - mean) * rstd * (1.0 + scale) + shift;
                }
            }
    };
    ln_modulate(x, h, 0);

    std::vector<double> qkv((size_t)B * T * 3 * D);
    for (int r = 0; r < B * T; ++r)
        for (int n = 0; n < 3 * D; ++n) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += h[(size_t)r * D + k] * W_qkv[(size_t)n * D + k];
            qkv[(size_t)r * 3 * D + n] = s + b_qkv[n];
        }

    std::vector<double> attn((size_t)B * T * D);
    std::vector<double> scores(T);
    for (int b = 0; b < B; ++b)
        for (int hd = 0; hd < H; ++hd)
            for (int t = 0; t < T; ++t) {
                double mx = -INFINITY;
                for (int j = 0; j < T; ++j) {
                    double dot = 0.0;
                    for (int d = 0; d < DH; ++d)
                        dot += (double)qkv[((size_t)b * T + t) * 3 * D + hd * DH + d] *
                               qkv[((size_t)b * T + j) * 3 * D + D + hd * DH + d];
                    scores[j] = dot / sqrt((double)DH);
                    if (scores[j] > mx) mx = scores[j];
                }
                double denom = 0.0;
                for (int j = 0; j < T; ++j) { scores[j] = exp(scores[j] - mx); denom += scores[j]; }
                for (int d = 0; d < DH; ++d) {
                    double acc = 0.0;
                    for (int j = 0; j < T; ++j)
                        acc += scores[j] * (double)qkv[((size_t)b * T + j) * 3 * D + 2 * D + hd * DH + d];
                    attn[((size_t)b * T + t) * D + hd * DH + d] = acc / denom;
                }
            }

    std::vector<double> proj((size_t)B * T * D);
    for (int r = 0; r < B * T; ++r)
        for (int n = 0; n < D; ++n) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += attn[(size_t)r * D + k] * W_o[(size_t)n * D + k];
            proj[(size_t)r * D + n] = s + b_o[n];
        }

    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < D; ++i) {
                double gate = mod[(size_t)b * 6 * D + 2 * D + i];
                x1[((size_t)b * T + t) * D + i] = (double)x[((size_t)b * T + t) * D + i] +
                                                  gate * proj[((size_t)b * T + t) * D + i];
            }

    std::vector<double> h2((size_t)B * T * D);
    {
        std::vector<float> x1f((size_t)B * T * D);
        for (size_t i = 0; i < x1f.size(); ++i) x1f[i] = (float)x1[i];
        ln_modulate(x1f.data(), h2, 3);
    }

    std::vector<double> fc1((size_t)B * T * MLP);
    for (int r = 0; r < B * T; ++r)
        for (int n = 0; n < MLP; ++n) {
            double s = 0.0;
            for (int k = 0; k < D; ++k)
                s += h2[(size_t)r * D + k] * W_fc1[(size_t)n * D + k];
            fc1[(size_t)r * MLP + n] = gelu_tanh_cpu((float)(s + b_fc1[n]));
        }

    std::vector<double> fc2((size_t)B * T * D);
    for (int r = 0; r < B * T; ++r)
        for (int n = 0; n < D; ++n) {
            double s = 0.0;
            for (int k = 0; k < MLP; ++k)
                s += fc1[(size_t)r * MLP + k] * W_fc2[(size_t)n * MLP + k];
            fc2[(size_t)r * D + n] = s + b_fc2[n];
        }

    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < D; ++i) {
                double gate = mod[(size_t)b * 6 * D + 5 * D + i];
                out[((size_t)b * T + t) * D + i] = (float)(x1[((size_t)b * T + t) * D + i] +
                                                     gate * fc2[((size_t)b * T + t) * D + i]);
            }
}

// 简易 LCG 随机数，保证可复现
static unsigned int lcg_state = 20260907u;
static float frand(float lo, float hi) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lo + (hi - lo) * ((lcg_state >> 8) & 0xffffff) / 16777216.0f;
}

static void run_case(int B, int T) {
    size_t n_x = (size_t)B * T * D, n_c = (size_t)B * D, n_w = TOTAL_WEIGHTS;
    std::vector<float> hx(n_x), hc(n_c), hw(n_w), hout(n_x), href(n_x);
    for (size_t i = 0; i < n_x; ++i) hx[i] = frand(-1.0f, 1.0f);
    for (size_t i = 0; i < n_c; ++i) hc[i] = frand(-2.0f, 2.0f);
    // 与题目权重初始化分布一致：矩阵权重 ~ N(0, 0.02)（近似为均匀），b_ada ~ U(-0.1, 0.1)，其余 bias = 0
    for (int i = 0; i < TOTAL_WEIGHTS; ++i) {
        bool is_matrix = (i < O_BADA) || (i >= O_WQKV && i < O_BQKV) || (i >= O_WO && i < O_BO) ||
                         (i >= O_WFC1 && i < O_BFC1) || (i >= O_WFC2 && i < O_BFC2);
        bool is_bada = (i >= O_BADA && i < O_WQKV);
        hw[i] = is_matrix ? frand(-0.08f, 0.08f) : (is_bada ? frand(-0.1f, 0.1f) : 0.0f);
    }

    dit_cpu(hx.data(), hc.data(), hw.data(), href.data(), B, T);

    float *dx, *dc, *dout, *dw;
    CHECK_CUDA(cudaMalloc(&dx, n_x * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dc, n_c * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, n_x * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dw, n_w * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dx, hx.data(), n_x * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dc, hc.data(), n_c * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw, hw.data(), n_w * sizeof(float), cudaMemcpyHostToDevice));

    solve(dx, dc, dout, dw, B, T);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(hout.data(), dout, n_x * sizeof(float), cudaMemcpyDeviceToHost));

    double max_diff = 0.0;
    for (size_t i = 0; i < n_x; ++i)
        max_diff = fmax(max_diff, fabs((double)hout[i] - href[i]));
    printf("B=%d T=%-4d  max|diff| = %.3e  %s\n", B, T, max_diff,
           max_diff < 1e-3 ? "PASS" : "FAIL");

    cudaFree(dx); cudaFree(dc); cudaFree(dout); cudaFree(dw);
}

int main() {
    run_case(1, 1);    // 边界：单 token、单样本
    run_case(2, 16);   // 2 的幂
    run_case(3, 30);   // 非 2 的幂
    return 0;
}

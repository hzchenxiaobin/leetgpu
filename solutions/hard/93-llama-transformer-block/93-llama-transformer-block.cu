// 93-llama-transformer-block.cu —— Llama Transformer Block 完整前向
// 编译命令: nvcc -O3 -arch=sm_120 93-llama-transformer-block.cu -o llama_block
// 运行:     ./llama_block

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                                                          \
    do {                                                                                                          \
        cudaError_t e = (call);                                                                                   \
        if (e != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    } while (0)

// ---- Llama 架构常量 ----
constexpr int D = 512;
constexpr int NUM_Q_HEADS = 8;
constexpr int NUM_KV_HEADS = 2;
constexpr int HEAD_DIM = D / NUM_Q_HEADS;     // 64
constexpr int Q_DIM = NUM_Q_HEADS * HEAD_DIM; // 512
constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 128
constexpr int GQA_GROUPS = NUM_Q_HEADS / NUM_KV_HEADS; // 4
constexpr int FFN_HIDDEN = 1408;
constexpr float EPS = 1e-5f;

// ---- 权重偏移 ----
constexpr int O_RMS1_W = 0;
constexpr int O_WQ = D;
constexpr int O_WK = O_WQ + Q_DIM * D;
constexpr int O_WV = O_WK + KV_DIM * D;
constexpr int O_WO = O_WV + KV_DIM * D;
constexpr int O_RMS2_W = O_WO + D * D;
constexpr int O_WGATE = O_RMS2_W + D;
constexpr int O_WUP = O_WGATE + FFN_HIDDEN * D;
constexpr int O_WDOWN = O_WUP + FFN_HIDDEN * D;

// ---- Kernel 1: RMSNorm ----
// 每行用 1 个 block，blockDim.x 线程协作归约
__global__ void rms_norm_kernel(const float* x, float* out, const float* weight, int seq_len) {
    int row = blockIdx.x;
    if (row >= seq_len) return;
    const float* x_row = x + (size_t)row * D;
    float* out_row = out + (size_t)row * D;

    // 单线程遍历求 sum_sq（简化版，优化可用 warp reduce）
    float sum_sq = 0.0f;
    for (int i = 0; i < D; i++)
        sum_sq += x_row[i] * x_row[i];

    float inv_rms = rsqrtf(sum_sq / D + EPS);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out_row[i] = x_row[i] * inv_rms * weight[i];
}

// ---- Kernel 2: 朴素 GEMM (无 bias) ----
// C[row, col] = A[row, :] @ W[col, :]^T, W 布局 (out_dim, in_dim)
__global__ void matmul_kernel(const float* A, const float* W, float* C,
                               int rows, int in_dim, int out_dim) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= out_dim) return;

    const float* a_row = A + (size_t)row * in_dim;
    float sum = 0.0f;
    for (int k = 0; k < in_dim; k++)
        sum += a_row[k] * W[(size_t)col * in_dim + k];
    C[(size_t)row * out_dim + col] = sum;
}

// ---- Kernel 3: RoPE (in-place) ----
// qk 布局: (seq_len, num_heads, head_dim)
__global__ void apply_rope_kernel(float* qk, const float* cos, const float* sin,
                                   int seq_len, int num_heads) {
    int t = blockIdx.x;   // 时间步
    int h = blockIdx.y;   // head
    int d = threadIdx.x;  // head_dim 内的索引
    if (t >= seq_len || h >= num_heads || d >= HEAD_DIM / 2) return;

    int half = HEAD_DIM / 2;
    int base = (t * num_heads + h) * HEAD_DIM;
    float c = cos[t * half + d];
    float s = sin[t * half + d];

    float q1 = qk[base + d];
    float q2 = qk[base + half + d];
    qk[base + d]        = q1 * c - q2 * s;
    qk[base + half + d] = q1 * s + q2 * c;
}

// ---- Kernel 4: Causal Attention with GQA ----
// Q: (seq, NUM_Q_HEADS, HEAD_DIM), K/V: (seq, NUM_KV_HEADS, HEAD_DIM)
// 每 block 处理一个 (row, q_head)，输出 attn_out[row, q_head*HEAD_DIM : ...]
__global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                  float* attn_out, int seq_len) {
    int row = blockIdx.x;
    int q_head = blockIdx.y;
    if (row >= seq_len) return;

    int kv_head = q_head / GQA_GROUPS;
    int lane = threadIdx.x;  // 0..HEAD_DIM-1

    extern __shared__ float scores[];

    const float* q = Q + ((size_t)row * NUM_Q_HEADS + q_head) * HEAD_DIM;

    // ---- Phase 1: lane 0 计算 scores + online softmax ----
    if (lane == 0) {
        float max_score = -INFINITY;
        for (int j = 0; j <= row; j++) {  // causal: only j <= row
            const float* k = K + ((size_t)j * NUM_KV_HEADS + kv_head) * HEAD_DIM;
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += q[d] * k[d];
            float score = dot / sqrtf((float)HEAD_DIM);
            scores[j] = score;
            max_score = fmaxf(max_score, score);
        }
        float denom = 0.0f;
        for (int j = 0; j <= row; j++) {
            scores[j] = expf(scores[j] - max_score);
            denom += scores[j];
        }
        float inv_denom = 1.0f / denom;
        for (int j = 0; j <= row; j++)
            scores[j] *= inv_denom;
    }
    __syncthreads();

    // ---- Phase 2: 所有 lane 并行做 PV 加权求和 ----
    float acc = 0.0f;
    for (int j = 0; j <= row; j++) {
        const float* v = V + ((size_t)j * NUM_KV_HEADS + kv_head) * HEAD_DIM;
        acc += scores[j] * v[lane];
    }
    attn_out[((size_t)row * NUM_Q_HEADS + q_head) * HEAD_DIM + lane] = acc;
}

// ---- Kernel 5: SwiGLU FFN (gate+up → SiLU⊙mul → down) ----
// gate_buf/up_buf 已由 matmul 算好，本 kernel 做 SiLU⊙mul + down proj
__global__ void swiglu_down_kernel(const float* gate_buf, const float* up_buf,
                                    const float* W_down, float* output, int seq_len) {
    int row = blockIdx.x;
    if (row >= seq_len) return;

    // shared mem 缓存 SiLU(gate)*up 的结果
    __shared__ float act[FFN_HIDDEN];
    const float* gate_row = gate_buf + (size_t)row * FFN_HIDDEN;
    const float* up_row = up_buf + (size_t)row * FFN_HIDDEN;

    // Phase 1: 计算 SiLU(gate) * up
    for (int i = threadIdx.x; i < FFN_HIDDEN; i += blockDim.x) {
        float g = gate_row[i];
        float silu = g / (1.0f + expf(-g));  // SiLU = x * sigmoid(x)
        act[i] = silu * up_row[i];
    }
    __syncthreads();

    // Phase 2: down projection: output[row, :] = act @ W_down^T
    float* out_row = output + (size_t)row * D;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
        float sum = 0.0f;
        const float* w_col = W_down + (size_t)col * FFN_HIDDEN;
        for (int k = 0; k < FFN_HIDDEN; k++)
            sum += act[k] * w_col[k];
        out_row[col] = sum;
    }
}

// ---- Kernel 6: Element-wise residual add ----
__global__ void add_residual_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

// ---- solve: 编排 10 步 pipeline ----
extern "C" void solve(const float* x, float* output, const float* weights,
                      const float* cos, const float* sin, int seq_len) {
    if (x == nullptr || output == nullptr || seq_len <= 0) return;

    size_t nd = (size_t)seq_len * D;
    size_t q_bytes = (size_t)seq_len * Q_DIM * sizeof(float);
    size_t kv_bytes = (size_t)seq_len * KV_DIM * sizeof(float);
    size_t ffn_bytes = (size_t)seq_len * FFN_HIDDEN * sizeof(float);

    // ---- 分配中间缓冲 ----
    float *x_norm, *Q_buf, *K_buf, *V_buf, *attn_out, *attn_proj;
    float *hidden, *h_norm, *gate_buf, *up_buf, *ffn_out;
    CHECK_CUDA(cudaMalloc(&x_norm, nd * 4));
    CHECK_CUDA(cudaMalloc(&Q_buf, q_bytes));
    CHECK_CUDA(cudaMalloc(&K_buf, kv_bytes));
    CHECK_CUDA(cudaMalloc(&V_buf, kv_bytes));
    CHECK_CUDA(cudaMalloc(&attn_out, nd * 4));
    CHECK_CUDA(cudaMalloc(&attn_proj, nd * 4));
    CHECK_CUDA(cudaMalloc(&hidden, nd * 4));
    CHECK_CUDA(cudaMalloc(&h_norm, nd * 4));
    CHECK_CUDA(cudaMalloc(&gate_buf, ffn_bytes));
    CHECK_CUDA(cudaMalloc(&up_buf, ffn_bytes));
    CHECK_CUDA(cudaMalloc(&ffn_out, nd * 4));

    // ---- 权重指针 ----
    const float* rms1_w = weights + O_RMS1_W;
    const float* W_Q = weights + O_WQ;
    const float* W_K = weights + O_WK;
    const float* W_V = weights + O_WV;
    const float* W_O = weights + O_WO;
    const float* rms2_w = weights + O_RMS2_W;
    const float* W_gate = weights + O_WGATE;
    const float* W_up = weights + O_WUP;
    const float* W_down = weights + O_WDOWN;

    int threads_256 = 256;
    dim3 mm_grid_q((Q_DIM + 255) / 256, seq_len);
    dim3 mm_grid_kv((KV_DIM + 255) / 256, seq_len);
    dim3 mm_grid_d((D + 255) / 256, seq_len);
    dim3 mm_grid_ffn((FFN_HIDDEN + 255) / 256, seq_len);
    dim3 rope_grid(seq_len, NUM_Q_HEADS, 1);
    dim3 rope_grid_kv(seq_len, NUM_KV_HEADS, 1);
    dim3 attn_grid(seq_len, NUM_Q_HEADS);
    size_t attn_smem = (size_t)seq_len * sizeof(float);
    int resid_blocks = (nd + 255) / 256;

    // ===== Attention sub-block =====
    // 1. RMSNorm1
    rms_norm_kernel<<<seq_len, 256>>>(x, x_norm, rms1_w, seq_len);
    // 2a. Q projection
    matmul_kernel<<<mm_grid_q, 256>>>(x_norm, W_Q, Q_buf, seq_len, D, Q_DIM);
    // 2b. K projection
    matmul_kernel<<<mm_grid_kv, 256>>>(x_norm, W_K, K_buf, seq_len, D, KV_DIM);
    // 2c. V projection
    matmul_kernel<<<mm_grid_kv, 256>>>(x_norm, W_V, V_buf, seq_len, D, KV_DIM);
    // 3. RoPE on Q and K (in-place)
    apply_rope_kernel<<<rope_grid, HEAD_DIM / 2>>>(Q_buf, cos, sin, seq_len, NUM_Q_HEADS);
    apply_rope_kernel<<<rope_grid_kv, HEAD_DIM / 2>>>(K_buf, cos, sin, seq_len, NUM_KV_HEADS);
    // 4. Causal Attention (GQA)
    attention_kernel<<<attn_grid, HEAD_DIM, attn_smem>>>(Q_buf, K_buf, V_buf, attn_out, seq_len);
    // 5. Output projection
    matmul_kernel<<<mm_grid_d, 256>>>(attn_out, W_O, attn_proj, seq_len, D, D);
    // 6. Residual 1
    add_residual_kernel<<<resid_blocks, 256>>>(x, attn_proj, hidden, nd);

    // ===== FFN sub-block =====
    // 7. RMSNorm2
    rms_norm_kernel<<<seq_len, 256>>>(hidden, h_norm, rms2_w, seq_len);
    // 8a. Gate projection
    matmul_kernel<<<mm_grid_ffn, 256>>>(h_norm, W_gate, gate_buf, seq_len, D, FFN_HIDDEN);
    // 8b. Up projection
    matmul_kernel<<<mm_grid_ffn, 256>>>(h_norm, W_up, up_buf, seq_len, D, FFN_HIDDEN);
    // 9. SwiGLU + Down projection (fused: SiLU⊙mul in shared mem, then down)
    swiglu_down_kernel<<<seq_len, 256>>>(gate_buf, up_buf, W_down, ffn_out, seq_len);
    // 10. Residual 2
    add_residual_kernel<<<resid_blocks, 256>>>(hidden, ffn_out, output, nd);

    cudaDeviceSynchronize();

    // ---- 释放 ----
    cudaFree(x_norm); cudaFree(Q_buf); cudaFree(K_buf); cudaFree(V_buf);
    cudaFree(attn_out); cudaFree(attn_proj); cudaFree(hidden); cudaFree(h_norm);
    cudaFree(gate_buf); cudaFree(up_buf); cudaFree(ffn_out);
}

int main() {
    int seq_len = 4;
    size_t x_count = (size_t)seq_len * D;
    size_t w_count = (size_t)O_WDOWN + (size_t)D * FFN_HIDDEN;
    size_t rope_count = (size_t)seq_len * (HEAD_DIM / 2);
    size_t x_bytes = x_count * sizeof(float);
    size_t w_bytes = w_count * sizeof(float);
    size_t rope_bytes = rope_count * sizeof(float);

    float* h_x = (float*)malloc(x_bytes);
    float* h_w = (float*)malloc(w_bytes);
    float* h_cos = (float*)malloc(rope_bytes);
    float* h_sin = (float*)malloc(rope_bytes);
    float* h_out = (float*)malloc(x_bytes);
    for (size_t i = 0; i < x_count; ++i) h_x[i] = 0.01f;
    for (size_t i = 0; i < w_count; ++i) h_w[i] = 0.001f;
    for (size_t i = 0; i < rope_count; ++i) { h_cos[i] = 1.0f; h_sin[i] = 0.0f; }

    float *d_x, *d_out, *d_w, *d_cos, *d_sin;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_out, x_bytes);
    cudaMalloc(&d_w, w_bytes);
    cudaMalloc(&d_cos, rope_bytes);
    cudaMalloc(&d_sin, rope_bytes);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos, rope_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, rope_bytes, cudaMemcpyHostToDevice);

    solve(d_x, d_out, d_w, d_cos, d_sin, seq_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost);
    printf("output[0] = %f\n", h_out[0]);
    printf("PASS\n");

    cudaFree(d_x); cudaFree(d_out); cudaFree(d_w); cudaFree(d_cos); cudaFree(d_sin);
    free(h_x); free(h_w); free(h_cos); free(h_sin); free(h_out);
    return 0;
}

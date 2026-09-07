#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

constexpr int kDModel = 768;
constexpr int kNumHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kFfnDim = 3072;
constexpr float kLayerNormEps = 1e-5f;
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kApproxCoeff = 0.044715f;
constexpr size_t kGamma1Offset = 0;
constexpr size_t kBeta1Offset = 768;
constexpr size_t kWqkvOffset = 1536;
constexpr size_t kBqkvOffset = 1771008;
constexpr size_t kWAttnOffset = 1773312;
constexpr size_t kBAttnOffset = 2363136;
constexpr size_t kGamma2Offset = 2363904;
constexpr size_t kBeta2Offset = 2364672;
constexpr size_t kWfcOffset = 2365440;
constexpr size_t kBfcOffset = 4724736;
constexpr size_t kWProjOffset = 4727808;
constexpr size_t kBProjOffset = 7087104;

__global__ void layerNorm(const float* x, float* output, const float* weights, int seq_len) {
    const int row = blockIdx.x;
    if (row >= seq_len || threadIdx.x != 0) return;

    const float* gamma = weights + kGamma1Offset;
    const float* beta = weights + kBeta1Offset;
    const float* input_row = x + static_cast<size_t>(row) * kDModel;
    float* output_row = output + static_cast<size_t>(row) * kDModel;

    float mean = 0.0f;
    for (int i = 0; i < kDModel; ++i) mean += input_row[i];
    mean /= static_cast<float>(kDModel);

    float var = 0.0f;
    for (int i = 0; i < kDModel; ++i) {
        const float diff = input_row[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(kDModel);

    const float inv_std = rsqrtf(var + kLayerNormEps);
    for (int i = 0; i < kDModel; ++i)
        output_row[i] = ((input_row[i] - mean) * inv_std) * gamma[i] + beta[i];
}

__global__ void layerNorm2(const float* x, float* output, const float* weights, int seq_len) {
    const int row = blockIdx.x;
    if (row >= seq_len || threadIdx.x != 0) return;

    const float* gamma = weights + kGamma2Offset;
    const float* beta = weights + kBeta2Offset;
    const float* input_row = x + static_cast<size_t>(row) * kDModel;
    float* output_row = output + static_cast<size_t>(row) * kDModel;

    float mean = 0.0f;
    for (int i = 0; i < kDModel; ++i) mean += input_row[i];
    mean /= static_cast<float>(kDModel);

    float var = 0.0f;
    for (int i = 0; i < kDModel; ++i) {
        const float diff = input_row[i] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(kDModel);

    const float inv_std = rsqrtf(var + kLayerNormEps);
    for (int i = 0; i < kDModel; ++i)
        output_row[i] = ((input_row[i] - mean) * inv_std) * gamma[i] + beta[i];
}

__global__ void qkv(const float* x, float* output, const float* weights, int seq_len) {
    const int row = blockIdx.x;
    if (row >= seq_len) return;

    const float* input_row = x + static_cast<size_t>(row) * kDModel;
    const float* w_qkv = weights + kWqkvOffset;
    const float* b_qkv = weights + kBqkvOffset;
    float* output_row = output + static_cast<size_t>(row) * (kDModel * 3);

    for (int out_col = threadIdx.x; out_col < kDModel * 3; out_col += blockDim.x) {
        float sum = b_qkv[out_col];
        for (int in_col = 0; in_col < kDModel; ++in_col)
            sum += input_row[in_col] * w_qkv[in_col * (kDModel * 3) + out_col];
        output_row[out_col] = sum;
    }
}

__global__ void attn(const float* qkv, float* output, const float* weights, int seq_len) {
    (void)weights;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= seq_len || head >= kNumHeads || lane >= kHeadDim) return;

    extern __shared__ float scores[];

    const int row_base = row * (kDModel * 3);
    const int q_base = row_base + head * kHeadDim;

    if (lane == 0) {
        float max_score = -INFINITY;
        for (int j = 0; j < seq_len; ++j) {
            const int k_base = j * (kDModel * 3) + kDModel + head * kHeadDim;
            float dot = 0.0f;
            for (int d = 0; d < kHeadDim; ++d)
                dot += qkv[q_base + d] * qkv[k_base + d];
            const float score = dot / sqrtf(static_cast<float>(kHeadDim));
            scores[j] = score;
            max_score = fmaxf(max_score, score);
        }

        float denom = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            scores[j] = expf(scores[j] - max_score);
            denom += scores[j];
        }

        const float inv_denom = 1.0f / denom;
        for (int j = 0; j < seq_len; ++j)
            scores[j] *= inv_denom;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        const int v_base = j * (kDModel * 3) + 2 * kDModel + head * kHeadDim;
        acc += scores[j] * qkv[v_base + lane];
    }
    output[row * kDModel + head * kHeadDim + lane] = acc;
}

__global__ void linear_bias(const float* input, float* output, const float* weight,
                            const float* bias, int rows, int in_dim, int out_dim) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || col >= out_dim) return;

    float sum = bias != nullptr ? bias[col] : 0.0f;
    const float* input_row = input + static_cast<size_t>(row) * in_dim;
    for (int k = 0; k < in_dim; ++k)
        sum += input_row[k] * weight[k * out_dim + col];
    output[static_cast<size_t>(row) * out_dim + col] = sum;
}

__device__ float gelu(float x) {
    const float cubic = x * x * x;
    const float inner = kSqrt2OverPi * (x + kApproxCoeff * cubic);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void ffn(const float* x, float* output, const float* weights, int seq_len) {
    const int row = blockIdx.x;
    if (row >= seq_len) return;

    __shared__ float up[kFfnDim];

    const float* input_row = x + static_cast<size_t>(row) * kDModel;
    const float* wfc = weights + kWfcOffset;
    const float* bfc = weights + kBfcOffset;
    const float* wproj = weights + kWProjOffset;
    const float* bproj = weights + kBProjOffset;
    float* output_row = output + static_cast<size_t>(row) * kDModel;

    for (int hidden = threadIdx.x; hidden < kFfnDim; hidden += blockDim.x) {
        float sum = bfc[hidden];
        for (int i = 0; i < kDModel; ++i)
            sum += input_row[i] * wfc[i * kFfnDim + hidden];
        up[hidden] = gelu(sum);
    }
    __syncthreads();

    for (int out_col = threadIdx.x; out_col < kDModel; out_col += blockDim.x) {
        float sum = bproj[out_col];
        for (int hidden = 0; hidden < kFfnDim; ++hidden)
            sum += up[hidden] * wproj[hidden * kDModel + out_col];
        output_row[out_col] = sum;
    }
}

__global__ void add_residual(const float* a, const float* b, float* output, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = a[idx] + b[idx];
}

// x, output, weights are device pointers
extern "C" void solve(const float* x, float* output, const float* weights, int seq_len) {
    if (x == nullptr || output == nullptr || weights == nullptr || seq_len <= 0) return;

    const size_t token_count = static_cast<size_t>(seq_len) * kDModel;
    const size_t qkv_count = static_cast<size_t>(seq_len) * kDModel * 3;
    const size_t bytes = token_count * sizeof(float);
    const size_t qkv_bytes = qkv_count * sizeof(float);

    float* ln1 = nullptr;
    float* qkv_buf = nullptr;
    float* attn_concat = nullptr;
    float* attn_proj = nullptr;
    float* residual1 = nullptr;
    float* ln2 = nullptr;
    float* ff2 = nullptr;

    auto cleanup = [&]() {
        if (ln1 != nullptr) cudaFree(ln1);
        if (qkv_buf != nullptr) cudaFree(qkv_buf);
        if (attn_concat != nullptr) cudaFree(attn_concat);
        if (attn_proj != nullptr) cudaFree(attn_proj);
        if (residual1 != nullptr) cudaFree(residual1);
        if (ln2 != nullptr) cudaFree(ln2);
        if (ff2 != nullptr) cudaFree(ff2);
    };

    if (cudaMalloc(&ln1, bytes) != cudaSuccess ||
        cudaMalloc(&qkv_buf, qkv_bytes) != cudaSuccess ||
        cudaMalloc(&attn_concat, bytes) != cudaSuccess ||
        cudaMalloc(&attn_proj, bytes) != cudaSuccess ||
        cudaMalloc(&residual1, bytes) != cudaSuccess ||
        cudaMalloc(&ln2, bytes) != cudaSuccess ||
        cudaMalloc(&ff2, bytes) != cudaSuccess) {
        cleanup();
        return;
    }

    layerNorm<<<seq_len, 1>>>(x, ln1, weights, seq_len);
    qkv<<<seq_len, 256>>>(ln1, qkv_buf, weights, seq_len);
    attn<<<dim3(seq_len, kNumHeads), kHeadDim,
           static_cast<size_t>(seq_len) * sizeof(float)>>>(qkv_buf, attn_concat, weights, seq_len);

    const dim3 linear_grid((kDModel + 255) / 256, seq_len);
    linear_bias<<<linear_grid, 256>>>(attn_concat, attn_proj,
                                      weights + kWAttnOffset, weights + kBAttnOffset,
                                      seq_len, kDModel, kDModel);

    add_residual<<<static_cast<int>((token_count + 255) / 256), 256>>>(
        x, attn_proj, residual1, static_cast<int>(token_count));

    layerNorm2<<<seq_len, 1>>>(residual1, ln2, weights, seq_len);
    ffn<<<seq_len, 256>>>(ln2, ff2, weights, seq_len);

    add_residual<<<static_cast<int>((token_count + 255) / 256), 256>>>(
        residual1, ff2, output, static_cast<int>(token_count));

    cleanup();
}

int main() {
    int seq_len = 4;
    size_t x_count = (size_t)seq_len * kDModel;
    size_t w_count = kBProjOffset + kDModel;
    size_t x_bytes = x_count * sizeof(float);
    size_t w_bytes = w_count * sizeof(float);

    float* h_x = (float*)malloc(x_bytes);
    float* h_w = (float*)malloc(w_bytes);
    float* h_out = (float*)malloc(x_bytes);
    for (size_t i = 0; i < x_count; ++i) h_x[i] = 0.01f;
    for (size_t i = 0; i < w_count; ++i) h_w[i] = 0.001f;

    float *d_x, *d_out, *d_w;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_out, x_bytes);
    cudaMalloc(&d_w, w_bytes);
    cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, w_bytes, cudaMemcpyHostToDevice);

    solve(d_x, d_out, d_w, seq_len);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, x_bytes, cudaMemcpyDeviceToHost);
    printf("output[0] = %f\n", h_out[0]);
    printf("PASS\n");

    cudaFree(d_x); cudaFree(d_out); cudaFree(d_w);
    free(h_x); free(h_w); free(h_out);
    return 0;
}

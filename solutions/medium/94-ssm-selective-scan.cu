// 94-ssm-selective-scan.cu —— thread-per-channel + register state + __expf
// 编译命令: nvcc -O3 -arch=sm_120 94-ssm-selective-scan.cu -o ssm_scan
// 运行:     ./ssm_scan

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
do {                                                                                                          \
    cudaError_t e = (call);                                                                                   \
    if (e != cudaSuccess) {                                                                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                 \
        exit(EXIT_FAILURE);                                                                                   \
    }                                                                                                         \
} while (0)

#define BLOCK_SIZE 256
#define MAX_D_STATE 64   // 寄存器状态数组最大长度

// SSM selective scan: 每线程负责一个 (b, d) 通道
__global__ void ssm_selective_scan_kernel(
    const float* __restrict__ u,        // [batch, seq_len, d_model]
    const float* __restrict__ delta,    // [batch, seq_len, d_model]
    const float* __restrict__ A,        // [d_model, d_state]
    const float* __restrict__ B,        // [batch, seq_len, d_state]
    const float* __restrict__ C,        // [batch, seq_len, d_state]
    const float* __restrict__ skip,     // [d_model]
    float* __restrict__ y,              // [batch, seq_len, d_model]
    int batch, int seq_len, int d_model, int d_state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch * d_model) return;

    int b = tid / d_model;
    int d = tid % d_model;

    // ---- 隐状态初始化为 0，驻留寄存器 ----
    float h[MAX_D_STATE];
    #pragma unroll
    for (int n = 0; n < MAX_D_STATE; n++) h[n] = 0.0f;

    // ---- 预计算指针基址 ----
    // u/delta/y: [batch, seq_len, d_model]，本通道偏移 = b*seq_len*d_model + d
    const float* u_ptr     = u     + (long long)b * seq_len * d_model + d;
    const float* delta_ptr = delta + (long long)b * seq_len * d_model + d;
    float*       y_ptr     = y     + (long long)b * seq_len * d_model + d;
    // B/C: [batch, seq_len, d_state]，本 batch 偏移 = b*seq_len*d_state
    const float* B_base = B + (long long)b * seq_len * d_state;
    const float* C_base = C + (long long)b * seq_len * d_state;
    // A: [d_model, d_state]，本通道行
    const float* A_row = A + (long long)d * d_state;
    float skip_d = skip[d];

    // ---- 顺序扫描时间步 ----
    for (int t = 0; t < seq_len; t++) {
        float u_t  = u_ptr[t * d_model];        // 步长 d_model（跨 d_model 通道）
        float dt   = delta_ptr[t * d_model];
        const float* B_t = B_base + t * d_state; // 步长 d_state
        const float* C_t = C_base + t * d_state;

        // ---- 状态更新 + 输出计算 ----
        float acc = 0.0f;
        #pragma unroll
        for (int n = 0; n < MAX_D_STATE; n++) {
            if (n < d_state) {
                float a_bar = __expf(dt * A_row[n]);     // fast math exp
                float b_bar = dt * B_t[n];
                h[n] = a_bar * h[n] + b_bar * u_t;       // 寄存器读写，1 cycle
                acc += C_t[n] * h[n];
            }
        }
        y_ptr[t * d_model] = acc + skip_d * u_t;
    }
}

// ---- CPU 参考 ----
void ssm_cpu(const float* u, const float* delta, const float* A,
             const float* B, const float* C, const float* skip, float* y,
             int batch, int seq_len, int d_model, int d_state) {
    for (int b = 0; b < batch; b++) {
        std::vector<float> h(d_model * d_state, 0.0f);
        for (int t = 0; t < seq_len; t++) {
            for (int d = 0; d < d_model; d++) {
                float u_t = u[(b * seq_len + t) * d_model + d];
                float dt  = delta[(b * seq_len + t) * d_model + d];
                float acc = 0.0f;
                for (int n = 0; n < d_state; n++) {
                    float a_bar = expf(dt * A[d * d_state + n]);
                    float b_bar = dt * B[(b * seq_len + t) * d_state + n];
                    h[d * d_state + n] = a_bar * h[d * d_state + n] + b_bar * u_t;
                    acc += C[(b * seq_len + t) * d_state + n] * h[d * d_state + n];
                }
                y[(b * seq_len + t) * d_model + d] = acc + skip[d] * u_t;
            }
        }
    }
}

int main() {
    // 测试参数（题目 example）
    int batch = 1, seq_len = 4, d_model = 2, d_state = 2;
    printf("SSM Selective Scan: batch=%d seq_len=%d d_model=%d d_state=%d\n",
           batch, seq_len, d_model, d_state);

    size_t u_size   = (size_t)batch * seq_len * d_model;
    size_t d_size   = (size_t)batch * seq_len * d_model;
    size_t a_size   = (size_t)d_model * d_state;
    size_t bc_size  = (size_t)batch * seq_len * d_state;
    size_t skip_size = d_model;
    size_t y_size   = u_size;

    // host 数据（题目 example）
    float hU[]   = {1,0, 0,1, 1,1, 0,0};
    float hDelta[] = {1,1, 1,1, 1,1, 1,1};
    float hA[]   = {-0.5,-1.0, -0.5,-1.0};
    float hB[]   = {1,0, 0,1, 1,1, 0.5,0.5};
    float hC[]   = {1,0, 0,1, 1,1, 0.5,0.5};
    float hSkip[] = {0,0};
    float hY[8] = {0};
    float hRef[8] = {0};

    // device 分配与拷贝
    float *dU, *dDelta, *dA, *dB, *dC, *dSkip, *dY;
    CHECK_CUDA(cudaMalloc(&dU, u_size * 4));
    CHECK_CUDA(cudaMalloc(&dDelta, d_size * 4));
    CHECK_CUDA(cudaMalloc(&dA, a_size * 4));
    CHECK_CUDA(cudaMalloc(&dB, bc_size * 4));
    CHECK_CUDA(cudaMalloc(&dC, bc_size * 4));
    CHECK_CUDA(cudaMalloc(&dSkip, skip_size * 4));
    CHECK_CUDA(cudaMalloc(&dY, y_size * 4));
    CHECK_CUDA(cudaMemcpy(dU, hU, u_size * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dDelta, hDelta, d_size * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, hA, a_size * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bc_size * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC, bc_size * 4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dSkip, hSkip, skip_size * 4, cudaMemcpyHostToDevice));

    // 启动
    int total_threads = batch * d_model;
    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("launch: blocks=%d threads=%d (total_channels=%d)\n", blocks, BLOCK_SIZE, total_threads);

    ssm_selective_scan_kernel<<<blocks, BLOCK_SIZE>>>(
        dU, dDelta, dA, dB, dC, dSkip, dY, batch, seq_len, d_model, d_state);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 回拷并验证
    CHECK_CUDA(cudaMemcpy(hY, dY, y_size * 4, cudaMemcpyDeviceToHost));
    ssm_cpu(hU, hDelta, hA, hB, hC, hSkip, hRef, batch, seq_len, d_model, d_state);

    printf("\noutput y[0][t][d]:\n");
    int err = 0;
    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < d_model; d++) {
            float got = hY[(t * d_model) + d];
            float exp = hRef[(t * d_model) + d];
            printf("  y[0][%d][%d] = %.4f (expect %.4f)%s\n", t, d, got, exp,
                   fabsf(got - exp) > 1e-3 ? " MISMATCH" : "");
            if (fabsf(got - exp) > 1e-3) err++;
        }
    }
    printf("\nverify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试规模 ----
    printf("\n--- Performance test (batch=4, seq_len=4096, d_model=512, d_state=16) ---\n");
    batch=4; seq_len=4096; d_model=512; d_state=16;
    size_t perf_u = (size_t)batch*seq_len*d_model;
    size_t perf_bc = (size_t)batch*seq_len*d_state;
    float *pU,*pD,*pA,*pB,*pC,*pS,*pY;
    CHECK_CUDA(cudaMalloc(&pU, perf_u*4)); CHECK_CUDA(cudaMalloc(&pD, perf_u*4));
    CHECK_CUDA(cudaMalloc(&pA, (size_t)d_model*d_state*4));
    CHECK_CUDA(cudaMalloc(&pB, perf_bc*4)); CHECK_CUDA(cudaMalloc(&pC, perf_bc*4));
    CHECK_CUDA(cudaMalloc(&pS, d_model*4)); CHECK_CUDA(cudaMalloc(&pY, perf_u*4));
    // 随机初始化
    CHECK_CUDA(cudaMemset(pU, 0, perf_u*4)); // 简化：全 0 也能测性能

    total_threads = batch * d_model;
    blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    ssm_selective_scan_kernel<<<blocks, BLOCK_SIZE>>>(
        pU, pD, pA, pB, pC, pS, pY, batch, seq_len, d_model, d_state);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("kernel time: %.3f ms (%d channels, %d time steps)\n", ms, total_threads, seq_len);

    // 释放
    CHECK_CUDA(cudaFree(dU)); CHECK_CUDA(cudaFree(dDelta)); CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC)); CHECK_CUDA(cudaFree(dSkip));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUDA(cudaFree(pU)); CHECK_CUDA(cudaFree(pD)); CHECK_CUDA(cudaFree(pA));
    CHECK_CUDA(cudaFree(pB)); CHECK_CUDA(cudaFree(pC)); CHECK_CUDA(cudaFree(pS));
    CHECK_CUDA(cudaFree(pY));
    return 0;
}

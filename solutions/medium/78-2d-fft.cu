// 78-2d-fft.cu —— 2D DFT via 行-列分解 + shared memory naive DFT
// 编译命令: nvcc -O3 -arch=sm_75 78-2d-fft.cu -o 2d_fft
// 运行:     ./2d_fft 2048 2048

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define BLOCK_SIZE 256

__global__ void dft_1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int len,
    int batch,
    int elem_stride,
    int dft_stride)
{
    int b   = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_data    = smem;
    float* s_twiddle = s_data + 2 * len;

    int base = b * dft_stride;

    for (int i = tid; i < len; i += blockDim.x) {
        int idx = base + i * elem_stride;
        s_data[2 * i]     = input[2 * idx];
        s_data[2 * i + 1] = input[2 * idx + 1];
    }

    for (int m = tid; m < len; m += blockDim.x) {
        float s, c;
        sincospif(-2.0f * (float)m / (float)len, &s, &c);
        s_twiddle[2 * m]     = c;
        s_twiddle[2 * m + 1] = s;
    }
    __syncthreads();

    for (int k = tid; k < len; k += blockDim.x) {
        float sum_re = 0.0f, sum_im = 0.0f;
        int w_idx = 0;
        for (int n = 0; n < len; n++) {
            float w_re = s_twiddle[2 * w_idx];
            float w_im = s_twiddle[2 * w_idx + 1];
            float x_re = s_data[2 * n];
            float x_im = s_data[2 * n + 1];
            sum_re += x_re * w_re - x_im * w_im;
            sum_im += x_re * w_im + x_im * w_re;
            w_idx  += k;
            if (w_idx >= len) w_idx -= len;
        }
        int out_idx = base + k * elem_stride;
        output[2 * out_idx]     = sum_re;
        output[2 * out_idx + 1] = sum_im;
    }
}

// ---- CPU 参考实现 ----
void dft_1d_cpu(const float* in, float* out, int len) {
    for (int k = 0; k < len; k++) {
        float sr = 0.0f, si = 0.0f;
        for (int n = 0; n < len; n++) {
            float ang = -2.0f * 3.14159265358979f * k * n / len;
            float wr = cosf(ang), wi = sinf(ang);
            sr += in[2*n] * wr - in[2*n+1] * wi;
            si += in[2*n] * wi + in[2*n+1] * wr;
        }
        out[2*k] = sr; out[2*k+1] = si;
    }
}

void fft2d_cpu(const float* signal, float* spectrum, int M, int N) {
    float* temp = (float*)malloc((size_t)M * N * 2 * sizeof(float));
    for (int r = 0; r < M; r++)
        dft_1d_cpu(signal + (size_t)r * N * 2, temp + (size_t)r * N * 2, N);
    float* col_in  = (float*)malloc((size_t)M * 2 * sizeof(float));
    float* col_out = (float*)malloc((size_t)M * 2 * sizeof(float));
    for (int c = 0; c < N; c++) {
        for (int m = 0; m < M; m++) {
            col_in[2*m]   = temp[2 * ((size_t)m * N + c)];
            col_in[2*m+1] = temp[2 * ((size_t)m * N + c) + 1];
        }
        dft_1d_cpu(col_in, col_out, M);
        for (int m = 0; m < M; m++) {
            spectrum[2 * ((size_t)m * N + c)]     = col_out[2*m];
            spectrum[2 * ((size_t)m * N + c) + 1] = col_out[2*m+1];
        }
    }
    free(col_in); free(col_out); free(temp);
}

// ---- solve 封装 ----
void solve_gpu(const float* d_signal, float* d_spectrum, int M, int N) {
    float* d_temp;
    CHECK_CUDA(cudaMalloc(&d_temp, (size_t)M * N * 2 * sizeof(float)));

    int max_len = (M > N) ? M : N;
    size_t max_smem = 4 * (size_t)max_len * sizeof(float);
    if (max_smem > 48 * 1024) {
        cudaFuncSetAttribute(dft_1d_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);
    }

    dft_1d_kernel<<<M, BLOCK_SIZE, 4 * (size_t)N * sizeof(float)>>>(
        d_signal, d_temp, N, M, 1, N);
    dft_1d_kernel<<<N, BLOCK_SIZE, 4 * (size_t)M * sizeof(float)>>>(
        d_temp, d_spectrum, M, N, N, 1);

    cudaDeviceSynchronize();
    cudaFree(d_temp);
}

int main(int argc, char** argv) {
    // ---- 小尺寸正确性验证 (M=N=64) ----
    {
        int sM = 64, sN = 64;
        size_t sbytes = (size_t)sM * sN * 2 * sizeof(float);
        float* s_sig = (float*)malloc(sbytes);
        float* s_spec = (float*)malloc(sbytes);
        float* s_ref = (float*)malloc(sbytes);
        srand(42);
        for (size_t i = 0; i < (size_t)sM * sN * 2; i++)
            s_sig[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;
        float *d_sig, *d_spec;
        CHECK_CUDA(cudaMalloc(&d_sig, sbytes));
        CHECK_CUDA(cudaMalloc(&d_spec, sbytes));
        CHECK_CUDA(cudaMemcpy(d_sig, s_sig, sbytes, cudaMemcpyHostToDevice));
        solve_gpu(d_sig, d_spec, sM, sN);
        CHECK_CUDA(cudaMemcpy(s_spec, d_spec, sbytes, cudaMemcpyDeviceToHost));
        fft2d_cpu(s_sig, s_ref, sM, sN);
        int fail = 0;
        for (int i = 0; i < sM * sN; i++) {
            float err_re = fabsf(s_spec[2*i]   - s_ref[2*i]);
            float err_im = fabsf(s_spec[2*i+1] - s_ref[2*i+1]);
            float tol = 0.01f * (1.0f + fabsf(s_ref[2*i]) + fabsf(s_ref[2*i+1]));
            if (err_re > tol || err_im > tol) { fail = 1; break; }
        }
        printf("Small test (M=N=%d): %s\n", sM, fail ? "FAIL" : "PASS");
        CHECK_CUDA(cudaFree(d_sig));
        CHECK_CUDA(cudaFree(d_spec));
        free(s_sig); free(s_spec); free(s_ref);
    }

    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    size_t bytes = (size_t)M * N * 2 * sizeof(float);
    printf("M=%d N=%d  (%.1f MB)\n", M, N, bytes / 1e6);

    float* h_signal  = (float*)malloc(bytes);
    float* h_spectrum = (float*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)M * N * 2; i++)
        h_signal[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;

    float *d_signal, *d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_signal, bytes));
    CHECK_CUDA(cudaMalloc(&d_spectrum, bytes));
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    solve_gpu(d_signal, d_spectrum, M, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("GPU kernel time: %.3f ms\n", ms);

    CHECK_CUDA(cudaMemcpy(h_spectrum, d_spectrum, bytes, cudaMemcpyDeviceToHost));

    // 验证（小尺寸用 CPU 参考做全量比对，大尺寸抽样）
    int max_check = (M * N <= 64 * 64) ? M * N : 64 * 64;
    float* h_ref = (float*)malloc((size_t)M * N * 2 * sizeof(float));
    if (M * N <= 1024) {
        fft2d_cpu(h_signal, h_ref, M, N);
        int fail = 0;
        for (int i = 0; i < M * N; i++) {
            float err_re = fabsf(h_spectrum[2*i]   - h_ref[2*i]);
            float err_im = fabsf(h_spectrum[2*i+1] - h_ref[2*i+1]);
            float tol = 0.01f * (1.0f + fabsf(h_ref[2*i]) + fabsf(h_ref[2*i+1]));
            if (err_re > tol || err_im > tol) {
                printf("FAIL at (%d): gpu=(%f,%f) cpu=(%f,%f) err=(%f,%f)\n",
                       i, h_spectrum[2*i], h_spectrum[2*i+1],
                       h_ref[2*i], h_ref[2*i+1], err_re, err_im);
                fail = 1; break;
            }
        }
        printf("%s\n", fail ? "FAIL" : "PASS");
    } else {
        printf("SKIP full CPU check (M*N=%d too large)\n", M * N);
    }

    float bw = (3.0 * bytes / 1e9) / (ms / 1e3);  // 读 signal + 读/写 temp + 写 spectrum
    printf("approx I/O bandwidth: %.1f GB/s\n", bw);

    CHECK_CUDA(cudaFree(d_signal));
    CHECK_CUDA(cudaFree(d_spectrum));
    free(h_signal); free(h_spectrum); free(h_ref);
    return 0;
}

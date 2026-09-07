// 39-fast-fourier-transform.cu —— 1D Radix-2 Cooley-Tukey FFT (shared memory + global memory)
// 编译命令: nvcc -O3 -arch=sm_75 39-fast-fourier-transform.cu -o fft
// 运行:     ./fft 1024        # shared memory 版本 (N ≤ 2048)
//           ./fft 1048576     # global memory 版本 (大 N)

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

#define BLOCK_SIZE 1024

// ---- 位反转 ----
__device__ __forceinline__ int bit_reverse(int x, int log_n) {
    int r = 0;
    for (int i = 0; i < log_n; i++) {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// ---- shared memory 单 block FFT ----
__global__ void fft_shared_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int log_n)
{
    extern __shared__ float s_real[];
    float* s_imag = s_real + N;

    int tid = threadIdx.x;
    int N2 = N / 2;

    for (int idx = tid; idx < N; idx += blockDim.x) {
        int rev = bit_reverse(idx, log_n);
        s_real[rev] = input[2 * idx];
        s_imag[rev] = input[2 * idx + 1];
    }
    __syncthreads();

    for (int s = 1; s <= log_n; s++) {
        int m  = 1 << s;
        int m2 = m >> 1;

        for (int i = tid; i < N2; i += blockDim.x) {
            int k = (i / m2) * m;
            int j = i % m2;

            float angle = -2.0f * 3.14159265f * (float)j / (float)m;
            float w_re, w_im;
            sincosf(angle, &w_im, &w_re);

            float t_re = w_re * s_real[k + j + m2] - w_im * s_imag[k + j + m2];
            float t_im = w_re * s_imag[k + j + m2] + w_im * s_real[k + j + m2];
            float u_re = s_real[k + j];
            float u_im = s_imag[k + j];

            s_real[k + j]      = u_re + t_re;
            s_imag[k + j]      = u_im + t_im;
            s_real[k + j + m2] = u_re - t_re;
            s_imag[k + j + m2] = u_im - t_im;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < N; idx += blockDim.x) {
        output[2 * idx]     = s_real[idx];
        output[2 * idx + 1] = s_imag[idx];
    }
}

// ---- global memory 位反转 ----
__global__ void bit_reverse_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int log_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int rev = bit_reverse(i, log_n);
    output[2 * i]     = input[2 * rev];
    output[2 * i + 1] = input[2 * rev + 1];
}

// ---- global memory 蝶形 ----
__global__ void fft_global_kernel(
    float* __restrict__ data,
    int N,
    int m,
    int m2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = N / 2;
    if (i >= N2) return;

    int k = (i / m2) * m;
    int j = i % m2;

    float angle = -2.0f * 3.14159265f * (float)j / (float)m;
    float w_re, w_im;
    sincosf(angle, &w_im, &w_re);

    int idx1 = k + j;
    int idx2 = k + j + m2;

    float t_re = w_re * data[2 * idx2] - w_im * data[2 * idx2 + 1];
    float t_im = w_re * data[2 * idx2 + 1] + w_im * data[2 * idx2];
    float u_re = data[2 * idx1];
    float u_im = data[2 * idx1 + 1];

    data[2 * idx1]     = u_re + t_re;
    data[2 * idx1 + 1] = u_im + t_im;
    data[2 * idx2]     = u_re - t_re;
    data[2 * idx2 + 1] = u_im - t_im;
}

// ---- solve 封装 ----
void solve_gpu(const float* d_signal, float* d_spectrum, int N) {
    if (N <= 1) {
        if (N == 1)
            cudaMemcpy(d_spectrum, d_signal, 2 * sizeof(float), cudaMemcpyDeviceToDevice);
        return;
    }

    int log_n = 0, tmp = N;
    while (tmp > 1) { log_n++; tmp >>= 1; }

    if (N <= 2048) {
        int threads = N / 2;
        if (threads < 32) threads = 32;
        size_t smem = 2 * (size_t)N * sizeof(float);
        if (smem > 48 * 1024) {
            cudaFuncSetAttribute(fft_shared_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }
        fft_shared_kernel<<<1, threads, smem>>>(d_signal, d_spectrum, N, log_n);
    } else {
        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        bit_reverse_kernel<<<blocks, threads>>>(d_signal, d_spectrum, N, log_n);

        int N2_blocks = (N / 2 + threads - 1) / threads;
        for (int s = 1; s <= log_n; s++) {
            int m = 1 << s;
            int m2 = m >> 1;
            fft_global_kernel<<<N2_blocks, threads>>>(d_spectrum, N, m, m2);
        }
    }
}

// ---- CPU 参考（朴素 DFT）----
void dft_cpu(const float* in, float* out, int N) {
    for (int k = 0; k < N; k++) {
        float sr = 0.0f, si = 0.0f;
        for (int n = 0; n < N; n++) {
            float ang = -2.0f * 3.14159265358979f * k * n / N;
            float wr = cosf(ang), wi = sinf(ang);
            sr += in[2*n] * wr - in[2*n+1] * wi;
            si += in[2*n] * wi + in[2*n+1] * wr;
        }
        out[2*k] = sr; out[2*k+1] = si;
    }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    // 确保 N 是 2 的幂
    int log_n = 0, tmp = N;
    while (tmp > 1) { log_n++; tmp >>= 1; }
    N = 1 << log_n;

    size_t bytes = (size_t)N * 2 * sizeof(float);
    printf("N = %d (2^%d), %.2f MB\n", N, log_n, bytes / 1e6);

    float* h_signal  = (float*)malloc(bytes);
    float* h_spectrum = (float*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)N * 2; i++)
        h_signal[i] = ((float)(rand() % 20000) - 10000.0f) / 1000.0f;

    float *d_signal, *d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_signal, bytes));
    CHECK_CUDA(cudaMalloc(&d_spectrum, bytes));
    CHECK_CUDA(cudaMemcpy(d_signal, h_signal, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    solve_gpu(d_signal, d_spectrum, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("GPU FFT time: %.3f ms\n", ms);

    CHECK_CUDA(cudaMemcpy(h_spectrum, d_spectrum, bytes, cudaMemcpyDeviceToHost));

    // 验证（小尺寸用 CPU DFT 全量比对）
    if (N <= 2048) {
        float* h_ref = (float*)malloc(bytes);
        dft_cpu(h_signal, h_ref, N);
        int fail = 0;
        for (int i = 0; i < N; i++) {
            float err_re = fabsf(h_spectrum[2*i]   - h_ref[2*i]);
            float err_im = fabsf(h_spectrum[2*i+1] - h_ref[2*i+1]);
            float tol = 0.01f * (1.0f + fabsf(h_ref[2*i]) + fabsf(h_ref[2*i+1]));
            if (err_re > tol || err_im > tol) {
                printf("FAIL at X[%d]: gpu=(%f,%f) cpu=(%f,%f) err=(%f,%f)\n",
                       i, h_spectrum[2*i], h_spectrum[2*i+1],
                       h_ref[2*i], h_ref[2*i+1], err_re, err_im);
                fail = 1; break;
            }
        }
        printf("%s\n", fail ? "FAIL" : "PASS");
        free(h_ref);
    } else {
        // 大尺寸：用 Parseval 定理验证能量守恒
        double energy_in = 0, energy_out = 0;
        for (int i = 0; i < N; i++)
            energy_in += (double)h_signal[2*i] * h_signal[2*i]
                       + (double)h_signal[2*i+1] * h_signal[2*i+1];
        for (int i = 0; i < N; i++)
            energy_out += (double)h_spectrum[2*i] * h_spectrum[2*i]
                        + (double)h_spectrum[2*i+1] * h_spectrum[2*i+1];
        energy_out /= N;
        double ratio = energy_in > 0 ? energy_out / energy_in : 1.0;
        printf("Parseval: E_in=%.2f E_out/N=%.2f ratio=%.6f %s\n",
               energy_in, energy_out, ratio,
               fabs(ratio - 1.0) < 0.001 ? "PASS" : "FAIL");
    }

    // 带宽估算
    double gb = (N <= 2048) ? 2.0 * bytes / 1e9 : (log_n + 1) * 2.0 * bytes / 1e9;
    printf("approx throughput: %.2f GB/s,  %.2f GFLOPS\n",
           gb / (ms / 1e3),
           (5.0 * N * log_n / 1e9) / (ms / 1e3));

    CHECK_CUDA(cudaFree(d_signal));
    CHECK_CUDA(cudaFree(d_spectrum));
    free(h_signal); free(h_spectrum);
    return 0;
}

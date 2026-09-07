// 69-jacobi-stencil-2d.cu —— 2D Jacobi 5 点 stencil（tiled + halo）
// 编译命令: nvcc -O3 -arch=sm_120 69-jacobi-stencil-2d.cu -o jacobi
// 运行:     ./jacobi 8192 8192

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 16

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// 朴素版：每 thread 从 global 读 4 邻（4 倍冗余读）
__global__ void jacobi_naive(const float* input, float* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows || j >= cols) return;
    int idx = i * cols + j;
    if (i == 0 || i == rows-1 || j == 0 || j == cols-1) {
        output[idx] = input[idx];   // 边界复制
    } else {
        output[idx] = 0.25f * (
            input[(i-1)*cols + j] + input[(i+1)*cols + j] +
            input[i*cols + j-1]    + input[i*cols + j+1]);
    }
}

// 优化版：tiled + 1 圈 halo —— input 载入 shared，stencil 只读 smem
__global__ void jacobi_tiled(const float* input, float* output, int rows, int cols) {
    __shared__ float smem[TILE + 2][TILE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * TILE + ty;   // 全局行
    int j = blockIdx.x * TILE + tx;   // 全局列
    int idx = i * cols + j;

    // ① 载入 tile 内部点
    if (i < rows && j < cols)
        smem[ty + 1][tx + 1] = input[idx];

    // ② 载入 halo 边界（边缘 thread 额外载 1 圈）
    // 上 halo
    if (ty == 0 && i > 0)
        smem[0][tx + 1] = input[(i - 1) * cols + j];
    // 下 halo
    if (ty == TILE - 1 && i < rows - 1)
        smem[TILE + 1][tx + 1] = input[(i + 1) * cols + j];
    // 左 halo
    if (tx == 0 && j > 0)
        smem[ty + 1][0] = input[i * cols + (j - 1)];
    // 右 halo
    if (tx == TILE - 1 && j < cols - 1)
        smem[ty + 1][TILE + 1] = input[i * cols + (j + 1)];

    __syncthreads();   // ③ 等 tile + halo 全部就位

    // ④ 计算 stencil 或复制边界
    if (i < rows && j < cols) {
        if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) {
            output[idx] = input[idx];   // 全局边界复制
        } else {
            output[idx] = 0.25f * (
                smem[ty][tx + 1]     +   // 上邻 smem[ty+1-1][tx+1]
                smem[ty + 2][tx + 1] +   // 下邻 smem[ty+1+1][tx+1]
                smem[ty + 1][tx]     +   // 左邻 smem[ty+1][tx+1-1]
                smem[ty + 1][tx + 2]);   // 右邻 smem[ty+1][tx+1+1]
        }
    }
}

int main(int argc, char** argv) {
    int rows = (argc > 1) ? atoi(argv[1]) : 8192;
    int cols = (argc > 2) ? atoi(argv[2]) : 8192;
    size_t bytes = (size_t)rows * cols * sizeof(float);
    printf("rows = %d, cols = %d  (%.1f MB)\n", rows, cols, bytes / 1e6);

    // ---- host ----
    float* hIn = (float*)malloc(bytes);
    srand(42);
    for (int i = 0; i < rows * cols; ++i)
        hIn[i] = (rand() % 2000) / 100.0f - 10.0f;

    // ---- device ----
    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, bytes));
    CHECK_CUDA(cudaMalloc(&dOut, bytes));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- CPU 验证（抽样）----
    float* hRef = (float*)malloc(bytes);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i == 0 || i == rows-1 || j == 0 || j == cols-1)
                hRef[i*cols+j] = hIn[i*cols+j];
            else
                hRef[i*cols+j] = 0.25f * (hIn[(i-1)*cols+j] + hIn[(i+1)*cols+j] +
                                          hIn[i*cols+j-1]    + hIn[i*cols+j+1]);
        }
    }

    // ---- 朴素版 ----
    cudaEventRecord(t0);
    jacobi_naive<<<grid, block>>>(dIn, dOut, rows, cols);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0.0f;
    cudaEventElapsedTime(&ms_naive, t0, t1);

    // ---- tiled halo 版 ----
    cudaEventRecord(t0);
    jacobi_tiled<<<grid, block>>>(dIn, dOut, rows, cols);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_tiled = 0.0f;
    cudaEventElapsedTime(&ms_tiled, t0, t1);

    // ---- 验证（抽样）----
    float* hOut = (float*)malloc(bytes);
    CHECK_CUDA(cudaMemcpy(hOut, dOut, bytes, cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int s = 0; s < 1000; ++s) {
        int i = rand() % rows, j = rand() % cols;
        float d = fabsf(hOut[i*cols+j] - hRef[i*cols+j]);
        if (d > max_err) max_err = d;
    }
    printf("[naive] time: %.3f ms\n", ms_naive);
    printf("[tiled ] time: %.3f ms  speedup: %.2fx  max_err: %.2e  %s\n",
           ms_tiled, ms_naive / ms_tiled, max_err, max_err < 1e-5 ? "PASS" : "FAIL");

    // 带宽估算（tiled：读 input + 写 output，各 rows×cols×4B）
    float bw_gbs = (2.0f * bytes / 1e9) / (ms_tiled / 1e3);
    printf("effective bandwidth (tiled): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dIn));
    CHECK_CUDA(cudaFree(dOut));
    free(hIn); free(hOut); free(hRef);
    return 0;
}

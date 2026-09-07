// 82-linear-recurrence.cu —— 关联扫描并行化线性递推（SSM 核心原语）
// 编译命令: nvcc -O3 -arch=sm_80 82-linear-recurrence.cu -o linear_recurrence

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP 32
#define BLOCK_SIZE 256
#define MAX_WARPS (BLOCK_SIZE / WARP)

// 仿射变换 pair: h -> a * h + b
struct Affine {
    float a, b;
};

// 关联算子 ⊙: 先应用 rhs，再应用 lhs
// f_lhs(f_rhs(h)) = a_lhs * (a_rhs * h + b_rhs) + b_lhs = (a_lhs * a_rhs) * h + (a_lhs * b_rhs + b_lhs)
__device__ __forceinline__ Affine compose(Affine lhs, Affine rhs) {
    Affine r;
    r.a = lhs.a * rhs.a;
    r.b = lhs.a * rhs.b + lhs.b;
    return r;
}

// warp 内 inclusive scan（用 __shfl_up_sync）
__device__ __forceinline__ Affine warp_inclusive_scan(Affine val) {
    int lane = threadIdx.x & (WARP - 1);
    for (int offset = 1; offset < WARP; offset <<= 1) {
        Affine other;
        other.a = __shfl_up_sync(0xFFFFFFFF, val.a, offset);
        other.b = __shfl_up_sync(0xFFFFFFFF, val.b, offset);
        if (lane >= offset) {
            val = compose(val, other);
        }
    }
    return val;
}

__global__ void linear_recurrence_kernel(
    const float* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ h,
    int B, int L)
{
    int batch = blockIdx.x;
    if (batch >= B) return;

    const float* a_row = a + (size_t)batch * L;
    const float* x_row = x + (size_t)batch * L;
    float* h_row = h + (size_t)batch * L;

    __shared__ Affine shared[MAX_WARPS];

    int tid = threadIdx.x;
    int E = (L + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ===== Phase 1: 线程内串行扫描 =====
    Affine carry = {1.0f, 0.0f};  // identity: h -> 1*h + 0
    for (int i = 0; i < E; i++) {
        int t = tid * E + i;
        if (t >= L) break;
        Affine elem;
        elem.a = (t == 0) ? 0.0f : a_row[t];  // t=0: a=0 确保 h[0]=x[0]
        elem.b = x_row[t];
        carry = compose(elem, carry);
    }

    // ===== Phase 2: block 级 exclusive scan =====
    int lane = tid & (WARP - 1);
    int warp_id = tid / WARP;

    // 2a: warp 内 inclusive scan
    Affine incl = warp_inclusive_scan(carry);

    // 2b: 每 warp 的最后一个 lane 写 carry 到 shared
    if (lane == WARP - 1)
        shared[warp_id] = incl;
    __syncthreads();

    // 2c: warp 0 对 8 个 warp carry 做 inclusive scan
    if (warp_id == 0) {
        Affine v = (lane < MAX_WARPS) ? shared[lane] : Affine{1.0f, 0.0f};
        v = warp_inclusive_scan(v);
        if (lane < MAX_WARPS)
            shared[lane] = v;
    }
    __syncthreads();

    // 2d: 计算每 thread 的 exclusive prefix
    //     = warp 间 prefix（shared[warp_id-1]）⊙ warp 内 prefix（incl[lane-1]）
    float prev_a = __shfl_up_sync(0xFFFFFFFF, incl.a, 1);
    float prev_b = __shfl_up_sync(0xFFFFFFFF, incl.b, 1);

    Affine prefix;
    if (warp_id == 0) {
        prefix = (lane == 0) ? Affine{1.0f, 0.0f} : Affine{prev_a, prev_b};
    } else {
        Affine warp_prefix = shared[warp_id - 1];
        if (lane == 0) {
            prefix = warp_prefix;
        } else {
            Affine prev = {prev_a, prev_b};
            prefix = compose(prev, warp_prefix);
        }
    }
    __syncthreads();

    // ===== Phase 3: 应用 prefix，重扫 chunk 写 h[t] =====
    carry = prefix;
    for (int i = 0; i < E; i++) {
        int t = tid * E + i;
        if (t >= L) break;
        Affine elem;
        elem.a = (t == 0) ? 0.0f : a_row[t];
        elem.b = x_row[t];
        carry = compose(elem, carry);
        h_row[t] = carry.b;
    }
}

// ===== Host 端 =====
int main() {
    // 测试: B=2, L=4
    int B = 2, L = 4;
    float h_a[]  = {0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f};
    float h_x[]  = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float h_h[8] = {0};

    // CPU 参考
    float ref[8];
    for (int b = 0; b < B; b++) {
        ref[b*L] = h_x[b*L];
        for (int t = 1; t < L; t++)
            ref[b*L+t] = h_a[b*L+t] * ref[b*L+t-1] + h_x[b*L+t];
    }

    float *d_a, *d_x, *d_h;
    cudaMalloc(&d_a, B * L * sizeof(float));
    cudaMalloc(&d_x, B * L * sizeof(float));
    cudaMalloc(&d_h, B * L * sizeof(float));
    cudaMemcpy(d_a, h_a, B * L * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, B * L * sizeof(float), cudaMemcpyHostToDevice);

    linear_recurrence_kernel<<<B, BLOCK_SIZE>>>(d_a, d_x, d_h, B, L);
    cudaDeviceSynchronize();
    cudaMemcpy(h_h, d_h, B * L * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Functional Test (B=%d, L=%d) ===\n", B, L);
    int pass = 1;
    for (int b = 0; b < B; b++) {
        printf("Batch %d: ", b);
        for (int t = 0; t < L; t++) {
            printf("%.4f ", h_h[b*L+t]);
            if (fabsf(ref[b*L+t] - h_h[b*L+t]) > 1e-5) pass = 0;
        }
        printf("\n  ref: ");
        for (int t = 0; t < L; t++) printf("%.4f ", ref[b*L+t]);
        printf("\n");
    }
    printf("%s\n\n", pass ? "✅ PASS" : "❌ FAIL");

    // ===== 性能测试: B=64, L=16384 =====
    int B2 = 64, L2 = 16384;
    float *d_a2, *d_x2, *d_h2;
    cudaMalloc(&d_a2, (size_t)B2 * L2 * sizeof(float));
    cudaMalloc(&d_x2, (size_t)B2 * L2 * sizeof(float));
    cudaMalloc(&d_h2, (size_t)B2 * L2 * sizeof(float));

    float *ha2 = (float*)malloc((size_t)B2 * L2 * sizeof(float));
    float *hx2 = (float*)malloc((size_t)B2 * L2 * sizeof(float));
    srand(42);
    for (size_t i = 0; i < (size_t)B2 * L2; i++) {
        ha2[i] = (float)rand() / RAND_MAX;  // [0, 1)
        hx2[i] = (float)(rand() % 200 - 100) / 10.0f;
    }
    cudaMemcpy(d_a2, ha2, (size_t)B2 * L2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, hx2, (size_t)B2 * L2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    linear_recurrence_kernel<<<B2, BLOCK_SIZE>>>(d_a2, d_x2, d_h2, B2, L2);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("=== Perf Test (B=%d, L=%d) ===\n", B2, L2);
    printf("Kernel time = %.3f ms\n", ms);
    printf("Data read = %.2f MB (2 passes × 2 arrays × %d×%d×4B)\n",
           2.0f * 2 * B2 * L2 * 4 / 1e6, B2, L2);
    printf("Effective bandwidth = %.2f GB/s\n",
           (2.0f * 2 * B2 * L2 * 4 + (float)B2 * L2 * 4) / (ms * 1e6));

    cudaFree(d_a); cudaFree(d_x); cudaFree(d_h);
    cudaFree(d_a2); cudaFree(d_x2); cudaFree(d_h2);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    free(ha2); free(hx2);
    return 0;
}

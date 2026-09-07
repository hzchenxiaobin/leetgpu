// 110-gae-reverse-scan.cu —— Parallel Reverse Scan (GAE)：反向 scan + 仿射复合算子
// 编译命令: nvcc -O3 -arch=sm_120 110-gae-reverse-scan.cu -o gae
// 运行:     ./gae

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

#define THREADS   256
#define WARP_SIZE 32
#define NUM_WARPS (THREADS / WARP_SIZE)        // 8
#define E_MAX     16                           // 每线程最多 16 步 → 覆盖 S ≤ 256*16 = 4096

// 仿射变换 (a,b) 表示 f(x)=a*x+b；combine(l,r)=l∘r（l 更早时刻，r 更晚时刻）
struct Op { float a, b; };
__device__ __forceinline__ Op combine(Op l, Op r) {
    return { l.a * r.a, l.a * r.b + l.b };
}

// warp 内反向 inclusive scan：lane L 持有 combine(data[L..31])
// 用 __shfl_down_sync（取右侧 lane+off）+ 递增 offset 1,2,4,8,16
__device__ __forceinline__ Op warp_rev_inclusive_scan(Op v) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    for (int off = 1; off < WARP_SIZE; off <<= 1) {
        Op r;
        r.a = __shfl_down_sync(0xffffffff, v.a, off);
        r.b = __shfl_down_sync(0xffffffff, v.b, off);
        if (lane + off < WARP_SIZE)
            v = combine(v, r);                         // v = combine(自身及左侧, 右侧累积)
    }
    return v;                                          // lane L = combine(data[L..31])
}

// 朴素版：一序列一线程，单线程串行递推（B 个线程，用于对比基准）
__global__ void gae_naive_kernel(const float* rewards, const float* values, float* adv,
                                 float gamma, float lam, int B, int S) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* rw = rewards + b * S;
    const float* vl = values  + b * S;
    float*       ad = adv     + b * S;
    float decay = gamma * lam;
    float last = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
        float nv = (t + 1 < S) ? vl[t + 1] : 0.0f;
        last = (rw[t] + gamma * nv - vl[t]) + decay * last;
        ad[t] = last;
    }
}

// 优化版：一序列一 block，三段式反向 scan（chunk→warp→block）
__global__ void gae_kernel(const float* rewards, const float* values, float* adv,
                           float gamma, float lam, int S) {
    int b = blockIdx.x;                          // 一个 block 处理一条序列
    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warpId = tid >> 5;

    const float* rw = rewards + b * S;
    const float* vl = values  + b * S;
    float*       ad = adv     + b * S;
    float decay = gamma * lam;
    int E = (S + THREADS - 1) / THREADS;         // 每线程步数（S≤4096 时 E≤16=E_MAX）

    float a_loc[E_MAX], b_loc[E_MAX];            // 本 chunk 反向 inclusive：(a=decay连乘, b=累积偏置)

    // ① chunk 内串行反向 scan（高 j → 低 j），并算出 chunk 聚合 local_inclusive[0]
    float na = 1.0f, nb = 0.0f;                  // a_loc[E], b_loc[E] = identity (1,0)
    for (int j = E - 1; j >= 0; --j) {
        int idx = tid * E + j;
        Op elem;
        if (idx < S) {
            float nv = (idx + 1 < S) ? vl[idx + 1] : 0.0f;   // next_value：末步为 0
            float delta = rw[idx] + gamma * nv - vl[idx];
            elem = { decay, delta };
        } else {
            elem = { 1.0f, 0.0f };               // 越界 step 补 identity，不污染复合
        }
        float a  = elem.a * na;                  // combine(elem, {na,nb})
        float bb = elem.a * nb + elem.b;
        a_loc[j] = a; b_loc[j] = bb;
        na = a; nb = bb;
    }
    Op chunk_agg = { na, nb };                   // 本 chunk 全体聚合 = local_inclusive[0]

    // ② warp 内反向 inclusive scan（聚合 32 个 chunk）
    Op inc = warp_rev_inclusive_scan(chunk_agg); // inc = combine(chunk[tid..warp末])
    // excl_lane = 本 warp 内"严格右侧" chunk 聚合 = inc[lane+1]，末 lane 为 identity
    Op excl_lane = { 1.0f, 0.0f };
    // 所有 lane 必须参与 __shfl_down_sync（mask=0xffffffff），否则死锁
    float excl_a = __shfl_down_sync(0xffffffff, inc.a, 1);
    float excl_b = __shfl_down_sync(0xffffffff, inc.b, 1);
    if (lane + 1 < WARP_SIZE) {
        excl_lane.a = excl_a;
        excl_lane.b = excl_b;
    }
    // 本 warp 总聚合（lane 0 持有 combine(整个 warp)）写 shared
    __shared__ Op s_warp[NUM_WARPS];
    if (lane == 0) s_warp[warpId] = inc;
    __syncthreads();

    // ③ block 级：对 8 个 warp 聚合做反向 inclusive scan → 每 warp 的"严格右侧 warp"聚合
    __shared__ Op s_carry[NUM_WARPS];
    if (tid == 0) {
        Op r = { 1.0f, 0.0f };                   // 严格右侧 warp 的累积，从 identity 起
        for (int w = NUM_WARPS - 1; w >= 0; --w) {
            s_carry[w] = r;                      // excl_warp_carry[w] = w+1..NUM_WARPS-1 的聚合
            r = combine(s_warp[w], r);           // inclusive[w] = combine(s_warp[w..end])
        }
    }
    __syncthreads();
    Op excl_warp = s_carry[warpId];              // 本 warp 严格右侧（更晚 chunk）的聚合

    // ④ chunk_carry = combine(warp内右侧, warp间右侧)
    Op carry = combine(excl_lane, excl_warp);

    // ⑤ 写回：A_t = a_loc[j]·carry.b + b_loc[j]
    for (int j = 0; j < E; ++j) {
        int idx = tid * E + j;
        if (idx < S)
            ad[idx] = a_loc[j] * carry.b + b_loc[j];
    }
}

// 串行 fallback：S 超出 THREADS*E_MAX 时保底（一序列一线程），保证任意 S 都正确
__global__ void gae_serial_kernel(const float* rewards, const float* values, float* adv,
                                  float gamma, float lam, int B, int S) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* rw = rewards + b * S;
    const float* vl = values  + b * S;
    float*       ad = adv     + b * S;
    float decay = gamma * lam, last = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
        float nv = (t + 1 < S) ? vl[t + 1] : 0.0f;
        last = (rw[t] + gamma * nv - vl[t]) + decay * last;
        ad[t] = last;
    }
}

int main(int argc, char** argv) {
    int B = (argc > 1) ? atoi(argv[1]) : 64;
    int S = (argc > 2) ? atoi(argv[2]) : 4096;
    float gamma = (argc > 3) ? (float)atof(argv[3]) : 0.99f;
    float lam   = (argc > 4) ? (float)atof(argv[4]) : 0.95f;
    size_t n = (size_t)B * S, bytes = n * sizeof(float);
    printf("B = %d, S = %d  (%.2f MB per tensor)\n", B, S, bytes / 1e6);

    float *hRw = (float*)malloc(bytes), *hVl = (float*)malloc(bytes),
          *hAd = (float*)malloc(bytes), *hRef = (float*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < n; ++i) {
        hRw[i] = (float)((rand() % 2000) - 1000) / 100.0f;
        hVl[i] = (float)((rand() % 2000) - 1000) / 100.0f;
    }

    // CPU 参考
    float decay = gamma * lam;
    for (int b = 0; b < B; ++b) {
        float last = 0.0f;
        for (int t = S - 1; t >= 0; --t) {
            int idx = b * S + t;
            float nv = (t + 1 < S) ? hVl[idx + 1] : 0.0f;
            last = (hRw[idx] + gamma * nv - hVl[idx]) + decay * last;
            hRef[idx] = last;
        }
    }

    float *dRw, *dVl, *dAd;
    CHECK_CUDA(cudaMalloc(&dRw, bytes));
    CHECK_CUDA(cudaMalloc(&dVl, bytes));
    CHECK_CUDA(cudaMalloc(&dAd, bytes));
    CHECK_CUDA(cudaMemcpy(dRw, hRw, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dVl, hVl, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // ---- 优化版 ----
    cudaEventRecord(t0);
    if (S <= THREADS * E_MAX)
        gae_kernel<<<B, THREADS>>>(dRw, dVl, dAd, gamma, lam, S);
    else
        gae_serial_kernel<<<(B + 255) / 256, 256>>>(dRw, dVl, dAd, gamma, lam, B, S);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_opt = 0; cudaEventElapsedTime(&ms_opt, t0, t1);
    CHECK_CUDA(cudaMemcpy(hAd, dAd, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)hAd[i] - hRef[i]);
        if (d > max_err) max_err = d;
    }
    printf("[parallel]  time: %.4f ms  max_err: %.3e  %s\n", ms_opt, max_err,
           max_err < 1e-3 * (1 + fabs(hRef[n - 1])) ? "PASS" : "FAIL");

    // ---- 朴素版对比 ----
    cudaEventRecord(t0);
    gae_naive_kernel<<<(B + 255) / 256, 256>>>(dRw, dVl, dAd, gamma, lam, B, S);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    float bw_gbs = (3.0 * bytes / 1e9) / (ms_opt / 1e3);   // 读 rewards+values + 写 advantages
    printf("[naive]     time: %.4f ms  speedup: %.2fx\n", ms_naive, ms_naive / ms_opt);
    printf("I/O bandwidth (parallel): %.1f GB/s\n", bw_gbs);

    CHECK_CUDA(cudaFree(dRw)); CHECK_CUDA(cudaFree(dVl)); CHECK_CUDA(cudaFree(dAd));
    free(hRw); free(hVl); free(hAd); free(hRef);
    return 0;
}

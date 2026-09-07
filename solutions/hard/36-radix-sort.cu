// 36-radix-sort.cu —— Radix-2 基数排序：32 趟稳定二分划分（count → scan → scatter）
// 编译命令: nvcc -O3 -arch=sm_120 36-radix-sort.cu -o radixsort
// 运行:     ./radixsort [N]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

#define TILE     1024
#define WARP     32
#define NUM_WARP (TILE / WARP)            // 32

// 块内 exclusive scan（复用 #16 Prefix Sum 的三阶段模板）
// 返回本线程的 exclusive 值；块总和（inclusive）留在 sh_warp[NUM_WARP-1]
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t val, uint32_t* sh_warp) {
    int tid    = threadIdx.x;
    int lane   = tid & (WARP - 1);
    int warpId = tid >> 5;
    uint32_t orig = val;
    // ① warp 内 inclusive scan（__shfl_up_sync，5 步）
    for (int off = 1; off < WARP; off <<= 1) {
        uint32_t t = __shfl_up_sync(0xffffffff, val, off);
        if (lane >= off) val += t;
    }
    if (lane == WARP - 1) sh_warp[warpId] = val;        // 各 warp 总和
    __syncthreads();
    // ② warp 0 对 32 个 warp 总和做 inclusive scan
    if (warpId == 0) {
        uint32_t w = sh_warp[lane];
        for (int off = 1; off < NUM_WARP; off <<= 1) {
            uint32_t t = __shfl_up_sync(0xffffffff, w, off);
            if (lane >= off) w += t;
        }
        sh_warp[lane] = w;                               // inclusive；sh_warp[31] = 块总和
    }
    __syncthreads();
    // ③ 加上本 warp 之前的总和 → block inclusive；exclusive = inclusive − own
    uint32_t warp_excl = (warpId == 0) ? 0 : sh_warp[warpId - 1];
    uint32_t incl = val + warp_excl;
    return incl - orig;
}

// ① count：每 block 算本 tile 的 0 计数 → d_count0[blockIdx]
__global__ void count_kernel(const uint32_t* src, int N, int bit, uint32_t* d_count0) {
    int tid = threadIdx.x;
    int gj  = blockIdx.x * TILE + tid;
    uint32_t key   = (gj < N) ? src[gj] : 0u;
    uint32_t pred0 = (gj < N) && (((key >> bit) & 1u) == 0u) ? 1u : 0u;
    __shared__ uint32_t sh_warp[NUM_WARP];
    block_excl_scan(pred0, sh_warp);                     // rank0 此处不需要，只用总和
    __syncthreads();
    if (tid == 0) d_count0[blockIdx.x] = sh_warp[NUM_WARP - 1];
}

// ② scan：对 d_count0[numBlocks] 做 exclusive scan → d_excl0[]；总和 → *d_total
__global__ void scan_count_kernel(const uint32_t* d_count0, uint32_t* d_excl0,
                                  uint32_t* d_total, int numBlocks) {
    int tid = threadIdx.x;
    uint32_t v = (tid < numBlocks) ? d_count0[tid] : 0u;
    __shared__ uint32_t sh_warp[NUM_WARP];
    uint32_t excl = block_excl_scan(v, sh_warp);
    __syncthreads();
    if (tid < numBlocks) d_excl0[tid] = excl;
    if (tid == 0) *d_total = sh_warp[NUM_WARP - 1];      // 全局 0 总数 Z
}

// ③ scatter：重算 rank0，按位稳定散列到 d_out
__global__ void scatter_kernel(const uint32_t* src, uint32_t* out, int N, int bit,
                               const uint32_t* d_excl0, const uint32_t* d_total) {
    int tid = threadIdx.x;
    int gj  = blockIdx.x * TILE + tid;
    uint32_t key   = (gj < N) ? src[gj] : 0u;
    uint32_t pred0 = (gj < N) && (((key >> bit) & 1u) == 0u) ? 1u : 0u;
    __shared__ uint32_t sh_warp[NUM_WARP];
    uint32_t rank0 = block_excl_scan(pred0, sh_warp);    // 本 block 内、本线程之前 0 的个数
    uint32_t excl0 = d_excl0[blockIdx.x];                // 全局在此 block 之前的 0 总数
    uint32_t Z     = *d_total;                           // 全局 0 总数
    if (gj < N) {
        uint32_t pos;
        if (((key >> bit) & 1u) == 0u) {
            pos = excl0 + rank0;                         // 0 → 左侧 [0, Z)
        } else {
            uint32_t rank1   = (uint32_t)tid - rank0;    // 本 block 内、本线程之前 1 的个数
            uint32_t one_base = Z + (uint32_t)blockIdx.x * (uint32_t)TILE - excl0;
            pos = one_base + rank1;                      // 1 → 右侧 [Z, N)
        }
        out[pos] = key;
    }
}

// LeetGPU 提交接口：原地升序排序 data[0..N)（uint32）
extern "C" void solve(uint32_t* data, int N) {
    if (N <= 1) return;
    int numBlocks = (N + TILE - 1) / TILE;
    uint32_t *buf, *d_count0, *d_excl0, *d_total;
    CHECK_CUDA(cudaMalloc(&buf,      (size_t)N * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_count0, (size_t)numBlocks * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_excl0,  (size_t)numBlocks * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_total,  sizeof(uint32_t)));

    uint32_t* src = data;
    uint32_t* out = buf;
    for (int bit = 0; bit < 32; ++bit) {
        count_kernel<<<numBlocks, TILE>>>(src, N, bit, d_count0);
        scan_count_kernel<<<1, TILE>>>(d_count0, d_excl0, d_total, numBlocks);
        scatter_kernel<<<numBlocks, TILE>>>(src, out, N, bit, d_excl0, d_total);
        uint32_t* tmp = src; src = out; out = tmp;       // ping-pong
    }
    // 32 趟为偶数 → 结果落回 data；若改奇数趟需补一次 D2D 拷贝
    if (src != data)
        CHECK_CUDA(cudaMemcpy(data, src, (size_t)N * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(buf));
    CHECK_CUDA(cudaFree(d_count0));
    CHECK_CUDA(cudaFree(d_excl0));
    CHECK_CUDA(cudaFree(d_total));
}

// ---------------- 本地自测 ----------------
int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t bytes = (size_t)N * sizeof(uint32_t);
    printf("N = %d  (%.2f MB)\n", N, bytes / 1e6);

    uint32_t *hData = (uint32_t*)malloc(bytes), *hRef = (uint32_t*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; ++i) {
        hData[i] = (uint32_t)((rand() << 16) ^ rand());  // 充分随机的 32 位
        hRef[i]  = hData[i];
    }
    std::sort(hRef, hRef + N);                           // CPU 参考

    uint32_t* dData;
    CHECK_CUDA(cudaMalloc(&dData, bytes));
    CHECK_CUDA(cudaMemcpy(dData, hData, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    solve(dData, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);

    CHECK_CUDA(cudaMemcpy(hData, dData, bytes, cudaMemcpyDeviceToHost));

    // 验证：逐元素比对（排序是精确置换，误差应为 0）
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (hData[i] != hRef[i]) { ok = false; break; }
    }
    printf("[radix-2] time: %.3f ms  %s\n", ms, ok ? "PASS" : "FAIL");
    printf("throughput: %.2f M elem/s\n", N / ms / 1000.0);

    CHECK_CUDA(cudaFree(dData));
    free(hData); free(hRef);
    return 0;
}

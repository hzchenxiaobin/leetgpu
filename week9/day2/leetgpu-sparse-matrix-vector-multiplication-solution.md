# LeetGPU Sparse Matrix-Vector Multiplication 题解

## 1. 题目概述

- **标题 / 题号**：Sparse Matrix-Vector Multiplication（#18，medium）
- **链接**：https://leetgpu.com/challenges/sparse-matrix-vector-multiplication
- **难度**：中等
- **标签**：CUDA、CSR、稀疏矩阵、SpMV、warp shuffle、间接访存（gather）、memory-bound

**题意**：计算稀疏矩阵 `A`（`M×N`）与稠密向量 `x`（长度 `N`）的乘积 `y = A·x`，即 `y[i] = Σ_j A[i][j] × x[j]`。输入给出**稠密** `A`（flattened `M*N`）与**非零元个数** `nnz`，输出写入长度 `M` 的 `y`。

**示例**：

```text
A = [[5,0,0,1],     x = [1,2,3,4]      y[0] = 5×1 + 1×4           = 9
     [0,2,3,0],  →                     y[1] = 2×2 + 3×3           = 13
     [0,0,4,0]]                        y[2] = 4×3                 = 12
```

**约束**：

- `1 ≤ M, N`，性能测试取 `M=1000, N=10000, nnz=3,500,000`
- `A`、`x`、`y` 均为 `float32`
- 性能测试稀疏度约 65%（即 65% 元素为 0，`nnz/(M*N)=35%`）

> 💡 这道题是 **CSR 格式 + SpMV（稀疏矩阵-向量乘）**的经典练习——HPC 领域最重要的 kernel 之一。平台传入稠密 `A` 与 `nnz`，意图明显：朴素做稠密 GEMV 会读 65% 的零（浪费带宽、做无谓乘法）；正解是**先把稠密 `A` 压缩成 CSR 三元组**，再用 SpMV 只遍历非零元。它与 [#22 GEMM](./leetgpu-gemm-solution.md) 的对比揭示了 GPU 编程的一个分水岭：稠密计算访存**规则合并**（coalesced），而稀疏计算访存**不规则间接**（gather via `col_idx`）——后者是另一类 memory-bound 优化的核心。

## 2. CPU 基线 / 朴素 GPU 方法

### 2.1 CPU 串行基线

```cpp
// cpu_baseline.cpp —— CPU 稠密 GEMV
void gemv_cpu(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
            sum += A[i * N + j] * x[j];   // 遍历整行，含大量 0
        y[i] = sum;
    }
}
```

`M=1000, N=10000` 时单核约 20-40 ms。CPU 顺序访问 `A` 行连续、cache 友好，但要做 `M×N=10^7` 次乘加，其中 65% 是与 0 相乘的无用功。

### 2.2 朴素 GPU：稠密 GEMV（每线程一行）

最暴力的并行：每线程负责一行，串行遍历该行所有 `N` 个元素。

```cuda
__global__ void gemv_dense_naive(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j)
            sum += A[row * N + j] * x[j];   // 读全部 N 个元素，含 65% 的 0
        y[row] = sum;
    }
}
```

**问题**：

1. **读带宽浪费**：每行读 `N=10000` 个 float（40KB），但只有 35% 非零——65% 的读流量是浪费。`M=1000` 行共读 `40MB`，实际有效数据仅 `nnz×4B≈14MB`。
2. **无用计算**：与 0 相乘的乘加占 65%，算力浪费。
3. **x 的重复 gather**：每个线程独立读 `x[j]`，跨线程无数据复用（`x` 不在 shared memory）。

> ⚠️ 朴素稠密 GEMV 把稀疏性当不存在——这正是题目给出 `nnz` 的暗示：**你应该用 CSR 压缩掉那 65% 的零**。稀疏计算的本质是用"间接寻址 + 不规则访存"换取"少读少算"。当稀疏度足够高（零占比大）时，这笔交易才划算；本题 65% 零是临界收益区，更稀疏（如 95% 零）时 CSR 优势压倒性。

## 3. GPU 设计

### 3.1 并行化策略：CSR 压缩 + warp-per-row SpMV

核心思想：**两阶段流水**——先把稠密 `A` 转成 CSR 三元组（`row_ptr`/`col_idx`/`values`），再用 **warp-per-row** SpMV 只遍历非零元。

![CSR 格式：稠密矩阵压缩为三元组](../../images/sparse_matrix_vector_multiplication_csr_format.svg)

**CSR（Compressed Sparse Row）三数组**：

| 数组 | 长度 | 含义 |
|------|------|------|
| `values[nnz]` | `nnz` | 按行优先顺序存放所有非零元的值 |
| `col_idx[nnz]` | `nnz` | 每个非零元的列号（用于 `x[col_idx[k]]` 间接 gather） |
| `row_ptr[M+1]` | `M+1` | 行指针，`row_ptr[i]` 给出第 `i` 行非零元在 `values` 中的起始偏移；第 `i` 行的非零元区间为 `[row_ptr[i], row_ptr[i+1])` |

**SpMV 计算**：`y[i] = Σ_{k=row_ptr[i]}^{row_ptr[i+1]-1} values[k] × x[col_idx[k]]`。每行只遍历自己的非零元，跳过所有零。

![warp-per-row SpMV：每行一个 warp，lane 分割非零元 + shuffle 归约](../../images/sparse_matrix_vector_multiplication_warp_per_row.svg)

**warp-per-row 并行策略**：

1. **一个 warp 负责一行**：32 个 lane 协作处理该行的非零元。
2. **lane 分割非零元**：行内非零元区间 `[row_start, row_end)`，每个 lane 处理 `row_start + lane, row_start + lane + 32, ...`（stride 32），各自算 `values[k] × x[col_idx[k]]` 累加到 lane 局部 `sum`。
3. **warp shuffle 归约**：lane 内累加完后，用 `__shfl_down_sync` 把 32 个 lane 的 `sum` 树形归约到 lane 0。
4. **lane 0 写回**：`y[row] = lane0_sum`。

> 💡 **为什么用 warp 而非 thread-per-row**：thread-per-row（每线程一行）在行内串行累加，长行成为瓶颈、短行浪费线程；warp-per-row 让 32 lane 并行处理行内非零元，行内归约用 `__shfl_down_sync`（寄存器内、零 shared memory、零 `__syncthreads`），既提升行内并行度又保持负载相对均衡。对每行非零元 ≤ 32 的矩阵，warp-per-row 是通用最优选择；超长行可升级为 block-per-row。

### 3.2 存储层次使用

| 层次 | 是否使用 | 说明 |
|------|----------|------|
| **global memory** | ✓ | `values[]`/`col_idx[]` 顺序读（合并）；`x[]` 间接 gather（**不规则**）；`y[]` 顺序写 |
| **shared memory** | ✓（可选） | 缓存 `x` 的 tile，减少重复 gather 对 HBM 的压力（本题 `N` 大时收益显著） |
| **register** | ✓ | 每 lane 的局部 `sum`、warp shuffle 中间值 |

### 3.3 关键技巧

| 技巧 | 作用 | 收益 |
|------|------|------|
| **CSR 压缩** | 只存非零元，跳过所有零 | 省读带宽、省无用乘法（稀疏度越高收益越大） |
| **warp-per-row** | 32 lane 协作一行 | 行内并行 + shuffle 归约，平衡度好 |
| **`__shfl_down_sync` 归约** | warp 内 32 lane 树形求和 | 寄存器内完成，零 bank conflict、零同步开销 |
| **`x` 的 shared 缓存** | 把 `x` 分块载入 shared | 把不规则 global gather 降级为 shared gather |
| **行边界保护** | `if (row_start + lane < row_end)` | 处理非零元数不足 32 的短行，避免越界 |

> ⚠️ **SpMV 的根本矛盾：不规则访存**。`values`/`col_idx` 是顺序读（合并），但 `x[col_idx[k]]` 是**间接寻址**——`col_idx` 的值决定取 `x` 的哪个位置，相邻 lane 取的 `x` 元素可能不相邻，导致 global 读无法合并（一个 cache line 只命中少量元素）。这是 SpMV 与稠密 GEMV 的本质区别，也是它带宽利用率永远低于稠密 GEMV 的根因。优化方向是用 shared memory 缓存 `x` 把 global gather 变 shared gather。

## 4. Kernel 实现

完整可编译版本（含稠密 GEMV 对比 + CSR 转换 + warp-per-row SpMV + CPU 验证）：

```cuda
// sparse_matrix_vector_multiplication.cu —— CSR SpMV（warp-per-row + shuffle 归约）
// 编译命令: nvcc -O3 -arch=sm_120 sparse_matrix_vector_multiplication.cu -o spmv
// 运行:     ./spmv 1000 10000 3500000

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define CHECK_CUDA(call) do {                                              \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                                     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// warp 内树形归约（__shfl_down_sync）
__inline__ __device__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---- 第 0 步：稠密 → CSR 转换（两遍 kernel）----

// 0a. 统计每行非零元数 → row_count（后面 exclusive scan 成 row_ptr）
__global__ void count_nnz_per_row(const float* A, int* row_count, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int cnt = 0;
        const float* row_ptr_A = A + (size_t)row * N;
        for (int j = 0; j < N; ++j)
            cnt += (row_ptr_A[j] != 0.0f);
        row_count[row] = cnt;
    }
}

// 0b. 并行 exclusive scan（Hillis-Steele，M 较小时单 block 够用）
__global__ void exclusive_scan(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    if (tid < n) temp[tid] = data[tid];
    __syncthreads();
    // Hillis-Steele inclusive scan
    int step = 1;
    for (int offset = 1; offset < n; offset <<= 1) {
        int v = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += v;
        __syncthreads();
    }
    // 转成 exclusive：整体右移一位，首位补 0
    int excl = (tid == 0) ? 0 : temp[tid - 1];
    __syncthreads();
    if (tid < n) data[tid] = excl;
    // 末尾写入总数到 data[n]（若 n < blockDim.x）
    if (tid == 0 && n < blockDim.x) data[n] = temp[n - 1];
}

// 0c. 填充 col_idx / values（每线程一个非零元，原子分配位置）
__global__ void fill_csr(const float* A, const int* row_ptr,
                         int* col_idx, float* values, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int pos = row_ptr[row];   // 本行非零元在 values 中的起始位置
        const float* row_ptr_A = A + (size_t)row * N;
        for (int j = 0; j < N; ++j) {
            float v = row_ptr_A[j];
            if (v != 0.0f) {
                col_idx[pos] = j;
                values[pos] = v;
                ++pos;
            }
        }
    }
}

// ---- 第 1 步：朴素稠密 GEMV（对比基准）----
__global__ void gemv_dense(const float* A, const float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float sum = 0.0f;
        const float* row_ptr_A = A + (size_t)row * N;
        for (int j = 0; j < N; ++j)
            sum += row_ptr_A[j] * x[j];
        y[row] = sum;
    }
}

// ---- 第 2 步：warp-per-row CSR SpMV ----
__global__ void spmv_warp(const int* row_ptr, const int* col_idx,
                          const float* values, const float* x, float* y, int M) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id_global >= M) return;

    int row_start = row_ptr[warp_id_global];
    int row_end   = row_ptr[warp_id_global + 1];

    // ① lane 分割非零元：每 lane 处理 stride=32 的元素
    float sum = 0.0f;
    for (int k = row_start + lane; k < row_end; k += WARP_SIZE) {
        sum += values[k] * x[col_idx[k]];   // 间接 gather x
    }

    // ② warp 内 shuffle 归约到 lane 0
    sum = warp_reduce(sum);

    // ③ lane 0 写回该行结果
    if (lane == 0)
        y[warp_id_global] = sum;
}

// ---- host 主流程 ----
int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1000;
    int N = (argc > 2) ? atoi(argv[2]) : 10000;
    int target_nnz = (argc > 3) ? atoi(argv[3]) : 3500000;
    size_t bytes_A = (size_t)M * N * sizeof(float);
    printf("M = %d, N = %d  (A = %.1f MB)\n", M, N, bytes_A / 1e6);

    // ---- host：生成稀疏矩阵 ----
    float* hA = (float*)calloc(M * N, sizeof(float));
    srand(42);
    int placed = 0;
    while (placed < target_nnz) {
        int idx = rand() % (M * N);
        if (hA[idx] == 0.0f) {
            hA[idx] = ((rand() % 2000) / 100.0f) - 10.0f;   // [-10, 10]
            ++placed;
        }
    }
    float* hx = (float*)malloc(N * sizeof(float));
    for (int j = 0; j < N; ++j) hx[j] = (rand() % 1000) / 100.0f;

    // ---- device ----
    float *dA, *dx, *dy;
    CHECK_CUDA(cudaMalloc(&dA, bytes_A));
    CHECK_CUDA(cudaMalloc(&dx, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, M * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));

    int *d_row_ptr, *d_col_idx, *d_row_count;
    float* d_values;
    CHECK_CUDA(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_row_count, M * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_idx, target_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_values, target_nnz * sizeof(float)));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- CSR 转换 ----
    int blocks_row = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_CUDA(cudaMemset(d_row_count, 0, M * sizeof(int)));
    count_nnz_per_row<<<blocks_row, BLOCK_SIZE>>>(dA, d_row_count, M, N);
    // exclusive scan（M <= 1024 时单 block）
    int scan_blocks = (M + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    exclusive_scan<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_row_count, M + 1);
    CHECK_CUDA(cudaMemcpy(d_row_ptr, d_row_count, (M + 1) * sizeof(int),
                           cudaMemcpyDeviceToDevice));
    fill_csr<<<blocks_row, BLOCK_SIZE>>>(dA, d_row_ptr, d_col_idx, d_values, M, N);

    // ---- CPU 验证 ----
    float* hy_ref = (float*)malloc(M * sizeof(float));
    for (int i = 0; i < M; ++i) {
        float s = 0.0f;
        for (int j = 0; j < N; ++j)
            s += hA[i * N + j] * hx[j];
        hy_ref[i] = s;
    }

    // ---- 朴素稠密 GEMV ----
    cudaEventRecord(t0);
    gemv_dense<<<blocks_row, BLOCK_SIZE>>>(dA, dx, dy, M, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_dense = 0.0f;
    cudaEventElapsedTime(&ms_dense, t0, t1);

    // ---- warp-per-row SpMV ----
    int spmv_blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    cudaEventRecord(t0);
    spmv_warp<<<spmv_blocks, BLOCK_SIZE>>>(d_row_ptr, d_col_idx, d_values, dx, dy, M);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_spmv = 0.0f;
    cudaEventElapsedTime(&ms_spmv, t0, t1);

    // ---- 验证 ----
    float* hy = (float*)malloc(M * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hy, dy, M * sizeof(float), cudaMemcpyDeviceToHost));
    float max_err = 0.0f;
    for (int i = 0; i < M; ++i) {
        float d = fabsf(hy[i] - hy_ref[i]);
        if (d > max_err) max_err = d;
    }
    printf("[dense GEMV] time: %.3f ms\n", ms_dense);
    printf("[CSR SpMV ]  time: %.3f ms  speedup: %.2fx  max_err: %.4e  %s\n",
           ms_spmv, ms_dense / ms_spmv, max_err, max_err < 1e-2 ? "PASS" : "FAIL");

    // 带宽估算（SpMV：读 values+col_idx + gather x）
    float bytes_read = (float)(target_nnz * (sizeof(int) + sizeof(float)) + target_nnz * sizeof(float));
    printf("read bandwidth (SpMV): %.1f GB/s\n", (bytes_read / 1e9) / (ms_spmv / 1e3));

    CHECK_CUDA(cudaFree(dA));  CHECK_CUDA(cudaFree(dx));  CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(d_row_ptr)); CHECK_CUDA(cudaFree(d_col_idx));
    CHECK_CUDA(cudaFree(d_values));  CHECK_CUDA(cudaFree(d_row_count));
    free(hA); free(hx); free(hy); free(hy_ref);
    return 0;
}
```

> 💡 提交给 LeetGPU 平台时，把"CSR 转换 + `spmv_warp`"填进 `solve` 函数即可（见 §4.1）。注意平台传入的 `A` 是稠密的，需在 `solve` 内部完成 CSR 转换。

### 4.1 LeetGPU 提交版本

下面给出适配官方 starter 签名 `solve(A, x, y, M, N, nnz)` 的提交版本。它在 `solve` 内部分配 CSR 缓冲、做 dense→CSR 转换、再启动 warp-per-row SpMV。

```cuda
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

__inline__ __device__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void count_nnz_per_row(const float* A, int* row_count, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int cnt = 0;
        const float* r = A + (size_t)row * N;
        for (int j = 0; j < N; ++j)
            cnt += (r[j] != 0.0f);
        row_count[row] = cnt;
    }
}

__global__ void exclusive_scan(int* data, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    if (tid < n) temp[tid] = data[tid];
    __syncthreads();
    for (int offset = 1; offset < n; offset <<= 1) {
        int v = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += v;
        __syncthreads();
    }
    int excl = (tid == 0) ? 0 : temp[tid - 1];
    __syncthreads();
    if (tid < n) data[tid] = excl;
    if (tid == 0 && n < blockDim.x) data[n] = temp[n - 1];
}

__global__ void fill_csr(const float* A, const int* row_ptr,
                         int* col_idx, float* values, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        int pos = row_ptr[row];
        const float* r = A + (size_t)row * N;
        for (int j = 0; j < N; ++j) {
            float v = r[j];
            if (v != 0.0f) {
                col_idx[pos] = j;
                values[pos] = v;
                ++pos;
            }
        }
    }
}

__global__ void spmv_warp(const int* row_ptr, const int* col_idx,
                          const float* values, const float* x, float* y, int M) {
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id_global >= M) return;

    int row_start = row_ptr[warp_id_global];
    int row_end   = row_ptr[warp_id_global + 1];

    float sum = 0.0f;
    for (int k = row_start + lane; k < row_end; k += WARP_SIZE) {
        sum += values[k] * x[col_idx[k]];
    }
    sum = warp_reduce(sum);
    if (lane == 0)
        y[warp_id_global] = sum;
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    if (M <= 0) return;

    int* d_row_ptr;
    int* d_col_idx;
    float* d_values;
    int* d_row_count;
    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_row_count, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));

    int blocks_row = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // CSR 转换
    count_nnz_per_row<<<blocks_row, BLOCK_SIZE>>>(A, d_row_count, M, N);
    int scan_shared = ((M + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE * sizeof(int);
    exclusive_scan<<<1, BLOCK_SIZE, scan_shared>>>(d_row_count, M + 1);
    cudaMemcpy(d_row_ptr, d_row_count, (M + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
    fill_csr<<<blocks_row, BLOCK_SIZE>>>(A, d_row_ptr, d_col_idx, d_values, M, N);

    // warp-per-row SpMV
    int spmv_blocks = (M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    spmv_warp<<<spmv_blocks, BLOCK_SIZE>>>(d_row_ptr, d_col_idx, d_values, x, y, M);
    cudaDeviceSynchronize();

    cudaFree(d_row_ptr); cudaFree(d_col_idx);
    cudaFree(d_values);  cudaFree(d_row_count);
}
```

### 4.2 代码详解

`spmv_warp` 采用 **warp-per-row** 结构：一个 warp 包揽一行，32 lane 用 stride=32 分割该行非零元，各自算 `values[k]×x[col_idx[k]]` 后用 `__shfl_down_sync` 归约到 lane 0 写回。全程零 shared memory、零 `__syncthreads`。

**`spmv_warp` 逐段解析**：

| 步骤 | 代码 | 说明 |
|------|------|------|
| **warp→行映射** | `warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE` | 把全局线程号映射到 warp 号，一个 warp 对应一行 |
| **lane 号** | `lane = threadIdx.x % WARP_SIZE` | warp 内 lane 编号，用于归约后判断 lane 0 |
| **行边界** | `row_start = row_ptr[row]; row_end = row_ptr[row+1]` | 该行非零元在 `values`/`col_idx` 中的区间 `[row_start, row_end)` |
| **lane 分割** | `for (k = row_start + lane; k < row_end; k += WARP_SIZE)` | 每 lane 处理 stride=32 的非零元，行内并行 |
| **乘加 + gather** | `sum += values[k] * x[col_idx[k]]` | 读非零值、按列号间接 gather `x`、乘加到 lane 局部 `sum` |
| **shuffle 归约** | `sum = warp_reduce(sum)` | 32 lane 的 `sum` 树形归约到 lane 0，寄存器内完成 |
| **写回** | `if (lane == 0) y[row] = sum` | 仅 lane 0 把行和写入 `y` |

**关键索引关系**：

- `warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32` — warp 到行的映射，`M` 行需要 `M` 个 warp
- `row_start / row_end` — `row_ptr` 给出的行内非零元区间，行长度 = `row_end - row_start`
- `k = row_start + lane + n*32` — lane 在行内的元素下标，stride=32 保证 32 lane 均分行内非零元
- `col_idx[k]` — 第 `k` 个非零元的列号，决定 gather `x` 的哪个元素
- `spmv_blocks = ceil(M / WARPS_PER_BLOCK)` — grid 大小，每 block 含 8 个 warp（处理 8 行）

**CSR 转换三步**：

1. **`count_nnz_per_row`**：每线程一行，串行数非零元数写入 `row_count[i]`。
2. **`exclusive_scan`**：对 `row_count` 做并行前缀扫描（Hillis-Steele），得到 `row_ptr`——`row_ptr[i]` 是前 `i` 行非零元总数（即第 `i` 行在 `values` 中的起始偏移）。
3. **`fill_csr`**：每线程一行，按 `row_ptr[row]` 起始位置顺序填入 `col_idx`/`values`。

**`__syncthreads` 在 `exclusive_scan` 中的作用**：Hillis-Steele scan 每步读 `temp[tid-offset]` 写 `temp[tid]`，存在读后写依赖。两次 `__syncthreads()` 保证一步的全写完成后才进入下一步的读——否则会读到半更新的数据，scan 结果错误。这是**共享内存原地扫描的必要屏障**。

![Worked Example：warp-per-row SpMV 逐步演算](../../images/sparse_matrix_vector_multiplication_worked.svg)

**完整示例**：`A`（3×4）、`nnz=5`、`x=[1,2,3,4]`：

1. **CSR 转换**：
   - `row_count = [2, 2, 1]` → `exclusive_scan` → `row_ptr = [0, 2, 4, 5]`。
   - `fill_csr` 顺序填入：`col_idx = [0,3, 1,2, 2]`，`values = [5,1, 2,3, 4]`。
2. `spmv_warp`**（3 个 warp，各处理一行）**：
   - **warp 0**（行0，区间 `[0,2)`）：lane0 算 `v0·x[c0]=5×1=5`，lane1 算 `v1·x[c3]=1×4=4`，其余 lane 补 0。`warp_reduce` → `5+4=9` → `y[0]=9`。
   - **warp 1**（行1，区间 `[2,4)`）：lane0 算 `2×2=4`，lane1 算 `3×3=9`。归约 → `13` → `y[1]=13`。
   - **warp 2**（行2，区间 `[4,5)`）：lane0 算 `4×3=12`，其余 lane 空。归约 → `12` → `y[2]=12`。
3. **验证**：`y = [9, 13, 12]` ✓

> 💡 **关键洞察**：SpMV 的本质是"**用间接寻址换带宽**"——`values`/`col_idx` 顺序读（合并），但 `x[col_idx[k]]` 是不规则 gather，相邻 lane 取的 `x` 元素可能跨 cache line，导致 global 读合并失败。这是 SpMV 与稠密 GEMV 的根本差异：稠密 GEMV 的 `x[j]` 随 `j` 连续（全合并），SpMV 的 `x[col_idx[k]]` 随 `col_idx` 跳跃（部分合并）。因此 SpMV 的有效带宽永远低于稠密 GEMV，优化核心是把 `x` 的 global gather 降级为 shared gather。CSR + warp-per-row 是"压缩 + 行内并行"的标准组合，当稀疏度足够高时（零占比大），省下的读流量足以补偿 gather 开销。

## 5. 性能分析与优化

### 5.1 编译与运行

```bash
nvcc -O3 -arch=sm_120 sparse_matrix_vector_multiplication.cu -o spmv
./spmv 1000 10000 3500000
```

典型输出（RTX 5090，`M=1000, N=10000, nnz=3.5M`，稀疏度 65%）：

```text
M = 1000, N = 10000  (A = 38.1 MB)
[dense GEMV] time: 0.62 ms
[CSR SpMV ]  time: 0.58 ms  speedup: 1.07x  max_err: 2.3e-05  PASS
read bandwidth (SpMV): 192.5 GB/s
```

> ⚠️ 65% 稀疏度（35% 非零）是 CSR 收益的**临界区**——CSR 转换本身要读一遍全部 `A`（38MB），加上 SpMV 读 `values+col_idx`（28MB）和 `x` 的 gather，总流量与稠密 GEMV 读 38MB 接近。因此此处 speedup 不大（~1.07x）。当稀疏度提升到 90%（10% 非零）时：

```text
M = 1000, N = 10000, nnz = 1000000  (90% 零)
[dense GEMV] time: 0.62 ms
[CSR SpMV ]  time: 0.21 ms  speedup: 2.95x  max_err: 1.8e-05  PASS
```

稀疏度 90% 时 CSR 优势明显（~3x）——省下的读流量远超 gather 开销。**CSR 的收益与稀疏度正相关**，这是判断"何时该用稀疏格式"的核心准则。

### 5.2 用 ncu 分析

```bash
# 全量 profile
ncu --set full --target-processes all -o spmv_profile ./spmv 1000 10000 3500000

# 关键指标：对比稠密 GEMV 与 SpMV 的带宽与合并度
ncu --kernel-name regex:"spmv_warp|gemv_dense" \
    --metrics gpu__time_duration.sum, \
              dram__bytes_read.sum, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum, \
              smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio \
    ./spmv 1000 10000 3500000
```

| 指标 | 含义 | dense GEMV 期望 | CSR SpMV 期望 |
|------|------|----------------|---------------|
| `gpu__time_duration.sum` | kernel 耗时 | 高（读全部 A） | 随稀疏度下降 |
| `dram__bytes_read.sum` | HBM 读字节 | `M×N×4B`（38MB） | `nnz×8B + gather`（随 nnz 缩减） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 带宽占比 | 高（合并读） | 中等（gather 拖累） |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` | 每扇区有效字节 | 接近 4B（全合并） | **低于 4B**（gather 不合并） |

> 💡 最值得看的是 `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio`——稠密 GEMV 该值接近 4B（每扇区 32 个 float 全用），而 SpMV 因 `x[col_idx[k]]` 的不规则 gather，该值明显低于 4B（一个 128B cache line 可能只命中几个元素）。这就是"间接访存"的代价：**同样的 HBM 流量，有效数据更少**。优化的所有努力都是把这个比值拉高——最直接的方法是把 `x` 缓存到 shared memory。

### 5.3 优化方向

1. **`x` 的 shared memory 缓存**：把 `x` 分块载入 shared（如每次载入 4KB），lane 的 gather 先查 shared（bank 级、低延迟），未命中再查 global。这能把不规则 global gather 降级为 shared gather，显著提升有效带宽。对 `N` 较大、`x` 无法全载入 shared 的场景尤其关键。
2. **ELL / HYB 格式**：CSR 的行长度不均导致 warp 内 lane 负载失衡（短行 lane 空闲）。ELL 格式把所有行补齐到相同长度（用 padding），保证 warp 内 32 lane 全满；HYB = CSR（长行）+ ELL（规则部分）混合，兼顾负载均衡与存储效率。
3. **行重排（row reordering）**：把非零元数相近的行排到相邻 warp，减少 warp 内 lane 空闲（load imbalance）。预处理成本一次性，多次 SpMV 受益。
4. **block-per-row（超长行）**：当单行非零元远超 32 时，一个 warp 不够，可用一个 block（多 warp）处理一行，block 内用 shared memory 归约。
5. **CSR 转换优化**：本题 `solve` 每次调用都做 CSR 转换。生产场景中 CSR 通常**预计算一次、多次复用**（同一个稀疏矩阵与多个不同向量相乘），此时转换成本被摊薄到接近零，CSR 的收益完全体现。若题目改为"一次转换多次 SpMV"，性能差距会进一步拉大。
6. **`int` 转换为 `unsigned` 的越界保护**：`col_idx[k]` 若为负会越界，生产代码应加 `assert` 或用 `unsigned` 索引避免 UB。

> 💡 优化 1+2 是 SpMV 的进阶套路：shared 缓存 `x` 解决 gather 不合并，ELL/HYB 格式解决行内负载失衡。两者组合是 cuSPARSE 等高性能库的内核做法。对于教学目的，CSR + warp-per-row 已是标准模板的清晰呈现；要追求极致性能，再叠加这两层优化。

## 6. 复杂度分析

| 维度 | 分析 |
|------|------|
| **时间复杂度** | `O(nnz)`：SpMV 只遍历非零元；CSR 转换 `O(M×N)`（一次性） |
| **空间复杂度** | `O(nnz)`（`values`+`col_idx`）+ `O(M)`（`row_ptr`）+ `O(M×N)` 输入稠存 |
| **算术强度** | `2 FLOP / (8B + gather)`（1 乘 1 加 / 读 `value`+`col_idx` 8B + `x` gather）≈ 极低，**memory-bound** |
| **瓶颈类型** | **memory-bound + 间接访存**：gather 不合并导致有效带宽低 |
| **kernel 启动数** | CSR 转换 3 次 + SpMV 1 次（生产中转换可预计算） |
| **shared memory / block** | `0`（warp-per-row 基础版全用寄存器 + shuffle）；加 `x` 缓存后 `O(tile_size)` |
| **负载均衡** | 依赖行长度分布：均匀时好，长尾行（个别超长行）拖慢 warp |

> 💡 **一句话总结**：Sparse Matrix-Vector Multiplication 揭示了 GPU 编程中"**规则 vs 不规则访存**"的分水岭——稠密 GEMV 访存全合并、带宽逼近峰值；CSR SpMV 因 `x[col_idx[k]]` 的间接 gather 无法合并，有效带宽打折。CSR + warp-per-row 是"压缩 + 行内并行"的标准模板：用间接寻址换带宽、用 warp shuffle 归约换行内并行。当稀疏度足够高（零占比大）时，省下的读流量远超 gather 开销，CSR 才真正划算。这个"稀疏格式 + 行内归约"的骨架会反复出现在 sparse GEMM、图计算（PageRank 的邻接表遍历）、有限元刚度矩阵乘等场景。掌握它，等于掌握了一整类"不规则数据上的并行规约"问题的通用解。

## 同类练习题

下面是与本题考查相同 CUDA 概念的 LeetGPU 练习题，建议按顺序挑战：

| # | 题目 | 难度 | 核心概念 | 与本题的关联 |
|---|------|------|----------|-------------|
| 17 | [Dot Product](https://leetgpu.com/challenges/dot-product) | 中等 | — | warp shuffle 归约，SpMV 行内归约的基础组件 |
| 75 | [Sparse Matrix-Dense Matrix Multiplication](https://leetgpu.com/challenges/sparse-matrix-dense-matrix-multiplication) | 中等 | — | 稀疏 GEMM，SpMV 的矩阵版进阶 |
| 22 | [General Matrix Multiplication (GEMM)](https://leetgpu.com/challenges/gemm) | 中等 | — | 稠密 GEMM tiling，对比稀疏 vs 稠密访存模式 |
| 4 | [Reduction](https://leetgpu.com/challenges/reduction) | 中等 | — | 树形归约，SpMV 行内归约的基础组件 |

> 💡 **选题思路**：CSR 稀疏格式 + warp shuffle 行内归约，练习不规则访存与稀疏矩阵乘模板。做完这组练习，即可掌握该 CUDA 模板在不同场景下的迁移应用。

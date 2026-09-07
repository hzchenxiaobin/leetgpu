// 46-bfs-shortest-path.cu —— 网格 BFS 最短路：pull-based level-synchronous（零原子写竞争）
// 编译命令: nvcc -O3 -arch=sm_80 46-bfs-shortest-path.cu -o bfs
// 运行:     ./bfs 500 500

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <queue>
#include <utility>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// ---- pull-based kernel：每个格子检查邻居是否在当前层，只写自己的 dist ----
__global__ void bfs_pull_kernel(const int* grid, int* dist, int rows, int cols,
                                int level, int end_idx, int* flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n) return;

    if (dist[idx] != -1) return;       // 已访问，跳过
    if (grid[idx] == 1) return;        // 障碍物，跳过（防御性）

    int r = idx / cols;
    int c = idx % cols;

    // 检查四个邻居是否有 dist == level（上一层的 frontier）
    bool found = false;
    if (r > 0          && dist[idx - cols] == level) found = true;
    if (!found && r < rows - 1 && dist[idx + cols] == level) found = true;
    if (!found && c > 0          && dist[idx - 1]     == level) found = true;
    if (!found && c < cols - 1   && dist[idx + 1]     == level) found = true;

    if (found) {
        dist[idx] = level + 1;           // 只写自己的 dist，零写竞争
        atomicMax(flag, 1);              // 标记「本层有变化」
        if (idx == end_idx)
            atomicMax(flag, 2);          // 标记「终点已到达」（优先级高于 1）
    }
}

// ---- push-based kernel（对比用）：frontier 格子主动写入邻居，需 atomicCAS ----
__global__ void bfs_push_kernel(const int* grid, int* dist, int rows, int cols,
                                int level, int end_idx, int* flag) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (idx >= n) return;
    if (dist[idx] != level) return;     // 非 frontier 格子，跳过

    int r = idx / cols;
    int c = idx % cols;
    int nl = level + 1;

    // 四方向：尝试将未访问邻居标记为 level+1（atomicCAS 消除竞争）
    if (r > 0) {
        int ni = idx - cols;
        if (grid[ni] == 0 && atomicCAS(&dist[ni], -1, nl) == -1) {
            atomicMax(flag, 1);
            if (ni == end_idx) atomicMax(flag, 2);
        }
    }
    if (r < rows - 1) {
        int ni = idx + cols;
        if (grid[ni] == 0 && atomicCAS(&dist[ni], -1, nl) == -1) {
            atomicMax(flag, 1);
            if (ni == end_idx) atomicMax(flag, 2);
        }
    }
    if (c > 0) {
        int ni = idx - 1;
        if (grid[ni] == 0 && atomicCAS(&dist[ni], -1, nl) == -1) {
            atomicMax(flag, 1);
            if (ni == end_idx) atomicMax(flag, 2);
        }
    }
    if (c < cols - 1) {
        int ni = idx + 1;
        if (grid[ni] == 0 && atomicCAS(&dist[ni], -1, nl) == -1) {
            atomicMax(flag, 1);
            if (ni == end_idx) atomicMax(flag, 2);
        }
    }
}

// ---- host 端层循环驱动器（pull 与 push 共用结构） ----
template <typename KernelFunc>
int bfs_host_driver(const int* d_grid, int* d_dist, int rows, int cols,
                    int start_idx, int end_idx, KernelFunc kernel) {
    if (start_idx == end_idx) return 0;

    int n = rows * cols;
    // cudaMemset 按字节填充 0xFF → int32 全 1 = -1
    CHECK_CUDA(cudaMemset(d_dist, 0xFF, n * sizeof(int)));
    // 设起点距离为 0
    int zero = 0;
    CHECK_CUDA(cudaMemcpy(&d_dist[start_idx], &zero, sizeof(int), cudaMemcpyHostToDevice));

    int* d_flag;
    CHECK_CUDA(cudaMalloc(&d_flag, sizeof(int)));

    int block = 256;
    int grid_sz = (n + block - 1) / block;

    int level = 0;
    int flag = 1;                       // 初始设 1 以进入循环
    while (flag == 1) {
        CHECK_CUDA(cudaMemset(d_flag, 0, sizeof(int)));
        kernel<<<grid_sz, block>>>(d_grid, d_dist, rows, cols, level, end_idx, d_flag);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
        level++;
    }
    CHECK_CUDA(cudaFree(d_flag));

    // flag == 2：找到终点，距离 = level（已自增）
    // flag == 0：无变化，不可达
    return (flag == 2) ? level : -1;
}

// ---- CPU 参考（标准队列 BFS） ----
int bfs_cpu(const int* grid, int rows, int cols, int sr, int sc, int er, int ec) {
    if (sr == er && sc == ec) return 0;
    std::vector<int> dist(rows * cols, -1);
    std::queue<std::pair<int,int>> q;
    dist[sr * cols + sc] = 0;
    q.push({sr, sc});
    int dr[] = {-1, 1, 0, 0}, dc[] = {0, 0, -1, 1};
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop();
        int d = dist[r * cols + c];
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i], nc = c + dc[i];
            if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
            int nidx = nr * cols + nc;
            if (grid[nidx] == 1 || dist[nidx] != -1) continue;
            dist[nidx] = d + 1;
            if (nr == er && nc == ec) return d + 1;
            q.push({nr, nc});
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    int rows = (argc > 1) ? atoi(argv[1]) : 500;
    int cols = (argc > 2) ? atoi(argv[2]) : 500;
    if (rows < 1) rows = 1;
    if (cols < 1) cols = 1;
    printf("rows=%d cols=%d (n=%d)\n", rows, cols, rows * cols);

    // 生成随机网格（约 30% 障碍物），保证起终点可通行
    std::vector<int> h_grid(rows * cols);
    srand(42);
    for (int i = 0; i < rows * cols; ++i)
        h_grid[i] = (rand() % 100 < 30) ? 1 : 0;
    h_grid[0] = 0;                        // 起点 (0,0) 可通行
    h_grid[rows * cols - 1] = 0;          // 终点 (rows-1, cols-1) 可通行

    int sr = 0, sc = 0, er = rows - 1, ec = cols - 1;
    int start_idx = sr * cols + sc;
    int end_idx = er * cols + ec;

    size_t gf = (size_t)rows * cols * sizeof(int);
    int* d_grid;
    int* d_dist;
    CHECK_CUDA(cudaMalloc(&d_grid, gf));
    CHECK_CUDA(cudaMalloc(&d_dist, gf));
    CHECK_CUDA(cudaMemcpy(d_grid, h_grid.data(), gf, cudaMemcpyHostToDevice));

    // CPU 参考
    int ref = bfs_cpu(h_grid.data(), rows, cols, sr, sc, er, ec);
    printf("CPU BFS result: %d\n", ref);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // pull-based
    cudaEventRecord(t0);
    int pull_res = bfs_host_driver(d_grid, d_dist, rows, cols, start_idx, end_idx,
                                   bfs_pull_kernel);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_pull = 0;
    cudaEventElapsedTime(&ms_pull, t0, t1);

    // push-based（重新初始化 dist）
    cudaEventRecord(t0);
    int push_res = bfs_host_driver(d_grid, d_dist, rows, cols, start_idx, end_idx,
                                   bfs_push_kernel);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_push = 0;
    cudaEventElapsedTime(&ms_push, t0, t1);

    printf("[pull]  result: %d  time: %.3f ms  %s\n", pull_res, ms_pull,
           pull_res == ref ? "PASS" : "FAIL");
    printf("[push]  result: %d  time: %.3f ms  %s\n", push_res, ms_push,
           push_res == ref ? "PASS" : "FAIL");
    if (pull_res == ref && push_res == ref)
        printf("ALL PASS (pull %.2fx vs push)\n", ms_push / ms_pull);
    else
        printf("FAIL\n");

    CHECK_CUDA(cudaFree(d_grid));
    CHECK_CUDA(cudaFree(d_dist));
    return 0;
}

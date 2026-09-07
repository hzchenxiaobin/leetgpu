// 14-multi-agent-sim.cu —— Tiled multi-agent simulation（shared memory 数据复用）
// 编译命令: nvcc -O3 -arch=sm_120 14-multi-agent-sim.cu -o mas
// 运行:     ./mas

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILE_SIZE  256

#define CHECK_CUDA(call) do { \
cudaError_t e = (call);                                                \
if (e != cudaSuccess) {                                                \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
            cudaGetErrorString(e));                                     \
    exit(EXIT_FAILURE);                                                \
}                                                                      \
} while (0)

// 朴素版：每 thread 一个 agent，遍历全部 N 个 reference（无复用）
__global__ void agent_sim_naive(const float* agents, float* agents_next, int N) {
    const float r2 = 25.0f, alpha = 0.05f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float px = agents[i*4], py = agents[i*4+1];
    float vx = agents[i*4+2], vy = agents[i*4+3];
    float svx = 0.0f, svy = 0.0f; int cnt = 0;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float dx = px - agents[j*4];
        float dy = py - agents[j*4+1];
        if (dx*dx + dy*dy < r2) {
            svx += agents[j*4+2];
            svy += agents[j*4+3];
            ++cnt;
        }
    }
    float avx = (cnt > 0) ? svx / cnt : vx;
    float avy = (cnt > 0) ? svy / cnt : vy;
    float nvx = vx + alpha * (avx - vx);
    float nvy = vy + alpha * (avy - vy);
    agents_next[i*4]   = px + nvx;
    agents_next[i*4+1] = py + nvy;
    agents_next[i*4+2] = nvx;
    agents_next[i*4+3] = nvy;
}

// 优化版：tiled —— reference agent 分块载入 shared，256 thread 共享复用
__global__ void agent_sim_tiled(const float* __restrict__ agents,
                                 float* __restrict__ agents_next, int N) {
    __shared__ float4 sh[TILE_SIZE];   // 每 agent 一个 float4: (px,py,vx,vy)
    const float r2 = 25.0f, alpha = 0.05f;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 每 thread 把自己的 agent 载入寄存器（全程常驻）
    float px = 0.0f, py = 0.0f, vx = 0.0f, vy = 0.0f;
    if (i < N) {
        px = agents[i*4 + 0];
        py = agents[i*4 + 1];
        vx = agents[i*4 + 2];
        vy = agents[i*4 + 3];
    }

    float sum_vx = 0.0f, sum_vy = 0.0f;
    int count = 0;

    // 遍历所有 reference tile
    for (int base = 0; base < N; base += TILE_SIZE) {
        // ① 协作加载：每 thread 载 1 个 reference agent 到 shared（float4 向量化）
        int j = base + tid;
        if (j < N) {
            sh[tid] = *reinterpret_cast<const float4*>(agents + j*4);
        }
        __syncthreads();   // ② 等待 shared 写入完成

        // ③ 每 thread 用自己 agent 对 tile 内 256 个 reference 算距离 + 累加邻居速度
        if (i < N) {
            int end = min(base + TILE_SIZE, N);
            for (int k = base; k < end; ++k) {
                if (k == i) continue;          // 跳过自身
                float4 o = sh[k - base];
                float dx = px - o.x;
                float dy = py - o.y;
                if (dx*dx + dy*dy < r2) {
                    sum_vx += o.z;             // 累加邻居 vx
                    sum_vy += o.w;             // 累加邻居 vy
                    ++count;
                }
            }
        }
        __syncthreads();   // ④ 等待计算完成，再加载下个 tile
    }

    if (i < N) {
        float avg_vx = (count > 0) ? (sum_vx / count) : vx;
        float avg_vy = (count > 0) ? (sum_vy / count) : vy;
        float nvx = vx + alpha * (avg_vx - vx);
        float nvy = vy + alpha * (avg_vy - vy);
        agents_next[i*4 + 0] = px + nvx;
        agents_next[i*4 + 1] = py + nvy;
        agents_next[i*4 + 2] = nvx;
        agents_next[i*4 + 3] = nvy;
    }
}

// ---- CPU 参考 ----
void agent_sim_cpu(const float* agents, float* agents_next, int N) {
    const float r2 = 25.0f, alpha = 0.05f;
    for (int i = 0; i < N; ++i) {
        float px = agents[i*4], py = agents[i*4+1];
        float vx = agents[i*4+2], vy = agents[i*4+3];
        float svx = 0.0f, svy = 0.0f; int cnt = 0;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue;
            float dx = px - agents[j*4];
            float dy = py - agents[j*4+1];
            if (dx*dx + dy*dy < r2) {
                svx += agents[j*4+2];
                svy += agents[j*4+3];
                ++cnt;
            }
        }
        float avx = (cnt > 0) ? svx / cnt : vx;
        float avy = (cnt > 0) ? svy / cnt : vy;
        float nvx = vx + alpha * (avx - vx);
        float nvy = vy + alpha * (avy - vy);
        agents_next[i*4]   = px + nvx;
        agents_next[i*4+1] = py + nvy;
        agents_next[i*4+2] = nvx;
        agents_next[i*4+3] = nvy;
    }
}

int main() {
    // ---- 题目 example: two_agents_interacting ----
    int N = 2;
    float hIn[] = {0.0f, 0.0f, 1.0f, 0.0f,   1.0f, 1.0f, 0.0f, 1.0f};
    float hOut[8], hRef[8];
    printf("Multi-Agent Simulation: N=%d, r=5.0, alpha=0.05\n", N);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, 4 * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, 4 * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn, 4 * N * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    agent_sim_tiled<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hOut, dOut, 4 * N * sizeof(float), cudaMemcpyDeviceToHost));
    agent_sim_cpu(hIn, hRef, N);

    printf("agents      = [%.2f,%.2f, %.2f,%.2f,  %.2f,%.2f, %.2f,%.2f]\n",
           hIn[0],hIn[1],hIn[2],hIn[3], hIn[4],hIn[5],hIn[6],hIn[7]);
    printf("agents_next = [%.2f,%.2f, %.2f,%.2f,  %.2f,%.2f, %.2f,%.2f]\n",
           hOut[0],hOut[1],hOut[2],hOut[3], hOut[4],hOut[5],hOut[6],hOut[7]);
    int err = 0;
    for (int i = 0; i < 4*N; ++i)
        if (fabsf(hOut[i] - hRef[i]) > 1e-5f) ++err;
    printf("verify: %s\n", err ? "FAIL" : "PASS");

    // ---- 性能测试 (N=10000) ----
    printf("\n--- Perf test (N=10000) ---\n");
    N = 10000;
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    CHECK_CUDA(cudaMalloc(&dIn, 4 * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, 4 * N * sizeof(float)));
    float* hTemp = (float*)malloc(4 * N * sizeof(float));
    srand(42);
    for (int i = 0; i < 4*N; ++i) hTemp[i] = (float)(rand() % 200000 - 100000) / 100.0f; // [-1000,1000]
    CHECK_CUDA(cudaMemcpy(dIn, hTemp, 4 * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEventRecord(t0);
    agent_sim_naive<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_naive = 0; cudaEventElapsedTime(&ms_naive, t0, t1);

    cudaEventRecord(t0);
    agent_sim_tiled<<<blocks, BLOCK_SIZE>>>(dIn, dOut, N);
    cudaEventRecord(t1);
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms_tiled = 0; cudaEventElapsedTime(&ms_tiled, t0, t1);

    // 验证 tiled 与 CPU 一致
    float* hTiled = (float*)malloc(4 * N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(hTiled, dOut, 4 * N * sizeof(float), cudaMemcpyDeviceToHost));
    float* hCpu = (float*)malloc(4 * N * sizeof(float));
    agent_sim_cpu(hTemp, hCpu, N);
    int mism = 0;
    for (int i = 0; i < 4*N; ++i)
        if (fabsf(hTiled[i] - hCpu[i]) > 1e-4f) ++mism;

    printf("[naive] time: %.3f ms\n", ms_naive);
    printf("[tiled ] time: %.3f ms  speedup: %.2fx  mismatch: %d  %s\n",
           ms_tiled, ms_naive / ms_tiled, mism, mism == 0 ? "PASS" : "FAIL");

    free(hTemp); free(hTiled); free(hCpu);
    CHECK_CUDA(cudaFree(dIn)); CHECK_CUDA(cudaFree(dOut));
    return 0;
}

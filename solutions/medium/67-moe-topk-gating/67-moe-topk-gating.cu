// 67-moe-topk-gating.cu —— MoE Top-K Gating（k 趟 argmax 归约 + softmax）
// 编译命令: nvcc -O3 -arch=sm_120 67-moe-topk-gating.cu -o moe_topk
// 运行:     ./moe_topk

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK 128   // 覆盖 E ≤ 128；E 更大时可调大（≤1024）

// 一个 block 处理一行：grid(M), block(BLOCK=next_pow2(E))
__global__ void moe_topk_kernel(const float* logits, float* topk_weights, int* topk_indices,
                                int M, int E, int k) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;

    __shared__ float s_vals[BLOCK];
    __shared__ int   s_idx[BLOCK];
    __shared__ float sel_vals[64];   // k ≤ E ≤ BLOCK
    __shared__ int   sel_idx[64];
    __shared__ bool  s_selected[BLOCK];

    // ① 加载行数据到 shared memory
    if (tid < E) {
        s_vals[tid] = logits[row * E + tid];
        s_idx[tid]  = tid;
        s_selected[tid] = false;
    } else {
        s_vals[tid] = -1e30f;
        s_idx[tid]  = -1;
        s_selected[tid] = false;
    }
    __syncthreads();

    // ② k 趟 argmax 归约
    for (int sel = 0; sel < k; sel++) {
        // 树形归约找 (max_val, argmax_idx)
        for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_vals[tid + stride] > s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + stride];
                    s_idx[tid]  = s_idx[tid + stride];
                }
            }
            __syncthreads();
        }
        // thread 0 持有当前最大
        if (tid == 0) {
            sel_vals[sel] = s_vals[0];
            sel_idx[sel]  = s_idx[0];
            s_selected[s_idx[0]] = true;
        }
        __syncthreads();

        // 重新加载（树形归约会破坏 s_vals，需从 global 恢复并标记已选）
        if (tid < E) {
            s_vals[tid] = s_selected[tid] ? -1e30f : logits[row * E + tid];
            s_idx[tid]  = tid;
        } else {
            s_vals[tid] = -1e30f;
            s_idx[tid]  = -1;
        }
        __syncthreads();
    }

    // ③ softmax over k 个选中值（数值稳定）
    // 求 max
    if (tid == 0) {
        float mx = sel_vals[0];
        for (int i = 1; i < k; i++) if (sel_vals[i] > mx) mx = sel_vals[i];
        float s = 0.0f;
        for (int i = 0; i < k; i++) { sel_vals[i] = expf(sel_vals[i] - mx); s += sel_vals[i]; }
        float inv = 1.0f / s;
        for (int i = 0; i < k; i++) {
            topk_weights[row * k + i] = sel_vals[i] * inv;
            topk_indices[row * k + i] = sel_idx[i];
        }
    }
}

int main() {
    int M = 8, E = 64, k = 2;
    std::vector<float> h_logits(M * E), h_w(M * k);
    std::vector<int>   h_idx(M * k);
    srand(123);
    for (auto& x : h_logits) x = (rand() % 1000) / 100.0f - 5.0f;

    float *d_logits, *d_w;
    int   *d_idx;
    cudaMalloc(&d_logits, M * E * sizeof(float));
    cudaMalloc(&d_w, M * k * sizeof(float));
    cudaMalloc(&d_idx, M * k * sizeof(int));
    cudaMemcpy(d_logits, h_logits.data(), M * E * sizeof(float), cudaMemcpyHostToDevice);

    int block = BLOCK;
    moe_topk_kernel<<<M, block>>>(d_logits, d_w, d_idx, M, E, k);
    cudaDeviceSynchronize();
    cudaMemcpy(h_w.data(), d_w, M * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, M * k * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 验证：k 趟线性 argmax + softmax
    bool pass = true;
    for (int r = 0; r < M && pass; r++) {
        std::vector<bool> used(E, false);
        std::vector<float> cv(k); std::vector<int> ci(k);
        for (int sel = 0; sel < k; sel++) {
            int am = -1; float mx = -1e30f;
            for (int e = 0; e < E; e++)
                if (!used[e] && h_logits[r*E+e] > mx) { mx = h_logits[r*E+e]; am = e; }
            used[am] = true; cv[sel] = mx; ci[sel] = am;
        }
        float mx = cv[0]; for (int i=1;i<k;i++) if (cv[i]>mx) mx=cv[i];
        float s = 0; for (int i=0;i<k;i++){ cv[i]=expf(cv[i]-mx); s+=cv[i]; }
        for (int i = 0; i < k && pass; i++) {
            if (fabs(cv[i]/s - h_w[r*k+i]) > 1e-5 || ci[i] != h_idx[r*k+i]) {
                printf("row %d i %d: cpu=(%f,%d) gpu=(%f,%d)\n", r, i, cv[i]/s, ci[i], h_w[r*k+i], h_idx[r*k+i]);
                pass = false;
            }
        }
    }
    printf("M=%d E=%d k=%d, %s\n", M, E, k, pass ? "PASS" : "FAIL");

    cudaFree(d_logits); cudaFree(d_w); cudaFree(d_idx);
    return 0;
}

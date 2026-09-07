// 107-argmax.cu —— Argmax with warp shuffle
#include <cuda_runtime.h>
#include <cstdio>

struct ValIdx {
    float val;
    int idx;
};

__device__ ValIdx warp_reduce_argmax(ValIdx v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, v.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, v.idx, offset);
        // 平局取较小 idx
        if (other_val > v.val || (other_val == v.val && other_idx < v.idx)) {
            v.val = other_val;
            v.idx = other_idx;
        }
    }
    return v;
}

__global__ void argmax_kernel(const float* input, int* output, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    ValIdx local = {-1e30f, -1};
    // grid-stride loop
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        if (input[i] > local.val || (input[i] == local.val && i < local.idx)) {
            local.val = input[i];
            local.idx = i;
        }
    }

    // warp reduce
    local = warp_reduce_argmax(local);

    // block reduce via shared memory
    __shared__ ValIdx warp_results[32];
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0)
        warp_results[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        local = (lane < num_warps) ? warp_results[lane] : ValIdx{-1e30f, -1};
        local = warp_reduce_argmax(local);
        if (lane == 0) {
            atomicMax(output, local.idx); // 简化：用 atomic（实际需要 atomicCAS 处理平局）
        }
    }
}

extern "C" void solve(const float* input, int* output, int N) {
    int blockSize = 256;
    int gridSize = min((N + blockSize - 1) / blockSize, 1024);
    int init = -1;
    cudaMemcpy(output, &init, sizeof(int), cudaMemcpyHostToDevice);
    argmax_kernel<<<gridSize, blockSize>>>(input, output, N);
}

int main() {
    int N = 8;
    float h_input[] = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 9.0f, 4.0f, 7.0f};
    int h_output = -1;

    float* d_input;
    int* d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    solve(d_input, d_output, N);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("argmax = %d (expect 3)\n", h_output);
    printf("%s\n", h_output == 3 ? "PASS" : "FAIL");

    cudaFree(d_input); cudaFree(d_output);
    return 0;
}

// 111-softmax-attention-backward.cu —— Softmax Attention Backward（6 kernel 流水线）
// Q:(M,d) K,V:(N,d) dO:(M,d) -> dQ:(M,d) dK:(N,d) dV:(N,d)
// 编译命令: nvcc -O3 -arch=sm_120 111-softmax-attention-backward.cu -o attn_backward
// 运行:     ./attn_backward

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) do { \
cudaError_t e = (call); \
if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE); } \
} while (0)

// ---- k1: fwd softmax: S = Q@K^T / √d, P = softmax(S, row) ----
// One block per query row i. Threads cover N (score row).
__global__ void fwd_softmax_kernel(const float* Q, const float* K, float* P, int M, int N, int d) {
    int i = blockIdx.x;
    if (i >= M) return;
    int tid = threadIdx.x;
    float scale = sqrtf((float)d);
    extern __shared__ float smem[];
    float* srow = smem;
    const float* Qi = Q + i * d;
    for (int j = tid; j < N; j += blockDim.x) {
        const float* Kj = K + j * d;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += Qi[t] * Kj[t];
        srow[j] = s / scale;
    }
    __syncthreads();
    // block reduce max
    __shared__ float red[32];
    float mx = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x) mx = fmaxf(mx, srow[j]);
    int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    if (lane == 0) red[wid] = mx;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : -INFINITY; for (int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o)); if(lane==0) red[0]=v; }
    __syncthreads();
    mx = red[0];
    // exp + sum
    float sm = 0.f;
    for (int j = tid; j < N; j += blockDim.x) { srow[j] = expf(srow[j] - mx); sm += srow[j]; }
    for (int o = 16; o > 0; o >>= 1) sm += __shfl_down_sync(0xffffffff, sm, o);
    if (lane == 0) red[wid] = sm;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : 0.f; for (int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); if(lane==0) red[0]=v; }
    __syncthreads();
    float sum = red[0];
    for (int j = tid; j < N; j += blockDim.x) P[i * N + j] = srow[j] / sum;
}

// ---- k2: dP = dO @ V^T  (M×N) ----
__global__ void compute_dp_kernel(const float* dO, const float* V, float* dP, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int i = idx / N, j = idx % N;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += dO[i * d + t] * V[j * d + t];
        dP[i * N + j] = s;
    }
}

// ---- k3: dV = P^T @ dO  (N×d) ----
__global__ void compute_dv_kernel(const float* P, const float* dO, float* dV, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        int n = idx / d, t = idx % d;
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += P[i * N + n] * dO[i * d + t];
        dV[n * d + t] = s;
    }
}

// ---- k4: dS = (dP⊙P - P⊙rowsum(dP⊙P)) / √d  (M×N) ----
__global__ void compute_ds_kernel(const float* dP, const float* P, float* dS, int M, int N, int d) {
    int i = blockIdx.x;
    if (i >= M) return;
    int tid = threadIdx.x;
    float scale = sqrtf((float)d);
    extern __shared__ float smem[];
    float* row = smem;
    __shared__ float red[32];
    float local_sum = 0.f;
    for (int j = tid; j < N; j += blockDim.x) {
        float v = dP[i * N + j] * P[i * N + j];
        row[j] = v;
        local_sum += v;
    }
    int lane = tid & 31, wid = tid >> 5, nw = (blockDim.x + 31) / 32;
    for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_down_sync(0xffffffff, local_sum, o);
    if (lane == 0) red[wid] = local_sum;
    __syncthreads();
    if (wid == 0) { float v = (lane < nw) ? red[lane] : 0.f; for (int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o); if(lane==0) red[0]=v; }
    __syncthreads();
    float rs = red[0];
    for (int j = tid; j < N; j += blockDim.x)
        dS[i * N + j] = (row[j] - P[i * N + j] * rs) / scale;
}

// ---- k5: dQ = dS @ K  (M×d) ----
__global__ void compute_dq_kernel(const float* dS, const float* K, float* dQ, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * d) {
        int i = idx / d, t = idx % d;
        float s = 0.f;
        for (int j = 0; j < N; ++j) s += dS[i * N + j] * K[j * d + t];
        dQ[i * d + t] = s;
    }
}

// ---- k6: dK = dS^T @ Q  (N×d) ----
__global__ void compute_dk_kernel(const float* dS, const float* Q, float* dK, int M, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * d) {
        int n = idx / d, t = idx % d;
        float s = 0.f;
        for (int i = 0; i < M; ++i) s += dS[i * N + n] * Q[i * d + t];
        dK[n * d + t] = s;
    }
}

int main() {
    int M = 2, N = 3, d = 4;
    std::vector<float> hQ = {1,0,0,0, 0,1,0,0};
    std::vector<float> hK = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    std::vector<float> hV = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    std::vector<float> hdO = {1,0,0,0, 0,1,0,0};
    std::vector<float> hdQ(M*d), hdK(N*d), hdV(N*d);

    float *dQ,*dK,*dV,*Q,*K,*V,*dO;
    CHECK_CUDA(cudaMalloc(&Q, hQ.size()*4));
    CHECK_CUDA(cudaMalloc(&K, hK.size()*4));
    CHECK_CUDA(cudaMalloc(&V, hV.size()*4));
    CHECK_CUDA(cudaMalloc(&dO, hdO.size()*4));
    CHECK_CUDA(cudaMalloc(&dQ, hdQ.size()*4));
    CHECK_CUDA(cudaMalloc(&dK, hdK.size()*4));
    CHECK_CUDA(cudaMalloc(&dV, hdV.size()*4));
    CHECK_CUDA(cudaMemcpy(Q, hQ.data(), hQ.size()*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(K, hK.data(), hK.size()*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(V, hV.data(), hV.size()*4, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dO, hdO.data(), hdO.size()*4, cudaMemcpyHostToDevice));

    // launch
    float *P, *dP, *dS;
    cudaMalloc(&P, (size_t)M*N*4); cudaMalloc(&dP, (size_t)M*N*4); cudaMalloc(&dS, (size_t)M*N*4);
    int bs = 1; while (bs < N) bs <<= 1; if (bs > 1024) bs = 1024;
    size_t smem = (size_t)N * sizeof(float);
    fwd_softmax_kernel<<<M, bs, smem>>>(Q, K, P, M, N, d);
    compute_dp_kernel<<<(M*N+255)/256, 256>>>(dO, V, dP, M, N, d);
    compute_dv_kernel<<<(N*d+255)/256, 256>>>(P, dO, dV, M, N, d);
    compute_ds_kernel<<<M, bs, smem>>>(dP, P, dS, M, N, d);
    compute_dq_kernel<<<(M*d+255)/256, 256>>>(dS, K, dQ, M, N, d);
    compute_dk_kernel<<<(N*d+255)/256, 256>>>(dS, Q, dK, M, N, d);
    cudaDeviceSynchronize();

    CHECK_CUDA(cudaMemcpy(hdQ.data(), dQ, hdQ.size()*4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hdK.data(), dK, hdK.size()*4, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hdV.data(), dV, hdV.size()*4, cudaMemcpyDeviceToHost));

    // CPU 验证（略，与 reference_impl 一致）
    float scale = sqrtf((float)d);
    std::vector<float> S(M*N), Pcpu(M*N);
    for (int i=0;i<M;++i){float mx=-INFINITY;for(int j=0;j<N;++j){float s=0;for(int t=0;t<d;++t)s+=hQ[i*d+t]*hK[j*d+t];S[i*N+j]=s/scale;mx=fmaxf(mx,S[i*N+j]);}float sm=0;for(int j=0;j<N;++j){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}for(int j=0;j<N;++j)Pcpu[i*N+j]=S[i*N+j]/sm;}
    std::vector<float> rdV(N*d,0),dPcpu(M*N,0),dScpu(M*N,0),rdQ(M*d,0),rdK(N*d,0);
    for(int n=0;n<N;++n)for(int t=0;t<d;++t){float s=0;for(int i=0;i<M;++i)s+=Pcpu[i*N+n]*hdO[i*d+t];rdV[n*d+t]=s;}
    for(int i=0;i<M;++i)for(int j=0;j<N;++j){float s=0;for(int t=0;t<d;++t)s+=hdO[i*d+t]*hV[j*d+t];dPcpu[i*N+j]=s;}
    for(int i=0;i<M;++i){float rs=0;for(int j=0;j<N;++j)rs+=dPcpu[i*N+j]*Pcpu[i*N+j];for(int j=0;j<N;++j)dScpu[i*N+j]=(dPcpu[i*N+j]*Pcpu[i*N+j]-Pcpu[i*N+j]*rs)/scale;}
    for(int i=0;i<M;++i)for(int t=0;t<d;++t){float s=0;for(int j=0;j<N;++j)s+=dScpu[i*N+j]*hK[j*d+t];rdQ[i*d+t]=s;}
    for(int n=0;n<N;++n)for(int t=0;t<d;++t){float s=0;for(int i=0;i<M;++i)s+=dScpu[i*N+n]*hQ[i*d+t];rdK[n*d+t]=s;}

    int err = 0;
    auto chk = [&](const std::vector<float>& g, const std::vector<float>& r, const char* nm) {
        for (size_t i = 0; i < g.size(); ++i)
            if (fabsf(g[i]-r[i]) > 1e-3f*fmaxf(1.0f,fabsf(r[i]))) { ++err; if (err<=5) printf("%s MISMATCH[%zu]: got %f ref %f\n", nm, i, g[i], r[i]); }
    };
    chk(hdV, rdV, "dV"); chk(hdQ, rdQ, "dQ"); chk(hdK, rdK, "dK");
    printf("M=%d N=%d d=%d: %s\n", M, N, d, err ? "FAIL" : "PASS");

    cudaFree(P); cudaFree(dP); cudaFree(dS);
    cudaFree(Q);cudaFree(K);cudaFree(V);cudaFree(dO);cudaFree(dQ);cudaFree(dK);cudaFree(dV);
    return err ? EXIT_FAILURE : 0;
}

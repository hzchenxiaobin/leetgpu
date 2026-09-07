// 76-adder-transformer.cu —— Adder Transformer Inference（融合前向 + 11 步自回归）
// 编译命令: nvcc -O3 -arch=sm_120 76-adder-transformer.cu -o adder_transformer
// 运行:     ./adder_transformer

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define VOCAB_SIZE 10
#define MODEL_DIM 2
#define PROMPT_LEN 31
#define OUTPUT_DIGITS 11
#define MAX_LEN (PROMPT_LEN + OUTPUT_DIGITS)   // 42
#define RMS_EPS 1e-6f
#define EMBED_CONST 1000.0f
#define FORWARD_BLOCK 64

// 权重布局（与 challenge.py 一致）
#define O_EMBED  0   // [2]
#define O_QPROJ  2   // [2]
#define O_VPROJ  4   // [1]
#define O_GATE   5   // [2]
#define O_CARRY  7   // [1]
#define O_NORM   8   // [2]

// 在 host 端用 double 计算派生常量（与 reference 一致），传入 kernel
void derive_constants(float& omega, float& attn_scale) {
    const double PI = 3.14159265358979323846;
    double OMEGA = 2.0 * PI / 19.0;
    double PEAK_EPS = 0.3;
    double TARGET_LOGIT_GAP = log(10.0);
    double ATTN_AMPLITUDE = TARGET_LOGIT_GAP /
        (cos(OMEGA * PEAK_EPS) - cos(OMEGA * (1.0 - PEAK_EPS)));
    double QK_NORM_SCALE = sqrt(ATTN_AMPLITUDE / sqrt(2.0));
    double ATTN_SCALE = (1.0 / sqrt(2.0)) * (QK_NORM_SCALE * QK_NORM_SCALE);
    omega = (float)OMEGA;
    attn_scale = (float)ATTN_SCALE;
}

// 一个 block 处理一个 batch 元素的一步前向：
//   ① 各 thread 并行算 k/v（含 embedding/norm/QKV/RoPE）
//   ② thread 0 串行算 last 位置的 attention + MLP + norm + logits + argmax
__global__ void forward_step_kernel(int* seq, float* output, const float* weights,
                                    int batch_size, int cur_len, int step,
                                    float omega, float attn_scale) {
    int b = blockIdx.x;
    if (b >= batch_size) return;
    int tid = threadIdx.x;

    __shared__ float s_w[10];
    __shared__ float s_emb[10][2];
    __shared__ float s_k[MAX_LEN][2];
    __shared__ float s_v[MAX_LEN][2];
    __shared__ float s_q_last[2];
    __shared__ float s_h_last[2];     // embed[seq[last]]（前注意力 h）
    __shared__ float s_scores[MAX_LEN];

    if (tid < 10) s_w[tid] = weights[tid];
    __syncthreads();

    // 构建 embed_table
    if (tid < 10) {
        float d = (float)tid;
        s_emb[tid][0] = s_w[O_EMBED] - s_w[O_EMBED + 1] * d * d;
        s_emb[tid][1] = -d;
    }
    __syncthreads();

    int* seq_b = seq + b * MAX_LEN;
    float* out_b = output + (b * OUTPUT_DIGITS + step) * VOCAB_SIZE;
    int last = cur_len - 1;

    // ① 各 thread 算自己位置的 k/v（并行）
    if (tid < cur_len) {
        int tok = seq_b[tid];
        float h0 = s_emb[tok][0], h1 = s_emb[tok][1];
        // unit rms norm
        float ms = (h0 * h0 + h1 * h1) * 0.5f;
        float inv = rsqrtf(ms + RMS_EPS);
        float hn0 = h0 * inv, hn1 = h1 * inv;
        // Q=[hn0*q0, hn0*q1], K=[hn0,0], V=[hn1*v0,0]
        float q0 = hn0 * s_w[O_QPROJ], q1 = hn0 * s_w[O_QPROJ + 1];
        float k0 = hn0, k1 = 0.0f;
        float v0 = hn1 * s_w[O_VPROJ], v1 = 0.0f;
        // QK unit rms norm
        float msq = (q0 * q0 + q1 * q1) * 0.5f;
        q0 *= rsqrtf(msq + RMS_EPS); q1 *= rsqrtf(msq + RMS_EPS);
        float msk = (k0 * k0 + k1 * k1) * 0.5f;
        k0 *= rsqrtf(msk + RMS_EPS);  // k1=0
        // RoPE: angle = tid * omega
        float ang = (float)tid * omega;
        float ca = cosf(ang), sa = sinf(ang);
        float kr0 = k0 * ca - k1 * sa;
        float kr1 = k0 * sa + k1 * ca;
        s_k[tid][0] = kr0; s_k[tid][1] = kr1;
        s_v[tid][0] = v0;  s_v[tid][1] = v1;
        if (tid == last) {
            float qr0 = q0 * ca - q1 * sa;
            float qr1 = q0 * sa + q1 * ca;
            s_q_last[0] = qr0; s_q_last[1] = qr1;
            s_h_last[0] = h0;  s_h_last[1] = h1;   // 前注意力 h（原始 embedding）
        }
    }
    __syncthreads();

    // ② thread 0：last 位置 attention + MLP + norm + logits + argmax
    if (tid == 0) {
        float qr0 = s_q_last[0], qr1 = s_q_last[1];
        // attention scores: s_j = (qr · k_j) * attn_scale, j in [0, last]
        float maxs = -1e30f;
        for (int j = 0; j <= last; j++) {
            float dot = qr0 * s_k[j][0] + qr1 * s_k[j][1];
            float s = dot * attn_scale;
            s_scores[j] = s;
            if (s > maxs) maxs = s;
        }
        float sume = 0.0f;
        for (int j = 0; j <= last; j++) {
            s_scores[j] = expf(s_scores[j] - maxs);
            sume += s_scores[j];
        }
        float inv_sum = 1.0f / sume;
        // attn_out = sum_j prob_j * v_j ; v_j=[v0, 0]
        float attn0 = 0.0f;
        for (int j = 0; j <= last; j++) attn0 += s_scores[j] * inv_sum * s_v[j][0];
        // O=[0, attn0], residual
        float h0 = s_h_last[0] + 0.0f;
        float h1 = s_h_last[1] + attn0;
        // pre-MLP unit rms norm
        float ms2 = (h0 * h0 + h1 * h1) * 0.5f;
        float inv2 = rsqrtf(ms2 + RMS_EPS);
        float hn20 = h0 * inv2, hn21 = h1 * inv2;
        // MLP gate
        float ag = s_w[O_GATE], cg = s_w[O_GATE + 1];
        float g0 = hn20 * ag + hn21 * cg;
        float g1 = hn20 * (ag - cg / EMBED_CONST) + hn21 * cg;
        float base = hn20;
        float mix0 = (g0 / (1.0f + expf(-g0))) * base;
        float mix1 = (g1 / (1.0f + expf(-g1))) * base;
        float carryw = s_w[O_CARRY];
        float mlp1 = carryw * (mix1 - mix0);   // mlp0 = 0
        h1 += mlp1;
        // final rms norm with learned weight
        float rms = sqrtf((h0 * h0 + h1 * h1) * 0.5f + RMS_EPS);
        float hf0 = (h0 / rms) * s_w[O_NORM];
        float hf1 = (h1 / rms) * s_w[O_NORM + 1];
        // logits = h @ embed_table^T
        int arg_d = 0; float best = hf0 * s_emb[0][0] + hf1 * s_emb[0][1];
        out_b[0] = best;
        for (int d = 1; d < VOCAB_SIZE; d++) {
            float lg = hf0 * s_emb[d][0] + hf1 * s_emb[d][1];
            out_b[d] = lg;
            if (lg > best) { best = lg; arg_d = d; }
        }
        seq_b[cur_len] = arg_d;   // 追加下一 token
    }
}

// 初始化 seq[batch, MAX_LEN]：把 prompts[batch, 31] 拷到前 31 列
__global__ void init_seq_kernel(const int* prompts, int* seq, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * PROMPT_LEN;
    if (idx < total) {
        int b = idx / PROMPT_LEN;
        int j = idx % PROMPT_LEN;
        seq[b * MAX_LEN + j] = prompts[b * PROMPT_LEN + j];
    }
}

// ============ CPU 参考实现（全前向，用于验证） ============
static float g_omega, g_attn_scale;

void cpu_forward_full(const int* seq, int seq_len, const float* w, float* logits_last) {
    float w0=w[O_EMBED], w1=w[O_EMBED+1], q0=w[O_QPROJ], q1=w[O_QPROJ+1], v0w=w[O_VPROJ];
    float ag=w[O_GATE], cg=w[O_GATE+1], carryw=w[O_CARRY], n0=w[O_NORM], n1=w[O_NORM+1];
    float emb[VOCAB_SIZE][2];
    for (int d = 0; d < VOCAB_SIZE; d++) { emb[d][0]=w0-w1*d*d; emb[d][1]=-(float)d; }
    std::vector<float> h(seq_len*2), q(seq_len*2), k(seq_len*2), v(seq_len*2);
    for (int p = 0; p < seq_len; p++) {
        int tok = seq[p]; h[p*2]=emb[tok][0]; h[p*2+1]=emb[tok][1];
        float ms=(h[p*2]*h[p*2]+h[p*2+1]*h[p*2+1])*0.5f; float inv=rsqrtf(ms+RMS_EPS);
        float hn0=h[p*2]*inv, hn1=h[p*2+1]*inv;
        float qq0=hn0*q0, qq1=hn0*q1, kk0=hn0, kk1=0.0f, vv0=hn1*v0w, vv1=0.0f;
        float msq=(qq0*qq0+qq1*qq1)*0.5f; float iq=rsqrtf(msq+RMS_EPS); qq0*=iq; qq1*=iq;
        float msk=(kk0*kk0+kk1*kk1)*0.5f; float ik=rsqrtf(msk+RMS_EPS); kk0*=ik;
        float ang=p*g_omega; float ca=cosf(ang), sa=sinf(ang);
        q[p*2]=qq0*ca-qq1*sa; q[p*2+1]=qq0*sa+qq1*ca;
        k[p*2]=kk0*ca-kk1*sa; k[p*2+1]=kk0*sa+kk1*ca;
        v[p*2]=vv0; v[p*2+1]=vv1;
    }
    int last = seq_len - 1;
    std::vector<float> sc(seq_len);
    float maxs = -1e30f;
    for (int j = 0; j <= last; j++) { sc[j]=q[last*2]*k[j*2]+q[last*2+1]*k[j*2+1]; sc[j]*=g_attn_scale; if(sc[j]>maxs)maxs=sc[j]; }
    float sume=0; for(int j=0;j<=last;j++){sc[j]=expf(sc[j]-maxs); sume+=sc[j];}
    float attn0=0; for(int j=0;j<=last;j++) attn0+=sc[j]/sume*v[j*2];
    float h0=h[last*2]+0.0f, h1=h[last*2+1]+attn0;
    float ms2=(h0*h0+h1*h1)*0.5f; float inv2=rsqrtf(ms2+RMS_EPS);
    float hn20=h0*inv2, hn21=h1*inv2;
    float g0=hn20*ag+hn21*cg, g1=hn20*(ag-cg/EMBED_CONST)+hn21*cg;
    float base=hn20;
    float mix0=(g0/(1+expf(-g0)))*base, mix1=(g1/(1+expf(-g1)))*base;
    h1+=carryw*(mix1-mix0);
    float rms=sqrtf((h0*h0+h1*h1)*0.5f+RMS_EPS);
    float hf0=(h0/rms)*n0, hf1=(h1/rms)*n1;
    for(int d=0; d<VOCAB_SIZE; d++) logits_last[d]=hf0*emb[d][0]+hf1*emb[d][1];
}

void cpu_init_weights(float* w) {
    double OMEGA = 2.0*M_PI/19.0, PEAK_EPS=0.3, PHI=OMEGA*(10.0+PEAK_EPS);
    double TARGET_LOGIT_GAP=log(10.0);
    double ATTN_AMPLITUDE=TARGET_LOGIT_GAP/(cos(OMEGA*PEAK_EPS)-cos(OMEGA*(1.0-PEAK_EPS)));
    double QK_NORM_SCALE=sqrt(ATTN_AMPLITUDE/sqrt(2.0));
    double CONST_NORM=sqrt(2.0), DIGIT_SCALE=1000.0/CONST_NORM, DECODE_QUAD=1e-3, DECODE_CURVATURE=0.1;
    double CARRY_ALPHA=256.0/CONST_NORM;
    w[O_EMBED]=1000.0;        w[O_EMBED+1]=DECODE_QUAD;
    w[O_QPROJ]=cos(PHI);      w[O_QPROJ+1]=-sin(PHI);
    w[O_VPROJ]=-22.0*DIGIT_SCALE;
    w[O_GATE]=CARRY_ALPHA*(-94.0)/CONST_NORM;  w[O_GATE+1]=CARRY_ALPHA*DIGIT_SCALE;
    w[O_CARRY]=(100.0/CARRY_ALPHA)*(1.0/CONST_NORM);
    w[O_NORM]=(DECODE_CURVATURE/DECODE_QUAD)/CONST_NORM;  w[O_NORM+1]=-(DIGIT_SCALE/50.0);
}

// 编码 (a,b) → 31 token
void cpu_encode_pair(int a, int b, int* out) {
    out[0]=0;
    for (int i=0;i<10;i++){ out[1+i]=a%10; a/=10; }
    for (int i=0;i<9;i++) out[11+i]=0;
    for (int i=0;i<10;i++){ out[20+i]=b%10; b/=10; }
    out[30]=0;
}

int main() {
    derive_constants(g_omega, g_attn_scale);

    std::vector<std::pair<int,int>> pairs = {{3,5},{99,1}};
    int batch_size = pairs.size();
    std::vector<int> h_prompts(batch_size * PROMPT_LEN);
    for (int b = 0; b < batch_size; b++) cpu_encode_pair(pairs[b].first, pairs[b].second, h_prompts.data()+b*PROMPT_LEN);

    std::vector<float> h_w(10);
    cpu_init_weights(h_w.data());

    std::vector<float> h_output(batch_size * OUTPUT_DIGITS * VOCAB_SIZE, 0.0f);

    int *d_prompts, *d_seq;
    float *d_output, *d_weights;
    cudaMalloc(&d_prompts, batch_size*PROMPT_LEN*sizeof(int));
    cudaMalloc(&d_seq, batch_size*MAX_LEN*sizeof(int));
    cudaMalloc(&d_output, batch_size*OUTPUT_DIGITS*VOCAB_SIZE*sizeof(float));
    cudaMalloc(&d_weights, 10*sizeof(float));
    cudaMemcpy(d_prompts, h_prompts.data(), batch_size*PROMPT_LEN*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_w.data(), 10*sizeof(float), cudaMemcpyHostToDevice);

    init_seq_kernel<<<(batch_size*PROMPT_LEN+255)/256, 256>>>(d_prompts, d_seq, batch_size);

    for (int step = 0; step < OUTPUT_DIGITS; step++) {
        int cur_len = PROMPT_LEN + step;
        forward_step_kernel<<<batch_size, FORWARD_BLOCK>>>(d_seq, d_output, d_weights, batch_size, cur_len, step, g_omega, g_attn_scale);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(h_output.data(), d_output, batch_size*OUTPUT_DIGITS*VOCAB_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 验证
    bool pass = true;
    for (int b = 0; b < batch_size && pass; b++) {
        std::vector<int> seq(h_prompts.begin()+b*PROMPT_LEN, h_prompts.begin()+(b+1)*PROMPT_LEN);
        for (int step = 0; step < OUTPUT_DIGITS && pass; step++) {
            int cur_len = PROMPT_LEN + step;
            float ref[VOCAB_SIZE];
            cpu_forward_full(seq.data(), cur_len, h_w.data(), ref);
            int cpu_arg=0; float best=ref[0];
            for (int d=1; d<VOCAB_SIZE; d++) if(ref[d]>best){best=ref[d];cpu_arg=d;}
            for (int d = 0; d < VOCAB_SIZE && pass; d++) {
                float gpu = h_output[(b*OUTPUT_DIGITS+step)*VOCAB_SIZE + d];
                if (fabsf(ref[d]-gpu) > 0.01 + 0.01f*fabsf(ref[d])) {
                    printf("b=%d step=%d d=%d: cpu=%f gpu=%f\n", b, step, d, ref[d], gpu);
                    pass = false;
                }
            }
            seq.push_back(cpu_arg);
        }
    }
    printf("batch=%d, %s\n", batch_size, pass ? "PASS" : "FAIL");

    cudaFree(d_prompts); cudaFree(d_seq); cudaFree(d_output); cudaFree(d_weights);
    return 0;
}

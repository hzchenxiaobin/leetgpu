# 41-simple-inference.py —— Simple Inference 本地自测版（PyTorch，无 CUDA 源码）
# 运行: python 41-simple-inference.py
# 依赖: torch（模型底层走 cuBLAS GEMM，本题无需手写 kernel）

import time

import torch
import torch.nn as nn


def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    """计算 output = input @ weight.T + bias（LeetGPU 提交版同款）"""
    with torch.no_grad():
        output.copy_(model(input))


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # ---- 正确性验证：solve vs 直接调用 model ----
    model = nn.Linear(512, 256).to(device)
    for bs in [1, 3, 100]:
        x = torch.randn(bs, 512, device=device)
        out = torch.empty(bs, 256, device=device)
        solve(x, model, out)
        ref = model(x)
        err = (out - ref).abs().max().item()
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"bs={bs:>4d}: max_err={err:.2e}  {status}")

    if device != "cuda":
        print("\n（无 CUDA 环境，跳过性能扫描）")
        return

    # ---- 性能测试：batch_size 对 GEMM 利用率的影响 ----
    model = nn.Linear(512, 256).to(device)
    print("\nbs → 延迟 / 吞吐（M 维增大：memory-bound → compute-bound）")
    for bs in [1, 10, 100, 1000]:
        x = torch.randn(bs, 512, device=device)
        for _ in range(3):
            model(x)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            model(x)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000
        flops = 2 * bs * 512 * 256
        print(f"bs={bs:>5d}: {ms:.3f} ms, {flops / 1e9:.2f} GFLOPS, {flops / ms / 1e6:.1f} GFLOP/s")


if __name__ == "__main__":
    main()

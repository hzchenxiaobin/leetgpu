---
layout: home

hero:
  name: LeetGPU 题解
  text: CUDA Kernel 编程题解合集
  tagline: 105 道题 · shared memory · tiling · bank conflict · 内存布局 · 性能优化
  actions:
    - theme: brand
      text: 开始刷题
      link: /easy.html
    - theme: alt
      text: GitHub 仓库
      link: https://github.com/hzchenxiaobin/leetgpu

features:
  - icon: 🟢
    title: Easy · 简单
    details: Vector Addition、Matrix Transpose、ReLU、Sigmoid 等入门题，掌握 kernel 基本写法与 coalesced 访存。
    link: /easy.html
    linkText: 进入 Easy
  - icon: 🟡
    title: Medium · 中等
    details: Softmax、Prefix Sum、GEMM、LayerNorm、RoPE、MoE Gating 等，深入 tiling、warp shuffle 与归约范式。
    link: /medium.html
    linkText: 进入 Medium
  - icon: 🔴
    title: Hard · 困难
    details: Multi-Head Attention、FFT、Radix Sort、LLaMA Transformer Block 等，逼近手写 kernel 的性能极限。
    link: /hard.html
    linkText: 进入 Hard
---

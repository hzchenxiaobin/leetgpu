#!/usr/bin/env python3
"""Generate SVG figures for the LeetGPU reduction solution note."""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent


def draw_block(color, x, y, width, height, ax, label="", text_color="white", fontsize=9, alpha=1.0):
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color, edgecolor="black", linewidth=1.5, alpha=alpha
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=fontsize, color=text_color, weight="bold")


def draw_arrow(ax, x1, y1, x2, y2, color="#374151", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


def generate_reduction_overview():
    """Two-stage reduction pipeline overview."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 5.7, "Reduction 两级归约流程", ha="center", va="center", fontsize=22, weight="bold")

    # Input array
    input_vals = ["3", "1", "4", "1", "5", "9", "2", "6"]
    n = len(input_vals)
    elem_w = 0.9
    start_x = 1.8
    y = 4.3
    for i, v in enumerate(input_vals):
        draw_block("#4C78A8", start_x + i * elem_w, y, elem_w - 0.05, 0.55, ax,
                   label=f"{v}", fontsize=13)
    ax.text(6, y + 0.8, "Input Array (N elements)", ha="center", fontsize=15)

    # Stage 1 kernel
    ax.text(3.0, 3.5, "Kernel 1: reduction_kernel<<<blocks, threads>>>",
            ha="center", fontsize=15, weight="bold")

    stages = [
        ("① Thread-level\ngrid-stride 累加", 1.2, 2.6, "#F58518"),
        ("② Warp-level\n__shfl_down_sync", 3.6, 2.6, "#E45756"),
        ("③ Block-level\nwarpSums[32]", 6.0, 2.6, "#72B7B2"),
        ("④ 输出 block 部分和", 9.0, 2.6, "#59A14F"),
    ]
    for label, x, y, color in stages:
        draw_block(color, x - 0.8, y - 0.35, 1.6, 0.7, ax, label=label, fontsize=12)

    # d_temp
    temp_vals = ["S0", "S1", "...", "S_n"]
    elem_w = 0.8
    start_x = 4.5
    y = 0.8
    for i, v in enumerate(temp_vals):
        draw_block("#72B7B2", start_x + i * elem_w, y, elem_w - 0.05, 0.5, ax,
                   label=v, fontsize=13)
    ax.text(6, y + 0.95, "d_temp[blockIdx.x] = 每个 block 的部分和", ha="center", fontsize=14)
    draw_arrow(ax, 6.0, 2.25, 6.0, 1.35)

    # Stage 2 kernel
    ax.text(3.0, 0.2, "Kernel 2: reduction_kernel<<<1, 256>>>(d_temp, d_out)",
            ha="center", fontsize=15, weight="bold")
    draw_block("#59A14F", 9.0, -0.1, 1.2, 0.5, ax, label="sum", fontsize=15)
    draw_arrow(ax, 7.8, 0.8, 9.0, 0.15)

    ax.text(6, -0.25,
            "第一次 kernel 把 N 个元素归约成 blocks 个部分和；第二次 kernel 再把 blocks 个部分和归约成最终结果",
            ha="center", fontsize=14,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "reduction_overview.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_grid_stride():
    """Illustrate grid-stride loop: each thread handles multiple elements."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6.5, 4.7, "Grid-Stride Loop 工作原理", ha="center", va="center",
            fontsize=21, weight="bold")

    # Input array elements
    n_elems = 16
    elem_w = 0.65
    start_x = 1.0
    y_input = 3.2
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]
    for i in range(n_elems):
        color = colors[i % 4]
        draw_block(color, start_x + i * elem_w, y_input, elem_w - 0.05, 0.5, ax,
                   label=str(i), fontsize=13)
    ax.text(6.5, y_input + 0.8, "Input Array (N elements)", ha="center", fontsize=15)

    # Thread assignments
    thread_labels = [
        ("T0", "0,4,8,12", 1.0, 1.8, colors[0]),
        ("T1", "1,5,9,13", 2.0, 1.8, colors[1]),
        ("T2", "2,6,10,14", 3.0, 1.8, colors[2]),
        ("T3", "3,7,11,15", 4.0, 1.8, colors[3]),
    ]
    for tid, elems, x, y, color in thread_labels:
        draw_block(color, x, y, 0.9, 0.6, ax, label=f"{tid}\n{elems}", fontsize=12)
        # arrows from thread to its first element
        first_idx = int(elems.split(",")[0])
        elem_x = start_x + first_idx * elem_w + elem_w / 2
        ax.annotate("", xy=(elem_x, y_input), xytext=(x + 0.45, y + 0.6),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    ax.text(6.5, 1.5, "每个线程从自己的 tid 开始，以 stride = gridDim.x * blockDim.x 为步长遍历数组",
            ha="center", fontsize=14)

    # Code snippet
    code_text = (
        "int tid = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "int stride = gridDim.x * blockDim.x;\n"
        "for (int i = tid; i < N; i += stride) {\n"
        "    sum += input[i];\n"
        "}"
    )
    ax.text(11.5, 3.5, code_text, fontsize=11, family="monospace",
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    ax.text(6.5, 0.5,
            "线程数可以远小于 N；每个线程负责多个元素，保证所有线程负载均衡且内存访问合并",
            ha="center", fontsize=14,
            bbox=dict(boxstyle="round", facecolor="#FFF3CD", edgecolor="#FFC107"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "reduction_grid_stride.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def generate_reduction_block_internal():
    """Detailed execution inside a single block (256 threads = 8 warps)."""
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(8.0, 7.7, "单个 Block 内部的归约过程（以 256 线程 = 8 warps 为例）",
            ha="center", va="center", fontsize=21, weight="bold")

    # Step 1: grid-stride loop
    ax.text(1.5, 7.1, "Step 1: Thread-level grid-stride 累加", fontsize=17, weight="bold")
    input_blocks = [
        ("T0", "3+1+...", 0.5, 6.4),
        ("T1", "4+1+...", 2.0, 6.4),
        ("...", "...", 3.5, 6.4),
        ("T255", "5+9+...", 5.0, 6.4),
    ]
    for label, val, x, y in input_blocks:
        draw_block("#4C78A8", x, y, 1.1, 0.45, ax, label=f"{label}\n{val}", fontsize=11)
    ax.text(3.5, 6.05, "每个线程负责 input 中不相交的一段元素，求局部和", ha="center", fontsize=13)

    # Step 2: warp shuffle
    ax.text(1.5, 5.6, "Step 2: Warp-level __shfl_down_sync 归约", fontsize=17, weight="bold")
    for w in range(8):
        x = 0.5 + w * 1.8
        y = 4.8
        # warp box
        draw_block("#F58518", x, y, 1.5, 0.5, ax, label=f"Warp {w}\n32 threads", fontsize=11)
        # arrows showing shuffle butterfly
        ax.annotate("", xy=(x + 0.75, y - 0.15), xytext=(x + 0.75, y),
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1.2))
        draw_block("#E45756", x + 0.3, y - 0.6, 0.9, 0.35, ax,
                   label=f"sum{w}", fontsize=12)
    ax.text(8.0, 3.9, "每个 warp 内部通过 shuffle 把 32 个线程的局部和归约成 1 个值",
            ha="center", fontsize=13)

    # Step 3: write to warpSums
    ax.text(1.5, 3.6, "Step 3: warp 部分和写入 Shared Memory", fontsize=17, weight="bold")
    for w in range(8):
        x = 0.7 + w * 1.8
        y = 3.0
        draw_block("#72B7B2", x, y, 1.4, 0.4, ax, label=f"warpSums[{w}]\n= sum{w}", fontsize=11)
    ax.text(8.0, 2.5, "只有每个 warp 的 lane 0 写入 warpSums[wid]，无 bank conflict",
            ha="center", fontsize=13)

    # Step 4: warp 0 final reduce
    ax.text(1.5, 2.1, "Step 4: Warp 0 读取 warpSums 做最终归约", fontsize=17, weight="bold")
    draw_block("#72B7B2", 4.5, 1.4, 5.0, 0.45, ax,
               label="warpSums[0..7] → __shfl_down_sync → block_sum", fontsize=13)
    draw_arrow(ax, 8.0, 3.0, 8.0, 1.9)
    draw_block("#59A14F", 11.0, 1.45, 1.8, 0.45, ax, label="output[blockIdx.x]", fontsize=12)
    draw_arrow(ax, 9.5, 1.65, 11.0, 1.65)

    ax.text(8.0, 0.7,
            "第二次 kernel 用同样的逻辑把 d_temp 中的 block 部分和再归约成全局总和",
            ha="center", fontsize=14,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "reduction_block_internal.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_reduction_overview()
    generate_grid_stride()
    generate_reduction_block_internal()
    print(f"Generated SVG figures in {OUT_DIR}:")
    print(f"  - reduction_overview.svg")
    print(f"  - reduction_grid_stride.svg")
    print(f"  - reduction_block_internal.svg")

#!/usr/bin/env python3
"""Generate figures for the LeetGPU argmax solution note."""

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


def generate_argmax_overview():
    """Two-stage reduction pipeline for argmax."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(6, 4.7, "Argmax 两级归约流程", ha="center", va="center", fontsize=15, weight="bold")

    # Input array
    input_vals = ["3.2", "1.5", "7.8", "2.1", "7.8", "0.9", "4.5", "7.8"]
    n = len(input_vals)
    elem_w = 1.0
    start_x = 1.0
    y = 3.5
    for i, v in enumerate(input_vals):
        draw_block("#4C78A8", start_x + i * elem_w, y, elem_w - 0.05, 0.6, ax,
                   label=f"{v}\nidx={i}", fontsize=7)
    ax.text(6, y + 0.85, "Input Array (N elements)", ha="center", fontsize=11)

    # Stage labels
    stages = [
        ("① Thread-level\ngrid-stride", 1.5, 2.4, "#F58518"),
        ("② Warp-level\n__shfl_down_sync", 4.5, 2.4, "#E45756"),
        ("③ Block-level\nShared Memory", 7.5, 2.4, "#72B7B2"),
        ("④ Cross-block\natomicMax", 10.5, 2.4, "#59A14F"),
    ]
    for label, x, y, color in stages:
        draw_block(color, x - 0.7, y - 0.35, 1.4, 0.7, ax, label=label, fontsize=8)

    # Output
    draw_block("#59A14F", 9.8, 0.8, 1.4, 0.6, ax, label="argmax idx\n= 2", fontsize=10)
    draw_arrow(ax, 10.5, 2.05, 10.5, 1.45)

    # annotations
    ax.text(6, 0.3,
            "每个线程维护 (max_val, max_idx)，逐级归约后最终输出全局最大值的索引",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "argmax_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_grid_stride():
    """Grid-stride loop: each thread handles multiple elements."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(5.5, 3.7, "Grid-Stride Loop：每个线程扫描多个元素", ha="center", fontsize=14, weight="bold")

    n = 16
    elem_w = 0.5
    start_x = 0.5
    y = 2.5
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2"]

    for i in range(n):
        tid = i % 4
        block = i // 4
        color = colors[tid]
        draw_block(color, start_x + i * elem_w, y, elem_w - 0.02, 0.5, ax,
                   label=str(i), fontsize=8)

    # Thread labels
    for tid in range(4):
        ax.text(start_x + tid * elem_w + 0.2, y + 0.7, f"T{tid}",
                ha="center", fontsize=9, color=colors[tid], weight="bold")

    # Stride arrows for T0
    for step in range(3):
        x1 = start_x + 0.2 + step * 4 * elem_w
        x2 = start_x + 0.2 + (step + 1) * 4 * elem_w
        draw_arrow(ax, x1, y - 0.3, x2, y - 0.3, color=colors[0], lw=2)

    ax.text(5.5, 1.5,
            r"stride = gridDim.x $\times$ blockDim.x" + "\n"
            r"for (i = tid; i < N; i += stride) 维护局部 (max_val, max_idx)",
            ha="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "argmax_grid_stride.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_warp_shuffle():
    """Warp shuffle reduction with butterfly pattern."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(5, 4.7, "Warp Shuffle 归约（Butterfly）", ha="center", fontsize=14, weight="bold")

    # 8 lanes
    lanes = 8
    x_positions = [1 + i * 1.0 for i in range(lanes)]
    y_start = 3.8
    row_gap = 0.8

    lane_labels = [f"L{i}\n(7,6)" for i in range(lanes)]
    for i, (x, label) in enumerate(zip(x_positions, lane_labels)):
        draw_block("#4C78A8", x - 0.35, y_start, 0.7, 0.5, ax, label=label, fontsize=7)

    # Reduction steps
    steps = [(4, "offset=4"), (2, "offset=2"), (1, "offset=1")]
    for step_idx, (offset, label) in enumerate(steps):
        y = y_start - (step_idx + 1) * row_gap
        # draw arrows from previous row
        prev_y = y_start - step_idx * row_gap + 0.25
        for i in range(lanes):
            x1 = x_positions[i]
            x2 = x_positions[(i + offset) % lanes]
            if i < lanes - offset:
                draw_arrow(ax, x1, prev_y - 0.1, x2, y + 0.35, color="#9CA3AF", lw=1)

        # draw result boxes
        for i in range(lanes):
            if step_idx == 2:
                text = "(7,6)" if i == 0 else ""
            else:
                text = ""
            color = "#E45756" if step_idx == 2 and i == 0 else "#F58518"
            draw_block(color, x_positions[i] - 0.35, y, 0.7, 0.5, ax, label=text, fontsize=7)

        ax.text(0.3, y + 0.25, label, ha="right", va="center", fontsize=9)

    # Lane 0 holds the warp result
    ax.text(5, 0.4,
            "Lane 0 保存当前 warp 的 (max_val, max_idx)，存入 Shared Memory s_val[wid] / s_idx[wid]",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "argmax_warp_shuffle.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_tie_breaking():
    """Tie-breaking: same value, smaller index wins."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    ax.axis("off")

    ax.text(4.5, 3.7, "平局处理：值相同时取下标更小者", ha="center", fontsize=14, weight="bold")

    # Two candidate boxes
    draw_block("#4C78A8", 1.0, 2.0, 1.8, 0.8, ax, label="A\nval=7.8\nidx=2", fontsize=10)
    draw_block("#F58518", 4.0, 2.0, 1.8, 0.8, ax, label="B\nval=7.8\nidx=5", fontsize=10)

    draw_arrow(ax, 2.8, 2.4, 4.0, 2.4, color="#374151", lw=2)

    # Comparison logic
    ax.text(6.2, 2.4,
            "if (B.val > A.val) 取 B\n"
            "else if (B.val == A.val && B.idx < A.idx) 取 B\n"
            "else 取 A",
            ha="left", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#FEF3C7", edgecolor="#F59E0B"))

    # Result
    draw_block("#59A14F", 3.0, 0.6, 2.0, 0.7, ax, label="Result\nval=7.8, idx=2", fontsize=11)
    draw_arrow(ax, 4.5, 2.0, 4.0, 1.35, color="#374151", lw=2)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "argmax_tie_breaking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_argmax_overview()
    generate_grid_stride()
    generate_warp_shuffle()
    generate_tie_breaking()
    print("Argmax figures generated in", OUT_DIR)

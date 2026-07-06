#!/usr/bin/env python3
"""Generate figures for the LeetGPU prefix sum solution note."""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent


def draw_block(color, x, y, width, height, ax, label="", text_color="white"):
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color, edgecolor="black", linewidth=1.5
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=9, color=text_color, weight="bold")


def generate_thread_block_mapping():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    ax.text(5, 3.2, "Input array 划分到 Block 与 Thread",
            ha="center", va="center", fontsize=14, weight="bold")

    colors = ["#4C78A8", "#F58518", "#E45756"]
    n_blocks = 3
    block_width = 2.8
    block_height = 1.0
    start_x = 0.4
    gap = 0.3

    values = ["a₀", "a₁", "a₂", "a₃", "a₄", "a₅", "a₆", "a₇", "a₈", "a₉", "a₁₀", "a₁₁"]
    elems_per_block = 4

    for b in range(n_blocks):
        bx = start_x + b * (block_width + gap)
        by = 1.6
        draw_block(colors[b], bx, by, block_width, block_height, ax,
                   label=f"Block {b}")

        # draw individual elements
        elem_w = block_width / elems_per_block
        for t in range(elems_per_block):
            idx = b * elems_per_block + t
            if idx >= len(values):
                break
            ex = bx + t * elem_w
            ey = by - 0.55
            draw_block("#FFFFFF", ex, ey, elem_w - 0.02, 0.45, ax,
                       label=values[idx], text_color="black")
            ax.text(ex + elem_w / 2, ey - 0.25, f"thread {t}",
                    ha="center", va="top", fontsize=7)

    # legend / formula
    ax.text(5, 0.45,
            r"global id = blockIdx.x $\times$ blockDim.x + threadIdx.x",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "thread_block_mapping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_blelloch_scan():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    data = [3, 1, 7, 0, 4, 1, 6, 3]
    n = len(data)
    levels = int(np.log2(n)) + 1

    def draw_tree(ax, title, arrows, highlight_nodes=None):
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.5, levels + 0.5)
        ax.set_title(title, fontsize=13, weight="bold", pad=10)
        ax.axis("off")

        # node positions
        pos = {}
        for level in range(levels):
            step = 2 ** level
            count = n // step
            y = levels - level - 1
            for i in range(count):
                x = i * step + (step - 1) / 2
                pos[(level, i)] = (x, y)
                circle = plt.Circle((x, y), 0.28, color="white", ec="black", linewidth=1.5, zorder=2)
                ax.add_patch(circle)
                # value text
                val = data[i * step] if level == 0 else ""
                ax.text(x, y, val, ha="center", va="center", fontsize=9, zorder=3)

        # arrows
        for (src, dst), color in arrows:
            x1, y1 = pos[src]
            x2, y2 = pos[dst]
            ax.annotate("", xy=(x2, y2 - 0.28), xytext=(x1, y1 + 0.28),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

        if highlight_nodes:
            for node, label, color in highlight_nodes:
                x, y = pos[node]
                circle = plt.Circle((x, y), 0.30, color=color, ec="black", linewidth=2, zorder=4, alpha=0.6)
                ax.add_patch(circle)
                ax.text(x, y, label, ha="center", va="center", fontsize=8, zorder=5, weight="bold")

    # Up-sweep arrows: each node adds from left child
    up_arrows = []
    for level in range(1, levels):
        step = 2 ** level
        count = n // step
        for i in range(count):
            up_arrows.append((((level - 1, 2 * i + 1), (level, i)), "#4C78A8"))

    # Down-sweep arrows: reverse, propagate to children
    down_arrows = []
    for level in range(levels - 1, 0, -1):
        step = 2 ** level
        count = n // step
        for i in range(count):
            down_arrows.append((((level, i), (level - 1, 2 * i)), "#F58518"))
            down_arrows.append((((level, i), (level - 1, 2 * i + 1)), "#E45756"))

    draw_tree(axes[0], "Up-sweep（归约）", up_arrows)
    draw_tree(axes[1], "Down-sweep（分发）", down_arrows)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "blelloch_scan.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_thread_block_mapping()
    generate_blelloch_scan()
    print("Figures generated in", OUT_DIR)

#!/usr/bin/env python3
"""Generate figures for the LeetGPU matrix addition solution note."""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent


def draw_block(color, x, y, width, height, ax, label="", text_color="white", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=color, edgecolor="black", linewidth=1.5
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=fontsize, color=text_color, weight="bold")


def generate_matrix_addition_mapping():
    """Element-wise matrix addition with thread mapping."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    ax.text(5.5, 4.2, "Matrix Addition：每个线程负责一个（或一组）元素",
            ha="center", va="center", fontsize=14, weight="bold")

    # Matrix A
    ax.text(1.5, 3.6, "Matrix A", ha="center", fontsize=11, weight="bold")
    values_a = [["1.2", "3.4"], ["5.6", "7.8"]]
    for i in range(2):
        for j in range(2):
            draw_block("#4C78A8", 0.5 + j * 1.0, 2.4 + i * 0.6, 0.95, 0.55, ax,
                       label=values_a[1 - i][j], fontsize=10)

    # Plus sign
    ax.text(3.0, 3.0, "+", ha="center", va="center", fontsize=20, weight="bold")

    # Matrix B
    ax.text(4.5, 3.6, "Matrix B", ha="center", fontsize=11, weight="bold")
    values_b = [["0.8", "0.6"], ["0.4", "0.2"]]
    for i in range(2):
        for j in range(2):
            draw_block("#F58518", 3.5 + j * 1.0, 2.4 + i * 0.6, 0.95, 0.55, ax,
                       label=values_b[1 - i][j], fontsize=10)

    # Equal sign
    ax.text(6.0, 3.0, "=", ha="center", va="center", fontsize=20, weight="bold")

    # Matrix C
    ax.text(7.5, 3.6, "Matrix C", ha="center", fontsize=11, weight="bold")
    values_c = [["2.0", "4.0"], ["6.0", "8.0"]]
    for i in range(2):
        for j in range(2):
            draw_block("#59A14F", 6.5 + j * 1.0, 2.4 + i * 0.6, 0.95, 0.55, ax,
                       label=values_c[1 - i][j], fontsize=10)

    # Thread mapping illustration
    ax.text(9.5, 3.3, "Thread\nMapping", ha="center", va="center", fontsize=10)
    for t in range(4):
        row = t // 2
        col = t % 2
        ax.text(9.0 + col * 0.9, 2.3 + row * 0.5, f"T{t}",
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#E5E7EB", edgecolor="#6B7280"))

    # Annotation
    ax.text(5.5, 0.7,
            "每个线程读取 A、B 对应位置的元素，相加后写入 C\n"
            "无数据依赖，适合用 float4 向量化加载以饱和显存带宽",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#F3F4F6", edgecolor="#9CA3AF"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matrix_addition_mapping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_matrix_addition_mapping()
    print("Matrix Addition figures generated in", OUT_DIR)

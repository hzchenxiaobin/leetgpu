#!/usr/bin/env python3
"""Generate hand-drawn sketch style figures for the LeetGPU matrix multiplication solution note."""

import random
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Bradley Hand', 'Comic Sans MS', 'Kaiti SC', 'Heiti TC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from pathlib import Path

OUT_DIR = Path(__file__).parent
random.seed(42)

# Sketchy marker / whiteboard palette
COLORS = {
    "A": "#5B9A8B",      # teal marker
    "B": "#E8A838",      # orange marker
    "C": "#7CB87C",      # green marker
    "accent": "#C75B5B", # red marker
    "ink": "#2C2C2C",    # dark ink
    "pencil": "#9E9E9E", # grey pencil
    "paper": "#FDFBF7",  # off-white paper
}


def sketched_font(size=10, color=None, weight="normal"):
    """Return FontProperties that look handwritten while supporting CJK fallback."""
    fp = FontProperties(family=['Bradley Hand', 'Comic Sans MS', 'Kaiti SC', 'Heiti TC'], size=size, weight=weight)
    return {"fontproperties": fp, "color": color or COLORS["ink"]}


def jitter(amount=0.012):
    return random.uniform(-amount, amount)


def sketch_line(x1, y1, x2, y2, n_segments=18):
    """Generate a wobbly polyline between two points."""
    xs = np.linspace(x1, x2, n_segments)
    ys = np.linspace(y1, y2, n_segments)
    # Add small perpendicular wobble
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy) + 1e-9
    perp_x, perp_y = -dy / length, dx / length
    for i in range(1, n_segments - 1):
        amp = random.uniform(0.008, 0.020) * length
        xs[i] += perp_x * amp + jitter(0.005)
        ys[i] += perp_y * amp + jitter(0.005)
    return xs, ys


def sketch_rect_path(x, y, width, height, n_points_per_side=12):
    """Generate a wobbly rectangle path; (x, y) is the bottom-left corner."""
    pts = []
    # bottom edge (left to right)
    xs = np.linspace(x, x + width, n_points_per_side)
    ys = np.full_like(xs, y)
    for xi, yi in zip(xs, ys):
        pts.append((xi + jitter(0.006), yi + jitter(0.010)))
    # right edge (bottom to top)
    ys = np.linspace(y, y + height, n_points_per_side)[1:]
    xs = np.full_like(ys, x + width)
    for xi, yi in zip(xs, ys):
        pts.append((xi + jitter(0.010), yi + jitter(0.006)))
    # top edge (right to left)
    xs = np.linspace(x + width, x, n_points_per_side)[1:]
    ys = np.full_like(xs, y + height)
    for xi, yi in zip(xs, ys):
        pts.append((xi + jitter(0.006), yi + jitter(0.010)))
    # left edge (top to bottom)
    ys = np.linspace(y + height, y, n_points_per_side)[1:]
    xs = np.full_like(ys, x)
    for xi, yi in zip(xs, ys):
        pts.append((xi + jitter(0.010), yi + jitter(0.006)))
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY]
    return MplPath(pts, codes)


def sketch_arrow(ax, x1, y1, x2, y2, color=None, lw=2.0):
    """Draw a hand-drawn arrow."""
    color = color or COLORS["ink"]
    angle = np.arctan2(y2 - y1, x2 - x1)
    head_len = 0.14
    # Tip sits slightly before the target point
    tip_x = x2 - 0.02 * np.cos(angle)
    tip_y = y2 - 0.02 * np.sin(angle)
    # Base of the arrowhead (where it meets the shaft)
    base_x = tip_x - head_len * np.cos(angle)
    base_y = tip_y - head_len * np.sin(angle)
    # Draw wobbly shaft
    xs, ys = sketch_line(x1, y1, base_x, base_y)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round")
    # Draw triangular arrowhead with two wobbly sides
    spread = 0.45
    perp_x, perp_y = -np.sin(angle), np.cos(angle)
    half_width = head_len * np.tan(spread)
    for sign in (-1, 1):
        corner_x = base_x + sign * half_width * perp_x
        corner_y = base_y + sign * half_width * perp_y
        ax.plot(*sketch_line(corner_x, corner_y, tip_x, tip_y, n_segments=5),
                color=color, lw=lw, solid_capstyle="round")


def draw_block(color, x, y, width, height, ax, label="", text_color="white", fontsize=9, alpha=0.9, edgecolor=None, lw=2.0):
    """Draw a hand-sketched filled block with rounded-ish corners via wobbly rect."""
    edgecolor = edgecolor or COLORS["ink"]
    path = sketch_rect_path(x, y, width, height)
    patch = mpatches.PathPatch(path, facecolor=color, edgecolor=edgecolor, linewidth=lw, alpha=alpha)
    ax.add_patch(patch)
    if label:
        ax.text(x + width / 2 + jitter(0.008), y + height / 2 + jitter(0.008), label,
                ha="center", va="center", **sketched_font(size=fontsize, color=text_color, weight="bold"))


def draw_matrix_outline(ax, x, y, n_rows, n_cols, cell_size, label="", lw=2.0):
    """Draw a hand-drawn matrix grid."""
    width = n_cols * cell_size
    height = n_rows * cell_size
    # Outer border
    path = sketch_rect_path(x, y - height, width, height)
    ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=COLORS["ink"], linewidth=lw))
    # Grid lines
    for i in range(1, n_rows):
        xs, ys = sketch_line(x, y - i * cell_size, x + width, y - i * cell_size, n_segments=12)
        ax.plot(xs, ys, color=COLORS["pencil"], lw=1.0, alpha=0.7)
    for j in range(1, n_cols):
        xs, ys = sketch_line(x + j * cell_size, y, x + j * cell_size, y - height, n_segments=12)
        ax.plot(xs, ys, color=COLORS["pencil"], lw=1.0, alpha=0.7)
    if label:
        ax.text(x + width / 2 + jitter(0.015), y + 0.30 + jitter(0.010), label,
                ha="center", va="bottom", **sketched_font(size=12, weight="bold"))


def highlight_cells(ax, x, y, n_rows, n_cols, cell_size, indices, color, alpha=0.75, label=None, label_size=10):
    """Highlight specific cells with a marker wash and wobbly borders."""
    for r, c in indices:
        cx = x + c * cell_size + jitter(0.004)
        cy = y - (r + 1) * cell_size + jitter(0.004)
        path = sketch_rect_path(cx, cy, cell_size, cell_size)
        ax.add_patch(mpatches.PathPatch(path, facecolor=color, edgecolor=COLORS["ink"], linewidth=1.0, alpha=alpha))
    if label:
        r, c = indices[len(indices) // 2]
        ax.text(x + (c + 0.5) * cell_size + jitter(0.005), y - (r + 0.5) * cell_size + jitter(0.005), label,
                ha="center", va="center", **sketched_font(size=label_size, color="white", weight="bold"))


def make_figure(figsize):
    """Create a figure with whiteboard-style background."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.axis("off")
    return fig, ax


def generate_naive_flow():
    """Naive GEMM: each thread computes one C element."""
    fig, ax = make_figure((12, 5))
    ax.text(6 + jitter(0.02), 4.7 + jitter(0.015), "Naive GEMM：每个线程计算 C 的一个元素",
            ha="center", va="center", **sketched_font(size=15, weight="bold"))

    cell = 0.36
    # Matrix A
    ax_A, ay_A = 0.8, 3.6
    draw_matrix_outline(ax, ax_A, ay_A, 5, 6, cell, label="A (M×K)")
    highlight_cells(ax, ax_A, ay_A, 5, 6, cell, [(1, c) for c in range(6)], COLORS["A"], label="row i", label_size=10)

    # Matrix B
    ax_B, ay_B = 4.8, 3.6
    draw_matrix_outline(ax, ax_B, ay_B, 6, 5, cell, label="B (K×N)")
    highlight_cells(ax, ax_B, ay_B, 6, 5, cell, [(r, 2) for r in range(6)], COLORS["B"], label="col j", label_size=10)

    # Matrix C
    ax_C, ay_C = 8.8, 3.6
    draw_matrix_outline(ax, ax_C, ay_C, 5, 5, cell, label="C (M×N)")
    highlight_cells(ax, ax_C, ay_C, 5, 5, cell, [(1, 2)], COLORS["C"], label="C[i][j]", label_size=10)

    # Arrows
    sketch_arrow(ax, ax_A + 6 * cell + 0.2, ay_A - (1 + 0.5) * cell,
                 ax_C + 2 * cell - 0.2, ay_C - 0.5 * cell, color=COLORS["A"], lw=2.2)
    sketch_arrow(ax, ax_B + (2 + 0.5) * cell, ay_B - 6 * cell - 0.2,
                 ax_C + 2.5 * cell, ay_C - (1 + 1) * cell + 0.2, color=COLORS["B"], lw=2.2)

    # Formula
    ax.text(6, 1.5,
            r"$C[i][j] = \sum_{k=0}^{K-1} A[i][k] \times B[k][j]$" + "\n"
            "每个线程：读 A 的一整行 + B 的一整列 → 写 C 的一个元素",
            ha="center", va="center", **sketched_font(size=11),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3F4F6", edgecolor=COLORS["pencil"], linewidth=2))

    # Bottleneck note
    ax.text(6, 0.55,
            "瓶颈：A 的同一行 / B 的同一列被多个线程重复读取",
            ha="center", va="center", **sketched_font(size=10, color=COLORS["accent"]),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEE2E2", edgecolor=COLORS["accent"], linewidth=2))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matmul_naive.png", dpi=200, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


def generate_tiled_flow():
    """Shared memory tiling: blocks load tiles of A/B and accumulate over K phases."""
    fig, ax = make_figure((14, 6))
    ax.text(7 + jitter(0.02), 5.7 + jitter(0.015), "Shared Memory Tiling：分块加载 + K 维复用",
            ha="center", va="center", **sketched_font(size=15, weight="bold"))

    cell = 0.36

    # Matrix A
    ax_A, ay_A = 0.8, 4.4
    draw_matrix_outline(ax, ax_A, ay_A, 6, 8, cell, label="A (M×K)")
    tile_a_indices = [(r, c) for r in range(1, 4) for c in range(2, 5)]
    highlight_cells(ax, ax_A, ay_A, 6, 8, cell, tile_a_indices, COLORS["A"], label="A tile", label_size=11)

    # Matrix B
    ax_B, ay_B = 4.8, 4.4
    draw_matrix_outline(ax, ax_B, ay_B, 8, 6, cell, label="B (K×N)")
    tile_b_indices = [(r, c) for r in range(2, 5) for c in range(1, 4)]
    highlight_cells(ax, ax_B, ay_B, 8, 6, cell, tile_b_indices, COLORS["B"], label="B tile", label_size=11)

    # Matrix C
    ax_C, ay_C = 9.6, 4.4
    draw_matrix_outline(ax, ax_C, ay_C, 6, 6, cell, label="C (M×N)")
    tile_c_indices = [(r, c) for r in range(1, 4) for c in range(1, 4)]
    highlight_cells(ax, ax_C, ay_C, 6, 6, cell, tile_c_indices, COLORS["C"], label="C tile", label_size=11)

    # Shared memory boxes (placed below matrices to avoid overlap)
    sx, sy = 4.3, 0.9
    draw_block(COLORS["A"], sx, sy, 2.2, 1.0, ax,
               label="s_A[TILE][TILE]\n(Shared Memory)", fontsize=9, lw=2.5)
    draw_block(COLORS["B"], sx + 2.8, sy, 2.2, 1.0, ax,
               label="s_B[TILE][TILE]\n(Shared Memory)", fontsize=9, lw=2.5)

    # Arrows
    sketch_arrow(ax, ax_A + (2 + 1.5) * cell, ay_A - (1 + 3) * cell - 0.15,
                 sx + 1.1, sy + 1.0 + 0.1, color=COLORS["A"], lw=2.2)
    sketch_arrow(ax, ax_B + (1 + 1.5) * cell, ay_B - (2 + 3) * cell - 0.15,
                 sx + 2.8 + 1.1, sy + 1.0 + 0.1, color=COLORS["B"], lw=2.2)
    sketch_arrow(ax, sx + 2.2 + 0.15, sy + 0.5,
                 ax_C + (1 + 0.5) * cell - 0.2, ay_C - (1 + 3) * cell - 0.15,
                 color=COLORS["C"], lw=2.2)

    # Phase annotation
    ax.text(7, 0.65,
            "① Block 协作把 A/B 的 TILE 加载到 Shared Memory\n"
            "② 每个线程用 s_A 的一行 × s_B 的一列累加部分和\n"
            "③ 沿 K 维滑动 tile，重复加载 → 累加，直到 K 遍历完成",
            ha="center", va="center", **sketched_font(size=11),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F3F4F6", edgecolor=COLORS["pencil"], linewidth=2))

    # K-dim labels
    ax.text(2.5 + jitter(0.015), 1.55, "K 维度分块", ha="center",
            **sketched_font(size=11, color=COLORS["A"], weight="bold"))
    ax.text(5.8 + jitter(0.015), 1.55, "K 维度分块", ha="center",
            **sketched_font(size=11, color=COLORS["B"], weight="bold"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matmul_tiled.png", dpi=200, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


def generate_thread_block_mapping():
    """Show how thread blocks and threads map to C tile."""
    fig, ax = make_figure((12, 5.5))
    ax.text(6 + jitter(0.02), 5.2 + jitter(0.015), "Thread / Block 映射：一个 Block 负责 C 的一个 TILE",
            ha="center", va="center", **sketched_font(size=15, weight="bold"))

    cell = 0.55
    n_tiles = 3
    tile_size = 3
    mx, my = 2.0, 5.0
    width = n_tiles * tile_size * cell
    height = n_tiles * tile_size * cell

    # Outer border
    path = sketch_rect_path(mx, my - height, width, height)
    ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=COLORS["ink"], linewidth=2.5))

    # Tile boundaries
    for i in range(1, n_tiles):
        xs, ys = sketch_line(mx, my - i * tile_size * cell, mx + width, my - i * tile_size * cell, n_segments=14)
        ax.plot(xs, ys, color=COLORS["pencil"], lw=1.5, alpha=0.8)
        xs, ys = sketch_line(mx + i * tile_size * cell, my, mx + i * tile_size * cell, my - height, n_segments=14)
        ax.plot(xs, ys, color=COLORS["pencil"], lw=1.5, alpha=0.8)

    # Blocks
    for bi in range(n_tiles):
        for bj in range(n_tiles):
            x = mx + bj * tile_size * cell
            y = my - bi * tile_size * cell
            path = sketch_rect_path(x, y - tile_size * cell, tile_size * cell, tile_size * cell)
            ax.add_patch(mpatches.PathPatch(path, facecolor="#E8E8E8", edgecolor=COLORS["pencil"],
                                            linewidth=1.2, alpha=0.5))
            ax.text(x + tile_size * cell / 2 + jitter(0.008), y - tile_size * cell / 2 + jitter(0.008),
                    f"Block\n({bj},{bi})", ha="center", va="center", **sketched_font(size=8, color=COLORS["ink"]))

    ax.text(mx + width / 2 + jitter(0.015), my - height - 0.25, "C 矩阵 (M×N)",
            ha="center", va="top", **sketched_font(size=12, weight="bold"))

    # Zoom into Block (1,1)
    target_bj, target_bi = 1, 1
    zx, zy = 8.6, 4.0
    zoom_size = tile_size * cell
    path = sketch_rect_path(zx, zy - zoom_size, zoom_size, zoom_size)
    ax.add_patch(mpatches.PathPatch(path, facecolor="none", edgecolor=COLORS["ink"], linewidth=2.5))

    for ti in range(tile_size):
        for tj in range(tile_size):
            x = zx + tj * cell
            y = zy - ti * cell
            color = COLORS["A"] if (ti, tj) == (1, 1) else COLORS["B"]
            path = sketch_rect_path(x, y - cell, cell, cell)
            ax.add_patch(mpatches.PathPatch(path, facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.85))
            ax.text(x + cell / 2 + jitter(0.005), y - cell / 2 + jitter(0.005),
                    f"T\n({tj},{ti})", ha="center", va="center", **sketched_font(size=8, color="white", weight="bold"))

    ax.text(zx + zoom_size / 2 + jitter(0.015), zy + 0.25,
            f"Block ({target_bj},{target_bi}) 内部：TILE_SIZE×TILE_SIZE 线程",
            ha="center", **sketched_font(size=12, weight="bold"))

    # Arrow
    src_x = mx + (target_bj + 0.5) * tile_size * cell
    src_y = my - (target_bi + 0.5) * tile_size * cell
    sketch_arrow(ax, src_x, src_y, zx - 0.1, zy - zoom_size / 2, color=COLORS["ink"], lw=2.5)

    # Formula
    ax.text(6, 0.35,
            "row = blockIdx.y × TILE_SIZE + threadIdx.y\n"
            "col = blockIdx.x × TILE_SIZE + threadIdx.x\n"
            "每个线程负责 C[row][col] 的最终累加结果",
            ha="center", va="center", **sketched_font(size=11),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF3C7", edgecolor=COLORS["B"], linewidth=2))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "matmul_thread_block_mapping.png", dpi=200, bbox_inches="tight", facecolor=COLORS["paper"])
    plt.close(fig)


if __name__ == "__main__":
    generate_naive_flow()
    generate_tiled_flow()
    generate_thread_block_mapping()
    print(f"Generated sketch-style matrix multiplication figures in {OUT_DIR}")

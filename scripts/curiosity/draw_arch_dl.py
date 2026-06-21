#!/usr/bin/env python3
"""Render the curiosity-detector architectures (MLP + DeepSets/SetNet) for the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/curiosity/report/architecture.png"


def box(ax, x, y, w, h, text, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=5)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.3, color="#444"))


fig, axes = plt.subplots(2, 1, figsize=(11, 9))
BLUE, ORANGE, GREEN, RED, GREY = "#cfe3f7", "#ffe0c2", "#d7f0d0", "#f7cfcf", "#eeeeee"

# ===== A) MLP =====
ax = axes[0]; ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis("off")
ax.set_title("A)  MLP detector  —  hand-crafted disagreement statistics", fontsize=12, loc="left", weight="bold")
box(ax, 0.2, 1.3, 2.4, 1.4,
    "8 disagreement\nstats / frame\n(disp_mean, disp_1s,\npathlen_std, …)\nNO ADE", GREY, 8)
box(ax, 3.4, 1.5, 1.9, 1.0, "Linear 8→64\nReLU + Drop0.3", BLUE)
box(ax, 5.9, 1.5, 1.9, 1.0, "Linear 64→32\nReLU + Drop0.3", BLUE)
box(ax, 8.4, 1.5, 1.7, 1.0, "Linear 32→1", ORANGE)
box(ax, 10.7, 1.5, 1.9, 1.0, "sigmoid", GREEN)
box(ax, 13.2, 1.5, 2.4, 1.0, "P(fail)\nADE top-10%?", RED)
for x1, x2 in [(2.6, 3.4), (5.3, 5.9), (7.8, 8.4), (10.1, 10.7), (12.6, 13.2)]:
    arrow(ax, x1, 2.0, x2, 2.0)

# ===== B) DeepSets / SetNet =====
ax = axes[1]; ax.set_xlim(0, 16); ax.set_ylim(0, 5.6); ax.axis("off")
ax.set_title("B)  SetNet (DeepSets)  —  permutation-invariant over M VLM trajectories (learns disagreement)",
             fontsize=12, loc="left", weight="bold")
# M per-model trajectories
for i, yy in enumerate([4.2, 3.4, 2.0]):
    lbl = "model M traj\n(6×2 → 12)" if i == 2 else ("model 2 traj\n(6×2 → 12)" if i == 1 else "model 1 traj\n(6×2 → 12)")
    box(ax, 0.2, yy, 2.0, 0.7, lbl, GREY, 8)
ax.text(1.2, 2.95, "⋮", ha="center", fontsize=16)
# shared phi
box(ax, 3.0, 2.0, 2.0, 2.9, "shared φ\n(per model)\nLinear 12→64\nReLU+Drop0.2\nLinear 64→64\nReLU", BLUE, 8.5)
for yy in [4.55, 3.75, 2.35]:
    arrow(ax, 2.2, yy, 3.0, 3.4 if yy > 3 else 3.0)
# embeddings
box(ax, 5.6, 2.2, 1.7, 2.4, "M × 64\nembeddings", GREY, 9)
arrow(ax, 5.0, 3.4, 5.6, 3.4)
# pooling
box(ax, 7.9, 2.2, 2.3, 2.4, "perm-invariant\npool\nmean ⊕ std ⊕ max\n→ 192\n(std/max = disagreement)", ORANGE, 8)
arrow(ax, 7.3, 3.4, 7.9, 3.4)
# rho
box(ax, 10.7, 2.4, 2.0, 2.0, "head ρ\nLinear 192→64\nReLU+Drop0.3\nLinear 64→1", BLUE, 8.5)
arrow(ax, 10.2, 3.4, 10.7, 3.4)
box(ax, 13.1, 2.7, 1.4, 1.0, "sigmoid", GREEN)
arrow(ax, 12.7, 3.4, 13.1, 3.2)
box(ax, 13.0, 1.3, 2.6, 1.0, "P(fail)\nADE top-10%?", RED)
arrow(ax, 13.8, 2.7, 13.8, 2.3)

fig.suptitle("Curiosity Detection — model architectures (label = ADE top-10%; ADE never an input)",
             fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)

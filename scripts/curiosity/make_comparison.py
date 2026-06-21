#!/usr/bin/env python3
"""Final comparison figure across signal sources (measured AUROC, label=fail_top10)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results/curiosity/figs/comparison_auroc.png"

# Measured AUROC (label = fail_top10), single-frame / CUSUM / EWMA
DATA = {
    "VLM ensemble\n(cross-model disagreement)": [0.798, 0.715, 0.760],
    "SparseDrive\n(native mode score)":          [0.551, 0.500, 0.539],
    "DiffusionDrive\n(native mode score)":        [0.606, 0.575, 0.591],
}
DETS = ["single-frame", "CUSUM", "EWMA"]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]

groups = list(DATA)
x = np.arange(len(groups))
w = 0.25
plt.figure(figsize=(8, 4.5))
for j, det in enumerate(DETS):
    vals = [DATA[g][j] for g in groups]
    plt.bar(x + (j - 1) * w, vals, w, label=det, color=COLORS[j])
plt.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.6)
plt.text(len(groups) - 0.5, 0.51, "chance", fontsize=8, alpha=0.6)
plt.xticks(x, groups, fontsize=8)
plt.ylim(0.45, 0.85)
plt.ylabel("Detection AUROC (failure = top-10% L2)")
plt.title("Cross-model disagreement >> single-model native uncertainty\n"
          "(and temporal accumulation never beats single-frame)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT, dpi=140)
print("wrote", OUT)

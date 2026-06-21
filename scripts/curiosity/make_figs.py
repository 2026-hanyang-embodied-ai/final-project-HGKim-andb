#!/usr/bin/env python3
"""Stage 4 — Figures + summary table for the detector comparison.

Fig 1  roc.png            ROC overlay (threshold / cusum / ewma), primary label
Fig 2  detrate_vs_fpr.png event detection rate vs false-alarm rate
Fig 3  event_triggered.png event-triggered average signal around failure onset
Fig 4  example_scene.png   one scene: signal + detector statistics + failure/alarm marks
table  summary_table.md    AUROC + matched-FPR detection/lead table
"""
from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import detectors as D

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results/curiosity"
FIGS = OUT / "figs"
PRIMARY = "fail_top10"
DETS = ["threshold", "cusum", "ewma"]
COLORS = {"threshold": "#1f77b4", "cusum": "#d62728", "ewma": "#2ca02c"}
NAMES = {"threshold": "single-frame (baseline)", "cusum": "CUSUM", "ewma": "EWMA"}


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    scenes, signal, rows = D.load_signals()
    stats = D.compute_statistics(scenes, signal)
    y = np.array([int(r[PRIMARY]) for r in rows])
    z = D.standardize(signal)

    # ---- Fig 1: ROC ----
    plt.figure(figsize=(5, 5))
    for det in DETS:
        fpr, tpr, _ = roc_curve(y, stats[det])
        auc = roc_auc_score(y, stats[det])
        plt.plot(fpr, tpr, color=COLORS[det], label=f"{NAMES[det]} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    plt.xlabel("False-alarm rate (FPR)"); plt.ylabel("Catch rate (TPR)")
    plt.title(f"Detection ROC (label = {PRIMARY})"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "roc.png", dpi=140); plt.close()

    # ---- Fig 2: detection rate vs FPR (event level) ----
    def event_detrate(alarms):
        ne = nd = 0
        for sc in OrderedDict.fromkeys(scenes.tolist()):
            idx = np.where(scenes == sc)[0]; lab = y[idx]; alm = alarms[idx]
            prev = 0
            for i, v in enumerate(lab):
                if v == 1 and prev == 0:
                    ne += 1
                    if alm[: i + 1].any(): nd += 1
                prev = v
        return nd / ne if ne else 0.0

    grid = np.linspace(0.01, 0.4, 30)
    plt.figure(figsize=(5, 4))
    for det in DETS:
        fpr, tpr, thr = roc_curve(y, stats[det])
        ys = []
        for tf in grid:
            ok = np.where(fpr <= tf)[0]
            t = thr[ok[-1]] if ok.size else thr[0]
            ys.append(event_detrate(stats[det] >= t))
        plt.plot(grid, ys, color=COLORS[det], label=NAMES[det])
    plt.xlabel("False-alarm rate (FPR)"); plt.ylabel("Event detection rate")
    plt.title("Event detection vs false-alarm (matched FPR)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGS / "detrate_vs_fpr.png", dpi=140); plt.close()

    # ---- Fig 3: event-triggered average signal ----
    W = 8
    prof = defaultdict(list)
    for sc in OrderedDict.fromkeys(scenes.tolist()):
        idx = np.where(scenes == sc)[0]; lab = y[idx]; zz = z[idx]
        prev = 0
        for i, v in enumerate(lab):
            if v == 1 and prev == 0:
                for d in range(-W, W + 1):
                    j = i + d
                    if 0 <= j < len(zz):
                        prof[d].append(zz[j])
            prev = v
    offs = sorted(prof)
    means = [np.mean(prof[d]) for d in offs]
    sems = [np.std(prof[d]) / np.sqrt(len(prof[d])) for d in offs]
    secs = [d * 0.5 for d in offs]
    plt.figure(figsize=(5.5, 4))
    plt.axvline(0, color="k", ls="--", lw=0.8, label="failure onset")
    plt.errorbar(secs, means, yerr=sems, color="#8000a0", capsize=2)
    plt.xlabel("time relative to failure onset (s)")
    plt.ylabel("standardized disagreement signal (z)")
    plt.title("Signal is concurrent, not a precursor")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGS / "event_triggered.png", dpi=140); plt.close()

    # ---- Fig 4: example scene timeline (pick a scene with a multi-frame failure) ----
    best = None
    for sc in OrderedDict.fromkeys(scenes.tolist()):
        idx = np.where(scenes == sc)[0]
        if y[idx].sum() >= 3 and len(idx) >= 15:
            best = sc; break
    idx = np.where(scenes == best)[0]
    t = np.arange(len(idx)) * 0.5
    plt.figure(figsize=(7, 4))
    plt.plot(t, z[idx], color="gray", label="signal z", lw=1)
    plt.plot(t, stats["cusum"][idx], color=COLORS["cusum"], label="CUSUM stat", lw=1)
    plt.plot(t, stats["ewma"][idx], color=COLORS["ewma"], label="EWMA stat", lw=1)
    fail_t = t[y[idx] == 1]
    for ft in fail_t:
        plt.axvspan(ft - 0.25, ft + 0.25, color="red", alpha=0.15)
    plt.scatter([], [], color="red", alpha=0.3, marker="s", label="failure frame")
    plt.xlabel("time in scene (s)"); plt.ylabel("statistic")
    plt.title(f"Example scene {best}")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGS / "example_scene.png", dpi=140); plt.close()

    # ---- summary table ----
    summary = json.loads((OUT / "eval_summary.json").read_text())
    lines = ["# Detector comparison — summary\n",
             f"label = `{PRIMARY}`; 1 frame = 0.5 s; nuScenes val, 150 scenes / 3908 frames\n",
             "## AUROC (frame-level)\n",
             "| label | single-frame | CUSUM | EWMA |", "|---|---|---|---|"]
    for lab, a in summary["auroc"].items():
        lines.append(f"| {lab} | {a['threshold']:.3f} | {a['cusum']:.3f} | {a['ewma']:.3f} |")
    lines += ["\n## Event detection & lead time @ matched FPR\n",
              "| target FPR | detector | det. rate | median lead (s) |", "|---|---|---|---|"]
    for key, ops in summary["operating_points"].items():
        for det in DETS:
            e = ops[det]
            lines.append(f"| {key} | {NAMES[det]} | {e['detection_rate']:.3f} | "
                         f"{e['median_lead_frames']*0.5:.1f} |")
    (OUT / "summary_table.md").write_text("\n".join(lines) + "\n")

    print("wrote figs:", sorted(p.name for p in FIGS.glob("*.png")))
    print("wrote", OUT / "summary_table.md")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Curiosity detector on the DriveLM-nuScenes VLM-ensemble set (strong disagreement).

Predict failure = mean-ADE top10%. Features = disagreement/dispersion across the 14 VLM
trajectories ONLY (NO ADE/L2 -> anti-circular). Logistic regression, 5-fold scene CV.
Target AUROC >= 0.7 (single disagreement feature already ~0.80).

Out: results/curiosity/report/detector_drivelm.csv + roc_drivelm.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[2]
SRC = PROJ / "results/unified/lightemma_test_150.jsonl"
OUT = PROJ / "results/curiosity/report"
MODELS = ["claude-3.7-sonnet", "claude-4.0-sonnet", "deepseek-vl2-16b", "deepseek-vl2-28b",
          "gemini-2.5-flash", "gemini-2.5-pro", "gpt-4.1", "gpt-4o", "gpt-5",
          "llama-3.2-11b", "llama-3.2-90b", "qwen-2.5-7b", "qwen-2.5-7b-local"]
N_WP = 6

FEATURES = ["disp_mean", "disp_1s", "disp_3s", "endpoint_lat_std",
            "pathlen_std", "heading_std", "max_pair_endpoint", "n_models"]


def valid(p):
    return isinstance(p, dict) and p.get("parse_ok") and isinstance(p.get("trajectory"), list) and len(p["trajectory"]) == N_WP


def ade(a, b):
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC)]
    feats, scenes, losses = [], [], []
    for d in rows:
        gt = np.asarray(d["gt_trajectory"], dtype=float)[:, :2]
        T, ades = [], []
        for m in MODELS:
            p = d["predictions"].get(m)
            if valid(p):
                tr = np.asarray(p["trajectory"], dtype=float)
                T.append(tr); ades.append(ade(gt, tr))
        if len(T) < 2:
            continue
        T = np.stack(T)                       # (M,6,2)
        mean_t = T.mean(0)
        per_wp_var = np.mean(np.sum((T - mean_t) ** 2, axis=2), axis=0)   # (6,)
        # path lengths per model
        seglen = np.linalg.norm(np.diff(T, axis=1), axis=2).sum(axis=1)   # (M,)
        # final heading per model
        last_vec = T[:, -1, :] - T[:, -2, :]
        headings = np.arctan2(last_vec[:, 1], last_vec[:, 0])
        # max pairwise endpoint distance
        ep = T[:, -1, :]
        maxpair = float(np.max([np.linalg.norm(ep[i] - ep[j]) for i in range(len(ep)) for j in range(i + 1, len(ep))]))
        feats.append([
            float(per_wp_var.mean()) ** 0.5,        # disp_mean
            float(per_wp_var[1]) ** 0.5,            # disp_1s (wp index1 ~1s)
            float(per_wp_var[-1]) ** 0.5,           # disp_3s (endpoint)
            float(np.std(ep[:, 1])),                # endpoint lateral (y) std
            float(np.std(seglen)),                  # path-length disagreement
            float(np.std(headings)),                # heading disagreement
            maxpair,                                # max pairwise endpoint dist
            float(len(T)),                          # n_models
        ])
        scenes.append(d["scene_name"])
        losses.append(float(np.mean(ades)))         # mean ADE = label source ONLY
    X = np.array(feats); scenes = np.array(scenes); losses = np.array(losses)
    y = (losses >= np.percentile(losses, 90)).astype(int)
    print(f"n={len(X)} frames, {len(set(scenes))} scenes | positives={y.sum()} ({100*y.mean():.1f}%)\n")

    def cv(cols):
        idx = [FEATURES.index(c) for c in cols]
        Xs = X[:, idx]
        ar, ap = [], []
        for tr, te in GroupKFold(5).split(Xs, y, scenes):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xs[tr]), y[tr])
            p = clf.predict_proba(sc.transform(Xs[te]))[:, 1]
            ar.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p))
        return np.mean(ar), np.std(ar), np.mean(ap), np.std(ap)

    table = []
    for name, cols in [("disagreement single (disp_mean)", ["disp_mean"]),
                       ("all dispersion features (learned)", FEATURES)]:
        ar, ars, ap, aps = cv(cols)
        table.append([name, "+".join(cols), f"{ar:.3f}", f"{ars:.3f}", f"{ap:.3f}", f"{aps:.3f}"])
        print(f"{name:38s} AUROC {ar:.3f}±{ars:.3f}  AUPRC {ap:.3f}±{aps:.3f}")
    print(f"(baseline AUPRC = {y.mean():.3f})")

    # importance (full fit)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    imp = sorted(zip(FEATURES, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("\nfeature importance (standardized LR coef):")
    for c, w in imp:
        print(f"  {c:18s} {w:+.3f}")

    # save csv
    with open(OUT / "detector_drivelm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["detector", "features", "AUROC", "AUROC_std", "AUPRC", "AUPRC_std"])
        w.writerows(table)
        w.writerow([])
        w.writerow(["feature_importance(LR coef)", "", "", "", "", ""])
        for c, ww in imp:
            w.writerow([c, f"{ww:+.3f}", "", "", "", ""])

    # ROC (OOF, fused)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, scenes):
        s = StandardScaler().fit(X[tr]); cl = LogisticRegression(max_iter=1000, class_weight="balanced").fit(s.transform(X[tr]), y[tr])
        oof[te] = cl.predict_proba(s.transform(X[te]))[:, 1]
    fpr, tpr, _ = roc_curve(y, oof)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#d62728", label=f"learned detector (AUROC={roc_auc_score(y, oof):.3f})")
    # single-feature reference
    fpr2, tpr2, _ = roc_curve(y, X[:, FEATURES.index("disp_mean")])
    plt.plot(fpr2, tpr2, color="#2ca02c", lw=1, label=f"disagreement only ({roc_auc_score(y, X[:, FEATURES.index('disp_mean')]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("DriveLM-nuScenes curiosity detector")
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(OUT / "roc_drivelm.png", dpi=140); plt.close()
    print("\nwrote detector_drivelm.csv + roc_drivelm.png")


if __name__ == "__main__":
    main()

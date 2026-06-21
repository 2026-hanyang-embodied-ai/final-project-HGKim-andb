#!/usr/bin/env python3
"""Curiosity Detection model — predict "does the model fail in this scene?" from signals.

Set: SparseDrive on nuScenes val (per-scene values already extracted, no re-inference).
  features (NO error/L2 — anti-circular; signals only):
    disagreement  : sig_mode_var, sig_mode_std  (spatial spread of the 6 candidate trajs)
    native uncert.: sig_entropy, sig_margin      (score-distribution confidence)
  label: fail_top10 = 1 if SparseDrive ADE in global worst 10%, else 0
Model: standardize + logistic regression, 5-fold GROUP CV by scene (no leakage).
Report: AUROC + AUPRC for [disagreement] vs [native] vs [fused] + LR coefficients.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CSV = Path(__file__).resolve().parents[2] / "results/curiosity/sd_signals.csv"
DISAGREE = ["sig_mode_var", "sig_mode_std"]
NATIVE = ["sig_entropy", "sig_margin"]
FUSED = DISAGREE + NATIVE
LABEL = "fail_top10"


def main():
    rows = list(csv.DictReader(open(CSV)))
    groups = np.array([r["scene"] for r in rows])
    y = np.array([int(r[LABEL]) for r in rows])
    feat = {c: np.array([float(r[c]) for r in rows]) for c in FUSED}
    print(f"n={len(rows)} frames, {len(set(groups))} scenes | positives={y.sum()} ({100*y.mean():.1f}%)\n")

    def evaluate(cols):
        X = np.column_stack([feat[c] for c in cols])
        gkf = GroupKFold(n_splits=5)
        aurocs, auprcs = [], []
        for tr, te in gkf.split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            aurocs.append(roc_auc_score(y[te], p))
            auprcs.append(average_precision_score(y[te], p))
        return np.mean(aurocs), np.std(aurocs), np.mean(auprcs), np.std(auprcs)

    print(f"{'detector':24s} {'AUROC':>14s} {'AUPRC':>14s}")
    print("-" * 54)
    for name, cols in [("disagreement only", DISAGREE),
                       ("native only", NATIVE),
                       ("fused (learned)", FUSED)]:
        ar, ars, ap, aps = evaluate(cols)
        print(f"{name:24s}  {ar:.3f}±{ars:.3f}   {ap:.3f}±{aps:.3f}")
    print(f"\n(baseline AUPRC = positive rate = {y.mean():.3f}; AUROC chance = 0.5)")

    # feature importance: standardized LR coefficients on full data
    X = np.column_stack([feat[c] for c in FUSED])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    print("\nfused LR standardized coefficients (which signal drives it):")
    for c, w in sorted(zip(FUSED, clf.coef_[0]), key=lambda kv: -abs(kv[1])):
        print(f"  {c:16s} {w:+.3f}")


if __name__ == "__main__":
    main()

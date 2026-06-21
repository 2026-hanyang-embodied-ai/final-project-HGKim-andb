#!/usr/bin/env python3
"""Stage 3 — Compare the three detectors.

Two views, both required to test the hypothesis fairly:

  (1) Frame-level ROC / AUROC: sweep each detector's threshold, measure
      TPR vs FPR against the failure labels. Detector-agnostic, threshold-free.

  (2) Event-level detection + lead time at MATCHED false-alarm rate: failures
      form events (consecutive failure frames); onset = first frame. For a
      threshold giving a target frame-level FPR, measure
        - detection rate: fraction of events with an alarm at/before onset
        - lead time: frames between the first pre-onset alarm and onset
      Matching FPR is essential: an accumulator that simply alarms more often
      would otherwise look better for free.

Outputs:
  results/curiosity/eval_summary.json   (AUROC + per-operating-point table)
  results/curiosity/roc_points.json     (ROC curves for plotting)
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import detectors as D

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "results/curiosity"

# CLI: eval_detectors.py [csv_path] [signal_col] [tag] [primary_label]
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else str(D.SIG_CSV)
SIGNAL_COL = sys.argv[2] if len(sys.argv) > 2 else "signal_std"
TAG = sys.argv[3] if len(sys.argv) > 3 else "vlm"
PRIMARY_LABEL = sys.argv[4] if len(sys.argv) > 4 else "fail_top10"
LABELS = ["fail_top05", "fail_top10", "fail_top15", "fail_top20", "fail_within_scene"]
TARGET_FPRS = [0.05, 0.10, 0.20]
DETECTORS = ["threshold", "cusum", "ewma"]


def scene_runs(labels: np.ndarray) -> list[int]:
    """Onset indices (start of each maximal run of label==1)."""
    onsets = []
    prev = 0
    for i, v in enumerate(labels):
        if v == 1 and prev == 0:
            onsets.append(i)
        prev = v
    return onsets


def event_metrics(scenes: np.ndarray, alarms: np.ndarray, labels: np.ndarray):
    """Detection rate + lead-time stats over failure events, with the alarm mask
    already thresholded. Alarm at/before onset (within the scene) = detected."""
    n_events = n_detected = 0
    leads = []
    for sc in OrderedDict.fromkeys(scenes.tolist()):
        idx = np.where(scenes == sc)[0]
        lab = labels[idx]
        alm = alarms[idx]
        for o in scene_runs(lab):
            n_events += 1
            pre = np.where(alm[: o + 1])[0]   # alarm frames at/before onset
            if pre.size:
                n_detected += 1
                leads.append(o - int(pre[0]))  # earliest pre-onset alarm
    det_rate = n_detected / n_events if n_events else 0.0
    leads = np.array(leads, dtype=float)
    return {
        "n_events": n_events,
        "detection_rate": det_rate,
        "median_lead_frames": float(np.median(leads)) if leads.size else 0.0,
        "mean_lead_frames": float(np.mean(leads)) if leads.size else 0.0,
    }


def threshold_for_fpr(stat: np.ndarray, label: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose frame-level FPR <= target (closest from below)."""
    fpr, tpr, thr = roc_curve(label, stat)
    ok = np.where(fpr <= target_fpr)[0]
    j = ok[-1] if ok.size else 0     # largest fpr not exceeding target
    return float(thr[j])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes, signal, rows = D.load_signals(CSV_PATH, SIGNAL_COL)
    stats = D.compute_statistics(scenes, signal)
    labels = {k: np.array([int(r[k]) for r in rows]) for k in LABELS}

    summary = {"primary_label": PRIMARY_LABEL, "auroc": {}, "operating_points": {}}
    roc_out = {}

    # --- AUROC for every detector x label
    for lab_name in LABELS:
        y = labels[lab_name]
        summary["auroc"][lab_name] = {
            det: float(roc_auc_score(y, stats[det])) for det in DETECTORS
        }

    # --- ROC curves (primary label) for plotting
    y = labels[PRIMARY_LABEL]
    for det in DETECTORS:
        fpr, tpr, _ = roc_curve(y, stats[det])
        roc_out[det] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # --- matched-FAR operating points (primary label)
    for tf in TARGET_FPRS:
        key = f"fpr={tf:.2f}"
        summary["operating_points"][key] = {}
        for det in DETECTORS:
            thr = threshold_for_fpr(stats[det], y, tf)
            alarms = (stats[det] >= thr).astype(int)
            actual_fpr = float(np.mean(alarms[y == 0]))
            em = event_metrics(scenes, alarms, y)
            em["threshold"] = thr
            em["actual_fpr"] = actual_fpr
            summary["operating_points"][key][det] = em

    (OUT_DIR / f"eval_summary_{TAG}.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / f"roc_points_{TAG}.json").write_text(json.dumps(roc_out))

    # --- console report
    print(f"[tag={TAG} signal={SIGNAL_COL}]")
    print(f"=== AUROC (label={PRIMARY_LABEL}) ===")
    for det in DETECTORS:
        print(f"  {det:10s} {summary['auroc'][PRIMARY_LABEL][det]:.3f}")
    print("\n=== AUROC across labels ===")
    print(f"  {'label':18s} " + " ".join(f"{d:>10s}" for d in DETECTORS))
    for lab_name in LABELS:
        a = summary["auroc"][lab_name]
        print(f"  {lab_name:18s} " + " ".join(f"{a[d]:10.3f}" for d in DETECTORS))
    print(f"\n=== Event detection & lead time @ matched FPR (label={PRIMARY_LABEL}) ===")
    print("  (lead in frames; 1 frame = 0.5 s)")
    for tf in TARGET_FPRS:
        key = f"fpr={tf:.2f}"
        print(f"  --- target {key} ---")
        print(f"    {'detector':10s} {'actFPR':>7s} {'detRate':>8s} "
              f"{'medLead':>8s} {'meanLead':>9s}  (n_events={summary['operating_points'][key]['threshold' if False else 'cusum']['n_events']})")
        for det in DETECTORS:
            e = summary["operating_points"][key][det]
            print(f"    {det:10s} {e['actual_fpr']:7.3f} {e['detection_rate']:8.3f} "
                  f"{e['median_lead_frames']:8.1f} {e['mean_lead_frames']:9.1f}")
    print(f"\nwrote {OUT_DIR}/eval_summary_{TAG}.json and roc_points_{TAG}.json")


if __name__ == "__main__":
    main()

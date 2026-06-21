#!/usr/bin/env python3
"""Stage 1 — Extract per-frame OOD signal + failure labels from the LightEMMA
nuScenes-val outputs (no re-inference).

Signal s_t (OOD / scene ambiguity): how much the ensemble of VLM trajectories
SPREADS at frame t. Computed "all at once" as positional variance across models
(NOT pairwise). High spread = models disagree = ambiguous/OOD-ish moment.

Failure label y_t (loss): how wrong the predictions are on average at frame t,
measured as the mean over models of the 3 s ADE (full-horizon average displacement
error vs GT). Labelled as failure when in the global top-{5,10,15,20}%, plus a
within-scene relative-spike variant.

Signal (spread) and label (mean error) are different quantities -> non-circular.

Output: results/curiosity/signals.csv  (one row per frame; everything downstream
reads only this file).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "results/unified/lightemma_test_150.jsonl"
OUT_DIR = PROJECT / "results/curiosity"
OUT_CSV = OUT_DIR / "signals.csv"

# Fixed high-coverage model set (~100% parse_ok). Excludes qwen-2.5-72b (52% parse)
# and qwen2.5-32b-awq (only 26 frames) so the ensemble is comparable across frames.
MODELS = [
    "claude-3.7-sonnet", "claude-4.0-sonnet",
    "deepseek-vl2-16b", "deepseek-vl2-28b",
    "gemini-2.5-flash", "gemini-2.5-pro",
    "gpt-4.1", "gpt-4o", "gpt-5",
    "llama-3.2-11b", "llama-3.2-90b",
    "qwen-2.5-7b", "qwen-2.5-7b-local",
]
N_WP = 6  # 6 waypoints @ 0.5 s = 3 s horizon

GLOBAL_PCTS = [5, 10, 15, 20]   # global top-% failure labels
WITHIN_SCENE_TOP = 0.20          # top 20% of frames within a scene -> relative spike


def valid_traj(pred: dict) -> bool:
    if not isinstance(pred, dict) or not pred.get("parse_ok"):
        return False
    tr = pred.get("trajectory")
    return isinstance(tr, list) and len(tr) == N_WP


def ade(a: np.ndarray, b: np.ndarray) -> float:
    """Average displacement error over all waypoints (a,b: (N_WP,2))."""
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC)]

    records = []
    for d in rows:
        gt = np.asarray(d["gt_trajectory"], dtype=float)  # (6,2)
        preds = d["predictions"]
        trajs, ades = [], []
        for m in MODELS:
            p = preds.get(m)
            if not valid_traj(p):
                continue
            tr = np.asarray(p["trajectory"], dtype=float)
            trajs.append(tr)
            ades.append(ade(gt, tr))           # recomputed, not trusting stored field
        n = len(trajs)
        if n < 2:
            continue  # need >=2 models to define spread
        T = np.stack(trajs)                    # (n,6,2)

        # --- OOD signal: positional variance across models, averaged over waypoints
        mean_traj = T.mean(axis=0)             # (6,2)
        # per-waypoint spread = E_models[ ||p - mean||^2 ] = var_x + var_y
        per_wp_var = np.mean(np.sum((T - mean_traj) ** 2, axis=2), axis=0)  # (6,)
        signal_var = float(per_wp_var.mean())          # mean positional variance
        signal_std = float(np.sqrt(signal_var))        # same on a length scale (m)
        signal_endpoint_var = float(per_wp_var[-1])    # spread at the 3 s endpoint

        # --- loss / failure source
        loss_mean = float(np.mean(ades))               # mean model error (m)
        loss_ensmean = ade(gt, mean_traj)              # error of the averaged path

        records.append({
            "scene": d["scene_name"],
            "frame_index": int(d["frame_index"]),
            "n_models": n,
            "signal_var": signal_var,
            "signal_std": signal_std,
            "signal_endpoint_var": signal_endpoint_var,
            "loss_mean": loss_mean,
            "loss_ensmean": loss_ensmean,
        })

    # --- global top-% labels on loss_mean
    losses = np.array([r["loss_mean"] for r in records])
    thresholds = {p: float(np.percentile(losses, 100 - p)) for p in GLOBAL_PCTS}
    for r in records:
        for p in GLOBAL_PCTS:
            r[f"fail_top{p:02d}"] = int(r["loss_mean"] >= thresholds[p])

    # --- within-scene relative spike: top-20% of loss within each scene
    by_scene: dict[str, list] = {}
    for r in records:
        by_scene.setdefault(r["scene"], []).append(r)
    for scene_rows in by_scene.values():
        sl = np.array([r["loss_mean"] for r in scene_rows])
        # percentile rank within scene (0..1)
        order = sl.argsort().argsort()
        rank = order / max(len(sl) - 1, 1)
        cut = 1.0 - WITHIN_SCENE_TOP
        for r, rk in zip(scene_rows, rank):
            r["within_scene_pct"] = float(rk)
            r["fail_within_scene"] = int(rk >= cut)

    # --- write CSV (sorted by scene then frame for readable timelines)
    records.sort(key=lambda r: (r["scene"], r["frame_index"]))
    fields = list(records[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    # --- console summary
    print(f"wrote {OUT_CSV}  ({len(records)} frames, {len(by_scene)} scenes)")
    print(f"models used: {len(MODELS)}  | n_models/frame: "
          f"min {min(r['n_models'] for r in records)} "
          f"max {max(r['n_models'] for r in records)}")
    print("global loss top-% thresholds (m):",
          {p: round(t, 3) for p, t in thresholds.items()})
    for p in GLOBAL_PCTS:
        k = sum(r[f"fail_top{p:02d}"] for r in records)
        print(f"  fail_top{p:02d}: {k} positives ({100*k/len(records):.1f}%)")
    kws = sum(r["fail_within_scene"] for r in records)
    print(f"  fail_within_scene: {kws} positives ({100*kws/len(records):.1f}%)")
    # quick non-circularity sanity: correlation signal vs loss (should be >0 but <1)
    s = np.array([r["signal_std"] for r in records])
    corr = float(np.corrcoef(s, losses)[0, 1])
    print(f"corr(signal_std, loss_mean) = {corr:.3f}  (expect modest +, not ~1)")


if __name__ == "__main__":
    main()

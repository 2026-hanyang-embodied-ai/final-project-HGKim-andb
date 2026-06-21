#!/usr/bin/env python3
"""Stage 1 (native model) — Extract SparseDrive native uncertainty + failure labels
on nuScenes val, by POST-PROCESSING the official test results pkl (no model surgery).

SparseDrive planning outputs, per frame:
  planning_score [3 cmd, 6 mode]   sigmoid scores over modes
  planning       [3, 6, 6, 2]      multimodal trajectories (absolute, cumsum'd)
  final_planning [6, 2]            the selected trajectory

Native uncertainty signal (from the selected command's 6 modes):
  - sig_mode_var : positional variance across the 6 mode trajectories  (analogous
                   to the VLM-ensemble disagreement signal -> directly comparable)
  - sig_entropy  : entropy of softmax over the 6 mode scores
  - sig_margin   : 1 - (top1 - top2) of softmax  (higher = less decisive)
Failure label (loss): L2(final_planning, GT) 3 s ADE, masked. Signal (mode spread /
score shape) and label (GT error) are different quantities -> non-circular.

GT note: infos['gt_ego_fut_trajs'] are PER-STEP DELTAS -> cumsum to absolute, matching
planning_eval. final_planning is already absolute.

Out: results/curiosity/sd_signals.csv
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

import sys

PROJECT = Path(__file__).resolve().parents[2]
SD = PROJECT / "models/sparse_drive"
# CLI: extract_sd_uncertainty.py [res_pkl] [infos_pkl] [out_csv]
# Defaults = SparseDrive. DiffusionDrive shares the exact output schema.
RES_PKL = Path(sys.argv[1]) if len(sys.argv) > 1 else SD / "work_dirs/sd_curiosity/sd_val_results.pkl"
INFOS = Path(sys.argv[2]) if len(sys.argv) > 2 else SD / "data/infos/nuscenes_infos_val.pkl"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else PROJECT / "results/curiosity/sd_signals.csv"

N_WP, N_MODE = 6, 6
GLOBAL_PCTS = [5, 10, 15, 20]
WITHIN_SCENE_TOP = 0.20


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def get_imgbbox(r):
    return r["img_bbox"] if isinstance(r, dict) and "img_bbox" in r else r


def main():
    results = pickle.load(open(RES_PKL, "rb"))
    infos_raw = pickle.load(open(INFOS, "rb"))
    infos = infos_raw["infos"] if isinstance(infos_raw, dict) else infos_raw
    # CRITICAL (CLAUDE.md step 3): the dataset's load_annotations sorts infos by
    # timestamp, so results[i] aligns with timestamp-sorted infos, NOT raw order.
    infos = sorted(infos, key=lambda e: e["timestamp"])
    assert len(results) == len(infos), f"len mismatch {len(results)} vs {len(infos)}"

    recs = []
    l2_1s = l2_2s = l2_3s = 0.0
    nval = 0
    for r, info in zip(results, infos):
        rb = get_imgbbox(r)
        pscore = np.asarray(rb["planning_score"])          # [3,6]
        planning = np.asarray(rb["planning"])              # [3,6,6,2]
        final = np.asarray(rb["final_planning"])           # [6,2]
        cmd = int(np.asarray(info["gt_ego_fut_cmd"]).argmax())

        gt = np.asarray(info["gt_ego_fut_trajs"], dtype=float).reshape(N_WP, 2).cumsum(0)
        mask = np.asarray(info["gt_ego_fut_masks"]).reshape(N_WP).astype(bool)

        # --- loss (label source): L2 of selected traj vs GT, masked
        step_l2 = np.linalg.norm(final[:N_WP] - gt, axis=1)
        m = mask if mask.any() else np.ones(N_WP, bool)
        loss_ade = float(step_l2[m].mean())
        # horizon metrics for validation against run.log (2 steps = 1 s)
        for acc, k in ((1, 2), (2, 4), (3, 6)):
            mm = m[:k]
            if mm.any():
                val = step_l2[:k][mm].mean()
                if acc == 1: l2_1s += val
                elif acc == 2: l2_2s += val
                else: l2_3s += val
        nval += 1

        # --- native uncertainty from the selected command's 6 modes
        sc = pscore[cmd]                                   # [6]
        modes = planning[cmd]                              # [6,6,2] absolute
        mean_mode = modes.mean(axis=0)                     # [6,2]
        per_wp_var = np.mean(np.sum((modes - mean_mode) ** 2, axis=2), axis=0)  # [6]
        sig_mode_var = float(per_wp_var.mean())
        sig_mode_std = float(np.sqrt(sig_mode_var))
        p = softmax(sc)
        sig_entropy = float(-(p * np.log(p + 1e-12)).sum())
        srt = np.sort(p)[::-1]
        sig_margin = float(1.0 - (srt[0] - srt[1]))        # higher = less decisive

        recs.append({
            "scene": info["scene_token"],
            "timestamp": int(info["timestamp"]),
            "token": info["token"],
            "cmd": cmd,
            "sig_mode_var": sig_mode_var,
            "sig_mode_std": sig_mode_std,
            "sig_entropy": sig_entropy,
            "sig_margin": sig_margin,
            "loss_l2": loss_ade,
        })

    # frame_index within scene by timestamp
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)
    for rows in by_scene.values():
        rows.sort(key=lambda r: r["timestamp"])
        for fi, r in enumerate(rows):
            r["frame_index"] = fi

    # labels
    losses = np.array([r["loss_l2"] for r in recs])
    thr = {p: float(np.percentile(losses, 100 - p)) for p in GLOBAL_PCTS}
    for r in recs:
        for p in GLOBAL_PCTS:
            r[f"fail_top{p:02d}"] = int(r["loss_l2"] >= thr[p])
    for rows in by_scene.values():
        sl = np.array([r["loss_l2"] for r in rows])
        rank = sl.argsort().argsort() / max(len(sl) - 1, 1)
        for r, rk in zip(rows, rank):
            r["within_scene_pct"] = float(rk)
            r["fail_within_scene"] = int(rk >= 1 - WITHIN_SCENE_TOP)

    recs.sort(key=lambda r: (r["scene"], r["frame_index"]))
    fields = ["scene", "frame_index", "timestamp", "token", "cmd",
              "sig_mode_var", "sig_mode_std", "sig_entropy", "sig_margin",
              "loss_l2", "within_scene_pct",
              "fail_top05", "fail_top10", "fail_top15", "fail_top20", "fail_within_scene"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(recs)

    print(f"wrote {OUT}  ({len(recs)} frames, {len(by_scene)} scenes)")
    print(f"VALIDATION — recomputed planning L2 (match run.log): "
          f"1s {l2_1s/nval:.3f}  2s {l2_2s/nval:.3f}  3s {l2_3s/nval:.3f}  "
          f"avg {(l2_1s+l2_2s+l2_3s)/3/nval:.3f}")
    for sig in ["sig_mode_std", "sig_entropy", "sig_margin"]:
        s = np.array([r[sig] for r in recs])
        c = np.corrcoef(s, losses)[0, 1]
        print(f"corr({sig}, loss) = {c:.3f}")


if __name__ == "__main__":
    main()

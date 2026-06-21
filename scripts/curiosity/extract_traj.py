#!/usr/bin/env python3
"""Extract per-frame final ego trajectory (token + 6x2 absolute, ego frame) from a
driving-model results pkl, for cross-model disagreement on nuScenes-val.

Usage: extract_traj.py <pkl> <infos.pkl> <out.csv> [--vad]
  SD/DD : img_bbox.final_planning (already absolute)
  VAD   : pts_bbox.ego_fut_preds[cmd], delta -> cumsum (absolute)
Mapping: results[i] <-> timestamp-sorted infos[i] (validated by recomputed ADE scale).
"""
import sys, pickle, csv
import numpy as np

pkl, infos_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
is_vad = "--vad" in sys.argv
res = pickle.load(open(pkl, "rb"))
infos = pickle.load(open(infos_p, "rb"))
infos = infos["infos"] if isinstance(infos, dict) else infos
infos = sorted(infos, key=lambda e: e["timestamp"])   # dataset order (validated below)
assert len(res) == len(infos), f"len {len(res)} vs {len(infos)}"

rows, ades = [], []
for r, info in zip(res, infos):
    tok = info["token"]
    if is_vad:
        pb = r["pts_bbox"]; ef = np.asarray(pb["ego_fut_preds"])
        cmd = int(np.asarray(info["gt_ego_fut_cmd"]).flatten()[:3].argmax())
        traj = ef[cmd][:6].cumsum(0)
    else:
        traj = np.asarray(r["img_bbox"]["final_planning"])[:6]
    rows.append([tok] + traj.reshape(-1).tolist())
    # validation: ADE vs GT (scale check)
    if "gt_ego_fut_trajs" in info:
        gt = np.asarray(info["gt_ego_fut_trajs"]).reshape(6, 2).cumsum(0)
        m = np.asarray(info.get("gt_ego_fut_masks", np.ones(6))).reshape(6).astype(bool)
        d = np.linalg.norm(traj - gt, axis=1)
        ades.append(d[m].mean() if m.any() else d.mean())

mean_ade = float(np.mean(ades)) if ades else float("nan")
hdr = ["token"] + [f"t{i}{ax}" for i in range(6) for ax in "xy"]
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
status = "OK" if (mean_ade != mean_ade or mean_ade < 2.5) else "SUSPECT(mismap?)"
print(f"wrote {out}: {len(rows)} frames | recomputed mean ADE = {mean_ade:.3f} [{status}]")

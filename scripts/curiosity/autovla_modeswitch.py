#!/usr/bin/env python3
"""AutoVLA fast/slow mode-switching analysis (Waymo v3, n=199 / 31 scenes).

AutoVLA is itself a fast-slow model (fast = no/short CoT, slow = full CoT). We have its
FORCED fast-mode and slow-mode runs on the same frames (its adaptive per-frame choice is
NOT logged). Questions:
  (1) Value of escalation: oracle per-frame fast/slow pick vs always-fast.
  (2) Can the cross-model DISAGREEMENT signal time the escalation? i.e. predict
      (a) "slow helps" (slow ADE < fast ADE - 0.5) and (b) "fast fails" (top-20% fast ADE).
Disagreement is computed across the OTHER (non-AutoVLA) models -> no self-reference.
Small n -> illustrative, not definitive.

Out: results/curiosity/report/autovla_modeswitch.csv + autovla_modeswitch.md
"""
import json, csv
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJ = Path(__file__).resolve().parents[2]
V3 = PROJ / "results/unified/all_frames_v3.jsonl"
OUT = PROJ / "results/curiosity/report"
OTHER = ["rap", "alpamayo", "alpamayo_v15_nav", "alpamayo_v15_nav_v2",
         "dd_waymo", "light_emma", "light_emma_local"]   # non-AutoVLA models for disagreement
NWP = 6

rows = [json.loads(l) for l in open(V3)]
def ade(r, m):
    p = r["predictions"].get(m); return p.get("ADE") if p and p.get("ADE") is not None else None

fast, slow, disag, scenes = [], [], [], []
for r in rows:
    f, s = ade(r, "autovla_waymo_fast"), ade(r, "autovla_waymo_slow")
    if f is None or s is None:
        continue
    T = []
    for m in OTHER:
        p = r["predictions"].get(m)
        if p and p.get("trajectory") and len(p["trajectory"]) >= NWP:
            T.append(np.asarray(p["trajectory"])[:NWP, :2])
    if len(T) < 2:
        continue
    T = np.stack(T)
    pw = np.mean(np.sum((T - T.mean(0))**2, axis=2), axis=0)
    fast.append(f); slow.append(s); disag.append(float(pw.mean())**0.5); scenes.append(r["scene_id"])

fast, slow, disag = np.array(fast), np.array(slow), np.array(disag)
n = len(fast)
oracle = np.minimum(fast, slow)
slow_helps = (slow < fast - 0.5).astype(int)
fast_fail = (fast >= np.percentile(fast, 80)).astype(int)   # top-20% (n small)

def auc(lab, sig):
    return roc_auc_score(lab, sig), average_precision_score(lab, sig)

a_help, p_help = auc(slow_helps, disag)
a_fail, p_fail = auc(fast_fail, disag)

table = [
    ["always-fast ADE", f"{fast.mean():.2f}", ""],
    ["always-slow ADE", f"{slow.mean():.2f}", ""],
    ["oracle per-frame mode ADE", f"{oracle.mean():.2f}", f"-{100*(1-oracle.mean()/fast.mean()):.0f}% vs fast"],
    ["disagreement -> 'slow helps' (AUROC/AUPRC)", f"{a_help:.3f}", f"AUPRC {p_help:.3f} (base {slow_helps.mean():.2f})"],
    ["disagreement -> 'fast fails' (AUROC/AUPRC)", f"{a_fail:.3f}", f"AUPRC {p_fail:.3f} (base {fast_fail.mean():.2f})"],
]
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "autovla_modeswitch.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv); w.writerow(["metric", "value", "note"]); w.writerows(table)

md = [f"# AutoVLA fast/slow mode switching (Waymo v3, n={n} / {len(set(scenes))} scenes)\n",
      "AutoVLA's adaptive per-frame choice is NOT logged; we use its forced fast/slow runs.",
      "Disagreement = spread of non-AutoVLA models' trajectories (no self-reference). Small n -> illustrative.\n",
      "| metric | value | note |", "|---|---|---|"]
md += [f"| {r[0]} | {r[1]} | {r[2]} |" for r in table]
slow_helps_rate = slow_helps.mean()
md += [f"\n**Value of escalation:** an oracle that picks fast-vs-slow per frame cuts ADE from "
       f"{fast.mean():.2f} to {oracle.mean():.2f} ({100*(1-oracle.mean()/fast.mean()):.0f}% lower) — "
       f"so *timing* the slow mode matters (always-slow barely helps: {slow.mean():.2f}).",
       f"**Can disagreement time it?** disagreement predicts 'slow helps' at AUROC {a_help:.3f} and "
       f"'fast fails' at {a_fail:.3f}. (n={n} small/noisy; Waymo; mid-training AutoVLA checkpoint.)",
       "\n**Reading:** consistent with the main study — cross-model disagreement carries the "
       "fast-slow signal — but n is too small here for a firm number; treat as a directional probe."]
(OUT / "autovla_modeswitch.md").write_text("\n".join(md) + "\n")

print(f"n={n}, scenes={len(set(scenes))}")
for r in table: print(f"  {r[0]:42s} {r[1]:>7s}  {r[2]}")
print("wrote autovla_modeswitch.csv + .md")

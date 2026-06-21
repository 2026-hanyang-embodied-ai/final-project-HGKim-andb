#!/usr/bin/env python3
"""Transfer test: does CROSS-MODEL DISAGREEMENT predict the actual driving model's
failure? Same set, same label (SparseDrive ADE top-10%), fused with SD native.

Disagreement is computed across real driving models {SparseDrive, DiffusionDrive, VAD}
on the official nuScenes-val frames (NOT the VLM ensemble — the 14 diverse VLMs are paid
APIs and were not re-run autonomously; logged). LABEL = SparseDrive's own ADE top-10%
(its trajectory error vs GT). ADE never used as a feature (anti-circular). 5-fold scene CV.

Out: results/curiosity/report/transfer_fusion.csv, transfer_RESULTS.md
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[2]
CUR = PROJ / "results/curiosity"
OUT = CUR / "report"; OUT.mkdir(parents=True, exist_ok=True)
log = []
def L(m): print(m); log.append(m)


def load_traj(p):
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["token"]] = np.array([float(r[f"t{i}{ax}"]) for i in range(6) for ax in "xy"]).reshape(6, 2)
    return out


def main():
    sd = load_traj(CUR / "traj/sd_trajs.csv")
    dd = load_traj(CUR / "traj/dd_trajs.csv")
    vad = load_traj(CUR / "traj/vad_trajs.csv")
    sig = {r["token"]: r for r in csv.DictReader(open(CUR / "sd_signals.csv"))}
    toks = [t for t in sig if t in sd and t in dd and t in vad]
    L(f"common frames (SD∩DD∩VAD∩signals): {len(toks)} / SD {len(sd)}")

    DIS, NAT, scenes, y = [], [], [], []
    for t in toks:
        T = np.stack([sd[t], dd[t], vad[t]])           # (3,6,2)
        mt = T.mean(0)
        pw = np.mean(np.sum((T - mt) ** 2, axis=2), axis=0)   # (6,)
        ep = T[:, -1]
        seg = np.linalg.norm(np.diff(T, axis=1), axis=2).sum(1)
        dd_vad = float(np.mean(np.linalg.norm(dd[t] - vad[t], axis=1)))  # leave-SD-out disagreement
        DIS.append([pw.mean() ** .5, pw[1] ** .5, pw[-1] ** .5, float(np.std(ep[:, 0])),
                    float(np.std(seg)), float(max(np.linalg.norm(ep[a] - ep[b])
                    for a in range(3) for b in range(a + 1, 3))), dd_vad])
        NAT.append([float(sig[t]["sig_margin"]), float(sig[t]["sig_entropy"]), float(sig[t]["sig_mode_std"])])
        scenes.append(sig[t]["scene"]); y.append(int(sig[t]["fail_top10"]))
    DIS = np.array(DIS); NAT = np.array(NAT); scenes = np.array(scenes); y = np.array(y)
    DIS_NAMES = ["dis_mean", "dis_1s", "dis_3s", "dis_lat_std", "dis_pathlen", "dis_maxpair", "dis_DDvsVAD"]
    NAT_NAMES = ["sig_margin", "sig_entropy", "sig_mode_std"]
    L(f"label = SparseDrive ADE top10%: positives {y.sum()} ({100*y.mean():.1f}%); "
      f"disagreement among 3 DRIVING models (SD/DD/VAD; SD&DD similar -> low diversity).")

    def cv(X):
        ar, ap = [], []
        for tr, te in GroupKFold(5).split(X, y, scenes):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            ar.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p))
        return float(np.mean(ar)), float(np.std(ar)), float(np.mean(ap)), float(np.std(ap))

    rows = []
    L("\n=== 5-fold scene-CV (predict SparseDrive failure) ===")
    for name, X in [("native-only", NAT), ("disagreement-only", DIS), ("fused", np.hstack([DIS, NAT]))]:
        ar, ars, ap, aps = cv(X)
        rows.append([name, f"{ar:.3f}", f"{ars:.3f}", f"{ap:.3f}", f"{aps:.3f}"])
        L(f"  {name:18s} AUROC {ar:.3f}±{ars:.3f}  AUPRC {ap:.3f}±{aps:.3f}")
    L(f"  (baseline AUPRC = {y.mean():.3f})")

    # importance on fused
    X = np.hstack([DIS, NAT]); sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X), y)
    imp = sorted(zip(DIS_NAMES + NAT_NAMES, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    L("  fused coefs: " + ", ".join(f"{c}{w:+.2f}" for c, w in imp[:6]))

    native_auroc = float(rows[0][1]); dis_auroc = float(rows[1][1]); fused_auroc = float(rows[2][1])
    verdict = ("TRANSFERS: disagreement/fused clearly beats native-only -> cross-model disagreement "
               "predicts the driving model's failure too."
               if max(dis_auroc, fused_auroc) >= native_auroc + 0.05 else
               "WEAK TRANSFER: ~native-only. With only 3 (similar) driving models, disagreement adds "
               "little -> DIVERSITY is the key; distilling the diverse-VLM signal is the open task.")

    with open(OUT / "transfer_fusion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["detector", "AUROC", "AUROC_std", "AUPRC", "AUPRC_std"]); w.writerows(rows)

    (OUT / "transfer_RESULTS.md").write_text(
        "# Transfer test — does cross-model disagreement predict SparseDrive failure?\n\n"
        f"Set: official nuScenes-val, n={len(toks)} frames. Label = SparseDrive ADE top-10% "
        "(driving planner failure, NOT VLM failure). ADE excluded from features. 5-fold scene CV.\n\n"
        "Disagreement source = 3 real driving models (SparseDrive/DiffusionDrive/VAD). The diverse "
        "14-VLM ensemble was NOT re-run on nuScenes-val (paid APIs; not run autonomously).\n\n"
        "| detector | AUROC | AUPRC |\n|---|---|---|\n"
        + "\n".join(f"| {r[0]} | {r[1]}±{r[2]} | {r[3]}±{r[4]} |" for r in rows)
        + f"\n\nbaseline AUPRC = {y.mean():.3f}; native-only baseline AUROC = {native_auroc:.3f}.\n\n"
        f"**Verdict:** {verdict}\n")
    L("\n" + verdict)
    L("wrote transfer_fusion.csv, transfer_RESULTS.md")
    (OUT / "transfer_log.md").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()

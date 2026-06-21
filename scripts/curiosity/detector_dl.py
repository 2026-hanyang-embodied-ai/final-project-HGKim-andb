#!/usr/bin/env python3
"""Deep-learning curiosity detector on DriveLM-nuScenes (VLM ensemble, n~3908).

Compares, same set & same scene-5fold protocol as the LR baseline (0.818):
  LR        : logistic regression on 8 disagreement stats (reference)
  A) MLP    : the 8 disagreement stats -> (64,32) ReLU+dropout -> sigmoid
  B) SetNet : DeepSets over the M per-model trajectories (perm-invariant)
              phi 12->64->64 ; pool mean(+)std(+)max ; rho 192->64->1

Anti-leakage: ADE (label) is NEVER an input feature. The 14 VLMs are ZERO-SHOT
(prompted), not fine-tuned on nuScenes -> no scene memorization (logged).
Early stopping uses a 15% scene-split INSIDE each train fold; the held-out test
fold is used for the final AUROC ONLY.

Out: results/curiosity/report/detector_dl.csv, roc_dl.png, learning_curve_dl.png
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"   # tiny models, small data; avoid GPU contention with ollama

PROJ = Path(__file__).resolve().parents[2]
SRC = PROJ / "results/unified/lightemma_test_150.jsonl"
OUT = PROJ / "results/curiosity/report"
MODELS = ["claude-3.7-sonnet", "claude-4.0-sonnet", "deepseek-vl2-16b", "deepseek-vl2-28b",
          "gemini-2.5-flash", "gemini-2.5-pro", "gpt-4.1", "gpt-4o", "gpt-5",
          "llama-3.2-11b", "llama-3.2-90b", "qwen-2.5-7b", "qwen-2.5-7b-local"]
N_WP, MMAX = 6, 13
FEATS = ["disp_mean", "disp_1s", "disp_3s", "endpoint_lat_std",
         "pathlen_std", "heading_std", "max_pair_endpoint", "n_models"]
log_lines = []
def log(m): print(m); log_lines.append(m)


def valid(p):
    return isinstance(p, dict) and p.get("parse_ok") and isinstance(p.get("trajectory"), list) and len(p["trajectory"]) == N_WP


def load():
    rows = [json.loads(l) for l in open(SRC)]
    Xset = np.zeros((len(rows), MMAX, 12), np.float32)
    mask = np.zeros((len(rows), MMAX), np.float32)
    Xmlp, scenes, losses, keep = [], [], [], []
    for i, d in enumerate(rows):
        gt = np.asarray(d["gt_trajectory"], float)[:, :2]
        T, ades = [], []
        for m in MODELS:
            p = d["predictions"].get(m)
            if valid(p):
                tr = np.asarray(p["trajectory"], float)[:, :2]
                T.append(tr); ades.append(float(np.mean(np.linalg.norm(gt - tr, axis=1))))
        if len(T) < 2:
            continue
        T = np.stack(T)
        for j in range(min(len(T), MMAX)):
            Xset[len(keep), j] = T[j].reshape(-1)
            mask[len(keep), j] = 1.0
        mt = T.mean(0)
        pw = np.mean(np.sum((T - mt) ** 2, axis=2), axis=0)
        seg = np.linalg.norm(np.diff(T, axis=1), axis=2).sum(1)
        lv = T[:, -1] - T[:, -2]; head = np.arctan2(lv[:, 1], lv[:, 0])
        ep = T[:, -1]
        mp = max(np.linalg.norm(ep[a] - ep[b]) for a in range(len(ep)) for b in range(a + 1, len(ep)))
        Xmlp.append([pw.mean() ** .5, pw[1] ** .5, pw[-1] ** .5, np.std(ep[:, 1]),
                     np.std(seg), np.std(head), mp, len(T)])
        scenes.append(d["scene_name"]); losses.append(float(np.mean(ades))); keep.append(i)
    n = len(keep)
    Xset, mask = Xset[:n], mask[:n]
    Xmlp = np.array(Xmlp, np.float32); scenes = np.array(scenes); losses = np.array(losses)
    y = (losses >= np.percentile(losses, 90)).astype(np.float32)
    return Xset, mask, Xmlp, y, scenes


class MLP(nn.Module):
    def __init__(s, d=8):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
                              nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(s, x, m=None): return s.net(x).squeeze(-1)


class SetNet(nn.Module):
    def __init__(s):
        super().__init__()
        s.phi = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Dropout(0.2),
                             nn.Linear(64, 64), nn.ReLU())
        s.rho = nn.Sequential(nn.Linear(192, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(s, x, m):                       # x:(B,M,12) m:(B,M)
        e = s.phi(x)                            # (B,M,64)
        mm = m.unsqueeze(-1)
        cnt = mm.sum(1).clamp(min=1)
        mean = (e * mm).sum(1) / cnt
        var = ((e - mean.unsqueeze(1)) ** 2 * mm).sum(1) / cnt
        std = torch.sqrt(var + 1e-6)
        mx = (e.masked_fill(mm == 0, -1e9)).max(1).values
        return s.rho(torch.cat([mean, std, mx], -1)).squeeze(-1)


def train_nn(ctor, Xtr, mtr, ytr, Xva, mva, yva, pos_w, max_ep=80, patience=12):
    model = ctor().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w))
    Xtr_t, mtr_t, ytr_t = (torch.tensor(a) for a in (Xtr, mtr, ytr))
    Xva_t, mva_t = torch.tensor(Xva), torch.tensor(mva)
    best, best_state, wait, curve = -1, None, 0, []
    n = len(Xtr_t); bs = 256
    for ep in range(max_ep):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr_t[idx], mtr_t[idx])
            lossf(out, ytr_t[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ptr = torch.sigmoid(model(Xtr_t, mtr_t)).numpy()
            pva = torch.sigmoid(model(Xva_t, mva_t)).numpy()
        atr = roc_auc_score(ytr, ptr) if ytr.sum() else .5
        ava = roc_auc_score(yva, pva) if yva.sum() else .5
        curve.append((atr, ava))
        if ava > best:
            best, best_state, wait = ava, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model, curve


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log("## Deep detector on DriveLM-nuScenes")
    log("Leakage: the 14 VLMs are ZERO-SHOT (prompted), NOT fine-tuned on nuScenes -> no "
        "scene memorization; ADE reflects genuine zero-shot difficulty. ADE never used as input.")
    Xset, mask, Xmlp, y, scenes = load()
    log(f"n={len(y)} frames, {len(set(scenes))} scenes, positives={int(y.sum())} ({100*y.mean():.1f}%); "
        f"M per frame up to {MMAX}.")
    pos_w = float((len(y) - y.sum()) / max(y.sum(), 1))

    gkf = GroupKFold(5)
    oof = {"LR": np.zeros(len(y)), "MLP": np.zeros(len(y)), "SetNet": np.zeros(len(y))}
    per = {k: [] for k in oof}
    curves = {}
    rng = np.random.RandomState(0)
    for fold, (tr, te) in enumerate(gkf.split(Xmlp, y, scenes)):
        # inner 15% scene split for early stopping
        tr_scenes = np.array(sorted(set(scenes[tr])))
        rng.shuffle(tr_scenes)
        n_va = max(1, int(0.15 * len(tr_scenes)))
        va_set = set(tr_scenes[:n_va])
        sub = tr[~np.isin(scenes[tr], list(va_set))]
        val = tr[np.isin(scenes[tr], list(va_set))]

        # LR (8 feats)
        sc = StandardScaler().fit(Xmlp[sub])
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xmlp[sub]), y[sub])
        oof["LR"][te] = lr.predict_proba(sc.transform(Xmlp[te]))[:, 1]

        # MLP (8 feats, standardized)
        mlp, c_mlp = train_nn(MLP, sc.transform(Xmlp[sub]).astype(np.float32), mask[sub], y[sub],
                              sc.transform(Xmlp[val]).astype(np.float32), mask[val], y[val], pos_w)
        with torch.no_grad():
            oof["MLP"][te] = torch.sigmoid(mlp(torch.tensor(sc.transform(Xmlp[te]).astype(np.float32)), torch.tensor(mask[te]))).numpy()

        # SetNet (raw per-model trajs, coord-standardized on train-sub valid points)
        pts = Xset[sub][mask[sub] == 1]
        mu, sd = pts.mean(0), pts.std(0) + 1e-6
        Xs = ((Xset - mu) / sd).astype(np.float32) * mask[..., None]   # zero padded rows
        setn, c_set = train_nn(SetNet, Xs[sub], mask[sub], y[sub], Xs[val], mask[val], y[val], pos_w)
        with torch.no_grad():
            oof["SetNet"][te] = torch.sigmoid(setn(torch.tensor(Xs[te]), torch.tensor(mask[te]))).numpy()

        for k in oof:
            per[k].append(roc_auc_score(y[te], oof[k][te]))
        if fold == 0:
            curves = {"MLP": c_mlp, "SetNet": c_set}
        log(f"  fold{fold}: LR {per['LR'][-1]:.3f} | MLP {per['MLP'][-1]:.3f} | SetNet {per['SetNet'][-1]:.3f}")

    # table
    table = []
    log("\n## Results (5-fold scene CV)")
    for k in ["LR", "MLP", "SetNet"]:
        ar = np.array(per[k]); auroc_oof = roc_auc_score(y, oof[k]); auprc = average_precision_score(y, oof[k])
        table.append([k, f"{ar.mean():.3f}", f"{ar.std():.3f}", f"{auprc:.3f}"])
        log(f"  {k:7s} AUROC {ar.mean():.3f}±{ar.std():.3f}  AUPRC {auprc:.3f}")
    with open(OUT / "detector_dl.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "AUROC_mean", "AUROC_std", "AUPRC"]); w.writerows(table)

    # ROC fig
    plt.figure(figsize=(5, 5))
    for k, c in [("LR", "#1f77b4"), ("MLP", "#ff7f0e"), ("SetNet", "#d62728")]:
        fpr, tpr, _ = roc_curve(y, oof[k]); plt.plot(fpr, tpr, color=c, label=f"{k} ({roc_auc_score(y, oof[k]):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.7); plt.legend(fontsize=9, loc="lower right")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("DriveLM curiosity detector — DL vs LR")
    plt.tight_layout(); plt.savefig(OUT / "roc_dl.png", dpi=140); plt.close()

    # learning curves (fold 0)
    plt.figure(figsize=(7, 4))
    for k, c in [("MLP", "#ff7f0e"), ("SetNet", "#d62728")]:
        cu = np.array(curves[k]); plt.plot(cu[:, 0], "--", color=c, alpha=.6, label=f"{k} train")
        plt.plot(cu[:, 1], "-", color=c, label=f"{k} val(ES)")
    plt.xlabel("epoch"); plt.ylabel("AUROC"); plt.title("Training curves (fold 0)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUT / "learning_curve_dl.png", dpi=140); plt.close()

    (OUT / "detector_dl_log.md").write_text("\n".join(log_lines) + "\n")
    log("\nwrote detector_dl.csv, roc_dl.png, learning_curve_dl.png, detector_dl_log.md")


if __name__ == "__main__":
    main()

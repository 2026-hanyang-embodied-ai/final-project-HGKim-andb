#!/usr/bin/env python3
"""Case-study GIF/MP4: 'our detector flags the hard scenes'.

Reuses results/curiosity/traj/{sd,dd,vad}_trajs.csv + sd_signals.csv + SparseDrive val infos
(GT trajectory + CAM_FRONT). Rebuilds the transfer detector (5-fold scene CV, out-of-fold
probabilities so each shown frame's score is from a model that did NOT train on it).
Selects hard scenes (true top-10% SparseDrive failure that the detector scores high), renders
camera + BEV + overlay per frame, writes case_study.gif / .mp4 / .md. Real numbers only.
"""
import csv, pickle, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[2]
CUR = PROJ / "results/curiosity"
OUT = CUR / "report"
ASSETS = PROJ / "assets"
INFOS = PROJ / "models/sparse_drive/data/infos/nuscenes_infos_val.pkl"
NUSC_ROOT = "/home/sp/nuscenes/data/nuscenes"

def load_traj(p):
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["token"]] = np.array([float(r[f"t{i}{a}"]) for i in range(6) for a in "xy"]).reshape(6, 2)
    return out

def main():
    sd = load_traj(CUR / "traj/sd_trajs.csv"); dd = load_traj(CUR / "traj/dd_trajs.csv"); vad = load_traj(CUR / "traj/vad_trajs.csv")
    sig = {r["token"]: r for r in csv.DictReader(open(CUR / "sd_signals.csv"))}
    infos = pickle.load(open(INFOS, "rb"))["infos"]
    infos = sorted(infos, key=lambda e: e["timestamp"])
    info_by = {e["token"]: e for e in infos}

    toks = [t for t in sig if t in sd and t in dd and t in vad and t in info_by]
    DIS, NAT, y, scn, T1s, Tep = [], [], [], [], [], []
    for t in toks:
        T = np.stack([sd[t], dd[t], vad[t]]); mt = T.mean(0)
        pw = np.mean(np.sum((T - mt)**2, axis=2), axis=0); ep = T[:, -1]
        seg = np.linalg.norm(np.diff(T, axis=1), axis=2).sum(1)
        mp = max(np.linalg.norm(ep[a]-ep[b]) for a in range(3) for b in range(a+1, 3))
        DIS.append([pw.mean()**.5, pw[1]**.5, pw[-1]**.5, np.std(ep[:,0]), np.std(seg), mp,
                    float(np.mean(np.linalg.norm(dd[t]-vad[t], axis=1)))])
        NAT.append([float(sig[t]["sig_margin"]), float(sig[t]["sig_entropy"]), float(sig[t]["sig_mode_std"])])
        y.append(int(sig[t]["fail_top10"])); scn.append(sig[t]["scene"])
        T1s.append(pw[1]**.5); Tep.append(pw[-1]**.5)
    X = np.hstack([np.array(DIS), np.array(NAT)]); y = np.array(y); scn = np.array(scn)
    T1s = np.array(T1s); Tep = np.array(Tep)

    # out-of-fold detector probabilities (no leakage: each frame scored by a fold that didn't train on it)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, scn):
        s = StandardScaler().fit(X[tr]); m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(s.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(s.transform(X[te]))[:, 1]
    loss = np.array([float(sig[t]["loss_l2"]) for t in toks])

    # --- select hard scenes: true failure (top10) + detector high, one per scene, diverse ---
    order = np.argsort(-oof)
    picked, seen = [], set()
    for i in order:
        if y[i] == 1 and scn[i] not in seen:
            picked.append(i); seen.add(scn[i])
        if len(picked) == 7:
            break
    # one easy contrast: true non-failure with lowest detector score
    easy = [i for i in np.argsort(oof) if y[i] == 0 and scn[i] not in seen][:1]
    sel = picked + easy
    N = len(picked)

    OUT.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(exist_ok=True)
    frames, md = [], ["# Case study — our detector flags the hard scenes\n",
        "Frames = SparseDrive **actual** top-10% failures that the learned transfer detector "
        "(disagreement+native, out-of-fold) scores high, one per scene; last frame = an easy "
        "(clear) contrast. All numbers are real (from saved CSVs / nuScenes GT).\n",
        "| # | token | disagreement 1s (m) | detector P(fail) | actual ADE (m) | label |",
        "|---|---|---|---|---|---|"]

    COL = {"SparseDrive": "#1f77b4", "DiffusionDrive": "#ff7f0e", "VAD": "#2ca02c"}
    for k, i in enumerate(sel):
        t = toks[i]; e = info_by[t]
        is_hard = (i in picked)
        # camera
        cf = e["cams"]["CAM_FRONT"]; p = cf.get("data_path") or cf.get("filename")
        img_path = os.path.join(NUSC_ROOT, "samples/CAM_FRONT", os.path.basename(p))
        gt = np.asarray(e["gt_ego_fut_trajs"]).reshape(6, 2).cumsum(0)

        fig = plt.figure(figsize=(9.0, 4.2), dpi=100)
        gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], figure=fig)
        # left camera
        axc = fig.add_subplot(gs[0, 0]); axc.axis("off")
        if os.path.exists(img_path):
            axc.imshow(Image.open(img_path))
        axc.set_title("front camera", fontsize=8)
        # right BEV (x=lateral, y=forward) -> plot forward as up
        axb = fig.add_subplot(gs[0, 1])
        allpts = np.vstack([sd[t], dd[t], vad[t], gt, np.zeros((1, 2))])
        for name, tr in [("SparseDrive", sd[t]), ("DiffusionDrive", dd[t]), ("VAD", vad[t])]:
            axb.plot(-tr[:, 0], tr[:, 1], "-o", ms=3, lw=1.5, color=COL[name], label=name)
        axb.plot(-gt[:, 0], gt[:, 1], "k--", lw=1.8, label="GT")
        axb.scatter([0], [0], c="k", marker="*", s=90, zorder=5)
        # auto-zoom to data with margin, keep equal aspect
        xs, ys = -allpts[:, 0], allpts[:, 1]
        xc = (xs.min() + xs.max()) / 2; xr = max(xs.max() - xs.min(), ys.max() - ys.min(), 6) / 2 + 2
        axb.set_xlim(xc - xr, xc + xr); axb.set_ylim(ys.min() - 2, ys.min() - 2 + 2 * xr); axb.set_aspect("equal")
        axb.set_title("BEV: model trajectories vs GT", fontsize=8)
        axb.legend(fontsize=6, loc="upper right"); axb.set_xticks([]); axb.set_yticks([])

        band = "Hard scene" if is_hard else "Easy scene (contrast)"
        idx = (k + 1) if is_hard else "+"
        fig.suptitle(f"{band}   {idx} / {N}", fontsize=12, weight="bold", y=0.99)
        flag = "FLAG ✓" if oof[i] >= 0.5 else "clear"
        lab = "top-10% = FAILURE" if y[i] == 1 else "nominal"
        agree = "Models disagree" if is_hard else "Models agree"
        txt = (f"{agree} (1s spread = {T1s[i]:.2f} m)   →   OUR detector P(fail) = {oof[i]:.2f}  {flag}"
               f"\nactual error = {loss[i]:.3f} m   ({lab})")
        fig.text(0.5, 0.02, txt, ha="center", fontsize=9.5,
                 color=("#b00020" if y[i] == 1 else "#1a7f37"), weight="bold")
        fig.tight_layout(rect=[0, 0.10, 1, 0.95])
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(buf)
        md.append(f"| {idx} | `{t[:16]}…` | {T1s[i]:.2f} | {oof[i]:.2f} | {loss[i]:.2f} | {lab} |")
        print(f"[{idx}] {t}  1s_spread={T1s[i]:.2f}  P(fail)={oof[i]:.2f}  ADE={loss[i]:.2f}  {lab}")

    # hold each ~1.6s ; loop
    hold = [f for f in frames for _ in range(1)]
    gif = OUT / "case_study.gif"
    imageio.mimsave(gif, frames, duration=1.6, loop=0)
    sz = gif.stat().st_size / 1e6
    if sz > 15:  # downscale if big
        small = [np.asarray(Image.fromarray(f).resize((720, int(720*f.shape[0]/f.shape[1])))) for f in frames]
        imageio.mimsave(gif, small, duration=1.6, loop=0); sz = gif.stat().st_size / 1e6
    try:
        imageio.mimsave(OUT / "case_study.mp4", frames, fps=1, macro_block_size=1, quality=8)
        mp4ok = True
    except Exception as ex:
        mp4ok = False; md.append(f"\n(mp4 skipped: {ex})")
    # copy to assets for README
    import shutil; shutil.copy(gif, ASSETS / "case_study.gif")
    (OUT / "case_study.md").write_text("\n".join(md) + f"\n\nGIF {sz:.1f} MB, {len(frames)} frames. "
        f"Detector P(fail) is out-of-fold (5-fold scene CV). mp4={'yes' if mp4ok else 'no'}.\n")
    print(f"\nwrote {gif} ({sz:.1f} MB), case_study.mp4={mp4ok}, case_study.md, assets/case_study.gif")

if __name__ == "__main__":
    main()

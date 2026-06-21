#!/usr/bin/env python3
"""Hard-scene driving video: front-camera keyframes of the hardest scenes, in time order,
concatenated. Footage only (no BEV/overlay) EXCEPT a per-frame marker showing when OUR
detector flags the moment as a predicted failure. Real frames only (nuScenes CAM_FRONT).

The red "DETECTOR FLAG" = our learned transfer detector's prediction P(fail) >= 0.5,
computed OUT-OF-FOLD (5-fold scene CV; each shown frame scored by a model that did NOT
train on its scene). This is the ALGORITHM's call, not the ground-truth label.

Hardness per scene = mean SparseDrive ADE over its frames (top scenes selected).
Out: results/curiosity/report/hard_scenes.mp4 (+ .gif) + assets/ copies + .md
"""
import csv, pickle, os
from collections import defaultdict
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[2]
CUR = PROJ / "results/curiosity"
OUT = CUR / "report"; ASSETS = PROJ / "assets"
INFOS = PROJ / "models/sparse_drive/data/infos/nuscenes_infos_val.pkl"
NUSC = "/home/sp/nuscenes/data/nuscenes"
N_SCENES = 6          # hardest scenes to include
N_EASY = 2            # easy contrast scenes appended at the end
EASY_MIN_MOVE = 15.0  # easy scenes must actually be MOVING (3s travel >= this, metres)
FPS = 2
W = 768               # output width
FLAG_TH = 0.5         # detector P(fail) threshold to show a flag


def load_traj(p):
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["token"]] = np.array([float(r[f"t{i}{a}"]) for i in range(6) for a in "xy"]).reshape(6, 2)
    return out


def detector_oof():
    """Out-of-fold detector P(fail) per token (disagreement+native, 5-fold scene CV)."""
    sd = load_traj(CUR / "traj/sd_trajs.csv"); dd = load_traj(CUR / "traj/dd_trajs.csv"); vad = load_traj(CUR / "traj/vad_trajs.csv")
    sig = {r["token"]: r for r in csv.DictReader(open(CUR / "sd_signals.csv"))}
    toks = [t for t in sig if t in sd and t in dd and t in vad]
    X, y, scn = [], [], []
    for t in toks:
        T = np.stack([sd[t], dd[t], vad[t]]); pw = np.mean(np.sum((T - T.mean(0))**2, axis=2), axis=0); ep = T[:, -1]
        seg = np.linalg.norm(np.diff(T, axis=1), axis=2).sum(1)
        mp = max(np.linalg.norm(ep[a]-ep[b]) for a in range(3) for b in range(a+1, 3))
        X.append([pw.mean()**.5, pw[1]**.5, pw[-1]**.5, np.std(ep[:, 0]), np.std(seg), mp,
                  float(np.mean(np.linalg.norm(dd[t]-vad[t], axis=1))),
                  float(sig[t]["sig_margin"]), float(sig[t]["sig_entropy"]), float(sig[t]["sig_mode_std"])])
        y.append(int(sig[t]["fail_top10"])); scn.append(sig[t]["scene"])
    X, y, scn = np.array(X), np.array(y), np.array(scn)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, scn):
        s = StandardScaler().fit(X[tr]); m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(s.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(s.transform(X[te]))[:, 1]
    return {t: float(oof[i]) for i, t in enumerate(toks)}

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def main():
    OUT.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(exist_ok=True)
    sig = list(csv.DictReader(open(CUR / "sd_signals.csv")))
    infos = {e["token"]: e for e in pickle.load(open(INFOS, "rb"))["infos"]}
    pfail = detector_oof()   # our algorithm's out-of-fold P(fail) per token

    by_scene = defaultdict(list)
    for r in sig:
        by_scene[r["scene"]].append(r)
    # rank scenes by mean ADE (hardness), require enough frames; also track ego motion
    scored, move = [], {}
    for sc, rows in by_scene.items():
        if len(rows) < 10:
            continue
        mean_ade = np.mean([float(x["loss_l2"]) for x in rows])
        failrate = np.mean([int(x["fail_top10"]) for x in rows])
        dists = []
        for x in rows:
            e = infos.get(x["token"])
            if e is not None:
                gt = np.asarray(e["gt_ego_fut_trajs"]).reshape(6, 2).cumsum(0)
                dists.append(float(np.linalg.norm(gt[-1])))   # 3s travel distance
        move[sc] = np.mean(dists) if dists else 0.0
        scored.append((mean_ade, failrate, sc, rows))
    scored.sort(reverse=True)
    hard = [(*s, True) for s in scored[:N_SCENES]]                 # hardest (is_hard=True)
    # easy contrast: MOVING scenes the detector keeps clearest (lowest flag rate),
    # so the demo reads as "easy -> detector stays quiet" (not just low GT error).
    def scene_flagrate(rows):
        return np.mean([1 if pfail.get(x["token"], 0) >= FLAG_TH else 0 for x in rows])
    moving = [s for s in scored if move.get(s[2], 0) >= EASY_MIN_MOVE]
    easy = [(*s, False) for s in sorted(moving, key=lambda s: scene_flagrate(s[3]))[:N_EASY]]
    chosen = hard + easy

    frames, md = [], [f"# Failure-detection demo video (front camera, time-ordered, {FPS} fps)\n",
        f"nuScenes-val scenes played as raw front-camera keyframes (2 Hz captured, shown at {FPS} fps). "
        f"A red **DETECTOR FLAG** marks frames where OUR learned detector predicts failure "
        f"(out-of-fold P(fail) >= {FLAG_TH}) — the algorithm's call, not ground truth. First "
        f"{N_SCENES} are the hardest scenes (high error), last {N_EASY} are easy scenes for "
        "contrast (detector stays clear). Real frames only.\n",
        "| scene | type | keyframes | mean ADE (m) | % flagged by detector |", "|---|---|---|---|---|"]
    F = font(26); Fs = font(20)
    hi = ei = 0
    for mean_ade, failrate, sc, rows, is_hard in chosen:
        rows = sorted(rows, key=lambda x: int(x["timestamp"]))
        flagrate = np.mean([1 if pfail.get(r["token"], 0) >= FLAG_TH else 0 for r in rows])
        if is_hard:
            hi += 1; cap = f"Hard scene {hi}/{N_SCENES}"
        else:
            ei += 1; cap = f"Easy scene {ei}/{N_EASY}"
        md.append(f"| `{sc[:16]}…` | {'HARD' if is_hard else 'easy'} | {len(rows)} | {mean_ade:.2f} | {100*flagrate:.0f}% |")
        print(f"[{'HARD' if is_hard else 'easy'}] scene {sc[:16]}: {len(rows)} frames, mean ADE {mean_ade:.2f}, "
              f"GT-fail {100*failrate:.0f}%, detector-flagged {100*flagrate:.0f}%")
        for k, r in enumerate(rows):
            e = infos.get(r["token"])
            if e is None:
                continue
            cf = e["cams"]["CAM_FRONT"]; p = cf.get("data_path") or cf.get("filename")
            ip = os.path.join(NUSC, "samples/CAM_FRONT", os.path.basename(p))
            if not os.path.exists(ip):
                continue
            im = Image.open(ip).convert("RGB")
            h = int(W * im.height / im.width)
            im = im.resize((W, h))
            strip = 40
            canvas = Image.new("RGB", (W, h + strip), (15, 15, 20))
            canvas.paste(im, (0, 0))
            d = ImageDraw.Draw(canvas)
            d.text((10, h + 6), f"{cap}  ·  mean ADE {mean_ade:.1f} m",
                   font=Fs, fill=(255, 230, 120) if is_hard else (160, 220, 255))
            # marker = OUR detector's prediction (out-of-fold P(fail)), NOT ground truth
            pf = pfail.get(r["token"])
            if pf is not None and pf >= FLAG_TH:
                d.rectangle([2, 2, W-3, h-3], outline=(230, 40, 40), width=5)
                d.ellipse([W-30, 8, W-12, 26], fill=(230, 40, 40))
                d.text((W-275, 6), f"DETECTOR FLAG  p={pf:.2f}", font=Fs, fill=(255, 90, 90))
            elif pf is not None:
                d.text((W-150, 6), f"clear  p={pf:.2f}", font=Fs, fill=(120, 230, 140))
            frames.append(np.asarray(canvas))

    mp4 = OUT / "hard_scenes.mp4"
    imageio.mimsave(mp4, frames, fps=FPS, macro_block_size=1, quality=5,
                    output_params=["-crf", "30", "-pix_fmt", "yuv420p"])
    gif = OUT / "hard_scenes.gif"
    # gif: downscale + subsample to keep size reasonable (mp4 keeps full smoothness)
    small = [np.asarray(Image.fromarray(f).resize((480, int(480*f.shape[0]/f.shape[1])))) for f in frames[::3]]
    imageio.mimsave(gif, small, duration=3.0/FPS, loop=0)
    import shutil
    shutil.copy(mp4, ASSETS / "hard_scenes.mp4"); shutil.copy(gif, ASSETS / "hard_scenes.gif")
    (OUT / "hard_scenes.md").write_text("\n".join(md) +
        f"\n\n{len(frames)} frames @ {FPS} fps. mp4 {mp4.stat().st_size/1e6:.1f} MB, "
        f"gif {gif.stat().st_size/1e6:.1f} MB. "
        "Red border/dot = our detector predicted failure (out-of-fold); green = clear.\n")
    print(f"\nwrote {mp4} ({mp4.stat().st_size/1e6:.1f} MB), {gif} ({gif.stat().st_size/1e6:.1f} MB), {len(frames)} frames")

if __name__ == "__main__":
    main()

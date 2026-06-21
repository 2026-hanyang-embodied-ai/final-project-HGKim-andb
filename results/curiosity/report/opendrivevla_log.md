# Bonus — OpenDriveVLA transfer test: got VERY far, STOPPED on a deep dependency conflict

Date: 2026-06-21. Bonus experiment (separate from submission). No submission files modified;
no git commit. Nothing left running.

## Goal
Test whether our cross-model **disagreement** signal predicts an actual nuScenes fast–slow VLA's
failure: run OpenDriveVLA on nuScenes-val → per-frame trajectory → ADE top-10% = failure label →
does disagreement (SD/DD/VAD) predict it (parallel to SparseDrive 0.76)?

## Candidate choice (recap)
FASIONAD / AdaDrive = code/weights not released. **OpenDriveVLA** = inference code + 0.5B
checkpoint released, **outputs trajectories** (12 waypoints, nuScenes) → the only viable one.

## How far it got (almost everything)
1. ✅ HF **gated access granted** by user; checkpoint downloaded (1.4 GB) using the project SSL
   workaround (proxy CA `myca.crt` + `@SECLEVEL=1`).
2. ✅ **Env** cloned from `sparsedrive` → `drivevla` (avoids the documented mmcv/mmdet3d source
   compile; mmdet3d 1.0.0rc6 already matches).
3. ✅ **Data** linked: nuScenes full + UniAD `nuscenes_infos_temporal_val.pkl`; downloaded
   GPT-Driver `cached_nuscenes_info.pkl` (716 MB) via `curl -k` (gdown/pip kept failing on the
   MITM proxy SSL — curl insecure worked).
4. ✅ Installed OpenDriveVLA package (`pip install -e . --no-deps`) + missing deps through the
   broken-SSL pip (`--trusted-host`): mmengine, deepspeed 0.14.2, open_clip, av, casadi,
   motmetrics, pytorch-lightning, torchmetrics. Imports of llava builder / mmengine / deepspeed
   all succeeded.
5. ❌ **BLOCKER — version conflict in UniAD `occ_head`:** its `metrics.py` imports the *old*
   `pytorch_lightning.metrics.metric` API (removed after pl 1.4). That API needs an old
   `torchmetrics`, but pl 1.4.9 + torchmetrics 0.11.4 + torch 2.0 are mutually incompatible
   (pl.metrics.utils import fails). Resolving it pulls a downgrade chain (old torchmetrics →
   conflicts again with torch 2.0). `occ_head` is the occupancy head — **not needed for planning**
   — but it is imported at `mmdet3d_plugin/uniad` package init, so inference won't start.

## Why STOP (per plan's stop-criteria)
This is a genuine dependency conflict. Clearing it cleanly requires building the **exact
documented env from scratch** (fresh torch 2.1.2 + source-compiled mmcv 1.7.2/mmdet3d 1.0.0rc6 +
the pinned transformers git commit + pl 1.2.5 + torchmetrics 0.11.4) — the multi-hour source
build the clone was meant to avoid. Hacking out the `occ_head` import in the third-party repo
would likely cascade. Not worth burning the night for a bonus.

## State left (reusable if pursued)
- `/home/sp/Project/_bonus_senna/OpenDriveVLA` (repo + `checkpoints/OpenDriveVLA-0.5B` 1.4 GB +
  `data/nuscenes/cached_nuscenes_info.pkl` 716 MB + data symlinks).
- conda env `drivevla` (clone of sparsedrive + the extra deps above).

## To finish later (est. 1 build session + ~1–3 h inference)
1. Build the **exact** documented `drivevla` env (`docs/1_INSTALL.md`: source mmcv 1.7.2 +
   mmdet3d 1.0.0rc6, pl 1.2.5, torchmetrics 0.11.4) — this resolves the occ_head conflict.
2. `bash scripts/eval_drivevla.sh checkpoints/OpenDriveVLA-0.5B 1` → `output/plan_conv_val.json`.
3. Parse predicted trajectories → ADE vs nuScenes GT → top-10% label; run the transfer detector
   (`scripts/curiosity/transfer_detector.py` pattern) with SD/DD/VAD disagreement features.

## Recommendation
Keep OpenDriveVLA as **Future Work** (everything but the exact env is in place). The submitted
real-driving-model transfer result — **SparseDrive, AUROC 0.76** — already answers the core
question; OpenDriveVLA would be a second, VLA-specific confirmation.

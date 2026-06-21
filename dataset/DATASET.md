# Derived Dataset — Curiosity Detection

This folder contains **only data we derived ourselves** (model trajectories, per-frame
signals/labels, and result tables). It does **not** redistribute the raw nuScenes or Waymo
sensor data — those must be obtained from their official sites under their own licenses
(see *Licensing* below). All values here come from running released model checkpoints /
zero-shot VLMs on those public datasets.

## Files & schema

### `trajectories/` — aligned predicted ego futures (nuScenes val, 6019 frames)
`sd_trajs.csv`, `dd_trajs.csv`, `vad_trajs.csv` — final planned ego trajectory of
SparseDrive / DiffusionDrive / VAD-Base.
| column | meaning |
|---|---|
| `token` | nuScenes sample token (join key across the 3 files) |
| `t0x,t0y … t5x,t5y` | 6 future waypoints (0.5–3.0 s), absolute ego frame (x=lateral, y=forward), metres |

VAD raw output is per-step delta and was cumulatively summed to absolute before saving, so
all three files share one convention. Alignment validated by recomputed ADE (~1.0–1.2 m,
matching each model's reported planning L2).

### `signals/` — per-frame detector inputs + labels
`vlm_ensemble_signals.csv` (DriveLM-nuScenes, 3908 frames): disagreement of ~14 zero-shot
VLM trajectories.
| column | meaning |
|---|---|
| `scene`, `frame_index` | scene id / ordered frame index |
| `signal_var`, `signal_std`, `signal_endpoint_var` | trajectory disagreement (positional spread across models) |
| `n_models` | # valid VLM predictions in the frame |
| `loss_mean`, `loss_ensmean` | mean ADE across models / ADE of the mean trajectory (**label source only**) |
| `fail_top05/10/15/20` | failure label = mean-ADE in worst 5/10/15/20 % (**top10 is primary**) |
| `within_scene_pct`, `fail_within_scene` | within-scene relative-difficulty variant |

`sparsedrive_signals.csv`, `diffusiondrive_signals.csv` (nuScenes val, 6019 frames):
single-model native uncertainty.
| column | meaning |
|---|---|
| `scene`, `frame_index`, `timestamp`, `token`, `cmd` | frame id / nuScenes token / driving command |
| `sig_mode_var`, `sig_mode_std` | spread of the model's 6 candidate trajectories |
| `sig_entropy`, `sig_margin` | entropy / 1−(top1−top2) of the mode scores |
| `loss_l2` | the model's own ADE vs GT (**label source only**) |
| `fail_top05/10/15/20`, `fail_within_scene` | failure labels (top10 primary) |

> ⚠ **Anti-circularity:** the ADE/loss columns are the *label source only* and are **never**
> used as model inputs. Detectors read only the disagreement / uncertainty signal columns.

### `results/` — final result tables (also under `results/curiosity/report/`)
`comparison_table.csv` (per-signal AUROC/AUPRC), `detector_dl.csv` (LR/MLP/DeepSets),
`detector_drivelm.csv` (feature importance), `detector_table.csv` (single-model internal),
`robustness_table.csv` (5/10/20 % thresholds), `transfer_fusion.csv` (transfer test).

## Provenance
- **Models:** SparseDrive (`sparsedrive_stage2`), DiffusionDrive (`diffusiondrive_nusc_stage2`),
  VAD-Base (`vad_base`) — official released checkpoints, run on nuScenes val.
  VLM ensemble = ~14 zero-shot vision-language models (Claude/GPT/Gemini/Qwen/Llama/DeepSeek)
  prompted via the LightEMMA pipeline on DriveLM-nuScenes frames (no fine-tuning → no scene memorization).
- **Labels:** measured trajectory error (ADE) vs logged GT; top-decile = failure.

## Licensing
Raw **nuScenes** (https://www.nuscenes.org) and **Waymo Open Dataset**
(https://waymo.com/open) are subject to their own non-commercial licenses and are **not**
included here. This derived data is shared for reproducibility of our analysis only.

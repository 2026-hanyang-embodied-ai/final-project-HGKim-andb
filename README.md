[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D1tTEX1K)

# Failure-Aware Mode Switching for E2E Autonomous Driving

**One line:** a fast end-to-end (E2E) driving policy runs by default; we study a *trigger*
that detects when the fast policy is about to fail so control can escalate to a slower,
stronger VLA. **This semester = the detection module only** (a passive failure monitor; no
closed-loop switching).

## Links
| Item | Link |
|---|---|
| Presentation video | https://youtu.be/zz4gnZANk2w |
| Slides | [`deliverables/Failure_Aware_Slides.pdf`](deliverables/Failure_Aware_Slides.pdf) |
| Report | `deliverables/report.pdf` _(TODO: compile from Overleaf)_ |
| Dataset | _TODO: Google Drive URL (upload `dataset.zip`)_ |
| Demo video | https://youtu.be/zz4gnZANk2w _(included within the presentation)_ |

*Demo video shows the hardest nuScenes-val scenes (front camera) with a red **DETECTOR FLAG**
on frames our detector predicts as failures (out-of-fold), then easy moving scenes where it
stays clear — i.e. the detector fires on hard moments and passes easy ones.*

## Overview
E2E planners are fast but degrade on rare, long-tail situations; VLA models reason more
robustly but are too slow to run every frame. A practical *fast–slow* design runs the cheap
policy by default and calls the expensive model only when needed — the open problem is the
**trigger**. We treat it as a **detection-only** task: a passive monitor scores each frame
for impending failure (it does not alter the trajectory or invoke the VLA). We compare
candidate signals — **cross-model disagreement**, single-model **native uncertainty**, and a
VLM's **perceived risk** — against a *measured* failure label, and train a small learned
detector. Cognitively, this mirrors how *ambiguity/conflict* (disagreement among plausible
interpretations) recruits deliberate attention.

## Key results
All numbers come from `results/curiosity/report/` (CSV/PNG); see `final-project.ipynb`.
Failure label = **measured** top-10 % trajectory error (ADE); **ADE is never an input
feature** (non-circular); the VLM ensemble is **zero-shot** (no leakage); CV is **scene-level**.

**Which signal predicts failure?** (`comparison_table.csv`)
| signal | AUROC | AUPRC |
|---|---|---|
| **Cross-model disagreement** (VLM ensemble) | **0.80** | 0.375 |
| Single-model native uncertainty | 0.55–0.61 | ~0.11–0.14 |
| VLM perceived risk (escalation) | 0.46 (≈ chance) | 0.10 |

**Learned detector** (DriveLM-nuScenes, 5-fold scene CV; `detector_dl.csv`): LR **0.815** /
MLP **0.812** / DeepSets **0.819** ≈ **0.82**. A single model's internal signals alone reach
only ≈ **0.61** (`detector_table.csv`). Top features: `pathlen_std` +1.37, `disp_1s` +0.98
(`detector_drivelm.csv`) — near-term/speed disagreement matters most.

**Transfer (core result)** — does disagreement predict an *actual driving model's* failure?
Official nuScenes-val, 6019 frames, label = **SparseDrive** ADE top-10 % (`transfer_fusion.csv`):
| detector | AUROC |
|---|---|
| native-only | 0.618 |
| **disagreement-only** | **0.755** |
| **fused** | **0.762** |
| leave-SD-out (DD vs VAD only → SD failure) | 0.667 |

→ Disagreement predicts the driving planner's failure at **0.76**, far above its own native
uncertainty (0.62). **Not self-referential:** excluding SparseDrive from the disagreement
(DD vs VAD only) still gives **0.667 > 0.62**. *Conservative:* the diverse 14-VLM ensemble was
**not** re-run on nuScenes-val (paid APIs), so this uses only 3 (partly similar) driving
models — more diversity (cf. VLM 0.80) would likely raise it.

**Negative results:** VLM perceived risk does **not** track failure (AUROC 0.46, Spearman
−0.03…+0.07); temporal accumulation does **not** help in open-loop (single-frame 0.798 >
EWMA 0.760 > CUSUM 0.715).

## Repository structure
```
final-project.ipynb        # executed notebook: loads CSV/PNG, reproduces all tables/figures
scripts/curiosity/         # analysis code (detectors, trajectory extraction, transfer test)
results/curiosity/report/  # result tables (*.csv) + figures (*.png)
results/curiosity/traj/    # aligned predicted trajectories (sd/dd/vad)
dataset/                   # derived data (trajectories, signals, results) + DATASET.md
                           #   (also packaged as dataset.zip on Google Drive — see Links)
deliverables/              # slides PDF (+ report.pdf to add); demo video on YouTube
```

## Reproduce
Heavy GPU inference is **already done**; the notebook only loads the saved CSV/PNG and
re-renders tables/figures (plus one quick `scikit-learn` recompute of the transfer AUROC).
Runs in seconds, CPU-only.

- Python 3.8+; packages: `pandas numpy matplotlib scikit-learn` (+ `jupyter nbconvert` to run).
- Execute end-to-end:
  ```bash
  jupyter nbconvert --to notebook --execute --inplace final-project.ipynb
  ```
  or open `final-project.ipynb` in Jupyter and *Run All*. Paths are relative to the repo root.

## Data & licensing
We share **only derived data** we produced (predicted trajectories, per-frame
signals/labels, result tables) under `dataset/` — see `dataset/DATASET.md` for schemas and
provenance. The **raw** nuScenes (https://www.nuscenes.org) and Waymo Open Dataset
(https://waymo.com/open) are **not** redistributed and must be downloaded from their official
sites under their own licenses.

---
*Course project — Failure-Aware Mode Switching (detection module).*

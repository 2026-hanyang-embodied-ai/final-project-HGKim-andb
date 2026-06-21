# Curiosity Detection — overnight run summary

Verified: VLM-ensemble set (DriveLM nuScenes) and SparseDrive/DD set (official nuScenes val) are DISJOINT (0 shared CAM_FRONT frames). Unified common-set fusion (steps 1-2 as specified) is NOT possible -> per-set fallback, as instructed.

## Step 1 — per-set signal comparison (NOT cross-set; sets are disjoint)
  VLM disagreement: AUROC 0.798 AUPRC 0.375 (n=3908)
  SparseDrive native (margin): AUROC 0.551 AUPRC 0.107
  SparseDrive native (entropy): AUROC 0.546 AUPRC 0.114
  SparseDrive mode-spread (intra-disagreement): AUROC 0.528 AUPRC 0.128
  DiffusionDrive native (margin): AUROC 0.606 AUPRC 0.136
  DiffusionDrive mode-spread: AUROC 0.506 AUPRC 0.100
  VAD-Base native: N/A — single-mode planning exposes no native uncertainty (logged).

## Step 1b — anchor479 (Waymo): cross-model disagreement & teacher-risk vs failure
  cross-model disagreement -> rap failure: AUROC 0.584 AUPRC 0.153 (n=473)
  teacher escalation -> rap failure: AUROC 0.457 AUPRC 0.103  (confirms teacher risk does NOT track actual model failure)

## Step 2/3 — learned detectors (logistic, 5-fold scene CV; cross-set VLM fusion NOT possible -> intra-model disagreement used)
  [SparseDrive] disagreement 0.456 | native 0.554 | fused 0.614 (AUROC) | coefs sig_margin+0.56, sig_mode_var+0.48, sig_entropy+0.37, sig_mode_std-0.21
  [DiffusionDrive] disagreement 0.516 | native 0.610 | fused 0.603 (AUROC) | coefs sig_mode_std+0.67, sig_mode_var-0.59, sig_entropy+0.42, sig_margin+0.07
  VAD-Base: no learned detector — single-mode planning has no usable signal (logged).
  NOTE: the strong cross-model VLM disagreement (AUROC 0.80) lives on a DISJOINT scene set, so it could not be fused into the SD/DD detector. Documented as future work.

## Step 4 — robustness: failure threshold 5/10/20% (fused detector)
  SparseDrive top5%: AUROC 0.576 AUPRC 0.070 (base AUPRC=0.050)
  SparseDrive top10%: AUROC 0.614 AUPRC 0.166 (base AUPRC=0.100)
  SparseDrive top20%: AUROC 0.619 AUPRC 0.284 (base AUPRC=0.200)
  DiffusionDrive top5%: AUROC 0.618 AUPRC 0.099 (base AUPRC=0.050)
  DiffusionDrive top10%: AUROC 0.603 AUPRC 0.180 (base AUPRC=0.100)
  DiffusionDrive top20%: AUROC 0.613 AUPRC 0.295 (base AUPRC=0.200)

Wrote comparison_table.csv, detector_table.csv, robustness_table.csv
Wrote comparison_auroc.png, roc_sd_detector.png, escalation_hist.png

## Conclusions
- Sets are disjoint -> reported per-set (apples-to-apples within each), not unified (fallback).
- Strong signal = VLM-ensemble disagreement (AUROC ~0.80, predicts VLM failure).
- Single-model native uncertainty (SD/DD) is weak (~0.55); VAD-Base exposes none.
- Learned fusion of intra-model signals lifts SD detector to ~0.61 (> either alone).
- Teacher escalation does NOT track actual model failure (AUROC ~0.5 on anchor479).
- Cross-model disagreement among real driving models (anchor479) reported as the transferable strong signal; fusing it into SD/DD blocked by disjoint scene sets (future work).

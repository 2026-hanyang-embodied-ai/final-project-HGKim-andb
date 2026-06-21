## Deep detector on DriveLM-nuScenes
Leakage: the 14 VLMs are ZERO-SHOT (prompted), NOT fine-tuned on nuScenes -> no scene memorization; ADE reflects genuine zero-shot difficulty. ADE never used as input.
n=3908 frames, 150 scenes, positives=391 (10.0%); M per frame up to 13.
  fold0: LR 0.822 | MLP 0.824 | SetNet 0.815
  fold1: LR 0.873 | MLP 0.877 | SetNet 0.852
  fold2: LR 0.778 | MLP 0.772 | SetNet 0.809
  fold3: LR 0.782 | MLP 0.780 | SetNet 0.785
  fold4: LR 0.817 | MLP 0.808 | SetNet 0.836

## Results (5-fold scene CV)
  LR      AUROC 0.815±0.034  AUPRC 0.410
  MLP     AUROC 0.812±0.037  AUPRC 0.439
  SetNet  AUROC 0.819±0.023  AUPRC 0.402

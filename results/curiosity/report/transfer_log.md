common frames (SD∩DD∩VAD∩signals): 6019 / SD 6019
label = SparseDrive ADE top10%: positives 602 (10.0%); disagreement among 3 DRIVING models (SD/DD/VAD; SD&DD similar -> low diversity).

=== 5-fold scene-CV (predict SparseDrive failure) ===
  native-only        AUROC 0.618±0.026  AUPRC 0.170±0.058
  disagreement-only  AUROC 0.755±0.018  AUPRC 0.290±0.055
  fused              AUROC 0.762±0.023  AUPRC 0.293±0.054
  (baseline AUPRC = 0.100)
  fused coefs: dis_maxpair+1.81, dis_mean-1.51, dis_DDvsVAD-1.05, dis_3s+0.62, dis_pathlen+0.48, sig_margin+0.46

TRANSFERS: disagreement/fused clearly beats native-only -> cross-model disagreement predicts the driving model's failure too.
wrote transfer_fusion.csv, transfer_RESULTS.md

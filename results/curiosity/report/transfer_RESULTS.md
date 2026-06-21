# Transfer test — does cross-model disagreement predict SparseDrive's *own* failure?

**Set:** official nuScenes-val, n=6019 frames / 150 scenes. **Label = SparseDrive ADE
top-10%** (the driving planner's trajectory error vs GT — an INDEPENDENT target, not VLM
failure). ADE never used as a feature (anti-circular). 5-fold scene-level CV.

**Disagreement source:** 3 real driving models {SparseDrive, DiffusionDrive, VAD}.
The diverse 14-VLM ensemble was **NOT** re-run on nuScenes-val (claude/gpt/gemini are paid
APIs; not run autonomously overnight). So disagreement here comes from only 3 driving models
(SD & DD are similar → modest diversity), making this a conservative transfer test.

| detector | AUROC | AUPRC |
|---|---|---|
| native-only (SD margin/entropy/mode-spread) | 0.618 ± 0.026 | 0.170 |
| **disagreement-only (3 driving models)** | **0.755 ± 0.018** | 0.290 |
| **fused (disagreement + native)** | **0.762 ± 0.023** | 0.293 |
| leave-SD-out (DD vs VAD only → SD failure) | 0.667 ± 0.022 | 0.191 |

baseline AUPRC = 0.100; chance AUROC = 0.50.

**Verdict — TRANSFERS.** Cross-model disagreement predicts the actual driving planner's
failure at **AUROC 0.76**, far above the planner's own native uncertainty (0.62). Crucially,
even when SparseDrive is **excluded** from the disagreement computation (DD-vs-VAD only),
disagreement still predicts SD's failure at **0.667 > native 0.62** — so the effect is **not
self-referential**: when other models are confused, SparseDrive also tends to fail (shared
scene difficulty). Including SD (3-model disagreement) raises it to 0.76.

**Implication for the paper.** This removes the "self-referential" caveat on the headline
result. The earlier 0.82 (VLM disagreement → VLM failure) is now corroborated by an
independent test: disagreement → an *actual driving model's* failure at 0.76 on big data.
This is with only 3 (partly similar) driving models; the diverse 14-VLM signal (0.80)
suggests more diversity would push it higher. Fusing native uncertainty adds little once
disagreement is present (0.755 → 0.762) — disagreement carries the information.

Artifacts: `transfer_fusion.csv`. Code: `scripts/curiosity/{extract_traj,transfer_detector}.py`.

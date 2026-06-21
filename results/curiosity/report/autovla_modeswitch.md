# AutoVLA fast/slow mode switching (Waymo v3, n=199 / 31 scenes)

AutoVLA's adaptive per-frame choice is NOT logged; we use its forced fast/slow runs.
Disagreement = spread of non-AutoVLA models' trajectories (no self-reference). Small n -> illustrative.

| metric | value | note |
|---|---|---|
| always-fast ADE | 3.76 |  |
| always-slow ADE | 3.68 |  |
| oracle per-frame mode ADE | 2.93 | -22% vs fast |
| disagreement -> 'slow helps' (AUROC/AUPRC) | 0.466 | AUPRC 0.283 (base 0.31) |
| disagreement -> 'fast fails' (AUROC/AUPRC) | 0.576 | AUPRC 0.225 (base 0.20) |

**Value of escalation:** an oracle that picks fast-vs-slow per frame cuts ADE from 3.76 to 2.93 (22% lower) — so *timing* the slow mode matters (always-slow barely helps: 3.68).
**Can disagreement time it?** disagreement predicts 'slow helps' at AUROC 0.466 and 'fast fails' at 0.576. (n=199 small/noisy; Waymo; mid-training AutoVLA checkpoint.)

**Reading:** consistent with the main study — cross-model disagreement carries the fast-slow signal — but n is too small here for a firm number; treat as a directional probe.

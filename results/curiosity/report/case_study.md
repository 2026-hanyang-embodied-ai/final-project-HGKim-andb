# Case study — our detector flags the hard scenes

Frames = SparseDrive **actual** top-10% failures that the learned transfer detector (disagreement+native, out-of-fold) scores high, one per scene; last frame = an easy (clear) contrast. All numbers are real (from saved CSVs / nuScenes GT).

| # | token | disagreement 1s (m) | detector P(fail) | actual ADE (m) | label |
|---|---|---|---|---|---|
| 1 | `bf8b54d2aa47416d…` | 0.94 | 1.00 | 4.64 | top-10% = FAILURE |
| 2 | `d62b0f9db06440c6…` | 4.28 | 1.00 | 6.06 | top-10% = FAILURE |
| 3 | `21bb21c84c3e4390…` | 4.16 | 1.00 | 5.26 | top-10% = FAILURE |
| 4 | `19ffc60b7e9f429c…` | 4.21 | 1.00 | 3.25 | top-10% = FAILURE |
| 5 | `a4fa9d49433944d8…` | 0.88 | 1.00 | 5.64 | top-10% = FAILURE |
| 6 | `269601cc1e4d45a3…` | 1.47 | 0.99 | 2.65 | top-10% = FAILURE |
| 7 | `15c59626df6b4f96…` | 3.39 | 0.99 | 2.20 | top-10% = FAILURE |
| + | `a572dd2e95e94e4d…` | 0.00 | 0.00 | 0.00 | nominal |

GIF 0.9 MB, 8 frames. Detector P(fail) is out-of-fold (5-fold scene CV). mp4=yes.

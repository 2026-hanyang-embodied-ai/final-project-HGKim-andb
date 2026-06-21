# Failure-detection demo video (front camera, time-ordered, 2 fps)

nuScenes-val scenes played as raw front-camera keyframes (2 Hz captured, shown at 2 fps). A red **DETECTOR FLAG** marks frames where OUR learned detector predicts failure (out-of-fold P(fail) >= 0.5) — the algorithm's call, not ground truth. First 6 are the hardest scenes (high error), last 2 are easy scenes for contrast (detector stays clear). Real frames only.

| scene | type | keyframes | mean ADE (m) | % flagged by detector |
|---|---|---|---|---|
| `6af9b75e439e4811…` | HARD | 40 | 2.62 | 45% |
| `7bd098ac88cb4221…` | HARD | 40 | 2.22 | 65% |
| `3dd2be428534403b…` | HARD | 40 | 2.12 | 62% |
| `7052d21b95fc4bae…` | HARD | 40 | 2.05 | 72% |
| `3363f396bb43405f…` | HARD | 40 | 1.97 | 35% |
| `f97bf749746c4c3a…` | HARD | 40 | 1.95 | 55% |
| `6a24a80e2ea3493c…` | easy | 41 | 1.07 | 7% |
| `5af9c7f124d84e7e…` | easy | 39 | 1.18 | 8% |

320 frames @ 2 fps. mp4 5.4 MB, gif 12.6 MB. Red border/dot = our detector predicted failure (out-of-fold); green = clear.

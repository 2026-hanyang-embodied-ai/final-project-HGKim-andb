# Senna transfer test — NOT PRODUCED

No results table (`senna_transfer.csv` intentionally not written — no fabricated numbers).

The experiment stopped at the Stage 0 feasibility gate: Senna outputs a **discrete
meta-action** (speed/path class), not a trajectory, so a "Senna ADE top-10% failure" label
is undefined from its native output (it would require the full Senna→E2E coupling). See
`senna_log.md` for details.

**Recommendation:** Senna stays Future Work; the SparseDrive transfer (AUROC 0.76) is the
submitted real-driving-model transfer result.

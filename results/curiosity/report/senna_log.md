# Bonus experiment — Senna transfer test: STOPPED at Stage 0 (feasibility gate)

Date: 2026-06-21. Status: **not attempted beyond Stage 0** (principled stop, per plan's
stop-criteria). No submission files were modified; no git commit.

## Stage 0 — feasibility gate
1. **Repo cloned OK:** `github.com/hustvl/Senna` → `/home/sp/Project/_bonus_senna/Senna`
   (code only; no weights downloaded). LLaVA-based (`llava`, `llava_next`).
2. **Checkpoint (domain OK):** Senna-VLM 7B (base `vicuna-7b-v1.5`, 6-view) on
   HuggingFace `rb93dett/Senna` — nuScenes domain, downloadable. ✔ (no domain-shift problem)
3. **BLOCKER 1 — output format mismatch (decisive):** Senna's output / evaluation is a
   **discrete meta-action**, not a trajectory. `eval_tools/senna_plan_cmd_eval_multi_img.py`
   scores classification accuracy over `SPEED ∈ {KEEP, ACCELERATE, DECELERATE, STOP}` ×
   `PATH ∈ {left, straight, right}` ("Planning Accuracy"). Senna emits no per-frame (x,y)
   trajectory, so **"Senna ADE → top-10% failure"** (the label Stage 1 requires, parallel to
   the SparseDrive 0.76 transfer) is **undefined from Senna's native output**. A trajectory
   would only come from coupling Senna's meta-action to a separate E2E planner (VAD), whose
   trajectory error would be the *planner's* failure, not Senna's — a different, much heavier
   pipeline and a muddier target.
4. **BLOCKER 2 — heavy infra:** running Senna needs the full LLaVA stack
   (deepspeed, bitsandbytes, accelerate, datasets, ...) + the 7B vicuna weights (~14 GB), and
   the documented nuScenes→Senna QA data conversion (`data_tools/senna_nusc_data_converter.py`)
   itself runs **LLaVA-v1.6-34B** to generate scene descriptions. Multi-hour, high-friction.

## Decision
**STOP.** The domain checkpoint exists, but Senna's native output (discrete meta-action) does
not yield a trajectory ADE, so the clean transfer test (disagreement → Senna's top-10% ADE
failure) cannot be defined without building the full Senna+VAD coupling — out of scope for a
timeboxed bonus. No numbers were produced (none fabricated).

## Recommendation
Keep Senna (and other nuScenes fast–slow systems: DriveVLM-Dual, FASIONAD, AdaDrive) as
**Future Work** in the report. The cloned repo is at `/home/sp/Project/_bonus_senna/Senna`
if pursued later. The submission's main transfer result stands on SparseDrive
(disagreement → driving-planner failure, AUROC 0.76), which already answers the core question
with a real driving model that *does* output trajectories.

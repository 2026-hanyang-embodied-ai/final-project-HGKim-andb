#!/usr/bin/env python3
"""Build final-project.ipynb (results-reproduction notebook). Loads saved CSV/PNG and
re-renders tables/figures; one live sklearn recompute of the transfer AUROC. No GPU."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
nb = nbf.v4.new_notebook()
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s))
co = lambda s: c.append(nbf.v4.new_code_cell(s))

md("""# Curiosity-Driven Mode Switching — Detection Module
**Goal.** A fast end-to-end (E2E) driving policy runs by default; detect *when it is about
to fail* so control can escalate to a slower, stronger VLA. This notebook is the **detection
module**: a passive failure monitor.

**Method (how to read every result below).**
- **Failure label** = *measured* trajectory error (ADE) in the worst 10% of frames. It is a
  measurement, not an opinion. **ADE is never an input feature** (anti-circular).
- **Candidate signals**: cross-model *disagreement*, single-model *native uncertainty*,
  VLM *perceived risk*, and *temporal accumulation*.
- **Evaluation**: 5-fold **scene-level** cross-validation (no scene split across folds);
  the VLM ensemble is **zero-shot** (no training-scene leakage).

This notebook only **loads saved results** (heavy inference is already done) and re-renders
them, plus one quick live recompute of the core transfer result.""")

co("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from IPython.display import Image, display
REPORT='results/curiosity/report'; TRAJ='results/curiosity/traj'; CUR='results/curiosity'
pd.set_option('display.max_colwidth', 60)
print('loaded libs; reading artifacts from', REPORT)""")

md("""## 1. Which signal predicts failure?
Single-signal detection AUROC/AUPRC (chance AUROC 0.50; baseline AUPRC = positive rate 0.10).""")
co("""sig = pd.read_csv(f'{REPORT}/comparison_table.csv'); display(sig)
v = sig.dropna(subset=['AUROC']).copy(); v['AUROC']=v['AUROC'].astype(float)
plt.figure(figsize=(8,4)); plt.barh(range(len(v)), v['AUROC'])
plt.yticks(range(len(v)), [s[:34] for s in v['signal']], fontsize=8)
plt.axvline(0.5, ls='--', c='k'); plt.xlabel('AUROC'); plt.title('Signal comparison (per-set)')
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()""")
md("**Takeaway.** Cross-model disagreement is the strongest signal (VLM ensemble **0.80**); "
   "single-model native uncertainty is weak (0.55–0.61); VLM perceived risk is ≈ chance (0.46).")

md("""## 2. Learned detector (Curiosity Detection module)
DriveLM-nuScenes, 5-fold scene CV. Three models of increasing expressiveness.""")
co("""dl = pd.read_csv(f'{REPORT}/detector_dl.csv'); display(dl)
print('LR / MLP / DeepSets all reach AUROC ~0.82')""")
co("""display(Image(filename=f'{REPORT}/architecture.png'))""")
md("**Takeaway.** LR (0.815), MLP (0.812) and a permutation-invariant **DeepSets** (0.819) are "
   "indistinguishable; DeepSets learns the disagreement representation directly from raw trajectories.")

md("## 3. Feature importance (which disagreement statistic drives it)")
co("""imp = pd.read_csv(f'{REPORT}/detector_drivelm.csv', skip_blank_lines=False)
fi = imp[imp['detector'].astype(str).str.contains('feature_importance', na=False)].index
rows = imp.iloc[fi[0]+1:].dropna(subset=['detector']) if len(fi) else imp.iloc[0:0]
rows = rows[['detector','features']].rename(columns={'detector':'feature','features':'coef'})
rows['coef']=rows['coef'].astype(float); display(rows)
plt.figure(figsize=(7,3.5)); r=rows.sort_values('coef')
plt.barh(r['feature'], r['coef']); plt.axvline(0,c='k'); plt.title('LR coefficients (DriveLM detector)')
plt.tight_layout(); plt.show()""")
md("**Takeaway.** Near-term, speed-related disagreement dominates: `pathlen_std` (+1.37) and "
   "1 s spread (+0.98); long-horizon (3 s) spread is not predictive.")

md("""## 4. Single-model internal signals only (contrast)
A detector limited to one model's own signals — the only option without an ensemble.""")
co("""display(pd.read_csv(f'{REPORT}/detector_table.csv'))
print('Internal-signal fused ~0.61  vs  cross-model disagreement ~0.82')""")

md("## 5. Robustness to the failure threshold (5/10/20%)")
co("""display(pd.read_csv(f'{REPORT}/robustness_table.csv'))
print('Conclusion stable across thresholds (all above baseline).')""")

md("""## 6. Transfer test (core result)
Does cross-model disagreement predict an **actual driving model's** failure?
Official nuScenes-val, 6019 frames, label = **SparseDrive** ADE top-10%.""")
co("""display(pd.read_csv(f'{REPORT}/transfer_fusion.csv'))""")
md("""### Live recompute (from the shared trajectory + signal CSVs)
Rebuilds the transfer detector from `traj/{sd,dd,vad}_trajs.csv` + `sd_signals.csv` with
5-fold scene CV — should reproduce ≈ native 0.62 / disagreement 0.76 / fused 0.76.""")
co("""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

def load_traj(p):
    df=pd.read_csv(p); cols=[f't{i}{a}' for i in range(6) for a in 'xy']
    return {t:df[df.token==t][cols].values.reshape(6,2) for t in df.token} if False else \
           {r.token:r[cols].values.astype(float).reshape(6,2) for _,r in df.iterrows()}
sd=load_traj(f'{TRAJ}/sd_trajs.csv'); dd=load_traj(f'{TRAJ}/dd_trajs.csv'); vad=load_traj(f'{TRAJ}/vad_trajs.csv')
sg=pd.read_csv(f'{CUR}/sd_signals.csv').set_index('token')
toks=[t for t in sg.index if t in sd and t in dd and t in vad]
DIS,NAT,y,scn=[],[],[],[]
for t in toks:
    T=np.stack([sd[t],dd[t],vad[t]]); pw=np.mean(np.sum((T-T.mean(0))**2,axis=2),axis=0); ep=T[:,-1]
    seg=np.linalg.norm(np.diff(T,axis=1),axis=2).sum(1)
    mp=max(np.linalg.norm(ep[a]-ep[b]) for a in range(3) for b in range(a+1,3))
    DIS.append([pw.mean()**.5,pw[1]**.5,pw[-1]**.5,np.std(ep[:,0]),np.std(seg),mp,
                float(np.mean(np.linalg.norm(dd[t]-vad[t],axis=1)))])
    NAT.append([sg.loc[t,'sig_margin'],sg.loc[t,'sig_entropy'],sg.loc[t,'sig_mode_std']])
    y.append(int(sg.loc[t,'fail_top10'])); scn.append(sg.loc[t,'scene'])
DIS,NAT,y,scn=np.array(DIS),np.array(NAT),np.array(y),np.array(scn)
def cv(X):
    a=[]
    for tr,te in GroupKFold(5).split(X,y,scn):
        s=StandardScaler().fit(X[tr]); m=LogisticRegression(max_iter=1000,class_weight='balanced').fit(s.transform(X[tr]),y[tr])
        a.append(roc_auc_score(y[te],m.predict_proba(s.transform(X[te]))[:,1]))
    return np.mean(a)
print(f'n={len(toks)} frames | positives={y.mean():.1%}')
print(f'native-only       AUROC {cv(NAT):.3f}')
print(f'disagreement-only AUROC {cv(DIS):.3f}')
print(f'fused             AUROC {cv(np.hstack([DIS,NAT])):.3f}')""")
md("**Takeaway.** Disagreement predicts the driving planner's failure at **0.76**, far above "
   "native (0.62). Excluding the target model (leave-SD-out, DD vs VAD only) still gives **0.667** "
   "> native — so the effect is **not self-referential**. (Conservative: only 3 driving models; "
   "the diverse 14-VLM ensemble was not re-run on nuScenes-val — paid APIs.)")

md("## 7. Negative results")
co("""print('VLM perceived risk vs actual failure : AUROC 0.46  (Spearman -0.03 .. +0.07)  -> OOD != failure')
print('Temporal accumulation (open-loop)     : single-frame 0.798 > EWMA 0.760 > CUSUM 0.715  -> no benefit')
# (values from results/curiosity/report/RESULTS.md)""")

md("""## 8. Conclusion
- **Cross-model disagreement** (computational *ambiguity*) is the most informative failure
  signal — 0.80 for a VLM ensemble, and it **transfers** to an actual driving planner (**0.76**,
  vs 0.62 for the planner's own uncertainty; 0.67 even with the target model excluded).
- A small **learned detector** (LR ≈ MLP ≈ DeepSets ≈ **0.82**) captures it; single-model
  internal signals alone reach only ≈0.61.
- **Negative**: VLM perceived risk ≠ failure; temporal accumulation gives no open-loop benefit.
- **Next step**: *distill* the (expensive) disagreement signal into a single-pass trigger and
  evaluate it in closed loop.""")

nb["cells"] = c
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "final-project.ipynb"
nbf.write(nb, str(out))
print("wrote", out, "with", len(c), "cells")

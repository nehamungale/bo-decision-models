"""tev1 'sample' mode on 20 dev seeds (seeds 10-19 added post hoc, flagged) vs static, rule, learned, random and the state-blind fixed-mixture control.
Reports both summaries because they can disagree: change in mean AUC (dominated by high-AUC datasets) and mean of per-dataset ratios (dominated by low-AUC ones).
Run: .venv/bin/python -m acq.analyse_phaseC20"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
H = os.path.dirname(os.path.abspath(__file__))
res = {}
for d in ("phaseA", "phaseB", "phaseC"):
    for f in glob.glob(os.path.join(H, "results", d, "*.pkl")):
        if d == "phaseC" and "dev__sample" not in f and "ctrl__" not in f: continue
        r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
DS = DEV[:8]; S = range(20)
M = lambda c: np.array([np.mean([res[(c, d, s)]["auc"] for s in S]) for d in DS])
cols = ["static_LogEI", "static_LogPI", "static_TS", "random", "rule", "learned_L2", "fixed_mixture_tev1", "typed_tev1_sample"]
print("20 seeds; mean AUC by dataset")
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_', '').replace('typed_tev1_', 'tev1_')[:11]:>12s}" for c in cols))
for i, d in enumerate(DS): print(f"{d:22s}" + "".join(f"{M(c)[i]:12.3f}" for c in cols))
print(f"{'MEAN':22s}" + "".join(f"{M(c).mean():12.4f}" for c in cols))
rng = np.random.default_rng(1)
def cmp(c, rc):
    a, b = M(c), M(rc)
    D = np.array([[res[(c, d, s)]["auc"] - res[(rc, d, s)]["auc"] for s in S] for d in DS])
    bs = [D[np.ix_(rng.integers(0, 8, 8), rng.integers(0, 20, 20))].mean() / b.mean() for _ in range(4000)]
    return (a.mean() - b.mean()) / b.mean(), ((a - b) / b).mean(), int((a < b).sum()), wilcoxon(a - b).pvalue, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print("\ntev1_sample (negative = tev1 better). change in mean AUC | mean of per-dataset ratios | better on k/8 | Wilcoxon p | 95% bootstrap CI of change in mean AUC")
for rc in ("static_TS", "static_LogEI", "rule", "random", "learned_L2", "fixed_mixture_tev1"):
    o = cmp("typed_tev1_sample", rc)
    print(f"  vs {rc:20s} {o[0]:+7.1%} | {o[1]:+7.1%} | {o[2]}/8 | p={o[3]:.3f} | [{o[4]:+.1%}, {o[5]:+.1%}]")

"""Clean check: tev1 'sample' on the FRESH seeds 10-19 only (seeds 0-9 were used to choose which mode to extend). Run: .venv/bin/python -m acq.analyse_phaseC_fresh"""
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
DS = DEV[:8]; S = range(10, 20)
M = lambda c: np.array([np.mean([res[(c, d, s)]["auc"] for s in S]) for d in DS])
cols = ["static_LogEI", "static_LogPI", "static_TS", "random", "rule", "learned_L2", "fixed_mixture_tev1", "typed_tev1_sample"]
print("FRESH seeds 10-19 only; mean AUC by dataset")
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_', '').replace('typed_tev1_', 'tev1_')[:11]:>12s}" for c in cols))
for i, d in enumerate(DS): print(f"{d:22s}" + "".join(f"{M(c)[i]:12.3f}" for c in cols))
print(f"{'MEAN':22s}" + "".join(f"{M(c).mean():12.4f}" for c in cols))
rng = np.random.default_rng(2)
print("\ntev1_sample vs each (negative = tev1 better): change in mean AUC | mean of ratios | better on k/8 | Wilcoxon p | 95% CI")
for rc in ("static_TS", "static_LogEI", "rule", "random", "learned_L2", "fixed_mixture_tev1"):
    a, b = M("typed_tev1_sample"), M(rc)
    D = np.array([[res[("typed_tev1_sample", d, s)]["auc"] - res[(rc, d, s)]["auc"] for s in S] for d in DS])
    bs = [D[np.ix_(rng.integers(0, 8, 8), rng.integers(0, 10, 10))].mean() / b.mean() for _ in range(4000)]
    print(f"  vs {rc:20s} {(a.mean() - b.mean()) / b.mean():+7.1%} | {((a - b) / b).mean():+7.1%} | {int((a < b).sum())}/8 | p={wilcoxon(a - b).pvalue:.3f} | [{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]")

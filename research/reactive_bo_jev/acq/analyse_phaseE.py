"""Amendment 4 analysis (pre-registered in PLAN_acquisition_choice.md): bandit_ts, sched_EI_PI, rule_dwell<D> each vs TS and vs LogEI,
paired Wilcoxon across the 8 per-dataset mean AUCs, Holm over the six tests. Context rows: rule, hedge, random.
Run: .venv/bin/python -m acq.analyse_phaseE"""
import sys, os, glob, pickle, json
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV

H = os.path.dirname(os.path.abspath(__file__))
res = {}
for d in ("phaseA", "phaseB", "phaseE"):
    for f in glob.glob(os.path.join(H, "results", d, "*.pkl")):
        r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
seeds = range(20); DS = DEV[:8]
D = json.load(open(os.path.join(H, "results", "phaseE_dwell.json")))
print("dwell tuning scores (seeds 100-104):", D["scores"], "-> D =", D["best_D"])
NEW = ["bandit_ts", "sched_EI_PI"] + ([f"rule_dwell{D['best_D']}"] if D["best_D"] != 1 else [])
A = lambda c, d: np.array([res[(c, d, s)]["auc"] for s in seeds])
cols = ["static_LogEI", "static_TS", "random", "hedge", "rule"] + NEW
print(f"\n{'dataset':22s}" + "".join(f"{c.replace('static_',''):>13s}" for c in cols))
for d in DS: print(f"{d:22s}" + "".join(f"{A(c, d).mean():13.3f}" for c in cols))
print(f"{'MEAN':22s}" + "".join(f"{np.mean([A(c, d).mean() for d in DS]):13.4f}" for c in cols))

def holm(p):
    o = np.argsort(p); h = np.empty(len(p))
    for r, i in enumerate(o): h[i] = min(1.0, (len(p) - r) * p[i])
    return np.maximum.accumulate(h[o])[np.argsort(o)]

tests = []
for rn, rc in (("TS", "static_TS"), ("LogEI", "static_LogEI")):
    for c in NEW:
        a = np.array([A(c, d).mean() for d in DS]); b = np.array([A(rc, d).mean() for d in DS])
        tests.append((c, rn, ((a - b) / b).mean(), int((a < b).sum()), wilcoxon(a - b).pvalue if (a != b).any() else 1.0))
hp = holm(np.array([t[4] for t in tests]))
print("\n== PRIMARY (Holm over 6): new selector vs comparator ==")
for (c, rn, dm, w, p), h in zip(tests, hp):
    print(f"  {c:14s} vs {rn:6s} mean relative AUC change {dm:+7.1%} | better on {w}/8 datasets | Wilcoxon p={p:.3f}, Holm p={h:.3f}")

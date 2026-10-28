"""Phase B analysis (pre-registered in PLAN_acquisition_choice.md, Amendment 1).
Primary family: hedge, rule, learned_L1, learned_L2, each vs TS (best pooled static): paired Wilcoxon across the 8 per-dataset
mean AUCs, Holm-corrected over the four. Secondary: vs LogEI and vs random-AF; per-dataset sign counts.
Run: .venv/bin/python -m acq.analyse_phaseB"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS

H = os.path.dirname(os.path.abspath(__file__))
res = {}
for d in ("phaseA", "phaseB"):
    for f in glob.glob(os.path.join(H, "results", d, "*.pkl")):
        r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
seeds = range(20); DS = DEV[:8]
SEL = ["hedge", "rule", "learned_L1", "learned_L2"]
A = lambda c, d: np.array([res[(c, d, s)]["auc"] for s in seeds])
ref = {"TS": "static_TS", "LogEI": "static_LogEI", "random": "random"}

print("== Mean AUC of normalised regret by dataset (lower is better) ==")
cols = ["static_LogEI", "static_TS", "random"] + SEL + ["oracle_onestep"]
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_',''):>12s}" for c in cols))
for d in DS: print(f"{d:22s}" + "".join(f"{A(c, d).mean():12.3f}" for c in cols))
print(f"{'MEAN over datasets':22s}" + "".join(f"{np.mean([A(c, d).mean() for d in DS]):12.4f}" for c in cols))

def holm(p):
    o = np.argsort(p); h = np.empty(len(p))
    for r, i in enumerate(o): h[i] = min(1.0, (len(p) - r) * p[i])
    return np.maximum.accumulate(h[o])[np.argsort(o)]

for rname, rc in ref.items():
    print(f"\n== {'PRIMARY: ' if rname == 'TS' else 'secondary: '}each selector vs {rname} ==")
    ps, rows = [], []
    for c in SEL:
        a = np.array([A(c, d).mean() for d in DS]); b = np.array([A(rc, d).mean() for d in DS])
        diff = (a - b) / b
        p = wilcoxon(a - b).pvalue if (a != b).any() else 1.0
        seed_p = [wilcoxon(A(c, d) - A(rc, d)).pvalue for d in DS]
        ps.append(p); rows.append((c, diff.mean(), int((a < b).sum()), int(sum(sp < .05 and A(c, d).mean() < A(rc, d).mean() for sp, d in zip(seed_p, DS)))))
    hp = holm(np.array(ps))
    for (c, dm, wins, sig), p, h in zip(rows, ps, hp):
        print(f"  {c:12s} mean relative AUC change {dm:+7.1%} | better on {wins}/8 datasets ({sig} individually significant) | Wilcoxon p={p:.3f}, Holm p={h:.3f}")

print("\n== AF usage shares by selector (pooled) ==")
for c in SEL:
    cnt = np.zeros(6)
    for d in DS:
        for s in seeds:
            for st in res[(c, d, s)]["log"]: cnt[st["af"]] += 1
    print(f"  {c:12s}", {a: f"{x/cnt.sum():.0%}" for a, x in zip(AFS, cnt)})

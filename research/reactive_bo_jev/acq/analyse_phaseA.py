"""Phase A headroom gate. Reads acq/results/phaseA. Gate thresholds as proposed in PLAN_acquisition_choice.md:
G1 one-step oracle beats the best static AF by >= 20 % relative AUC regret on >= 6/8 dev datasets;
G2 static AFs differ (Friedman across seeds, Holm over datasets, alpha .05) on >= 5/8 datasets.
Run: .venv/bin/python -m acq.analyse_phaseA"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phaseA")
STATICS = [f"static_{a}" for a in AFS]
res = {}
for f in glob.glob(os.path.join(R, "*.pkl")):
    r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
seeds = sorted({k[2] for k in res})
print(f"{len(res)} campaigns, {len(seeds)} seeds")


def auc(c, d): return np.array([res[(c, d, s)]["auc"] for s in seeds])
def fin(c, d): return np.array([res[(c, d, s)]["final"] for s in seeds])


print("\n== Mean AUC of normalised regret (lower is better), 50 steps after 15 initial points ==")
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_',''):>9s}" for c in STATICS) + f"{'random':>9s}{'oracle':>9s}")
g1, g2, rows = 0, 0, []
pvals = []
for d in DEV[:8]:
    m = [auc(c, d).mean() for c in STATICS]
    print(f"{d:22s}" + "".join(f"{x:9.3f}" for x in m) + f"{auc('random', d).mean():9.3f}{auc('oracle_onestep', d).mean():9.3f}")
    pvals.append(friedmanchisquare(*[auc(c, d) for c in STATICS]).pvalue)
order = np.argsort(pvals); holm = np.empty(len(pvals))
for rank, i in enumerate(order): holm[i] = min(1.0, (len(pvals) - rank) * pvals[i])
holm = np.maximum.accumulate(holm[order])[np.argsort(order)]

print("\n== Gate quantities per dataset ==")
print(f"{'dataset':22s}{'best static':>13s}{'AUC':>7s}{'oracle gain':>13s}{'Friedman p (Holm)':>19s}{'rand vs best':>14s}")
for i, d in enumerate(DEV[:8]):
    m = {c: auc(c, d).mean() for c in STATICS}; bs = min(m, key=m.get)
    gain = (m[bs] - auc('oracle_onestep', d).mean()) / m[bs]
    rv = (auc('random', d).mean() - m[bs]) / m[bs]
    g1 += gain >= 0.20; g2 += holm[i] < 0.05
    print(f"{d:22s}{bs.replace('static_',''):>13s}{m[bs]:7.3f}{gain:13.1%}{holm[i]:19.4f}{rv:14.1%}")
print(f"\nG1 (oracle gain >= 20 % on >= 6/8): {g1}/8 -> {'PASS' if g1 >= 6 else 'FAIL'}")
print(f"G2 (statics differ on >= 5/8): {g2}/8 -> {'PASS' if g2 >= 5 else 'FAIL'}")

print("\n== Proposal agreement (static LogEI trajectories) ==")
dist = [len(set(st["proposals"])) for d in DEV[:8] for s in seeds for st in res[("static_LogEI", d, s)]["log"]]
print("mean distinct proposals per step:", round(float(np.mean(dist)), 2), "of 6;",
      "steps with all six agreeing:", f"{np.mean([x == 1 for x in dist]):.1%}")

print("\n== Which AF is best by dataset (pooled over seeds), and best-AF consistency across datasets ==")
best_by_ds = {d: min(STATICS, key=lambda c: auc(c, d).mean()).replace('static_', '') for d in DEV[:8]}
print(best_by_ds)
print("\n== Final regret (after 50 steps), mean over seeds ==")
for d in DEV[:8]:
    print(f"{d:22s}" + "".join(f"{fin(c, d).mean():9.3f}" for c in STATICS) + f"{fin('random', d).mean():9.3f}{fin('oracle_onestep', d).mean():9.3f}")

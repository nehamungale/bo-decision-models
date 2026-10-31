"""T1 analysis (rule declared in sp_tune.py): per configuration, ratio of mean AUC SP/W1 within each source (TabRepo, LCBench) and per TabRepo family; selection and stop rule."""
import glob, os, pickle, json
import numpy as np
from meta.sp_tune import GRID, OUT, H
R = {}
for f in glob.glob(os.path.join(OUT, "*.pkl")):
    r = pickle.load(open(f, "rb")); R.setdefault(r["landscape"], []).append(r)
T = json.load(open(os.path.join(H, "results", "sp_tuning_set.json")))
def mean_auc(ls, arm): return np.mean([np.mean([u["auc"][arm] for u in R[l]]) for l in ls])
fam = lambda l: l.split("/")[0].split("_", 1)[1]
out = []
def P(*a): s = " ".join(str(x) for x in a); out.append(s); print(s)
P(f"tuning landscapes: {len(T['tabrepo'])} TabRepo, {len(T['lcbench'])} LCBench; units with results {sum(len(v) for v in R.values())}")
P(f"W1 mean AUC: TabRepo {mean_auc(T['tabrepo'], 'W1'):.4f}, LCBench {mean_auc(T['lcbench'], 'W1'):.4f} | SP-uniform vs W1: TabRepo {mean_auc(T['tabrepo'],'uniform')/mean_auc(T['tabrepo'],'W1')-1:+.1%}, LCBench {mean_auc(T['lcbench'],'uniform')/mean_auc(T['lcbench'],'W1')-1:+.1%}")
P(f"\n{'kappa':>6s}{'lambda':>8s}{'TabRepo':>10s}{'LCBench':>10s}{'mean':>8s}{'eff.src':>9s}{'volat.':>8s}   (SP vs W1, ratio of means; negative = SP better)")
rows = []
for c in GRID:
    rt = mean_auc(T["tabrepo"], c) / mean_auc(T["tabrepo"], "W1") - 1; rl = mean_auc(T["lcbench"], c) / mean_auc(T["lcbench"], "W1") - 1
    eff = np.mean([u["eff"][c] for l in R for u in R[l]]); vol = np.mean([u["vol"][c] for l in R for u in R[l]])
    rows.append((c, rt, rl, (rt + rl) / 2)); P(f"{c[0]:6.0f}{c[1]:8.2f}{rt:+10.1%}{rl:+10.1%}{(rt+rl)/2:+8.1%}{eff:9.1f}{vol:8.3f}")
best = min(rows, key=lambda r: r[3]); P(f"\nSelected configuration (lowest mean of the two source ratios): kappa {best[0][0]:.0f}, lambda {best[0][1]}: TabRepo {best[1]:+.1%}, LCBench {best[2]:+.1%}")
P("\nPer TabRepo family at the selected configuration (SP vs W1):")
for f in sorted({fam(l) for l in T["tabrepo"]}):
    ls = [l for l in T["tabrepo"] if fam(l) == f]; P(f"   {f:14s} {mean_auc(ls, best[0])/mean_auc(ls, 'W1')-1:+.1%}  (W1 {mean_auc(ls,'W1'):.4f})")
stop = not any(rt <= -0.02 and rl <= -0.02 for _, rt, rl, _ in rows)
P("\nSTOP RULE: " + ("no configuration is at least 2 % below W1 on both sources -> STOP before the test run" if stop else "at least one configuration is >= 2 % below W1 on both sources -> proceed to T2 with the selected configuration"))
json.dump(dict(kappa=best[0][0], lam=best[0][1], stop=bool(stop)), open(os.path.join(H, "results", "sp_selected.json"), "w"))
open(os.path.join(H, "results", "sp_tune.txt"), "w").write("\n".join(out) + "\n")

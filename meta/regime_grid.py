"""Descriptive regime grid: how much does the choice of AF matter, by regime? Uses existing runs only (no new compute):
acq/results/phaseA (8 LCBench dev datasets x 20 seeds) and meta/results/stage2 (30 landscapes, behaviour = fixed AF or random AF).
Strata: family, dimension (d <= 6, 7-10, > 10), budget (< 50 vs 50). Noise cannot be stratified: every one of these runs is noise-free.
Statistics per stratum (no gate): mean AUC of LogEI, TS, random AF; relative gap random vs LogEI; relative spread of the five complete static AFs
(max-min over LogEI, LogPI, UCB, TS, PosMean, relative to LogEI). In-sample, descriptive. Run: .venv/bin/python -m meta.regime_grid"""
import sys, os, glob, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meta.landscapes import inventory, budget

H = os.path.dirname(os.path.abspath(__file__)); AF5 = ["LogEI", "LogPI", "UCB", "TS", "PosMean"]
inv = inventory().set_index("landscape")
rows = []
for f in glob.glob(os.path.join(H, "results", "stage2", "*.pkl")):
    r = pickle.load(open(f, "rb")); rows.append(dict(src="stage2", landscape=r["landscape"], seed=r["seed"], pol=r["behaviour"], auc=r["auc"]))
sys.path.insert(0, os.path.dirname(H))
for f in glob.glob(os.path.join(os.path.dirname(H), "acq", "results", "phaseA", "*.pkl")):
    r = pickle.load(open(f, "rb")); c = r["cond"]
    if c.startswith("static_"): pol = c[7:]
    elif c == "random": pol = "random"
    else: continue
    rows.append(dict(src="phaseA", landscape="lcbench/" + r["dataset"], seed=r["seed"], pol=pol, auc=r["auc"]))
d = pd.DataFrame(rows)
# per landscape x policy mean over seeds; use only landscapes with every policy present
pl = d.groupby(["landscape", "pol"]).auc.mean().unstack()
pl = pl[AF5 + ["random"]].dropna()
meta_ = []
for l in pl.index:
    key = l if l in inv.index else None
    if key is None: continue
    meta_.append(dict(landscape=l, family=inv.loc[l, "family"], d=int(inv.loc[l, "d"]), n=int(inv.loc[l, "n"]), budget=min(50, max(10, int(0.25 * inv.loc[l, "n"])))))
m = pd.DataFrame(meta_).set_index("landscape"); pl = pl.loc[m.index]
m["dim"] = pd.cut(m.d, [0, 6, 10, 10**6], labels=["d<=6", "d7-10", "d>10"]); m["bud"] = np.where(m.budget < 50, "budget<50", "budget=50")
pl["spread"] = pl[AF5].max(1) - pl[AF5].min(1)
# best fixed AF chosen leave-one-landscape-out vs random AF (winner's-curse-free comparison)
loo = []
for l in pl.index:
    others = pl.drop(l); best = others[AF5].mean().idxmin(); loo.append(pl.loc[l, best])
pl["loo_best"] = loo
out = [f"{len(pl)} landscapes with all of {AF5 + ['random']} (phaseA: 8 LCBench, stage2: rest). All noise-free. In-sample, descriptive; no gate.\n"]
def show(title, g):
    out.append(f"== {title} ==")
    out.append(f"{'stratum':16s}{'n':>4s}{'LogEI':>8s}{'TS':>8s}{'random':>8s}{'LOO-best':>9s}{'rand/LogEI':>11s}{'LOObest/rand':>13s}{'spread(abs)':>12s}{'spread/LogEI':>13s}")
    for k, idx in g:
        x = pl.loc[idx]
        out.append(f"{str(k):16s}{len(x):4d}{x.LogEI.mean():8.3f}{x.TS.mean():8.3f}{x.random.mean():8.3f}{x.loo_best.mean():9.3f}{x.random.mean()/x.LogEI.mean()-1:+11.1%}{x.loo_best.mean()/x.random.mean()-1:+13.1%}{x.spread.mean():12.3f}{x.spread.mean()/x.LogEI.mean():13.1%}")
    out.append("")
show("all", [("all", pl.index)])
show("family", [(k, v.index) for k, v in m.groupby("family")])
show("dimension", [(k, v.index) for k, v in m.groupby("dim", observed=True)])
show("budget", [(k, v.index) for k, v in m.groupby("bud")])
out.append("All columns are ratios of stratum means (mean AUC), not means of per-landscape ratios, which blow up where AUC is near zero. rand/LogEI: random AF vs LogEI (positive = random worse). LOO-best: AUC of the fixed AF chosen on the other landscapes; LOObest/rand: relative to random AF (negative = best fixed better). spread: worst minus best of the five static AFs, per landscape, averaged.")
txt = "\n".join(out); print(txt); open(os.path.join(H, "results", "regime_grid.txt"), "w").write(txt + "\n")

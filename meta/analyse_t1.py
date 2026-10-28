import glob, os, pickle
import numpy as np
from scipy.stats import wilcoxon
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "t1")
R = {}
for f in glob.glob(os.path.join(H, "*.pkl")):
    r = pickle.load(open(f, "rb")); R.setdefault((r["landscape"], r["arm"]), []).append(r)
Ls = sorted({k[0] for k in R})
fam = lambda l: l.split("/")[0].split("_")[0]
print(f"{'landscape':44s} {'random':>8s} {'W1b':>8s} {'W1a':>8s} {'W0':>8s} | start regret rand/W1b | n_train")
rel = {a: {} for a in ("W1b", "W1a", "W0")}; tab = {}
for l in Ls:
    m = {a: np.mean([r["auc"] for r in R[(l, a)]]) for a in ("random", "W1b", "W1a", "W0")}
    tab[l] = m
    for a in rel: rel[a][l] = (m["random"] - m[a]) / m["random"] if m["random"] > 0 else 0.0
    sr = np.mean([r["start_regret"] for r in R[(l, "random")]]); sw = R[(l, "W1b")][0]["start_regret"]
    print(f"{l:44s} {m['random']:8.4f} {m['W1b']:8.4f} {m['W1a']:8.4f} {m['W0']:8.4f} | {sr:.3f} / {sw:.3f} | {R[(l,'W1b')][0]['n_train_tasks']}")
# determinism of the portfolio arms across the 3 seeds
dev = max(np.ptp([r["auc"] for r in R[(l, "W1b")]]) for l in Ls); print(f"\nmax AUC range across 3 seeds for W1b: {dev:.4f}")
print("\nrelative AUC-regret reduction vs random start (positive = better); mean of per-landscape ratios | change in mean AUC | wins | Wilcoxon one-sided p")
for a in rel:
    v = np.array([rel[a][l] for l in Ls]); mr = np.mean([tab[l]["random"] for l in Ls]); ma = np.mean([tab[l][a] for l in Ls])
    p = wilcoxon(v, alternative="greater").pvalue
    byf = {f_: round(float(np.mean([rel[a][l] for l in Ls if fam(l) == f_])), 3) for f_ in sorted({fam(l) for l in Ls})}
    print(f"{a:4s} ratios {v.mean():+.3f} | mean AUC {mr:.4f} -> {ma:.4f} ({(mr-ma)/mr:+.3f}) | wins {int((v>0).sum())}/{len(v)} | p={p:.4f} | by family {byf}")
v = rel["W1b"]; fams = {f_: np.mean([v[l] for l in Ls if fam(l) == f_]) for f_ in {fam(l) for l in Ls}}
print("\nT1 stop rule (primary arm W1b): close W if gain < 3% on all families ->", "CLOSE" if all(x < 0.03 for x in fams.values()) else "CONTINUE", fams)
print("Preview of TH-W (>=10% on >=2 families, p<0.05):", "on track" if sum(x >= 0.10 for x in fams.values()) >= 2 else "not on track at 10-landscape scale")

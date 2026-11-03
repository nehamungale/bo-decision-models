import glob, os, pickle
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w150")
A = {}; F = {}; skipped = set()
for f in glob.glob(os.path.join(H, "*.pkl")):
    r = pickle.load(open(f, "rb"))
    if r["skipped"]: skipped.add(r["landscape"]); continue
    A.setdefault((r["landscape"], r["arm"], r["noise"]), []).append(r["auc"])
    if r["arm"] == "W1" and r["noise"] == 0.0: F[r["landscape"]] = r
Ls = sorted({k[0] for k in A if k[1] == "W1"}); fam = lambda l: "lcbench" if l.startswith("lcbench") else l.split("/")[0]
print(f"{len(Ls)} landscapes with W1 (skipped for lack of training tasks: {len(skipped)})")
for n in (0.0, 0.05):
    w = np.array([np.mean(A[(l, "W1", n)]) for l in Ls]); r = np.array([np.mean(A[(l, "random", n)]) for l in Ls])
    worse = w > r; ex = np.maximum(w - r, 0)
    print(f"\n=== noise {n:.0%} ===  always-W1 {w.mean():.4f} | random {r.mean():.4f} | hindsight best-of {np.minimum(w, r).mean():.4f} | fall-back headroom {(w.mean()-np.minimum(w,r).mean())/w.mean():.1%} of W1's AUC")
    print(f"W1 worse than random on {worse.sum()}/{len(Ls)} ({worse.mean():.1%}); excess regret on those {ex.sum()/w.sum():.1%} of total W1 AUC; mean-of-ratios gain {np.mean((r-w)/r):+.3f}; Wilcoxon p={wilcoxon(r-w, alternative='greater').pvalue:.2e}")
    for f in sorted({fam(l) for l in Ls}):
        i = [k for k, l in enumerate(Ls) if fam(l) == f]
        print(f"   {f:24s} n={len(i):3d} W1 {w[i].mean():.4f} random {r[i].mean():.4f} worse on {int(worse[i].sum())} hindsight {np.minimum(w[i], r[i]).mean():.4f}")
    if n == 0.0:
        lab = worse.astype(int); print("failures:", [Ls[k] for k in np.where(worse)[0]])
        feats = ["spearman_src", "r2_src", "obs_cv", "obs_skew", "obs_max_z", "best_is_src_top", "loo_rmse", "ls_mean", "ls_min", "gp_noise"]
        if lab.sum() >= 5:
            print("single-feature AUROC for predicting 'W1 worse than random' (0.5 = no signal; sign-free = max(a, 1-a)):")
            for ft in feats:
                x = np.array([F[l].get(ft, np.nan) for l in Ls], float); ok = ~np.isnan(x)
                if ok.sum() > 20 and 0 < lab[ok].sum() < ok.sum():
                    a = roc_auc_score(lab[ok], x[ok]); print(f"   {ft:16s} AUROC {a:.2f}  (n={ok.sum()}, failures {lab[ok].sum()})")

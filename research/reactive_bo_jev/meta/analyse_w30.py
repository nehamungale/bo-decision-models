import glob, os, pickle
import numpy as np
from scipy.stats import wilcoxon, spearmanr
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w30")
R = {}; skipped = set()
for f in glob.glob(os.path.join(H, "*.pkl")):
    r = pickle.load(open(f, "rb"))
    arm = "W0" if r["arm"].startswith("W0_") else r["arm"]
    if r["skipped"]:
        if arm == "W1": skipped.add(r["landscape"])
        continue
    R.setdefault((r["landscape"], arm, r["noise"]), []).append(r)
Ls = sorted({k[0] for k in R}); fam = lambda l: l.split("/")[0].split("_")[0]
print("landscapes with W1 skipped (fewer than 8 training tasks):", sorted(skipped))
Lw = [l for l in Ls if l not in skipped]
print(f"{len(Lw)} landscapes used for W: ", {f: sum(fam(l) == f for l in Lw) for f in ("lcbench", "hpob", "tabrepo")})
auc = lambda l, a, n: np.mean([r["auc"] for r in R[(l, a, n)]])
def rel(l, a, n): m = auc(l, "random", n); return (m - auc(l, a, n)) / m if m > 0 else 0.0
for n in (0.0, 0.05):
    print(f"\n=== observation noise {n:.0%} of span ===")
    for a in ("W1", "FPS", "W0", "W2x"):
        Lx = [l for l in Lw if (l, a, n) in R]
        if len(Lx) < 3: continue
        v = np.array([rel(l, a, n) for l in Lx]); mr = np.mean([auc(l, "random", n) for l in Lx]); ma = np.mean([auc(l, a, n) for l in Lx])
        p = wilcoxon(v, alternative="greater").pvalue if np.any(v != 0) else 1.0
        byf = {f: (round(float(np.mean([rel(l, a, n) for l in Lx if fam(l) == f])), 3), sum(fam(l) == f for l in Lx)) for f in ("lcbench", "hpob", "tabrepo") if any(fam(l) == f for l in Lx)}
        print(f"{a:4s} n={len(Lx):2d} mean-of-ratios {v.mean():+.3f} | mean AUC {mr:.4f} -> {ma:.4f} ({(mr-ma)/mr:+.3f}) | wins {int((v>0).sum())}/{len(v)} | p={p:.5f} | by family (ratio, n) {byf}")
    v = {l: rel(l, "W1", n) for l in Lw}; fams = {f: np.mean([v[l] for l in Lw if fam(l) == f]) for f in ("lcbench", "hpob", "tabrepo")}
    p = wilcoxon(list(v.values()), alternative="greater").pvalue
    ok = sum(x >= 0.10 for x in fams.values()) >= 2 and p < 0.05
    print(f"TH-W (primary W1 vs random start; >=10% on >=2 families and Wilcoxon p<0.05): {'PASS' if ok else 'FAIL'} | families {({k: round(float(x),3) for k,x in fams.items()})} p={p:.5f}")
    if n == 0.0:
        w = [l for l in Lw if (l, "W1", n) in R]
        best = np.mean([min(auc(l, "W1", n), auc(l, "random", n)) for l in w]); alw = np.mean([auc(l, "W1", n) for l in w]); rnd = np.mean([auc(l, "random", n) for l in w])
        print(f"fall-back headroom (hindsight best of W1/random per landscape): always-W1 {alw:.4f}, hindsight {best:.4f}, random {rnd:.4f}; landscapes where W1 worse than random: {[l for l in w if auc(l,'W1',n) > auc(l,'random',n)]}")
tr = [(l, R[(l, "W1", 0.0)][0]["trust_spearman"], rel(l, "W1", 0.0)) for l in Lw if fam(l) != "hpob" and (l, "W1", 0.0) in R]
tr = [t for t in tr if not np.isnan(t[1])]
if len(tr) > 4:
    print(f"\ntrust statistic (Spearman between portfolio's observed values and source mean score) vs W1 relative gain, {len(tr)} exact-grid landscapes: rho={spearmanr([t[1] for t in tr],[t[2] for t in tr])[0]:.3f}")
    lo = sorted(tr, key=lambda t: t[1])[:5]; print("lowest-trust landscapes:", [(t[0], round(t[1], 2), round(t[2], 2)) for t in lo])

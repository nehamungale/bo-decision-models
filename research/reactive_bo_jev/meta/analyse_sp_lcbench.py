"""Gates of PLAN_source_prior_lcbench.md, written before any test result existed. Ratio of means over the 24 landscapes (seed-averaged AUC per landscape), paired Wilcoxon, Holm over TH-L1..L3.
Run: python -m meta.analyse_sp_lcbench  -> meta/results/sp_lcbench.txt"""
import os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "results")
D = {}
for f in glob.glob(os.path.join(R, "sp_lcbench", "*.pkl")):
    r = pickle.load(open(f, "rb")); D.setdefault(r["landscape"], []).append(r)
L = sorted(D); assert all(len(D[l]) == 4 for l in L), "incomplete run"
arms = list(D[L[0]][0]["auc"])
A = {a: np.array([np.mean([u["auc"][a] for u in D[l]]) for l in L]) for a in arms}
out = []
def P(*a): s = " ".join(str(x) for x in a); out.append(s); print(s)
def rom(a, b): return A[a].sum() / A[b].sum() - 1          # ratio of means
def mor(a, b): return float(np.mean(A[a] / np.maximum(A[b], 1e-12)) - 1)   # mean of ratios
def pval(a, b):
    d = A[a] - A[b]
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(A[a], A[b], alternative="less").pvalue)
P(f"{len(L)} landscapes, 4 seeds each; mean AUC per arm:")
for a in arms: P(f"   {a:9s} {A[a].mean():.4f}  vs W1 ratio-of-means {rom(a,'W1'):+.1%}, mean-of-ratios {mor(a,'W1'):+.1%}")
tests = [("TH-L1 SP vs W1", "SP", "W1", 0.05), ("TH-L2 SP vs SP-uniform", "SP", "uniform", 0.02), ("TH-L3 SP vs SP-frozen", "SP", "frozen", 0.02)]
ps = [pval(a, b) for _, a, b, _ in tests]; order = np.argsort(ps); holm = [0.0] * 3; run = 0.0
for rank, i in enumerate(order): run = max(run, min(1.0, (3 - rank) * ps[i])); holm[i] = run
res = {}
P("\nGates (need effect below -threshold and Holm p < 0.05):")
for (nm, a, b, th), p_, h in zip(tests, ps, holm):
    eff = rom(a, b); ok = eff <= -th and h < 0.05; res[nm] = ok
    P(f"   {nm}: ratio-of-means {eff:+.1%} (mean-of-ratios {mor(a,b):+.1%}), raw p {p_:.4f}, Holm p {h:.4f}, threshold -{th:.0%} -> {'PASS' if ok else 'FAIL'}")
harm = float(np.mean(A["SP"] > 1.05 * A["W1"])); worst = float(np.max(A["SP"] / A["W1"]) - 1)
P(f"\nSP worse than W1 by >5 % on {harm:.0%} of landscapes (worst {worst:+.0%}); better by >5 % on {np.mean(A['SP'] < 0.95*A['W1']):.0%}")
if res["TH-L1 SP vs W1"] and harm > 0.25: P("   -> L1 passes on average but is harmful on a sizeable minority (>25 %)")
c1 = rom("shuffled", "W1") >= -0.02; c2 = rom("permw", "SP") >= 0.02
P(f"\nControls: SP-shuffled not >2 % better than W1: {rom('shuffled','W1'):+.1%} -> {'OK' if c1 else 'FAIL'}; SP-permw >=2 % worse than SP: {rom('permw','SP'):+.1%} -> {'OK' if c2 else 'FAIL'}")
P(f"Sensitivity (not gated): SP c=0 vs W1 {rom('sp_c0','W1'):+.1%}")
fl = [l for l, w, r_ in zip(L, A["W1"], A["random"]) if w > 1.05 * r_]
P(f"\nW1 failure landscapes on new seeds (W1 > 1.05 x random): {len(fl)}")
if fl:
    ex = sum(A['W1'][L.index(l)] - A['random'][L.index(l)] for l in fl); cl = sum(A['W1'][L.index(l)] - A['SP'][L.index(l)] for l in fl)
    P(f"   SP closes {cl/ex:.0%} of W1's excess over random on them")
allo = {k: np.mean([u[k][a] for l in L for u in D[l]]) for k in ("eff", "vol") for a in ["SP"]}
P(f"\nSP diagnostics: effective sources {allo['eff']:.1f}, weight volatility {allo['vol']:.3f}")
P("\nControls " + ("hold" if c1 and c2 else "FAIL: result not interpretable"))
open(os.path.join(R, "sp_lcbench.txt"), "w").write("\n".join(out) + "\n")

"""Phase C analysis (tev1 typed selector, modes argmax/sample/gated; dev seeds 0..S-1). Pre-registered in PLAN_acquisition_choice.md.
H1': each mode vs TS (best static) ; H2 (primary): each mode vs the tuned rule, with 95% bootstrap CI against the +/-5 % equivalence margin;
H4: does the model's probability for AF a predict that a's proposal would have beaten the incumbent (AUROC, ECE), does confidence track outcome?
Run: .venv/bin/python -m acq.analyse_phaseC"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS, load_pool

H = os.path.dirname(os.path.abspath(__file__))
res = {}
for d in ("phaseA", "phaseB", "phaseD"):
    for f in glob.glob(os.path.join(H, "results", d, "*.pkl")):
        r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
for f in glob.glob(os.path.join(H, "results", "phaseC", "dev__*.pkl")):
    r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
modes = sorted({k[0] for k in res if k[0].startswith("typed_")})
DS = DEV[:8]
seeds = sorted({s for (c, d, s) in res if c == modes[0]})
seeds = [s for s in seeds if all((m, d, s) in res for m in modes for d in DS)]
print("modes", modes, "complete seeds", seeds)
A = lambda c, d: np.array([res[(c, d, s)]["auc"] for s in seeds])
cols = ["static_LogEI", "static_LogPI", "static_TS", "random", "rule", "learned_L2", "llm_gptoss"] + modes
cols = [c for c in cols if all((c, DS[0], s) in res for s in seeds)]
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_','').replace('typed_tev1_','tev1_'):>12s}" for c in cols))
for d in DS: print(f"{d:22s}" + "".join(f"{A(c, d).mean():12.3f}" for c in cols))
print(f"{'MEAN':22s}" + "".join(f"{np.mean([A(c, d).mean() for d in DS]):12.4f}" for c in cols))

rng = np.random.default_rng(0)
def holm(p):
    o = np.argsort(p); h = np.empty(len(p))
    for r, i in enumerate(o): h[i] = min(1.0, (len(p) - r) * p[i])
    return np.maximum.accumulate(h[o])[np.argsort(o)]
def cmp(c, rc):
    a = np.array([A(c, d).mean() for d in DS]); b = np.array([A(rc, d).mean() for d in DS])
    p = wilcoxon(a - b).pvalue if (a != b).any() else 1.0
    D = np.array([A(c, d) - A(rc, d) for d in DS]); base = np.mean([A(rc, d).mean() for d in DS])
    bs = [D[np.ix_(rng.integers(0, 8, 8), rng.integers(0, len(seeds), len(seeds)))].mean() / base for _ in range(3000)]
    return ((a - b) / b).mean(), int((a < b).sum()), p, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for title, rc in (("H2 PRIMARY: vs tuned rule", "rule"), ("vs best static TS", "static_TS"), ("vs random-AF", "random"), ("vs LLM gpt-oss", "llm_gptoss")):
    if rc not in cols: continue
    out = [cmp(m, rc) for m in modes]; hp = holm(np.array([o[2] for o in out]))
    print(f"\n== {title} (negative = tev1 better; Holm over three modes) ==")
    for m, o, h in zip(modes, out, hp):
        eq = "EQUIVALENT within +/-5%" if (o[3] > -0.05 and o[4] < 0.05) else "equivalence not shown"
        print(f"  {m:22s} {o[0]:+7.1%} | better on {o[1]}/8 | p={o[2]:.3f} Holm {h:.3f} | 95% CI [{o[3]:+.1%}, {o[4]:+.1%}] -> {eq}")

print("\n== H4: calibration of the model's probabilities (all calls, all modes) ==")
pools = {d: load_pool(d) for d in DS}
P, Y, CONF, IMP = [], [], [], []
for (c, d, s), r in res.items():
    if not c.startswith("typed_") or s not in seeds: continue
    p = pools[d]; span = max(p.y_max - p.y_med, 1e-12)
    best = float(p.y[np.random.default_rng(s).choice(len(p.y), 2 * p.X.shape[1] + 1, replace=False)].max())
    for st, call in zip(r["log"], r["calls"]):
        if call["probs"] is None: best = max(best, st["y"]); continue
        lab = [float(y > best) for y in st["proposal_y"]]
        P.extend(call["probs"]); Y.extend(lab)
        CONF.append(call["conf"]); IMP.append(float(st["y"] > best))
        best = max(best, st["y"])
P, Y = np.array(P), np.array(Y)
print(f"{len(P)//6} calls; base rate that an AF's proposal beats the incumbent: {Y.mean():.3f}")
print(f"AUROC of p_a for 'a's proposal beats incumbent': {roc_auc_score(Y, P):.3f}")
pa = P.reshape(-1, 6); ya = Y.reshape(-1, 6)
print("per-AF AUROC:", {a: round(roc_auc_score(ya[:, i], pa[:, i]), 3) if 0 < ya[:, i].sum() < len(ya) else None for i, a in enumerate(AFS)})
print("per-AF mean p vs observed rate:", {a: (round(pa[:, i].mean(), 3), round(ya[:, i].mean(), 3)) for i, a in enumerate(AFS)})
print(f"confidence vs (chosen AF improved): spearman = {spearmanr(CONF, IMP)[0]:+.3f} (n={len(CONF)})")

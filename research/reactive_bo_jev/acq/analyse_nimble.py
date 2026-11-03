"""nimble (9B) sample mode vs static, rule, learned, random, tev1-sample and a state-blind mixture built from nimble's own AF shares (weights fixed from the shares, then run as a control).
Reports both summaries (change in mean AUC; mean of per-dataset ratios), Wilcoxon over 8 dataset means, bootstrap CI, AF usage, calibration of the model's probabilities, fallback rate.
Run: .venv/bin/python -m acq.analyse_nimble"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS, load_pool
H = os.path.dirname(os.path.abspath(__file__))
res = {}
for d in ("phaseA", "phaseB", "phaseD", "phaseC"):
    for f in glob.glob(os.path.join(H, "results", d, "*.pkl")):
        if d == "phaseC" and not any(k in os.path.basename(f) for k in ("nimble__sample", "dev__sample", "ctrl__nimble")): continue
        r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
DS = DEV[:8]; S = range(10)
N = "typed_nimble_sample"
cnt = np.zeros(6); fb = n = 0; lat = []
for d in DS:
    for s in S:
        for c in res[(N, d, s)]["calls"]:
            cnt[c["af"]] += 1; n += 1; fb += c["fallback"]
            if c["latency"]: lat.append(c["latency"])
shares = cnt / cnt.sum()
print(f"nimble-sample: {n} calls, fallbacks {fb}, AF shares {dict(zip(AFS, np.round(shares, 3)))}")
sw = np.mean([np.mean([a["af"] != b["af"] for a, b in zip(res[(N, d, s)]["calls"][:-1], res[(N, d, s)]["calls"][1:])]) for d in DS for s in S])
print(f"switch rate between consecutive steps {sw:.1%}")
for lo, hi in ((0, 17), (17, 34), (34, 50)):
    c2 = np.zeros(6)
    for d in DS:
        for s in S:
            for c in res[(N, d, s)]["calls"][lo:hi]: c2[c["af"]] += 1
    print(f"  steps {lo:2d}-{hi:2d}:", {a: f"{x/c2.sum():.0%}" for a, x in zip(AFS, c2)})
np.save(os.path.join(H, "results", "nimble_sample_shares.npy"), shares)

A = lambda c: np.array([np.mean([res[(c, d, s)]["auc"] for s in S]) for d in DS])
cols = ["static_LogEI", "static_LogPI", "static_TS", "random", "rule", "learned_L2", "llm_gptoss", "typed_tev1_sample", N]
cols = [c for c in cols if all((c, d, s) in res for d in DS for s in S)]
print("\nmean AUC by dataset (seeds 0-9)")
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_', '').replace('typed_', '')[:11]:>12s}" for c in cols))
for i, d in enumerate(DS): print(f"{d:22s}" + "".join(f"{A(c)[i]:12.3f}" for c in cols))
print(f"{'MEAN':22s}" + "".join(f"{A(c).mean():12.4f}" for c in cols))
rng = np.random.default_rng(5)
print("\nnimble-sample vs each (negative = nimble better): change in mean AUC | mean of per-dataset ratios | better on k/8 | Wilcoxon p | 95% CI")
for rc in ("static_TS", "static_LogEI", "static_LogPI", "rule", "random", "learned_L2", "llm_gptoss", "typed_tev1_sample"):
    if rc not in cols: continue
    a, b = A(N), A(rc)
    D = np.array([[res[(N, d, s)]["auc"] - res[(rc, d, s)]["auc"] for s in S] for d in DS])
    bs = [D[np.ix_(rng.integers(0, 8, 8), rng.integers(0, 10, 10))].mean() / b.mean() for _ in range(4000)]
    print(f"  vs {rc:18s} {(a.mean() - b.mean()) / b.mean():+7.1%} | {((a - b) / b).mean():+7.1%} | {int((a < b).sum())}/8 | p={wilcoxon(a - b).pvalue:.3f} | [{np.percentile(bs, 2.5):+.1%}, {np.percentile(bs, 97.5):+.1%}]")

print("\n== calibration: does P(a) predict that a's proposal beats the incumbent? ==")
pools = {d: load_pool(d) for d in DS}
P, Y, CONF, IMP = [], [], [], []
for d in DS:
    p = pools[d]
    for s in S:
        r = res[(N, d, s)]
        best = float(p.y[np.random.default_rng(s).choice(len(p.y), 2 * p.X.shape[1] + 1, replace=False)].max())
        for st, call in zip(r["log"], r["calls"]):
            if call["probs"] is not None:
                P.extend(call["probs"]); Y.extend([float(y > best) for y in st["proposal_y"]]); CONF.append(call["conf"]); IMP.append(float(st["y"] > best))
            best = max(best, st["y"])
P, Y = np.array(P), np.array(Y); pa, ya = P.reshape(-1, 6), Y.reshape(-1, 6)
print(f"base rate {Y.mean():.3f}; AUROC of p_a {roc_auc_score(Y, P):.3f}; per-AF AUROC {dict(zip(AFS, [round(roc_auc_score(ya[:, i], pa[:, i]), 3) for i in range(6)]))}")
print(f"mean p vs observed rate per AF: {dict(zip(AFS, [(round(pa[:, i].mean(), 3), round(ya[:, i].mean(), 3)) for i in range(6)]))}")
print(f"confidence vs chosen-AF improved: spearman {spearmanr(CONF, IMP)[0]:+.3f} (n={len(CONF)})")

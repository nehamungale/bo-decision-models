"""Phase D analysis: the generative LLM selector against static, numeric selectors and random-AF on the SAME seeds (0..S-1 present for the LLM).
Primary: LLM vs TS (best pooled static) and LLM vs the tuned rule, paired Wilcoxon across the 8 dataset means, Holm over the two. Secondary: vs LogEI, UCB,
random-AF, learned_L2. Also: AF usage, fallback rate, call latency, nondeterminism proxy.
Run: .venv/bin/python -m acq.analyse_phaseD"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS

H = os.path.dirname(os.path.abspath(__file__))
res = {}
for d in ("phaseA", "phaseB", "phaseD"):
    for f in glob.glob(os.path.join(H, "results", d, "*.pkl")):
        r = pickle.load(open(f, "rb")); res[(r["cond"], r["dataset"], r["seed"])] = r
llm = sorted({k[0] for k in res if k[0].startswith("llm_")})[0]
seeds = sorted({k[2] for k in res if k[0] == llm and all((llm, d, k[2]) in res for d in DEV[:8])})
DS = DEV[:8]
print(f"LLM condition {llm}; complete seeds: {seeds}")
A = lambda c, d: np.array([res[(c, d, s)]["auc"] for s in seeds])
cols = ["static_LogEI", "static_UCB", "static_TS", "random", "rule", "learned_L2", llm]
print(f"{'dataset':22s}" + "".join(f"{c.replace('static_',''):>12s}" for c in cols))
for d in DS: print(f"{d:22s}" + "".join(f"{A(c, d).mean():12.3f}" for c in cols))
print(f"{'MEAN':22s}" + "".join(f"{np.mean([A(c, d).mean() for d in DS]):12.4f}" for c in cols))

def holm(p):
    o = np.argsort(p); h = np.empty(len(p))
    for r, i in enumerate(o): h[i] = min(1.0, (len(p) - r) * p[i])
    return np.maximum.accumulate(h[o])[np.argsort(o)]

def cmp(rc):
    a = np.array([A(llm, d).mean() for d in DS]); b = np.array([A(rc, d).mean() for d in DS])
    p = wilcoxon(a - b).pvalue if (a != b).any() else 1.0
    sig = sum(wilcoxon(A(llm, d) - A(rc, d)).pvalue < .05 and A(llm, d).mean() < A(rc, d).mean() for d in DS)
    return ((a - b) / b).mean(), int((a < b).sum()), sig, p

print("\n== PRIMARY (negative change = LLM better) ==")
prim = ["static_TS", "rule"]; out = [cmp(c) for c in prim]; hp = holm(np.array([o[3] for o in out]))
for c, o, h in zip(prim, out, hp):
    print(f"  LLM vs {c:10s} mean relative AUC change {o[0]:+7.1%} | better on {o[1]}/8 datasets ({o[2]} individually significant) | p={o[3]:.3f}, Holm p={h:.3f}")
print("== secondary ==")
for c in ["static_LogEI", "static_UCB", "random", "learned_L2"]:
    o = cmp(c); print(f"  LLM vs {c:12s} {o[0]:+7.1%} | better on {o[1]}/8 ({o[2]} sig) | p={o[3]:.3f}")

cnt = np.zeros(6); fb = 0; n = 0; lat = []
for d in DS:
    for s in seeds:
        for c in res[(llm, d, s)]["llm_calls"]:
            cnt[c["af"]] += 1; n += 1; fb += c["fallback"]
            if c["latency"]: lat.append(c["latency"])
print("\nAF usage:", {a: f"{x/cnt.sum():.0%}" for a, x in zip(AFS, cnt)})
print(f"fallback (malformed or failed) rate {fb/n:.2%} of {n} calls; latency median {np.median(lat):.2f}s p95 {np.quantile(lat, .95):.2f}s")
sw = np.mean([np.mean([c["af"] != p["af"] for p, c in zip(res[(llm, d, s)]["llm_calls"][:-1], res[(llm, d, s)]["llm_calls"][1:])]) for d in DS for s in seeds])
print(f"switch rate between consecutive steps: {sw:.1%}")

"""T1 stop gates of PLAN_fallback_detection.md (no detector is trained here). Run: python -m meta.analyse_detect_t1"""
import glob, os, pickle
import numpy as np
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
U = {}
for f in glob.glob(os.path.join(H, "detect", "*.pkl")):
    r = pickle.load(open(f, "rb")); U[(r["landscape"], r["seed"])] = r
Ls = sorted({l for l, _ in U}); fam = lambda l: "lcbench" if l.startswith("lcbench") else l.split("/")[0]
AR = ["random", "W1", "hedge", "W1R10", "W1F10", "W1F10h", "W1R5", "W1F5"]
rows = {}; skipped = []; noinj = []
for l in Ls:
    s4 = [U[(l, s)] for s in range(4300, 4304) if (l, s) in U]
    if any(u.get("skipped", True) for u in s4): skipped.append(l); continue
    if any("W1F10" not in u["arms"] for u in s4): noinj.append(l); continue
    d = {a: np.array([u["arms"][a]["auc"] for u in s4]) for a in AR if a != "random" and all(a in u["arms"] for u in s4)}
    d["random"] = np.array([U[(l, s)]["arms"]["random"]["auc"] for s in range(4300, 4312)])
    rows[l] = d
L = sorted(rows); out = []
P = lambda *a: (out.append(" ".join(str(x) for x in a)), print(*a))[1]
P(f"{len(Ls)} landscapes run; skipped (fewer than 8 training tasks): {len(skipped)}; no room for injection (k+m > budget): {len(noinj)}; analysed: {len(L)}")
m = lambda a, idx=None: np.mean([rows[l][a].mean() for l in (L if idx is None else idx)])
P("\nMean AUC over landscapes (seed-averaged; ratio vs W1 in brackets):")
for a in AR:
    if all(a in rows[l] for l in L): P(f"  {a:8s} {m(a):.4f}  ({m(a)/m('W1')-1:+.1%})")
W = np.array([rows[l]["W1"].mean() for l in L]); R = np.array([rows[l]["random"].mean() for l in L])
fail = W > 1.05 * R
P(f"\nFailure (W1 worse than random by > 5 % of random AUC): {fail.sum()}/{len(L)}  | by family:", {f: f"{int(fail[[fam(l)==f for l in L]].sum())}/{sum(fam(l)==f for l in L)}" for f in sorted({fam(l) for l in L})})
for eps in (0.0, 0.01, 0.02):
    hR = np.array([rows[l]["W1R10"].mean() < (1 - eps) * rows[l]["W1"].mean() for l in L]); hF = np.array([rows[l]["W1F10"].mean() < (1 - eps) * rows[l]["W1"].mean() for l in L]); hE = hR | hF
    P(f"\nCross-tab, epsilon {eps:.0%} (k=10, m=n0):   injection helps | among {fail.sum()} failing | among {(~fail).sum()} non-failing")
    for nm, h in (("random", hR), ("fps", hF), ("either", hE)):
        P(f"   {nm:7s} {h.sum():4d}/{len(L)} landscapes | {h[fail].sum()}/{fail.sum()} = {h[fail].mean():.0%} | {h[~fail].sum()}/{(~fail).sum()} = {h[~fail].mean():.0%}")
# realised oracle
def oracle(sel_idx=None, score_idx=None):
    tot = 0.0
    for l in L:
        d = rows[l]; sel = sel_idx if sel_idx is not None else slice(None); sc = score_idx if score_idx is not None else slice(None)
        arms = ["W1", "W1R10", "W1F10"]; best = min(arms, key=lambda a: d[a][sel].mean()); tot += d[best][sc].mean()
    return tot / len(L)
w1m = m("W1"); o_in = oracle()
o_split = 0.5 * (oracle(np.array([0, 1]), np.array([2, 3])) + oracle(np.array([2, 3]), np.array([0, 1])))
w1_split = w1m
P(f"\nRealised oracle (best of W1, W1R10, W1F10 per landscape): in-sample {o_in:.4f} = {1 - o_in/w1m:.1%} below always-W1 ({w1m:.4f}); selected on 2 seeds, scored on the other 2: {o_split:.4f} = {1 - o_split/w1m:.1%}")
cost_free = np.mean(np.minimum(W, R)); P(f"Cost-free hindsight (min of W1, random per landscape): {cost_free:.4f} = {1 - cost_free/w1m:.1%} below always-W1 (the earlier w150 number)")
gain = w1m - o_in
P("\nShare of the realised oracle gain captured by fixed arms (negative = worse than always-W1):")
for a in ("hedge", "W1F10", "W1R10", "W1F10h", "random"):
    P(f"   always-{a:7s} {(w1m - m(a))/gain:+.0%}  (AUC {m(a):.4f})")
bf = min(["W1", "random", "hedge", "W1R10", "W1F10"], key=m); P(f"   best fixed arm: {bf} ({m(bf):.4f}); share {(w1m - m(bf))/gain:+.0%}")
P("\nStop gates: (1) injection helps in >= 30 % of failing landscapes; (2) realised oracle gain >= 5 % of W1; (3) hedge captures <= 70 % of the oracle gain.")
open(os.path.join(H, "detect_t1.txt"), "w").write("\n".join(out) + "\n")

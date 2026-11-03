"""T1 pilot analysis for PLAN_nonstationary.md (written before results). Per pair, post-change AUC is averaged over seeds and over T_c in {25 %, 50 %}; summary = ratio of means over pairs; Wilcoxon (one-sided) over pairs.
TH-N0: oracle_time >= 10 % below oblivious, p < 0.05, same sign in LCBench and TabRepo strata. Gated regime: unrelated x abrupt. Other regimes reported (robustness). False-alarm cost c_FA from stationary runs.
Run: python -m meta.analyse_nonstat_t1 -> meta/results/nonstat_t1.txt"""
import os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon
from meta.nonstat_pairs import load_pairs
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "results")
pairs = {(p["A"], p["B"]): p for p in load_pairs()["pilot"]}
C, S = {}, {}
for f in glob.glob(os.path.join(R, "nonstat_t1", "*.pkl")):
    r = pickle.load(open(f, "rb"))
    (C if r["kind"] == "change" else S).setdefault((r["A"], r["B"], r["a"]), []).append(r)
out = []
def P(*a): s = " ".join(str(x) for x in a); out.append(s); print(s)
ARMS = ["oblivious", "oracle_time", "oracle_target", "oracle_keep", "window", "periodic"]; FR = (0.25, 0.5)
def per_pair(keys, arm, frs=FR, field="post_auc"):
    return np.array([np.mean([r["runs"][(arm, fr)][field] for r in C[k] for fr in frs]) for k in keys])
def wil(x, y):
    d = x - y
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y, alternative="less").pvalue)
regimes = {}
for (A, B, a), rs in C.items():
    p = pairs[(A, B)]; regimes.setdefault((p["relatedness"], "abrupt" if a == 1.0 else "partial"), []).append((A, B, a))
P(f"units: {sum(len(v) for v in C.values())} change, {sum(len(v) for v in S.values())} stationary")
for reg in sorted(regimes):
    ks = regimes[reg]; P(f"\n== {reg[0]} x {reg[1]} ({len(ks)} pairs, 4 seeds, both T_c): post-change AUC, ratio of means vs oblivious")
    ob = per_pair(ks, "oblivious")
    for arm in ARMS[1:]:
        x = per_pair(ks, arm); P(f"   {arm:14s} {x.sum()/ob.sum()-1:+7.1%}  (p {wil(x, ob):.4f}; mean-of-ratios {np.mean(x/np.maximum(ob,1e-12))-1:+.1%})")
    for fr in FR:
        o2 = per_pair(ks, "oblivious", (fr,)); P(f"   by T_c={fr:.0%}: oracle_time {per_pair(ks,'oracle_time',(fr,)).sum()/o2.sum()-1:+.1%}, oracle_target {per_pair(ks,'oracle_target',(fr,)).sum()/o2.sum()-1:+.1%}, periodic {per_pair(ks,'periodic',(fr,)).sum()/o2.sum()-1:+.1%}")
    for st in ("lcbench", "tabrepo"):
        kk = [k for k in ks if pairs[(k[0], k[1])]["stratum"] == st]
        if kk:
            o3 = per_pair(kk, "oblivious"); P(f"   stratum {st} ({len(kk)} pairs): oracle_time {per_pair(kk,'oracle_time').sum()/o3.sum()-1:+.1%}, oracle_target {per_pair(kk,'oracle_target').sum()/o3.sum()-1:+.1%}, oblivious AUC {o3.mean():.3f}")
    best_t = {a: per_pair(ks, "oracle_time").sum() / ob.sum() - 1 for a in [0]}
P("\n== TH-N0 gate (unrelated x abrupt, oracle_time vs oblivious)")
ks = regimes[("unrelated", "abrupt")]; ob = per_pair(ks, "oblivious"); ot = per_pair(ks, "oracle_time"); eff = ot.sum() / ob.sum() - 1; pv = wil(ot, ob)
signs = {}
for st in ("lcbench", "tabrepo"):
    kk = [i for i, k in enumerate(ks) if pairs[(k[0], k[1])]["stratum"] == st]; signs[st] = ot[kk].sum() / ob[kk].sum() - 1
ok = eff <= -0.10 and pv < 0.05 and all(v < 0 for v in signs.values())
P(f"   effect {eff:+.1%}, p {pv:.4f}, strata {signs} -> {'PASS' if ok else 'FAIL'} (threshold -10 %, p < 0.05, both strata negative)")
P("   passing regimes for N1 (oracle_time >= 10 % below oblivious, p < 0.05, ratio of means):")
for reg in sorted(regimes):
    k2 = regimes[reg]; o = per_pair(k2, "oblivious"); x = per_pair(k2, "oracle_time"); e = x.sum() / o.sum() - 1; p_ = wil(x, o)
    P(f"      {reg}: {e:+.1%} p {p_:.4f} -> {'passes' if e <= -0.10 and p_ < 0.05 else 'fails'}")
P("\n== False-alarm cost from stationary runs (pilot pairs, 4 seeds)")
ks = sorted({(k[0], k[1]) for k in S}); ob = np.array([np.mean([r["runs"]["oblivious"]["auc"] for a in (1.0,) for r in S[(A, B, a)]]) for A, B in ks])
for nm in ("restart50", "periodic25"):
    x = np.array([np.mean([r["runs"][nm]["auc"] for r in S[(A, B, 1.0)]]) for A, B in ks]); P(f"   {nm}: AUC {x.mean():.4f} vs oblivious {ob.mean():.4f} -> {x.sum()/ob.sum()-1:+.1%} (absolute +{x.mean()-ob.mean():.4f}); mean restarts {np.mean([r['runs'][nm]['n_restarts'] for k in S for r in S[k]]):.2f}")
x = np.array([np.mean([r["runs"]["restart50"]["auc"] for r in S[(A, B, 1.0)]]) for A, B in ks]); cfa = max(x.sum() / ob.sum() - 1, 1e-9)
P(f"   c_FA (one forced restart at 50 %, relative AUC increase) = {cfa:.3f}; allowed false-alarm rate = min(10 %, 0.02 / c_FA) = {min(0.10, 0.02/cfa):.1%}" + ("  -> detector infeasible (< 2 %)" if min(0.10, 0.02/cfa) < 0.02 else ""))
open(os.path.join(R, "nonstat_t1.txt"), "w").write("\n".join(out) + "\n")

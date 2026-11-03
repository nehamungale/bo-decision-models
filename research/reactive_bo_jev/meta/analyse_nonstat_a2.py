"""One analysis pass for A2 (PLAN_nonstationary.md 9d/9e/9h): test pairs only. python -m meta.analyse_nonstat_a2 -> meta/results/nonstat_a2.txt
Per-pair value = mean over seeds and T_c fractions of post-change AUC; ratio of means over pairs; one-sided paired Wilcoxon over pairs."""
import os, glob, pickle, sys
import numpy as np
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "nonstat_a2"); FRS = (0.25, 0.5)
GATED = ("unrelated", "abrupt")


def load():
    from meta.nonstat_pairs import load_pairs
    pairs = {(p["A"], p["B"]): p for p in load_pairs()["test"]}; C, S = {}, {}
    for f in glob.glob(os.path.join(OUT, "*.pkl")):
        r = pickle.load(open(f, "rb"))
        for nm, v in r["runs"].items():
            if r["kind"] == "change": C.setdefault((r["A"], r["B"], r["a"]), {}).setdefault(nm[0], {}).setdefault(nm[1], {})[r["seed"]] = v
            else: S.setdefault((r["A"], r["B"]), {}).setdefault(nm, {})[r["seed"]] = v
    return pairs, C, S


def regime(pairs, k): return (pairs[k[:2]]["relatedness"], "abrupt" if k[2] == 1.0 else "partial")
def stratum(pairs, k):
    p = pairs[k[:2]]; return None if p["relatedness"] == "related" else ("anti" if p["spearman"] < 0 else "lowpos")


def keys(pairs, C, reg=None, strat=None, arms=()):
    return [k for k in sorted(C) if (reg is None or regime(pairs, k) == reg) and (strat is None or stratum(pairs, k) == strat or strat == "all") and all(a in C[k] for a in arms)]


def pv(C, ks, arm, field="post_auc"): return np.array([np.mean([v[field] for fr in FRS for v in C[k][arm][fr].values()]) for k in ks])
def sv(S, ks, arm): return np.array([np.mean([v["auc"] for v in S[k][arm].values()]) for k in ks])


def cmp(num, den):
    """ratio of means - 1, one-sided Wilcoxon p that num < den, and the number of pairs"""
    d = num - den; p = wilcoxon(d, alternative="less").pvalue if np.any(d != 0) else 1.0
    return float(num.sum() / den.sum() - 1), float(p), len(num)


def boot_ci(num, den, B=4000, seed=1):
    r = np.random.default_rng(seed); n = len(num); x = [num[i].sum() / den[i].sum() - 1 for i in (r.integers(0, n, n) for _ in range(B))]
    return float(np.quantile(x, .05)), float(np.quantile(x, .95))


def a0_check():
    pairs, C, S = load(); arms = ("oracle_keep", "refresh"); out = {}
    ks = keys(pairs, C, GATED, arms=arms); g = cmp(pv(C, ks, "oracle_keep"), pv(C, ks, "refresh")); out["gated"] = g
    for st in ("anti", "lowpos"):
        k2 = keys(pairs, C, GATED, st, arms=arms); out[st] = cmp(pv(C, k2, "oracle_keep"), pv(C, k2, "refresh")) if k2 else None
    out["pass"] = bool(g[0] <= -0.10 and g[1] < 0.05 and all(out[s] and out[s][0] < 0 for s in ("anti", "lowpos"))); return out


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def main():
    pairs, C, S = load(); L = []
    def P(*a): s = " ".join(str(x) for x in a); L.append(s); print(s)
    R = [(r, w) for r in ("unrelated", "related") for w in ("abrupt", "partial")]
    have = lambda *a: all(any(x in C[k] for k in C) for x in a)
    P(f"A2 analysis; test pairs {len(pairs)}, campaigns per cell = pairs x 4 seeds x 2 T_c")
    a0 = a0_check(); P("\nTH-A0 (oracle_keep vs refresh, unrelated x abrupt; need <= -10 %, p < .05, both strata < 0):", {k: v for k, v in a0.items()})
    P("\n== post-change AUC ratio of means vs refresh and vs oblivious, by regime (n pairs, p)")
    for arm in ("remeasure_keep", "random_keep_rm", "oracle_keep", "sp_stale_remeasure", "sp_passive", "sp_stale_full", "oracle_time", "oracle_target", "refresh"):
        if not have(arm): continue
        row = []
        for reg in R:
            ks = keys(pairs, C, reg, arms=(arm, "refresh", "oblivious"))
            if not ks: continue
            a, b = cmp(pv(C, ks, arm), pv(C, ks, "refresh")), cmp(pv(C, ks, arm), pv(C, ks, "oblivious"))
            row.append(f"{reg[0][:3]}x{reg[1][:3]}: vsRef {a[0]:+.1%} (p {a[1]:.3f}) vsObl {b[0]:+.1%}")
        P(f"  {arm:20s}", " | ".join(row))
    P("\n== unrelated x abrupt, split (note: anti-correlated pairs are almost all TabRepo)")
    for st in ("anti", "lowpos"):
        for arm in ("remeasure_keep", "oracle_keep", "sp_stale_full", "sp_passive"):
            if not have(arm): continue
            ks = keys(pairs, C, GATED, st, arms=(arm, "refresh")); P(f"  {st:7s} {arm:16s} n={len(ks)} vs refresh {cmp(pv(C, ks, arm), pv(C, ks, 'refresh'))[0]:+.1%}")
    P("\n== stationary AUC ratio vs oblivious and vs refresh")
    for arm in ("refresh", "remeasure_keep", "random_keep_rm", "sp_stale_remeasure", "sp_passive", "sp_stale_full"):
        ks = [k for k in sorted(S) if arm in S[k] and "oblivious" in S[k]]
        if ks: P(f"  {arm:20s} vs oblivious {sv(S, ks, arm).sum() / sv(S, ks, 'oblivious').sum() - 1:+.1%}; trigger share {np.mean([v['ntrig'] > 0 for k in ks for v in S[k][arm].values()]):.1%}" + (f"; vs refresh {sv(S, ks, arm).sum() / sv(S, ks, 'refresh').sum() - 1:+.1%}" if all("refresh" in S[k] for k in ks) else ""))
    P("\n== TH-A1 (remeasure_keep vs refresh; gated need <= -5 %, p < .05; related and stationary harm <= +2 %)")
    ks = keys(pairs, C, GATED, arms=("remeasure_keep", "refresh")); a1 = cmp(pv(C, ks, "remeasure_keep"), pv(C, ks, "refresh")); P("  gated", a1)
    harm = {}
    for reg in R:
        k2 = keys(pairs, C, reg, arms=("remeasure_keep", "refresh")); harm[reg] = float(pv(C, k2, "remeasure_keep").sum() / pv(C, k2, "refresh").sum() - 1)
    P("  per regime vs refresh:", {f"{a[:3]}x{b[:3]}": f"{v:+.1%}" for (a, b), v in harm.items()})
    k3 = [k for k in sorted(S) if "remeasure_keep" in S[k] and "refresh" in S[k]]; sh = sv(S, k3, "remeasure_keep").sum() / sv(S, k3, "refresh").sum() - 1; P(f"  stationary vs refresh {sh:+.1%}")
    ok1 = a1[0] <= -0.05 and a1[1] < 0.05 and sh <= 0.02 and all(harm[r] <= 0.02 for r in R if r[0] == "related")
    kc = keys(pairs, C, GATED, arms=("remeasure_keep", "random_keep_rm")); ctl = cmp(pv(C, kc, "remeasure_keep"), pv(C, kc, "random_keep_rm")); P(f"  vs matching random control (gated): {ctl[0]:+.1%} (p {ctl[1]:.3f}); need <= -2 %")
    if have("oracle_keep"):
        ko = keys(pairs, C, GATED, arms=("remeasure_keep", "refresh", "oracle_keep")); g_r = pv(C, ko, "remeasure_keep").sum() - pv(C, ko, "refresh").sum(); g_o = pv(C, ko, "oracle_keep").sum() - pv(C, ko, "refresh").sum(); P(f"  share of oracle_keep's gain recovered (gated): {g_r / g_o:.1%}")
    P("\n== detection (remeasure_keep): trigger share, pre-T_c triggers, delay, gain given trigger vs refresh")
    for reg in R:
        ks = keys(pairs, C, reg, arms=("remeasure_keep", "refresh")); tot = trg = pre = 0; dl = []; num = den = 0.0; dr = []
        for k in ks:
            for fr in FRS:
                for sd, v in C[k]["remeasure_keep"][fr].items():
                    tot += 1
                    if v["first"] is not None:
                        if v["first"] < v["Tc"]: pre += 1
                        else: trg += 1; dl.append(v["delay"]); dr.append(v["dreg"]); num += v["post_auc"]; den += C[k]["refresh"][fr][sd]["post_auc"]
        P(f"  {reg[0][:3]}x{reg[1][:3]}: triggered post-T_c {trg/tot:.1%}, pre-T_c {pre/tot:.1%}, missed {1 - (trg + pre)/tot:.1%}, mean delay {np.mean(dl) if dl else float('nan'):.1f}, delay-window regret {np.mean(dr) if dr else float('nan'):.2f}, gain given trigger vs refresh {num/den - 1 if den else float('nan'):+.1%}")
    P("\n== TH-A3 and the passive reading (gated; passive arms are untuned designs, 9e)")
    ps = []
    pas = [a for a in ("sp_passive", "sp_stale_full") if have(a)]
    best = None
    for a in pas:
        ks = keys(pairs, C, GATED, arms=("remeasure_keep", a)); c = cmp(pv(C, ks, "remeasure_keep"), pv(C, ks, a)); ci = boot_ci(pv(C, ks, "remeasure_keep"), pv(C, ks, a))
        ks2 = [k for k in sorted(S) if a in S[k] and "oblivious" in S[k]]; sth = sv(S, ks2, a).sum() / sv(S, ks2, "oblivious").sum() - 1
        rel = {f"{r[0][:3]}x{r[1][:3]}": float(pv(C, keys(pairs, C, r, arms=("remeasure_keep", a)), "remeasure_keep").sum() / pv(C, keys(pairs, C, r, arms=("remeasure_keep", a)), a).sum() - 1) for r in R}
        P(f"  remeasure_keep vs {a}: {c[0]:+.1%} (p {c[1]:.3f}), 90 % bootstrap CI [{ci[0]:+.1%}, {ci[1]:+.1%}]; {a} stationary harm vs oblivious {sth:+.1%}; per regime {rel}")
        tot = pv(C, ks, a).sum()
        if best is None or tot < best[1]: best = (a, tot, c, ci, sth)
    if best:
        a, _, c, ci, sth = best; ok3 = c[0] <= -0.02 and c[1] < 0.05
        P(f"  better passive arm (lower gated post-change AUC): {a}; A3 {'PASS' if ok3 else 'FAIL'}; 'passive suffices' needs CI inside +-2 % ({ci[0] > -0.02 and ci[1] < 0.02}) and passive stationary harm <= 2 % ({sth <= 0.02})")
        adj = holm(np.array([a0["gated"][1], a1[1], c[1], 1.0]))
        P(f"  Holm (m = 4, A2 equivalence test entered as p = 1): A0 {adj[0]:.3f}, A1 {adj[1]:.3f}, A3 {adj[2]:.3f}")
    P(f"\nSummary: A0 {'PASS' if a0['pass'] else 'FAIL'}; A1 {'PASS' if ok1 else 'FAIL'}"); open(os.path.join(H, "results", "nonstat_a2.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__": main()

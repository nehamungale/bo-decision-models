"""Noise stage tuning (PLAN_nonstationary.md 9i): 5 % noise, 40 tuning pairs, seeds 5100-5103. Stages: n1a v, n1b (r, m) grid + refresh + refresh_rep, n1c random control. Rules are declared in 9i.
Run: python -m meta.nonstat_n1 [workers] -> meta/results/nonstat_n1/{decisions.json, report.txt}. Stops on the pinned rules."""
import sys, os, time, pickle, json, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "nonstat_n1"); os.makedirs(OUT, exist_ok=True)
SEEDS = [5100, 5101, 5102, 5103]; FRS = (0.25, 0.5); VS = (0.01, 0.1, 0.3); RS = (5, 10, 20); MS = (1, 2, 3, 4); NZ = 0.05; B_STEPS = 50
DEC = os.path.join(OUT, "decisions.json")
tolm = lambda m: m * 2 ** 0.5 * NZ


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta import nonstat as ns
    stage, kind, la, lb, aa, seed, cfgs = a
    f = os.path.join(OUT, f"{stage}__{kind}__{la.replace('/', '__')}__{lb.replace('/', '__')}__{aa}__{seed}.pkl")
    if os.path.exists(f): return f
    pair = ns.make_pair(la, lb, aa); res = dict(kind=kind, A=la, B=lb, a=aa, seed=seed, runs={})
    for name, arm, kw in cfgs:
        kw = dict(kw, noise=NZ)
        if kind == "change":
            for fr in FRS:
                r = ns.run_ns(pair, seed, arm, frac=fr, **kw)
                res["runs"][(name, fr)] = dict(post_auc=r["post_auc"], auc=r["auc"], ntrig=len(r["triggers"]), first=(r["triggers"][0] if r["triggers"] else None), delay=r["delay"], dreg=r["delay_regret"], Tc=r["Tc"])
        else:
            r = ns.run_ns(pair, seed, arm, stationary=True, **kw); res["runs"][name] = dict(auc=r["auc"], ntrig=len(r["triggers"]))
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


def run_stage(stage, cfgs, W, kinds=("change", "stationary")):
    from meta.nonstat_pairs import load_pairs
    P = load_pairs()["tuning"]; J = []
    if "change" in kinds: J += [(stage, "change", p["A"], p["B"], aa, s, cfgs) for s in SEEDS for p in P for aa in (1.0, 0.5)]
    if "stationary" in kinds: J += [(stage, "stationary", p["A"], p["B"], 1.0, s, cfgs) for s in SEEDS for p in P]
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(unit, J, chunksize=1):
            k += 1
            if k % 40 == 0: print(f"  {stage} {k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"{stage} done {len(J)} units {time.time()-t0:.0f}s", flush=True)


def load_all():
    from meta.nonstat_pairs import load_pairs
    pairs = {(p["A"], p["B"]): p for p in load_pairs()["tuning"]}; C, S = {}, {}
    for f in glob.glob(os.path.join(OUT, "*.pkl")):
        r = pickle.load(open(f, "rb"))
        for nm, v in r["runs"].items():
            if r["kind"] == "change": C.setdefault((r["A"], r["B"], r["a"]), {}).setdefault(nm, []).append(v)
            else: S.setdefault((r["A"], r["B"]), {}).setdefault(nm, []).append(v)
    return pairs, C, S


def regimes(pairs, C):
    R = {}
    for (A, B, a) in C: R.setdefault((pairs[(A, B)]["relatedness"], "abrupt" if a == 1.0 else "partial"), []).append((A, B, a))
    return R
def post_by_pair(C, ks, name): return np.array([np.mean([v["post_auc"] for fr in FRS for v in C[k][(name, fr)]]) for k in ks])
def pooled_ratio(pairs, C, num, den): R = regimes(pairs, C); return float(np.mean([post_by_pair(C, ks, num).sum() / post_by_pair(C, ks, den).sum() - 1 for ks in R.values()]))
def stat_ratio(S, num, den): ks = list(S); return float(sum(np.mean([v["auc"] for v in S[k][num]]) for k in ks) / sum(np.mean([v["auc"] for v in S[k][den]]) for k in ks) - 1)
def stat_fa(S, name): return float(np.mean([v["ntrig"] > 0 for k in S for v in S[k][name]]))


def trig_rates(pairs, C, S, name):
    R = regimes(pairs, C)
    ch = {f"{r[0]}x{r[1]}": float(np.mean([v["ntrig"] > 0 for k in ks for fr in FRS for v in C[k][(name, fr)]])) for r, ks in R.items()}
    pre = float(np.mean([(v["first"] is not None and v["first"] < v["Tc"]) for k in C for fr in FRS for v in C[k][(name, fr)]]))
    dl = [v["delay"] for k in C for fr in FRS for v in C[k][(name, fr)] if v["delay"] is not None]
    return dict(change=ch, pre_Tc=pre, stationary=stat_fa(S, name), mean_delay=float(np.mean(dl)) if dl else None)


def load_dec(): return json.load(open(DEC)) if os.path.exists(DEC) else {}
def save_dec(d): json.dump(d, open(DEC, "w"), indent=1)


def main(W):
    D = load_dec(); rep = []
    def P(*a):
        s = " ".join(str(x) for x in a); rep.append(s); print(s, flush=True)
    def stop(why):
        D["stopped"] = why; save_dec(D); P("\nN1 STOPPED:", why); open(os.path.join(OUT, "report.txt"), "w").write("\n".join(rep) + "\n"); sys.exit(0)
    run_stage("n1a", [("oblivious", "oblivious", {})] + [(f"oracle_keep_v{v}", "oracle_keep", dict(var_stale=v)) for v in VS], W, kinds=("change",))
    run_stage("n1as", [("oblivious", "oblivious", {})] + [(f"forced_keep_v{v}", "forced_keep", dict(var_stale=v)) for v in VS], W, kinds=("stationary",))
    pairs, C, S = load_all(); J_v = {v: pooled_ratio(pairs, C, f"oracle_keep_v{v}", "oblivious") + stat_ratio(S, f"forced_keep_v{v}", "oblivious") for v in VS}
    vstar = min(VS, key=lambda v: (round(J_v[v], 6), abs(v - 0.1))); D["v_star"] = vstar; D["J_v"] = {str(k): x for k, x in J_v.items()}
    P("N1a keep variance under 5 % noise:", {k: round(x, 4) for k, x in J_v.items()}, "-> v* =", vstar)
    for v in VS: P(f"   v={v}: oracle_keep post {pooled_ratio(pairs, C, f'oracle_keep_v{v}', 'oblivious'):+.1%}, forced_keep stationary {stat_ratio(S, f'forced_keep_v{v}', 'oblivious'):+.1%}")
    cfa = max(stat_ratio(S, f"forced_keep_v{vstar}", "oblivious"), 1e-9); D["c_FA_keep"] = cfa; D["fa_budget_keep"] = min(0.10, 0.02 / cfa); save_dec(D); P(f"   c_FA(keep, noise) = {cfa:.3f}; FA budget = {D['fa_budget_keep']:.1%}")
    cfgs = [(f"refresh_r{r}", "refresh", dict(r=r)) for r in RS] + [(f"rk_r{r}_m{m}", "remeasure_keep", dict(r=r, tol=tolm(m), var_stale=vstar)) for r in RS for m in MS] + [("refresh_rep_r10", "refresh_rep", dict(r=10))]
    run_stage("n1b", cfgs, W)
    pairs, C, S = load_all(); grid = {}
    for r in RS:
        for m in MS:
            n = f"rk_r{r}_m{m}"; grid[(r, m)] = dict(J=pooled_ratio(pairs, C, n, "oblivious") + stat_ratio(S, n, "oblivious"), post=pooled_ratio(pairs, C, n, "oblivious"), stat=stat_ratio(S, n, "oblivious"), fa=stat_fa(S, n), tr=trig_rates(pairs, C, S, n))
            P(f"   r={r} m={m}: J {grid[(r, m)]['J']:+.3f}  post vs obl {grid[(r, m)]['post']:+.1%}  stat vs obl {grid[(r, m)]['stat']:+.1%}  stationary FA {grid[(r, m)]['fa']:.1%}  change triggers {({k: round(x, 2) for k, x in grid[(r, m)]['tr']['change'].items()})}  pre-Tc {grid[(r, m)]['tr']['pre_Tc']:.1%}  delay {grid[(r, m)]['tr']['mean_delay']}")
    for r in RS: P(f"   refresh r={r}: post vs oblivious {pooled_ratio(pairs, C, f'refresh_r{r}', 'oblivious'):+.1%}, stationary {stat_ratio(S, f'refresh_r{r}', 'oblivious'):+.1%}")
    P(f"   replicate effect (exploratory) r=10: refresh_rep vs refresh post-change {pooled_ratio(pairs, C, 'refresh_rep_r10', 'refresh_r10'):+.1%}, stationary {stat_ratio(S, 'refresh_rep_r10', 'refresh_r10'):+.1%}")
    D["grid"] = {f"{r}_{m}": {k: v for k, v in g.items() if k != "tr"} for (r, m), g in grid.items()}; save_dec(D)
    if all(pooled_ratio(pairs, C, f"refresh_r{r}", "oblivious") > 0 for r in RS): stop("(a) refresh is net-harmful vs oblivious in change scenarios for every r")
    feas = [(g["J"], k) for k, g in grid.items() if g["fa"] <= D["fa_budget_keep"]]
    if not feas: stop("(d) no (r, m) cell is within the false-alarm budget: remeasure_keep infeasible at 5 % noise")
    rstar, mstar = min(feas)[1]; D["r_star"], D["m_star"] = rstar, mstar; save_dec(D); P(f"   frozen cell: r* = {rstar}, m* = {mstar} (feasible cells {sorted((round(j, 3), k) for j, k in feas)})")
    R = regimes(pairs, C); ks = R[("unrelated", "abrupt")]; ok = post_by_pair(C, ks, f"oracle_keep_v{vstar}").sum() / post_by_pair(C, ks, f"refresh_r{rstar}").sum() - 1; D["oracle_keep_vs_refresh_gated"] = ok
    P(f"   oracle_keep(v*) vs refresh(r*) gated: {ok:+.1%} (stop if above -5 %)"); save_dec(D)
    if ok > -0.05: stop("(c) oracle_keep less than 5 % below refresh in the gated regime")
    n = f"rk_r{rstar}_m{mstar}"; tr = trig_rates(pairs, C, S, n); q = 0.5 * float(np.mean(list(tr["change"].values()))) + 0.5 * tr["stationary"]; D["q"], D["rate_rm"] = q, q / B_STEPS
    P(f"   control rate: q = {q:.3f} -> per step {q / B_STEPS:.5f}"); save_dec(D)
    run_stage("n1c", [("random_keep_rm", "random_keep_rm", dict(r=rstar, rate=D["rate_rm"], var_stale=vstar, tol=tolm(mstar)))], W)
    pairs, C, S = load_all(); gap = pooled_ratio(pairs, C, n, "random_keep_rm"); D["frozen_vs_control"] = gap
    P(f"   frozen detector vs matching random control: post-change {gap:+.1%} (stop if above -2 %); frozen vs refresh {pooled_ratio(pairs, C, n, f'refresh_r{rstar}'):+.1%}, stationary vs refresh {stat_ratio(S, n, f'refresh_r{rstar}'):+.1%}")
    save_dec(D)
    if gap > -0.02: stop("(b) the frozen detector does not beat its matching random control by 2 %")
    P("\nN1 complete. Manifest:", {k: D[k] for k in ("v_star", "r_star", "m_star", "rate_rm")}); open(os.path.join(OUT, "report.txt"), "w").write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)

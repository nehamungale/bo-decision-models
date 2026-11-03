"""A1 of PLAN_nonstationary.md Amendment 1 (pins in sections 9b/9c): tuning stage on the 40 tuning pairs, seeds 4900-4903, T_c in {25 %, 50 %}, a in {1, .5}.
Stages: S1a keep variance v, S1b interval r, S2 CUSUM threshold h + c_FA(spstale), S3 remaining arms, random controls, detector selection. All decision rules are in this file (declared before the data).
Run: python -m meta.nonstat_a1 [workers]  -> meta/results/nonstat_a1/{decisions.json, report.txt}; stops early on the pinned stop rules."""
import sys, os, time, pickle, json, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "nonstat_a1"); os.makedirs(OUT, exist_ok=True)
SEEDS = [4900, 4901, 4902, 4903]; FRS = (0.25, 0.5); VS = (0.01, 0.1, 0.3); RS = (5, 10, 20); HS = (2.0, 4.0, 8.0, 16.0, 32.0); B_STEPS = 50
DEC = os.path.join(OUT, "decisions.json")


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta import nonstat as ns
    stage, kind, la, lb, aa, seed, cfgs = a
    f = os.path.join(OUT, f"{stage}__{kind}__{la.replace('/', '__')}__{lb.replace('/', '__')}__{aa}__{seed}.pkl")
    if os.path.exists(f): return f
    pair = ns.make_pair(la, lb, aa); res = dict(kind=kind, A=la, B=lb, a=aa, seed=seed, runs={})
    for name, arm, kw in cfgs:
        if kind == "change":
            for fr in FRS:
                r = ns.run_ns(pair, seed, arm, frac=fr, **kw)
                res["runs"][(name, fr)] = dict(post_auc=r["post_auc"], auc=r["auc"], ntrig=len(r["triggers"]), first=(r["triggers"][0] if r["triggers"] else None), delay=r["delay"], dreg=r["delay_regret"], Tc=r["Tc"])
        else:
            r = ns.run_ns(pair, seed, arm, stationary=True, **kw); res["runs"][name] = dict(auc=r["auc"], ntrig=len(r["triggers"]))
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


def run_stage(stage, cfgs, W, kinds=("change", "stationary")):
    from meta.nonstat_pairs import load_pairs
    P = load_pairs()["tuning"]
    J = []
    if "change" in kinds: J += [(stage, "change", p["A"], p["B"], aa, s, cfgs) for s in SEEDS for p in P for aa in (1.0, 0.5)]
    if "stationary" in kinds: J += [(stage, "stationary", p["A"], p["B"], 1.0, s, cfgs) for s in SEEDS for p in P]
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(unit, J, chunksize=1):
            k += 1
            if k % 40 == 0: print(f"  {stage} {k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"{stage} done {len(J)} units {time.time()-t0:.0f}s", flush=True)


# ---- aggregation (tuning pairs only)
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


def post_by_pair(C, ks, name, field="post_auc"):
    return np.array([np.mean([v[field] for fr in FRS for v in C[k][(name, fr)]]) for k in ks])


def pooled_ratio(pairs, C, num, den):
    """mean over the four regimes of (sum num / sum den over pairs - 1)"""
    R = regimes(pairs, C); return float(np.mean([post_by_pair(C, ks, num).sum() / post_by_pair(C, ks, den).sum() - 1 for ks in R.values()]))


def stat_ratio(S, num, den):
    ks = list(S); return float(sum(np.mean([v["auc"] for v in S[k][num]]) for k in ks) / sum(np.mean([v["auc"] for v in S[k][den]]) for k in ks) - 1)


def trig_rates(pairs, C, S, name):
    R = regimes(pairs, C)
    ch = {f"{r[0]}x{r[1]}": float(np.mean([v["ntrig"] > 0 for k in ks for fr in FRS for v in C[k][(name, fr)]])) for r, ks in R.items()}
    pre = float(np.mean([(v["first"] is not None and v["first"] < v["Tc"]) for k in C for fr in FRS for v in C[k][(name, fr)]]))
    st = float(np.mean([v["ntrig"] > 0 for k in S for v in S[k][name]])) if S and all(name in S[k] for k in S) else None
    dl = [v["delay"] for k in C for fr in FRS for v in C[k][(name, fr)] if v["delay"] is not None]
    return dict(change=ch, pre_Tc=pre, stationary=st, mean_delay=float(np.mean(dl)) if dl else None)


def load_dec(): return json.load(open(DEC)) if os.path.exists(DEC) else {}
def save_dec(d): json.dump(d, open(DEC, "w"), indent=1)


def main(W):
    D = load_dec(); rep = []
    def P(*a):
        s = " ".join(str(x) for x in a); rep.append(s); print(s, flush=True)
    def stop(why):
        D["stopped"] = why; save_dec(D); P("\nA1 STOPPED:", why); open(os.path.join(OUT, "report.txt"), "w").write("\n".join(rep) + "\n"); sys.exit(0)
    # ---- S1a: keep variance v
    run_stage("s1a", [("oblivious", "oblivious", {})] + [(f"oracle_keep_v{v}", "oracle_keep", dict(var_stale=v)) for v in VS], W, kinds=("change",))
    run_stage("s1as", [("oblivious", "oblivious", {})] + [(f"forced_keep_v{v}", "forced_keep", dict(var_stale=v)) for v in VS], W, kinds=("stationary",))
    pairs, C, S = load_all(); J_v = {}
    for v in VS: J_v[v] = pooled_ratio(pairs, C, f"oracle_keep_v{v}", "oblivious") + stat_ratio(S, f"forced_keep_v{v}", "oblivious")
    vstar = min(VS, key=lambda v: (round(J_v[v], 6), abs(v - 0.1))); D["v_star"] = vstar; D["J_v"] = {str(k): x for k, x in J_v.items()}; save_dec(D)
    P("S1a keep variance (J = pooled post-change ratio of oracle_keep vs oblivious + stationary forced_keep cost):", {k: round(x, 4) for k, x in J_v.items()}, "-> v* =", vstar)
    for v in VS: P(f"   v={v}: oracle_keep post {pooled_ratio(pairs, C, f'oracle_keep_v{v}', 'oblivious'):+.1%}, forced_keep stationary {stat_ratio(S, f'forced_keep_v{v}', 'oblivious'):+.1%}")
    c_fa_keep = max(stat_ratio(S, f"forced_keep_v{vstar}", "oblivious"), 1e-9); D["c_FA_keep"] = c_fa_keep; D["fa_budget_keep"] = min(0.10, 0.02 / c_fa_keep); save_dec(D)
    P(f"   c_FA(keep) = {c_fa_keep:.3f}; FA budget = {D['fa_budget_keep']:.1%}")
    # ---- S1b: interval r
    run_stage("s1b", [(f"refresh_r{r}", "refresh", dict(r=r)) for r in RS] + [(f"remeasure_keep_r{r}", "remeasure_keep", dict(r=r, var_stale=vstar)) for r in RS], W)
    pairs, C, S = load_all(); J_r = {}
    for r in RS: J_r[r] = pooled_ratio(pairs, C, f"remeasure_keep_r{r}", "oblivious") + stat_ratio(S, f"remeasure_keep_r{r}", "oblivious")
    rstar = min(RS, key=lambda r: round(J_r[r], 6)); D["r_star"] = rstar; D["J_r"] = {str(k): x for k, x in J_r.items()}; save_dec(D)
    P("\nS1b interval r (J = pooled post-change ratio + stationary ratio of remeasure_keep vs oblivious):", {k: round(x, 4) for k, x in J_r.items()}, "-> r* =", rstar)
    for r in RS: P(f"   r={r}: refresh post {pooled_ratio(pairs, C, f'refresh_r{r}', 'oblivious'):+.1%} (stationary {stat_ratio(S, f'refresh_r{r}', 'oblivious'):+.1%}); remeasure_keep post {pooled_ratio(pairs, C, f'remeasure_keep_r{r}', 'oblivious'):+.1%} (stationary {stat_ratio(S, f'remeasure_keep_r{r}', 'oblivious'):+.1%}); triggers {trig_rates(pairs, C, S, f'remeasure_keep_r{r}')['change']}")
    zero_fa = trig_rates(pairs, C, S, f"remeasure_keep_r{rstar}")["stationary"]; P(f"   remeasure_keep r*={rstar} stationary false-alarm rate: {zero_fa}")
    if all(pooled_ratio(pairs, C, f"refresh_r{r}", "oblivious") > 0 for r in RS): stop("(a) refresh is net-harmful vs oblivious in change scenarios for every r")
    R = regimes(pairs, C); ks = R[("unrelated", "abrupt")]
    head = post_by_pair(C, ks, f"refresh_r{rstar}").sum()
    # oracle_keep(v*) shares oblivious's trajectory but with the keep response; refresh baseline needs oracle_keep+refresh: compare with refresh as defined (re-measurement only)
    ok = post_by_pair(C, ks, f"oracle_keep_v{vstar}").sum() / head - 1; D["oracle_keep_vs_refresh_gated"] = ok; save_dec(D)
    P(f"   early A0 check, unrelated x abrupt: oracle_keep(v*) vs refresh(r*) {ok:+.1%} (stop if above -5 %)")
    if ok > -0.05: stop("(c) oracle_keep is less than 5 % below refresh in the gated regime on the tuning pairs")
    # ---- S2: forced_spstale cost, CUSUM h
    run_stage("s2", [(f"cusum_keep_h{h}", "cusum_keep", dict(cusum_h=h, var_stale=vstar)) for h in HS], W)
    run_stage("s2s", [("forced_spstale", "forced_spstale", {})], W, kinds=("stationary",))
    pairs, C, S = load_all()
    D["c_FA_spstale"] = stat_ratio(S, "forced_spstale", "oblivious"); P(f"\nS2 c_FA(spstale) = {D['c_FA_spstale']:+.3f}")
    feas = []
    for h in HS:
        fa = trig_rates(pairs, C, S, f"cusum_keep_h{h}"); g = pooled_ratio(pairs, C, f"cusum_keep_h{h}", f"refresh_r{rstar}")
        P(f"   h={h}: stationary FA {fa['stationary']:.1%}, post-change vs refresh {g:+.1%}, pre-T_c triggers {fa['pre_Tc']:.1%}, mean delay {fa['mean_delay']}, change triggers {fa['change']}")
        if fa["stationary"] <= D["fa_budget_keep"]: feas.append((g, h))
    D["h_star"] = min(feas)[1] if feas else None; save_dec(D)
    P("   h* =", D["h_star"], "" if feas else "(no h meets the FA budget: cusum_keep infeasible)")
    # ---- S3: remaining arms
    cfg3 = [("sp_stale_remeasure", "sp_stale_remeasure", dict(r=rstar)), ("sp_passive", "sp_passive", {}), ("oblivious", "oblivious", {})]
    if D["h_star"]: cfg3.append(("shuffled_keep", "shuffled_keep", dict(cusum_h=D["h_star"], var_stale=vstar)))
    run_stage("s3a", cfg3, W)
    pairs, C, S = load_all()
    P(f"\nS3 sp_passive stationary harm vs oblivious {stat_ratio(S, 'sp_passive', 'oblivious'):+.1%}; post-change vs oblivious {pooled_ratio(pairs, C, 'sp_passive', 'oblivious'):+.1%}" + ("  [FLAG: above 5 %]" if stat_ratio(S, 'sp_passive', 'oblivious') > 0.05 else ""))
    cands = {"remeasure_keep": (f"remeasure_keep_r{rstar}", f"remeasure_keep_r{rstar}"), "sp_stale_remeasure": ("sp_stale_remeasure", "sp_stale_remeasure")}
    if D["h_star"]: cands["cusum_keep"] = (f"cusum_keep_h{D['h_star']}", f"cusum_keep_h{D['h_star']}")
    sel = []
    for nm, (cn, sn) in cands.items():
        g = pooled_ratio(pairs, C, cn, f"refresh_r{rstar}"); sh = stat_ratio(S, sn, f"refresh_r{rstar}"); tr = trig_rates(pairs, C, S, cn)
        P(f"   {nm}: post-change vs refresh {g:+.1%}, stationary vs refresh {sh:+.1%}, stationary FA {tr['stationary']}, pre-T_c triggers {tr['pre_Tc']:.1%}, delay {tr['mean_delay']}, change triggers {tr['change']}")
        if sh <= 0.02: sel.append((g, nm))
    D["detector"] = min(sel)[1] if sel else None; save_dec(D); P("   frozen detector:", D["detector"])
    if D["detector"] is None: stop("no detector has stationary harm <= 2 % vs refresh")
    # ---- random controls at the frozen detectors' tuning rates
    def q_of(cn):
        tr = trig_rates(pairs, C, S, cn); return 0.5 * float(np.mean(list(tr["change"].values()))) + 0.5 * (tr["stationary"] or 0.0)
    q_rm, q_cu = q_of(f"remeasure_keep_r{rstar}"), (q_of(f"cusum_keep_h{D['h_star']}") if D["h_star"] else 0.0)
    D["q_remeasure"], D["q_cusum"] = q_rm, q_cu; D["rate_rm"], D["rate_cu"] = q_rm / B_STEPS, q_cu / B_STEPS; save_dec(D)
    P(f"   control rates: q(remeasure_keep) {q_rm:.3f} -> per step {q_rm/B_STEPS:.4f}; q(cusum_keep) {q_cu:.3f} -> {q_cu/B_STEPS:.4f}")
    cfg3b = [("random_keep_rm", "random_keep_rm", dict(r=rstar, rate=D["rate_rm"], var_stale=vstar))] + ([("random_keep", "random_keep", dict(rate=D["rate_cu"], var_stale=vstar))] if D["h_star"] else [])
    run_stage("s3b", cfg3b, W)
    pairs, C, S = load_all()
    ctrl = {"remeasure_keep": "random_keep_rm", "cusum_keep": "random_keep", "sp_stale_remeasure": "random_keep_rm"}[D["detector"]]
    dn = cands[D["detector"]][0]; gap = pooled_ratio(pairs, C, dn, ctrl)
    P(f"\n   frozen detector {D['detector']} vs matching random control {ctrl}: post-change {gap:+.1%} (stop if above -2 %)")
    for c_ in ("random_keep_rm", "random_keep"):
        if all(c_ in C[k] or (c_, 0.25) in C[k] for k in C): P(f"   {c_}: post vs refresh {pooled_ratio(pairs, C, c_, f'refresh_r{rstar}'):+.1%}, stationary vs refresh {stat_ratio(S, c_, f'refresh_r{rstar}'):+.1%}")
    D["frozen_vs_control"] = gap; save_dec(D)
    if gap > -0.02: stop("(b) the frozen detector does not beat its matching random control by 2 % on the tuning pairs")
    P("\nA1 complete. Manifest:", {k: D[k] for k in ("v_star", "r_star", "detector", "h_star", "q_remeasure", "q_cusum", "rate_rm", "rate_cu")})
    open(os.path.join(OUT, "report.txt"), "w").write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)

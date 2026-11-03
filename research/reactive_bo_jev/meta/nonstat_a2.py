"""A2 of PLAN_nonstationary.md (sections 9d frozen settings, 9e Amendment 2, 9h pins): the 100 test pairs, seeds 5000-5003, T_c in {25 %, 50 %}, a in {1, .5}, plus stationary controls.
Stages: active, passive, ref. After 'active' the runner evaluates TH-A0 and stops if it fails. Run: python -m meta.nonstat_a2 [workers]; analysis: python -m meta.analyse_nonstat_a2"""
import sys, os, time, pickle, json
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "nonstat_a2"); os.makedirs(OUT, exist_ok=True)
SEEDS = [5000, 5001, 5002, 5003]; FRS = (0.25, 0.5)
D1 = json.load(open(os.path.join(H, "results", "nonstat_a1", "decisions.json")))
V, R, RATE = D1["v_star"], D1["r_star"], D1["rate_rm"]
assert (V, R, D1["detector"]) == (0.01, 10, "remeasure_keep") and abs(RATE - 0.008171875) < 1e-12      # the frozen manifest of 9d
STAGES = {
    "active": ([("oblivious", "oblivious", {}), ("refresh", "refresh", dict(r=R)), ("remeasure_keep", "remeasure_keep", dict(r=R, var_stale=V)),
                ("random_keep_rm", "random_keep_rm", dict(r=R, rate=RATE, var_stale=V)), ("oracle_keep", "oracle_keep", dict(var_stale=V)),
                ("sp_stale_remeasure", "sp_stale_remeasure", dict(r=R))], ("change", "stationary")),
    "passive": ([("sp_passive", "sp_passive", {}), ("sp_stale_full", "sp_stale_full", {})], ("change", "stationary")),
    "ref": ([("oracle_time", "oracle_time", {}), ("oracle_target", "oracle_target", {})], ("change",)),
}


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
                res["runs"][(name, fr)] = dict(post_auc=r["post_auc"], pre_auc=r["pre_auc"], auc=r["auc"], ntrig=len(r["triggers"]), first=(r["triggers"][0] if r["triggers"] else None), delay=r["delay"], dreg=r["delay_regret"], Tc=r["Tc"])
        else:
            r = ns.run_ns(pair, seed, arm, stationary=True, **kw); res["runs"][name] = dict(auc=r["auc"], ntrig=len(r["triggers"]))
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


def run_stage(stage, W):
    from meta.nonstat_pairs import load_pairs
    P = load_pairs()["test"]; cfgs, kinds = STAGES[stage]; J = []
    if "change" in kinds: J += [(stage, "change", p["A"], p["B"], aa, s, cfgs) for s in SEEDS for p in P for aa in (1.0, 0.5)]
    if "stationary" in kinds: J += [(stage, "stationary", p["A"], p["B"], 1.0, s, cfgs) for s in SEEDS for p in P]
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(unit, J, chunksize=1):
            k += 1
            if k % 100 == 0: print(f"  {stage} {k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"{stage} done {len(J)} units {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    run_stage("active", W)
    from meta import analyse_nonstat_a2 as an
    a0 = an.a0_check()
    print("TH-A0:", a0, flush=True)
    if not a0["pass"]: print("A2 STOPPED: TH-A0 failed; no further stages (reported as the result)", flush=True); sys.exit(0)
    run_stage("passive", W); run_stage("ref", W); print("A2 complete", flush=True)

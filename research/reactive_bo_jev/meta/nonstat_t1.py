"""T1 pilot of PLAN_nonstationary.md: 20 pilot pairs x a in {1, .5} x T_c in {25 %, 50 %} x seeds 4800-4803; arms oblivious, oracle_time, oracle_target, oracle_keep, window, periodic (25 %).
Stationary controls (20 pairs x 4 seeds): oblivious, forced single restart at 50 % (periodic with period 50 %: the cost of one false alarm), periodic 25 %.
Run: python -m meta.nonstat_t1 [workers]"""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "nonstat_t1"); os.makedirs(OUT, exist_ok=True)
SEEDS = [4800, 4801, 4802, 4803]; ARMS = ["oblivious", "oracle_time", "oracle_target", "oracle_keep", "window", "periodic"]


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta import nonstat as ns
    kind, la, lb, aa, seed = a
    f = os.path.join(OUT, f"{kind}__{la.replace('/', '__')}__{lb.replace('/', '__')}__{aa}__{seed}.pkl")
    if os.path.exists(f): return f
    pair = ns.make_pair(la, lb, aa); res = dict(kind=kind, A=la, B=lb, a=aa, seed=seed, runs={})
    if kind == "change":
        for frac in (0.25, 0.5):
            for arm in ARMS:
                r = ns.run_ns(pair, seed, arm, frac=frac); res["runs"][(arm, frac)] = dict(post_auc=r["post_auc"], pre_auc=r["pre_auc"], auc=r["auc"], n_restarts=r["n_restarts"])
    else:
        for name, arm, kw in (("oblivious", "oblivious", {}), ("restart50", "periodic", dict(period_frac=0.5)), ("periodic25", "periodic", {})):
            r = ns.run_ns(pair, seed, arm, stationary=True, **kw); res["runs"][name] = dict(auc=r["auc"], n_restarts=r["n_restarts"])
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    from meta.nonstat_pairs import load_pairs
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    P = load_pairs()["pilot"]
    J = [("change", p["A"], p["B"], aa, s) for s in SEEDS for p in P for aa in (1.0, 0.5)] + [("stationary", p["A"], p["B"], 1.0, s) for s in SEEDS for p in P]
    print(len(J), "units", flush=True); t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(unit, J, chunksize=1):
            k += 1
            if k % 20 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(J)} {time.time()-t0:.0f}s", flush=True)

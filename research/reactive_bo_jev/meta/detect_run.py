"""T1 of PLAN_fallback_detection.md: arms random / W1 / hedge / W1+R(k) / W1+F(k) on the w150 landscapes, noise 0, with traces and observed indices stored.
Unit = (landscape, seed). Seeds 4300-4303: all arms; seeds 4304-4311: random only. Injection arms are skipped where k + m > budget.
Run from reactive_bo_jev/: python -m meta.detect_run pilot | full [workers]"""
import sys, os, time, pickle, functools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "detect"); os.makedirs(OUT, exist_ok=True)
SEEDS_ARMS = list(range(4300, 4304)); SEEDS_RANDOM = list(range(4300, 4312))


@functools.lru_cache(maxsize=4)
def portfolio(ln, n0):
    from meta import bo
    from meta.w_expand import train_pools
    tp = train_pools(ln)
    return None if len(tp) < 8 else (bo.portfolio_greedy(tp, n0), len(tp))


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo
    ln, seed = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{seed}.pkl")
    if os.path.exists(f): return f, 0.0
    t0 = time.time(); p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p); res = dict(landscape=ln, seed=seed, n0=n0, B=B, arms={})
    def go(name, **kw):
        r = bo.run(p, seed, budget=B, n_init=n0, **kw)
        res["arms"][name] = dict(auc=r["auc"], final=r["final"], regret=r["regret"], obs_idx=r["obs_idx"], y_obs=r["y_obs"])
    go("random")
    if seed in SEEDS_ARMS:
        pf = portfolio(ln, n0)
        if pf is None: res["skipped"] = True
        else:
            port, ntrain = pf; res["skipped"] = False; res["ntrain"] = ntrain; res["portfolio"] = port
            go("W1", init_idx=port)
            go("hedge", init_idx=bo.hedge_start(len(p.X), port, n0, seed))
            for k in (10, 5):
                for mode, tag in (("random", "R"), ("fps", "F")):
                    if k + n0 <= B: go(f"W1{tag}{k}", init_idx=port, inject=(k, n0, mode))
            if 10 + n0 // 2 <= B: go("W1F10h", init_idx=port, inject=(10, n0 // 2, "fps"))
    res["secs"] = time.time() - t0
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f, res["secs"]


if __name__ == "__main__":
    from meta.w150 import landscapes
    from meta.landscapes import load
    mode = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    L = landscapes()
    if mode == "pilot":
        sizes = sorted(L, key=lambda l: -(len(load(l).y) * load(l).X.shape[1]))[:5]
        print("pilot landscapes (largest pool x dim):", sizes, flush=True)
        with ProcessPoolExecutor(5) as ex:
            for f, s in ex.map(unit, [(l, 4300) for l in sizes]): print(os.path.basename(f), f"{s:.0f}s", flush=True)
    else:
        J = [(l, s) for s in SEEDS_RANDOM for l in L]
        t0 = time.time(); k = 0
        with ProcessPoolExecutor(W) as ex:
            for _ in ex.map(unit, J, chunksize=1):
                k += 1
                if k % 50 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
        print(f"done {len(J)} {time.time()-t0:.0f}s", flush=True)

"""PLAN_source_prior_lcbench.md: pre-registered LCBench-only test of the source prior. 24 LCBench dev landscapes (w150 minus the tuning set), seeds 4700-4703, noise 0.
Fixed configuration kappa 10, lambda .25, c .25 (no tuning). Arms: random, W1, SP, SP-frozen, SP-uniform, SP-shuffled, SP-permw, SP-c0 (sensitivity).
Run: python -m meta.sp_lcbench [workers] [--time]   (--time: one landscape, one seed, prints seconds per arm)"""
import sys, os, time, json, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "sp_lcbench"); os.makedirs(OUT, exist_ok=True)
SEEDS = [4700, 4701, 4702, 4703]; KAPPA, LAM, C = 10.0, 0.25, 0.25
ARMS = {"SP": dict(mode="sp", c=C), "frozen": dict(mode="frozen", c=C), "uniform": dict(mode="uniform", c=C),
        "shuffled": dict(mode="shuffled", c=C), "permw": dict(mode="permw", c=C), "sp_c0": dict(mode="sp", c=0.0)}


def test_landscapes():
    from meta.w150 import landscapes
    tune = set(json.load(open(os.path.join(H, "results", "sp_tuning_set.json")))["lcbench"])
    L = sorted(l for l in landscapes() if l.startswith("lcbench") and l not in tune)
    assert len(L) == 24 and not (set(L) & tune), len(L)
    return L


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo, sp
    from meta.w_expand import train_pools
    ln, seed = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{seed}.pkl")
    if os.path.exists(f): return f
    p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p); tp = train_pools(ln); S = bo._score_matrix(tp); port = bo.portfolio_greedy(tp, n0)
    res = dict(landscape=ln, seed=seed, auc={}, secs={}, eff={}, vol={})
    t = time.time(); res["auc"]["random"] = bo.run(p, seed, budget=B, n_init=n0)["auc"]; res["secs"]["random"] = time.time() - t
    t = time.time(); res["auc"]["W1"] = bo.run(p, seed, budget=B, n_init=n0, init_idx=port)["auc"]; res["secs"]["W1"] = time.time() - t
    for k, kw in ARMS.items():
        t = time.time(); r = sp.run_sp(p, S, seed, budget=B, n_init=n0, init_idx=port, kappa=KAPPA, lam=LAM, **kw)
        res["auc"][k] = r["auc"]; res["secs"][k] = time.time() - t; res["eff"][k] = float(r["eff_sources"].mean()); res["vol"][k] = float(r["volatility"].mean())
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    L = test_landscapes(); print(len(L), "landscapes", flush=True)
    if "--time" in sys.argv:
        t = time.time(); f = unit((L[0], SEEDS[0])); r = pickle.load(open(f, "rb")); print({k: round(v, 1) for k, v in r["secs"].items()}, f"total {time.time()-t:.0f}s")
        print({k: round(v, 4) for k, v in r["auc"].items()}); sys.exit()
    W = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 12
    J = [(l, s) for s in SEEDS for l in L]; t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(unit, J, chunksize=1):
            k += 1
            if k % 8 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(J)} {time.time()-t0:.0f}s", flush=True)

"""T1 of PLAN_source_prior.md: tune kappa in {3,10,30} x lambda in {.25,.5,1} on a tuning set disjoint from the test: 36 fresh TabRepo landscapes (6 per family, not in w150, >= 8 sources)
+ 6 LCBench dev landscapes (fixed-seed draw; they leave the test set). Seeds 4500-4501. Arms: W1 (portfolio start + standard GP, LogEI), SP per configuration, SP-uniform (diagnostic).
Selection rule (declared here, before the run): the configuration with the lowest mean over the two sources (TabRepo, LCBench) of AUC(SP)/AUC(W1) (ratio of means within a source).
Stop rule: no configuration at least 2 % below W1 on BOTH sources -> stop. Run: python -m meta.sp_tune [workers]"""
import sys, os, time, json, pickle, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "sp_tune"); os.makedirs(OUT, exist_ok=True)
SEEDS = [4500, 4501]; GRID = list(itertools.product([3.0, 10.0, 30.0], [0.25, 0.5, 1.0]))


def tuning_set():
    f = os.path.join(H, "results", "sp_tuning_set.json")
    if os.path.exists(f): return json.load(open(f))
    from meta.landscapes import kept
    from meta.w150 import landscapes
    from meta.w_expand import train_pools
    used = set(landscapes()); k = kept(("tabrepo",)); tab = []
    for g, h in k[~k.landscape.isin(used)].groupby("group"):
        if g.split("_", 1)[1] not in ("CatBoost", "ExtraTrees", "LightGBM", "NeuralNetTorch", "RandomForest", "XGBoost"): continue
        pool = h.sample(frac=1.0, random_state=77).landscape.tolist(); got = []
        for l in pool:
            if len(train_pools(l)) >= 8: got.append(l)
            if len(got) == 6: break
        tab += got
    lc = sorted(l for l in landscapes() if l.startswith("lcbench")); rng = np.random.default_rng(2026); lcb = sorted(rng.choice(lc, 6, replace=False).tolist())
    d = dict(tabrepo=tab, lcbench=lcb); json.dump(d, open(f, "w"), indent=1); return d


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo, sp
    from meta.w_expand import train_pools
    ln, seed = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{seed}.pkl")
    if os.path.exists(f): return f
    t0 = time.time(); p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p); tp = train_pools(ln); S = bo._score_matrix(tp); port = bo.portfolio_greedy(tp, n0)
    res = dict(landscape=ln, seed=seed, auc={}, eff={}, vol={})
    res["auc"]["W1"] = bo.run(p, seed, budget=B, n_init=n0, init_idx=port)["auc"]
    r = sp.run_sp(p, S, seed, "uniform", budget=B, n_init=n0, init_idx=port); res["auc"]["uniform"] = r["auc"]
    for kappa, lam in GRID:
        r = sp.run_sp(p, S, seed, "sp", budget=B, n_init=n0, init_idx=port, kappa=kappa, lam=lam)
        res["auc"][(kappa, lam)] = r["auc"]; res["eff"][(kappa, lam)] = float(r["eff_sources"].mean()); res["vol"][(kappa, lam)] = float(r["volatility"].mean())
    res["secs"] = time.time() - t0
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    T = tuning_set(); L = T["tabrepo"] + T["lcbench"]; print(len(T["tabrepo"]), "TabRepo +", len(T["lcbench"]), "LCBench tuning landscapes", flush=True)
    J = [(l, s) for s in SEEDS for l in L]; t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(unit, J, chunksize=1):
            k += 1
            if k % 10 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(J)} {time.time()-t0:.0f}s", flush=True)

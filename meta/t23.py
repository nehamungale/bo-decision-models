"""T2 (surrogate K) and T3 (search region R) with noise arms, PLAN_component_triage.md. 10 landscapes (3 LCBench, 4 HPO-B, 3 TabRepo from the Stage 2b list; plan said the T1 landscapes,
changed before data to include HPO-B since K/R do not depend on shared grids). Entries: K rbf_ard (default, also R global), matern52_ard, rbf_iso, rf; R box 0.15, 0.30, adaptive (GP rbf_ard).
Seeds A 4000-4004 (selection), B 4100-4119 (scoring); noise 0 and 5 % of span. Run: python -m meta.t23 [workers]"""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "t23"); os.makedirs(OUT, exist_ok=True)
SEEDS_A = list(range(4000, 4005)); SEEDS_B = list(range(4100, 4120))
CFG = {"K:rbf_ard": dict(kernel="rbf_ard"), "K:matern52_ard": dict(kernel="matern52_ard"), "K:rbf_iso": dict(kernel="rbf_iso"), "K:rf": dict(kernel="rf"),
       "R:0.15": dict(region=0.15), "R:0.30": dict(region=0.30), "R:adaptive": dict(region="adaptive")}


def targets():
    from meta.stage2 import landscapes
    L = landscapes()
    return [l for l in L if l.startswith("lcbench")][:3] + [l for l in L if l.startswith("hpob")][:4] + [l for l in L if l.startswith("tabrepo")][:3]


def job(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo
    ln, cfg, seed, noise = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{cfg}__{seed}__{noise}.pkl")
    if os.path.exists(f): return f
    p = load(ln); n0 = 2 * p.X.shape[1] + 1
    r = bo.run(p, seed, budget=budget(p), n_init=n0, noise_frac=noise, **CFG[cfg])
    pickle.dump(dict(landscape=ln, cfg=cfg, seed=seed, noise=noise, auc=r["auc"], final=r["final"]), open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    L = targets(); print(L, flush=True)
    J = [(l, c, s, n) for s in SEEDS_A + SEEDS_B for l in L for c in CFG for n in (0.0, 0.05)]
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(job, J, chunksize=2):
            k += 1
            if k % 200 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(J)} {time.time()-t0:.0f}s", flush=True)

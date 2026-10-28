"""Stage 1: label reliability (G0). 6 landscapes (2 per family, fixed-seed random draw) x 5 seeds, every 3rd step, H=4, two continuations.
Run from reactive_bo_jev/: python -m meta.stage1 [workers]. Pre-registered: G0 passes if mean within-state Spearman(LogEI-cont, UCB-cont) > 0.3
and exceeds the cross-state shuffled control."""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HZ = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stage1" if HZ == 4 else f"stage1_H{HZ}"); os.makedirs(OUT, exist_ok=True)
SEEDS = range(2000, 2005)


def job(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta.labels import LabelCollector, BEHAVIOURS
    from acq.engine import run_campaign
    name, seed = a
    f = os.path.join(OUT, f"{name.replace('/', '__')}__{seed}.pkl")
    if os.path.exists(f): return f
    p = load(name); B = budget(p); beh = BEHAVIOURS[(seed - 2000) % len(BEHAVIOURS)]
    c = LabelCollector(beh, H=HZ, budget=B)
    r = run_campaign(p, seed, c, budget=B, n_init=2 * p.X.shape[1] + 1, record=False)
    pickle.dump(dict(landscape=name, seed=seed, behaviour=beh, rows=c.rows, final=r["final"], auc=r["auc"]), open(f, "wb"))
    return f


if __name__ == "__main__":
    from meta.landscapes import kept
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    k = kept()
    pick = [k[k.family == fam].sample(2, random_state=11).landscape.tolist() for fam in ["lcbench", "hpob", "tabrepo"]]
    pick = [x for g in pick for x in g]; print("landscapes:", pick, flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(W) as ex: list(ex.map(job, [(n, s) for n in pick for s in SEEDS]))
    print(f"done {time.time()-t0:.0f}s", flush=True)

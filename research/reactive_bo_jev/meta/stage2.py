"""Stage 2b: powered offline H1 check. 30 landscapes (10 per family, fixed-seed draw, disjoint from the 6 pilot landscapes), 20 seeds (2100-2119),
behaviour policy cycles over [random, six statics], every 3rd step, H=4, LogEI continuation only. Resumable, one pickle per campaign.
Run from reactive_bo_jev/: python -m meta.stage2 [workers]"""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "stage2"); os.makedirs(OUT, exist_ok=True)
SEEDS = range(2100, 2120)


def landscapes():
    from meta.landscapes import kept
    pilot = set(pickle.load(open(f, "rb"))["landscape"] for f in __import__("glob").glob(os.path.join(os.path.dirname(OUT), "stage1", "*.pkl")))
    k = kept(); k = k[~k.landscape.isin(pilot)]
    return [x for fam in ["lcbench", "hpob", "tabrepo"] for x in k[k.family == fam].sample(10, random_state=21).landscape.tolist()]


def job(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta.labels import LabelCollector, BEHAVIOURS
    from acq.engine import run_campaign
    name, seed = a
    f = os.path.join(OUT, f"{name.replace('/', '__')}__{seed}.pkl")
    if os.path.exists(f): return f
    p = load(name); B = budget(p); beh = BEHAVIOURS[(seed - 2100) % len(BEHAVIOURS)]
    c = LabelCollector(beh, H=4, conts=(0,), budget=B)
    r = run_campaign(p, seed, c, budget=B, n_init=2 * p.X.shape[1] + 1, record=False)
    pickle.dump(dict(landscape=name, seed=seed, behaviour=beh, rows=c.rows, final=r["final"], auc=r["auc"]), open(f + ".tmp", "wb"))
    os.rename(f + ".tmp", f)
    return f


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    L = landscapes(); print("landscapes:", L, flush=True)
    jobs = [(n, s) for s in SEEDS for n in L]            # seed-major so partial results cover all landscapes
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(job, jobs):
            k += 1
            if k % 60 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)

"""T1 (PLAN_component_triage.md): warm start W on 10 landscapes (5 LCBench + 5 TabRepo from the Stage 2b list). Arms: random start (20 seeds, 4100-4119),
W1b greedy portfolio (primary, added before any data: standard zero-shot portfolio), W1a top-n0 by mean score (plan), W0 fixed random set (control).
Portfolio arms are deterministic given the start, so 3 seeds each (determinism check). Leave-one-task-out: portfolios use only other tasks sharing the held-out task's config grid.
Run: python -m meta.t1_warmstart [workers]"""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "t1"); os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(4100, 4120))


def targets():
    from meta.stage2 import landscapes
    L = landscapes()
    return [l for l in L if l.startswith("lcbench")][:5] + [l for l in L if l.startswith("tabrepo")][:5]


def train_pools(ln):
    from meta.landscapes import kept, load
    p = load(ln); k = kept(("lcbench", "tabrepo"))
    grp = k[k.landscape == ln].group.iloc[0]
    out = []
    for l in k[k.group == grp].landscape:
        if l == ln: continue
        q = load(l)
        if q.X.shape == p.X.shape and np.allclose(q.X, p.X): out.append(q)
    return out


def arm_init(ln, arm, p, n0):
    from meta import bo
    if arm == "random": return None
    if arm == "W0":
        return list(np.random.default_rng(0).choice(len(p.y), n0, replace=False))
    tp = train_pools(ln)
    return bo.portfolio_greedy(tp, n0) if arm == "W1b" else bo.portfolio_rank(tp, n0)


def job(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo
    ln, arm, seed = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{arm}__{seed}.pkl")
    if os.path.exists(f): return f
    p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p)
    init = arm_init(ln, arm, p, n0)
    r = bo.run(p, seed, budget=B, n_init=n0, init_idx=init)
    ntrain = len(train_pools(ln)) if arm in ("W1a", "W1b") else 0
    start = p.regret(float(p.y[init].max())) if init is not None else p.regret(float(p.y[np.random.default_rng(seed).choice(len(p.y), n0, replace=False)].max()))
    pickle.dump(dict(landscape=ln, arm=arm, seed=seed, auc=r["auc"], final=r["final"], start_regret=start, n_train_tasks=ntrain, regret=r["regret"]), open(f + ".tmp", "wb"))
    os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    L = targets(); print("targets:", L, flush=True)
    jobs = [(l, "random", s) for s in SEEDS for l in L] + [(l, arm, s) for s in SEEDS[:3] for l in L for arm in ("W1b", "W1a", "W0")]
    t0 = time.time()
    with ProcessPoolExecutor(W) as ex: list(ex.map(job, jobs))
    print(f"done {len(jobs)} campaigns {time.time()-t0:.0f}s", flush=True)

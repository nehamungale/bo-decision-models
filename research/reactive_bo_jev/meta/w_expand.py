"""T4 for W (PLAN_component_triage.md): all 30 Stage 2b landscapes, noise 0 and 5 % of span.
Arms: random start; W1b/W2 (primary: greedy exact-grid portfolio for LCBench/TabRepo, region prior for HPO-B); FPS (farthest-point space-filling start, coverage control);
W0 (10 fixed random sets per landscape, averaged: replaces the single-draw control of T1); W2x (HPO-B only: region prior from a different search space with the same input dimension).
Training tasks per target: same group, same config grid (exact families) or same input dimension (HPO-B), held-out task excluded; need >= 8 else the landscape is skipped for W and listed.
Added before data (deviations from the plan, recorded in notes.md): FPS control, W0 averaged over 10 draws, W2x, trust statistic logging.
Run: python -m meta.w_expand [workers]"""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w30"); os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(4100, 4120)); MIN_TRAIN = 8


def train_pools(ln, other_group=False):
    from meta.landscapes import kept, load
    p = load(ln); k = kept(("lcbench", "hpob", "tabrepo")); row = k[k.landscape == ln].iloc[0]
    exact = row.family in ("lcbench", "tabrepo")
    if other_group:
        cands = k[(k.family == "hpob") & (k.group != row.group)]
        for g, h in cands.groupby("group"):
            ps = [load(l) for l in h.landscape]; ps = [q for q in ps if q.X.shape[1] == p.X.shape[1]]
            if len(ps) >= MIN_TRAIN: return ps
        return []
    out = []
    for l in k[k.group == row.group].landscape:
        if l == ln: continue
        q = load(l)
        if exact and q.X.shape == p.X.shape and np.allclose(q.X, p.X): out.append(q)
        if not exact and q.X.shape[1] == p.X.shape[1]: out.append(q)
    return out


def init_for(ln, arm, p, n0, seed):
    from meta import bo
    fam = ln.split("/")[0].split("_")[0]
    if arm == "random": return None, {}
    if arm == "FPS": return bo.farthest_point_start(p.X, n0, seed), {}
    if arm.startswith("W0_"): return list(np.random.default_rng(int(arm[3:])).choice(len(p.y), n0, replace=False)), {}
    if arm == "W1":
        tp = train_pools(ln)
        if len(tp) < MIN_TRAIN: return "skip", {}
        if fam == "hpob":
            return bo.portfolio_region([q.X for q in tp], [q.y for q in tp], p.X, n0), {"ntrain": len(tp)}
        init = bo.portfolio_greedy(tp, n0); S = bo._score_matrix(tp).mean(0)
        from scipy.stats import spearmanr
        yo = p.y[init]; trust = float(spearmanr(yo, S[init])[0]) if np.ptp(yo) > 0 and np.ptp(S[init]) > 0 else float("nan")
        return init, {"ntrain": len(tp), "trust_spearman": trust, "obs_rank_frac": float((p.y < yo.max()).mean())}
    if arm == "W2x":
        if fam != "hpob": return "skip", {}
        tp = train_pools(ln, other_group=True)
        if not tp: return "skip", {}
        return bo.portfolio_region([q.X for q in tp], [q.y for q in tp], p.X, n0), {"ntrain": len(tp)}


def job(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo
    ln, arm, seed, noise = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{arm}__{seed}__{noise}.pkl")
    if os.path.exists(f): return f
    p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p)
    init, extra = init_for(ln, arm, p, n0, seed)
    if isinstance(init, str):
        pickle.dump(dict(landscape=ln, arm=arm, seed=seed, noise=noise, skipped=True), open(f, "wb")); return f
    r = bo.run(p, seed, budget=B, n_init=n0, init_idx=init, noise_frac=noise)
    start = p.regret(float(p.y[init if init is not None else np.random.default_rng(seed).choice(len(p.y), n0, replace=False)].max()))
    pickle.dump(dict(landscape=ln, arm=arm, seed=seed, noise=noise, auc=r["auc"], final=r["final"], start_regret=start, skipped=False, **extra), open(f + ".tmp", "wb"))
    os.rename(f + ".tmp", f); return f


def all_jobs(L):
    J = []
    for si, s in enumerate(SEEDS):                    # seed-major so partial results cover every landscape
        for l in L:
            J += [(l, "random", s, 0.0), (l, "random", s, 0.05)]
            if si < 3: J += [(l, "W1", s, 0.0), (l, "FPS", s, 0.0), (l, "W2x", s, 0.0)]
            J += [(l, "W1", s, 0.05), (l, "FPS", s, 0.05), (l, "W2x", s, 0.05)]
            if si == 0: J += [(l, f"W0_{d}", 4100, 0.0) for d in range(10)]
            if si < 2: J += [(l, f"W0_{d}", s, 0.05) for d in range(10)]
    return J


if __name__ == "__main__":
    from meta.stage2 import landscapes
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    L = landscapes(); J = all_jobs(L); print(len(L), "landscapes,", len(J), "campaigns", flush=True)
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(job, J, chunksize=2):
            k += 1
            if k % 200 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)

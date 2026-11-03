"""EXPLORATORY, post hoc diagnostic after the T1 stop rule (PLAN_source_prior.md): which part of the SP surrogate hurts? Tuning set only (never the test landscapes), seeds 4500-4501.
Arms: b0 (prior switched off: b = 0, c irrelevant; should reproduce W1 = engine check), uniform with c = 0 and c = 0.25 (c = 1 exists in sp_tune), selected SP (kappa 10, lambda .25) with c = 0 and c = 0.25.
Run: python -m meta.sp_diag [workers]"""
import sys, os, time, json, pickle, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "sp_diag"); os.makedirs(OUT, exist_ok=True)
ARMS = {"b0": dict(mode="uniform", c=0.0, force_b0=True), "uni_c0": dict(mode="uniform", c=0.0), "uni_c.25": dict(mode="uniform", c=0.25),
        "sp_c0": dict(mode="sp", c=0.0, kappa=10.0, lam=0.25), "sp_c.25": dict(mode="sp", c=0.25, kappa=10.0, lam=0.25)}


def unit(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo, sp
    from meta.w_expand import train_pools
    ln, seed = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{seed}.pkl")
    if os.path.exists(f): return f
    p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p); tp = train_pools(ln); S = bo._score_matrix(tp); port = bo.portfolio_greedy(tp, n0)
    res = dict(landscape=ln, seed=seed, auc={})
    for k, kw in ARMS.items(): res["auc"][k] = sp.run_sp(p, S, seed, budget=B, n_init=n0, init_idx=port, **kw)["auc"]
    pickle.dump(res, open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    T = json.load(open(os.path.join(H, "results", "sp_tuning_set.json"))); L = T["tabrepo"] + T["lcbench"]
    J = [(l, s) for s in (4500, 4501) for l in L]
    with ProcessPoolExecutor(W) as ex: list(ex.map(unit, J, chunksize=1))
    print("done", len(J), flush=True)
    base = {}
    for f in glob.glob(os.path.join(H, "results", "sp_tune", "*.pkl")):
        r = pickle.load(open(f, "rb")); base.setdefault(r["landscape"], []).append(r)
    D = {}
    for f in glob.glob(os.path.join(OUT, "*.pkl")):
        r = pickle.load(open(f, "rb")); D.setdefault(r["landscape"], []).append(r)
    out = []
    def P(*a): s = " ".join(str(x) for x in a); out.append(s); print(s)
    for nm, ls in (("TabRepo", T["tabrepo"]), ("LCBench", T["lcbench"])):
        w1 = np.mean([np.mean([u["auc"]["W1"] for u in base[l]]) for l in ls]); P(f"\n{nm}: W1 {w1:.4f}")
        rows = [("uniform c=1 (T1)", np.mean([np.mean([u["auc"]["uniform"] for u in base[l]]) for l in ls])),
                ("sp k10 l.25 c=1 (T1)", np.mean([np.mean([u["auc"][(10.0, 0.25)] for u in base[l]]) for l in ls]))]
        rows += [(k, np.mean([np.mean([u["auc"][k] for u in D[l]]) for l in ls])) for k in ARMS]
        for k, v in rows: P(f"   {k:22s} {v:.4f}  ({v/w1-1:+.1%} vs W1)")
    open(os.path.join(H, "results", "sp_diag.txt"), "w").write("\n".join(out) + "\n")

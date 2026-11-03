"""Fall-back triage, step 1: portfolio start (W1) vs random start on ~150 exact-grid landscapes (30 LCBench dev + 20 per TabRepo model family x 6, fixed-seed draw).
Noise 0: W1 1 run (deterministic), random 8 seeds (4200-4207). Noise 5 %: W1 3 seeds (4200-4202), random 8 seeds. Logs trust features computed from the n0 portfolio observations
(available before any further decision): rank agreement with the source ranking, R^2 of observed values on the source score, GP fit after the portfolio (leave-one-out error,
lengthscales), skew and standardised spread of observed values.  Run: python -m meta.w150 [workers]"""
import sys, os, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "w150"); os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(4200, 4208))


def landscapes():
    from meta.landscapes import kept
    k = kept(("lcbench", "tabrepo"))
    L = k[k.family == "lcbench"].landscape.tolist()
    for g, h in k[k.family == "tabrepo"].groupby("group"):
        L += h.sample(min(20, len(h)), random_state=31).landscape.tolist()
    return L


def trust_features(p, init, S):
    import torch
    from scipy.stats import spearmanr
    from meta import bo
    yo = p.y[init]; s = S[init]; z = (yo - yo.mean()) / (yo.std() + 1e-12)
    f = {"obs_cv": float(yo.std() / (abs(yo.mean()) + 1e-12)), "obs_skew": float((z ** 3).mean()), "obs_max_z": float(z.max())}
    f["spearman_src"] = float(spearmanr(yo, s)[0]) if np.ptp(yo) > 0 and np.ptp(s) > 0 else float("nan")
    A = np.column_stack([s, np.ones_like(s)]); coef, *_ = np.linalg.lstsq(A, yo, rcond=None); res = yo - A @ coef
    f["r2_src"] = float(1 - res.var() / (yo.var() + 1e-12)) if yo.var() > 0 else float("nan")
    f["best_is_src_top"] = float(np.argmax(yo) == np.argmax(s))
    m = bo._gp(p.X[init], yo, 0, "rbf_ard"); ys = torch.tensor((yo - yo.mean()) / (yo.std(ddof=1) + 1e-12))
    with torch.no_grad():
        K = m.covar_module(torch.tensor(p.X[init])).evaluate() + m.likelihood.noise.reshape(-1)[0] * torch.eye(len(init))
        Ki = torch.linalg.inv(K); loo = ((Ki @ ys) / torch.diag(Ki)).numpy()
        ls = m.covar_module.lengthscale.reshape(-1).numpy()
    f.update(loo_rmse=float(np.sqrt((loo ** 2).mean())), ls_mean=float(ls.mean()), ls_min=float(ls.min()), gp_noise=float(m.likelihood.noise.reshape(-1)[0]))
    return f


def job(a):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta import bo
    from meta.w_expand import train_pools
    ln, arm, seed, noise = a
    f = os.path.join(OUT, f"{ln.replace('/', '__')}__{arm}__{seed}__{noise}.pkl")
    if os.path.exists(f): return f
    p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = budget(p); extra = {}
    if arm == "W1":
        tp = train_pools(ln)
        if len(tp) < 8: pickle.dump(dict(landscape=ln, arm=arm, seed=seed, noise=noise, skipped=True), open(f, "wb")); return f
        init = bo.portfolio_greedy(tp, n0); S = bo._score_matrix(tp).mean(0)
        extra = dict(ntrain=len(tp), **(trust_features(p, init, S) if noise == 0.0 else {}))
    else: init = None
    r = bo.run(p, seed, budget=B, n_init=n0, init_idx=init, noise_frac=noise)
    pickle.dump(dict(landscape=ln, arm=arm, seed=seed, noise=noise, auc=r["auc"], final=r["final"], skipped=False, **extra), open(f + ".tmp", "wb")); os.rename(f + ".tmp", f); return f


if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    L = landscapes(); print(len(L), "landscapes", flush=True)
    J = []
    for l in L:
        J += [(l, "W1", SEEDS[0], 0.0)] + [(l, "W1", s, 0.05) for s in SEEDS[:3]] + [(l, "random", s, n) for s in SEEDS for n in (0.0, 0.05)]
    J.sort(key=lambda j: (j[2], j[0]))
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(job, J, chunksize=2):
            k += 1
            if k % 300 == 0: print(f"{k}/{len(J)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(J)} {time.time()-t0:.0f}s", flush=True)

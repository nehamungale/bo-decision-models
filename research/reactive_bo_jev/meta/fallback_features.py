"""Fall-back triage step 2: features for the portfolio-trust decision, from source data only (SRC, available before any evaluation) and from the target's n0 portfolio observations (TRG).
SRC: leave-one-source-task-out success of the portfolio start against the expected random start, over up to 40 source tasks; mean start regret; spread; number of source tasks.
TRG: the trust features logged in results/w150 plus similarity of the target's observed values to each source task at the same configs (max, mean of top 5, mean).
Writes meta/results/fallback_features.csv. Run: python -m meta.fallback_features"""
import os, sys, pickle, glob, zlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meta.landscapes import load, kept
from meta.w150 import landscapes
H = os.path.dirname(os.path.abspath(__file__))


def greedy_S(S, n0):
    chosen = []; cur = np.full(S.shape[0], -np.inf)
    for _ in range(n0):
        gain = np.maximum(S, cur[:, None]).mean(0); gain[chosen] = -np.inf
        j = int(np.argmax(gain)); chosen.append(j); cur = np.maximum(cur, S[:, j])
    return chosen


def main():
    k = kept(("lcbench", "tabrepo")); cache = {}; rows = []
    for ln in landscapes():
        row = k[k.landscape == ln].iloc[0]; p = load(ln); n0 = 2 * p.X.shape[1] + 1
        key = row.group
        if key not in cache:
            names, S = [], []
            for l in k[k.group == key].landscape:
                q = load(l)
                if q.X.shape == p.X.shape and np.allclose(q.X, p.X): names.append(l); S.append((q.y - q.y_med) / max(q.y_max - q.y_med, 1e-12))
            cache[key] = (names, np.stack(S))
        names, S = cache[key]; ti = names.index(ln); keep = np.array([i for i in range(len(names)) if i != ti]); Ss = S[keep]
        rng = np.random.default_rng(__import__('zlib').crc32(ln.encode())); pick = rng.choice(len(Ss), min(40, len(Ss)), replace=False)
        succ, sreg, ratio = [], [], []
        for t in pick:
            tr = np.delete(np.arange(len(Ss)), t); init = greedy_S(Ss[tr], n0)
            reg = 1 - Ss[t, init].max()
            rr = np.mean([1 - Ss[t, rng.choice(Ss.shape[1], n0, replace=False)].max() for _ in range(100)])
            succ.append(reg < rr); sreg.append(reg); ratio.append(reg / (rr + 1e-9))
        init = greedy_S(Ss, n0); best_src = Ss[:, init].max(1)
        yo = p.y[init]; sims = np.array([spearmanr(yo, Ss[i, init])[0] if np.ptp(Ss[i, init]) > 0 and np.ptp(yo) > 0 else np.nan for i in range(len(Ss))])
        sims = sims[~np.isnan(sims)]
        r = dict(landscape=ln, family=ln.split("/")[0].split("_")[0] + ("" if ln.startswith("lcbench") else "_" + ln.split("/")[0].split("_", 1)[1]),
                 dataset=ln.split("/")[1], n_train=len(Ss), src_success=float(np.mean(succ)), src_mean_start_regret=float(np.mean(sreg)), src_median_ratio=float(np.median(ratio)),
                 src_best_mean=float(best_src.mean()), src_best_std=float(best_src.std()),
                 sim_max=float(sims.max()) if len(sims) else np.nan, sim_top5=float(np.sort(sims)[-5:].mean()) if len(sims) else np.nan, sim_mean=float(sims.mean()) if len(sims) else np.nan)
        f = os.path.join(H, "results", "w150", f"{ln.replace('/', '__')}__W1__4200__0.0.pkl"); t = pickle.load(open(f, "rb"))
        for kk in ("spearman_src", "r2_src", "obs_cv", "obs_skew", "obs_max_z", "best_is_src_top", "loo_rmse", "ls_mean", "ls_min", "gp_noise"): r[kk] = t.get(kk, np.nan)
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(H, "results", "fallback_features.csv"), index=False); print("wrote", len(rows))


if __name__ == "__main__":
    main()

"""T2 features for PLAN_fallback_detection.md. Per landscape (seed-averaged over the 4 W1 seeds), from the first n0 + K = n0 + 10 observations of the stored W1 runs only:
spearman_k / r2_k (observed y vs source score), gain_k (incumbent gain over the best portfolio point, in std of observed y), inc_z_k (incumbent z among observed), since_k (steps since incumbent changed within K),
loo_k, ls_mean_k, ls_min_k, gpn_k (GP fit on the n0+K points); ref_src_pct (percentile of W1's inc_z among random-subset inc_z of up to 40 source tasks; deployable);
ref_tgt_pct (percentile of W1's incumbent y among the 12 random-start runs' incumbent y at the same evaluation count; uses the target's own random runs, an upper bound, not deployable).
Writes meta/results/detect_features.csv. Run: python -m meta.detect_features"""
import os, sys, glob, pickle, zlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meta.landscapes import load, kept
from meta.w150 import landscapes
from meta import bo
import torch
H = os.path.dirname(os.path.abspath(__file__)); K = 10


def gpfit(X, y):
    m = bo._gp(X, y, 0, "rbf_ard"); ys = torch.tensor((y - y.mean()) / (y.std(ddof=1) + 1e-12))
    with torch.no_grad():
        Kk = m.covar_module(torch.tensor(X)).evaluate() + m.likelihood.noise.reshape(-1)[0] * torch.eye(len(y))
        Ki = torch.linalg.inv(Kk); loo = ((Ki @ ys) / torch.diag(Ki)).numpy(); ls = m.covar_module.lengthscale.reshape(-1).numpy()
    return float(np.sqrt((loo ** 2).mean())), float(ls.mean()), float(ls.min()), float(m.likelihood.noise.reshape(-1)[0])


def main():
    k = kept(("lcbench", "tabrepo")); cache = {}; rows = []
    for ln in landscapes():
        row = k[k.landscape == ln].iloc[0]; p = load(ln); key = row.group
        if key not in cache:
            names, S = [], []
            for l in k[k.group == key].landscape:
                q = load(l)
                if q.X.shape == p.X.shape and np.allclose(q.X, p.X): names.append(l); S.append((q.y - q.y_med) / max(q.y_max - q.y_med, 1e-12))
            cache[key] = (names, np.stack(S), {l: load(l).y for l in names})
        names, S, Y = cache[key]; ti = names.index(ln); keep = [i for i in range(len(names)) if i != ti]; Ss = S[keep]; src = Ss.mean(0)
        units = [pickle.load(open(os.path.join(H, "results", "detect", f"{ln.replace('/', '__')}__{s}.pkl"), "rb")) for s in range(4300, 4304)]
        n0 = units[0]["n0"]; rng = np.random.default_rng(zlib.crc32(ln.encode()))
        # source reference: inc_z of random subsets of n0+K configs from other tasks
        ref = []
        for i in rng.choice(len(keep), min(40, len(keep)), replace=False):
            y = Y[names[keep[i]]]
            for _ in range(5):
                sub = y[rng.choice(len(y), n0 + K, replace=False)]; ref.append((sub.max() - sub.mean()) / (sub.std() + 1e-12))
        ref = np.array(ref)
        per = []
        for u in units:
            a = u["arms"]["W1"]; idx = np.array(a["obs_idx"][: n0 + K]); y = np.array(a["y_obs"][: n0 + K]); s = src[idx]
            f = dict(spearman_k=spearmanr(y, s)[0] if np.ptp(y) > 0 and np.ptp(s) > 0 else np.nan)
            A = np.column_stack([s, np.ones_like(s)]); coef, *_ = np.linalg.lstsq(A, y, rcond=None); f["r2_k"] = float(1 - (y - A @ coef).var() / (y.var() + 1e-12))
            sd = y.std() + 1e-12; f["gain_k"] = float((y.max() - y[:n0].max()) / sd); f["inc_z_k"] = float((y.max() - y.mean()) / sd)
            best = np.maximum.accumulate(y[n0:]); ch = np.where(np.diff(np.concatenate([[y[:n0].max()], best])) > 0)[0]
            f["since_k"] = float(K - 1 - ch[-1]) if len(ch) else float(K)
            f["loo_k"], f["ls_mean_k"], f["ls_min_k"], f["gpn_k"] = gpfit(p.X[idx], y)
            f["ref_src_pct"] = float((ref < f["inc_z_k"]).mean())
            rinc = np.array([max(U["arms"]["random"]["y_obs"][: n0 + K]) for U in (pickle.load(open(os.path.join(H, "results", "detect", f"{ln.replace('/', '__')}__{s_}.pkl"), "rb")) for s_ in range(4300, 4312))])
            f["ref_tgt_pct"] = float((rinc < y.max()).mean() + 0.5 * (rinc == y.max()).mean())
            per.append(f)
        r = {"landscape": ln}; r.update(pd.DataFrame(per).mean(numeric_only=True).to_dict()); rows.append(r)
        print(ln, flush=True) if len(rows) % 25 == 0 else None
    pd.DataFrame(rows).to_csv(os.path.join(H, "results", "detect_features.csv"), index=False); print("wrote", len(rows))


if __name__ == "__main__":
    main()

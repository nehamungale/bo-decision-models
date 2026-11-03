"""Similarity-weighted source prior (PLAN_source_prior.md). Separate from meta/bo.py so the default engine is untouched.
Target value = a + b * m(x) + GP residual, m = weighted source score; predictive variance = GP variance + c * b^2 * v(x), v = weighted source spread.
Modes: sp (weights updated every step), frozen (weights from the initial observations, then fixed), uniform, shuffled (sources permuted across configs, then sp), permw (sp weights at step 0 permuted over sources once, then fixed)."""
import numpy as np
from scipy.stats import rankdata
from meta import bo

MODES = ["sp", "frozen", "uniform", "shuffled", "permw"]


def weights(S, obs_idx, y_obs, kappa, lam):
    """w = (1 - lam) * uniform + lam * softmax(kappa * Spearman(S_i at observed configs, observed y))."""
    N = S.shape[0]; ry = rankdata(y_obs); ry = ry - ry.mean()
    R = np.apply_along_axis(rankdata, 1, S[:, obs_idx]); R = R - R.mean(1, keepdims=True)
    den = np.linalg.norm(R, axis=1) * np.linalg.norm(ry)
    rho = np.where(den > 0, (R @ ry) / np.where(den > 0, den, 1.0), 0.0)
    z = kappa * rho; e = np.exp(z - z.max()); sm = e / e.sum()
    return (1 - lam) / N + lam * sm, rho


def prior_moments(S, w):
    m = w @ S; v = np.maximum(w @ (S ** 2) - m ** 2, 0.0)
    return m, v


def predict(S, w, X, obs_idx, y_obs, cand, seed, t, c, force_b0=False):
    """Return mu, sd on candidates and the fitted (a, b)."""
    m, v = prior_moments(S, w); mo = m[obs_idx]
    if np.ptp(mo) > 1e-12 and not force_b0:
        A = np.column_stack([mo, np.ones_like(mo)]); (b, a), *_ = np.linalg.lstsq(A, y_obs, rcond=None)
        if b < 0: b, a = 0.0, float(y_obs.mean())
    else: b, a = 0.0, float(y_obs.mean())
    r = y_obs - (a + b * mo)
    if np.std(r) < 1e-9 * (np.std(y_obs) + 1e-12) or np.std(r) < 1e-12:
        mu_r = np.zeros(len(cand)); sd_r = np.full(len(cand), 1e-3 * (np.std(y_obs) + 1e-12))
    else:
        import torch
        model = bo._gp(X[obs_idx], r, seed * 100003 + t, "rbf_ard")
        Xc = torch.tensor(X[cand])
        with torch.no_grad():
            post = model.posterior(Xc.unsqueeze(1)); mu_r = post.mean.reshape(-1).numpy(); sd_r = post.variance.clamp_min(1e-18).sqrt().reshape(-1).numpy()
    mu = a + b * m[cand] + mu_r; sd = np.sqrt(sd_r ** 2 + c * (b ** 2) * v[cand])
    return mu, sd, sd_r, (a, b)


def run_sp(pool, S, seed, mode="sp", budget=50, n_init=None, init_idx=None, kappa=10.0, lam=0.5, c=1.0, noise_frac=0.0, force_b0=False):
    """One campaign. S: sources x configs (normalised scores on the target's grid). Returns the regret trace plus diagnostics (weights, effective number of sources, volatility)."""
    assert mode in MODES
    n_init = n_init if n_init is not None else 2 * pool.X.shape[1] + 1
    rng = np.random.default_rng(seed)
    obs_idx = list(rng.choice(pool.X.shape[0], n_init, replace=False)) if init_idx is None else list(init_idx)
    span = max(pool.y_max - pool.y_med, 1e-12)
    nrng = np.random.default_rng(seed + 424242)
    noise = nrng.normal(0.0, noise_frac * span, pool.X.shape[0]) if noise_frac > 0 else np.zeros(pool.X.shape[0])
    obs = lambda i: pool.y[i] + noise[i]
    y_obs = [obs(i) for i in obs_idx]
    Su = S
    if mode == "shuffled":
        prng = np.random.default_rng(seed + 777); Su = np.stack([row[prng.permutation(len(row))] for row in S])
    N = S.shape[0]; fixed_w = None; trace = []; W = []; eff = []; tv = []; prev = None
    for t in range(budget):
        ia = np.array(obs_idx); ya = np.array(y_obs)
        mask = np.ones(pool.X.shape[0], bool); mask[ia] = False; cand = np.where(mask)[0]
        if mode == "uniform": w = np.full(N, 1.0 / N)
        elif mode in ("frozen", "permw"):
            if fixed_w is None:
                w0, _ = weights(Su, ia, ya, kappa, lam)
                fixed_w = w0 if mode == "frozen" else w0[np.random.default_rng(seed + 555).permutation(N)]
            w = fixed_w
        else: w, _ = weights(Su, ia, ya, kappa, lam)
        W.append(w.copy()); eff.append(float(np.exp(-(w * np.log(np.maximum(w, 1e-300))).sum()))); tv.append(0.5 * float(np.abs(w - prev).sum()) if prev is not None else 0.0); prev = w
        mu, sd, sd_r, ab = predict(Su, w, pool.X, ia, ya, cand, seed, t, c, force_b0)
        s = bo._log_ei_gauss(mu, np.maximum(sd, 1e-12), float(ya.max()))
        chosen = int(cand[int(np.argmax(s))]); obs_idx.append(chosen); y_obs.append(obs(chosen))
        trace.append(pool.regret(float(pool.y[obs_idx[int(np.argmax(y_obs))]])))
    reg = np.array(trace)
    return dict(regret=reg, auc=float(reg.mean()), final=float(reg[-1]), obs_idx=[int(i) for i in obs_idx], weights=np.array(W), eff_sources=np.array(eff), volatility=np.array(tv))


def source_matrix(landscape):
    """Normalised score matrix (sources x configs) of the training tasks of `landscape` (held-out task excluded) and the W1 portfolio; None if fewer than 8 sources."""
    from meta.w_expand import train_pools
    tp = train_pools(landscape)
    return None if len(tp) < 8 else bo._score_matrix(tp)

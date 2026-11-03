"""Component-triage BO engine (PLAN_component_triage.md): LogEI campaigns with a swappable surrogate (K), search region (R), initial design (W) and observation noise.
With default arguments it reproduces acq.engine.run_campaign(pool, seed, Static("LogEI")) exactly (tested in meta/test_bo.py)."""
import numpy as np
import torch
from scipy.special import log_ndtr
from scipy.stats import norm
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.models.utils.gpytorch_modules import SQRT2, SQRT3
from gpytorch.kernels import RBFKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.ensemble import RandomForestRegressor
from math import log

KERNELS = ["rbf_ard", "matern52_ard", "rbf_iso", "rf"]


def _gp(X, y, seed, kernel):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        Xt = torch.tensor(X); Yt = torch.tensor(y).unsqueeze(-1); D = X.shape[1]
        if kernel == "rbf_ard":
            m = SingleTaskGP(Xt, Yt, outcome_transform=Standardize(m=1))
        elif kernel == "matern52_ard":
            from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
            m = SingleTaskGP(Xt, Yt, covar_module=get_covar_module_with_dim_scaled_prior(D, use_rbf_kernel=False), outcome_transform=Standardize(m=1))
        elif kernel == "rbf_iso":
            pr = LogNormalPrior(loc=SQRT2 + log(D) * 0.5, scale=SQRT3)
            k = RBFKernel(lengthscale_prior=pr, lengthscale_constraint=GreaterThan(2.5e-2, transform=None, initial_value=pr.mode))
            m = SingleTaskGP(Xt, Yt, covar_module=k, outcome_transform=Standardize(m=1))
        else:
            raise ValueError(kernel)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    return m.eval()


def _log_ei_gauss(mu, sd, best):
    """Stable log EI for a Gaussian predictive (used by the RF surrogate)."""
    z = (mu - best) / sd
    out = np.empty_like(z)
    big = z > -5
    h = norm.pdf(z[big]) + z[big] * norm.cdf(z[big]); out[big] = np.log(np.maximum(h, 1e-300))
    zs = z[~big]; out[~big] = -0.5 * zs ** 2 - 0.5 * np.log(2 * np.pi) - 2 * np.log(-zs)
    return out + np.log(sd)


def scores(pool, obs_idx, y_obs, cand, seed, t, kernel):
    """LogEI over candidates, plus posterior mean/sd."""
    best = float(np.max(y_obs))
    if kernel == "rf":
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=1, max_features=1.0, random_state=int(seed * 100003 + t) % (2 ** 31), n_jobs=1)
        rf.fit(pool.X[obs_idx], y_obs)
        P = np.stack([e.predict(pool.X[cand]) for e in rf.estimators_])
        mu = P.mean(0); sd = np.maximum(P.std(0), 1e-3 * (np.std(y_obs) + 1e-12))
        return _log_ei_gauss(mu, sd, best), mu, sd
    model = _gp(pool.X[obs_idx], y_obs, seed * 100003 + t, kernel)
    Xc = torch.tensor(pool.X[cand])
    with torch.no_grad():
        post = model.posterior(Xc.unsqueeze(1))
        mu = post.mean.reshape(-1).numpy(); sd = post.variance.clamp_min(1e-18).sqrt().reshape(-1).numpy()
        s = LogExpectedImprovement(model, best_f=best)(Xc.unsqueeze(1)).numpy()
    return s, mu, sd


class TrustBox:
    """TuRBO-style adaptive box: half-length L around the incumbent; grows after `succ_tol` successes, shrinks after `fail_tol` failures; restart when too small."""
    def __init__(self, L0=0.4, succ_tol=3, fail_tol=4, L_min=0.03):
        self.L0, self.L, self.s, self.f, self.succ_tol, self.fail_tol, self.L_min = L0, L0, 0, 0, succ_tol, fail_tol, L_min
    def update(self, improved):
        if improved: self.s += 1; self.f = 0
        else: self.f += 1; self.s = 0
        if self.s >= self.succ_tol: self.L = min(self.L * 2, 1.0); self.s = 0
        if self.f >= self.fail_tol: self.L = self.L / 2; self.f = 0
        if self.L < self.L_min: self.L = self.L0


def run(pool, seed, budget=50, n_init=None, kernel="rbf_ard", region=None, init_idx=None, noise_frac=0.0, inject=None):
    """region: None (global), a float radius (fixed box half-length), or "adaptive". noise_frac: observation noise sd as a fraction of (y_max - y_median).
    inject: None or (k, m, mode): at BO step k, evaluate m extra points chosen by mode ("random": uniform without replacement from the unevaluated points;
    "fps": farthest from the observed set, max-min distance) instead of m BO steps, so the total number of evaluations (n_init + budget) is unchanged.
    Steps before k are identical to the run without injection. Regret is measured on the true value of the recommended point (the one with the best *observed* value)."""
    n_init = n_init if n_init is not None else 2 * pool.X.shape[1] + 1
    rng = np.random.default_rng(seed)
    obs_idx = list(rng.choice(pool.X.shape[0], n_init, replace=False)) if init_idx is None else list(init_idx)
    span = max(pool.y_max - pool.y_med, 1e-12)
    nrng = np.random.default_rng(seed + 424242)
    noise = nrng.normal(0.0, noise_frac * span, pool.X.shape[0]) if noise_frac > 0 else np.zeros(pool.X.shape[0])   # one fixed draw per point
    obs = lambda i: pool.y[i] + noise[i]
    y_obs = [obs(i) for i in obs_idx]
    box = TrustBox() if region == "adaptive" else None
    trace = []
    if inject is not None:
        k0, m_inj, mode = inject; assert mode in ("random", "fps") and 0 <= k0 and k0 + m_inj <= budget
        inj_rng = np.random.default_rng(seed + 999)
    for t in range(budget):
        ia = np.array(obs_idx); ya = np.array(y_obs)
        mask = np.ones(pool.X.shape[0], bool); mask[ia] = False
        cand = np.where(mask)[0]
        if inject is not None and k0 <= t < k0 + m_inj:
            if mode == "random": chosen = int(inj_rng.choice(cand))
            else: chosen = int(cand[int(np.argmax(np.min(np.linalg.norm(pool.X[cand][:, None, :] - pool.X[ia][None, :, :], axis=2), axis=1)))])
            obs_idx.append(chosen); y_obs.append(obs(chosen))
            trace.append(pool.regret(float(pool.y[obs_idx[int(np.argmax(y_obs))]])))
            continue
        s, mu, sd = scores(pool, ia, ya, cand, seed, t, kernel)
        if region is not None:
            half = box.L if box is not None else float(region)
            inc = pool.X[ia[int(np.argmax(ya))]]
            inside = np.max(np.abs(pool.X[cand] - inc), axis=1) <= half
            if inside.any(): s = np.where(inside, s, -np.inf)
        k = int(np.argmax(s)); chosen = int(cand[k]); yn = obs(chosen)
        improved = yn > ya.max()
        if box is not None: box.update(improved)
        obs_idx.append(chosen); y_obs.append(yn)
        rec = obs_idx[int(np.argmax(y_obs))]
        trace.append(pool.regret(float(pool.y[rec])))
    reg = np.array(trace)
    return dict(regret=reg, auc=float(reg.mean()), final=float(reg[-1]), obs_idx=[int(i) for i in obs_idx], y_obs=[float(v) for v in y_obs])


# ── warm start (W) ───────────────────────────────────────────────────────────────────────────────

def _score_matrix(pools):
    return np.stack([(p.y - p.y_med) / max(p.y_max - p.y_med, 1e-12) for p in pools])      # tasks x configs


def portfolio_rank(pools, n0):
    """W1a: top-n0 configs by mean normalised score over the training tasks."""
    S = _score_matrix(pools); return list(np.argsort(-S.mean(0))[:n0])


def portfolio_greedy(pools, n0):
    """W1b (primary): greedy forward selection maximising the mean over training tasks of the best score within the set."""
    S = _score_matrix(pools); chosen = []; cur = np.full(S.shape[0], -np.inf)
    for _ in range(n0):
        gain = np.maximum(S, cur[:, None]).mean(0); gain[chosen] = -np.inf
        j = int(np.argmax(gain)); chosen.append(j); cur = np.maximum(cur, S[:, j])
    return chosen


def portfolio_region(train_X_list, train_y_list, target_X, n0, top_frac=0.05, seed=0):
    """W2 (HPO-B, different config sets per task): pool the best `top_frac` points of each training task, cluster into n0 centres (k-means),
    start from the distinct target-pool points nearest to the centres."""
    from sklearn.cluster import KMeans
    pts = []
    for X, y in zip(train_X_list, train_y_list):
        k = max(1, int(round(top_frac * len(y)))); pts.append(X[np.argsort(-y)[:k]])
    pts = np.vstack(pts)
    km = KMeans(n_clusters=n0, n_init=5, random_state=seed).fit(pts)
    chosen = []
    for c in km.cluster_centers_:
        d = np.linalg.norm(target_X - c, axis=1); d[chosen] = np.inf; chosen.append(int(np.argmin(d)))
    return chosen


def farthest_point_start(X, n0, seed):
    """Space-filling control: random first point, then repeatedly the point farthest from the chosen set."""
    rng = np.random.default_rng(seed); chosen = [int(rng.integers(len(X)))]
    dmin = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < n0:
        j = int(np.argmax(dmin)); chosen.append(j); dmin = np.minimum(dmin, np.linalg.norm(X - X[j], axis=1))
    return chosen


def hedge_start(n_pool, portfolio, n0, seed):
    """Hedge initial design: the first ceil(n0/2) portfolio points plus floor(n0/2) uniformly random other points (same n0)."""
    h = list(portfolio[: (n0 + 1) // 2]); rng = np.random.default_rng(seed + 31337)
    rest = np.setdiff1d(np.arange(n_pool), h)
    return h + [int(i) for i in rng.choice(rest, n0 - len(h), replace=False)]

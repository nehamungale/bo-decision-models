"""Counterfactual-rollout labels (PLAN_meta_selector.md section 4) and landscape descriptors.

At a logged state, for each AF a: evaluate a's proposal, then continue H-1 steps with a fixed continuation AF.
Label V_a = (incumbent after - incumbent before) / (y_max - y_median). Proposals shared by several AFs share one rollout."""
import numpy as np
import torch
from acq.engine import AFS, propose_all, run_campaign, RandomAF, Static, CAND_KEYS
from scipy.stats import norm as _norm

BEHAVIOURS = ["random"] + AFS


def rollout(pool, ia, ya, first, seed, t, H, cont_af):
    ia = np.append(ia, first); ya = np.append(ya, pool.y[first])
    for j in range(1, H):
        _, cand, _, _, props = propose_all(pool, ia, ya, seed, t + j)
        p = props[cont_af]; ia = np.append(ia, p); ya = np.append(ya, pool.y[p])
    return float(ya.max())


def labels(step, H=4, conts=(0, 2), seed=0):
    """V[c][a] for each continuation AF index c (0 = LogEI, 2 = UCB) and each AF a."""
    pool, ia, ya = step.pool, step.obs_idx, step.y_obs
    span = max(pool.y_max - pool.y_med, 1e-12); before = float(ya.max())
    out = {}
    for c in conts:
        cache = {}
        v = []
        for p in step.proposals:
            if p not in cache: cache[p] = rollout(pool, ia, ya, p, seed, step.t, H, c)
            v.append((cache[p] - before) / span)
        out[c] = v
    return out


def descriptors(step):
    """Landscape descriptors from the campaign so far (beyond step.state): GP LOO error, noise-to-signal, local structure."""
    m, ia, ya, pool = step.model, step.obs_idx, step.y_obs, step.pool
    X = m.train_inputs[0]; yt = m.train_targets
    with torch.no_grad():
        K = m.covar_module(X).evaluate() + m.likelihood.noise.reshape(-1)[0] * torch.eye(len(X))
        Ki = torch.linalg.inv(K); alpha = Ki @ yt
        loo = (alpha / torch.diag(Ki)).numpy()                       # LOO residuals in standardised units
    inc = pool.X[ia[int(np.argmax(ya))]]
    d_inc = np.linalg.norm(pool.X[ia] - inc, axis=1)
    near = float((d_inc < 0.25 * np.sqrt(pool.X.shape[1])).mean())
    z = (ya - ya.mean()) / (ya.std() + 1e-12)
    loc = z[d_inc <= np.median(d_inc)].mean() if (d_inc <= np.median(d_inc)).any() else 0.0
    return dict(dim=pool.X.shape[1], n_pool=len(pool.y), loo_rmse=float(np.sqrt(np.mean(loo ** 2))), loo_max=float(np.abs(loo).max()),
                noise_to_signal=float(step.state["noise"]), near_inc_frac=near, local_vs_global=float(loc),
                y_skew=float(((z ** 3).mean())), y_range_ratio=float((ya.max() - ya.min()) / (ya.std() + 1e-12)))


def cand_features(step):
    """Nine candidate numbers per AF proposal (same as run_campaign(record=True))."""
    pool, ia, ya, mu, sd, cand = step.pool, step.obs_idx, step.y_obs, step.mu, step.sd, step.cand
    best_y = float(ya.max()); ystd = float(ya.std() + 1e-12); inc = pool.X[ia[int(np.argmax(ya))]]; Xe = pool.X[ia]
    rows = []
    for p_ in step.proposals:
        k_ = int(np.where(cand == p_)[0][0]); m_, s_ = mu[k_], sd[k_]; z_ = (m_ - best_y) / s_
        rows.append([(m_ - best_y) / ystd, s_ / ystd, z_, float(_norm.cdf(z_)), float(s_ * (_norm.pdf(z_) + z_ * _norm.cdf(z_)) / ystd),
                     float(np.linalg.norm(pool.X[p_] - inc)), float(np.linalg.norm(Xe - pool.X[p_], axis=1).min()),
                     float((mu < m_).mean()), float((sd < s_).mean())])
    return rows


class LabelCollector:
    """Behaviour policy that also records labelled states every `every` steps. `behaviour` in BEHAVIOURS."""
    def __init__(self, behaviour, H=4, every=3, conts=(0, 2), budget=50):
        self.beh = RandomAF() if behaviour == "random" else Static(behaviour)
        self.name = f"collect_{behaviour}"; self.H, self.every, self.conts, self.budget = H, every, conts, budget
        self.behaviour = behaviour
    def reset(self, seed):
        self.seed = seed; self.rows = []
        r = getattr(self.beh, "reset", None)
        if r: r(seed)
    def __call__(self, step, peek):
        a = self.beh(step, peek)
        if step.t % self.every == 0 and step.t + self.H <= self.budget:
            V = labels(step, self.H, self.conts, self.seed)
            self.rows.append(dict(t=step.t, behaviour=self.behaviour, state=step.state, cand=cand_features(step), desc=descriptors(step),
                                  proposals=list(step.proposals), V={int(c): v for c, v in V.items()}))
        return a

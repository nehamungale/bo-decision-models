"""Proposers and reaction policies. Policy signature: (state, run, metric, rng) -> (action, latency)."""
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=Warning, module="sklearn")
from .engine import CONTINUE, STOP


def random_proposer(st, rng):
    free = np.setdiff1d(np.arange(st.pool.n), np.fromiter(st.done_cfgs, int, len(st.done_cfgs)))
    return int(rng.choice(free))


def make_gp_rung_proposer(rung=10, n_init=8, n_cand=500):
    """GP-EI on the metric at epoch `rung` (observed by every run that reached it): a cheap
    multi-fidelity proxy so early-stopped runs still inform the surrogate."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from scipy.stats import norm

    def prop(st, rng):
        obs = [(c, r[rung - 1]) for c, r in st.reports.items() if len(r) >= rung]
        if len(obs) < n_init:
            return random_proposer(st, rng)
        X = st.pool.X[[c for c, _ in obs]]; y = np.array([v for _, v in obs])
        gp = GaussianProcessRegressor(C(1.0) * Matern(nu=2.5, length_scale=np.ones(X.shape[1]), length_scale_bounds=(0.05, 20.0)) + WhiteKernel(1e-3, noise_level_bounds=(1e-6, 1.0)),
                                      normalize_y=True, n_restarts_optimizer=0, random_state=int(rng.integers(1e9)))
        gp.fit(X, y)
        free = np.setdiff1d(np.arange(st.pool.n), np.fromiter(st.done_cfgs, int, len(st.done_cfgs)))
        cand = rng.choice(free, size=min(n_cand, len(free)), replace=False)
        mu, sd = gp.predict(st.pool.X[cand], return_std=True)
        z = (mu - y.max()) / np.maximum(sd, 1e-9)
        ei = (mu - y.max()) * norm.cdf(z) + sd * norm.pdf(z)
        return int(cand[np.argmax(ei)])
    return prop


def no_stop(st, run, metric, rng):
    return CONTINUE, 0.0


def make_asha(eta=3, min_epoch=1, latency=0.0):
    """ASHA stopping variant: at rung epochs min_epoch*eta^k, stop unless the metric is in the top
    1/eta of all values seen at that rung so far (continue while fewer than eta values exist)."""
    rungs = []; r = min_epoch
    while r < 50:
        rungs.append(r); r *= eta

    def pol(st, run, metric, rng):
        if run.epoch not in rungs:
            return CONTINUE, 0.0
        vals = np.array([rep[run.epoch - 1] for rep in st.reports.values() if len(rep) >= run.epoch])
        if len(vals) < eta:
            return CONTINUE, 0.0
        thresh = np.quantile(vals, 1 - 1 / eta)
        return (CONTINUE if metric >= thresh else STOP), latency
    return pol

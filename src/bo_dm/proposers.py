from typing import Protocol

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


class Proposer(Protocol):
    def propose(self, X: np.ndarray, y: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Return the next point to evaluate, given observations (minimization)."""


def _sample(rng: np.random.Generator, bounds: np.ndarray, n: int) -> np.ndarray:
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, len(bounds)))


class RandomProposer:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, X, y, bounds):
        return _sample(self.rng, bounds, 1)[0]


class GPEIProposer:
    """Decision model: GP posterior + expected improvement, maximized over candidates."""

    def __init__(self, seed: int = 0, n_candidates: int = 2000):
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates

    def propose(self, X, y, bounds):
        if len(X) < 2:
            return _sample(self.rng, bounds, 1)[0]
        gp = GaussianProcessRegressor(
            ConstantKernel() * Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=2,
            random_state=int(self.rng.integers(2**31)),
        ).fit(X, y)
        cand = _sample(self.rng, bounds, self.n_candidates)
        mu, sd = gp.predict(cand, return_std=True)
        sd = np.maximum(sd, 1e-12)
        z = (y.min() - mu) / sd
        ei = (y.min() - mu) * norm.cdf(z) + sd * norm.pdf(z)
        return cand[int(np.argmax(ei))]

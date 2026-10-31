from typing import Callable

import numpy as np

from .objectives import BRANIN_BOUNDS, BRANIN_MIN, branin
from .proposers import GPEIProposer, Proposer, RandomProposer


def run_bo(
    objective: Callable[[np.ndarray], float],
    proposer: Proposer,
    bounds: np.ndarray,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.empty((0, len(bounds)))
    y = np.empty(0)
    for _ in range(budget):
        x = proposer.propose(X, y, bounds)
        X = np.vstack([X, x])
        y = np.append(y, objective(x))
    return X, y


if __name__ == "__main__":
    for name, p in [("random", RandomProposer(0)), ("gp-ei", GPEIProposer(0))]:
        _, y = run_bo(branin, p, BRANIN_BOUNDS, budget=30)
        print(f"{name:8s} best regret: {y.min() - BRANIN_MIN:.4f}")

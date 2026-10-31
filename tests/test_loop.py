import numpy as np

from bo_dm.loop import run_bo
from bo_dm.objectives import BRANIN_BOUNDS, BRANIN_MIN, branin
from bo_dm.proposers import GPEIProposer, RandomProposer


def test_branin_minimum():
    assert abs(branin(np.array([np.pi, 2.275])) - BRANIN_MIN) < 1e-4


def test_loop_respects_budget_and_bounds():
    X, y = run_bo(branin, RandomProposer(1), BRANIN_BOUNDS, budget=5)
    assert X.shape == (5, 2) and y.shape == (5,)
    assert (X >= BRANIN_BOUNDS[:, 0]).all() and (X <= BRANIN_BOUNDS[:, 1]).all()


def test_gp_ei_beats_random():
    _, y_r = run_bo(branin, RandomProposer(0), BRANIN_BOUNDS, budget=25)
    _, y_g = run_bo(branin, GPEIProposer(0), BRANIN_BOUNDS, budget=25)
    assert y_g.min() < y_r.min()

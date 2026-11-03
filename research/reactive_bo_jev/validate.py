"""Step-2 sanity checks of the simulator core, before any decision model is added.
Expected, well-known behaviour: (1) more workers -> better quality at equal wall-clock;
(2) ASHA-style early stopping beats running everything to completion; (3) accounting is exact.
Run: python validate.py [n_seeds]"""
import sys
import numpy as np
from splits import DEV
from sim.lcbench_env import load_pool
from sim.engine import simulate
from sim.policies import random_proposer, make_gp_rung_proposer, no_stop, make_asha

n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
T_MAX = 300.0                         # epoch units; one full run = 50 units, so 4 workers ~ 24 full runs
CONDS = {
    "rand/no-stop K=1": (random_proposer, no_stop, 1),
    "rand/no-stop K=4": (random_proposer, no_stop, 4),
    "rand/no-stop K=8": (random_proposer, no_stop, 8),
    "rand/ASHA    K=4": (random_proposer, make_asha(), 4),
    "gp/ASHA      K=4": (make_gp_rung_proposer(), make_asha(), 4),
}
res = {k: [] for k in CONDS}
for ds in DEV:
    pool = load_pool(ds)
    best = pool.y[:, -1].max(); span = best - np.median(pool.y[:, -1])
    for k, (prop, pol, K) in CONDS.items():
        regs = []
        for s in range(n_seeds):
            out = simulate(pool, prop, pol, K, T_MAX, np.random.default_rng(1000 + s))
            assert out["worker_time"] <= K * T_MAX + 1e-9            # accounting check
            regs.append((best - max(out["final"], np.median(pool.y[:, -1]))) / span)  # normalised regret
        res[k].append(np.mean(regs))
    print(f"{ds:22s}", "  ".join(f"{i}={res[k][-1]:.3f}" for i, k in enumerate(CONDS)), flush=True)
print("\nmean normalised regret across dev datasets (lower is better; 0 = pool optimum, 1 = pool median):")
for i, k in enumerate(CONDS): print(f"  [{i}] {k}: {np.mean(res[k]):.3f}   per-dataset: {np.round(res[k], 3).tolist()}")

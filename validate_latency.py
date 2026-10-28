"""Sanity check: ASHA's advantage should erode monotonically as the stop-decision latency grows
(epoch units). Run: python validate_latency.py [n_seeds]"""
import sys
import numpy as np
from splits import DEV
from sim.lcbench_env import load_pool
from sim.engine import simulate
from sim.policies import random_proposer, make_asha, no_stop

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
LATS = [0.0, 0.1, 1.0, 5.0, 20.0, 86.0]          # 86 ~ a 327 s generative call at 3.8 s/epoch
print("latency (epochs):", LATS, "| last column = no early stopping")
tot = np.zeros(len(LATS) + 1)
for ds in DEV:
    pool = load_pool(ds); best = pool.y[:, -1].max(); span = best - np.median(pool.y[:, -1])
    row = []
    for pol in [make_asha(latency=l) for l in LATS] + [no_stop]:
        r = [(best - max(simulate(pool, random_proposer, pol, 4, 300.0, np.random.default_rng(1000 + s))["final"],
                         np.median(pool.y[:, -1]))) / span for s in range(n)]
        row.append(np.mean(r))
    tot += row; print(f"{ds:22s}", np.round(row, 3), flush=True)
print("mean regret          ", np.round(tot / len(DEV), 3))

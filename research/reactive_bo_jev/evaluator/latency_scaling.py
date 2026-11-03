"""H4 (latency vs number of questions) and H5 (bundle interaction). Run AFTER collection so the
timings are not contended. Run: python -m evaluator.latency_scaling [n_states]"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from sim.lcbench_env import load_pool
from evaluator.states import build
from evaluator.questions import build_questions, NOULS
from evaluator.client import ask

SEED = 20261024
n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(SEED + 1)
pools = {ds: load_pool(ds) for ds in DEV[:4]}
states = []
for _ in range(n):
    ds = list(pools)[rng.integers(len(pools))]; p = pools[ds]
    states.append(build(p, int(rng.integers(p.n)), int(rng.choice([5, 10, 25])), SEED)[0])
allq = build_questions(); names = list(allq)
ask(states[0], allq)                                           # warm-up
print("H4: latency by number of questions (median / p95 seconds over states)")
for k in (1, 2, 4, 8):
    sub = {q: allq[q] for q in names[:k]}
    lat = [ask(s, sub)[1] for s in states]
    print(f"  {k} questions: median {np.median(lat):.3f}  p95 {np.quantile(lat, .95):.3f}")
print("H5: will_finish_top10 alone vs inside the 8-question bundle")
d = []
for s in states:
    a1, _ = ask(s, {"will_finish_top10": allq["will_finish_top10"]})
    a8, _ = ask(s, allq)
    d.append(a8["will_finish_top10"]["noul"] - a1["will_finish_top10"]["noul"])
d = np.array(d); print(f"  mean |diff| {np.abs(d).mean():.4f}, mean diff {d.mean():+.4f}, max |diff| {np.abs(d).max():.3f}")

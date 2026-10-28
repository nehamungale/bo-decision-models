"""Phase B dev evaluation: GP-Hedge, tuned rule, learned L1/L2 on 8 dev datasets x dev seeds 0..S-1. Resumable.
Run after tune_rule: .venv/bin/python -m acq.run_phaseB [n_seeds] [workers]"""
import sys, os, json, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "phaseB")
_cache = {}


def make(cond, ds):
    from acq.selectors import GPHedge, RuleSelector, LearnedSelector
    if cond == "hedge": return GPHedge()
    if cond == "rule":
        return RuleSelector(**json.load(open(os.path.join(HERE, "results", "tune_rule_best.json")))["best_params"])
    v = {"learned_L1": "L1", "learned_L2": "L2"}[cond]
    if (ds, v) not in _cache: _cache[(ds, v)] = LearnedSelector(ds, v)       # trained leave-one-dataset-out
    return _cache[(ds, v)]


def job(args):
    import torch; torch.set_num_threads(1)
    cond, ds, seed = args
    path = os.path.join(OUT, f"{cond}__{ds}__{seed}.pkl")
    if os.path.exists(path): return path
    from acq.engine import load_pool, run_campaign
    r = run_campaign(load_pool(ds), seed, make(cond, ds)); r["cond"] = cond
    with open(path + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(path + ".tmp", path); return path


if __name__ == "__main__":
    from splits import DEV
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    os.makedirs(OUT, exist_ok=True)
    conds = ["hedge", "rule", "learned_L1", "learned_L2"]
    # group by (dataset, cond) so each worker trains a learned model once per dataset
    jobs = [(c, d, s) for d in DEV[:8] for c in conds for s in range(S)
            if not os.path.exists(os.path.join(OUT, f"{c}__{d}__{s}.pkl"))]
    print(f"{len(jobs)} campaigns on {W} workers", flush=True)
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            fut.result(); k += 1
            if k % 80 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)

"""Tune the rule selector by random search (budget fixed in the plan: 100 configs x 8 dev datasets x 5 tuning seeds 100..104).
Resumable. Run from reactive_bo_jev/: .venv/bin/python -m acq.tune_rule [n_configs] [workers]"""
import sys, os, json, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "tune_rule.jsonl")
TUNE_SEEDS = range(100, 105)


def job(args):
    import torch; torch.set_num_threads(1)
    from splits import DEV
    from acq.engine import load_pool, run_campaign
    from acq.selectors import RuleSelector
    cid, params, ds, seed = args
    r = run_campaign(load_pool(ds), seed, RuleSelector(**params), record=False)
    return cid, ds, seed, r["auc"], r["final"]


if __name__ == "__main__":
    import numpy as np
    from splits import DEV
    from acq.selectors import sample_rule_params
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = np.random.default_rng(2026)
    configs = [sample_rule_params(rng) for _ in range(n)]
    done = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line); done[(r["cid"], r["ds"], r["seed"])] = r
    jobs = [(c, configs[c], d, s) for c in range(n) for d in DEV[:8] for s in TUNE_SEEDS if (c, d, s) not in done]
    print(f"{n} configs, {len(jobs)} campaigns to run", flush=True)
    t0 = time.time(); k = 0
    with open(OUT, "a") as fh, ProcessPoolExecutor(W) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            cid, ds, seed, auc, fin = fut.result()
            fh.write(json.dumps(dict(cid=cid, ds=ds, seed=seed, auc=auc, final=fin)) + "\n"); fh.flush(); k += 1
            if k % 200 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(OUT)]
    score = {}
    for c in range(n):
        per_ds = [np.mean([r["auc"] for r in rows if r["cid"] == c and r["ds"] == d]) for d in DEV[:8]]
        score[c] = float(np.mean(per_ds))
    best = min(score, key=score.get)
    json.dump(dict(best_params=configs[best], best_score=score[best], all_scores=score, configs=configs),
              open(os.path.join(os.path.dirname(OUT), "tune_rule_best.json"), "w"), indent=1)
    print("best config", best, configs[best], "score", round(score[best], 4), "| median config score", round(float(np.median(list(score.values()))), 4))

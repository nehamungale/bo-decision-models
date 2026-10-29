"""Amendment 4 baselines. Stages: `tune` (dwell D in {3,5}, tuning seeds 100-104), `eval` (bandit_ts, sched, best dwell; seeds 0-19). Resumable.
Run from reactive_bo_jev/: .venv/bin/python -m acq.run_phaseE tune|eval [workers]"""
import sys, os, json, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "phaseE")


def make(cond):
    from acq.selectors import BetaBandit, Schedule, DwellRule, RuleSelector
    if cond == "bandit_ts": return BetaBandit()
    if cond == "sched_EI_PI": return Schedule()
    D = int(cond.replace("rule_dwell", ""))
    return DwellRule(RuleSelector(**json.load(open(os.path.join(HERE, "results", "tune_rule_best.json")))["best_params"]), D)


def job(args):
    import torch; torch.set_num_threads(1)
    cond, ds, seed = args
    path = os.path.join(OUT, f"{cond}__{ds}__{seed}.pkl")
    if os.path.exists(path): return path
    from acq.engine import load_pool, run_campaign
    r = run_campaign(load_pool(ds), seed, make(cond), record=False); r["cond"] = cond
    with open(path + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(path + ".tmp", path); return path


def run(jobs, W):
    print(f"{len(jobs)} campaigns on {W} workers", flush=True); t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            fut.result(); k += 1
            if k % 40 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    from splits import DEV
    import numpy as np
    stage = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    os.makedirs(OUT, exist_ok=True)
    if stage == "tune":
        run([(c, d, s) for c in ("rule_dwell3", "rule_dwell5") for d in DEV[:8] for s in range(100, 105)], W)
        base = json.load(open(os.path.join(HERE, "results", "tune_rule_best.json")))["best_score"]
        sc = {1: base}
        for D in (3, 5):
            sc[D] = float(np.mean([np.mean([pickle.load(open(os.path.join(OUT, f"rule_dwell{D}__{d}__{s}.pkl"), "rb"))["auc"] for s in range(100, 105)]) for d in DEV[:8]]))
        best = min(sc, key=sc.get); print("tuning scores", sc, "-> D =", best)
        json.dump(dict(scores=sc, best_D=best), open(os.path.join(HERE, "results", "phaseE_dwell.json"), "w"))
    else:
        D = json.load(open(os.path.join(HERE, "results", "phaseE_dwell.json")))["best_D"]
        conds = ["bandit_ts", "sched_EI_PI"] + ([f"rule_dwell{D}"] if D != 1 else [])
        run([(c, d, s) for c in conds for d in DEV[:8] for s in range(20)], W)

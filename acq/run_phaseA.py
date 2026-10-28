"""Phase A grid: six static AFs + random-AF + one-step oracle (upper bound), dev datasets, dev seeds 0..S-1.
Resumable: one pickle per campaign. Run from reactive_bo_jev/: .venv/bin/python -m acq.run_phaseA [n_seeds] [workers]"""
import sys, os, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phaseA")


def conditions():
    from acq.engine import Static, RandomAF, OracleOneStep, AFS
    return {**{f"static_{a}": (lambda a=a: Static(a)) for a in AFS}, "random": RandomAF, "oracle_onestep": OracleOneStep}


def job(args):
    import torch; torch.set_num_threads(1)
    cond, ds, seed = args
    path = os.path.join(OUT, f"{cond}__{ds}__{seed}.pkl")
    if os.path.exists(path):
        return path, 0.0
    from acq.engine import load_pool, run_campaign
    t0 = time.time()
    r = run_campaign(load_pool(ds), seed, conditions()[cond]())
    r["cond"] = cond
    with open(path + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(path + ".tmp", path)
    return path, time.time() - t0


if __name__ == "__main__":
    from splits import DEV
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    os.makedirs(OUT, exist_ok=True)
    jobs = [(c, d, s) for s in range(S) for d in DEV[:8] for c in conditions()]
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}__{j[1]}__{j[2]}.pkl"))]
    print(f"{len(jobs)} campaigns, {len(todo)} to run on {W} workers", flush=True)
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(W) as ex:
        for fut in as_completed([ex.submit(job, j) for j in todo]):
            fut.result(); done += 1
            if done % 100 == 0: print(f"{done}/{len(todo)} done, {time.time()-t0:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)

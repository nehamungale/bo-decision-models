"""nimble (9B) typed selector on dev datasets. Resumable. Run: .venv/bin/python -m acq.run_phaseC_nimble MODE [n_seeds] [workers]
MODE in sample | argmax | gated_adaptive. The endpoint serves one request at a time (~1.3 s each), so extra workers only overlap the GP compute."""
import sys, os, pickle, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phaseC")


def job(a):
    import torch; torch.set_num_threads(1)
    mode, ds, seed = a
    path = os.path.join(OUT, f"nimble__{mode}__{ds}__{seed}.pkl")
    if os.path.exists(path): return path
    from acq.engine import load_pool, run_campaign
    from acq.typed_selector import TypedSelector
    sel = TypedSelector(mode, None, model="nimble:latest")
    r = run_campaign(load_pool(ds), seed, sel); r["cond"] = sel.name; r["calls"] = sel.calls
    with open(path + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(path + ".tmp", path); return path


if __name__ == "__main__":
    from splits import DEV
    mode = sys.argv[1]; S = int(sys.argv[2]) if len(sys.argv) > 2 else 10; W = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    jobs = [(mode, d, s) for s in range(S) for d in DEV[:8] if not os.path.exists(os.path.join(OUT, f"nimble__{mode}__{d}__{s}.pkl"))]
    print(f"nimble {mode}: {len(jobs)} campaigns on {W} workers", flush=True)
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for _ in ex.map(job, jobs):
            k += 1
            if k % 8 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("finished", round(time.time() - t0), flush=True)

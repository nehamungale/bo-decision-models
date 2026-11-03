"""State-blind control for tev1-sample: fixed-mixture AF draws, 8 dev datasets x seeds 0..19. Run: .venv/bin/python -m acq.run_control_mixture [workers]"""
import sys, os, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phaseC")

def job(a):
    import torch; torch.set_num_threads(1)
    ds, seed = a; path = os.path.join(OUT, f"ctrl__mixture__{ds}__{seed}.pkl")
    if os.path.exists(path): return
    from acq.engine import load_pool, run_campaign
    from acq.selectors import FixedMixture
    r = run_campaign(load_pool(ds), seed, FixedMixture()); r["cond"] = "fixed_mixture_tev1"
    pickle.dump(r, open(path, "wb"))

if __name__ == "__main__":
    from splits import DEV
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(d, s) for s in range(20) for d in DEV[:8]]
    t0 = time.time()
    with ProcessPoolExecutor(W) as ex: list(ex.map(job, jobs))
    print("done", round(time.time() - t0), "s")

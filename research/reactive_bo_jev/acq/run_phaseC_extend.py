"""Post-hoc extension (flagged in notes.md): tev1 'sample' mode on dev seeds 10..19, same prompt, same tau, run regardless of outcome.
Run: .venv/bin/python -m acq.run_phaseC_extend [workers]"""
import sys, os, json, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acq.run_phaseC import job, TAU

if __name__ == "__main__":
    from splits import DEV
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tau = json.load(open(TAU))["tau"]
    jobs = [("dev", "sample", d, s, tau) for s in range(10, 20) for d in DEV[:8]]
    t0 = time.time()
    with ProcessPoolExecutor(W) as ex: list(ex.map(job, jobs))
    print("done", round(time.time() - t0), "s", flush=True)

"""Phase C: tev1 typed selector. Stages: 'pilot' (tuning seeds 100,101, sample mode; fixes the gate tau = median confidence) then 'dev'
(modes argmax/sample/gated, dev seeds 0..S-1). Resumable. Run: .venv/bin/python -m acq.run_phaseC pilot|dev [workers] [n_seeds]"""
import sys, os, json, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "phaseC"); TAU = os.path.join(HERE, "results", "phaseC_tau.json")


def job(args):
    import torch; torch.set_num_threads(1)
    stage, mode, ds, seed, tau = args
    path = os.path.join(OUT, f"{stage}__{mode}__{ds}__{seed}.pkl")
    if os.path.exists(path): return path
    from acq.engine import load_pool, run_campaign
    from acq.typed_selector import TypedSelector
    sel = TypedSelector(mode, tau)
    r = run_campaign(load_pool(ds), seed, sel); r["cond"] = sel.name; r["calls"] = sel.calls
    with open(path + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(path + ".tmp", path); return path


if __name__ == "__main__":
    import numpy as np
    from splits import DEV
    stage = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 4; S = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(OUT, exist_ok=True)
    if stage == "pilot":
        jobs = [("pilot", "sample", d, s, None) for d in DEV[:8] for s in (100, 101)]
    else:
        tau = json.load(open(TAU))["tau"]
        jobs = [("dev", m, d, s, tau) for s in range(S) for d in DEV[:8] for m in ("argmax", "sample", "gated")]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}__{j[1]}__{j[2]}__{j[3]}.pkl"))]
    print(f"{stage}: {len(jobs)} campaigns on {W} workers", flush=True)
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            fut.result(); k += 1
            if k % 12 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    if stage == "pilot":
        import glob
        conf = [c["conf"] for f in glob.glob(os.path.join(OUT, "pilot__*.pkl")) for c in pickle.load(open(f, "rb"))["calls"] if c["conf"] is not None]
        fb = sum(c["fallback"] for f in glob.glob(os.path.join(OUT, "pilot__*.pkl")) for c in pickle.load(open(f, "rb"))["calls"])
        tau = float(np.median(conf)); json.dump(dict(tau=tau, n=len(conf), fallbacks=fb, quantiles=np.quantile(conf, [.1, .25, .5, .75, .9]).tolist()), open(TAU, "w"))
        print("pilot done; tau (median confidence) =", round(tau, 4), "n =", len(conf), "fallbacks", fb, flush=True)
    print("finished", time.time() - t0, flush=True)

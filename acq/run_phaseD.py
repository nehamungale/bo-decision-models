"""Phase D (reduced): LMABO-style generative LLM selector on dev datasets, dev seeds 0..S-1, 2 worker processes
(the cloud endpoint returns 429 above ~3 concurrent requests). Resumable.
Run: .venv/bin/python -m acq.run_phaseD [n_seeds] [workers] [model]"""
import sys, os, pickle, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "phaseD")


def job(args):
    import torch; torch.set_num_threads(1)
    ds, seed, model = args
    tag = model.split(":")[0].replace("-", "")
    path = os.path.join(OUT, f"llm_{tag}__{ds}__{seed}.pkl")
    if os.path.exists(path): return path
    from acq.engine import load_pool, run_campaign
    from acq.llm_selector import LLMSelector
    sel = LLMSelector(model)
    r = run_campaign(load_pool(ds), seed, sel); r["cond"] = f"llm_{tag}"; r["llm_calls"] = sel.calls
    with open(path + ".tmp", "wb") as f: pickle.dump(r, f)
    os.replace(path + ".tmp", path); return path


if __name__ == "__main__":
    from splits import DEV
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-oss:120b-cloud"
    tag = model.split(":")[0].replace("-", "")
    os.makedirs(OUT, exist_ok=True)
    jobs = [(d, s, model) for s in range(S) for d in DEV[:8] if not os.path.exists(os.path.join(OUT, f"llm_{tag}__{d}__{s}.pkl"))]
    print(f"{len(jobs)} campaigns on {W} workers ({model})", flush=True)
    t0 = time.time(); k = 0
    with ProcessPoolExecutor(W) as ex:
        for fut in as_completed([ex.submit(job, j) for j in jobs]):
            fut.result(); k += 1
            if k % 8 == 0: print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("finished", time.time() - t0, flush=True)

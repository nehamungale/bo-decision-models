"""Replay every Phase A campaign with candidate-level logging. Asserts the replay reproduces the logged proposals and outcomes exactly. Output: results/phaseA_cand.
Run: .venv/bin/python -m acq.replay_cand [workers]"""
import sys, os, glob, pickle, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "results", "phaseA_cand")


def job(f):
    import torch; torch.set_num_threads(1)
    out = os.path.join(OUT, os.path.basename(f))
    if os.path.exists(out): return 0
    from acq.engine import load_pool, run_campaign, Replay
    r = pickle.load(open(f, "rb"))
    n = run_campaign(load_pool(r["dataset"]), r["seed"], Replay([x["af"] for x in r["log"]]))
    for a, b in zip(r["log"], n["log"]):
        assert a["proposals"] == b["proposals"] and abs(a["y"] - b["y"]) < 1e-12, (f, a["t"])
    pickle.dump(dict(cond=r["cond"], dataset=r["dataset"], seed=r["seed"], log=[dict(t=x["t"], af=x["af"], proposals=x["proposals"], proposal_y=x["proposal_y"], y=x["y"],
                     improved=x["improved"], state=x["state"], cand_feats=x["cand_feats"]) for x in n["log"]]), open(out, "wb"))
    return 1

if __name__ == "__main__":
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    os.makedirs(OUT, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(H, "results", "phaseA", "*.pkl")))
    t0 = time.time()
    with ProcessPoolExecutor(W) as ex: done = sum(ex.map(job, fs, chunksize=4))
    print(f"replayed {done} campaigns (all proposals and outcomes matched), {time.time()-t0:.0f}s", flush=True)

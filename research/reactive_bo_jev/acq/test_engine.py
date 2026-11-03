"""Engine sanity tests (Phase A step 1). Run: .venv/bin/python -m acq.test_engine   (from reactive_bo_jev/)"""
import sys, os, json, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acq.engine import load_pool, run_campaign, Static, RandomAF, OracleOneStep, AFS, propose_all
import torch
torch.set_num_threads(1)


def determinism():
    """Same seed in two separate processes must give the identical trajectory."""
    code = ("import sys,json; sys.path.insert(0,'.');"
            "from acq.engine import *; import torch; torch.set_num_threads(1);"
            "p=load_pool('vehicle'); r=run_campaign(p,3,Static('LogEI'),budget=8);"
            "print(json.dumps([round(float(x),12) for x in r['regret']]))")
    outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip() for _ in range(2)]
    assert outs[0] and outs[0] == outs[1], f"non-deterministic: {outs}"
    print("determinism: OK (two processes identical)")


def reference_agreement():
    """Our LogEI proposal must equal BoTorch's own discrete optimiser on the same fitted model."""
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.optim.optimize import optimize_acqf_discrete
    p = load_pool("vehicle"); rng = np.random.default_rng(0)
    idx = rng.choice(len(p.y), 15, replace=False)
    model, cand, mu, sd, props = propose_all(p, idx, p.y[idx], 0, 0)
    af = LogExpectedImprovement(model, best_f=float(p.y[idx].max()))
    pts, _ = optimize_acqf_discrete(af, q=1, choices=torch.tensor(p.X[cand]))
    ref = int(cand[np.argmin(np.linalg.norm(p.X[cand] - pts.numpy(), axis=1))])
    assert ref == props[0], (ref, props[0])
    print("reference agreement: OK (LogEI argmax equals BoTorch optimize_acqf_discrete)")


def shapes_and_sanity():
    p = load_pool("vehicle")
    r = run_campaign(p, 1, Static('LogEI'), budget=10)
    assert len(r["regret"]) == 10 and (np.diff(r["regret"]) <= 1e-12).all(), "regret must be non-increasing"
    s = r["log"][-1]["state"]; assert all(np.isfinite(v) for v in s.values())
    assert len(set(r["log"][0]["proposals"])) >= 1
    print("shapes/monotone regret/finite state: OK; distinct proposals at step 0:", len(set(r['log'][0]['proposals'])))


def logei_beats_random(n=6):
    p = load_pool("vehicle")
    a = [run_campaign(p, s, Static('LogEI'), budget=25, record=False)["auc"] for s in range(n)]
    b = [run_campaign(p, s, RandomAF(), budget=25, record=False)["auc"] for s in range(n)]
    print(f"LogEI AUC {np.mean(a):.3f} vs random-AF {np.mean(b):.3f} (vehicle, {n} seeds, 25 steps)")


if __name__ == "__main__":
    determinism(); reference_agreement(); shapes_and_sanity(); logei_beats_random()

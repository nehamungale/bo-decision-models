"""Stage 0: (a) cost of one labelled state; (b) triviality of kept landscapes under static LogEI.
Run from reactive_bo_jev/: .venv python -m meta.stage0 [workers] [n_landscapes]"""
import sys, os, time, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)


def triv(args):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from acq.engine import run_campaign, Static
    name, seed = args
    p = load(name); n0 = 2 * p.X.shape[1] + 1; B = budget(p)
    t = time.time(); r = run_campaign(p, seed, Static("LogEI"), budget=B, n_init=n0, record=False)
    reg0 = p.regret(float(p.y[np.random.default_rng(seed).choice(len(p.y), n0, replace=False)].max()))
    return dict(landscape=name, seed=seed, n=len(p.y), d=p.X.shape[1], budget=B, init_regret=reg0, auc=r["auc"], final=r["final"], secs=time.time() - t)


def timing(name):
    import torch; torch.set_num_threads(1)
    from meta.landscapes import load, budget
    from meta.labels import LabelCollector
    from acq.engine import run_campaign
    p = load(name); B = budget(p); c = LabelCollector("random", every=B // 2 or 1, budget=B)
    t = time.time(); run_campaign(p, 2000, c, budget=B, n_init=2 * p.X.shape[1] + 1, record=False); tot = time.time() - t
    return dict(landscape=name, n_states=len(c.rows), secs_total=tot, secs_per_state=tot / max(len(c.rows), 1), budget=B)


if __name__ == "__main__":
    from meta.landscapes import kept
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    k = kept()
    print("kept pool:", k.family.value_counts().to_dict(), flush=True)
    for nm in ["lcbench/vehicle", k[k.family == "hpob"].landscape.iloc[0], k[k.family == "tabrepo"].landscape.iloc[0]]:
        print("timing", timing(nm), flush=True)
    rng = np.random.default_rng(0)
    pick = k.groupby("family", group_keys=False).apply(lambda g: g.sample(min(len(g), N // 3), random_state=0))
    jobs = [(n, s) for n in pick.landscape for s in (3000, 3001, 3002)]
    t0 = time.time()
    with ProcessPoolExecutor(W) as ex: rows = list(ex.map(triv, jobs, chunksize=2))
    json.dump(rows, open(os.path.join(OUT, "stage0_triviality.json"), "w"))
    import pandas as pd
    d = pd.DataFrame(rows); d["family"] = d.landscape.str.split("_|/", n=1).str[0]
    g = d.groupby("landscape").agg(auc=("auc", "mean"), init=("init_regret", "mean"), fam=("family", "first"))
    print(f"{len(g)} landscapes x 3 seeds, {time.time()-t0:.0f}s wall")
    print("mean campaign secs:", d.groupby("family").secs.mean().round(1).to_dict())
    for fam, h in g.groupby("fam"):
        print(fam, "n", len(h), "| init regret==0:", f"{(h.init==0).mean():.2f}", "| LogEI AUC==0:", f"{(h.auc==0).mean():.2f}", "| AUC<0.01:", f"{(h.auc<0.01).mean():.2f}",
              "| AUC quartiles", np.round(h.auc.quantile([.25,.5,.75]).values, 3).tolist())
    print("overall AUC<0.01 fraction:", f"{(g.auc<0.01).mean():.2f}", "| init==0:", f"{(g.init==0).mean():.2f}")

"""Phase 1 of PLAN_meta_selector.md: export every real tabular landscape to plain .npz pools (X in [0,1], y higher-is-better)
and write an inventory with validity checks. Needs syne-tune. Run with the syne-tune venv:
    python meta/export_landscapes.py [out_dir]
One file per family: <out_dir>/<family>.npz with keys "<landscape>__X" and "<landscape>__y"; inventory.csv alongside."""
import sys, os, csv
import numpy as np
import pandas as pd
from syne_tune.blackbox_repository import load_blackbox, blackbox_list

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pools")
os.makedirs(OUT, exist_ok=True)

HPOB = [n for n in blackbox_list() if n.startswith("hpob_")]
TABREPO = [n for n in blackbox_list() if n.startswith("tabrepo_")]
SKIPPED = []
MAX_POOL = 3000          # subsample very large tables (FCNet, NAS-Bench-201), fixed seed


def encode(h: pd.DataFrame) -> np.ndarray:
    """Numeric -> [0,1] (log first if positive and spanning > 2 decades); two-level categorical -> 0/1; others one-hot.
    Constant columns are dropped."""
    cols = []
    for c in h.columns:
        s = h[c]
        if s.dtype == object or str(s.dtype).startswith("str") or s.dtype == bool:
            lv = sorted(s.astype(str).unique())
            if len(lv) < 2: continue
            if len(lv) == 2: cols.append((s.astype(str) == lv[1]).to_numpy(float))
            else: cols += [(s.astype(str) == l).to_numpy(float) for l in lv]
        else:
            v = s.to_numpy(float)
            if np.ptp(v) == 0: continue
            if v.min() > 0 and v.max() / v.min() > 100: v = np.log(v)
            cols.append((v - v.min()) / np.ptp(v))
    return np.column_stack(cols)


def check(X, y):
    """Validity checks; returns dict of flags."""
    top = np.isclose(y, y.max())
    return dict(n=len(y), d=X.shape[1], nan=int(np.isnan(X).sum() + np.isnan(y).sum()),
                dup_x=int(len(X) - len(np.unique(np.round(X, 9), axis=0))), n_at_max=int(top.sum()),
                y_min=float(y.min()), y_med=float(np.median(y)), y_max=float(y.max()),
                span=float(y.max() - np.median(y)))


def add(store, rows, family, group, task, h, y, rng):
    X = encode(h)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if len(y) < 20:                      # failed / mostly-missing landscape: record and skip
        SKIPPED.append((family, group, task, int(len(y)))); return
    if len(y) > MAX_POOL:
        idx = np.sort(rng.choice(len(y), MAX_POOL, replace=False)); X, y = X[idx], y[idx]
    name = f"{group}/{task}"
    store[f"{name}__X"] = X; store[f"{name}__y"] = y
    rows.append(dict(family=family, group=group, landscape=name, **check(X, y)))


def main():
    rng = np.random.default_rng(0)
    rows = []
    # HPO-B: accuracy (higher is better); hyperparameters already in [0,1]
    st = {}
    for g in HPOB:
        bb = load_blackbox(g)
        for t, b in bb.items():
            add(st, rows, "hpob", g, t, b.hyperparameters, b.objectives_evaluations[:, :, -1, b.objectives_names.index("metric_accuracy")].mean(1), rng)
    np.savez_compressed(os.path.join(OUT, "hpob.npz"), **st)
    # TabRepo: validation error averaged over the 3 seeds, negated
    st = {}
    for g in TABREPO:
        bb = load_blackbox(g)
        for t, b in bb.items():
            e = b.objectives_evaluations[:, :, -1, b.objectives_names.index("metric_error_val")].mean(1)
            add(st, rows, "tabrepo", g, t, b.hyperparameters, -e, rng)
    np.savez_compressed(os.path.join(OUT, "tabrepo.npz"), **st)
    # FCNet: final validation loss at last epoch, mean over 4 seeds, negated
    st = {}
    bb = load_blackbox("fcnet")
    for t, b in bb.items():
        e = b.objectives_evaluations[:, :, -1, b.objectives_names.index("metric_valid_loss")].mean(1)
        add(st, rows, "fcnet", "fcnet", t, b.hyperparameters, -e, rng)
    np.savez_compressed(os.path.join(OUT, "fcnet.npz"), **st)
    # NAS-Bench-201: final validation error, mean over 3 seeds, negated
    st = {}
    bb = load_blackbox("nasbench201")
    for t, b in bb.items():
        e = b.objectives_evaluations[:, :, -1, b.objectives_names.index("metric_valid_error")].mean(1)
        add(st, rows, "nb201", "nb201", t, b.hyperparameters, -e, rng)
    np.savez_compressed(os.path.join(OUT, "nb201.npz"), **st)
    # LCBench: reuse the repo encoder (final validation balanced accuracy), all 35 datasets
    sys.path.insert(0, os.path.dirname(HERE))
    from sim.lcbench_env import load_pool
    from splits import CONFIRM
    st = {}
    for t in load_blackbox("lcbench").keys():
        p = load_pool(t)
        name = f"lcbench/{t}"
        st[f"{name}__X"] = p.X.astype(float); st[f"{name}__y"] = p.y[:, -1].astype(float)
        rows.append(dict(family="lcbench", group="lcbench_reserved" if t in CONFIRM else "lcbench", landscape=name, **check(p.X, p.y[:, -1])))
    np.savez_compressed(os.path.join(OUT, "lcbench.npz"), **st)
    with open(os.path.join(OUT, "inventory.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("skipped (fewer than 20 valid configs):", SKIPPED)
    print("wrote", len(rows), "landscapes to", OUT)


if __name__ == "__main__":
    main()

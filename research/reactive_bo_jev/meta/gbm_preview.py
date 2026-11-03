"""Stage 2 preview: leave-one-landscape-out GBM on the 480 pilot states (H=4, LogEI-continuation labels).
Pre-declared: HistGradientBoostingRegressor defaults (random_state 0), one row per (state, AF), features = state without AF-history keys + descriptors
+ that AF's nine candidate numbers + AF one-hot; target V_a; selection = argmax over the six AFs. Score = mean achieved label on ALL held-out states."""
import glob, os, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from acq.engine import AFS, STATE_KEYS
from acq.selectors import NOHIST_KEYS
H = os.path.dirname(os.path.abspath(__file__))
S = []
for f in sorted(glob.glob(os.path.join(H, "results", "stage1", "*.pkl"))):
    r = pickle.load(open(f, "rb"))
    for x in r["rows"]: S.append((r["landscape"], r["seed"], x))
DK = ["dim", "n_pool", "loo_rmse", "loo_max", "noise_to_signal", "near_inc_frac", "local_vs_global", "y_skew", "y_range_ratio"]
def feats(x, groups=("state", "desc", "cand", "afid")):
    rows = []
    for a in range(6):
        v = []
        if "state" in groups: v += [x["state"][k] for k in NOHIST_KEYS]
        if "desc" in groups: v += [x["desc"][k] for k in DK]
        if "cand" in groups: v += list(x["cand"][a])
        if "afid" in groups: v += [1.0 if i == a else 0.0 for i in range(6)]
        rows.append(v)
    return rows
L = np.array([s[0] for s in S]); V = np.array([s[2]["V"][0] for s in S]); names = sorted(set(L))
def run(groups, shuffle=False, seed=0):
    F = np.array([feats(s[2], groups) for s in S]); out = {}
    rng = np.random.default_rng(seed)
    for ln in names:
        tr = L != ln; te = ~tr
        Xtr = F[tr].reshape(-1, F.shape[2]); Vtr = V[tr].copy()
        if shuffle: Vtr = Vtr[rng.permutation(len(Vtr))]
        m = HistGradientBoostingRegressor(random_state=0).fit(Xtr, Vtr.reshape(-1))
        P = m.predict(F[te].reshape(-1, F.shape[2])).reshape(-1, 6)
        fixed = int(V[tr].mean(0).argmax())                         # best fixed AF chosen on training landscapes
        Vt = V[te]
        out[ln] = dict(sel=Vt[np.arange(len(Vt)), P.argmax(1)].mean(), fixed=Vt[:, fixed].mean(), mean=Vt.mean(), oracle=Vt.max(1).mean(), fixed_af=AFS[fixed],
                       sel_rk=None)
    return out
def report(tag, o):
    sel = np.array([v["sel"] for v in o.values()]); fx = np.array([v["fixed"] for v in o.values()])
    print(f"{tag:34s} selected {sel.mean():.4f} | best-fixed(train) {fx.mean():.4f} | diff {np.mean(sel-fx):+.4f} | landscapes where selected > fixed: {(sel>fx).sum()}/{len(sel)}")
    return sel, fx
o = run(("state", "desc", "cand", "afid"))
print("per landscape (label units = fraction of y_max - y_median):")
for ln, v in o.items(): print(f"  {ln:42s} selected {v['sel']:.4f} fixed({v['fixed_af']}) {v['fixed']:.4f} mean-AF {v['mean']:.4f} oracle {v['oracle']:.4f}")
print("overall:"); report("M1 all features", o)
print("mean-AF (random choice)       ", np.mean([v["mean"] for v in o.values()]).round(4), "| per-state oracle", np.mean([v["oracle"] for v in o.values()]).round(4))
for g in [("afid",), ("cand", "afid"), ("state", "desc", "afid"), ("state", "afid"), ("desc", "afid")]:
    report("C2/C3 " + "+".join(g), run(g))
sh = [report(f"C1 label-shuffled (rep {r})", run(("state", "desc", "cand", "afid"), True, r)) for r in range(3)]

"""Pre-declared analysis of Stage 2b (offline H1 proxy). Leave-one-landscape-out HistGradientBoostingRegressor (defaults, random_state 0), one row per (state, AF),
features = state without AF-history keys + descriptors + that AF's nine candidate numbers + AF one-hot; target = LogEI-continuation label V_a; selection = argmax.
Per held-out landscape: mean achieved label over ALL its states for (GBM, best fixed AF chosen on the other landscapes, random AF, per-state oracle, label-shuffled GBM x3).
PASS (stop rule given to the user): GBM > best-fixed on >= 20 of 30 landscapes, one-sided Wilcoxon p < 0.05, and GBM-minus-fixed exceeds the shuffled-control difference (paired Wilcoxon p < 0.05).
Caveat: the label is a 4-step incumbent gain, a proxy for the online AUC-of-regret criterion in H1, which is tested later by campaigns."""
import glob, os, pickle, sys
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from acq.engine import AFS
from acq.selectors import NOHIST_KEYS
H = os.path.dirname(os.path.abspath(__file__))
DK = ["dim", "n_pool", "loo_rmse", "loo_max", "noise_to_signal", "near_inc_frac", "local_vs_global", "y_skew", "y_range_ratio"]
S = []
for f in sorted(glob.glob(os.path.join(H, "results", "stage2", "*.pkl"))):
    r = pickle.load(open(f, "rb"))
    for x in r["rows"]: S.append((r["landscape"], x))
L = np.array([s[0] for s in S]); V = np.array([s[1]["V"][0] for s in S]); names = sorted(set(L))
def feats(x, groups):
    out = []
    for a in range(6):
        v = []
        if "state" in groups: v += [x["state"][k] for k in NOHIST_KEYS]
        if "desc" in groups: v += [x["desc"][k] for k in DK]
        if "cand" in groups: v += list(x["cand"][a])
        if "afid" in groups: v += [1.0 if i == a else 0.0 for i in range(6)]
        out.append(v)
    return out
print(f"{len(S)} states, {len(names)} landscapes; all-zero label share {np.mean(V.max(1)==0):.2f}; ranked share {np.mean(np.ptp(V,axis=1)>0):.2f}")
def run(groups, shuffle=False, seed=0):
    F = np.array([feats(s[1], groups) for s in S]); rng = np.random.default_rng(seed); res = {}
    for ln in names:
        tr = L != ln; te = ~tr
        Vtr = V[tr].copy()
        if shuffle: Vtr = Vtr[rng.permutation(len(Vtr))]
        m = HistGradientBoostingRegressor(random_state=0).fit(F[tr].reshape(-1, F.shape[2]), Vtr.reshape(-1))
        P = m.predict(F[te].reshape(-1, F.shape[2])).reshape(-1, 6); Vt = V[te]; fixed = int(V[tr].mean(0).argmax())
        res[ln] = dict(sel=Vt[np.arange(len(Vt)), P.argmax(1)].mean(), fixed=Vt[:, fixed].mean(), rand=Vt.mean(), oracle=Vt.max(1).mean(), fixed_af=AFS[fixed], n=len(Vt))
    return res
def summ(tag, res, base=None):
    sel = np.array([v["sel"] for v in res.values()]); fx = np.array([v["fixed"] for v in res.values()]); d = sel - fx
    p = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
    print(f"{tag:36s} selected {sel.mean():.4f} fixed {fx.mean():.4f} diff {d.mean():+.4f} | wins {int((d>0).sum())}/{len(d)} | Wilcoxon p={p:.3f}")
    return d, p
res = run(("state", "desc", "cand", "afid"))
d, p = summ("M1 (all features)", res)
print(f"random-AF {np.mean([v['rand'] for v in res.values()]):.4f} | per-state oracle {np.mean([v['oracle'] for v in res.values()]):.4f}")
fams = {}
for ln, v in res.items(): fams.setdefault(ln.split("/")[0].split("_")[0], []).append(v["sel"] - v["fixed"])
print("by family (mean diff, wins):", {k: (round(float(np.mean(x)), 4), int(np.sum(np.array(x) > 0)), len(x)) for k, x in fams.items()})
print("best fixed AF chosen when each landscape is held out:", {a: sum(v["fixed_af"] == a for v in res.values()) for a in AFS})
ds = []
for r in range(3):
    dd, _ = summ(f"C1 label-shuffled rep {r}", run(("state", "desc", "cand", "afid"), True, r)); ds.append(dd)
dsh = np.mean(ds, axis=0); pp = wilcoxon(d - dsh, alternative="greater").pvalue
print(f"M1 diff minus mean shuffled diff: {np.mean(d-dsh):+.4f}, paired Wilcoxon p={pp:.3f}")
for g in [("cand", "afid"), ("state", "desc", "afid"), ("state", "afid"), ("desc", "afid")]: summ("C3 " + "+".join(g), run(g))
ok = (d > 0).sum() >= 20 and p < 0.05 and pp < 0.05
print("\nSTAGE 2b VERDICT:", "PASS" if ok else "FAIL", f"(wins {int((d>0).sum())}/30 needed 20; p={p:.3f}; vs shuffled p={pp:.3f})")

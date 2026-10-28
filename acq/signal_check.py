"""Is there ANY learnable signal in the state about which AF's next proposal will improve the incumbent? Leave-one-dataset-out gradient boosting on Phase A logs
(all conditions pooled, 64k states), per-AF binary label 'proposal beats the incumbent' and the one-step gain. Run: .venv/bin/python -m acq.signal_check"""
import sys, os, glob, pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS, STATE_KEYS, load_pool
from acq.selectors import NOHIST_KEYS

H = os.path.dirname(os.path.abspath(__file__))
rows = []
pools = {d: load_pool(d) for d in DEV[:8]}
for f in glob.glob(os.path.join(H, "results", "phaseA", "*.pkl")):
    r = pickle.load(open(f, "rb")); p = pools[r["dataset"]]; span = max(p.y_max - p.y_med, 1e-12)
    best = float(p.y[np.random.default_rng(r["seed"]).choice(len(p.y), 2 * p.X.shape[1] + 1, replace=False)].max())
    for st in r["log"]:
        rows.append((r["dataset"], r["cond"], [st["state"][k] for k in STATE_KEYS], [max(0.0, y - best) / span for y in st["proposal_y"]]))
        best = max(best, st["y"])
ds = np.array([r[0] for r in rows]); cond = np.array([r[1] for r in rows])
X = np.array([r[2] for r in rows], float); G = np.array([r[3] for r in rows], float)
cols = [STATE_KEYS.index(k) for k in NOHIST_KEYS]
print(f"{len(X)} states; base rate of 'proposal beats incumbent' per AF: {dict(zip(AFS, np.round((G > 0).mean(0), 3)))}")
auc = {a: [] for a in AFS}; achieved = {"learned argmax": [], "random AF": [], "oracle": [], **{f"static {a}": [] for a in AFS}}
for held in DEV[:8]:
    tr, te = ds != held, ds == held
    pred = np.zeros((te.sum(), 6))
    for i, a in enumerate(AFS):
        y = (G[tr, i] > 0).astype(int)
        m = HistGradientBoostingClassifier(random_state=0).fit(X[tr][:, cols], y)
        pred[:, i] = m.predict_proba(X[te][:, cols])[:, 1]
        yt = (G[te, i] > 0).astype(int)
        if 0 < yt.sum() < len(yt): auc[a].append(roc_auc_score(yt, pred[:, i]))
    g = G[te]
    achieved["learned argmax"].append(g[np.arange(len(g)), pred.argmax(1)].mean()); achieved["random AF"].append(g.mean())
    achieved["oracle"].append(g.max(1).mean())
    for i, a in enumerate(AFS): achieved[f"static {a}"].append(g[:, i].mean())
print("\nLODO AUROC of state -> 'AF a's proposal beats the incumbent' (mean over 8 held-out datasets):")
print("  ", {a: round(float(np.mean(v)), 3) for a, v in auc.items()})
print("\nMean one-step normalised gain per step (mean over held-out datasets; higher is better):")
for k, v in achieved.items(): print(f"   {k:16s} {np.mean(v):.4f}")

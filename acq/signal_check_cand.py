"""Candidate-level signal check (no language model). Question: if a selector may look at the NUMBERS OF EACH AF's PROPOSAL (posterior mean/std gap to the incumbent,
PI, EI, distances, ranks) in addition to the global state, can it choose a proposal with a better one-step outcome than random or static AF choice?
Defined before running: (1) univariate within-state selection by each candidate feature; (2) leave-one-dataset-out gradient boosting on candidate + global features (and candidate only);
scored by achieved one-step normalised gain (max(0, y_proposal - incumbent)/(y_max - y_median)) against random AF, each static AF, and the peeking oracle; per-dataset paired Wilcoxon.
Run: .venv/bin/python -m acq.signal_check_cand"""
import sys, os, glob, pickle
import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from acq.engine import AFS, STATE_KEYS, CAND_KEYS, load_pool
from acq.selectors import NOHIST_KEYS

H = os.path.dirname(os.path.abspath(__file__))
pools = {d: load_pool(d) for d in DEV[:8]}
D, SG, CF, G = [], [], [], []
for f in glob.glob(os.path.join(H, "results", "phaseA_cand", "*.pkl")):
    r = pickle.load(open(f, "rb")); p = pools[r["dataset"]]; span = max(p.y_max - p.y_med, 1e-12)
    best = float(p.y[np.random.default_rng(r["seed"]).choice(len(p.y), 2 * p.X.shape[1] + 1, replace=False)].max())
    for st in r["log"]:
        D.append(r["dataset"]); SG.append([st["state"][k] for k in NOHIST_KEYS]); CF.append(st["cand_feats"])
        G.append([max(0.0, y - best) / span for y in st["proposal_y"]]); best = max(best, st["y"])
D = np.array(D); SG = np.array(SG, float); CF = np.array(CF, float); G = np.array(G, float)        # CF: (n, 6 AFs, 9 feats)
n = len(D); print(f"{n} states x 6 AF proposals; base rate proposal beats incumbent {np.mean(G > 0):.3f}")
ach = lambda pick: G[np.arange(n), pick]
print("\nOne-step normalised gain per step, pooled over states (higher is better); reference points:")
ref = {"random AF": G.mean(), "oracle (peeks)": G.max(1).mean(), **{f"static {a}": G[:, i].mean() for i, a in enumerate(AFS)}}
for k, v in ref.items(): print(f"   {k:16s} {v:.4f}")

print("\n(1) Univariate: pick the AF whose proposal has the highest value of one candidate feature (within each state)")
for j, k in enumerate(CAND_KEYS):
    for sign, nm in ((1, "highest"), (-1, "lowest")):
        pick = np.argmax(sign * CF[:, :, j], axis=1); print(f"   {nm:7s} {k:9s} -> gain {ach(pick).mean():.4f}   AF shares {dict(zip(AFS, np.round(np.bincount(pick, minlength=6) / n, 2)))}")

def lodo(feat_fn, name):
    per = {"learned": [], "random": [], "UCB": [], "LogEI": [], "oracle": []}; aucs = []
    pick_all = np.zeros(n, int)
    for held in DEV[:8]:
        tr, te = D != held, D == held
        Xtr = np.vstack([feat_fn(i, tr) for i in range(6)]); ytr = np.concatenate([(G[tr, i] > 0).astype(int) for i in range(6)])
        m = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
        pred = np.column_stack([m.predict_proba(feat_fn(i, te))[:, 1] for i in range(6)])
        pk = pred.argmax(1); pick_all[np.where(te)[0]] = pk; g = G[te]
        per["learned"].append(g[np.arange(len(g)), pk].mean()); per["random"].append(g.mean()); per["UCB"].append(g[:, 2].mean()); per["LogEI"].append(g[:, 0].mean()); per["oracle"].append(g.max(1).mean())
        aucs.append(roc_auc_score((g > 0).ravel(), pred.ravel()))
    m_ = {k: float(np.mean(v)) for k, v in per.items()}
    wr = wilcoxon(np.array(per["learned"]) - np.array(per["random"])).pvalue; wu = wilcoxon(np.array(per["learned"]) - np.array(per["UCB"])).pvalue
    print(f"\n(2) {name}: LODO AUROC (pooled AF rows) {np.mean(aucs):.3f}; one-step gain learned {m_['learned']:.4f} | random {m_['random']:.4f} | static UCB {m_['UCB']:.4f} | static LogEI {m_['LogEI']:.4f} | oracle {m_['oracle']:.4f}")
    print(f"     per-dataset learned minus random: {np.round(np.array(per['learned']) - np.array(per['random']), 4).tolist()}; learned > random on {int(np.sum(np.array(per['learned']) > np.array(per['random'])))}/8 (Wilcoxon p={wr:.3f}); vs UCB p={wu:.3f}")
    print(f"     AF shares {dict(zip(AFS, np.round(np.bincount(pick_all, minlength=6) / n, 2)))}")

oh = lambda i, m: np.tile(np.eye(6)[i], (m.sum(), 1))
lodo(lambda i, m: np.hstack([CF[m, i, :], oh(i, m)]), "candidate features + AF identity")
lodo(lambda i, m: np.hstack([CF[m, i, :], oh(i, m), SG[m]]), "candidate + AF identity + global state")
lodo(lambda i, m: np.hstack([CF[m, i, :], oh(i, m), CF[m, :, :].reshape(m.sum(), -1)]), "candidate + AF identity + all six proposals' features (context)")

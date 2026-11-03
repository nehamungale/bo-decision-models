import sys, os, glob, pickle
import numpy as np
from scipy.stats import spearmanr
H = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
data = {}
for f in sorted(glob.glob(os.path.join(H, "results", "stage1", "*.pkl"))):
    r = pickle.load(open(f, "rb"))
    data.setdefault(r["landscape"], []).extend(r["rows"])

def rho(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0: return np.nan
    return spearmanr(a, b)[0]

print(f"{'landscape':40s} states  allzero(LogEI) allzero(UCB)  defined  mean_rho  shuffled  top1_agree  nonzero_mean")
per = {}
for ln, rows in data.items():
    A = np.array([r["V"][0] for r in rows]); B = np.array([r["V"][2] for r in rows])
    z0 = (A.max(1) == 0).mean(); z2 = (B.max(1) == 0).mean()
    rh = np.array([rho(a, b) for a, b in zip(A, B)])
    perm = rng.permutation(len(rows))
    sh = np.array([rho(a, b) for a, b in zip(A, B[perm])])
    t1 = np.mean([A[i].argmax() == B[i].argmax() for i in range(len(A)) if A[i].max() > 0 and B[i].max() > 0] or [np.nan])
    per[ln] = (np.nanmean(rh), np.nanmean(sh), np.mean(~np.isnan(rh)))
    print(f"{ln:40s} {len(rows):5d}   {z0:10.2f} {z2:12.2f}  {np.mean(~np.isnan(rh)):7.2f}  {np.nanmean(rh):8.3f}  {np.nanmean(sh):8.3f}  {t1:9.2f}   {np.mean(np.r_[A.ravel(), B.ravel()]):.4f}")
m = np.array([v[0] for v in per.values()]); s = np.array([v[1] for v in per.values()])
print(f"\nmean within-state Spearman across landscapes {m.mean():.3f} (landscape-bootstrap 95% CI "
      f"{np.percentile([rng.choice(m, len(m)).mean() for _ in range(5000)], [2.5, 97.5]).round(3)}), shuffled control {s.mean():.3f}")
# state-level pooled
allr = []; alls = []
for ln, rows in data.items():
    A = np.array([r["V"][0] for r in rows]); B = np.array([r["V"][2] for r in rows]); perm = rng.permutation(len(rows))
    allr += [rho(a, b) for a, b in zip(A, B)]; alls += [rho(a, b) for a, b in zip(A, B[perm])]
print(f"state-pooled: rho {np.nanmean(allr):.3f} on {np.sum(~np.isnan(allr))} defined of {len(allr)} states; shuffled {np.nanmean(alls):.3f}")
zs = np.mean([np.mean(np.array([r["V"][0] for r in rows]).max(1) == 0) for rows in data.values()])
print(f"mean share of states with all-zero labels (LogEI continuation): {zs:.2f}; share of states with a defined ranking: {np.mean(~np.isnan(allr)):.2f}")
sp = np.nanmean(alls)
print("G0 (as pre-registered, on states where a ranking exists):", "PASS" if np.nanmean(allr) > 0.3 and np.nanmean(allr) > sp + 0.1 else "FAIL", f"| rho {np.nanmean(allr):.3f} vs shuffled {sp:.3f}")
print("Sparsity check (plan: >~90% all-zero would mean too sparse):", "ok" if zs < 0.9 else "SPARSE", f"at the landscape mean; per-landscape range printed above")

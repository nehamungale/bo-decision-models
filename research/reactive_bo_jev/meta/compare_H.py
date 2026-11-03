import os, glob, pickle
import numpy as np
from scipy.stats import spearmanr
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(H, d, "*.pkl"))):
        r = pickle.load(open(f, "rb"))
        for row in r["rows"]: out[(r["landscape"], r["seed"], row["t"])] = row
    return out
a, b = load("stage1"), load("stage1_H8")
keys = sorted(set(a) & set(b)); print("states in common:", len(keys), "(H=4:", len(a), "H=8:", len(b), ")")
def rho(x, y): return np.nan if np.ptp(x) == 0 or np.ptp(y) == 0 else spearmanr(x, y)[0]
for nm, d in (("H=4", a), ("H=8", b)):
    A = np.array([d[k]["V"][0] for k in keys]); B = np.array([d[k]["V"][2] for k in keys])
    r = np.array([rho(x, y) for x, y in zip(A, B)])
    nz = (A.max(1) > 0)
    print(f"{nm}: all-zero {1-nz.mean():.2f} | ranking defined (both conts) {np.mean(~np.isnan(r)):.2f} | mean rho {np.nanmean(r):.3f} | mean label (nonzero states) {A[nz].mean():.4f} | mean spread max-min {np.mean(A.max(1)-A.min(1)):.4f}")
    if nm == "H=8":
        for ln in sorted({k[0] for k in keys}):
            ii = [i for i, k in enumerate(keys) if k[0] == ln]
            print(f"   {ln:40s} all-zero {1-nz[ii].mean():.2f} defined {np.mean(~np.isnan(r[ii])):.2f} rho {np.nanmean(r[ii]):.3f}")
A4 = np.array([a[k]["V"][0] for k in keys]); A8 = np.array([b[k]["V"][0] for k in keys])
m = (A4.max(1) > 0) & (A8.max(1) > 0)
print("states with signal at both horizons:", m.sum(), "| top-1 AF agrees H4 vs H8 there:", np.mean(A4[m].argmax(1) == A8[m].argmax(1)).round(2),
      "| mean Spearman H4 vs H8 labels:", np.nanmean([rho(x, y) for x, y in zip(A4[m], A8[m])]).round(3))

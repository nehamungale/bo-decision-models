"""Pre-declared diagnostic of PLAN_nonstationary.md 9g: why does surprise-based CUSUM exceed the false-alarm budget? cusum_keep h = 32, v* on the 40 tuning pairs, seed 4900.
Output meta/results/diag_cusum.txt. Run: python -m meta.diag_cusum"""
import os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meta import nonstat as ns
from meta.nonstat_pairs import load_pairs
H = os.path.dirname(os.path.abspath(__file__)); L = []
def P(*a): s = " ".join(str(x) for x in a); L.append(s); print(s, flush=True)
rows = []
for p in load_pairs()["tuning"]:
    pair = ns.make_pair(p["A"], p["B"], 1.0)
    for stat in (True, False):
        r = ns.run_ns(pair, 4900, "cusum_keep", stationary=stat, frac=0.5, cusum_h=32.0, var_stale=0.01)
        for (t, z, d, sd, rev) in r["zlog"]: rows.append((p["A"] + p["B"], stat, t, abs(z), d, sd, rev, r["Tc"], bool(r["triggers"])))
Z = np.array([(x[3], x[4], x[5]) for x in rows if np.isfinite(x[4])]); st = np.array([x[1] for x in rows if np.isfinite(x[4])])
P(f"steps logged: {len(Z)}; median |z| {np.median(Z[:,0]):.2f} (calibrated reference 0.67 median, 0.8 mean); mean |z| {Z[:,0].mean():.2f}")
camp = {}
for x in rows:
    if np.isfinite(x[4]): camp.setdefault((x[0], x[1]), []).append(x)
for lab, flag in (("stationary", True), ("change", False)):
    rd, rs = [], []
    for (n, s_), v in camp.items():
        if s_ != flag or len(v) < 8: continue
        a = np.array([[q[3], q[4], q[5]] for q in v])
        if a[:, 1].std() > 0: rd.append(spearmanr(a[:, 0], a[:, 1])[0])
        if a[:, 2].std() > 0: rs.append(spearmanr(a[:, 0], a[:, 2])[0])
    P(f"{lab}: mean within-campaign Spearman(|z|, distance to nearest earlier observation) {np.nanmean(rd):+.2f}, Spearman(|z|, posterior sd) {np.nanmean(rs):+.2f}  ({len(rd)} campaigns)")
d = Z[st, 1]; z_ = Z[st, 0]; lo, hi = np.quantile(d, [1 / 3, 2 / 3])
P(f"stationary |z| median by distance tertile: low {np.median(z_[d <= lo]):.2f}, mid {np.median(z_[(d > lo) & (d < hi)]):.2f}, high {np.median(z_[d >= hi]):.2f}")
cross = 0; top = 0
for (n, s_), v in camp.items():
    if not s_: continue
    c = 0.0
    for q in v:
        c = max(0.0, c + q[3] - 1.0)
        if c > 32.0:
            cross += 1; top += q[4] >= hi; break
P(f"stationary campaigns with a CUSUM crossing at h = 32 (from the logged steps): {cross} of {sum(1 for k in camp if k[1])}; crossing step in top distance tertile: {top}")
cs = [q for (n, s_), v in camp.items() if not s_ for q in v]
at = [q[3] for q in cs if 0 <= q[2] - q[7] <= 3]; oth = [q[3] for q in cs if not (0 <= q[2] - q[7] <= 3)]
P(f"change runs: mean |z| at T_c..T_c+3 {np.mean(at):.2f} vs other steps {np.mean(oth):.2f}")
P("reading rule (9g): H_explore if mean Spearman(|z|, distance) >= 0.3 and high-tertile median >= 1.5 x low-tertile; H_miscal if low-tertile median |z| > 1.2")
open(os.path.join(H, "results", "diag_cusum.txt"), "w").write("\n".join(L) + "\n")

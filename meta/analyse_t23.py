"""Pre-declared analysis (PLAN_component_triage.md section 3): per component and noise level, hindsight per-landscape best entry (chosen on seeds A, scored on seeds B) vs the best fixed
entry (chosen on seeds A over the other landscapes by mean ratio to the default). TH: mean relative gain >= 5 %, Wilcoxon p < 0.05, gain present in >= 2 landscape families (mean > 0)."""
import glob, os, pickle
import numpy as np
from scipy.stats import wilcoxon
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "t23")
D = {}
for f in glob.glob(os.path.join(H, "*.pkl")):
    r = pickle.load(open(f, "rb")); D.setdefault((r["landscape"], r["cfg"], r["noise"], "A" if r["seed"] < 4100 else "B"), []).append(r["auc"])
Ls = sorted({k[0] for k in D}); fam = lambda l: l.split("/")[0].split("_")[0]
COMP = {"K": ["K:rbf_ard", "K:matern52_ard", "K:rbf_iso", "K:rf"], "R": ["K:rbf_ard", "R:0.15", "R:0.30", "R:adaptive"]}
m = lambda l, c, n, s: float(np.mean(D[(l, c, n, s)]))
for n in (0.0, 0.05):
    for comp, menu in COMP.items():
        print(f"\n=== {comp} menu, noise {n:.0%} ({'entries ' + ', '.join(menu)}) ===")
        print(f"{'landscape':42s} " + " ".join(f"{c[2:]:>9s}" for c in menu) + "  | A-best  B(A-best) B(fixed)  B(default)")
        gains = {}; sel_count = {}
        for l in Ls:
            ratio = {c: np.mean([m(o, c, n, "A") / m(o, "K:rbf_ard", n, "A") for o in Ls if o != l]) for c in menu}      # ratio to default on other landscapes (A seeds)
            fixed = min(ratio, key=ratio.get)
            bestA = min(menu, key=lambda c: m(l, c, n, "A"))
            hb, fb, db = m(l, bestA, n, "B"), m(l, fixed, n, "B"), m(l, "K:rbf_ard", n, "B")
            gains[l] = (fb - hb) / fb if fb > 0 else 0.0; sel_count[fixed] = sel_count.get(fixed, 0) + 1
            print(f"{l:42s} " + " ".join(f"{m(l,c,n,'B'):9.4f}" for c in menu) + f"  | {bestA[2:]:>8s} {hb:9.4f} {fb:9.4f} {db:10.4f}")
        v = np.array(list(gains.values())); p = wilcoxon(v, alternative="greater").pvalue if np.any(v != 0) else 1.0
        fams = {f: float(np.mean([gains[l] for l in Ls if fam(l) == f])) for f in sorted({fam(l) for l in Ls})}
        ok = v.mean() >= 0.05 and p < 0.05 and sum(x > 0 for x in fams.values()) >= 2
        mean_def = np.mean([m(l, "K:rbf_ard", n, "B") for l in Ls]); mean_fix = np.mean([m(l, min(menu, key=lambda c: np.mean([m(o, c, n, 'A') / m(o, 'K:rbf_ard', n, 'A') for o in Ls if o != l])), n, "B") for l in Ls])
        print(f"hindsight(A->B) vs best fixed: mean gain {v.mean():+.3f}, wins {int((v>0).sum())}/{len(v)}, p={p:.4f}, by family {({k: round(x,3) for k,x in fams.items()})}; fixed entry chosen across folds {sel_count}; mean B AUC default {mean_def:.4f} vs fixed-chosen {mean_fix:.4f}")
        print(f"TH-{comp} (noise {n:.0%}):", "PASS" if ok else "FAIL")

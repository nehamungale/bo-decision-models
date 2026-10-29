"""T2 of PLAN_fallback_detection.md. Primary k = 10, m = n0, remedies R (random) and F (farthest-point). Landscape-level rows (150): AUCs are seed-averaged (W1 and injection arms, 4 seeds).
Label: injection arm beats W1 by > 1 % of W1's AUC. Decision realised AUC = injection arm's AUC if flagged else W1's (injection cost already inside the arm's AUC).
Detectors: S0 step-0 features only (logreg, the required comparator); ON online (step-0 + step-k, logreg); RS online + source reference percentile (deployable); RT online + target random-run percentile
(upper bound, not deployable); thresholds on gain_k, ref_src_pct (deployable) and ref_tgt_pct (upper bound): flag if feature <= theta, theta from training-fold quantiles 5..50 % minimising training realised AUC.
No GBM (fewer than 20 positives). logreg: L2, C = 1, standardised, median-imputed, flag if P > 0.5. CV: GroupKFold(5) by dataset and leave-one-family-out. Shuffled controls x5.
Gate (per remedy, per detector, both CV schemes): captures >= 30 % of the realised oracle gain over always-W1 (oracle = best of W1, R10, F10 per landscape), realised AUC below the S0 detector and below always-W1
(best fixed arm), paired Wilcoxon p < 0.05 versus S0 and versus always-W1, and below the shuffled control. Raw p-values; 6 deployable tests per scheme, so a Bonferroni level of 0.0083 is also shown."""
import os, glob, pickle
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
H = os.path.dirname(os.path.abspath(__file__))
F0 = pd.read_csv(os.path.join(H, "results", "fallback_features.csv")); F1 = pd.read_csv(os.path.join(H, "results", "detect_features.csv"))
F = F0.merge(F1, on="landscape")
U = {}
for f in glob.glob(os.path.join(H, "results", "detect", "*.pkl")):
    r = pickle.load(open(f, "rb")); U[(r["landscape"], r["seed"])] = r
auc = lambda l, a: np.mean([U[(l, s)]["arms"][a]["auc"] for s in range(4300, 4304)])
W1 = np.array([auc(l, "W1") for l in F.landscape]); AR = {"R": np.array([auc(l, "W1R10") for l in F.landscape]), "F": np.array([auc(l, "W1F10") for l in F.landscape])}
orc = np.minimum(W1, np.minimum(AR["R"], AR["F"])); gain = W1.mean() - orc.mean()
SRC = ["src_success", "src_mean_start_regret", "src_median_ratio", "src_best_mean", "src_best_std", "n_train"]
TRG = ["sim_max", "sim_top5", "sim_mean", "spearman_src", "r2_src", "obs_cv", "obs_skew", "obs_max_z", "best_is_src_top", "loo_rmse", "ls_mean", "ls_min", "gp_noise"]
STK = ["spearman_k", "r2_k", "gain_k", "inc_z_k", "since_k", "loo_k", "ls_mean_k", "ls_min_k", "gpn_k"]
SETS = {"S0": SRC + TRG, "ON": SRC + TRG + STK, "RS": SRC + TRG + STK + ["ref_src_pct"], "RT": SRC + TRG + STK + ["ref_src_pct", "ref_tgt_pct"]}
THR = {"thr_gain": "gain_k", "thr_refsrc": "ref_src_pct", "thr_reftgt": "ref_tgt_pct"}
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

def splits(scheme):
    g = F["dataset"].to_numpy() if scheme == "dataset" else F["family"].to_numpy()
    if scheme == "dataset": return list(GroupKFold(5).split(F, groups=g))
    return [(np.where(g != x)[0], np.where(g == x)[0]) for x in np.unique(g)]

def decide(det, X, y, arm, scheme, shuffle=False, seed=0):
    rng = np.random.default_rng(seed); flag = np.zeros(len(F), bool)
    for tr, te in splits(scheme):
        ytr = rng.permutation(y[tr]) if shuffle else y[tr]
        if det in SETS:
            Xa = F[SETS[det]].to_numpy(float)
            if ytr.min() == ytr.max(): flag[te] = bool(ytr[0]); continue
            flag[te] = mk().fit(Xa[tr], ytr).predict_proba(Xa[te])[:, 1] > 0.5
        else:
            x = F[THR[det]].to_numpy(float); xt = np.where(np.isnan(x), np.nanmedian(x[tr]), x)
            best, th = W1[tr].mean(), -np.inf
            for q in range(5, 55, 5):
                t_ = np.quantile(xt[tr], q / 100); a = np.where(xt[tr] <= t_, arm[tr], W1[tr]).mean()
                if shuffle: a = np.where(xt[tr] <= t_, arm[tr], W1[tr]).mean()
                if a < best: best, th = a, t_
            flag[te] = xt[te] <= th
    return np.where(flag, arm, W1), flag

lines = []
def P(*a): s = " ".join(str(x) for x in a); lines.append(s); print(s)
P(f"{len(F)} landscapes. always-W1 {W1.mean():.4f}; realised oracle {orc.mean():.4f} ({gain/W1.mean():.1%} below W1); 30 % of it = {0.3*gain/W1.mean():.1%} of W1's AUC.")
for rem, arm in AR.items():
    y = (arm < 0.99 * W1).astype(int); P(f"\n##### remedy {rem}: positives (injection helps by >1 %) {y.sum()}/{len(y)}; always-inject {arm.mean():.4f} ({arm.mean()/W1.mean()-1:+.1%})")
    for scheme in ("dataset", "family"):
        P(f"--- CV by {scheme} ---")
        s0, _ = decide("S0", None, y, arm, scheme)
        for det in list(SETS) + list(THR):
            real, flag = decide(det, None, y, arm, scheme)
            sh = np.mean([decide(det, None, y, arm, scheme, True, s)[0].mean() for s in range(5)])
            cap = (W1.mean() - real.mean()) / gain
            d1 = W1 - real; p1 = wilcoxon(d1, alternative="greater").pvalue if np.any(d1 != 0) else 1.0
            d0 = s0 - real; p0 = wilcoxon(d0, alternative="greater").pvalue if np.any(d0 != 0) and det != "S0" else 1.0
            dep = det not in ("RT", "thr_reftgt")
            ok = cap >= 0.30 and real.mean() < s0.mean() and p1 < 0.05 and p0 < 0.05 and real.mean() < sh and det != "S0"
            P(f"{det:10s} flagged {flag.sum():3d} (true positives {int((flag & (y==1)).sum()):2d}) | realised {real.mean():.4f} ({(real.mean()-W1.mean())/W1.mean():+.1%} vs W1) | captures {cap:+.0%} of oracle | shuffled {sh:.4f} | p vs W1 {p1:.3f}, vs S0 {p0:.3f} | {'PASS' if ok else 'FAIL'}{'' if dep else ' (upper bound, not deployable)'}")
P("\nA PASS needs: captures >= 30 % of the realised oracle gain, below the step-0 detector and below always-W1 with p < 0.05 each, and below its shuffled control.")
open(os.path.join(H, "results", "decide_detect.txt"), "w").write("\n".join(lines) + "\n")

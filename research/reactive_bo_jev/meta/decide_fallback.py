"""Pre-declared decision test (fall-back triage step 2). Label: W1 worse than random (mean AUC over seeds). Models: L2 logistic regression (C=1, standardised features, median-imputed) and
shallow HistGradientBoosting (max_depth 3, 100 iters), feature sets SRC (source-only), TRG (target observations, incl. similarity), ALL. Decision: fall back to random start if P(fail) > 0.5.
Realised cost-free: AUC = random-start mean if fall-back else W1 AUC (optimistic: a true post-portfolio fall-back also spends the n0 portfolio evaluations).
CV: GroupKFold(5) by dataset name (TabRepo datasets recur across families), and leave-one-family-out. Controls: label-shuffled (x5), always-W1, hindsight.
PASS: mean realised AUC at least 5 % below always-W1, paired Wilcoxon p<0.05 across landscapes, and better than the shuffled-label mean. TRG/ALL evaluated at noise 0 only (features logged noise-free)."""
import glob, os, pickle
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
H = os.path.dirname(os.path.abspath(__file__))
F = pd.read_csv(os.path.join(H, "results", "fallback_features.csv"))
A = {}
for f in glob.glob(os.path.join(H, "results", "w150", "*.pkl")):
    r = pickle.load(open(f, "rb"))
    if not r["skipped"]: A.setdefault((r["landscape"], r["arm"], r["noise"]), []).append(r["auc"])
SRC = ["src_success", "src_mean_start_regret", "src_median_ratio", "src_best_mean", "src_best_std", "n_train"]
TRG = ["sim_max", "sim_top5", "sim_mean", "spearman_src", "r2_src", "obs_cv", "obs_skew", "obs_max_z", "best_is_src_top", "loo_rmse", "ls_mean", "ls_min", "gp_noise"]
SETS = {"SRC": SRC, "TRG": TRG, "ALL": SRC + TRG}
def mk(kind): 
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)) if kind == "logreg" else \
           make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=0))
def cv(kind, cols, y, w, r, scheme, shuffle=False, seed=0):
    X = F[cols].to_numpy(float); pred = np.zeros(len(F), int); rng = np.random.default_rng(seed)
    groups = F["dataset"].to_numpy() if scheme == "dataset" else F["family"].to_numpy()
    splits = GroupKFold(5).split(X, y, groups) if scheme == "dataset" else [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in np.unique(groups)]
    for tr, te in splits:
        ytr = y[tr].copy()
        if shuffle: ytr = rng.permutation(ytr)
        if ytr.min() == ytr.max(): pred[te] = int(ytr[0]); continue
        pred[te] = (mk(kind).fit(X[tr], ytr).predict_proba(X[te])[:, 1] > 0.5).astype(int)
    return np.where(pred == 1, r, w), pred
for n in (0.0, 0.05):
    w = np.array([np.mean(A[(l, "W1", n)]) for l in F.landscape]); r = np.array([np.mean(A[(l, "random", n)]) for l in F.landscape]); y = (w > r).astype(int)
    print(f"\n=== noise {n:.0%}: always-W1 {w.mean():.4f}, random {r.mean():.4f}, hindsight {np.minimum(w, r).mean():.4f}, failures {y.sum()}/{len(y)} ===")
    for sname in ("SRC", "TRG", "ALL"):
        if n > 0 and sname != "SRC": continue
        for kind in ("logreg", "gbm"):
            for scheme in ("dataset", "family"):
                real, pred = cv(kind, SETS[sname], y, w, r, scheme)
                sh = np.mean([cv(kind, SETS[sname], y, w, r, scheme, True, s)[0].mean() for s in range(5)])
                d = w - real; p = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
                ok = (w.mean() - real.mean()) / w.mean() >= 0.05 and p < 0.05 and real.mean() < sh
                print(f"{sname} {kind:6s} CV-by-{scheme:7s}: realised {real.mean():.4f} ({(w.mean()-real.mean())/w.mean():+.1%} vs always-W1) | shuffled {sh:.4f} | fall-backs {pred.sum():3d} | p={p:.3f} | {'PASS' if ok else 'FAIL'}")

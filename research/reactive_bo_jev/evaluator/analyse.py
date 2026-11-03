"""Analysis for the pre-registered offline evaluator study (see notes.md). No tuning: logistic
regression C=1.0, leave-one-dataset-out. Run: python -m evaluator.analyse evaluator/data/states_tev1.jsonl"""
import sys, json, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

CODE = ["value", "pct", "gap_median", "gap_best_done", "slope3", "slope_mean", "noise", "max_drop", "change_half"]
NOUL = ["will_finish_top10", "above_most_peers", "plateaued", "improving_fast", "unstable", "will_beat_best_so_far"]
MODEL = NOUL + ["promise", "promise_conf", "p_stop", "action_conf"]


def load(path):
    rows = [json.loads(l) for l in open(path)]
    df = pd.DataFrame([{**{k: r[k] for k in ("dataset", "config", "epoch", "y", "latency")},
                        **r["feats"], **r["model"]} for r in rows])
    for k in NOUL:                                   # logit of probabilities, clipped
        p = df[k].clip(1e-4, 1 - 1e-4); df[k + "_logit"] = np.log(p / (1 - p))
    return df


def logit_cols(cols): return [c + "_logit" if c in NOUL else c for c in cols]


def lodo_scores(df, cols):
    """Out-of-fold predicted probability, fitting on all other datasets, separately per epoch."""
    out = pd.Series(np.nan, index=df.index)
    for e, de in df.groupby("epoch"):
        for ds in de.dataset.unique():
            tr, te = de[de.dataset != ds], de[de.dataset == ds]
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
            m.fit(tr[cols], tr.y); out[te.index] = m.predict_proba(te[cols])[:, 1]
    return out


def per_ds_auc(df, score):
    r = {}
    for ds, d in df.assign(s=score).groupby("dataset"):
        if 0 < d.y.sum() < len(d): r[ds] = roc_auc_score(d.y, d.s)
    return pd.Series(r)


def ece(p, y, bins=10):
    idx = np.minimum((np.asarray(p) * bins).astype(int), bins - 1); tot, rel = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(np.mean(np.asarray(y)[m]) - np.mean(np.asarray(p)[m]))
            rel.append((b / bins, int(m.sum()), float(np.mean(np.asarray(p)[m])), float(np.mean(np.asarray(y)[m]))))
    return tot, rel


def main(path):
    df = load(path)
    print(f"{len(df)} states, {df.dataset.nunique()} datasets, base rate {df.y.mean():.3f}")
    sets = {"A code": CODE, "B model": logit_cols(MODEL), "C code+model": CODE + logit_cols(MODEL)}
    print("\n== Per-dataset AUROC, mean over datasets (leave-one-dataset-out), by epoch ==")
    res = {}
    for e in [None] + sorted(df.epoch.unique()):
        d = df if e is None else df[df.epoch == e]
        line = {}
        for name, cols in sets.items():
            s = pd.concat([lodo_scores(d, cols)]) if e is not None else pd.concat(
                [lodo_scores(df[df.epoch == x], cols) for x in sorted(df.epoch.unique())]).sort_index()
            res[(e, name)] = per_ds_auc(d, s); line[name] = res[(e, name)].mean()
        line["D will_top10 raw"] = per_ds_auc(d, d.will_finish_top10).mean()
        line["E peer pct"] = per_ds_auc(d, d.pct).mean()
        print("epoch", "all" if e is None else e, {k: round(v, 3) for k, v in line.items()})
        if e is None:
            res[(e, "D")] = per_ds_auc(d, d.will_finish_top10)
    print("\n== H1: will_finish_top10 AUROC vs 0.5 (per dataset, pooled epochs) ==")
    D = res[(None, "D")]; print(D.round(3).to_dict()); print("Wilcoxon vs 0.5 p =", round(wilcoxon(D - 0.5).pvalue, 4))
    print("\n== H2: incremental value, C minus A, per-dataset AUROC difference ==")
    for e in [None] + sorted(df.epoch.unique()):
        diff = res[(e, "C code+model")] - res[(e, "A code")]
        p = wilcoxon(diff).pvalue if (diff != 0).any() else float("nan")
        print("epoch", "all" if e is None else e, f"mean diff {diff.mean():+.4f}, datasets C>A {int((diff > 0).sum())}/{len(diff)}, Wilcoxon p={p:.3f}")
    print("\n== H3: calibration of will_finish_top10 (raw) ==")
    e_, rel = ece(df.will_finish_top10, df.y)
    brier = ((df.will_finish_top10 - df.y) ** 2).mean(); base = ((df.y.mean() - df.y) ** 2).mean()
    print(f"ECE {e_:.3f}; Brier {brier:.3f} vs base-rate Brier {base:.3f}; mean predicted {df.will_finish_top10.mean():.3f} vs base {df.y.mean():.3f}")
    for lo, n, pm, ym in rel: print(f"  bin>={lo:.1f} n={n:5d} mean_pred={pm:.3f} observed={ym:.3f}")
    err = -(df.will_finish_top10 - df.y).abs()
    print("Spearman(promise_conf, -|error|) =", round(spearmanr(df.promise_conf, err)[0], 3),
          "| Spearman(action_conf, -|error|) =", round(spearmanr(df.action_conf, err)[0], 3))
    print("\n== State sensitivity: does each answer track its matching code feature? (Spearman, pooled) ==")
    pairs = [("above_most_peers", "pct"), ("plateaued", "slope3"), ("improving_fast", "slope3"),
             ("unstable", "noise"), ("will_beat_best_so_far", "gap_best_done"), ("will_finish_top10", "pct"),
             ("promise", "pct"), ("p_stop", "pct")]
    for q, f in pairs: print(f"  {q:24s} vs {f:14s} rho = {spearmanr(df[q], df[f])[0]:+.3f}")
    print("\n== Latency of the 8-question call (s) ==", df.latency.describe()[["mean", "50%", "max"]].round(3).to_dict(),
          "p95", round(df.latency.quantile(0.95), 3))


if __name__ == "__main__":
    main(sys.argv[1])

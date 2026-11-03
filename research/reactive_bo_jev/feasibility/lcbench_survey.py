"""Survey all LCBench datasets: how informative are partial curves, how long are runs.
Objective: val_balanced_accuracy (higher is better). Run: python -I lcbench_survey.py [out.csv]"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from syne_tune.blackbox_repository import load_blackbox

OBJ, EPOCHS = "val_balanced_accuracy", (1, 5, 10, 25)
bb = load_blackbox("lcbench")
rows = []
for name in sorted(bb):
    b = bb[name]
    ev = b.objectives_evaluations[:, 0]                  # (configs, epochs, objs)
    y = ev[:, :, b.objectives_names.index(OBJ)]
    t = ev[:, :, b.objectives_names.index("time")]
    fin = y[:, -1]
    top = fin >= np.quantile(fin, 0.9)                   # true top-10 % by final
    r = dict(dataset=name, n_nan=int(np.isnan(y).sum()), final_std=fin.std(),
             final_best=fin.max(), run_time_med=np.median(t[:, -1]),
             epoch_time_med=np.median(t[:, -1] / y.shape[1]))
    for e in EPOCHS:
        r[f"rho@{e}"] = spearmanr(y[:, e - 1], fin)[0]
    # early-stop risk: share of true top-10 % outside the top-30 % at epoch 10
    early_top30 = y[:, 9] >= np.quantile(y[:, 9], 0.7)
    r["top10_lost@e10_cut30"] = float((top & ~early_top30).sum() / top.sum())
    rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(df.round(3).to_string(index=False))
print(df[[f"rho@{e}" for e in EPOCHS] + ["run_time_med", "top10_lost@e10_cut30"]].describe().round(3))
if len(sys.argv) > 1:
    df.to_csv(sys.argv[1], index=False)

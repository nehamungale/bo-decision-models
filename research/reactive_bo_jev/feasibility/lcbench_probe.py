# Feasibility probe for LCBench via Syne Tune (run in a throwaway venv: syne-tune huggingface_hub pandas pyarrow fastparquet scipy)
# Run with: python -I lcbench_probe.py
import numpy as np
from syne_tune.blackbox_repository import load_blackbox
bb = load_blackbox("lcbench")
names = list(bb.keys()); print(len(names), "datasets; first 3:", names[:3])
b = bb[names[0]]
print("n configs:", len(b.hyperparameters), "| objectives:", b.objectives_names)
print("fidelity:", b.fidelity_name, "n=", len(b.fidelity_values), "last=", b.fidelity_values[-1])
print("train_runs tensor shape (configs, seeds?, fidelities, objs):", getattr(b,'objectives_evaluations',np.zeros(0)).shape)
ev = b.objectives_evaluations
ie = b.objectives_names.index("val_balanced_accuracy")
it = [i for i,n in enumerate(b.objectives_names) if "time" in n]
print("time objs:", [b.objectives_names[i] for i in it])
# final-vs-early rank correlation: how predictive is a partial curve?
from scipy.stats import spearmanr
fin = ev[:,:,-1,ie].mean(1)
for e in (1,3,5,10,25):
    print(f"epoch {e:2d}: spearman(early, final) = {spearmanr(ev[:,:,e-1,ie].mean(1), fin)[0]:.2f}")
print("seed noise: mean std across seeds at final =", np.round(ev[:,:,-1,ie].std(1).mean(),4), "vs spread across configs", np.round(fin.std(),4))
if it:
    t = ev[:,:,:,it[0]]; print("per-epoch time (s): median", np.round(np.median(np.diff(t[...,:],axis=2)),2), " total median", np.round(np.median(t[...,-1]),1))

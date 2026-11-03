"""One-off: export LCBench (x -> final validation balanced accuracy) pools to a plain .npz so the BoTorch
environment does not need syne-tune. Run with the venv that has syne-tune: python export_pools.py"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV, CONFIRM
from sim.lcbench_env import load_pool

out = {}
for ds in DEV + CONFIRM:
    p = load_pool(ds)
    out[f"{ds}__X"] = p.X.astype(np.float64)
    out[f"{ds}__y"] = p.y[:, -1].astype(np.float64)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcbench_final_pools.npz")
np.savez_compressed(path, **out)
print("wrote", path, os.path.getsize(path) // 1024, "KB;", len(DEV + CONFIRM), "datasets")

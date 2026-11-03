"""Access to the exported landscapes (meta/pools). Kept landscapes follow the rules in PLAN_meta_selector.md (Phase 1 result)."""
import os
import numpy as np
import pandas as pd
from acq.engine import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS = os.path.join(HERE, "pools")
_cache = {}


def inventory():
    return pd.read_csv(os.path.join(POOLS, "inventory_ext.csv"))


def kept(families=("lcbench", "hpob", "tabrepo")):
    d = inventory()
    return d[d.keep & d.family.isin(families) & (d.group != "lcbench_reserved")].reset_index(drop=True)


def load(landscape):
    fam = inventory().set_index("landscape").loc[landscape, "family"]
    z = _cache.setdefault(fam, np.load(os.path.join(POOLS, f"{fam}.npz")))
    return Pool(landscape, z[f"{landscape}__X"], z[f"{landscape}__y"])


def budget(pool):
    return int(min(50, max(10, 0.25 * len(pool.y))))

"""LCBench replay environment. Time is in units of the dataset's median epoch time, so a
decision latency of 0.1 means 'a tenth of an epoch' regardless of the dataset."""
from dataclasses import dataclass
import numpy as np

OBJ = "val_balanced_accuracy"   # higher is better


@dataclass
class Pool:
    name: str
    X: np.ndarray        # (N, 7) encoded hyperparameters in [0, 1]
    y: np.ndarray        # (N, E) objective after each epoch
    cum_t: np.ndarray    # (N, E) cumulative training time, in epoch units
    unit_s: float        # seconds per time unit (median epoch time)

    @property
    def n(self): return self.y.shape[0]
    @property
    def n_epochs(self): return self.y.shape[1]
    def dt(self, i, e):  # duration of epoch e (1-indexed) of config i
        return self.cum_t[i, e - 1] - (self.cum_t[i, e - 2] if e > 1 else 0.0)


def _encode(h):
    log = lambda s: np.log(s.astype(float))
    cols = [log(h.batch_size), h.max_dropout.astype(float), log(h.max_units),
            # LCBench stores 1 layer as the string 'True' (parsed from '1'); map it back.
            h.num_layers.map(lambda v: 1 if v == "True" else int(v)).astype(float),
            log(h.learning_rate), h.momentum.astype(float), log(h.weight_decay)]
    X = np.column_stack([np.asarray(c, float) for c in cols])
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def load_pool(name):
    from syne_tune.blackbox_repository import load_blackbox
    b = load_blackbox("lcbench")[name]
    ev = b.objectives_evaluations[:, 0].astype(np.float64)   # float32 times accumulate clock drift
    y = ev[:, :, b.objectives_names.index(OBJ)]
    t = ev[:, :, b.objectives_names.index("time")]
    unit = float(np.median(t[:, -1] / t.shape[1]))
    assert not np.isnan(y).any() and (np.diff(t, axis=1) > 0).all()
    return Pool(name, _encode(b.hyperparameters), y, t / unit, unit)

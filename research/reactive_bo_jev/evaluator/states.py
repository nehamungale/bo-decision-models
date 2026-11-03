"""Build the state text and code-computed features for one (pool, config, epoch)."""
import numpy as np

N_PEERS, N_COMPLETED = 30, 10


def peers_for(pool, i, e, seed):
    rng = np.random.default_rng([seed, i, 7])          # same peers at every epoch for a config
    others = np.delete(np.arange(pool.n), i)
    return rng.choice(others, N_PEERS, replace=False)


def build(pool, i, e, seed):
    """Returns (text, features dict, label). Label: config ends in the top 10 % by final metric."""
    peers = peers_for(pool, i, e, seed)
    curve = pool.y[i, :e]
    pe = pool.y[peers, e - 1]
    best_done = float(pool.y[peers[:N_COMPLETED], -1].max())
    pct = float((pe < curve[-1]).mean())
    q = np.quantile(pe, [0.5, 0.75, 0.9])
    text = (
        f"Hyperparameter tuning of a neural network classifier. One training run is at epoch {e} "
        f"of {pool.n_epochs}. Validation balanced accuracy after each epoch: "
        + ", ".join(f"{v:.3f}" for v in curve) + ". "
        f"For comparison, {N_PEERS} other runs at the same epoch have median {q[0]:.3f}, "
        f"75th percentile {q[1]:.3f}, 90th percentile {q[2]:.3f}, best {pe.max():.3f}. "
        f"This run is better than {100 * pct:.0f}% of them. "
        f"The best fully trained run so far finished at {best_done:.3f}."
    )
    d = np.diff(curve) if e > 1 else np.zeros(1)
    k = min(3, len(d))
    feats = dict(
        value=float(curve[-1]), pct=pct, gap_median=float(curve[-1] - q[0]),
        gap_best_done=float(curve[-1] - best_done), slope3=float(d[-k:].mean()),
        slope_mean=float(d.mean()), noise=float(d.std()), max_drop=float(min(d.min(), 0.0)),
        change_half=float(curve[-1] - curve[max(e // 2 - 1, 0)]),
    )
    thr = np.quantile(pool.y[:, -1], 0.9)
    return text, feats, int(pool.y[i, -1] >= thr)

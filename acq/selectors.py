"""Numeric AF selectors for Phase B. All see only the Step (state, history, posterior), never y of unevaluated points."""
import os, json, pickle, glob
import numpy as np
import torch
from acq.engine import AFS, STATE_KEYS, HERE

HIST_KEYS = [f"{a}_used" for a in AFS] + [f"{a}_improved" for a in AFS] + ["last_af", "last_improved"]
NOHIST_KEYS = [k for k in STATE_KEYS if k not in HIST_KEYS]
AF_IDX = {a: i for i, a in enumerate(AFS)}


class GPHedge:
    """Hoffman, Brochu & de Freitas (2011) portfolio: gain of AF i is the standardised posterior mean at its past proposals."""
    name = "hedge"
    def __init__(self, eta=1.0): self.eta = eta
    def reset(self, seed):
        self.G = np.zeros(len(AFS)); self.prev = None; self.rng = np.random.default_rng(seed + 777)
    def __call__(self, step, peek):
        if self.prev is not None:
            X = torch.tensor(step.pool.X[self.prev])
            with torch.no_grad():
                m = step.model.posterior(X.unsqueeze(1)).mean.reshape(-1).numpy()
            self.G += (m - step.y_obs.mean()) / (step.y_obs.std() + 1e-12)
        w = np.exp(self.eta * (self.G - self.G.max())); p = w / w.sum()
        self.prev = list(step.proposals)
        return int(self.rng.choice(len(AFS), p=p))


class RuleSelector:
    """Transparent decision list on the shared state. See PLAN_acquisition_choice.md, Amendment 1."""
    def __init__(self, explore="UCB", exploit="PosMean", default="LogEI", a=0.0, b=0.0, k=3, win_stay=False):
        self.p = dict(explore=explore, exploit=exploit, default=default, a=a, b=b, k=k, win_stay=win_stay)
        self.name = "rule"
    def __call__(self, step, peek):
        p, s = self.p, step.state
        used = 1.0 - s["frac_left"]
        if s["frac_left"] < p["b"]: return AF_IDX[p["exploit"]]
        if used < p["a"]: return AF_IDX[p["explore"]]
        if p["win_stay"] and s["last_improved"] == 1: return int(s["last_af"])
        if s["since_improve"] >= p["k"]: return AF_IDX[p["explore"]]
        return AF_IDX[p["default"]]


RULE_SPACE = dict(explore=["UCB", "TS", "PosSTD"], exploit=["PosMean", "LogEI", "LogPI"],
                  default=["LogEI", "LogPI", "TS", "UCB"], a=[0.0, 0.1, 0.2, 0.3], b=[0.0, 0.1, 0.2, 0.3],
                  k=[1, 2, 3, 5, 8], win_stay=[True, False])


def sample_rule_params(rng):
    return {k: v[int(rng.integers(len(v)))] for k, v in RULE_SPACE.items()}


# ── learned numeric selector ───────────────────────────────────────────────────

def load_training_table(exclude_dataset=None, cond_filter=None):
    """Rows from Phase A logs: state features, and for each AF the normalised one-step gain of its proposal."""
    from acq.engine import load_pool
    X, Y = [], []
    pools = {}
    for f in glob.glob(os.path.join(HERE, "results", "phaseA", "*.pkl")):
        r = pickle.load(open(f, "rb"))
        if r["dataset"] == exclude_dataset: continue
        if cond_filter and not cond_filter(r["cond"]): continue
        p = pools.setdefault(r["dataset"], load_pool(r["dataset"]))
        span = max(p.y_max - p.y_med, 1e-12)
        rng = np.random.default_rng(r["seed"]); n0 = 2 * p.X.shape[1] + 1
        best = float(p.y[rng.choice(len(p.y), n0, replace=False)].max())
        for st in r["log"]:
            gains = [max(0.0, y - best) / span for y in st["proposal_y"]]
            X.append([st["state"][k] for k in STATE_KEYS]); Y.append(gains)
            best = max(best, st["y"])
    return np.array(X, float), np.array(Y, float)


class LearnedSelector:
    """Argmax of per-AF predicted one-step gain; trained leave-one-dataset-out on Phase A logs (off-policy)."""
    def __init__(self, heldout, variant="L1"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.name = f"learned_{variant}"
        if variant == "L1": keys, filt = NOHIST_KEYS, None
        else:               keys, filt = STATE_KEYS, (lambda c: c == "random")
        self.cols = [STATE_KEYS.index(k) for k in keys]
        X, Y = load_training_table(heldout, filt)
        self.models = [HistGradientBoostingRegressor(random_state=0).fit(X[:, self.cols], Y[:, i]) for i in range(len(AFS))]
        self.n_train = len(X)
    def __call__(self, step, peek):
        x = np.array([[step.state[k] for k in STATE_KEYS]], float)[:, self.cols]
        pred = np.array([m.predict(x)[0] for m in self.models])
        return int(np.argmax(pred))


class FixedMixture:
    """State-blind control: draw the AF i.i.d. from fixed weights. Weights are tev1-sample's observed AF shares on the dev run
    (LogEI .070, LogPI .348, UCB .159, TS .170, PosMean .048, PosSTD .205), fixed before this control was run."""
    name = "fixed_mixture_tev1"
    W = np.array([0.070, 0.348, 0.159, 0.170, 0.048, 0.205])
    def reset(self, seed): self.rng = np.random.default_rng(seed + 9091)
    def __call__(self, step, peek): return int(self.rng.choice(len(AFS), p=self.W / self.W.sum()))


# ── Amendment 4 baselines ──────────────────────────────────────────────────────

class BetaBandit:
    """Non-contextual Thompson sampling over the six AFs; reward = the evaluated point improved the incumbent."""
    name = "bandit_ts"
    def reset(self, seed):
        self.a = np.ones(len(AFS)); self.b = np.ones(len(AFS)); self.rng = np.random.default_rng(seed + 4242)
    def __call__(self, step, peek): return int(np.argmax(self.rng.beta(self.a, self.b)))
    def observe(self, step, af, y, improved):
        if improved: self.a[af] += 1
        else: self.b[af] += 1


class Schedule:
    """Fixed schedule: `first` for the first `frac` of the budget, then `then` (Benjamins et al. 2022: LogEI 25 %, then LogPI)."""
    def __init__(self, first="LogEI", then="LogPI", frac=0.25):
        self.first, self.then, self.frac = AF_IDX[first], AF_IDX[then], frac; self.name = f"sched_{first}_{then}"
    def __call__(self, step, peek): return self.first if 1.0 - step.state["frac_left"] < self.frac else self.then


class DwellRule:
    """Wraps a RuleSelector: a chosen AF is held for at least `dwell` steps before the rule may change it."""
    def __init__(self, rule, dwell):
        self.rule, self.dwell = rule, dwell; self.name = f"rule_dwell{dwell}"
    def reset(self, seed): self.cur, self.held = None, 0
    def __call__(self, step, peek):
        want = self.rule(step, peek)
        if self.cur is None or self.held >= self.dwell: self.cur, self.held = want, 0
        self.held += 1
        return self.cur

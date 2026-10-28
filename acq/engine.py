"""Sequential BO on a discrete candidate pool with a menu of acquisition functions (AFs).

One GP (BoTorch 0.18 SingleTaskGP default: ARD RBF kernel with dimension-scaled lengthscale priors, standardised y) is fitted per step. All six AFs are scored on every
unevaluated pool point, so every step yields six proposals plus a state vector; a *selector* then picks which AF's
proposal is evaluated. Maximisation. Common random numbers: GP fit and Thompson sampling are seeded per (seed, step),
so two selectors that reach the same data see identical proposals.
"""
import os
import warnings
from dataclasses import dataclass, field
import numpy as np
import torch
from scipy.stats import norm as _norm

warnings.filterwarnings("ignore")
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)
HERE = os.path.dirname(os.path.abspath(__file__))

CAND_KEYS = ["gap", "sdn", "z", "pi", "ei", "dist_inc", "dist_nn", "mu_rank", "sd_rank"]   # per-proposal features (record=True only)
AFS = ["LogEI", "LogPI", "UCB", "TS", "PosMean", "PosSTD"]
UCB_SIGMAS = 2.0           # UCB = mean + 2 * std
AF_GLOSS = {               # one plain sentence per AF, used as menu criteria for model selectors
    "LogEI": "Expected improvement: balances a high predicted value against uncertainty; the standard default.",
    "LogPI": "Probability of improvement: picks the point most likely to beat the incumbent by any amount; tends to exploit.",
    "UCB": "Upper confidence bound: predicted value plus two standard deviations; explores uncertain regions.",
    "TS": "Thompson sampling: optimum of one random function drawn from the model; randomised exploration.",
    "PosMean": "Posterior mean: the point with the best predicted value; pure exploitation.",
    "PosSTD": "Posterior standard deviation: the most uncertain point; pure exploration.",
}


def load_pool(name, path=os.path.join(HERE, "lcbench_final_pools.npz")):
    z = np.load(path)
    return Pool(name, z[f"{name}__X"], z[f"{name}__y"])


@dataclass
class Pool:
    name: str
    X: np.ndarray       # (N, D) in [0, 1]
    y: np.ndarray       # (N,) higher is better

    def __post_init__(self):
        self.y_max = float(self.y.max())
        self.y_med = float(np.median(self.y))

    def regret(self, best):                       # normalised: 0 = pool optimum, 1 = pool median
        return (self.y_max - best) / max(self.y_max - self.y_med, 1e-12)


def _fit(X, y, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        Xt = torch.tensor(X); Yt = torch.tensor(y).unsqueeze(-1)
        model = SingleTaskGP(Xt, Yt, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model.eval()


@dataclass
class Step:
    t: int
    proposals: list            # pool index proposed by each AF (order of AFS)
    mu: np.ndarray             # posterior mean on candidates
    sd: np.ndarray             # posterior std on candidates
    cand: np.ndarray           # candidate pool indices (the unevaluated ones)
    state: dict
    model: object = None
    pool: object = None
    obs_idx: np.ndarray = None
    y_obs: np.ndarray = None


def propose_all(pool, obs_idx, y_obs, seed, t):
    """Fit the GP and return the six proposals + posterior marginals on the unevaluated pool points."""
    mask = np.ones(pool.X.shape[0], bool); mask[obs_idx] = False
    cand = np.where(mask)[0]
    model = _fit(pool.X[obs_idx], y_obs, seed * 100003 + t)
    Xc = torch.tensor(pool.X[cand])
    best = float(np.max(y_obs))
    with torch.no_grad():
        post = model.posterior(Xc.unsqueeze(1))                       # marginals: (n, 1, 1)
        mu = post.mean.reshape(-1).numpy(); sd = post.variance.clamp_min(1e-18).sqrt().reshape(-1).numpy()
        logei = LogExpectedImprovement(model, best_f=best)(Xc.unsqueeze(1)).numpy()
        logpi = LogProbabilityOfImprovement(model, best_f=best)(Xc.unsqueeze(1)).numpy()
        with torch.random.fork_rng():                                  # joint Thompson draw over all candidates
            torch.manual_seed(seed * 7919 + t)
            ts = model.posterior(Xc).rsample(torch.Size([1])).reshape(-1).numpy()
    scores = [logei, logpi, mu + UCB_SIGMAS * sd, ts, mu, sd]
    proposals = [int(cand[np.argmax(s)]) for s in scores]
    return model, cand, mu, sd, proposals


def gp_summary(model):
    ls = model.covar_module.lengthscale.detach().reshape(-1).numpy()      # default kernel has no output-scale term
    return dict(ls_min=float(ls.min()), ls_max=float(ls.max()), ls_mean=float(ls.mean()), ls_std=float(ls.std()),
                noise=float(model.likelihood.noise.detach().reshape(-1)[0]))


STATE_KEYS = ["t", "frac_left", "n_obs", "best_z", "since_improve", "n_improve_last10", "last_improve_size",
              "ls_min", "ls_max", "ls_mean", "ls_std", "noise", "sd_mean", "sd_max", "dist_last",
              "n_distinct", "surprise_last"] + [f"{a}_used" for a in AFS] + [f"{a}_improved" for a in AFS] + \
             ["last_af", "last_improved"]


@dataclass
class History:
    best_trace: list = field(default_factory=list)       # incumbent value after each evaluation (post-init)
    improve_steps: list = field(default_factory=list)
    used: np.ndarray = field(default_factory=lambda: np.zeros(len(AFS)))
    improved: np.ndarray = field(default_factory=lambda: np.zeros(len(AFS)))
    last_af: int = -1
    last_improved: int = -1
    last_gain: float = 0.0
    surprise: float = 0.0


def build_state(pool, obs_idx, y_obs, t, T, model, mu, sd, proposals, hist):
    z = (y_obs - y_obs.mean()) / (y_obs.std() + 1e-12)
    best_z = float(z.max())
    since = t - (hist.improve_steps[-1] + 1) if hist.improve_steps else t
    d_last = float(np.min(np.linalg.norm(pool.X[obs_idx[:-1]] - pool.X[obs_idx[-1]], axis=1))) if len(obs_idx) > 1 else 0.0
    s = dict(t=t, frac_left=(T - t) / T, n_obs=len(obs_idx), best_z=best_z, since_improve=since,
             n_improve_last10=sum(1 for k in hist.improve_steps if k >= t - 10),
             last_improve_size=hist.last_gain, **gp_summary(model), sd_mean=float(sd.mean()), sd_max=float(sd.max()),
             dist_last=d_last, n_distinct=len(set(proposals)), surprise_last=hist.surprise,
             last_af=hist.last_af, last_improved=hist.last_improved)
    for i, a in enumerate(AFS):
        s[f"{a}_used"] = float(hist.used[i]); s[f"{a}_improved"] = float(hist.improved[i])
    return s


def run_campaign(pool, seed, selector, budget=50, n_init=None, record=True):
    """Run one campaign. `selector(step: Step, pool_y_peek) -> AF index`. Returns dict with regret trace and a step log."""
    n_init = n_init if n_init is not None else 2 * pool.X.shape[1] + 1
    rng = np.random.default_rng(seed)
    obs_idx = list(rng.choice(pool.X.shape[0], n_init, replace=False))   # shared init for a given seed
    y_obs = list(pool.y[obs_idx])
    hist = History()
    log = []
    selector_reset = getattr(selector, "reset", None)
    if selector_reset: selector_reset(seed)
    for t in range(budget):
        ya = np.array(y_obs); ia = np.array(obs_idx)
        model, cand, mu, sd, proposals = propose_all(pool, ia, ya, seed, t)
        state = build_state(pool, ia, ya, t, budget, model, mu, sd, proposals, hist)
        step = Step(t, proposals, mu, sd, cand, state, model, pool, ia, ya)
        cand_feats = None
        if record:      # candidate-level numbers for each AF's proposal, in units of the observed std; does not touch any RNG
            best_y = float(ya.max()); ystd = float(ya.std() + 1e-12); inc = pool.X[ia[int(np.argmax(ya))]]; Xe = pool.X[ia]
            cand_feats = []
            for p_ in proposals:
                k_ = int(np.where(cand == p_)[0][0]); m_, s_ = mu[k_], sd[k_]; z_ = (m_ - best_y) / s_
                cand_feats.append([(m_ - best_y) / ystd, s_ / ystd, z_, float(_norm.cdf(z_)), float(s_ * (_norm.pdf(z_) + z_ * _norm.cdf(z_)) / ystd),
                                   float(np.linalg.norm(pool.X[p_] - inc)), float(np.linalg.norm(Xe - pool.X[p_], axis=1).min()),
                                   float((mu < m_).mean()), float((sd < s_).mean())])
        peek = pool.y if getattr(selector, "needs_peek", False) else None
        a = int(selector(step, peek))
        chosen = proposals[a]
        k = int(np.where(cand == chosen)[0][0])
        yn = float(pool.y[chosen])
        hist.surprise = float((yn - mu[k]) / (sd[k] + 1e-12))
        prev_best = float(ya.max()); improved = yn > prev_best
        hist.used[a] += 1; hist.improved[a] += int(improved); hist.last_af = a; hist.last_improved = int(improved)
        hist.last_gain = float((yn - prev_best) / (ya.std() + 1e-12)) if improved else 0.0
        if improved: hist.improve_steps.append(t)
        obs_idx.append(chosen); y_obs.append(yn)
        hist.best_trace.append(max(prev_best, yn))
        if hasattr(selector, "observe"):
            selector.observe(step, a, yn, improved)
        if record:
            log.append(dict(t=t, af=a, proposals=proposals, proposal_y=[float(pool.y[p]) for p in proposals], cand_feats=cand_feats,
                            y=yn, improved=bool(improved), state=state))
    reg = np.array([pool.regret(b) for b in hist.best_trace])
    return dict(dataset=pool.name, seed=seed, regret=reg, final=float(reg[-1]), auc=float(reg.mean()), log=log)


# ── selectors ───────────────────────────────────────────────────────────────────

class Static:
    def __init__(self, af): self.af = AFS.index(af) if isinstance(af, str) else af; self.name = f"static_{AFS[self.af]}"
    def __call__(self, step, peek): return self.af


class RandomAF:
    name = "random"
    def reset(self, seed): self.rng = np.random.default_rng(seed + 12345)
    def __call__(self, step, peek): return int(self.rng.integers(len(AFS)))


class OracleOneStep:
    """Upper bound only: peeks at the true value of each AF's proposal and takes the best. Not a method."""
    name = "oracle_onestep"
    needs_peek = True
    def __call__(self, step, peek):
        ys = [peek[p] for p in step.proposals]
        return int(np.argmax(ys))


class Replay:
    """Replays a logged AF sequence (to re-run a deterministic campaign with extra logging)."""
    name = "replay"
    def __init__(self, afs): self.afs = list(afs)
    def __call__(self, step, peek): return self.afs[step.t]

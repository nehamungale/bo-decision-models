"""Non-stationary campaign engine (PLAN_nonstationary.md v2). Separate from meta/bo.py so the stationary path is untouched.
A pair (A, B) of tasks on one exact grid; y_pre = normalised score of A, y_post = (1 - a) * sA + a * sB (normalised scores, (y - median)/(max - median)).
Evaluations at budget step t < T_c return y_pre, at t >= T_c return y_post. Regret before the change is measured against y_pre, after it against y_post (its own max and median).
Arms (T0): oblivious, oracle_time, oracle_target, oracle_keep, window, periodic, cusum, shuffled_det, random_trigger, sp_stale_oracle, sp_stale_cusum.
Arms (A0, Amendment 1/1b): refresh, remeasure_keep, random_keep_rm, cusum_keep, random_keep, shuffled_keep, sp_stale_remeasure, sp_passive, forced_keep, forced_spstale.
Restart = discard the active data and spend n_restart steps on a generic random design (oracle_target: the W1 portfolio built for B), like the injection experiments (the steps count against the budget)."""
import numpy as np
import torch
from dataclasses import dataclass
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from meta import bo, sp
from meta.landscapes import load, budget
from meta.w_expand import train_pools

ARMS = ["oblivious", "oracle_time", "oracle_target", "oracle_keep", "window", "periodic", "cusum", "shuffled_det", "random_trigger", "sp_stale_oracle", "sp_stale_cusum",
        "refresh", "remeasure_keep", "random_keep_rm", "cusum_keep", "random_keep", "shuffled_keep", "sp_stale_remeasure", "sp_passive", "forced_keep", "forced_spstale", "sp_stale_full", "refresh_rep"]
RM_ARMS = ("refresh", "remeasure_keep", "random_keep_rm", "sp_stale_remeasure", "refresh_rep")      # arms that re-measure the incumbent every r steps
DEFAULTS = dict(noise=0.0, var_stale=0.1, var_fresh=1e-4, window=15, period_frac=0.25, cusum_k=1.0, cusum_h=8.0, trig_rate=0.02, n_restart=None, kappa=10.0, lam=0.25, c=0.25, r=10, tol=1e-9, rate=0.02)


@dataclass
class Pair:
    la: str; lb: str; a: float
    X: np.ndarray; sA: np.ndarray; ypost: np.ndarray
    post_max: float; post_med: float
    S: np.ndarray          # source scores (sources x configs); A and B excluded
    n0: int; B: int
    port_b: list = None    # optional explicit oracle-target design (tests)

    def reg_pre(self, v): return (self.sA.max() - v) / max(self.sA.max() - np.median(self.sA), 1e-12)
    def reg_post(self, v): return (self.post_max - v) / max(self.post_max - self.post_med, 1e-12)


def norm_scores(p): return (p.y - p.y_med) / max(p.y_max - p.y_med, 1e-12)


def make_pair(la, lb, a):
    pa, pb = load(la), load(lb)
    assert pa.X.shape == pb.X.shape and np.allclose(pa.X, pb.X), "tasks must share the grid"
    sA, sB = norm_scores(pa), norm_scores(pb); yp = (1 - a) * sA + a * sB
    tp = [q for q in train_pools(la) if q.name != lb]          # train_pools already excludes A
    S = bo._score_matrix(tp) if len(tp) >= 8 else None
    return Pair(la, lb, a, pa.X, sA, yp, float(yp.max()), float(np.median(yp)), S, 2 * pa.X.shape[1] + 1, budget(pa))


def target_portfolio(pair):
    """Oracle-target design: greedy W1 portfolio for B (sources = other tasks of the grid, A excluded)."""
    if pair.port_b is not None: return list(pair.port_b)
    tp = [q for q in train_pools(pair.lb) if q.name != pair.la]
    return [int(i) for i in bo.portfolio_greedy(tp, pair.n0)]


def _fixed_noise_scores(X, idx, y, var, cand, seed, t, best_pos):
    """LogEI under a GP with per-point fixed noise variance (z-units); y standardised here. best_pos: indices (into idx/y) allowed to define the incumbent."""
    mu_y, sd_y = float(np.mean(y)), float(np.std(y, ddof=1) + 1e-12); z = (np.asarray(y) - mu_y) / sd_y
    with torch.random.fork_rng():
        torch.manual_seed(seed * 100003 + t)
        m = SingleTaskGP(torch.tensor(X[idx]), torch.tensor(z).unsqueeze(-1), train_Yvar=torch.tensor(np.asarray(var, float)).unsqueeze(-1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    m.eval(); Xc = torch.tensor(X[cand])
    with torch.no_grad():
        post = m.posterior(Xc.unsqueeze(1)); mu = post.mean.reshape(-1).numpy(); sd = post.variance.clamp_min(1e-18).sqrt().reshape(-1).numpy()
        s = LogExpectedImprovement(m, best_f=float(z[best_pos].max()))(Xc.unsqueeze(1)).numpy()
    return s, mu * sd_y + mu_y, sd * sd_y


def stale_row(X, idx, y, seed):
    """GP mean of the stale observations over the whole grid, as an extra source row (already in normalised-score units)."""
    m = bo._gp(X[idx], np.asarray(y, float), seed + 4242, "rbf_ard")
    with torch.no_grad(): return m.posterior(torch.tensor(X).unsqueeze(1)).mean.reshape(-1).numpy()


def run_ns(pair, seed, arm, frac=0.5, stationary=False, **opts):
    assert arm in ARMS
    o = dict(DEFAULTS); o.update(opts); N, n0, B = len(pair.sA), pair.n0, pair.B
    n_r = o["n_restart"] or n0
    Tc = None if stationary else int(round(frac * B))
    rng = np.random.default_rng(seed)
    nz = np.random.default_rng(seed + 777); sig_pre = o["noise"] * max(float(pair.sA.max() - np.median(pair.sA)), 1e-12); sig_post = o["noise"] * max(pair.post_max - pair.post_med, 1e-12)
    def meas(truth_i, post_):      # observation = true value + fresh Gaussian noise (sd = noise x (max - median) of the active objective); no random draw at noise 0
        return float(truth_i) if o["noise"] <= 0 else float(truth_i) + (sig_post if post_ else sig_pre) * float(nz.normal())
    idx = [int(i) for i in rng.choice(N, n0, replace=False)]; val = [meas(pair.sA[i], False) for i in idx]    # initial design: pre-change objective
    stale = [False] * n0; aware = False; queue = []; n_restarts = 0; stale_src = None; kept_once = False
    cus = 0.0; zhist = []; trig = []; trace = []; phase = []; pending = None; n_remeasure = 0; zlog = []; rmlog = []
    det_rng = np.random.default_rng(seed + 9091); per = max(int(round(o["period_frac"] * B)), 1); t_forced = int(round(0.5 * B))
    detector = arm in ("cusum", "shuffled_det", "sp_stale_cusum", "cusum_keep", "shuffled_keep")
    random_fire = arm in ("random_trigger", "random_keep", "random_keep_rm")
    use_stale_src = (arm.startswith("sp_stale") and arm != "sp_stale_full") or arm == "forced_spstale"

    def restart(design):
        nonlocal idx, val, stale, queue, n_restarts, aware, stale_src, cus
        if use_stale_src: stale_src = stale_row(pair.X, idx, val, seed)
        idx, val, stale = [], [], []; aware = use_stale_src or arm in ("oracle_target", "oracle_time", "periodic", "random_trigger") or detector
        queue = list(design); n_restarts += 1; cus = 0.0

    def keep_response(except_k=None):
        """Mark every stored observation stale (variance inflated, permanent, re-evaluable); at most one keep event per campaign."""
        nonlocal stale, aware, kept_once, cus
        if kept_once: return False
        stale = [k != except_k for k in range(len(idx))]; aware = True; kept_once = True; cus = 0.0; return True

    for t in range(B):
        post = Tc is not None and t >= Tc
        truth = pair.ypost if post else pair.sA
        rec_override = None
        # ---- events at the start of step t
        if pending is not None: restart(pending); pending = None      # trigger at the end of the previous step
        if arm in ("oracle_time", "oracle_target", "sp_stale_oracle") and t == Tc:
            if arm == "oracle_target": design = target_portfolio(pair)[:n_r]
            else: design = [int(i) for i in np.random.default_rng(seed + 5151 + n_restarts).choice(N, n_r, replace=False)]
            restart(design)
        elif arm == "oracle_keep" and t == Tc:
            stale = [True] * len(idx); aware = True
        elif arm == "periodic" and t > 0 and t % per == 0:
            restart([int(i) for i in np.random.default_rng(seed + 5151 + n_restarts).choice(N, n_r, replace=False)])
        elif arm == "forced_keep" and t == t_forced: keep_response(); trig.append(t)
        elif arm == "forced_spstale" and t == t_forced:
            restart([int(i) for i in np.random.default_rng(seed + 5151 + n_restarts).choice(N, n_r, replace=False)]); trig.append(t)
        is_rm = arm in RM_ARMS and (t + 1) % o["r"] == 0 and not queue and len(idx) > 0
        # ---- design phase of a restart
        if queue:
            i = queue.pop(0); idx.append(i); val.append(meas(truth[i], post)); stale.append(False)
        elif is_rm:     # re-measure the incumbent: a paired observation of the (possibly changed) objective
            valid = [k for k in range(len(idx)) if not (aware and stale[k])] or list(range(len(idx)))
            ik = max(valid, key=lambda k: val[k]); newv = meas(truth[idx[ik]], post); rmlog.append((t, idx[ik], val[ik], newv, float(pair.sA[idx[ik]]), float(pair.ypost[idx[ik]]))); mism = abs(newv - val[ik]) > o["tol"]; n_remeasure += 1
            if arm == "refresh_rep": idx.append(idx[ik]); val.append(newv); stale.append(False)      # keep the replicate as an extra row
            else: val[ik] = newv; stale[ik] = False
            if mism and not kept_once and arm in ("remeasure_keep", "sp_stale_remeasure"):
                trig.append(t)
                if arm == "remeasure_keep": keep_response(except_k=ik)
                else:
                    kept_once = True
                    pending = [int(i) for i in np.random.default_rng(seed + 5151 + n_restarts).choice(N, n_r, replace=False)]
        else:
            ia = np.array(idx); ya = np.array(val); mu_all = None
            if arm == "window": ia, ya = ia[-o["window"]:], ya[-o["window"]:]
            passive = arm in ("sp_passive", "sp_stale_full") and len(ia) - o["window"] >= 3
            full = passive and arm == "sp_stale_full"      # Amendment 2: no window; every observation stays in the surrogate and the weights; only the history row uses the older ones
            if passive: iw, yw = (ia, ya) if full else (ia[-o["window"]:], ya[-o["window"]:])
            if passive: used = set(int(j) for j in iw)
            elif arm == "window": used = set(int(j) for j in ia)
            else: used = set(int(j) for j, s_ in zip(idx, stale) if not (aware and s_))
            cand = np.array([j for j in range(N) if j not in used])
            if any(stale) and aware:
                fv = o["var_fresh"] if o["noise"] <= 0 else max(o["var_fresh"], ((sig_post if post else sig_pre) / max(float(np.std(ya, ddof=1)), 1e-12)) ** 2)
                var = [(o["var_stale"] + (fv if o["noise"] > 0 else 0.0)) if s_ else fv for s_ in stale]; fpos = [k for k, s_ in enumerate(stale) if not s_] or list(range(len(idx)))
                s, mu, sd = _fixed_noise_scores(pair.X, ia, ya, var, cand, seed, t, np.array(fpos))
            elif stale_src is not None and use_stale_src:
                S = np.vstack([pair.S, stale_src[None, :]]); w, _ = sp.weights(S, ia, ya, o["kappa"], o["lam"])
                mu, sd, _, _ = sp.predict(S, w, pair.X, ia, ya, cand, seed, t, o["c"])
                s = bo._log_ei_gauss(mu, np.maximum(sd, 1e-12), float(ya.max()))
            elif passive:
                row = stale_row(pair.X, ia[:-o["window"]], ya[:-o["window"]], seed); S = np.vstack([pair.S, row[None, :]]); w, _ = sp.weights(S, iw, yw, o["kappa"], o["lam"])
                mu_all, sd_all, _, _ = sp.predict(S, w, pair.X, iw, yw, np.arange(N), seed, t, o["c"]); mu, sd = mu_all[cand], sd_all[cand]
                s = bo._log_ei_gauss(mu, np.maximum(sd, 1e-12), float(yw.max()))
                ev = np.array(idx); rec_override = int(ev[int(np.argmax(mu_all[ev]))])
            else:
                s, mu, sd = bo.scores(pair, ia, ya, cand, seed, t, "rbf_ard")      # pair.X is all scores() reads
            k = int(np.argmax(s)); chosen = int(cand[k]); yn = meas(truth[chosen], post)
            if detector:
                z = (yn - mu[k]) / max(sd[k], 1e-12)
                prior = pair.X[idx] if len(idx) else None; zlog.append((t, float(z), float(np.min(np.linalg.norm(prior - pair.X[chosen], axis=1))) if prior is not None else float('nan'), float(sd[k]), float(chosen in set(idx))))
                if arm in ("shuffled_det", "shuffled_keep"): zhist.append(z); z = float(det_rng.choice(zhist[:-1])) if len(zhist) > 1 else 0.0
                cus = max(0.0, cus + abs(z) - o["cusum_k"])
            if chosen in idx:      # a stored observation of the same config is superseded by the fresh one (the paired signal)
                keep = [k_ for k_ in range(len(idx)) if idx[k_] != chosen]
                idx = [idx[k_] for k_ in keep]; val = [val[k_] for k_ in keep]; stale = [stale[k_] for k_ in keep]
            idx.append(chosen); val.append(yn); stale.append(False)
            fire = (detector and cus > o["cusum_h"]) or (random_fire and det_rng.random() < o["rate" if arm != "random_trigger" else "trig_rate"])
            if fire and t + 1 < B:
                if arm in ("cusum_keep", "shuffled_keep", "random_keep", "random_keep_rm"):
                    if keep_response(): trig.append(t)
                else:
                    trig.append(t)
                    pending = [int(i) for i in np.random.default_rng(seed + 5151 + n_restarts).choice(N, n_r, replace=False)]
        # ---- recommendation: best stored value among data the controller treats as valid (sp_passive: best fused posterior mean over evaluated configs)
        if rec_override is not None: rec = rec_override
        else:
            valid = [k for k in range(len(idx)) if not (aware and stale[k])] or list(range(len(idx)))
            rec = idx[max(valid, key=lambda k: val[k])]
        trace.append(pair.reg_post(float(truth[rec])) if post else pair.reg_pre(float(truth[rec]))); phase.append(post)
    tr = np.array(trace); Tcc = Tc if Tc is not None else B
    first = next((x for x in trig if Tc is not None and x >= Tc), None)
    return dict(regret=tr, pre_auc=float(tr[:Tcc].mean()) if Tcc > 0 else float("nan"), post_auc=float(tr[Tcc:].mean()) if Tcc < B else float("nan"), auc=float(tr.mean()),
                Tc=Tc, triggers=trig, n_restarts=n_restarts, n_remeasure=n_remeasure, delay=(first - Tc) if first is not None else None,
                delay_regret=float(tr[Tc:first + 1].sum()) if first is not None else None, zlog=zlog, rmlog=rmlog, obs_idx=[int(i) for i in idx], val=[float(v) for v in val], stale=list(stale))

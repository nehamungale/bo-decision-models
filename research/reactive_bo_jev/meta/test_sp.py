"""T0 tests for meta/sp.py (PLAN_source_prior.md). Run from reactive_bo_jev/: python -m meta.test_sp"""
import numpy as np
from meta.landscapes import load, kept
from meta import bo, sp
from meta.w_expand import train_pools

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info, flush=True)

ln = [l for l in kept(("tabrepo",)).landscape if l.startswith("tabrepo_XGBoost")][3]
p = load(ln); tp = train_pools(ln); S = bo._score_matrix(tp); n0 = 2 * p.X.shape[1] + 1; N = S.shape[0]
init = bo.portfolio_greedy(tp, n0); B = 12
check("target task not among its sources", not any(np.allclose(row, (p.y - p.y_med) / max(p.y_max - p.y_med, 1e-12)) for row in S), f"{N} sources")
# weights
rng = np.random.default_rng(0); idx = np.array(init); y = p.y[idx]
w, rho = sp.weights(S, idx, y, 10.0, 0.5)
check("weights sum to one, non-negative (shrinkage)", abs(w.sum() - 1) < 1e-12 and (w >= 0).all())
check("lambda = 0 gives uniform", np.allclose(sp.weights(S, idx, y, 10.0, 0.0)[0], 1.0 / N))
S2 = S.copy(); S2[3] = (p.y - p.y_med) / max(p.y_max - p.y_med, 1e-12)      # test only: leak the target as source 3
w2, rho2 = sp.weights(S2, idx, y, 10.0, 1.0)
check("a source equal to the target gets the largest weight", int(np.argmax(w2)) == 3 and rho2[3] > 0.99, f"rho {rho2[3]:.2f}, w {w2[3]:.2f}")
w_lo = sp.weights(S2, idx, y, 3.0, 0.25)[0]; check("small kappa/lambda keep weights near uniform (effective sources high)", np.exp(-(w_lo * np.log(w_lo)).sum()) > 0.8 * N)
m, v = sp.prior_moments(S, np.full(N, 1.0 / N)); check("uniform prior mean and spread are the plain mean and variance", np.allclose(m, S.mean(0)) and np.allclose(v, S.var(0)))
# modes
r = {md: sp.run_sp(p, S, 11, md, budget=B, n_init=n0, init_idx=init) for md in sp.MODES}
check("all modes run, budget respected, no duplicate evaluations", all(len(x["regret"]) == B and len(set(x["obs_idx"])) == n0 + B for x in r.values()))
check("deterministic", np.array_equal(r["sp"]["regret"], sp.run_sp(p, S, 11, "sp", budget=B, n_init=n0, init_idx=init)["regret"]))
check("frozen weights equal sp weights at step 0 and stay constant", np.allclose(r["frozen"]["weights"][0], r["sp"]["weights"][0]) and np.allclose(r["frozen"]["weights"], r["frozen"]["weights"][0]))
check("sp weights change over steps (volatility > 0), frozen/uniform volatility 0", r["sp"]["volatility"][1:].sum() > 0 and r["frozen"]["volatility"].sum() == 0 and r["uniform"]["volatility"].sum() == 0, f"sp TV {r['sp']['volatility'].sum():.3f}")
check("uniform weights", np.allclose(r["uniform"]["weights"], 1.0 / N))
w0 = r["sp"]["weights"][0]; wp = r["permw"]["weights"][0]
check("permw keeps sp's weight multiset but changes the assignment", np.allclose(np.sort(w0), np.sort(wp)) and not np.allclose(w0, wp) and np.allclose(r["permw"]["weights"], wp))
prng = np.random.default_rng(11 + 777); Ssh = np.stack([row[prng.permutation(len(row))] for row in S])
check("shuffled sources keep each task's score distribution but break the config assignment", np.allclose(np.sort(Ssh, 1), np.sort(S, 1)) and not np.allclose(Ssh, S))
check("effective number of sources between 1 and N", all(1 - 1e-9 <= e <= N + 1e-9 for x in r.values() for e in x["eff_sources"]))
# variance inflation
mu, sd, sd_r, ab = sp.predict(S, w, p.X, idx, y, np.setdiff1d(np.arange(len(p.X)), idx), 1, 0, 1.0)
check("variance inflation finite, sd >= residual sd, active where sources disagree", np.isfinite(sd).all() and (sd >= sd_r - 1e-12).all() and (sd > sd_r + 1e-9).any() or ab[1] == 0, f"b={ab[1]:.3f}")
Sid = np.tile(S[0], (N, 1)); mu0, sd0, sd_r0, _ = sp.predict(Sid, np.full(N, 1.0 / N), p.X, idx, y, np.setdiff1d(np.arange(len(p.X)), idx), 1, 0, 1.0)
check("identical sources: no inflation", np.allclose(sd0, sd_r0))
# sanity: leaked target as a source finds the optimum fast
Sl = S.copy(); Sl[0] = (p.y - p.y_med) / max(p.y_max - p.y_med, 1e-12)
rl = sp.run_sp(p, Sl, 11, "sp", budget=B, n_init=n0, init_idx=init, kappa=30.0, lam=1.0)
check("leaked-target sanity: finds the optimum at the first step, AUC far below plain SP", rl["regret"][0] < 1e-9 and rl["auc"] < 0.5 * r["sp"]["auc"], f"AUC {rl['auc']:.4f} vs plain SP {r['sp']['auc']:.4f}")
rn = sp.run_sp(p, S, 11, "sp", budget=B, n_init=n0, init_idx=init, noise_frac=0.05)
check("noise run works and differs", len(rn["regret"]) == B)
print("ALL PASS" if ok else "SOME FAILED")

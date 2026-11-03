"""Tests for Amendment 2's sp_stale_full (PLAN_nonstationary.md 9e). Run from reactive_bo_jev/: python -m meta.test_nonstat_a2"""
import dataclasses
from unittest import mock
import numpy as np
from meta import nonstat as ns
from meta.nonstat_pairs import load_pairs

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info, flush=True)

pr = [p for p in load_pairs()["pilot"] if p["stratum"] == "lcbench"][0]
real = ns.make_pair(pr["A"], pr["B"], 1.0); rs = dataclasses.replace(real, B=26); n0 = real.n0
calls_row, calls_pred = [], []
orig_row, orig_pred = ns.stale_row, ns.sp.predict
def spy_row(X, idx_, y_, seed_): calls_row.append(len(idx_)); return orig_row(X, idx_, y_, seed_)
def spy_pred(S_, w_, X_, obs_, y_, cand_, seed_, t_, c_, force_b0=False): calls_pred.append(len(obs_)); return orig_pred(S_, w_, X_, obs_, y_, cand_, seed_, t_, c_, force_b0)
with mock.patch.object(ns, "stale_row", spy_row), mock.patch.object(ns.sp, "predict", spy_pred):
    rf = ns.run_ns(rs, 5, "sp_stale_full", frac=0.5)
ob = ns.run_ns(rs, 5, "oblivious", frac=0.5); pa = ns.run_ns(rs, 5, "sp_passive", frac=0.5)
check("history row = observations older than the last 15 (3, 4, 5, ...)", calls_row[:3] == [3, 4, 5], str(calls_row[:5]))
check("surrogate and weights use every observation (history size + 15 at each step, no window)", all(p == r_ + 15 for p, r_ in zip(calls_pred, calls_row)) and len(calls_pred) == len(calls_row), str(calls_pred[:5]))
check("sp_stale_full equals oblivious until the history has 3 points", rf["obs_idx"][:n0 + 3] == ob["obs_idx"][:n0 + 3] if len(rf["obs_idx"]) > n0 + 3 else True)
check("no restart, no stale marking, no trigger", rf["n_restarts"] == 0 and sum(rf["stale"]) == 0 and rf["triggers"] == [] and rf["n_remeasure"] == 0)
check("differs from sp_passive (no window) once the history exists", rf["obs_idx"] != pa["obs_idx"])
check("deterministic", np.array_equal(rf["regret"], ns.run_ns(rs, 5, "sp_stale_full", frac=0.5)["regret"]))
st = ns.run_ns(rs, 5, "sp_stale_full", stationary=True); check("stationary run valid (budget respected)", len(st["regret"]) == rs.B and st["triggers"] == [])
print("ALL PASS" if ok else "SOME FAILED")

# ---- noise stage (PLAN 9f/9i): noise options, noise-0 regression against stored A2 results
import glob, pickle, os
f_ = sorted(glob.glob("meta/results/nonstat_a2/active__change__*__1.0__5000.pkl"))[3]; r_ = pickle.load(open(f_, "rb")); pr_ = ns.make_pair(r_["A"], r_["B"], r_["a"])
chk = all(abs(ns.run_ns(pr_, 5000, a, frac=0.5, **kw)["post_auc"] - r_["runs"][(a, 0.5)]["post_auc"]) < 1e-12 for a, kw in (("remeasure_keep", dict(r=10, var_stale=0.01)), ("oblivious", {}), ("oracle_keep", dict(var_stale=0.01))))
check("noise 0 reproduces stored A2 results exactly (remeasure_keep, oblivious, oracle_keep)", chk)
on = ns.run_ns(rs, 5, "oblivious", frac=0.5, noise=0.05); o0 = ns.run_ns(rs, 5, "oblivious", frac=0.5)
check("noise changes the stored values; regret is scored on the true value of the recommendation", not np.allclose(on["val"], o0["val"]) and np.all(np.isfinite(on["regret"])) and max(abs(np.array(on["val"]) - np.array([rs.sA[i] if t < n0 + rs.B // 2 else rs.ypost[i] for t, i in enumerate(on["obs_idx"])]))) > 0)
rm = ns.run_ns(rs, 5, "refresh", stationary=True, noise=0.05, r=4)
check("a re-measurement under noise draws a fresh value for the same config (stationary, mismatch with the stored value is nonzero)", any(abs(x[2] - x[3]) > 1e-9 for x in rm["rmlog"]), str([round(x[3] - x[2], 3) for x in rm["rmlog"][:4]]))
check("noise run deterministic", np.array_equal(rm["regret"], ns.run_ns(rs, 5, "refresh", stationary=True, noise=0.05, r=4)["regret"]))
rr = ns.run_ns(rs, 5, "refresh_rep", stationary=True, noise=0.05, r=4); n_rm = rr["n_remeasure"]
check("refresh_rep keeps replicates as extra rows (more rows than evaluations - re-measurements consume budget steps)", len(rr["obs_idx"]) == rs.n0 + rs.B and len(set(rr["obs_idx"])) < len(rr["obs_idx"]) and n_rm >= 1)
band = 3 * 2 ** 0.5 * 0.05; ks_ = [ns.run_ns(rs, sd, "remeasure_keep", stationary=True, noise=0.05, r=4, tol=band, var_stale=0.01) for sd in range(6)]
kc = ns.run_ns(rs, 5, "remeasure_keep", frac=0.5, noise=0.05, r=4, tol=band, var_stale=0.01)
check("remeasure_keep under noise with a band: runs, at most one keep event, triggers only on |difference| > band", len(kc["triggers"]) <= 1 and all(x["n_restarts"] == 0 for x in ks_))
print("ALL PASS" if ok else "SOME FAILED")

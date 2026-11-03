"""T0 tests for meta/nonstat.py and meta/nonstat_pairs.py (PLAN_nonstationary.md v2). Run from reactive_bo_jev/: python -m meta.test_nonstat"""
import dataclasses, time
import numpy as np
from meta.landscapes import load
from meta import bo, nonstat as ns
from meta.nonstat_pairs import load_pairs
from meta.w_expand import train_pools
from acq.engine import Pool

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info, flush=True)

# ---- synthetic mini pair (fast; pool small so pre-change configs get re-chosen)
def mini(seed=0, N=40, a=1.0, B=16):
    r = np.random.default_rng(seed); X = r.random((N, 3)); sA = r.random(N); sA = (sA - np.median(sA)) / (sA.max() - np.median(sA)); sB = np.cos(5 * X[:, 0]) + 0.3 * r.random(N)
    sB = (sB - np.median(sB)) / (sB.max() - np.median(sB)); yp = (1 - a) * sA + a * sB
    return ns.Pair("A", "B", a, X, sA, yp, float(yp.max()), float(np.median(yp)), r.random((9, N)), 7, B, port_b=[int(i) for i in r.choice(N, 7, replace=False)])

# ---- real LCBench pair
P = load_pairs(); pr = [p for p in P["pilot"] if p["stratum"] == "lcbench"][0]
real = ns.make_pair(pr["A"], pr["B"], 1.0); real_short = dataclasses.replace(real, B=16)
n0 = real.n0

# 1. stationary oblivious equals the stationary engine exactly
pool = Pool("A", real.X, real.sA); t0 = time.time()
ref = bo.run(pool, 5, budget=16, n_init=n0); mine = ns.run_ns(real_short, 5, "oblivious", stationary=True)
check("oblivious without change == bo.run (indices and regret)", ref["obs_idx"] == mine["obs_idx"] and np.allclose(ref["regret"], mine["regret"], atol=1e-12), f"{time.time()-t0:.0f}s")
# 2. change applied at T_c exactly
r = ns.run_ns(real_short, 5, "oblivious", frac=0.5); Tc = r["Tc"]; ob = r["obs_idx"]; v = np.array(r["val"])
pre = np.arange(len(ob)) < n0 + Tc
check("values before T_c are y_pre, from T_c on are y_post", np.allclose(v[pre], real.sA[np.array(ob)[pre]]) and np.allclose(v[~pre], real.ypost[np.array(ob)[~pre]]), f"T_c={Tc}")
check("oblivious keeps every observation and never re-evaluates", len(ob) == n0 + 16 and len(set(ob)) == len(ob))
# 3. before T_c the arms that act only at T_c follow the oblivious trajectory
same = {arm: np.allclose(ns.run_ns(real_short, 5, arm, frac=0.5)["regret"][:Tc], r["regret"][:Tc], atol=1e-12) for arm in ("oracle_time", "oracle_target", "oracle_keep", "sp_stale_oracle")}
check("pre-change trace identical for oracle arms", all(same.values()), str(same))
# 4. regret on the blend objective, not on B
m = mini(1, a=0.5); best = int(np.argmax(m.ypost))
check("blend regret is zero at the blend optimum and uses the blend's own max/median", abs(m.reg_post(float(m.ypost[best]))) < 1e-12 and abs(m.post_max - m.ypost.max()) < 1e-15 and m.post_med == float(np.median(m.ypost)))
cnt = sum(int(np.argmax(mini(s, a=0.5).ypost)) != int(np.argmax(mini(s, a=1.0).ypost)) for s in range(30))
check("across 30 draws the blend optimum is often not the B optimum", cnt >= 5, f"{cnt}/30")
# 5. oracle_time discards exactly the pre-change data and uses a generic design
ro = ns.run_ns(real_short, 5, "oracle_time", frac=0.5); gen = [int(i) for i in np.random.default_rng(5 + 5151 + 0).choice(len(real.sA), n0, replace=False)]
check("oracle_time: one restart, active data are only post-change, design is the seeded generic one",
      ro["n_restarts"] == 1 and len(ro["obs_idx"]) == 16 - Tc and ro["obs_idx"][:n0] == gen[:len(ro["obs_idx"])], f"active {len(ro['obs_idx'])}")
check("oracle_time uses no source/target information (design independent of the pair)", ns.run_ns(dataclasses.replace(real_short, S=None, sA=real.sA), 5, "oracle_time", frac=0.5)["obs_idx"][:n0] == ro["obs_idx"][:n0])
rt = ns.run_ns(real_short, 5, "oracle_target", frac=0.5); tp = ns.target_portfolio(real)
check("oracle_target restarts from B's portfolio (sources exclude A and B)", rt["obs_idx"][:n0] == tp[:len(rt["obs_idx"])] and not any(q.name in (real.la, real.lb) for q in [q for q in train_pools(real.lb) if q.name != real.la]))
# 6. re-evaluation after a restart returns y_post (small pool so overlap with pre-change configs is certain)
mm = mini(2); hits = 0; check_ = True
for sd in range(6):
    rr = ns.run_ns(mm, sd, "oracle_time", frac=0.5); Tm = rr["Tc"]
    first = np.random.default_rng(sd).choice(40, 7, replace=False)
    ok_vals = all(abs(v - mm.ypost[i]) < 1e-12 for i, v in zip(rr["obs_idx"], rr["val"]))
    hits += len(set(first.tolist()) & set(rr["obs_idx"])); check_ &= ok_vals
check("after restart every stored value is y_post, including configs that were evaluated before the change", check_ and hits > 0, f"{hits} re-evaluated configs")
rk = [ns.run_ns(mm, sd, "oracle_keep", frac=0.5) for sd in range(6)]
check("oracle_keep: stale observation of a config is replaced by the fresh one (no duplicate configs), fresh values are y_post",
      all(len(set(x["obs_idx"])) == len(x["obs_idx"]) for x in rk) and all(abs(v - mm.ypost[i]) < 1e-12 for x in rk for i, v, s_ in zip(x["obs_idx"], x["val"], x["stale"]) if not s_), f"{sum(sum(x['stale']) for x in rk)} stale kept")
# 7. determinism and budget for every arm (mini pair, fast)
det = {}; 
for arm in ns.ARMS:
    a1 = ns.run_ns(mm, 3, arm, frac=0.5, cusum_h=1.0, trig_rate=0.2); a2 = ns.run_ns(mm, 3, arm, frac=0.5, cusum_h=1.0, trig_rate=0.2)
    det[arm] = np.array_equal(a1["regret"], a2["regret"]) and len(a1["regret"]) == mm.B and a1["triggers"] == a2["triggers"]
check("every arm deterministic and respects the budget", all(det.values()), str([k for k, v in det.items() if not v]))
# 8. stationary controls: oracle arms never act, equal oblivious; periodic does restart
so = ns.run_ns(mm, 3, "oblivious", stationary=True); 
check("oracle_time/oracle_keep without a change equal oblivious", all(np.array_equal(ns.run_ns(mm, 3, a, stationary=True)["regret"], so["regret"]) for a in ("oracle_time", "oracle_keep", "oracle_target", "sp_stale_oracle")))
check("periodic restarts on stationary runs (the false-alarm cost is measurable)", ns.run_ns(mm, 3, "periodic", stationary=True)["n_restarts"] >= 2)
# 9. detector controls
never = ns.run_ns(mm, 3, "cusum", frac=0.5, cusum_h=1e9); always = ns.run_ns(mm, 3, "cusum", frac=0.5, cusum_h=0.0, cusum_k=0.0)
check("cusum with huge threshold never fires and equals oblivious; with zero threshold fires", never["n_restarts"] == 0 and np.array_equal(never["regret"], ns.run_ns(mm, 3, "oblivious", frac=0.5)["regret"]) and always["n_restarts"] >= 1)
rf = ns.run_ns(mm, 3, "cusum", frac=0.5, cusum_h=0.0, cusum_k=0.0)
check("delay bookkeeping: first trigger at or after T_c gives delay >= 0 and a delay-window regret", rf["delay"] is None or (rf["delay"] >= 0 and rf["delay_regret"] is not None), f"delay {rf['delay']}")
rs = ns.run_ns(mm, 3, "shuffled_det", frac=0.5, cusum_h=0.5); check("shuffled-state detector runs causally (z drawn only from the past)", len(rs["regret"]) == mm.B)
# 10. SP-stale
S = real.S; sAn, sBn = ns.norm_scores(load(real.la)), ns.norm_scores(load(real.lb))
check("SP-stale sources exclude A and B and are fewer than or equal to A's usual sources", not any(np.allclose(row, sAn) or np.allclose(row, sBn) for row in S) and S.shape[0] == len([q for q in train_pools(real.la) if q.name != real.lb]), f"{S.shape[0]} sources")
rng = np.random.default_rng(0); ii = rng.choice(len(real.sA), 15, replace=False); row = ns.stale_row(real.X, ii, real.sA[ii], 0)
from scipy.stats import spearmanr
check("stale row (GP on stale data) correlates with the stale objective over the grid", spearmanr(row, real.sA)[0] > 0.3, f"rho {spearmanr(row, real.sA)[0]:.2f}")
rsp = ns.run_ns(real_short, 5, "sp_stale_oracle", frac=0.5)
check("SP-stale oracle runs on a real pair; active data post-change only", rsp["n_restarts"] == 1 and len(rsp["obs_idx"]) == 16 - Tc)
rsc = ns.run_ns(real_short, 5, "sp_stale_cusum", frac=0.5, cusum_h=1.0); check("SP-stale cusum runs", len(rsc["regret"]) == 16, f"restarts {rsc['n_restarts']}")
# 11. window arm keeps only the last w points; random-trigger rate
rw = ns.run_ns(real_short, 5, "window", frac=0.5, window=10); check("window arm runs under its budget", len(rw["regret"]) == 16)
# 12. pair lists
allp = P["pilot"] + P["tuning"] + P["test"]; keyset = [frozenset((p["A"], p["B"])) for p in allp]
check("pairs unique across pilot/tuning/test; counts 20/40/100", len(set(keyset)) == len(keyset) and (len(P["pilot"]), len(P["tuning"]), len(P["test"])) == (20, 40, 100))
check("A != B, every cell balanced", all(p["A"] != p["B"] for p in allp) and all(sum(1 for p in P[s] if p["stratum"] == st and p["relatedness"] == rl) == P["quota"][s] for s in ("pilot", "tuning", "test") for st in ("lcbench", "tabrepo") for rl in ("related", "unrelated")))
rel = [p["spearman"] for p in allp if p["relatedness"] == "related"]; unr = [p["spearman"] for p in allp if p["relatedness"] == "unrelated"]
check("related pairs have higher Spearman than unrelated pairs on average within lcbench", np.mean([p["spearman"] for p in allp if p["stratum"] == "lcbench" and p["relatedness"] == "related"]) > np.mean([p["spearman"] for p in allp if p["stratum"] == "lcbench" and p["relatedness"] == "unrelated"]), f"overall mean {np.mean(rel):.2f} vs {np.mean(unr):.2f}")
fam = {g: sum(1 for p in P["test"] if p["group"] == g) for g in sorted({p["group"] for p in P["test"]})}
check("test set covers all seven units, TabRepo families roughly balanced (spread <= 4)", len(fam) == 7 and max(v for g, v in fam.items() if g != "lcbench") - min(v for g, v in fam.items() if g != "lcbench") <= 4, str(fam))
ok_src = True
for p in allp:
    n_src = len([q for q in train_pools(p["A"]) if q.name != p["B"]]); ok_src &= n_src >= 8
check("every A has at least 8 sources after excluding B", ok_src)


# =================== A0 (Amendment 1 / 1b) ===================
from unittest import mock
import torch
# 13. re-measurement mechanics
rf1 = ns.run_ns(real_short, 5, "refresh", frac=0.5, r=1000); ob1 = ns.run_ns(real_short, 5, "oblivious", frac=0.5)
check("refresh with r larger than the budget equals oblivious", rf1["obs_idx"] == ob1["obs_idx"] and np.array_equal(rf1["regret"], ob1["regret"]) and rf1["n_remeasure"] == 0)
rs4 = ns.run_ns(mm, 3, "refresh", stationary=True, r=4)
check("each re-measurement consumes exactly one budget step", rs4["n_remeasure"] == mm.B // 4 and len(rs4["obs_idx"]) == mm.n0 + mm.B - rs4["n_remeasure"], f"{rs4['n_remeasure']} re-measurements of {mm.B} steps")
zero = True; same_as_refresh = True
for sd in range(6):
    for pr_ in (mm, real_short):
        a_ = ns.run_ns(pr_, sd, "remeasure_keep", stationary=True, r=3); b_ = ns.run_ns(pr_, sd, "refresh", stationary=True, r=3)
        zero &= a_["triggers"] == []; same_as_refresh &= np.array_equal(a_["regret"], b_["regret"]) and a_["obs_idx"] == b_["obs_idx"]
check("zero false alarms at noise 0 on stationary runs (tested, 12 campaigns); remeasure_keep == refresh there", zero and same_as_refresh)
# 14. detection after a change; no trigger before it; paired with refresh until the first trigger
fired = []; paired = True; sched = True
for sd in range(8):
    rk_ = ns.run_ns(mm, sd, "remeasure_keep", frac=0.5, r=3); rr_ = ns.run_ns(mm, sd, "refresh", frac=0.5, r=3); rkr = ns.run_ns(mm, sd, "random_keep_rm", frac=0.5, r=3, rate=0.3)
    if rk_["triggers"]:
        fired.append(rk_["triggers"][0]); f0 = rk_["triggers"][0]
        paired &= np.array_equal(rk_["regret"][:f0], rr_["regret"][:f0])
    if rkr["triggers"]: paired &= np.array_equal(rkr["regret"][:rkr["triggers"][0]], rr_["regret"][:rkr["triggers"][0]])
    sched &= bool(rk_["n_remeasure"] == rr_["n_remeasure"] == rkr["n_remeasure"])
check("remeasure_keep fires only on a re-measurement step at or after T_c", len(fired) >= 5 and all((f + 1) % 3 == 0 and f >= 8 for f in fired), str(fired))
check("re-measuring arms share r and RNG streams: identical to refresh until their first trigger", paired and sched)
# 15. keep response: permanent marking, one event only, variance inflation
rkd = ns.run_ns(real_short, 5, "cusum_keep", frac=0.5, cusum_h=0.0, cusum_k=0.0)
check("keep event fires at most once per campaign even when the detector fires every step", len(rkd["triggers"]) <= 1 and rkd["n_restarts"] == 0, str(rkd["triggers"]))
check("keep response leaves stale (inflated-variance) observations in the data, no restart", sum(rkd["stale"]) > 0 or not rkd["triggers"], f"{sum(rkd['stale'])} stale of {len(rkd['stale'])}")
Xq = np.random.default_rng(0).random((12, 2)); idq = np.array([0, 1, 2, 3, 4]); yq = np.array([0.2, 0.9, 0.4, 0.1, 0.6]); cq = np.arange(5, 12)
_, _, sd_a = ns._fixed_noise_scores(Xq, idq, yq, [1e-4] * 5, np.array([4]), 1, 0, np.arange(5)); _, _, sd_b = ns._fixed_noise_scores(Xq, idq, yq, [1e-4, 1e-4, 1e-4, 1e-4, 0.1], np.array([4]), 1, 0, np.arange(4))
check("inflating one observation's variance (fixed-noise GP) raises the predictive sd at that config", np.isfinite(sd_b).all() and sd_b[0] > sd_a[0], f"{sd_a[0]:.4f} -> {sd_b[0]:.4f}")
# 16. re-measure detector under SP-stale and random controls
rsr = [ns.run_ns(mm, sd, "sp_stale_remeasure", frac=0.5, r=3) for sd in range(8)]
check("sp_stale_remeasure restarts once with a stale source after detection, never on stationary runs", all(x["n_restarts"] <= 1 for x in rsr) and any(x["n_restarts"] == 1 for x in rsr) and all(ns.run_ns(mm, sd, "sp_stale_remeasure", stationary=True, r=3)["n_restarts"] == 0 for sd in range(4)))
rr1 = ns.run_ns(mm, 3, "random_keep_rm", frac=0.5, r=3, rate=1.0); check("random_keep_rm with rate 1 fires once, on a non-re-measurement step", len(rr1["triggers"]) == 1 and (rr1["triggers"][0] + 1) % 3 != 0)
# 17. forced events (cost controls)
fk = ns.run_ns(mm, 3, "forced_keep", stationary=True); fs = ns.run_ns(mm, 3, "forced_spstale", stationary=True)
check("forced events fire at 50 % of the budget on stationary runs", fk["triggers"] == [round(0.5 * mm.B)] and fs["n_restarts"] == 1 and fs["triggers"] == [round(0.5 * mm.B)])
# 18. sp_passive: history row from older observations only; surrogate sees the last 15; equals the plain engine until the history has 3 points
calls_row, calls_pred = [], []
orig_row, orig_pred = ns.stale_row, ns.sp.predict
def spy_row(X, idx_, y_, seed_): calls_row.append(len(idx_)); r_ = orig_row(X, idx_, y_, seed_); calls_row.append(bool(np.isfinite(r_).all() and r_.shape == (len(X),))); return r_
def spy_pred(S_, w_, X_, obs_, y_, cand_, seed_, t_, c_, force_b0=False): calls_pred.append(len(obs_)); return orig_pred(S_, w_, X_, obs_, y_, cand_, seed_, t_, c_, force_b0)
with mock.patch.object(ns, "stale_row", spy_row), mock.patch.object(ns.sp, "predict", spy_pred):
    rp = ns.run_ns(real_short, 5, "sp_passive", frac=0.5)
lens = calls_row[0::2]; fin = calls_row[1::2]
check("sp_passive: history row built from observations older than the last 15 (3, 4, 5, ...), finite on the full grid", lens[:3] == [3, 4, 5] and all(fin), f"{lens[:5]}")
check("sp_passive: the surrogate uses exactly the last 15 observations", set(calls_pred) == {15} and len(calls_pred) == len(lens), f"{set(calls_pred)}")
check("sp_passive equals the plain engine until the history has 3 points", rp["obs_idx"][:n0 + 3] == ob1["obs_idx"][:n0 + 3])
check("sp_passive never restarts or marks stale", rp["n_restarts"] == 0 and sum(rp["stale"]) == 0)
check("history row == GP posterior mean of the old observations (same function as the stale row)", np.allclose(ns.stale_row(real.X, ii, real.sA[ii], 0), orig_row(real.X, ii, real.sA[ii], 0)))
# 19. determinism and budget for the new arms on the real pair (short)
newarms = [a for a in ns.ARMS if a in ("refresh", "remeasure_keep", "random_keep_rm", "cusum_keep", "random_keep", "shuffled_keep", "sp_stale_remeasure", "sp_passive", "forced_keep", "forced_spstale")]
dn = {a: np.array_equal(ns.run_ns(real_short, 7, a, frac=0.25, r=4, cusum_h=2.0, rate=0.15)["regret"], ns.run_ns(real_short, 7, a, frac=0.25, r=4, cusum_h=2.0, rate=0.15)["regret"]) for a in newarms}
check("new arms deterministic on a real pair", all(dn.values()), str([k for k, v in dn.items() if not v]))
print("ALL PASS" if ok else "SOME FAILED")

"""T0 tests for injection and hedge start in meta/bo.py (PLAN_fallback_detection.md). Run from reactive_bo_jev/: python -m meta.test_inject"""
import numpy as np
from meta.landscapes import load
from meta import bo

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info, flush=True)

for ln in ["lcbench/vehicle", "tabrepo_XGBoost/adult"]:
    p = load(ln); n0 = 2 * p.X.shape[1] + 1; B = 16; k = 5; m = n0
    if k + m > B: m = B - k
    init = bo.farthest_point_start(p.X, n0, 1)
    base = bo.run(p, 7, budget=B, n_init=n0, init_idx=init)
    for mode in ("random", "fps"):
        r = bo.run(p, 7, budget=B, n_init=n0, init_idx=init, inject=(k, m, mode))
        check(f"{ln} {mode}: default path untouched when inject is None", np.array_equal(base["regret"], bo.run(p, 7, budget=B, n_init=n0, init_idx=init)["regret"]))
        check(f"{ln} {mode}: identical to no-injection through step k", base["obs_idx"][: n0 + k] == r["obs_idx"][: n0 + k] and np.array_equal(base["regret"][:k], r["regret"][:k]))
        check(f"{ln} {mode}: total evaluations and trace length unchanged", len(r["obs_idx"]) == n0 + B and len(r["regret"]) == B, f"{len(r['obs_idx'])} evals, trace {len(r['regret'])}")
        check(f"{ln} {mode}: no point evaluated twice (exact-grid dedup)", len(set(r["obs_idx"])) == len(r["obs_idx"]))
        check(f"{ln} {mode}: deterministic", r["obs_idx"] == bo.run(p, 7, budget=B, n_init=n0, init_idx=init, inject=(k, m, mode))["obs_idx"])
        check(f"{ln} {mode}: regret never increases (noise-free incumbent)", np.all(np.diff(r["regret"]) <= 1e-12))
    rr = bo.run(p, 7, budget=B, n_init=n0, init_idx=init, inject=(k, m, "random"))["obs_idx"][n0 + k: n0 + k + m]
    rf = bo.run(p, 7, budget=B, n_init=n0, init_idx=init, inject=(k, m, "fps"))["obs_idx"][n0 + k: n0 + k + m]
    check(f"{ln}: random and fps inject different points", rr != rf)
    # fps: first injected point is the max-min-distance unevaluated point
    seen = bo.run(p, 7, budget=B, n_init=n0, init_idx=init, inject=(k, m, "fps"))["obs_idx"][: n0 + k]
    cand = np.setdiff1d(np.arange(len(p.X)), seen)
    d = np.array([np.linalg.norm(p.X[seen] - p.X[c], axis=1).min() for c in cand])
    check(f"{ln}: fps first injected point has maximal min-distance", rf[0] == int(cand[int(np.argmax(d))]))
    # hedge start
    port = bo.farthest_point_start(p.X, n0, 3)
    h = bo.hedge_start(len(p.X), port, n0, 7)
    check(f"{ln}: hedge start has n0 distinct points, first half from the portfolio", len(h) == n0 and len(set(h)) == n0 and h[: (n0 + 1) // 2] == port[: (n0 + 1) // 2])
    check(f"{ln}: hedge start deterministic", h == bo.hedge_start(len(p.X), port, n0, 7))
try:
    bo.run(load("lcbench/vehicle"), 1, budget=5, inject=(3, 5, "fps")); check("injection past the budget is refused", False)
except AssertionError:
    check("injection past the budget is refused", True)
print("ALL PASS" if ok else "SOME FAILED")

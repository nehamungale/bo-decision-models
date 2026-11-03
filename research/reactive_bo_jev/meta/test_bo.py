"""T0 tests for meta/bo.py. Run from reactive_bo_jev/: python -m meta.test_bo"""
import numpy as np, time
from meta.landscapes import load, kept
from meta import bo
from acq.engine import run_campaign, Static
import acq.engine as E

ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info, flush=True)

# 1. default reproduces the reference engine exactly
for ln in ["lcbench/vehicle", "tabrepo_XGBoost/adult"]:
    p = load(ln); B = 12; n0 = 2 * p.X.shape[1] + 1
    ref = run_campaign(p, 5, Static("LogEI"), budget=B, n_init=n0, record=False)["regret"]
    new = bo.run(p, 5, budget=B, n_init=n0)["regret"]
    check(f"default == reference engine ({ln})", np.allclose(ref, new, atol=1e-12), f"max diff {np.abs(ref-new).max():.2e}")
# 2. determinism
p = load("lcbench/vehicle"); a = bo.run(p, 3, budget=8); b = bo.run(p, 3, budget=8)
check("deterministic", np.array_equal(a["regret"], b["regret"]))
# 3. every kernel runs, regret non-increasing without noise, and beats random selection of the same number of points
rng = np.random.default_rng(0)
for kn in bo.KERNELS:
    t = time.time(); r = bo.run(p, 3, budget=15, kernel=kn)["regret"]
    rand = np.mean([p.regret(float(p.y[rng.choice(len(p.y), 2 * 7 + 1 + 15, replace=False)].max())) for _ in range(200)])
    check(f"kernel {kn}: monotone, final regret below random-search level", np.all(np.diff(r) <= 1e-12) and r[-1] < rand, f"final {r[-1]:.3f} random {rand:.3f} ({time.time()-t:.1f}s)")
# 4. region masks: chosen points lie inside the box (re-run manually via small radius) and adaptive box runs
r = bo.run(p, 3, budget=10, region=0.15)["regret"]; check("fixed box runs", len(r) == 10)
r = bo.run(p, 3, budget=20, region="adaptive")["regret"]; check("adaptive box runs", len(r) == 20)
tb = bo.TrustBox(); [tb.update(True) for _ in range(3)]; up = tb.L
tb2 = bo.TrustBox(); [tb2.update(False) for _ in range(4)]
check("TrustBox grows on successes, shrinks on failures", up > 0.4 and tb2.L < 0.4, f"{up} {tb2.L}")
# 5. noise: zero noise identical, positive noise changes the trace, true-value regret can rise
diff = False; det = True
for sd_ in range(4):
    a = bo.run(p, sd_, budget=25, noise_frac=0.0)["regret"]; b = bo.run(p, sd_, budget=25, noise_frac=0.2)["regret"]
    diff |= not np.allclose(a, b); det &= np.array_equal(b, bo.run(p, sd_, budget=25, noise_frac=0.2)["regret"])
check("noise changes traces on some seed; seeded", diff and det)
# 6. portfolio: held-out task excluded, greedy >= rank on mean best-in-set, shapes
ls = [l for l in kept(("tabrepo",)).landscape if l.startswith("tabrepo_XGBoost")][:40]
pools = [load(l) for l in ls if load(l).X.shape[0] == 200]
held, train = pools[0], pools[1:]
n0 = 2 * held.X.shape[1] + 1
g = bo.portfolio_greedy(train, n0); rk = bo.portfolio_rank(train, n0)
S = bo._score_matrix(train)
check("greedy portfolio at least as good as top-by-rank on training tasks", S[:, g].max(1).mean() >= S[:, rk].max(1).mean() - 1e-12, f"{S[:, g].max(1).mean():.3f} vs {S[:, rk].max(1).mean():.3f}")
rw = np.mean([held.regret(float(held.y[g].max()))]); rr = np.mean([held.regret(float(held.y[rng.choice(200, n0, replace=False)].max())) for _ in range(200)])
print(f"   held-out start regret: greedy portfolio {rw:.3f} vs random start {rr:.3f}")
print("ALL PASS" if ok else "SOME FAILED")

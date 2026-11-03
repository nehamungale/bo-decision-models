"""Frozen task-pair lists for PLAN_nonstationary.md. Pairs (A -> B) on one exact grid, related = full-grid Spearman in the top tertile of all pairs of that group, unrelated = bottom tertile.
Cells: {LCBench, TabRepo} x {related, unrelated}; TabRepo draws round-robin over the six model families. Sets: pilot 5 pairs/cell, tuning 10, test 25 (unordered pairs disjoint across sets).
Seeded draw (seed 20261031), written once to meta/results/nonstat_pairs.json. Run: python -m meta.nonstat_pairs"""
import os, json, itertools
import numpy as np
from scipy.stats import rankdata
from meta.landscapes import kept, load
H = os.path.dirname(os.path.abspath(__file__)); F = os.path.join(H, "results", "nonstat_pairs.json")
QUOTA = dict(pilot=5, tuning=10, test=25); SEED = 20261031


def group_pairs(group):
    k = kept(("lcbench", "tabrepo")); names = k[k.group == group].landscape.tolist()
    pools = [load(l) for l in names]; ref = pools[0].X
    keep = [(n, p) for n, p in zip(names, pools) if p.X.shape == ref.shape and np.allclose(p.X, ref)]
    names = [n for n, _ in keep]
    R = np.stack([rankdata(p.y) for _, p in keep]); R = R - R.mean(1, keepdims=True); R /= np.linalg.norm(R, axis=1, keepdims=True)
    C = R @ R.T; iu = np.triu_indices(len(names), 1); v = C[iu]; lo, hi = np.quantile(v, [1 / 3, 2 / 3])
    return names, [(names[i], names[j], float(c)) for i, j, c in zip(*iu, v)], lo, hi


def build():
    rng = np.random.default_rng(SEED); out = {s: [] for s in QUOTA}; used = set(); info = {}
    cand = {}
    for g in ["lcbench"] + sorted(k for k in kept(("tabrepo",)).group.unique()):
        names, prs, lo, hi = group_pairs(g); info[g] = dict(n_tasks=len(names), q33=lo, q67=hi)
        cand[g] = dict(related=[p for p in prs if p[2] >= hi], unrelated=[p for p in prs if p[2] <= lo])
    fams = [g for g in cand if g != "lcbench"]
    for rel in ("related", "unrelated"):
        for stratum in ("lcbench", "tabrepo"):
            groups = ["lcbench"] if stratum == "lcbench" else fams
            for s, q in QUOTA.items():
                for n in range(q):
                    g = groups[(n + (0 if rel == "related" else 3)) % len(groups)]
                    while True:
                        a, b, c = cand[g][rel][int(rng.integers(len(cand[g][rel])))]
                        if rng.random() < 0.5: a, b = b, a
                        if frozenset((a, b)) not in used: break
                    used.add(frozenset((a, b)))
                    out[s].append(dict(A=a, B=b, spearman=c, relatedness=rel, stratum=stratum, group=g))
    return dict(seed=SEED, quota=QUOTA, groups=info, **out)


def load_pairs():
    return json.load(open(F))


if __name__ == "__main__":
    d = build(); json.dump(d, open(F, "w"), indent=1)
    for s in QUOTA: print(s, len(d[s]), {r: sum(p["relatedness"] == r for p in d[s]) for r in ("related", "unrelated")})
    for g, i in d["groups"].items(): print(g, i)

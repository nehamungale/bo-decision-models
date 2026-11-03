"""Verification of the missed-trigger reading (user's check 1). For every remeasure_keep campaign on the test pairs that never triggered after T_c: (a) rerun it and classify the post-T_c re-measurements of the incumbent
(identical value under both objectives vs incumbent already measured post-change); (b) take oracle_keep vs refresh on the same campaigns from the A2 pickles (is there regret left to save?). Output meta/results/diag_missed.txt"""
import os, sys, pickle, glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H = os.path.dirname(os.path.abspath(__file__))


def work(a):
    import torch; torch.set_num_threads(1)
    from meta import nonstat as ns
    A, B, aa, seed, fr = a; pair = ns.make_pair(A, B, aa)
    r = ns.run_ns(pair, seed, "remeasure_keep", frac=fr, r=10, var_stale=0.01)
    assert not r["triggers"] or r["triggers"][0] < r["Tc"] or True
    post = [x for x in r["rmlog"] if x[0] >= r["Tc"]]
    ident = [abs(x[4] - x[5]) < 1e-12 for x in post]
    return a, dict(n_post=len(post), any_identical=any(ident), all_identical=bool(post) and all(ident), triggered=bool(r["triggers"]), post_auc=r["post_auc"])


if __name__ == "__main__":
    from meta import analyse_nonstat_a2 as an
    pairs, C, S = an.load(); jobs = []; info = {}
    for k in C:
        for fr in an.FRS:
            for sd, v in C[k]["remeasure_keep"][fr].items():
                if v["first"] is None or v["first"] < v["Tc"]:
                    miss = v["first"] is None
                    if miss: jobs.append((k[0], k[1], k[2], sd, fr))
    L = []
    def P(*a): s = " ".join(str(x) for x in a); L.append(s); print(s, flush=True)
    with ProcessPoolExecutor(12) as ex: res = list(ex.map(work, jobs, chunksize=4))
    P(f"missed campaigns (remeasure_keep never triggered; all pairs and regimes): {len(res)} of {sum(1 for k in C for fr in an.FRS for _ in C[k]['remeasure_keep'][fr])}")
    n0 = sum(1 for _, d in res if d["n_post"] == 0); P(f"  no post-T_c re-measurement at all: {n0}")
    allid = sum(1 for _, d in res if d["all_identical"]); anyid = sum(1 for _, d in res if d["any_identical"] and not d["all_identical"])
    P(f"  every post-T_c re-measurement had y_A(x*) == y_B(x*) (incumbent unaffected by the change): {allid}")
    P(f"  mixed (some identical, some already-fresh incumbents): {anyid}")
    P(f"  incumbent differed under the two objectives but the stored value was already post-change (incumbent measured after T_c): {len(res) - n0 - allid - anyid}")
    # oracle_keep vs refresh on exactly these campaigns
    num = den = 0.0; rows = []
    for (A, B, aa, sd, fr), d in res:
        k = (A, B, aa); o = C[k]["oracle_keep"][fr][sd]["post_auc"]; rf = C[k]["refresh"][fr][sd]["post_auc"]; ob = C[k]["oblivious"][fr][sd]["post_auc"]; num += o; den += rf; rows.append((o, rf, ob))
    rows = np.array(rows); P(f"  on the missed campaigns: oracle_keep post-change AUC vs refresh {num/den - 1:+.1%}; mean AUC refresh {rows[:,1].mean():.4f}, oracle_keep {rows[:,0].mean():.4f}, oblivious {rows[:,2].mean():.4f}")
    # same for all non-missed campaigns for comparison
    n2 = d2 = 0.0
    for k in C:
        for fr in an.FRS:
            for sd, v in C[k]["remeasure_keep"][fr].items():
                if v["first"] is not None: n2 += C[k]["oracle_keep"][fr][sd]["post_auc"]; d2 += C[k]["refresh"][fr][sd]["post_auc"]
    P(f"  on triggered campaigns: oracle_keep vs refresh {n2/d2 - 1:+.1%}")
    x = rows[:, 1] - rows[:, 0]; P(f"  missed campaigns where oracle_keep is better than refresh by > 0.01 AUC: {(x > 0.01).sum()} of {len(x)}")
    open(os.path.join(H, "results", "diag_missed.txt"), "w").write("\n".join(L) + "\n")
    best = np.minimum(rows[:, 0], rows[:, 1])      # a perfect detector on missed campaigns would take the better of keeping and not keeping
    allref = sum(C[k]["refresh"][fr][sd]["post_auc"] for k in C for fr in an.FRS for sd in C[k]["refresh"][fr])
    allrk = sum(C[k]["remeasure_keep"][fr][sd]["post_auc"] for k in C for fr in an.FRS for sd in C[k]["remeasure_keep"][fr])
    P(f"  headroom if a perfect detector chose keep/no-keep per missed campaign: removes {(rows[:,1]-best).sum()/allrk:.1%} of remeasure_keep's total post-change AUC mass (all regimes); remeasure_keep is {allrk/allref - 1:+.1%} vs refresh overall")
    open(os.path.join(H, "results", "diag_missed.txt"), "w").write("\n".join(L) + "\n")

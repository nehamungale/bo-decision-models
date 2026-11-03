"""Collect typed-model answers over sampled states. Resumable: appends to a jsonl keyed by
(dataset, config, epoch). Run: python -m evaluator.collect OUT.jsonl [n_configs] [datasets...]"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from splits import DEV
from sim.lcbench_env import load_pool
from evaluator.states import build
from evaluator.questions import build_questions
from evaluator.client import ask, flatten

EPOCHS = (5, 10, 25)
SEED = 20261024


def main(out, n_cfg=250, datasets=None):
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line); done.add((r["dataset"], r["config"], r["epoch"]))
    qs = build_questions()
    with open(out, "a") as fh:
        for ds in datasets or DEV:
            pool = load_pool(ds)
            cfgs = np.random.default_rng([SEED, DEV.index(ds)]).permutation(pool.n)[:n_cfg]
            for i in cfgs:
                for e in EPOCHS:
                    if (ds, int(i), e) in done:
                        continue
                    text, feats, y = build(pool, int(i), e, SEED)
                    try:
                        ans, dt = ask(text, qs)
                    except Exception as ex:
                        print("FAIL", ds, i, e, str(ex)[:100], flush=True); continue
                    rec = dict(dataset=ds, config=int(i), epoch=e, y=y, latency=dt, feats=feats,
                               model=flatten(ans))
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
            print("done", ds, flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 250, a[3:] or None)

"""Does a typed model understand the actual prompt? Part A: the prompt's own jargon (true/false implications). Part B: reading facts out of REAL rendered states
(ground truth from the state dict). Noul calls. Run: .venv/bin/python -m acq.prompt_probe [model ...]"""
import sys, os, glob, pickle
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acq.engine import AFS
from acq.llm_selector import render_state
from acq.knowledge_probe import call

JARGON = [
    ("If the best value found has not improved for many iterations, the optimiser should usually explore more.", 1),
    ("A high mean predictive standard deviation over untested points means the model is still uncertain about many of them.", 1),
    ("A large positive standardised residual of the last observation means the value was much higher than the model predicted.", 1),
    ("Short Gaussian-process lengthscales mean the objective is assumed to change quickly with the inputs.", 1),
    ("The incumbent is the best value observed so far.", 1),
    ("When the acquisition functions propose many distinct points, they disagree about where to evaluate next.", 1),
    ("Near-zero Gaussian-process noise means measurements are treated as almost exact.", 1),
    ("A large positive standardised residual means the model predicted the value accurately.", 0),
    ("The incumbent is the worst value observed so far.", 0),
    ("When all acquisition functions propose the same point, they disagree strongly.", 0),
    ("Near-zero Gaussian-process noise means measurements are very unreliable.", 0),
    ("A high mean predictive standard deviation over untested points means the model is very certain about them.", 0),
    ("Little remaining budget is a reason to explore more widely rather than refine the best result.", 0),
    ("An acquisition function that has repeatedly failed to improve the best value is a good reason to keep using it.", 0),
]
FACTS = [  # (statement, truth(state))
    ("The best value found has not improved for more than 5 iterations.", lambda s: s["since_improve"] > 5),
    ("More than half of the evaluation budget remains.", lambda s: s["frac_left"] > 0.5),
    ("The last experiment improved the best value.", lambda s: s["last_improved"] == 1),
    ("The last observation was surprising: it differed from the model's prediction by more than one standard deviation.", lambda s: abs(s["surprise_last"]) > 1),
    ("At least five different points are currently proposed by the six acquisition functions.", lambda s: s["n_distinct"] >= 5),
    ("LogEI has been used more often than any other acquisition function.", lambda s: s["LogEI_used"] > 0 and s["LogEI_used"] >= max(s[f"{a}_used"] for a in AFS)),
]

def noul(model, state, stmt): return call(model, state, {"q": {"type": "noul", "instructions": stmt}})["q"]["noul"]

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    fs = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "phaseA", "*.pkl"))
    states = []
    for f in rng.choice(fs, 120, replace=False):
        r = pickle.load(open(f, "rb")); states.append(r["log"][int(rng.integers(1, 50))]["state"])
    for model in (sys.argv[1:] or ["tev1:0.8b", "nimble:latest"]):
        print(f"\n=== {model} ===")
        p = np.array([noul(model, "Notes on Bayesian optimisation.", st) for st, _ in JARGON]); y = np.array([t for _, t in JARGON])
        print(f"Part A (prompt jargon): accuracy {np.mean((p > .5) == y):.0%} ({int(np.sum((p > .5) == y))}/14), AUROC {roc_auc_score(y, p):.2f}, mean P(true) on true {p[y == 1].mean():.2f} / false {p[y == 0].mean():.2f}")
        for (st, t), pi in zip(JARGON, p):
            if (pi > .5) != t: print(f"   wrong: P={pi:.2f} truth={t}: {st[:95]}")
        print("Part B (reading real states), 120 states:")
        txt = [render_state(s, menu=False, question=False) for s in states]
        for stmt, fn in FACTS:
            truth = np.array([bool(fn(s)) for s in states]); pr = np.array([noul(model, t, stmt) for t in txt])
            acc = np.mean((pr > .5) == truth); maj = max(truth.mean(), 1 - truth.mean())
            auc = roc_auc_score(truth, pr) if 0 < truth.sum() < len(truth) else float("nan")
            print(f"   {stmt[:78]:78s} acc {acc:.0%} (always-majority {maj:.0%}) AUROC {auc:.2f} base-rate true {truth.mean():.0%}")

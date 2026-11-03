"""Does a typed decision model know the BO vocabulary, and can it use it? Part 1: true/false concept statements (Noul). Part 2: synthetic states with an
unambiguous right action (Choice between explore and exploit). Run: .venv/bin/python -m acq.knowledge_probe [model ...]"""
import sys, os, json, urllib.request
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URL = "http://localhost:11434/v1/systemone"

CONCEPTS = [  # (statement, truth)
    ("Expected improvement is an acquisition function used in Bayesian optimisation to decide which experiment to run next.", 1),
    ("A Pareto front is the set of solutions where no objective can be improved without making another objective worse.", 1),
    ("Hypervolume measures how much of the objective space is dominated by the Pareto front.", 1),
    ("Pure exploitation chooses the point with the best predicted value.", 1),
    ("Upper confidence bound adds a multiple of the predictive uncertainty to the predicted mean, which encourages exploring uncertain regions.", 1),
    ("A long Gaussian-process lengthscale means the function is assumed to vary slowly and smoothly across inputs.", 1),
    ("Thompson sampling picks the next point by drawing one random function from the model's posterior and maximising it.", 1),
    ("Expected improvement ignores the model's uncertainty and uses only the predicted mean.", 0),
    ("A point on the Pareto front is dominated by another point.", 0),
    ("A larger hypervolume means the Pareto front is worse.", 0),
    ("Pure exploration chooses the point with the best predicted value.", 0),
    ("A short Gaussian-process lengthscale means the function is assumed to be very smooth and slowly varying.", 0),
    ("Simple regret goes down when the best value found so far gets worse.", 0),
    ("A standardised residual of +3 on the last observation means the model predicted that value very accurately.", 0),
]
OPTS = {"explore": "Choose the point with the highest uncertainty, to learn about unexplored regions.",
        "exploit": "Choose the point with the best predicted value, to refine around the best result found so far."}
def S(t, T, since, improved_last, sd_mean, n_obs):
    return (f"Bayesian optimisation of an expensive function. Iteration {t} of {T}. {n_obs} evaluations so far. "
            f"The best value found has not improved for {since} iterations. The last experiment {'improved' if improved_last else 'did not improve'} the best value. "
            f"Mean predictive uncertainty over the remaining candidates is {sd_mean}.")
STATES = [  # (state text, correct answer)
    (S(48, 50, 0, True, "very low (0.01)", 63), "exploit"),
    (S(49, 50, 3, False, "low (0.02)", 64), "exploit"),
    (S(47, 50, 1, True, "low (0.02)", 62), "exploit"),
    (S(46, 50, 0, True, "very low (0.01)", 61), "exploit"),
    (S(2, 50, 2, False, "very high (0.90)", 17), "explore"),
    (S(3, 50, 1, False, "very high (0.85)", 18), "explore"),
    (S(1, 50, 0, True, "very high (0.95)", 16), "explore"),
    (S(30, 50, 28, False, "high (0.40)", 45), "explore"),
    (S(25, 50, 24, False, "high (0.45)", 40), "explore"),
    (S(40, 50, 35, False, "high (0.35)", 55), "explore"),
]

def call(model, state, questions):
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read().decode())["answers"]

if __name__ == "__main__":
    for model in (sys.argv[1:] or ["tev1:0.8b", "nimble:latest"]):
        p, y = [], []
        for stmt, truth in CONCEPTS:
            a = call(model, "General knowledge about Bayesian optimisation and multi-objective optimisation.", {"q": {"type": "noul", "instructions": stmt}})
            p.append(a["q"]["noul"]); y.append(truth)
        p, y = np.array(p), np.array(y)
        print(f"\n{model}\n  Part 1 concepts: accuracy {np.mean((p > .5) == y):.0%} ({int(np.sum((p > .5) == y))}/{len(y)}), AUROC {roc_auc_score(y, p):.2f}; "
              f"mean P(true) on true statements {p[y == 1].mean():.2f}, on false {p[y == 0].mean():.2f}")
        for (stmt, t), pi in zip(CONCEPTS, p):
            if (pi > .5) != t: print(f"    wrong: P={pi:.2f} truth={t}: {stmt[:90]}")
        ok, conf = [], []
        for st, ans in STATES:
            a = call(model, st, {"q": {"type": "choice", "instructions": "What should the optimiser do next?", "criteria": OPTS}})["q"]
            pr = a["probabilities"]; pick = max(pr, key=pr.get); ok.append(pick == ans); conf.append(pr[ans])
        print(f"  Part 2 states: accuracy {np.mean(ok):.0%} ({sum(ok)}/{len(ok)}; chance 50%); mean P(correct) {np.mean(conf):.2f}; per-state correct: {[int(o) for o in ok]}")

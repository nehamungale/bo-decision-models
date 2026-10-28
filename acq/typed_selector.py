"""Phase C: typed decision-model selector (tev1 via Ollama /v1/systemone, a stand-in for Jev). One Choice call per step over the six AFs.
Modes (fixed in PLAN_acquisition_choice.md): 'argmax', 'sample' (draw from the returned probabilities), 'gated' (argmax if confidence >= tau, else LogEI).
Every call is logged with all six probabilities and the confidence for the calibration study. Fallback on any error: LogEI (flagged)."""
import json, time, urllib.request
import numpy as np
from acq.engine import AFS, AF_GLOSS
from acq.llm_selector import render_state

URL = "http://localhost:11434/v1/systemone"
INSTR = ("Which acquisition function should the Bayesian optimisation loop use for its next experiment? "
         "Avoid acquisition functions that have repeatedly failed to improve the incumbent.")


def ask_choice(state_text, model="tev1:0.8b", timeout=90):
    body = json.dumps({"model": model, "state": state_text, "questions": {"af": {
        "type": "choice", "instructions": INSTR, "criteria": {a: AF_GLOSS[a] for a in AFS}}}}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    a = out["answers"]["af"]
    return np.array([a["probabilities"][k] for k in AFS], float), float(a.get("confidence", np.nan)), time.perf_counter() - t0


class TypedSelector:
    needs_peek = False
    def __init__(self, mode="sample", tau=None, model="tev1:0.8b"):
        self.mode, self.tau, self.model = mode, tau, model
        self.name = f"typed_{model.split(':')[0]}_{mode}"
        self.calls = []
    def reset(self, seed): self.calls = []; self.rng = np.random.default_rng(seed + 4242)
    def __call__(self, step, peek):
        try:
            p, conf, dt = ask_choice(render_state(step.state, menu=False, question=False), self.model)
            p = np.clip(p, 0, None); p = p / p.sum(); err = None
        except Exception as e:
            self.calls.append(dict(t=step.t, probs=None, conf=None, latency=None, af=0, fallback=True, error=str(e)[:80])); return 0
        if self.mode == "argmax": af = int(np.argmax(p))
        elif self.mode == "sample": af = int(self.rng.choice(len(AFS), p=p))
        elif self.mode == "gated_adaptive":      # regime-invariant gate: follow argmax if confidence >= median of this campaign's earlier confidences (first 5 steps always follow)
            prev = [c["conf"] for c in self.calls if c["conf"] is not None]
            af = int(np.argmax(p)) if (len(prev) < 5 or conf >= np.median(prev)) else 0
        else: af = int(np.argmax(p)) if conf >= self.tau else 0
        self.calls.append(dict(t=step.t, probs=p.tolist(), conf=conf, latency=dt, af=af, fallback=False, error=None))
        return af

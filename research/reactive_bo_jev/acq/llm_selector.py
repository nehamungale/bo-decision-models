"""LMABO-style generative LLM selector (reference for Phase D). The state is rendered as text, the model answers with the
name of one AF. Fallback on a malformed answer: UCB (as in LMABO). Backend: Ollama /api/chat (local or *-cloud models)."""
import json, re, time, urllib.request
import numpy as np
from acq.engine import AFS, AF_GLOSS

URL = "http://localhost:11434/api/chat"
SYSTEM = (
    "You are the strategy module of a Bayesian optimisation loop that maximises an expensive black-box function over a "
    "finite candidate set. At every iteration you choose ONE acquisition function from the menu to pick the next experiment. "
    "Base your choice on the optimisation state. Avoid acquisition functions that have repeatedly failed to improve the "
    "incumbent. Answer with JSON only, exactly: {\"acquisition\": \"<name>\"} where <name> is one of: " + ", ".join(AFS) + "."
)


def render_state(s, D=7, menu=True, question=True):
    af_lines = "\n".join(
        f"  {a}: used {int(s[f'{a}_used'])}x, improved the incumbent {int(s[f'{a}_improved'])}x" for a in AFS)
    last = "none yet" if s["last_af"] < 0 else f"{AFS[int(s['last_af'])]} ({'improved' if s['last_improved'] else 'did not improve'} the incumbent)"
    return (
        f"Iteration {s['t']} of {int(round(s['t'] / (1 - s['frac_left']))) if s['frac_left'] < 1 else 50}; "
        f"{s['frac_left']:.0%} of the budget remains; {int(s['n_obs'])} evaluations so far; {D} input dimensions (scaled to [0,1]).\n"
        f"Incumbent is {s['best_z']:.2f} standard deviations above the mean of all observed values.\n"
        f"Iterations since the incumbent last improved: {int(s['since_improve'])}; improvements in the last 10 iterations: {int(s['n_improve_last10'])}.\n"
        f"Surprise of the last observation (standardised residual of the model's prediction): {s['surprise_last']:+.2f}.\n"
        f"Gaussian-process model: lengthscales min {s['ls_min']:.2f}, mean {s['ls_mean']:.2f}, max {s['ls_max']:.2f}; "
        f"noise {s['noise']:.4f}; predictive std over unevaluated points: mean {s['sd_mean']:.2f}, max {s['sd_max']:.2f}.\n"
        f"Distance from the last evaluated point to its nearest earlier point: {s['dist_last']:.2f}.\n"
        f"The six acquisition functions currently propose {int(s['n_distinct'])} distinct points.\n"
        f"Last acquisition function used: {last}.\nUsage history:\n{af_lines}\n\n"
        + ("Menu:\n" + "\n".join(f"  {a}: {AF_GLOSS[a]}" for a in AFS) + "\n\n" if menu else "")
        + ("Which acquisition function should be used next?" if question else "")
    )


def parse(text):
    m = re.search(r'"acquisition"\s*:\s*"([A-Za-z]+)"', text or "")
    if m and m.group(1) in AFS: return AFS.index(m.group(1))
    for a in AFS:                                   # lenient: a bare name
        if re.fullmatch(rf"\s*{a}\s*", text or ""): return AFS.index(a)
    return None


def ask(state, model, think="low", timeout=180, temperature=0.0):
    body = json.dumps({"model": model, "stream": False, "think": think,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render_state(state)}],
                       "options": {"temperature": temperature}}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    text = out.get("message", {}).get("content", "")
    return text, dt, out


class LLMSelector:
    """Selector wrapper: retries on HTTP 429/5xx with backoff, falls back to UCB on a malformed answer, logs every call."""
    needs_peek = False
    def __init__(self, model="gpt-oss:120b-cloud", think="low", max_retries=6):
        self.model, self.think, self.max_retries = model, think, max_retries
        self.name = "llm_" + model.split(":")[0].replace("-", "")
        self.calls = []
    def reset(self, seed): self.calls = []
    def __call__(self, step, peek):
        import urllib.error
        last = None
        for attempt in range(self.max_retries):
            try:
                text, dt, out = ask(step.state, self.model, think=self.think); break
            except urllib.error.HTTPError as e:
                last = e.code; time.sleep(min(30, 2 ** attempt))
            except Exception as e:
                last = str(e)[:80]; time.sleep(min(30, 2 ** attempt))
        else:
            self.calls.append(dict(t=step.t, text=None, latency=None, af=AFS.index("UCB"), fallback=True, error=str(last))); return AFS.index("UCB")
        af = parse(text)
        self.calls.append(dict(t=step.t, text=text, latency=dt, af=af if af is not None else AFS.index("UCB"), fallback=af is None, error=None))
        return af if af is not None else AFS.index("UCB")

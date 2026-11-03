import json, time, urllib.request

URL = "http://localhost:11434/v1/systemone"


def ask(state, questions, model="tev1:0.8b", timeout=90, url=URL):
    """POST one typed call. Returns (answers dict, wall seconds). Raises on HTTP/JSON errors."""
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    if "answers" not in out:
        raise RuntimeError(out)
    return out["answers"], dt


def flatten(ans):
    """Answers -> flat dict of numbers: noul probs, score expectation/confidence, p(stop)/confidence."""
    f = {}
    for k, a in ans.items():
        if a["type"] == "noul":
            f[k] = a["noul"]
        elif a["type"] == "score":
            f[k] = a["score"]; f[k + "_conf"] = a.get("confidence")
        elif a["type"] == "choice":
            f["p_stop"] = a["probabilities"]["stop"]; f["action_conf"] = a.get("confidence")
    return f

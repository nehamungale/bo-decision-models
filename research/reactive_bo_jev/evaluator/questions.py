"""The typed question set asked about each state."""
NOULS = {
    "will_finish_top10": "This run will finish training among the best 10% of all runs.",
    "above_most_peers": "This run is currently better than most of the other runs.",
    "plateaued": "This run's accuracy curve has stopped improving.",
    "improving_fast": "This run's accuracy is still improving quickly.",
    "unstable": "This run's accuracy curve is noisy or unstable.",
    "will_beat_best_so_far": "This run will finish above the best fully trained run so far.",
}
PROMISE = ["hopeless", "weak", "average", "good", "excellent"]


def build_questions(names=None):
    """Return the questions dict for /v1/systemone. `names` restricts to a subset (for latency scaling)."""
    qs = {k: {"type": "noul", "instructions": v} for k, v in NOULS.items()}
    qs["promise"] = {"type": "score", "instructions": "How promising is this training run?", "criteria": PROMISE}
    qs["action"] = {"type": "choice", "instructions": "What should be done with this training run?",
                    "criteria": {"continue": "Keep training this run.", "stop": "Stop this run and free the worker."}}
    return qs if names is None else {k: qs[k] for k in names}

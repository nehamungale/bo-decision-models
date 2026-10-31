# bo-decision-models

Experiment: in a Bayesian optimization loop, let a **decision model** choose the
next point to evaluate, instead of asking an LLM to propose candidates.

A decision model here = a surrogate (GP) plus an explicit utility/acquisition
rule (e.g. expected improvement), so each choice is the argmax of an expected
utility rather than free-form text generation.

## Layout

- `src/bo_dm/objectives.py` – test functions (Branin)
- `src/bo_dm/proposers.py` – `Proposer` protocol; `RandomProposer`, `GPEIProposer`
- `src/bo_dm/loop.py` – the BO loop, proposer-agnostic
- An LLM proposer can be added later by implementing the same `Proposer`
  protocol, so all strategies are compared on the same loop and budget.

## Run

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m bo_dm.loop
```

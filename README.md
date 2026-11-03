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

## Research: reactive BO with typed decision models (`research/reactive_bo_jev/`)

Thesis experiments on whether a component of a BO campaign (acquisition function, surrogate, search region, warm start, source prior, and response to a mid-run objective change) can be changed during the run to find the optimum better. History is carried over from the `llm_af_evo` repository (split of commit 44c04c8). Start with `research/reactive_bo_jev/APPROACHES_SO_FAR.md`, `notes.md` and `PLAN_nonstationary.md`; the thesis drafts are `THESIS_*.md`.

The folder is self-contained and is run from inside it:

```
cd research/reactive_bo_jev
pip install -r requirements.txt            # botorch, torch, gpytorch, pandas, syne-tune
python -m meta.test_nonstat_a2              # fast checks (about 1 minute)
python -m meta.analyse_nonstat_a2           # re-derives meta/results/nonstat_a2.txt from the stored runs
```

Landscape data (`meta/pools/*.npz`, `acq/lcbench_final_pools.npz`) and raw run files (`meta/results/*/*.pkl`) are git-ignored; keep a local copy or regenerate with the scripts. The scaffold in `src/bo_dm/` is separate.

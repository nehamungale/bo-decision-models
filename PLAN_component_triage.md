# Component headroom triage — plan for approval (2026-10-28)

Status: DRAFT, nothing built. Supersedes the "try AF choice again with richer inputs" option in [`notes.md`](notes.md).
Context: AF choice failed its stop rule (Stage 2b: GBM +0.0010 over best fixed, indistinguishable from label-shuffled control; random AF ≈ best fixed AF). A web check
(notes.md, same date) suggests the AF is the least differentiated component in a noise-free, low-dimensional, GP-on-pool setting, that the gains reported for "learned acquisition" come from
transferring structure across related tasks, and that the LLM evidence is about priors and regions, not AF choice. This plan measures where the room is **before** any model is trained.

## 1. Question

For each reactive component, how much better could the campaign be with hindsight knowledge of the best setting, compared with the best **fixed** setting chosen without it? A component earns a decision model
only if that headroom exceeds a threshold fixed now. No model, LLM or fine-tune is involved in this plan.

## 2. Components and menus (each compared on the same landscapes, budget rule and metric as before)

Metric: AUC of normalised regret over the campaign (as in all earlier work); budget 2D+1 initial points then min(50, 25 % of the pool) steps; regret normalised by (y_max − y_median).

| ID | Component | Menu | Fixed baseline | Hindsight reference |
|---|---|---|---|---|
| K | Surrogate / kernel | ARD RBF (current default), ARD Matern-5/2, isotropic RBF, random-forest surrogate (mean + across-tree std) with LogEI-style score | the single best menu entry chosen on training landscapes | per-landscape best entry (selected on seeds A, scored on seeds B) |
| R | Search region | global (current), box of radius r around the incumbent with r in {0.15, 0.30} (candidates outside are masked), TuRBO-style adaptive box (grow on success, shrink on failure) | best entry chosen on training landscapes | per-landscape best entry (A/B seeds) |
| W | Warm start from related tasks | random initial design (current); zero-shot portfolio: initial points = best configs by mean rank over the other tasks of the same search space (exact shared grids); region prior for HPO-B (below) | random initial design | not an oracle: a real method (see section 4) |
| A | Acquisition function (done) | six AFs | LogEI | one-step oracle, per-landscape best: already measured |

Not in this plan: early termination and algorithm switching (need multi-fidelity learning curves; only LCBench has them, so they use the earlier simulator), batch strategy (needs a batch engine). They come after.

## 3. Hypotheses and thresholds (fixed before data)

- **TH-K / TH-R (landscape-level hindsight headroom).** Per-landscape-best menu entry, selected on seeds A (4000–4004) and scored on fresh seeds B (4100–4119), beats the best fixed entry (chosen on the other landscapes) by at least **5 % relative AUC regret**,
  with Wilcoxon p < 0.05 across landscapes and the gain present in at least two of the three landscape families. The A/B split removes the winner's-curse bias that inflated the earlier "12 % headroom" to a true ≈ 4.5 %.
  The 5 % threshold is my judgment: it is about the AF headroom that proved unlearnable, so a component should clear that bar, not merely reach it.
- **TH-W (warm start).** Zero-shot portfolio start beats the random initial design by at least **10 % relative AUC regret** on held-out tasks (leave-one-task-out within each search space), paired by seed, Wilcoxon p < 0.05, on at least two families. Set higher than K/R because W is a real method, not an upper bound.
- A passing component moves on to a step-level question (does knowing *when* to switch add anything beyond the best landscape-level setting), with a counterfactual label, a fixed-setting baseline and a shuffled-label control, as in Stage 2b. A failing component is closed and reported as a negative result.
- Reported either way: per-family results, both summaries (change in mean AUC and mean of per-landscape ratios, which can disagree in sign), and the fraction of landscapes where the hindsight entry equals the fixed one.

## 4. How W works (exact grids vs regions)

Config grids are identical across tasks for LCBench (30 of 30) and TabRepo (almost all; the few tasks with a smaller grid are excluded from W) and different for HPO-B (each task has its own sampled points in a shared continuous space). So:
- **W1, exact portfolio (LCBench, TabRepo):** rank configs by mean normalised score over the training tasks; the first `n0` points of a campaign are the top `n0` of that ranking (n0 = 2D+1, same number of evaluations as the random start, which they replace).
- **W2, region prior (HPO-B):** pool the best 5 % of points of each training task in the search space; the initial design is the `n0` pool points nearest to the cluster centres of that set (k-means, k = n0). Same evaluation count.
- Controls: random start with the same budget, and a **mismatched-portfolio control** (portfolio from a different search space or model family, where applicable), which must not help.
- Leave-one-task-out: the held-out task never contributes to its own portfolio.

## 5. Seeds, landscapes, cost, stages (stop and ask at each)

- Landscapes: the 30 from Stage 2b (10 per family) as dev; the reserved LCBench datasets, FCNet and NAS-Bench-201 stay untouched for a later final test. Seeds: A = 4000–4004, B = 4100–4119 (disjoint from all earlier blocks).
- Cost per campaign 5 to 25 s without labels. Estimates, to be measured in stage T0.

| Stage | What | Estimated cost (12 cores) | Stop rule |
|---|---|---|---|
| T0 | Implement menus K, R, W in the engine; unit tests (determinism, W excludes the held-out task, masks work, RF surrogate is sane); time each | no compute | any menu entry not reproducing the default when set to the default |
| T1 | W on 10 landscapes (LCBench and TabRepo first), 20 seeds, 3 arms | about 20 min | W gain < 3 % on all families: close W |
| T2 | K on the same 10 landscapes (4 entries, 5 A seeds + 20 B seeds) | about 40 min | no entry beats the best fixed by > 2 % on the A seeds: close K |
| T3 | R on the same 10 landscapes | about 30 min | as T2 |
| T4 | Expand the passing components to all 30 landscapes; test TH-K/R/W | up to about 3 h per component | TH fails: close |
| T5 (optional) | Repeat the best component under observation noise (Gaussian, 5 % of the span) | about the same as T4 | reported as a robustness arm, not a gate |

## 6. Where a decision model or an LLM would enter (only after a component passes)

- W: a typed Score or Choice question to rank source tasks as similar, using hyperparameter *names* (restored from the source tables; the pools currently carry anonymous numbers), and to judge whether the prior is still trustworthy as observations arrive (the fall-back pattern of meta-learned acquisition).
- R: when to shrink or expand the region (trust-region state is a small, interpretable input).
- K: which surrogate to trust given leave-one-out error (already logged as a descriptor).
- Any step-level label would use trajectory-level regret rather than the 4-step incumbent gain, which was 70 % zeros.

## 7. Risks

- Hindsight headroom measured at landscape level says nothing about step-level switching; a component can pass T4 and still fail the step-level question.
- Warm-start gains on TabRepo/LCBench rely on shared grids and similar tasks; real labs rarely have 200 comparable prior campaigns. The mismatched-portfolio control and the leave-family-out reading cover part of that.
- RF surrogate and masks add implementation surface; the T0 tests are there for that.
- Everything is noise-free until T5.

## 8. Open decisions for the user

1. Approve or change the thresholds (5 % for K/R, 10 % for W) and the menus.
2. Is it acceptable to start with W (largest expected effect, cheapest), then K, then R?
3. Include T5 (noise) now or later?

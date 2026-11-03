# Component 1b: a landscape-aware, outcome-trained acquisition selector (hybrid decision model + LLM) — plan for approval (2026-10-28)

Status: DRAFT, nothing built. Supersedes the zero-shot line of [`PLAN_acquisition_choice.md`](PLAN_acquisition_choice.md), whose result is in [`notes.md`](notes.md)
(no zero-shot selector beat the best fixed AF; the global state carried no usable signal; candidate-level numbers carried signal, AUROC 0.70, already used by the AFs).
Goal remains **better optima** (regret against evaluations), not speed.

## 1. Why this is not a repeat

The earlier nulls were about: 8 landscapes, one-step improvement labels (6 % base rate), zero-shot models, a state without landscape descriptors. Per the user's critique and the
literature (BioDecision-4B v2 on a 1.75M-decision pool; CS1-4B on 12.4M pairs, arXiv 2610.27099) a decision model earns its place by being trained on outcome labels at scale, and the realistic
division of labour is repeated typed judgements for the decision model, planning and synthesis for the LLM. The data check supports the premise: the best fixed AF differs by landscape and
per-dataset best-static knowledge is worth about 4.5 % AUC at 10 seeds (stable on only 3/8 datasets), so the available gain is real but modest, and 8 landscapes cannot teach it.

## 2. Question and hypotheses (fixed before building)

Question: trained on many landscapes with outcome labels, can a selector choose the AF (equivalently the proposal) better than the best fixed AF on **held-out landscapes**, and what does each
model class add?

- **H1 (gate, model-free):** a gradient-boosted control (GBM) trained on counterfactual-rollout labels beats the best fixed AF (chosen on the training landscapes) on held-out landscapes by at least
  `delta` relative AUC regret, paired across landscapes (Wilcoxon p < 0.05). `delta` is set after the pilot variance estimate and before any selector is evaluated; default 3 % (LMABO's unreplicated claim is about 10 %;
  3 % is the smallest effect I would spend a fine-tune on). Failing H1 closes the component and fine-tuning is not run.
- **H2:** a fine-tuned decision head is non-inferior to the GBM (CI of the difference within +/-2 % AUC) and calibrated after recalibration (ECE < 0.05). Reported either way; it answers "does a language-model head add anything".
- **H3 (hybrid roles):** each LLM role in section 6 changes AUC relative to the head alone by more than the equivalence margin; exploratory, Holm over the roles tested.
- **H4 (transfer):** the gain survives leave-family-out (a held-out HPO-B search space, a held-out TabRepo model family, FCNet, NAS-Bench-201) and the untouched 5-dataset LCBench confirmation set.

## 3. Landscape set (the effective sample size is the number of landscapes, not states)

Real tabular landscapes are available through the Syne Tune blackbox repository (inventory loaded 2026-10-28; load time and download size vary, sizes not measured). Each is a table
configuration -> measured metric, i.e. a discrete pool with the same interface as the LCBench pools:

| Family | Landscapes | Inputs | Pool size | Note |
|---|---|---|---|---|
| LCBench | 35 datasets | 7, mixed | 2000 | neural-net hyperparameters; 5 reserved (`splits.CONFIRM`) |
| HPO-B (`hpob_*`, 22 search spaces) | e.g. 69 tasks (`hpob_5859`, 4 inputs, 1157 configs), 44 tasks (`hpob_4796`, 3 inputs, 279 configs); hundreds in total | 3-6 | 279-1157+ | OpenML tasks, one accuracy value per config |
| TabRepo (`tabrepo_*`, 8 model families) | about 240 datasets per family, so roughly 1900 | 3-10 | 200 | tree/boosting/linear/NN hyperparameters; error to be negated |
| FCNet | 4 | 9 | 62208 (subsample to about 3000) | held out as a whole family |
| NAS-Bench-201 | 3 | 6 categorical | 15625 (subsample) | held out as a whole family; categorical inputs need care |

**Phase 1 result (2026-10-28), exclusion rules fixed from landscape properties only, before any campaign was run on them.** Exporter: `meta/export_landscapes.py` (pools in `meta/pools/*.npz`, `inventory.csv`, `inventory_ext.csv`).
Encoding: log then min-max for positive inputs spanning more than two decades, 0/1 for two-level categoricals, one-hot otherwise (NAS-Bench-201 becomes 30 inputs). Metrics: HPO-B accuracy; TabRepo validation error averaged over 3 seeds, negated; FCNet final validation loss averaged over 4 seeds, negated;
NAS-Bench-201 validation error averaged over 3 seeds, negated; LCBench unchanged (verified identical to the earlier pools). Tables above 3000 configs subsampled to 3000 with a fixed seed. 2629 landscapes exported, none with NaNs or duplicated inputs.
Keep a landscape if: at least 90 configs; best value above the median (regret defined); fewer than 50 % of configs tied at the median (otherwise the regret scale is set by a plateau); at most 5 % of configs tied at the optimum. Result: 1983 kept
(LCBench 35, HPO-B 532 over 14 of 16 search spaces, TabRepo 1409, FCNet 4, NAS-Bench-201 3). Dropped wholesale: TabRepo LinearModel (every value missing in this repository version), TabRepo KNeighbors (50 configs), and five HPO-B search spaces that are
mostly plateaus (5636, 5859, 5860, 5970, 6766 keep only 1 to 4 tasks each). Not yet checked: how many kept landscapes are trivially easy (all AFs reach the optimum within the initial design); this is measured in the pilot and reported, not used to filter.

- **Meta-train/validation:** a sampled set of about 150 landscapes across LCBench (30, excluding the 5 reserved), HPO-B (about 60, spread over several search spaces) and TabRepo (about 60, spread over model families), with
  group cross-validation by landscape; folds never split a landscape, and a search space or model family is held out whole in the leave-family-out test.
- **Final test:** the 5 reserved LCBench datasets, FCNet, NAS-Bench-201, and one held-out HPO-B search space and TabRepo model family; run once.
- **Budget per landscape:** 2D+1 initial points then min(50, 25 % of the pool) steps, so a 200-config pool is not exhausted; regret stays normalised by (y_max - y_median).
- **Synthetic analytic functions are dropped** unless the real families prove too few; they are not lab landscapes either.
- Caveats: HPO-B and TabRepo are low-dimensional and hyperparameter-tuning landscapes, still not chemistry or materials; small pools make the problem easier and change the regret scale; the real-lab transfer question stays open.

## 4. Labels: counterfactual rollouts (the "outcome after the decision" label)

For a logged state `s` (from campaigns under a mixture of behaviour policies: six statics, random-AF, and a mixed policy), for each AF `a`: take `a`'s proposal at `s`, then continue with deterministic LogEI for
`H - 1` further steps (H = 4 primary, H = 8 sensitivity on a subsample). Label `V_a(s)` = (incumbent after H steps - incumbent before) / (y_max - y_median) of that landscape. LogEI as the fixed continuation removes
rollout RNG noise (TS and random-AF are not used in continuation).
- **G0, label reliability gate:** repeat the labels with a second continuation policy (UCB). Mean within-state Spearman across the six AFs between the two label sets must exceed 0.3 on the pilot landscapes,
  and labels must beat a shuffled-label control. If labels are noise, stop and report; more data would not fix it.
- Sampling: every 3rd step of 20 seeds per landscape (about 340 states), 6 AFs, H = 4. Estimated cost per landscape 340 x 6 x 3 steps x 0.14 s ~ 15 min of one core (estimate; the pilot measures it). About 150 landscapes
  is then roughly 40 core-hours, around 3 to 4 hours on the 12 cores. Scale the number of states, not landscapes, if over budget.

## 5. Inputs, models, controls

Inputs (all numeric, rendered as text for the language head): per-AF candidate numbers (the nine from `CAND_KEYS`: gap, std, z, PI, EI, distances, ranks), AF id, progress, improvement history, and **landscape
descriptors** computed from the campaign so far: ARD lengthscale statistics and effective dimension, noise-to-signal ratio, GP leave-one-out error, fraction of observations near the incumbent, incumbent-local vs global
gap, input dimension. Recorded in a replay run exactly as `acq/replay_cand.py` did.

| ID | Model | Role |
|---|---|---|
| B0 | six static AFs, random-AF, GP-Hedge, best-static-chosen-on-train | baselines |
| B1 | per-landscape oracle best static (hindsight) | upper reference for regime knowledge |
| M1 | gradient-boosted regression of `V_a` per candidate, argmax | model-free control (the gate) |
| M2 | small decoder (Qwen-family 0.5B-class) + LoRA + candidate head, CLM-style: each (state, candidate) scored, softmax over the six, soft-cross-entropy against the softmax of standardised `V_a`; temperature + log-odds recalibration on held-out landscapes | the fine-tuned decision model |
| C1 | label-shuffled M1 | must be at random level |
| C2 | AF-id-only M1 | must reproduce the best fixed AF |
| C3 | feature-group ablations: candidate-only, descriptors-only, history-only, both | what carries the signal |
| M3 | zero-shot nimble/tev1/gpt-oss on the same text (from the earlier work) | showing what training adds |

## 6. Hybrid roles to test after H1 and H2 (exploratory, one at a time)

A. head alone. B. LLM invoked only when the head's confidence is low, choosing between its top two. C. LLM writes a short regime description from the descriptors, appended to the head's input.
D. LLM maps a stated campaign goal (screening vs refinement, a spec threshold) to a bound on the head's choices (needs a context channel; synthetic text, labelled as such).
Each is compared with A on held-out landscapes; the question "what is needed where and when" is answered by which roles move AUC.

## 7. Evaluation, seeds, statistics

- **Offline:** achieved H-step label on held-out landscapes vs baselines. **Online (primary):** full campaigns (15 or 2D+1 initial points, 50 steps) on held-out landscapes, 20 seeds, AUC of normalised regret,
  paired by seed; summaries reported both as change in mean AUC and mean of per-landscape ratios (they can disagree in sign, seen in Phase C). Wilcoxon across landscapes, Holm correction within each hypothesis family.
- Seed blocks (disjoint): label generation 2000+, model selection/tuning 3000+, evaluation 0-19, final confirmation 1000+.
- Behaviour-policy shift: training states come from static/random policies; the learned selector visits other states. After the first M1 evaluation, one DAgger-style round (collect states under M1, relabel, retrain) is allowed and counted as part of the tuning budget.

## 8. Phases and checkpoints (stop and ask the user at each)

1. Export the sampled landscapes (22 more LCBench, about 60 HPO-B, about 60 TabRepo, FCNet and NAS-Bench-201 subsamples) to plain `.npz` pools; check each (no NaNs, the optimum is not duplicated, metric sign fixed so higher is better); add descriptor logging to the replay. *Checkpoint: landscape inventory.*
2. **Pilot labels on 6 landscapes** (2 LCBench, 2 HPO-B, 2 TabRepo): measure cost, label reliability (G0), variance for setting `delta`. *Checkpoint: G0 pass/fail.*
3. Full labels; M1 and controls C1-C3 with group CV; **H1 gate** on the validation folds. *Checkpoint: continue or close.*
4. Only if H1 passes: online campaigns for M1; then M2 fine-tune (budget and backbone confirmed with the user first); recalibration; H2.
5. Hybrid roles (section 6); final test on the untouched confirmation set (run once).

## 9. Risks

- Effective sample is the landscape count (30 + about 120); a selector could overfit to landscape identity even under group CV; the leave-family-out and confirmation tests guard this.
- Counterfactual labels with a LogEI continuation reward AFs that complement LogEI; H = 8 sensitivity checks this.
- LCBench is noise-free and 7-dimensional; real labs are noisy and different. The in-distribution vs transfer gap in the CS1 paper (+16 vs +5 points) is the realistic expectation.
- Fine-tuning a Qwen head on Apple silicon is slow; a gate before M2 avoids wasting it.
- Synthetic text for role D is invented; results show a possibility, not that real lab text helps.

## 10. Decisions (resolved 2026-10-28)

1. Gates approved by the user: G0 (label reliability, Spearman > 0.3) and H1 at `delta` = 3 % set after the pilot variance estimate.
2. M2 backbone deferred until H1 passes.
3. Real tabular landscapes preferred over synthetic ones: HPO-B, TabRepo, LCBench as the main families, FCNet and NAS-Bench-201 as held-out families.
4. The menu stays the six AFs: LogEI, LogPI, UCB (mean + 2 std), TS, PosMean, PosSTD. Not yet decided: whether to alter the menu before labels are generated (see notes).

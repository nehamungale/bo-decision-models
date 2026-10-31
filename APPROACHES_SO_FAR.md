# Reactive BO with typed decision models: every approach tried so far (as of 2026-10-28)

Written for: you, as a thesis record. Numbers come from `notes.md`, the plans, and `meta/results/*.txt`. The earliest phase (section 1) is reconstructed from commit messages and saved notes, so check it against `notes.md` before quoting it.

## 0. Goal and protocol

Make BO reactive: add or change one component during a run (acquisition function, search region, surrogate, warm start, batch, early stop) so the optimum is found with lower regret per evaluation. Typed Jev-style decision models (Noul, Choice, Score) are the proposed decision layer. Quality first, speed later, real lab eventually.

- Metric: AUC of normalised regret, (y_max - best) / (y_max - y_median). Lower is better.
- Engine: pool-based sequential BO, BoTorch GP (ARD RBF), six AFs (LogEI, LogPI, UCB, TS, PosMean, PosSTD), budget 2D+1 initial points then min(50, 25% of pool) steps.
- Testbeds: LCBench simulator first, then 1983 kept real landscapes (LCBench, HPO-B, TabRepo, FCNet, NAS-Bench-201).
- Stand-ins: tev1:0.8b, nimble (9B), gpt-oss:120b as generative reference. None is Jev itself.
- Rules: pre-register hypotheses and gates, strong numeric baselines on the same state, shuffled-label and state-blind controls, report negatives, flag post-hoc deviations.

## 1. Earlier work (before the AF-choice plan)

| # | Approach | Outcome |
|---|---|---|
| 1.1 | CLM-8B/Jev-style decision models for BO, CALIPER-trust (commit 5951f77) | Negative result |
| 1.2 | Does tev1's trust capital predict per-seed hypervolume uplift (commit 1e00812) | No |
| 1.3 | sdl_adaptive: LLM routing vs rule router vs EGBO | LLM routing is about equal to the rule router, both below EGBO. This is the baseline to beat |
| 1.4 | Knowledge probe of the stand-ins (`acq/results/prompt_probe_output.txt`) | Jargon statements: tev1 9/14, nimble 13/14. Reading real states: nimble AUROC about 1.0 on six statements, tev1 weaker. nimble became the fairer stand-in |

## 2. Approach A: reactive acquisition-function choice (`PLAN_acquisition_choice.md`)

Question: at each step, can a selector pick the AF so regret beats any fixed AF, and can a typed model beat a tuned numeric rule on the same state?

1. **Phase A, headroom gate (no models).** Oracle selector and static AF spread on 8 LCBench dev datasets. Gate G1 (oracle gain at least 20% on 6/8) failed at 5/8; G2 (static AFs differ) passed at 6/8. You chose to continue anyway, recorded as a post-hoc override (Amendment 1). Best static AF was TS: 0.1441 vs LogEI 0.1491, UCB 0.1490, random AF 0.1517.
2. **Phase B, numeric baselines.** GP-Hedge, a tuned decision-list rule (100 random configs, 5 tuning seeds), learned numeric selectors (per-AF gradient-boosted regressors trained on other datasets).
3. **Phase C, typed model.** tev1:0.8b as Choice over six AFs, modes argmax / sample / gated. The fixed-tau gate degenerated into static LogEI (design flaw, not rerun). Added a state-blind fixed-mixture control, which replaced the shuffled-state control.
4. **Phase D, generative reference.** gpt-oss:120b-cloud with an LMABO-style prompt, 10 seeds.
5. **nimble (9B) as a better stand-in** (Amendment 3): sample, argmax and gated-adaptive modes, with a state-blind control using nimble's own AF shares.

Result: no typed model beat the tuned rule or the best static AF. The confirmatory run on held-out datasets was not performed because no dev result justified it. Typed-model AF selection is a null result (commit 0d7f681).

## 3. Approach B: landscape-aware, outcome-trained AF selector (`PLAN_meta_selector.md`)

Reason: the zero-shot models saw no outcome signal. Train a selector on counterfactual rollouts across many real landscapes.

1. **Phase 1:** export every real tabular landscape to `.npz` pools and keep 1983 of 2629 by validity rules.
2. **Stage 0 and Stage 1:** counterfactual labels (horizon H=4, LogEI continuation) and gate G0 (label reliability, Spearman above 0.3 and above shuffled). G0 passes only on states where a ranking exists.
3. **Label sparsity:** 70 to 76% of states are all-zero (no AF improves within 4 steps), 13 to 24% have a ranking. A longer horizon (H=8) did not help. Trajectory-level labels were tried.
4. **GBM preview and powered 30-landscape check:** landscape-aware gradient-boosted selector, group CV, against best fixed AF.
5. **Stage 2b:** gate H1 failed. The GBM gained +0.0010 over the best fixed AF, indistinguishable from the label-shuffled control. A random AF per step is about as good as the best fixed AF.

Result: AF choice is closed from three angles (zero-shot typed models, learned numeric, landscape-aware GBM on 9460 states). The H2 fine-tuned head was never run because H1 failed.

## 4. Approach C: component headroom triage (`PLAN_component_triage.md`)

Reframe after a web check: AF is the least differentiated component in a noise-free, low-dimensional GP-on-pool setting, and learned-acquisition gains in the literature come from transfer across related tasks. Measure headroom for each component before training any model.

| Component | What was tried | Result |
|---|---|---|
| K, surrogate/kernel | ARD RBF vs Matern-5/2, isotropic, random forest; per-landscape best chosen on seeds A, scored on seeds B vs best fixed | Closed at noise 0 and 5% (hindsight no better than best fixed) |
| R, search region | Global vs boxes around incumbent (radius 0.15, 0.30), TuRBO-style adaptive box | Closed at both noise levels |
| W, warm start from related tasks | Zero-shot greedy portfolio as initial design vs random start, with controls (farthest-point start, fixed random sets, mismatched search space) | Passes (TH-W) at noise 0 and 5%. Large gains on LCBench and TabRepo, no gain from the HPO-B region version |
| A, acquisition | Done in Approaches A and B | Closed |

Also done: noise arm at 5% of span, T2/T3 winner's-curse control (hindsight chosen on seeds A, scored on B), and an engine rewrite (`meta/bo.py`, 11 tests) that reproduces the default exactly.

On 150 exact-grid landscapes the portfolio start gives about 39% lower AUC regret without noise and about 23% at 5% noise.

## 5. Approach D: fall-back decision for the warm start

Question: can the model tell when the portfolio start will lose to a random start?

- Failures are common: 33 of 150 landscapes without noise (22 to 30% across settings). An earlier "failures are rare" claim came from a 30-landscape set dominated by HPO-B and was corrected.
- Hindsight headroom is real and stable (label agreement 0.90 and 0.78 across seed halves): always-W1 0.0329, random 0.0535, hindsight 0.0247.
- Predictors: source-task features and target-observation features (rank agreement with source, R-squared, GP leave-one-out error, lengthscales, noise, similarity to sources), logistic regression and shallow GBM, grouped CV by dataset and by family, shuffled controls.
- **Every pre-declared cell failed.** Best source-only GBM: +3.6% (p = 0.38); target and combined features between +1.9% and -10.4%; shuffled controls 0.0327 to 0.0413. At 5% noise, best +2.7%, worst -21%.
- Failures are concentrated (top 10 hold 78% of excess regret) and depend on the (task, family) pair; no dataset fails in all its landscapes.

Committed as 8e61ba5. Not pushed.

## 6. Scoreboard

| Approach | Verdict |
|---|---|
| Typed zero-shot AF choice (tev1, nimble, gpt-oss) | Null: no better than tuned rule or best static AF |
| Learned numeric AF selector | Null |
| Landscape-aware outcome-trained AF selector | Fails H1; equals shuffled control |
| Surrogate choice (K) | Closed |
| Search region (R) | Closed |
| Warm start from related tasks (W) | Works, strongest finding; best case for transfer |
| Fall-back from warm start | Not predictable from cheap features |

## 7. Caveats

- Warm start is the best case for transfer: shared config grids, 29 to 238 source tasks. A real lab rarely has that.
- Mostly noise-free; 5% noise arms only for K, R, W and the fall-back.
- Fall-back gains were counted as cost-free, which is optimistic. A hedge arm (portfolio plus random at equal total budget) was designed but not run.
- Only simple descriptors tested. T2/T3 used few selection seeds.
- Never run: final tests on reserved LCBench, FCNet and NAS-Bench-201; confirmatory runs; early termination, algorithm switching and batch strategy; hyperparameter names for LLM roles.

## 8. Open options

1. Similarity-weighted source prior (RGPE style) that re-weights source tasks as observations arrive; the natural place for a typed Score/Choice question.
2. Cost-weighted fall-back target and a hedge arm (post-hoc, to be flagged).
3. Close the warm-start line and write up.
4. Components not yet touched: early termination, batch strategy, algorithm switching.

## 9. Update 2026-10-29

- **Approach E, extra AF-selection baselines (Amendment 4, `acq/results/phaseE_analysis.txt`):** non-contextual Thompson bandit (0.1441, equal to TS), LogEI-then-LogPI schedule (0.1498), minimum dwell rule (D = 1 won tuning, so it equals the tuned rule). None beats TS or LogEI (Holm p = 1.0). The AF-selection baseline table is complete.
- **Regime grid (`meta/results/regime_grid.txt`):** AF choice matters within a landscape (best-worst spread 69 % of LogEI, in-sample), but the fixed AF chosen on other landscapes is no better than random (-2.3 % overall, +42 % on HPO-B). Claim: no learnable AF headroom, not no AF effect. Noise could not be stratified from existing runs.
- **Approach F, online fall-back detection (`PLAN_fallback_detection.md`, `meta/results/detect_t1.txt`, `decide_detect.txt`):** 150 landscapes, 1800 units, 58 min. Insurance costs more than it saves (hedge +13 %, injection +14 to +16 %, FPS with n0/2 +5 % vs always-W1). Realised oracle 10.6 % below W1 (cost-free hindsight 26.6 %). All T2 detector cells FAIL, including the upper-bound detector that uses the target's own random runs; only 7 to 11 positives, so power is low.
- **Next:** similarity-weighted source prior (`PLAN_source_prior.md`, draft).

| Approach | Verdict |
|---|---|
| Bandit, schedule, dwell AF selection | Null |
| Online fall-back detection | Fails; ceiling about 10 % |

## 10. Update 2026-10-31
- **Approach G, similarity-weighted source prior (`PLAN_source_prior.md`, `PLAN_source_prior_lcbench.md`).** T1 tuning stop rule fired (SP worse than W1 on TabRepo in all 9 configs). Exploratory diagnostic suggested a gain on 6 LCBench landscapes (-25 %). Pre-registered confirmation on 24 fresh LCBench landscapes failed: SP +7.0 % vs W1, frozen +3.4 %, uniform +0.7 %, all gates FAIL (`meta/results/sp_lcbench.txt`). On half the landscapes W1 already finds the optimum at step 0. Line closed.

| Approach | Verdict |
|---|---|
| Similarity-weighted source prior (reactive weights) | Fails; diagnostic gain did not replicate |

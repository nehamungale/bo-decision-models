# Component 1: reactive acquisition-function choice — plan for approval (2026-10-24)

Status: DRAFT, nothing built. Goal: **better** optima (regret against number of evaluations), not faster. Latency is out
of scope until later. Context: [`notes.md`](notes.md) (literature by component, offline evaluator study, earlier pivots).

## 1. Question

At each BO iteration, can a selector that sees the campaign state and history choose the acquisition function (AF) so that
regret is lower than any fixed AF, and can a typed decision model do this better than a tuned numeric rule that sees the
**same** state? The second half is the missing control in LMABO (arXiv 2602.07904) and the point of the chapter.

## 2. Testbed

One engine: sequential BO on a discrete candidate pool. Every AF is scored on every pool point, the argmax is evaluated.
- **T1, LCBench final accuracy** (real objective): x = 7 hyperparameters, y = final validation balanced accuracy,
  2000 points per dataset, noise-free (one seed per config). Dev: the 8 datasets in `splits.py`. Confirmatory: the 5 held out.
- **T2, synthetic functions** for comparability with LMABO (Ackley, Levy, Hartmann6, Branin, Rosenbrock, Michalewicz, ...):
  5000 Sobol points per instance, so they run through the same pool engine. This approximates continuous optimisation of the AF
  and will be stated as such.
- T3 (later): excipient pool, after scalarising objectives. Out of scope here.

Budget: 2D+1 initial points (LMABO convention, 15 for D=7) then 50 iterations. Seeds: 20 per dataset (LMABO used 10); final n
set by a pilot power calculation on the static AFs, fixed before any selector is run.

## 3. Menu of AFs (six, interpretable, cheap on a pool)

LogEI, LogPI, UCB (beta = 2), Thompson sampling, posterior mean (pure exploit), posterior std (pure explore).
Dropped: KG/PES/MES/JES (need continuous optimisation, slow). GP: BoTorch SingleTaskGP, Matern-5/2 ARD, standardised y,
refit every step. Fixed seeds everywhere (`torch.manual_seed` per run; lesson from the earlier reproducibility bug).

## 4. State (identical for every selector; about 20 numbers plus history)

iteration and fraction of budget left; incumbent as a fraction of the observed range; iterations since last improvement and the
last improvements; GP lengthscales (min, max, mean, std), outputscale, noise; mean and max posterior std over the pool;
distance from the last point to its nearest earlier point; **history of (AF used, did it improve the incumbent)**, the field whose
removal hurt LMABO most; **agreement among the six AFs' proposals** (number of distinct argmaxes). For the typed model the same
numbers are rendered as text, with each AF described in one plain sentence as menu criteria.

## 5. Phases and gates

**Phase A: no model calls at all.**
1. Engine tests: determinism across processes, LogEI beats random, agreement with a reference BoTorch run on a toy problem.
2. **Headroom analysis.** (a) *Oracle selector*: at each step compute the six proposed points, look up their true y, take the best
   one; this peeks at y, so it is an upper bound on one-step selection, not a method. (b) *Spread among static AFs*: do the six AFs differ
   at all? (c) Mean number of distinct proposals per step.
   **Gate G**: continue only if the oracle selector beats the best static AF by a clear margin on most dev datasets AND the static AFs
   differ significantly on most datasets. Thresholds proposed: oracle gain >= 20 % relative AUC regret (large because it peeks) on
   >= 6/8 datasets; static spread significant (paired test, Holm) on >= 5/8. These numbers are my judgment, not derived; please
   challenge them. If G fails the component is closed for this testbed and we move to the next component.

**Phase B: numeric baselines** (same state).
- Static: the six AFs. Random AF per step. GP-Hedge. No-PASt-BO if cheap.
- **Tuned rule selector**: a small transparent rule on the state (stagnation count, budget fraction, ineffective-AF history,
  agreement), thresholds tuned by search on dev datasets with a tuning budget stated in advance and equal to the prompt-variant budget
  given to the model.
- **Learned numeric selector**: contextual bandit / gradient-boosted policy trained offline on simulated campaigns of other datasets
  (leave-one-dataset-out), reward = next-step improvement in normalised regret. The strong competitor and the proxy for a
  trained decision model.

**Phase C: typed decision model.** Choice over the six AFs; stand-ins tev1:0.8b and nimble (9B). Selection modes fixed in advance:
argmax; sample from the probabilities; gated (use argmax only if confidence >= tau, otherwise fall back to LogEI). Controls:
state shuffled across steps (does the choice depend on the state?), and the same prompt with fields removed (LMABO-style ablation),
only if the model beats the rule. Log every (state, probabilities, choice, improved) for a calibration study.

**Phase D (needs your decision): generative reference.** LMABO used Gemini 2.5 Flash, about 1 s and $0.01 per 50-iteration run, and
found Qwen3-8B significantly worse than GPT-4o-mini or gpt-oss-120B. Local qwen3:32b would take hours per condition; Ollama cloud
models or an API send data off the machine (the data is non-sensitive synthetic/benchmark numbers). Run on a reduced subset only.

## 6. Pre-registered hypotheses (fixed before Phase B/C data)

- **H1** Some adaptive selector (Hedge, rule, learned) beats the best static AF. Expected small effect; replicates the literature.
- **H2 (primary)** The typed model matches or beats the tuned rule on the same state. Predicted tie; reported with an equivalence margin
  of 5 % relative AUC regret, so a tie is a stated result and not just a non-significant one.
- **H3** The model's choices depend on the state (shuffled-state control is worse than the real state).
- **H4** The model's probabilities for "this AF improves the incumbent now" are better than chance and calibrated (ECE, reliability bins);
  the confidence field tracks reliability. Offline study as before.
- **H5** If the model beats the rule, each state field it relies on matters (ablation).

Primary metric: AUC of normalised simple regret over iterations (as in LMABO); secondary: final regret, regret at 25/50 %, per-dataset
win counts. Paired tests per dataset then Wilcoxon across datasets, Holm correction across the pre-registered family. Exploration
(dev datasets, seeds A) is kept strictly separate from the confirmatory run (held-out datasets, new seeds), which is run once.

## 7. Cost and environment

Estimated 30 to 60 s per campaign with a GP refit each step; 8 datasets x 20 seeds = 160 campaigns per condition, about 1.5 to 3 hours per
condition on one core, parallelisable. Needs torch + BoTorch, which exist in the repo's main environment and not in the scratch venv; pools
will be exported once from LCBench to `.npz`. The typed-model phase adds roughly 0.1 to 0.3 s per decision.

## 8. Risks

- The gate may close the component (no headroom). That is a result.
- tev1:0.8b is probably too small for the task (LMABO: 8B already worse); nimble is the fairer stand-in, and neither is Jev.
- Numeric states favour trained heads over zero-shot text models (our offline study); a null on H2 is the likely outcome.
- LCBench is noise-free and 7-dimensional, so AF-choice effects may be small; the synthetic functions add range.
- LMABO's numbers are a preprint, not replicated here.

## 9. Open decisions for the user

1. Approve or change the gate thresholds (section 5).
2. Allowed models for Phase D: local only, Ollama cloud, or an API with a spend cap?
3. Include the synthetic functions (T2) from the start, or LCBench only first?
4. Which Python environment to use for BoTorch (the repo's main one is assumed).

---

## Amendment 1 (2026-10-24, after Phase A): gate overridden, Phase B protocol fixed before running

**Deviation, recorded openly.** Gate G1 failed as pre-specified (5/8 datasets, needed 6/8); G2 passed (6/8). The user chose to
proceed to Phase B (decision made after seeing the Phase A result, so this is a post-hoc override, not a pre-registered path).
Reason: G1 is lenient by construction (the oracle peeks at y) and failed narrowly; cost of proceeding is compute only. The confirmatory
run on held-out datasets with new seeds remains the final judge.

**Seed blocks** (disjoint): tuning seeds 100-104; dev evaluation seeds 0-19 (already used by Phase A); confirmatory seeds 1000+ on the 5 held-out datasets.

**Comparators fixed now** (from Phase A): LogEI (the default); TS (best pooled static, mean AUC 0.1441 vs LogEI 0.1491, UCB 0.1490,
random-AF 0.1517); random-AF.

**Phase B selectors** (all see the same state; none sees y of unevaluated points):
1. *GP-Hedge*: cumulative gains G_i += standardised posterior mean (current GP) at AF i's previous proposal; P(i) proportional to exp(G_i), eta = 1.
2. *Tuned rule* (decision list): (a) last fraction b of the budget -> exploit AF; (b) first fraction a -> explore AF; (c) if win_stay and the last
   AF improved -> repeat it; (d) if iterations since last improvement >= k -> explore AF; (e) else default AF. Grid: explore in {UCB, TS, PosSTD},
   exploit in {PosMean, LogEI, LogPI}, default in {LogEI, LogPI, TS, UCB}, a, b in {0, .1, .2, .3}, k in {1, 2, 3, 5, 8}, win_stay in {yes, no}.
   Tuning budget: 100 random configurations x (8 dev datasets x 5 tuning seeds); criterion = mean over datasets of mean AUC.
   Tuned and evaluated on the same datasets but different seeds, which favours the rule over the model (conservative for H2).
   The typed model gets at most 10 prompt/wording variants evaluated the same way, a smaller tuning budget.
3. *Learned numeric*: per-AF gradient-boosted regressors (sklearn HistGradientBoosting, default settings, fixed random_state) predicting the one-step
   normalised gain max(0, y_proposal - incumbent)/(y_max - y_median) from the state, trained on Phase A logs of other datasets (leave-one-dataset-out),
   then argmax. L1: all Phase A logs, without AF-history features (those identify the logging policy in static runs). L2: random-AF logs only, with history
   features. Phase A logs are off-policy for the learned selectors; stated limitation.

**Primary tests** (family = four selectors; Holm correction): for each selector, per-dataset mean AUC over the 20 dev seeds compared with TS (best pooled
static) and with LogEI by paired Wilcoxon across the 8 datasets, plus per-dataset sign counts; also versus random-AF. H1 supported if any selector beats the
pooled best static after correction. H2 (typed model vs tuned rule) comes in Phase C.


---

## Amendment 2 (2026-10-25): deviations and exploratory additions made during Phases C and D

- Phase D ran before Phase C (the strongest reference first, to bound what is achievable); gpt-oss:120b-cloud, LMABO-style prompt, 10 seeds, 2 workers.
- Phase C gated mode degenerated into static LogEI (tau set on a different history regime). Reported as a design flaw, not rerun.
- Added after seeing results, so exploratory: the state-blind fixed-mixture control (weights fixed before it was run); extension of tev1-sample to seeds 10-19, with seeds 10-19 alone reported as the unbiased check.
- The shuffled-state control (H3) was replaced by the stronger fixed-mixture control. H5 (field ablations) was not run since no model beat the rule or static.
- The confirmatory run (5 held-out datasets, seeds 1000+) was not performed: no dev result justified a confirmatory test.


---

## Amendment 3 (2026-10-25): nimble (9B) as a better typed stand-in; protocol fixed before the run

Reason: knowledge probe (notes.md) showed tev1:0.8b misses basic BO/Pareto vocabulary (8/14) while nimble gets 14/14; the tev1 nulls partly reflect a weak stand-in.
- Same state text, same Choice instruction and criteria as tev1; dev datasets, seeds 0-9.
- Latency: 1.31 s per call, no concurrency speed-up (16 fresh calls at 1, 2, 4 and 8 threads: 1.31-1.37 s each), so 4000 calls take about 90 minutes.
- Modes: `sample` first (primary, the live mode for tev1); `argmax` and `gated_adaptive` only afterwards if time allows. gated_adaptive replaces the failed fixed-tau gate:
  follow the model's argmax if its confidence >= the median of this campaign's earlier confidences (first 5 steps always follow), else LogEI. It needs no cross-regime calibration.
- Comparators unchanged (TS, LogEI, tuned rule, random, learned_L2, tev1-sample) plus a state-blind fixed-mixture control using nimble-sample's own AF shares.
- Probe on 16 states: argmax LogPI 12, LogEI 2, UCB 1, PosSTD 1; probability SD across states up to 0.14, larger than tev1's.


---

## Amendment 4 (2026-10-28): two extra numeric baselines from the portfolio/bandit literature, fixed before running

Reason: the AF-choice list lacked a plain non-contextual bandit over AFs and a rule with an enforced minimum dwell time (suggested in a literature
summary pasted by the user). Both are cheap. Same 8 dev datasets, dev seeds 0-19, same state and engine, so they are directly comparable with the
existing static, random, hedge, rule and learned arms. The confirmatory run on held-out datasets is still not performed.

- **bandit_ts** (no tuning): Beta(1,1) Thompson sampling over the six AFs; reward 1 if the evaluated point improved the incumbent, else 0; no discount; no state.
- **rule_dwell**: the tuned rule of Amendment 1 with its parameters unchanged, wrapped so that a chosen AF is held for at least D steps before the rule may change it.
  D in {3, 5} compared with D = 1 (the existing rule, tuning score 0.13978) on tuning seeds 100-104 over the 8 dev datasets; the best D is run on seeds 0-19.
- **sched_EI_PI** (extra, literature default, no tuning): LogEI for the first 25 % of the budget, LogPI after (Benjamins et al. 2022).
- Primary tests: each of the three vs TS (best pooled static) and vs LogEI, paired Wilcoxon across the 8 per-dataset mean AUCs, Holm over the six tests.
  Prediction (written before the run): null, as for hedge, rule and learned. An improvement counts only if it survives Holm.

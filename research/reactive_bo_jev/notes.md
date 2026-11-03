# Reactive BO with a typed decision model (Jev stand-in) — working notes

Started 2026-10-24. Follows on from [`../clm_jev_bo_exploration/notes.md`](../clm_jev_bo_exploration/notes.md),
which found that using decision models to re-rank BO candidates adds nothing
(5 backends all at or below baseline; trust-capital r≈0).

Target: a thesis chapter. Eventual deployment: a real lab. Simulator first.

## Definitions (agreed)

- **Reactive BO** = real-time, event-driven, asynchronous BO: decisions are made
  *while experiments run*, triggered by events (new partial reading, instrument
  freeing up, stagnation, failure), under a latency budget.
- Not found as an established term in the literature. Closest: multi-stage BO with
  intermediate proxy measurements (Torresi & Friederich, arXiv 2512.15483) and
  freeze-thaw / DyHPO / ifBO for early stopping.

## Why a typed decision model (Jev) — and what must be tested, not assumed

Claimed advantages: (1) 70–500 ms latency vs ~327 s for a thinking generative LLM,
(2) calibrated probability + confidence returned directly, no thinking-token cost,
(3) schema-constrained output.

Caveats:
- Latency only matters when decision window ≈ experiment duration. Between
  multi-hour experiments, a slow LLM is fine. The key variable is
  **decision latency ÷ experiment duration**; sweep it.
- Confidence is only useful if calibrated on *these* decisions. tev1's confidence
  showed no outcome signal in the earlier exploration. Test with reliability curves.
- Fair baseline: a **non-thinking small LLM with constrained decoding and token
  log-probs**, which also yields a distribution cheaply.
- Jev's RLCD calibration cannot be reproduced by a stand-in (tev1 / CLM-8B).
  Chapter must not claim Jev results from stand-in runs.
- Simulated state is numeric only, so rules can match the model. Jev's real edge is
  unstructured text (operator notes, error messages). Add a clearly-labelled
  synthetic text channel later.

## Step 0 findings: current simulator can't support this

`excipient_oracle*.py`, `sdl_adaptive/simulator.py`: no time, no durations, no
parallel instruments, no partial observations, noise frozen into the pool (no
re-measurement), no failures, no text. Only "time" is step-based
`controller_interval`. Reusable: EGBO candidate pool, seeded campaign loop,
typed-call plumbing, comparison harness, `power_calc.py`. Prior art in repo:
`sdl_adaptive` controllers A–D (step-interval LLM routing).

## Existing simulators surveyed

| Source | Offers | Gap |
|---|---|---|
| Syne Tune simulator backend | simulated clock, event queue, N workers, charged decision delay, tabular benchmarks | ML HPO, not lab chemistry |
| LCBench / YAHPO-Gym / HPOBench / JAHS | real learning curves; freeze-thaw, DyHPO, ifBO, Hyperband baselines | not chemistry |
| Olympus, Atlas | chemistry/materials emulators, SDL planner | instant queries, no time/partials |
| Async delayed-feedback SDL sim (arXiv 2312.03466) | multi-stage SDL, 177 coatings experiments, delayed feedback | no mid-run reactions |
| MSBO (arXiv 2512.15483) | intermediate proxy measurements | method paper, not a simulator |

No benchmark found measuring decision latency for LLM-guided BO in SDLs — a
possible gap for the chapter.

## Decisions so far

- Landscape: **domain-agnostic**. Real lab timing/readout details not needed now;
  they only choose the highlighted point on a latency-ratio sweep, and can be added
  later as parameters.
- Base simulator on **LCBench** (via Syne Tune blackbox repo) with a Syne Tune–style
  clock/event queue (own implementation).
- Stand-in backend: tev1 via Ollama `/v1/systemone`, with injected Jev-like latency
  (sweep ~70–500 ms).

## Feasibility check (2026-10-24) — `feasibility/lcbench_probe.py`

- `syne-tune` 0.16.0 (Apache-2.0) + `huggingface_hub pandas pyarrow fastparquet scipy`
  installs and loads on this arm64 Mac, Python 3.12.
- LCBench: 35 datasets × 2000 configs × 50 epochs; 7 hyperparameters; 7 recorded
  quantities (val/test accuracy, balanced accuracy, cross-entropy, cumulative time).
  **Tabular**: only the stored configs can be queried (discrete pool).
- APSFailure: rank correlation of epoch-e accuracy with final accuracy =
  0.36 (e=1), 0.47 (5), 0.60 (10), 0.83 (25). Early-stop is a real decision.
- Timing: ~3.8 s/epoch, ~188 s/run median. Jev-class 0.07–0.5 s ≈ 2–13 % of an epoch;
  a 327 s generative call ≈ 86 epochs.
- Limits: one seed per config (no re-measurement), timings are one machine's,
  curves smooth (no failures/drift), only APSFailure inspected, LCBench data
  licence not checked, download size not measured.

## Dataset survey and splits (2026-10-24) — `feasibility/lcbench_survey.py`, `splits.py`

All 35 datasets: no NaNs, cumulative time strictly increasing. Across datasets, rank
correlation of partial with final accuracy has mean 0.65 / 0.83 / 0.89 / 0.97 at epochs
1 / 5 / 10 / 25 (APSFailure is among the hardest at 0.36 / 0.47 / 0.60 / 0.83). Median run
time 123 s (range 45 s - 1885 s). Share of the true top-10 % that an epoch-10 top-30 % cut
would discard: median 6 %, mean 10 %, max 48 % (numerai28.6, kc1, christine are the traps).

Splits fixed by a rule on curve statistics only, before any method ran (`splits.py`):
sort by rho@10; dev = ranks 2, 8, 14, ...; confirmatory = ranks 5, 11, 17, ... (5 datasets,
the rule yields 5 not 6). Dev set skews easy; only numerai28.6 is a hard trap case. Revisit
before the confirmatory run if the hard cases matter for the hypotheses.

Data quirk: `num_layers` is stored as strings and 1 layer appears as `'True'`; mapped back to 1.

## Simulator core (2026-10-24) — `sim/`, `validate.py`, `validate_latency.py`

`sim/engine.py` is a discrete-event simulator. Assumptions, each needing defence in the chapter:
- Time unit = the dataset's median epoch time, so latency is a ratio, not a number of seconds.
- Training does not pause while a decision is computed. A STOP takes effect `latency` after
  the report (epochs finished meanwhile are wasted); CONTINUE is free. Proposing a new
  configuration costs `propose_latency` (default 0) with the worker idle.
- Score = best FINAL metric among runs that completed all 50 epochs (deployable quality).
  Stopped runs never count. Reported as normalised regret: 0 = pool optimum, 1 = pool median.
- Budget `T_MAX` = 300 epoch units (6 full runs per worker). Float64 clock (float32 caused
  0.0007 drift, caught by the worker-time accounting assertion).
- Surrogate in `gp_rung`: sklearn GP-EI on the metric at epoch 10, over a discrete pool.
  Used sklearn rather than BoTorch to avoid pulling torch into this environment and the
  seeding problems recorded in the clm exploration. Revisit if the chapter needs BoTorch.

Validation (20 seeds x 6 dev datasets, mean normalised regret, lower is better;
`validate_output.txt`): random/no-stop K=1 0.598; K=4 0.379; K=8 0.292; random/ASHA K=4
0.171; GP-rung/ASHA K=4 0.133. More workers help; early stopping beats finishing every run;
the GP proposer beats random on 4 of 6 datasets and is flat on the other two. These match
known behaviour, so the core is trusted enough to add a decision layer.

Latency check (`validate_latency_output.txt`; ASHA, K=4, mean regret by stop-decision latency
in epochs): 0 -> 0.171, 0.1 -> 0.173, 1 -> 0.184, 5 -> 0.218, 20 -> 0.336, 86 -> 0.362,
no stopping -> 0.379. Early stopping is worth almost nothing once latency reaches ~20
epochs, and 86 epochs (a 327 s call at 3.8 s/epoch) is almost the same as not stopping.
Below ~1 epoch there is almost no loss. Implication: the speed argument for Jev only
needs "well under an epoch"; the difference between 70 ms and 500 ms barely matters unless
epochs are seconds long. The chapter's speed claim is "fast enough" versus "far too slow".

## Evaluator framing and new sources (2026-10-24)

Sources read: LangChain/LangSmith agent-evals post; Langfuse evals post; Deng et al.,
"Jev for Scientific Decisions" (arXiv 2609.24965). The medical-RAG Medium post returned 403.

What they support (all vendor-adjacent or small studies; none independently verifies calibration):
- Jev as a **judge**: many typed questions (Noul / Choice / Score) on one state, answered in
  parallel and in isolation; cheap enough to run on every event. LangChain: 0.44 s and
  $0.00035/call, lowest score variance of the judges tested, but only 5 runs, 1 agent.
  Langfuse: vendor latency/cost claims unmeasured there; 6,003 rubric checks, 91.5 % agreement
  with a frontier model's verdict.
- Deng et al.: median latency 0.335 s, p95 0.442 s, ~$6e-5 per response; matched five other
  configurations at full semantic correctness on 20 curated Choices. Its design principle is a
  **code/model division**: code does counting and arithmetic, the model answers atomic semantic
  questions. Wrong selections can change downstream numbers while the final label stays right.
- Caveats from Langfuse: cannot abstain (forced answer), no rationale, context rot on long or
  irrelevant state, context limit unclear (32K vs 64K), waitlist access.

Design consequences:
1. Use the typed model as an **evaluator of BO state**, not a ranker: at each event, code computes
   numbers and a text summary; the model answers several atomic questions (e.g. "has this curve
   plateaued", "is it unstable", "will it finish in the top 10 %", a Score for promise); a small
   deterministic combiner turns the probabilities into CONTINUE / STOP.
2. **Offline evaluator study before the closed loop.** Collect (state -> model probabilities) over
   many (run, epoch) pairs and measure calibration and discrimination against the true outcome
   (does the config finish in the top 10 %), versus code-only features, and the combination.
   Key question: does the model add information beyond cheap code features? This is also the
   calibration study (plan step 5), moved first. It needs no campaign simulation.
3. Latency of a multi-question call is an assumption (parallel => ~one call). Measure how the
   stand-in's latency scales with the number of questions before charging it in the simulator.
4. Because a question answerable from the numbers can be answered by code, the rules ablation must
   use the same questions as code-computed features and the same combiner.

## Offline evaluator study — pre-registered design (written before any data collected, 2026-10-24)

Backend: tev1:0.8b via Ollama `/v1/systemone` (stand-in; NOT Jev). API quirks found: Noul takes
`instructions` (the statement) and returns only a probability (no confidence); Score takes
`criteria` as an array of 2-10 levels and returns probabilities + confidence; Choice takes
`criteria` as a dict. (An early hand test seemed to show bundling shifts a Noul answer, 0.814 alone vs
0.803 bundled; that was my error, the two calls used different question wording. Re-tested below: no
interaction.)

Unit: a state = (dataset, config, epoch e in {5, 10, 25}). 250 configs per dev dataset sampled
uniformly (seeded), same configs at every epoch. 8 dev datasets. Text = curve so far + 30 random
peers' statistics at the same epoch + the best final accuracy among 10 of those peers treated as
fully trained. Outcome y = config finishes in the top 10 % of all 2000 by final accuracy.

Questions (one call per state): Noul: will_finish_top10 (the target, asked directly),
above_most_peers, plateaued, improving_fast, unstable, will_beat_best_so_far. Score: promise (5
levels). Choice: action in {continue, stop}.

Code features computed from the same information: value at e, percentile among peers, gap to
peer median, gap to best completed, last-3-epoch slope, mean slope, noise of increments, largest
drop, value change since epoch e/2.

Models: standardised logistic regression, leave-one-dataset-out across the 8 dev datasets.
Feature sets: A code only; B model answers only; C code + answers; D will_finish_top10 alone;
E peer percentile alone.

Hypotheses (stated before looking):
- H1 (discrimination): D has AUROC > 0.5 with a dataset-level paired test. Expected to hold weakly.
- H2 (incremental value): C beats A in LODO AUROC, judged by the per-dataset AUROC difference
  (mean, sign count out of 8, Wilcoxon). Expected NULL: the questions are answerable from the
  numbers, so code should already capture it. A null is a valid result.
- H3 (calibration): the raw will_finish_top10 probability is miscalibrated (ECE > 0.05, reported
  with reliability bins); the reported Score confidence does not track correctness.
- H4 (latency scaling): per-call latency of the bundled 8-question call vs 1, 2, 4 questions.
- H5 (interaction): will_finish_top10 asked alone vs inside the 8-question bundle differ by more
  than 0.02 on average (measured on 200 states).
Metrics: AUROC, Brier, ECE (10 bins), per epoch and pooled. No hypothesis is tuned after seeing results.

### Collection slowdown, diagnosed and fixed (2026-10-24)

Per-call latency drifted from 0.31 s to ~0.9 s (throughput 1.1 states/s). Cause: memory
pressure. Swap was 16.6 of 17.4 GB; two idle servers left running for ~7 days in tmux from the
CLM-8B work (`vllm serve mlx-community/Qwen3-8B-4bit` on :8090, `clm-serve` on :8700, ~4.5 GB
resident plus GPU-wired memory) competed with Ollama's tev1 runner. Stopped both with SIGTERM
(tmux sessions left in place). Afterwards: swap 9.3 GB, latency 0.25 s, 3.8 states/s.
**Latencies in the first ~600 collected states are inflated and must not be used as the model's
speed**; use `latency_scaling.py` on an idle machine instead. To restore the servers:
`vllm serve mlx-community/Qwen3-8B-4bit --served-model-name qwen3-8b --runner pooling
--max-model-len 2048 --port 8090` (tmux `vllm-serve`) and `KMP_DUPLICATE_LIB_OK=TRUE clm-serve`
from `~/.venvs/clm-serve` (tmux `clm-serve`).

## Offline evaluator study — results (tev1:0.8b, 6000 states, 8 dev datasets; `evaluator/data/analysis_output.txt`)

Base rate of "finishes in top 10 %": 0.105. All figures are mean per-dataset AUROC, leave-one-dataset-out.
- Pooled: A code 0.871; B model answers 0.843; C code+model 0.870; D raw will_finish_top10 0.77;
  E peer percentile alone 0.860. By epoch (A / C): epoch 5 0.812 / 0.815; 10 0.834 / 0.829; 25 0.941 / 0.941.
- **H1 supported.** D beats 0.5 on all 8 datasets (0.59 to 0.92), Wilcoxon p = 0.0078 (the minimum possible with 8).
  Weakest: numerai28.6 0.59, christine 0.61, kc1 0.62 (the hard early-stopping datasets).
- **H2 null, as predicted.** C minus A = -0.0004 (C>A on 4/8 datasets, p = 0.84); no epoch differs. One code
  feature (peer percentile) already gives 0.860 of A's 0.871. The model adds no information beyond cheap code features.
- **H3 supported, severely.** Mean predicted probability 0.83 vs observed base rate 0.105; ECE 0.725; Brier 0.615
  vs 0.094 for always predicting the base rate. Even the p >= 0.9 bin is right only 37 % of the time.
  The probabilities carry rank information only. Confidence fields point the wrong way: Spearman(confidence,
  -|error|) = -0.09 (promise) and -0.29 (action). This error metric is crude and confounded with the
  inflated output level, so read it as "no evidence confidence flags reliability", not as proof of anti-calibration.
- State sensitivity (Spearman with matching code feature): promise vs percentile +0.61, p_stop -0.68,
  above_most_peers +0.50, improving_fast vs slope +0.52, will_beat_best +0.45 (all right sign). plateaued vs
  slope -0.26 (right sign, weak). **unstable vs noise -0.25 is the wrong sign**, unexplained.
- Confound to state in the chapter: the state text spells out "better than X % of them", the very quantity in
  feature E, so the model may mostly be reading that sentence (p_stop correlates -0.68 with it). A C-over-A
  gain was unlikely by construction; the harder test is a state without that sentence, or text-only information.
- Latency in this file (mean 0.31 s, median 0.25 s, p95 0.88 s) includes the early contended period; not
  the model's speed. H4/H5 (`latency_scaling.py`) still to run on an idle machine.

Consequence for the plan: on purely numeric LCBench states, a closed-loop policy built from tev1's answers cannot
be expected to beat a policy built from the code features, and its raw probabilities cannot be used as
probabilities without recalibration (which the same code features would also receive). The pre-registered logic
said to stop here for numeric states.

### H4 and H5 results (idle machine, tev1:0.8b, 200 states; `evaluator/data/latency_scaling_output.txt`)

- **H4: latency grows roughly linearly with the number of questions**, not flat: median (p95) seconds for
  1 / 2 / 4 / 8 questions = 0.056 (0.081) / 0.080 (0.119) / 0.129 (0.166) / 0.257 (0.290). That is ~0.03 s base
  plus ~0.028 s per extra question. The vendor claim that extra questions add little latency does NOT hold for
  this local stand-in; whether it holds for Jev (a hosted service, claimed parallel) is untested. The simulator
  should charge `base + n_questions * per_question` rather than one flat call.
- **H5: no bundle interaction.** A question asked alone and inside the 8-question bundle gives identical
  answers (mean |diff| 0.0000 over 200 states for will_finish_top10, and over 60 states for positions 1, 3, 5, 6),
  and repeated identical calls are deterministic. Not rejected: this is the stand-in; Jev's isolation is
  claimed by the vendor, not tested here.
- The 0.25 s for 8 questions matches the quiet-machine collection speed; the 0.9 s seen mid-collection was
  contention (see above).

## Pivot (2026-10-24): from "ask the model what numbers already say" to "what can it do that numbers cannot"

Decision after the offline study: nimble rerun DROPPED for now. Asking a typed model about numeric curve
states duplicates normal BO (study H2 null). Target role is now (1) interpreting events whose cause is not in the
numbers (failed run vs bad config, drift, hidden constraints, operator/instrument text) and (2) campaign-level
control (explore / exploit / switch algorithm), in the manner of `sdl_adaptive`, in the async latency-charged
simulator. No real lab notes or logs are available (user), so any text is synthetic and must be labelled so.

Prior art in this repo (`sdl_adaptive/COMPREHENSIVE_LOG.md`): hand-coded EGBO beat every LLM controller (A/B/C/C-evo/D)
and the rule router in every sampled row (typically by 0.05-0.13 AUC); approach_d tracked rule_router within
0.02-0.05, i.e. no routing value over its own fallback; isolated judgment calls (lengthscale mode) were WORSE
than the default and worse with the bigger model; rule thresholds were "initial guesses". Those controllers were
slow generative models on numeric-only state, with no latency charged. Implication: on numeric campaign signals
expect parity at best between a typed model and a tuned rule; value, if any, must come from information the rule
cannot see (text context) or from a policy trained on outcomes, not zero-shot prompting.

## Track 1 design: event interpretation (2026-10-24, before any building)

> STATUS: on hold. User corrected the framing the same day: reactive BO = adding or modifying a BO component
> during the run so the optimum is found faster, not fault diagnosis. Fault interpretation is at most one
> instance. See "Reframe" below once written.

Terminology: a bad RUN is an invalid measurement (crash, preemption, drift); a bad CONFIGURATION is a valid
measurement of a poor or infeasible config. The model's job is to tell them apart when the numbers cannot.

Causes (ground truth known because the simulator injects them on top of real LCBench curves):
1. transient_failure: infrastructure fault (preempted, node lost, timeout). Invalid. Action: re-run the same config.
2. config_infeasible: caused by the config itself (out-of-memory for a large config, NaN from a bad learning rate).
   Re-running fails again. Action: mark the neighbourhood infeasible.
3. genuine_poor: ran fine, result is just low. Valid. Action: record normally, learn from it.
(4. drift / contamination: valid run, shifted metric. Later.)

What guides the model: only the state text (numbers + note) and the question wording (instructions and explicit
true/false criteria). No GP, no learning during the campaign. Its judgment is its pretraining/fine-tuning prior plus
the literal cues in the text. Questions and criteria wording are frozen on a dev set of phrasings before the test set
of unseen phrasings is touched. What guides the EFFECT of its judgment is deterministic code: a fixed mapping from
probabilities to bounded actions, with a confidence gate and a safe default. Fine-tuning a head on simulator-labelled
events (as with CLM-8B) is a later option.

Making numbers ambiguous on purpose: a `separability` knob controls how different crash curves are from real
divergence curves. At 0 they are statistically matched and only the note can disambiguate; at 1 numbers suffice.
Notes carry a `reliability` knob: informative / ambiguous / misleading / absent. Hard cases where keywords mislead
(e.g. "out of memory" is a config property if the config is huge, a transient fault otherwise) are included.

Good vs bad, three levels:
- Per event: accuracy, macro-F1, calibration (ECE, Brier), cost-weighted loss against the generator's true cause.
- Per campaign: normalised regret at the time budget; fraction of "regret recovered" = (B0 - X) / (B0 - oracle);
  wasted instrument-time fraction; good configs wrongly excluded.
- Cost: latency and calls per event, charged to the simulated clock.

Baseline ladder: B0 treat every anomaly as a normal observation (floor); B1 always re-run once; B2 numeric-only
classifier plus outlier-robust GP (what numbers can do); B3 hand-written keyword/regex rules on the notes (the real
competitor, frozen before any model output is seen); B4 tev1 typed with deterministic mapping and confidence gate;
B5 generative LLM with real latency charged; oracle: true cause given (upper bound).

## Literature by BO component (searched 2026-10-24; search snippets and fetched pages only, replication not checked)

| Component | Key evidence | Adaptive during a run? | Numeric competitor |
|---|---|---|---|
| Acquisition choice | LMABO (arXiv 2602.07904, ICLR 2026): zero-shot LLM picks 1 of 12 AFs from a state summary; 50 problems; -9.7 % AUC vs best static, -16.6 % vs best adaptive portfolio. Ablations: dropping budget, GP lengthscales, distance, or the "avoid AFs that failed" instruction each hurts (the instruction most). Qwen3-8B significantly worse than GPT-4o-mini / gpt-oss-120B. Never tested a rule selector on the same state. GP-Hedge (arXiv 1009.5419), No-PASt-BO (1908.00361) | yes | GP-Hedge, No-PASt-BO, static LogEI |
| Search region | TuRBO: success/failure counters grow or shrink the trust region; bandit allocation across regions | yes (rule) | the counters |
| Kernel / surrogate | rank-based online kernel selection (2206.01409), GP ensemble + Thompson sampling (2205.14090), CAKE/BAKER LLM-evolved kernels (2509.17998, preprint) | yes | likelihood-based selection |
| Learned acquisition | MetaBO (1904.02642): RL meta-trained AF beats EI esp. early; needs related source tasks | trained offline | EI |
| Batch strategy | local penalization (González 2015), improved/asynchronous LP (1901.10452), Kriging believer / constant liar theory (2605.18819) | no work on adapting batch size or strategy during a run found | LP, KB |
| Noise model | homoscedastic assumption can mislead BO; heteroscedastic GP + noise-penalised EI helps on toy/real data; needs replicates, which costs sample efficiency | little | heteroscedastic GP |
| Input warping | Snoek 2014 Beta-CDF warping (1402.0929); Kumaraswamy warping used in PFNs4BO | learned automatically via marginal likelihood | the same |
| Priors | piBO (2204.11051) multiplies AF by a user prior with decaying weight, regret guarantees, up to 12.5x speed-up on a deep learning task; BOPrO (2006.14608); Hvarfner 2024 (2402.02229) dimension-scaled lengthscale prior makes vanilla BO strong in high dimensions | decays automatically | the same |
| Early termination | ASHA, freeze-thaw, ifBO (see earlier) | yes | ASHA |

| LLM-derived priors | LLAMBO (2402.03921, ICLR 2024): zero-shot warm start, surrogate and candidate sampling; gains mainly early. Independent reproduction (2511.18891): warm start improves early regret and variance; LLM surrogate weaker than GP/SMAC as a single-task regressor; sampler better than TPE/random; 8B-27B backbones unstable or invalid. LABO (2605.22054), LILO (2510.17671, natural-language feedback), LGBO (shifts surrogate mean to keep LLM guidance through the loop) | mostly at initialisation; LGBO and CAKE persist through the loop | metadata warm-starts, uniform init |
| Output transformation | HEBO (2012.03826): Box-Cox / Yeo-Johnson output plus Kumaraswamy input warping, large ablation gains, refit each step. mlr3mbo docs: log-transform of heavy right-tailed targets improves GP fit. Chowdhury & Gopalan 2019: heavy-tailed noise theory | refit automatically each step | the same |

No source found that measures an output transform's effect on regret in isolation. Repo history: llm_bo warm-start improved sample efficiency but hurt final hypervolume (-2.5 % to -12.8 %), consistent with LLM priors helping early and narrowing the search.

Reading: priors, warping, output transforms and lengthscale scaling have mostly been solved by better static or self-decaying defaults, leaving little for a controller. Real regime-dependence, with bandit/portfolio literature, exists for acquisition, search region and kernel. Adapting batch strategy during a run appears under-studied, which matters for asynchronous SDLs.

## Acquisition choice, Phase A result (2026-10-24; `acq/`, `acq/results/phaseA_analysis.txt`)

Engine checks passed (identical trajectories across processes; LogEI argmax equals BoTorch optimize_acqf_discrete; BO beats
random search, e.g. albert AUC 0.054 vs 0.324). GP = BoTorch 0.18 default (ARD RBF, dimension-scaled priors), not Matern-5/2.
1280 campaigns: 6 static AFs, random AF, one-step oracle x 8 dev datasets x 20 seeds, 15 initial points + 50 steps.

Mean AUC of normalised regret (lower better): best static differs by dataset (PosSTD, PosMean, TS, UCB, UCB, PosMean, TS, PosMean);
LogEI is best on none. Best-vs-worst static spread 9 % (kc1) to 244 % (vehicle).
- **G2 PASS**: statics differ significantly (Friedman, Holm) on 6/8 datasets (not on numerai28.6, kc1).
- **G1 FAIL as pre-specified**: one-step oracle gain over best static >= 20 % on 5/8 datasets (needed 6/8):
  numerai 26.5 %, KDD 35.2 %, albert 38.0 %, helena 40.1 %, vehicle 67.1 %, sylvine 20.0 % (bootstrap CI -25 % to 45 %,
  counted as a miss by float rounding), kc1 19.1 %, christine 8.5 % (CI -5 % to 22 %).
- Random-AF per step is close to the best static on several datasets (KDD -1.5 %, kc1 +5.6 %, christine +17 %), so a useful
  selector must beat random-AF, not only LogEI.
- Mean 4.4 of 6 distinct proposals per step; all six never agree, so the agreement feature is informative only through its range.
- One-step-oracle choice shares: PosMean 28 %, TS 26 %, LogPI 18 %, LogEI 17 %, UCB 7 %, PosSTD 4 %.

How to read the gate: both headroom measures are optimistic (the oracle peeks at true y; a per-campaign hindsight-best static
picks the minimum of six noisy AUCs and so is biased low even if all AFs were equal). A narrow fail of an optimistic gate is weak
evidence against headroom, not strong. Thresholds were my judgment, not derived.

## Acquisition choice, Phase B result: no numeric selector beats the best static AF (2026-10-24; `acq/results/phaseB_analysis.txt`)

Dev datasets, dev seeds 0-19, 640 new campaigns. Mean AUC of normalised regret over the 8 datasets (lower is better): TS 0.1441 (best static),
learned_L2 0.1444, learned_L1 0.1482, rule 0.1488, LogEI 0.1491, random-AF 0.1517, GP-Hedge 0.1603; one-step oracle 0.0975 (peeks, not a method).
Pre-registered family (each selector vs TS, paired Wilcoxon across 8 dataset means, Holm over four): none better than TS; relative AUC change
+14.8 % (hedge), +7.5 % (rule), +4.5 % (L1), +8.2 % (L2); Holm p = 1.0 for all. None beats LogEI or random-AF either. **H1 not supported.**
- Tuned rule: tuning-seed score 0.1398 (best of 100) fell to 0.1488 on dev seeds, equal to LogEI: winner's curse, tuning gains did not transfer across seeds.
  Chosen rule uses UCB 65 % of the time.
- GP-Hedge is worst (62 % PosMean): posterior-mean rewards drift it to exploitation. This matches LMABO, where GP-Hedge (RP 1.45) was also worse than static EI (1.34).
- learned_L2 (random-AF logs, with history) is a statistical tie with TS and better on 6/8 datasets but with mean change +8.2 % driven by the datasets with tiny AUC;
  nothing individually significant. Relative changes are dominated by low-AUC datasets (vehicle, helena).
- Knowing the best static AF per dataset would give mean AUC about 0.127 (12 % below TS), so the signal "which AF suits this dataset" exists but is not recovered online
  within 50 steps by these selectors.
Power note: with 8 datasets the smallest possible two-sided Wilcoxon p is 0.0078; per-seed tests (20 seeds) rarely detect differences of a few percent.
Implication: the numeric half of the ladder is flat, so a typed model equal to the rule would equal static. The informative next experiment is the strongest reference
selector (LMABO-style large LLM), which bounds what is achievable, before spending effort on small typed stand-ins.

### Phase D feasibility test, gpt-oss:120b-cloud via Ollama (2026-10-24; `acq/llm_selector.py`)

LMABO-style prompt: rendered state (iteration/budget, incumbent z, stagnation, last-observation surprise, GP lengthscales/noise, predictive std, distance,
number of distinct proposals, per-AF usage and improvement history), the six-AF menu with one-sentence glosses, a system instruction to avoid AFs that
repeatedly failed to improve the incumbent (LMABO's most important ablation), JSON-only answer, think="low", temperature 0, UCB fallback on a malformed answer.
Measured: 0.5-0.8 s per call (median about 0.6 s); 22/22 answers parsed. Sequential calls never failed. HTTP 429 (too many requests) appeared with 4 or more
concurrent threads; 2-3 threads worked but gave no speed-up (about 0.75 s per call overall), so throughput is about 1.3-1.6 calls/s regardless. NOT deterministic
at temperature 0: the same state gave 2,2,2,3,2 (UCB,UCB,UCB,TS,UCB) in five calls. Choices on the sampled states were mostly UCB (about 14 of 17), some TS.
Estimate: 4000 decisions (8 datasets x 10 seeds x 50 steps) is about 40-60 minutes, much less than the first guess of a few hours (which assumed 3 s/call).
Data sent off the machine: rendered numeric benchmark states only.

## Acquisition choice, Phase D result: a 120B LLM selector is statistically indistinguishable from random AF choice (2026-10-25; `acq/results/phaseD_analysis.txt`)

gpt-oss:120b-cloud, LMABO-style prompt, think=low, temperature 0, 8 dev datasets x seeds 0-9, 4000 decisions, 0 malformed or failed calls, median 1.13 s per call (2 workers).
Mean AUC over datasets on these 10 seeds (lower better): TS 0.1358, learned_L2 0.1358, rule 0.1397, random-AF 0.1451, UCB 0.1455, LLM 0.1457, LogEI 0.1500.
- Primary: LLM vs TS +4.0 % (better on 3/8 datasets, p = 0.547); LLM vs tuned rule +1.5 % (p = 0.25; Holm 0.50). Not supported.
- Bootstrap 95 % CIs of the relative AUC difference (resampling datasets and seeds): vs TS +7.3 % [-13 %, +34 %]; vs LogEI -2.9 % [-18 %, +10 %]; vs random-AF +0.4 % [-15 %, +13 %];
  vs rule +4.3 % [-10 %, +21 %]. The run cannot exclude effects of about 10 % in either direction (LMABO reported -9.7 % vs best static), so this is "no evidence of benefit", not
  "proof of no benefit".
- Behaviour: the LLM almost never repeats an AF after a non-improving step (P = 0.01, n = 3683) and repeats after an improving step 39 % of the time (n = 237); 97 % of
  consecutive steps switch AF. It is executing the prompt's "avoid AFs that failed" instruction as near round-robin rotation with weak win-stay. AF shares are close to uniform
  (LogEI 17, LogPI 6, UCB 19, TS 24, PosMean 19, PosSTD 15 %); mild budget response (PosMean 18 -> 24 %, LogEI 19 -> 14 % from early to late). My earlier prediction from 17 states
  that it would be UCB-heavy was wrong.
- Overall for this component on this testbed: no selector (GP-Hedge, tuned rule, two learned numeric, 120B LLM) beats the best fixed AF, and the LLM is no better than random AF choice.
  Informative but unrepeated caveats: 7 input dimensions, noise-free LCBench pools, 10 seeds for the LLM, a prompt borrowed from LMABO and not tuned, tev1/Jev not yet tested.

## Acquisition choice, Phase C: tev1 typed selector (2026-10-25; `acq/results/phaseC_analysis.txt`, `phaseC20_analysis.txt`, `phaseC_fresh_analysis.txt`)

Setup fixed before the dev run: Choice over the six AFs, state text without the menu (menu given as criteria), instruction to avoid AFs that failed, gate tau = 0.1359 from a
16-campaign pilot (tuning seeds 100-101), modes argmax / sample / gated, dev seeds 0-9, 12000 calls, 0 fallbacks, median 0.37 s per call.
- **argmax mode picked LogPI on 100 % of steps**: identical to static LogPI. **gated mode picked LogEI on 99.8 % of steps**: identical to static LogEI. Design flaw: tau was set
  on pilot states with a mixed AF history, but the model's confidence depends on that history; in gated mode (history all LogEI) mean confidence was 0.100 < tau. Gated never switched on.
- Only **sample** mode varies; its AF shares (LogEI .07, LogPI .35, UCB .16, TS .17, PosMean .05, PosSTD .21) equal the model's mean probabilities. The model assigns LogEI the lowest probability and
  LogPI the highest, regardless of dataset.
- **H4 (calibration) fails**: AUROC of p_a for "a's proposal beats the incumbent" = 0.483 (chance); PosSTD 0.33 (below chance: mean p .175, observed .03); confidence vs chosen-AF improvement
  spearman -0.027. The probabilities carry no information about which AF will help.
- H2 (vs tuned rule), seeds 0-9: no mode shown equivalent within +/-5 %; 95 % CIs span about 30 points. Ranking by mean AUC put sample first (0.1297, TS 0.1358, rule 0.1397) but the pre-registered
  mean-of-per-dataset-ratios summary was +1.7 % vs TS (the two summaries disagree in sign because low-AUC datasets dominate ratios and high-AUC datasets dominate mean AUC).
- State-blind control (post hoc, weights fixed beforehand): i.i.d. draws from those shares scored 0.1454 on seeds 0-9 (tev1-sample 0.1297; lower on 6/8 datasets, p = 0.195, kc1 alone -0.069).
- Post-hoc extension (flagged): sample mode run on seeds 10-19 because it looked best on 0-9. 20 seeds: tev1-sample 0.1391, TS 0.1441, learned_L2 0.1444, mixture 0.1488, rule 0.1488,
  LogEI 0.1491, random 0.1517; vs TS -3.5 % (p = 0.46, CI [-17 %, +9 %]), vs rule -6.5 % (p = 0.15). **On the fresh seeds 10-19 alone (the unbiased check)**: tev1-sample 0.1484, LogEI 0.1482,
  TS 0.1524, mixture 0.1522, rule 0.1579, random 0.1584; vs LogEI +0.2 %, vs TS -2.6 % (p = 0.38, CI [-22 %, +15 %]), vs mixture -2.5 % (p = 1.0). The seed-0-9 lead was largely selection
  of the best-looking mode plus seed luck. The same condition's rank moves between seed blocks (LogEI best on 10-19, TS best on 0-9).
- Conclusion: tev1 (0.8B, stand-in for Jev) gives no evidence of beating a fixed AF, the rule, or a state-blind mixture; its probabilities are uninformative and one of three modes degenerated.

### Acquisition choice, overall (all phases)
No selector beat the best static AF on the dev datasets: GP-Hedge, tuned rule, two learned numeric selectors, gpt-oss-120B (LMABO-style), tev1 (three modes). The 120B LLM was indistinguishable
from random AF choice; tev1's probabilities were at chance. Honest limits: dev datasets only (confirmatory run on the 5 held-out datasets was NOT run and is not justified by any dev result), 7 input
dimensions, noise-free LCBench, 10-20 seeds with 95 % CIs of roughly +/-15-20 % so effects under ~10 % cannot be excluded, prompts not tuned, no Jev access.

## Does the typed model know the BO vocabulary? (2026-10-25; `acq/knowledge_probe.py`)

Part 1: 14 true/false statements (7 true, 7 false) about acquisition functions, Pareto front, hypervolume, lengthscale, regret, answered with Noul. Part 2: 10 synthetic
plain-language states with an unambiguous right action (4 late/converged -> exploit, 6 early or stagnating with high uncertainty -> explore), answered with a 2-way Choice.
- tev1:0.8b: concepts 8/14 (57 %), AUROC 0.71, acquiescent (mean P(true) 0.72 on true, 0.56 on FALSE statements). Missed: Pareto-front points are not dominated (P = .61), larger
  hypervolume means worse (.55), short lengthscale means smooth (.63), regret falls when the best gets worse (.72), EI ignores uncertainty (.59). States 10/10, mean P(correct) 0.72.
- nimble (9B): concepts 14/14, AUROC 1.00 (P(true) 1.00 on true, 0.11 on false). States 10/10, mean P(correct) 0.87.
Reading: tev1:0.8b does not reliably know the vocabulary (including Pareto/hypervolume), but it does read simple plain-language state cues correctly. nimble knows both. The acquisition-choice nulls
for tev1 therefore reflect a weak stand-in as well as a hard task; the 120B LLM (which knows the vocabulary) was also at random-AF level, so knowledge alone is not sufficient. Caveat: 24 easy items;
real BO states are ambiguous, and a probe score says nothing about usefulness in the loop. nimble was set aside earlier for a different reason; this result reopens it.

### Prompt-level check: does the model understand the actual prompt? (2026-10-25; `acq/prompt_probe.py`, `acq/results/prompt_probe_output.txt`)

Part A: 14 true/false statements using the prompt's own jargon. Part B: 6 facts read out of 120 real rendered states, ground truth from the state data, scored against an always-majority baseline.
- **nimble (9B)**: Part A 13/14 (93 %), AUROC 1.00; the one miss is "little remaining budget is a reason to explore more widely" (P = 0.79, wrong). Part B: AUROC 1.00 on all six facts (0.99 for
  "LogEI used most"); accuracy 95-100 % on five of them and 66 % on "more than half the budget remains" (ranks perfectly but its 0.5 cut-off is misplaced). It reads the real prompt correctly.
- **tev1:0.8b**: Part A 9/14 (64 %), AUROC 0.67; wrong on "the incumbent is the best value observed so far" (P = 0.33). Part B: ranks some facts well (budget AUROC 0.93, last improved 1.00,
  >=5 distinct proposals 0.96) but is below the majority baseline in accuracy on four of six (uncalibrated thresholds), and weak on stagnation (AUROC 0.65) and surprise (0.75).
- Reading: nimble is a fair test of whether a typed decision model can use this prompt; tev1:0.8b only partly is. The one nimble error is a heuristic that matters for the task (budget and exploration).
- Process note: the probe showed a wall-clock of about 9 h (probably the laptop sleeping; direct calls took 0.5 s afterwards). Swap was near full (14.7 of 15.4 GB) with nimble loaded at 17 GB.

## Acquisition choice, nimble result and root cause (2026-10-25; `acq/results/nimble_analysis.txt`, `signal_check_output.txt`)

nimble (9B) sample mode, 8 dev datasets x seeds 0-9, 4000 calls, 0 fallbacks, ~1.5 s per call overall. Mean AUC: tev1-sample 0.1297, TS 0.1358, learned_L2 0.1358, **nimble 0.1372**, rule 0.1397,
random 0.1451, gpt-oss-120B 0.1457, LogEI 0.1500. nimble vs TS +1.0 % (p = 1.0), vs LogEI -8.6 % (p = 0.195; mean of per-dataset ratios -1.9 %), vs rule -1.8 % (p = 0.74), vs random -5.5 % (p = 0.74).
Nothing significant; 95 % CIs about +/-20 %. The state-blind mixture control for nimble was not run (root-cause check below made it low priority).
- Behaviour: switches AF on 86.7 % of steps; AF shares LogEI 26, LogPI 21, UCB 17, PosMean 16, PosSTD 12, TS 7 %, **identical across early / middle / late budget** (no budget response, although it reads the budget correctly).
- Calibration: AUROC of P(a) for "a's proposal beats the incumbent" = 0.543 (per-AF 0.50-0.56); confidence vs improvement spearman -0.037. Mean P (0.07-0.27) exceeds the observed rate (0.03-0.08).
- So three models of very different size (0.8B, 9B, 120B), two of which understand the prompt's vocabulary, all behave like AF mixtures with probabilities at chance.

**Root-cause check, no model involved (`acq/signal_check.py`)**: 64000 Phase A states, leave-one-dataset-out gradient boosting from the state (without AF history) to "AF a's proposal beats the incumbent":
AUROC LogEI .56, LogPI .57, UCB .58, TS .62, PosMean .58, PosSTD .71 (mostly a general "early steps improve more" effect). But choosing the AF by the learned predictions achieves a one-step normalised
gain of 0.0056, equal to random AF (0.0055) and below static UCB (0.0069) and LogEI (0.0063); the peeking oracle gets 0.0182. **The global state carries no usable information about WHICH AF's proposal will improve
(beyond when improvement is likelier overall), so no selector reading this state, numeric or language model, can beat random or static choice at one step.** This explains every null in the acquisition-choice component.
Caveats: one-step improvement is a myopic, noisy target; the state has no per-candidate information (the posterior mean and std at each AF's proposal, distance to incumbent), which is where signal would sit.

### Candidate-level signal check (2026-10-25; `acq/replay_cand.py`, `acq/signal_check_cand.py`, `acq/results/signal_check_cand_output.txt`)

All 1280 Phase A campaigns replayed with per-proposal logging (replay asserted identical proposals and outcomes). 64000 states x 6 proposals; per-proposal features: gap of posterior mean to incumbent,
posterior std, z, PI, EI, distance to incumbent and nearest tested point, mean and std ranks (all in units of the observed std). Scored by one-step normalised gain, as before.
- Reference one-step gains: random AF 0.0055, static LogEI 0.0063, static UCB 0.0069 (best static), peeking oracle 0.0182.
- Univariate within-state picks: highest EI = LogEI by construction (0.0063); lowest distance to the incumbent 0.0065 (slightly above LogEI); highest PI/z 0.0054; highest posterior mean gap 0.0050; highest std 0.0041.
- Leave-one-dataset-out gradient boosting on candidate features + AF identity: pooled AUROC 0.70 (vs about 0.5 for the global state), one-step gain **0.0069**: above random AF on 8/8 held-out datasets (Wilcoxon p = 0.008),
  **equal to static UCB (p = 1.0)** and 10 % above static LogEI. Adding global state (0.0064) or the other proposals' features (0.0069) does not help.
- Conclusion: the candidate numbers do carry signal (AUROC 0.70), but it is the signal the acquisition functions already use. The best reachable one-step selection with this information is about the best
  static AF; the oracle's headroom (0.0182) is the luck of which proposal happens to land on a good point. Closes the acquisition-choice component: no information source tested (global state, history, candidate
  numbers; numeric, learned, or language model) supports selection beating a good fixed AF. Lesson for the next component: run the model-free signal check first.

## Reframe after user critique (2026-10-25): "no signal" is relative to this state, target and data

User points, all accepted: (1) a null for zero-shot tev1/nimble says little about a model FINE-TUNED on the right target (BioDecision-4B v2 per Featherless/LinkedIn sources: Qwen3.5-4B-based biomedical decision model,
75.3 % on 63,432 held-out decisions, trained on a 1.75M-decision pool; architecture details such as a "head" unverified); (2) conditional-best selection should beat the global best when the best AF varies by landscape;
(3) a hybrid (LLM for semantic/rare decisions, decision model for frequent calibrated ones) is untested here; (4) candidate-level numbers carry signal (AUROC 0.70) and should be kept as inputs.
Check of (2) on existing data: per-dataset best static AF chosen on seeds 0-9 and scored on seeds 10-19: mean AUC 0.1455 vs pooled-best (TS) 0.1524 (-4.5 %, better on 4/8) and vs LogEI 0.1482 (-1.8 %); the best static
AF is the same on both seed halves for only 3/8 datasets (chance about 1.3). So regime knowledge helps modestly and the earlier "12 %" figure was inflated by choosing the best of six on the same seeds. Eight landscapes
cannot teach a landscape-to-AF mapping at all, so the failure to learn it is partly a data limitation. Earlier fine-tune attempt in `research/clm_jev_bo_exploration` (LoRA + head on Qwen2.5-0.5B) used 80 train / 40 dev
examples and was a loop-validation pass only. Claims in this file about "no signal" apply to: this state, one-step improvement labels, 8 LCBench landscapes, 10-20 seeds.

### Reference: a fine-tuned System One model plus LLM hybrid (arXiv 2610.27099, 19 Oct 2026; read from a fetch-tool summary of the first 100k of 151k characters, numbers to be checked in the paper)

"A general-purpose System One decision model for clinical records" (Mass General Brigham, Harvard, MIT): CS1-4B, 4.2B parameters, full fine-tune of a Qwen3.5-based decision model on 12.4M document-question pairs
(3.6M patients), 211 GPU-hours on eight B300s. Architecture: encode the document once; each question is a segment attending to the document and itself but not to other questions; a candidate head gives one logit per
option, softmax gives the distribution. Three label types: teacher-LLM soft labels averaged over option orders, exact rule-computed labels from structured data, and outcomes observed after the decision point.
Results: held-out patients balanced accuracy 89.7 vs 73.3 for its initialisation and 81.7 for the best open model; 21 public datasets 86.7 vs 81.7 (initialisation) and 88.6 (a 27B model), 6.4x faster; ECE 0.029,
and a one-parameter log-odds shift cut outcome-risk ECE from 0.203 to 0.014. Hybrid harness: a general LLM plans typed sub-questions and composes cited answers, the decision model answers every question over every
document, code aggregates; on note-only criteria the LLM was better (0.796 vs 0.766). No Bayesian optimisation content. Lessons for this project: (1) in-distribution fine-tuning gave +16 points, transfer to public data +5,
the same gap we should expect from simulator to lab; (2) labels from exact computation and observed outcomes map to counterfactual-rollout labels from the simulator; teacher-LLM labels do not, since the LLM selector was at random level;
(3) "encode once, ask many isolated questions" matches the many-questions-per-state idea; (4) recalibration on held-out data is part of the recipe; (5) the division of labour is repeated typed judgements for the decision model,
planning and synthesis for the LLM; (6) scale is orders of magnitude beyond one Mac, so a small-backbone head on 1e5-1e6 simulated decisions is the realistic analogue.

## Plan

1. Simulator core on LCBench: clock, K workers, event queue, charged decision
   latency, partial-curve observation.
2. Validate with ASHA/Hyperband-style early stopping against known behaviour.
3. Typed-decision layer (Choice: continue / stop / etc.), confidence gate, safe default.
4. Conditions, scored on HV/best-value vs **wall-clock** and instrument-hours:
   sync batch; async no-reaction; rule early-stopper (GP curve extrapolation);
   generative LLM with real latency; non-thinking logprob LLM; typed model;
   random-action typed model.
5. Calibration study as its own question (reliability curves).
6. Pre-registered hypotheses, separate exploration/confirmatory seed blocks,
   power analysis.
7. Later: synthetic text channel; lab-flavoured curves; backend-agnostic adapter
   with confidence gate and logging for real-lab use.

## Open questions

- (Resolved) kc1 and christine added to the dev set (see `splits.py`).
- Is tev1 (Ollama) still installed and running here, and does Jev early access exist? Stand-in assumed.
- Whether to also build a lab-flavoured second simulator (e.g. Tm ramp, coatings).
- LCBench data licence.


## 2026-10-28 Phase 1 of PLAN_meta_selector.md: landscape export and validation
- Exported 2629 real tabular landscapes (HPO-B 16 search spaces, TabRepo 7 usable model families, FCNet, NAS-Bench-201, LCBench) to `meta/pools/`; 1983 pass the property-based keep rules in the plan (LCBench 35, HPO-B 532, TabRepo 1409, FCNet 4, NB201 3).
- Problems found: TabRepo LinearModel is all-missing; HPO-B accuracies are heavily tied in several search spaces (median plateau over 50 % of configs in 5 of them); TabRepo ExtraTrees/RandomForest have many tied optima (about 20 % unique values). These are handled by the keep rules, listed in the plan.
- Engine smoke test on one landscape per family with budget min(50, 25 % of the pool): runs fine. hpob_7609/3492 reached zero regret in the initial design (AUC 0.0); easy landscapes will give uninformative labels, so the pilot reports the share of landscapes where all AFs tie.
- NAS-Bench-201 with 30 one-hot inputs costs about 2 minutes per 50-step campaign (versus 5 to 25 s elsewhere); it is held-out only, so acceptable.

## 2026-10-28 Stage 0 of the pilot (cost and triviality), code in `meta/` (landscapes.py, labels.py, stage0.py)
- Cost of one labelled state (6 AFs, H=4, two continuation policies LogEI and UCB, shared rollouts for equal proposals): LCBench 9.0 s, HPO-B 1.6 s, TabRepo 8.2 s (single core, 2 states each, so rough). About 340 states per landscape with a single continuation is therefore roughly 15 to 25 core-minutes, close to the plan's estimate; the pilot's double continuation costs twice that.
- Triviality (60 kept landscapes x 3 seeds, static LogEI, budget min(50, 25 % pool)): initial design never contained the optimum (0 of 60); LogEI AUC < 0.01 on 10 % of landscapes overall (TabRepo 25 %, LCBench 5 %, HPO-B 0 %). Median AUC: HPO-B 0.085, LCBench 0.072, TabRepo 0.022. Stop rule was >40 % trivial; passed.
- Caveat: TabRepo's small pools make it the easiest family; labels there will be smaller in magnitude.

## 2026-10-28 Stage 1 (G0, label reliability), 6 landscapes x 5 seeds, 480 states, H=4, 293 s wall on 12 cores
- Where a ranking exists the labels are reliable: mean within-state Spearman between LogEI- and UCB-continuation labels 0.61 across landscapes (bootstrap CI 0.50 to 0.76), state-pooled 0.59, versus -0.09 for labels paired across different states. Top-1 agreement 0.57 to 0.94.
- But the labels are sparse: 57 to 94 % of states (mean 0.76) have all six labels equal to zero, i.e. no AF's proposal improves the incumbent within 4 steps; a ranking is defined for only 63 of 480 states (13 %). So G0 as written passes on the states that have a ranking, while the sparsity condition (landscape mean under 90 %) passes narrowly and fails for LCBench blood-transfusion (94 %).
- The first version of analyse_stage1.py reported FAIL only because the per-landscape shuffled control was NaN (undefined rankings); fixed to compare on states where defined. Result file: meta/results/stage1_analysis.txt.
- Implication: with H=4 about 7 in 8 states carry no training signal for a ranking model; a full run would waste most of its compute. Options listed to the user (H=8 as in the plan's sensitivity check, denser label, or fewer but targeted states).

## 2026-10-28 H=8 sensitivity on the same states (meta/results/compare_H.txt, ranked_by_distinct.txt)
- 450 common states. All-zero share: H=4 0.74, H=8 0.66. Ranking defined (both continuations vary): 0.14 at both horizons. Mean within-state Spearman falls from 0.59 to 0.47 (kr-vs-k negative, -0.22 on 1 % of states). Where both horizons have signal (115 states) the best AF agrees 83 % of the time.
- Conclusion: a longer horizon does not fix sparsity; it adds some nonzero labels that are identical across AFs and lowers reliability. The sparsity is mostly structural: the six AFs often propose the same or equivalent points.

## 2026-10-28 Stage 2 preview: leave-one-landscape-out GBM on the 480 pilot states (meta/gbm_preview.py, results/gbm_preview.txt)
- Mean achieved H=4 label on all held-out states (units: fraction of y_max - y_median): GBM with all features 0.0122; best fixed AF chosen on the training landscapes 0.0118 (difference +0.0005, better on 2 of 6 landscapes); random AF 0.0128; per-state oracle 0.0213.
- Controls: AF-id-only reproduces the best fixed AF (0.0118, as it must); label-shuffled GBM ranges 0.0114 to 0.0139 over three repeats (difference -0.0004 to +0.0021), which is as large as any real-feature variant (candidate numbers + AF id: +0.0014; state+descriptors: +0.0013).
- Reading: a null at this size. No variant separates from the shuffled control; random AF choice even beats the best fixed AF on this label. The preview has 5 training landscapes and about 31 ranked states per landscape pool, so it cannot confirm or reject H1; it only says there is no large, easily learned effect.

## 2026-10-28 Stage 2b (powered offline H1 proxy): FAIL by the pre-declared rule (meta/results/stage2_analysis.txt)
- 30 landscapes (10 LCBench, 10 HPO-B, 10 TabRepo), 600 campaigns, 9460 labelled states (H=4, LogEI continuation); 70 % of states all-zero, 24 % ranked. About 1 h on 12 cores.
- Leave-one-landscape-out GBM: selected label 0.0229 vs best fixed AF (LogEI chosen on 26 of 30 training folds) 0.0219, diff +0.0010, wins 21/30, Wilcoxon p=0.046. Label-shuffled GBM: diff +0.0007, +0.0006, -0.0001 (wins 16, 18, 14 of 30). M1 minus shuffled: +0.0006, p=0.214. So the real-feature model is not distinguishable from a model trained on shuffled labels. Feature-group ablations (candidate, state, descriptors) all inside the same band.
- Random AF choice 0.0223 vs best fixed 0.0219: the six AFs are nearly interchangeable on this label. Per-state oracle 0.0373 (peeks) shows the headroom a selector would have to capture.
- Verdict per the stop rule given before the run: AF choice is closed as a learnable component on these landscapes with these features. Full label generation and the fine-tuned head (M2) are not run.
- Caveats: the label is a 4-step incumbent-gain proxy for online AUC of regret; behaviour-policy states only (no DAgger round); GBM at defaults, untuned; landscape descriptors are simple. A pass would have needed all of them to be adequate; a fail does not rule out better descriptors or labels, it rules out cheap gains.

## 2026-10-28 T0 and T1 of PLAN_component_triage.md (code: meta/bo.py, test_bo.py, t1_warmstart.py, analyse_t1.py; results/t1_analysis.txt)
- T0: meta/bo.py (swappable surrogate rbf_ard / matern52_ard / rbf_iso / rf, fixed and TuRBO-style adaptive box, initial-design override, observation noise as a fraction of the span, greedy and top-rank portfolios). Default arguments reproduce acq.engine exactly (max diff 0). All 11 tests pass.
- Deviations from the plan, fixed before any T1 data: greedy zero-shot portfolio (W1b) is the primary warm-start arm (plan had top-n0 by mean rank, kept as W1a); the "mismatched portfolio" control is replaced by a fixed random set (W0), since exact grids cannot be mismatched. W0 is a single draw (rng 0), so it is noisy as a control.
- T1 (10 landscapes: 5 LCBench, 5 TabRepo; random start 20 seeds, portfolio arms 3 seeds; portfolio arms are deterministic, AUC range across seeds 0.0000): mean AUC 0.0812 (random start) -> 0.0115 (W1b), wins 9/10, Wilcoxon p=0.0068; by family LCBench -96 %, TabRepo -46 % (mean of per-landscape ratios +0.71; change in mean AUC +0.86). W1a (top-by-rank) 0.0158 but negative on TabRepo ratios; W0 fixed random set 0.0432 (+0.47; ratios +0.14, 7/10, p=0.28).
- Failures worth noting: tabrepo_RandomForest/walking-activity (W1b 0.0758 vs random 0.0409) and ExtraTrees/APSFailure (start regret 0.343 vs 0.151): the portfolio does not transfer on some tasks. That is where a "trust the prior or fall back" decision has a job.
- Caveats: LCBench datasets and TabRepo tasks within a family share grids and have correlated optima, which is exactly the situation where a portfolio works; real labs rarely have 29 to 238 comparable prior campaigns. The comparison is portfolio start vs random start with the same number of evaluations, not vs a transfer-GP.
- Stop rule: continue (W1b gain far above 3 % on both families); TH-W (>=10 % on >=2 families, p<0.05) is on track at 10-landscape scale; the formal test on 30 landscapes is not yet run.

## 2026-10-28 W expanded to 30 landscapes, noise 0 and 5 % (meta/w_expand.py, analyse_w30.py; results/w30_analysis.txt)
- Bug found and fixed during the run: the first launch compared the seed value (4100+) with small constants, so the noise-free portfolio arms and the W0/FPS/W2x controls were never created; only the noise arms ran. Fixed with the seed index; resumable rerun produced the missing campaigns (4170 files). The first analysis attempt crashed on the missing keys, so no results were read from the incomplete data.
- Deviations from the plan (all fixed before any W30 data): HPO-B region prior (W2, k-means on top 5 % of source points), W2x mismatched-search-space control, farthest-point-start control (FPS), W0 averaged over 10 fixed random sets, minimum 8 training tasks, trust-statistic logging.
- Noise 0 %, 30 landscapes: W1 vs random start: mean AUC 0.0735 -> 0.0457 (-38 %), mean-of-ratios +0.44, wins 23/30, Wilcoxon p=0.0034. By family: LCBench +0.74, TabRepo +0.61, HPO-B -0.02 (no gain). TH-W PASS (>=10 % on two families, p<0.05). Controls: FPS -0.06 (p=0.50), W0 fixed random sets -0.07 (p=0.93), mismatched region W2x -4.56 (wins 2/9): the gain is information from related tasks, not coverage, and a wrong prior is badly harmful.
- Noise 5 % of span: W1 mean AUC 0.0881 -> 0.0608 (-31 %), mean-of-ratios +0.29, wins 22/30, p=0.0013; LCBench +0.57, TabRepo +0.32, HPO-B -0.03. TH-W PASS under noise too. FPS -0.02, W0 -0.07, W2x -1.38.
- HPO-B: the region-prior version gave nothing (and the mismatched version is harmful). This is one implementation (k-means centres from top 5 % of other tasks' points, input columns assumed aligned across tasks of a search space, accuracies heavily tied); it does not show that transfer is impossible there.
- Fall-back headroom (noise 0): always-W1 0.0457, hindsight best-of(W1, random) per landscape 0.0289, random 0.0735. W1 is worse than random on 7/30 landscapes (5 HPO-B, lcbench/APSFailure, tabrepo_RandomForest/walking-activity). Hindsight selection from 3-seed vs 20-seed means is optimistic (winner's-curse), so treat 0.0289 as an upper bound.
- Trust statistic (Spearman of portfolio's observed values vs source mean score): rho 0.22 with W1's relative gain on 20 exact-grid landscapes; the lowest-trust landscapes include ones where W1 helped a lot (madeline +0.82, eating +1.0). So this statistic, as defined, does not identify the failures; a better signal is needed.

## 2026-10-28 T2 (surrogate K) and T3 (search region R), noise 0 and 5 % (meta/t23.py, analyse_t23.py; results/t23_analysis.txt, t23_winners_curse.txt)
- 10 landscapes (3 LCBench, 4 HPO-B, 3 TabRepo; changed from the plan's T1 landscapes before data), seeds A 4000-4004 select, B 4100-4119 score, 3500 campaigns.
- All four gates FAIL. Per-landscape best entry chosen on A seeds and scored on B seeds is no better than the best fixed entry (chosen on the other landscapes): K noise 0 %: +0.003 (3/10 wins, p=0.50); R noise 0 %: -0.037 (2/10, p=0.77); K noise 5 %: -0.070 (2/10, p=0.97); R noise 5 %: -0.078 (4/10, p=0.88). The best fixed entry is the default ARD RBF GP with global search in 10/10 folds noise-free, and 4-6/10 under noise.
- The raw B-seed tables show large per-landscape differences (for example hpob_7609/145972 with box 0.30: AUC 0.1137 vs default 0.1532; lcbench Amazon with box 0.30: 0.2867 vs 0.1698), but the A-seed choice does not predict which entry wins on B, so the landscape-level differences are mostly seed variance in this design (5 selection seeds, random initial designs).
- Caveat: only 5 A seeds and 10 landscapes; a design with more selection seeds could find stable per-landscape differences. As pre-registered, K and R close for noise 0 and 5 %. The step-level "when to switch" question for these components is not tested.

## 2026-10-28 Fall-back triage step 1: W1 vs random start on 150 exact-grid landscapes (meta/w150.py, analyse_w150.py; results/w150_analysis.txt, w150_stability.txt)
- Incident: the first launch crashed (a `noise` keyword clash with a trust feature; the macOS `sed -i` fix silently failed), leaving only 30 random-arm files; I misread the completion notice. Fixed in Python, single job verified, full rerun (3000 files). The first analysis crashed on empty data, so nothing was read from the broken run.
- 150 landscapes (30 LCBench + 20 each of six TabRepo families), noise 0: always-W1 AUC 0.0329 vs random 0.0535 (-39 %); W1 worse than random on 33/150 (22 %); hindsight best-of 0.0247 (fall-back headroom 25 % of W1's AUC, optimistic). Noise 5 %: 0.0505 vs 0.0657; worse on 45/150 (30 %); headroom 20.7 %. Failures are common across TabRepo families (CatBoost 7/20, LightGBM 7/20, XGBoost 6/20) and LCBench (4/30), not rare as the 30-landscape view suggested.
- Single trust features (computed from the n0 portfolio observations) do not predict failure: AUROC 0.41 (rank agreement with source), 0.41 (R^2), 0.52 (GP LOO error), 0.56 (GP noise); the largest departure is obs_cv at 0.29 (0.71 sign-free), untested for multiplicity (10 features) and confounded with family and metric scale.

## 2026-10-28 Fall-back triage step 2: can the portfolio failure be predicted? (meta/fallback_features.py, decide_fallback.py; results/decide_fallback.txt, failure_by_dataset.txt): NO
- Features: SRC (leave-one-source-task-out success of the portfolio start, mean start regret, spread, n tasks), TRG (target observations: similarity to source tasks, rank agreement, R^2, skew, GP LOO error, lengthscales, noise). Decision: fall back to random start if P(fail)>0.5; models logistic regression and shallow GBM; grouped CV by dataset name and leave-one-family-out; label-shuffled controls; realised cost-free (optimistic).
- Result: every cell fails the pre-declared pass rule. Noise 0 %: best case SRC GBM CV-by-dataset +3.6 % vs always-W1 (p=0.38, 16 fall-backs); TRG and ALL range from +1.9 % to -10.4 %; shuffled controls 0.0327 to 0.0413. Noise 5 %: SRC only, +2.7 % at best, -21 % at worst. Hindsight stays at 25 % but is not reachable with these features.
- Structure of the failures: the top 10 failures hold 78 % of the total excess regret; failure is partly dataset-linked (Buzzinsocialmedia_Twitter 4 of 6 families, blood-transfusion 2/3, kc1 2/3) but no dataset fails in all of its landscapes, and 15 of 34 repeated datasets fail in some but not all. So failure depends on the (task, model family) pair, not on the dataset alone.
- Not tried: a cost-weighted (excess-regret) target, a hedge arm with real cost, richer similarity (task-embedding) features. A post-hoc cost-weighted variant would be flagged as such.

## 2026-10-28 regime grid (descriptive, existing runs, `meta/regime_grid.py`, `meta/results/regime_grid.txt`)
36 landscapes with all of five static AFs and random AF (8 LCBench from phaseA, rest from stage2 with about 3 seeds each), all noise-free, so noise cannot be stratified from existing data.
Ratios of stratum means. Random AF vs LogEI: +4.2 % overall, +16.7 % HPO-B, +34.7 % TabRepo, -5.2 % LCBench. The fixed AF chosen on the other landscapes vs random AF: -2.3 % overall (LCBench -14 %, TabRepo -7 %, HPO-B +42 %).
Per-landscape spread between the best and worst of five static AFs averages 0.067 AUC (69 % of LogEI): AFs differ a lot within a landscape, but which one is best does not transfer from other landscapes. The spread is an in-sample best-minus-worst, so it overstates usable headroom (winner's curse, 3 seeds per stage2 landscape).
So "random AF matches the best fixed AF" holds only on average and for LCBench; on HPO-B the chosen-elsewhere fixed AF is worse than random.

## 2026-10-29 detection T1 (PLAN_fallback_detection.md; `meta/detect_run.py`, `meta/analyse_detect_t1.py`, `meta/results/detect_t1.txt`)
Baselines (earlier, Amendment 4, `acq/results/phaseE_analysis.txt`): bandit_ts, sched_EI_PI, dwell rule all null vs TS and LogEI (Holm p = 1.0); dwell D=1 won tuning so the dwell arm equals the existing rule.
Regime grid (`meta/results/regime_grid.txt`): AF choice matters within a landscape (best-worst spread 69 % of LogEI, in-sample) but the winner does not transfer (fixed AF chosen elsewhere vs random AF: -2.3 % overall, +42 % on HPO-B); noise not stratifiable.
T1: 150 landscapes, 4 seeds per arm, 12 random seeds, 1800 units in 58 min. Mean AUC: W1 0.0329, W1+FPS(10, n0/2) +4.8 %, hedge +13.3 %, W1+FPS(10) +13.6 %, W1+random(10) +15.7 %, random +51.6 %. Failures (W1 worse than random by >5 %): 29/150.
Gate 1: injection helps in 24 % (random) and 24 % (FPS) of failing landscapes individually, 38-41 % for either-of-two (the union is a post-hoc reading; the pre-declared wording did not say). Gate 2: realised oracle 10.6 % below always-W1 (8.7 % with selection on 2 seeds, scoring on 2), vs 26.6 % cost-free. Gate 3: hedge captures -126 % of the oracle gain (it is worse than W1).
Positives (injection helps): 14-16 of 150 landscapes, below the 20 needed for a GBM; 3-4 of them are non-failing landscapes.

## 2026-10-29 detection T2 (`meta/detect_features.py`, `meta/decide_detect.py`, `meta/results/decide_detect.txt`)
Every cell FAIL (remedies R and F, k=10, m=n0; step-0 logreg S0, online ON, online + source reference RS, online + target random-run reference RT (upper bound), three one-feature threshold rules; CV by dataset and leave-one-family-out).
Best realised results: RT +3 % of the oracle gain (CV by dataset, remedy R), S0 +4 % (remedy F); the 30 % bar needed 3.2 % of W1's AUC. Detectors flag 0-7 landscapes, almost never correctly (0-2 true positives); threshold rules mostly never fire because any flag loses in training folds.
Power caveat: positives are 11 (R) and 7 (F) of 150, so a fail here is weak evidence about detectability in general; the ceiling is also only about 10 % (realised oracle), so the practical conclusion is that online fall-back is not worth building on this testbed.
The label-shuffled control is a no-op for the threshold rules (they use no labels as classifiers), so it is not informative for them.

## 2026-10-29 source prior: plan v2 and T0 (`PLAN_source_prior.md`, `meta/sp.py`, `meta/test_sp.py`)
Plan amended after an external review before any run: shrinkage weights (lambda) instead of pure softmax, variance inflation by the weighted source spread (c = 1), SP-frozen arm and TH-SP4 (reactivity), SP-shuffled expected at W1 level (shared portfolio start), SP-permw (weights permuted once), failure-group gate on new seeds with a margin, 6 LCBench landscapes moved from the test set into tuning, family units corrected to 7.
T0: 18 checks pass (weights, modes, frozen/permw/shuffled properties, variance inflation, determinism, budget, no duplicates, noise, leaked-target sanity). Nothing run beyond tests.

## 2026-10-29 source prior T1 tuning (`meta/sp_tune.py`, `meta/analyse_sp_tune.py`, `meta/results/sp_tune.txt`): STOP RULE TRIGGERED
42 tuning landscapes (36 fresh TabRepo, 6 LCBench), seeds 4500-4501, 9 configurations kappa {3,10,30} x lambda {.25,.5,1}. SP vs W1 (ratio of means, negative = better): TabRepo +21 % to +58 % worse in all 9 configurations; LCBench from -13.7 % to +12.7 % (not smooth in the configuration, so mostly noise on 6 landscapes). SP-uniform (same prior mean and variance term, equal weights) is also worse than W1: TabRepo +12.0 %, LCBench +16.6 %. Selected by the declared rule: kappa 10, lambda 0.25 (TabRepo +21.5 %, LCBench -13.7 %). No configuration is 2 % better on both sources, so the test run is not performed.
Effective number of sources stays high (11 to 197 of about 235), so the hedge mechanism works as designed; weight volatility is small. The loss is therefore not from winner-take-all weights. The uniform arm also losing suggests that the prior-mean surrogate form (linear fit on m(x) plus GP on residual plus source-spread variance inflation) is worse than the standard GP after the portfolio start; not yet diagnosed.

## 2026-10-29 source prior diagnostic (EXPLORATORY, post hoc, tuning set only; `meta/sp_diag.py`, `meta/results/sp_diag.txt`)
Engine check: SP with the prior switched off (b = 0) reproduces W1 exactly (+0.0 % on both sources), so the machinery is correct.
TabRepo (36 landscapes, W1 AUC 0.0253): every prior-mean variant is worse than W1 (uniform +7 % to +15 %, selected SP +4.2 % at c = 0.25, +21.5 % at c = 1, +23.5 % at c = 0).
LCBench (6 landscapes, W1 AUC 0.1917): the prior mean helps: uniform -6 % (c = 0) / -3.9 % (c = 0.25), selected SP -26.1 % (c = 0) / -24.6 % (c = 0.25) / -13.7 % (c = 1). Weighting adds to the gain on LCBench (SP vs uniform at c = 0: -26 % vs -6 %).
Reading: the prior mean pays where the portfolio start leaves room (LCBench) and does not where the portfolio is already near-optimal (TabRepo). The LCBench evidence is 6 landscapes and 2 seeds, and the variants were looked at after the stop rule fired; any follow-up is a new hypothesis that needs its own pre-registration and fresh landscapes (24 LCBench dev landscapes remain after the tuning set).

## 2026-10-31 Source prior, LCBench-only pre-registered test (PLAN_source_prior_lcbench.md): FAIL
24 LCBench dev landscapes not in the tuning set, seeds 4700-4703, noise 0, fixed kappa 10 / lambda .25 / c .25, 96 units, 436 s. Results: `meta/results/sp_lcbench.txt`, code `meta/sp_lcbench.py`, `meta/analyse_sp_lcbench.py` (gates written before the run; one timing unit was run first and kept).
Ratio of means vs W1 (0.0272): SP +7.0 %, frozen +3.4 %, uniform +0.7 %, permw +3.3 %, shuffled +15.7 %, SP c=0 +3.6 %. TH-L1 FAIL (p 0.78), TH-L2 FAIL, TH-L3 FAIL. SP worse than W1 by >5 % on 25 % of landscapes, better by >5 % on 8 %. Control SP-permw (>=2 % worse than SP) fails (-3.5 %), but L1 already fails so nothing rests on it. Zero W1 failures vs random on these landscapes.
Reading: the diagnostic's -25 % on 6 LCBench landscapes was selection on a hard subset, not a transferable effect. Descriptive, post hoc: on 12 of 24 test landscapes W1 already has AUC 0 (optimum in the portfolio start), so no room; the 6 with largest W1 AUC show SP -4.5 % vs W1 (not gated). The tuning set's W1 AUC (0.19) was 7x the test average (0.027).
Source-prior line closed: T1 stop (TabRepo), diagnostic (hypothesis), confirmation failed (LCBench). T2 (noise) and T3 (typed Score) not run.

## 2026-10-31 Non-stationary T0 (PLAN_nonstationary.md v2): engine, pair lists, tests
`meta/nonstat.py` (11 arms: oblivious, oracle_time, oracle_target, oracle_keep, window, periodic, cusum, shuffled_det, random_trigger, sp_stale_oracle, sp_stale_cusum), `meta/nonstat_pairs.py` -> `meta/results/nonstat_pairs.json` (pilot 20, tuning 40, test 100 pairs; cells LCBench/TabRepo x related/unrelated, tertiles of full-grid Spearman within group; TabRepo families round-robin), `meta/test_nonstat.py` (all checks pass, incl. oblivious == bo.run, regret on the blend y_post, oracle-time uses no pair information, paired re-evaluation returns y_post, stale replaced by fresh, A and B excluded from sources, determinism of all arms, stationary controls).
Implementation choices to flag: (1) SP-stale needed no change to sp.py: the stale row (GP mean of the pre-trigger data over the grid) is built in nonstat.py; the "continuous" SP-stale variant in the plan was dropped (stale data do not change after the trigger, so it would equal the triggered version). (2) A detector trigger takes effect at the start of the next step. (3) Shuffled-state detector draws z by bootstrap from its own past z values (causal), not a permutation of the whole stream. (4) "Unrelated" is relative: Spearman tertiles differ by group (ExtraTrees' bottom tertile is still about 0.5). (5) Pairs are unique across sets, but A tasks repeat across sets (LCBench has only 30 tasks), so tuning and test share some A tasks.
No compute beyond tests; T1 pilot not run.

## 2026-10-31 Non-stationary T1 pilot (PLAN_nonstationary.md v2): N0 passes in the gated regime; the false-alarm rule makes restart-based detection infeasible
240 units (160 change, 80 stationary), 1112 s, `meta/results/nonstat_t1.txt`, code `meta/nonstat_t1.py`, `meta/analyse_nonstat_t1.py` (written before the run). 20 pilot pairs, seeds 4800-4803, T_c 25 % and 50 %, post-change AUC (ratio of means over pairs; per-pair average over seeds and T_c).
- TH-N0, unrelated x abrupt (10 pairs): oracle_time -80.0 % vs oblivious (p 0.007), LCBench -41 %, TabRepo -89 % -> PASS. Unrelated x partial: -68 % (p 0.04). Related x abrupt/partial: oracle_time is worse than oblivious (+37 %, +143 %), so restart hurts when tasks agree. oracle_keep: -50 % related x abrupt (p 0.053), -75 % unrelated x abrupt, -82 % unrelated x partial, i.e. keeping stale data with inflated noise dominates restarting in every regime. oracle_target (B's portfolio) is best in the unrelated regimes (-91 %, -87 %). Window ~0. Periodic restart -60 % in unrelated x abrupt (p 0.12), +125 % related x abrupt.
- False-alarm cost: one forced restart at 50 % in a stationary campaign raises AUC +208 % (0.075 -> 0.232). Pre-declared rule: allowed false-alarm rate = min(10 %, 0.02 / 2.08) = 1.0 % < 2 % -> restart-based detector "infeasible" by the plan. Periodic restart every 25 %: +510 %.
- Caveats (not in the plan): the oblivious arm recommends the best stored value, which stays the stale optimum until a fresh value beats it, so part of the headroom is the recommendation rule; a cheap deployable baseline (re-measure the incumbent every few steps, which returns y_post directly) is not in the arm list. TabRepo "unrelated" pairs are anti-correlated (AUC of oblivious above 1), so effects there are very large. 10 pairs per regime. The false-alarm cost was measured for restart only, not for the soft (keep / SP-stale) response, which looks far cheaper.

## 2026-10-31 Non-stationary A0 (Amendment 1 / 1b of PLAN_nonstationary.md): new arms and tests
Pins P1-P4 written into the plan (section 9b) before building. `meta/nonstat.py` now has 21 arms: added refresh, remeasure_keep, random_keep_rm, cusum_keep, random_keep, shuffled_keep, sp_stale_remeasure, sp_passive, forced_keep, forced_spstale. `meta/test_nonstat.py`: 44 checks pass (27 old + 17 new): refresh with r > budget == oblivious; each re-measurement costs one step; zero false alarms at noise 0 on 12 stationary campaigns and remeasure_keep == refresh there; remeasure_keep fires only on a re-measurement step at or after T_c; re-measuring arms identical to refresh until their first trigger (the trigger step itself differs because the keep response changes the recommendation immediately); one keep event per campaign; inflated variance raises the posterior sd at the stale config; sp_passive builds its history row from observations older than 15 (3, 4, 5, ...), the surrogate sees exactly the last 15, equals the plain engine until the history has 3 points; determinism of the new arms on a real pair.
Implementation notes: a re-evaluated config now replaces any stored entry of that config in every arm (previously only for aware arms; window arm affected only in that case, tests unchanged). sp_passive recommends the evaluated config with the highest fused posterior mean (design choice, flagged in 9b). No compute beyond tests; A1 not run.

## 2026-10-31 Non-stationary A1 started (tuning pairs, pins in PLAN_nonstationary.md 9c)
Pipeline `meta/nonstat_a1.py` (S1a keep variance, S1b interval r, S2 CUSUM h + c_FA(spstale), S3 sp_stale_remeasure / sp_passive / shuffled_keep, random controls, detector selection). Seeds 4900-4903, 40 tuning pairs. Smoke test on one pair: all arms run. Log: meta/results/nonstat_a1_run.log; report/decisions in meta/results/nonstat_a1/.

## 2026-11-01 Non-stationary A1 complete (tuning pairs)
v*=0.01, r*=10, detector remeasure_keep, cusum_keep infeasible, no stop rule fired; manifest in PLAN_nonstationary.md 9d. remeasure_keep -46.8 % vs refresh post-change, stationary +3.3 % vs oblivious (0 false alarms); sp_passive stationary +140 %. A2 (test pairs) awaiting approval.

## 2026-11-01 Non-stationary A2 complete (100 test pairs, seeds 5000-5003); output meta/results/nonstat_a2.txt, diag meta/results/diag_cusum.txt
A0 PASS (oracle_keep -55.6 % vs refresh gated, both strata). A1 PASS (remeasure_keep -39.7 % vs refresh gated, p 3e-11; related regimes -38 / -49 %, no harm; stationary +0.0 % vs refresh, +4.5 % vs oblivious; vs random_keep_rm -30.6 %; recovers 71.5 % of oracle_keep gain). A3 FAIL as pre-declared: better passive arm = sp_passive, remeasure_keep -19.4 % (Wilcoxon p 0.089, Holm 0.178; 90 % CI [-34.1, -0.4]); vs sp_stale_full -46.1 % (p<0.001). Passive-suffices fails: sp_passive stationary +172.5 %, sp_stale_full +14.0 %. CUSUM diagnosis: H_explore and H_miscal both rejected (Spearman |z|-distance -0.07; median |z| 0.73 but mean 1.89, heavy tails). Noise stage (9f) awaiting approval.

## 2026-11-01 Missed-trigger check and noise tuning started
diag_missed: of 263 never-triggered campaigns (of 1600), 47 had y_A(x*)=y_B(x*), 213 had an incumbent already measured post-change; perfect keep/no-keep on the misses would remove 0.9 % of remeasure_keep's post-change AUC mass. The literal "identical value" claim is false, the substantive one holds. Noise-stage pins in PLAN_nonstationary.md 9i/9k; N1 tuning (meta/nonstat_n1.py, seeds 5100-5103) started, manifest 9j after it.

## 2026-11-01 Noise tuning N1 complete
v*=0.01, r*=5, m*=3 (delta 0.212), stationary FA 0.0 % (predicted 5-25 %: too pessimistic), post-change -45 % vs oblivious, oracle_keep -42.6 % vs refresh gated, vs matched random control -33.8 %. refresh_rep: +30.6 % post-change, -5.7 % stationary. Manifest in PLAN_nonstationary.md 9j. Test (seeds 5200-5203) awaiting approval.

# Similarity-weighted source prior (reactive transfer): plan v2 (2026-10-29)

Status: v2 after an external review (gaps G1-G8, decisions in section 8); T0 only, nothing run. Follows the closed fall-back line (`PLAN_fallback_detection.md`, T1/T2): W1's failures are common (29/150 landscapes), insuring against them costs more than it saves, and they are not detectable from step-0 or step-10 information with binary rules. This plan tests the soft version: instead of committing to or abandoning the portfolio, keep using the source tasks as a prior but re-weight them every step by how well they agree with what has been observed. Numeric method first; a typed model enters only if the numeric one passes.

## 1. Question
Does a source prior whose weights are updated from the campaign's own observations (a) beat the portfolio start with a standard GP (W1), (b) protect the landscapes where W1 fails, and (c) do so because of the re-weighting, not merely because the source scores are used as a mean function?

## 2. Method (all numeric, exact shared grids as in W1; LCBench and TabRepo)
- Source matrix S: normalised scores of the other tasks of the same search space on the shared grid (the held-out task is excluded, as in W1). Initial design: the same greedy portfolio as W1, so every arm starts identically.
- Weights at every step from the n observed target points (all observed points, pre-declared): rho_i = Spearman(S_i at observed configs, observed y); **w = (1 - lambda) * uniform + lambda * softmax(kappa * rho)** (shrinkage, so the hedge holds by construction; lambda = 0 is uniform, lambda = 1 pure softmax). Weighted prior score m(x) = sum_i w_i S_i(x); weighted source spread v(x) = sum_i w_i S_i(x)^2 - m(x)^2.
- Surrogate: target value = a + b * m(x) + GP residual (a, b by least squares each step, b >= 0; default ARD RBF on the residual). **Predictive variance = GP variance + c * b^2 * v(x), c = 1 fixed**, so an uncertain prior mean is not treated as known. LogEI from the resulting Gaussian.
- Diagnostics logged per step: effective number of sources exp(entropy(w)); total variation of w between consecutive steps.

## 3. Arms (same landscapes, same seeds, same initial design)
| Arm | What it tests | Expected level |
|---|---|---|
| random | random start + standard GP | reference |
| W1 | portfolio start + standard GP | baseline to beat |
| SP | weights updated every step (the method) | |
| SP-frozen | weights computed once from the n0 initial observations, then fixed | the test of reactivity (equals SP at step 0) |
| SP-uniform | same prior mean and variance term, equal weights | isolates the value of weighting |
| SP-shuffled | each source task's scores permuted across configs, then the method | adds nothing: within 5 % of W1 (shared portfolio start, so not random level) |
| SP-permw | SP's weights at step n0 permuted over source tasks once, then fixed (same weight distribution, wrong assignment) | isolates weight information from weight volatility |
Count: 150 x 7 x 4 seeds, minus tuning-set landscapes (below): about 4000 campaigns, under 2 hours.

## 4. Data, seeds, tuning
- Tuning set: 36 fresh TabRepo landscapes (6 per family, not in w150) + **6 LCBench dev landscapes held out of the test** (fixed-seed draw), seeds 4500-4501. Grid: kappa in {3, 10, 30} x lambda in {0.25, 0.5, 1.0}. Stop rule: if SP is not at least 2 % better than W1 on the tuning set for any configuration, **on both TabRepo and LCBench**, stop before the test run. Per-family sensitivity reported.
- Test: the remaining 144 w150 landscapes (24 LCBench + 120 TabRepo), seeds 4600-4603, noise 0. Failure group, pre-selection from T1 seeds 4300-4311: the 29 landscapes where W1 lost to random (those in the tuning set removed). **The gate uses the new seeds:** TH-SP2 applies to pre-selected landscapes where W1's seed-averaged AUC on 4600-4603 exceeds the random arm's by more than 5 % of the random AUC; the recomposition of the group is reported, plus the same test on all new-seed failures (secondary).
- If the noise-free test passes: the same with 5 % observation noise (T3). Reserved LCBench, FCNet, NAS-Bench-201 untouched.

## 5. Hypotheses and gates (fixed now; seed-averaged AUC per landscape, ratio of means, paired Wilcoxon over landscapes; family-level secondary with 7 units: LCBench + 6 TabRepo families)
- **TH-SP1:** SP at least 5 % below W1, p < 0.05, same sign in at least two families.
- **TH-SP2:** on the new-seed failure group, SP closes at least 50 % of W1's excess over the random arm; on the other landscapes SP is no more than 2 % worse than W1 on average **and** the per-landscape distribution of harm (share of landscapes worse by more than 5 %, worst case) and per-family breakdown are reported.
- **TH-SP3 (weighting matters):** SP at least 2 % below SP-uniform, p < 0.05.
- **TH-SP4 (reactivity):** SP at least 2 % below SP-frozen, p < 0.05. If TH-SP1 to TH-SP3 hold but TH-SP4 fails, the finding is "a static similarity-weighted prior captures the gain; per-step re-weighting adds nothing", and the reactive claim is closed. That outcome must be reportable as a result.
- **Controls:** SP-shuffled within 5 % of W1 (a permuted prior adds nothing), SP-permw at least 2 % worse than SP.
- Both summaries (change in mean AUC and mean of per-landscape ratios) reported. The hedge mechanism is checked from the diagnostics (effective number of sources and weight volatility).
- Not predictive: T2 passing narrowly says little about T3; the prior's value and risk can both grow under noise.

## 6. Stages (stop and ask at each)
- T0: implement `meta/sp.py` (separate from `meta/bo.py`, so the default path is untouched); tests: default path unchanged, weights sum to one under shrinkage and lambda = 0 gives uniform, a source equal to the target gets the largest weight, SP-frozen equals SP at step 0 and keeps constant weights while SP's change, shuffled sources keep per-task score distributions but break the config assignment, permw keeps SP's weight multiset, variance inflation active and finite, held-out task excluded, a leaked-target sanity check finds the optimum quickly, determinism, budget respected, noise runs.
- T1: tuning run and stop rule. T2: test run and gates, noise 0. T3: noise 5 % if T2 passes. T4 (only if numeric passes and leaves a gap): typed Score question per source task with restored hyperparameter names against the numeric weights; gate: equivalence within 2 % or better, plus a shuffled-state control.

## 7. Risks
- Best case for transfer again (shared grids, 29 to 238 sources). The shuffled-source control and the leave-family-out reading bound this partly.
- Spearman on few points is noisy early; the shrinkage, the frozen arm and the weight-permutation control address that.
- The portfolio points were chosen using the source scores, so early rank agreement is partly built in (a selection effect to state in the write-up). Late in the campaign the points cluster near the incumbent, so weights reflect local agreement; a trailing-window variant is reported on a subset as a diagnostic.
- The failure group was chosen from earlier seeds on these landscapes; gating on the new seeds handles regression to the mean.

## 8. Review gaps and decisions
G1 adopted (shuffled control expectation). G2 adopted (gate on new-seed failures). G3 adopted as shrinkage (a), with lambda tuned and the effective-sources diagnostic. G4 adopted in a stronger form (weights permuted once; same distribution). G5 adopted (SP-frozen, TH-SP4). G6 adopted inside SP with c = 1. G7 adopted as a pre-declared correlation set (all observed points) with a trailing-window diagnostic. G8 adopted (6 LCBench landscapes moved to tuning, removed from the test). Family-unit arithmetic corrected to 7 (LCBench plus 6 TabRepo families), including the detection plan's earlier secondary (never run).

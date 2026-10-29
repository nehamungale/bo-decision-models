# Online fall-back detection for the warm start: plan v2 (2026-10-28)

Status: T0 built (engine + tests), nothing run. v2 incorporates an external review (gaps G1-G18, listed in section 8 with the decision taken on each).
Builds on `meta/results/decide_fallback.txt` (prediction from step-0 and portfolio-start features failed every pre-declared cell) and `w150`.
Correction: the stored w150 results hold only AUC/final regret (one noise-free W1 seed per landscape), so detection cannot be tested on existing data.

## 1. Question
After k further steps, does what has been observed tell us W1 is failing, early enough that an insurance of m evaluations costs less than the regret it avoids? Realised cost counted. Headline claim sought: **an online detector beats a step-0-only detector and every fixed arm.** Scope: a fixed-k snapshot analysis. A pass does not validate a step-by-step monitoring rule (section 5).

## 2. Arms (150 exact-grid landscapes of w150; noise 0; W1 and injection arms seeds 4300-4303; random arm seeds 4300-4311; n0 = 2D+1; total evaluations n0 + B in every arm)
| Arm | Definition |
|---|---|
| random | random initial design |
| W1 | greedy portfolio initial design, BO to the budget |
| hedge | first ceil(n0/2) portfolio points + floor(n0/2) random |
| W1+R(k), W1+F(k) | identical to W1 for n0 + k evaluations, then m = n0 injected points, R = uniform random without replacement from unevaluated points, F = farthest-point (max-min distance), then BO |
| W1+F(10, n0/2) | m sensitivity |
**Primary: k = 10, m = n0, both remedies.** k = 5 is sensitivity only (one primary k, no alpha splitting). Injection never repeats an evaluated point (tested).
Action space: the detector chooses between W1 and the injection arm. Full abandonment is out of scope: injection keeps the n0 + k evaluations already paid for in the GP.
Count: 150 landscapes x (4 seeds x 6 non-random arms + 12 random) = 5400 campaigns. Pilot on the 5 slowest landscapes first to get a real time estimate.

## 3. Definitions fixed before data
- **Failure** (independent of any remedy): seed-averaged AUC of W1 exceeds the seed-averaged random-arm AUC by more than 5 % of the random AUC.
- **Injection helps**: seed-averaged AUC of the injection arm is below W1's by more than epsilon = 1 % of W1's AUC. Seed-averaged labels are primary; per-seed labels are sensitivity. Epsilon 0 and 2 % reported.
- **Realised oracle**: per landscape, the best of {W1, W1+R(10), W1+F(10)} by seed-averaged AUC (in-sample, so a ceiling; the same quantity with arm selection on 2 seeds and scoring on the other 2 is reported).

## 4. Stop gates at T1 (before any detector is trained)
1. **Cross-tab** failure x injection-helps. If, among failing landscapes, injection helps in fewer than 30 %, stop and redesign the remedy, not the detector.
2. **Realised oracle gain** over always-W1 below 5 % of W1's AUC: stop (nothing to detect).
3. **Hedge captures** more than 70 % of the realised oracle gain: detection has little room; report and ask.
4. Fewer than 20 positive landscapes in a training fold: logistic regression and one-feature threshold only, no GBM.

## 5. Detectors, features, test
Detectors (observable information at step k only): (a) **step-0 only** (the decide_fallback source/target features, the required baseline); (b) **online**: step-0 features plus step-k features: Spearman and R-squared of observed y on the source score, standardised incumbent gain over the best portfolio point, steps since improvement, GP leave-one-out RMSE, lengthscales, noise; (c) **reference-aware**: (b) plus the percentile of W1's incumbent among random-start incumbents at the same step. Two versions of the reference: from the **target's own** random runs (12 seeds; not available in a real lab, reported as an upper bound) and from **source tasks'** random runs of the same family (deployable). (d) one-feature "no progress" threshold. Models: logistic regression, shallow GBM (subject to gate 4). Label shuffled controls for all.
CV: grouped by dataset name (stricter than by landscape: it also separates the same dataset in different model families, and all seeds of a landscape fall in one fold; verified in T2) and leave-one-family-out.
**Pass for the primary (k = 10) in both CV schemes, for the online or reference-aware detector:**
1. captures at least **30 % of the realised oracle gain** over always-W1 (fraction fixed now; changed only by a recorded amendment if T1 shows the oracle is tiny or huge);
2. beats the **step-0-only detector** and the best fixed arm (selected on the test data, which is conservative), paired Wilcoxon over landscapes p < 0.05;
3. beats its label-shuffled control.
Secondary, pre-declared: same test with families as units (8 units: LCBench + 7 TabRepo families); per-seed labels; k = 5; m = n0/2. Exploratory only: a two-look sequential rule (look at k = 5, else k = 10, at most one injection).

## 6. Stages (stop and ask at each)
- **T0 done:** `meta/bo.py` `run(..., inject=(k, m, mode))`, `hedge_start`; `meta/test_inject.py` (trajectory identity up to k, budget, no duplicates, determinism, FPS first point, hedge start, refusal past budget).
- **T1:** pilot on 5 slowest landscapes (time), then the full run; cross-tab, realised oracle, hedge fraction (stop gates 1-3). Features are computed from the stored `obs_idx` of the W1 arm.
- **T2:** detector analysis and the gates.
- **T3:** 5 % noise block (W1, hedge, injection arms, same seeds) run only if T2 passes; scope limit stated otherwise (noise-free only; noise arms changed conclusions for K, R and W).

## 7. Risks
Realised headroom after k steps and m insertions can be much below the cost-free 25 %; the target-random reference is a simulation luxury; four seeds per landscape make per-seed labels noisy; conclusions are specific to m and to the two remedies tried.

## 8. Review gaps and decisions
G1, G2, G4, G5, G7, G8, G10, G11, G12, G13, G14, G16, G17, G18 adopted as written. G3 adopted, with the added point that the target's own random runs are not deployable, so a source-based reference is the deployable version. G6 not adopted as written: grouping by dataset name is stricter than grouping by landscape, because the same dataset appears in several model families; all seeds of a landscape sit in one fold in both. G9 adopted as a secondary. G15 adopted as a conditional T3 block, not run unconditionally, to save 2-3 h if the noise-free pass fails.

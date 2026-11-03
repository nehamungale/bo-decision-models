# Section 3 draft: stationary campaigns (2026-10-31)

All numbers are AUC of normalised regret, (y_max - best) / (y_max - y_median) averaged over the campaign, lower is better, unless stated. Sources: `APPROACHES_SO_FAR.md`, `notes.md`, `meta/results/`. Items marked [check] come from the early-phase reconstruction and need verification against `notes.md` before submission.

## 3.0 Protocol in one paragraph
Each campaign is pool-based BO on a real tabular landscape: a BoTorch ARD-RBF GP, LogEI unless the acquisition function (AF) is the object of study, 2D+1 initial points and min(50, 25 % of the pool) steps. Every comparison is paired by landscape and seed, uses a numeric baseline on the same state, and a shuffled-label or state-blind control. Gates (effect size and Wilcoxon p across landscapes) were fixed before the data were collected. Summaries are ratios of means, because mean-of-ratios blows up where the AUC is near zero. Where a setting was chosen on some seeds, it was scored on others, to remove winner's-curse bias.

## 3.1 Results table
| # | Component | Question | Evidence | Verdict |
|---|---|---|---|---|
| 1 | AF choice, typed zero-shot | Can a typed model (tev1 0.8B, nimble 9B, gpt-oss) choosing among six AFs beat a tuned rule or the best static AF? | Best static AF (TS) 0.1441 vs LogEI 0.1491 on 8 dev LCBench datasets; no typed mode beat the tuned rule or best static AF | Null |
| 2 | AF choice, learned numeric | Does a selector trained on counterfactual outcomes beat the best fixed AF? | 9,460 labelled states; 70 to 76 % of states have no signal; GBM gain +0.0010 over the best fixed AF, indistinguishable from the label-shuffled control | Null |
| 3 | AF choice, extra baselines | Thompson bandit, LogEI-then-LogPI schedule, minimum-dwell rule | bandit 0.1441 (= TS), schedule 0.1498, dwell best D = 1 (= tuned rule); Holm p = 1.0 vs TS and LogEI | Null |
| 4 | AF regime grid | Does the best AF transfer across landscapes? | In-sample best-worst spread within a landscape is 69 % of LogEI; the AF chosen on other landscapes is no better than random (-2.3 % overall, +42 % on HPO-B) | No learnable headroom (not "no AF effect") |
| 5 | Surrogate, search region | Per-landscape best kernel or box vs best fixed | Hindsight no better than best fixed, noise 0 and 5 % | Closed |
| 6 | **Warm start from related tasks** | Portfolio initial design vs random start | About 39 % lower AUC without noise, about 23 % at 5 % noise, on 150 exact-grid landscapes; controls (farthest-point start, mismatched portfolio) do not help | **Works** |
| 7 | Fall-back from the warm start | Predict where the portfolio loses to random start | Failures on 33 of 150 landscapes (always-W1 0.0329, random 0.0535, hindsight 0.0247); best predictor +3.6 % (p = 0.38), shuffled controls 0.0327 to 0.0413 | Not predictable |
| 8 | Online fall-back detection | Detect failure from the first steps, or insure against it | Hedge +13.3 %, injection +13.6 to +15.7 % vs always-W1; realised oracle 10.6 % vs 26.6 % cost-free; all detector cells fail (7 to 11 positives, low power) | Insurance costs more than it saves |
| 9 | Similarity-weighted source prior | Re-weight source tasks every step from observations | Tuning: worse than W1 on TabRepo in all 9 configurations. Diagnostic on 6 LCBench landscapes: -25 %. Pre-registered confirmation on 24 fresh LCBench landscapes: SP +7.0 %, frozen +3.4 %, uniform +0.7 % vs W1; all gates fail | Fails; the diagnostic gain did not replicate |

## 3.2 What the table says
The five per-step ideas (rows 1 to 4 and 9, and the detection in row 8) each failed against a fixed or static baseline of the same information; the surrogate and region (row 5) failed at the landscape level. The two positive results are both about what the campaign knows at step zero: the portfolio start (row 6) works, and attempts to improve it by deciding differently later (rows 7 to 9) do not.

## 3.3 Why the nulls are credible
- **Gates before data.** Thresholds, controls and stop rules were written into the plans first. Post-hoc steps (the Phase A override, the diagnostic after the T1 stop) are flagged where they occur.
- **Controls caught real artefacts.** Shuffled-label controls equalled the learned selector (row 2). The winner's-curse split shrank an apparent 12 % hindsight headroom to about 4.5 %. A 6-landscape tuning subset showed -25 % that vanished on fresh landscapes (row 9).
- **The effect is not absent, only unlearnable.** AF choice matters within a landscape (row 4); the winner does not transfer.

## 3.4 Limits specific to this section
- The warm start is the best case for transfer: shared configuration grids and 29 to 238 source tasks. A real lab rarely has this.
- Mostly noise-free. A 5 % noise arm exists for the surrogate, region, warm-start and fall-back studies, not for the AF and source-prior studies.
- Several early-phase numbers [check] come from the earliest sessions and should be re-read from `notes.md`.
- Reserved LCBench datasets, FCNet and NAS-Bench-201 were never touched; a final test for the warm-start claim is still open.
- The typed models are stand-ins, not Jev itself; none was fine-tuned on BO states.

## 3.5 Placeholder
The non-stationary section (`PLAN_nonstationary.md`) and the synthesis ("adaptivity pays at initialisation, except ...") wait for its result.

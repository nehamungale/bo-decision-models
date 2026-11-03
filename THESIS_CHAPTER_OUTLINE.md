# Thesis chapter outline: reactive BO with typed decision models (draft 2026-10-31)

Written for: your thesis committee, as an outline you edit. Numbers come from `APPROACHES_SO_FAR.md`, `notes.md` and `meta/results/*.txt`; check each against its source before quoting.

## Main claim
Across six acquisition, surrogate, region and transfer components on real tabular landscapes, per-step reactive decisions did not beat tuned fixed settings, and typed zero-shot decision models did not beat numeric baselines on the same state. The one adaptive component that paid was the warm start from related tasks, which acts before the campaign starts. Reactivity is therefore a question of regime: the stationary case is closed; the non-stationary case is the open test (section 7).

## 1. Introduction and research question
Reactive BO: change one component during a run so the optimum is found with lower regret per evaluation. Decision layer: typed Jev-style models (Noul, Choice, Score). Quality first, then speed; real lab later.

## 2. Setting and protocol
- Pool-based BO, GP surrogate, six AFs, metric (AUC of normalised regret), budgets.
- Testbeds: LCBench simulator, 1983 kept real landscapes; w150 set; family units.
- Stand-ins for Jev (tev1, nimble, gpt-oss) and their limits; open Jev-style implementations (Kev, SemIf) exist but were not benchmarked on BO states.
- Protocol: pre-registered gates, strong numeric baselines, shuffled and state-blind controls, A/B seed splits for winner's curse, ratio of means.

## 3. Results, stationary campaigns (a sequence of negative results)
| Section | Question | Verdict |
|---|---|---|
| 3.1 | Typed zero-shot AF choice vs tuned rule and best static AF | Null |
| 3.2 | Learned numeric and landscape-aware AF selectors, 9,460 labelled states | Null; equals shuffled control |
| 3.3 | Extra AF baselines: Thompson bandit, EI-to-PI schedule, dwell rule | Null |
| 3.4 | Regime grid: AF effect within a landscape is large, winner does not transfer | No learnable AF headroom |
| 3.5 | Surrogate and search-region choice | Closed at noise 0 and 5 % |
| 3.6 | Warm start from related tasks | **Passes**: about 39 % lower AUC (noise 0), 23 % (5 % noise) |
| 3.7 | Fall-back from the warm start; online detection | Not predictable; insurance costs more than it saves |
| 3.8 | Similarity-weighted source prior | T1 stop on TabRepo; LCBench diagnostic did not replicate (+7 % vs W1) |

## 4. Why the nulls (discussion)
- Noise-free, low-dimensional GP-on-pool: AF is the least differentiated component; fixed settings are hard to beat.
- Label sparsity: 70 to 76 % of counterfactual states have no signal.
- Gains live in what the model knows before the campaign (transfer), not in per-step choices.
- Selection effects seen in this work: winner's curse (12 % headroom shrank to about 4.5 %), tuning-set subset (LCBench -25 % did not replicate).

## 5. Limits
Best case for transfer (shared grids, many sources); mostly noise-free; simulator landscapes; stand-in models; reserved LCBench, FCNet and NAS-Bench-201 final tests not run; no batch, early-termination or algorithm-switching components.

## 6. Typed decision models: what was and was not shown
Zero-shot models read BO state fine on probes (nimble AUROC about 1.0 on six statements) but added no decision value. A fine-tuned or logit-read model over the same signal-free labels would not change this. Methods contribution: a reproducible decision-layer protocol with shuffled and state-blind controls.

## 7. Open test: non-stationary campaigns (`PLAN_nonstationary.md`, if run)
Gates N0 to N2; typed model compared with a numeric change detector. Insert results here, whatever they are.

## 8. Conclusion
Report the regime map: where reactivity pays (pre-campaign transfer; possibly drift) and where it does not.

## Open writing tasks
- Reconstruct section 1.1 to 1.4 of `APPROACHES_SO_FAR.md` (earlier phases) from `notes.md` before quoting.
- Figures: per-component headroom bars, warm-start gain by family, regime grid.
- Decide whether to run reserved final tests (LCBench CONFIRM, FCNet, NAS-Bench-201) for the warm-start claim only.

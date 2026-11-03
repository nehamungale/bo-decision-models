"""Pre-fixed LCBench dataset splits (rule: sort the 35 datasets by rho@10 ascending from
feasibility/lcbench_survey.csv; dev = ranks 2,8,14,..., confirmatory = ranks 5,11,17,...).
Chosen from curve statistics only, before any method was run."""
DEV = ["numerai28.6", "KDDCup09_appetency", "albert", "helena", "vehicle", "sylvine"]
CONFIRM = ["shuttle", "airlines", "MiniBooNE", "dionis", "fabert"]
# Added 2026-10-24 (before any decision-model run) because the rule-based dev set skews easy:
# kc1 and christine have the lowest early-stopping signal in the survey (rho@10 0.71, 0.78;
# top-10% lost at an epoch-10 cut 42%, 45%). Neither is in CONFIRM.
DEV = DEV + ["kc1", "christine"]

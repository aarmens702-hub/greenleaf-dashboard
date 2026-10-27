---
name: Experiment
about: A pre-registered training or tuning run, written in full before the first run
labels: task, experiment
---

## Type
experiment

## Question
<!-- The one question this answers, in plain words. -->

## Measurement
<!-- Every field filled before filing. /experiment refuses an incomplete one. Never edited after registration. -->
- **Hypothesis:**
- **Arms:** <!-- each: name, method (ppo_paper | ppo | residual | residual_lag | rule_based), config diff from control/configs/ppo.yml -->
- **Seeds:** <!-- e.g. 0-4 -->
- **Timesteps per run:** <!-- the same for every arm unless stated and justified -->
- **Scenarios:** <!-- validation or train_eval only; never a 2012 test split -->
- **Primary metric:** <!-- e.g. IQM profit €/m² over seeds × validation scenarios -->
- **Guard metrics:** <!-- e.g. violation-hours per constraint ≤ tuned RB's; solver failures = 0 -->
- **Baselines in the comparison:** <!-- e.g. tuned RB, do-nothing, random -->
- **Decision rule:** <!-- e.g. adopt arm X if its IQM 95% CI lies entirely above tuned RB's profit and every guard holds -->
- **Budget:** <!-- CPU-hours and dollars (worst case); the backend (modal or local) -->

## Code needed first
<!-- Files a builder must change before the run, or "none". -->
-

## Done when
- [ ] Registered, run and resulted in the ledger (row ids in a comment)
- [ ] Every registered run accounted for (finished, crashed or budget-stopped)
- [ ] Reviewer checked diagnostics and one trajectory per arm
- [ ] Aarmen's decision recorded

## Out of scope
-

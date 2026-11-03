Made by `uv run python -m greenleaf_control.hypotheses` (in control/) from control/results/final/final-registered-429132de/seasons.parquet, the 2013 final (final-result-7d2876de; 2013 test weather, scored once). Each 2013 split is graded alone; nothing is pooled. Every number is in hypotheses.csv at full precision.

The hypotheses, registered as issue #73's text (plan approved 2026-11-02), word for word (graded per split; verified only if every part holds on that split):

Conventions: like-for-like baselines = the baseline's 8 seasons tiled to the arm's seed count through stats.iqm_ci; a paired gain = stats.iqm_ci(arm n×8 − baseline 1×8), same seasons; "CI above/below X" = the 95% CI's low end > X / high end < X; Σ = the sum of the three per-constraint hour IQMs (hours don't depend on prices).

- H1 (the price regime reverses the ranking): ppo's paired profit gain over rb_default has its CI entirely below 0 at agc2020 AND entirely above 0 at sim.
- H2 (the residual agent's edge is robust to prices): residual_w05's paired profit gain over rb_tuned has its CI entirely above 0 at sim AND at agc2020.
- H3 (price-matched RL beats the price-matched rules): for each of ppo_agc and residual_agc, at agc2020: (a) its profit IQM CI lies entirely above max(rb_default, rb_tuned, rb_tuned_agc2020)'s like-for-like profit IQM; (b) its Σ ≤ rb_tuned's like-for-like Σ (no slack); (c) 0 invalid, 0 solver failures.
- H4 (replication of #45 on fresh weather): #45's rule at sim for residual_w05 and lag_b100: (a) profit CI above rb_tuned's like-for-like profit IQM; (b) Σ ≤ rb_tuned's like-for-like Σ; (c) 0 invalid, 0 solver failures.

- A part reads value, relation, bar: it passes when the relation holds. H1's and H2's parts are named by their price regime: the value is the paired profit gain's 95% CI high end (below 0) or low end (above 0), EUR/m².
- a: the arm's profit IQM 95% CI low end vs the like-for-like baseline profit IQM, EUR/m². H3(a)'s bar is the largest of the three baselines' like-for-like profit IQMs, each through stats.iqm_ci(its one-run row tiled to the arm's n runs)[0]; H4(a)'s is rb_tuned's, the same way.
- b: Σ of the CO₂, temp and RH violation-hour IQMs vs rb_tuned's like-for-like Σ, h.
- c: the arm's invalid seasons and its solver failures at the regime graded; it passes when both are 0.
- 95% stratified-bootstrap CIs over the arm's n runs x 8 seasons. A baseline is deterministic, one run. A verdict on fewer than 5 runs is flagged n<5: no claim.

| split | hypothesis | arm | n | verdict | parts |
|---|---|---|---|---|---|
| amsterdam_2013 | H1 | ppo | 5 | verified | agc2020: -3.752 < 0.000 (pass); sim: 7.005 > 0.000 (pass) |
| amsterdam_2013 | H2 | residual_w05 | 10 | verified | sim: 0.340 > 0.000 (pass); agc2020: 0.397 > 0.000 (pass) |
| amsterdam_2013 | H3 | ppo_agc | 10 | verified | a: 15.778 > 12.541 (pass); b: 219.763 ≤ 301.188 (pass); c: invalid 0, solver failures 0 (pass) |
| amsterdam_2013 | H3 | residual_agc | 10 | fails (a) | a: 10.390 > 12.541 (fail); b: 169.450 ≤ 301.188 (pass); c: invalid 0, solver failures 0 (pass) |
| amsterdam_2013 | H4 | residual_w05 | 10 | verified | a: 0.401 > 0.074 (pass); b: 250.194 ≤ 301.188 (pass); c: invalid 0, solver failures 0 (pass) |
| amsterdam_2013 | H4 | lag_b100 | 10 | fails (b) | a: 0.526 > 0.074 (pass); b: 322.013 ≤ 301.188 (fail); c: invalid 0, solver failures 0 (pass) |
| london_2013 | H1 | ppo | 5 | verified | agc2020: -4.387 < 0.000 (pass); sim: 10.963 > 0.000 (pass) |
| london_2013 | H2 | residual_w05 | 10 | verified | sim: 0.408 > 0.000 (pass); agc2020: 0.300 > 0.000 (pass) |
| london_2013 | H3 | ppo_agc | 10 | verified | a: 14.606 > 11.856 (pass); b: 237.269 ≤ 322.812 (pass); c: invalid 0, solver failures 0 (pass) |
| london_2013 | H3 | residual_agc | 10 | fails (a) | a: 9.282 > 11.856 (fail); b: 187.450 ≤ 322.812 (pass); c: invalid 0, solver failures 0 (pass) |
| london_2013 | H4 | residual_w05 | 10 | verified | a: -2.369 > -2.766 (pass); b: 267.494 ≤ 322.812 (pass); c: invalid 0, solver failures 0 (pass) |
| london_2013 | H4 | lag_b100 | 10 | fails (b) | a: -2.244 > -2.766 (pass); b: 347.606 ≤ 322.812 (fail); c: invalid 0, solver failures 0 (pass) |
| abbotsford_2013 | H1 | ppo | 5 | verified | agc2020: -3.977 < 0.000 (pass); sim: 7.592 > 0.000 (pass) |
| abbotsford_2013 | H2 | residual_w05 | 10 | verified | sim: 0.279 > 0.000 (pass); agc2020: 0.230 > 0.000 (pass) |
| abbotsford_2013 | H3 | ppo_agc | 10 | verified | a: 14.527 > 12.022 (pass); b: 482.625 ≤ 534.000 (pass); c: invalid 0, solver failures 0 (pass) |
| abbotsford_2013 | H3 | residual_agc | 10 | fails (a) | a: 10.283 > 12.022 (fail); b: 309.263 ≤ 534.000 (pass); c: invalid 0, solver failures 0 (pass) |
| abbotsford_2013 | H4 | residual_w05 | 10 | verified | a: 0.269 > -0.017 (pass); b: 415.519 ≤ 534.000 (pass); c: invalid 0, solver failures 0 (pass) |
| abbotsford_2013 | H4 | lag_b100 | 10 | verified | a: 0.412 > -0.017 (pass); b: 507.944 ≤ 534.000 (pass); c: invalid 0, solver failures 0 (pass) |

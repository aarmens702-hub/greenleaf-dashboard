Made by `uv run python -m greenleaf_control.report --split abbotsford_2012` (in control/) from control/results/final/final-registered-4d0a4ba9/seasons.parquet, checked against final-result-aee23915 (2012 test weather, scored once); every number is in summary.csv, at full precision.

IQM [95% stratified-bootstrap CI] over each row's runs x seasons. Profit and gain in EUR/m², hours in h, yield in kg/m². rb_tuned like-for-like: rb_tuned's one-run row repeated to the row's n runs, then its IQM. Gain and hours saved pair each season with rb_tuned's same season; P is the share of same-season wins over rb_tuned, ties half.

- Total violation hours: the IQM of each season's CO₂ + temp + RH hours, with a CI.
- Σ per-constraint IQMs: the CO₂, temp and RH IQMs added up, the number the registered rule (b) of E2 to E7 (#31 to #39) uses. It has no CI; rb_tuned's like-for-like Σ is in parentheses.
- Gain and hours saved are IQMs of per-season differences, not differences of IQMs: hours saved is not rb_tuned's hours minus the row's, in either hours column.
- decision: the result's single decision row, shown on every arm of its registration, so an arm the decision note drops can still read "adopted". Read the note.
- A 1-run row's CI is zero-width because there is one run, not because the value is certain. That matters for random, which is stochastic.
- PROTOCOL: the first 8 hex of the docs/PROTOCOL.md hash at registration. Rows under different hashes aren't fingerprint-comparable (#38).
- rb_tuned's own row compares it with itself (gain 0, P 0.5).

| label | decision | PROTOCOL | n | crashed | invalid | profit IQM [95% CI] | rb_tuned like-for-like | gain [CI] | P(profit > rb_tuned) | total violation hours IQM [CI] | Σ per-constraint IQMs (rb_tuned like-for-like) | hours saved [CI] | P(fewer hours) | yield IQM [CI] | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_controls |  |  | 1 |  | 0 | 1.907 [1.907, 1.907] | -3.315 | 5.293 [5.293, 5.293] | 1.000 | 3007.94 [3007.94, 3007.94] | 3013.12 (559.81) | -2453.44 [-2453.44, -2453.44] | 0.000 | 1.192 [1.192, 1.192] | deterministic, 1 run |
| random |  |  | 1 |  | 3 | -30.282 [-30.282, -30.282] | -3.315 | -27.185 [-27.185, -27.185] | 0.000 | 1633.81 [1633.81, 1633.81] | 1638.81 (559.81) | -1066.44 [-1066.44, -1066.44] | 0.000 | 4.120 [4.120, 4.120] | n<5: no claim |
| rb_default |  |  | 1 |  | 0 | -12.518 [-12.518, -12.518] | -3.315 | -9.203 [-9.203, -9.203] | 0.000 | 922.12 [922.12, 922.12] | 923.31 (559.81) | -363.56 [-363.56, -363.56] | 0.000 | 11.342 [11.342, 11.342] | deterministic, 1 run |
| rb_tuned |  |  | 1 |  | 0 | -3.315 [-3.315, -3.315] | -3.315 | 0.000 [0.000, 0.000] | 0.500 | 558.56 [558.56, 558.56] | 559.81 (559.81) | 0.00 [0.00, 0.00] | 0.500 | 8.282 [8.282, 8.282] | deterministic, 1 run |
| ppo_paper | adopted decision-471d1e0e | 849d5407 | 5 |  | 0 | 0.076 [-0.124, 0.281] | -3.315 | 3.362 [3.196, 3.528] | 1.000 | 264.68 [229.86, 329.84] | 256.91 (559.81) | 295.84 [233.75, 332.56] | 0.850 | 5.653 [5.554, 5.739] |  |
| ppo | adopted decision-b1319a3a | 849d5407 | 5 |  | 0 | -0.064 [-0.139, 0.014] | -3.315 | 3.253 [3.180, 3.326] | 1.000 | 230.30 [218.95, 245.62] | 232.81 (559.81) | 326.07 [311.39, 337.01] | 1.000 | 5.646 [5.621, 5.670] |  |
| residual_w05 | adopted decision-eab26197 | 849d5407 | 10 |  | 0 | -2.797 [-2.860, -2.735] | -3.315 | 0.540 [0.467, 0.605] | 1.000 | 445.54 [440.02, 450.89] | 444.32 (559.81) | 110.42 [105.32, 115.24] | 1.000 | 8.423 [8.407, 8.439] |  |
| lag_b100 | adopted decision-960905e6 | af2f687e | 10 |  | 0 | -2.655 [-2.705, -2.601] | -3.315 | 0.633 [0.587, 0.669] | 1.000 | 552.16 [547.11, 557.38] | 551.45 (559.81) | 5.74 [0.15, 11.62] | 0.575 | 8.429 [8.404, 8.453] |  |

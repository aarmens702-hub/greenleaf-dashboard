Made by `uv run python -m greenleaf_control.report --split amsterdam_2012` (in control/) from control/results/final/final-registered-4d0a4ba9/seasons.parquet, checked against final-result-aee23915 (2012 test weather, scored once); every number is in summary.csv, at full precision.

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
| zero_controls |  |  | 1 |  | 0 | 3.103 [3.103, 3.103] | -1.918 | 5.020 [5.020, 5.020] | 1.000 | 2978.38 [2978.38, 2978.38] | 2977.50 (365.19) | -2610.44 [-2610.44, -2610.44] | 0.000 | 1.939 [1.939, 1.939] | deterministic, 1 run |
| random |  |  | 1 |  | 0 | -32.369 [-32.369, -32.369] | -1.918 | -30.054 [-30.054, -30.054] | 0.000 | 1761.38 [1761.38, 1761.38] | 1755.81 (365.19) | -1401.44 [-1401.44, -1401.44] | 0.000 | 3.173 [3.173, 3.173] | n<5: no claim |
| rb_default |  |  | 1 |  | 0 | -9.343 [-9.343, -9.343] | -1.918 | -7.426 [-7.426, -7.426] | 0.000 | 763.94 [763.94, 763.94] | 761.88 (365.19) | -400.00 [-400.00, -400.00] | 0.000 | 10.977 [10.977, 10.977] | deterministic, 1 run |
| rb_tuned |  |  | 1 |  | 0 | -1.918 [-1.918, -1.918] | -1.918 | 0.000 [0.000, 0.000] | 0.500 | 366.25 [366.25, 366.25] | 365.19 (365.19) | 0.00 [0.00, 0.00] | 0.500 | 8.588 [8.588, 8.588] | deterministic, 1 run |
| ppo_paper | adopted decision-471d1e0e | 849d5407 | 5 |  | 0 | 0.667 [0.556, 0.776] | -1.918 | 2.638 [2.517, 2.730] | 1.000 | 255.55 [240.32, 274.64] | 253.81 (365.19) | 110.72 [92.05, 124.06] | 0.875 | 5.936 [5.852, 6.005] |  |
| ppo | adopted decision-b1319a3a | 849d5407 | 5 |  | 0 | 0.803 [0.742, 0.860] | -1.918 | 2.735 [2.687, 2.782] | 1.000 | 261.54 [254.46, 271.39] | 260.79 (365.19) | 103.40 [93.60, 111.05] | 1.000 | 5.955 [5.929, 5.980] |  |
| residual_w05 | adopted decision-eab26197 | 849d5407 | 10 |  | 0 | -1.638 [-1.694, -1.585] | -1.918 | 0.304 [0.252, 0.353] | 0.913 | 292.00 [288.50, 295.32] | 292.56 (365.19) | 75.56 [71.82, 79.18] | 1.000 | 8.766 [8.752, 8.780] |  |
| lag_b100 | adopted decision-960905e6 | af2f687e | 10 |  | 0 | -1.507 [-1.558, -1.457] | -1.918 | 0.375 [0.347, 0.415] | 1.000 | 400.96 [394.15, 407.03] | 399.93 (365.19) | -33.06 [-39.13, -26.27] | 0.188 | 8.762 [8.745, 8.778] |  |

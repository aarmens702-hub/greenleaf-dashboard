Made by `uv run python -m greenleaf_control.report --split london_2012` (in control/) from control/results/final/final-registered-4d0a4ba9/seasons.parquet, checked against final-result-aee23915 (2012 test weather, scored once); every number is in summary.csv, at full precision.

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
| zero_controls |  |  | 1 |  | 0 | 3.211 [3.211, 3.211] | -0.087 | 3.306 [3.306, 3.306] | 1.000 | 3046.00 [3046.00, 3046.00] | 3058.69 (456.94) | -2590.00 [-2590.00, -2590.00] | 0.000 | 2.007 [2.007, 2.007] | deterministic, 1 run |
| random |  |  | 1 |  | 0 | -30.792 [-30.792, -30.792] | -0.087 | -30.069 [-30.069, -30.069] | 0.000 | 1777.88 [1777.88, 1777.88] | 1775.19 (456.94) | -1325.88 [-1325.88, -1325.88] | 0.000 | 4.158 [4.158, 4.158] | n<5: no claim |
| rb_default |  |  | 1 |  | 0 | -6.555 [-6.555, -6.555] | -0.087 | -6.469 [-6.469, -6.469] | 0.000 | 767.31 [767.31, 767.31] | 766.38 (456.94) | -312.31 [-312.31, -312.31] | 0.000 | 11.000 [11.000, 11.000] | deterministic, 1 run |
| rb_tuned |  |  | 1 |  | 0 | -0.087 [-0.087, -0.087] | -0.087 | 0.000 [0.000, 0.000] | 0.500 | 456.62 [456.62, 456.62] | 456.94 (456.94) | 0.00 [0.00, 0.00] | 0.500 | 8.889 [8.889, 8.889] | deterministic, 1 run |
| ppo_paper | adopted decision-471d1e0e | 849d5407 | 5 |  | 0 | 1.808 [1.671, 1.954] | -0.087 | 2.080 [1.922, 2.230] | 1.000 | 242.00 [218.03, 273.09] | 227.06 (456.94) | 214.07 [182.77, 237.18] | 1.000 | 6.380 [6.297, 6.455] |  |
| ppo | adopted decision-b1319a3a | 849d5407 | 5 |  | 0 | 1.896 [1.837, 1.959] | -0.087 | 2.080 [2.020, 2.143] | 1.000 | 239.99 [228.06, 255.45] | 239.25 (456.94) | 215.19 [200.50, 227.04] | 1.000 | 6.407 [6.387, 6.430] |  |
| residual_w05 | adopted decision-eab26197 | 849d5407 | 10 |  | 0 | 0.251 [0.204, 0.297] | -0.087 | 0.396 [0.350, 0.438] | 1.000 | 340.79 [336.46, 346.26] | 343.07 (456.94) | 114.49 [108.48, 119.71] | 1.000 | 9.153 [9.140, 9.165] |  |
| lag_b100 | adopted decision-960905e6 | af2f687e | 10 |  | 0 | 0.381 [0.351, 0.414] | -0.087 | 0.460 [0.433, 0.490] | 1.000 | 445.28 [441.64, 449.32] | 444.16 (456.94) | 10.32 [6.42, 14.13] | 0.762 | 9.132 [9.111, 9.152] |  |

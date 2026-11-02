Made by `uv run python -m greenleaf_control.report --split validation` (in control/) from the ledger, the runs' selected.json and evaluate's baseline parquets; every number is in summary.csv, at full precision.

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
| zero_controls |  |  | 1 |  | 0 | 2.949 [2.949, 2.949] | -0.517 | 3.466 [3.466, 3.466] | 1.000 | 2942.42 [2942.42, 2942.42] | 2942.42 (355.67) | -2586.75 [-2586.75, -2586.75] | 0.000 | 1.843 [1.843, 1.843] | deterministic, 1 run |
| random |  |  | 1 |  | 0 | -30.308 [-30.308, -30.308] | -0.517 | -29.791 [-29.791, -29.791] | 0.000 | 1652.92 [1652.92, 1652.92] | 1652.92 (355.67) | -1297.25 [-1297.25, -1297.25] | 0.000 | 4.409 [4.409, 4.409] | n<5: no claim |
| rb_default |  |  | 1 |  | 0 | -7.414 [-7.414, -7.414] | -0.517 | -6.897 [-6.897, -6.897] | 0.000 | 743.17 [743.17, 743.17] | 743.17 (355.67) | -387.50 [-387.50, -387.50] | 0.000 | 11.338 [11.338, 11.338] | deterministic, 1 run |
| rb_tuned | adopted decision-b2221004 | dcfd7a58 | 1 |  | 0 | -0.517 [-0.517, -0.517] | -0.517 | 0.000 [0.000, 0.000] | 0.500 | 355.67 [355.67, 355.67] | 355.67 (355.67) | 0.00 [0.00, 0.00] | 0.500 | 9.064 [9.064, 9.064] | deterministic, 1 run |
| ppo_paper #31/2d02 | adopted decision-471d1e0e | 849d5407 | 5 | 0 | 0 | 2.299 [2.146, 2.468] | 0.092 | 2.320 [2.155, 2.488] | 1.000 | 255.94 [238.05, 277.42] | 251.44 (353.44) | 95.58 [71.86, 121.03] | 1.000 | 6.801 [6.626, 6.937] |  |
| ppo #33/0221 | adopted decision-b1319a3a | 849d5407 | 5 | 0 | 0 | 2.395 [2.327, 2.458] | 0.092 | 2.399 [2.335, 2.489] | 1.000 | 295.36 [275.55, 310.53] | 289.75 (353.44) | 56.72 [41.55, 75.31] | 1.000 | 6.690 [6.652, 6.730] |  |
| residual_w05 #34/4d20 | adopted decision-b159ac9a | 849d5407 | 5 | 0 | 0 | 0.629 [0.482, 0.740] | 0.092 | 0.437 [0.343, 0.567] | 1.000 | 266.72 [259.05, 274.45] | 265.25 (353.44) | 94.72 [84.39, 102.28] | 1.000 | 9.572 [9.545, 9.604] |  |
| residual_w1 #34/4d20 | adopted decision-b159ac9a | 849d5407 | 5 | 0 | 0 | 0.396 [0.321, 0.449] | 0.092 | 0.225 [0.161, 0.301] | 0.933 | 130.83 [119.72, 140.08] | 129.22 (353.44) | 224.86 [216.94, 233.53] | 1.000 | 9.572 [9.541, 9.603] |  |
| residual_w4 #34/4d20 | adopted decision-b159ac9a | 849d5407 | 5 | 0 | 0 | -0.367 [-0.482, -0.265] | 0.092 | -0.579 [-0.645, -0.475] | 0.000 | 43.56 [40.69, 46.86] | 42.42 (353.44) | 307.86 [304.39, 312.14] | 1.000 | 9.585 [9.559, 9.615] |  |
| lag_b100 #35/976c | dropped decision-c26a2bc9 | 849d5407 | 5 | 0 | 0 | 0.258 [0.114, 0.428] | 0.092 | 0.216 [0.100, 0.325] | 0.800 | 278.75 [250.35, 298.86] | 277.31 (353.44) | 78.47 [65.63, 97.06] | 1.000 | 9.503 [9.444, 9.561] |  |
| lag_b50 #35/976c | dropped decision-c26a2bc9 | 849d5407 | 5 | 0 | 0 | 0.175 [0.113, 0.252] | 0.092 | 0.093 [0.011, 0.178] | 0.667 | 174.28 [158.22, 189.08] | 174.00 (353.44) | 185.42 [169.50, 202.25] | 1.000 | 9.536 [9.505, 9.568] |  |
| lag_b25 #35/976c | dropped decision-c26a2bc9 | 849d5407 | 5 | 0 | 0 | -0.218 [-0.290, -0.152] | 0.092 | -0.356 [-0.389, -0.293] | 0.000 | 96.25 [92.47, 100.61] | 96.19 (353.44) | 261.11 [256.28, 265.42] | 1.000 | 9.581 [9.559, 9.603] |  |
| residual_w05 #36/f2fd | adopted decision-eab26197 | 849d5407 | 10 | 0 | 0 | 0.765 [0.702, 0.836] | 0.282 | 0.492 [0.403, 0.570] | 1.000 | 262.97 [257.25, 269.00] | 262.56 (352.75) | 92.66 [86.52, 99.11] | 1.000 | 9.625 [9.602, 9.645] |  |
| lag_b100 #35/157d | dropped decision-3a99310a | af2f687e | 4 | 1 | 0 | 1.200 [1.093, 1.274] | 0.396 | 0.742 [0.654, 0.839] | 1.000 | 384.92 [376.67, 393.12] | 385.08 (352.33) | -29.46 [-33.75, -25.42] | 0.083 | 9.662 [9.635, 9.691] | n<5: no claim |
| lag_b50 #35/157d | dropped decision-3a99310a | af2f687e | 5 | 0 | 0 | 0.543 [0.448, 0.615] | 0.092 | 0.393 [0.323, 0.470] | 1.000 | 220.72 [210.30, 232.50] | 221.56 (353.44) | 133.06 [120.42, 145.58] | 1.000 | 9.573 [9.547, 9.598] |  |
| lag_b25 #35/157d | dropped decision-3a99310a | af2f687e | 5 | 0 | 0 | -0.099 [-0.188, -0.032] | 0.092 | -0.272 [-0.343, -0.188] | 0.067 | 87.19 [80.47, 93.70] | 86.33 (353.44) | 267.33 [262.47, 272.42] | 1.000 | 9.609 [9.589, 9.626] |  |
| lag_b100 #39/6d92 | adopted decision-960905e6 | af2f687e | 10 | 0 | 0 | 0.937 [0.856, 1.003] | 0.282 | 0.569 [0.517, 0.636] | 1.000 | 363.31 [356.47, 370.77] | 362.28 (352.75) | -10.55 [-17.52, -1.44] | 0.300 | 9.609 [9.573, 9.640] |  |

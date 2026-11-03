Made by `uv run python -m greenleaf_control.prices` (in control/): exploratory, re-priced recorded seasons, post hoc. No season is played and the 2012 seal is not reopened; every number is in regimes.csv, sweep.csv and crossovers.csv at full precision.

Sources: control/results/analysis/validation/scores.parquet; control/results/final/final-registered-4d0a4ba9/seasons.parquet. The 2012 seasons are the final's (2012 test weather, scored once).

Prices (fruit EUR/kg, heat and elec EUR/kWh, CO₂ EUR/kg; dry matter 0.065 throughout):
- sim: fruit 1.6, heat 0.09, elec 0.3, co2 0.3; the simulator's (PROTOCOL §1), at which every season was recorded.
- agc2020: fruit 1.6, heat 0.03, elec 0.06, co2 0.08; heat 0.03 €/kWh and CO₂ 0.08 €/kg (its ≤ 12 kg/m² tier) from the 2nd Autonomous Greenhouse Challenge's 2020 net-profit scoring (Hemming et al. 2020, Sensors 20(22):6430); electricity 0.06 €/kWh, the unweighted mean of its 0.08 on-peak (07-23 h) and 0.04 off-peak tariffs, flat, not time-of-use; fruit 1.6 €/kg and dmfm 0.065 are the simulator's.

- IQM [95% stratified-bootstrap CI] over each controller's runs x seasons; profit and gains in EUR/m², hours in h, yield in kg/m². rank 1 is the highest profit IQM in its split and regime.
- A gain pairs each season with rb_tuned's or rb_default's same season. rb_tuned, rb_default and zero_controls are deterministic, one run each, so a baseline-vs-baseline gain has a zero-width CI: one run, not certainty.
- Σ hours (each season's CO₂ + temp + RH violation hours) and yield don't move with price.
- random is left out: noise, not a controller (3 invalid Abbotsford seasons).

## validation (checkpoints were selected on these)

| controller | n | Σ hours IQM | yield IQM | sim rank | sim profit | sim gain_rb_tuned | sim gain_rb_default | agc2020 rank | agc2020 profit | agc2020 gain_rb_tuned | agc2020 gain_rb_default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_controls | 1 | 2942.42 | 1.843 | 1 | 2.949 [2.949, 2.949] | 3.466 [3.466, 3.466] | 10.362 [10.362, 10.362] | 7 | 2.949 [2.949, 2.949] | -7.170 [-7.170, -7.170] | -8.806 [-8.806, -8.806] |
| ppo | 5 | 295.36 | 6.690 | 2 | 2.395 [2.327, 2.458] | 2.399 [2.335, 2.489] | 8.913 [8.836, 8.985] | 6 | 7.927 [7.881, 7.975] | -2.508 [-2.549, -2.465] | -4.065 [-4.105, -4.021] |
| ppo_paper | 5 | 255.94 | 6.801 | 3 | 2.299 [2.146, 2.468] | 2.320 [2.155, 2.488] | 8.839 [8.661, 9.006] | 5 | 8.036 [7.796, 8.237] | -2.489 [-2.714, -2.257] | -4.042 [-4.254, -3.876] |
| lag_b100 | 10 | 363.31 | 9.609 | 4 | 0.937 [0.856, 1.003] | 0.569 [0.517, 0.636] | 7.125 [7.060, 7.186] | 2 | 11.118 [11.071, 11.157] | 0.591 [0.539, 0.631] | -0.940 [-0.993, -0.900] |
| residual_w05 | 10 | 262.97 | 9.625 | 5 | 0.765 [0.702, 0.836] | 0.492 [0.403, 0.570] | 6.981 [6.915, 7.056] | 3 | 11.084 [11.048, 11.111] | 0.557 [0.519, 0.586] | -0.974 [-1.012, -0.946] |
| rb_tuned | 1 | 355.67 | 9.064 | 6 | -0.517 [-0.517, -0.517] | 0.000 [0.000, 0.000] | 6.897 [6.897, 6.897] | 4 | 10.118 [10.118, 10.118] | 0.000 [0.000, 0.000] | -1.636 [-1.636, -1.636] |
| rb_default | 1 | 743.17 | 11.338 | 7 | -7.414 [-7.414, -7.414] | -6.897 [-6.897, -6.897] | 0.000 [0.000, 0.000] | 1 | 11.754 [11.754, 11.754] | 1.636 [1.636, 1.636] | 0.000 [0.000, 0.000] |

## amsterdam_2012 (2012 test weather, scored once)

| controller | n | Σ hours IQM | yield IQM | sim rank | sim profit | sim gain_rb_tuned | sim gain_rb_default | agc2020 rank | agc2020 profit | agc2020 gain_rb_tuned | agc2020 gain_rb_default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_controls | 1 | 2978.38 | 1.939 | 1 | 3.103 [3.103, 3.103] | 5.020 [5.020, 5.020] | 12.446 [12.446, 12.446] | 7 | 3.103 [3.103, 3.103] | -6.095 [-6.095, -6.095] | -7.795 [-7.795, -7.795] |
| ppo | 5 | 261.54 | 5.955 | 2 | 0.803 [0.742, 0.860] | 2.735 [2.687, 2.782] | 10.146 [10.085, 10.203] | 5 | 6.621 [6.597, 6.645] | -2.583 [-2.605, -2.561] | -4.207 [-4.231, -4.182] |
| ppo_paper | 5 | 255.55 | 5.936 | 3 | 0.667 [0.556, 0.776] | 2.638 [2.517, 2.730] | 10.010 [9.899, 10.120] | 6 | 6.560 [6.430, 6.665] | -2.612 [-2.755, -2.500] | -4.300 [-4.413, -4.190] |
| lag_b100 | 10 | 400.96 | 8.762 | 4 | -1.507 [-1.558, -1.457] | 0.375 [0.347, 0.415] | 7.834 [7.784, 7.886] | 2 | 9.508 [9.492, 9.522] | 0.323 [0.312, 0.332] | -1.320 [-1.335, -1.306] |
| residual_w05 | 10 | 292.00 | 8.766 | 5 | -1.638 [-1.694, -1.585] | 0.304 [0.252, 0.353] | 7.701 [7.647, 7.757] | 3 | 9.460 [9.447, 9.472] | 0.276 [0.265, 0.288] | -1.363 [-1.376, -1.352] |
| rb_tuned | 1 | 366.25 | 8.588 | 6 | -1.918 [-1.918, -1.918] | 0.000 [0.000, 0.000] | 7.426 [7.426, 7.426] | 4 | 9.196 [9.196, 9.196] | 0.000 [0.000, 0.000] | -1.630 [-1.630, -1.630] |
| rb_default | 1 | 763.94 | 10.977 | 7 | -9.343 [-9.343, -9.343] | -7.426 [-7.426, -7.426] | 0.000 [0.000, 0.000] | 1 | 10.822 [10.822, 10.822] | 1.630 [1.630, 1.630] | 0.000 [0.000, 0.000] |

## london_2012 (2012 test weather, scored once)

| controller | n | Σ hours IQM | yield IQM | sim rank | sim profit | sim gain_rb_tuned | sim gain_rb_default | agc2020 rank | agc2020 profit | agc2020 gain_rb_tuned | agc2020 gain_rb_default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_controls | 1 | 3046.00 | 2.007 | 1 | 3.211 [3.211, 3.211] | 3.306 [3.306, 3.306] | 9.767 [9.767, 9.767] | 7 | 3.211 [3.211, 3.211] | -6.812 [-6.812, -6.812] | -8.277 [-8.277, -8.277] |
| ppo | 5 | 239.99 | 6.407 | 2 | 1.896 [1.837, 1.959] | 2.080 [2.020, 2.143] | 8.489 [8.425, 8.548] | 5 | 7.469 [7.441, 7.496] | -2.561 [-2.585, -2.540] | -3.982 [-4.009, -3.953] |
| ppo_paper | 5 | 242.00 | 6.380 | 3 | 1.808 [1.671, 1.954] | 2.080 [1.922, 2.230] | 8.490 [8.341, 8.628] | 6 | 7.414 [7.279, 7.538] | -2.556 [-2.733, -2.410] | -4.079 [-4.208, -3.949] |
| lag_b100 | 10 | 445.28 | 9.132 | 4 | 0.381 [0.351, 0.414] | 0.460 [0.433, 0.490] | 6.953 [6.919, 6.988] | 2 | 10.451 [10.422, 10.477] | 0.429 [0.417, 0.442] | -1.051 [-1.079, -1.024] |
| residual_w05 | 10 | 340.79 | 9.153 | 5 | 0.251 [0.204, 0.297] | 0.396 [0.350, 0.438] | 6.835 [6.792, 6.873] | 3 | 10.428 [10.414, 10.442] | 0.392 [0.375, 0.405] | -1.069 [-1.085, -1.054] |
| rb_tuned | 1 | 456.62 | 8.889 | 6 | -0.087 [-0.087, -0.087] | 0.000 [0.000, 0.000] | 6.469 [6.469, 6.469] | 4 | 10.034 [10.034, 10.034] | 0.000 [0.000, 0.000] | -1.474 [-1.474, -1.474] |
| rb_default | 1 | 767.31 | 11.000 | 7 | -6.555 [-6.555, -6.555] | -6.469 [-6.469, -6.469] | 0.000 [0.000, 0.000] | 1 | 11.423 [11.423, 11.423] | 1.474 [1.474, 1.474] | 0.000 [0.000, 0.000] |

## abbotsford_2012 (2012 test weather, scored once)

| controller | n | Σ hours IQM | yield IQM | sim rank | sim profit | sim gain_rb_tuned | sim gain_rb_default | agc2020 rank | agc2020 profit | agc2020 gain_rb_tuned | agc2020 gain_rb_default |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero_controls | 1 | 3007.94 | 1.192 | 1 | 1.907 [1.907, 1.907] | 5.293 [5.293, 5.293] | 14.496 [14.496, 14.496] | 7 | 1.907 [1.907, 1.907] | -6.574 [-6.574, -6.574] | -8.758 [-8.758, -8.758] |
| ppo_paper | 5 | 264.68 | 5.653 | 2 | 0.076 [-0.124, 0.281] | 3.362 [3.196, 3.528] | 12.594 [12.389, 12.800] | 5 | 6.062 [5.883, 6.217] | -2.449 [-2.632, -2.273] | -4.642 [-4.806, -4.516] |
| ppo | 5 | 230.30 | 5.646 | 3 | -0.064 [-0.139, 0.014] | 3.253 [3.180, 3.326] | 12.454 [12.379, 12.532] | 6 | 6.002 [5.968, 6.033] | -2.556 [-2.586, -2.526] | -4.734 [-4.767, -4.702] |
| lag_b100 | 10 | 552.16 | 8.429 | 4 | -2.655 [-2.705, -2.601] | 0.633 [0.587, 0.669] | 9.863 [9.814, 9.917] | 2 | 8.902 [8.871, 8.931] | 0.346 [0.322, 0.369] | -1.857 [-1.888, -1.826] |
| residual_w05 | 10 | 445.54 | 8.423 | 5 | -2.797 [-2.860, -2.735] | 0.540 [0.467, 0.605] | 9.721 [9.658, 9.783] | 3 | 8.839 [8.820, 8.856] | 0.285 [0.265, 0.301] | -1.909 [-1.928, -1.890] |
| rb_tuned | 1 | 558.56 | 8.282 | 6 | -3.315 [-3.315, -3.315] | 0.000 [0.000, 0.000] | 9.203 [9.203, 9.203] | 4 | 8.552 [8.552, 8.552] | 0.000 [0.000, 0.000] | -2.192 [-2.192, -2.192] |
| rb_default | 1 | 922.12 | 11.342 | 7 | -12.518 [-12.518, -12.518] | -9.203 [-9.203, -9.203] | 0.000 [0.000, 0.000] | 1 | 10.733 [10.733, 10.733] | 2.192 [2.192, 2.192] | 0.000 [0.000, 0.000] |

## Crossovers

Where a pair's difference changes sign along a sweep, linearly interpolated between grid points: the paired-gain IQM against rb_default or rb_tuned, the difference of profit IQMs for ppo vs residual_w05 (different runs). λ moves heat, elec and CO₂ together from agc2020 (λ = 0) to sim (λ = 1) in 41 points; the elec sweep moves electricity alone, 0.02 to 0.40 EUR/kWh, with heat and CO₂ at agc2020's. Fruit stays 1.6 EUR/kg.

| split | controller vs versus | λ | elec [EUR/kWh] |
|---|---|---|---|
| validation | ppo vs rb_default | 0.311 (controller ahead above x) | 0.146 (controller ahead above x) |
| validation | ppo vs residual_w05 | 0.659 (controller ahead above x) | 0.363 (controller ahead above x) |
| validation | zero_controls vs rb_default | 0.459 (controller ahead above x) | 0.237 (controller ahead above x) |
| validation | rb_tuned vs rb_default | 0.192 (controller ahead above x) | 0.104 (controller ahead above x) |
| validation | residual_w05 vs rb_tuned | none in range | none in range |
| amsterdam_2012 | ppo vs rb_default | 0.293 (controller ahead above x) | 0.139 (controller ahead above x) |
| amsterdam_2012 | ppo vs residual_w05 | 0.537 (controller ahead above x) | 0.281 (controller ahead above x) |
| amsterdam_2012 | zero_controls vs rb_default | 0.383 (controller ahead above x) | 0.204 (controller ahead above x) |
| amsterdam_2012 | rb_tuned vs rb_default | 0.180 (controller ahead above x) | 0.101 (controller ahead above x) |
| amsterdam_2012 | residual_w05 vs rb_tuned | none in range | none in range |
| london_2012 | ppo vs rb_default | 0.317 (controller ahead above x) | 0.145 (controller ahead above x) |
| london_2012 | ppo vs residual_w05 | 0.643 (controller ahead above x) | 0.343 (controller ahead above x) |
| london_2012 | zero_controls vs rb_default | 0.457 (controller ahead above x) | 0.238 (controller ahead above x) |
| london_2012 | rb_tuned vs rb_default | 0.178 (controller ahead above x) | 0.100 (controller ahead above x) |
| london_2012 | residual_w05 vs rb_tuned | none in range | none in range |
| abbotsford_2012 | ppo vs rb_default | 0.275 (controller ahead above x) | 0.129 (controller ahead above x) |
| abbotsford_2012 | ppo vs residual_w05 | 0.509 (controller ahead above x) | 0.238 (controller ahead above x) |
| abbotsford_2012 | zero_controls vs rb_default | 0.377 (controller ahead above x) | 0.188 (controller ahead above x) |
| abbotsford_2012 | rb_tuned vs rb_default | 0.192 (controller ahead above x) | 0.103 (controller ahead above x) |
| abbotsford_2012 | residual_w05 vs rb_tuned | none in range | none in range |

Reference non-household electricity prices beside the elec column (Eurostat nrg_pc_205, 500-1999 MWh band, excluding taxes, retrieved 2026-11-03; an annotation, no input): NL 2020-S1 0.0739 EUR/kWh, NL 2022-S2 0.151 EUR/kWh, EU27 2020-S1 0.0818 EUR/kWh, EU27 2022-S2 0.1987 EUR/kWh.

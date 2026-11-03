Made by `uv run python -m greenleaf_control.rule_family horizon --lengths 60 120 180 240 300` (in control/): VALIDATION weather; exploratory analysis; nothing trains or is selected.

rb_default and rb_tuned (evaluate.run_rb) and zero_controls (envs.run_baseline, graded by metrics) on each VALIDATION season (Amsterdam 2010, start days 45, 59, 75) at each season length, played once at sim prices and re-priced at agc2020. All three are deterministic: one run per cell, so no interval.

- fruit is metrics' yield_kg_m2, the net change in standing fruit over the season as fresh weight; Σ hours is the season's CO₂ + temperature + RH violation hours.
- A cell whose season would end on or past day 365 is skipped, never played: start day 75 + 300 days = day 375, on or past day 365: with upstream's 0.5-day weather lookahead it would read the 2011 file, past its 2010 weather year.
- Prices (heat and elec EUR/kWh, CO₂ EUR/kg; fruit 1.6 EUR/kg in both), metrics.PRICE_REGIMES: sim: heat 0.09, elec 0.3, co2 0.3; agc2020: heat 0.03, elec 0.06, co2 0.08.

## profit at sim [EUR/m²]

| controller | start day | 60 days | 120 days | 180 days | 240 days | 300 days |
|---|---|---|---|---|---|---|
| Default rules | 45 | -15.05 | -16.87 | -19.51 | -41.21 | -85.86 |
| Default rules | 59 | -4.70 | -6.93 | -12.93 | -39.98 | -86.24 |
| Default rules | 75 | -2.49 | 0.06 | -9.80 | -43.99 | skipped |
| Tuned rules | 45 | -5.03 | -5.29 | -10.68 | -22.84 | -48.15 |
| Tuned rules | 59 | 1.31 | -1.43 | -7.70 | -22.15 | -49.59 |
| Tuned rules | 75 | 2.17 | -0.63 | -6.42 | -25.35 | skipped |
| Do nothing | 45 | 2.07 | 4.49 | 2.05 | 2.58 | 2.36 |
| Do nothing | 59 | 2.89 | 4.12 | 2.05 | 2.63 | 2.34 |
| Do nothing | 75 | 3.88 | 2.61 | 2.20 | 2.73 | skipped |

## profit at agc2020 [EUR/m²]

| controller | start day | 60 days | 120 days | 180 days | 240 days | 300 days |
|---|---|---|---|---|---|---|
| Default rules | 45 | 9.60 | 19.34 | 23.71 | 21.31 | 14.29 |
| Default rules | 59 | 12.44 | 22.17 | 24.85 | 22.27 | 14.14 |
| Default rules | 75 | 13.22 | 23.41 | 25.46 | 21.48 | skipped |
| Tuned rules | 45 | 7.49 | 15.69 | 16.60 | 14.92 | 7.37 |
| Tuned rules | 59 | 11.04 | 16.90 | 17.64 | 14.96 | 6.49 |
| Tuned rules | 75 | 11.82 | 15.65 | 18.19 | 13.56 | skipped |
| Do nothing | 45 | 2.07 | 4.49 | 2.05 | 2.58 | 2.36 |
| Do nothing | 59 | 2.89 | 4.12 | 2.05 | 2.63 | 2.34 |
| Do nothing | 75 | 3.88 | 2.61 | 2.20 | 2.73 | skipped |

## fruit [kg/m²]

| controller | start day | 60 days | 120 days | 180 days | 240 days | 300 days |
|---|---|---|---|---|---|---|
| Default rules | 45 | 10.930 | 19.628 | 23.938 | 26.151 | 28.795 |
| Default rules | 59 | 11.418 | 20.142 | 23.634 | 26.668 | 28.765 |
| Default rules | 75 | 11.666 | 19.825 | 23.517 | 26.677 | skipped |
| Tuned rules | 45 | 7.810 | 15.157 | 17.367 | 18.892 | 18.176 |
| Tuned rules | 59 | 9.471 | 15.361 | 17.591 | 18.769 | 17.740 |
| Tuned rules | 75 | 9.910 | 14.062 | 17.745 | 18.178 | skipped |
| Do nothing | 45 | 1.292 | 2.807 | 1.280 | 1.615 | 1.473 |
| Do nothing | 59 | 1.809 | 2.577 | 1.283 | 1.644 | 1.461 |
| Do nothing | 75 | 2.428 | 1.632 | 1.374 | 1.706 | skipped |

## Σ hours [h]

| controller | start day | 60 days | 120 days | 180 days | 240 days | 300 days |
|---|---|---|---|---|---|---|
| Default rules | 45 | 804.75 | 1495.50 | 2205.00 | 3100.50 | 4082.00 |
| Default rules | 59 | 709.25 | 1396.50 | 2179.50 | 3094.00 | 4023.50 |
| Default rules | 75 | 715.50 | 1388.50 | 2199.75 | 3211.25 | skipped |
| Tuned rules | 45 | 324.50 | 748.25 | 1416.00 | 2084.00 | 2431.50 |
| Tuned rules | 59 | 349.00 | 808.00 | 1536.50 | 2135.00 | 2404.25 |
| Tuned rules | 75 | 393.50 | 948.75 | 1627.25 | 2215.00 | skipped |
| Do nothing | 45 | 2930.00 | 5464.00 | 6442.50 | 8064.25 | 10997.50 |
| Do nothing | 59 | 2919.25 | 5240.25 | 6079.00 | 8188.00 | 11049.00 |
| Do nothing | 75 | 2978.00 | 4779.25 | 5687.50 | 8271.25 | skipped |

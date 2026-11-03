Made by `uv run python -m greenleaf_control.rule_family frontier --n 1000 --seed 0 --season-length 60` (in control/): VALIDATION weather; exploratory analysis; nothing trains or is selected.

1002 rule settings, each played once at sim prices on the 3 VALIDATION seasons (Amsterdam 2010, start days 45, 59, 75) of 60 days and re-priced at agc2020: setting 0 is the default rules, 1 the tuned rules, and the 1000 after them uniform draws over tune_rb.TUNE_SPACE from default_rng(0) (setting i is sample_settings(1000, 0)[i]). A setting's numbers are means over its 3 seasons; frontier_seasons.csv has each season.

- The comparison is in-sample for the agents: VALIDATION selected their checkpoints. Nothing is selected or tuned from this frontier.
- 0 of 1002 settings excluded from the comparison for an invalid season (a solver failure or a short season); they stay in frontier.csv with valid False.
- A point is a controller's IQM (rliable's) over its recorded validation runs x seasons (validation's scores.parquet, by label; n seasons): its profit re-priced at the regime, its fruit, and Σ hours as the sum of its CO₂, temperature and RH hours IQMs (Σ per-constraint IQMs).
- best: among the valid settings with fruit ≥ the point's and Σ hours ≤ the point's (within 1e-9), the one with the highest profit at the regime; qualifying counts them; gap = the point's profit − best's profit (positive: the controller earns more).
- best is in-sample too: a maximum over the qualifying settings on the same 3 VALIDATION seasons, so it is optimistic for the rules, more so as qualifying grows.
- gap is descriptive, not an out-of-sample estimate of what an agent earns over the rules.
- Do nothing (zero_controls) and random are left out of the comparison: random is a single stochastic run, and Do nothing grows ~2 kg/m² of fruit in the recorded validation seasons (1.843, validation's summary.md), far below every agent, so it is no comparison point.
- Prices (heat and elec EUR/kWh, CO₂ EUR/kg; fruit 1.6 EUR/kg in both), metrics.PRICE_REGIMES: sim: heat 0.09, elec 0.3, co2 0.3; agc2020: heat 0.03, elec 0.06, co2 0.08.

## sim

| controller | n | profit IQM [EUR/m²] | fruit IQM [kg/m²] | Σ per-constraint IQMs [h] | best setting | best profit [EUR/m²] | gap [EUR/m²] | qualifying |
|---|---|---|---|---|---|---|---|---|
| PPO, no rules (ppo #33/0221) | 15 | 2.395 | 6.690 | 289.75 | 497 | -3.974 | 6.368 | 82 |
| PPO, paper setup (ppo_paper #31/2d02) | 15 | 2.299 | 6.801 | 251.44 | 897 | -4.943 | 7.242 | 46 |
| Residual agent, fixed penalty (residual_w05 #36/f2fd) | 30 | 0.765 | 9.625 | 262.56 | 926 | -5.575 | 6.340 | 46 |
| Residual agent, constrained (lag_b100 #39/6d92) | 30 | 0.937 | 9.609 | 362.28 | 325 | -2.759 | 3.696 | 128 |
| Default rules (rb_default) | 3 | -7.414 | 11.338 | 743.17 | 739 | -3.765 | -3.649 | 76 |
| Tuned rules (rb_tuned) | 3 | -0.517 | 9.064 | 355.67 | 1 | -0.517 | 0.000 | 136 |

## agc2020

| controller | n | profit IQM [EUR/m²] | fruit IQM [kg/m²] | Σ per-constraint IQMs [h] | best setting | best profit [EUR/m²] | gap [EUR/m²] | qualifying |
|---|---|---|---|---|---|---|---|---|
| PPO, no rules (ppo #33/0221) | 15 | 7.927 | 6.690 | 289.75 | 192 | 11.852 | -3.924 | 82 |
| PPO, paper setup (ppo_paper #31/2d02) | 15 | 8.036 | 6.801 | 251.44 | 457 | 11.182 | -3.146 | 46 |
| Residual agent, fixed penalty (residual_w05 #36/f2fd) | 30 | 11.084 | 9.625 | 262.56 | 457 | 11.182 | -0.098 | 46 |
| Residual agent, constrained (lag_b100 #39/6d92) | 30 | 11.118 | 9.609 | 362.28 | 192 | 11.852 | -0.733 | 128 |
| Default rules (rb_default) | 3 | 11.754 | 11.338 | 743.17 | 620 | 12.732 | -0.978 | 76 |
| Tuned rules (rb_tuned) | 3 | 10.118 | 9.064 | 355.67 | 192 | 11.852 | -1.733 | 136 |

# spike_3d early-warning model — validation report (Phase 2A)

**Operational early-warning for triage**, not a financial-return model. Phase 1
(`docs/eda_findings.md`) showed plot-level responsiveness does **not** predict profit, so
this model makes **no profit/return claims**. It answers one operational question:
*"Which plots are likely to have their plant-stress spike in the next 3 days, so a manager
should watch or act now?"*

Reproduce: `python analysis/spike_model.py` (seed 42). Saved artifact:
`models/spike_model.joblib` (+ `models/spike_model_features.json`).

---

## 1. Target & framing

`spike_3d` = 1 if `plant_stress_index` rises **≥ 0.10 over the next 3 days**, else 0
(already built in `daily.parquet`). Base rate **0.1812** over 24,840 valid plot-days (the
360 season-end rows with no 3-day-forward target are dropped).

**A structural fact that shapes everything (from EDA + confirmed here):** stress is capped
near 0.85, so a spike is *more* likely when current stress is **low** (room to rise) and
rare when stress is already high. Spike rate by current-stress quintile: 0.265 → 0.286 →
0.192 → 0.109 → **0.053**; `corr(current stress, spike) = −0.224`. The task therefore has a
strong **mean-reversion** component, and even pest signals correlate slightly *negatively*
with spikes (high pest → already-high stress → less headroom). This is why we report a
"current-stress baseline" — a good chunk of predictability is just mean-reversion, and the
model has to earn its keep *above* that.

---

## 2. Leakage handling (the most important methodological point)

The target is forward-looking, so features use **only at/before-prediction-day**
information. The script asserts at runtime that no excluded column appears in the feature
set (`assert_no_leakage()`), so a future edit can't silently reintroduce leakage.

**Excluded (16 columns), with reason:**
- *Target:* `spike_3d`.
- *Forward-looking:* `stress_lead_3`, `stress_delta_3d_forward`,
  `post_action_stress_delta_3d` (all use the future — the last is the 3-day **post-action**
  stress change).
- *Intervention / post-decision* (leaks the response the model is meant to inform, and
  isn't known before the manager decides to act): `action_taken`, `action_delay_days`,
  `recommended_action`, `daily_precision_cost`, `daily_routine_cost`,
  `daily_precision_actions_count`, `daily_total_actions_count`, `daily_total_input_cost`.
- *Identity / time index:* `plot_id`, `greenhouse_id`, `farm_id`, `date`.

**Used (37 features):** current-day sensors (`air_temp_c`, `relative_humidity_pct`,
`vpd_kpa`, `co2_ppm`, `light_ppfd`, `substrate_moisture`, `substrate_ec`, `substrate_ph`,
`canopy_temp_c`, `pest_pressure_index`, `disease_risk_index`, `plant_stress_index`,
`canopy_air_diff`, `alert_flag`); trailing lags/rolling means (`stress_lag_1/3/7`,
`vpd_kpa_7d_mean`, `substrate_moisture_7d_mean`, `pest_pressure_7d_mean`,
`canopy_air_diff_7d_mean`); derived trailing momentum (`stress_trend_3` =
`plant_stress_index − stress_lag_3`; `pest_dev_7` = `pest_pressure_index −
pest_pressure_7d_mean`); static plot attributes (density, area, sensor density, acres,
capacity); and categoricals (`crop`, `treatment`, `structure_type`, `heating_system`,
`irrigation_system`, `climate_zone`, `region`, `production_system`, `alert_type`). **None
are forward-looking.** Scouting is excluded entirely (decoupled, 2026 dates).

**Time-aware split (no random splitting):**
| split | dates | rows | base rate |
|---|---|---|---|
| TRAIN | 2025-02-15 … 2025-07-15 | 18,120 | 0.1776 |
| *buffer (dropped)* | 2025-07-16 … 2025-07-31 | 1,920 | — |
| TEST | 2025-08-01 … 2025-09-09 | 4,800 | 0.1821 |

The 16-day buffer far exceeds the 3-day forward horizon, so the target never straddles the
boundary. Test is the **late season** — the realistic deployment scenario (predict the
future from the past). No class-imbalance reweighting is applied: we optimize/observe
**ranking** metrics (ROC/PR) and a tuned threshold, which preserves probability calibration
(needed because the manager acts on the probability).

---

## 3. Metrics vs baselines (held-out test set, n=4,800, threshold = 0.18)

| model | AUC-ROC | AUC-PR | Brier | Prec@0.18 | Recall | F1 | Prec@top20% |
|---|---|---|---|---|---|---|---|
| B1 base-rate | 0.500 | 0.182 | 0.149 | 0.000 | 0.000 | 0.000 | 0.200 |
| B2 current-stress | 0.626 | 0.246 | 0.149 | 0.226 | 0.866 | 0.358 | 0.244 |
| B3 pest/alert | 0.579 | 0.212 | 0.146 | 0.204 | 0.978 | 0.337 | 0.215 |
| Logistic (all feats) | 0.775 | **0.408** | 0.128 | 0.335 | 0.759 | 0.465 | 0.402 |
| **HGB (primary)** | **0.775** | 0.393 | 0.130 | 0.364 | 0.637 | 0.464 | **0.412** |

*No-skill references: AUC-PR = base rate = 0.182; AUC-ROC = 0.5.*

**Does the model earn its keep? Yes, clearly.** Over the current-stress baseline it lifts
AUC-ROC **0.626 → 0.775** and AUC-PR **0.246 → ~0.40** (more than **double** the no-skill PR
of 0.182). Brier improves 0.149 → 0.130. This is not a "barely beats baseline" case — the
multi-feature view adds real, out-of-time signal.

**Honest nuances:**
- **The interpretable logistic ties the GBT** (AUC-ROC 0.7748 vs 0.7750; AUC-PR *higher* at
  0.408 vs 0.393). The signal is largely linear — the gradient-boosted tree buys almost
  nothing on ranking. HGB is kept as the saved primary (slightly better precision at the
  alarm threshold and at top-20%), but **logistic is a co-equal, more transparent
  alternative** the dashboard could use instead.
- **Much of the base predictability is mean-reversion**, not agronomy: the current-stress
  baseline alone reaches AUC-ROC 0.626 because low-stress plots tend to bounce up. The
  model's genuine *agronomic* value-add is identifying *which* low-stress plots are at
  risk — driven by substrate-moisture and environment (see §4).

---

## 4. What drives the early warning (permutation importance, test, scored on AUC-PR)

| rank | feature | importance | ± std |
|---|---|---|---|
| 1 | `substrate_moisture_7d_mean` | 0.088 | 0.007 |
| 2 | `substrate_moisture` | 0.039 | 0.004 |
| 3 | `stress_lag_7` | 0.039 | 0.003 |
| 4 | `treatment` | 0.035 | 0.003 |
| 5 | `plant_stress_index` | 0.024 | 0.007 |
| 6 | `co2_ppm` | 0.017 | 0.003 |
| 7 | `vpd_kpa` | 0.014 | 0.002 |
| 8 | `alert_type` | 0.013 | 0.006 |
| 9 | `stress_lag_1` | 0.009 | 0.003 |
| 10 | `stress_trend_3` | 0.004 | 0.002 |

**Key finding — and it complements rather than contradicts the EDA.** The EDA found that
the **level** of season stress is driven by **pest-alert burden** (corr 0.99). But the
**short-term spike** (a forward *change*) is driven mostly by **substrate moisture** (the
trailing 7-day mean dominates), recent stress history (`stress_lag_7`), and **treatment**,
with `pest_pressure_index` near the bottom (≈0.002). Reading: **pest pressure sets the
baseline stress level; water/irrigation status and environment (VPD, CO₂) drive when a plot
is about to spike.** Operationally that's encouraging — the early-warning signal points at a
*controllable* lever (irrigation/substrate management), not just exogenous pest pressure.

---

## 5. Calibration (HGB, test, 10 quantile bins)

| mean predicted | observed fraction |
|---|---|
| 0.014 | 0.000 |
| 0.033 | 0.023 |
| 0.055 | 0.077 |
| 0.077 | 0.090 |
| 0.101 | 0.104 |
| 0.130 | 0.167 |
| 0.167 | 0.252 |
| 0.221 | 0.285 |
| 0.307 | 0.371 |
| 0.474 | 0.452 |

Predicted probabilities track observed spike frequencies closely across the whole range
(mildly under-confident in the mid-range, e.g. predicts 0.167 where 0.252 occur). **Brier
0.130 vs 0.149 base-rate.** The probabilities are meaningful — a manager can read "31%
chance of spike" as roughly a real 1-in-3, which is what makes a probability-ranked watch
list trustworthy.

**Confusion @ threshold 0.18** (recall-oriented operational alarm): TP=557, FP=972, FN=317,
TN=2,954. Recall **0.64**, precision **0.36** (2× the 0.18 base rate). At this threshold it
is a sensitive *watch list*, not a precise predictor — it flags ~32% of plot-days to catch
64% of spikes. **Triage framing (the intended use): precision@top-20% = 0.41** — a manager
who checks the riskiest fifth of plot-days each cycle finds a spike 41% of the time vs 18%
at random (a **2.3× lift**).

---

## 6. Secondary check — GroupKFold(5) by plot_id

Out-of-fold over all valid rows: **AUC-ROC 0.840, AUC-PR 0.535** (base rate 0.181) — *higher*
than the time-split. Interpretation: the model **generalizes across plots even better than
across time**. The gap (0.840 grouped vs 0.775 time-held-out) is **temporal drift** — the
late-season test period is genuinely harder to predict from spring/early-summer data. The
**time-split number (0.775) is the honest deployment estimate**; the GroupKFold result just
confirms there's no plot-identity overfitting.

---

## 7. What this model can and cannot be used for

**CAN (supported):**
- **Operational triage / watch list:** rank plot-days by spike probability so managers act
  on the riskiest first (2.3× lift over random in the top-20%).
- **A calibrated 3-day spike probability** per plot-day, meaningful enough to act on.
- **Pointing attention at a controllable lever:** the dominant driver is substrate moisture,
  i.e. irrigation/water status — actionable, not just "pest happened."

**CANNOT (do not claim):**
- ❌ **Any profit / ROI / financial-return prediction.** Per the EDA, plot-level
  responsiveness and stress do not predict profit once treatment/pest burden are controlled.
  This model predicts a *stress event*, not money.
- ❌ **A precise per-plot guarantee.** At the operational threshold precision is ~0.36 —
  expect false alarms; it is a sensitivity-first watch list.
- ❌ **Causal attribution.** Importances show association, not that changing substrate
  moisture *causes* fewer spikes (that would need the Aim-6 response-effect analysis).
- ❌ **Extrapolation beyond the observed season** (Feb–Sept 2025) or to other crops/regions
  not represented; late-season drift already costs ~0.06 AUC.

## Verdict

**The model earns its keep.** It beats the current-stress baseline decisively (AUC-ROC
0.626 → 0.775; AUC-PR 0.25 → ~0.40, >2× no-skill) and is well-calibrated, giving a usable
2.3× triage lift for an operational manager. Two honest caveats kept in front: (1) a simple
**logistic regression matches the gradient-boosted tree**, so the story is interpretable and
we don't need the black box; (2) the early-warning signal is **substrate-moisture/environment
driven (a controllable lever), distinct from the pest-driven stress *level*** the EDA
surfaced — and it is strictly an operational stress-spike predictor, never a profit model.

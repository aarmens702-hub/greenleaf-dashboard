# Top 10 aims for stress indicators & risk modeling

Companion to `docs/eda_findings.md`. Each aim = **what** we'd build, **why it matters**
(narrative for farmer/lender), and **how to make it from our data** (target, features,
method, baseline, validation, sample size, caveats). Ordered as an arc: strongest/most
feasible first → exploratory/small-N last. Feasibility flag: 🟢 solid · 🟡 doable with care
· 🔴 exploratory, report as such.

The agreed prediction target `spike_3d` (= `stress_delta_3d_forward >= 0.10`) is preserved
as the flagship (Aim 1). The rest are complementary indicators/models, not a replacement.

---

## Ground rules that apply to every model here (read first)

These come straight from the Phase-1 data structure and are what keep the numbers
defensible to an RBC statistician:

- **Validate with `GroupKFold` on `plot_id`, and/or a time-ordered split** (train early
  season, test late). Rows are plot×day and heavily autocorrelated — a random row-wise
  split leaks tomorrow into today and inflates every metric.
- **Baseline before model, always.** Spike → base rate; stress level → *persistence*
  (tomorrow = today). A model only "wins" if it beats persistence out-of-sample.
- **Never feed future-derived columns as features:** `stress_lead_3`,
  `stress_delta_3d_forward`, `spike_3d` (target), and `post_action_stress_delta_3d`
  (a post-action, forward-looking field) are leakage for a pre-emptive predictor.
- **Never use `precision_benefit_cad` as a target** — it's `0.65 × profit` (circular).
- **Report metrics with bootstrap CIs over plots**, and flag any cell with n<5 / n<10.
- **Mind the edges:** first 7 days have NaN stress lags (840 rows), last 3 days NaN leads
  (360 rows) — drop or handle explicitly.
- **Control for treatment/crop** whenever relating stress to an *outcome* (Phase 1: the
  stress→profit link is treatment, not stress).
- **Don't use `scouting` features for sensor-aligned models** — it's decoupled (2026 dates,
  corr ≈0 with sensor signals).

---

## Aim 1 — Stress-spike early-warning classifier 🟢 *(flagship)*

**What:** Predict `spike_3d` — will plant stress jump ≥0.10 within the next 3 days? — from
*today's* sensor state.

**Narrative:** Turns the sensor wall into a forward-looking alarm. Instead of reacting once
stress is already high, the farmer gets a "stress likely to spike in 72h" flag and can act
same-day — which Phase 1 proved roughly halves the stress hit. This is the operational
embodiment of precision ag and the natural anchor for the dashboard.

**How with our data:**
- Target: `spike_3d` (already built; ~24,840 usable rows after dropping the 360 NaN-lead
  tail).
- Features: `pest_pressure_index`, `pest_pressure_7d_mean`, `disease_risk_index`, `vpd_kpa`,
  `vpd_kpa_7d_mean`, `substrate_moisture`(+7d), `canopy_air_diff`(+7d), `co2_ppm`,
  `light_ppfd`, `substrate_ec`, `substrate_ph`, current `plant_stress_index` + `stress_lag_1/3/7`,
  and plot meta (`crop`, `treatment`, `structure_type`, `irrigation_system`, `climate_zone`).
- Method: logistic regression (interpretable baseline) → gradient-boosted trees (lift).
  Report PR-AUC/ROC-AUC vs base rate, plus a calibrated probability for the gauge.
- Baseline: base spike rate (~6–7% of action days; compute overall rate first) and a
  "high pest pressure today → spike" rule.
- Caveat: confirm sensors beat persistence/pest-pressure-alone; if pest pressure carries
  almost all the signal (likely, given Phase 1), that's itself the headline, not a flaw.

---

## Aim 2 — 3-day stress-level forecast (regression) 🟢

**What:** Forecast the *level* of `plant_stress_index` 3 days out (`stress_lead_3`), not just
the binary spike.

**Narrative:** Severity, not just yes/no. A farmer triaging 6 greenhouses wants "which plots
will be worst, and how bad," to rank where to spend the day. Feeds a heatmap of projected
stress.

**How with our data:**
- Target: `stress_lead_3` (or `stress_delta_3d_forward` for the change).
- Features: same as Aim 1.
- Method: linear/GBM regression. **Baseline = persistence** (`plant_stress_index` today);
  report R²/MAE *improvement over persistence* out-of-sample (grouped CV).
- Caveat: stress is strongly autocorrelated, so persistence is a hard baseline — be honest
  about marginal lift. The value is in the sensor features' early warning, not raw R².

---

## Aim 3 — Pest-pressure / alert-onset forecaster 🟢

**What:** Predict whether a High-Pest-Pressure alert (`alert_flag`) fires tomorrow, or that
`pest_pressure_index` crosses its alert threshold.

**Narrative:** Phase 1 showed stress ≈ pest burden (corr 0.99) and 99% of fired alerts are
High Pest Pressure. So *forecasting pest pressure is essentially forecasting stress* — and
pest is the one thing the whole operation is fighting. A 1-day pest lead time is directly
actionable (pre-emptive IPM).

**How with our data:**
- Target: `alert_flag` at t+1 (or `pest_pressure_index` rising), built by shifting within
  `plot_id` (mirror `build_derived.py`'s lag pattern — but compute in `analysis/`, don't
  modify the derived data without asking).
- Features: environment that precedes pest pressure — `vpd_kpa`, `relative_humidity_pct`,
  `air_temp_c`, `canopy_air_diff`, `disease_risk_index`, prior `pest_pressure_7d_mean`,
  crop/treatment/structure.
- Method: logistic/GBM; grouped CV; baseline = persistence of yesterday's alert state.
- Caveat: if the simulator drives pest pressure largely off treatment + a stochastic term,
  environmental features may add little — test and report honestly.

---

## Aim 4 — GreenLeaf composite Stress-Risk Index (explainable gauge) 🟢

**What:** A single transparent 0–1 plot-day risk score blending the known drivers
(pest pressure, VPD deviation, moisture deficit, canopy-air diff, disease risk), shown as a
gauge with its component breakdown.

**Narrative:** Farmers and lenders don't want a black box — they want one number they trust
and can decompose ("risk is high *because* pest pressure + dry substrate"). It's the
dashboard's emotional center and is explainable by construction.

**How with our data:**
- Build: normalize each driver to 0–1 (min–max or z within crop), combine with transparent
  weights. *Calibrate* the weights against forward risk — choose weights so the index best
  separates `spike_3d` (e.g., logistic-regression coefficients re-expressed as weights), so
  it's data-driven yet interpretable.
- Validate: AUC of the index vs `spike_3d`; show it tracks `plant_stress_index`.
- Caveat: present weights openly; don't overfit — a 4–5 component index is plenty.

---

## Aim 5 — Environmental "out-of-band" indicators (agronomic flags) 🟡

**What:** Per-crop flags for when `vpd_kpa`, `substrate_moisture`, `substrate_ec`,
`substrate_ph`, `canopy_air_diff`, `co2_ppm`, `light_ppfd` fall outside agronomic optimal
ranges.

**Narrative:** The most *actionable* indicators — each flag maps to a concrete lever
("VPD too high → vent/humidify"). Grounds the dashboard in real agronomy, not just a model
score, and connects to the precision-ag literature (responsive vs scheduled control).

**How with our data:**
- Ranges from `docs/greenleaf_corpus/crop_growing_characteristics.md` and
  `bc_benchmarks.parquet` (per crop where available); compute % of season each plot spends
  out-of-band per variable.
- Validate: do out-of-band days precede stress spikes? (cross-check vs `spike_3d`).
- Caveat: optimal ranges are external assumptions — cite the source; flag where the corpus
  lacks a crop-specific range.

---

## Aim 6 — Response-speed effect model (the precision proof, formalized) 🟢

**What:** Model `post_action_stress_delta_3d` as a function of `action_delay_days` with
controls, producing a defensible effect estimate with CI.

**Narrative:** This is Phase 1's single unassailable result (same-day response ≈ halves the
stress hit, p≈10⁻⁹⁸). Formalizing it — controlling for confounders and reporting an effect
size with CI — is the strongest, most lender-credible piece of the whole precision case.

**How with our data:**
- Sample: action days (n=5,509). Outcome `post_action_stress_delta_3d`; key regressor
  `action_delay_days` (0/1/2); controls: baseline `plant_stress_index`, `alert_type`,
  `pest_pressure_index`, crop, treatment.
- Method: OLS with controls + bootstrap CI over plots; report the per-day marginal effect
  (Phase 1 raw: 0→−0.049, 1→−0.029, 2→−0.011).
- Caveat: observational — same-day responses may occur under arrestable conditions; state
  this. Speed lowers the *average* delta but not the spike *rate* (Phase 1) — don't overclaim.

---

## Aim 7 — "Cost of delay" counterfactual KPI (stress-days avoided) 🟡

**What:** Using Aim 6's delay→relief curve, estimate the extra stress reduction (or
stress-days avoided) if every delayed response had been same-day.

**Narrative:** Translates the mechanism into an operational, board-room KPI: "responding
same-day instead of next-day would have removed X stress-days across the portfolio." Gives
the lender a tangible lever without inventing a dollar figure.

**How with our data:**
- Apply the estimated per-delay effect to the 3,151 delayed responses to get a portfolio
  counterfactual; express in stress-units × days.
- Caveat: a partial-equilibrium estimate (assumes the effect transfers); label it an
  estimate. **Do not** convert to CAD via `precision_benefit_cad` (circular) or the daily
  cost log (doesn't reconcile with season $).

---

## Aim 8 — Cumulative stress-burden / trajectory indicator 🟡

**What:** Per-plot season risk profile: cumulative stress-days above a threshold, total
time-in-alert, peak-stress timing, and the stress trajectory shape.

**Narrative:** A plot's risk isn't one number — it's a curve. This links in-season
monitoring to the end-of-season outcome and lets the dashboard show "this plot was under
chronic pressure from week 6" rather than a single average.

**How with our data:**
- Build: threshold `plant_stress_index` (e.g., >0.5), cumulative `alert_days`, rolling
  peaks per `plot_id` over the 210 days; relate to `season` outcomes.
- Caveat: any link to profit must carry the Phase-1 treatment-confound caveat — present as
  description/segmentation, not causation.

---

## Aim 9 — Plot-level financial-risk screen (loss / bottom-tercile) 🔴

**What:** Flag plots at financial risk (negative profit, or bottom-ROI tercile) from
*early-season* stress/pest-burden + plot meta.

**Narrative:** The lender's literal question — "which operations are at risk?" Even a rough
early-warning screen is valuable for a precision-ag loan program.

**How with our data:**
- Target: `season_profit_cad < 0` (only **6 of 120** — extremely small) or bottom ROI
  tercile (n≈40, more stable). Features: first-60–90-day mean stress, alert burden, crop,
  treatment, structure.
- Method: keep it simple (logistic / risk-tier table), bootstrap heavily, **report as
  exploratory**. With 120 plots and 6 positives, a fitted classifier will overfit —
  prefer a transparent risk-tier descriptive screen over a flashy model.
- Caveat: highest-overfitting risk on the list; lead with the small-N warning.

---

## Aim 10 — Alert-to-action operational KPI / backlog monitor 🟢

**What:** Management-quality dashboard metrics per plot/greenhouse over time: alert-response
rate, mean response delay, and unaddressed-alert backlog.

**Narrative:** This is the *controllable* behavior a precision-ag loan is meant to improve.
Phase 1 found responsiveness doesn't *yet* separate plot profit (it's uniformly high) — but
that's precisely the honest framing: it's the lever to monitor and improve, and the metric
that shows whether financed equipment is actually being used responsively.

**How with our data:**
- Build from `alert_flag`, `action_taken`, `action_delay_days` aggregated by
  plot/greenhouse/week; trend over the season.
- Caveat: descriptive/operational, not predictive of profit (per Phase 1) — present it as a
  process metric, and note the compressed response-rate range (min 0.47).

---

## At a glance

| # | Aim | Type | Feas. | Key data target |
|---|---|---|---|---|
| 1 | Stress-spike early-warning | model | 🟢 | `spike_3d` |
| 2 | 3-day stress-level forecast | model | 🟢 | `stress_lead_3` |
| 3 | Pest-pressure / alert-onset forecast | model | 🟢 | `alert_flag` t+1 |
| 4 | Composite Stress-Risk Index | indicator | 🟢 | sensors → `spike_3d` |
| 5 | Out-of-band agronomic flags | indicator | 🟡 | sensors vs ranges |
| 6 | Response-speed effect (precision proof) | model | 🟢 | `post_action_stress_delta_3d` |
| 7 | Cost-of-delay counterfactual | indicator | 🟡 | delay curve |
| 8 | Cumulative stress-burden trajectory | indicator | 🟡 | `plant_stress_index` series |
| 9 | Plot-level financial-risk screen | model | 🔴 | `season_profit_cad`<0 / ROI tercile |
| 10 | Alert-to-action operational KPI | indicator | 🟢 | `alert_flag`/`action_taken`/`delay` |

**Recommended build order:** 1 + 4 (flagship model + its explainable gauge), then 6
(precision proof) and 10 (operational KPI) — these four form a complete, defensible story.
2, 3, 5, 8 deepen it; 7 and 9 are estimate/exploratory — build only if time allows and
label honestly.

# GreenLeaf EDA — Does precision agriculture beat routine management?

**Phase 1 findings. Analyst's brief for an RBC lender who knows statistics.**

Every number below is computed from `data/derived/*.parquet` and reproducible via
`python analysis/eda.py` (bootstrap seed = 42). Where a figure is an
estimate-with-assumptions rather than a hard measurement, it is flagged **[ASSUMPTION]**.

Headline up front, in one line: **the *mechanism* of precision is real and cheap and
provable; the *bottom-line dollar return* claimed in the data is an assumption, not a
measurement; and the honest value proposition is risk reduction against pest-driven
stress, concentrated where pest pressure is highest.**

---

## Executive summary — the 5 strongest defensible claims

1. **Acting on an alert the *same day* cuts plant stress about twice as much as
   responding a day or two later.** Post-action 3-day stress change: **−0.0494** for
   same-day response (n=2,358) vs **−0.0236** for delayed (n=3,151); difference
   **−0.0258, 95% CI [−0.0282, −0.0235]**, Welch t=−21.5, p≈2×10⁻⁹⁸. Clean monotonic
   dose-response: delay 0 / 1 / 2 days → −0.049 / −0.029 / −0.011. *This is the one
   result a skeptic cannot dismiss — it is the mechanism by which precision works.*

2. **Plant stress is almost entirely a restatement of pest-alert burden, not a free
   lever.** Across 120 plots, `corr(alert_days, mean_stress) = +0.99`. 99.2% of the
   6,460 fired alerts are "High Pest Pressure." Stress is downstream of how many pest
   alerts a plot faced.

3. **Precision is a small, cheap overlay on mostly-routine management — not a separate
   regime.** Precision (alert-response) actions are **8.4%** of all input applications
   (799 / 9,559) and **4.6%** of input cost (18,646 / 404,050 CAD). Every plot is
   managed with a mix; there is no "pure routine" vs "pure precision" plot to compare.

4. **Treatment, driven by pest burden, orders the outcomes.** High Light earns the most
   (mean profit **838.7 CAD/plot**, lowest stress 0.413, fewest alerts 20.9); Shade the
   least (**379.2 CAD**, highest stress 0.554, most alerts 100.2). High Light − Shade
   profit gap = **+460.6 CAD/plot, 95% CI [242, 705]** (n=26 vs 13).

5. **The headline "precision benefit" figure is circular.** `precision_benefit_cad` =
   **0.65 × season_profit_cad, floored at 0** (corr 0.996; median ratio exactly 0.650;
   the 6 loss-making plots get 0). It totals **47,034 CAD = 65.6% of total profit**.
   **[ASSUMPTION]** It is the simulation's *assumed* attribution of profit to precision,
   not an independent measurement, and carries no information beyond profit itself.

**Bottom line for the lender:** precision's *mechanism* is demonstrably real and low-cost.
But this dataset does **not** let us prove that more-precise plots earned more money —
season-level profit variation is dominated by exogenous pest pressure and treatment, and
the dollar "benefit" field is profit relabelled. Pitch precision as **proven, cheap risk
management against pest-driven stress**, strongest where pest pressure is highest — not
as a measured 65% profit uplift.

---

## 1. Data orientation

| Table | Grain | Rows | Notable |
|---|---|---|---|
| `season` | 1 row / plot | 120 | outcomes + plot metadata; `precision_benefit_cad` lives here |
| `daily` | plot × day | 25,200 (120 × 210) | 2025-02-15 → 2025-09-12; sensors, costs, alerts, actions, engineered lags |
| `actions` | input application | 4,613 | `is_precision_action`: 799 precision / 3,814 routine |
| `scouting` | plot × visit | 1,800 | **dates are 2026-02-15 → 2026-08-30** (≈1 yr offset from sensors) |
| `plot_meta` | 1 row / plot | 120 | metadata only |
| `bc_benchmarks` | benchmark | **1** | only metric = greenhouse tomato annual yield (60 kg/m²/yr) |

**Dimensions available to group by:** crop (4: Strawberry 69, Tomato 29, Cucumber 13,
Pepper 9), treatment (7), treatment_block (4), region (4), climate_zone (3),
structure_type (2: Glass 66 / Polyfilm 54), heating_system (3), irrigation_system (3),
row_orientation (4), production_system (2), farm_id (8), greenhouse_id (20).

**Data quality issues found (computed):**
- **Scouting is decoupled from the sensor season.** Scouting dates are a year offset, and
  per-plot scouting `pest_severity` correlates ≈0 with sensor `alert_days` (0.04),
  `mean_stress` (0.04), and `profit` (−0.03), and is flat across treatments. *Scouting
  cannot be used to corroborate the pest-pressure story.*
- **Season financials don't reconcile with the daily cost log.**
  `corr(season total_cost_cad, summed daily_total_input_cost) = −0.21` (negative), a 2.4×
  scale gap; area- and capacity-normalization don't fix it (corr stays −0.16 to −0.22).
  They are two independent accounting universes.
- **Benchmark is thin and not season-matched:** one metric, *annual* tomato yield, vs our
  ≈7-month (210-day) partial season and only 29 tomato plots → any normalization is an
  **[ASSUMPTION]**.
- **Known/expected:** 6 plots have negative profit; the market-price CSV (2026+) feeds no
  derived table; 99.2% of *fired* alerts are High Pest Pressure (the documented quirk).
- **Two distinct "action" logs (do not conflate):** the **sensor-response log**
  (`action_taken`/`action_delay_days`/`post_action_stress_delta_3d`) has **5,509**
  response-days (4,820 on alert days); the **input-application log**
  (`is_precision_action` / `daily_precision_cost`) has **799** precision applications.
  The mechanism analysis uses the first; the cost analysis uses the second.

---

## 2. The core question — does precision beat routine?

Because precision = "respond to an alert" at the action level (not a plot-level regime),
"does precision beat routine" decomposes into three testable links: **(a) does responding
faster reduce stress?**, **(b) does lower stress raise outcomes?**, **(c) do more-precise
plots earn more?** The answers differ sharply by level of analysis — that contrast *is*
the finding.

| Link | Level | Result | Verdict |
|---|---|---|---|
| (a) Faster response → less stress | action event (n=5,509) | −0.0258 [−0.0282,−0.0235], p≈1e-98 | **Strong** |
| (b) Lower season stress → more profit | plot (n=120) | r=−0.348 (p≈1e-4) **but vanishes within treatment** | **Confounded** |
| (c) More-responsive plots → more profit | plot (n=120) | response-rate r=+0.02; resp-delay r=+0.23 (wrong sign) | **Absent** |

The mechanism (a) is real; the bottom-line (c) is not demonstrable here. Details in §3–4.

`precision_benefit_cad` (the field that *names* the answer): total **47,034 CAD**, mean
**391.9 / plot**, median 365.8. **[ASSUMPTION — see §5/Confounder A]:** it is
0.65 × profit floored at 0, so it asserts rather than measures that precision beats routine.

---

## 3. Why it wins — mechanism

### 3a. Response speed → stress relief (the strong link)

Restricting to action days (n=5,509), grouped by `action_delay_days`:

| delay (days) | mean post-action 3-day stress Δ | std | n |
|---|---|---|---|
| 0 (same day) | **−0.0494** | 0.043 | 2,358 |
| 1 | −0.0291 | 0.044 | 2,178 |
| 2 | −0.0112 | 0.045 | 973 |

Same-day vs delayed (≥1): difference **−0.0258, bootstrap 95% CI [−0.0282, −0.0235]**,
Welch t=−21.5, p≈2×10⁻⁹⁸. Each day of delay roughly **halves** the stress relief. This is
the cleanest causal-flavored evidence that responsiveness — the essence of precision ag —
does something measurable.

**Honest caveat:** speed reduces the *magnitude* of the post-action stress change but does
**not** reduce the rate of large forward stress *spikes* (`spike_3d` ≈ 0.068 / 0.068 /
0.071 across delay 0/1/2 — flat). And this is observational: same-day responses might
systematically occur under conditions where stress was already easier to arrest.

### 3b. Stress → outcomes (the moderate, confounded link)

Per-plot season-mean stress vs outcomes (Pearson, n=120):

| outcome | r | p |
|---|---|---|
| revenue | −0.404 | 5e-6 |
| profit | **−0.348** | 1e-4 |
| ROI | −0.268 | 3e-3 |
| yield (kg/m²) | −0.190 | 0.037 |
| marketable ratio | −0.140 | 0.126 (**n.s.**) |

Monotonic across stress terciles (profit 720 → 624 → 446; ROI 0.51 → 0.43 → 0.35, n=40
each). **But the link to yield is weak and to marketable ratio not significant** — and
marketable ratio is compressed (0.84–0.98, mean 0.908), with little room to move. The
stress→profit link runs more through *revenue* than *yield*, the first hint of a crop/price
confounder (see §4). **This link does not survive controlling for treatment (§4) — read §4
before trusting it.**

### 3c. Treatment effects (profit *and* stress)

| treatment | n | profit | ROI | stress | alert_days | marketable |
|---|---|---|---|---|---|---|
| **High Light** | 26 | **838.7** | 0.585 | **0.413** | **20.9** | 0.908 |
| High N | 16 | 650.6 | 0.414 | 0.447 | 35.1 | 0.900 |
| Control | 15 | 580.5 | 0.430 | 0.486 | 63.4 | 0.916 |
| Low N | 14 | 571.4 | 0.453 | 0.516 | 83.8 | 0.891 |
| Integrated Pest | 22 | 555.3 | 0.381 | 0.431 | 27.2 | **0.927** |
| Reduced Pest | 14 | 399.0 | 0.310 | 0.536 | 95.0 | 0.907 |
| **Shade** | 13 | **379.2** | 0.310 | **0.554** | **100.2** | 0.896 |

Bootstrap profit difference **vs Control** (95% CI): High Light **+259.9 [+48, +483]**
(sig), Reduced Pest **−181.5 [−358, −5]** (sig), Shade **−200.3 [−385, −23]** (sig);
High N / Low N / Integrated Pest indistinguishable from Control.
**The pattern: low pest burden → low stress → high profit (High Light), high burden → high
stress → low profit (Shade).** Integrated Pest delivers the best marketable ratio (0.927) —
a quality-protection angle. **Caveat:** treatments are *not randomized* (confounded with
crop/greenhouse), so these are associations, not causal treatment effects.

---

## 4. Confounders & threats to validity

This is where the precision story is won or lost. Each suspected confounder was tested.

**Confounder A — `precision_benefit_cad` is profit relabelled.** It equals
`clip(0.65 × profit, 0, ∞)`: corr 0.996, median ratio exactly 0.650 (range 0.560–0.650,
the deviations *unexplained* by any precision metric, all |r|<0.12), 6 loss plots → 0.
Worse, *across treatments* it correlates **−0.83 with alert burden** and only **+0.32 with
precision actions** — i.e., it credits the plots with the **fewest** pest problems with the
**most** "precision benefit." It measures profit, not precision effort. **Any claim built
on this field is circular.**

**Confounder B (the decisive one) — the stress→profit link is treatment, not stress.**
Raw `corr(mean_stress, profit) = −0.348`. Controlling for treatment (within-group
demeaning) it **vanishes: r=+0.094, p=0.31**. It *survives* within crop (−0.357),
greenhouse (−0.389), and region (−0.356) — so crop/site are **not** the confounder;
**treatment is**. Within a treatment, stress does not predict profit.

**Confounder C — stress ≈ pest burden.** `corr(alert_days, mean_stress) = +0.99`.
Partial correlation of stress→profit controlling for `alert_days` = **−0.103, p=0.26**.
Stress and profit are both downstream of exogenous pest-alert burden; stress is not an
independent lever you can pull for profit.

**Confounder D — crop is a price story, not a precision story.** Strawberry is the most
profitable crop (678.7) on the *lowest* yield (5.16 kg/m²) because of price: revenue per
kg/m² is 404 (Strawberry) vs 210 (Tomato). Stress is flat across crops (0.466–0.495), so
crop drives profit through price, independent of the stress channel.

**Confounder E — the plot-level "responsiveness pays" story is absent or wrong-signed.**
- `alert_response_rate` vs profit r=+0.02; vs stress r=−0.05 (≈ nothing).
- `mean_resp_delay` vs profit r=**+0.23** (slower → *higher* profit), and this survives
  controlling for alert burden (partial r=+0.22, p=0.015). Within-treatment signs flip
  (High Light +0.48, Shade +0.57, Control −0.33, Integrated Pest −0.17) — noise at n=13–26.
- Even among the **highest-burden** plots (top alert tercile, n=40), responsiveness doesn't
  help (resp-rate vs profit −0.03; resp-delay vs profit +0.24).
- `missed_alerts` (alerts not acted on), controlling for alert burden, predicts neither
  stress (partial r=0.12, p=0.19) nor profit (r=−0.01, p=0.93).
- **Caveat that softens this:** response rate is compressed (min 0.47, mostly 0.67–0.83) —
  there is no genuinely negligent plot in the sample to contrast against, so absence of a
  plot-level effect is partly absence of variation.

**Threats to validity (general):** (i) treatments are observational, not randomized;
(ii) the data is simulated — `precision_benefit_cad` and `post_action_stress_delta_3d` are
constructed fields whose generating assumptions we cannot see; (iii) season financials and
the daily cost log are inconsistent (corr −0.21), so no cross-log dollar ROI is valid;
(iv) scouting is decoupled (§1); (v) small cells — several treatment×crop cells have n<10,
and Pepper (n=9), Farm F07 (n=2), several greenhouses (n=2) are too small for stand-alone
claims.

**What survived:** the action-level mechanism (3a) survives everything — it is within-event
and not a between-plot comparison. The treatment *ordering* (3c) is robust as an
*association*. Everything that depends on `precision_benefit_cad` or on a plot-level
"responsiveness → profit" link did **not** survive.

---

## 5. Segmentation — where precision pays most

- **By mechanism leverage:** precision's lever (fast response cutting stress) has the most
  *room to act* where pest burden is highest — Shade (100 alert-days), Reduced Pest (95),
  Low N (84), Control (63). These plots generate the most alert-response events.
- **But high-burden plots still earn less** (pest burden dominates profit), and within a
  burden level responsiveness does not separate winners from losers (§4E). So precision is
  best framed as **loss mitigation in high-pest contexts**, not profit maximization.
- **Structural dimensions are weak differentiators** (all stress ≈0.47): Polyfilm slightly
  out-earns Glass (629 vs 571), EBB-Flow irrigation tops Fertigation (666 vs 518), Interior
  climate edges Coastal/Fraser Valley (663 vs 590/545) — but none of these move stress, so
  they're not precision channels.
- **Quality angle:** Integrated Pest gives the best marketable ratio (0.927) at moderate
  profit — relevant if the lender's farmer sells into a premium/marketable-grade market.

---

## 6. The financial argument

**Defensible (hard numbers):**
- Precision is a **cheap overlay**: 4.6% of input cost (18,646 / 404,050 CAD) and 8.4% of
  actions (799 / 9,559).
- Daily costs decompose exactly: `total_input = energy+fertilizer+labor+pesticide+water`
  and `total_input = precision_cost + routine_cost` (both exact to the cent).
- Total season profit **71,654 CAD** across 120 plots (mean 597, median 563); 6 plots
  unprofitable.

**[ASSUMPTION] — handle with care:**
- "Total precision benefit = **47,034 CAD** (65.6% of profit)" is `precision_benefit_cad`,
  which is 0.65 × profit floored at 0 (Confounder A). It is not an independent return.
- **No defensible ROI-on-precision exists in this data.** Dividing benefit (47,034, a
  season-level field) by precision cost (18,646, a daily-log field) gives ~2.5 — but the
  two accounting universes do not reconcile (corr −0.21), and the numerator is circular. We
  should **not** present a precision ROI multiple.
- Any comparison to the BC tomato benchmark (60 kg/m²/yr) requires annualizing a 7-month
  partial season for 29 plots — an estimate, not a measurement.

---

## What the data does NOT support (claims to avoid)

1. ❌ **"Precision agriculture adds ~65% to profit" / "delivered 47k CAD."** That figure is
   profit × 0.65 by construction — circular, not evidence.
2. ❌ **"Lower plant stress causes higher profit."** The link disappears once you control
   for treatment (r=+0.09, p=0.31) or pest burden (partial r=−0.10, p=0.26). Stress is a
   *symptom* of pest pressure, not an independent profit lever.
3. ❌ **"Plots that respond faster / more often to alerts earn more."** Plot-level
   responsiveness→profit is ~0 or wrong-signed (+0.23 for slower response), and fails even
   among high-burden plots.
4. ❌ **"Precision ROI is ~2.5×."** Crosses two non-reconciling cost universes; do not use.
5. ❌ **"Scouting confirms the pest-pressure pattern."** Scouting is decoupled (corr ≈0).
6. ❌ **"Treatment X *causes* +Y profit."** Treatments are observational/confounded with
   crop and greenhouse; report as associations only.
7. ❌ **"Faster response prevents stress spikes."** Spike rate is flat across response delay.
8. ⚠️ **Avoid stand-alone claims on Pepper (n=9), small farms/greenhouses (n≤3), or any
   treatment×crop cell with n<5** — flag small-N instead.

---

## Strongest story for the presentation

**Lead with the mechanism, be honest about the ceiling, reframe the value as risk.**

**Act 1 — The proof (unassailable).** "When GreenLeaf acts on a sensor alert the *same
day*, plant stress falls about twice as much as when the response slips a day or two:
−0.049 vs −0.024 stress units, 95% CI excludes zero, across 5,509 real response events,
p≈10⁻⁹⁸. Every day of delay roughly halves the benefit." This is the within-event result no
statistician can dismiss. It *is* the value of precision: turning fixed schedules into
same-day, data-driven responses (exactly the mechanism the precision-ag literature
describes).

**Act 2 — What stress is, and what it costs.** "Stress is overwhelmingly pest-driven —
99% of alerts are High Pest Pressure, and alert burden explains stress almost perfectly
(r=0.99). The operations facing the most pest pressure (Shade-type, high-alert plots) earn
~460 CAD/plot less than the lowest-pressure ones [bootstrap CI 242–705]." This sizes the
problem precision is built to fight.

**Act 3 — The honest ceiling (this is what earns the lender's trust).** "We will *not*
claim precision added 65% to profit — that figure in the data is profit relabelled. And at
the whole-season level we cannot yet show that more-responsive plots earned more, because
responsiveness is already high across the board and pest pressure dominates the variance.
What we *can* prove is the mechanism and its cost." Volunteering this is what makes Acts 1
and 4 credible.

**Act 4 — The defensible pitch.** "Precision is a **cheap, proven, real-time lever against
pest-driven stress** — 4.6% of input spend, demonstrably effective at the moment it's used,
and most valuable for the high-pest-pressure operations that have the most to lose. The
financing case is **risk management**: precision blunts the stress shocks that turn a good
season into a loss-making one (6 of 120 plots lost money), and the farmers who need it most
are exactly those under heaviest pest pressure."

**Where to focus the dashboard:** (1) the response-delay → stress-relief curve (the proof);
(2) the alert-burden → stress → profit gradient by treatment (the problem); (3) an honest
"precision is a 4.6%-of-cost overlay" framing; and (4) a clearly-labelled treatment of
`precision_benefit_cad` as a *planning assumption* (0.65 × profit), never as a measured
return. Avoid any chart that implies a measured precision ROI multiple.

---

*Reproduce all numbers: `python analysis/eda.py`. Functions are importable for Phase-2
chart code. No derived data, `build_derived.py`, or app code was modified.*

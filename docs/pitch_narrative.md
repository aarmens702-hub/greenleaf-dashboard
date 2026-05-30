# GreenLeaf — 5-Minute Pitch Narrative (deck + live demo)

A slide-by-slide plan tying the **GreenLeaf_Pitch** deck to a **live demo** of the rebuilt
dashboards (`streamlit run app/streamlit_app/main.py`). The spine is the honest 4-act arc:
**Proof → Problem → Honest ceiling → Risk-lever pitch.** Every claim maps to a source; we
win the room by being the team that *refused* to overclaim.

Audience: farmers, RBC lenders, technical judges. Tone: calm, evidence-first, candid.
Demo tabs: **Operations** (technical), **Financials** (lender), **Ask GreenLeaf** (chat).

---

## The arc at a glance
| Act | Message in one line | Live moment |
|---|---|---|
| 1 · Proof | Fast, sensor-driven response measurably cuts crop stress. | Operations → *Does acting work?* |
| 2 · Problem | Stress is pest-burden driven; it concentrates losses. | Operations → *Alert exposure* / Financials → *loss tail* |
| 3 · Honest ceiling | We will **not** claim precision raises profit — and here's why. | Financials → *precision-benefit honesty* |
| 4 · Risk-lever pitch | Precision is a cheap, proven **risk-management** lever — finance it as such. | Operations → *watch-list + driver* |

---

## Slide-by-slide (≈5 min)

**Slide 1 — Title / framing (20s).** *"GreenLeaf: turning a sensor wall into a financeable risk decision."*
Spoken: "We were asked whether precision agriculture pays. Our answer is honest and specific — here's what the data does and doesn't support." → *Transition:* "Start with the one thing that's unarguable."

**Slide 2 — ACT 1 · PROOF (45s).** Message: **"Acting same-day on an alert cuts plant stress about twice as much as waiting."**
- **Live:** Operations tab → **03 · Does acting work?** (the dot-and-CI dose-response).
- Numbers: −0.049 (same-day) vs −0.024 (delayed); diff −0.0258, 95% CI [−0.028, −0.024], p≈2×10⁻⁹⁸, n=5,509 actions. Each day of delay ≈ halves the benefit.
- Spoken: "Across 5,509 real responses, in every treatment, speed wins — and the confidence interval doesn't go near zero." → *Transition:* "So speed works. The question a lender asks next is: speed against *what*?"

**Slide 3 — ACT 2 · PROBLEM (45s).** Message: **"Stress is overwhelmingly pest-burden, and that burden is where the money is lost."**
- **Live:** Operations → **06 · Alert-burden exposure** (treatment-clustered), then Financials → **01 · loss tail** (6 named loss plots, worst-5% = −$123/plot).
- Numbers: alert-days↔stress r=0.99; 6/120 plots unprofitable; 5 of 6 in high-pest treatments (Shade / Reduced Pest).
- Spoken: "The plots that lose money are the high-pest ones — a *risk* problem, not a yield problem." → *Transition:* "Now, the tempting move is to claim precision fixes the P&L. We won't — and that's the point."

**Slide 4 — ACT 3 · HONEST CEILING (60s — the trust-builder).** Message: **"We refuse three claims the data can't support."**
- **Live:** Financials → **04 · The 'precision benefit' figure** (the scatter: benefit = 0.65 × profit) and the *defensible vs won't-claim* panel.
- The three refusals (with sources): (a) **No precision ROI** — `precision_benefit_cad` is 0.65×profit, circular (corr 0.996). (b) **Stress doesn't drive profit** — the link vanishes within treatment (r −0.35 → +0.09, p=0.31). (c) **No cross-ledger return** — season vs daily costs don't reconcile (r=−0.21).
- Spoken: "If a vendor shows you a precision-ROI number on this data, they computed profit times 0.65 and renamed it. We won't insult your underwriting team." → *Transition:* "So what *can* we sell you? A lever — priced honestly."

**Slide 5 — ACT 4 · THE RISK-LEVER PITCH (60s).** Message: **"Precision is a cheap, proven risk-management lever: a triage watch-list pointed at a controllable cause."**
- **Live:** Operations → **01 · watch-list** (rank by 3-day spike risk) → **05 · driver** (substrate-moisture trend is the #1 actionable lever) → **04 · early-warning performance** (gains curve).
- Numbers: precision = **4.6%** of input cost; watch-list top-20% catches **44%** of spikes at **2.2×** the base rate; #1 lever = substrate-moisture drift (a knob, not exogenous pests).
- Spoken: "For under 5% of input spend, you get a calibrated 3-day watch-list that doubles your hit-rate and points at irrigation — something the grower controls." → *Transition:* "That's the financing case."

**Slide 6 — The financing ask (40s).** Message: **"Finance precision as risk mitigation: bounded, named downside; cheap, validated tooling."**
- **Live:** Financials KPI row (total profit $71,654; 6 loss plots; 30% blended margin) + the loss-maker table.
- Spoken: "You're underwriting a known, shallow tail — 6 plots, worst −$253 — and funding a 4.6%-of-cost tool that demonstrably reduces the stress shocks behind those losses." → *Transition:* "And anyone can interrogate it."

**Slide 7 — Ask GreenLeaf (20s).** Message: **"Every number is one question away."**
- **Live:** Ask tab → ask *"How do same-day responses compare to delayed ones?"* → cited answer.
- Spoken: "Lender, farmer, or judge — ask in plain English; every answer cites its source." → *Close.*

**Slide 8 — Close (15s).** *"Honest, validated, financeable. Precision isn't a profit miracle — it's risk infrastructure, and that's exactly what a lender funds."*

---

## 30-second Q&A prep (own the ceiling)

- **"So precision doesn't increase profit?"** — "Not measurably in this data, and we won't pretend otherwise. It reduces *stress shocks* (proven), which is the risk a lender cares about. Profit is dominated by crop/price and pest burden, not response speed."
- **"Your model's precision is only 34%."** — "At the wide alarm, yes — it's a *sensitivity-first watch-list*: catch 76% of spikes, accept review volume. For triage we run the top-20% view: 44% hit-rate, 2.2× lift. We always show precision next to recall."
- **"Isn't alert-burden just pest pressure?"** — "Alert-days are a stress-threshold count (r=0.99 with stress), and 98% of the between-plot spread is treatment assignment, not organic pest exposure. We label it as exposure, not a pest discovery."
- **"Why logistic, not a fancier model?"** — "It ties the gradient-boosted tree on accuracy (AUC 0.78) and every flag is explained by named factors. Interpretability beats a 0.00x AUC gain."
- **"Is it validated honestly?"** — "Time-held-out on the late season (the real forecasting setup), with a calibration curve shown. We report the honest 0.78, not the rosier cross-plot 0.84."

---

## Claim → source map (nothing unsupported)
| Pitch claim | Number | Source (frozen) |
|---|---|---|
| Same-day ≈2× relief | −0.049 vs −0.024; p≈2e-98; n=5,509 | `eda.mechanism_delay_stress` (Operations §03) |
| Stress = pest burden | alert-days↔stress r=0.99 | `eda.merged` / `confounders` (Operations §06) |
| 6 loss plots, shallow tail | 6/120; worst −$253; ES₅ −$123 | `finance.loss_makers` / `profit_tail` (Financials §01) |
| precision_benefit circular | =0.65×profit, corr 0.996 | `finance.precision_benefit_check` (Financials §04) |
| stress→profit vanishes | r −0.35 → +0.09, p=0.31 | `eda.confounders` (frozen finding) |
| ledgers don't reconcile | r=−0.21 | `finance.precision_benefit_check` |
| precision = 4.6% of cost | 4.6% / labor 63% | `finance.cost_structure` (Financials §02) |
| watch-list lift | top-20% 44%, 2.2× | `insights.gains_curve` / model metrics (Operations §04) |
| #1 actionable lever | substrate-moisture trend, |contrib| 0.64 | `risk_score` logistic coefficients (Operations §05) |
| model performance | AUC-ROC 0.78, AUC-PR 0.41, Brier 0.13 | `risk_score` validation_metrics (Operations §07) |
| season totals | $71,654 profit, 30% margin, 43% ROI | `finance.headline` (Financials KPIs) |

*Every figure on a slide is reproducible via `analysis/*` + `app/streamlit_app/insights.py`. No statistic is computed in the slide or the UI layer.*

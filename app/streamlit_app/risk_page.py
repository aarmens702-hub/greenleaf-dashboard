"""Operations tab — technical-manager view of stress-spike risk.

Built ONLY from theme.py primitives and insights.py read-only helpers (which compose the
frozen analysis layer). No data/number is computed here. Validated insights only:
S1 (response dose-response), S2 (driver breakdown), Y1 (early-warning gains, caveat),
Y2 (seasonal trajectory, caveat), Y3 (alert exposure, caveat), Y5 (moisture drift),
plus the watch-list (operational core) and an honest model-transparency section.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.streamlit_app import theme as T
from app.streamlit_app import insights as I


def _treat_colors():
    order = I.weekly_stress_trajectory()["order"]   # ascending season-mean stress
    return {t: T.CAT[i % len(T.CAT)] for i, t in enumerate(order)}


def _page_title():
    st.markdown(
        '<div class="gl-h1">Stress-Risk Operations</div>'
        '<div class="gl-sub">A validated 3-day early-warning for triage — where to act first, '
        'why, and proof that acting works. Operational, never profit prediction.</div>'
        '<hr class="gl-hair">', unsafe_allow_html=True)


def _kpis(wl):
    """Single consolidated operational KPI row — six cards rendered as 2×3 (one visual
    group, no header between) so they stay legible at full width. Every value traces to a
    source helper; nothing is recomputed here."""
    dose = I.response_dose()
    by = dose["by"].set_index("delay")
    delayed = float((by.loc[[1, 2], "mean"] * by.loc[[1, 2], "count"]).sum() / by.loc[[1, 2], "count"].sum())
    adv = by.loc[0, "mean"] / delayed
    k = I.operational_kpis()
    elevated = int((wl.spike_prob >= 0.25).sum()) if not wl.empty else 0
    cards = [
        T.kpi(f"{elevated}", "High-risk plots today", "≥25% spike probability"),
        T.kpi(f"{adv:.1f}×", "Same-day response advantage",
              f"speed payoff · {by.loc[0,'mean']:.3f} vs {delayed:.3f} delayed", T.TERRA),
        T.kpi(f"{k['healthy_rate']*100:.1f}%", "Healthy plot-day rate", "stress &lt; 0.50"),
        T.kpi(f"{k['performance_score']*100:.1f}%", "Avg plot performance", "5-factor blend score", T.INDIGO),
        T.kpi(f"{k['alert_rate']*100:.1f}%", "Alert rate", "plot-days with an alert", T.INDIGO),
        T.kpi(f"{k['alert_response_rate']*100:.1f}%", "Alert response rate", "of alerts, share acted on"),
    ]
    T.kpi_row(cards[:3])
    T.kpi_row(cards[3:])


def _watchlist_panel(wl, top_n):
    T.card_title("Watch-list · direct fast response here first")
    view = wl.head(top_n)[["plot_id", "crop", "treatment", "greenhouse_id", "spike_pct", "top_driver"]]
    maxv = max(5.0, float(view.spike_pct.max()))
    st.dataframe(view, hide_index=True, use_container_width=True, height=430, column_config={
        "plot_id": st.column_config.TextColumn("Plot"), "crop": st.column_config.TextColumn("Crop"),
        "treatment": st.column_config.TextColumn("Treatment"), "greenhouse_id": st.column_config.TextColumn("GH"),
        "spike_pct": st.column_config.ProgressColumn("Spike risk · 3d", format="%.1f%%", min_value=0, max_value=maxv),
        "top_driver": st.column_config.TextColumn("Top environmental driver", width="medium")})


def _today_dist(wl, base):
    T.card_title("Today's risk spread (all 120 plots)")
    fig = go.Figure(go.Histogram(x=wl.spike_prob * 100, nbinsx=20, marker_color=T.SAGE,
                                 marker_line_color=T.PAPER, marker_line_width=1))
    fig.add_vline(x=base * 100, line_color=T.TERRA, line_dash="dot")
    st.plotly_chart(T.style_fig(fig, 200, xtitle="spike probability (%)", ytitle="plots"), use_container_width=True)


def _drilldown(wl, date_iso):
    if wl.empty:
        return
    labels = {r.plot_id: f"{r.plot_id} · {r.crop} · {r.treatment} ({r.spike_pct:.1f}%)" for r in wl.itertuples()}
    pid = st.selectbox("Plot", list(wl.plot_id), index=0, format_func=lambda p: labels.get(p, p))
    c = st.columns(2, gap="large")
    with c[0]:
        T.card_title("Why this plot — signed risk drivers")
        ct = I.plot_contributions(pid, date_iso)
        ct = ct.reindex(ct.contribution.abs().sort_values(ascending=False).index).head(8).sort_values("contribution")
        fig = go.Figure(go.Bar(x=ct.contribution, y=ct.phrase, orientation="h",
                               marker_color=[T.SAGE if v > 0 else T.MUTED for v in ct.contribution]))
        st.plotly_chart(T.style_fig(fig, 280, xtitle="← lowers risk     log-odds contribution     raises risk →"),
                        use_container_width=True)
    with c[1]:
        T.card_title("Substrate-moisture drift (early-warning lever)")
        md = I.moisture_drift(pid, date_iso)
        fig = go.Figure()
        if len(md):
            fig.add_trace(go.Scatter(x=md.date, y=[md.upper.iloc[0]] * len(md), line=dict(color=T.HAIR, width=1),
                                     name="±3σ"))
            fig.add_trace(go.Scatter(x=md.date, y=[md.lower.iloc[0]] * len(md), line=dict(color=T.HAIR, width=1),
                                     fill="tonexty", fillcolor="rgba(217,212,199,0.35)", name="band"))
            fig.add_trace(go.Scatter(x=md.date, y=md.substrate_moisture, mode="markers",
                                     marker=dict(color=T.MUTED, size=5), name="moisture"))
            fig.add_trace(go.Scatter(x=md.date, y=md.ewma, line=dict(color=T.SAGE, width=2), name="EWMA"))
        st.plotly_chart(T.style_fig(fig, 280, ytitle="substrate moisture"), use_container_width=True)
    T.caveat_note("The drift monitor is operational triage only — it flags <b>recurring</b> drift, but does NOT "
                  "pre-empt a plot's first hard alert. No profit/ROI claim.")


def _response_proof():
    # Heat map: rows = response delay, cols = treatment, colour = mean post-action relief.
    mean, cnt = I.delay_treatment_relief()
    ylab = {0: "same day", 1: "1-day delay", 2: "2-day delay"}
    treatments = list(mean.columns)
    z = mean.values
    z_shown = np.where(cnt.values < 30, np.nan, z)   # de-emphasize small-n cells (NaN -> blank)
    fig = go.Figure(go.Heatmap(
        z=z_shown, x=treatments, y=[ylab[i] for i in mean.index],
        colorscale=[[0.0, "#EFE9DC"], [1.0, T.SAGE_SOFT]], reversescale=True,
        xgap=3, ygap=3, hoverongaps=False,
        colorbar=dict(title=dict(text="relief", font=dict(size=9)), thickness=9,
                      outlinewidth=0, tickfont=dict(size=9)),
        hovertemplate="%{y} · %{x}<br>mean relief %{z:.3f}<extra></extra>"))
    for i, ridx in enumerate(mean.index):
        for j, t in enumerate(treatments):
            n = int(cnt.values[i, j])
            small = n < 30
            fig.add_annotation(
                x=t, y=ylab[ridx], showarrow=False,
                text=f"{z[i, j]:.3f}<br><span style='font-size:8px'>n={n}{' ⚠' if small else ''}</span>",
                font=dict(size=11, color=T.MUTED if small else T.INK))
    fig = T.style_fig(fig, 300, xgrid=False, ygrid=False)
    fig.update_yaxes(autorange="reversed")   # same-day at top
    st.plotly_chart(fig, use_container_width=True)
    T.caveat_note("<b>Descriptive only.</b> Cells show observed relief by delay×treatment — <b>not</b> a causal "
                  "claim that any treatment 'responds better'. Treatments weren’t randomized and differ in pest "
                  "burden and crop mix; small-n cells (n &lt; 30, de-emphasized) would be unreliable (none here — "
                  "all cells n ≥ 86). The confounder-safe result is the overall same-day ≈ 2× relief; the "
                  "per-treatment cells are exploratory.")


def _early_warning():
    g = I.gains_curve()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=T.MUTED, width=1, dash="dot"), name="random"))
    fig.add_trace(go.Scatter(x=g["frac"], y=g["capture"], line=dict(color=T.SAGE, width=2.5), name="model"))
    for f, key in [(0.10, "top10"), (0.20, "top20"), (0.30, "top30")]:
        fig.add_trace(go.Scatter(x=[f], y=[g[key]], mode="markers+text", marker=dict(color=T.TERRA, size=8),
                                 text=[f"  {g[key]*100:.0f}%"], textposition="middle right",
                                 textfont=dict(size=10, color=T.INK)))
    st.plotly_chart(T.style_fig(fig, 260, xtitle="% of plot-days inspected", ytitle="% of upcoming spikes caught"),
                    use_container_width=True)
    T.caveat_note("Operational early-warning triage only — <b>NOT profit prediction</b>; target is up to 3 days "
                  "forward. Top-20% catches ~44% of spikes (you miss ~56%); at the wider 0.18 alarm you catch "
                  "~76% but only ~1 in 3 alarms is real. Always read precision next to recall. Time-held-out "
                  "(Aug–Sep 2025), in-season validation only.")


def _drivers_and_trajectory(tcolors):
    c = st.columns(2, gap="large")
    with c[0]:
        T.card_title("What raises spike risk — the actionable levers")
        di = I.driver_importance()
        env = di[di.env].head(6).sort_values("mean_abs")
        fig = go.Figure(go.Bar(x=env.mean_abs, y=env.label, orientation="h", marker_color=T.SAGE,
                               text=[f"{v:.2f}" for v in env.mean_abs], textposition="outside"))
        st.plotly_chart(T.style_fig(fig, 280, xtitle="mean |log-odds contribution|"), use_container_width=True)
    with c[1]:
        T.card_title("Stress over the season, by treatment")
        w = I.weekly_stress_trajectory()
        piv = w["piv"]
        fig = go.Figure()
        for t in w["order"]:
            sub = piv[piv.treatment == t].sort_values("week")
            fig.add_trace(go.Scatter(x=sub.week, y=sub.plant_stress_index, mode="lines",
                                     line=dict(color=tcolors[t], width=2), name=t))
        st.plotly_chart(T.style_fig(fig, 280, legend=True, xtitle="ISO week", ytitle="mean plant stress"),
                        use_container_width=True)
        T.caveat_note(f"Descriptive operational view. Day-of-season explains ~{w['eta_timing']*100:.0f}% of stress "
                      f"variance vs ~{w['eta_treat']*100:.0f}% for treatment (one-way η²) — the shared seasonal "
                      f"decline dominates; treatment is a stable ~{w['spread']:.2f}-wide offset. stress→profit "
                      f"collapses within treatment (r +0.09, p=0.31): never imply lower-stress treatments earn more.")


def _recommendation_uptake():
    g, overall, _ = I.recommendation_adherence()
    fig = go.Figure(go.Bar(
        x=g.adhere * 100, y=g.rec, orientation="h", marker_color=T.SAGE,
        text=[f"{a*100:.0f}%  (n={n:,})" for a, n in zip(g.adhere, g.n)], textposition="auto"))
    fig.add_vline(x=overall * 100, line_color=T.MUTED, line_dash="dot")
    st.plotly_chart(T.style_fig(fig, 240, xtitle="% of recommendations acted on"), use_container_width=True)
    T.caveat_note("Two honest limits: (1) ‘acted on’ = an action was logged on a recommendation-day — an "
                  "<b>uptake proxy</b>; the data can’t confirm the action matched the specific recommendation "
                  "(the input-application log doesn’t align with this one — only ~5% same-day overlap). "
                  "(2) Relief-when-acted is the observed 3-day stress change, not a randomized causal effect. "
                  "Descriptive, never a profit claim.")


def _transparency():
    metrics, cal, _ = I.model_metrics()
    c = st.columns(2, gap="large")
    with c[0]:
        T.card_title("Metrics vs baselines (held-out test)")
        show = metrics.set_index("model")[["AUC_ROC", "AUC_PR", "Brier", "Recall", "Prec@top20%"]].round(3)
        st.dataframe(show, use_container_width=True)
    with c[1]:
        T.card_title("Calibration — predicted vs observed")
        mx = float(cal.mean_predicted.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], line=dict(color=T.MUTED, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=cal.mean_predicted, y=cal.fraction_positive, mode="lines+markers",
                                 line=dict(color=T.SAGE, width=2)))
        st.plotly_chart(T.style_fig(fig, 240, xtitle="predicted risk", ytitle="observed spike rate"),
                        use_container_width=True)
    T.caveat_note("<b>Honest ceiling:</b> much of the raw signal is mean-reversion (low-stress plots rebound) — the "
                  "model's value-add is the moisture/environment read. It is a sensitivity-first watch list "
                  "(~1 in 3 flags is a real spike), validated only in-season, and predicts a stress spike — "
                  "<b>never profit</b> (plot-level responsiveness does not predict profit).")


def render_risk() -> None:
    T.inject_css()
    T.brand_bar("OPERATIONS")
    _page_title()

    base = float(I.model_metrics()[2])  # base rate — still used by the risk-spread caption below
    dmin, dmax = I.date_range()
    default = min(max(dmin.date().replace(month=8, day=15), dmin.date()), dmax.date())
    ctl = st.columns([2, 1, 2], gap="large")
    with ctl[0]:
        as_of = st.date_input("As-of date (predict next 3 days)", value=default,
                              min_value=dmin.date(), max_value=dmax.date())
    with ctl[1]:
        top_n = st.slider("Watch-list rows", 5, 30, 12)
    date_iso = as_of.isoformat()
    wl = I.watchlist(date_iso)
    tcolors = _treat_colors()

    # One consolidated operational KPI row (after the date control so "today's" count is correct)
    _kpis(wl)

    T.section_header("01 · Where to act first", "The daily watch-list")
    a = st.columns([3, 2], gap="large")
    with a[0]:
        _watchlist_panel(wl, top_n)
    with a[1]:
        _today_dist(wl, base)

    T.section_header("02 · Why this plot", "Drill-down: drivers & moisture drift")
    _drilldown(wl, date_iso)

    T.section_header("03 · Does acting work?", "Response speed → stress relief")
    _response_proof()

    T.section_header("04 · Early-warning performance", "How much the watch-list earns its keep")
    _early_warning()

    T.section_header("05 · What drives risk & how stress moves", "Levers and the seasonal picture")
    _drivers_and_trajectory(tcolors)

    T.section_header("06 · Are the recommendations followed?", "Decision-support uptake — and whether it helps")
    _recommendation_uptake()

    with st.expander("Model details & validation", expanded=False):
        _transparency()

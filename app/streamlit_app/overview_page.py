"""Overview tab — a curated "best of both worlds" front door to the dashboard.

Built ONLY from theme.py primitives and insights.py read-only helpers; every number traces to a
frozen analysis function or the clean all_scores_complete.csv (no recompute in the UI). The 5
headline KPIs sit at the very top; global Crop + Date-range filters then drive every panel:
  01 Input spending over time — daily total input cost (canonical ledger), as a line
  02 Plots at risk           — composite HEALTH-SCORE risk for the selected day (NOT the spike model)
  03 Plot-health mix          — plots by plot_health_category (Excellent…Critical), all_scores_complete donut
  04 Plot scores over season  — daily mean plot-health / env-stability / risk (all_scores_complete) line
  05 Plot health by treatment — mean plot_health_90 per treatment (all_scores_complete) bar
  06 Health vs composite risk — per-plot scatter coloured by crop (all_scores_complete)

The date control is a range: the time series span it; the snapshots use the latest day in range.
Defaults: all crops, full season.
"""
import plotly.graph_objects as go
import streamlit as st

from app.streamlit_app import theme as T
from app.streamlit_app import insights as I

_CRITICAL = "#7A3B2E"   # deepest warm tone (shared with the Financials risk-level mix)
# plot_health_category colours: healthiest (green) -> worst (deep warm)
_HEALTH_CAT_COLORS = {"Excellent": T.SAGE, "Good": T.SAGE_SOFT, "Fair": T.CAT[2],
                      "Poor": T.TERRA, "Critical": _CRITICAL}
_SCORE_COLORS = {"plot_health_90": T.SAGE, "env_stability_90": T.INDIGO, "risk_score_90": T.TERRA}


def _page_title():
    st.markdown(
        '<div class="gl-h1">At a Glance</div>'
        '<div class="gl-sub">The whole operation on one screen — headline numbers up top, then spend, risk and '
        'plant-health scores. Filter by crop and date; every panel responds.</div>'
        '<hr class="gl-hair">', unsafe_allow_html=True)


def _kpis():
    """The five headline KPIs, pinned at the very top of the tab (one visual group, rendered 3+2 so
    five-across never gets cramped). Every value traces to I.headline_kpis(); the performance score is
    the PROFIT-FREE recompute (clean_plot_score), not the contaminated composite."""
    k = I.headline_kpis()
    wk = f"{k['week_start'].strftime('%b %-d')}–{k['week_end'].strftime('%-d')}"
    cards = [
        T.kpi(f"{k['avg_performance']:.1f}", "Avg plot performance", "profit-free score · 0–90", T.INDIGO),
        T.kpi(f"{k['avg_health']*100:.1f}%", "Avg plot health", "health component (H)"),
        T.kpi(f"{k['avg_roi']*100:.1f}%", "Avg ROI", "return on season cost", T.INDIGO),
        T.kpi(f"{k['weekly_response_rate']*100:.1f}%", "Weekly response rate", f"alerts acted on · {wk}"),
        T.kpi(f"${k['profit_per_m2']:.2f}", "Profit per m²", "season profit ÷ total area", T.TERRA),
    ]
    T.kpi_row(cards[:3])
    T.kpi_row(cards[3:])


def _filters():
    """Global Crop + Date-range controls. Defaults: all crops, full season (snapshots use the latest
    day in range)."""
    dmin, dmax = I.date_range()
    all_crops = I.crops_list()
    c = st.columns([3, 2], gap="large")
    with c[0]:
        picked = st.multiselect("Crop", all_crops, default=all_crops,
                                help="Filters every panel. Empty = all crops.")
    with c[1]:
        rng = st.date_input("Date range (snapshots use the latest day)",
                            value=(dmin.date(), dmax.date()),
                            min_value=dmin.date(), max_value=dmax.date())
    sel_crops = picked or all_crops                       # empty selection = all
    if isinstance(rng, (list, tuple)):                    # tuple while a range is set
        start, end = (rng[0], rng[-1]) if len(rng) >= 2 else (rng[0], rng[0])
    else:                                                 # single date mid-selection
        start = end = rng
    return sel_crops, start, end, all_crops


def _scope(sel_crops, all_crops):
    return "all crops" if set(sel_crops) == set(all_crops) else ", ".join(sel_crops)


# --- 01 · input spend (full-season line) -------------------------------------
def _spend_line(sel_crops, start, end, all_crops):
    s = I.overview_spend(sel_crops, start, end)
    scope = _scope(sel_crops, all_crops)
    T.card_title(f"Daily total input spend · {scope} · {start.strftime('%b %-d')}–{end.strftime('%-d, %Y')}")
    fig = go.Figure(go.Scatter(
        x=s.date, y=s.spend, mode="lines", line=dict(color=T.SAGE, width=2),
        fill="tozeroy", fillcolor="rgba(61,90,61,0.06)",
        hovertemplate="%{x|%b %-d}<br>$%{y:,.0f}<extra></extra>"))
    st.plotly_chart(T.style_fig(fig, 280, xtitle="date", ytitle="CAD / day"), use_container_width=True)


# --- 02 · composite-risk table -----------------------------------------------
def _risk_table(sel_crops, end, all_crops):
    day, target = I.composite_risk_today(end.isoformat(), sel_crops)
    c = st.columns([3, 2], gap="large")
    with c[0]:
        T.card_title(f"Highest composite-risk plots · {target.strftime('%b %-d, %Y')} · "
                     f"{_scope(sel_crops, all_crops)}")
        view = day.head(12)[["plot_id", "crop", "risk_score", "risk_level"]]
        st.dataframe(view, hide_index=True, use_container_width=True, height=430, column_config={
            "plot_id": st.column_config.TextColumn("Plot"),
            "crop": st.column_config.TextColumn("Crop"),
            "risk_score": st.column_config.ProgressColumn("Composite risk · today", format="%.0f",
                                                          min_value=0, max_value=90),
            "risk_level": st.column_config.TextColumn("Level")})
    with c[1]:
        T.card_title("Risk-level mix")
        order = ["Low", "Moderate", "Elevated", "High", "Critical"]   # ascending risk
        ramp = [T.SAGE_SOFT, T.CAT[2], T.CAT[6], T.TERRA, _CRITICAL]   # calm green -> deep warm
        counts = day.risk_level.value_counts()
        ys = [int(counts.get(lv, 0)) for lv in order]
        fig = go.Figure(go.Bar(x=order, y=ys, marker_color=ramp))
        st.plotly_chart(T.style_fig(fig, 230, ytitle="plots", xgrid=False), use_container_width=True)


# --- 03 · plot-health donut ---------------------------------------------------
def _health_donut(sel_crops, end, all_crops):
    counts, target = I.health_category_distribution(end.isoformat(), sel_crops)
    n = int(counts.n.sum())
    cats = list(counts.category)
    c = st.columns([3, 2], gap="large")
    with c[0]:
        T.card_title(f"Plot-health mix · {target.strftime('%b %-d, %Y')} · {_scope(sel_crops, all_crops)}")
        fig = go.Figure(go.Pie(
            labels=cats, values=counts.n, hole=0.55, sort=False, direction="clockwise",
            marker=dict(colors=[_HEALTH_CAT_COLORS.get(b, T.MUTED) for b in cats],
                        line=dict(color=T.PAPER, width=2)),
            texttemplate="%{label}<br>%{value} · %{percent}", textfont=dict(size=11),
            hovertemplate="%{label}: %{value} plots (%{percent})<extra></extra>"))
        fig = T.style_fig(fig, 300, legend=False)
        fig.update_layout(annotations=[dict(text=f"{n}<br>plots", x=0.5, y=0.5, showarrow=False,
                                            font=dict(family=T.SERIF, size=18, color=T.INK))])
        st.plotly_chart(fig, use_container_width=True)
    with c[1]:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        for cat in cats:
            v = int(counts.loc[counts.category == cat, "n"].iloc[0])
            col = _HEALTH_CAT_COLORS.get(cat, T.MUTED)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;font-size:0.82rem;color:{T.INK}">'
                f'<span style="width:11px;height:11px;background:{col};display:inline-block"></span>'
                f'<b>{cat}</b> — {v} plots</div>', unsafe_allow_html=True)


# --- 04 · scores time series (all_scores_complete) ---------------------------
def _scores_trend(sel_crops, start, end, all_crops):
    g = I.scores_timeseries(sel_crops, start, end)
    T.card_title(f"Plot scores over the season · {_scope(sel_crops, all_crops)} · "
                 f"{start.strftime('%b %-d')}–{end.strftime('%-d, %Y')}")
    fig = go.Figure()
    for col, lab in I.SCORE_SERIES:
        fig.add_trace(go.Scatter(x=g.date, y=g[col], mode="lines", name=lab,
                                 line=dict(color=_SCORE_COLORS[col], width=1.8),
                                 hovertemplate=f"{lab}<br>%{{x|%b %-d}}: %{{y:.0f}}<extra></extra>"))
    st.plotly_chart(T.style_fig(fig, 300, legend=True, xtitle="date", ytitle="score · 0–90"),
                    use_container_width=True)


# --- 05 · plot health by treatment (all_scores_complete) ---------------------
def _scores_bar(sel_crops, start, end, all_crops):
    g = I.scores_by_dimension("treatment", "plot_health_90", sel_crops, start, end)
    T.card_title(f"Average plot health by treatment · {_scope(sel_crops, all_crops)}")
    fig = go.Figure(go.Bar(
        x=g.value, y=g.category, orientation="h", marker_color=T.SAGE,
        text=[f"{v:.0f}" for v in g.value], textposition="auto",
        customdata=g.n, hovertemplate="%{y}: %{x:.1f} (n=%{customdata})<extra></extra>"))
    fig = T.style_fig(fig, max(240, 60 + 26 * len(g)), xtitle="mean plot health · 0–90 (higher = better)")
    fig.update_traces(cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


# --- 06 · health vs risk scatter (all_scores_complete) -----------------------
def _scores_scatter(sel_crops, end, all_crops):
    df, target = I.scores_scatter(end.isoformat(), sel_crops)
    T.card_title(f"Plot health vs composite risk · {target.strftime('%b %-d, %Y')} · {_scope(sel_crops, all_crops)}")
    fig = go.Figure()
    for i, (crop, sub) in enumerate(df.groupby("crop")):
        fig.add_trace(go.Scatter(
            x=sub.plot_health_90, y=sub.risk_score_90, mode="markers", name=crop, text=sub.plot_id,
            marker=dict(size=8, color=T.CAT[i % len(T.CAT)], line=dict(width=0.5, color=T.PAPER)),
            hovertemplate=f"{crop} %{{text}}<br>health %{{x:.0f}} · risk %{{y:.0f}}<extra></extra>"))
    st.plotly_chart(T.style_fig(fig, 320, legend=True, xtitle="plot health · 0–90 (higher = better)",
                                ytitle="composite risk · 0–90 (higher = worse)"), use_container_width=True)


def render_overview() -> None:
    T.inject_css()
    T.brand_bar("OVERVIEW")
    _page_title()

    _kpis()                                              # KPIs pinned at the very top
    sel_crops, start, end, all_crops = _filters()

    T.section_header("01 · Input spending over time", "Full-season daily input cost")
    _spend_line(sel_crops, start, end, all_crops)

    T.section_header("02 · Plots at risk", "Composite health-score risk for the selected day")
    _risk_table(sel_crops, end, all_crops)

    T.section_header("03 · Plot-health mix", "How the greenhouse is holding up, by stress band")
    _health_donut(sel_crops, end, all_crops)

    T.section_header("04 · Plot scores over the season", "Health, stability & risk trends — from all_scores_complete")
    _scores_trend(sel_crops, start, end, all_crops)

    T.section_header("05 · Average plot health by treatment", "Which treatment keeps plants healthiest")
    _scores_bar(sel_crops, start, end, all_crops)

    T.section_header("06 · Plot health vs composite risk", "Per-plot snapshot, coloured by crop")
    _scores_scatter(sel_crops, end, all_crops)

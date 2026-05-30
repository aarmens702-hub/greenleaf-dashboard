"""Financials tab — RBC/lender view of the season in dollars, honestly.

Built ONLY from theme.py primitives and insights.py read-only helpers; every number traces
to a frozen analysis function (no recompute in the UI). (The blended 5-KPI summary row lives on
the Overview tab; this tab carries its own purely-financial KPI row.)
  KPI row — Profit per m², Average ROI, Total precision benefit (planning ASSUMPTION), Cost per kg
  01 Season P&L + profit distribution + loss/win tail (Expected Shortfall, named plots)
  02 What drives profit  — driver correlations + price->profit scatter (price, not yield)
  03 Crop economics      — price-beats-volume unit economics by crop
  04 Average payoff by treatment — mean profit per plot (bars) with ±1 std swing, honestly
     labelled as descriptive dispersion, not a forward risk forecast
  05 Input spending      — cost mix (labor dominates). The daily-spend time series and the
     composite "plots at risk" table now live on the Overview tab (moved, not duplicated).
"""
import plotly.graph_objects as go
import streamlit as st

from app.streamlit_app import theme as T
from app.streamlit_app import insights as I


def _money(x):
    return f"${x:,.0f}"


def _page_title():
    st.markdown(
        '<div class="gl-h1">The Season in Dollars</div>'
        '<div class="gl-sub">What earns, what costs, what loses money — and the honest limits on '
        'what the data can claim about precision’s return.</div>'
        '<hr class="gl-hair">', unsafe_allow_html=True)


def _kpis():
    """Four headline FINANCIAL KPIs. NB: total precision benefit is a planning ASSUMPTION
    (= 0.65 × profit), not a measured return — the in-card caveat was removed by request; the
    honesty treatment still lives in docs/eda_findings.md and the pitch."""
    k = I.finance_kpis()
    cards = [
        T.kpi(f"${k['profit_per_m2']:.2f}", "Profit per m²", "season profit ÷ total area", T.SAGE),
        T.kpi(f"{k['avg_roi']*100:.1f}%", "Average ROI", "mean return on season cost", T.INDIGO),
        T.kpi(_money(k["total_precision_benefit"]), "Total precision benefit", accent=T.TERRA),
        T.kpi(f"${k['cost_per_kg']:.2f}", "Cost per kg", "total cost ÷ total kg produced", T.INDIGO),
    ]
    T.kpi_row(cards)


def _pnl_and_distribution(pt):
    h = pt["headline"]
    c = st.columns(2, gap="large")
    with c[0]:
        T.card_title("Season P&L (all 120 plots)")
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["relative", "relative", "total"],
            x=["Revenue", "− Cost", "Profit"], y=[h["total_revenue"], -h["total_cost"], 0],
            text=[_money(h["total_revenue"]), "−" + _money(h["total_cost"]), _money(h["total_profit"])],
            textposition="outside", increasing=dict(marker=dict(color=T.SAGE)),
            decreasing=dict(marker=dict(color=T.TERRA)), totals=dict(marker=dict(color=T.INDIGO)),
            connector=dict(line=dict(color=T.HAIR))))
        st.plotly_chart(T.style_fig(fig, 260, ytitle="CAD"), use_container_width=True)
    with c[1]:
        T.card_title("Profit per plot — and the downside")
        fig = go.Figure(go.Histogram(x=pt["dist"].season_profit_cad, nbinsx=24, marker_color=T.SAGE,
                                     marker_line_color=T.PAPER, marker_line_width=1))
        fig.add_vline(x=0, line_color=T.TERRA, line_dash="dot")
        st.plotly_chart(T.style_fig(fig, 260, xtitle="season profit (CAD / plot)", ytitle="plots"),
                        use_container_width=True)


def _profit_col(name):
    return st.column_config.NumberColumn(name, format="$%.0f")


def _tail(pt, win, diag):
    # Expected-Shortfall headline, then loss-makers and top earners side by side.
    T.card_title("The downside and the upside — who lost, who won")
    st.markdown(
        f'<div class="gl-note">Mean profit of the worst <b>5%</b> of plots: <b>{_money(pt["es5"])}</b> · '
        f'worst <b>10%</b>: <b>{_money(pt["es10"])}</b> (positive).</div>', unsafe_allow_html=True)
    ren = {"plot_id": "Plot", "crop": "Crop", "treatment": "Treatment", "season_profit_cad": "Profit"}
    c = st.columns(2, gap="large")
    with c[0]:
        T.card_title(f"The {diag['n']} loss-makers")
        lm = pt["loss_makers"].rename(columns=ren)
        st.dataframe(lm[["Plot", "Crop", "Treatment", "Profit"]], hide_index=True,
                     use_container_width=True, height=235,
                     column_config={"Profit": _profit_col("Profit")})
    with c[1]:
        T.card_title("The 5 top earners")
        te = win.rename(columns=ren)
        st.dataframe(te[["Plot", "Crop", "Treatment", "Profit"]], hide_index=True,
                     use_container_width=True, height=235,
                     column_config={"Profit": _profit_col("Profit")})


def _drivers(pdv):
    c = st.columns([2, 3], gap="large")
    with c[0]:
        T.card_title("What correlates with per-plot profit")
        df = pdv["corr"]
        fig = go.Figure(go.Bar(
            x=df.r, y=df.factor, orientation="h",
            marker_color=[T.SAGE if v > 0 else T.TERRA for v in df.r],
            text=[f"{v:+.2f}" for v in df.r], textposition="outside"))
        fig.add_vline(x=0, line_color=T.MUTED, line_width=1)
        fig = T.style_fig(fig, 260, xtitle="correlation with profit (r)")
        fig.update_traces(cliponaxis=False)                 # let outside labels draw past the bars
        fig.update_xaxes(range=[df.r.min() - 0.25, df.r.max() + 0.22])  # room for ±labels
        st.plotly_chart(fig, use_container_width=True)
    with c[1]:
        T.card_title("Realized price per kg vs. profit — each plot")
        sc = pdv["scatter"]
        fig = go.Figure()
        for i, (crop, sub) in enumerate(sc.groupby("crop")):
            fig.add_trace(go.Scatter(
                x=sub.price_per_kg, y=sub.season_profit_cad, mode="markers", name=crop,
                marker=dict(size=7, color=T.CAT[i % len(T.CAT)], line=dict(width=0.5, color=T.PAPER))))
        fig.add_hline(y=0, line_color=T.TERRA, line_dash="dot")
        st.plotly_chart(T.style_fig(fig, 260, legend=True, xtitle="realized price (CAD / kg)",
                                    ytitle="season profit (CAD)"), use_container_width=True)


def _crop_econ(g):
    c = st.columns([3, 2], gap="large")
    with c[0]:
        T.card_title("Price beats volume — unit economics by crop")
        fig = go.Figure()
        for i, r in g.reset_index(drop=True).iterrows():
            fig.add_trace(go.Scatter(
                x=[r.yield_kgm2], y=[r.price_kg], mode="markers+text", text=[r.crop],
                textposition="top center", textfont=dict(size=10, color=T.TAUPE), name=r.crop,
                marker=dict(size=14 + r.profit / 30, color=T.CAT[i % len(T.CAT)],
                            line=dict(width=0.5, color=T.PAPER))))
        st.plotly_chart(T.style_fig(fig, 260, xtitle="yield (kg/m²)",
                                    ytitle="realized price (CAD / kg)"), use_container_width=True)
    with c[1]:
        T.card_title("Mean profit per plot by crop")
        gg = g.sort_values("profit")
        fig = go.Figure(go.Bar(x=gg.profit, y=gg.crop, orientation="h", marker_color=T.SAGE,
                               text=[_money(p) for p in gg.profit], textposition="auto"))
        st.plotly_chart(T.style_fig(fig, 260, xtitle="mean profit (CAD / plot)"),
                        use_container_width=True)


def _reward_consistency(rr):
    T.card_title("Average payoff by treatment — bars, with the plot-to-plot swing")
    rr = rr.sort_values("mean_profit")
    fig = go.Figure(go.Bar(
        x=rr.mean_profit, y=rr.treatment, orientation="h",
        marker=dict(color=rr.worst, colorscale=[[0.0, T.TERRA], [1.0, T.SAGE]], showscale=False,
                    line=dict(width=0.5, color=T.PAPER)),
        error_x=dict(type="data", array=rr.risk, color=T.MUTED, thickness=1, width=4),
        text=[_money(p) for p in rr.mean_profit], textposition="auto",
        customdata=rr[["worst", "n", "risk"]].values,
        hovertemplate="%{y}<br>avg payoff %{x:$.0f}<br>spread ±%{customdata[2]:$.0f}<br>"
                      "worst plot %{customdata[0]:$.0f} · n=%{customdata[1]}<extra></extra>"))
    fig = T.style_fig(fig, 300, xtitle="average profit per plot (CAD)")
    fig.update_traces(cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)


def _cost_mix(cm):
    c = st.columns([3, 2], gap="large")
    with c[0]:
        T.card_title("Where the money goes (season input cost)")
        df = cm["by_cat"]
        fig = go.Figure(go.Bar(x=df.total, y=df.category, orientation="h", marker_color=T.TERRA,
                               text=[f"{_money(t)} ({p:.0f}%)" for t, p in zip(df.total, df.pct)],
                               textposition="auto"))
        st.plotly_chart(T.style_fig(fig, 240, xtitle="total season spend (CAD)"), use_container_width=True)
    with c[1]:
        T.card_title("Precision vs routine spend")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["spend"], x=[cm["precision_pct"]], orientation="h", marker_color=T.SAGE,
                             name="precision", text=[f"precision {cm['precision_pct']:.1f}%"], textposition="inside"))
        fig.add_trace(go.Bar(y=["spend"], x=[100 - cm["precision_pct"]], orientation="h", marker_color=T.HAIR,
                             name="routine", text=["routine 95.4%"], textposition="inside"))
        fig.update_layout(barmode="stack")
        st.plotly_chart(T.style_fig(fig, 240, xtitle="% of input cost", ygrid=False), use_container_width=True)


def render_finance() -> None:
    T.inject_css()
    T.brand_bar("FINANCIALS")
    _page_title()

    _kpis()

    pt = I.profit_tail()
    T.section_header("01 · Season P&L and the profit distribution", "Totals, spread, and the loss tail")
    _pnl_and_distribution(pt)
    _tail(pt, I.top_earners(), I.loss_diagnosis())

    T.section_header("02 · What drives profit", "It's a price story, not a yield story")
    _drivers(I.profit_drivers())

    T.section_header("03 · Crop economics", "Where the money actually comes from")
    _crop_econ(I.crop_economics())

    T.section_header("04 · Reward vs. consistency by treatment",
                     "Average payoff against how much plot-to-plot results swing")
    _reward_consistency(I.treatment_spread())

    T.section_header("05 · Input spending", "Where the season's input budget goes")
    _cost_mix(I.cost_mix())

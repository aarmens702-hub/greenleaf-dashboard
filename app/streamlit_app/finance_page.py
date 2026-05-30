"""Financials tab — RBC/lender view of the season in dollars, honestly.

Built ONLY from theme.py primitives and insights.py read-only helpers. Validated/defensible
content only: S3 (cost mix), S4 (profit distribution + portfolio tail), Y4 (treatment
economics with CIs, caveat), plus the mandated honesty element (precision_benefit shown as
an assumption, never ROI) and an explicit defensible-vs-won't-claim panel.
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


def _kpis(h):
    margin = h["total_profit"] / h["total_revenue"]
    T.kpi_row([
        T.kpi(_money(h["total_profit"]), "Total season profit", f"{h['n_plots']} plots"),
        T.kpi(_money(h["mean_profit"]), "Mean profit / plot", f"median {_money(h['median_profit'])}"),
        T.kpi(f"{margin*100:.0f}%", "Blended profit margin", "of revenue", T.INDIGO),
        T.kpi(f"{h['mean_roi']*100:.0f}%", "Mean ROI", "return on cost", T.INDIGO),
        T.kpi(f"{h['n_unprofitable']}", "Plots that lost money", f"of {h['n_plots']}", T.TERRA),
    ])


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
    T.note(f"{_money(h['total_revenue'])} revenue − {_money(h['total_cost'])} cost = "
           f"{_money(h['total_profit'])} profit. <b>{h['n_unprofitable']} of {h['n_plots']}</b> plots lost money; "
           f"losses are shallow (worst −$253) and revenue-driven, not cost-driven.")


def _tail(pt):
    c = st.columns([3, 2], gap="large")
    with c[0]:
        T.card_title("Concentration — Lorenz curve (revenue)")
        pop, val = pt["lorenz_rev"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=T.MUTED, width=1, dash="dot"), name="equal"))
        fig.add_trace(go.Scatter(x=pop, y=val, line=dict(color=T.SAGE, width=2.5), fill="tonexty",
                                 fillcolor="rgba(61,90,61,0.10)", name="revenue"))
        st.plotly_chart(T.style_fig(fig, 250, xtitle="cumulative share of plots", ytitle="cumulative share of revenue"),
                        use_container_width=True)
        T.note(f"Gini(revenue) = {pt['gini_revenue']:.2f} (fairly even); Gini(profit) = {pt['gini_profit']:.2f}. "
               f"Top 20% of plots hold {pt['top20_share']*100:.0f}% of profit.")
    with c[1]:
        T.card_title("Worst-case tail (Expected Shortfall)")
        st.markdown(
            f'<div class="gl-note">Mean profit of the worst <b>5%</b> of plots: '
            f'<b>{_money(pt["es5"])}</b><br>Mean profit of the worst <b>10%</b>: '
            f'<b>{_money(pt["es10"])}</b> (positive)</div>', unsafe_allow_html=True)
        lm = pt["loss_makers"].rename(columns={"plot_id": "Plot", "crop": "Crop", "treatment": "Treatment",
                                               "season_profit_cad": "Profit"})
        st.dataframe(lm[["Plot", "Crop", "Treatment", "Profit"]], hide_index=True, use_container_width=True,
                     height=235, column_config={"Profit": st.column_config.NumberColumn("Profit", format="$%.0f")})
    T.note("The 6 loss-makers and the worst-5% shortfall (−$123/plot) are the concrete downside a lender "
           "underwrites; 5 of 6 sit in high-pest treatments (Shade / Reduced Pest).")


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
    T.note(f"<b>Labor is ~{cm['labor_pct']:.0f}%</b> of input cost. Precision (alert-response) actions are only "
           f"<b>{cm['precision_pct']:.1f}%</b> ({_money(cm['precision'])}) — a cheap overlay on mostly-routine spend. "
           f"Source: daily cost ledger (<code>finance.cost_structure</code>) — the canonical cost source.")


def _treatment_economics(te):
    rows = te["rows"]
    fig = go.Figure()
    fig.add_vline(x=0, line_color=T.MUTED, line_width=1)
    for _, r in rows.iterrows():
        if r["diff"] is None:
            fig.add_trace(go.Scatter(x=[0], y=[r["treatment"]], mode="markers",
                                     marker=dict(color=T.MUTED, size=10, symbol="line-ns-open"),
                                     text=["Control (baseline)"]))
        else:
            sig = (r["lo"] > 0) or (r["hi"] < 0)
            fig.add_trace(go.Scatter(
                x=[r["diff"]], y=[r["treatment"]], mode="markers",
                marker=dict(color=T.SAGE if sig else T.MUTED, size=11),
                error_x=dict(type="data", symmetric=False, array=[r["hi"] - r["diff"]],
                             arrayminus=[r["diff"] - r["lo"]], color=T.TAUPE, thickness=1.5, width=6)))
    st.plotly_chart(T.style_fig(fig, 280, xtitle="profit vs Control (CAD/plot) — 95% bootstrap CI"),
                    use_container_width=True)
    gap = te["hl_shade_gap"]
    T.note(f"High Light − Shade = +{gap[0]:.0f} CAD/plot [95% CI {gap[1]:.0f}, {gap[2]:.0f}]. Only High Light "
           f"(+260 [48, 483]), Reduced Pest (−182 [−358, −5]) and Shade (−200 [−385, −23]) differ from Control "
           f"with CIs excluding zero.")
    T.caveat_note("Observed per-arm economics, <b>not a causal treatment or precision-quality effect</b>. "
                  "Treatments weren’t randomized: the profit ranking tracks pest/stress burden "
                  "(profit vs alert-days r=−0.83) and crop mix differs across arms. Keep CIs visible; currency CAD.")


def _precision_honesty(pb, h):
    c = st.columns(2, gap="large")
    with c[0]:
        T.card_title("“Precision benefit” vs profit — read this")
        s = pb["scatter"]
        xmax = float(s.season_profit_cad.max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[min(0, float(s.season_profit_cad.min())), xmax], y=[0, 0.65 * xmax],
                                 mode="lines", line=dict(color=T.MUTED, width=1, dash="dot"), name="0.65 × profit"))
        fig.add_trace(go.Scatter(x=s.season_profit_cad, y=s.precision_benefit_cad, mode="markers",
                                 marker=dict(color=T.SAGE, size=6, line=dict(width=0.5, color=T.PAPER)), name="plots"))
        st.plotly_chart(T.style_fig(fig, 270, xtitle="season profit (CAD)", ytitle="precision_benefit_cad (CAD)"),
                        use_container_width=True)
        T.note(f"Every plot sits on the line: precision_benefit_cad = <b>0.65 × profit</b> "
               f"(corr {pb['corr_benefit_profit']:.3f}), floored at 0. It is profit relabelled — an assumption.")
    with c[1]:
        T.card_title("What we can — and cannot — claim")
        st.markdown(
            f'<div class="gl-note" style="font-size:0.88rem;">'
            f'<b style="color:{T.SAGE};">✓ Defensible</b><br>'
            f'• Precision is a cheap overlay: <b>4.6%</b> of input cost.<br>'
            f'• Fast response measurably cuts <b>stress</b> (same-day ≈2×, see Operations).<br>'
            f'• The downside is bounded & named: 6 loss plots, worst-5% −$123.<br><br>'
            f'<b style="color:{T.TERRA};">✗ Won’t claim</b><br>'
            f'• A precision <b>ROI</b> — precision_benefit_cad is 0.65×profit (circular).<br>'
            f'• That precision/stress raises <b>profit</b> — stress→profit vanishes within treatment (r +0.09, p=0.31).<br>'
            f'• A cross-ledger return — season vs daily cost don’t reconcile (corr {pb["corr_ledger"]:.2f}).'
            f'</div>', unsafe_allow_html=True)
    T.caveat_note("The dataset’s <code>precision_benefit_cad</code> totals "
                  f"{_money(h['total_precision_benefit'])} ({h['benefit_share']*100:.0f}% of profit) but is an "
                  "<b>assumption</b>, not a measured return. Sell precision as risk management, not a profit multiple.")


def render_finance() -> None:
    T.inject_css()
    T.brand_bar("FINANCIALS")
    _page_title()

    h = I.headline()
    _kpis(h)

    pt = I.profit_tail()
    T.section_header("01 · Season P&L and the profit distribution", "Totals, spread, and the loss tail")
    _pnl_and_distribution(pt)
    _tail(pt)

    T.section_header("02 · Where the money goes", "Cost mix — labor dominates, precision is cheap")
    _cost_mix(I.cost_mix())

    T.section_header("03 · Treatment economics", "Per-arm profit vs Control, with bootstrap CIs")
    _treatment_economics(I.treatment_economics())

    T.section_header("04 · The “precision benefit” figure", "Shown honestly — an assumption, not a return")
    _precision_honesty(I.precision_benefit(), h)

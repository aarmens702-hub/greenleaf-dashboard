"""Financials tab: aggregations reproduce the EDA, page renders offline, honesty visible."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import finance as fin


def test_headline_reproduces_eda():
    h = fin.headline()
    assert h["n_plots"] == 120
    assert h["n_unprofitable"] == 6
    assert round(h["total_profit"]) == 71654
    assert 0.64 < h["benefit_share"] < 0.67     # precision_benefit ~ 0.65 x profit (circular)


def test_precision_benefit_is_circular_and_ledger_mismatch():
    bc = fin.precision_benefit_check()
    assert bc["corr_benefit_profit"] > 0.99
    assert bc["corr_ledger"] < 0                 # season vs daily ledgers don't reconcile


def test_finance_page_imports_cleanly():
    from app.streamlit_app.finance_page import render_finance

    assert callable(render_finance)


def test_render_finance_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    script = (
        f"import sys; sys.path.insert(0, r'{root}')\n"
        "from app.streamlit_app.finance_page import render_finance\n"
        "render_finance()\n"
    )
    at = AppTest.from_string(script).run(timeout=120)
    assert not at.exception, at.exception
    md = " ".join(b.value for b in at.markdown)
    # narrative sections
    assert "Season P&L" in md
    assert "What drives profit" in md
    assert "Crop economics" in md
    assert "Reward vs. consistency" in md
    assert "Average payoff by treatment" in md     # §04 is now a bar chart (was a scatter)
    assert "Where the money goes" in md           # cost-mix card (now under '05 Input spending')
    # the tab's own four FINANCIAL KPIs (the blended 5-KPI row stays on Overview)
    for card in ["Profit per m", "Average ROI", "Total precision benefit", "Cost per kg"]:
        assert card in md, card
    assert "Avg plot performance" not in md        # the Overview blended-KPI labels stay on Overview
    assert "profit-free score" not in md
    # previously deleted sections stay gone
    assert "Treatment economics" not in md
    assert "Lorenz" not in md
    # loss + win tail (card titles; the price-not-spending note caption was removed by request)
    assert "loss-makers" in md.lower()
    assert "top earners" in md.lower()
    # The daily-spend time series and the composite "plots at risk" table were MOVED to the
    # Overview tab — they must no longer render on Financials (moved, not duplicated).
    assert "Daily total input spend" not in md
    assert "Plots at risk today" not in md
    assert "composite health-score risk" not in md
    assert len(at.dataframe) == 2                 # only loss-makers + top-earners remain

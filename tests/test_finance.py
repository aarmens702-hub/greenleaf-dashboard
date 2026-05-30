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
    assert "Where the money goes" in md
    assert "Treatment economics" in md
    assert "precision benefit" in md.lower()
    assert "Total season profit" in md           # KPI card
    # honesty must be visible, not buried:
    assert "assumption" in md.lower()
    assert "Won’t claim" in md or "Defensible" in md
    assert len(at.dataframe) >= 1                 # loss-maker table

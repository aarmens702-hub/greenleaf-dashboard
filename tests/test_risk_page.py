"""Operations tab: imports cleanly and renders end-to-end (offline) without exceptions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_risk_page_imports_cleanly():
    from app.streamlit_app.risk_page import render_risk

    assert callable(render_risk)


def test_render_risk_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    script = (
        f"import sys; sys.path.insert(0, r'{root}')\n"
        "from app.streamlit_app.risk_page import render_risk\n"
        "render_risk()\n"
    )
    at = AppTest.from_string(script).run(timeout=120)
    assert not at.exception, at.exception
    md = " ".join(b.value for b in at.markdown)
    assert "Where to act first" in md
    assert "Does acting work" in md
    # consolidated single KPI row: exactly these six cards present...
    for card in ["High-risk plots today", "Same-day response advantage", "Healthy plot-day rate",
                 "Avg plot performance", "Alert rate", "Alert response rate"]:
        assert card in md, card
    # ...and the removed cards are gone (deleted, not hidden)
    for gone in ["Plots monitored", "3-day spike base rate", "Watch-list hit rate"]:
        assert gone not in md, gone
    # Section 06 is now the recommendation-uptake view; the by-greenhouse
    # "Where the work lands" view and old alert-burden bars are both gone.
    assert "Alert-burden exposure" not in md
    assert "Where the work lands" not in md
    assert "Are the recommendations followed" in md
    # Model transparency collapsed into a default-collapsed expander, content intact
    assert any("Model details" in (e.label or "") for e in at.expander)
    assert "Honest ceiling" in md            # the caveat still rendered inside the expander
    assert len(at.dataframe) >= 1            # watch-list (+ metrics) table
    assert len(at.selectbox) == 1            # plot drill-down

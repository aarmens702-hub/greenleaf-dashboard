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
    assert "Model transparency" in md
    assert "Plots monitored" in md           # KPI card (HTML)
    assert len(at.dataframe) >= 1            # watch-list (+ metrics) table
    assert len(at.selectbox) == 1            # plot drill-down

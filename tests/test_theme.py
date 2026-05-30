"""Smoke tests for the shared design system (app/streamlit_app/theme.py)."""
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.streamlit_app import theme as T


def test_primitives_exist():
    for f in ["inject_css", "brand_bar", "section_header", "kpi", "kpi_row",
              "card_title", "caveat_note", "note", "hairline", "style_fig"]:
        assert callable(getattr(T, f)), f


def test_kpi_dict_shape():
    k = T.kpi("120", "Plots", "sub", accent=T.TERRA)
    assert k == {"value": "120", "label": "Plots", "sub": "sub", "accent": T.TERRA}


def test_style_fig_applies_theme():
    fig = go.Figure(go.Bar(x=[1, 2], y=[1, 2]))
    T.style_fig(fig, height=250, xtitle="x", ytitle="y")
    assert fig.layout.paper_bgcolor == T.PAPER
    assert fig.layout.plot_bgcolor == T.PAPER
    assert tuple(fig.layout.colorway) == tuple(T.CAT)          # never Plotly blue
    assert fig.layout.height == 250
    fam = fig.layout.font.family
    assert "Helvetica" in fam or "Arial" in fam

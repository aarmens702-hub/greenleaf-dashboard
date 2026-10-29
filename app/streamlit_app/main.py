"""GreenLeaf dashboard — entry point and shell.

Run:  .venv/bin/python -m streamlit run app/streamlit_app/main.py

One cohesive product, four tabs, each a render_*() drawn into its tab (tabs share no
state — they read cached functions only). All visual styling comes from theme.py; every
number comes from the frozen analysis layer via insights.py. No width hacks: theme.py
owns the page width. The Ask tab splits into a conversation column + a suggested-prompt rail.
    Overview      -> render_overview()  (curated summary: spend, at-risk, health — filterable)
    Operations    -> render_risk()     (stress-spike early-warning, technical manager)
    Financials    -> render_finance()  (the season in dollars, RBC/lender)
    Ask GreenLeaf -> render_chat()     (Stage-4 chat analyst — logic unchanged)
"""
from pathlib import Path
import sys

import streamlit as st

# `streamlit run` puts only this file's folder on sys.path; the app imports from the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_derived  # noqa: E402
from app.streamlit_app.chat_page import render_chat  # noqa: E402
from app.streamlit_app.finance_page import render_finance  # noqa: E402
from app.streamlit_app.overview_page import render_overview  # noqa: E402
from app.streamlit_app.risk_page import render_risk  # noqa: E402

# The six parquets build_derived.main() writes; data/derived/ is gitignored, so a fresh clone has none.
DERIVED = ["plot_meta", "season", "daily", "actions", "scouting", "bc_benchmarks"]

st.set_page_config(page_title="GreenLeaf", layout="wide")

if not all((build_derived.DERIVED_DIR / f"{n}.parquet").exists() for n in DERIVED):
    with st.spinner("Building derived data (first start)..."):
        build_derived.main()

tabs = st.tabs(["Overview", "Operations", "Financials", "Ask GreenLeaf"])
with tabs[0]:
    render_overview()
with tabs[1]:
    render_risk()
with tabs[2]:
    render_finance()
with tabs[3]:
    render_chat()

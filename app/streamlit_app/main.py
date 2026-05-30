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
import streamlit as st

from app.streamlit_app.chat_page import render_chat
from app.streamlit_app.finance_page import render_finance
from app.streamlit_app.overview_page import render_overview
from app.streamlit_app.risk_page import render_risk

st.set_page_config(page_title="GreenLeaf", layout="wide")

tabs = st.tabs(["Overview", "Operations", "Financials", "Ask GreenLeaf"])
with tabs[0]:
    render_overview()
with tabs[1]:
    render_risk()
with tabs[2]:
    render_finance()
with tabs[3]:
    render_chat()

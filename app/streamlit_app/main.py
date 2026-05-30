"""GreenLeaf dashboard — entry point and shell.

Run:  .venv/bin/python -m streamlit run app/streamlit_app/main.py

One cohesive product, three tabs, each a render_*() drawn into its tab (tabs share no
state — they read cached functions only). All visual styling comes from theme.py; every
number comes from the frozen analysis layer via insights.py. No width hacks: theme.py
owns the page width. The Ask tab is centred to its editorial measure.
    Operations    -> render_risk()     (stress-spike early-warning, technical manager)
    Financials    -> render_finance()  (the season in dollars, RBC/lender)
    Ask GreenLeaf -> render_chat()     (Stage-4 chat analyst — logic unchanged)
"""
import streamlit as st

from app.streamlit_app.chat_page import render_chat
from app.streamlit_app.finance_page import render_finance
from app.streamlit_app.risk_page import render_risk

st.set_page_config(page_title="GreenLeaf", layout="wide")

tabs = st.tabs(["Operations", "Financials", "Ask GreenLeaf"])
with tabs[0]:
    render_risk()
with tabs[1]:
    render_finance()
with tabs[2]:
    _, mid, _ = st.columns([1, 5, 1])
    with mid:
        render_chat()

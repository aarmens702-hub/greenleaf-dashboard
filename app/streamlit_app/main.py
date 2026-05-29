"""Temporary single-page entry point for the Stage 4 chat.

Run with:
    .venv/bin/python -m streamlit run app/streamlit_app/main.py

This file is a stopgap so the chat can be demoed standalone. When the 4-tab
dashboard (Story, Predictor, Ask, Under the Hood) is built, render_chat() will
be called from inside the "Ask" tab and this main.py will be replaced.
"""
import streamlit as st

from app.streamlit_app.chat_page import render_chat

st.set_page_config(page_title="GreenLeaf — Ask", layout="wide")
render_chat()

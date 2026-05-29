"""Stage 4 smoke test: the chat page must import cleanly.

Streamlit rendering is verified manually in the browser; here we only guard
against import errors / dependency typos in chat_page.py.
"""


def test_chat_page_imports_cleanly():
    from app.streamlit_app.chat_page import render_chat

    assert callable(render_chat)

"""Public entry point for the GreenLeaf chat.

As of Stage 3 v2 the single-shot router was replaced by an agentic tool-calling
loop (see app/chat/agent.py). This module re-exports answer() so existing imports
keep working unchanged:

    from app.chat.answer import answer   # used by the Streamlit UI

answer(question, max_iterations=6) -> {answer, sources, trace}.
"""
from app.chat.agent import answer  # noqa: F401  (re-export)

__all__ = ["answer"]

"""Stage 3 orchestrator: the public answer() function.

Routes a question (router.classify), gathers document chunks (rag.retrieve) and/or
computed data (data_queries), builds a grounded context block, and asks Claude Opus
for a cited answer. answer() is the single entry point the Stage 4 UI will call.
"""
import json

import pandas as pd

from app.chat import router
from app.chat.router import classify
from app.rag.retrieve import retrieve
from app.chat.data_queries import (
    season_overview,
    precision_benefit_summary,
    response_time_payoff,
    avg_profit_by_crop,
    treatment_summary,
    plot_status,
    top_at_risk_plots,
    greenleaf_vs_benchmark,
    cost_breakdown,
    alert_response_summary,
)

# DeepSeek V3 (cheap, strong at grounded RAG). Swap this one constant to change model.
ANSWER_MODEL = "deepseek-chat"

DISPATCH = {
    "season_overview": season_overview,
    "precision_benefit_summary": precision_benefit_summary,
    "response_time_payoff": response_time_payoff,
    "avg_profit_by_crop": avg_profit_by_crop,
    "treatment_summary": treatment_summary,
    "plot_status": plot_status,
    "top_at_risk_plots": top_at_risk_plots,
    "greenleaf_vs_benchmark": greenleaf_vs_benchmark,
    "cost_breakdown": cost_breakdown,
    "alert_response_summary": alert_response_summary,
}

_TRUNC = 2000

ANSWER_SYSTEM = """You are GreenLeaf's analytics assistant for a BC controlled-environment agriculture operator. Answer the user's question using ONLY the context provided below (documents and/or computed data). Rules:
- If the context does not contain enough information, say so plainly and suggest what specific question they could ask instead.
- Cite every factual claim. For doc facts, include the source_url in parentheses. For data facts, name the function in brackets, e.g., "[precision_benefit_summary]".
- Never invent numbers or sources. If a number isn't in the context, you don't have it.
- Keep answers under 200 words unless the user explicitly asks for more.
- Use plain language; this may be read by farmers and lenders, not only technical users."""


def _format_result(result):
    if isinstance(result, pd.DataFrame):
        s = result.head(20).to_string(index=False)
    elif isinstance(result, (dict, list)):
        s = json.dumps(result, indent=2, default=str)
    else:
        s = str(result)
    return s[:_TRUNC]


def _generate_answer(system, user):
    """Call the LLM for the final written answer. Patched in tests."""
    return router._call_llm(ANSWER_MODEL, system, user, max_tokens=700)


def answer(question):
    """Answer a question using retrieved docs and/or computed data, with citations.

    Args: question (str).
    Returns: dict {answer: str, sources: list[dict], trace: dict}.
    Example: answer("What is GreenLeaf's overall precision benefit?")["answer"]
    """
    routing = classify(question)

    chunks = retrieve(question, k=5) if routing.get("needs_docs") else []

    data_calls = []
    data_results = []  # (name, result) for successful calls
    used_calls = []    # (name, args) for successful calls
    if routing.get("needs_data"):
        data_args = routing.get("data_args", {}) or {}
        for name in routing.get("data_functions", []) or []:
            args = data_args.get(name, {}) or {}
            fn = DISPATCH.get(name)
            if fn is None:
                data_calls.append({"name": name, "status": "error", "error_msg": f"unknown function '{name}'"})
                continue
            try:
                result = fn(**args)
            except ValueError as e:
                data_calls.append({"name": name, "status": "error", "error_msg": str(e)})
                continue
            data_calls.append({"name": name, "status": "ok"})
            data_results.append((name, result))
            used_calls.append((name, args))

    sections = []
    if chunks:
        doc_lines = [
            f"[Doc {i}] {c.get('title')} ({c.get('source_url')})\n{(c.get('text') or '')[:_TRUNC]}"
            for i, c in enumerate(chunks, 1)
        ]
        sections.append("DOCUMENTS:\n" + "\n\n".join(doc_lines))
    if data_results:
        data_lines = [f"=== [{name}] ===\n{_format_result(res)}" for name, res in data_results]
        sections.append("COMPUTED DATA:\n" + "\n\n".join(data_lines))
    context = "\n\n".join(sections) if sections else "(no context available)"

    answer_text = _generate_answer(ANSWER_SYSTEM, f"Question: {question}\n\nCONTEXT:\n{context}")

    sources = []
    seen_urls = set()
    for c in chunks:
        url = c.get("source_url") or ""
        if url and url in answer_text and url not in seen_urls:
            seen_urls.add(url)
            sources.append({"type": "doc", "title": c.get("title"), "source_url": url})
    for name, args in used_calls:
        sources.append({"type": "data", "function": name, "args": args})

    return {
        "answer": answer_text,
        "sources": sources,
        "trace": {
            "routing": routing,
            "n_doc_chunks": len(chunks),
            "data_calls": data_calls,
        },
    }

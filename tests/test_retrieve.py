"""Retrieval tests — a metric BUDGET over the golden set, not brittle filename asserts.

The old version hard-coded which source_file must appear for two specific queries, so
any corpus edit (or the Phase-2 hybrid/rerank/embedder changes that legitimately
reorder results) broke it. Instead we assert a recall/hit budget over the whole golden
set plus the basic retrieve() contract — robust to reordering, sensitive to real drops.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.retrieve import retrieve
from eval import run_eval


def test_retrieve_contract():
    """retrieve() returns k records with the expected keys, scored descending."""
    results = retrieve("greenhouse energy costs", k=5)
    assert len(results) == 5
    for r in results:
        assert {"source_file", "text", "score"} <= set(r)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recall_budget_over_golden():
    """Recall/hit over the labeled golden questions must clear a floor."""
    rows = run_eval.load_golden()
    summary, _ = run_eval.evaluate(rows, k=5, retrieve_fn=retrieve)
    assert summary["n_labeled"] >= 20
    assert summary["hit@k"] >= 0.9, summary
    assert summary["recall@k"] >= 0.8, summary

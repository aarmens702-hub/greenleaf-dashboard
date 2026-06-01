"""IR metrics + eval aggregation (Phase 0). No model: retrieve is faked."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics, run_eval


def test_metrics_basic():
    ranked = ["a", "b", "c"]
    assert metrics.hit_at_k(ranked, ["b"], 3) == 1.0
    assert metrics.hit_at_k(ranked, ["b"], 1) == 0.0
    assert metrics.recall_at_k(ranked, ["a", "b"], 3) == 1.0
    assert metrics.recall_at_k(ranked, ["a", "x"], 3) == 0.5
    assert metrics.reciprocal_rank(ranked, ["b"]) == 0.5
    assert metrics.ndcg_at_k(["b", "a"], ["b"], 2) == 1.0
    assert metrics.dedup_preserve_order(["a", "a", "b", "a"]) == ["a", "b"]


def test_golden_file_loads_and_is_well_formed():
    rows = run_eval.load_golden()
    assert len(rows) >= 30
    for r in rows:
        assert r["id"] and r["question"] and r["type"] in {"doc", "data", "hybrid"}
        if r["type"] in {"doc", "hybrid"}:
            assert r.get("relevant_sources")


def test_evaluate_aggregates_over_labeled_rows():
    rows = [
        {"id": "1", "type": "doc", "question": "q1", "relevant_sources": ["d1.md"]},
        {"id": "2", "type": "data", "question": "q2", "expected_tools": ["x"]},
    ]

    def fake_retrieve(_q, _k):
        return [{"source_file": "d1.md"}, {"source_file": "d2.md"}]

    summary, per_q = run_eval.evaluate(rows, k=5, retrieve_fn=fake_retrieve)
    assert summary["n_labeled"] == 1  # the data row has no labels
    assert summary["n_total"] == 2
    assert summary["hit@k"] == 1.0
    assert per_q[0]["mrr"] == 1.0

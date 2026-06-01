"""Offline retrieval eval for the GreenLeaf RAG — free IR metrics, no LLM/API cost.

For every golden question that has labeled relevant sources (type 'doc' or 'hybrid'),
run retrieve() and score recall@k / hit@k / MRR / nDCG@k at the source-file level.
Aggregate means and write eval/baseline.json (the regression budget CI checks in
Phase 1). Data-tool questions are reported as coverage only (no retrieval label).

Run:   .venv/bin/python -m eval.run_eval [--k 5] [--out eval/baseline.json]
Generation metrics (faithfulness/answer-relevance) are a later phase; this stays
LLM-free so it can gate every PR cheaply.
"""
import argparse
import json
from pathlib import Path

from eval import metrics

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent / "baseline.json"


def load_golden(path=GOLDEN):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(rows, k, retrieve_fn):
    """Score the labeled rows. retrieve_fn(question, k) -> list of records with
    'source_file'. Returns (summary_dict, per_question_list)."""
    labeled = [r for r in rows if r.get("relevant_sources")]
    per_q = []
    for r in labeled:
        recs = retrieve_fn(r["question"], k)
        ranked = metrics.dedup_preserve_order([rec.get("source_file") for rec in recs])
        rel = r["relevant_sources"]
        per_q.append({
            "id": r["id"],
            "type": r.get("type"),
            "hit@k": metrics.hit_at_k(ranked, rel, k),
            "recall@k": metrics.recall_at_k(ranked, rel, k),
            "mrr": metrics.reciprocal_rank(ranked, rel),
            "ndcg@k": metrics.ndcg_at_k(ranked, rel, k),
            "top": ranked[:k],
        })

    def _mean(key):
        return round(sum(q[key] for q in per_q) / len(per_q), 4) if per_q else 0.0

    summary = {
        "k": k,
        "n_labeled": len(per_q),
        "n_total": len(rows),
        "hit@k": _mean("hit@k"),
        "recall@k": _mean("recall@k"),
        "mrr": _mean("mrr"),
        "ndcg@k": _mean("ndcg@k"),
    }
    return summary, per_q


def _index_meta():
    """Stamp the result with which corpus/model produced it (best-effort)."""
    man = ROOT / "app" / "rag" / "index_data" / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        return {"embedding_model": m.get("embedding_model"),
                "corpus_version": m.get("corpus_version"),
                "chunk_count": m.get("chunk_count")}
    return {}


def main():
    ap = argparse.ArgumentParser(description="GreenLeaf retrieval eval (free IR metrics).")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    # Imported here so the pure metrics stay testable without loading the model.
    from app.rag.retrieve import retrieve

    rows = load_golden()
    summary, per_q = evaluate(rows, args.k, retrieve)
    report = {**summary, "index": _index_meta(), "per_question": per_q}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[eval] {summary['n_labeled']} labeled / {summary['n_total']} total @ k={args.k}")
    print(f"[eval] hit@k={summary['hit@k']}  recall@k={summary['recall@k']}  "
          f"mrr={summary['mrr']}  ndcg@k={summary['ndcg@k']}")
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()

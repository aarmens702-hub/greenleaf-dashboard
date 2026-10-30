"""Pure information-retrieval metrics for the GreenLeaf RAG eval (no model, no deps).

All operate on a ranked list of source_file labels (best first, deduplicated) and a
set of relevant source_files. Source-file granularity is deliberate: it is stable
across re-chunking / re-embedding (a chunk_id is not), so the metric budget survives
the Phase-2 retrieval changes that renumber chunks.
"""
import math


def hit_at_k(ranked, relevant, k):
    """1.0 if any relevant source appears in the top-k, else 0.0."""
    rel = set(relevant)
    return 1.0 if any(s in rel for s in ranked[:k]) else 0.0


def recall_at_k(ranked, relevant, k):
    """Fraction of the relevant sources that appear in the top-k."""
    rel = set(relevant)
    if not rel:
        return 0.0
    found = len({s for s in ranked[:k] if s in rel})
    return found / len(rel)


def reciprocal_rank(ranked, relevant):
    """1 / rank of the first relevant source (0.0 if none retrieved)."""
    rel = set(relevant)
    for i, s in enumerate(ranked, start=1):
        if s in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """Binary-relevance nDCG@k over source labels."""
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, s in enumerate(ranked[:k], start=1) if s in rel)
    ideal_n = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def dedup_preserve_order(items):
    """Collapse a chunk-level ranked list to unique source labels, best rank first."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

"""Retrieval for the GreenLeaf chat.

Loads the prebuilt index (embeddings + chunk metadata + manifest) and the embedding
model ONCE at import, validates the index fails-loud on a model/dim/count mismatch
(so a stale or half-built index never silently serves wrong results), then exposes
retrieve(question, k). Run build_index.py first.

CLI: python app/rag/retrieve.py <your question>
"""
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core import config
from app.rag import index_io

INDEX_DIR = Path(__file__).resolve().parent / "index_data"
MODEL_NAME = config.EMBEDDING_MODEL

# Loaded ONCE per process at import (not per call). Validation here means a mismatched
# or corrupt index raises at startup instead of returning garbage at query time.
_EMBEDDINGS, _CHUNKS, _MANIFEST = index_io.load_index(INDEX_DIR)
index_io.validate_index(_EMBEDDINGS, _CHUNKS, _MANIFEST, config.EMBEDDING_DIM, config.EMBEDDING_MODEL)
_MODEL = SentenceTransformer(MODEL_NAME)


def _search_topk(query_vec, k):
    """Return the indices of the top-k most similar chunks.

    SWAP POINT: this is the ONLY place the search backend lives. Today it's a
    plain numpy dot-product scan. To move to a vector DB later (FAISS, chroma,
    pgvector, ...), replace just the body of this function to query that store
    and return chunk indices — retrieve() and everything else stay unchanged.

    The dot product equals cosine similarity here because both _EMBEDDINGS and
    query_vec are L2-normalized (normalize_embeddings=True at build + query time).
    """
    scores = _EMBEDDINGS @ query_vec
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return top[np.argsort(-scores[top])].tolist()


def retrieve(question, k=config.RETRIEVE_K):
    q_vec = _MODEL.encode([question], normalize_embeddings=True)[0].astype(np.float32)
    results = []
    for i in _search_topk(q_vec, k):
        rec = dict(_CHUNKS[i])
        rec["score"] = float(_EMBEDDINGS[i] @ q_vec)
        results.append(rec)
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "is precision irrigation good for strawberries"
    for r in retrieve(q):
        print(f"{r['score']:.3f}  {r['source_file']}: {r['text'][:80]}...")

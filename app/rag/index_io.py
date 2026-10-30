"""On-disk index I/O + validation (numpy + json only — no embedding model).

Split out of retrieve.py so the fail-loud validation is unit-testable without loading
the heavy sentence-transformers model, and so the (future) vector-DB backend has a
clean place to live. retrieve.py and build_index.py share these helpers.
"""
from pathlib import Path
import json

import numpy as np

MANIFEST_NAME = "manifest.json"


def load_index(index_dir):
    """Load (embeddings, chunks, manifest) from index_dir.

    embeddings: np.ndarray | None, chunks: list[dict], manifest: dict | None.
    Raises FileNotFoundError if the core files are missing (manifest is optional so a
    legacy pre-manifest index still loads).
    """
    index_dir = Path(index_dir)
    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.json"
    if not emb_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Index not found in {index_dir}. Run: python app/rag/build_index.py"
        )
    embeddings = np.load(emb_path)
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    manifest = None
    man_path = index_dir / MANIFEST_NAME
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    return embeddings, chunks, manifest


def validate_index(embeddings, chunks, manifest, expected_dim, expected_model):
    """Fail loud on a corrupt or mismatched index rather than serving wrong results.

    Checks (raises RuntimeError with an actionable message on any failure):
      - embeddings is a 2-D array whose row count equals len(chunks)
      - embedding dimension equals expected_dim (catches a model swap w/o reindex)
      - if a manifest is present: its embedding_dim/chunk_count match the files, and
        its embedding_model matches expected_model
    A missing manifest is tolerated (legacy index): the dim + count checks still apply.
    """
    if embeddings is None or getattr(embeddings, "ndim", None) != 2:
        raise RuntimeError("Index embeddings are missing or not a 2-D array; rebuild the index.")
    n_rows, dim = embeddings.shape
    if n_rows != len(chunks):
        raise RuntimeError(
            f"Index corrupt: {n_rows} embeddings vs {len(chunks)} chunk records "
            f"(a partial/mid-crash write?). Re-run: python app/rag/build_index.py"
        )
    if dim != expected_dim:
        raise RuntimeError(
            f"Index embedding dim {dim} != configured EMBEDDING_DIM {expected_dim}. "
            f"The embedding model likely changed without a reindex. "
            f"Re-run: python app/rag/build_index.py"
        )
    if manifest is not None:
        if manifest.get("embedding_dim") != dim:
            raise RuntimeError(
                f"Manifest embedding_dim {manifest.get('embedding_dim')} != actual {dim}; "
                f"rebuild the index."
            )
        if manifest.get("chunk_count") != len(chunks):
            raise RuntimeError(
                f"Manifest chunk_count {manifest.get('chunk_count')} != actual {len(chunks)}; "
                f"rebuild the index."
            )
        if manifest.get("embedding_model") != expected_model:
            raise RuntimeError(
                f"Index was built with model '{manifest.get('embedding_model')}' but config "
                f"expects '{expected_model}'. Re-run build_index.py with the intended model."
            )

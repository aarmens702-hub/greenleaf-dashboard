"""Fail-loud index validation (Phase 0). Pure — no embedding model loaded."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag import index_io


def _emb(n, d):
    return np.zeros((n, d), dtype=np.float32)


def test_valid_index_passes():
    man = {"embedding_dim": 4, "chunk_count": 2, "embedding_model": "m"}
    index_io.validate_index(_emb(2, 4), [{}, {}], man, expected_dim=4, expected_model="m")


def test_missing_manifest_tolerated():
    # Legacy index (no manifest): dim + count checks still apply, no model check.
    index_io.validate_index(_emb(2, 4), [{}, {}], None, expected_dim=4, expected_model="m")


def test_dim_mismatch_raises():
    with pytest.raises(RuntimeError, match="dim"):
        index_io.validate_index(_emb(2, 4), [{}, {}], None, expected_dim=8, expected_model="m")


def test_count_mismatch_raises():
    with pytest.raises(RuntimeError, match="corrupt"):
        index_io.validate_index(_emb(3, 4), [{}, {}], None, expected_dim=4, expected_model="m")


def test_manifest_model_mismatch_raises():
    man = {"embedding_dim": 4, "chunk_count": 2, "embedding_model": "old-model"}
    with pytest.raises(RuntimeError, match="model"):
        index_io.validate_index(_emb(2, 4), [{}, {}], man, expected_dim=4, expected_model="new-model")

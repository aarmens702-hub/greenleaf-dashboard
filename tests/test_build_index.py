"""Index build: metadata persistence, content hashing, delta reuse map (Phase 0).

No embedding model is loaded — these exercise the pure build helpers and the reuse-map
decision (which is what makes an unchanged-corpus rebuild a no-op).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag import build_index as bi
from app.core import config


def _write_doc(p, body="para one is here.\n\npara two is here.", **front):
    fm = "".join(f"{k}: {v}\n" for k, v in front.items())
    p.write_text(f"---\n{fm}---\n\n{body}\n", encoding="utf-8")


def test_build_records_persists_previously_dropped_metadata(tmp_path):
    p = tmp_path / "x.md"
    _write_doc(p, title="T", source_name="N", source_url="u",
               category="energy_cost", region="BC", fetch_date="2026-05-27")
    records, docs_meta = bi.build_records([p])
    r = records[0]
    assert r["category"] == "energy_cost"
    assert r["region"] == "BC"
    assert r["fetch_date"] == "2026-05-27"
    assert [rec["chunk_id"] for rec in records] == list(range(len(records)))
    assert docs_meta["x.md"]["n_chunks"] == len(records)
    assert len(docs_meta["x.md"]["content_hash"]) == 64


def test_content_hash_is_stable_and_change_sensitive():
    assert bi._content_hash("a\nb") == bi._content_hash("a\nb  ")  # trailing ws per line ignored
    assert bi._content_hash("alpha") != bi._content_hash("beta")


def test_reuse_map_returns_text_to_vec_when_compatible(tmp_path, monkeypatch):
    dim = config.EMBEDDING_DIM
    monkeypatch.setattr(bi, "INDEX_DIR", tmp_path)
    np.save(tmp_path / "embeddings.npy", np.ones((2, dim), dtype=np.float32))
    (tmp_path / "chunks.json").write_text(json.dumps([{"text": "a"}, {"text": "b"}]))
    (tmp_path / "manifest.json").write_text(json.dumps({"embedding_model": config.EMBEDDING_MODEL}))
    m = bi._load_reuse_map()
    assert set(m) == {"a", "b"}
    assert m["a"].shape == (dim,)


def test_reuse_map_skips_on_dim_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "INDEX_DIR", tmp_path)
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 8), dtype=np.float32))  # wrong dim
    (tmp_path / "chunks.json").write_text(json.dumps([{"text": "a"}, {"text": "b"}]))
    assert bi._load_reuse_map() == {}


def test_reuse_map_skips_on_model_mismatch(tmp_path, monkeypatch):
    dim = config.EMBEDDING_DIM
    monkeypatch.setattr(bi, "INDEX_DIR", tmp_path)
    np.save(tmp_path / "embeddings.npy", np.zeros((1, dim), dtype=np.float32))
    (tmp_path / "chunks.json").write_text(json.dumps([{"text": "a"}]))
    (tmp_path / "manifest.json").write_text(json.dumps({"embedding_model": "some-other-model"}))
    assert bi._load_reuse_map() == {}


def test_reuse_map_empty_when_no_index(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "INDEX_DIR", tmp_path)
    assert bi._load_reuse_map() == {}

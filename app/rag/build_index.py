"""Build the searchable index for the GreenLeaf chat from docs/greenleaf_corpus/*.md.

Run: python app/rag/build_index.py   (use the project's .venv interpreter)

Reads the markdown docs, strips YAML frontmatter, chunks each body into ~300-word
passages, embeds them with the configured sentence-transformers model (L2-normalized),
and writes app/rag/index_data/{embeddings.npy, chunks.json, manifest.json}.

Production hardening (Phase 0):
- Chunk records now persist ALL frontmatter (category/region/fetch_date), not just
  title/source_name/source_url — needed for metadata-filtered retrieval later.
- A manifest.json stamps the embedding model + version + dim + corpus_version so a
  reader can refuse a mismatched index, and so we can tell what produced a store.
- DELTA builds: a chunk whose exact text already exists in the previous index reuses
  its embedding instead of re-encoding. Removed docs' chunks simply fall out. So an
  unchanged corpus re-runs without loading the model at all, and adding one doc only
  embeds that doc's chunks.
- ATOMIC publish: everything is written to a .staging/ dir and os.replace'd into place
  with manifest.json LAST (the commit point). A crash mid-write never publishes a
  half-built index; the reader validates manifest vs files and fails loud on mismatch.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import yaml

from app.core import config

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "docs" / "greenleaf_corpus"
INDEX_DIR = Path(__file__).resolve().parent / "index_data"
STAGING_DIR = INDEX_DIR / ".staging"
MANIFEST_NAME = "manifest.json"

# Kept as module-level names (read from config) so existing call sites / tests that
# reference them keep working; env can override via app.core.config.
MODEL_NAME = config.EMBEDDING_MODEL
TARGET_WORDS = config.CHUNK_TARGET_WORDS
MIN_CHUNK_CHARS = config.CHUNK_MIN_CHARS


def parse_frontmatter(raw_text):
    """Split a markdown doc into (meta_dict, body).

    Frontmatter is the block between the first two '---' lines. If absent or
    malformed, returns ({}, original_text).
    """
    text = raw_text.lstrip("﻿")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            return meta, parts[2].lstrip("\n")
    return {}, text


def chunk_text(body, target_words=TARGET_WORDS, min_chunk_chars=MIN_CHUNK_CHARS):
    """Chunk body into ~target_words passages on paragraph boundaries.

    Greedily accumulate paragraphs (split on blank lines) until adding the next
    would exceed ~target_words, then start a new chunk. Splits only on paragraph
    boundaries, so sentences are never cut. Drops chunks under min_chunk_chars.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks = []
    current = []
    count = 0
    for para in paragraphs:
        words = len(para.split())
        if current and count + words > target_words:
            chunks.append("\n\n".join(current))
            current = [para]
            count = words
        else:
            current.append(para)
            count += words
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c) >= min_chunk_chars]


def _content_hash(text):
    """Stable sha256 of a document's raw text (whitespace-normalized)."""
    norm = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def build_records(md_files):
    """Parse + chunk each doc into chunk records and per-doc metadata.

    Returns (records, docs_meta) where records is the ordered list of chunk dicts
    (chunk_id assigned by global position) and docs_meta maps source_file ->
    {content_hash, n_chunks}. Pure: no model load, fully unit-testable.
    """
    records = []
    docs_meta = {}
    for path in md_files:
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        chunks = chunk_text(body)
        for text in chunks:
            records.append({
                "chunk_id": len(records),
                "text": text,
                "source_file": path.name,
                "title": meta.get("title"),
                "source_name": meta.get("source_name"),
                "source_url": meta.get("source_url"),
                # Previously DROPPED — now persisted for metadata-filtered retrieval.
                "category": meta.get("category"),
                "region": meta.get("region"),
                "fetch_date": str(meta.get("fetch_date")) if meta.get("fetch_date") is not None else None,
            })
        docs_meta[path.name] = {"content_hash": _content_hash(raw), "n_chunks": len(chunks)}
    return records, docs_meta


def _load_reuse_map():
    """Build {chunk_text: embedding_vector} from the EXISTING index, but only if it is
    compatible with the current model + dim (else changing the embedder must re-embed
    everything). Returns {} when there is no usable prior index."""
    emb_path = INDEX_DIR / "embeddings.npy"
    chunks_path = INDEX_DIR / "chunks.json"
    man_path = INDEX_DIR / MANIFEST_NAME
    if not (emb_path.exists() and chunks_path.exists()):
        return {}
    try:
        old_emb = np.load(emb_path)
        old_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError):
        return {}
    if len(old_emb) != len(old_chunks) or old_emb.ndim != 2:
        return {}  # corrupt/mismatched — don't trust it for reuse
    if old_emb.shape[1] != config.EMBEDDING_DIM:
        return {}  # dim changed -> incompatible vectors
    if man_path.exists():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            if man.get("embedding_model") != config.EMBEDDING_MODEL:
                return {}  # model changed -> incompatible vectors
        except (OSError, json.JSONDecodeError):
            pass  # no/legacy manifest: dim check above is our only guard
    return {rec["text"]: old_emb[i].astype(np.float32) for i, rec in enumerate(old_chunks)}


def _atomic_publish(embeddings, records, manifest):
    """Write embeddings/chunks/manifest to a staging dir, then os.replace into place
    with the manifest LAST so a partial write never becomes the live index."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    np.save(STAGING_DIR / "embeddings.npy", embeddings)
    (STAGING_DIR / "chunks.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (STAGING_DIR / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Commit order: data files first, manifest (the pointer) last.
    os.replace(STAGING_DIR / "embeddings.npy", INDEX_DIR / "embeddings.npy")
    os.replace(STAGING_DIR / "chunks.json", INDEX_DIR / "chunks.json")
    os.replace(STAGING_DIR / MANIFEST_NAME, INDEX_DIR / MANIFEST_NAME)
    try:
        STAGING_DIR.rmdir()
    except OSError:
        pass


def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    md_files = sorted(CORPUS_DIR.glob("*.md"))

    records, docs_meta = build_records(md_files)
    for name, info in docs_meta.items():
        print(f"[index] {name}: {info['n_chunks']} chunks")

    if not records:
        print("[index] FAILED: no chunks produced (is docs/greenleaf_corpus/ empty?)")
        sys.exit(1)

    texts = [r["text"] for r in records]
    reuse = _load_reuse_map()
    missing = sorted({t for t in texts if t not in reuse})

    if missing:
        # Import here so parse_frontmatter / chunk_text / build_records stay unit-testable
        # without loading the (heavy) embedding model. With a full cache hit we never get here.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBEDDING_MODEL)
        new_vecs = model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
        new_map = {t: np.asarray(v, dtype=np.float32) for t, v in zip(missing, new_vecs)}
    else:
        new_map = {}

    embeddings = np.zeros((len(records), config.EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        embeddings[i] = reuse[t] if t in reuse else new_map[t]

    try:
        import importlib.metadata as _md
        st_version = _md.version("sentence-transformers")
    except Exception:  # noqa: BLE001 — version stamp is best-effort
        st_version = "unknown"

    corpus_version = hashlib.sha256(
        "".join(f"{n}:{docs_meta[n]['content_hash']}" for n in sorted(docs_meta)).encode("utf-8")
    ).hexdigest()

    manifest = {
        "schema_version": config.INDEX_SCHEMA_VERSION,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_model_version": st_version,
        "embedding_dim": config.EMBEDDING_DIM,
        "chunk_count": len(records),
        "doc_count": len(md_files),
        "corpus_version": corpus_version,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "docs": docs_meta,
    }

    _atomic_publish(embeddings, records, manifest)

    reused = len(records) - len(missing) if missing else len(records)
    print(f"[index] total: {len(records)} chunks ({reused} reused, {len(missing)} embedded), "
          f"embeddings shape {embeddings.shape}, model {config.EMBEDDING_MODEL} v{st_version}")
    print(f"[index] corpus_version {corpus_version[:12]}… published atomically")
    print("[index] done")


if __name__ == "__main__":
    build()

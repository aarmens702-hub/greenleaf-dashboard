"""Build the searchable index for the GreenLeaf chat from docs/corpus/*.md.

Run once: python app/rag/build_index.py

Reads the markdown docs in docs/corpus/, strips YAML frontmatter, chunks each
body into ~300-word passages, embeds them with all-MiniLM-L6-v2 (normalized),
and writes app/rag/index_data/{embeddings.npy, chunks.json}. Idempotent:
rerunning overwrites with identical output (no timestamps in the data).
"""
from pathlib import Path
import json
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "docs" / "greenleaf_corpus"
INDEX_DIR = Path(__file__).resolve().parent / "index_data"
MODEL_NAME = "all-MiniLM-L6-v2"
TARGET_WORDS = 300
MIN_CHUNK_CHARS = 20


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


def chunk_text(body):
    """Chunk body into ~TARGET_WORDS passages on paragraph boundaries.

    Greedily accumulate paragraphs (split on blank lines) until adding the next
    would exceed ~TARGET_WORDS, then start a new chunk. Splits only on paragraph
    boundaries, so sentences are never cut. Drops chunks under MIN_CHUNK_CHARS.
    """
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks = []
    current = []
    count = 0
    for para in paragraphs:
        words = len(para.split())
        if current and count + words > TARGET_WORDS:
            chunks.append("\n\n".join(current))
            current = [para]
            count = words
        else:
            current.append(para)
            count += words
    if current:
        chunks.append("\n\n".join(current))
    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


def build():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    md_files = sorted(CORPUS_DIR.glob("*.md"))

    records = []
    for path in md_files:
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        chunks = chunk_text(body)
        for text in chunks:
            records.append({
                "chunk_id": len(records),
                "text": text,
                "source_file": path.name,
                "title": meta.get("title"),
                "source_name": meta.get("source_name"),
                "source_url": meta.get("source_url"),
            })
        print(f"[index] {path.name}: {len(chunks)} chunks")

    if not records:
        print("[index] FAILED: no chunks produced (is docs/corpus/ empty?)")
        sys.exit(1)

    # Import here so parse_frontmatter / chunk_text stay unit-testable without
    # loading the (heavy) embedding model.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    texts = [r["text"] for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[index] total: {len(records)} chunks, embeddings shape {embeddings.shape}")
    print("[index] done")


if __name__ == "__main__":
    build()

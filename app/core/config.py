"""Central, env-overridable config for GreenLeaf (stdlib only — no new deps).

Replaces constants that were duplicated/hardcoded across the RAG and chat modules
(embedding model + chunk params in app/rag/*, the DeepSeek base URL/model + the 120s
timeout + max_iterations + the 4000-char truncation + k in app/chat/*). A real
environment variable always wins, so a deployment can tune any of these without a
code edit; the defaults reproduce today's behavior exactly.

Phase 0 of the production-RAG pipeline. Kept dependency-free on purpose: a heavier
settings library (pydantic-settings) is a later-phase concern.
"""
import os


def _str(name, default):
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _int(name, default):
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


# --- Embedding / index -----------------------------------------------------
# Default reproduces the hackathon index exactly. The dim is asserted against the
# loaded embeddings at retrieve time so a model swap can't silently serve a
# mismatched index (see app/rag/retrieve.py).
EMBEDDING_MODEL = _str("GREENLEAF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = _int("GREENLEAF_EMBEDDING_DIM", 384)
CHUNK_TARGET_WORDS = _int("GREENLEAF_CHUNK_TARGET_WORDS", 300)
CHUNK_MIN_CHARS = _int("GREENLEAF_CHUNK_MIN_CHARS", 20)

# Bump when the on-disk index/manifest layout changes in a non-backward way.
INDEX_SCHEMA_VERSION = "1"

# --- LLM provider seam -----------------------------------------------------
DEEPSEEK_BASE_URL = _str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
CHAT_MODEL = _str("GREENLEAF_CHAT_MODEL", "deepseek-chat")
# Split the single dangerous 120s timeout into connect + total (read) budgets.
LLM_CONNECT_TIMEOUT = _float("GREENLEAF_LLM_CONNECT_TIMEOUT", 5.0)
LLM_TOTAL_TIMEOUT = _float("GREENLEAF_LLM_TOTAL_TIMEOUT", 60.0)
LLM_MAX_RETRIES = _int("GREENLEAF_LLM_MAX_RETRIES", 2)        # retries on 429/5xx
LLM_BACKOFF_BASE = _float("GREENLEAF_LLM_BACKOFF_BASE", 0.5)  # seconds, exponential

# Approx blended price per 1M tokens (CAD), used only to attach a cost estimate to
# each answer's trace. Override via env when prices change; this is telemetry, not billing.
LLM_PRICE_CAD_PER_MTOK_IN = _float("GREENLEAF_LLM_PRICE_IN", 0.40)
LLM_PRICE_CAD_PER_MTOK_OUT = _float("GREENLEAF_LLM_PRICE_OUT", 1.10)

# --- Agent loop ------------------------------------------------------------
AGENT_MAX_ITERATIONS = _int("GREENLEAF_AGENT_MAX_ITERATIONS", 6)
TOOL_RESULT_TRUNC = _int("GREENLEAF_TOOL_RESULT_TRUNC", 4000)
RETRIEVE_K = _int("GREENLEAF_RETRIEVE_K", 5)

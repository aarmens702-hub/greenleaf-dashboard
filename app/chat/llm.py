"""Single LLM provider seam for GreenLeaf (Phase 0 of the production pipeline).

Every model call goes through chat(). Centralizing it buys, in one place:
  - .env loading + API-key resolution (DeepSeek primary; ANTHROPIC_API_KEY tolerated
    as the stand-in where the key currently lives — a real env var always wins),
  - retry with exponential backoff on 429/5xx,
  - a SPLIT connect/total timeout instead of one blunt 120 s,
  - token-usage capture -> an estimated CAD cost + wall-clock latency on every call,
  - an optional fallback to a second OpenAI-compatible endpoint when the primary fails.

chat() returns a ChatResult; callers read .message exactly as they read the raw
assistant message before, and may record .usage/.cost_cad/.latency_ms for telemetry.

Provider is DeepSeek's OpenAI-compatible chat-completions API. To change providers,
point the env at an OpenAI-compatible gateway (e.g. OpenRouter/LiteLLM proxy) — no
caller changes. A native Anthropic-SDK fallback (format translation) is deferred.
"""
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.core import config

_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path):
    """Minimal .env loader (no dependency): KEY=VALUE lines -> os.environ.

    A real environment variable always wins (we only setdefault). The .env file is
    gitignored, so the key never enters version control.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(_ROOT / ".env")

# Prefer DEEPSEEK_API_KEY; fall back to ANTHROPIC_API_KEY (where the key currently lives).
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
USE_LIVE_API = bool(API_KEY)

# Optional secondary OpenAI-compatible endpoint, tried when the primary exhausts retries.
FALLBACK_BASE_URL = os.environ.get("GREENLEAF_FALLBACK_BASE_URL")
FALLBACK_API_KEY = os.environ.get("GREENLEAF_FALLBACK_API_KEY")
FALLBACK_MODEL = os.environ.get("GREENLEAF_FALLBACK_MODEL")

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_NO_KEY_MSG = (
    "No LLM API key is set, so live calls are disabled. Get a key at "
    "https://platform.deepseek.com and export DEEPSEEK_API_KEY."
)


@dataclass
class ChatResult:
    """One model turn: the assistant message plus telemetry."""
    message: dict
    usage: dict = field(default_factory=dict)
    cost_cad: float = 0.0
    latency_ms: int = 0
    provider: str = "deepseek"
    model: str = ""


def estimate_cost_cad(usage):
    """Rough CAD cost from a usage block (telemetry only, not billing)."""
    pin = usage.get("prompt_tokens", 0) or 0
    pout = usage.get("completion_tokens", 0) or 0
    return round(
        pin / 1e6 * config.LLM_PRICE_CAD_PER_MTOK_IN
        + pout / 1e6 * config.LLM_PRICE_CAD_PER_MTOK_OUT,
        6,
    )


def _post_once(base_url, api_key, model, messages, tools, max_tokens, timeout):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if tools is not None:
        payload["tools"] = tools
    return httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )


def _call_with_retry(base_url, api_key, model, messages, tools, max_tokens, timeout, provider):
    """POST with exponential backoff on transport errors and 429/5xx. Raises on final failure."""
    last_exc = None
    for attempt in range(config.LLM_MAX_RETRIES + 1):
        try:
            resp = _post_once(base_url, api_key, model, messages, tools, max_tokens, timeout)
            if resp.status_code in _RETRYABLE_STATUS and attempt < config.LLM_MAX_RETRIES:
                time.sleep(config.LLM_BACKOFF_BASE * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {}) or {}
            return ChatResult(
                message=data["choices"][0]["message"],
                usage=usage,
                cost_cad=estimate_cost_cad(usage),
                provider=provider,
                model=model,
            )
        except httpx.TransportError as e:  # timeouts, connection errors -> retryable
            last_exc = e
            if attempt < config.LLM_MAX_RETRIES:
                time.sleep(config.LLM_BACKOFF_BASE * (2 ** attempt))
                continue
            raise
    raise last_exc  # pragma: no cover


def chat(messages, tools=None, max_tokens=1024, model=None):
    """One chat-completions call with retry/timeout/fallback. Returns a ChatResult.

    Args: messages (list of role dicts); tools (OpenAI tool schemas or None);
    max_tokens (int); model (override; defaults to config.CHAT_MODEL).
    Raises RuntimeError if no API key is configured.
    """
    if not USE_LIVE_API:
        raise RuntimeError(_NO_KEY_MSG)
    model = model or config.CHAT_MODEL
    timeout = httpx.Timeout(config.LLM_TOTAL_TIMEOUT, connect=config.LLM_CONNECT_TIMEOUT)

    started = time.monotonic()
    try:
        result = _call_with_retry(
            config.DEEPSEEK_BASE_URL, API_KEY, model, messages, tools, max_tokens, timeout, "deepseek"
        )
    except (httpx.HTTPError,) as primary_exc:
        if FALLBACK_BASE_URL and FALLBACK_API_KEY:
            result = _call_with_retry(
                FALLBACK_BASE_URL, FALLBACK_API_KEY, FALLBACK_MODEL or model,
                messages, tools, max_tokens, timeout, "fallback",
            )
        else:
            raise primary_exc
    result.latency_ms = int((time.monotonic() - started) * 1000)
    return result

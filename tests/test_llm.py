"""LLM seam: cost estimate, retry-then-success, usage capture, no-key guard (Phase 0).

No network: llm._post_once is monkeypatched with a scripted response sequence.
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat import llm
from app.core import config


class _FakeResp:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)

    def json(self):
        return self._payload


_OK_PAYLOAD = {
    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
}


def test_estimate_cost_cad():
    cost = llm.estimate_cost_cad({"prompt_tokens": 1_000_000, "completion_tokens": 0})
    assert cost == round(config.LLM_PRICE_CAD_PER_MTOK_IN, 6)


def test_chat_raises_without_key(monkeypatch):
    monkeypatch.setattr(llm, "USE_LIVE_API", False)
    with pytest.raises(RuntimeError, match="API key"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_chat_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm, "USE_LIVE_API", True)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)  # no real backoff wait
    seq = iter([_FakeResp(429), _FakeResp(200, _OK_PAYLOAD)])
    monkeypatch.setattr(llm, "_post_once", lambda *a, **k: next(seq))

    res = llm.chat([{"role": "user", "content": "hi"}])
    assert res.message["content"] == "hi"
    assert res.usage["prompt_tokens"] == 1000
    assert res.cost_cad > 0
    assert res.provider == "deepseek"
    assert res.latency_ms >= 0


def test_chat_propagates_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm, "USE_LIVE_API", True)
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)
    monkeypatch.setattr(llm, "FALLBACK_BASE_URL", None)  # no fallback configured
    monkeypatch.setattr(llm, "_post_once", lambda *a, **k: _FakeResp(503))
    with pytest.raises(httpx.HTTPStatusError):
        llm.chat([{"role": "user", "content": "hi"}])

"""Observability primitives (Phase 1)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import obs


def test_new_request_id_unique_and_short():
    a, b = obs.new_request_id(), obs.new_request_id()
    assert a != b
    assert len(a) == 12


def test_log_answer_never_raises():
    obs.log_answer({"event": "answer", "request_id": "x", "cost_cad": 0.0})


def test_append_jsonl_writes_lines(tmp_path):
    p = tmp_path / "sub" / "traces.jsonl"
    assert obs.append_jsonl(str(p), {"a": 1}) is True
    assert obs.append_jsonl(str(p), {"a": 2}) is True
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1


def test_append_jsonl_noop_on_empty_path():
    assert obs.append_jsonl("", {"a": 1}) is False

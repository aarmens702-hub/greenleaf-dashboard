"""Config defaults + env-override mechanism (Phase 0)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import config


def test_defaults_reproduce_hackathon_behavior():
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert config.EMBEDDING_DIM == 384
    assert config.CHUNK_TARGET_WORDS == 300
    assert config.CHAT_MODEL == "deepseek-chat"
    assert config.AGENT_MAX_ITERATIONS == 6
    assert config.TOOL_RESULT_TRUNC == 4000


def test_int_helper_overrides_and_falls_back(monkeypatch):
    monkeypatch.setenv("GL_TEST_INT", "7")
    assert config._int("GL_TEST_INT", 3) == 7
    assert config._int("GL_TEST_MISSING", 3) == 3
    monkeypatch.setenv("GL_TEST_BAD", "abc")
    assert config._int("GL_TEST_BAD", 3) == 3  # bad value -> default, not a crash


def test_str_and_float_helpers(monkeypatch):
    monkeypatch.setenv("GL_TEST_STR", "x")
    assert config._str("GL_TEST_STR", "d") == "x"
    assert config._str("GL_TEST_MISSING", "d") == "d"
    monkeypatch.setenv("GL_TEST_EMPTY", "")
    assert config._str("GL_TEST_EMPTY", "d") == "d"  # empty -> default
    monkeypatch.setenv("GL_TEST_FLOAT", "1.5")
    assert config._float("GL_TEST_FLOAT", 0.0) == 1.5

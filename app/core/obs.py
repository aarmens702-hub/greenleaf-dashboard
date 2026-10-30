"""Observability primitives for GreenLeaf (Phase 1) — stdlib only, no new deps.

Three tiny things every later phase builds on:
  - new_request_id(): a short id stamped on each answer so logs, traces, and user
    feedback can be joined.
  - log_answer(record): emit ONE JSON line per answer to the 'greenleaf' logger (the
    always-on, cheap backstop — ships to stdout in a container, greppable anywhere).
  - append_jsonl(path, record): best-effort append a JSON line to a trace/feedback
    store. Best-effort means a logging failure NEVER breaks answering.
"""
import json
import logging
import uuid
from pathlib import Path

from app.core import config

_LOGGER_NAME = "greenleaf"
_configured = False


def get_logger():
    """A stdlib logger with a single stream handler. We pass it pre-serialized JSON
    strings, so the formatter is just the message."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False
        _configured = True
    return logger


def new_request_id():
    return uuid.uuid4().hex[:12]


def log_answer(record):
    """Emit one structured JSON log line for an answer. Never raises."""
    try:
        get_logger().info(json.dumps(record, default=str, sort_keys=True))
    except Exception:  # noqa: BLE001 — telemetry must never break the request
        pass


def append_jsonl(path, record):
    """Append one JSON line to `path` (creating parent dirs). Best-effort: returns
    True on success, False on any failure (never raises)."""
    if not path:
        return False
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        return True
    except Exception:  # noqa: BLE001
        return False

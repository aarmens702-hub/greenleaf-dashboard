"""Fixtures every test module gets.

The money guard: no test may reach Modal, spend on it or read its token (~/.modal.toml). It never
imports the SDK; it only points the SDK's config at an absent file, drops its token env vars, and
makes run_grid._run_remote (the one seam that opens a remote session) raise, so a test that forgets
to fake the session fails instead of spending.
"""

import os

import pytest


def _remote_not_faked(*args, **kwargs):
    raise AssertionError("remote session not faked")


@pytest.fixture(autouse=True)
def _never_the_remote_service(monkeypatch, tmp_path):
    from greenleaf_control import run_grid

    monkeypatch.setenv("MODAL_CONFIG_PATH", str(tmp_path / "absent.toml"))
    for name in list(os.environ):
        if name.upper().startswith("MODAL_TOKEN"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(run_grid, "_run_remote", _remote_not_faked)

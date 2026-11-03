"""Tests for the public wording (issue #64, S6): nothing a visitor reads from the README onward says
"co-pilot". The README, the write-up and brief it links (docs/REPORT.md, docs/PRD.md), AGENTS.md,
the package's description in control/pyproject.toml and greenleaf_control's module docstring say
"residual agent" (and "constrained variant" for lag_b100), never "co-pilot", "copilot" or
"co pilot" (#63's public wording, docs/STATUS.md).

The conventions pinned here.
- The match is #63's site test's: CO_PILOT, run on the text with its whitespace collapsed and
  lowercased, so a line break, any case or a Unicode hyphen (U+2010, U+2011) inside the word is
  still caught.
- The description is pyproject's [project].description (tomllib); the docstring is
  ast.get_docstring of greenleaf_control/__init__.py. Each source is its own case, so a failure
  names the file.
- Each source also says "residual": the swap renames the agent, it doesn't drop it.
- Code comments and test docstrings are out of scope (#64): they aren't read from the README.

Known answer: on dev at 4691468, docs/REPORT.md, docs/PRD.md, AGENTS.md, the pyproject description
and the package docstring say "co-pilot" and fail; README.md (renamed by e8d8a6b) passes. Every
file is read inside the test body, so a missing one FAILS on an assertion rather than ERRORs.
Nothing here imports greenleaf_control, builds an env or touches the network.
"""

import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path(__file__).resolve().parents[1]
# "co-pilot", "copilot" or "co pilot" (any Unicode hyphen), on lowercased, collapsed text (#63,
# test_site.py's CO_PILOT, copied rather than imported so this module stands alone).
CO_PILOT = re.compile(r"co[-‐‑ ]?pilot")


def _read(path: Path) -> str:
    assert path.is_file(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def _description() -> str:
    """control/pyproject.toml's [project].description."""
    project = tomllib.loads(_read(CONTROL / "pyproject.toml")).get("project", {})
    description = project.get("description")
    assert isinstance(description, str) and description.strip(), "no [project].description"
    return description


def _docstring() -> str:
    """greenleaf_control/__init__.py's module docstring."""
    doc = ast.get_docstring(ast.parse(_read(CONTROL / "greenleaf_control" / "__init__.py")))
    assert doc, "greenleaf_control/__init__.py has no module docstring"
    return doc


SOURCES = {
    "README.md": lambda: _read(ROOT / "README.md"),
    "docs/REPORT.md": lambda: _read(ROOT / "docs" / "REPORT.md"),
    "docs/PRD.md": lambda: _read(ROOT / "docs" / "PRD.md"),
    "AGENTS.md": lambda: _read(ROOT / "AGENTS.md"),
    "pyproject.toml-description": _description,
    "greenleaf_control-docstring": _docstring,
}


@pytest.mark.parametrize("source", SOURCES)
def test_public_text_says_residual_agent_not_co_pilot(source):
    """Done when (#64): the source, whitespace collapsed, any case, says no "co-pilot" (nor
    "copilot" or "co pilot", any Unicode hyphen), and says "residual". On 4691468 REPORT.md,
    PRD.md, AGENTS.md, the description ("a reinforcement-learning co-pilot") and the docstring
    ("an RL co-pilot") fail; README.md passes."""
    text = " ".join(SOURCES[source]().split()).lower()
    stale = sorted({m.group(0) for m in CO_PILOT.finditer(text)})
    assert not stale, f"{source} still says {stale}: the public word is residual agent (#63, #64)"
    assert "residual" in text, f"{source} never says residual: the swap dropped the agent"

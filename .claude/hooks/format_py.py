"""PostToolUse hook: ruff-format the Python file Claude just edited, when it
is part of the control/ project (GreenLeaf 2.0). The hackathon dashboard at
the repo root is frozen and keeps its own style.

Format only, never `ruff check --fix`: the fixer strips an import added in
one edit before the next edit adds its first use. /ship runs the lint.
A formatter failure must never block Claude, so every error exits 0.
Standard library only. Ported from crivo.
"""

import json
import os
import shutil
import subprocess
import sys

try:
    path = json.load(sys.stdin).get("tool_input", {}).get("file_path") or ""
except (ValueError, AttributeError):
    sys.exit(0)
if not path.endswith(".py") or not os.path.isfile(path):
    sys.exit(0)

# The control/ project that holds this file: its nearest ancestor named control
# with a pyproject.toml (works from the main checkout and from any worktree).
d = os.path.dirname(os.path.abspath(path))
project = None
while d and d != os.path.dirname(d):
    if os.path.basename(d) == "control" and os.path.isfile(os.path.join(d, "pyproject.toml")):
        project = d
        break
    d = os.path.dirname(d)
if project is None:
    sys.exit(0)

uv = os.path.expanduser("~/.local/bin/uv")
if not os.access(uv, os.X_OK):
    uv = shutil.which("uv")
if not uv:
    sys.exit(0)

try:
    subprocess.run(
        [uv, "run", "--quiet", "--project", project, "ruff", "format", path],
        cwd=project,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=20,
        check=False,
    )
except (OSError, subprocess.SubprocessError):
    pass
sys.exit(0)

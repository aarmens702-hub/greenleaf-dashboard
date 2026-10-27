"""Stop hook: when Claude is about to finish with uncommitted Python changes
in control/ (GreenLeaf 2.0), run its fast test suite first. If it fails,
block the stop (exit 2) and hand Claude the failures, so "done" always means
"tests pass".

It tests the tree the session is working in (stdin's cwd, resolved to its
git top level), so parallel worktrees don't block each other. Skips quietly
when nothing relevant changed, when uv or control/.venv is missing, or when
this stop was itself caused by this hook (stop_hook_active), so it can't loop.

When the tests pass but a scorer file changed (SCORER_FILES: what decides a
result's number), it blocks once more: results recorded before the change
are no longer comparable, and the user has to hear that plainly. Standard
library only. Ported from crivo.
"""

SCORER_FILES = (
    "control/greenleaf_control/scenarios.py",
    "control/greenleaf_control/rewards.py",
    "control/greenleaf_control/envs.py",
    "control/greenleaf_control/metrics.py",
    "docs/PROTOCOL.md",
)

import json
import os
import shutil
import subprocess
import sys

try:
    data = json.load(sys.stdin)
except ValueError:
    data = {}
if data.get("stop_hook_active"):
    sys.exit(0)


def git(*args, cwd):
    return subprocess.run(
        ["git", "-C", cwd, *args],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
if data.get("cwd"):
    try:
        top = git("rev-parse", "--show-toplevel", cwd=data["cwd"]).stdout.strip()
        if top and os.path.isdir(os.path.join(top, "control")):
            root = top
    except (OSError, subprocess.SubprocessError):
        pass

project = os.path.join(root, "control")
try:
    changed = git("status", "--porcelain", "--", "control", "docs/PROTOCOL.md", cwd=root)
except (OSError, subprocess.SubprocessError):
    sys.exit(0)
lines = changed.stdout.splitlines()
touched_scorer = [f for f in SCORER_FILES if any(line.endswith(f) for line in lines)]
if not any(line.endswith(".py") for line in lines) and not touched_scorer:
    sys.exit(0)

uv = os.path.expanduser("~/.local/bin/uv")
if not os.access(uv, os.X_OK):
    uv = shutil.which("uv")
if not uv or not os.path.isdir(os.path.join(project, ".venv")):
    sys.exit(0)

try:
    run = subprocess.run(
        [uv, "run", "--quiet", "pytest", "-q", "-x", "-m", "not slow", "-p", "no:cacheprovider"],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=280,
        check=False,
    )
except subprocess.TimeoutExpired:
    sys.stderr.write(
        "The fast test suite did not finish in 280 s. Run `uv run pytest -q -m 'not slow'` "
        "in control/ yourself and tell the user the result before finishing.\n"
    )
    sys.exit(2)
except OSError:
    sys.exit(0)

# pytest exit 5 = no tests collected (an empty project): not a failure.
if run.returncode in (0, 5):
    if touched_scorer:
        sys.stderr.write(
            f"Tests pass, but {', '.join(touched_scorer)} decides how results are scored. "
            "Results recorded before this change are no longer comparable with new ones. "
            "Before finishing, tell the user that plainly, and say whether any ledger "
            "result needs re-running. This is a core change: Aarmen sees the diff.\n"
        )
        sys.exit(2)
    sys.exit(0)

tail = "\n".join((run.stdout + run.stderr).splitlines()[-40:])
sys.stderr.write(
    "The control/ test suite is failing (uv run pytest -q -m 'not slow'). Fix it "
    f"before finishing, or tell the user plainly which tests fail and why.\n\n{tail}\n"
)
sys.exit(2)

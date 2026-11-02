"""Stop hook: when Claude is about to finish with uncommitted Python changes
in control/ (GreenLeaf 2.0), run its fast test suite first. If it fails,
block the stop (exit 2) and hand Claude the failures, so "done" always means
"tests pass".

It tests the tree the session is working in (stdin's cwd, resolved to its
git top level), so parallel worktrees don't block each other. Skips quietly
when nothing relevant changed, or when this stop was itself caused by this
hook (stop_hook_active, read before any git call), so it can't loop. When
control/ changed but uv is missing or can't run, or control/.venv is
missing, it blocks and asks Claude to run `uv sync` and the fast tests
itself.

When the tests pass but a scorer file changed (SCORER_FILES: what decides a
result's number), it blocks once more: results recorded before the change
are no longer comparable, and the user has to hear that plainly.

It fails closed: stdin that isn't a JSON object (empty or not JSON
included), a cwd that isn't a string, a git that times out or can't be run,
a git status that fails, or a crash blocks the stop and asks Claude to run
the fast tests itself. A git rev-parse that fails is not an error: a cwd
outside any repo falls back to CLAUDE_PROJECT_DIR (or the hook's cwd), and
the git status there still fails closed. In theory unreadable stdin blocks
every stop, since it can't show stop_hook_active; Claude Code always sends
an object, and Esc interrupts. Standard library only. Ported from crivo.

Its slowest path, both git calls (GIT_TIMEOUT each) and the suite
(PYTEST_TIMEOUT) timing out, plus ~20 s of its own, must end inside the Stop
hook's timeout in .claude/settings.json (360 s): Claude Code kills a hook at
that timeout and lets the stop through.
"""

import json
import os
import shutil
import subprocess
import sys

SCORER_FILES = (
    "control/greenleaf_control/scenarios.py",
    "control/greenleaf_control/rewards.py",
    "control/greenleaf_control/envs.py",
    "control/greenleaf_control/metrics.py",
    "docs/PROTOCOL.md",
)

# Seconds the hook waits on each git call, and on the fast suite.
GIT_TIMEOUT = 20
PYTEST_TIMEOUT = 280

# What Claude does before finishing when the hook can't run the fast tests itself.
RUN_THEM = (
    "`uv run pytest -q -m 'not slow'` in your worktree's control/, tell the user the result, "
    f"and say whether a scorer file ({', '.join(SCORER_FILES)}) changed."
)
# The end of every message for a stop the hook can't judge.
RUN_THEM_YOURSELF = "so it can't tell whether control/ changed. Before finishing, run " + RUN_THEM


def git(*args, cwd):
    return subprocess.run(
        ["git", "-C", cwd, *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT,
        check=False,
    )


def main():
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return "stop_tests: a Stop payload that isn't a JSON object, " + RUN_THEM_YOURSELF
    if data.get("stop_hook_active"):
        return None
    cwd = data.get("cwd")
    if not isinstance(cwd, (str, type(None))):
        return "stop_tests: a Stop payload whose cwd isn't a string, " + RUN_THEM_YOURSELF

    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        if cwd:
            top = git("rev-parse", "--show-toplevel", cwd=cwd).stdout.strip()
            if top and os.path.isdir(os.path.join(top, "control")):
                root = top
        changed = git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "control",
            "docs/PROTOCOL.md",
            cwd=root,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"stop_tests: git failed ({type(exc).__name__}: {exc}), " + RUN_THEM_YOURSELF
    if changed.returncode != 0:
        return (
            f"stop_tests: git status exited {changed.returncode} "
            f"({changed.stderr.strip()}), " + RUN_THEM_YOURSELF
        )

    project = os.path.join(root, "control")
    lines = changed.stdout.splitlines()
    touched_scorer = [f for f in SCORER_FILES if any(line.endswith(f) for line in lines)]
    if not any(line.endswith(".py") for line in lines) and not touched_scorer:
        return None

    uv = os.path.expanduser("~/.local/bin/uv")
    if not os.access(uv, os.X_OK):
        uv = shutil.which("uv")
    if not uv:
        return (
            "stop_tests: control/ changed, but there's no uv in ~/.local/bin or on PATH to run "
            "its fast tests. Install uv and run `uv sync` in control/, then before finishing run "
            + RUN_THEM
        )
    if not os.path.isdir(os.path.join(project, ".venv")):
        return (
            "stop_tests: control/ changed, but control/.venv is missing, so its fast tests can't "
            "run. Run `uv sync` in control/, then before finishing run " + RUN_THEM
        )

    try:
        run = subprocess.run(
            [
                uv,
                "run",
                "--quiet",
                "pytest",
                "-q",
                "-x",
                "-m",
                "not slow",
                "-p",
                "no:cacheprovider",
            ],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=PYTEST_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            f"The fast test suite did not finish in {PYTEST_TIMEOUT} s. Run "
            "`uv run pytest -q -m 'not slow'` in control/ yourself and tell the user the result "
            "before finishing."
        )
    except OSError as exc:
        return (
            f"stop_tests: uv at {uv} couldn't run ({type(exc).__name__}: {exc}). Fix uv, then "
            "before finishing run " + RUN_THEM
        )

    # pytest exit 5 = no tests collected (an empty project): not a failure.
    if run.returncode in (0, 5):
        if touched_scorer:
            return (
                f"Tests pass, but {', '.join(touched_scorer)} decides how results are scored. "
                "Results recorded before this change are no longer comparable with new ones. "
                "Before finishing, tell the user that plainly, and say whether any ledger "
                "result needs re-running. This is a core change: Aarmen sees the diff."
            )
        return None

    tail = "\n".join((run.stdout + run.stderr).splitlines()[-40:])
    return (
        "The control/ test suite is failing (uv run pytest -q -m 'not slow'). Fix it "
        f"before finishing, or tell the user plainly which tests fail and why.\n\n{tail}"
    )


try:
    message = main()
except Exception as exc:  # noqa: BLE001 - fail closed: a crash must block the stop, not pass it
    message = f"stop_tests crashed ({type(exc).__name__}: {exc}), " + RUN_THEM_YOURSELF
if message:
    sys.stderr.write(message + "\n")
    sys.exit(2)
sys.exit(0)

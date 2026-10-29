"""PreToolUse hook for the test-writer and builder subagents (set in their
.claude/agents/*.md frontmatter), so their split of work is enforced, not
trusted:

    python3 guard_tests.py only-tests  -> test-writer may edit test files only
    python3 guard_tests.py no-tests    -> builder may not edit test files

A test file is anything under a tests/ folder (control/tests/ included, and
its golden/ and probes/ folders), a test_*.py or *_test.py file, or a
conftest.py. The path is judged once resolved (against CLAUDE_PROJECT_DIR,
or the cwd, and through symlinks), so tests/../ or a link can't pass app
code off as a test, or a test as app code. Exit code 2 blocks the edit and
tells the agent why, in one stderr line. It fails closed: a missing or
unknown mode, stdin that isn't a JSON object, a tool_input that isn't an
object, a file_path that isn't a string, or a crash all block; an empty
file_path passes. Standard library only, so it runs on any python3. Ported
from crivo.
"""

import json
import os
import re
import sys

MODES = ("only-tests", "no-tests")


def block(reason):
    sys.stderr.write("Blocked: " + " ".join(reason.split()) + "\n")
    sys.exit(2)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in MODES:
        return (
            f"guard_tests needs a mode ({', '.join(MODES)}) and got {mode or 'none'!r}, so it "
            "fails closed — wire it as guard_tests.py <mode> in the agent's frontmatter"
        )
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = None
    args = data.get("tool_input") if isinstance(data, dict) else None
    path = args.get("file_path") if isinstance(args, dict) else None
    if not isinstance(path, str):
        return (
            f"guard_tests ({mode}): an edit it can't read, so it fails closed — it needs a JSON "
            "object with a tool_input object whose file_path is a string"
        )
    if not path:
        return None

    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    p = os.path.realpath(os.path.join(root, path)).replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    is_test = bool(
        re.search(r"(^|/)tests/", p) or re.fullmatch(r"test_.*\.py|.*_test\.py|conftest\.py", name)
    )

    if mode == "only-tests" and not is_test:
        return (
            f"the test-writer only writes tests (tests/, test_*.py, "
            f"conftest.py). {path} is app code; the builder changes it."
        )
    if mode == "no-tests" and is_test:
        return (
            f"the builder doesn't edit tests, golden numbers or probes. If "
            f"{path} looks wrong, stop and report why instead of changing it."
        )
    return None


try:
    reason = main()
except Exception as exc:  # noqa: BLE001 - fail closed: a crash must block the call, not pass it
    reason = f"guard_tests crashed ({type(exc).__name__}: {exc}), so it fails closed — report it"
if reason:
    block(reason)
sys.exit(0)

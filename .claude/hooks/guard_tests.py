"""PreToolUse hook for the test-writer and builder subagents (set in their
.claude/agents/*.md frontmatter), so their split of work is enforced, not
trusted:

    python3 guard_tests.py only-tests  -> test-writer may edit test files only
    python3 guard_tests.py no-tests    -> builder may not edit test files

A test file is anything under a tests/ folder (control/tests/ included, and
its golden/ and probes/ folders), a test_*.py or *_test.py file, or a
conftest.py. Exit code 2 blocks the edit and tells the agent why.
Standard library only, so it runs on any python3. Ported from crivo.
"""

import json
import re
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    path = json.load(sys.stdin).get("tool_input", {}).get("file_path") or ""
except (ValueError, AttributeError):
    sys.exit(0)
if not path:
    sys.exit(0)

p = path.replace("\\", "/")
name = p.rsplit("/", 1)[-1]
is_test = bool(
    re.search(r"(^|/)tests/", p)
    or re.fullmatch(r"test_.*\.py|.*_test\.py|conftest\.py", name)
)

if mode == "only-tests" and not is_test:
    sys.stderr.write(
        f"Blocked: the test-writer only writes tests (tests/, test_*.py, "
        f"conftest.py). {path} is app code; the builder changes it.\n"
    )
    sys.exit(2)
if mode == "no-tests" and is_test:
    sys.stderr.write(
        f"Blocked: the builder doesn't edit tests, golden numbers or probes. If "
        f"{path} looks wrong, stop and report why instead of changing it.\n"
    )
    sys.exit(2)
sys.exit(0)

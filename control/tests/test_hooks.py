"""Tests for the PreToolUse hooks in .claude/hooks/ (issue #5): guard_runs, guard_tests,
protect_secrets.

guard_runs.py (new in #5) keeps every specialist off the ledger, results, runs, Modal, real
training and the sealed 2012 test weather, with the same rules in all four modes. guard_tests.py and
protect_secrets.py already exist; their tests pin today's behaviour so #5 can't change it.

Each hook runs as a subprocess with Claude Code's PreToolUse JSON on stdin: exit 2 blocks the call,
exit 0 lets it through. `python` on a missing script also exits 2, so every block is checked for its
one "Blocked" line on stderr too; until guard_runs.py exists its tests fail on that assertion.

Issue #26 makes the safety net fail closed: guard_tests.py and protect_secrets.py block a call they
can't read, the Stop hook (stop_tests.py) sees a .py file inside a brand-new folder under control/,
and control-ci.yml refuses a stale lockfile, reads the repo only and pins its uv.

Issue #27 closes three gaps: guard_tests.py judges the resolved path, so `tests/..` or a symlink
can't pass app code off as a test; protect_secrets.py reads a Grep's glob and type, not only its
path; and stop_tests.py blocks (exit 2) when it can't read its input or git can't say what changed.

Issue #29 closes the gaps left after #27. guard_tests.py and protect_secrets.py casefold before they
match, since the Mac's filesystem ignores case; protect_secrets.py judges a symlink by its target as
well as its own name, blocks a Grep of the home folder with no filter, and calls the brace cap's
block what it is, not a crash; stop_tests.py blocks the stop when control/.venv or uv is missing or
uv can't run; and the Stop hook's settings timeout covers its own worst case.

Issue #40 opens a checkout's control/results/final/ to Read, Grep and Glob, in every mode, once the
ledger beside it (that checkout's control/ledger/ledger.jsonl) holds a final-result row: the seal is
spent, so reading the results can't steer anything. No ledger, an unreadable one, or any line that
isn't a JSON object keeps it sealed. Writes there and the 2012 weather stay blocked for good. These
tests use fake checkouts under tmp_path, so the real ledger's seal decides none of them: the day F1's
final-result lands on dev, a test that read the real repo's results/final would turn red.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / ".claude" / "hooks"
AGENTS = ROOT / ".claude" / "agents"
CONTROL = ROOT / "control"

# guard_runs.py's modes, one per specialist; each is also the agent's file name in AGENTS.
MODES = ["builder", "test-writer", "scout", "reviewer"]
GUARD_RUNS_COMMAND = 'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/guard_runs.py" {}'
GUARD_TESTS_COMMAND = 'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/guard_tests.py" {}'
GC = "uv run python -m greenleaf_control"
TEST_SPLITS = ["amsterdam_2012", "london_2012", "abbotsford_2012"]
# gl_gym ships its own copy of the test weather inside the venv.
GL_GYM_2012 = CONTROL / ".venv/lib/python3.12/site-packages/gl_gym/data/weather/Amsterdam/2012.csv"
WRITE_TOOLS = ["Write", "Edit", "MultiEdit", "NotebookEdit"]


def _call(tool: str, cwd: Path = CONTROL, **tool_input) -> dict:
    """A PreToolUse payload as Claude Code sends it on stdin."""
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input,
        "cwd": str(cwd),
    }


def _bash(command: str, cwd: Path = CONTROL) -> dict:
    return _call("Bash", cwd, command=command, description="run it")


def _file_call(tool: str, path, cwd: Path = CONTROL) -> dict:
    """A call on one file, with the input fields each file tool really sends."""
    path = str(path)
    tool_input = {
        "Read": {"file_path": path},
        "Write": {"file_path": path, "content": "x\n"},
        "Edit": {"file_path": path, "old_string": "a", "new_string": "b"},
        "MultiEdit": {"file_path": path, "edits": [{"old_string": "a", "new_string": "b"}]},
        "NotebookEdit": {"notebook_path": path, "new_source": "x"},
    }[tool]
    return _call(tool, cwd, **tool_input)


def _hook(name: str, stdin, *argv: str) -> subprocess.CompletedProcess:
    """Run one hook with stdin as its input: a dict is JSON-encoded, a str is sent as is.

    -S keeps site-packages off sys.path: the agents run the hooks on whatever python3 is on PATH,
    so a hook may need nothing beyond the standard library.
    """
    return subprocess.run(
        [sys.executable, "-S", str(HOOKS / name), *argv],
        input=stdin if isinstance(stdin, str) else json.dumps(stdin),
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(ROOT)},
    )


def _assert_blocked(proc: subprocess.CompletedProcess, prefix: str = "Blocked") -> None:
    """Exit 2 with exactly one stderr line starting prefix.

    A missing script also exits 2 ("can't open file"), so the code alone proves nothing.
    """
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    lines = proc.stderr.splitlines()
    assert len(lines) == 1 and lines[0].startswith(prefix), proc.stderr


def _assert_allowed(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, (proc.returncode, proc.stderr)


def _pre_tool_use(agent: str) -> list[tuple]:
    """(matcher, type, command) of every PreToolUse hook in an agent file's YAML frontmatter."""
    _, front, _ = (AGENTS / f"{agent}.md").read_text().split("---\n", 2)
    entries = (yaml.safe_load(front).get("hooks") or {}).get("PreToolUse") or []
    return [
        (entry.get("matcher"), hook.get("type"), hook.get("command"))
        for entry in entries
        for hook in entry.get("hooks") or []
    ]


# ---------------------------------------------------------------------------------------------
# guard_runs.py
# ---------------------------------------------------------------------------------------------

# Paths only the CLIs write, as an agent might name them: from this checkout, relative to the
# payload's cwd, in another worktree, and in mixed case.
PROTECTED_PATHS = [
    pytest.param(str(CONTROL / "ledger/ledger.jsonl"), CONTROL, id="ledger"),
    pytest.param(str(CONTROL / "results/iqm.json"), CONTROL, id="results"),
    pytest.param(str(CONTROL / "runs/run-0123abcd/manifest.json"), CONTROL, id="runs"),
    pytest.param(str(CONTROL / "configs/rule_based_tuned.yml"), CONTROL, id="rule_based_tuned"),
    pytest.param("ledger/ledger.jsonl", CONTROL, id="relative to cwd"),
    pytest.param("/x/.worktrees/9/control/ledger/ledger.jsonl", CONTROL, id="another worktree"),
    pytest.param(str(ROOT / "CONTROL/Ledger/ledger.jsonl"), CONTROL, id="mixed case"),
]

BLOCKED_BASH = [
    # Shell writes to the CLI-owned paths.
    pytest.param("echo x > control/ledger/ledger.jsonl", ROOT, id=">"),
    pytest.param("echo x >> control/ledger/ledger.jsonl", ROOT, id=">>"),
    pytest.param("echo x | tee -a control/ledger/ledger.jsonl", ROOT, id="tee -a"),
    pytest.param("cp a control/results/x", ROOT, id="cp"),
    pytest.param("mv a control/runs/x", ROOT, id="mv"),
    pytest.param("rm -rf control/runs", ROOT, id="rm -rf"),
    pytest.param("sed -i s/a/b/ control/ledger/ledger.jsonl", ROOT, id="sed -i"),
    pytest.param("truncate -s0 control/ledger/ledger.jsonl", ROOT, id="truncate"),
    pytest.param("cd control && rm ledger/ledger.jsonl", ROOT, id="cd then rm"),
    pytest.param(
        "cp x.yml control/configs/rule_based_tuned.yml", ROOT, id="cp onto rule_based_tuned"
    ),
    # Modal, anywhere in the command and in any case.
    pytest.param("modal run x.py", CONTROL, id="modal run"),
    pytest.param("MODAL run x.py", CONTROL, id="MODAL in capitals"),
    pytest.param(f"{GC}.run_grid --backend modal", CONTROL, id="run_grid on modal"),
    # Real training, grids and tuning: the orchestrator runs these through /experiment.
    pytest.param(f"{GC}.train --method ppo --seed 0", CONTROL, id="train"),
    pytest.param(f"{GC}.train --registered r", CONTROL, id="train --registered"),
    pytest.param(
        f"{GC}.train --smoke && {GC}.train --registered r", CONTROL, id="smoke then real train"
    ),
    pytest.param(f"{GC}.run_grid --registered r --backend local", CONTROL, id="run_grid"),
    pytest.param(f"{GC}.tune_rb --registered r", CONTROL, id="tune_rb"),
    # The sealed test weather and the final evaluation.
    *[
        pytest.param(
            f"{GC}.evaluate --controller rb_default --split {split} --out /tmp/t.parquet",
            CONTROL,
            id=f"evaluate {split}",
        )
        for split in TEST_SPLITS
    ],
    pytest.param(f"{GC}.evaluate --final x", CONTROL, id="evaluate --final"),
    # Every ledger subcommand but show writes a row.
    pytest.param(f"{GC}.ledger register --issue 5 --hypothesis h", CONTROL, id="ledger register"),
    pytest.param(
        f"{GC}.ledger result --registered r --numbers n.json", CONTROL, id="ledger result"
    ),
    pytest.param(f"{GC}.ledger decide --result r --adopted --note ok", CONTROL, id="ledger decide"),
    pytest.param(
        f"{GC}.ledger final-register --issue 9 --approved-by Aarmen",
        CONTROL,
        id="ledger final-register",
    ),
]

# The 2012 weather only: control/results/final/ opens with the ledger beside it (#40), so its cases
# use fake checkouts under tmp_path (below), never this repo's.
SEALED_READS = [
    pytest.param(
        _file_call("Read", CONTROL / "data/weather/Amsterdam/2012.csv"), id="read 2012 weather"
    ),
    pytest.param(_file_call("Read", GL_GYM_2012), id="read gl_gym's 2012 weather"),
    pytest.param(
        _call("Glob", ROOT, pattern="control/data/weather/*/2012.csv"), id="glob 2012 weather"
    ),
]

ALLOWED = [
    pytest.param(_bash(f"{GC}.train --method ppo --seed 0 --smoke"), id="train --smoke"),
    pytest.param(
        _bash(f"{GC}.evaluate --controller rb_default --split validation --out /tmp/v.parquet"),
        id="evaluate on validation",
    ),
    pytest.param(_bash(f"{GC}.ledger show"), id="ledger show"),
    pytest.param(_bash(f"{GC}.ledger show --issue 3"), id="ledger show --issue"),
    pytest.param(_bash(f"{GC}.run_grid --issue 5 --dry"), id="run_grid --dry"),
    pytest.param(_bash(f"{GC}.tune_rb --trials 5 --smoke"), id="tune_rb --smoke"),
    pytest.param(_bash('uv run pytest -q -m "not slow"'), id="fast tests"),
    # Naming a module's file isn't running it.
    pytest.param(
        _bash("git diff -- control/greenleaf_control/train.py", ROOT), id="git diff of train.py"
    ),
    pytest.param(_file_call("Read", CONTROL / "tests/test_ledger.py"), id="read a test"),
    pytest.param(_file_call("Edit", CONTROL / "greenleaf_control/envs.py"), id="edit envs.py"),
    # Reading the ledger is fine; writing it isn't. The reviewer diffs it to check it only grows.
    pytest.param(_file_call("Read", CONTROL / "ledger/ledger.jsonl"), id="read the ledger"),
    pytest.param(
        _bash("git diff origin/dev...HEAD -- control/ledger/ledger.jsonl", ROOT),
        id="git diff of the ledger",
    ),
    pytest.param(
        _file_call("Read", CONTROL / "data/weather/Amsterdam/2010.csv"), id="read 2010 weather"
    ),
    pytest.param(_file_call("Edit", HOOKS / "guard_runs.py"), id="edit guard_runs.py"),
]

# Input the hook can't trust: it must block, not guess.
MALFORMED = [
    pytest.param("not json", id="not json"),
    pytest.param("[]", id="a list"),
    pytest.param(json.dumps({"tool_name": "Bash", "tool_input": "ls"}), id="tool_input a string"),
    pytest.param(
        json.dumps({"tool_name": "Write", "tool_input": {"file_path": 5, "content": "x"}}),
        id="file_path an int",
    ),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("path", "cwd"), PROTECTED_PATHS)
@pytest.mark.parametrize("tool", WRITE_TOOLS)
def test_guard_runs_blocks_edits_to_cli_owned_paths(tool, path, cwd, mode):
    """Write, Edit, MultiEdit and NotebookEdit can't touch the ledger, results, runs or the tuned
    rule-based config. Paths are realpath-ed against the payload's cwd, case-folded and matched
    anywhere, so a relative path, another worktree and mixed case are all caught.
    """
    _assert_blocked(_hook("guard_runs.py", _file_call(tool, path, cwd), mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("command", "cwd"), BLOCKED_BASH)
def test_guard_runs_blocks_bash_that_writes_results_or_runs_for_real(command, cwd, mode):
    """Bash can't write the CLI-owned paths, mention modal, train, grid or tune without --smoke or
    --dry, evaluate on a 2012 split or --final, or run a ledger subcommand other than show.
    """
    _assert_blocked(_hook("guard_runs.py", _bash(command, cwd), mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("payload", SEALED_READS)
def test_guard_runs_blocks_reads_of_the_test_weather(payload, mode):
    """Read, Grep and Glob can't reach any weather/<dir>/2012.csv, gl_gym's copy in the venv
    included. control/results/final/'s reads are tested on fake checkouts (#40, below).
    """
    _assert_blocked(_hook("guard_runs.py", payload, mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("payload", ALLOWED)
def test_guard_runs_allows_the_specialists_everyday_work(payload, mode):
    """Smoke runs, validation evaluation, ledger show, dry grids, the tests, reading or diffing the
    ledger, reading non-test weather, and editing library code or the hook itself all pass in every
    mode.
    """
    _assert_allowed(_hook("guard_runs.py", payload, mode))


@pytest.mark.parametrize("mode", MODES)
def test_guard_runs_lets_an_unknown_tool_through(mode):
    """Wired with matcher "*", the hook sees every tool. One it has no rule for, like the report
    back to the orchestrator, passes even when its text names modal and 2012.
    """
    payload = _call("SubagentHandback", message="modal 2012 results")
    _assert_allowed(_hook("guard_runs.py", payload, mode))


@pytest.mark.parametrize("argv", [[], [""], ["orchestrator"]], ids=["missing", "empty", "unknown"])
def test_guard_runs_fails_closed_without_a_known_mode(argv):
    """A missing, empty or unknown mode blocks even a call every mode allows."""
    payload = _file_call("Read", CONTROL / "tests/test_ledger.py")
    _assert_blocked(_hook("guard_runs.py", payload, *argv))
    _assert_allowed(_hook("guard_runs.py", payload, "builder"))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("stdin", MALFORMED)
def test_guard_runs_fails_closed_on_malformed_input(stdin, mode):
    """stdin that isn't a JSON object, a tool_input that isn't an object, or a file_path that isn't
    a string blocks the call.
    """
    _assert_blocked(_hook("guard_runs.py", stdin, mode))


@pytest.mark.parametrize("agent", MODES)
def test_every_specialist_runs_guard_runs_on_every_tool(agent):
    """Each agent's frontmatter wires guard_runs.py in its own mode as a PreToolUse hook on "*"."""
    assert ("*", "command", GUARD_RUNS_COMMAND.format(agent)) in _pre_tool_use(agent)


@pytest.mark.parametrize(
    ("agent", "mode"), [("builder", "no-tests"), ("test-writer", "only-tests")]
)
def test_guard_tests_stays_wired(agent, mode):
    """Wiring guard_runs keeps the builder's and test-writer's existing guard_tests hook."""
    expected = ("Edit|Write|MultiEdit", "command", GUARD_TESTS_COMMAND.format(mode))
    assert expected in _pre_tool_use(agent)


@pytest.mark.parametrize(
    ("doc", "stale"),
    [
        ("CLAUDE.md", "Planned (Stage 0)"),
        (".claude/agents/builder.md", "until the `guard_runs` hook lands"),
    ],
)
def test_docs_say_guard_runs_has_landed(doc, stale):
    """CLAUDE.md and the builder's rule 10 describe guard_runs as enforced now, not planned."""
    assert stale not in (ROOT / doc).read_text()


# ---------------------------------------------------------------------------------------------
# guard_tests.py: today's behaviour, pinned
# ---------------------------------------------------------------------------------------------

TEST_FILES = [
    "control/tests/x.py",
    "control/tests/golden/x.json",
    "control/tests/probes/x.py",
    "scripts/test_x.py",
    "scripts/x_test.py",
    "control/conftest.py",
]
APP_FILES = [
    "control/greenleaf_control/envs.py",
    ".claude/hooks/guard_runs.py",
    ".claude/agents/builder.md",
    "CLAUDE.md",
]


@pytest.mark.parametrize("rel", TEST_FILES)
def test_guard_tests_lets_the_test_writer_edit_test_files(rel):
    """only-tests passes anything under tests/, a test_*.py or *_test.py file, or a conftest.py."""
    _assert_allowed(_hook("guard_tests.py", _file_call("Edit", ROOT / rel), "only-tests"))


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit"])
def test_guard_tests_blocks_the_test_writer_from_app_code(tool):
    payload = _file_call(tool, CONTROL / "greenleaf_control/envs.py")
    _assert_blocked(_hook("guard_tests.py", payload, "only-tests"), "Blocked: the test-writer")


@pytest.mark.parametrize("rel", APP_FILES)
def test_guard_tests_lets_the_builder_edit_app_code_hooks_and_docs(rel):
    _assert_allowed(_hook("guard_tests.py", _file_call("Edit", ROOT / rel), "no-tests"))


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit"])
@pytest.mark.parametrize(
    "rel", ["control/tests/test_hooks.py", "control/tests/golden/x", "control/conftest.py"]
)
def test_guard_tests_blocks_the_builder_from_tests(rel, tool):
    payload = _file_call(tool, ROOT / rel)
    _assert_blocked(_hook("guard_tests.py", payload, "no-tests"), "Blocked: the builder")


@pytest.mark.parametrize("mode", ["only-tests", "no-tests"])
def test_guard_tests_lets_an_empty_file_path_through(mode):
    _assert_allowed(_hook("guard_tests.py", _call("Write", file_path="", content="x"), mode))


# ---------------------------------------------------------------------------------------------
# protect_secrets.py: today's behaviour, pinned
# ---------------------------------------------------------------------------------------------

SECRET_FILES = [
    pytest.param(ROOT / ".env", id="root .env"),
    pytest.param(ROOT / "app/.env", id="app/.env"),
    pytest.param(ROOT / ".env.local", id=".env.local"),
    pytest.param(ROOT / ".env.production", id=".env.production"),
    pytest.param(Path.home() / ".modal.toml", id="~/.modal.toml"),
]


@pytest.mark.parametrize("tool", ["Read", "Edit", "Write"])
@pytest.mark.parametrize("path", SECRET_FILES)
def test_protect_secrets_blocks_a_secret_file(path, tool):
    """Any .env or .env.<suffix> at any depth, and Modal's token file, are off limits."""
    _assert_blocked(_hook("protect_secrets.py", _file_call(tool, path)))


def test_protect_secrets_blocks_grep_of_a_dotenv():
    payload = _call("Grep", ROOT, pattern="KEY", path=str(ROOT / ".env"))
    _assert_blocked(_hook("protect_secrets.py", payload))


@pytest.mark.parametrize("rel", [".env.example", "config.env", "control/greenleaf_control/envs.py"])
def test_protect_secrets_lets_non_secret_files_through(rel):
    """.env.example only lists names; a name that merely contains "env" isn't a secret."""
    _assert_allowed(_hook("protect_secrets.py", _file_call("Read", ROOT / rel)))


@pytest.mark.parametrize("command", ["cat .env", "source app/.env", "cat ~/.modal.toml"])
def test_protect_secrets_blocks_a_shell_command_on_a_secret_file(command):
    _assert_blocked(_hook("protect_secrets.py", _bash(command, ROOT)))


@pytest.mark.parametrize("command", ["cat .env.example", 'python -c "load_dotenv()"'])
def test_protect_secrets_lets_other_shell_commands_through(command):
    _assert_allowed(_hook("protect_secrets.py", _bash(command, ROOT)))


# ---------------------------------------------------------------------------------------------
# Fail closed (#26): guard_tests.py and protect_secrets.py block a call they can't read
# ---------------------------------------------------------------------------------------------


def _real(payload: dict) -> dict:
    """payload with the other top-level fields Claude Code sends on every PreToolUse call."""
    return {
        "session_id": "abc123",
        "transcript_path": "/tmp/transcript.jsonl",
        "permission_mode": "default",
        "tool_use_id": "toolu_01",
        **payload,
    }


# stdin neither hook can read: not JSON, not an object, no tool_input object, or a path that isn't a
# string. Before #26 both hooks exited 0 on these, or crashed with exit 1, which Claude Code also lets
# through. JSON nested past the recursion limit makes json raise RecursionError, not ValueError, so
# it reaches each hook's crash handler.
UNREADABLE = [
    pytest.param("[" * 100_000 + "]" * 100_000, id="nested past the recursion limit"),
    pytest.param("", id="empty"),
    pytest.param("not json", id="not json"),
    pytest.param("[]", id="a list"),
    pytest.param("null", id="null"),
    pytest.param("{}", id="an empty object"),
    pytest.param(json.dumps({"tool_name": "Edit"}), id="no tool_input"),
    pytest.param(json.dumps({"tool_name": "Edit", "tool_input": None}), id="tool_input null"),
    pytest.param(json.dumps({"tool_name": "Edit", "tool_input": "x.py"}), id="tool_input a string"),
    pytest.param(
        json.dumps({"tool_name": "Write", "tool_input": {"file_path": 5, "content": "x"}}),
        id="file_path an int",
    ),
]
# guard_tests decides on the edited file alone, so an edit that doesn't name one is unreadable too.
UNREADABLE_EDIT = [
    *UNREADABLE,
    pytest.param(
        json.dumps({"tool_name": "Edit", "tool_input": {"old_string": "a", "new_string": "b"}}),
        id="no file_path",
    ),
    pytest.param(
        json.dumps({"tool_name": "Write", "tool_input": {"file_path": None, "content": "x"}}),
        id="file_path null",
    ),
]
# protect_secrets picks its rule by tool_name (the command for Bash, the paths for the rest), so a
# call without a string tool_name, or with a command that isn't a string, is unreadable too.
UNREADABLE_CALL = [
    *UNREADABLE,
    pytest.param(json.dumps({"tool_input": {"command": "ls"}}), id="no tool_name"),
    pytest.param(
        json.dumps({"tool_name": 5, "tool_input": {"file_path": "README.md"}}),
        id="tool_name an int",
    ),
    pytest.param(
        json.dumps({"tool_name": "Bash", "tool_input": {"command": ["cat", "README.md"]}}),
        id="command a list",
    ),
    # Grep's glob and type pick the files it searches (#27), so they must be strings too.
    pytest.param(
        json.dumps({"tool_name": "Grep", "tool_input": {"pattern": "KEY", "glob": ["*.py"]}}),
        id="Grep glob a list",
    ),
    pytest.param(
        json.dumps({"tool_name": "Grep", "tool_input": {"pattern": "KEY", "type": 5}}),
        id="Grep type an int",
    ),
]

# A file each guard_tests mode lets its agent edit.
PERMITTED_EDIT = {
    "only-tests": CONTROL / "tests/test_x.py",
    "no-tests": CONTROL / "greenleaf_control/envs.py",
}


@pytest.mark.parametrize("mode", ["only-tests", "no-tests"])
@pytest.mark.parametrize("stdin", UNREADABLE_EDIT)
def test_guard_tests_fails_closed_on_unreadable_input(stdin, mode):
    """stdin that isn't a JSON object, or a call without a tool_input object holding a string
    file_path, blocks with one "Blocked" line in both modes; a permitted edit still passes.
    """
    _assert_blocked(_hook("guard_tests.py", stdin, mode))
    _assert_allowed(_hook("guard_tests.py", _file_call("Edit", PERMITTED_EDIT[mode]), mode))


@pytest.mark.parametrize(
    "argv",
    [(), ("",), ("builder",), ("no-test",)],
    ids=["missing", "empty", "guard_runs' mode", "misspelt"],
)
def test_guard_tests_fails_closed_without_a_known_mode(argv):
    """A missing, empty or unknown mode (guard_runs' builder, or no-test for no-tests) blocks an
    edit no-tests allows, with one "Blocked" line; the same edit in no-tests passes.
    """
    payload = _file_call("Edit", PERMITTED_EDIT["no-tests"])
    _assert_blocked(_hook("guard_tests.py", payload, *argv))
    _assert_allowed(_hook("guard_tests.py", payload, "no-tests"))


@pytest.mark.parametrize("mode", ["only-tests", "no-tests"])
@pytest.mark.parametrize("tool", ["Edit", "Write", "MultiEdit"])
def test_guard_tests_still_reads_a_real_call(tool, mode):
    """A full payload, with Claude Code's other top-level fields and the tool's non-path inputs
    (booleans, a list of edits), is readable: a permitted edit passes in both modes.
    """
    path = str(PERMITTED_EDIT[mode])
    tool_input = {
        "Edit": {"file_path": path, "old_string": "a", "new_string": "b", "replace_all": False},
        "Write": {"file_path": path, "content": "x\n"},
        "MultiEdit": {
            "file_path": path,
            "edits": [{"old_string": "a", "new_string": "b", "replace_all": True}],
        },
    }[tool]
    _assert_allowed(_hook("guard_tests.py", _real(_call(tool, **tool_input)), mode))


@pytest.mark.parametrize("stdin", UNREADABLE_CALL)
def test_protect_secrets_fails_closed_on_unreadable_input(stdin):
    """stdin that isn't a JSON object, a tool_name that isn't a string, a tool_input that isn't an
    object, or a path or command that isn't a string blocks with one "Blocked" line. Reading
    README.md still passes and reading .env is still blocked.
    """
    _assert_blocked(_hook("protect_secrets.py", stdin))
    _assert_allowed(_hook("protect_secrets.py", _file_call("Read", ROOT / "README.md")))
    _assert_blocked(_hook("protect_secrets.py", _file_call("Read", ROOT / ".env")))


# One harmless call per tool protect_secrets is wired on (.claude/settings.json), with the non-path
# inputs Claude Code really sends (numbers, booleans, lists), so failing closed can't block them.
HARMLESS_REAL_CALLS = [
    pytest.param(_call("Read", file_path=str(ROOT / "README.md"), offset=1, limit=50), id="Read"),
    pytest.param(
        _call(
            "Edit",
            file_path=str(CONTROL / "greenleaf_control/envs.py"),
            old_string="a",
            new_string="b",
            replace_all=True,
        ),
        id="Edit",
    ),
    pytest.param(_call("Write", file_path=str(CONTROL / "x.py"), content="x\n"), id="Write"),
    pytest.param(
        _call(
            "MultiEdit",
            file_path=str(CONTROL / "x.py"),
            edits=[{"old_string": "a", "new_string": "b", "replace_all": False}],
        ),
        id="MultiEdit",
    ),
    pytest.param(
        _call("Grep", ROOT, pattern="def ", output_mode="content", head_limit=20, **{"-n": True}),
        id="Grep without a path",
    ),
    pytest.param(
        _call("Grep", ROOT, pattern="def ", path="control", glob="*.py"), id="Grep with a path"
    ),
    pytest.param(
        _call(
            "Bash",
            ROOT,
            command="git status",
            description="status",
            timeout=60000,
            run_in_background=False,
        ),
        id="Bash",
    ),
    pytest.param(
        _call(
            "NotebookEdit",
            notebook_path=str(ROOT / "x.ipynb"),
            new_source="x",
            cell_id="c1",
            edit_mode="replace",
        ),
        id="NotebookEdit",
    ),
]


@pytest.mark.parametrize("payload", HARMLESS_REAL_CALLS)
def test_protect_secrets_still_reads_every_real_call(payload):
    _assert_allowed(_hook("protect_secrets.py", _real(payload)))


# ---------------------------------------------------------------------------------------------
# stop_tests.py (#26): a .py file in a brand-new folder under control/ is a change
# ---------------------------------------------------------------------------------------------

# Stands in for uv: says it ran and exits 1, as a failing fast suite would, so the Stop hook blocks
# (exit 2) with this line in its message exactly when it decided control/ changed.
FAKE_UV = '#!/bin/sh\necho "FAKE-UV-RAN $*"\nexit 1\n'


def _stop_repo(tmp_path: Path) -> tuple[Path, dict]:
    """A git repo shaped like this one (a committed control/ package, a control/.venv folder) and the
    environment to run the Stop hook in: HOME's ~/.local/bin/uv, also first on PATH, is FAKE_UV,
    and git reads no global or system config, so a user's status.showUntrackedFiles can't decide
    the result.
    """
    uv = tmp_path / "home/.local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text(FAKE_UV)
    uv.chmod(0o755)
    repo = tmp_path / "repo"
    package = repo / "control/greenleaf_control"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text("A = 1\n")
    (repo / "control/pyproject.toml").write_text('[project]\nname = "x"\n')
    (repo / "control/.venv").mkdir()
    env = {
        k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k != "XDG_CONFIG_HOME"
    }
    env.update(
        HOME=str(tmp_path / "home"),
        PATH=f"{uv.parent}{os.pathsep}{os.environ['PATH']}",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        CLAUDE_PROJECT_DIR=str(repo),
    )
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    for args in (["init", "-q"], ["add", "-A"], [*identity, "commit", "-q", "-m", "init"]):
        subprocess.run(
            ["git", "-C", str(repo), *args], env=env, check=True, capture_output=True, timeout=20
        )
    return repo, env


def _stop(
    repo: Path,
    env: dict,
    stdin: str | None = None,
    stop_hook_active: bool = False,
    timeout: float = 30,
) -> subprocess.CompletedProcess:
    """Run the Stop hook as Claude Code does when a session in repo is about to finish.

    stdin, when given, is sent as is in place of Claude Code's payload. stop_hook_active says this
    stop follows one the hook itself blocked.
    """
    payload = {
        "session_id": "abc123",
        "transcript_path": str(repo.parent / "transcript.jsonl"),
        "cwd": str(repo),
        "hook_event_name": "Stop",
        "stop_hook_active": stop_hook_active,
    }
    return subprocess.run(
        [sys.executable, "-S", str(HOOKS / "stop_tests.py")],
        input=json.dumps(payload) if stdin is None else stdin,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=repo,
        env=env,
    )


@pytest.mark.parametrize(
    "rel",
    [
        pytest.param("control/greenleaf_control/a.py", id="an edited tracked file"),
        pytest.param("control/greenleaf_control/b.py", id="a new file in a tracked folder"),
        pytest.param("control/newpkg/mod.py", id="a new file in a new folder"),
        pytest.param("control/newpkg/sub/mod.py", id="a new file two new folders deep"),
    ],
)
def test_stop_tests_runs_the_fast_tests_on_a_python_change(tmp_path, rel):
    """An edited or new .py file under control/ makes the Stop hook run the fast tests, inside a new
    folder too. Plain `git status --porcelain` lists a new folder as `?? control/newpkg/`, never
    the .py inside it, so the hook used to skip it; --untracked-files=all lists the file.
    """
    repo, env = _stop_repo(tmp_path)
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("B = 2\n")
    proc = _stop(repo, env)
    assert proc.returncode == 2 and "FAKE-UV-RAN" in proc.stderr, (proc.returncode, proc.stderr)


@pytest.mark.parametrize(
    "rel",
    [
        pytest.param(None, id="nothing changed"),
        pytest.param("control/notes/todo.md", id="a new folder without python"),
        pytest.param("app/new/mod.py", id="a new folder outside control"),
    ],
)
def test_stop_tests_skips_when_no_python_under_control_changed(tmp_path, rel):
    """No .py change under control/ (and no scorer file): the hook exits 0 without running the
    suite.
    """
    repo, env = _stop_repo(tmp_path)
    if rel:
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n")
    proc = _stop(repo, env)
    assert proc.returncode == 0 and "FAKE-UV-RAN" not in proc.stderr, (proc.returncode, proc.stderr)


# ---------------------------------------------------------------------------------------------
# guard_tests.py (#27): the path is resolved before the test-file rules apply
# ---------------------------------------------------------------------------------------------

# A scorer file reached through tests/..: it reads as a test but is app code. The relative form is
# resolved against the project (CLAUDE_PROJECT_DIR, the repo root in _hook).
SCORER_BEHIND_DOTDOT = [
    pytest.param(str(CONTROL / "tests/../greenleaf_control/envs.py"), id="absolute"),
    pytest.param("control/tests/../greenleaf_control/envs.py", id="relative"),
]


@pytest.mark.parametrize("path", SCORER_BEHIND_DOTDOT)
def test_guard_tests_blocks_the_test_writer_from_app_code_behind_dotdot(path):
    """`tests/../greenleaf_control/envs.py` is envs.py, a scorer file: the test-writer is blocked.
    A real test file reached through `..` (control/tests/../tests/test_x.py) still passes.
    """
    payload = _file_call("Edit", path)
    _assert_blocked(_hook("guard_tests.py", payload, "only-tests"), "Blocked: the test-writer")
    real_test = _file_call("Edit", "control/tests/../tests/test_x.py")
    _assert_allowed(_hook("guard_tests.py", real_test, "only-tests"))


@pytest.mark.parametrize("path", SCORER_BEHIND_DOTDOT)
def test_guard_tests_lets_the_builder_edit_app_code_behind_dotdot(path):
    """The builder may edit envs.py however the path is spelt; a golden number reached through
    `greenleaf_control/..` is still a test file, so the builder is blocked from it.
    """
    _assert_allowed(_hook("guard_tests.py", _file_call("Edit", path), "no-tests"))
    golden = _file_call("Edit", "control/greenleaf_control/../tests/golden/x.json")
    _assert_blocked(_hook("guard_tests.py", golden, "no-tests"), "Blocked: the builder")


@pytest.mark.parametrize(
    ("mode", "link", "target"),
    [
        pytest.param(
            "only-tests", "tests/link.py", "greenleaf_control/envs.py", id="test-writer, tests/link"
        ),
        pytest.param(
            "no-tests", "greenleaf_control/link.py", "tests/test_x.py", id="builder, package link"
        ),
    ],
)
def test_guard_tests_judges_a_symlink_by_its_target(tmp_path, mode, link, target):
    """A symlink is the file it points at: tests/link.py pointing at app code is app code, so the
    test-writer is blocked; greenleaf_control/link.py pointing at a test is a test, so the builder
    is blocked.
    """
    (tmp_path / target).parent.mkdir(parents=True)
    (tmp_path / target).write_text("x\n")
    (tmp_path / link).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / link).symlink_to(tmp_path / target)
    _assert_blocked(_hook("guard_tests.py", _file_call("Edit", tmp_path / link), mode))


# ---------------------------------------------------------------------------------------------
# protect_secrets.py (#27): a Grep's glob and type pick files too
# ---------------------------------------------------------------------------------------------

# Grep filters (glob or ripgrep type) that select .env, .env.<suffix> or .modal.toml, bare or at any
# depth. `*` and `.*` select them too and block: an accepted false positive. ripgrep's toml type is
# *.toml, which takes in .modal.toml, and its all type is every file. A glob through a real folder
# (control/*, **/control/*) reaches the secret files in it: ripgrep's -g overrides gitignore and
# hidden-file skipping.
SECRET_GREP_FILTERS = [
    pytest.param("glob", ".env", id=".env"),
    pytest.param("glob", "**/.env*", id="**/.env*"),
    pytest.param("glob", "*.env", id="*.env"),
    pytest.param("glob", "{.env,foo}", id="brace group"),
    pytest.param("glob", "app/.env", id="app/.env"),
    pytest.param("glob", "*.py .env", id="two globs in one field"),
    pytest.param("glob", ".*", id=".*"),
    pytest.param("glob", "*", id="*"),
    pytest.param("glob", ".modal.toml", id=".modal.toml"),
    pytest.param("glob", ".env.*", id=".env.*"),
    pytest.param("glob", "control/*", id="control/*"),
    pytest.param("glob", "control/.env*", id="control/.env*"),
    pytest.param("glob", "x/y/z/.env*", id="x/y/z/.env*"),
    pytest.param("glob", "**/control/*", id="**/control/*"),
    pytest.param("type", "toml", id="type toml"),
    pytest.param("type", "all", id="type all"),
]

# (field, a normal filter, its twin that selects a secret file).
NORMAL_GREP_FILTERS = [
    pytest.param("glob", "*.py", "*.env", id="*.py"),
    pytest.param("glob", "*.{py,md}", "*.{py,env}", id="*.{py,md}"),
    pytest.param("glob", "envs.py", ".env", id="envs.py"),
    pytest.param("glob", ".env.example", ".env.local", id=".env.example"),
    pytest.param("glob", "**/test_*.py", "**/.env*", id="**/test_*.py"),
    pytest.param("glob", "control/**/*.py", "control/**/.env*", id="control/**/*.py"),
    pytest.param("type", "py", "toml", id="type py"),
]


def _grep(**tool_input) -> dict:
    """A Grep for KEY from the repo root, without a path, so only its filter picks the files."""
    return _call("Grep", ROOT, pattern="KEY", output_mode="content", **tool_input)


@pytest.mark.parametrize(("field", "value"), SECRET_GREP_FILTERS)
def test_protect_secrets_blocks_a_grep_filtered_to_a_secret_file(field, value):
    """A Grep whose glob or type selects a secret file would print its lines: blocked, like a Grep
    whose path is one.
    """
    _assert_blocked(_hook("protect_secrets.py", _grep(**{field: value})))


@pytest.mark.parametrize(("field", "normal", "secret"), NORMAL_GREP_FILTERS)
def test_protect_secrets_lets_a_grep_with_a_normal_filter_through(field, normal, secret):
    """A Grep filtered to Python, Markdown, envs.py, .env.example or test files passes, and its twin
    filtered to a secret file is blocked, so the pass isn't the hook ignoring the filter.
    """
    _assert_allowed(_hook("protect_secrets.py", _grep(**{field: normal})))
    _assert_blocked(_hook("protect_secrets.py", _grep(**{field: secret})))


def test_protect_secrets_blocks_a_grep_globbed_to_the_repo_folder():
    """A Grep of the repo by absolute path, globbed to **/<the repo's folder name>/*, selects every
    file at the repo's top level, the root .env among them: blocked.
    """
    payload = _grep(path=str(ROOT), glob=f"**/{ROOT.name}/*")
    _assert_blocked(_hook("protect_secrets.py", payload))


def test_protect_secrets_blocks_a_brace_blow_up_in_time():
    """30 empty {,} groups before x.py spell one glob 2**30 ways. Expanding every spelling runs past
    any hook timeout, and Claude Code lets a call through when its hook times out, so the hook must
    block a glob it can't judge, well inside _hook's 10 s. A TimeoutExpired here is the failure.
    """
    payload = _grep(glob="{,}" * 30 + "x.py")
    _assert_blocked(_hook("protect_secrets.py", payload))


# ---------------------------------------------------------------------------------------------
# stop_tests.py (#27): input it can't read, or a git that fails, blocks the stop
# ---------------------------------------------------------------------------------------------

# stdin the Stop hook can't read. Before #27 the first two passed as an empty payload (exit 0) and
# the rest crashed (exit 1); Claude Code lets the stop through on both. Only exit 2 blocks it.
UNREADABLE_STOP = [
    pytest.param("not json", id="not json"),
    pytest.param("", id="empty"),
    pytest.param("[]", id="a list"),
    pytest.param("null", id="null"),
    pytest.param(json.dumps({"cwd": 5}), id="cwd an int"),
    pytest.param("[" * 100_000 + "]" * 100_000, id="nested past the recursion limit"),
]

# Stands in for a git that fails: says so on stderr and exits 128, as git does outside a repo.
FAILING_GIT = '#!/bin/sh\necho "fatal: FAKE-GIT-FAILED" >&2\nexit 128\n'
# Stands in for a git that hangs. exec, so the hook's timeout kills sleep itself: a sleep left behind
# by a killed sh would hold the pipes open and the hook would wait on it.
HANGING_GIT = "#!/bin/sh\nexec sleep 30\n"


def _with_git(tmp_path: Path, env: dict, script: str | None) -> dict:
    """env with git replaced, once _stop_repo has built the repo with the real one: script becomes
    the only git on PATH, or, with None, PATH holds no git at all.
    """
    bin_dir = tmp_path / "fake-git-bin"
    bin_dir.mkdir()
    if script is None:
        return {**env, "PATH": str(bin_dir)}
    git = bin_dir / "git"
    git.write_text(script)
    git.chmod(0o755)
    return {**env, "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}"}


@pytest.mark.parametrize("stdin", UNREADABLE_STOP)
def test_stop_tests_fails_closed_on_unreadable_input(tmp_path, stdin):
    """On a clean repo, where a readable payload passes, stdin that isn't a JSON object or whose cwd
    isn't a string blocks the stop (exit 2) with the hook's own stop_tests message on stderr. A
    missing script or an uncaught crash would say "python: can't open file" or "Traceback" instead.
    """
    repo, env = _stop_repo(tmp_path)
    proc = _stop(repo, env, stdin=stdin)
    assert proc.returncode == 2 and proc.stderr.startswith("stop_tests"), (
        proc.returncode,
        proc.stderr,
    )


@pytest.mark.parametrize(
    "script", [pytest.param(FAILING_GIT, id="git exits 128"), pytest.param(None, id="no git")]
)
def test_stop_tests_blocks_when_git_cant_say_what_changed(tmp_path, script):
    """A git that fails or isn't there can't tell a clean tree from a changed one, so the stop is
    blocked (exit 2) with the hook's own stop_tests message naming git, not passed as "nothing
    changed". The stop that follows, stop_hook_active, still passes (exit 0), so the hook can't
    trap the session in a loop.
    """
    repo, env = _stop_repo(tmp_path)
    env = _with_git(tmp_path, env, script)
    proc = _stop(repo, env)
    assert proc.returncode == 2 and proc.stderr.startswith("stop_tests"), (
        proc.returncode,
        proc.stderr,
    )
    # A whole word: tmp_path's folder (test_..._when_git_c0) doesn't count.
    assert re.search(r"\bgit\b", proc.stderr, re.IGNORECASE), proc.stderr
    again = _stop(repo, env, stop_hook_active=True)
    assert again.returncode == 0, (again.returncode, again.stderr)


@pytest.mark.slow
def test_stop_tests_blocks_when_git_times_out(tmp_path):
    """A git that hangs past the hook's 20 s timeout blocks the stop (exit 2) with the hook's own
    stop_tests message naming git. Takes 20 to 40 s: one or two git calls time out before the hook
    decides.
    """
    repo, env = _stop_repo(tmp_path)
    env = _with_git(tmp_path, env, HANGING_GIT)
    proc = _stop(repo, env, timeout=90)
    assert proc.returncode == 2 and proc.stderr.startswith("stop_tests"), (
        proc.returncode,
        proc.stderr,
    )
    # A whole word: tmp_path's folder (test_..._when_git_t0) doesn't count.
    assert re.search(r"\bgit\b", proc.stderr, re.IGNORECASE), proc.stderr


# ---------------------------------------------------------------------------------------------
# control-ci.yml (#26): a stale lock fails, the token reads only, uv is pinned
# ---------------------------------------------------------------------------------------------

CI = ROOT / ".github/workflows/control-ci.yml"


def _ci() -> dict:
    """The workflow, parsed (PyYAML reads its `on:` key as True)."""
    return yaml.safe_load(CI.read_text())


def _ci_steps() -> list[dict]:
    return [step for job in _ci()["jobs"].values() for step in job.get("steps") or []]


def test_ci_install_refuses_a_stale_lockfile():
    """Every `uv sync` uses --locked, which fails when uv.lock no longer matches pyproject.toml;
    --frozen installs the stale lock without checking it.
    """
    syncs = [step["run"] for step in _ci_steps() if "uv sync" in step.get("run", "")]
    assert syncs, "control-ci.yml has no uv sync step"
    for run in syncs:
        assert "--locked" in run.split() and "--frozen" not in run.split(), run


def test_ci_token_reads_the_repo_only():
    """Every job's GITHUB_TOKEN is `contents: read` and nothing more, set at the top of the
    workflow or on the job.
    """
    ci = _ci()
    for name, job in ci["jobs"].items():
        assert job.get("permissions", ci.get("permissions")) == {"contents": "read"}, name


def test_ci_pins_the_uv_version():
    """setup-uv's `version` input names one uv release (x.y.z), not latest or a range, so a new uv
    can't change how CI resolves and installs.
    """
    setup = [
        step for step in _ci_steps() if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert len(setup) == 1, setup
    version = (setup[0].get("with") or {}).get("version")
    assert isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version), version


# ---------------------------------------------------------------------------------------------
# guard_tests.py and protect_secrets.py (#29): case doesn't change what a file is
# ---------------------------------------------------------------------------------------------

# Test files spelt in capitals. The Mac's filesystem ignores case, so control/TESTS/golden/x.json is
# the golden number control/tests/golden/x.json. Both hooks casefold before matching, on every
# platform, so Linux CI judges these paths as the Mac does. Paths only: nothing is created.
TEST_FILES_IN_CAPITALS = [
    "control/TESTS/golden/x.json",
    "control/Tests/probes/x.py",
    "control/greenleaf_control/TEST_X.PY",
    "control/CONFTEST.PY",
]


@pytest.mark.parametrize("rel", TEST_FILES_IN_CAPITALS)
def test_guard_tests_blocks_the_builder_from_tests_in_capitals(rel):
    """A golden number, probe or test reached through capitals is still a test file: the builder
    is blocked, as it is from control/tests/golden/x.json.
    """
    payload = _file_call("Edit", ROOT / rel)
    _assert_blocked(_hook("guard_tests.py", payload, "no-tests"), "Blocked: the builder")


@pytest.mark.parametrize("rel", TEST_FILES_IN_CAPITALS)
def test_guard_tests_lets_the_test_writer_edit_tests_in_capitals(rel):
    """The same casefolded rule lets the test-writer edit them."""
    _assert_allowed(_hook("guard_tests.py", _file_call("Edit", ROOT / rel), "only-tests"))


def test_guard_tests_still_judges_app_code_in_capitals_as_app_code():
    """Casefolding makes nothing a test that wasn't one: GREENLEAF_CONTROL/ENVS.PY is envs.py, so
    the builder may edit it and the test-writer is blocked.
    """
    payload = _file_call("Edit", CONTROL / "GREENLEAF_CONTROL/ENVS.PY")
    _assert_allowed(_hook("guard_tests.py", payload, "no-tests"))
    _assert_blocked(_hook("guard_tests.py", payload, "only-tests"), "Blocked: the test-writer")


# Secret files spelt in capitals: on the Mac each name opens the real file. Paths only: nothing
# named like this is ever created, since inside the repo it would alias the real .env.
SECRET_FILES_IN_CAPITALS = [
    pytest.param(ROOT / ".ENV", id=".ENV"),
    pytest.param(ROOT / "app/.Env", id="app/.Env"),
    pytest.param(ROOT / ".ENV.LOCAL", id=".ENV.LOCAL"),
    pytest.param(Path.home() / ".MODAL.TOML", id="~/.MODAL.TOML"),
]


@pytest.mark.parametrize("tool", ["Read", "Edit", "Write"])
@pytest.mark.parametrize("path", SECRET_FILES_IN_CAPITALS)
def test_protect_secrets_blocks_a_secret_file_in_capitals(path, tool):
    """A Read of .ENV on the Mac prints the real .env: the name is casefolded before it is judged."""
    _assert_blocked(_hook("protect_secrets.py", _file_call(tool, path)))


@pytest.mark.parametrize("rel", [".ENV.EXAMPLE", "CONFIG.ENV", "control/GREENLEAF_CONTROL/ENVS.PY"])
def test_protect_secrets_lets_non_secret_files_in_capitals_through(rel):
    """Casefolded, these are .env.example, config.env and envs.py: not secrets."""
    _assert_allowed(_hook("protect_secrets.py", _file_call("Read", ROOT / rel)))


@pytest.mark.parametrize(
    "tool_input",
    [
        pytest.param({"path": str(ROOT / ".ENV")}, id="path .ENV"),
        pytest.param({"glob": ".ENV"}, id="glob .ENV"),
        pytest.param({"glob": "*.ENV"}, id="glob *.ENV"),
        pytest.param({"glob": "**/.Env*"}, id="glob **/.Env*"),
        pytest.param({"glob": ".MODAL.TOML"}, id="glob .MODAL.TOML"),
    ],
)
def test_protect_secrets_blocks_a_grep_of_a_secret_file_in_capitals(tool_input):
    """A Grep whose path names .ENV opens the real .env on the Mac: blocked. Its glob's last part is
    casefolded too, one rule for every field. ripgrep matches globs case-sensitively, so a glob in
    capitals may select nothing: an accepted false positive.
    """
    _assert_blocked(_hook("protect_secrets.py", _grep(**tool_input)))


@pytest.mark.parametrize("command", ["cat .ENV", "source app/.Env.Local", "cat ~/.MODAL.TOML"])
def test_protect_secrets_blocks_a_shell_command_on_a_secret_file_in_capitals(command):
    """The orchestrator's call for #29: a Bash command is casefolded too, so `cat .ENV` is blocked
    like `cat .env`.
    """
    _assert_blocked(_hook("protect_secrets.py", _bash(command, ROOT)))


@pytest.mark.parametrize("command", ["cat .ENV.EXAMPLE", 'python -c "LOAD_DOTENV()"'])
def test_protect_secrets_lets_other_shell_commands_in_capitals_through(command):
    _assert_allowed(_hook("protect_secrets.py", _bash(command, ROOT)))


# ---------------------------------------------------------------------------------------------
# protect_secrets.py (#29): a symlink is judged by its target and by its own name
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("secret", [".env", ".env.local", ".modal.toml"])
@pytest.mark.parametrize("tool", ["Read", "Edit", "NotebookEdit", "Grep"])
def test_protect_secrets_judges_a_symlink_by_its_target(tmp_path, tool, secret):
    """notes.txt linked to a .env (or a .modal.toml) is that file, so reading or searching it
    through the link prints the keys. The path is resolved before it is judged: blocked. The
    secret is a dummy in tmp_path.
    """
    (tmp_path / secret).write_text("FAKE=1\n")
    link = tmp_path / "notes.txt"
    link.symlink_to(tmp_path / secret)
    if tool == "Grep":
        payload = _call("Grep", ROOT, pattern="FAKE", path=str(link))
    else:
        payload = _file_call(tool, link)
    _assert_blocked(_hook("protect_secrets.py", payload))


def test_protect_secrets_judges_a_symlink_by_its_own_name_too(tmp_path):
    """A link to a normal file passes, so the block above comes from the target, not from being a
    link; a link named .env is still blocked by its own name, wherever it points.
    """
    (tmp_path / "readme.md").write_text("x\n")
    (tmp_path / "notes.txt").symlink_to(tmp_path / "readme.md")
    (tmp_path / ".env").symlink_to(tmp_path / "readme.md")
    _assert_allowed(_hook("protect_secrets.py", _file_call("Read", tmp_path / "notes.txt")))
    _assert_blocked(_hook("protect_secrets.py", _file_call("Read", tmp_path / ".env")))


# ---------------------------------------------------------------------------------------------
# protect_secrets.py (#29): a Grep of the home folder, and the brace cap's message
# ---------------------------------------------------------------------------------------------

# A Grep whose path is the home folder or a folder above it, with no type and no glob that selects
# anything (a !-glob only excludes). If Claude Code's Grep passes --hidden (unverified, so treated
# as possible), ripgrep would read ~/.modal.toml: home has no .gitignore to skip it. Each path is
# resolved against the project (CLAUDE_PROJECT_DIR, the repo root in _hook) and casefolded.
GREP_OF_HOME = [
    pytest.param({"path": str(Path.home())}, id="home"),
    pytest.param({"path": "~"}, id="~"),
    pytest.param({"path": "~/"}, id="~/"),
    pytest.param({"path": "/"}, id="/"),
    pytest.param({"path": str(Path.home().parent)}, id="home's parent"),
    pytest.param({"path": str(Path.home()).upper()}, id="home in capitals"),
    pytest.param({"path": os.path.relpath(Path.home(), ROOT)}, id="home relative to the project"),
    pytest.param({"path": str(Path.home()), "glob": "!*.py"}, id="home, a negation-only glob"),
]

# The same searches narrowed to a type or a positive glob, or a path that isn't home or above it.
GREP_NOT_OF_HOME = [
    pytest.param({"path": str(Path.home()), "glob": "*.py"}, id="home, *.py"),
    pytest.param({"path": str(Path.home()), "type": "py"}, id="home, type py"),
    pytest.param({"path": "/", "glob": "!*.md *.py"}, id="/, a negation and *.py"),
    pytest.param({"path": str(ROOT)}, id="the project"),
    pytest.param({"path": str(Path.home() / "Desktop")}, id="a folder inside home"),
    pytest.param({"path": str(Path.home())[:-1]}, id="a sibling named like home, one letter short"),
]


@pytest.mark.parametrize("tool_input", GREP_OF_HOME)
def test_protect_secrets_blocks_an_unfiltered_grep_of_home(tool_input):
    """A Grep of home, or of a folder above it, that no type or positive glob narrows could reach
    ~/.modal.toml: blocked. The hook judges the path; it never opens anything there.
    """
    _assert_blocked(_hook("protect_secrets.py", _grep(**tool_input)))


@pytest.mark.parametrize("tool_input", GREP_NOT_OF_HOME)
def test_protect_secrets_lets_a_narrowed_grep_or_one_below_home_through(tool_input):
    """A type or a positive glob that selects no secret file narrows a Grep of home; a path inside
    home, or a sibling whose name is a prefix of home's, isn't home or above it. All pass.
    """
    _assert_allowed(_hook("protect_secrets.py", _grep(**tool_input)))


def test_protect_secrets_says_the_brace_cap_block_is_deliberate():
    """A glob over the brace cap blocks by design, so its one "Blocked" line says so plainly
    rather than "protect_secrets crashed", which reads as a bug to report.
    """
    proc = _hook("protect_secrets.py", _grep(glob="{,}" * 30 + "x.py"))
    _assert_blocked(proc)
    assert "crashed" not in proc.stderr, proc.stderr


# ---------------------------------------------------------------------------------------------
# stop_tests.py (#29): a stop it can't test blocks; its worst case fits its timeout
# ---------------------------------------------------------------------------------------------


def test_stop_tests_blocks_when_control_changed_and_the_venv_is_missing(tmp_path):
    """control/ changed but control/.venv is gone, so the fast tests can't run: the stop is
    blocked (exit 2) with the hook's own stop_tests message telling Claude to run `uv sync`, then
    the fast tests ("not slow"), where it used to pass quietly; not its crash handler's. uv never
    runs. The hook knows control/ changed, so it doesn't say it can't tell. The stop that follows,
    stop_hook_active, passes, so the hook can't loop.
    """
    repo, env = _stop_repo(tmp_path)
    (repo / "control/.venv").rmdir()
    (repo / "control/greenleaf_control/a.py").write_text("A = 2\n")
    proc = _stop(repo, env)
    assert proc.returncode == 2 and proc.stderr.startswith("stop_tests"), (
        proc.returncode,
        proc.stderr,
    )
    assert "uv sync" in proc.stderr and "not slow" in proc.stderr, proc.stderr
    assert "FAKE-UV-RAN" not in proc.stderr and "crashed" not in proc.stderr, proc.stderr
    assert "can't tell whether control/ changed" not in proc.stderr, proc.stderr
    again = _stop(repo, env, stop_hook_active=True)
    assert again.returncode == 0, (again.returncode, again.stderr)


def test_stop_tests_skips_a_missing_venv_when_nothing_changed(tmp_path):
    """No .venv and no change under control/: nothing to test, so the stop passes (exit 0)."""
    repo, env = _stop_repo(tmp_path)
    (repo / "control/.venv").rmdir()
    proc = _stop(repo, env)
    assert proc.returncode == 0, (proc.returncode, proc.stderr)


def test_stop_tests_blocks_when_control_changed_and_uv_is_missing(tmp_path):
    """The orchestrator's call for #29: the "no uv at all" skip is dropped. With control/ changed
    and no uv in ~/.local/bin or on PATH, the stop is blocked (exit 2) with the hook's own
    stop_tests message naming uv, not its crash handler's. PATH holds only git, so the real uv (the
    Mac's, or CI's setup-uv) can't be found. The stop that follows, stop_hook_active, passes.
    """
    repo, env = _stop_repo(tmp_path)
    (tmp_path / "home/.local/bin/uv").unlink()
    bin_dir = tmp_path / "git-only-bin"
    bin_dir.mkdir()
    (bin_dir / "git").symlink_to(shutil.which("git"))
    env = {**env, "PATH": str(bin_dir)}
    (repo / "control/greenleaf_control/a.py").write_text("A = 2\n")
    # The git on this PATH sees the change, so the block below is about uv, not a failing git.
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "a.py" in status.stdout, (status.returncode, status.stderr)
    proc = _stop(repo, env)
    assert proc.returncode == 2 and proc.stderr.startswith("stop_tests"), (
        proc.returncode,
        proc.stderr,
    )
    # A whole word: tmp_path's folder (test_..._uv_is_missing0) doesn't count.
    assert re.search(r"\buv\b", proc.stderr) and "crashed" not in proc.stderr, proc.stderr
    again = _stop(repo, env, stop_hook_active=True)
    assert again.returncode == 0, (again.returncode, again.stderr)


def test_stop_tests_blocks_when_uv_cant_run(tmp_path):
    """A uv that exists but can't start (its #! interpreter is missing, so the exec raises OSError)
    blocks the stop (exit 2) with the hook's own stop_tests message naming uv, not its crash
    handler's, where it used to pass as if the tests had. The stop that follows, stop_hook_active,
    passes.
    """
    repo, env = _stop_repo(tmp_path)
    uv = tmp_path / "home/.local/bin/uv"
    uv.write_text("#!/nonexistent/interpreter\n")
    uv.chmod(0o755)
    (repo / "control/greenleaf_control/a.py").write_text("A = 2\n")
    proc = _stop(repo, env)
    assert proc.returncode == 2 and proc.stderr.startswith("stop_tests"), (
        proc.returncode,
        proc.stderr,
    )
    # A whole word: tmp_path's folder (test_..._when_uv_cant_run0) doesn't count.
    assert re.search(r"\buv\b", proc.stderr) and "crashed" not in proc.stderr, proc.stderr
    again = _stop(repo, env, stop_hook_active=True)
    assert again.returncode == 0, (again.returncode, again.stderr)


def _stop_hook_timeout() -> int:
    """The timeout .claude/settings.json gives the Stop hook that runs stop_tests.py, in seconds."""
    settings = json.loads((ROOT / ".claude/settings.json").read_text())
    hooks = [
        hook
        for entry in settings["hooks"]["Stop"]
        for hook in entry["hooks"]
        if "stop_tests.py" in hook.get("command", "")
    ]
    assert len(hooks) == 1, hooks
    return hooks[0]["timeout"]


def test_stop_tests_worst_case_fits_its_settings_timeout():
    """Claude Code kills a Stop hook at its settings timeout and lets the stop through, so the
    hook's slowest path must end first: both git calls time out (GIT_TIMEOUT each), then the fast
    suite does (PYTEST_TIMEOUT), plus 20 s for python and the hook's own work. #29's decision: git
    20 s, pytest 280 s, settings 360 s, so 2 * 20 + 280 + 20 = 340 s <= 360 s. Before it the steps
    alone took 20 + 20 + 280 = 320 s against 300 s.

    stop_tests.py runs at import, so it is parsed, not imported. Every timeout= in it must be one
    of the two module-level constants, so the sum covers everything the hook waits on.
    """
    tree = ast.parse((HOOKS / "stop_tests.py").read_text())
    constants = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert {"GIT_TIMEOUT", "PYTEST_TIMEOUT"} <= constants.keys(), (
        "stop_tests.py needs module-level GIT_TIMEOUT and PYTEST_TIMEOUT",
        sorted(constants),
    )
    git_timeout = ast.literal_eval(constants["GIT_TIMEOUT"])
    pytest_timeout = ast.literal_eval(constants["PYTEST_TIMEOUT"])
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    git_calls = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id == "git"]
    assert len(git_calls) == 2, [ast.unparse(c) for c in git_calls]
    timeouts = sorted(ast.unparse(k.value) for c in calls for k in c.keywords if k.arg == "timeout")
    assert timeouts == ["GIT_TIMEOUT", "PYTEST_TIMEOUT"], timeouts
    worst_case = len(git_calls) * git_timeout + pytest_timeout + 20
    assert worst_case <= _stop_hook_timeout(), (worst_case, _stop_hook_timeout())


# ---------------------------------------------------------------------------------------------
# guard_runs.py (#40): control/results/final/ opens once the ledger beside it has a final-result
# ---------------------------------------------------------------------------------------------

LEDGER = "control/ledger/ledger.jsonl"


def _row(kind: str, **fields) -> str:
    """One ledger line of kind, with the base fields the ledger CLI stamps on every row."""
    return json.dumps(
        {
            "schema_version": 1,
            "row_id": f"{kind}-0000abcd",
            "kind": kind,
            "at": "2026-11-01T00:00:00+00:00",
            "issue": 40,
            "fingerprints": {},
            **fields,
        }
    )


# A ledger up to an approved final that hasn't run yet, then the same ledger with the final's row.
UNSPENT = [
    _row("registered", hypothesis="h"),
    _row("result", registered="registered-0000abcd"),
    _row("final-registered", approved_by="Aarmen"),
]
FINAL_RESULT = _row(
    "final-result", **{"final-registered": "final-registered-0000abcd", "per_split": {}}
)
SPENT = [*UNSPENT, FINAL_RESULT]

# Ledgers that keep the seal: no final-result row, or a line that isn't a JSON object anywhere in
# the file (fail closed), or bytes that aren't text.
SEALED_LEDGERS = [
    pytest.param(None, id="no ledger"),
    pytest.param([], id="an empty ledger"),
    pytest.param(UNSPENT, id="registered, result and final-registered rows only"),
    pytest.param([*SPENT, "not json"], id="a final-result row, then a line that isn't JSON"),
    pytest.param(["not json", *SPENT], id="a line that isn't JSON, then a final-result row"),
    pytest.param(
        [*UNSPENT, json.dumps({"kind": "result", "note": "final-result"})],
        id="final-result as a value, not the kind",
    ),
    pytest.param([*UNSPENT, json.dumps("final-result")], id="a JSON string, not an object"),
    pytest.param(
        [*UNSPENT, json.dumps([json.loads(FINAL_RESULT)])], id="a final-result row inside a list"
    ),
    pytest.param(
        "".join(f"{line}\n" for line in SPENT).encode() + b"\xff\n",
        id="a final-result row, then bytes that aren't UTF-8",
    ),
]

# How Read, Grep and Glob name a checkout's sealed results: by absolute path ({checkout} is filled
# in), relative to the call's cwd (the checkout), and in mixed case past the checkout. Only
# Control/Results/Final changes case, never the tmp_path part: Linux CI's filesystem is
# case-sensitive, so a re-cased tmp_path would be another folder there.
FINAL_READS_ABSOLUTE = [
    pytest.param("Read", {"file_path": "{checkout}/control/results/final/x.json"}, id="Read"),
    pytest.param("Grep", {"pattern": "iqm", "path": "{checkout}/control/results/final"}, id="Grep"),
    pytest.param(
        "Glob", {"pattern": "*.json", "path": "{checkout}/control/results/final"}, id="Glob"
    ),
]
FINAL_READS = [
    *FINAL_READS_ABSOLUTE,
    pytest.param("Read", {"file_path": "control/results/final/x.json"}, id="Read relative"),
    pytest.param("Grep", {"pattern": "iqm", "path": "control/results/final"}, id="Grep relative"),
    pytest.param("Glob", {"pattern": "control/results/final/*.json"}, id="Glob relative, no path"),
    pytest.param(
        "Read", {"file_path": "{checkout}/Control/Results/Final/x.json"}, id="Read mixed case"
    ),
    pytest.param(
        "Grep", {"pattern": "iqm", "path": "{checkout}/Control/Results/Final"}, id="Grep mixed case"
    ),
    pytest.param(
        "Glob",
        {"pattern": "*.json", "path": "{checkout}/Control/Results/Final"},
        id="Glob mixed case",
    ),
]

# Read, Grep and Glob of the checkout's 2012 weather, which no ledger row opens.
WEATHER_READS = [
    pytest.param(
        "Read", {"file_path": "{checkout}/control/data/weather/Amsterdam/2012.csv"}, id="Read"
    ),
    pytest.param(
        "Grep",
        {"pattern": "x", "path": "{checkout}/control/data/weather/Amsterdam/2012.csv"},
        id="Grep",
    ),
    pytest.param(
        "Glob", {"pattern": "*/2012.csv", "path": "{checkout}/control/data/weather"}, id="Glob"
    ),
]


def _checkout(root: Path, ledger: list[str] | bytes | None) -> Path:
    """A fake checkout at root, inside tmp_path: control/results/final/x.json and
    control/data/weather/Amsterdam/2012.csv (dummies), and, unless ledger is None, LEDGER holding
    ledger's lines, or its bytes as they are. Never this repo, so the real seal decides nothing.

    Callers name root with capitals (Repo, A): a hook that opened the case-folded ledger path would
    miss it on Linux CI, where case matters, and keep the seal it should open.
    """
    for rel in ("control/results/final/x.json", "control/data/weather/Amsterdam/2012.csv"):
        (root / rel).parent.mkdir(parents=True)
        (root / rel).write_text("x\n")
    if ledger is not None:
        if not isinstance(ledger, bytes):
            ledger = "".join(f"{line}\n" for line in ledger).encode()
        (root / LEDGER).parent.mkdir(parents=True)
        (root / LEDGER).write_bytes(ledger)
    return root


def _in_checkout(tool: str, tool_input: dict, checkout: Path, cwd: Path | None = None) -> dict:
    """tool's call with {checkout} filled in, from cwd (the checkout itself by default)."""
    filled = {key: value.format(checkout=checkout) for key, value in tool_input.items()}
    return _call(tool, cwd or checkout, **filled)


def _assert_blocked_not_crashed(proc: subprocess.CompletedProcess) -> None:
    """Blocked by a rule. A crash blocks too, since the hook fails closed, so exit 2 alone can't
    tell a seal kept on purpose from a ledger check that raised.
    """
    _assert_blocked(proc)
    assert "crashed" not in proc.stderr, proc.stderr


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("tool", "tool_input"), FINAL_READS_ABSOLUTE)
@pytest.mark.parametrize("ledger", SEALED_LEDGERS)
def test_guard_runs_keeps_results_final_sealed_without_a_final_result_row(
    tmp_path, ledger, tool, tool_input, mode
):
    """Read, Grep and Glob of control/results/final/ stay blocked, by the hook's rule and not its
    crash handler, while the ledger beside it has no final-result row, or has one plus any line
    that isn't a JSON object (fail closed): no ledger, an empty one, registered, result and
    final-registered rows only, a line that isn't JSON before or after the row, "final-result" as
    a value rather than the kind, a JSON string or list where an object should be, and bytes that
    aren't UTF-8.
    """
    checkout = _checkout(tmp_path / "Repo", ledger)
    _assert_blocked_not_crashed(
        _hook("guard_runs.py", _in_checkout(tool, tool_input, checkout), mode)
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("tool", "tool_input"), FINAL_READS_ABSOLUTE)
def test_guard_runs_keeps_results_final_sealed_when_the_ledger_cant_be_opened(
    tmp_path, tool, tool_input, mode
):
    """A ledger.jsonl that is a folder raises an OSError when opened: still sealed, by the hook's
    rule and not its crash handler.
    """
    checkout = _checkout(tmp_path / "Repo", None)
    (checkout / LEDGER).mkdir(parents=True)
    _assert_blocked_not_crashed(
        _hook("guard_runs.py", _in_checkout(tool, tool_input, checkout), mode)
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("tool", "tool_input"), FINAL_READS)
def test_guard_runs_opens_results_final_once_the_seal_is_spent(tmp_path, tool, tool_input, mode):
    """The same Read, Grep or Glob of control/results/final/ is blocked while the checkout's
    ledger holds registered, result and final-registered rows, and passes in every mode once a
    final-result row is appended: by absolute path, relative to the cwd (the checkout), and with
    Control/Results/Final in mixed case. The block first shows the pass comes from the row, not
    from the hook missing the path.
    """
    checkout = _checkout(tmp_path / "Repo", UNSPENT)
    payload = _in_checkout(tool, tool_input, checkout)
    _assert_blocked_not_crashed(_hook("guard_runs.py", payload, mode))
    with (checkout / LEDGER).open("a") as ledger:
        ledger.write(FINAL_RESULT + "\n")
    _assert_allowed(_hook("guard_runs.py", payload, mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("tool", "tool_input"), FINAL_READS_ABSOLUTE)
def test_guard_runs_asks_the_ledger_beside_the_results(tmp_path, tool, tool_input, mode):
    """Checkout A's seal is spent. Checkout B sits inside it, as .worktrees/<n> sits inside the
    main checkout, and has no ledger. B's results/final stays blocked with the cwd in A, and A's
    opens with the cwd in B: the ledger beside the results decides, not the cwd's, not this repo's,
    and not the nearest one found by climbing above B.
    """
    spent = _checkout(tmp_path / "A", SPENT)
    sealed = _checkout(spent / ".worktrees/9", None)
    _assert_blocked_not_crashed(
        _hook("guard_runs.py", _in_checkout(tool, tool_input, sealed, cwd=spent), mode)
    )
    _assert_allowed(_hook("guard_runs.py", _in_checkout(tool, tool_input, spent, cwd=sealed), mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tool", [*WRITE_TOOLS, "Bash"])
def test_guard_runs_keeps_results_final_unwritable_after_the_seal_is_spent(tmp_path, tool, mode):
    """A spent seal opens reads only: Write, Edit, MultiEdit and NotebookEdit of
    control/results/final/x.json, and `echo x > control/results/final/x.json` from the checkout,
    stay blocked.
    """
    checkout = _checkout(tmp_path / "Repo", SPENT)
    if tool == "Bash":
        payload = _bash("echo x > control/results/final/x.json", checkout)
    else:
        payload = _file_call(tool, checkout / "control/results/final/x.json", checkout)
    _assert_blocked_not_crashed(_hook("guard_runs.py", payload, mode))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("tool", "tool_input"), WEATHER_READS)
def test_guard_runs_keeps_the_2012_weather_sealed_after_the_seal_is_spent(
    tmp_path, tool, tool_input, mode
):
    """A final-result row opens control/results/final/ only: Read, Grep and Glob of the same
    checkout's control/data/weather/Amsterdam/2012.csv stay blocked.
    """
    checkout = _checkout(tmp_path / "Repo", SPENT)
    _assert_blocked_not_crashed(
        _hook("guard_runs.py", _in_checkout(tool, tool_input, checkout), mode)
    )

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
"""

import json
import os
import re
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

SEALED_READS = [
    pytest.param(_file_call("Read", CONTROL / "results/final/x.parquet"), id="read results/final"),
    pytest.param(
        _file_call("Read", CONTROL / "data/weather/Amsterdam/2012.csv"), id="read 2012 weather"
    ),
    pytest.param(_file_call("Read", GL_GYM_2012), id="read gl_gym's 2012 weather"),
    pytest.param(
        _call("Grep", ROOT, pattern="iqm", path="control/results/final"), id="grep results/final"
    ),
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
def test_guard_runs_blocks_reads_of_the_sealed_results_and_test_weather(payload, mode):
    """Read, Grep and Glob can't reach control/results/final/ or any weather/<dir>/2012.csv,
    gl_gym's copy in the venv included.
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


def _stop(repo: Path, env: dict) -> subprocess.CompletedProcess:
    """Run the Stop hook as Claude Code does when a session in repo is about to finish."""
    payload = {
        "session_id": "abc123",
        "transcript_path": str(repo.parent / "transcript.jsonl"),
        "cwd": str(repo),
        "hook_event_name": "Stop",
        "stop_hook_active": False,
    }
    return subprocess.run(
        [sys.executable, "-S", str(HOOKS / "stop_tests.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
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

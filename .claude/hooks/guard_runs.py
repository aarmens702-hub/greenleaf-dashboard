"""PreToolUse hook that keeps every specialist off the ledger, results, runs, Modal, real training
and the sealed 2012 test weather. Each agent's .claude/agents/*.md frontmatter runs it on every tool
(matcher "*"), in its own mode:

    python3 guard_runs.py builder|test-writer|scout|reviewer

The four modes get the same rules; the mode only names the agent in the message. Blocked:

- Edit, Write, MultiEdit and NotebookEdit of control/ledger/, control/results/, control/runs/ and
  control/configs/rule_based_tuned.yml, which only the CLIs write;
- Bash that writes those paths: a > or >> redirect onto one, or one named in a command that runs
  rm, cp, mv, tee, truncate or sed -i;
- Bash that contains "modal", anywhere, in any case;
- Bash that runs greenleaf_control.train or tune_rb without --smoke, run_grid without --dry,
  evaluate on amsterdam_2012, london_2012 or abbotsford_2012 or with --final (or a prefix of it),
  or a ledger subcommand other than show;
- Read, Grep and Glob of control/results/final/ and any weather/<site>/2012.csv, gl_gym's copy in
  site-packages included.

Paths are joined to the call's cwd (a Bash `cd DIR` moves it for the commands after it), followed
through links (os.path.realpath), case-folded and matched anywhere in the result, so the main
checkout, every .worktrees/<n> and another worktree meet the same rule. Any other tool
(SubagentHandback, Task, TodoWrite) passes. It fails closed: a missing or unknown mode, stdin that
isn't a JSON object, a tool_input that isn't an object, a path or command that isn't a string, or a
crash all block. Exit code 2 blocks the call with one stderr line saying what was blocked and what
is allowed instead. Standard library only and Python 3.9 syntax, so it runs on any python3.

The hole, stated plainly: hooks see tool calls, not what code runs. A script, a test or a
`python -c` that an agent writes and runs can read any file, write the ledger, train or reach Modal.
The Bash rules read words, so variables, $(...), heredocs, shells fed on stdin, and writers not in
the list (dd, ln, rsync, perl -i, git checkout, find -delete) pass, as do `cat` of a sealed file and
a Grep from above a sealed folder. The code-level refusals in train, evaluate and ledger are the
real lock; the reviewer checks the diff.
"""

import json
import os
import re
import shlex
import sys

MODES = ("builder", "test-writer", "scout", "reviewer")
# The tool_input fields read as paths or commands: each must be a string or absent.
FIELDS = ("file_path", "notebook_path", "path", "pattern", "glob", "command")
WRITE_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")

# Searched in a real, case-folded path, so they match wherever the checkout is.
WRITE_PROTECTED = re.compile(
    r"(^|/)control/(ledger|results|runs)(/|$)|(^|/)control/configs/rule_based_tuned\.yml$"
)
SEALED = re.compile(r"(^|/)control/results/final(/|$)|(^|/)weather/[^/]+/2012\.csv$")

OPERATORS = "();<>|&\n"
WRITERS = ("rm", "cp", "mv", "tee", "truncate")
SED_IN_PLACE = re.compile(r"-[^-]*[iI]|--i")  # -i, -i.bak, -Ei, BSD's -I, --in-place, --in
CLI_MODULE = re.compile(r"greenleaf_control\.(\w+)")
CLI_SCRIPT = re.compile(r"(?:^|/)greenleaf_control/(\w+)\.py$")
TEST_SPLITS = ("amsterdam_2012", "london_2012", "abbotsford_2012")

CLI_ONLY = (
    "control/ledger/, control/results/, control/runs/ and control/configs/rule_based_tuned.yml "
    "are written by the CLIs only; reading them is fine"
)
SEALED_UNTIL_FINAL = (
    "control/results/final/ and the 2012 test weather stay sealed until the final issue; other "
    "results and other weather years are fine"
)
EXPERIMENT_ONLY = "real runs are experiments the orchestrator runs through /experiment"


def block(reason):
    sys.stderr.write("Blocked: " + " ".join(reason.split()) + "\n")
    sys.exit(2)


def norm(path, cwd):
    """path joined to cwd, ~ expanded, links followed, case-folded for matching."""
    return os.path.realpath(os.path.join(cwd, os.path.expanduser(path))).casefold()


def segments(command):
    """Each command between && || ; | & ( ) and newlines, as its de-quoted words and the files its
    > and >> redirects write. Ported from crivo's guard_reads.segments()."""
    lex = shlex.shlex(command, posix=True, punctuation_chars=OPERATORS)
    lex.whitespace = " \t\r"  # a newline ends a command, so it is an operator here
    lex.whitespace_split = True
    lex.commenters = ""  # a mid-word # is not a comment
    try:
        tokens = list(lex)
    except ValueError:  # an unclosed quote: fall back to plain words
        tokens = re.findall(r"[();<>|&\n]+|[^\s();<>|&]+", command)
        tokens = [t.replace("'", "").replace('"', "") for t in tokens]
    found, words, writes, redirect = [], [], [], None
    for token in tokens + [";"]:
        if redirect is not None:
            if ">" in redirect and not token.isdigit() and token != "-":  # not 2>&1 or >&-
                writes.append(token)
            redirect = None
        elif token and set(token) <= set(OPERATORS) and set(token) & set("<>"):
            redirect = token
        elif token and set(token) <= set(OPERATORS):
            found.append((words, writes))
            words, writes = [], []
        else:
            words.append(token)
    return found


def is_program(words, i):
    """Whether words[i] is what its command runs: the first word, or the first after python or
    uv run, past their options."""
    j = i - 1
    while j >= 0 and words[j].startswith("-"):
        j -= 1
    return j < 0 or bool(re.fullmatch(r"python[0-9.]*|run", os.path.basename(words[j])))


def cli_run(words):
    """The greenleaf_control module a command runs and the words after it, or (None, []).

    -m M, -mM and a greenleaf_control/<module>.py that is the program count; a path merely named,
    as in `git diff -- control/greenleaf_control/train.py`, doesn't.
    """
    words = [w.casefold() for w in words]
    for i, word in enumerate(words):
        if word == "-m" and i + 1 < len(words):
            match, rest = CLI_MODULE.fullmatch(words[i + 1]), words[i + 2 :]
        elif word.startswith("-m"):
            match, rest = CLI_MODULE.fullmatch(word[2:]), words[i + 1 :]
        elif is_program(words, i):
            match, rest = CLI_SCRIPT.search(word), words[i + 1 :]
        else:
            continue
        if match:
            return match.group(1), rest
    return None, []


def sealed_evaluation(word):
    """A 2012 test split, or --final or a prefix argparse expands to it (--fi, --fin=row)."""
    option = word.split("=", 1)[0]
    return any(s in word for s in TEST_SPLITS) or (len(option) > 2 and "--final".startswith(option))


def cli_reason(words):
    module, rest = cli_run(words)
    if module in ("train", "tune_rb") and "--smoke" not in rest:
        return f"greenleaf_control.{module} without --smoke — {EXPERIMENT_ONLY}; add --smoke"
    if module == "run_grid" and "--dry" not in rest:
        return f"greenleaf_control.run_grid without --dry — {EXPERIMENT_ONLY}; add --issue N --dry"
    if module == "evaluate" and any(sealed_evaluation(w) for w in rest):
        return (
            "greenleaf_control.evaluate on a 2012 test split or with --final — the test weather "
            "stays sealed until the final issue; evaluate on train_eval, validation or paper"
        )
    if module == "ledger" and rest[:1] != ["show"]:
        subcommand = rest[0] if rest else "without a subcommand"
        return (
            f"greenleaf_control.ledger {subcommand} — ledger rows are written by /experiment, "
            "by the orchestrator; ledger show is allowed"
        )
    return None


def bash_reason(command, cwd):
    if "modal" in command.casefold():
        return (
            "a command that contains modal — Modal runs only through /experiment, by the "
            "orchestrator; run it locally with --smoke or --dry"
        )
    for words, writes in segments(command):
        for target in writes:
            if WRITE_PROTECTED.search(norm(target, cwd)):
                return f"a redirect onto {target} — {CLI_ONLY}"
        names = [os.path.basename(w).casefold() for w in words]
        writer = any(n in WRITERS for n in names) or (
            "sed" in names and any(SED_IN_PLACE.match(w) for w in words)
        )
        for word in words if writer else []:
            if WRITE_PROTECTED.search(norm(word, cwd)):
                return f"rm, cp, mv, tee, truncate or sed -i on {word} — {CLI_ONLY}"
        reason = cli_reason(words)
        if reason:
            return reason
        if len(words) > 1 and words[0] == "cd":
            cwd = os.path.realpath(os.path.join(cwd, os.path.expanduser(words[-1])))
    return None


def tool_reason(tool, args, cwd):
    """Why this call is blocked, or None to let it through."""
    if tool in WRITE_TOOLS:
        for key in ("file_path", "notebook_path"):
            path = args.get(key)
            if path and WRITE_PROTECTED.search(norm(path, cwd)):
                return f"{tool} of {path} — {CLI_ONLY}"
    elif tool == "Read":
        path = args.get("file_path")
        if path and SEALED.search(norm(path, cwd)):
            return f"Read of {path} — {SEALED_UNTIL_FINAL}"
    elif tool in ("Grep", "Glob"):
        base = args.get("path") or ""
        extra = (args.get("glob") if tool == "Grep" else args.get("pattern")) or ""
        searched = [norm(base, cwd)] + ([norm(os.path.join(base, extra), cwd)] if extra else [])
        if any(SEALED.search(p) for p in searched):
            return f"{tool} of {os.path.join(base or cwd, extra)} — {SEALED_UNTIL_FINAL}"
    elif tool == "Bash":
        return bash_reason(args.get("command") or "", cwd)
    return None


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in MODES:
        return (
            f"guard_runs needs a mode ({', '.join(MODES)}) and got {mode or 'none'!r}, so it "
            "fails closed — wire it as guard_runs.py <mode> in the agent's frontmatter"
        )
    try:
        data = json.load(sys.stdin)
    except ValueError:
        data = None
    tool = data.get("tool_name") if isinstance(data, dict) else None
    args = data.get("tool_input") if isinstance(data, dict) else None
    if (
        not isinstance(tool, str)
        or not isinstance(args, dict)
        or any(not isinstance(args.get(k), (str, type(None))) for k in FIELDS)
    ):
        return (
            f"guard_runs ({mode}): a tool call it can't read, so it fails closed — it needs a JSON "
            "object with a string tool_name and a tool_input object whose paths and command are "
            "strings"
        )
    cwd = data.get("cwd")
    cwd = os.path.realpath(cwd if isinstance(cwd, str) and cwd else os.getcwd())
    reason = tool_reason(tool, args, cwd)
    return f"guard_runs ({mode}): {reason}" if reason else None


try:
    reason = main()
except Exception as exc:  # noqa: BLE001 - fail closed: a crash must block the call, not pass it
    reason = f"guard_runs crashed ({type(exc).__name__}: {exc}), so it fails closed — report it"
if reason:
    block(reason)
sys.exit(0)

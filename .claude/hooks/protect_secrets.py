"""PreToolUse hook: keep API keys and cloud tokens out of every agent's
context. Blocks reading, searching, editing or shelling out to any .env file
(at any depth) except .env.example, which only lists variable names, and to
Modal's token file (.modal.toml). Permission deny rules don't cover grep or
subprocesses; this hook does. Exit code 2 blocks the tool call and shows
Claude the reason in one stderr line. It fails closed: stdin that isn't a
JSON object, a tool_name that isn't a string, a tool_input that isn't an
object, a path, command, glob or type that isn't a string (an absent one is
fine), or a crash all block. Standard library only. Ported from crivo.

Names, paths and commands are casefolded before they are matched, on every
platform, since the Mac's filesystem ignores case: .ENV opens .env there. A
path is judged by its own name and by its target's, once resolved (~
expanded, against CLAUDE_PROJECT_DIR or the cwd, through symlinks), so a link
to a .env blocks too. The hook only lstats paths, never opens them.

A Grep picks its files by glob and type as well as path, so a glob or a
ripgrep type that could select a secret file blocks too. Its {a,b} groups are
expanded first, then it is split on spaces and commas; a !-glob only excludes.
Each glob's last path part is matched against the secret file names, since
ripgrep's -g reaches hidden and gitignored files in any folder. Broad filters
such as *, <dir>/*, .*, *env* or *.toml (and type toml or all) select a
secret file too, so they block: an accepted false positive. The glob is
casefolded as well, though ripgrep matches globs case-sensitively, so a glob
like .ENV blocks while it may select nothing: the same accepted false
positive. A glob with more than MAX_SPELLINGS {a,b} spellings blocks rather
than outrun the timeout.

A Grep whose path is the home folder or a folder above it ("/" included)
blocks unless a type or a glob that selects something (not only !-globs)
narrows it: if Grep passes --hidden, ripgrep would read ~/.modal.toml, since
home has no .gitignore. A Grep with no path is left alone.
"""

import fnmatch
import json
import os
import re
import sys

# The tool_input fields read as paths, commands or Grep filters: each must be a string or absent.
FIELDS = ("file_path", "path", "notebook_path", "command", "glob", "type")

# Secret file names a Grep glob's last path part is tried on.
SECRET_NAMES = (".env", ".env.x", ".env.local", ".env.production.local", ".modal.toml")
# More {a,b} spellings than this and the glob blocks, rather than run past the hook's timeout.
MAX_SPELLINGS = 1000
# The ripgrep types that can select a secret file (rg --type-list): toml is *.toml, and all is
# every type. No type covers .env.
TYPE_GLOBS = {"toml": "*.toml", "all": "*"}


def block(reason):
    sys.stderr.write("Blocked: " + " ".join(reason.split()) + "\n")
    sys.exit(2)


def is_secret_file(path) -> bool:
    name = os.path.basename(str(path or "")).casefold()
    if name == ".modal.toml":
        return True
    return bool(re.fullmatch(r"\.env(\..+)?", name)) and name != ".env.example"


def norm(path, cwd):
    """path joined to cwd, ~ expanded, links followed, case-folded for matching."""
    return os.path.realpath(os.path.join(cwd, os.path.expanduser(path))).casefold()


# A .env path inside a shell command: ".env", "app/.env", ".env.local", but
# not ".env.example" and not a name that merely contains "env" (load_dotenv).
SECRET_IN_COMMAND = re.compile(
    r"""(^|[\s'"=/\\<>|;&(])\.env(?!\.example\b)(\.[A-Za-z0-9_.-]+)?(?=$|[\s'";|&)>])"""
)
MODAL_TOKEN_IN_COMMAND = re.compile(r"\.modal\.toml")


def expand_braces(glob):
    """Every spelling of glob with its {a,b} groups expanded, innermost first."""
    done, todo = [], [glob]
    while todo:
        if len(done) + len(todo) > MAX_SPELLINGS:
            raise ValueError(f"a glob with over {MAX_SPELLINGS} {{a,b}} spellings")
        text = todo.pop()
        group = re.search(r"\{([^{}]*)\}", text)
        if group is None:
            done.append(text)
        else:
            head, tail = text[: group.start()], text[group.end() :]
            todo.extend(head + alt + tail for alt in group.group(1).split(","))
    return done


def patterns(glob):
    """The patterns of a Grep glob (ripgrep syntax) that select files: its {a,b} groups expanded,
    then split on spaces and commas, leaving out the !-globs, which only exclude."""
    return [
        pattern
        for text in expand_braces(glob or "")
        for pattern in re.split(r"[\s,]+", text)
        if pattern and not pattern.startswith("!")
    ]


def selects_secret(glob) -> bool:
    """True if a Grep glob (ripgrep syntax) could select a secret file."""
    for pattern in patterns(glob):
        last = pattern.replace("[^", "[!").replace("\\", "").rsplit("/", 1)[-1].casefold()
        if is_secret_file(last) or any(fnmatch.fnmatchcase(n, last) for n in SECRET_NAMES):
            return True
    return False


def main():
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
            "protect_secrets: a tool call it can't read, so it fails closed — it needs a JSON "
            "object with a string tool_name and a tool_input object whose paths and command are "
            "strings"
        )

    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    if tool == "Bash":
        command = (args.get("command") or "").casefold()
        blocked = bool(SECRET_IN_COMMAND.search(command) or MODAL_TOKEN_IN_COMMAND.search(command))
    else:
        paths = [args[k] for k in ("file_path", "path", "notebook_path") if args.get(k)]
        blocked = any(is_secret_file(p) or is_secret_file(norm(p, root)) for p in paths)
        if tool == "Grep" and not blocked:
            glob, kind, where = args.get("glob"), args.get("type"), args.get("path")
            try:
                blocked = selects_secret(glob) or selects_secret(TYPE_GLOBS.get(kind))
                narrowed = bool(kind or patterns(glob))
            except ValueError:
                return (
                    f"protect_secrets: a Grep glob with over {MAX_SPELLINGS} {{a,b}} spellings is "
                    "too many to check in time, so it blocks; narrow the glob"
                )
            if where and not blocked and not narrowed:
                where, home = norm(where, root), norm("~", root)
                if os.path.commonpath([where, home]) == where:
                    return (
                        "protect_secrets: a Grep of the home folder, or a folder above it, that no "
                        "type or glob narrows could read ~/.modal.toml; add a glob or a type, or "
                        "search a folder inside home"
                    )

    if blocked:
        return (
            ".env files and ~/.modal.toml hold keys and stay out of agent "
            "context. Use .env.example for variable names, and ask the user to "
            "edit .env (or run `modal setup`) themselves."
        )
    return None


try:
    reason = main()
except Exception as exc:  # noqa: BLE001 - fail closed: a crash must block the call, not pass it
    reason = (
        f"protect_secrets crashed ({type(exc).__name__}: {exc}), so it fails closed — report it"
    )
if reason:
    block(reason)
sys.exit(0)

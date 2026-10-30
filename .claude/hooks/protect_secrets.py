"""PreToolUse hook: keep API keys and cloud tokens out of every agent's
context. Blocks reading, searching, editing or shelling out to any .env file
(at any depth) except .env.example, which only lists variable names, and to
Modal's token file (.modal.toml). Permission deny rules don't cover grep or
subprocesses; this hook does. Exit code 2 blocks the tool call and shows
Claude the reason in one stderr line. It fails closed: stdin that isn't a
JSON object, a tool_name that isn't a string, a tool_input that isn't an
object, a path, command, glob or type that isn't a string (an absent one is
fine), or a crash all block. Standard library only. Ported from crivo.

A Grep picks its files by glob and type as well as path, so a glob or a
ripgrep type that could select a secret file blocks too. Its {a,b} groups are
expanded first, then it is split on spaces and commas; a !-glob only excludes.
Each glob's last path part is matched against the secret file names, since
ripgrep's -g reaches hidden and gitignored files in any folder. Broad filters
such as *, <dir>/*, .*, *env* or *.toml (and type toml or all) select a
secret file too, so they block: an accepted false positive. A glob with more
than MAX_SPELLINGS {a,b} spellings blocks rather than outrun the timeout.
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
    name = os.path.basename(str(path or ""))
    if name == ".modal.toml":
        return True
    return bool(re.fullmatch(r"\.env(\..+)?", name)) and name != ".env.example"


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


def selects_secret(glob) -> bool:
    """True if a Grep glob (ripgrep syntax) could select a secret file."""
    for text in expand_braces(glob or ""):
        for pattern in re.split(r"[\s,]+", text):
            if not pattern or pattern.startswith("!"):
                continue
            last = pattern.replace("[^", "[!").replace("\\", "").rsplit("/", 1)[-1]
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

    if tool == "Bash":
        command = args.get("command") or ""
        blocked = bool(SECRET_IN_COMMAND.search(command) or MODAL_TOKEN_IN_COMMAND.search(command))
    else:
        blocked = any(is_secret_file(args.get(k)) for k in ("file_path", "path", "notebook_path"))
        if tool == "Grep":
            blocked = (
                blocked
                or selects_secret(args.get("glob"))
                or selects_secret(TYPE_GLOBS.get(args.get("type")))
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

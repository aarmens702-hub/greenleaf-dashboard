"""PreToolUse hook: keep API keys and cloud tokens out of every agent's
context. Blocks reading, searching, editing or shelling out to any .env file
(at any depth) except .env.example, which only lists variable names, and to
Modal's token file (.modal.toml). Permission deny rules don't cover grep or
subprocesses; this hook does. Exit code 2 blocks the tool call and shows
Claude the reason in one stderr line. It fails closed: stdin that isn't a
JSON object, a tool_name that isn't a string, a tool_input that isn't an
object, a path or command that isn't a string (an absent one is fine), or a
crash all block. Standard library only. Ported from crivo.
"""

import json
import os
import re
import sys

# The tool_input fields read as paths or commands: each must be a string or absent.
FIELDS = ("file_path", "path", "notebook_path", "command")


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

"""PreToolUse hook: keep API keys and cloud tokens out of every agent's
context. Blocks reading, searching, editing or shelling out to any .env file
(at any depth) except .env.example, which only lists variable names, and to
Modal's token file (.modal.toml). Permission deny rules don't cover grep or
subprocesses; this hook does. Exit code 2 blocks the tool call and shows
Claude the reason. Standard library only. Ported from crivo.
"""

import json
import os
import re
import sys

try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
tool = data.get("tool_name", "")
args = data.get("tool_input") or {}


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

if tool == "Bash":
    command = str(args.get("command", ""))
    blocked = bool(SECRET_IN_COMMAND.search(command) or MODAL_TOKEN_IN_COMMAND.search(command))
else:
    blocked = any(
        is_secret_file(args.get(k)) for k in ("file_path", "path", "notebook_path")
    )

if blocked:
    sys.stderr.write(
        "Blocked: .env files and ~/.modal.toml hold keys and stay out of agent "
        "context. Use .env.example for variable names, and ask the user to "
        "edit .env (or run `modal setup`) themselves.\n"
    )
    sys.exit(2)
sys.exit(0)

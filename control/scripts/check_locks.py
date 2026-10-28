"""Prove the ledger's locks from git history: exit 0 when clean, 1 with one line per violation.

It proves that every commit only appended to control/ledger/ledger.jsonl, that every result
names an earlier registration with the same fingerprints, that every run has a run-end, and that
the seal was spent at most once, after its final-registered row. /ship and /promote run it.
It walks the whole history, so it needs a full clone and refuses a shallow one.

Standalone on purpose: it reads the ledger through git and never imports greenleaf_control, so
the lock isn't defined by the writer it polices.
"""

import argparse
import json
import subprocess
from pathlib import Path

LEDGER = "control/ledger/ledger.jsonl"  # a repo-relative pathspec


def _git(top: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(top), *args], check=True, capture_output=True).stdout


def _blob(top: Path, rev: str) -> bytes | None:
    """The ledger's bytes at rev, or None where it doesn't exist; a git error raises."""
    if not _git(top, "ls-tree", "--name-only", rev, "--", LEDGER).strip():
        return None
    return _git(top, "cat-file", "blob", f"{rev}:{LEDGER}")


def _first_difference(before: bytes, after: bytes) -> int:
    """The 1-based number of the first line of before that after doesn't keep."""
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    for n, (a, b) in enumerate(zip(old, new, strict=False), 1):
        if a != b:
            return n
    return min(len(old), len(new)) + 1


def check_history(top: Path) -> tuple[list[str], int]:
    """Each commit touching the ledger keeps its first parent's version as a byte prefix.

    Returns the violations and the number of commits that touched the ledger.
    """
    commits = _git(top, "rev-list", "--reverse", "--full-history", "HEAD", "--", LEDGER)
    commits = commits.decode().split()
    violations = []
    for c in commits:
        parents = _git(top, "rev-list", "--parents", "-n", "1", c).decode().split()
        before = _blob(top, parents[1]) if len(parents) > 1 else None
        after = _blob(top, c)
        if (after or b"").startswith(before or b""):
            continue
        if after is None:
            violations.append(f"commit {c[:12]}: deletes {LEDGER} (append-only)")
        else:
            n = _first_difference(before, after)
            violations.append(
                f"commit {c[:12]}: edits or removes line {n} of {LEDGER} (append-only)"
            )
    return violations, len(commits)


def check_head(top: Path) -> tuple[list[str], int]:
    """The rows at HEAD link up: results to earlier registrations, runs to run-ends, one seal.

    Returns the violations and the number of rows at HEAD.
    """
    blob = _blob(top, "HEAD")
    violations, rows = [], []
    for n, line in enumerate(blob.splitlines() if blob is not None else [], 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        if isinstance(row, dict):
            rows.append(row)
        else:
            violations.append(f"line {n} of {LEDGER} at HEAD is not a JSON object")

    ended = {r["run"] for r in rows if r["kind"] == "run-end"}
    earlier = {}  # (kind, row_id) -> row, for the rows above the current one
    finals = 0
    for row in rows:
        kind, row_id = row["kind"], row["row_id"]
        if kind == "result":
            reg = earlier.get(("registered", row["registered"]))
            if reg is None:
                violations.append(
                    f"{row_id}: names registered {row['registered']}, "
                    "which is not an earlier registered row"
                )
            elif row["fingerprints"] != reg["fingerprints"]:
                mine, theirs, absent = row["fingerprints"], reg["fingerprints"], object()
                keys = sorted(
                    k
                    for k in mine.keys() | theirs.keys()
                    if mine.get(k, absent) != theirs.get(k, absent)
                )
                violations.append(
                    f"{row_id}: fingerprints differ from {reg['row_id']}'s at {','.join(keys)}"
                )
        elif kind == "run" and row_id not in ended:
            violations.append(f"{row_id}: no run-end; close it with run_end(status='crashed')")
        elif kind == "final-result":
            finals += 1
            if finals > 1:
                violations.append(f"{row_id}: a second final-result; the seal is spent")
            elif ("final-registered", row["final-registered"]) not in earlier:
                violations.append(
                    f"{row_id}: names final-registered {row['final-registered']}, "
                    "which is not an earlier final-registered row"
                )
        earlier[(kind, row_id)] = row
    return violations, len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_locks.py", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="any directory in the git repo to check (default: the repo holding this script)",
    )
    args = parser.parse_args(argv)
    top = Path(_git(args.repo, "rev-parse", "--show-toplevel").decode().strip())
    if _git(top, "rev-parse", "--is-shallow-repository").decode().strip() == "true":
        print("check_locks: shallow clone; needs the full history (git fetch --unshallow)")
        return 1

    history, commits = check_history(top)
    head, rows = check_head(top)
    for line in history + head:
        print(line)
    if history or head:
        return 1
    print(f"check_locks: clean, {rows} rows at HEAD, {commits} commits touched {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

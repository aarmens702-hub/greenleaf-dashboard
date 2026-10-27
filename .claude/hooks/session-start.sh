#!/usr/bin/env bash
# SessionStart hook: tell Claude where this checkout stands relative to dev,
# whether the test seal is intact, and what the ledger last recorded, so a
# session doesn't build on a stale branch or forget an experiment in flight.
# Output goes into context. Ported from crivo.
cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

branch=$(git branch --show-current)
echo "Git: on branch '${branch:-detached}'. Read docs/STATUS.md before starting new work."

# A linked worktree has its own git dir, apart from the shared common dir.
if [ "$(git rev-parse --path-format=absolute --git-dir 2>/dev/null)" != "$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" ]; then
  echo "Worktree: $(git rev-parse --show-toplevel)."
elif [ -n "$branch" ] && [ "$branch" != "dev" ]; then
  echo "The main checkout should stay on dev. Start work with /start-task, which makes a worktree in .worktrees/<n>."
fi

if git remote get-url origin >/dev/null 2>&1; then
  GIT_TERMINAL_PROMPT=0 git fetch --quiet origin dev 2>/dev/null
  if git rev-parse --verify --quiet origin/dev >/dev/null; then
    behind=$(git rev-list --count HEAD..origin/dev 2>/dev/null || echo 0)
    [ "$behind" -gt 0 ] && echo "Branch is $behind commit(s) behind origin/dev. Sync before starting new work."
  fi
fi

case "$branch" in
  master) echo "On master (public): switch to dev (git switch dev). master only changes through /promote." ;;
  dev) echo "On dev (the workspace): start work with /start-task <issue#>; /ship pushes it back to dev after the checks." ;;
esac

# The experiment ledger: the last rows, and whether the test seal is intact.
ledger="control/ledger/ledger.jsonl"
if [ -f "$ledger" ]; then
  n=$(wc -l < "$ledger" | tr -d ' ')
  echo "Ledger: $n row(s). Last 3:"
  tail -n 3 "$ledger" | python3 -c 'import json,sys
for line in sys.stdin:
    try: r = json.loads(line)
    except ValueError: continue
    print("  ", r.get("row_id","?"), r.get("kind","?"), "issue", r.get("issue","?"), r.get("decision", r.get("status","")))' 2>/dev/null
  if grep -q '"kind": *"final-result"' "$ledger" 2>/dev/null; then
    echo "Seal: OPENED (a final result exists). The test weather is spent; never evaluate on it again."
  else
    echo "Seal: intact. Test weather (2012) is scored only through a final issue with Aarmen's yes."
  fi
fi

# The user's open issues, so the agent knows what is in flight.
if command -v gh >/dev/null 2>&1 && git remote get-url origin >/dev/null 2>&1; then
  mine=$(gh issue list --assignee @me --state open --limit 5 --json number,title \
    --jq '.[] | "#\(.number) \(.title)"' 2>/dev/null)
  [ -n "$mine" ] && printf 'Your open issues:\n%s\n' "$mine"
fi

changes=$(git status --porcelain | wc -l | tr -d ' ')
[ "$changes" -gt 0 ] && echo "$changes uncommitted change(s) in the working tree."
exit 0

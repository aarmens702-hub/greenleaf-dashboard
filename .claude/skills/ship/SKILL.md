---
name: ship
description: Finish the current task - run checks, review, stop for Aarmen on scorer-file changes, commit, rebase onto the latest dev and push straight to dev, then close the issue. Use when the user says ship it, push this, or is done with a task.
---

# Ship the current task

`dev` is the workspace: shipping means pushing to it once the checks pass. No PR. Only `/promote` touches `master`.

1. **Branch check.** On a task branch (`aarmen/<n>-<slug>`, normally in `.worktrees/<n>`): take the issue number from its name. On `dev`: ask which issue this closes. On `master`: stop.
2. **Check.** This is the gate. From `control/` in the worktree:
   - `uv run ruff check --fix . && uv run ruff format .`, then `uv run ruff check .` must be clean.
   - `uv run pytest -q -m "not slow"`, then `uv run pytest -q -m slow` if the diff touched envs, wrappers, the Lagrangian, training or a probe. Check each exit status itself; never through `tail` or a `;` chain.
   - Once `scripts/check_locks.py` exists: `uv run python scripts/check_locks.py` must exit 0.
   - If the change is visible (a plot, a CLI table, the README), run it once and look.
   - Never push red.
3. **By type.**
   - **code, probe, analysis:** nothing more.
   - **experiment:** only what `/experiment` recorded ships to `dev`: the ledger rows, `control/results/validation/`, and any `control/configs/rule_based_tuned.yml` the experiment wrote, plus code the reviewer passed. Run directories stay out of git.
   - **final:** as experiment, plus `control/results/final/`. Confirm the ledger has exactly one `final-result` row.
4. **Review.** If `/start-task`'s reviewer already said SHIP IT for this exact diff, skip this. Otherwise run the **reviewer** subagent. On FIX, fix the items, rerun steps 2–3, review once more; if it still says FIX, stop and show the user.
5. **The scorer gate.** Run `git diff --name-only origin/dev...HEAD` plus `git diff --name-only`. If any scorer file is in it (`control/greenleaf_control/scenarios.py`, `rewards.py`, `envs.py`, `metrics.py`, `docs/PROTOCOL.md`), or a golden number under `control/tests/golden/` changed:
   - Show Aarmen the diff of those files, two or three lines on what it changes about scoring, and which recorded results stop being comparable.
   - Wait for his explicit yes in this conversation. A yes for an earlier diff doesn't count.
   - On no, stop. Leave the branch and worktree as they are.
6. **Commit.** Stage only this task's files by name, never `.env*` and never `git add -A`. Short imperative message (`feat: ...`, `fix: ...`, `exp: ...`, `docs: ...`) with a body that says why, ending with `Closes #<n>`.
7. **Rebase and push to `dev`.**
   - `git fetch origin dev && git rebase origin/dev`. On conflicts, keep both sides' intent and show the user. A conflict in `control/ledger/ledger.jsonl` keeps both sides' new lines, `dev`'s first; never edit or drop a line. If the tree changed, rerun step 2.
   - Re-read the issue (`gh issue view <n> --comments`). If it's closed or claimed by a newer session, stop.
   - `git push origin HEAD:dev`. Never force-push. If rejected because `dev` moved, fetch, rebase, rerun step 2, retry once.
8. **Tidy up.** `root=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")`; leave the worktree (ExitWorktree, or `cd "$root"`); `git -C "$root" worktree remove .worktrees/<n>`; `git -C "$root" branch -d <task branch>`. Then `gh issue close <n> --comment "Shipped to dev in <sha>"`.
9. **Report.** The commit on `dev`, the ledger rows on experiments, and that CI runs on the push (`gh run list --branch dev -L 1`); if it goes red, fixing it comes first.

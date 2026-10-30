---
name: start-task
description: Begin work on a GitHub issue as its orchestrator - claim it, branch off dev in a worktree, route it by type and size, then run the specialist subagents (scout, test-writer, builder, reviewer) its route names until it's ready for /ship. Use when the user says start/pick up/work on issue N, or asks what to work on next.
argument-hint: "[issue number] [full]"
---

# Start a task (orchestrator)

The issue is an approved design, so don't brainstorm or write a spec. You are the orchestrator: you don't write the code yourself. You run the specialists in `.claude/agents/`, pass each one the previous one's output, and keep the user posted with one line per step.

## 1. Set up
1. **Pick the issue.**
   - If no number was given, list open issues with `gh issue list --state open --label task` and `... --label bug`, and let the user choose. Leave out any with a "Claimed by" comment.
   - Suggest `bug` issues first, then the lowest-numbered `task` in the current stage of `specs/2026-10-27-the-plan.md` (`docs/STATUS.md` says which stage).
   - **The ladder gate:** no `experiment` issue starts while any Stage 1 ladder issue is open. Say so and suggest the ladder issue instead.
2. **Read it.** `gh issue view <n> --comments`. If Type, Goal, Files or Done-when is missing, or an `experiment` issue has an incomplete Measurement, or it contradicts the docs, stop: tell the user what's unclear.
3. **Claim it,** before making the worktree: if it's closed or claimed by another session, stop. Otherwise `gh issue comment <n> --body "Claimed by: <machine> session, worktree .worktrees/<n>, branch <branch>, <date>"` and `gh issue edit <n> --add-assignee @me`.
4. **Branch.**
   - Never switch branches in the main checkout. It stays on `dev`.
   - `root=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")`, then `git -C "$root" fetch origin dev && git -C "$root" worktree add "$root/.worktrees/<n>" -b aarmen/<n>-<short-slug> origin/dev`.
   - Move this session into it with the EnterWorktree tool (`path: "$root/.worktrees/<n>"`), otherwise `cd` there. Never call EnterWorktree with `name` or no arguments.
   - `uv sync` in `.worktrees/<n>/control` (it gets its own `.venv`).
   - Pass the worktree's absolute path to every specialist, and tell each to work only there.

## 2. Route it (say which path and why, in one line)

| Type | Route |
|---|---|
| `code` | Full or lean path below, by size. |
| `probe` | scout → test-writer (the probe, with its known answer) → builder (only the plumbing the probe needs) → reviewer. |
| `experiment` | builder (only if the issue needs code) → reviewer on that code → `/experiment` (you run it) → reviewer on the result → Aarmen decides. |
| `analysis` | scout → builder (the analysis script) → reviewer. Nothing trains. |
| `final` | Only with Aarmen's yes in this conversation. `/experiment final` (you run it) → reviewer → Aarmen. |
| test-only (`code` issue whose every file is under `control/tests/`) | scout → test-writer → reviewer. |

**Full path: scout → test-writer → builder → reviewer.** Use it if any of these is true:
- It touches a scorer file (`scenarios.py`, `rewards.py`, `envs.py`, `metrics.py`, `docs/PROTOCOL.md`), or carries the `core` label.
- It changes logic, a CLI, the ledger, the seal, or anything in `docs/API-CONTRACT.md`.
- It creates a file or module, or touches more than 2 files.
- It's a `bug` issue. The test-writer first reproduces it.

**Lean path: builder → reviewer,** only if all of these hold: 2 files or fewer; docs, text or comments only; no scorer file.

If the user says `full`, use the full path.

## 3. Run the specialists
Call each one the route names with the Agent tool, one at a time, and give it everything it needs, since it starts with no context:
1. **scout.** Pass the issue text and the worktree path. If its map has Blockers, stop and show them.
2. **test-writer.** Pass the issue text, the change map and the worktree path. Check that every new test fails for the right reason (on test-only: passes, and the fast suite is green). If it can't, stop and show why.
3. **builder.** Pass the issue text, the worktree path, the change map and the test report. It has 2 attempts. If it reports "Stuck" or a test it thinks is wrong, stop and show the user.
4. **`/experiment`** (experiment, final). Run the skill yourself from `.worktrees/<n>/control`. Never hand it to an agent.
5. **reviewer** (every route). Pass the issue text, the branch and the worktree path, plus on experiments the run directories and ledger rows.
   - **SHIP IT:** move on. On experiment and final, show Aarmen the result first: he decides.
   - **FIX:** send `[tests]` items to the **test-writer** first, then the rest to the builder, each **once**. Run the reviewer once more. If it still says FIX, stop and show the user both lists.
   - **Test notes** don't block.

Every subagent reply is a report: check it before passing anything on. Never fake a step. If an agent failed or was skipped, say so.

## 4. End
Tell the user what was built, which tests prove it, the ledger rows on experiments, and anything left to check by hand. If the reviewer listed scorer files, say plainly that `/ship` will stop for Aarmen's yes on the diff, and that older results stop being comparable.

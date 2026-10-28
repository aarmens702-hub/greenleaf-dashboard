@AGENTS.md

## Claude Code specifics
- **A GitHub issue is an approved design.** Working an issue counts as bounded work: skip brainstorming and spec writing, restate the plan in 3 to 5 bullets, then implement. Brainstorming is for architecture and planning only. Aarmen prefers decisions stated as defaults over rounds of questions.
- **Project skills:**
  - `/start-task`: begin an issue (the orchestrator; routes by type).
  - `/ship`: finish it. Runs the checks, stops at the scorer gate when it applies, then pushes straight to `dev`.
  - `/promote`: full checks on `dev`, numbers in the README matched to the ledger, then a `dev → master` PR.
  - `/new-task`: write an issue (task or experiment template).
  - `/experiment`: run a pre-registered experiment, or the one-time final evaluation. Orchestrator only.
- **`/start-task` routes by type** (AGENTS.md, "Work types"), then runs the specialists in `.claude/agents/`:
  - **Full path** (logic, a scorer file, the contract, the ledger or seal, bugs, more than 2 files): scout → test-writer → builder → reviewer.
  - **Lean path** (at most 2 files of docs or text, no scorer file): builder → reviewer.
  - **probe:** scout → test-writer → builder → reviewer. **experiment / final:** code first if needed, then `/experiment`, then the reviewer, then Aarmen.
- **Limits are enforced, not trusted:**
  - scout and reviewer can't edit files.
  - `guard_tests.py` keeps the test-writer inside test files (`tests/`, `test_*.py`, `conftest.py`, so golden numbers and probes too), and keeps the builder out of them.
  - `guard_runs.py` keeps every specialist off the ledger, results, runs, Modal, real training and the 2012 test weather. The code-level refusals in `train`, `evaluate` and `ledger` stay the real lock.
  - The hole, stated plainly: hooks read tool calls, not what code runs. Code an agent writes and runs can read any file. The reviewer checks the diff.
- **Other hooks:**
  - `.env` files and `~/.modal.toml` are blocked for every agent.
  - Every edited `.py` file under `control/` is formatted with ruff.
  - Claude can't finish while `control/`'s fast tests fail, and when a scorer file changed it must tell the user that older results are no longer comparable.
- **Subagents run on the strongest model.** Never downgrade an agent's model to save tokens.
- Look up library APIs (Stable-Baselines3, Gymnasium, CasADi, Optuna, rliable, Modal) with context7 instead of relying on memory.
- Every task gets its own git worktree, **inside the repo** at `.worktrees/<issue#>`; `/start-task` makes it. The main checkout stays on `dev`. `.worktrees/` is git-ignored. Temporary files go in your scratchpad or `/tmp`, never on the Desktop.

## Behaviour guidelines (kept from the hackathon)

**Think before coding.** State assumptions. If several readings exist, say so instead of picking silently. If a simpler approach exists, say so. If something is unclear, stop and name it.

**Simplicity first.** Minimum code that solves the problem: no speculative features, no abstractions for single-use code, no configurability nobody asked for, no error handling for impossible cases. If 200 lines could be 50, rewrite it.

**Surgical changes.** Touch only what the issue needs. Don't improve adjacent code, comments or formatting; match the existing style. Remove only the orphans your change created; mention other dead code instead of deleting it.

**Goal-driven execution.** Turn every task into a check that can fail: "add validation" becomes "tests for invalid inputs, then make them pass"; "fix the bug" becomes "a test that reproduces it, then make it pass". For multi-step work, state each step with its check.

**Analysis discipline.** Compute a baseline before comparing to a model, and report both. Show intervals or sample sizes with every claim. Use bootstrap CIs, not parametric ones. Never make a claim on n < 5; flag it instead.

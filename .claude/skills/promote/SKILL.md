---
name: promote
description: Promote dev to master after a quick test - run the full local checks on dev (fast and slow tests, the probe ladder, check_locks), confirm the README's numbers match the ledger, then open a dev → master PR and merge it once CI is green. Use when the user says promote, push to master, release, or "is dev good to go".
---

# Promote dev → master

`master` is what the public sees: the README, the pinned repo and the LinkedIn link all read it.

1. **Sync.** `git switch dev && git pull`. If `git log --oneline origin/master..dev` is empty, say so and stop.
2. **Checks.** From `control/`: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q` (fast and slow, the whole ladder), `uv run python scripts/check_locks.py`. Stop on any failure.
3. **Numbers match.** Every number the README or `docs/` states about results must trace to a ledger row (`uv run python -m greenleaf_control.ledger show`). A number with no row, or a row the text misreports, stops the promotion: fix the text on `dev` first.
4. **The dashboard still runs** (the hackathon app at the repo root): if anything outside `control/`, `docs/`, `specs/` or `.claude/` changed since the last promotion, start it once (`streamlit run app/streamlit_app/main.py` with `PYTHONPATH=.`) and open each tab.
5. **PR.** `gh pr create --base master --head dev --title "Promote dev → master (<date>)" --body "<what's new from git log origin/master..dev; checks; ledger rows added>"`. Wait for CI (`gh pr checks`), then `gh pr merge --merge`. Never delete `dev`. Then `git switch dev`.
6. **Report.** What reached `master`, with the PR link, or why not.

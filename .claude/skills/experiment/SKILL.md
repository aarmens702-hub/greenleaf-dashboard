---
name: experiment
description: Run a pre-registered experiment (or the one-time final evaluation) - refuse an incomplete Measurement, check the debug ladder is green, dry-run the cost, register in the ledger, launch the runs (Modal or local), record every run including crashes, select checkpoints on validation, judge against the rule as registered, and record Aarmen's decision. Use when an experiment or final issue needs running, or the user says run the experiment, train it, or record the result.
argument-hint: "<issue> [final]"
---

# Run a pre-registered experiment

Only the orchestrating session runs this, never a specialist, and always from `control/` in the task's worktree (`.worktrees/<n>/control`). The ledger, the manifests and the seal are enforced in code (`greenleaf_control.ledger`, `train`, `evaluate`); this skill is the order of operations around them.

1. **Refuse.** Stop and say why if any of these holds:
   - The tree is dirty (`git status --porcelain`). Commit first; real runs record the commit, and `train` refuses a dirty tree anyway.
   - The issue's Type (`gh issue view <n>`) is not `experiment` (or `final` with the `final` argument).
   - The Measurement lacks any of: hypothesis, arms, seeds, timesteps, scenarios, primary metric, guard metrics, decision rule, budget (CPU-hours and dollars). Ask for what's missing on the issue; never fill it in yourself.
   - The scenarios include a 2012 test scenario, and this isn't a `final` run.
   - **The ladder isn't green:** `uv run pytest -q -m slow tests/probes tests/golden` must pass on this tree. An experiment on a broken ladder measures the bug.
2. **Dry run.** `uv run python -m greenleaf_control.run_grid --issue <n> --dry` (with the arms, seeds and timesteps from the Measurement). It prints the run count, the worst-case CPU-hours, the worst-case dollars on Modal, and the wall-clock estimate from the last measured run.
   - Estimates have run ~3x low before, so show Aarmen the worst case ×3 next to the printed one.
   - Wait for his answer. Register only once he has confirmed the cap. A new cap gets a fresh `--dry`.
   - The project's running total of Modal spend (`ledger show --spend`) plus this cap must stay under $40. Over that, stop.
3. **Register.** `uv run python -m greenleaf_control.ledger register --issue <n> ...` with the Measurement's fields one for one. It stamps fingerprints (commit, scorer-file hashes, weather hashes, gl-gym commit, package versions) and prints the row id.
   - An issue gets 2 results. A third registration needs `--owner-ok`, only with Aarmen's yes in this conversation.
   - Commit `control/ledger/ledger.jsonl` by name (`exp: register <row> for #<n>`), then `uv run python scripts/check_locks.py` must exit 0.
4. **Run.** `uv run python -m greenleaf_control.run_grid --registered <row> --backend modal` (or `--backend local` if the Measurement says so). Until #19 lands, `--backend modal` refuses (remote runs can't record ledger rows, fingerprints or cost yet), so run local. E1 (rule-based tuning) doesn't go through run_grid: launch it with `uv run python -m greenleaf_control.tune_rb --registered <row> --trials 200 --jobs 1 --seed 0`.
   - Every run writes its manifest and a `run` ledger row at launch, and an end row when it finishes, crashes or hits its budget. A crashed run stays in the record; never re-launch it silently. A re-run is a new run row with a note.
   - When the runs finish: `modal volume get greenleaf-runs <row> runs/<row>`, then `uv run python -m greenleaf_control.selection --registered <row>` (validation only).
   - Check every run's diagnostics before judging: approx-KL never negative, entropy not collapsed early, solver failures 0, clip fraction sane. A run that fails one is reported as such, not dropped.
5. **Result.** `uv run python -m greenleaf_control.selection --registered <row>` computes the primary and guard metrics over every registered seed and VALIDATION scenario (IQM with 95% stratified-bootstrap CIs, `stats.iqm_ci`) from each run's selected checkpoint, and writes `control/results/validation/<row>.json`. Then `uv run python -m greenleaf_control.ledger result --registered <row> --numbers control/results/validation/<row>.json` records it. The ledger refuses if any registered arm × seed lacks a run-end, and adds each arm's crashed count and spend. Crashed runs count as failures, not as missing: they're never scored, they're listed, and they block adoption (PROTOCOL §6). Baselines (rb_default, zero_controls, random) aren't result arms: compare against `evaluate --controller <baseline> --split validation`.
   - Show Aarmen each arm's line as printed: primary IQM [CI], guards, crashed runs, solver failures, wall time, spend.
   - Commit the ledger again, then `check_locks`.
6. **Judge.** Hold the result to the rule as registered; never reword the rule after the run. Overlapping CIs are not a win. If the best setting came from a sweep, the claim waits for its fresh-seed confirmation run (a new experiment issue, 10 seeds).
7. **Hand off.** The reviewer gets the run directories, the ledger rows, and one seed's eval trajectory per arm for the reward-hacking check.
8. **Decide,** only once Aarmen says so: `uv run python -m greenleaf_control.ledger decide --result <row> --adopted|--dropped --note "<his call>, <date>"`. Commit, then `check_locks`.

## Final (the seal)
`/experiment <issue> final` follows the same steps, with these differences:
- The registration lists the exact selected checkpoints (from validation selection rows) and both rule-based controllers, and carries Aarmen's yes in this conversation (`--approved-by Aarmen`).
- `evaluate --final <row>` scores them on the three 2012 test splits, once.
- The `final-result` row consumes the seal. Every later attempt is refused, by the code and by `check_locks`.
- Report every number, including the ones that don't support the headline.

## Report
Give the registered, run, result and decision row ids, each arm's line, the spend against the cap, and how the result stands against the rule. Say plainly what crashed, what failed diagnostics, and what the reviewer was given.

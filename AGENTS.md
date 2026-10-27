# GreenLeaf

Two projects in one repo:
- **GreenLeaf 2.0** (`control/`, the active work): a reinforcement-learning co-pilot that runs a tomato greenhouse on top of the grower's own rules, built on Wageningen's GreenLight-Gym simulator.
- **The hackathon dashboard** (the repo root): the Streamlit app that placed 2nd at the RBC × BCCAI × SFU Beedie Agribusiness Analytics Hackathon (May 2026). Frozen, except its cleanup issue.

We win on **believable numbers**:
- every number we publish comes from a registered run, on recorded code, under a protocol fixed before the run;
- the test weather is opened once;
- a claim names its seeds and its interval;
- a component is proven on a problem with a known answer before it touches the greenhouse.

Everything below serves that.

## Team
- **Aarmen** (`@aarmens702-hub`): owner. Decides the plan, the architecture, the protocol, the contracts and every experiment's adopt or drop. Reviews every scorer-file change before it lands. Runs `/promote`.
- **Agents** work one GitHub issue at a time through `/start-task` and `/ship`.

If you are an agent: your job is the issue in front of you, nothing more. When the issue is unclear or seems wrong, stop and tell Aarmen. Don't guess at architecture.

## Deciding: Aarmen decides
- **Aarmen decides** code direction, architecture, `docs/API-CONTRACT.md`, `docs/PROTOCOL.md`, dependencies, the plan, the experiments and their decision rules. Say what changed in the commit.
- **Scorer files are reviewed, not delegated:** `control/greenleaf_control/scenarios.py`, `rewards.py`, `envs.py`, `metrics.py`, and `docs/PROTOCOL.md`, plus the golden numbers in `control/tests/golden/`. They decide how results are scored. An agent may draft a change on its branch, as a small diff with the tradeoffs stated, but `/ship` stops and shows Aarmen the diff. Changing one makes earlier results incomparable; say so. Issues that touch them carry the `core` label.
- Write big calls under "Key decisions" in `docs/ARCHITECTURE.md` so they aren't re-argued.

## Work types: the type sets what "done" means

| Type | Covers | Done when |
|---|---|---|
| `code` | Library code and CLIs: scenarios, rewards, envs, wrappers, metrics, the Lagrangian, the ledger, manifests, the seal, hooks | A failing test goes green (test-writer and builder split). Scorer-file changes wait for Aarmen's yes |
| `probe` | A known-answer check of one component on a toy problem (the ladder in `docs/PROTOCOL.md`) | The probe passes, seeded and deterministic, in under ~2 min on the Mac; its known answer is stated in its docstring |
| `experiment` | Anything that trains or tunes | The Measurement (hypothesis, arms, seeds, timesteps, validation scenarios, primary and guard metrics, decision rule, budget) is on the issue before the first run; `/experiment` registers, runs and records it; Aarmen adopts or drops |
| `analysis` | Stats, figures and write-ups from recorded results | Tables and figures regenerate from the ledger and runs with one command; every number in the text comes from that output; nothing trains |
| `final` | The one-time evaluation on the 2012 test weather | Checkpoints already selected on validation and recorded; Aarmen's yes in the conversation; the seal is consumed once |

- **Never claim an experimental result without its ledger row.** Grade every claim verified, partial or unverified.
- **No experiment while the ladder is red.** An experiment on a broken component measures the bug.

## Branches: dev is the workspace, master is public
- **`dev`** is the workspace.
  - Work on a local branch off `dev`, in its own worktree at `.worktrees/<n>` (`/start-task` makes both). The main checkout stays on `dev`.
  - `/ship` runs the checks, rebases onto the latest `dev`, and pushes straight to `dev`. No PR.
  - CI (`.github/workflows/control-ci.yml`) runs ruff and the fast tests on every push. `/ship`'s local checks are still the gate.
- **`master`** is what the public sees: the README and the pinned repo. Only `/promote` changes it, through a `dev → master` PR.

## Commands
Run from `control/` (or `.worktrees/<n>/control`).

| Do | Command |
|---|---|
| Install | `uv sync` |
| One test | `uv run pytest tests/test_x.py::test_name -q` |
| Fast tests | `uv run pytest -q -m "not slow"` (what the Stop hook and CI run) |
| Whole ladder | `uv run pytest -q` (fast + slow: probes and golden numbers) |
| Lint and format | `uv run ruff check --fix . && uv run ruff format .` |
| Ledger | `uv run python -m greenleaf_control.ledger show [--issue N]` |
| Locks | `uv run python scripts/check_locks.py` |
| Smoke train | `uv run python -m greenleaf_control.train --method <m> --seed 0 --smoke` |
| Real runs | Only through `/experiment`, by the orchestrator |

The hackathon dashboard (repo root): `PYTHONPATH=. .venv/bin/streamlit run app/streamlit_app/main.py`.

## Docs (read the one you need, when you need it)
- `docs/STATUS.md`: **read first when you pick the project up.** Where things stand, what's next, what we learned. Update it at the end of a session.
- `docs/PRD.md`: the research brief: the claim, the questions, non-goals, the demo story.
- `docs/ARCHITECTURE.md`: modules, data flow, who works where, Key decisions.
- `docs/PROTOCOL.md`: **the evaluation protocol** (splits, seeds, metrics, statistics, selection, budgets, the debug ladder, the seal). Read before any experiment, metric or env work. Core.
- `docs/API-CONTRACT.md`: the interfaces between modules: function signatures, info keys, manifest and ledger schemas, CLI flags. **Build against it exactly.**
- `specs/2026-10-27-the-plan.md`: the plan, stage by stage, and the only sequencing authority.
- The pitch: https://claude.ai/artifact/SpQ5Dxb5dU8Pt513s3HNJk

## Task loop (every task)
1. Start from an issue: `/start-task <number>`. No issue means no work, so write one with `/new-task`.
2. One issue, one branch off `dev` in its own worktree, one small change (about 300 lines or fewer; split bigger issues).
3. Verify: run the tests. For anything a person sees (a plot, a table), run it and look.
4. Finish with `/ship`.

## Code rules
- **Execution beats self-report.** Nothing works until pytest or an assertion says so.
- **Plain functions, CleanRL-style simplicity.** No RL framework beyond Stable-Baselines3. No Hydra, MLflow or class hierarchies. No new abstraction until something uses it twice. Configs are YAML plus dataclasses.
- **Never swallow errors.** No bare `except`, no `except: pass`, no silent fallback (ruff E722, BLE001, S110 are on). The upstream env swallows solver errors; we count them instead.
- **The upstream simulator is untrusted code.** It's pinned to one commit, and every reward term and metric we rely on has our own test with hand-computed values. Two upstream bugs found so far: the lamp penalty never fires, and the CO₂ and temperature penalty labels are swapped.
- **The grader can't be written from the graded code.** `metrics.py` computes profit and violation-hours from raw states and controls, and never imports `rewards.py`.
- **One evaluation path.** `evaluate.run_policy` rebuilds the training wrapper chain from the run's manifest and freezes VecNormalize. The rule-based controller sits below `VecNormalize`, on raw observations.
- **Don't "improve" SB3.** Hyperparameters come from `control/configs/ppo.yml`; PPO's defaults are part of the result (Engstrom et al.).
- **Data is immutable.** `control/data/weather/` and the dashboard's `data/raw/` are read-only; their hashes stamp every run.
- **Results come from code only.** `control/ledger/`, `control/results/` and `control/runs/` are written by the CLIs, never by hand.
- **Tests are keyless and offline.** No Modal, no network, no API keys in tests.

## Boundaries
**Always**
- Stay within the files the issue lists.
- Build against `docs/API-CONTRACT.md` and `docs/PROTOCOL.md`.
- Pull `dev` before starting and before shipping.

**Ask first**
- Aarmen, on anything the issue lists as out of scope, any scorer-file change, any new dependency, and any spend.

**Never**
- Train, tune, select or evaluate on the 2012 test weather outside a `final` issue.
- Edit or shell-write the ledger, results, runs or `configs/rule_based_tuned.yml`.
- Run anything on Modal from a specialist agent.
- Change an assertion, tolerance, golden number or metric to make a check pass.
- Push to `master` directly, or push to `dev` without the checks.
- Force-push or rewrite history on `dev` or `master`.
- Commit, stash, reset or push from a specialist agent. The orchestrating session owns git.
- Read, print or commit secrets: `.env` files and `~/.modal.toml` are blocked by a hook.
- Leave `dev` red.

## The hackathon dashboard (repo root, frozen)
Kept as the record of what placed 2nd. Its rules from the hackathon still hold if its cleanup issue touches it:
- Raw CSVs in `data/raw/` are read-only. Derived data lives in `data/derived/`, built only by `build_derived.py`.
- IDs: plots `P0001`, greenhouses `GH001`, farms `F01`. Dates are pandas datetimes. Currency is CAD, labelled as such.
- Known data quirks are real, not bugs: alerts are 99.5% "High Pest Pressure"; sensor data is 2025 and prices 2026+; some treatment × crop cells have n ≤ 1; strawberry is 57% of plots.
- The data is synthetic (the organisers' case). Never present a dashboard number as real-world evidence.

## Gotchas
<!-- Add one line each time an agent makes the same mistake twice. Say what to do instead. -->
- **Never mask a pytest exit code** with `;` or `| tail` in a verify chain. Check the exit status itself.
- **A pytest fixture error shows as ERROR, not FAILED.** A new failing test must fail inside its body, on its assertion.
- **`ruff check --fix` strips an unused import.** Add an import and its first use in the same edit.
- **GreenLight-Gym's info labels:** `temp_penalty` is really CO₂ and `co2_penalty` is really temperature (upstream bug). Use our `cost_co2` / `cost_temp` / `cost_rh`.

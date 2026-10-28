# Status and handoff

**Read this first when you pick the project up on any machine.** It's the shared memory: where things stand, what's next, what we learned, and how Aarmen likes to work. Update it at the end of each session.

*Last updated: 2026-10-28, Mac session.*

## Where things are
- **The repo** is `github.com/aarmens702-hub/greenleaf-dashboard`: public, default branch `master`, MIT at the root and AGPL-3.0-or-later in `control/`. It holds the hackathon dashboard that placed 2nd (RBC × BCCAI × SFU, May 2026). GreenLeaf 2.0 lives in `control/`.
- **Stage 0 (the harness in code) is done**, all on `dev` with CI green:
  - #1 `control/` uv project, licenses, CI (`86b1b52`);
  - #2 `manifest.py`, run receipts with fingerprints (`f4d4623`);
  - #3 `ledger.py`, the append-only experiment ledger (`2fa45c4`);
  - #4 `scripts/check_locks.py`, the ledger's history lock (`4c9790c`);
  - #5 `.claude/hooks/guard_runs.py`, the specialists' guard hook (`916c883`).
- **Stage 1, shipped to `dev` with CI green:**
  - #8 `rule_based.py` and upstream's configs (`d2db5cc`);
  - #7 `rewards.py` (`eda7af5`, scorer file, approved by Aarmen);
  - #6 `scenarios.py` and the weather data (`b136399`, scorer file, approved by Aarmen: 2 m wind, plain git);
  - #12 `lagrangian.py` (`6554dc0`, with the approved integral clamp in the contract).
- **Stage 1, in flight:** #9 `envs.py` (`.worktrees/9`, being built; core, so it stops for Aarmen) with decision B (seasons of exactly N steps); #10 metrics scouted ahead.
- **Decided 2026-10-28:** episode length B (5,760 steps; envs.py ends the season itself; 3104.58 gets re-measured in #11); the PID integral clamp; #6's wind height (2 m), data in plain git, and format/range tests on the fetched 2012 files.

## To do, in order
1. **Aarmen:** say yes or no to #7's diff; then #6's when it reaches the gate.
2. **Stage 1, by dependency wave:**
   - now: #6, #7 (in flight);
   - next: #9 envs (needs #6, #7, #8), #12 Lagrangian (needs #7);
   - then: #10 metrics (needs #9), then #11 golden numbers (needs #10) and #13 train/evaluate/selection (needs #9, #10, #12);
   - last: #14, #15, #16 probes, #17 stats/tune/grid/Modal, #18 plots (all need #13).
   - No experiment before all of Stage 1 is shipped.
3. **Stage 2** (E1 … E6): experiments, each pre-registered. File them with `/new-task` once Stage 1 is green. Ask GreenLeaf (D6) is built in parallel.
4. **Stage 3:** the final (once), then the analysis.
5. **Stage 4:** deploy and polish (D1–D7 in the plan). Visual roadmap: https://claude.ai/artifact/19obed2CuttzDAPZzbFDuw

**Run the harness from a Claude Code session started inside this repo** (`cd ~/Desktop/greenleaf-dashboard && claude`). Its agents, hooks and skills load only there.

**To run several issues at once,** keep the orchestrating session in the main checkout (on `dev`). Don't `EnterWorktree` or `cd` into a worktree: subagents inherit the session's worktree lock and then can't write to any other worktree. Address worktrees with `git -C <path>` and `uv --directory <path>/control run ...`.

## Open decisions for Aarmen
### Decide before the stage that needs it
1. **Ledger vs the dirty-tree check (before local training, S1-8/S1-12).** The experiment skill appends a `run` row to the tracked ledger at launch, so the next local run's `write_manifest(smoke=False)` sees ` M control/ledger/ledger.jsonl` and refuses. Options: ignore the ledger file in the dirty check; write every manifest before any ledger row; or commit between runs.
2. **casadi 3.8.2 (before #11).** The simulator's ODE solver was uploaded the same day the lock was made; the bench that gave 3104.58 didn't record its casadi version. If #11 can't reproduce 3104.58, casadi is the first suspect. Options: pin the bench's version, or a uv `exclude-newer` cooldown.
3. **Root CI breaks `master` on the first `/promote`.** `.github/workflows/ci.yml:33` runs a bare `pytest -q` from the root, which collects `control/tests/` (import-file mismatch with the root `tests/test_smoke.py`, and no `gl_gym` there). The PR path filter hides it, so the PR looks green and master goes red on merge. Fix: `pytest -q tests`. It's a dashboard file, so it needs an issue and a yes.
4. **The "modal" rule in `guard_runs` (before #17).** Any Bash command containing "modal" is blocked for specialists, including `pytest tests/test_modal_app.py`, `ruff check .../modal_app.py` and `grep modal_app`. Option: block `modal` as a program word and `--backend modal` instead.
5. **The `final` reviewer can't read `control/results/final/`** (blocked in every guard_runs mode). Decide how the final's review reads its results before the final issue.
6. **PID recovery speed (before E5).** Decided 2026-10-28: the integral is clamped to [0, `lambda_max`] (#12). That stops unbounded windup, but it does not make λ recover within a run at the defaults: costs are ≥ 0, so the integral drops by at most `ki` per update, and from the cap λ needs (`lambda_max` − `kp`)/`ki` updates to reach 0 at zero cost: 1,990 at the defaults (100, 0.5, 0.05), against ~244 updates per run. A budget of 0 sends λ to the cap on the first violation. Prompt recovery needs a smaller `lambda_max` or a larger `ki`, both per-run settings; choose them when registering E5.

### Decided 2026-10-28
- **Episode length:** B. Seasons are exactly N steps (5,760 for 60 days), as PROTOCOL §1 says; envs.py ends the season itself because upstream's env runs N+1. The paper's 3104.58 (5,761 steps) is re-measured in #11.
- **#6 at its gate:** 2 m wind for London and Abbotsford (matches upstream; the Amsterdam training files look like 10 m, about 20% windier; say so in the write-up); the weather CSVs in plain git; #6's format and range tests on the fetched 2012 files are fine.
- **PID integral clamp** to [0, `lambda_max`] (see 6 above for its limit).

### Defaults taken (veto any)
- **Manifest (#2):** `manifest.REPO_ROOT` is the git seam (the contract has no repo argument); `fingerprints(root=None)` gains the issue's optional argument (the contract text still says `fingerprints()`); fingerprint keys are repo-relative POSIX paths, a missing file is `null`; `seeds` come from `config["seed"]` (not in the contract); a dirty tree raises `RuntimeError`.
- **Fingerprint set:** the four scorer modules, `docs/PROTOCOL.md` and `configs/ppo.yml`. It excludes `tests/golden/`, `configs/rule_based_default.yml` and `configs/rule_based_tuned.yml` (which drives `rb_tuned` and both residual arms). Adding keys later changes every row's dict.
- **Ledger (#3):** row `fingerprints` = `manifest.fingerprints()` exactly (the experiment skill's step 3 list of commit, weather and versions is stale against the contract); the CLI's `result` takes a new `--numbers FILE.json` (not in the contract); `per_arm` is keyed by arm name and covers exactly the registered arms; the ledger computes `crashed` and `spend_usd`; `budget-stop` isn't a crash; a result is refused while any run of its registration is still open.
- **check_locks (#4):** standalone (doesn't import the ledger it polices); strict open-run rule (every committed `run` needs a `run-end`); exits 0 when no ledger file exists yet; CI can't run it (shallow clone), `/ship` and `/promote` do.
- **guard_runs (#5):** all four modes share one rule set; the sealed pattern is any `weather/<dir>/2012.csv`, which also catches gl_gym's site-packages copy; a crash fails closed.
- **rewards (#7):** upstream's mislabelled `temp_penalty`/`co2_penalty`/`rh_penalty` are dropped from info; `info["reward"]` is ours and upstream's is `paper_reward`; upstream's unweighted `penalty` key stays.
- **rule_based (#8):** configs are byte-identical upstream copies under a 3-line AGPL header; `load_rb_params` unwraps the `GreenLightEnv:` key, so `tune_rb` (#17) must write the tuned file in that shape.

### Follow-ups worth an issue
- `ledger.run_start` doesn't check the arm and seed against the registration, so a run under a mistyped arm name can crash without counting in its result (it still shows in `rows()` and `--spend`).
- `manifest.py`'s `weather_hashes` fallback and two `test_manifest.py` tests become dead once #6 ships; every manifest will then hash about 65 MB of weather.
- `manifest.fingerprints()` silently returns all `null` under a non-editable install (e.g. a Modal image built `--no-editable`); raise if `docs/PROTOCOL.md` is missing.
- `check_locks` doesn't compare a `final-result`'s fingerprints with its `final-registered` row's (within spec; one line).
- `ledger.register` and `check_locks` don't refuse a registration whose scenarios include a test split.
- `stop_tests.py:60` uses `git status --porcelain`, so brand-new untracked folders show as `?? dir/` and the Stop hook skips them; use `--untracked-files=all`.
- CI: `uv sync --frozen` doesn't fail on a stale lock (`--locked` would); no `permissions: contents: read`; uv version unpinned.
- The venv uses Anaconda's CPython 3.12.7; a `.python-version` or `UV_PYTHON_PREFERENCE=only-managed` would decouple it.
- `guard_tests.py` and `protect_secrets.py` fail open on unparseable input.
- `builder.md` rule 10's bullets are stricter than the hook (`run_grid` outright, `tune_rb` over 5 trials); the hook also blocks read-only `cp control/runs/x /tmp/y` and `rm -rf runs/smoke`.
- MIT copyright names only Aarmen; confirm for the hackathon team.

## What we've learned
- **GreenLight-Gym at `2d3febb`:**
  - Installs cleanly on the M3 (Python 3.12, no build step).
  - Runs ~2,000 steps/s; a 60-day season with the rule-based controller takes 2.9 s, return 3104.58.
  - The bench script is `bench.py` in the 2026-10-27 session scratchpad; S1-6 re-measures it in-repo.
  - Episodes end `terminated` after N+1 steps (see decision 1).
  - Its configs (`configs/agents/*.yml`) and its weather fetcher are not in the installed package; its bundled weather (Amsterdam 2008–2012, including the sealed 2012) is.
  - `expand_weather_data` silently loads `growth_year + 1` when a season runs past the end of its year; no registered non-test season reaches it.
  - The rule-based controller returns absolute controls; it needs an env with `normalize_actions=False`.
  - `ppo.yml` keeps strings that #13 must map: `log_std_init: np.log(1)`, `optimizer_class: adam`, `activation_fn: silu`; it also carries `n_envs` and `total_timesteps`.
- **Upstream bugs (read from source, pinned in tests):**
  - The lamp penalty always returns 0.
  - The `temp_penalty` and `co2_penalty` info labels are swapped.
  - A solver error is swallowed by a bare `except` and the episode is truncated.
- **Prior work that shaped the contribution:**
  - Plain PPO-Lagrangian on this simulator: SafeGreenRL (Jun 2026, "conservative economics").
  - Unconstrained residual on the rule-based controller: arXiv 2607.11959.
  - Simple optimizers beat RL in the Autonomous Greenhouse Challenge: iGrow, AggieAI.
  - The paper's rule-based baseline is untuned.
- **Data sources differ by city:** Amsterdam comes from GL-Gym's bundled files, London and Abbotsford from NASA POWER. TEST cities differ by data source as well as climate; say so in the write-up.
- **Compute:** CPU-bound, so Modal CPU containers (~$10 expected), capped at $40 total. GPUs don't help here.

## How Aarmen likes to work
- **Decisions stated as defaults**, not rounds of questions; he vetoes what he doesn't like.
- **Plans before changes;** casual tone; no em dashes in public-facing text.
- **He does logins, key pastes and anything outward-facing** (repo settings, publishing); merges to `master` need his explicit yes. `/ship` pushes to `dev` after the checks.
- **Small paid runs only;** check the real spend; estimates have under-counted ~3x before.
- **Wants many issues in flight at once** where the dependency graph allows.

## Set up a new laptop
1. Install uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`) and the GitHub CLI; `gh auth login`.
2. `git clone https://github.com/aarmens702-hub/greenleaf-dashboard && cd greenleaf-dashboard && git switch dev`.
3. `cd control && uv sync`.
4. `modal setup` (Aarmen only; the token stays out of agent context).

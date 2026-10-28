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
  - #12 `lagrangian.py` (`6554dc0`, with the approved integral clamp in the contract);
  - #9 `envs.py` (`ad7c769`, seasons of exactly N steps), #10 `metrics.py` (`964a194`), #11 golden numbers (`1a220cc`);
  - #13a `train.py` (`d3844cb`), #13b `evaluate.py` (`ec0f96e`, with `pyarrow<26`);
  - #16 overfit probe (`57e113a`), #14 probe ladder (`c39b482`), #18 `viz.py` (`d6955d1`).
- **Stage 1, in flight:** #13c `selection.py` (`.worktrees/13c`), #17a `stats`/`run_grid`/`modal_app` (`.worktrees/17`), #17b `tune_rb` (`.worktrees/17b`).
- **Filed 2026-10-28:** #19, recording Modal runs (ledger rows, fingerprints, cost, configs and weather in the image). Due before E2. Until it lands, `run_grid --backend modal` refuses.

## To do, in order
1. **Finish Stage 1:** #13c, #17a, #17b, then #15 (residual-offset and Lagrangian probes; needs #14's `toys.py`), #19, and a `selection` CLI (the experiment skill calls `python -m greenleaf_control.selection --registered <row>`; no contract line yet). No experiment before all of Stage 1 is shipped.
2. **Aarmen, before E1:** veto or keep the TUNE_SPACE default (Defaults taken, #17b).
3. **Stage 2** (E1 … E6): experiments, each pre-registered. File them with `/new-task` once Stage 1 is green. Ask GreenLeaf (D6) is built in parallel.
4. **Stage 3:** the final (once), then the analysis.
5. **Stage 4:** deploy and polish (D1–D7 in the plan). Visual roadmap: https://claude.ai/artifact/19obed2CuttzDAPZzbFDuw

**Run the harness from a Claude Code session started inside this repo** (`cd ~/Desktop/greenleaf-dashboard && claude`). Its agents, hooks and skills load only there.

**To run several issues at once,** keep the orchestrating session in the main checkout (on `dev`). Don't `EnterWorktree` or `cd` into a worktree: subagents inherit the session's worktree lock and then can't write to any other worktree. Address worktrees with `git -C <path>` and `uv --directory <path>/control run ...`.

## Open decisions for Aarmen
### Decide before the stage that needs it
1. **Root CI breaks `master` on the first `/promote`.** `.github/workflows/ci.yml:33` runs a bare `pytest -q` from the root, which collects `control/tests/` (import-file mismatch with the root `tests/test_smoke.py`, and no `gl_gym` there). The PR path filter hides it, so the PR looks green and master goes red on merge. Fix: `pytest -q tests`. It's a dashboard file, so it needs an issue and a yes.
2. **The probe bar vs rung 7.** AGENTS.md says a probe runs "in under ~2 min on the Mac", but #16's overfit probe needs the issue's ~200k steps: `model.learn` alone took 130 s at load 7 to 10, and the whole test 118 to 280 s depending on load. Default: accept it for rung 7 (it's `@slow`, so off the Stop hook and CI) and add "rung 7 excepted" to the bar.
3. **The `final` reviewer can't read `control/results/final/`** (blocked in every guard_runs mode). Decide how the final's review reads its results before the final issue.
4. **PID recovery speed (before E5).** Decided 2026-10-28: the integral is clamped to [0, `lambda_max`] (#12). That stops unbounded windup, but it does not make λ recover within a run at the defaults: costs are ≥ 0, so the integral drops by at most `ki` per update, and from the cap λ needs (`lambda_max` − `kp`)/`ki` updates to reach 0 at zero cost: 1,990 at the defaults (100, 0.5, 0.05), against ~244 updates per run. A budget of 0 sends λ to the cap on the first violation. Prompt recovery needs a smaller `lambda_max` or a larger `ki`, both per-run settings; choose them when registering E5.
5. **Zero-cost budgets for residual_lag (before E5).** rb_default never violates CO₂ on the TRAIN_EVAL seasons, so `budget_frac` × 0 gives a CO₂ budget of 0; the PID's error is then cost / 1e-8 and λ_co2 jumps to its cap (100) at the first CO₂ cost. If rb_tuned also has zero CO₂ cost, residual_lag is pinned at λ = 100 on CO₂. Option: a budget floor.

### Decided 2026-10-28
- **The "modal" rule in `guard_runs` stays as is.** Specialists built and reviewed #17a's `modal_app` without loosening it: they write with the Write tool, check with `ruff check .` and plain `pytest`, and read with the Read and Grep tools. No guard change.
- **Run automated (Aarmen, 2026-10-28):** scorer-file and golden-number diffs ship once the reviewer says SHIP IT, and the diff is shown to Aarmen in the summary afterward (he can revert). Paid Modal runs, the final, each experiment's adopt/drop, and merges to `master` still wait for his explicit yes.
- **Ledger vs the dirty-tree check:** the manifest's dirty check exempts exactly `control/ledger/ledger.jsonl` (append-only, guarded by `check_locks`), alongside `control/runs/`; otherwise every registered run after the first in a grid was refused (found in #13a's review; shipped with #13a).
- **Episode length:** B. Seasons are exactly N steps (5,760 for 60 days), as PROTOCOL §1 says; envs.py ends the season itself because upstream's env runs N+1. PROTOCOL §7's 3104.58 (the 2026-10-27 bench, 5,761 steps on upstream's env) is re-measured in #11.
- **#6 at its gate:** 2 m wind for London and Abbotsford (matches upstream; the Amsterdam training files look like 10 m, about 20% windier; say so in the write-up); the weather CSVs in plain git; #6's format and range tests on the fetched 2012 files are fine.
- **PID integral clamp** to [0, `lambda_max`] (see 4 above for its limit).
- **casadi 3.8.2 is fine:** on the locked environment, upstream's raw `rb_default` PAPER season returns 3104.5771 over 5,761 steps (the #9 reviewer's run), matching the original bench's 3104.58.

### Defaults taken (veto any)
- **Metrics seam (#10, shipped):** Metrics grades from raw states, but upstream's info carries only `controls`. Defaults: #9's envs puts `info["x_prev"]` and `info["x"]` (float64 copies) on every step of every chain; `trace_from_infos(infos)` drops its unused `raw_env` argument (a contract edit); `solver_failures` is counted from states (upstream leaves `x` unchanged on a failed step), so `episode_metrics` keeps its signature; PRICES, BOUNDS and upstream's normalisers (2500, 15, 15) become named constants in `metrics.py`.
- **Manifest (#2):** `manifest.REPO_ROOT` is the git seam (the contract has no repo argument); `fingerprints(root=None)` gains the issue's optional argument (the contract text still says `fingerprints()`); fingerprint keys are repo-relative POSIX paths, a missing file is `null`; `seeds` come from `config["seed"]` (not in the contract); a dirty tree raises `RuntimeError`.
- **Fingerprint set:** the four scorer modules, `docs/PROTOCOL.md` and `configs/ppo.yml`. It excludes `tests/golden/`, `configs/rule_based_default.yml` and `configs/rule_based_tuned.yml` (which drives `rb_tuned` and both residual arms). Adding keys later changes every row's dict.
- **Ledger (#3):** row `fingerprints` = `manifest.fingerprints()` exactly (the experiment skill's step 3 list of commit, weather and versions is stale against the contract); the CLI's `result` takes a new `--numbers FILE.json` (not in the contract); `per_arm` is keyed by arm name and covers exactly the registered arms; the ledger computes `crashed` and `spend_usd`; `budget-stop` isn't a crash; a result is refused while any run of its registration is still open.
- **check_locks (#4):** standalone (doesn't import the ledger it polices); strict open-run rule (every committed `run` needs a `run-end`); exits 0 when no ledger file exists yet; CI can't run it (shallow clone), `/ship` and `/promote` do.
- **guard_runs (#5):** all four modes share one rule set; the sealed pattern is any `weather/<dir>/2012.csv`, which also catches gl_gym's site-packages copy; a crash fails closed.
- **rewards (#7):** upstream's mislabelled `temp_penalty`/`co2_penalty`/`rh_penalty` are dropped from info; `info["reward"]` is ours and upstream's is `paper_reward`; upstream's unweighted `penalty` key stays.
- **train/evaluate/selection (#13), shipped in three parts (13a train, 13b evaluate, 13c selection):** `train --rb {tuned,default}` (default tuned; the resolved params are stored in the manifest, so evaluation rebuilds from them); VecNormalize normalises obs and reward with PPO's gamma (0.9631) and SB3's default clips; a mid-season solver failure keeps SB3's time-limit bootstrap and is counted; after training, one deterministic PAPER episode with the in-memory model is logged to `final_eval.json` for the reload check; the path that opens the test weather (`evaluate --final`) waits for the final (Stage 3), so #13 ships only its refusals.
- **metrics (#10):** `metrics.py` imports nothing from gl_gym at all (importing `gl_gym.environments.utils` loads `gl_gym.components.rewards` through its package `__init__`), so its unit conversions are reimplemented. `yield_kg_m2` is upstream's `fruit_growth_ffw`: the net change in standing fruit dry matter (growth minus maintenance and harvest), not harvested yield; say so in the write-up.
- **Probe ladder (#14):**
  - The toys are swapped in for `envs.make_env`, and train.py is unchanged.
  - The constant-reward probe asserts V / `normalize_reward(1)` ≈ 1 ± 0.1. VecNormalize clips that reward to 10, so V_norm = 10, and de-normalising by sqrt(var) is meaningless there.
  - The action × obs probe needs > 0.9 / < −0.9 at the two contexts. The issue's sign-only check passes untrained, because the initial policy is nearly odd-symmetric.
  - A fast negative control shows that every criterion fails untrained.
- **Overfit (#16):**
  - The gate is zero_controls + 1.0 return unit (€0.0202/m²), fixed on the issue before PPO trained.
  - The baselines are computed live, and each must be a valid 192-step season.
- **evaluate (#13b):**
  - `paper_reward_sum` is summed with `+=` like `train._final_eval`. Python 3.12's `sum()` is compensated, so the two differ in the last bits.
  - `--final` is refused until F1.
  - A baseline row's checkpoint is None.
  - `pyarrow<26`, a dependency change under run-automated.
- **viz (#18):**
  - Each function returns its Figure, built without pyplot on a white facecolor.
  - Actuator axes are padded to −0.05 to 1.05, so a control pinned at a bound stays visible.
  - Learning curves are labelled as the env reward at the run's w, which for residual_lag is profit only.
- **stats, run_grid, modal_app (#17a):**
  - `iqm_ci` writes out rliable 1.2.0's percentile stratified bootstrap on a local `RandomState(seed)`. `rliable.library` can't import here: arch 7.2.0 against pandas 3. Checked equal to rliable itself (shimmed) to 6e-17.
  - `run_grid --dry` takes `--issue --arms --seeds --timesteps`, since it runs before registering.
  - Prices come from modal.com/pricing, fetched 2026-10-28.
  - `train_remote` has hard limits `cpu=(4, 4)` and `memory=(8192, 8192)`. Modal bills max(request, usage), so the limits make the dry run's worst case a real bound.
  - `--backend modal` refuses until #19.
- **tune_rb (#17b), veto before E1:**
  - TUNE_SPACE: temp_setpoint_day (16, 24), temp_setpoint_night (14, 20), heat_deadzone (1, 10), co2_day (400, 1200), rh_max (70, 90), t_vent_off (0, 4), thScrSpDay and thScrSpNight (0, 15), thScrDeadZone (1, 8), lamps_on (0, 6), lamps_off (12, 24), lamps_off_sun (200, 600).
  - P-bands and flags stay at their defaults. Trial 0 runs at the defaults.
  - E1 is registered as arm `rb_tuned` with `{"trials": 200}`, seed 0, scenarios `train_eval`, `--jobs 1`.
- **selection (#13c):**
  - The scoring is §5 as written.
  - "Total cost" is the plain sum of the mean per-step costs. The ≤ is inclusive. A tie goes to the earliest checkpoint.
  - Invalid episodes are recorded, never dropped.
  - Seasons are 60 days.
  - `selected.json` is written once and refuses an existing file.
  - The function only, no CLI yet.
- **rule_based (#8):** configs are byte-identical upstream copies under a 3-line AGPL header; `load_rb_params` unwraps the `GreenLightEnv:` key, so `tune_rb` (#17) must write the tuned file in that shape.

### Follow-ups worth an issue
- `envs._refuse_test` truncates the start day with `int()`, but upstream uses a fractional start day as given on the fixed and sampler paths, so a 2011 scenario with start day 363.9 and a 0.9-day season would read 2012. Nothing produces fractional start days (the contract types them int). Fix: `float(s["start_day"])` plus a test.
- `evaluate --final` (#13) has to rebuild make_env's outer chain around `make_raw_env(allow_test=True)` by hand, since `make_env` has no `allow_test`; test that it can't drift from the training chain.
- `/ship` runs the slow tests only when the diff touches envs, wrappers, the Lagrangian, training or a probe; a change to `metrics.py`, `rewards.py`, `scenarios.py`, `rule_based.py` or the RB configs would ship without `tests/golden` running. Run `tests/golden` whenever a scorer file or rule_based changes.
- The golden numbers are pinned on arm64 macOS only; CI (Linux x86_64) never runs them. If they're ever run on Linux, a failure may be the platform, not a bug: re-pin per platform with Aarmen's yes, never loosen the tolerance.
- The ledger's dirty-check exemption covers any working-tree change to `control/ledger/ledger.jsonl`, and `train --registered` reads that working file, so a hand-edited registration (e.g. extra seeds) that was never committed runs with `dirty: false`; check_locks only catches it once committed. /experiment commits registrations first, so it's closed by procedure. Code fix: exempt the ledger only when HEAD's ledger is a byte prefix of the working file, and refuse a `--registered` row that isn't in HEAD's ledger.
- API-CONTRACT's train section doesn't yet state that a registered run's unlisted flags must be at their CLI defaults and its timesteps must equal the registration's (#13a enforces both).
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
- `evaluate.run_policy` is tested only on a `residual` run. A 1-day smoke over `ppo_paper` (other obs shape, normalized actions) and `residual_lag` (VecLagrangianReward in the stack) would lock it for every method.
- `.claude/skills/experiment/SKILL.md` step 4 says `run_grid --backend modal`, which refuses until #19; run local until then.
- `selection.py` isn't in `manifest.FINGERPRINTED`, so a change to it won't show in ledger rows, though it applies PROTOCOL §5.
- viz's season strip counts 15-minute steps (to 5,760); days would read better in the README.

## What we've learned
- **GreenLight-Gym at `2d3febb`:**
  - Installs cleanly on the M3 (Python 3.12, no build step).
  - Runs ~2,000 steps/s; a 60-day season with the rule-based controller takes 2.9 s, return 3104.58.
  - The bench script is `bench.py` in the 2026-10-27 session scratchpad; S1-6 re-measures it in-repo.
  - Episodes end `terminated` after N+1 steps; we end them after N (decided). Dropping the extra step moves `rb_default`'s PAPER return from 3104.5771 to 3104.2159 (one run by the #9 reviewer, no ledger row; #11 records it).
  - A season that runs past the end of its year silently reads the next year's file (`expand_weather_data`), so a 2011 season starting after about day 304 (60 days) reads the sealed 2012 file. #9's envs refuses such seasons; no registered split crosses (the latest ends on day 135.5).
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
- **The `random` baseline hits real solver failures:** in 2 of 6 TRAIN_EVAL seasons the #10 reviewer ran (2008 day 75 at step 5,128; 2011 day 75 at step 3,693), CVODES failed, so those `random` episodes are invalid (reported, never dropped). #11's golden numbers and every table must show invalid counts.
- **Metrics matches upstream:** on 21 full seasons (PAPER and TRAIN_EVAL, three baselines), `metrics.profit_eur_m2` is within 1e-13 of upstream's summed profit and violation counts are equal, though the closest values sit within 1 to 2 float32 ulps of a bound (unrecorded reviewer check). A count split on another platform is a finding: our float64 state is the truth.
- **Data sources differ by city:** Amsterdam comes from GL-Gym's bundled files, London and Abbotsford from NASA POWER. TEST cities differ by data source as well as climate; say so in the write-up.
- **Compute:** CPU-bound, so Modal CPU containers (~$10 expected), capped at $40 total. GPUs don't help here.
- **Our training plumbing learns:** the five rung 6 toys pass at seeds 0 and 1 within 12,288 steps each, at ppo.yml's lr 2e-5 (#14).
- **A 2-day PAPER season at w = 0 (#16):**
  - Doing nothing beats the rule-based controller: zero_controls 126.80 vs rb_default 86.82 return (random 67.58).
  - The lamps cost more per step than the fruit they add.
  - PPO (196,608 steps, seed 0) reaches 129.30 by shutting the costly actuators and using the free thermal screen.
  - Deterministic over 4 runs.
- **Library traps in the locked env:**
  - pyarrow 26 refuses numpy 1.x;
  - `rliable.library` breaks on pandas 3 (through arch 7.2.0);
  - Python 3.12's `sum()` over floats is compensated, so it can differ from a `+=` loop.
- **Parallel slow suites contend:** four test suites at once on the M3 (plus other load) slowed probes up to 5×. Ship one slow suite at a time.

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

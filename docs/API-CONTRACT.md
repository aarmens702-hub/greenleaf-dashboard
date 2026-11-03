# API contract (GreenLeaf 2.0, `control/`)

The interfaces between modules. Build against them exactly; a change is proposed on the issue and decided by Aarmen.
Paths are relative to `control/`. Scenario dicts are `{"location": str, "growth_year": int, "start_day": int}`.

## `greenleaf_control.scenarios` (scorer file)
- `WEATHER_DIR: Path` (`control/data/weather`)
- `TRAIN_SAMPLER_KWARGS: dict`: `{"locations": ["Amsterdam"], "growth_years": [2008, 2009, 2011], "start_days": list(range(45, 76))}`
- `TRAIN_EVAL: list[dict]` (9), `VALIDATION: list[dict]` (3), `PAPER: dict`, `TEST: dict[str, list[dict]]` with keys
  `amsterdam_2012`, `london_2012`, `abbotsford_2012` (8 each), and `TEST_2013: dict[str, list[dict]]` with keys
  `amsterdam_2013`, `london_2013`, `abbotsford_2013` (8 each). Values per `docs/PROTOCOL.md` §2.
- `SEALS: dict[int, dict[str, list[dict]]]`: `{2012: TEST, 2013: TEST_2013}`, each test year's seal (PROTOCOL §8).
- `SPLITS: dict[str, list[dict]]`: every named split above (`train_eval`, `validation`, `paper`, plus the six test
  keys).
- `is_test(scenario: dict) -> bool`: True for any scenario whose `growth_year` is a `SEALS` key (an int: 2012 or
  2013), read at call time.
- `weather_hashes() -> dict[str, str]`: `"<location>/<year>.csv"` → sha256 for every weather file.
- `require_weather(scenarios: list[dict]) -> None`: raises `FileNotFoundError` naming the missing
  `<WEATHER_DIR>/<location>/<year>.csv`.

## `greenleaf_control.rewards` (scorer file)
- `CONSTRAINTS = ("co2", "temp", "rh")`
- `class ConstraintCostReward(GreenhouseReward)`, constructed as
  `__init__(self, *, penalty_weight: float = 1.0, **greenhouse_reward_kwargs)`.
  - Reward = `scaled_profit - penalty_weight * (cost_co2 + cost_temp + cost_rh)`.
  - Its info dict has upstream's money keys (`profit`, `revenue`, `heat_cost`, `elec_cost`, `co2_cost`,
    `fruit_growth_ffw`), plus `scaled_profit`, `cost_co2`, `cost_temp`, `cost_rh`, and `paper_reward` (upstream's
    reward value).
- `constraint_costs(info: dict) -> np.ndarray` of shape (3,), in CONSTRAINTS order.

## `greenleaf_control.rule_based`
- `load_rb_params(name: str) -> dict`: `name` is one of `NAMES = ("default", "tuned", "tuned_agc2020")`
  (`configs/rule_based_<name>.yml`; `tuned_agc2020` is `tune_rb --price-regime agc2020`'s).
- `make_controller(params: dict) -> RuleBasedController`
- `rb_action(raw_env, controller) -> np.ndarray`: float32, shape (6,), clipped to [0, 1]; computed from the current
  step's state (StepContext built from `raw_env.unwrapped`).

## `greenleaf_control.envs` (scorer file)
- `METHODS = ("ppo_paper", "ppo", "residual", "residual_lag")`
- `BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned")`
- `REWARD_PRICES = {"sim": {}, "agc2020": {"fruit_price": 1.6, "heating_price": 0.03, "elec_price": 0.06,
  "co2_price": 0.08}}`: PROTOCOL §1's price regimes in upstream's reward kwarg names, written apart from
  `metrics.PRICE_REGIMES`.
- `make_raw_env(*, scenario: dict | None = None, sampler_kwargs: dict | None = None, season_length: int = 60,
  normalize_actions: bool, penalty_weight: float = 1.0, season_progress: bool = False, allow_test: bool = False,
  price_regime: str = "sim") -> gym.Env`
  - Always uses `ConstraintCostReward` and `weather_data_dir=WEATHER_DIR`.
  - `reward_kwargs` = upstream's registered ones, then `REWARD_PRICES[price_regime]` over them, then
    `penalty_weight`; `sim` overrides nothing. An unknown `price_regime` raises ValueError naming it.
  - A fixed `scenario` uses the fixed sampler; `sampler_kwargs` uses the random one.
  - Refuses a TEST scenario unless `allow_test=True` is passed. Only `evaluate --final` sets it True (plus the
    reset-only weather test). `make_env`, `run_baseline`, `train.build_vec_env`, `evaluate.run_rb` and `run_policy`
    pass it through, default False.
  - A season that would read a TEST year's file (its own year, or the next year's via upstream's year-end expansion)
    is refused unless `allow_test`, at construction (every scenario the sampler can draw) and at `reset`.
  - Ends the season after exactly N steps (decision B) and returns a flat `Box` observation (`FlattenObservation`,
    then `SeasonProgress` if asked).
  - Every step's info carries `x_prev` and `x`: float64 copies of the raw state before and after the step.
- `make_env(method: str, *, seed: int, scenario: dict | None = None, season_length: int = 60, penalty_weight: float = 1.0,
  rb_params: dict | None = None, residual_scale: float = 0.15, allow_test: bool = False, price_regime: str = "sim")
  -> gym.Env`
  - `allow_test` passes through to `make_raw_env`; only `evaluate --final` sets it. `price_regime` passes through too.
  - Returns a flat `Box` observation.
  - `ppo_paper`: PAPER, normalized rate actions, upstream observations.
  - `ppo`: TRAIN sampler, normalized rate actions, plus season progress.
  - `residual` / `residual_lag`: TRAIN sampler, `normalize_actions=False`, wrapped in `ResidualRBWrapper`, plus season
    progress. `residual_lag` forces `penalty_weight=0.0`.
  - Every method is wrapped in `SolverFailureFlag`.
- `class ResidualRBWrapper(gym.Wrapper)`, constructed as `__init__(env, controller, scale: float)`.
  - Action space `Box(-1, 1, (6,))`. Observation = `concat(flat_obs, u_rb_next)`, so 6 extra dims.
  - `step(a)`: `u = clip(u_rb + scale * a, 0, 1)`.
  - `info["u_rb"]` (the RB action this step) and `info["u_residual"]` (the unclipped `scale * a`).
- `class SeasonProgress(gym.ObservationWrapper)`: appends `timestep / N` in [0, 1].
- `class SolverFailureFlag(gym.Wrapper)`: sets `info["solver_failure"] = bool(truncated)` (nothing truncates for time;
  a season ends terminated after exactly N steps).
- `run_baseline(name: str, scenario: dict, season_length: int = 60, *, allow_test: bool = False,
  price_regime: str = "sim") -> list[dict]`: a full episode's per-step info dicts for a BASELINES entry (`random` is
  seeded by the scenario's start day). `allow_test` and `price_regime` pass through to `make_raw_env`; only
  `evaluate --final` sets `allow_test`.

## `greenleaf_control.metrics` (scorer file; never imports `rewards`)
- `episode_metrics(states: np.ndarray, controls: np.ndarray, params: np.ndarray, *, dt: float = 900.0,
  season_steps: int, prices: dict = PRICES, bounds: dict = BOUNDS) -> dict`
  - `states` has shape (T + 1, nx); `controls` has shape (T, 6).
  - `prices` and `bounds` default to the module constants `PRICES` and `BOUNDS` (`docs/PROTOCOL.md` §1).
- `PRICE_REGIMES = {"sim": PRICES, "agc2020": {"fruit": 1.6, "dmfm": 0.065, "heat": 0.03, "elec": 0.06,
  "co2": 0.08}}` (PROTOCOL §1); `PRICES` is `sim`, upstream's prices. A season at regime r is graded with
  `prices=PRICE_REGIMES[r]`.
  - Keys per `docs/PROTOCOL.md` §4, plus `steps` and `valid`.
- `trace_from_infos(infos: list[dict]) -> tuple[np.ndarray, np.ndarray]`: collects an episode's states
  (`infos[0]["x_prev"]`, then each step's `x`) and controls (each step's `controls`), both float64.

## `greenleaf_control.lagrangian`
- `class PIDLagrangian`, constructed as `__init__(self, budget: float, kp: float = 0.5, ki: float = 0.05,
  kd: float = 0.1, lambda_max: float = 100.0)`.
  - `update(mean_cost: float) -> float`. `lam` starts at 0.0.
  - The error is relative to the budget: `e = (cost - budget) / max(budget, 1e-8)`.
  - The integral is clamped to [0, lambda_max] (anti-windup); the derivative acts only on increases; λ is clamped to [0, lambda_max].
- `class FixedMultiplier`, constructed as `__init__(self, value: float = 1.0)`, for a constraint whose budget is 0.
  - `update(mean_cost: float) -> float` returns `value` whatever the cost. `lam` is `value` from construction.
- `class VecLagrangianReward(VecEnvWrapper)`
  - `lambdas: np.ndarray` (3,)
  - Each reward becomes `(r - Σλᵢcᵢ) / (1 + Σλᵢ)`.
  - `pop_mean_costs() -> np.ndarray` (3,), the mean per step since the last pop.
- `class LagrangianCallback(BaseCallback)`, constructed as `__init__(wrapper, pids, log_path: Path)`.
  - `pids` holds a `PIDLagrangian` or a `FixedMultiplier` per constraint.
  - At training start it sets `wrapper.lambdas` from the multipliers' `lam`.
  - At each rollout end it updates the PIDs, sets `wrapper.lambdas`, and logs `lag/lambda_<c>` and `lag/cost_<c>`.
- Stack order: `SubprocVecEnv → VecMonitor → VecLagrangianReward → VecNormalize`.

## `greenleaf_control.manifest`
- `code_stamp(*, smoke: bool) -> dict`: `{commit, dirty, fingerprints, weather_hashes}` of this checkout. It
  refuses a dirty tree unless `smoke`, exempting `control/runs/` and `control/ledger/ledger.jsonl`.
- `write_manifest(run_dir: Path, *, config: dict, issue: int, registered: str | None, smoke: bool,
  stamp: dict | None = None) -> dict`
  - Refuses an existing `run_dir`.
  - Without `stamp`, takes the four fields from `code_stamp(smoke=smoke)`.
  - With `stamp` (a remote run, stamped on the Mac):
    - runs no git and takes the four fields verbatim;
    - refuses, before any run dir exists, a dirty stamp unless `smoke`, and any weather file whose hash on this
      machine differs from the stamp's (the RuntimeError names the files).
  - Writes `manifest.json` with:
    - `commit`, `dirty`;
    - `fingerprints` (sha256 of the scorer files and `configs/ppo.yml`), `weather_hashes`;
    - `versions` (python, gl-gym commit, stable-baselines3, torch, gymnasium, casadi, numpy);
    - `config`, `seeds` (`model`, `env`), `host`, `started_at`, `issue`, `registered`, `smoke`.
- `finish_manifest(run_dir: Path, *, status: str, wall_s: float, timesteps: int, solver_failures: int,
  cost_usd: float | None, diagnostics: dict | None = None) -> dict`
  - `status` is one of `finished`, `crashed`, `budget-stop`.
  - A train run's `diagnostics` equal its run-end's.
- `fingerprints() -> dict`: the scorer and config hashes alone (the ledger uses the same function).

## `greenleaf_control.ledger` (the only writer of `ledger/ledger.jsonl`)
- **Every row** has `schema_version`, `row_id` (`<kind>-<8 hex>`), `kind`, `at` (UTC ISO), `issue`, `fingerprints`.
- **Kinds and their extra fields:**

  | Kind | Extra fields |
  |---|---|
  | `registered` | `hypothesis`, `arms` (list of {name, method, config}), `seeds`, `timesteps`, `scenarios` (split names), `primary`, `guards`, `rule`, `budget` ({cpu_hours, usd, backend}), `owner_ok` |
  | `run` | `registered`, `arm`, `seed`, `run_dir`, `status: "running"` |
  | `run-end` | `run`, `status`, `wall_s`, `timesteps`, `solver_failures`, `cost_usd`, `diagnostics` ({min_approx_kl, final_entropy, final_explained_variance}) |
  | `result` | `registered`, `per_arm` ({iqm, ci_low, ci_high (null when no run was scored), guards ({guard: {iqm, ci_low, ci_high}}), crashed, invalid_episodes, spend_usd (the arm's run-ends' cost_usd, null as 0)}), `selected` ({run row_id → `"<steps>"` \| `"final"`}, learning runs only) |
  | `decision` | `result`, `decision` (adopted or dropped), `note` |
  | `final-registered` | `checkpoints` ({run row_id → `"<steps>"` \| `"final"`}; each pair is exactly a value of some `result` row's `selected`; a learning arm is named by its run rows' `arm` and comes from one registration; a learning arm's name is not one of the row's `baselines`), `baselines` (a list drawn from `FINAL_BASELINES`, no duplicates, with `rb_default` and `rb_tuned`; `rb_tuned_agc2020` allowed, not required), `approved_by`, `approved_at`; optional `seal` (an int key of `SEALS`, written only when given; absent means 2012) |
  | `final-result` | `final-registered`, `per_split` ({split: {arm: {iqm, ci_low, ci_high, guards ({g: {iqm, ci_low, ci_high}}, g in F1a-3's `FINAL_GUARDS`), invalid_episodes, prob_improvement, scored}}}). Never carries `seal`: its seal is its final-registered row's. split is exactly `SEALS[<that seal>]` (`TEST_SPLITS` for 2012). A 2012 split holds its arms (the shape above); a 2013 split holds each price regime, `{split: {regime: {arm: {...}}}}`, regime exactly `PRICE_REGIMES`, each regime's entries computed on its own rows (PROTOCOL §8). In every (split) or (split, regime), arm is exactly `final_arms` plus `baselines`; iqm and ci_* are on the primary `profit_eur_m2`; prob_improvement is P(arm > rb_tuned) on the primary, null for rb_tuned; scored is {run row_id → checkpoint} as registered, `{}` for a baseline |

- **CLI** (`python -m greenleaf_control.ledger`):
  - `register ... [--dry]`
  - `result --registered <row> --numbers <file>`: `<file>` is `{per_arm, selected}` as `selection` writes it; the ledger adds `crashed` and `spend_usd` from the run-ends and refuses them in the file
  - `decide --result <row> --adopted|--dropped --note "..."`
  - `show [--issue N] [--spend]`
  - `final-register --issue N --checkpoints JSON --baselines B... --approved-by Aarmen [--seal YEAR]`: `--seal` (an
    int) passes `seal=` through the same checks; without it the row is 2012's
  - `final-result --final-registered <row> --numbers <file>`: `<file>` is exactly `{per_split}` as F1a-3's `evaluate --final` will write it; prints the row id
- **Names:** `TEST_SPLITS` and `FINAL_BASELINES` (written out; tests pin them to `tuple(scenarios.TEST)` and
  `(*envs.BASELINES, "rb_tuned_agc2020")`, evaluate's `BASELINES`), `SEALS` (`{2012: TEST_SPLITS, 2013:
  ("amsterdam_2013", "london_2013", "abbotsford_2013")}`, written out; a test pins it to `scenarios.SEALS`),
  `PRICE_REGIMES = ("sim", "agc2020")` (written out; a test pins it to `tuple(metrics.PRICE_REGIMES)`; a 2013
  split's keys in `per_split`), `FINAL_ARM_KEYS`, `final_arms(final_registered: dict, rows:
  list[dict]) -> dict[str, dict[str, str]]` (arm → {run row_id → checkpoint}), `seal_of(final_registered: dict) ->
  int` (its `seal`, or 2012), `seal_spent(rows: list[dict], seal: int) -> bool` (a `final-result` under one of the
  seal's rows), and `final_register(*, issue, checkpoints, baselines, approved_by, seal=None, path=LEDGER)`.
- **Appends only.** `register` refuses a third registration on one issue without `--owner-ok`, and any `primary`
  or `guards` entry not in `ledger.RESULT_KEYS` (checked on every append). `result` refuses
  unless every registered arm × seed has a `run-end` row. `final-register` refuses a missing `approved_by`, a `seal`
  that isn't an int key of `SEALS` (a ValueError naming seal; on every append), its seal spent, empty `checkpoints`,
  a pair no result's `selected` holds, one arm from two registrations, bad `baselines`, and a learning arm named like
  one of its baselines. `final-result` refuses a second final-result of its seal first, then a `per_split` whose
  splits (its seal's), a 2013 split's regimes (exactly `PRICE_REGIMES`: a flat 2013 split, a missing or an extra
  regime; a 2012 split nested by regime is refused too), arms or entry keys aren't the above, a `prob_improvement`
  that isn't null exactly for `rb_tuned`, and a `scored` that differs from the registration, each checked in every
  (split) or (split, regime) and named in the ValueError (guard names are checked in F1a-3 via `FINAL_GUARDS`,
  not here). Seals are independent: spending one leaves the others as they were.
  `check_locks` holds a result's and a final-result's fingerprints to their parent's, and counts final-results per
  seal (the parent's; absent 2012): an unknown seal, a `per_split` not keyed by exactly that seal's splits, or a
  second final-result of one seal is a violation.

## `greenleaf_control.train`
- `python -m greenleaf_control.train --method M --seed S [--penalty-weight W] [--budget-frac B] [--timesteps N]
  [--season-length D] [--n-envs 4] [--runs-dir runs] [--rb {default,tuned,tuned_agc2020}]
  [--price-regime {sim,agc2020}] (--smoke | --registered <row>) [--remote JSON]`
- `--smoke` caps timesteps at 20,000 and the season at 2 days, writes under `runs/smoke/`, and records no ledger row.
- Without `--smoke`, `--registered` is required, and the arm and seed must be in that registration.
- `--remote JSON` (set only by `modal_app.train_remote`) is `{"stamp": <code_stamp>, "registered": <the registered
  row>}`.
  - It's refused with `--smoke`, when the row's id isn't `--registered`, and wherever `REPO_ROOT/.git` exists (a file
    check, not a git call), so a local caller can't skip the dirty-tree check.
  - With it, train reads neither git nor the ledger: the registration check runs against that row, the manifest
    takes the stamp, and no `run` or `run-end` row is written (the Mac's run_grid records them).
  - Without it, a registered run outside a git repo refuses before any run dir.
- `--rb` picks the residual arms' RB params (`configs/rule_based_<rb>.yml`, default `tuned`); the resolved params are
  stored in the manifest as `config["rb_params"]`.
- `--price-regime` (default `sim`; another name is argparse's refusal) prices the training envs' reward and is stored
  as `config["price_regime"]`, always present. A registered run's arm that names no `price_regime` is a `sim` arm.
- **Outputs:**
  - checkpoints `ckpt_<steps>.zip` + `vecnorm_<steps>.pkl` every 250,000 steps, and `ckpt_final.zip` +
    `vecnorm_final.pkl`;
  - `tb/` (TensorBoard: SB3's train/* diagnostics plus `lag/*`);
  - `manifest.json`;
  - `lambdas.csv` (residual_lag only);
  - `final_eval.json`: `{scenario, season_length, checkpoint: "final", paper_return, steps}`, one deterministic PAPER
    season at the run's season length, played by the end-of-training model with frozen copies of its VecNormalize
    stats. Its `paper_return` is Σ `info["paper_reward"]` summed in step order with `+=` (evaluate's
    `paper_reward_sum`, a reload check), not `metrics`' `paper_return`; selection uses the metrics one.
- `build_vec_env(config: dict, n_envs: int, *, scenario: dict | None = None, season_length: int | None = None,
  allow_test: bool = False) -> VecNormalize`: `SubprocVecEnv` (`DummyVecEnv` for one env) → `VecMonitor` →
  `VecLagrangianReward` (residual_lag only) → `VecNormalize`, env i = `make_env(method, seed=seed + i, ...)` from the
  manifest's config; `scenario` and `season_length` override the config's (for evaluation). `allow_test` passes
  through to `make_env`; only `evaluate --final` sets it. The envs are built at `config.get("price_regime", "sim")`,
  so a config recorded before v3 builds `sim`.
- `build_model(config: dict, venv) -> PPO`: `configs/ppo.yml`'s PPO on cpu, seeded, with only `log_std_init` changed
  (-1.0 for the residual arms).

## `greenleaf_control.evaluate`
- `run_rb(params: dict, scenario: dict, season_length: int = 60, record_steps: bool = False,
  return_trace: bool = False, *, allow_test: bool = False, price_regime: str = "sim")
  -> tuple[dict, pd.DataFrame | None]`: plays the env at `price_regime` and grades the season at it.
- `run_policy(run_dir: Path, checkpoint: str, scenario: dict, season_length: int = 60, record_steps: bool = False,
  return_trace: bool = False, *, allow_test: bool = False) -> tuple[dict, pd.DataFrame | None]`
  - Rebuilds the training stack with `train.build_vec_env(manifest["config"], 1, scenario=..., season_length=...)`
    (`DummyVecEnv` → `VecMonitor` → `VecLagrangianReward` for residual_lag), loads
    `VecNormalize.load(run_dir/"vecnorm_<checkpoint>.pkl", <that stack>)`, sets `training = False` and
    `norm_reward = False` before reset, then `PPO.load(run_dir/"ckpt_<checkpoint>.zip", device="cpu")` and
    deterministic predict until the season ends.
  - Takes no `price_regime`: the stack is built and the season graded at the manifest's
    `config.get("price_regime", "sim")`.
- **`return_trace`** (both, default False): with True, `(result, frame, (states, controls, params))`: the same
  season's raw trace that metrics graded (no second season).
- **`allow_test`** (both): passes through, `run_rb` to `make_raw_env` and `run_policy` to `train.build_vec_env`;
  default False, and only `evaluate --final` sets it.
- **Result dict** (both): `metrics.episode_metrics(...)` graded with the env's own `p`, `season_steps = N` and
  `prices = metrics.PRICE_REGIMES[<the season's regime>]`, plus
  `paper_reward_sum` = Σ `info["paper_reward"]` in step order with `+=` (a reproduction check, not a graded metric).
- **`main(argv=None, *, ledger_path=ledger.LEDGER)`**; `ledger_path` is the ledger `--final` reads.
- **CLI, scoring:** `--controller {zero_controls|random|rb_default|rb_tuned|rb_tuned_agc2020|<run_dir>}`
  `--split {train_eval|validation|paper|<the six test splits>} --out <parquet>`, all three required (argparse's
  message) when there is no `--final`; `[--price-regime {sim,agc2020}]`.
  - `BASELINES = (*envs.BASELINES, "rb_tuned_agc2020")`, the CLI's baselines; `rb_tuned_agc2020` plays
    `load_rb_params("tuned_agc2020")`.
  - `--price-regime` (default `sim`) plays and grades a baseline at that regime. With a run dir it is refused (the
    manifest's regime stands), and so it is with `--final`.
  - The six test splits (every `scenarios.SEALS` seal's) are refused as sealed (PROTOCOL §8) without `--final`.
  - `--out` must not exist (created exclusively). One row per scenario in split order, 60-day seasons. Identity
    columns `controller, checkpoint` (None for a baseline, `final` for a run dir), `split, location, growth_year,
    start_day, season_length, price_regime` (a baseline's `--price-regime`, `sim` when unflagged; a run dir's
    manifest's `config.get("price_regime", "sim")`), then the result keys.
  - `<run_dir>` is a directory holding `manifest.json`, scored at its `final` checkpoint. `zero_controls` and
    `random` go through `envs.run_baseline`, `rb_*` through `run_rb`.
- **CLI, the final:** `--final <final-registered row>`, with no `--controller`, `--split`, `--out` or
  `--price-regime` (refused, naming the flag).
  - **Refuses, in this order**, before any env is built, any weather is read or any file is written: (1) a row that
    isn't a `final-registered` row; (2) its seal (`ledger.seal_of`) spent, a `final-result` under one of the seal's
    rows ("the seal is spent"); (3) a row that isn't its seal's newest (names the newer); (4)
    `manifest.fingerprints()` unequal to the row's (names the keys); (5) a `FINAL_RESULTS` holding anything but the
    dirs of spent seals' final-registered rows ("`<FINAL_RESULTS>` is not empty"; an unspent seal's own dir is an
    interrupted final); (6) a registered `rb_*` baseline whose `rule_based.CONFIGS / "rule_based_<name>.yml"` (read
    at call time) isn't a file (names each missing file's path); (7) a missing weather file of the seal's scenarios
    (`scenarios.require_weather`'s `FileNotFoundError`, naming the path); (8) for any run in
    `ledger.final_arms(row)`: a run dir without `manifest.json`, `ckpt_<c>.zip`, `vecnorm_<c>.pkl` or
    `selected.json`, or a `selected.json` whose `checkpoint` isn't `<c>` (names the run's row id, arm, seed and dir);
    (9) a dirty tree (`manifest.code_stamp(smoke=False)` raises). The seal's splits are `scenarios.SEALS[seal]`,
    read at call time.
  - **Scores**, for each of the seal's splits in `scenarios.SEALS[seal]` order and each scenario in split order,
    60-day seasons:
    each learning arm (by name), its runs by ascending seed, with `run_policy(run_dir, c, s, 60, allow_test=True)`;
    then each of the row's `baselines` in its order, as the scoring CLI scores it, with `allow_test=True`.
    Traced seasons, each arm's lowest registered seed and every baseline, are scored with `record_steps=True,
    return_trace=True` (`zero_controls` and `random`: the frame and trace from `run_baseline`'s infos); each
    season is played once.
  - **A 2013 final grades each play at both regimes** (PROTOCOL §8): every season is played once with
    `return_trace=True`, a run at its manifest's `config.get("price_regime", "sim")` (as `run_policy` plays it) and
    a baseline at `sim` (`rb_tuned_agc2020` plays `load_rb_params("tuned_agc2020")`), then graded at each of
    `metrics.PRICE_REGIMES` in order: at the play's regime the row is the play's own result; at the other it is
    `_regraded(result, trace, regime, season_steps)`, `metrics.episode_metrics(states, controls, params,
    season_steps=season_length × 96, prices=metrics.PRICE_REGIMES[regime])` plus the play's own `paper_reward_sum`
    (a play quantity, the reward at the play's regime, so equal in both rows).
  - **Writes `FINAL_RESULTS/<row>/` once** (`mkdir(exist_ok=False)`, each file opened exclusively): `stamp.json`
    (the code stamp) before the first season; `seasons.parquet`, one row per split × scenario × arm × run, identity
    columns `arm, run, seed, checkpoint` (null for a baseline), `split, location, growth_year, start_day,
    season_length`, then the result keys; then `per_split.json`, `{"per_split": ...}` in the ledger's `final-result`
    shape with sorted keys, indent 2 and no NaN. Per split and arm, `stats.iqm_ci` (defaults: 2,000 reps, seed 0) on
    the runs × scenarios array (a baseline is one run) gives `iqm, ci_low, ci_high` on `FINAL_PRIMARY` and `guards`
    on each of `FINAL_GUARDS`; `invalid_episodes` counts seasons with `valid == False`; `prob_improvement` is
    `stats.prob_improvement(arm, rb_tuned)` on the primary, null for `rb_tuned`; `scored` is `final_arms`' (`{}` for
    a baseline). A 2013 final's `seasons.parquet` has a `price_regime` column after `season_length` and two rows per
    season, one per regime in `metrics.PRICE_REGIMES` order (one per split × scenario × arm × run × regime), and its
    `per_split` is `{split: {regime: {arm: ...}}}`, each regime's entries computed as above on that regime's rows
    alone (`prob_improvement` against `rb_tuned` at the same regime).
  - **Then `replays/`** (after `per_split.json`, before the `next:` line): one
    `<prefix>_<location>_<growth_year>_<start_day>.json` per traced season, `<prefix>` the arm's name or the
    baseline's, each `export_replay.export(..., prices=metrics.PRICE_REGIMES[<its play's regime>])` of its scored
    season (so `totals` is its `seasons.parquet` row, and its hourly counters are graded at the same regime; a
    baseline plays at `sim`, a run at its manifest's), written exclusively, compact, no NaN, under 250 KB.
    `meta.controller` is `<prefix>`; the stamp is `{commit, dirty}` of the one `code_stamp(smoke=False)` plus
    `gl_gym`; `meta.run` is export's {id, registered, result, arm, seed, checkpoint, commit} for the traced run (`result` the result row selecting
    its checkpoint), `None` for a baseline; `meta.note` is `"registered final <row>: <arm> seed <k> at checkpoint
    <c>"` or `"registered final <row>: <baseline>"`. A 2013 final writes one replay per traced play (not per graded
    row): its note ends `", played at <regime> prices"`, the play's regime, and its `totals` is the play's own
    result, its `seasons.parquet` row at that regime. The stamp and each `meta.run` are built before
    `FINAL_RESULTS/<row>/` is made.
  - **A short traced season** (its frame isn't `season_length × 96` steps, e.g. a solver failure ended it) gets no
    replay; every full-length one is still written. After the replays, `--final` raises one `RuntimeError` naming
    each short season (`<prefix> <split> day <start_day>: <n> of <N> steps`), saying `seasons.parquet` and
    `per_split.json` are complete, and giving the `final-result --final-registered <row> --numbers
    <per_split.json>` command in place of the `next:` line.
  - **Prints** one line per split × arm, a 2013 final's per split × regime × arm naming its regime after the split
    (primary and each guard as `iqm [lo, hi]`, `P(improvement)`, `n_runs`, `invalid_episodes`, and `n<5: no claim`
    for a learning arm under 5 runs), then last
    `next: python -m greenleaf_control.ledger final-result --final-registered <row> --numbers <per_split.json>`.
  - **Writes no ledger row.** Every `allow_test=True` in evaluate is inside `_final`, this path.
- **Names:** `FINAL_RESULTS` (`control/results/final/`), `FINAL_PRIMARY = "profit_eur_m2"`, `FINAL_GUARDS =
  ("viol_hours_co2", "viol_hours_temp", "viol_hours_rh", "paper_return", "yield_kg_m2", "solver_failures")`, each a
  `ledger.RESULT_KEYS` entry.
- **Per-step frame** (`record_steps`): `t, co2, temp, rh, u_*` (6), `u_rb_*` (residual only), `profit`, `cost_*`.
  - `t` is 0..T-1; co2 [ppm], temp [°C] and rh [%] from the state after the step; `u_*` (the applied absolute
    controls) and `u_rb_*` in upstream order: boil, co2, thscr, vent, lamp, blscr; `profit` and `cost_*` are the
    step's info values (reward-side diagnostics, not metrics).

## `greenleaf_control.export_replay` (D2, for the demo site)
- `export(frame, result, states, controls, params, *, controller, scenario, stamp, split, run,
  prices=metrics.PRICES) -> dict`:
  - `{meta, hourly, totals}`.
  - `meta` = {controller, scenario, split, run, commit, dirty, gl_gym, note, bounds, units}:
    - `split` is the split's name; `run` is `None` for a baseline, and for a run dir {id, registered, result, arm,
      seed, checkpoint, commit}: its ledger run row, registration and result row, and `commit` the run's own training
      commit, from its `manifest.json`;
    - `commit`, `dirty` and `gl_gym` are the export code's stamp;
    - `note` = `f"a simulator replay of a {split} season, not the final result"`.
  - `hourly` holds H = season_length × 24 points of:
    - t_hour;
    - co2/temp/rh at the end of each hour (frame row 4h+3, 3 significant digits);
    - the six u_* as the hour's mean (3 decimals);
    - profit and viol_co2/temp/rh as metrics on the prefix to that hour (4 decimals), profit at `prices` (pass the
      regime `result` was graded at, so the last hour's profit is the season's).
  - `totals` = the result exactly.
- CLI `--controller {zero_controls|random|rb_default|rb_tuned|<run_dir>} --split ... --out <dir> [--final ROW]`,
  `main(argv=None, *, ledger_path=ledger.LEDGER)` as `selection.main`:
  - writes `<controller>_<location>_<year>_<startday>.json` per scenario (compact, no NaN), each under 250 KB. A
    baseline's `<controller>` is its name, a run dir's `<arm>_seed<seed>` (e.g. `residual_w05_seed100`);
  - a run dir is a registered run, looked up in `ledger_path`, which is only read (a baseline never reads it):
    - the `run` row whose `run_dir` resolves to the dir; none is refused as "not a registered run", naming the dir;
    - its registration's `result` rows: exactly one, whose `selected` names the run; otherwise refused, naming the
      run id and the count;
  - a run dir plays at that row's `selected[run id]` checkpoint (not `selected.json`'s) through
    `evaluate.run_policy(..., return_trace=True)`;
  - refusals, in order and all before any env: `--final` (until F1), a test split without `--final`, a non-baseline
    with no `manifest.json`, the run lookup, an existing file.

## `greenleaf_control.selection`, `stats`, `tune_rb`, `run_grid`, `modal_app`
- `select_checkpoint(run_dir: Path) -> str`: the PROTOCOL §5 rule on VALIDATION; writes `run_dir/selected.json` once.
  - **Write-once:** an existing `selected.json` raises FileExistsError before any scoring.
  - **Candidates:** `ckpt_<steps>.zip` in numeric order of steps, then `ckpt_final.zip`. Any other `ckpt_*.zip` is
    ignored; none raises FileNotFoundError.
  - **Scoring:** each candidate is scored with `evaluate.run_policy(run_dir, name, s, 60, record_steps=<residual_lag>)`
    for s in `scenarios.VALIDATION`, in order. Seasons are always 60 days, whatever the manifest's season length.
  - **residual_lag:**
    - A candidate qualifies when, for every constraint c with a nonzero budget, the mean of its frames' `cost_<c>`,
      pooled over the three seasons, is ≤ `config["budgets"][c] × 1.10`. With every budget 0, every candidate
      qualifies.
    - The qualifying candidate with the highest mean `profit_eur_m2` wins.
    - If none qualifies, the lowest total cost wins: the plain sum over c of those means.
  - **Other arms:** the highest mean `paper_return` (metrics', not `paper_reward_sum`).
  - **Ties and bad values:** a tie goes to the earliest candidate. Invalid seasons are scored, never dropped. A
    non-finite value raises ValueError, and no file is written.
  - **Returns** `"<steps>"` or `"final"`, the name `run_policy` takes.
  - **`selected.json`:** JSON with sorted keys, indent 2, no NaN or Infinity:
    - `checkpoint`, `method`;
    - `rule` (`max_mean_paper_return` | `max_mean_profit_within_budget` | `min_total_cost`);
    - `scenarios` (VALIDATION), `season_length` (60);
    - `budgets` and `tolerance` (1.1), both residual_lag only, else null;
    - `candidates: {name: {per_scenario: [run_policy result dicts], mean_paper_return, mean_profit_eur_m2, mean_cost:
      {co2, temp, rh} or null, qualifies: bool or null}}`.
- `RESULTS` = `control/results/validation`. `RESULT_KEYS` (defined in `ledger`, dependency-free; selection re-exports it)
  = a scored season's numeric keys (`metrics.episode_metrics`' keys except `valid`, plus `paper_reward_sum`).
- `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`; CLI `python -m greenleaf_control.selection --registered <row>
  [--out FILE]`, `--out` defaulting to `RESULTS/<row>.json`, created exclusively.
  - **Refused before anything is scored or written:**
    - an existing `--out`;
    - an unknown row;
    - scenarios other than exactly `["validation"]`;
    - an arm whose method is neither in `envs.METHODS` nor `rb_tuned` (`rb_default`, `zero_controls`, `random` have no
      run rows; compare them with `evaluate --controller ... --split validation`);
    - a primary or guard not in `RESULT_KEYS`;
    - any run without a `run-end`;
    - a registered arm × seed never run;
    - an arm × seed with more than one run that didn't crash;
    - an existing `selected.json` that isn't VALIDATION at 60 days;
    - arms that mix price regimes (an arm whose config names no `price_regime` is `sim`); the message names
      `price_regime` and the regimes.
  - **Scored**, per arm and registered seed (sorted): the run whose run-end isn't `crashed` (a crashed run beside a
    finished rerun scores the rerun).
    - A learning arm uses its `selected.json`, or `select_checkpoint(run_dir)`: the chosen checkpoint's
      `per_scenario`, with no second evaluation.
    - `rb_tuned` uses `evaluate.run_rb(<run_dir>/rule_based_tuned.yml["GreenLightEnv"], s, 60,
      price_regime=<the arms' regime>)` for s in VALIDATION, so it is graded like for like.
  - **Writes** `{per_arm, selected}` (sorted keys, indent 2, no NaN):
    - `per_arm[arm]` = {iqm, ci_low, ci_high, guards: {g: {iqm, ci_low, ci_high}}, invalid_episodes};
    - each is `stats.iqm_ci` on the runs (by seed) × VALIDATION array, all null when no run was scored;
    - no `crashed` or `spend_usd`.
  - **Prints** one line per arm (primary and guards as iqm [lo, hi], n_runs, invalid_episodes, crashed N). It ends with
    `n<5: no claim` under 5 scored runs, or `deterministic, 1 run` for a one-run rb_tuned arm.
- `stats.iqm_ci(scores: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float, float]`
  - `scores` is 2-D, runs × tasks (seeds × scenarios); anything else raises ValueError.
  - Returns `(iqm, ci_low, ci_high)` as Python floats.
  - The IQM is `rliable.metrics.aggregate_iqm`.
  - The CI is the percentile 95% stratified bootstrap (runs resampled within each task) from a local
    `RandomState(seed)`, equal to rliable 1.2.0's `get_interval_estimates(method="percentile")` after
    `np.random.seed(seed)`.
  - The global RNG is untouched.
- `stats.prob_improvement(a: np.ndarray, b: np.ndarray) -> float`: rliable's `probability_of_improvement`.
  - Both arrays are 2-D with the same task count (else ValueError); their run counts may differ.
- `tune_rb`:
  - `TUNE_SPACE: dict[str, tuple[float, float]]` (12 setpoints, PROTOCOL §3): temp_setpoint_day, temp_setpoint_night,
    heat_deadzone, co2_day, rh_max (the vents' threshold only; rhMax isn't tuned), t_vent_off, thScrSpDay,
    thScrSpNight, thScrDeadZone, lamps_on, lamps_off, lamps_off_sun. Every other key stays at
    `configs/rule_based_default.yml`'s value.
  - `InvalidSeason(RuntimeError)`: a season metrics marks invalid.
  - `rb_return(params, scenarios, season_length=60, *, price_regime="sim") -> float`: the mean over scenarios of
    `evaluate.run_rb(params, s, season_length, price_regime=price_regime)[0]["paper_return"]` (metrics', not
    `paper_reward_sum`). An invalid season raises `InvalidSeason` and is never scored.
  - **Tuning:**
    - Optuna `TPESampler(seed=--seed)`, maximizing; trial 0 is enqueued at the default setpoints.
    - Every trial is scored by `rb_return(params, scenarios.TRAIN_EVAL, season_length, price_regime=--price-regime)`.
      TRAIN_EVAL is hard-wired.
    - `optimize(..., n_jobs=--jobs, catch=(InvalidSeason,))`: an invalid-season trial is FAIL and never scored; any
      other exception crashes the run.
    - The tuned params are the best COMPLETE trial's. With none, the run crashes and writes no yml.
    - Deterministic at `--jobs 1` only.
  - `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`; CLI `--trials N --seed S [--jobs 1] [--runs-dir runs]
    [--price-regime {sim,agc2020}] (--smoke | --registered <row>)`.
  - `--smoke`: min(N, 2) trials on 1-day seasons into `<runs-dir>/smoke/tune_rb-<UTC stamp>-<hex>`. No ledger row;
    configs are untouched.
  - `--registered <row>`, refused before anything is written unless all of these hold:
    - the row has exactly one arm of method `rb_tuned`, whose config is exactly `{"trials": N}`, optionally with a
      `price_regime`;
    - `--price-regime` is the arm's `price_regime` (`sim` if it names none);
    - `--trials N`;
    - scenarios exactly `["validation"]`, the split the result is scored on (PROTOCOL §2, §9);
    - seeds `[0]` and `--seed 0`;
    - timesteps 0;
    - `--jobs 1`;
    - its configs file doesn't exist yet: `configs/rule_based_tuned.yml` for `sim`,
      `configs/rule_based_tuned_<regime>.yml` (e.g. `rule_based_tuned_agc2020.yml`) otherwise.
  - Then a registered run plays N trials on 60-day seasons into `<runs-dir>/<row>/<arm>/seed0`:
    - a `run` row before the trials;
    - after them, a `run-end`: finished, or crashed and re-raised; timesteps 0; `solver_failures` = the count of
      FAIL trials (trials, not steps; null if the tune raised); `cost_usd` and diagnostics null;
    - then that configs file, written once; a tune at another regime never touches `rule_based_tuned.yml`.
  - **Run dir outputs:**
    - `manifest.json`: config `{method: "rb_tuned", seed, trials, jobs, season_length, price_regime,
      tuned_on: "train_eval", tune_space, run_dir}`;
    - `trials.csv`: Optuna's `trials_dataframe()`, with state COMPLETE/FAIL and `params_<key>`;
    - `rule_based_tuned.yml` (this name at every regime; selection reads it): `{"GreenLightEnv": defaults with the
      best trial's 12 setpoints}`, as `load_rb_params` reads it.
- `run_grid`: `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`. Module constants `CPU=4.0`, `MEMORY_MB=8192`,
  `TIMEOUT_S=14400`, `USD_PER_CORE_S=0.0000131`, `USD_PER_GIB_S=0.00000222`, `PRICES_AS_OF="2026-10-28"`
  (modal.com/pricing; one core = 2 vCPU).
  - `--registered <row> --backend {local,modal} [--parallel 2]`:
    - Before any launch, refuses an unknown row, a method not in `envs.METHODS`, a config key outside `CONFIG_KEYS`
      (train's `penalty_weight, budget_frac, season_length, n_envs, rb, price_regime`) and any scenario that is a
      split of a `scenarios.SEALS` seal.
    - Then runs `python -m greenleaf_control.train --method M [--<key> V ...] --seed S --timesteps T --registered
      <row>` once per arm × seed, `--parallel` at a time, and exits 1 naming the arm and seed of any failure.
    - Locally, run_grid writes no ledger row; each train run records its own.
    - `--backend modal`: the Mac is the only ledger writer.
      - It runs the same pre-launch refusals, then stamps the code with `manifest.code_stamp(smoke=False)` (refusing a
        dirty tree), and refuses any cell whose local run dir `RUNS/<row>/<arm>/seed<S>` already exists. All of this
        happens before any call.
      - It opens one remote session (`_run_remote()`) and runs the cells `--parallel` at a time. For each cell it
        writes a `run` row (run_dir = that absolute local path, where `modal volume get greenleaf-runs <row> runs`
        from `control/` lands it), calls `train_remote(argv, {"stamp", "registered": row}, "<row>/<arm>/seed<S>")`,
        then writes a `run-end`:
        - the returned manifest's status, timesteps, solver_failures and diagnostics;
        - `crashed` with null fields if the call raised anything (a crash, OOM at the hard memory limit, the 4 h
          timeout, a cancellation, a preemption rerun hitting the existing run dir).
      - `wall_s` is the Mac's monotonic wall around the call, and `cost_usd = wall_s × (CPU × USD_PER_CORE_S +
        MEMORY_MB/1024 × USD_PER_GIB_S)` ($0.00007016/s).
      - Also refused before any call: any cell that already has a `run` row for this registration (a remote run dir
        lives only on the volume until it's downloaded).
      - The pool outlives the session. On a Ctrl-C or any early exit, cells still queued are cancelled (no ledger
        rows, listed as not launched), and the error re-raises.
      - It exits 1 naming the failed cells.
  - `--issue N --arms '<JSON list of {name, method, config}>' --seeds S [S ...] --timesteps T --dry [--parallel 2]`,
    run before registering. It writes nothing and prints:
    - the run count;
    - the worst case: CPU-hours = runs × TIMEOUT_S × CPU, and $ = runs × TIMEOUT_S × (CPU × USD_PER_CORE_S +
      MEMORY_MB/1024 × USD_PER_GIB_S). A true bound per attempt, because Modal bills max(request, usage) and
      `train_remote`'s limits equal its requests;
    - the wall clock from the newest `finished` run-end with timesteps > 0, so never a tune_rb run (wall_s / timesteps
      × T, × ceil(runs / parallel) locally); with none, "no measured run".
- `modal_app` (import contacts nothing; only run_grid's `_run_remote` calls it):
  - `app = modal.App("greenleaf")`;
  - image: `debian_slim(python_version="3.12")` + `apt_install("git")` + `uv_sync(<absolute control/>, frozen=True)` +
    `add_local_python_source("greenleaf_control")` + `add_local_dir(control/configs, "/root/configs")` +
    `add_local_dir(control/data/weather, "/root/data/weather")`. Every weather file ships, the sealed 2012 and 2013 included:
    they're hashed against the stamp, never trained on.
  - `train_remote(argv: list[str], meta: dict, run_rel: str) -> dict`:
    - hard limits `cpu=(4.0, 4.0)`, `memory=(8192, 8192)` ((request, limit) pairs), `timeout=4*3600`, `retries=0`,
      `single_use_containers=True` (`nonpreemptible` stays False, pending Aarmen's spend call);
    - runs `train.main([*argv, "--runs-dir", "/runs", "--remote", json.dumps(meta)])`, commits the volume in a
      `finally`, and returns `/runs/<run_rel>/manifest.json`, read by its own path (the commit reloads the volume, so
      other containers' manifests are visible).
  - volume `greenleaf-runs` (`create_if_missing=True`) mounted at `/runs`.

## `greenleaf_control.report` (A1a, A1c; plays no season, writes no ledger row)
- `ANALYSIS` = `control/results/analysis`; `BASELINE` = `"rb_tuned"`; `METRICS` = `(profit_eur_m2, viol_hours_total,
  viol_hours_co2, viol_hours_temp, viol_hours_rh, yield_kg_m2, paper_return)`; `SCORED_ONCE` = `"2012 test weather,
  scored once"`.
- **Splits:** validation, and the three TEST splits (A1c) once the ledger holds the 2012 `final-result` row (#73: the
  one whose final-registered row has `ledger.seal_of(...) == 2012`; a 2013 final-result doesn't open them, wherever
  it sits in the file, and every TEST-split read below is of the 2012 final). Until then a test split is refused (a
  ValueError naming it and the seal, PROTOCOL §8) before any parquet is read (only the ledger is read). `train_eval` and `paper` are always refused (a ValueError naming the split from `load_scores`, an argparse
  error from the CLI).
- `load_scores(split, *, ledger_path=ledger.LEDGER) -> pd.DataFrame`: one row per label × run × scenario.
  - **Columns:** `label, arm, issue, registered, result, decision, protocol, run, seed, crashed, split, location,
    growth_year, start_day`, every `ledger.RESULT_KEYS` key, `valid`, `viol_hours_total` (co2 + temp + rh). `issue`,
    `seed` and `crashed` are nullable Int64.
  - **Order** (selection's; the bootstrap depends on it): baselines in `envs.BASELINES` order, then each `registered`
    row with scenarios `["validation"]` and a `result` row in ledger order, arms in registration order, runs by seed,
    scenarios in `scenarios.SPLITS` order.
  - **Baselines:** every `*.parquet` in `selection.RESULTS`, rows with split `validation` (others ignored), labelled by
    name; ids, `run`, `seed` and `crashed` null. Refused: a controller not in `envs.BASELINES` (a run dir included), a
    (controller, scenario) in two parquets, a baseline without every scenario of the split.
  - **Arms:** runs from `selection._registration` (every run that didn't crash, by seed); each run's seasons are
    `selection._read`'s `candidates[checkpoint]["per_scenario"]`. Refused: a run with no `selected.json` ("run
    selection first"), a stale one, an arm whose runs all crashed, two result rows on a registration, two decision rows
    on a result. A registration with no result row prints `skipped <row>: no result row yet`, and none of its files
    are read. Validation is priced at `sim` (#74, PROTOCOL §4): a registration any of whose arms' config names another
    `price_regime` prints `skipped <row>: priced at <regime>; ...`, and none of its files are read.
  - **Ids:** `label` = `<arm> #<issue>/<first 4 hex of the registered id>`; `decision` = `"<adopted|dropped> <decision
    row id>"` or null; `protocol` = the first 8 hex of the registration's `fingerprints["docs/PROTOCOL.md"]`; `crashed`
    = `ledger._arm_tally`'s count for the arm (null for a baseline).
  - **E1:** an arm of method `rb_tuned` is no row of its own; its arm name, `registered`, `result`, `decision` and
    `protocol` go on the `rb_tuned` baseline rows. Refused: a second such arm, one with no run that didn't crash, one
    with no `rb_tuned` parquet.
  - **A TEST split** reads only `evaluate.FINAL_RESULTS/<the final-registered row the final-result names>/
    seasons.parquet`, that split's rows; no selected.json and no evaluate parquet. The same columns: `label` = `arm` =
    the learning arm's or baseline's name; a learning arm's `issue, registered, result, decision, protocol` as above
    (from the registration its runs name), `run` and `seed` from its run rows; `crashed` null; a baseline's ids null.
    Order: the final's baselines in `envs.BASELINES` order, then its arms (`ledger.final_arms`) by registration in
    ledger order and arm in registration order, runs by seed, scenarios in `scenarios.SPLITS` order. Refused (a
    ValueError naming it): no seasons.parquet; a registered run or baseline without each of the split's scenarios
    exactly once.
- `summarize(scores) -> pd.DataFrame`: one row per label (a `label` column), in the scores' first-appearance order.
  Each statistic is on the label's runs (by seed) × scenarios (split order) array, whatever the input's row order.
  - `label, decision, protocol, n_runs, crashed, invalid_episodes` (the count of `valid` False; PROTOCOL §4).
  - `<m>_{iqm,lo,hi}` = `stats.iqm_ci` for each m in `METRICS`; `viol_hours_sum_iqm` = the sum of the three
    per-constraint IQMs (no CI). The table carries both "total" violation hours.
  - **rb_tuned like-for-like:** `rb_tuned_profit_iqm` = `stats.iqm_ci(np.tile(<rb_tuned's 1 × S profits>, (n_runs,
    1)))[0]`; `rb_tuned_viol_sum_iqm` = the same per constraint, summed.
  - **Paired gains over rb_tuned's same scenario:** `profit_gain_{iqm,lo,hi}` = `iqm_ci(arm − rb_tuned)` on profit;
    `viol_saved_{iqm,lo,hi}` = `iqm_ci(rb_tuned − arm)` on `viol_hours_total`.
  - `p_improve_profit` = `stats.prob_improvement(arm, rb_tuned)`; `p_improve_viol` = the same on negated
    `viol_hours_total`. Against a 1-run baseline it's the share of same-season wins (ties half), with no CI.
  - `flag`: `deterministic, 1 run` for zero_controls, rb_default and rb_tuned (PROTOCOL §6); `n<5: no claim` under 5
    runs, random included; otherwise `""`.
  - No `rb_tuned` rows raises ValueError.
- `check_ledger(scores, *, ledger_path) -> int`: for every `result` row of a registration in the scores, each arm's
  primary and every guard (`iqm, ci_low, ci_high`) must equal `stats.iqm_ci` on the scores within 1e-12, else
  ValueError naming the result row, arm and key. Returns the number of (arm, key) pairs checked.
  - **A TEST split's scores** are checked against the `final-result` row instead: every arm in its
    `per_split[split]`, `evaluate.FINAL_PRIMARY` and each of its guards, within 1e-12 of `stats.iqm_ci` on that
    arm's scores, else ValueError naming the final-result row, split, arm and key (an arm with no score rows too).
- `figures(scores, summary, out_dir) -> None` (A1b): three PNGs drawn by `viz`, derived from the scores and the
  summary, each with the footer `source: python -m greenleaf_control.report --split <split>; derived from
  summary.csv / scores.parquet`. On a TEST split the footer adds `SCORED_ONCE` on a second line, and each figure
  gets `viz`'s `title=` = `<viz's default title> · <split>, 2012 test weather, scored once`. Validation passes no
  `title=`, so `viz` draws its default (`viz.PER_SCENARIO_TITLE`, `PROFILES_TITLE`, `TRADEOFF_TITLE`).
  - **Status** of a label: its decision's word (`adopted` or `dropped`), unless a decision note calls its arm otherwise:
    `report.NOTE_CALLS[(decision row id, arm)]` (E4's `decision-b159ac9a` drops `residual_w4`, E8's
    `decision-a5b2fbd3` drops `residual_agc`) overrides the row's word. None for a baseline (rb_tuned's E1 fold
    included) or an undecided arm.
  - `per_scenario.png`: `viz.per_scenario`, panels `profit [€/m²]` and `total violation hours [h]`, every label's raw
    runs × scenarios array, scenarios labelled `<location> day <start_day>`;
  - `profiles.png`: `viz.profiles`, every label but rb_tuned, panels `profit gain over rb_tuned's same season [€/m²]`
    (arm − rb_tuned) and `violation hours saved vs rb_tuned's same season [h]` (rb_tuned − arm); `highlight` = the
    labels whose status is `adopted`;
  - `tradeoff.png`: `viz.tradeoff`, one point per summary row at (`viol_saved_*`, `profit_gain_*`) with its status and
    `n` = `n_runs` (a learning label under 5 runs reads `(n=<k>, no claim)`), so rb_tuned is the origin; x label
    `violation hours saved vs rb_tuned [h]`, y label `profit gain over rb_tuned [€/m²]`.
- `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`; CLI `python -m greenleaf_control.report --split
  {validation,amsterdam_2012,london_2012,abbotsford_2012} [--out DIR]`, `--out` defaulting to `ANALYSIS`.
  - Loads, checks and summarizes before a dir is made or a file is written; a refusal writes nothing.
  - Writes `<out>/<split>/scores.parquet` (load_scores' frame), `summary.csv` (summarize's, full precision) and
    `summary.md`, then `figures`' `per_scenario.png`, `profiles.png` and `tradeoff.png`, replacing any previous
    version. `summary.md`: line 1 names the command and `summary.csv` (on a TEST split also
    `control/results/final/<final-registered row>/seasons.parquet`, the final-result row and `SCORED_ONCE`), a
    legend, then a pipe table with one row per label (3 decimals, hours 2). Neither text file carries a timestamp or
    the out path.
  - Prints `matches the ledger: <k> (arm, key) pairs`, then one line per label.

## `greenleaf_control.site_build` (S1, for the demo site; plays no season, writes no ledger row)
- Here `site/` and `control/...` paths are relative to the repo root (`root`), not to `control/`.
- **Reads only recorded files:** imports stdlib, `ledger` and `manifest` only (no env, `evaluate`, `report`,
  `export_replay` or `rewards`, so no `gl_gym`); opens no weather file; writes nothing outside `out`.
- `CONTROLLERS`: the page's controllers, id → display name, in order: `rb_tuned` "Tuned rules", `rb_default`
  "Default rules", `residual_w05` "Residual agent, fixed penalty", `lag_b100` "Residual agent, constrained", `ppo` "PPO, no
  rules", `ppo_paper` "PPO, paper setup", `zero_controls` "Do nothing". `random` is left out everywhere.
- `build(out: Path, *, root=manifest.REPO_ROOT, ledger_path=ledger.LEDGER, force=False) -> dict`:
  - returns `{"final_result": <final-result row id>, "replays": <int, copied into out/data/final>, "files": <int,
    files written>, "bytes": <int, their total size>}`.
  - **The final** is the ledger's 2012 `final-result` row (#73: the one whose final-registered row has
    `ledger.seal_of(...) == 2012`; a 2013 final-result changes nothing); its replays are `root/control/results/final/<the
    final-registered row it names>/replays`, never typed.
  - **Copies, byte for byte:** every plain file directly in `root/site` and `root/site/data/*.json`, to the same
    relative path under `out`; and each page controller's `<id>_<Location>_2012_<day>.json` from the final's replays to
    `out/data/final/` under its own name (7 × 3 cities × 8 start days = 168 today).
  - **`out/data/index.json`:** `{"final_result", "splits"}`; splits `amsterdam_2012`, `london_2012`, `abbotsford_2012`,
    `validation` in that order, each `{"id", "location", "year", "label", "dir", "start_days", "controllers"}`:
    - `dir` is `data/final` for a 2012 split, `data` for validation; a replay is
      `out/<dir>/<file_prefix>_<location>_<year>_<day>.json`;
    - `start_days` are the days whose files are present, ascending; refused unless every controller has the same set;
    - a controller is `{"id", "name", "file_prefix", "seed"}`, in `CONTROLLERS` order on a 2012 split (`file_prefix` =
      id); validation lists `rb_tuned` and `residual_w05` (`file_prefix` `residual_w05_seed100`), what `site/data`
      holds; `seed` = its replays' `meta.run.seed`, null when `meta.run` is null; refused if they disagree.
  - **`out/data/results.json`:** `{<split id>: {"final_result" (2012 splits only), "note", "rows"}}`, in index.json's
    split order:
    - `note`: `"2012 test weather, scored once in the final (<row id>). Simulated, not a real greenhouse."`;
      validation `"Validation seasons: checkpoints were selected on these. Simulated."`;
    - `rows`: `root/control/results/analysis/<split>/summary.csv`'s labels in its order, `random` dropped; each
      `{"controller", "n_runs", "invalid_episodes", "decision", <m>_{iqm,lo,hi} for m in profit_eur_m2, profit_gain,
      viol_hours_total, viol_hours_co2, viol_hours_temp, viol_hours_rh, yield_kg_m2, "viol_hours_sum_iqm",
      "rb_tuned_profit_iqm", "rb_tuned_viol_sum_iqm"}`: floats are `float(<the csv text>)`, the counts `int`, a blank
      decision null; never recomputed or rounded;
    - `controller`: the label on a 2012 split; on validation the page controller id of the final's baselines (by
      name) and of each arm it scored (`ledger.final_arms`, by report's `<arm> #<issue>/<4 hex>` label of its
      registration), null for every other label.
  - Neither JSON holds NaN or Infinity (`allow_nan=False`).
  - **Refusals** (`SystemExit` with a message), all before anything is made or written: the ledger has no 2012
    `final-result` row (the message names it; a ledger whose only final-result is 2013's included); `out`, resolved
    and casefolded, is in or equal to `root/site`, `root/control/results`, `root/control/runs` or
    `root/control/ledger`; `out` is `root` or an ancestor of it
    (`force` included); `out` is non-empty without `force`. `force` overwrites same-named files and never deletes.
- `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`; CLI `python -m greenleaf_control.site_build --out DIR
  [--force]` (`--out` required). Prints one line per file group (the site files, `<n> replays`, `index.json`,
  `results.json`) and `<files> files, <bytes> bytes`; returns 0.

## `greenleaf_control.prices` (#68, exploratory; plays no season, writes no ledger row)
- **Re-prices recorded seasons only:** profit is linear in each price, so a season's recorded `yield_kg_m2`,
  `heat_cost`, `elec_cost` and `co2_cost` (at `metrics.PRICES`) give its profit at other prices exactly. No env,
  `evaluate`, `report`, `selection` or `export_replay` is imported; no weather file is opened; the 2012 seal is not
  reopened (the test seasons are the final's recorded ones). Every output says "exploratory, re-priced recorded
  seasons".
- `REGIMES`: `sim` (a copy of `metrics.PRICES`) and `agc2020` (heat 0.03 €/kWh and CO₂ 0.08 €/kg (its ≤ 12 kg/m²
  tier) from the 2nd Autonomous Greenhouse Challenge's 2020 net-profit scoring (Hemming et al. 2020, Sensors
  20(22):6430); electricity 0.06 €/kWh, the unweighted mean of its 0.08 on-peak (07:00-23:00) and 0.04 off-peak
  tariffs, flat, not time-of-use; fruit 1.6 €/kg and dmfm 0.065 are the simulator's), the numbers #67 puts in
  `metrics.PRICE_REGIMES`. `PAIRS`: (ppo, rb_default), (ppo,
  residual_w05), (zero_controls, rb_default), (rb_tuned, rb_default), (residual_w05, rb_tuned).
- `reprice(seasons, prices) -> np.ndarray`: `yield × fruit − heat_cost × heat/0.09 − elec_cost × elec/0.30 −
  co2_cost × co2/0.30` per row. A `dmfm` other than `metrics.PRICES["dmfm"]` is a ValueError.
- `load_seasons(*, ledger_path=ledger.LEDGER) -> pd.DataFrame`: one row per recorded season (`split, controller,
  seed, location, growth_year, start_day, yield_kg_m2, heat_cost, elec_cost, co2_cost, profit_eur_m2,
  viol_hours_{co2,temp,rh}, valid, run, source`; `seed` float, NaN for a baseline). Validation: the rows of
  `control/results/analysis/validation/scores.parquet` whose label `site_build._validation_arms` maps, `controller` =
  the final's arm id. The 2012 splits: every row of `control/results/final/<final-registered row>/seasons.parquet`,
  found from the ledger's 2012 `final-result` row (#73: the one whose final-registered row has `ledger.seal_of(...)
  == 2012`; a 2013 final-result changes nothing), random included. No 2012 `final-result` row: ValueError naming
  it.
- `tables(seasons) -> {"regimes", "sweep", "crossovers"}`, per split present. A controller's matrix is its runs (by
  seed) × scenarios (`scenarios.SPLITS` order), as `report._matrix`.
  - `regimes`: `split, regime, rank, controller, n_runs, profit_{iqm,lo,hi}, gain_rb_tuned_{iqm,lo,hi},
    gain_rb_default_{iqm,lo,hi}, viol_hours_iqm, yield_iqm`: `stats.iqm_ci` of the profit matrix and of it minus the
    baseline's 1 × S row; rank 1 the highest profit IQM in its split and regime; Σ hours (co2 + temp + rh) and yield
    IQMs are price-free.
  - `sweep`: `split, sweep, x, heat, elec, co2, controller, profit_iqm` (rliable's point IQM). `lambda`: x in
    `linspace(0, 1, 41)`, heat, elec and co2 moved x of the way from agc2020 to sim; `elec`: x = 0.02 to 0.40 €/kWh
    in 0.01 steps, heat and co2 at agc2020's. Fruit and dmfm fixed.
  - `crossovers`: `split, sweep, controller, versus, x, note`, one row per sign change along each sweep of each pair's
    difference, x linearly interpolated between the two grid points; none is one row, x NaN, note `none in range`.
    The difference is the paired-gain IQM against rb_default or rb_tuned, the difference of point IQMs for (ppo,
    residual_w05).
- `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`; CLI `python -m greenleaf_control.prices [--out DIR]`, `--out`
  defaulting to `control/results/analysis/prices`.
  - Loads, then checks every row (random included): `reprice` at sim within 1e-9 €/m² of the recorded
    `profit_eur_m2`, else ValueError naming the split, controller, run, seed and start day with both values. Drops
    random, then refuses (ValueError) an invalid season of a kept controller. Everything is computed before `out` is
    made; a refusal writes nothing.
  - Writes `regimes.csv`, `sweep.csv`, `crossovers.csv` (full precision), `summary.md` (line 1 names the command and
    says "exploratory, re-priced recorded seasons"; no timestamp or out path; the electricity-only crossovers beside
    Eurostat's non-household electricity prices, an annotation) and `sweep.png` (profit IQM vs λ per controller, one
    panel per 2012 split, `site_build.CONTROLLERS` display names, every text ≥ 9 pt); returns 0.

## `greenleaf_control.rule_family` (#69, exploratory; VALIDATION weather only, writes no ledger row)
- **Plays the rule family on VALIDATION weather (Amsterdam 2010, start days 45, 59, 75); nothing trains and nothing
  is selected.** Every season is played once at sim and re-priced at agc2020 with `prices.reprice` and
  `metrics.PRICE_REGIMES`. VALIDATION selected the agents' checkpoints, so the comparison is in-sample for them; every
  output says "VALIDATION weather; exploratory analysis; nothing trains or is selected". Only `Amsterdam/2010.csv`
  is read.
- `OUT = control/results/analysis/rule_family` (both commands' default `--out`); `VALIDATION_SCORES =
  control/results/analysis/validation/scores.parquet`, read at call time. `LABELS`: `ppo #33/0221`, `ppo_paper
  #31/2d02`, `residual_w05 #36/f2fd`, `lag_b100 #39/6d92`, `rb_default`, `rb_tuned` → their controller ids.
- `sample_settings(n, seed) -> list[dict]`: `[load_rb_params("default"), load_rb_params("tuned")]`, then `n` draws
  from one `np.random.default_rng(seed)`: the defaults overlaid with one `rng.uniform(lo, hi)` per
  `tune_rb.TUNE_SPACE` key, in its order. A setting's id is its index.
- `skip_reason(scenario, season_length) -> str | None`: a reason naming day 365 when `start_day + season_length ≥
  365` (upstream's loader reads the weather 0.5 day past a season's end, so a season ending on day 365 already opens
  the next year's file), else None.
- `compare(frontier, points) -> pd.DataFrame`: `points` (`controller, label, regime, profit_iqm, yield_iqm,
  viol_hours_iqm`) plus `best_setting, best_profit, gap, n_qualifying, note`: among `frontier`'s valid settings with
  `yield_kg_m2 ≥ yield_iqm` and `viol_hours ≤ viol_hours_iqm` (both within 1e-9), the one with the highest
  `profit_<regime>`; `gap = profit_iqm − best_profit`. None qualifying: NA, NaN, NaN, 0, `no setting qualifies`.
  `best` is in-sample too (a maximum over the qualifying settings on the same 3 VALIDATION seasons, optimistic for
  the rules, more so as `n_qualifying` grows); `gap` is descriptive, not an out-of-sample estimate.
- `main(argv=None) -> int`; CLI `python -m greenleaf_control.rule_family {frontier,horizon}`. `--jobs J` (default 1):
  1 plays every season in the calling process through `evaluate.run_rb` / `envs.run_baseline`, looked up at call
  time; more uses J spawned worker processes, results in task order, so any `--jobs` writes the same bytes.
  Everything is played, checked and computed before `--out` is made: every played season and every recorded season
  of a `LABELS` entry must re-price at sim to its recorded `profit_eur_m2` within 1e-9 €/m², and no recorded one may
  be invalid, else ValueError and nothing is written.
  - `frontier [--n 1000] [--seed 0] [--season-length 60]`: every setting of `sample_settings(n, seed)` on the 3
    seasons via `evaluate.run_rb(params, scenario, season_length)`; a season length with a `skip_reason` (ending on
    or past day 365) for any season is a ValueError naming 365, before anything is played or written.
    Writes `frontier_seasons.csv` (`setting, start_day, profit_eur_m2, yield_kg_m2, heat_cost, elec_cost, co2_cost,
    viol_hours_{co2,temp,rh}, valid`; an invalid season is flagged, never dropped), `frontier.csv` (`setting, valid,
    profit_sim, profit_agc2020, yield_kg_m2, viol_hours, heat_cost, elec_cost, co2_cost`: means over the 3 seasons,
    Σ hours = co2 + temp + rh, valid = all 3 valid; an invalid setting stays, excluded from `compare`),
    `frontier_summary.csv` (one row per `LABELS` entry × regime: rliable's IQM over the label's runs × seasons of
    re-priced profit and of yield, Σ hours = the sum of the 3 per-constraint IQMs, then `compare`'s columns),
    `frontier.md` (line 1 the command with `--n`, `--seed`, `--season-length`; in-sample; the excluded count; best
    in-sample and optimistic for the rules, gap descriptive; zero_controls and random left out, and why; the hours
    column named "Σ per-constraint IQMs") and `frontier.png` (per regime, valid settings as grey points at (fruit,
    profit), each label's IQM as a marker; the footer says in-sample and "Σ hours not shown").
  - `horizon [--lengths 60 120 180 240 300]`: rb_default and rb_tuned (`evaluate.run_rb`) and zero_controls
    (`envs.run_baseline`, graded by `metrics` on the raw env's params and N) on each season × length; a cell with a
    `skip_reason` is never played. Writes `horizon.csv` (`controller, start_day, length, profit_sim, profit_agc2020,
    yield_kg_m2, viol_hours, valid, note`; a skipped cell has NaN numbers and its reason in `note`), `horizon.md`
    (line 1 the command with `--lengths`; the skip reasons; profit, fruit and Σ hours tables) and `horizon.png`
    (profit vs season length per controller, a line per start day, per regime).
  - Figures: `Figure(layout="constrained")`, 200 dpi, every text ≥ 9 pt, `site_build.CONTROLLERS` display names, one
    panel per regime titled with its name. No clock time or out path in the `.md` files. Returns 0.

## `greenleaf_control.hypotheses` (#73; plays no season, writes no ledger row)
- **Grades the registered 2013 hypotheses H1-H4 from the 2013 final's recorded seasons**, so none is graded by hand.
  The 2013 final is found from the ledger, never typed: the `final-result` row whose final-registered row has
  `ledger.seal_of(...) == 2013`, then `FINAL_RESULTS/<that final-registered row>/seasons.parquet` (#72's shape, one
  row per season and `price_regime`, `sim` and `agc2020`). `FINAL_RESULTS` = `control/results/final`, read at call
  time. No env, `evaluate`, `report` or `selection` is imported; no weather file is opened; the 2012 final's files are
  never read.
- **Per 2013 split** (`scenarios.SEALS[2013]` order), each regime from its own rows; nothing is pooled.
  - An arm's matrix is its runs (by seed) × the split's seasons (`scenarios.SEALS[2013][split]` order, as
    `report._matrix`); a baseline is 1 run. Refused (ValueError naming the arm) unless every run holds each season
    once, so a missing graded arm or baseline is refused.
  - A paired gain is `stats.iqm_ci(arm − the baseline's 1 × 8 row)`; a baseline's like-for-like profit IQM is
    `stats.iqm_ci(np.tile(<its row>, (n_runs, 1)))[0]`; "CI above X" is `ci_low > X`, "below X" `ci_high < X`.
  - Σ is the sum of the CO₂, temp and RH violation-hour IQMs (rliable's point IQM); rb_tuned's like-for-like Σ
    tiles its rows the same way.
  - **H1** (ppo vs rb_default): gain CI below 0 at agc2020 and above 0 at sim. **H2** (residual_w05 vs rb_tuned): gain
    CI above 0 at sim and at agc2020. Parts are named by their regime.
  - **H3** (ppo_agc, residual_agc, at agc2020) and **H4** (residual_w05, lag_b100, at sim): (a) the arm's profit
    `ci_low` > the highest like-for-like profit IQM of rb_default, rb_tuned and rb_tuned_agc2020 (H3) or rb_tuned's
    (H4); (b) Σ ≤ rb_tuned's like-for-like Σ, no slack; (c) its invalid seasons plus solver failures = 0.
- `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`; CLI `python -m greenleaf_control.hypotheses [--out DIR]`,
  `--out` defaulting to `OUT` = `control/results/analysis/final_2013`.
  - **Refusals**, all before `out` is made: no 2013 `final-result` row (ValueError naming "2013" and "final-result",
    before any parquet is read); the 2013 final's `seasons.parquet` missing; a graded arm or baseline missing.
  - Writes `hypotheses.csv`: exactly `split, hypothesis, arm, regime, part, value, bar, pass`, one row per split ×
    hypothesis × arm × part (H1/H2: `agc2020`, `sim`; H3/H4: `a`, `b`, `c`), full precision. `value` and `bar`: a
    gain's `ci_high` (below) or `ci_low` (above) and 0; a: the arm's profit `ci_low` and the bar; b: the arm's Σ and
    rb_tuned's like-for-like Σ; c: the count and 0. `pass` is a boolean.
  - Writes `summary.md`: line 1 names the command, `control/results/final/<final-registered row>/seasons.parquet` and
    the 2013 final-result row; the hypotheses word for word as registered (issue #73's text, named on a source line)
    and a legend (including how H3(a)'s bar is read); then one table row per split × hypothesis × arm with n runs,
    its verdict (`verified` when every part passes, else `fails (<parts>)`, with `n<5: no claim` appended for an arm
    with fewer than 5 runs) and each part's value, relation and bar (part c as invalid seasons and solver failures,
    each at the graded regime). No timestamp or out path. Prints each file written and each verdict; returns 0.
- **The 2012 readers keep 2012** (#73): `site_build.build`, `prices.load_seasons`, `report`'s TEST splits and
  `media`'s note read the `final-result` whose final-registered row has `ledger.seal_of(...) == 2012`, wherever a 2013
  final-result sits in the ledger, and a ledger whose only final-result is 2013's is refused (or sealed) as one with
  none.

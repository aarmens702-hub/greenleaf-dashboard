# API contract (GreenLeaf 2.0, `control/`)

The interfaces between modules. Build against them exactly; a change is proposed on the issue and decided by Aarmen.
Paths are relative to `control/`. Scenario dicts are `{"location": str, "growth_year": int, "start_day": int}`.

## `greenleaf_control.scenarios` (scorer file)
- `WEATHER_DIR: Path` (`control/data/weather`)
- `TRAIN_SAMPLER_KWARGS: dict`: `{"locations": ["Amsterdam"], "growth_years": [2008, 2009, 2011], "start_days": list(range(45, 76))}`
- `TRAIN_EVAL: list[dict]` (9), `VALIDATION: list[dict]` (3), `PAPER: dict`, `TEST: dict[str, list[dict]]` with keys
  `amsterdam_2012`, `london_2012`, `abbotsford_2012` (8 each). Values per `docs/PROTOCOL.md` §2.
- `SPLITS: dict[str, list[dict]]`: every named split above (`train_eval`, `validation`, `paper`, plus the three test
  keys).
- `is_test(scenario: dict) -> bool`: True for any 2012 scenario.
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
- `load_rb_params(name: str) -> dict`: `name` is `"default"` or `"tuned"` (`configs/rule_based_<name>.yml`).
- `make_controller(params: dict) -> RuleBasedController`
- `rb_action(raw_env, controller) -> np.ndarray`: float32, shape (6,), clipped to [0, 1]; computed from the current
  step's state (StepContext built from `raw_env.unwrapped`).

## `greenleaf_control.envs` (scorer file)
- `METHODS = ("ppo_paper", "ppo", "residual", "residual_lag")`
- `BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned")`
- `make_raw_env(*, scenario: dict | None = None, sampler_kwargs: dict | None = None, season_length: int = 60,
  normalize_actions: bool, penalty_weight: float = 1.0, season_progress: bool = False, allow_test: bool = False)
  -> gym.Env`
  - Always uses `ConstraintCostReward` and `weather_data_dir=WEATHER_DIR`.
  - A fixed `scenario` uses the fixed sampler; `sampler_kwargs` uses the random one.
  - Refuses a TEST scenario unless `allow_test=True` is passed. That flag exists only for `evaluate --final` and the
    reset-only weather test.
  - A season that would read a TEST year's file (its own year, or the next year's via upstream's year-end expansion)
    is refused unless `allow_test`, at construction (every scenario the sampler can draw) and at `reset`.
  - Ends the season after exactly N steps (decision B) and returns a flat `Box` observation (`FlattenObservation`,
    then `SeasonProgress` if asked).
  - Every step's info carries `x_prev` and `x`: float64 copies of the raw state before and after the step.
- `make_env(method: str, *, seed: int, scenario: dict | None = None, season_length: int = 60, penalty_weight: float = 1.0,
  rb_params: dict | None = None, residual_scale: float = 0.15) -> gym.Env`
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
- `run_baseline(name: str, scenario: dict, season_length: int = 60) -> list[dict]`: a full episode's per-step info
  dicts for a BASELINES entry (`random` is seeded by the scenario's start day).

## `greenleaf_control.metrics` (scorer file; never imports `rewards`)
- `episode_metrics(states: np.ndarray, controls: np.ndarray, params: np.ndarray, *, dt: float = 900.0,
  season_steps: int, prices: dict = PRICES, bounds: dict = BOUNDS) -> dict`
  - `states` has shape (T + 1, nx); `controls` has shape (T, 6).
  - `prices` and `bounds` default to the module constants `PRICES` and `BOUNDS` (`docs/PROTOCOL.md` §1).
  - Keys per `docs/PROTOCOL.md` §4, plus `steps` and `valid`.
- `trace_from_infos(infos: list[dict]) -> tuple[np.ndarray, np.ndarray]`: collects an episode's states
  (`infos[0]["x_prev"]`, then each step's `x`) and controls (each step's `controls`), both float64.

## `greenleaf_control.lagrangian`
- `class PIDLagrangian`, constructed as `__init__(self, budget: float, kp: float = 0.5, ki: float = 0.05,
  kd: float = 0.1, lambda_max: float = 100.0)`.
  - `update(mean_cost: float) -> float`. `lam` starts at 0.0.
  - The error is relative to the budget: `e = (cost - budget) / max(budget, 1e-8)`.
  - The integral is clamped to [0, lambda_max] (anti-windup); the derivative acts only on increases; λ is clamped to [0, lambda_max].
- `class VecLagrangianReward(VecEnvWrapper)`
  - `lambdas: np.ndarray` (3,)
  - Each reward becomes `(r - Σλᵢcᵢ) / (1 + Σλᵢ)`.
  - `pop_mean_costs() -> np.ndarray` (3,), the mean per step since the last pop.
- `class LagrangianCallback(BaseCallback)`, constructed as `__init__(wrapper, pids, log_path: Path)`.
  - At each rollout end it updates the PIDs, sets `wrapper.lambdas`, and logs `lag/lambda_<c>` and `lag/cost_<c>`.
- Stack order: `SubprocVecEnv → VecMonitor → VecLagrangianReward → VecNormalize`.

## `greenleaf_control.manifest`
- `write_manifest(run_dir: Path, *, config: dict, issue: int, registered: str | None, smoke: bool) -> dict`
  - Refuses an existing `run_dir`, and a dirty git tree unless `smoke`.
  - Writes `manifest.json` with:
    - `commit`, `dirty`;
    - `fingerprints` (sha256 of the scorer files and `configs/ppo.yml`), `weather_hashes`;
    - `versions` (python, gl-gym commit, stable-baselines3, torch, gymnasium, casadi, numpy);
    - `config`, `seeds` (`model`, `env`), `host`, `started_at`, `issue`, `registered`, `smoke`.
- `finish_manifest(run_dir: Path, *, status: str, wall_s: float, timesteps: int, solver_failures: int,
  cost_usd: float | None) -> dict`
  - `status` is one of `finished`, `crashed`, `budget-stop`.
- `fingerprints() -> dict`: the scorer and config hashes alone (the ledger uses the same function).

## `greenleaf_control.ledger` (the only writer of `ledger/ledger.jsonl`)
- **Every row** has `schema_version`, `row_id` (`<kind>-<8 hex>`), `kind`, `at` (UTC ISO), `issue`, `fingerprints`.
- **Kinds and their extra fields:**

  | Kind | Extra fields |
  |---|---|
  | `registered` | `hypothesis`, `arms` (list of {name, method, config}), `seeds`, `timesteps`, `scenarios` (split names), `primary`, `guards`, `rule`, `budget` ({cpu_hours, usd, backend}), `owner_ok` |
  | `run` | `registered`, `arm`, `seed`, `run_dir`, `status: "running"` |
  | `run-end` | `run`, `status`, `wall_s`, `timesteps`, `solver_failures`, `cost_usd`, `diagnostics` ({min_approx_kl, final_entropy, final_explained_variance}) |
  | `result` | `registered`, `per_arm` ({iqm, ci_low, ci_high, guards, crashed, invalid_episodes, spend_usd}), `selected` ({run → checkpoint}) |
  | `decision` | `result`, `decision` (adopted or dropped), `note` |
  | `final-registered` | `checkpoints`, `baselines`, `approved_by`, `approved_at` |
  | `final-result` | `final-registered`, `per_split` |

- **CLI** (`python -m greenleaf_control.ledger`):
  - `register ... [--dry]`
  - `result --registered <row>`
  - `decide --result <row> --adopted|--dropped --note "..."`
  - `show [--issue N] [--spend]`
  - `final-register ... --approved-by Aarmen`
- **Appends only.** `register` refuses a third registration on one issue without `--owner-ok`. `result` refuses
  unless every registered arm × seed has a `run-end` row.

## `greenleaf_control.train`
- `python -m greenleaf_control.train --method M --seed S [--penalty-weight W] [--budget-frac B] [--timesteps N]
  [--season-length D] [--n-envs 4] [--runs-dir runs] [--rb {tuned,default}] (--smoke | --registered <row>)`
- `--smoke` caps timesteps at 20,000 and the season at 2 days, writes under `runs/smoke/`, and records no ledger row.
- Without `--smoke`, `--registered` is required, and the arm and seed must be in that registration.
- `--rb` picks the residual arms' RB params (`configs/rule_based_<rb>.yml`, default `tuned`); the resolved params are
  stored in the manifest as `config["rb_params"]`.
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
- `build_vec_env(config: dict, n_envs: int, *, scenario: dict | None = None, season_length: int | None = None)
  -> VecNormalize`: `SubprocVecEnv` (`DummyVecEnv` for one env) → `VecMonitor` → `VecLagrangianReward` (residual_lag
  only) → `VecNormalize`, env i = `make_env(method, seed=seed + i, ...)` from the manifest's config; `scenario` and
  `season_length` override the config's (for evaluation).
- `build_model(config: dict, venv) -> PPO`: `configs/ppo.yml`'s PPO on cpu, seeded, with only `log_std_init` changed
  (-1.0 for the residual arms).

## `greenleaf_control.evaluate`
- `run_rb(params: dict, scenario: dict, season_length: int = 60, record_steps: bool = False) -> tuple[dict, pd.DataFrame | None]`
- `run_policy(run_dir: Path, checkpoint: str, scenario: dict, season_length: int = 60, record_steps: bool = False)
  -> tuple[dict, pd.DataFrame | None]`
  - Rebuilds the training stack with `train.build_vec_env(manifest["config"], 1, scenario=..., season_length=...)`
    (`DummyVecEnv` → `VecMonitor` → `VecLagrangianReward` for residual_lag), loads
    `VecNormalize.load(run_dir/"vecnorm_<checkpoint>.pkl", <that stack>)`, sets `training = False` and
    `norm_reward = False` before reset, then `PPO.load(run_dir/"ckpt_<checkpoint>.zip", device="cpu")` and
    deterministic predict until the season ends.
- **Result dict** (both): `metrics.episode_metrics(...)` graded with the env's own `p` and `season_steps = N`, plus
  `paper_reward_sum` = Σ `info["paper_reward"]` in step order with `+=` (a reproduction check, not a graded metric).
- **CLI:** `--controller {zero_controls|random|rb_default|rb_tuned|<run_dir>}`
  `--split {train_eval|validation|paper|<the three test splits>} --out <parquet>`.
  - The three test splits are refused as sealed (PROTOCOL §8) without `--final <final-registered row>`; `--final`
    itself is refused until F1 (Stage 3) implements it.
  - `--out` must not exist (created exclusively). One row per scenario in split order, 60-day seasons. Identity
    columns `controller, checkpoint` (None for a baseline, `final` for a run dir), `split, location, growth_year,
    start_day, season_length`, then the result keys.
  - `<run_dir>` is a directory holding `manifest.json`, scored at its `final` checkpoint. `zero_controls` and
    `random` go through `envs.run_baseline`, `rb_*` through `run_rb`.
- **Per-step frame** (`record_steps`): `t, co2, temp, rh, u_*` (6), `u_rb_*` (residual only), `profit`, `cost_*`.
  - `t` is 0..T-1; co2 [ppm], temp [°C] and rh [%] from the state after the step; `u_*` (the applied absolute
    controls) and `u_rb_*` in upstream order: boil, co2, thscr, vent, lamp, blscr; `profit` and `cost_*` are the
    step's info values (reward-side diagnostics, not metrics).

## `greenleaf_control.selection`, `stats`, `tune_rb`, `run_grid`, `modal_app`
- `select_checkpoint(run_dir: Path) -> str`: the PROTOCOL §5 rule on VALIDATION; writes `run_dir/selected.json` once.
  - **Write-once:** an existing `selected.json` raises FileExistsError before any scoring.
  - **Candidates:** `ckpt_<steps>.zip` in numeric order of steps, then `ckpt_final.zip`. Any other `ckpt_*.zip` is
    ignored; none raises FileNotFoundError.
  - **Scoring:** each candidate is scored with `evaluate.run_policy(run_dir, name, s, 60, record_steps=<residual_lag>)`
    for s in `scenarios.VALIDATION`, in order. Seasons are always 60 days, whatever the manifest's season length.
  - **residual_lag:**
    - A candidate qualifies when, for every constraint c, the mean of its frames' `cost_<c>`, pooled over the three
      seasons, is ≤ `config["budgets"][c] × 1.10`.
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
  - `TUNE_SPACE: dict[str, tuple[float, float]]` (12 setpoints, PROTOCOL §3).
  - `rb_return(params, scenarios, season_length=60) -> float`.
  - CLI `--trials N --jobs J --seed 0 (--smoke | --registered <row>)`.
- `run_grid`: `main(argv=None, *, ledger_path=ledger.LEDGER) -> int`. Module constants `CPU=4.0`, `MEMORY_MB=8192`,
  `TIMEOUT_S=14400`, `USD_PER_CORE_S=0.0000131`, `USD_PER_GIB_S=0.00000222`, `PRICES_AS_OF="2026-10-28"`
  (modal.com/pricing; one core = 2 vCPU).
  - `--registered <row> --backend {local,modal} [--parallel 2]`:
    - Before any launch, refuses an unknown row, a method not in `envs.METHODS`, a config key outside `CONFIG_KEYS`
      (train's `penalty_weight, budget_frac, season_length, n_envs, rb`) and any scenario in `scenarios.TEST`.
    - Then runs `python -m greenleaf_control.train --method M [--<key> V ...] --seed S --timesteps T --registered
      <row>` once per arm × seed, `--parallel` at a time, and exits 1 naming the arm and seed of any failure.
    - run_grid writes no ledger row; each train run records its own.
    - `--backend modal` refuses until #19 (remote runs can't record ledger rows, fingerprints or cost yet).
  - `--issue N --arms '<JSON list of {name, method, config}>' --seeds S [S ...] --timesteps T --dry [--parallel 2]`,
    run before registering. It writes nothing and prints:
    - the run count;
    - the worst case: CPU-hours = runs × TIMEOUT_S × CPU, and $ = runs × TIMEOUT_S × (CPU × USD_PER_CORE_S +
      MEMORY_MB/1024 × USD_PER_GIB_S). A true bound per attempt, because Modal bills max(request, usage) and
      `train_remote`'s limits equal its requests;
    - the wall clock from the newest `finished` run-end (wall_s / timesteps × T, × ceil(runs / parallel) locally).
- `modal_app` (import contacts nothing; nothing calls it before #19):
  - `app = modal.App("greenleaf")`;
  - image: `debian_slim(python_version="3.12")` + `apt_install("git")` + `uv_sync(<absolute control/>, frozen=True)` +
    `add_local_python_source("greenleaf_control")`;
  - `train_remote(argv: list[str]) -> str`, with hard limits `cpu=(4.0, 4.0)`, `memory=(8192, 8192)` ((request,
    limit) pairs) and `timeout=4*3600`;
  - volume `greenleaf-runs` (`create_if_missing=True`) mounted at `/runs`.

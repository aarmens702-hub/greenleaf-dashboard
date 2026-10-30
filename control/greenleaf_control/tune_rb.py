"""Tuning the rule-based controller: its 12 setpoints, Optuna TPE, on TRAIN_EVAL (PROTOCOL §3).

Every trial plays the controller with configs/rule_based_default.yml's params, the trial's 12
setpoints over them, on each TRAIN_EVAL season and scores the mean metrics paper_return. Trial 0 is
the defaults. --jobs is Optuna's n_jobs; a run is deterministic at --jobs 1 only. A trial with an
invalid season is FAIL, never scored, and the tune goes on; the tuned params are the best COMPLETE
trial's, and a tune with none crashes. Any other exception crashes the run.

- --smoke runs min(N, 2) trials on 1-day seasons into <runs-dir>/smoke/tune_rb-<UTC stamp>-<hex>
  and records no ledger row.
- --registered <row> runs that row's rb_tuned arm on 60-day seasons into
  <runs-dir>/<row>/<arm>/seed<S>, records its run and run-end rows, and writes the tuned params to
  configs/rule_based_tuned.yml, once. The row's scenarios are where the result is scored
  (["validation"]); the tune itself always runs on TRAIN_EVAL.

Both write manifest.json, trials.csv (Optuna's trials_dataframe) and rule_based_tuned.yml
({"GreenLightEnv": params}, as load_rb_params reads it) into the run dir.
"""

import argparse
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import optuna
import yaml

from greenleaf_control import evaluate, ledger, manifest, rule_based, scenarios, train

# The 12 setpoints and their [lo, hi]; every other controller key stays at its default.
TUNE_SPACE = {
    "temp_setpoint_day": (16, 24),  # °C
    "temp_setpoint_night": (14, 20),  # °C
    "heat_deadzone": (1, 10),  # °C over the heating setpoint before venting for heat
    "co2_day": (400, 1200),  # ppm
    "rh_max": (70, 90),  # %, the vents' humidity threshold
    "t_vent_off": (0, 4),  # °C under the heating setpoint where the vents close
    "thScrSpDay": (0, 15),  # °C outdoors, below which the screen closes by day
    "thScrSpNight": (0, 15),  # °C outdoors, by night
    "thScrDeadZone": (1, 8),  # °C over the heating setpoint before the screen opens for heat
    "lamps_on": (0, 6),  # hour of day
    "lamps_off": (12, 24),  # hour of day
    "lamps_off_sun": (200, 600),  # W/m² of sun above which the lamps switch off
}
SEASON_LENGTH = 60  # days; a registered tune scores full seasons
SMOKE_TRIALS = 2
SMOKE_DAYS = 1


class InvalidSeason(RuntimeError):
    """A season metrics marks invalid (a solver failure, or a short season): never scored."""


def rb_return(params: dict, scenarios: list[dict], season_length: int = 60) -> float:
    """The mean metrics paper_return of the controller with params over scenarios; an invalid
    season raises InvalidSeason: it is never scored."""
    returns = []
    for s in scenarios:
        result = evaluate.run_rb(params, s, season_length)[0]
        if not result["valid"]:
            raise InvalidSeason(
                f"invalid season {s}: solver_failures {result['solver_failures']}, steps"
                f" {result['steps']} of {season_length} days"
            )
        returns.append(result["paper_return"])
    return float(np.mean(returns))


def _tune(trials: int, jobs: int, seed: int, season_length: int) -> optuna.Study:
    """A seeded TPE study maximizing rb_return on TRAIN_EVAL, trial 0 at the defaults."""
    defaults = rule_based.load_rb_params("default")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial({key: defaults[key] for key in TUNE_SPACE})

    def objective(trial: optuna.Trial) -> float:
        tuned = {key: trial.suggest_float(key, lo, hi) for key, (lo, hi) in TUNE_SPACE.items()}
        return rb_return({**defaults, **tuned}, scenarios.TRAIN_EVAL, season_length)

    # Only an invalid season fails a trial and lets the tune go on; anything else crashes it.
    study.optimize(objective, n_trials=trials, n_jobs=jobs, catch=(InvalidSeason,))
    return study


def _registration(
    row_id: str, trials: int, seed: int, jobs: int, ledger_path: Path
) -> tuple[dict, dict]:
    """The registered row and its rb_tuned arm; refuses a tune that isn't exactly its
    registration, and a second registered tune (configs/rule_based_tuned.yml is written once)."""
    found = [r for r in ledger.rows(kind="registered", path=ledger_path) if r["row_id"] == row_id]
    if not found:
        raise ValueError(f"--registered {row_id}: no registered row with that id in {ledger_path}")
    [row] = found
    arms = [a for a in row["arms"] if a["method"] == "rb_tuned"]
    if len(arms) != 1:
        raise ValueError(
            f"{row_id}: tune_rb runs exactly one arm of method rb_tuned; its arms are {row['arms']}"
        )
    [arm] = arms
    extra = sorted(set(arm["config"]) - {"trials"})
    if extra:
        raise ValueError(
            f"{row_id}: arm {arm['name']}'s config keys {extra} are not tune_rb's; an rb_tuned"
            " arm's config is exactly {'trials': N}"
        )
    if trials != arm["config"].get("trials"):
        raise ValueError(
            f"{row_id}: --trials {trials} differs from arm {arm['name']}'s trials"
            f" {arm['config'].get('trials')!r}"
        )
    if row["scenarios"] != ["validation"]:
        raise ValueError(
            f"{row_id}: an rb_tuned result is scored on exactly ['validation'] (PROTOCOL §2, §9;"
            f" the tune itself runs on TRAIN_EVAL); its scenarios are {row['scenarios']}"
        )
    if row["seeds"] != [0] or seed != 0:
        raise ValueError(
            f"{row_id}: rb_tuned is tuned at seed 0 (PROTOCOL §3); its seeds are {row['seeds']}"
            f" and --seed is {seed}"
        )
    if row["timesteps"] != 0:
        raise ValueError(
            f"{row_id}: rb_tuned trains nothing; its timesteps must be 0, not {row['timesteps']}"
        )
    if jobs != 1:
        raise ValueError(
            f"{row_id}: --jobs {jobs}; a registered tune runs at --jobs 1, the only n_jobs at which"
            " seed 0 is deterministic"
        )
    tuned = rule_based.CONFIGS / "rule_based_tuned.yml"
    if tuned.exists():
        raise ValueError(f"{tuned} already exists: a registered tune writes it once")
    return row, arm


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.tune_rb")
    p.add_argument("--trials", type=int, required=True, help=f"--smoke caps it at {SMOKE_TRIALS}")
    p.add_argument("--jobs", type=int, default=1, help="Optuna's n_jobs (default 1, deterministic)")
    p.add_argument("--seed", type=int, required=True, help="the TPE sampler's seed")
    p.add_argument("--runs-dir", type=Path, default=train.RUNS, help="default control/runs")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke", action="store_true", help=f"{SMOKE_DAYS}-day seasons, no ledger row"
    )
    mode.add_argument("--registered", metavar="ROW", help="the registered row this tune belongs to")
    return p


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    args = _parser().parse_args(argv)
    if args.smoke:
        trials, season_length = min(args.trials, SMOKE_TRIALS), SMOKE_DAYS
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.runs_dir / "smoke" / f"tune_rb-{stamp}-{secrets.token_hex(4)}"
        row, arm = None, None
    else:
        row, arm = _registration(args.registered, args.trials, args.seed, args.jobs, ledger_path)
        trials, season_length = args.trials, SEASON_LENGTH
        run_dir = args.runs_dir / row["row_id"] / arm["name"] / f"seed{args.seed}"
    config = {
        "method": "rb_tuned",
        "seed": args.seed,
        "trials": trials,
        "jobs": args.jobs,
        "season_length": season_length,
        "tuned_on": "train_eval",
        "tune_space": TUNE_SPACE,
        "run_dir": str(run_dir),
    }

    manifest.write_manifest(
        run_dir,
        config=config,
        issue=0 if row is None else row["issue"],
        registered=None if row is None else row["row_id"],
        smoke=args.smoke,
    )
    start = time.monotonic()
    run, status, failed = None, "crashed", None
    try:
        if row is not None:
            run = ledger.run_start(
                registered=row["row_id"],
                arm=arm["name"],
                seed=args.seed,
                run_dir=str(run_dir),
                path=ledger_path,
            )
        study = _tune(trials, args.jobs, args.seed, season_length)
        failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
        study.trials_dataframe().to_csv(run_dir / "trials.csv", index=False)
        # best_params raises ValueError when no trial is COMPLETE: no tuned params, a crash.
        params = {**rule_based.load_rb_params("default"), **study.best_params}
        text = yaml.safe_dump({"GreenLightEnv": params}, sort_keys=False)
        (run_dir / "rule_based_tuned.yml").write_text(text)
        if row is not None:
            with (rule_based.CONFIGS / "rule_based_tuned.yml").open("x") as f:  # written once
                f.write(text)
        status = "finished"
    finally:  # a crash is recorded in the manifest and the run-end, then propagates
        end = {
            "status": status,
            "wall_s": time.monotonic() - start,
            "timesteps": 0,  # nothing trains
            # The FAIL trials, each an invalid season; None if the tune itself crashed.
            "solver_failures": failed,
            "cost_usd": None,
        }
        manifest.finish_manifest(run_dir, **end)
        if run is not None:
            diagnostics = dict.fromkeys(ledger.DIAGNOSTIC_KEYS)  # no PPO update to diagnose
            ledger.run_end(run=run["row_id"], **end, diagnostics=diagnostics, path=ledger_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

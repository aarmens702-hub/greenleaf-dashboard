"""Tests for greenleaf_control.export_replay (issue #24, D2; issue #48, D3b-1): a recorded season
as hourly JSON for the demo site (D3).

export(frame, result, states, controls, params, *, controller, scenario, stamp, split, run) -> dict
builds the payload {"meta", "hourly", "totals"} from one season: evaluate's per-step frame (one row
per step t = 0..T-1, read from the state after step t), its result dict, and the same season's raw
trace (states (T + 1, nx), controls (T, 6), as metrics.trace_from_infos gives them) and model
params, which feed the running counters through metrics, the grader. scenario is a split's
scenario dict (location, growth_year, start_day); stamp holds the export code's "commit", "dirty"
and "gl_gym"; split is the split's name; run is a run dir's provenance (None for a baseline).
main(argv=None, *, ledger_path=ledger.LEDGER) -> int is the CLI.

The conventions pinned here. A season of T steps has H = T / 4 hours; hour h is steps 4h..4h+3.
- co2 [ppm], temp [°C] and rh [%] at hour h are the frame's row 4h + 3: the state at the END of the
  hour, states[4h + 4], rounded to 3 significant digits (an error of at most half a unit in the
  3rd significant digit, and no more digits than 3).
- Each actuator u_* at hour h is the mean of frame rows 4h..4h+3, rounded to 3 decimals.
- profit [€/m²] and viol_<c> [h] at hour h are metrics on the season's first 4h + 4 steps:
  metrics.episode_metrics(states[:4h + 5], controls[:4h + 4], params, season_steps=4h + 4)
  (metrics' convention: step t runs from states[t] to states[t + 1] and is graded on
  states[t + 1]), rounded to 4 decimals. So they are read at the same instant as the climate, and
  the last hour's are the season's own.
- totals is the result dict, unrounded.
- meta is exactly {controller, scenario, split, run, commit, dirty, gl_gym, note, bounds, units};
  note is "a simulator replay of a <split> season, not the final result".
- The file is <controller>_<location>_<growth_year>_<start_day>.json, written as
  json.dumps(payload, allow_nan=False, separators=(",", ":")). A baseline's controller is its
  name; a run dir's is <arm>_seed<seed> (residual_w05_seed100), not the dir's name.

A run dir is a registered run (issue #48). Its lookup reads ledger_path, and only reads it:
- the run row whose run_dir resolves to the given dir (both sides resolved); none is refused,
  naming the dir and "not a registered run";
- its registration's result rows: exactly one, whose selected names the run; otherwise refused,
  naming the run id and the count;
- it plays at that result's selected[run id] (e.g. "1500000", not "final") through
  evaluate.run_policy(..., return_trace=True), and meta.run is {id, registered, result, arm, seed,
  checkpoint, commit}, commit from the run dir's manifest.json (the run's training commit, not the
  export's stamp).
Refusals come in this order, all before any env: --final, a test split, a non-baseline with no
manifest.json, the lookup, an existing file. A baseline never reads the ledger.

Every ledger here is a tmp one, built with ledger.register, run_start, run_end and record_result as
test_selection_cli.py builds them, and every CLI call passes main its ledger_path.

Known answers used throughout:
- One day is 96 steps (86,400 s / 900 s), so 24 hours; 60 days are 5,760 steps, 1,440 hours.
- A violating step is 900 s = 0.25 h, so viol_<c> at hour h is 0.25 h x the out-of-bounds steps
  among the first 4h + 4.
- rb default never leaves the CO₂ band on PAPER (the golden 60-day season's viol_hours_co2 is 0.0);
  co2_day 2,500 doses toward 2,500 ppm by day, above the 1,600 ppm bound, so it does.

The 2012 weather is sealed (PROTOCOL §8). Every test that could build an env runs under a tripwire
on pandas.read_csv that fails before any *2012.csv is read; the CLI refusals also run with every
env builder (and evaluate's run_rb, run_policy) replaced by one that fails, so a refusal must come
before any env or file. Nothing here may write the real ledger, control/runs/ or control/results/;
the last test checks all three are as they were when this module was collected.
"""

import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenleaf_control import (
    envs,
    evaluate,
    ledger,
    manifest,
    metrics,
    rule_based,
    scenarios,
    train,
)

# greenleaf_control.export_replay is imported inside each test body, so while it is missing each
# test FAILS rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"
REAL_RESULTS = CONTROL / "results"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger, runs and results, read when this module is collected, before any test in it runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)
REAL_RESULTS_BEFORE = _listing(REAL_RESULTS)

N_DAY = 96  # 86,400 s / 900 s
PAPER = scenarios.PAPER  # Amsterdam 2010, start day 59
CONSTRAINTS = metrics.CONSTRAINTS  # ("co2", "temp", "rh")
RB_DEFAULT = rule_based.load_rb_params("default")
# Upstream's control order: uBoil, uCO2, uThScr, uVent, uLamp, uBlScr.
U_COLS = ["u_boil", "u_co2", "u_thscr", "u_vent", "u_lamp", "u_blscr"]
CLIMATE = ["co2", "temp", "rh"]
HOURLY_KEYS = ["t_hour", *CLIMATE, *U_COLS, "profit", *(f"viol_{c}" for c in CONSTRAINTS)]
META_KEYS = {
    "controller",
    "scenario",
    "split",
    "run",
    "commit",
    "dirty",
    "gl_gym",
    "note",
    "bounds",
    "units",
}
NOTE_PAPER = "a simulator replay of a paper season, not the final result"
NOTE_VALIDATION = "a simulator replay of a validation season, not the final result"
UNITS = {"co2": "ppm", "temp": "°C", "rh": "%", "profit": "€/m²", "hours": "h"}
# PROTOCOL §1's bounds: CO₂ 300-1,600 ppm, air temperature 15-34 °C, RH 50-85 %.
BOUNDS_JSON = {"co2": [300.0, 1600.0], "temp": [15.0, 34.0], "rh": [50.0, 85.0]}
# A stamp export copies into meta: a made-up commit (so a stamp read from git instead shows), and
# the installed gl-gym commit (so a builder that reads it from the install agrees).
STAMP = {
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "dirty": True,
    "gl_gym": manifest._versions()["gl-gym"],
}
TEST_SPLITS = ("amsterdam_2012", "london_2012", "abbotsford_2012")
CLI_FLAGS = ("--controller", "--split", "--out", "--final")
PAPER_FILE = "rb_default_Amsterdam_2010_59.json"  # <controller>_<location>_<year>_<start day>
MAX_BYTES = 250_000  # the plan's ceiling, 250 KB, in decimal kilobytes

# A registered run, shaped like E6's seed 100 (#36): arm residual_w05, its result row selecting it
# at checkpoint "1500000". Its manifest's commit is made up, so it differs from HEAD and STAMP's.
ARM = "residual_w05"
CHECKPOINT = "1500000"
RUN_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
RUN_FILE = "residual_w05_seed100_Amsterdam_2010_59.json"  # <arm>_seed<seed>_<location>_<year>_<day>
# meta.run as export is handed it: E6's ids for seed 100, with the made-up commit.
RUN = {
    "id": "run-5de00091",
    "registered": "registered-f2fd71e9",
    "result": "result-d1f0b8e7",
    "arm": ARM,
    "seed": 100,
    "checkpoint": CHECKPOINT,
    "commit": RUN_COMMIT,
}
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}
# A result row's per-arm numbers as selection hands them over (the ledger adds crashed, spend_usd).
PER_ARM = {
    "iqm": 0.077,
    "ci_low": -0.1,
    "ci_high": 0.2,
    "guards": {"viol_hours_temp": {"iqm": 1.0, "ci_low": 0.5, "ci_high": 1.5}},
    "invalid_episodes": 0,
}


@pytest.fixture
def no_2012_reads(monkeypatch) -> None:
    """Fail loudly if the test reads any 2012 weather file. pandas.read_csv is the only file read
    upstream's weather loader makes (its own year, and the next year's file past the year's end);
    the check runs before the real read, and every other file goes through the real one."""
    real_read_csv = pd.read_csv

    def read_csv(filepath_or_buffer, *args, **kwargs):
        if Path(str(filepath_or_buffer)).name.endswith("2012.csv"):
            raise AssertionError(f"the sealed 2012 test weather was read: {filepath_or_buffer}")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)


def _rb_season(params: dict, days: int) -> dict:
    """One rule-based season on PAPER as export's inputs: run_rb's result and per-step frame, and
    the same season's raw trace and model params, replayed the way run_rb plays it (the controller
    is deterministic, so it is the same season: its frame's u_* are its controls exactly)."""
    result, frame = evaluate.run_rb(params, PAPER, days, record_steps=True)
    env = envs.SolverFailureFlag(
        envs.make_raw_env(scenario=PAPER, season_length=days, normalize_actions=False)
    )
    controller = rule_based.make_controller(params)
    env.reset()
    infos, done = [], False
    while not done:
        _, _, terminated, truncated, info = env.step(rule_based.rb_action(env, controller))
        infos.append(info)
        done = terminated or truncated
    states, controls = metrics.trace_from_infos(infos)
    p = np.asarray(env.unwrapped.p, dtype=np.float64)
    env.close()
    return {"frame": frame, "result": result, "states": states, "controls": controls, "params": p}


@pytest.fixture(scope="module")
def seasons() -> dict:
    """1-day PAPER seasons (96 steps, 24 hours) of rb default and of rb default with co2_day 2,500.
    Built from evaluate and metrics (which exist), so this fixture never ERRORs. About 0.5 s."""
    return {
        "rb_default": _rb_season(RB_DEFAULT, 1),
        "co2_day_2500": _rb_season({**RB_DEFAULT, "co2_day": 2500}, 1),
    }


@pytest.fixture(scope="module")
def paper_60() -> tuple[dict, pd.DataFrame]:
    """run_rb(default params, PAPER, 60 days, record_steps=True): the reference for the CLI's file.
    About 3 s."""
    return evaluate.run_rb(RB_DEFAULT, PAPER, 60, record_steps=True)


def _reject_constant(name: str):
    raise AssertionError(f"{name} in the JSON: the payload must be finite numbers only")


def _exported(
    export_replay,
    season: dict,
    controller: str = "rb_default",
    split: str = "paper",
    run: dict | None = None,
) -> dict:
    """export's payload for a season, through the file's own JSON round trip (dumps then loads).
    While export takes no split or run, this fails on that assertion, not on a TypeError."""
    keywords = set(inspect.signature(export_replay.export).parameters)
    assert {"split", "run"} <= keywords, f"export takes no split or run: {sorted(keywords)}"
    payload = export_replay.export(
        season["frame"],
        season["result"],
        season["states"],
        season["controls"],
        season["params"],
        controller=controller,
        scenario=PAPER,
        stamp=STAMP,
        split=split,
        run=run,
    )
    text = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    return json.loads(text, parse_constant=_reject_constant)


def _half_sig3(v: float) -> float:
    """Half a unit in v's 3rd significant digit: the most rounding to 3 significant digits moves v
    (21.37 °C -> 21.4 moves 0.03 <= 0.05; 812.3 ppm -> 812 moves 0.3 <= 0.5)."""
    return 0.0 if v == 0 else 0.5 * 10 ** (math.floor(math.log10(abs(v))) - 2)


def _assert_climate_is_the_end_of_hour_state(hourly: dict, frame: pd.DataFrame) -> None:
    """hourly co2, temp, rh at hour h are frame row 4h + 3 to 3 significant digits, with no more
    digits than 3 (the file stays small)."""
    for c in CLIMATE:
        want = frame[c].to_numpy(dtype=np.float64)[3::4]
        got = hourly[c]
        assert len(got) == len(want), c
        for h, (r, v) in enumerate(zip(got, want, strict=True)):
            # 1e-6 of slack: a 3-digit rounding of a float32 or float64 copy of v both pass.
            assert abs(r - v) <= _half_sig3(v) * (1 + 1e-6), f"{c} hour {h}: {r} for {v}"
            assert float(f"{r:.3g}") == r, f"{c} hour {h}: {r} has more than 3 significant digits"


def _assert_actuators_are_the_hourly_means(hourly: dict, frame: pd.DataFrame) -> None:
    """Each u_* at hour h is the mean of frame rows 4h..4h+3 within 5e-4 (3 decimals), and has no
    more than 3 decimals."""
    n_hours = len(frame) // 4
    for col in U_COLS:
        want = frame[col].to_numpy(dtype=np.float64).reshape(n_hours, 4).mean(axis=1)
        got = hourly[col]
        assert len(got) == n_hours, col
        for h, (r, v) in enumerate(zip(got, want, strict=True)):
            # 3 decimals move a mean at most 5e-4; 1e-6 of slack for a float32 mean
            assert abs(r - v) <= 5e-4 + 1e-6, f"{col} hour {h}: {r} for the hourly mean {v}"
            assert round(r, 3) == r, f"{col} hour {h}: {r} has more than 3 decimals"


def _prefix(season: dict, k: int) -> dict:
    """metrics on the season's first k steps: states[:k + 1] (the initial state and the k states
    after each step), controls[:k]."""
    return metrics.episode_metrics(
        season["states"][: k + 1], season["controls"][:k], season["params"], season_steps=k
    )


def _assert_profit_within_4_decimals(r: float, v: float, where: str) -> None:
    # 4 decimals move a profit [€/m²] at most 5e-5
    assert abs(r - v) <= 5e-5 * (1 + 1e-9), f"{where}: {r} for {v}"
    assert round(r, 4) == r, f"{where}: {r} has more than 4 decimals"


def _trip(monkeypatch, export_replay, names) -> None:
    """Replace the env builders and season players named (on envs, train and evaluate, and on
    export_replay itself in case it imported them by name) with ones that fail."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any env")

        return refuse

    homes = {
        "make_raw_env": envs,
        "make_env": envs,
        "run_baseline": envs,
        "build_vec_env": train,
        "run_rb": evaluate,
        "run_policy": evaluate,
    }
    for name in names:
        monkeypatch.setattr(homes[name], name, trip(f"{homes[name].__name__}.{name}"))
        if hasattr(export_replay, name):
            monkeypatch.setattr(export_replay, name, trip(f"export_replay.{name}"))


ALL_ENV_ROUTES = (
    "make_raw_env",
    "make_env",
    "run_baseline",
    "build_vec_env",
    "run_rb",
    "run_policy",
)


def _refusal(excinfo, capsys) -> str:
    """A refusal's text: the exception's message plus stderr (argparse's error goes there). A
    SystemExit's int code is left out: argparse's 2 is not part of the message."""
    value = excinfo.value
    if isinstance(value, SystemExit):
        assert value.code not in (0, None), "a refusal must exit nonzero"
    message = "" if isinstance(value, SystemExit) and isinstance(value.code, int) else str(value)
    return message + capsys.readouterr().err


def _names_count(text: str, count: int, tmp_path: Path) -> bool:
    """count is in text as a number of its own, outside the tmp paths (pytest-1/ could be one)."""
    return re.search(rf"\b{count}\b", text.replace(str(tmp_path), "<tmp>")) is not None


def _argv(controller: str, split: str, out: Path, *extra: str) -> list[str]:
    return ["--controller", controller, "--split", split, "--out", str(out), *extra]


def _main(export_replay, argv: list[str], ledger_path: Path) -> int:
    """export_replay.main(argv, ledger_path=ledger_path): every CLI call here reads a tmp ledger,
    never the real one. While main takes no ledger_path, this fails on that assertion, not on a
    TypeError."""
    params = inspect.signature(export_replay.main).parameters
    assert "ledger_path" in params, f"main takes no ledger_path: {list(params)}"
    return export_replay.main(argv, ledger_path=ledger_path)


def _fake_run_dir(path: Path) -> Path:
    """A dir that looks like a run dir to a CLI check: it holds a manifest.json with a config and
    the run's training commit (RUN_COMMIT, made up)."""
    path.mkdir(parents=True)
    manifest_json = {"commit": RUN_COMMIT, "config": {"method": "residual"}}
    (path / "manifest.json").write_text(json.dumps(manifest_json))
    return path


def _register(path: Path, issue: int) -> dict:
    """A registration on the tmp ledger of one arm, residual_w05, over seeds 100 and 101."""
    return ledger.register(
        issue=issue,
        hypothesis="residual_w05 beats rb_tuned on validation profit",
        arms=[{"name": ARM, "method": "residual", "config": {}}],
        seeds=[100, 101],
        timesteps=2_000_000,
        scenarios=["validation"],
        primary="profit_eur_m2",
        guards=["viol_hours_temp"],
        rule="adopt if the profit IQM CI excludes rb_tuned's",
        budget={"cpu_hours": 1.0, "usd": 0.0, "backend": "local"},
        path=path,
    )


def _runs(path: Path, root: Path, reg: dict, *, hop: bool = False) -> dict[int, tuple[dict, Path]]:
    """reg's runs at seeds 100 and 101, as train records them: a fake run dir
    root/<registered>/residual_w05/seed<S>, a run row naming it, and a finished run-end. With hop,
    the run row names its dir through a ".." (absolute but not resolved, as /var is to
    /private/var). Returns {seed: (run row, run dir)}."""
    runs = {}
    for seed in reg["seeds"]:
        run_dir = _fake_run_dir(root / reg["row_id"] / ARM / f"seed{seed}")
        named = run_dir.parent / ".." / ARM / run_dir.name if hop else run_dir
        run = ledger.run_start(
            registered=reg["row_id"], arm=ARM, seed=seed, run_dir=str(named), path=path
        )
        ledger.run_end(
            run=run["row_id"],
            status="finished",
            wall_s=12.5,
            timesteps=reg["timesteps"],
            solver_failures=0,
            cost_usd=None,
            diagnostics=DIAGNOSTICS,
            path=path,
        )
        runs[seed] = (run, run_dir)
    return runs


def _e6(tmp_path: Path, results: list[dict], *, hop: bool = False) -> tuple:
    """A tmp ledger shaped like E6's (#36), after a decoy experiment, written in this order:
    - the decoy (issue 35): residual_w05 at seeds 100 and 101 in its own run dirs, and one result
      row selecting both at "1750000";
    - E6 (issue 36): residual_w05 at seeds 100 and 101 (hop as in _runs), then one result row per
      entry of results, each {seed: checkpoint} turned into its selected {run row_id: checkpoint}.
    A lookup that matches a run by its dir's name, or by arm and seed, or counts every result row,
    finds the decoy's. Returns (ledger path, E6's registration, E6's {seed: (run row, run dir)},
    E6's result rows)."""
    path, root = tmp_path / "ledger.jsonl", tmp_path / "runs"
    decoy = _register(path, 35)
    decoy_runs = _runs(path, root, decoy)
    ledger.record_result(
        registered=decoy["row_id"],
        per_arm={ARM: PER_ARM},
        selected={run["row_id"]: "1750000" for run, _ in decoy_runs.values()},
        path=path,
    )
    reg = _register(path, 36)
    runs = _runs(path, root, reg, hop=hop)
    rows = [
        ledger.record_result(
            registered=reg["row_id"],
            per_arm={ARM: PER_ARM},
            selected={runs[seed][0]["row_id"]: ckpt for seed, ckpt in by_seed.items()},
            path=path,
        )
        for by_seed in results
    ]
    return path, reg, runs, rows


# --- export: the round trip (Done when) ---------------------------------------------------------


def test_export_hourly_climate_is_the_end_of_hour_state_to_3_significant_digits(seasons):
    """Done when (round trip): on a 1-day PAPER rb default season, the JSON round trip of export's
    payload gives 24 hourly co2, temp and rh, each the frame's row 4h + 3 (the state at the end of
    hour h) to 3 significant digits, and no more digits than 3.

    Non-vacuous: the 15-minute state (row 4h) is outside that rounding for some hour, so the check
    tells the end of the hour from its start."""
    import greenleaf_control.export_replay as export_replay

    season = seasons["rb_default"]
    hourly = _exported(export_replay, season)["hourly"]

    _assert_climate_is_the_end_of_hour_state(hourly, season["frame"])
    starts = {c: season["frame"][c].to_numpy(dtype=np.float64)[0::4] for c in CLIMATE}
    assert any(
        abs(r - v) > _half_sig3(v)
        for c in CLIMATE
        for r, v in zip(hourly[c], starts[c], strict=True)
    ), "the hour's first row fits as well as its last, so the sampling isn't pinned"


def test_export_hourly_actuators_are_the_hourly_means_to_3_decimals(seasons):
    """Done when (round trip): on a 1-day PAPER rb default season, each of the six actuators at
    hour h is the mean of the hour's four 15-minute steps (frame rows 4h..4h+3) within 5e-4, the
    most rounding to 3 decimals moves it, and has no more than 3 decimals.

    Non-vacuous: some hourly mean differs from the hour's last step by more than 1e-3, so a sampled
    value instead of the mean fails."""
    import greenleaf_control.export_replay as export_replay

    season = seasons["rb_default"]
    hourly = _exported(export_replay, season)["hourly"]

    _assert_actuators_are_the_hourly_means(hourly, season["frame"])
    u = season["frame"][U_COLS].to_numpy(dtype=np.float64)
    means = u.reshape(24, 4, 6).mean(axis=1)
    assert np.any(np.abs(means - u[3::4]) > 1e-3), "every hour is constant: means prove nothing"


def test_export_totals_are_the_result_exactly_and_the_metrics_exactly(seasons):
    """Done when (round trip): the JSON round trip of totals is the result dict passed in, exactly
    (unrounded: Python floats survive json.dumps and json.loads bit for bit), and its metrics keys
    are metrics.episode_metrics on the whole season (97 states, 96 controls, the model's params,
    96 season steps), exactly."""
    import greenleaf_control.export_replay as export_replay

    season = seasons["rb_default"]
    totals = _exported(export_replay, season)["totals"]

    assert totals == season["result"]
    graded = metrics.episode_metrics(
        season["states"], season["controls"], season["params"], season_steps=N_DAY
    )
    assert {k: totals[k] for k in graded} == graded


# --- export: the running counters (Done when) ---------------------------------------------------

# Per season, the constraints whose counter must both rise in some hour (a violating step) and
# stay flat in another (none): rb default leaves the temperature and RH bands on its first day but
# never the CO₂ band; co2_day 2,500 leaves the CO₂ band by day.
COUNTER_CASES = [
    pytest.param("rb_default", ("temp", "rh"), id="rb-default"),
    pytest.param("co2_day_2500", ("co2",), id="rb-co2-day-2500"),
]


@pytest.mark.parametrize(("name", "moving"), COUNTER_CASES)
def test_export_running_counters_are_metrics_on_each_hours_prefix(name, moving, seasons):
    """Done when: each hour's running profit and viol_<c> are metrics on that hour's prefix, and
    the last hour's are the season's.

    - Every hour h of a 1-day PAPER season: profit within 5e-5 (4 decimals) of, and each viol_<c>
      exactly, metrics.episode_metrics(states[:4h + 5], controls[:4h + 4], params) (viol hours are
      multiples of 0.25 h, exact at 4 decimals).
    - By hand, hours 0, 11 and 23 (states[:5] / controls[:4], [:49] / [:48], [:97] / [:96]): their
      prefixes graded literally, and each viol_<c> is 0.25 h x the frame's out-of-bounds rows
      among 0..4h+3 (metrics.BOUNDS; on a bound is in bounds).
    - Hour 23's profit is the result's profit_eur_m2 within 5e-5, and its viol_<c> the result's
      viol_hours_<c> exactly.

    Non-vacuous: in this season each constraint in `moving` has an hour its counter rises and an
    hour it stays flat."""
    import greenleaf_control.export_replay as export_replay

    season = seasons[name]
    hourly = _exported(export_replay, season)["hourly"]

    want = [_prefix(season, 4 * h + 4) for h in range(24)]
    for h, m in enumerate(want):
        _assert_profit_within_4_decimals(hourly["profit"][h], m["profit_eur_m2"], f"hour {h}")
        for c in CONSTRAINTS:
            assert hourly[f"viol_{c}"][h] == m[f"viol_hours_{c}"], f"viol_{c} hour {h}"

    frame = season["frame"]
    for h, (n_states, n_controls) in {0: (5, 4), 11: (49, 48), 23: (97, 96)}.items():
        m = metrics.episode_metrics(
            season["states"][:n_states],
            season["controls"][:n_controls],
            season["params"],
            season_steps=n_controls,
        )
        _assert_profit_within_4_decimals(hourly["profit"][h], m["profit_eur_m2"], f"hour {h}")
        for c in CONSTRAINTS:
            low, high = metrics.BOUNDS[c]
            v = frame[c].to_numpy(dtype=np.float64)[: 4 * h + 4]
            # 900 s per violating step / 3,600 s per hour = 0.25 h each
            by_hand = 0.25 * int(np.count_nonzero((v < low) | (v > high)))
            assert hourly[f"viol_{c}"][h] == by_hand == m[f"viol_hours_{c}"], f"viol_{c} hour {h}"

    result = season["result"]
    _assert_profit_within_4_decimals(hourly["profit"][-1], result["profit_eur_m2"], "last hour")
    for c in CONSTRAINTS:
        assert hourly[f"viol_{c}"][-1] == result[f"viol_hours_{c}"], c

    for c in moving:
        rises = np.diff([0.0, *(m[f"viol_hours_{c}"] for m in want)])
        assert np.any(rises > 0) and np.any(rises == 0), f"viol_{c} never rises or never rests"


# --- export: the meta block and the shape -------------------------------------------------------


META_CASES = [
    pytest.param("rb_default", "paper", None, NOTE_PAPER, id="baseline-paper"),
    pytest.param("residual_w05_seed100", "validation", RUN, NOTE_VALIDATION, id="run-validation"),
]


@pytest.mark.parametrize(("controller", "split", "run", "note"), META_CASES)
def test_export_meta_names_the_season_its_split_its_run_and_its_code(
    controller, split, run, note, seasons
):
    """Done when: meta is exactly {controller, scenario, split, run, commit, dirty, gl_gym, note,
    bounds, units}:
    - controller as given; scenario the given scenario plus season_length in days (96 frame rows
      -> 1);
    - split the split's name as given; run as given: None for a baseline, the 7-key dict for a run
      dir (its commit is the run's own, RUN_COMMIT);
    - commit, dirty and gl_gym the stamp's (the export code's), so meta.commit is not run's commit;
    - note "a simulator replay of a <split> season, not the final result";
    - bounds metrics.BOUNDS as lists (PROTOCOL §1: 300-1,600 ppm, 15-34 °C, 50-85 %);
    - units co2 "ppm", temp "°C", rh "%", profit "€/m²", hours "h"."""
    import greenleaf_control.export_replay as export_replay

    payload = _exported(export_replay, seasons["rb_default"], controller, split=split, run=run)
    meta = payload["meta"]

    assert set(meta) == META_KEYS
    assert meta["controller"] == controller
    assert meta["scenario"] == {
        "location": "Amsterdam",
        "growth_year": 2010,
        "start_day": 59,
        "season_length": 1,
    }
    assert meta["split"] == split
    assert meta["run"] == run
    assert (meta["commit"], meta["dirty"], meta["gl_gym"]) == (
        STAMP["commit"],
        STAMP["dirty"],
        STAMP["gl_gym"],
    )
    assert meta["commit"] != RUN_COMMIT
    assert meta["note"] == note
    assert meta["bounds"] == BOUNDS_JSON
    assert meta["bounds"] == {c: list(bounds) for c, bounds in metrics.BOUNDS.items()}
    assert meta["units"] == UNITS


def test_export_is_finite_plain_json_with_one_value_per_hour_in_every_array(seasons):
    """export returns plain JSON: json.dumps(payload, allow_nan=False) takes it as is (no numpy
    arrays, no NaN or infinity), and the parsed payload is {"meta", "hourly", "totals"}. hourly has
    exactly t_hour, co2, temp, rh, the six u_*, profit and the three viol_<c>, each 24 long for a
    1-day season, and t_hour is 0..23."""
    import greenleaf_control.export_replay as export_replay

    payload = _exported(export_replay, seasons["rb_default"])

    assert set(payload) == {"meta", "hourly", "totals"}
    hourly = payload["hourly"]
    assert set(hourly) == set(HOURLY_KEYS)
    assert {k: len(v) for k, v in hourly.items()} == dict.fromkeys(HOURLY_KEYS, 24)
    assert hourly["t_hour"] == list(range(24))
    assert all(math.isfinite(x) for k in HOURLY_KEYS for x in hourly[k])


# --- The CLI: the written file (Done when) ------------------------------------------------------


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_writes_one_compact_60_day_paper_file_under_250_kb(tmp_path):
    """Done when: export_replay --controller rb_default --split paper --out <dir> returns 0 and
    writes exactly one file, rb_default_Amsterdam_2010_59.json (PAPER is one scenario), under
    250,000 bytes. It is json.dumps(payload, allow_nan=False, separators=(",", ":")) byte for byte
    (a trailing newline allowed): compact, and with no NaN or infinity. Its hourly arrays are 1,440
    long (60 days x 24 h) and t_hour is 0..1,439.

    Unrounded, the hourly arrays alone are about 255 KB, so the ceiling also catches a file that
    skips the rounding. About 4 s."""
    import greenleaf_control.export_replay as export_replay

    out = tmp_path / "site"
    out.mkdir()
    assert _main(export_replay, _argv("rb_default", "paper", out), tmp_path / "ledger.jsonl") == 0

    assert [p.name for p in out.iterdir()] == [PAPER_FILE]
    path = out / PAPER_FILE
    size = path.stat().st_size
    assert size < MAX_BYTES, f"{size:,} bytes"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text, parse_constant=_reject_constant)
    compact = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    assert text in (compact, compact + "\n"), "not written as compact json.dumps"
    hourly = payload["hourly"]
    assert {k: len(v) for k, v in hourly.items()} == dict.fromkeys(HOURLY_KEYS, 60 * 24)
    assert hourly["t_hour"] == list(range(60 * 24))


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_file_round_trips_the_season_and_its_totals_are_the_metrics(tmp_path, paper_60):
    """Done when (round trip, through the file): json.loads of the CLI's 60-day PAPER rb default
    file reproduces run_rb(default params, PAPER, 60, record_steps=True)'s season:
    - the hourly climate is its frame's row 4h + 3 to 3 significant digits;
    - the actuators are its hourly means within 5e-4;
    - totals is its result exactly: metrics' keys alone, or with paper_reward_sum too;
    - the last hour's profit is totals' profit_eur_m2 within 5e-5, its viol_<c> totals'
      viol_hours_<c> exactly;
    - meta: controller "rb_default", scenario PAPER with season_length 60, split "paper", run None
      (a baseline), the checkout's commit (git rev-parse HEAD), dirty a bool, and the note "a
      simulator replay of a paper season, not the final result".
    A baseline never reads the ledger: its ledger_path holds a line that isn't JSON, which
    ledger.rows would fail on.
    About 4 s, plus the fixture's 3 s."""
    import greenleaf_control.export_replay as export_replay

    want, frame = paper_60
    out = tmp_path / "site"
    out.mkdir()
    unreadable = tmp_path / "ledger.jsonl"
    unreadable.write_text("not a ledger row\n")
    assert _main(export_replay, _argv("rb_default", "paper", out), unreadable) == 0

    payload = json.loads(
        (out / PAPER_FILE).read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    hourly, totals, meta = payload["hourly"], payload["totals"], payload["meta"]
    _assert_climate_is_the_end_of_hour_state(hourly, frame)
    _assert_actuators_are_the_hourly_means(hourly, frame)
    graded = {k: v for k, v in want.items() if k != "paper_reward_sum"}
    assert totals in (graded, want)
    _assert_profit_within_4_decimals(hourly["profit"][-1], totals["profit_eur_m2"], "last hour")
    for c in CONSTRAINTS:
        assert hourly[f"viol_{c}"][-1] == totals[f"viol_hours_{c}"], c
    head = subprocess.run(
        ["git", "-C", str(manifest.REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert set(meta) == META_KEYS
    assert meta["controller"] == "rb_default"
    assert meta["scenario"] == {**PAPER, "season_length": 60}
    assert meta["split"] == "paper"
    assert meta["run"] is None
    assert meta["commit"] == head
    assert isinstance(meta["dirty"], bool)
    assert meta["note"] == NOTE_PAPER


# --- The CLI: the seal and the refusals (Done when; PROTOCOL §8) --------------------------------


def _controller(controller: str, tmp_path: Path) -> str:
    """controller as given, or for "<unregistered run dir>" a fake run dir (a manifest.json) that
    no ledger row names: tmp_path/runs/registered-00000000/residual_w05/seed100."""
    if controller != "<unregistered run dir>":
        return controller
    return str(_fake_run_dir(tmp_path / "runs" / "registered-00000000" / ARM / "seed100"))


@pytest.mark.parametrize("controller", ["rb_default", "<unregistered run dir>"])
@pytest.mark.parametrize("split", TEST_SPLITS)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_a_test_split_without_final(split, controller, tmp_path, monkeypatch, capsys):
    """Done when: --split <a test split> without --final exits non-zero, naming the seal.

    --split takes every split in scenarios.SPLITS, so argparse doesn't refuse it as an unknown
    choice; export_replay refuses it, with SystemExit (code not 0) whose text says "sealed
    (PROTOCOL §8)" and names --final, before any env is built, any weather is read or any file is
    written: the out dir stays empty. The seal comes before the run lookup (#48): an unregistered
    run dir on a test split is refused as sealed, not as unregistered."""
    import greenleaf_control.export_replay as export_replay

    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)
    controller = _controller(controller, tmp_path)
    out = tmp_path / "site"
    out.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        _main(export_replay, _argv(controller, split, out), tmp_path / "ledger.jsonl")
    text = _refusal(excinfo, capsys)
    assert "sealed (PROTOCOL §8)" in text, text
    assert "--final" in text, text
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("controller", ["rb_default", "<unregistered run dir>"])
@pytest.mark.parametrize("split", ["paper", "amsterdam_2012"])
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_final_as_f1s_not_implemented(split, controller, tmp_path, monkeypatch, capsys):
    """Done when: any --final is refused until F1. On a non-test split and on a test split alike,
    with a well-formed final-registered row id, it is refused naming F1 and "not implemented",
    before any env, weather read or file. It comes before the run lookup (#48): an unregistered
    run dir with --final is refused as F1's, not as unregistered."""
    import greenleaf_control.export_replay as export_replay

    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)
    controller = _controller(controller, tmp_path)
    out = tmp_path / "site"
    out.mkdir()
    argv = _argv(controller, split, out, "--final", "final-registered-deadbeef")

    with pytest.raises((SystemExit, NotImplementedError)) as excinfo:
        _main(export_replay, argv, tmp_path / "ledger.jsonl")
    text = _refusal(excinfo, capsys)
    assert "F1" in text, text
    assert "not implemented" in text.lower(), text
    assert list(out.iterdir()) == []


EXISTING_CASES = [
    pytest.param("rb_default", "paper", PAPER_FILE, id="baseline-paper"),
    # VALIDATION is Amsterdam 2010 from days 45, 59 and 75: its last file exists, so the refusal
    # must check every file before the first season is played.
    pytest.param(
        "rb_default", "validation", "rb_default_Amsterdam_2010_75.json", id="baseline-last-of-3"
    ),
    # A registered run dir's files are named <arm>_seed<seed>, not by its dir (seed100) (#48).
    pytest.param("<registered run dir>", "paper", RUN_FILE, id="run-dir-named-by-arm-and-seed"),
]


@pytest.mark.parametrize(("controller", "split", "existing"), EXISTING_CASES)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_an_existing_output_file_before_any_env(
    controller, split, existing, tmp_path, monkeypatch, capsys
):
    """Done when: an output file that already exists is refused, naming it, and left as it was.
    The refusal comes before any env is built or season played, so nothing else is written either:
    the out dir holds only that file. A registered run dir (E6-shaped tmp ledger, its result row
    selecting seed 100 at "1500000") passes the lookup and is refused on its new name,
    residual_w05_seed100_Amsterdam_2010_59.json."""
    import greenleaf_control.export_replay as export_replay

    ledger_path = tmp_path / "ledger.jsonl"
    if controller == "<registered run dir>":
        ledger_path, _, runs, _ = _e6(tmp_path, [{100: CHECKPOINT, 101: "final"}])
        controller = str(runs[100][1])
    out = tmp_path / "site"
    out.mkdir()
    (out / existing).write_bytes(b"an earlier export\n")
    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)

    with pytest.raises((SystemExit, FileExistsError)) as excinfo:
        _main(export_replay, _argv(controller, split, out), ledger_path)
    assert existing in _refusal(excinfo, capsys)
    assert (out / existing).read_bytes() == b"an earlier export\n"
    assert [p.name for p in out.iterdir()] == [existing]


# --- The CLI: a run dir goes through the one evaluation path (Done when) ------------------------


class _Routed(Exception):
    """Raised by the spy on run_policy once it has recorded its call."""


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_plays_a_run_dir_through_run_policy_at_its_selected_checkpoint(tmp_path, monkeypatch):
    """Done when: --controller <run_dir> goes through evaluate.run_policy, the one evaluation path,
    at the checkpoint its registration's result row selected, not "final".

    The tmp ledger is E6-shaped (_e6): E6's one result row selects seed 100 at "1500000" and seed
    101 at "final"; the decoy experiment's selects its own seed 100 at "1750000"; and the run dir
    holds a selected.json naming "1750000" too, so only the ledger's E6 result row gives "1500000".
    The CLI is handed the dir through a ".." (resolved, it is the run row's).

    run_policy is replaced by a spy that records its arguments and raises _Routed (so what export
    does with the season isn't tested here); run_rb, run_baseline and build_vec_env fail if called,
    so the run dir can't be played any other way. On --split paper, run_policy is called once
    before anything is written, with the run dir, checkpoint "1500000", the PAPER scenario, a
    60-day season, record_steps True (the per-step frame), return_trace True (the raw trace for
    the counters) and allow_test False (it never opens the 2012 weather)."""
    import greenleaf_control.export_replay as export_replay

    ledger_path, _, runs, _ = _e6(tmp_path, [{100: CHECKPOINT, 101: "final"}])
    run_dir = runs[100][1]
    (run_dir / "selected.json").write_text(json.dumps({"checkpoint": "1750000"}))
    given = run_dir.parent / ".." / ARM / run_dir.name
    out = tmp_path / "site"
    out.mkdir()
    signature = inspect.signature(evaluate.run_policy)
    _trip(monkeypatch, export_replay, ("run_baseline", "build_vec_env", "run_rb"))
    calls = []

    def run_policy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        raise _Routed

    monkeypatch.setattr(evaluate, "run_policy", run_policy)
    if hasattr(export_replay, "run_policy"):
        monkeypatch.setattr(export_replay, "run_policy", run_policy)

    with pytest.raises(_Routed):
        _main(export_replay, _argv(str(given), "paper", out), ledger_path)
    [call] = calls
    assert Path(call["run_dir"]).resolve() == run_dir.resolve()
    assert call["checkpoint"] == CHECKPOINT
    assert call["scenario"] == PAPER
    assert call["season_length"] == 60
    assert call["record_steps"] is True
    assert call["return_trace"] is True
    assert call["allow_test"] is False
    assert list(out.iterdir()) == []


# --- The CLI: a run dir is a registered run, and meta names it (Done when, #48) -----------------


def _bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


UNREGISTERED_CASES = [
    # No ledger file: ledger.rows reads no rows, so no run row names the dir.
    pytest.param(False, id="no-ledger"),
    # The E6-shaped ledger, and a copy of seed 100's dir under another root: E6's registration, the
    # arm, the seed and the dir's name are all in its path, but no run row names it.
    pytest.param(True, id="same-name-elsewhere"),
]


@pytest.mark.parametrize("e6", UNREGISTERED_CASES)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_an_unregistered_run_dir_before_any_env(e6, tmp_path, monkeypatch, capsys):
    """Done when: a run dir (it holds a manifest.json) that no ledger run row names is refused
    before any env, naming the dir and "not a registered run" (the site replays registered runs
    only), with nothing written: the out dir stays empty and the tmp ledger is as it was.
    - no-ledger: ledger_path doesn't exist.
    - same-name-elsewhere: the E6-shaped ledger (_e6), its result row selecting seed 100; the dir
      is tmp/copy/<E6's registered>/residual_w05/seed100, so a lookup by the dir's name, or by
      the arm and seed in its path, finds a run row it shouldn't."""
    import greenleaf_control.export_replay as export_replay

    ledger_path, registered = tmp_path / "ledger.jsonl", "registered-00000000"
    if e6:
        ledger_path, reg, _, _ = _e6(tmp_path, [{100: CHECKPOINT, 101: "final"}])
        registered = reg["row_id"]
    run_dir = _fake_run_dir(tmp_path / "copy" / registered / ARM / "seed100")
    before = _bytes(ledger_path)
    out = tmp_path / "site"
    out.mkdir()
    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        _main(export_replay, _argv(str(run_dir), "validation", out), ledger_path)
    text = _refusal(excinfo, capsys)
    assert "not a registered run" in text, text
    assert str(run_dir) in text, text
    assert list(out.iterdir()) == []
    assert _bytes(ledger_path) == before


LOOKUP_CASES = [
    # E6's registration has no result row yet (the decoy experiment's has one: it doesn't count).
    pytest.param([], 0, id="no-result-row"),
    # One result row, selecting seed 101 only.
    pytest.param([{101: "final"}], 1, id="result-selects-another-seed"),
    # One result row, selecting no run.
    pytest.param([{}], 1, id="result-selects-no-run"),
    # Two result rows, both selecting seed 100 at different checkpoints: either would be a guess.
    pytest.param(
        [{100: CHECKPOINT, 101: "final"}, {100: "1750000", 101: "final"}], 2, id="two-result-rows"
    ),
]


@pytest.mark.parametrize(("results", "count"), LOOKUP_CASES)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_a_run_dir_without_exactly_one_result_row_selecting_it(
    results, count, tmp_path, monkeypatch, capsys
):
    """Done when: a registered run dir (seed 100 of the E6-shaped tmp ledger, its run row naming
    the dir) is refused before any env unless its registration has exactly one result row and
    that row's selected names the run. The refusal names the run's row id and the count of its
    registration's result rows, as a number (0, 1 or 2 here; the decoy experiment's result row is
    another registration's), and nothing is written: the out dir stays empty and the tmp ledger is
    as it was."""
    import greenleaf_control.export_replay as export_replay

    ledger_path, _, runs, _ = _e6(tmp_path, results)
    run, run_dir = runs[100]
    before = _bytes(ledger_path)
    out = tmp_path / "site"
    out.mkdir()
    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        _main(export_replay, _argv(str(run_dir), "validation", out), ledger_path)
    text = _refusal(excinfo, capsys)
    assert run["row_id"] in text, text
    assert _names_count(text, count, tmp_path), f"the count {count} isn't named: {text}"
    assert list(out.iterdir()) == []
    assert _bytes(ledger_path) == before


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_run_dir_meta_names_its_registered_run_and_its_own_commit(
    tmp_path, monkeypatch, seasons
):
    """Done when: a run dir's file is <arm>_seed<seed>_<location>_<year>_<start day>.json, and its
    meta.run is the 7-key dict from the ledger, whose commit is the run dir's manifest's, not the
    stamp's.

    The E6-shaped tmp ledger's run row names seed 100's dir through a ".." (resolved, it is the dir
    the CLI is handed). run_policy is a spy that returns the 1-day PAPER rb default season as
    run_policy(..., return_trace=True) does, (result, frame, (states, controls, params)), so no
    checkpoint loads. main(--controller <seed 100's dir> --split paper) returns 0, calls run_policy
    once at "1500000", and writes exactly residual_w05_seed100_Amsterdam_2010_59.json, whose meta:
    - has exactly the 10 keys;
    - controller "residual_w05_seed100", scenario PAPER with season_length 1 (96 frame rows),
      split "paper", note "a simulator replay of a paper season, not the final result";
    - run {id: seed 100's run row, registered: E6's registration, result: E6's result row, arm
      "residual_w05", seed 100, checkpoint "1500000", commit RUN_COMMIT (the manifest's)};
    - commit the checkout's (git rev-parse HEAD), which is not RUN_COMMIT.
    The tmp ledger is as it was: the lookup only reads it."""
    import greenleaf_control.export_replay as export_replay

    ledger_path, reg, runs, [res] = _e6(tmp_path, [{100: CHECKPOINT, 101: "final"}], hop=True)
    run, run_dir = runs[100]
    before = _bytes(ledger_path)
    out = tmp_path / "site"
    out.mkdir()
    season = seasons["rb_default"]
    signature = inspect.signature(evaluate.run_policy)
    _trip(monkeypatch, export_replay, ("run_baseline", "build_vec_env", "run_rb"))
    calls = []

    def run_policy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        trace = (season["states"], season["controls"], season["params"])
        return season["result"], season["frame"], trace

    monkeypatch.setattr(evaluate, "run_policy", run_policy)
    if hasattr(export_replay, "run_policy"):
        monkeypatch.setattr(export_replay, "run_policy", run_policy)

    assert _main(export_replay, _argv(str(run_dir), "paper", out), ledger_path) == 0

    assert [call["checkpoint"] for call in calls] == [CHECKPOINT]
    assert [p.name for p in out.iterdir()] == [RUN_FILE]
    meta = json.loads((out / RUN_FILE).read_text(encoding="utf-8"))["meta"]
    head = subprocess.run(
        ["git", "-C", str(manifest.REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert set(meta) == META_KEYS
    assert meta["controller"] == "residual_w05_seed100"
    assert meta["scenario"] == {**PAPER, "season_length": 1}
    assert meta["split"] == "paper"
    assert meta["note"] == NOTE_PAPER
    assert meta["run"] == {
        "id": run["row_id"],
        "registered": reg["row_id"],
        "result": res["row_id"],
        "arm": ARM,
        "seed": 100,
        "checkpoint": CHECKPOINT,
        "commit": RUN_COMMIT,
    }
    assert meta["commit"] == head != RUN_COMMIT
    assert _bytes(ledger_path) == before


def test_cli_runs_as_a_module_and_lists_its_flags_and_every_split(tmp_path):
    """python -m greenleaf_control.export_replay --help exits 0 and lists --controller, --split,
    --out and --final, and every split in scenarios.SPLITS (the test splits too: refusing them is
    export_replay's seal, not an unknown choice)."""
    result = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.export_replay", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    missing = [word for word in (*CLI_FLAGS, *scenarios.SPLITS) if word not in result.stdout]
    assert not missing, f"--help doesn't list {missing}"


# --- Last: the signatures, and nothing touched the real ledger, runs or results ----------------


def test_signatures_match_the_design_and_no_test_wrote_the_real_ledger_runs_or_results():
    """Done when: export(frame, result, states, controls, params, *, controller, scenario, stamp,
    split, run) and main(argv=None, *, ledger_path=ledger.LEDGER), as selection.main.

    This test runs last in the module and checks the real ledger's size and mtime, and the
    listings of control/runs/ and control/results/, are as they were when the module was
    collected."""
    import greenleaf_control.export_replay as export_replay

    export = inspect.signature(export_replay.export).parameters
    positional = [n for n, p in export.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    keyword = [n for n, p in export.items() if p.kind is p.KEYWORD_ONLY]
    assert positional == ["frame", "result", "states", "controls", "params"]
    assert keyword == ["controller", "scenario", "stamp", "split", "run"]
    assert len(export) == 10
    main = inspect.signature(export_replay.main).parameters
    assert list(main) == ["argv", "ledger_path"]
    assert main["argv"].default is None
    assert main["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert main["ledger_path"].default == ledger.LEDGER
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE
    assert _listing(REAL_RESULTS) == REAL_RESULTS_BEFORE

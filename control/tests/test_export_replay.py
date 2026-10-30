"""Tests for greenleaf_control.export_replay (issue #24, D2): a recorded season as hourly JSON for
the demo site (D3).

export(frame, result, states, controls, params, *, controller, scenario, stamp) -> dict builds the
payload {"meta", "hourly", "totals"} from one season: evaluate's per-step frame (one row per step
t = 0..T-1, read from the state after step t), its result dict, and the same season's raw trace
(states (T + 1, nx), controls (T, 6), as metrics.trace_from_infos gives them) and model params,
which feed the running counters through metrics, the grader. scenario is a split's scenario dict
(location, growth_year, start_day); stamp holds the code's "commit", "dirty" and "gl_gym".
main(argv=None) -> int is the CLI.

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
- The file is <controller>_<location>_<growth_year>_<start_day>.json, written as
  json.dumps(payload, allow_nan=False, separators=(",", ":")); for a run dir, controller is the
  dir's name.

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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenleaf_control import envs, evaluate, manifest, metrics, rule_based, scenarios, train

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
META_KEYS = {"controller", "scenario", "commit", "dirty", "gl_gym", "note", "bounds", "units"}
NOTE = "a simulator run, not a registered result"
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


def _exported(export_replay, season: dict, controller: str = "rb_default") -> dict:
    """export's payload for a season, through the file's own JSON round trip (dumps then loads)."""
    payload = export_replay.export(
        season["frame"],
        season["result"],
        season["states"],
        season["controls"],
        season["params"],
        controller=controller,
        scenario=PAPER,
        stamp=STAMP,
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
    """A refusal's text: the exception's message plus stderr (argparse's error goes there)."""
    if isinstance(excinfo.value, SystemExit):
        assert excinfo.value.code not in (0, None), "a refusal must exit nonzero"
    return str(excinfo.value) + capsys.readouterr().err


def _argv(controller: str, split: str, out: Path, *extra: str) -> list[str]:
    return ["--controller", controller, "--split", split, "--out", str(out), *extra]


def _fake_run_dir(path: Path) -> Path:
    """A dir that looks like a run dir to a CLI check (it holds a manifest.json)."""
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"config": {"method": "residual"}}))
    return path


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


def test_export_meta_names_the_season_and_its_code_and_says_it_is_not_a_registered_result(seasons):
    """meta is exactly {controller, scenario, commit, dirty, gl_gym, note, bounds, units}:
    - controller as given; scenario the given scenario plus season_length in days (96 frame rows
      -> 1);
    - commit, dirty and gl_gym the stamp's;
    - note "a simulator run, not a registered result";
    - bounds metrics.BOUNDS as lists (PROTOCOL §1: 300-1,600 ppm, 15-34 °C, 50-85 %);
    - units co2 "ppm", temp "°C", rh "%", profit "€/m²", hours "h"."""
    import greenleaf_control.export_replay as export_replay

    meta = _exported(export_replay, seasons["rb_default"], controller="rb_default")["meta"]

    assert set(meta) == META_KEYS
    assert meta["controller"] == "rb_default"
    assert meta["scenario"] == {
        "location": "Amsterdam",
        "growth_year": 2010,
        "start_day": 59,
        "season_length": 1,
    }
    assert (meta["commit"], meta["dirty"], meta["gl_gym"]) == (
        STAMP["commit"],
        STAMP["dirty"],
        STAMP["gl_gym"],
    )
    assert meta["note"] == NOTE
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
    assert export_replay.main(_argv("rb_default", "paper", out)) == 0

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
    - meta: controller "rb_default", scenario PAPER with season_length 60, the checkout's commit
      (git rev-parse HEAD), dirty a bool, and the note.
    About 4 s, plus the fixture's 3 s."""
    import greenleaf_control.export_replay as export_replay

    want, frame = paper_60
    out = tmp_path / "site"
    out.mkdir()
    assert export_replay.main(_argv("rb_default", "paper", out)) == 0

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
    assert meta["controller"] == "rb_default"
    assert meta["scenario"] == {**PAPER, "season_length": 60}
    assert meta["commit"] == head
    assert isinstance(meta["dirty"], bool)
    assert meta["note"] == NOTE


# --- The CLI: the seal and the refusals (Done when; PROTOCOL §8) --------------------------------


@pytest.mark.parametrize("split", TEST_SPLITS)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_a_test_split_without_final(split, tmp_path, monkeypatch, capsys):
    """Done when: --split <a test split> without --final exits non-zero, naming the seal.

    --split takes every split in scenarios.SPLITS, so argparse doesn't refuse it as an unknown
    choice; export_replay refuses it, with SystemExit (code not 0) whose text says "sealed
    (PROTOCOL §8)" and names --final, before any env is built, any weather is read or any file is
    written: the out dir stays empty."""
    import greenleaf_control.export_replay as export_replay

    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)
    out = tmp_path / "site"
    out.mkdir()

    with pytest.raises(SystemExit) as excinfo:
        export_replay.main(_argv("rb_default", split, out))
    text = _refusal(excinfo, capsys)
    assert "sealed (PROTOCOL §8)" in text, text
    assert "--final" in text, text
    assert list(out.iterdir()) == []


@pytest.mark.parametrize("split", ["paper", "amsterdam_2012"])
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_final_as_f1s_not_implemented(split, tmp_path, monkeypatch, capsys):
    """Done when: any --final is refused until F1. On a non-test split and on a test split alike,
    with a well-formed final-registered row id, it is refused naming F1 and "not implemented",
    before any env, weather read or file."""
    import greenleaf_control.export_replay as export_replay

    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)
    out = tmp_path / "site"
    out.mkdir()
    argv = _argv("rb_default", split, out, "--final", "final-registered-deadbeef")

    with pytest.raises((SystemExit, NotImplementedError)) as excinfo:
        export_replay.main(argv)
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
    # A run dir's files are named by the dir's name, not its path.
    pytest.param("<run_dir>", "paper", "my-run_Amsterdam_2010_59.json", id="run-dir-named-by-dir"),
]


@pytest.mark.parametrize(("controller", "split", "existing"), EXISTING_CASES)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_an_existing_output_file_before_any_env(
    controller, split, existing, tmp_path, monkeypatch, capsys
):
    """An output file that already exists is refused, naming it, and left as it was. The refusal
    comes before any env is built or season played, so nothing else is written either: the out dir
    holds only that file."""
    import greenleaf_control.export_replay as export_replay

    if controller == "<run_dir>":
        controller = str(_fake_run_dir(tmp_path / "my-run"))
    out = tmp_path / "site"
    out.mkdir()
    (out / existing).write_bytes(b"an earlier export\n")
    _trip(monkeypatch, export_replay, ALL_ENV_ROUTES)

    with pytest.raises((SystemExit, FileExistsError)) as excinfo:
        export_replay.main(_argv(controller, split, out))
    assert existing in _refusal(excinfo, capsys)
    assert (out / existing).read_bytes() == b"an earlier export\n"
    assert [p.name for p in out.iterdir()] == [existing]


# --- The CLI: a run dir goes through the one evaluation path (Done when) ------------------------


class _Routed(Exception):
    """Raised by the spy on run_policy once it has recorded its call."""


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_plays_a_run_dir_through_run_policy_final_checkpoint_60_days(tmp_path, monkeypatch):
    """Done when: --controller <run_dir> goes through evaluate.run_policy, the one evaluation path.

    run_policy is replaced by a spy that records its arguments and raises _Routed (so what export
    does with the season isn't tested here); run_rb, run_baseline and build_vec_env fail if called,
    so the run dir can't be played any other way. On --split paper, run_policy is called once
    before anything is written, with the run dir, checkpoint "final", the PAPER scenario, a
    60-day season and record_steps True (the per-step frame)."""
    import greenleaf_control.export_replay as export_replay

    run_dir = _fake_run_dir(tmp_path / "my-run")
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
        export_replay.main(_argv(str(run_dir), "paper", out))
    [call] = calls
    assert Path(call["run_dir"]).resolve() == run_dir.resolve()
    assert call["checkpoint"] == "final"
    assert call["scenario"] == PAPER
    assert call["season_length"] == 60
    assert call["record_steps"] is True
    assert list(out.iterdir()) == []


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
    """export(frame, result, states, controls, params, *, controller, scenario, stamp) and
    main(argv=None).

    This test runs last in the module and checks the real ledger's size and mtime, and the
    listings of control/runs/ and control/results/, are as they were when the module was
    collected."""
    import greenleaf_control.export_replay as export_replay

    export = inspect.signature(export_replay.export).parameters
    positional = [n for n, p in export.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    keyword = [n for n, p in export.items() if p.kind is p.KEYWORD_ONLY]
    assert positional == ["frame", "result", "states", "controls", "params"]
    assert keyword == ["controller", "scenario", "stamp"]
    assert len(export) == 8
    main = inspect.signature(export_replay.main).parameters
    assert list(main) == ["argv"]
    assert main["argv"].default is None
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE
    assert _listing(REAL_RESULTS) == REAL_RESULTS_BEFORE

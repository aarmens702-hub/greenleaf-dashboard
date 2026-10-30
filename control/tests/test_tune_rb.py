"""Tests for greenleaf_control.tune_rb (issue #17, part 17b): tuning the rule-based controller's 12
setpoints with Optuna on TRAIN_EVAL (PROTOCOL §3, rb_tuned; docs/API-CONTRACT.md).

tune_rb, as these tests pin it:
- TUNE_SPACE: dict[str, tuple[float, float]], the 12 setpoints and their [lo, hi]. Every other
  controller key (the P-bands, flags and limits) stays at configs/rule_based_default.yml's value.
- rb_return(params, scenarios, season_length=60) -> float: the mean over scenarios of metrics'
  paper_return from evaluate.run_rb(params, s, season_length). An invalid season (a solver failure,
  or a short season) raises tune_rb.InvalidSeason (a RuntimeError): it is never scored.
- The tuner: Optuna TPESampler(seed=--seed), direction maximize, trial 0 enqueued at the default
  setpoints, every trial scored by rb_return(params, scenarios.TRAIN_EVAL, season_length). --jobs
  is Optuna's n_jobs; determinism is claimed at --jobs 1 only. optimize catches InvalidSeason only:
  that trial is FAIL in trials.csv, never scored, and the run goes on; the tuned yml is the best
  COMPLETE trial's. Any other exception crashes the run, and so does a tune with no COMPLETE trial
  (it writes no yml). A finished run-end's solver_failures is its count of FAIL trials.
- main(argv=None, *, ledger_path=ledger.LEDGER); CLI --trials N --jobs J --seed S --runs-dir DIR
  (--smoke | --registered ROW):
  - --smoke: min(N, 2) trials on 1-day seasons. Writes <runs-dir>/smoke/tune_rb-<stamp>/ with
    rule_based_tuned.yml ({"GreenLightEnv": every default key, the best trial's setpoints over
    them}, the shape rule_based.load_rb_params unwraps) and trials.csv (Optuna's
    trials_dataframe: one row per trial, with number, value, state and params_<key>). No ledger row.
  - --registered ROW: refused before any trial unless ROW matches the registration exactly, as
    train refuses a run its row doesn't hold: ROW is a registered row in the ledger whose arm's
    method is rb_tuned and whose arm config is exactly {"trials": N}; its scenarios are
    ["validation"] (where the RESULT is scored, PROTOCOL §2 and §9; the tuning itself stays
    hard-wired to TRAIN_EVAL), its seeds [0] and its timesteps 0 (nothing trains); --trials is N,
    --seed is in its seeds and --jobs is 1 (the seed-0 claim holds at n_jobs 1 only); and
    CONFIGS/rule_based_tuned.yml doesn't exist yet (write-once). A registered run
    records itself as train.main does: manifest.write_manifest into
    <runs-dir>/<row>/<arm>/seed<S>, a ledger run row before the trials and a run-end row after
    (finished, or crashed on an exception). It writes trials.csv and a copy of the tuned yml there,
    and the tuned yml to rule_based.CONFIGS / "rule_based_tuned.yml". So every registered test
    here has CONFIGS at a tmp copy and a clean temp git repo as manifest.REPO_ROOT (as
    test_train_smoke's registered tests do), and the one that runs its trials fakes rb_return.

Known answers used throughout:
- One day is 96 steps (86,400 s / 900 s); TRAIN_EVAL is 9 scenarios (Amsterdam 2008, 2009, 2011 x
  start days 45, 60, 75), so a smoke trial is nine 96-step seasons, about 1 s.
- Optuna 5.0's trials_dataframe names a parameter's column params_<name> (checked on optuna 5.0.0).
- metrics' paper_return is not the env's paper_reward_sum: on the 1-day PAPER season they are
  31.058082956... and 31.058083635..., so an exact check tells them apart.
- The mean of two floats is the same bits however it is summed: a + b rounds once and / 2 is exact.

Every test runs under a tripwire on pandas.read_csv that fails before any *2012.csv is read, and
one that refuses any append to the real ledger. Every main() gets a tmp --runs-dir, a tmp
ledger_path and rule_based.CONFIGS pointed at a tmp copy of the configs, and the last test checks
the real ledger, control/runs/ and configs/rule_based_tuned.yml are as they were when this module
was collected.
"""

import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
import yaml
from gl_gym.components.rule_based import RuleBasedController

from greenleaf_control import evaluate, ledger, manifest, rule_based, scenarios

# greenleaf_control.tune_rb is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"
REAL_TUNED = CONTROL / "configs" / "rule_based_tuned.yml"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger, runs dir and tuned yml, read when this module is collected, before any test.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)
REAL_TUNED_BEFORE = _stat(REAL_TUNED)

PAPER = scenarios.PAPER
RB_DEFAULT = rule_based.load_rb_params("default")
# The orchestrator's default space (issue #17; Aarmen may veto before E1): 12 setpoints, [lo, hi].
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
SMOKE_TRIALS = 2
CLI_FLAGS = ("--trials", "--jobs", "--seed", "--runs-dir", "--smoke", "--registered")
# E1's registration shape (orchestrator default): one rb_tuned arm of 200 trials, seed 0, its
# result scored on validation (it is tuned on TRAIN_EVAL regardless). ledger.register needs every
# registered field; timesteps is 0 (nothing trains).
E1 = {
    "issue": 17,
    "hypothesis": "plumbing: E1's registration shape, a test registration",
    "arms": [{"name": "rb_tuned", "method": "rb_tuned", "config": {"trials": 200}}],
    "seeds": [0],
    "timesteps": 0,
    "scenarios": ["validation"],
    "primary": "paper_return",
    "guards": ["viol_hours_temp"],
    "rule": "none: a test registration",
    "budget": {"cpu_hours": 0.01, "usd": 0.0, "backend": "local"},
}


@pytest.fixture(autouse=True)
def _never_the_real_ledger(monkeypatch):
    """A ledger call that forgets path= would append to the real ledger; this refuses it."""
    real_append = ledger.append

    def append(row: dict, path: Path = ledger.LEDGER) -> None:
        if Path(path).resolve() == REAL_LEDGER.resolve():
            raise AssertionError(f"a test tried to append a {row['kind']} row to the real ledger")
        real_append(row, path)

    monkeypatch.setattr(ledger, "append", append)


@pytest.fixture(autouse=True)
def _no_2012_reads(monkeypatch) -> None:
    """Fail loudly if a test reads any 2012 weather file. pandas.read_csv is the only file read
    upstream's weather loader makes; the check runs before the real read, and every other file
    (trials.csv too) goes through the real one."""
    real_read_csv = pd.read_csv

    def read_csv(filepath_or_buffer, *args, **kwargs):
        if Path(str(filepath_or_buffer)).name.endswith("2012.csv"):
            raise AssertionError(f"the sealed 2012 test weather was read: {filepath_or_buffer}")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)


@pytest.fixture(scope="module")
def default_season() -> pd.DataFrame:
    """The default RB's 2-day PAPER season frame, made by evaluate (which exists), so this fixture
    never ERRORs on tune_rb. Two days, not one: see the liveness test."""
    _, frame = evaluate.run_rb(RB_DEFAULT, PAPER, 2, record_steps=True)
    return frame


def _configs(tmp_path: Path, monkeypatch) -> Path:
    """rule_based.CONFIGS pointed at tmp_path/configs holding a copy of rule_based_default.yml, so a
    tuner that writes configs/rule_based_tuned.yml by mistake writes it there."""
    configs = tmp_path / "configs"
    configs.mkdir()
    shutil.copy(rule_based.CONFIGS / "rule_based_default.yml", configs)
    monkeypatch.setattr(rule_based, "CONFIGS", configs)
    return configs


def _smoke_argv(runs: Path, *, seed: int = 0, jobs: int = 1, trials: int = 5) -> list[str]:
    """--smoke with every flag explicit; --trials 5 is over the smoke cap, so 2 trials run."""
    return [
        "--smoke",
        *("--trials", str(trials)),
        *("--jobs", str(jobs)),
        *("--seed", str(seed)),
        *("--runs-dir", str(runs)),
    ]


def _registered_argv(
    row_id: str, runs: Path, *, trials: str = "200", seed: str = "0", jobs: str = "1"
) -> list[str]:
    """--registered; by default E1's 200 trials, seed 0 and one job."""
    return [
        *("--registered", row_id),
        *("--trials", trials),
        *("--jobs", jobs),
        *("--seed", seed),
        *("--runs-dir", str(runs)),
    ]


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], check=True, cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(repo: Path) -> None:
    """Stage and commit everything. CI has no git identity and may sign commits globally."""
    _git(repo, "add", ".")
    _git(
        repo,
        *("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"),
        *("commit", "-q", "-m", "commit"),
    )


def _make_repo(tmp_path: Path, monkeypatch) -> Path:
    """A temp git repo with one committed file, control/README.md, set as manifest.REPO_ROOT (as
    test_train_smoke's registered tests do), so a registered run's manifest sees a clean tree.

    runs/, configs/ and the ledger live beside it under tmp_path, so a run leaves its tree clean.
    """
    # A git hook's environment would point these git calls at the real repo.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    (repo / "control").mkdir(parents=True)
    (repo / "control" / "README.md").write_text("temp repo\n")
    _git(repo, "init", "-q")
    _commit(repo)
    monkeypatch.setattr(manifest, "REPO_ROOT", repo)
    return repo


def _recorder(monkeypatch, tune_rb) -> list[tuple[dict, list, int]]:
    """tune_rb.rb_return replaced by one that records (params, scenarios, season_length) and
    returns a number that depends on the params (the closer co2_day to 900 ppm, the higher)."""
    calls = []

    def rb_return(params, scenarios, season_length=60):
        calls.append((dict(params), list(scenarios), season_length))
        return -abs(float(params["co2_day"]) - 900.0)

    monkeypatch.setattr(tune_rb, "rb_return", rb_return)
    return calls


def _invalid_when(monkeypatch, tune_rb, fails) -> list[int]:
    """tune_rb.rb_return replaced by one that raises tune_rb.InvalidSeason on its i-th call (from
    0) when fails(i), and otherwise returns the recorder's number. At --jobs 1, call i is trial i.

    Returns [the number of calls so far], updated as it is called."""
    count = [0]

    def rb_return(params, scenarios, season_length=60):
        i, count[0] = count[0], count[0] + 1
        if fails(i):
            raise tune_rb.InvalidSeason(f"invalid season (faked) on call {i}")
        return -abs(float(params["co2_day"]) - 900.0)

    monkeypatch.setattr(tune_rb, "rb_return", rb_return)
    return count


def _no_season(monkeypatch, tune_rb) -> None:
    """evaluate.run_rb (and tune_rb.run_rb, if imported by name) fails: with rb_return faked, a
    registered tune of 60-day seasons must simulate nothing."""

    def run_rb(*args, **kwargs):
        raise AssertionError("evaluate.run_rb was called: the objective is rb_return, faked here")

    monkeypatch.setattr(evaluate, "run_rb", run_rb)
    if hasattr(tune_rb, "run_rb"):
        monkeypatch.setattr(tune_rb, "run_rb", run_rb)


def _no_trial(monkeypatch, tune_rb) -> None:
    """Every way a trial could start fails: Optuna's Study.optimize, tune_rb.rb_return and
    evaluate.run_rb (and tune_rb.run_rb, if imported by name)."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any trial")

        return refuse

    monkeypatch.setattr(optuna.study.Study, "optimize", trip("optuna Study.optimize"))
    monkeypatch.setattr(tune_rb, "rb_return", trip("tune_rb.rb_return"))
    monkeypatch.setattr(evaluate, "run_rb", trip("evaluate.run_rb"))
    if hasattr(tune_rb, "run_rb"):
        monkeypatch.setattr(tune_rb, "run_rb", trip("tune_rb.run_rb"))


def _refusal(excinfo, capsys) -> str:
    """A refusal's text: the exception's message plus stderr (argparse's error goes there)."""
    if isinstance(excinfo.value, SystemExit):
        assert excinfo.value.code not in (0, None), "a refusal must exit nonzero"
    return str(excinfo.value) + capsys.readouterr().err


def _out_dir(runs: Path) -> Path:
    """The one smoke output dir under runs: runs/smoke/tune_rb-<stamp>, and nothing else."""
    assert _listing(runs) == ["smoke"], _listing(runs)
    [out] = list((runs / "smoke").iterdir())
    assert out.name.startswith("tune_rb-"), out.name
    return out


def _params_columns(trials: pd.DataFrame) -> set[str]:
    return {c for c in trials.columns if c.startswith("params_")}


# --- TUNE_SPACE (Done when: TUNE_SPACE has the 12 keys) ----------------------------------------


def test_tune_space_is_the_12_setpoints_with_their_ranges():
    """Done when: TUNE_SPACE has the 12 keys. It is exactly the 12 setpoints above with their
    (lo, hi) tuples; each key is a key of the default params (a name the controller takes, so
    make_controller would reject a typo), lo < hi, and the default lies in [lo, hi] (lamps_on's
    default 0 is its lo), so trial 0, the defaults, is a point of the space."""
    import greenleaf_control.tune_rb as tune_rb

    assert tune_rb.TUNE_SPACE == TUNE_SPACE
    for key, (lo, hi) in tune_rb.TUNE_SPACE.items():
        assert key in RB_DEFAULT, key
        assert lo < hi, key
        assert lo <= RB_DEFAULT[key] <= hi, (key, RB_DEFAULT[key], (lo, hi))


# Each setpoint and the control upstream's rule computes from it (gl_gym @2d3febb,
# components/rule_based.py): u_boil from the heating setpoint; u_co2 from co2_day; u_vent from
# heatMax (heat_deadzone), ventRh (rh_max) and ventCold (t_vent_off); u_thscr from thScrCold
# (thScrSpDay, thScrSpNight) and thScrHeat (thScrDeadZone); u_lamp from the lamp schedule and the
# sun limit.
LIVE = [
    pytest.param("temp_setpoint_day", "u_boil", id="temp_setpoint_day"),
    pytest.param("temp_setpoint_night", "u_boil", id="temp_setpoint_night"),
    pytest.param("heat_deadzone", "u_vent", id="heat_deadzone"),
    pytest.param("co2_day", "u_co2", id="co2_day"),
    pytest.param("rh_max", "u_vent", id="rh_max"),
    pytest.param("t_vent_off", "u_vent", id="t_vent_off"),
    pytest.param("thScrSpDay", "u_thscr", id="thScrSpDay"),
    pytest.param("thScrSpNight", "u_thscr", id="thScrSpNight"),
    pytest.param("thScrDeadZone", "u_thscr", id="thScrDeadZone"),
    pytest.param("lamps_on", "u_lamp", id="lamps_on"),
    pytest.param("lamps_off", "u_lamp", id="lamps_off"),
    pytest.param("lamps_off_sun", "u_lamp", id="lamps_off_sun"),
]


@pytest.mark.parametrize(("key", "control"), LIVE)
def test_each_tune_space_key_moves_the_control_it_sets(key, control, default_season):
    """Every TUNE_SPACE key is live: the controller reads it, and nothing shadows it. Setting it to
    its lo, or to its hi, with every other key at its default, moves the control the rule computes
    from it by at least 0.01 (1 % of full actuation) on some step of a 2-day PAPER season, against
    the default's season.

    Known answers (measured on the 2-day PAPER season, Amsterdam 2010 from day 59; the lamps are
    scheduled 00:00 to 18:00, so "day inside" starts at midnight):
    - temp_setpoint_day, u_boil: both bounds, from step 0 (00:00, day inside under the lamps).
    - temp_setpoint_night, u_boil: both bounds, from 18:15 on day 1, once the lamps are off.
    - heat_deadzone, u_vent: at 1, from 08:45 on day 1, when the sunlit house (20.9 °C) passes the
      heating setpoint + 1 = 20.5 °C and vents for heat (|Δ| up to 0.95).
    - co2_day, u_co2: both bounds, from the first steps.
    - rh_max, u_vent: both bounds, from 01:30 on day 1 (RH 86 to 88 %, near the 85 % default).
      rh_max (snake case) is the vents' threshold and is read only there; rhMax (camel case, 85 by
      default, not tuned) is the screen's and the lamps'. Both are read, so neither shadows the
      other.
    - t_vent_off, u_vent: both bounds, from the first steps.
    - thScrSpDay, u_thscr: both bounds, from 05:00 on day 1 (3.6 to 4 °C outdoors).
    - thScrSpNight, u_thscr: at 0, from step 0 (4.2 °C outdoors is under the default 10 °C, which
      closes the screen, and over 0 °C, which opens it). At 15 the screen is already closed.
    - thScrDeadZone, u_thscr: at 1, on 4 steps from 07:45 on day 2 (20.8 °C and RH 88.6 % inside,
      3.2 °C outdoors: humid enough that the screen's RH term lets the heat term decide, warm enough
      to pass the heating setpoint + 1). On day 1 alone it moves u_thscr by 3.5e-5 at most, which
      is why these seasons are 2 days.
    - lamps_on, u_lamp: at 6, from 00:15 on day 1 (the schedule starts at 06:00, not 00:00); its lo
      is the default.
    - lamps_off, u_lamp: at 12 from 12:00 on day 1, at 24 from 18:00 on day 1.
    - lamps_off_sun, u_lamp: at 200 from 08:00 on day 1 (231 W/m² of sun switches the lamps off);
      at 600 from 10:15 on day 2 (414 W/m² no longer does).
    Twenty-four 2-day seasons, about 4 s in all.
    """
    import greenleaf_control.tune_rb as tune_rb

    assert key in tune_rb.TUNE_SPACE, f"{key} is not in TUNE_SPACE"
    default = default_season[control].to_numpy(dtype=np.float64)
    moved = {}
    for bound in tune_rb.TUNE_SPACE[key]:
        _, frame = evaluate.run_rb({**RB_DEFAULT, key: bound}, PAPER, 2, record_steps=True)
        moved[bound] = float(np.max(np.abs(frame[control].to_numpy(dtype=np.float64) - default)))
    assert max(moved.values()) >= 0.01, f"{key} is dead: max |Δ {control}| per bound {moved}"


# --- rb_return ---------------------------------------------------------------------------------


def test_rb_return_is_the_mean_paper_return_of_run_rb():
    """rb_return(params, [PAPER], 1) is run_rb(params, PAPER, 1)'s metrics paper_return exactly
    (not the env's paper_reward_sum, which differs from it in the 7th digit on this day), a float;
    and over [PAPER, TRAIN_EVAL[0]] it is the mean of the two seasons' paper_return, exactly (two
    floats: a + b rounds once, / 2 is exact). Three 1-day seasons."""
    import greenleaf_control.tune_rb as tune_rb

    paper = evaluate.run_rb(RB_DEFAULT, PAPER, 1)[0]
    train_eval = evaluate.run_rb(RB_DEFAULT, scenarios.TRAIN_EVAL[0], 1)[0]
    assert paper["paper_return"] != paper["paper_reward_sum"]

    one = tune_rb.rb_return(RB_DEFAULT, [PAPER], 1)
    assert isinstance(one, float)
    assert one == paper["paper_return"]
    two = tune_rb.rb_return(RB_DEFAULT, [PAPER, scenarios.TRAIN_EVAL[0]], 1)
    assert two == (paper["paper_return"] + train_eval["paper_return"]) / 2


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"solver_failures": 1, "valid": False}, id="solver-failure"),
        pytest.param({"steps": 95, "valid": False}, id="short-season"),
    ],
)
def test_rb_return_raises_on_an_invalid_season(bad, monkeypatch):
    """An invalid season is never scored: rb_return over [PAPER, TRAIN_EVAL[0]] raises
    tune_rb.InvalidSeason, a RuntimeError of its own (so the tuner can catch exactly it), its
    message saying "invalid", when TRAIN_EVAL[0]'s season is invalid, whether from a solver failure
    or a short season, though PAPER's is valid.

    The solver is faked, not rb_return: evaluate.run_rb (and tune_rb.run_rb, if imported by name)
    is wrapped so TRAIN_EVAL[0]'s real result comes back marked invalid."""
    import greenleaf_control.tune_rb as tune_rb

    invalid_season = getattr(tune_rb, "InvalidSeason", None)
    assert invalid_season is not None, "tune_rb has no InvalidSeason"
    assert issubclass(invalid_season, RuntimeError)
    assert invalid_season is not RuntimeError
    real_run_rb = evaluate.run_rb

    def run_rb(params, scenario, season_length=60, record_steps=False):
        result, frame = real_run_rb(params, scenario, season_length, record_steps)
        return ({**result, **bad} if scenario == scenarios.TRAIN_EVAL[0] else result), frame

    monkeypatch.setattr(evaluate, "run_rb", run_rb)
    if hasattr(tune_rb, "run_rb"):
        monkeypatch.setattr(tune_rb, "run_rb", run_rb)

    with pytest.raises(invalid_season) as excinfo:
        tune_rb.rb_return(RB_DEFAULT, [PAPER, scenarios.TRAIN_EVAL[0]], 1)
    assert "invalid" in str(excinfo.value).lower(), str(excinfo.value)


# --- The tuner ---------------------------------------------------------------------------------


def test_tuning_scores_every_trial_on_exactly_train_eval(tmp_path, monkeypatch):
    """Done when: a test asserts the tuning scenarios are exactly TRAIN_EVAL. With rb_return
    recorded, a smoke tune (--trials 5, capped at 2) calls it once per trial, 2 calls, each with
    scenarios == scenarios.TRAIN_EVAL (no test scenario among them) and a 1-day season."""
    import greenleaf_control.tune_rb as tune_rb

    calls = _recorder(monkeypatch, tune_rb)
    _configs(tmp_path, monkeypatch)

    assert tune_rb.main(_smoke_argv(tmp_path / "runs"), ledger_path=tmp_path / "l.jsonl") == 0

    assert len(calls) == SMOKE_TRIALS
    for _, tuned_on, season_length in calls:
        assert tuned_on == scenarios.TRAIN_EVAL
        assert not any(scenarios.is_test(s) for s in tuned_on)
        assert season_length == 1


def test_trial_0_is_the_defaults_and_only_the_12_setpoints_move(tmp_path, monkeypatch):
    """Trial 0 is enqueued at the defaults: rb_return's first call gets the default params exactly.
    Every call gets every default key; the keys outside TUNE_SPACE (P-bands, flags, limits) are
    always their defaults, and each setpoint lies in its [lo, hi]. trials.csv agrees: row 0 is
    trial 0 at the defaults."""
    import greenleaf_control.tune_rb as tune_rb

    calls = _recorder(monkeypatch, tune_rb)
    _configs(tmp_path, monkeypatch)
    runs = tmp_path / "runs"

    assert tune_rb.main(_smoke_argv(runs), ledger_path=tmp_path / "l.jsonl") == 0

    assert calls[0][0] == RB_DEFAULT
    for params, _, _ in calls:
        assert set(params) == set(RB_DEFAULT)
        assert {k: v for k, v in params.items() if k not in TUNE_SPACE} == {
            k: v for k, v in RB_DEFAULT.items() if k not in TUNE_SPACE
        }
        for key, (lo, hi) in TUNE_SPACE.items():
            assert lo <= params[key] <= hi, (key, params[key])
    trials = pd.read_csv(_out_dir(runs) / "trials.csv")
    row0 = trials[trials["number"] == 0].iloc[0]
    assert {k: row0[f"params_{k}"] for k in TUNE_SPACE} == {k: RB_DEFAULT[k] for k in TUNE_SPACE}


def test_smoke_writes_the_tuned_yml_and_trials_csv_under_runs_dir(tmp_path, monkeypatch):
    """Done when: tune_rb --smoke writes a tuned yml and trials.csv to a tmp path. The real tuner
    (rb_return not faked), --trials 5 --jobs 1 --seed 0: returns 0 and writes exactly one dir,
    <runs-dir>/smoke/tune_rb-<stamp>, holding rule_based_tuned.yml and trials.csv; no ledger row,
    nothing in configs.

    - trials.csv: 2 rows (the smoke cap), numbers 0 and 1, all COMPLETE, a params_<key> column for
      each of the 12 setpoints and no other params_ column. Trial 0's value is the defaults' mean
      paper_return over TRAIN_EVAL's nine 1-day seasons, recomputed here from evaluate.run_rb
      (rel 1e-12: the mean's summation order is the tuner's).
    - rule_based_tuned.yml is {"GreenLightEnv": params}: every default key, the 17 untuned ones at
      their defaults, the 12 setpoints those of the best trial (the highest value: maximize; rel
      1e-12, a CSV round trip). make_controller accepts it, and load_rb_params("tuned") with
      CONFIGS at that dir returns it, so E1's file will load as the residual arms read it.
    Twenty-seven 1-day seasons, about 4 s."""
    import greenleaf_control.tune_rb as tune_rb

    configs = _configs(tmp_path, monkeypatch)
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"

    assert tune_rb.main(_smoke_argv(runs), ledger_path=ledger_path) == 0

    out = _out_dir(runs)
    assert {"rule_based_tuned.yml", "trials.csv"} <= set(_listing(out))
    assert not ledger_path.exists()
    assert _listing(configs) == ["rule_based_default.yml"]

    trials = pd.read_csv(out / "trials.csv")
    assert trials["number"].tolist() == [0, 1]
    assert (trials["state"] == "COMPLETE").all()
    assert _params_columns(trials) == {f"params_{k}" for k in TUNE_SPACE}
    want = np.mean(
        [evaluate.run_rb(RB_DEFAULT, s, 1)[0]["paper_return"] for s in scenarios.TRAIN_EVAL]
    )
    assert trials["value"].iloc[0] == pytest.approx(want, rel=1e-12, abs=0)

    data = yaml.safe_load((out / "rule_based_tuned.yml").read_text())
    assert list(data) == ["GreenLightEnv"]
    tuned = data["GreenLightEnv"]
    assert set(tuned) == set(RB_DEFAULT)
    assert {k: v for k, v in tuned.items() if k not in TUNE_SPACE} == {
        k: v for k, v in RB_DEFAULT.items() if k not in TUNE_SPACE
    }
    best = trials.loc[trials["value"].idxmax()]
    for key in TUNE_SPACE:
        assert tuned[key] == pytest.approx(best[f"params_{key}"], rel=1e-12, abs=0), key
    assert isinstance(rule_based.make_controller(tuned), RuleBasedController)
    monkeypatch.setattr(rule_based, "CONFIGS", out)
    assert rule_based.load_rb_params("tuned") == tuned


def test_smoke_at_jobs_1_is_reproducible(tmp_path, monkeypatch):
    """Seed 0 at --jobs 1, run twice (into two runs dirs) with rb_return recorded, so no season is
    simulated, gives the same trials: rb_return gets the same params in the same order (trial 1,
    TPE's first draw, is not the defaults), trials.csv's number, value and params_<key> columns
    are equal exactly, and so is the tuned yml."""
    import greenleaf_control.tune_rb as tune_rb

    calls = _recorder(monkeypatch, tune_rb)
    _no_season(monkeypatch, tune_rb)
    _configs(tmp_path, monkeypatch)
    outs, params = [], []
    for name in ("a", "b"):
        calls.clear()
        runs = tmp_path / name
        assert tune_rb.main(_smoke_argv(runs), ledger_path=tmp_path / "l.jsonl") == 0
        outs.append(_out_dir(runs))
        params.append([p for p, _, _ in calls])

    assert len(params[0]) == SMOKE_TRIALS
    assert params[0] == params[1]
    assert params[0][1] != RB_DEFAULT
    a, b = (pd.read_csv(out / "trials.csv") for out in outs)
    columns = ["number", "value", *sorted(_params_columns(a))]
    pd.testing.assert_frame_equal(a[columns], b[columns])
    ymls = [yaml.safe_load((out / "rule_based_tuned.yml").read_text()) for out in outs]
    assert ymls[0] == ymls[1]


def test_seed_reaches_the_tpe_sampler(tmp_path, monkeypatch):
    """--seed seeds the sampler: with rb_return recorded, seed 0 and seed 1 both start at the
    defaults (trial 0 is enqueued), and their trial 1, TPE's first random draw, differs in every
    setpoint (12 continuous draws from two seeded RNGs)."""
    import greenleaf_control.tune_rb as tune_rb

    calls = _recorder(monkeypatch, tune_rb)
    _configs(tmp_path, monkeypatch)
    by_seed = {}
    for seed in (0, 1):
        calls.clear()
        argv = _smoke_argv(tmp_path / f"seed{seed}", seed=seed)
        assert tune_rb.main(argv, ledger_path=tmp_path / "l.jsonl") == 0
        by_seed[seed] = [params for params, _, _ in calls]

    assert by_seed[0][0] == by_seed[1][0] == RB_DEFAULT
    differ = [k for k in TUNE_SPACE if by_seed[0][1][k] != by_seed[1][1][k]]
    assert differ == list(TUNE_SPACE), (
        f"seeds 0 and 1 drew the same {set(TUNE_SPACE) - set(differ)}"
    )


def test_jobs_is_optunas_n_jobs_and_the_study_maximizes_with_tpe(tmp_path, monkeypatch):
    """--jobs J is Optuna's n_jobs, and the study is a TPESampler study that maximizes. Study.optimize
    is spied on (the real one runs) during a smoke tune at --jobs 2 --trials 5 with rb_return
    recorded: every call has n_jobs 2, the calls' n_trials sum to 2 (the smoke cap), the study's
    sampler is a TPESampler and its direction MAXIMIZE."""
    import greenleaf_control.tune_rb as tune_rb

    _recorder(monkeypatch, tune_rb)
    _configs(tmp_path, monkeypatch)
    real_optimize = optuna.study.Study.optimize
    seen = []

    def optimize(self, func, *args, **kwargs):
        bound = inspect.signature(real_optimize).bind(self, func, *args, **kwargs)
        bound.apply_defaults()
        seen.append((self, bound.arguments["n_trials"], bound.arguments["n_jobs"]))
        return real_optimize(self, func, *args, **kwargs)

    monkeypatch.setattr(optuna.study.Study, "optimize", optimize)

    argv = _smoke_argv(tmp_path / "runs", jobs=2)
    assert tune_rb.main(argv, ledger_path=tmp_path / "l.jsonl") == 0

    assert seen, "Study.optimize was never called"
    assert [n_jobs for _, _, n_jobs in seen] == [2] * len(seen)
    assert sum(n_trials for _, n_trials, _ in seen) == SMOKE_TRIALS
    for study, _, _ in seen:
        assert isinstance(study.sampler, optuna.samplers.TPESampler)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE


# --- --registered: the refusals ----------------------------------------------------------------


REFUSALS = [
    pytest.param("missing", {}, {}, "registered-deadbeef", id="no-such-row"),
    pytest.param(
        "row",
        {"arms": [{"name": "a", "method": "ppo", "config": {"trials": 200}}]},
        {},
        "rb_tuned",
        id="arm-method-ppo",
    ),
    # The registration's scenarios are where its result is scored: exactly ["validation"].
    pytest.param("row", {"scenarios": ["train_eval"]}, {}, "validation", id="scenarios-train_eval"),
    pytest.param(
        "row", {"scenarios": ["train_eval", "paper"]}, {}, "validation", id="scenarios-te-and-paper"
    ),
    pytest.param(
        "row",
        {"scenarios": ["validation", "paper"]},
        {},
        "validation",
        id="scenarios-val-and-paper",
    ),
    pytest.param("row", {"seeds": [1]}, {}, "seed", id="seeds-1"),
    pytest.param("row", {"seeds": [0, 1]}, {}, "seed", id="seeds-0-and-1"),
    pytest.param(
        "row",
        {
            "arms": [
                {
                    "name": "rb_tuned",
                    "method": "rb_tuned",
                    "config": {"trials": 200, "season_length": 10},
                }
            ]
        },
        {},
        "season_length",
        id="arm-config-extra-key",
    ),
    pytest.param("row", {"timesteps": 2_000_000}, {}, "timesteps", id="timesteps-2m-not-0"),
    # E1's own row, run off its registration: 2 trials of its 200, a seed it doesn't hold, or two
    # jobs (the seed-0 determinism holds at n_jobs 1 only).
    pytest.param("row", {}, {"trials": "2"}, "trials", id="trials-2-not-200"),
    pytest.param("row", {}, {"seed": "1"}, "seed", id="seed-1-not-in-0"),
    pytest.param("row", {}, {"jobs": "2"}, "jobs", id="jobs-2-not-1"),
]


@pytest.mark.parametrize(("which", "over", "flags", "named"), REFUSALS)
def test_registered_is_refused_before_any_trial(
    which, over, flags, named, tmp_path, monkeypatch, capsys
):
    """--registered ROW is refused (SystemExit nonzero or ValueError), its text naming what's wrong,
    unless the tune matches its registration exactly, as train refuses a run its registration
    doesn't hold: ROW is a registered row in the ledger with one arm of method rb_tuned whose config
    is exactly {"trials": N}, scenarios exactly ["validation"], seeds [0] and timesteps 0, run with
    --trials N, a --seed in its seeds and --jobs 1. The tmp ledger holds one row, E1's shape with
    over replacing its fields; ROW is that row, or for no-such-row a well-formed id the ledger
    doesn't hold; flags replace the argv's --trials 200 / --seed 0 / --jobs 1. The refusal comes
    before any trial (Study.optimize, rb_return and run_rb all trip), writes no file in --runs-dir
    or configs (no manifest either, though the temp repo is clean), and adds no ledger row."""
    import greenleaf_control.tune_rb as tune_rb

    _no_trial(monkeypatch, tune_rb)
    _make_repo(tmp_path, monkeypatch)
    configs = _configs(tmp_path, monkeypatch)
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"
    row = ledger.register(**{**E1, **over}, path=ledger_path)
    before = ledger_path.read_bytes()
    row_id = "registered-deadbeef" if which == "missing" else row["row_id"]

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        tune_rb.main(_registered_argv(row_id, runs, **flags), ledger_path=ledger_path)
    text = _refusal(excinfo, capsys)
    assert named in text.lower(), text
    assert _listing(runs) in (None, [])
    assert _listing(configs) == ["rule_based_default.yml"]
    assert ledger_path.read_bytes() == before


def test_registered_e1_row_passes_the_checks_and_a_crash_is_recorded(tmp_path, monkeypatch):
    """The refusals above are the fields', not --registered's: E1's own row (arm rb_tuned, method
    rb_tuned, config {"trials": 200}; seeds [0]; timesteps 0; scenarios ["validation"]) with
    --trials 200 --seed 0 --jobs 1 is not refused. It records its run row and goes on to tune
    (Optuna's real optimize), and trial 0's rb_return raises a plain RuntimeError: not an
    InvalidSeason, so optimize must not catch it (a catch of RuntimeError would). It propagates, as
    train.main lets a crash propagate, after a crashed run-end closes the run and the manifest says
    crashed. configs gets no tuned yml."""
    import greenleaf_control.tune_rb as tune_rb

    def rb_return(params, scenarios, season_length=60):
        raise RuntimeError("a crash that isn't an invalid season")

    monkeypatch.setattr(tune_rb, "rb_return", rb_return)
    _no_season(monkeypatch, tune_rb)
    _make_repo(tmp_path, monkeypatch)
    configs = _configs(tmp_path, monkeypatch)
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"
    row = ledger.register(**E1, path=ledger_path)

    with pytest.raises(RuntimeError, match="a crash that isn't an invalid season"):
        tune_rb.main(_registered_argv(row["row_id"], runs), ledger_path=ledger_path)

    assert _listing(configs) == ["rule_based_default.yml"]
    reg, run, end = ledger.rows(path=ledger_path)
    assert reg == row
    assert (run["kind"], run["registered"], run["arm"], run["seed"]) == (
        "run",
        row["row_id"],
        "rb_tuned",
        0,
    )
    assert (end["kind"], end["run"], end["status"]) == ("run-end", run["row_id"], "crashed")
    run_dir = runs / row["row_id"] / "rb_tuned" / "seed0"
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "crashed"


def test_registered_run_records_itself_and_writes_the_tuned_yml_to_configs(tmp_path, monkeypatch):
    """A registered tune of E1's row (--trials 200 --jobs 1 --seed 0) from a clean temp repo, with
    rb_return replaced by a cheap deterministic function of the params (recorded; the closer
    co2_day to 900 ppm, the higher) and evaluate.run_rb tripped, so 200 TPE trials take about 3 s
    and no season is simulated. It returns 0 and records itself as train.main does:

    - the ledger reads registered, run, run-end: the run row names the registration, arm rb_tuned,
      seed 0 and run_dir <runs>/<row>/rb_tuned/seed0; the run-end closes it, finished, with
      solver_failures 0 (no FAIL trial);
    - run_dir/manifest.json names the registration and issue 17, not smoke, clean, finished;
    - run_dir/trials.csv has 200 rows, numbers 0..199, all COMPLETE, row 0 the defaults;
    - rb_return was called 200 times, each on scenarios.TRAIN_EVAL with 60-day seasons (the
      registered run tunes full seasons; only --smoke shortens them);
    - CONFIGS/rule_based_tuned.yml (CONFIGS a tmp copy) is {"GreenLightEnv": every default key,
      the untuned ones at their defaults, the 12 setpoints the best trial's (the highest value;
      rel 1e-12, a CSV round trip)}, load_rb_params("tuned") returns it, and run_dir holds an
      identical copy;
    - no smoke dir, and the real configs/rule_based_tuned.yml is as it was when this module was
      collected (absent until E1).
    """
    import greenleaf_control.tune_rb as tune_rb

    calls = _recorder(monkeypatch, tune_rb)
    _no_season(monkeypatch, tune_rb)
    _make_repo(tmp_path, monkeypatch)
    configs = _configs(tmp_path, monkeypatch)
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"
    row = ledger.register(**E1, path=ledger_path)

    assert tune_rb.main(_registered_argv(row["row_id"], runs), ledger_path=ledger_path) == 0

    run_dir = runs / row["row_id"] / "rb_tuned" / "seed0"
    reg, run, end = ledger.rows(path=ledger_path)
    assert reg == row
    assert (run["kind"], run["registered"], run["arm"], run["seed"]) == (
        "run",
        row["row_id"],
        "rb_tuned",
        0,
    )
    assert Path(run["run_dir"]) == run_dir
    assert (end["kind"], end["run"], end["status"], end["solver_failures"]) == (
        "run-end",
        run["row_id"],
        "finished",
        0,
    )
    m = json.loads((run_dir / "manifest.json").read_text())
    assert (m["registered"], m["smoke"], m["dirty"], m["issue"], m["status"]) == (
        row["row_id"],
        False,
        False,
        17,
        "finished",
    )

    trials = pd.read_csv(run_dir / "trials.csv")
    assert trials["number"].tolist() == list(range(200))
    assert (trials["state"] == "COMPLETE").all()
    assert _params_columns(trials) == {f"params_{k}" for k in TUNE_SPACE}
    row0 = trials[trials["number"] == 0].iloc[0]
    assert {k: row0[f"params_{k}"] for k in TUNE_SPACE} == {k: RB_DEFAULT[k] for k in TUNE_SPACE}
    assert len(calls) == 200
    assert all(on == scenarios.TRAIN_EVAL and days == 60 for _, on, days in calls)

    assert _listing(configs) == ["rule_based_default.yml", "rule_based_tuned.yml"]
    data = yaml.safe_load((configs / "rule_based_tuned.yml").read_text())
    assert list(data) == ["GreenLightEnv"]
    tuned = data["GreenLightEnv"]
    assert set(tuned) == set(RB_DEFAULT)
    assert {k: v for k, v in tuned.items() if k not in TUNE_SPACE} == {
        k: v for k, v in RB_DEFAULT.items() if k not in TUNE_SPACE
    }
    best = trials.loc[trials["value"].idxmax()]
    for key in TUNE_SPACE:
        assert tuned[key] == pytest.approx(best[f"params_{key}"], rel=1e-12, abs=0), key
    assert rule_based.load_rb_params("tuned") == tuned
    assert yaml.safe_load((run_dir / "rule_based_tuned.yml").read_text()) == data
    assert not (runs / "smoke").exists()
    assert _stat(REAL_TUNED) == REAL_TUNED_BEFORE


def test_registered_refuses_to_overwrite_a_tuned_yml(tmp_path, monkeypatch, capsys):
    """Write-once: with CONFIGS/rule_based_tuned.yml already there (CONFIGS a tmp copy), E1's own
    row, run exactly as registered, is refused before any trial (SystemExit nonzero or ValueError;
    Study.optimize, rb_return and run_rb all trip), its text naming rule_based_tuned.yml. The file's
    bytes are unchanged, --runs-dir gets nothing and the ledger no row."""
    import greenleaf_control.tune_rb as tune_rb

    _no_trial(monkeypatch, tune_rb)
    _make_repo(tmp_path, monkeypatch)
    configs = _configs(tmp_path, monkeypatch)
    tuned = configs / "rule_based_tuned.yml"
    tuned.write_text("# an earlier tune's params\nGreenLightEnv: {}\n")
    tuned_before = tuned.read_bytes()
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"
    row = ledger.register(**E1, path=ledger_path)
    before = ledger_path.read_bytes()

    with pytest.raises((SystemExit, ValueError)) as excinfo:
        tune_rb.main(_registered_argv(row["row_id"], runs), ledger_path=ledger_path)
    text = _refusal(excinfo, capsys)
    assert "rule_based_tuned.yml" in text, text
    assert tuned.read_bytes() == tuned_before
    assert _listing(runs) in (None, [])
    assert ledger_path.read_bytes() == before


def _e1_with(trials: int) -> dict:
    """E1's registration with its arm's config at {"trials": trials}."""
    return {
        **E1,
        "arms": [{"name": "rb_tuned", "method": "rb_tuned", "config": {"trials": trials}}],
    }


def test_an_invalid_season_fails_its_trial_and_the_tune_finishes(tmp_path, monkeypatch):
    """An invalid season is never scored and doesn't end the tune: optimize catches
    tune_rb.InvalidSeason, Optuna marks that trial FAIL, and the next trial runs. E1's row at 5
    trials (config {"trials": 5}, --trials 5 --jobs 1 --seed 0), rb_return faked to raise
    InvalidSeason on trial 1 only and otherwise score the params. It returns 0 and:

    - trials.csv: trials 0..4, trial 1 FAIL with no value (Optuna keeps its params), the rest
      COMPLETE;
    - the tuned yml (CONFIGS, and run_dir's copy) is the best COMPLETE trial's setpoints over the
      defaults (rel 1e-12, a CSV round trip), and not trial 1's;
    - the run-end is finished with solver_failures 1, the count of FAIL trials; so is the manifest.
    """
    import greenleaf_control.tune_rb as tune_rb

    assert hasattr(tune_rb, "InvalidSeason"), "tune_rb has no InvalidSeason"
    _invalid_when(monkeypatch, tune_rb, lambda i: i == 1)
    _no_season(monkeypatch, tune_rb)
    _make_repo(tmp_path, monkeypatch)
    configs = _configs(tmp_path, monkeypatch)
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"
    row = ledger.register(**_e1_with(5), path=ledger_path)

    argv = _registered_argv(row["row_id"], runs, trials="5")
    assert tune_rb.main(argv, ledger_path=ledger_path) == 0

    run_dir = runs / row["row_id"] / "rb_tuned" / "seed0"
    trials = pd.read_csv(run_dir / "trials.csv")
    assert trials["number"].tolist() == [0, 1, 2, 3, 4]
    assert trials["state"].tolist() == ["COMPLETE", "FAIL", "COMPLETE", "COMPLETE", "COMPLETE"]
    assert np.isnan(trials["value"].iloc[1])
    complete = trials[trials["state"] == "COMPLETE"]
    best = complete.loc[complete["value"].idxmax()]
    failed = trials.iloc[1]

    data = yaml.safe_load((configs / "rule_based_tuned.yml").read_text())
    tuned = data["GreenLightEnv"]
    assert {k: v for k, v in tuned.items() if k not in TUNE_SPACE} == {
        k: v for k, v in RB_DEFAULT.items() if k not in TUNE_SPACE
    }
    for key in TUNE_SPACE:
        assert tuned[key] == pytest.approx(best[f"params_{key}"], rel=1e-12, abs=0), key
    assert any(tuned[k] != pytest.approx(failed[f"params_{k}"], rel=1e-12) for k in TUNE_SPACE)
    assert yaml.safe_load((run_dir / "rule_based_tuned.yml").read_text()) == data

    _, _, end = ledger.rows(path=ledger_path)
    assert (end["kind"], end["status"], end["solver_failures"]) == ("run-end", "finished", 1)
    m = json.loads((run_dir / "manifest.json").read_text())
    assert (m["status"], m["solver_failures"]) == ("finished", 1)


def test_a_tune_with_no_complete_trial_crashes_and_writes_no_yml(tmp_path, monkeypatch):
    """When every trial's season is invalid, trial 0 at the defaults included, there is no best
    trial, so no tuned params: E1's row at 3 trials, rb_return faked to raise
    tune_rb.InvalidSeason on every call. All 3 trials run (each FAIL, caught), then the run crashes:
    a RuntimeError or ValueError propagates that is not the InvalidSeason itself (Optuna's
    best_params raises ValueError "No trials are completed yet"). The run-end and the manifest say
    crashed, and neither CONFIGS nor the run dir gets a tuned yml."""
    import greenleaf_control.tune_rb as tune_rb

    assert hasattr(tune_rb, "InvalidSeason"), "tune_rb has no InvalidSeason"
    calls = _invalid_when(monkeypatch, tune_rb, lambda i: True)
    _no_season(monkeypatch, tune_rb)
    _make_repo(tmp_path, monkeypatch)
    configs = _configs(tmp_path, monkeypatch)
    runs, ledger_path = tmp_path / "runs", tmp_path / "l.jsonl"
    row = ledger.register(**_e1_with(3), path=ledger_path)

    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        tune_rb.main(_registered_argv(row["row_id"], runs, trials="3"), ledger_path=ledger_path)

    assert calls == [3], f"rb_return was called {calls[0]} times; every trial must run"
    assert not isinstance(excinfo.value, tune_rb.InvalidSeason), excinfo.value
    run_dir = runs / row["row_id"] / "rb_tuned" / "seed0"
    assert _listing(configs) == ["rule_based_default.yml"]
    assert not (run_dir / "rule_based_tuned.yml").exists()
    _, _, end = ledger.rows(path=ledger_path)
    assert (end["kind"], end["status"]) == ("run-end", "crashed")
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "crashed"


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param([], id="neither"),
        pytest.param(["--smoke", "--registered", "registered-deadbeef"], id="both"),
    ],
)
def test_cli_needs_exactly_one_of_smoke_or_registered(mode, tmp_path, monkeypatch, capsys):
    """--smoke and --registered are one required choice: neither, or both, exits nonzero before
    any trial and writes nothing."""
    import greenleaf_control.tune_rb as tune_rb

    _no_trial(monkeypatch, tune_rb)
    configs = _configs(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    argv = [*mode, "--trials", "2", "--jobs", "1", "--seed", "0", "--runs-dir", str(runs)]

    with pytest.raises(SystemExit) as excinfo:
        tune_rb.main(argv, ledger_path=tmp_path / "l.jsonl")
    _refusal(excinfo, capsys)
    assert _listing(runs) in (None, [])
    assert _listing(configs) == ["rule_based_default.yml"]


def test_cli_runs_as_a_module_and_lists_its_flags(tmp_path):
    """python -m greenleaf_control.tune_rb --help exits 0 and lists --trials, --jobs, --seed,
    --runs-dir, --smoke and --registered."""
    result = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.tune_rb", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    missing = [flag for flag in CLI_FLAGS if flag not in result.stdout]
    assert not missing, f"--help doesn't list {missing}"


def test_signatures_match_the_contract_and_nothing_real_was_written():
    """rb_return(params, scenarios, season_length=60) and main(argv=None, *,
    ledger_path=ledger.LEDGER), as train.main.

    This test runs last in the module and checks the real ledger's size and mtime, the listing of
    control/runs/, and configs/rule_based_tuned.yml (absent until E1) are as they were when the
    module was collected.
    """
    import greenleaf_control.tune_rb as tune_rb

    rb = inspect.signature(tune_rb.rb_return).parameters
    assert list(rb) == ["params", "scenarios", "season_length"]
    assert rb["season_length"].default == 60
    main = inspect.signature(tune_rb.main).parameters
    assert list(main) == ["argv", "ledger_path"]
    assert main["argv"].default is None
    assert main["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert main["ledger_path"].default == ledger.LEDGER
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE
    assert _stat(REAL_TUNED) == REAL_TUNED_BEFORE

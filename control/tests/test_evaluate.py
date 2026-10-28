"""Tests for greenleaf_control.evaluate (issue #13, part 13b): the one evaluation path, the RB
baseline, the per-step frame, the CLI and the seal.

evaluate scores a controller on fixed scenarios (docs/API-CONTRACT.md). run_rb plays the grower's
rules on the raw env. run_policy rebuilds a run's env stack from its manifest.json with
train.build_vec_env, loads vecnorm_<checkpoint>.pkl onto the stack UNDER VecNormalize, freezes it
(training=False, norm_reward=False), loads ckpt_<checkpoint>.zip on cpu, resets once and predicts
deterministically to the season's end. Both return metrics.episode_metrics(...) plus
"paper_reward_sum" (the season's info["paper_reward"] summed left to right with +=, as
train._final_eval sums final_eval.json's paper_return; not sum(), which compensates its float sums
in Python 3.12, so differs in the last bits), and, with record_steps, one frame row per step.

Known answers used throughout:
- One day is 96 steps (86,400 s / 900 s), so season_length=1 is 96 steps, 2 is 192, 60 is 5,760.
- A policy whose action_net weight and bias are zero predicts exactly 0 for every observation, and
  the co-pilot applies clip(u_rb + 0.15 * 0, 0, 1) = u_rb (rb_action is already in [0, 1]); so its
  season through the full VecNormalize stack is the RB's season, step for step.
- A violating step is 900 s = 0.25 h, so viol_hours_<c> is the frame's out-of-bounds count x 0.25.
- Upstream's control order is uBoil, uCO2, uThScr, uVent, uLamp, uBlScr (gl_gym @2d3febb).

One residual smoke run (seed 0, one env, 2,048 steps = one rollout, 1-day seasons, rb default),
made once by a module fixture under a tmp runs dir and a tmp ledger, is shared by the tests that
load a policy.

The 2012 weather is sealed (PROTOCOL §8). Every test that could build an env runs under a tripwire
on pandas.read_csv that fails before any *2012.csv is read; the CLI refusals also run with every
env builder (and evaluate's run_rb, run_policy) replaced by one that fails, so a refusal must come
before any env or file. Nothing here may write the real ledger or control/runs/; the last test
checks both are as they were when this module was collected.
"""

import inspect
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecMonitor, VecNormalize

from greenleaf_control import envs, metrics, rule_based, scenarios, train

# greenleaf_control.evaluate is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger and runs dir, read when this module is collected, before any test in it runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)

N_DAY = 96  # 86,400 s / 900 s
PAPER = scenarios.PAPER
CONSTRAINTS = metrics.CONSTRAINTS  # ("co2", "temp", "rh")
RB_DEFAULT = rule_based.load_rb_params("default")
# Upstream's control order: uBoil, uCO2, uThScr, uVent, uLamp, uBlScr.
U_COLS = ["u_boil", "u_co2", "u_thscr", "u_vent", "u_lamp", "u_blscr"]
U_RB_COLS = [f"u_rb_{c.removeprefix('u_')}" for c in U_COLS]
COST_COLS = [f"cost_{c}" for c in CONSTRAINTS]
RB_FRAME = ["t", "co2", "temp", "rh", *U_COLS, "profit", *COST_COLS]
RESIDUAL_FRAME = ["t", "co2", "temp", "rh", *U_COLS, *U_RB_COLS, "profit", *COST_COLS]
# A --out row's identity columns; the rest are the result dict's keys.
ROW_KEYS = {
    "controller",
    "checkpoint",
    "split",
    "location",
    "growth_year",
    "start_day",
    "season_length",
}
TEST_SPLITS = ("amsterdam_2012", "london_2012", "abbotsford_2012")
CLI_FLAGS = ("--controller", "--split", "--out", "--final")


@pytest.fixture(scope="module")
def residual_run(tmp_path_factory) -> Path:
    """One residual smoke run, made by train (which exists), so this fixture never ERRORs on
    evaluate. Seed 0, one env, 2,048 steps (one rollout), 1-day seasons, rb default; its runs dir
    and ledger are tmp. Its final_eval.json is one deterministic 96-step PAPER season. About 6 s."""
    root = tmp_path_factory.mktemp("residual_run")
    runs = root / "runs"
    argv = [
        *("--method", "residual"),
        *("--seed", "0"),
        "--smoke",
        *("--runs-dir", str(runs)),
        *("--n-envs", "1"),
        *("--timesteps", "2048"),
        *("--season-length", "1"),
        *("--rb", "default"),
    ]
    assert train.main(argv, ledger_path=root / "ledger.jsonl") == 0
    [run_dir] = list((runs / "smoke").iterdir())
    return run_dir


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


def _no_env_built(monkeypatch, evaluate) -> None:
    """Every way evaluate could build an env or play a season fails: envs.make_raw_env, make_env,
    run_baseline, train.build_vec_env, and the same names (plus run_rb, run_policy) on evaluate
    itself, in case it imported them by name."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any env")

        return refuse

    for module, name in [
        (envs, "make_raw_env"),
        (envs, "make_env"),
        (envs, "run_baseline"),
        (train, "build_vec_env"),
    ]:
        monkeypatch.setattr(module, name, trip(f"{module.__name__}.{name}"))
    for name in (
        "make_raw_env",
        "make_env",
        "run_baseline",
        "build_vec_env",
        "run_rb",
        "run_policy",
    ):
        if hasattr(evaluate, name):
            monkeypatch.setattr(evaluate, name, trip(f"evaluate.{name}"))


def _refusal(excinfo, capsys) -> str:
    """A refusal's text: the exception's message plus stderr (argparse's error goes there)."""
    if isinstance(excinfo.value, SystemExit):
        assert excinfo.value.code not in (0, None), "a refusal must exit nonzero"
    return str(excinfo.value) + capsys.readouterr().err


def _argv(controller: str, split: str, out: Path, *extra: str) -> list[str]:
    return ["--controller", controller, "--split", split, "--out", str(out), *extra]


def _graded(infos: list[dict], scenario: dict, days: int) -> dict:
    """A baseline season's result dict, graded the way the golden numbers are: trace_from_infos,
    then episode_metrics with the model's params and season_steps = days x 96, plus
    paper_reward_sum, info["paper_reward"] summed left to right with +=, as train._final_eval
    sums paper_return; not sum(), which compensates its float sums in Python 3.12."""
    states, controls = metrics.trace_from_infos(infos)
    p = envs.make_raw_env(
        scenario=scenario, season_length=days, normalize_actions=False
    ).unwrapped.p
    graded = metrics.episode_metrics(states, controls, np.asarray(p), season_steps=days * N_DAY)
    paper_reward_sum = 0.0
    for info in infos:
        paper_reward_sum += info["paper_reward"]
    return {**graded, "paper_reward_sum": paper_reward_sum}


def _chain(venv) -> list:
    """The VecEnv stack from the outside in, ending at the DummyVecEnv or SubprocVecEnv."""
    chain = [venv]
    while isinstance(chain[-1], VecEnvWrapper):
        chain.append(chain[-1].venv)
    return chain


def _residual_wrapper(env) -> envs.ResidualRBWrapper:
    """The ResidualRBWrapper in a gym env's wrapper chain, walking .env from the outside in."""
    while isinstance(env, gym.Wrapper):
        if isinstance(env, envs.ResidualRBWrapper):
            return env
        env = env.env
    raise AssertionError("no ResidualRBWrapper in the env's wrapper chain")


def _assert_frame_violations_are_the_metrics(frame: pd.DataFrame, result: dict) -> None:
    """Per constraint, the frame's out-of-bounds steps (metrics.BOUNDS; a value on a bound is in
    bounds) x 0.25 h are the result's viol_hours_<c>."""
    for c in CONSTRAINTS:
        low, high = metrics.BOUNDS[c]
        v = frame[c].to_numpy(dtype=np.float64)
        violating = int(np.count_nonzero((v < low) | (v > high)))
        # 900 s per step / 3,600 s per hour = 0.25 h per violating step
        assert violating * 0.25 == result[f"viol_hours_{c}"], c


# --- The golden evaluation path (Done when; PROTOCOL §7, ladder rung 5) ------------------------


@pytest.mark.usefixtures("no_2012_reads")
def test_zero_output_policy_through_the_full_vecnormalize_stack_is_run_rb(residual_run):
    """Done when (golden eval path): the residual smoke run's final checkpoint with action_net's
    weight and bias zeroed, saved as ckpt_zero.zip beside a copy of vecnorm_final.pkl as
    vecnorm_zero.pkl, played by run_policy through the residual env's full VecNormalize stack on a
    1-day PAPER season, gives exactly run_rb's season with the same (default) RB params.

    Known answer: the zeroed policy predicts exactly 0 for any observation, so the co-pilot applies
    clip(u_rb + 0.15 x 0, 0, 1) = u_rb every step. So the result dicts are ==, the frames are equal
    exactly on run_rb's columns, and the residual frame's u_rb_* are its u_*.

    Non-vacuous: the unzeroed final checkpoint's controls differ from run_rb's somewhere.
    """
    import greenleaf_control.evaluate as evaluate

    model = PPO.load(residual_run / "ckpt_final.zip", device="cpu")
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    obs = np.random.default_rng(0).normal(size=(4, 29)).astype(np.float32)  # 23 + 6 RB dims
    action, _ = model.predict(obs, deterministic=True)
    np.testing.assert_array_equal(action, np.zeros((4, 6), dtype=np.float32))
    model.save(residual_run / "ckpt_zero.zip")
    shutil.copy(residual_run / "vecnorm_final.pkl", residual_run / "vecnorm_zero.pkl")

    got, frame = evaluate.run_policy(residual_run, "zero", PAPER, 1, record_steps=True)
    want, rb_frame = evaluate.run_rb(RB_DEFAULT, PAPER, 1, record_steps=True)

    assert got == want
    assert (got["steps"], got["valid"]) == (N_DAY, True)
    assert list(rb_frame.columns) == RB_FRAME
    np.testing.assert_array_equal(
        frame[RB_FRAME].to_numpy(dtype=np.float64), rb_frame.to_numpy(dtype=np.float64)
    )
    np.testing.assert_array_equal(
        frame[U_RB_COLS].to_numpy(dtype=np.float64), frame[U_COLS].to_numpy(dtype=np.float64)
    )

    _, trained = evaluate.run_policy(residual_run, "final", PAPER, 1, record_steps=True)
    assert not np.array_equal(
        trained[U_COLS].to_numpy(dtype=np.float64), rb_frame[U_COLS].to_numpy(dtype=np.float64)
    ), "the trained policy plays exactly the RB, so the zeroed-policy check above proves nothing"


def test_run_policy_rebuilds_the_training_stack_under_a_frozen_vecnormalize(
    residual_run, monkeypatch
):
    """Done when: run_policy loads VecNormalize with training False and norm_reward False.

    VecNormalize.load is spied on (the real one runs). run_policy(run, "final", VALIDATION[0], 2)
    loads vecnorm_final.pkl exactly once, onto the stack under VecNormalize that build_vec_env
    rebuilds from manifest.json (VecMonitor over a DummyVecEnv; a VecNormalize there would
    normalize twice), and the loaded VecNormalize ends the season frozen: training False,
    norm_reward False, norm_obs True, and obs_rms (mean, var, count) exactly the pickle's, so not
    even the reset updated it. The env is the residual co-pilot at the manifest's residual_scale,
    on the given scenario for the given 2-day season (192 steps), not the manifest's.
    """
    import greenleaf_control.evaluate as evaluate

    real_load = VecNormalize.load
    loads = []

    def load(load_path, venv):
        vecnorm = real_load(load_path, venv)
        loads.append((Path(load_path), venv, vecnorm))
        return vecnorm

    monkeypatch.setattr(VecNormalize, "load", staticmethod(load))
    s = scenarios.VALIDATION[0]
    result, frame = evaluate.run_policy(residual_run, "final", s, 2)

    [(path, under, vecnorm)] = loads
    assert path.resolve() == (residual_run / "vecnorm_final.pkl").resolve()
    assert vecnorm.training is False and vecnorm.norm_reward is False
    assert vecnorm.norm_obs is True
    with (residual_run / "vecnorm_final.pkl").open("rb") as f:
        saved = pickle.load(f)
    np.testing.assert_array_equal(vecnorm.obs_rms.mean, saved.obs_rms.mean)
    np.testing.assert_array_equal(vecnorm.obs_rms.var, saved.obs_rms.var)
    assert vecnorm.obs_rms.count == saved.obs_rms.count

    assert vecnorm.venv is under
    assert [type(v) for v in _chain(under)] == [VecMonitor, DummyVecEnv]
    env = _chain(under)[-1].envs[0]
    config = json.loads((residual_run / "manifest.json").read_text())["config"]
    assert _residual_wrapper(env).scale == config["residual_scale"]
    assert vars(env.unwrapped.weather_scenario_sampler) == s
    assert env.unwrapped.N == 2 * N_DAY
    assert result["steps"] == 2 * N_DAY
    assert frame is None


@pytest.mark.usefixtures("no_2012_reads")
def test_reloaded_final_checkpoint_reproduces_its_end_of_training_evaluation(residual_run):
    """Done when: a reloaded smoke checkpoint reproduces the evaluation number logged at the end of
    its training. final_eval.json holds one deterministic 1-day PAPER season (96 steps) played by
    the in-memory model with frozen copies of its VecNormalize stats; run_policy(run, "final",
    PAPER, 1) replays it from ckpt_final.zip and vecnorm_final.pkl, so its paper_reward_sum is
    final_eval.json's paper_return EXACTLY (same weights, same stats, same env, and both summed left
    to right with +=, as train._final_eval does; not sum(), which compensates in Python 3.12)."""
    import greenleaf_control.evaluate as evaluate

    final = json.loads((residual_run / "final_eval.json").read_text())
    assert (final["scenario"], final["season_length"], final["steps"]) == (PAPER, 1, N_DAY)

    result, frame = evaluate.run_policy(residual_run, "final", PAPER, 1)

    assert result["paper_reward_sum"] == final["paper_return"]
    assert (result["steps"], result["valid"]) == (N_DAY, True)
    assert frame is None


@pytest.mark.usefixtures("no_2012_reads")
def test_run_rb_is_the_rb_default_baseline_graded_the_same_way():
    """run_rb(default params, PAPER, 1) is envs.run_baseline("rb_default", PAPER, 1) graded with
    trace_from_infos + episode_metrics (the model's params, 96 season steps) + paper_reward_sum
    (summed left to right with +=, see _graded): the same season, so the result dicts are ==
    exactly. Valid, 96 steps, no frame by default."""
    import greenleaf_control.evaluate as evaluate

    result, frame = evaluate.run_rb(RB_DEFAULT, PAPER, 1)

    assert result == _graded(envs.run_baseline("rb_default", PAPER, 1), PAPER, 1)
    assert (result["steps"], result["valid"]) == (N_DAY, True)
    assert frame is None


# --- The per-step frame ------------------------------------------------------------------------


@pytest.mark.usefixtures("no_2012_reads")
def test_run_rb_frame_is_one_row_per_step_read_from_the_raw_state():
    """run_rb(..., record_steps=True)'s frame on a 1-day PAPER season: exactly the contract's
    columns in order (t, co2, temp, rh, the 6 u_*, profit, the 3 cost_*; no u_rb_*), 96 rows, and
    row t is step t (t = 0..95, as metrics numbers steps), read from the same season as
    envs.run_baseline("rb_default", PAPER, 1) (test_run_rb_is_the_rb_default_baseline...):

    - co2, temp, rh from the state after the step, info["x"], with upstream's conversions:
      co2dens2ppm(x[2], x[0] x 1e-6) [ppm], x[2] [C], vaporPres2rh(x[2], x[15]) [%];
    - u_* the applied controls, info["controls"], in upstream's order;
    - profit and cost_* the step's info["profit"] and info["cost_<c>"].

    co2 and rh are compared at rel 1e-12 (an exp vectorized or not may differ in its last bit);
    everything else is exact.
    """
    import greenleaf_control.evaluate as evaluate

    _, frame = evaluate.run_rb(RB_DEFAULT, PAPER, 1, record_steps=True)

    assert list(frame.columns) == RB_FRAME
    assert len(frame) == N_DAY
    assert frame["t"].tolist() == list(range(N_DAY))
    infos = envs.run_baseline("rb_default", PAPER, 1)
    x = np.stack([info["x"] for info in infos])  # (96, 28): the state after each step
    controls = np.stack([info["controls"] for info in infos]).astype(np.float64)
    exact = {
        "temp": x[:, 2],
        **{col: controls[:, j] for j, col in enumerate(U_COLS)},
        "profit": np.array([info["profit"] for info in infos], dtype=np.float64),
        **{f"cost_{c}": np.array([info[f"cost_{c}"] for info in infos]) for c in CONSTRAINTS},
    }
    for col, want in exact.items():
        np.testing.assert_array_equal(frame[col].to_numpy(dtype=np.float64), want, err_msg=col)
    converted = {
        "co2": co2dens2ppm(x[:, 2], x[:, 0] * 1e-6),  # x[0] CO2 [mg/m3] -> [kg/m3] -> [ppm]
        "rh": vaporPres2rh(x[:, 2], x[:, 15]),  # x[15] vapour pressure [Pa] -> RH [%]
    }
    for col, want in converted.items():
        np.testing.assert_allclose(
            frame[col].to_numpy(dtype=np.float64), want, rtol=1e-12, atol=0, err_msg=col
        )


@pytest.mark.usefixtures("no_2012_reads")
def test_run_policy_frame_adds_the_rb_action_for_the_residual_arm(residual_run):
    """run_policy(..., record_steps=True) on the residual run: the contract's columns in order with
    the 6 u_rb_* after the 6 u_* (residual only), 96 rows, t = 0..95, and every u_* and u_rb_* in
    [0, 1] (both are clipped there)."""
    import greenleaf_control.evaluate as evaluate

    _, frame = evaluate.run_policy(residual_run, "final", PAPER, 1, record_steps=True)

    assert list(frame.columns) == RESIDUAL_FRAME
    assert len(frame) == N_DAY
    assert frame["t"].tolist() == list(range(N_DAY))
    u = frame[U_COLS + U_RB_COLS].to_numpy(dtype=np.float64)
    assert np.all((u >= 0.0) & (u <= 1.0))


FRAME_CASES = [
    pytest.param("rb", RB_DEFAULT, id="rb-default"),
    pytest.param("rb", {**RB_DEFAULT, "co2_day": 2500}, id="rb-co2-day-2500"),
    pytest.param("policy", None, id="residual-final-policy"),
]


@pytest.mark.parametrize(("kind", "params"), FRAME_CASES)
@pytest.mark.usefixtures("no_2012_reads")
def test_frame_out_of_bounds_steps_x_a_quarter_hour_are_the_violation_hours(kind, params, request):
    """Per constraint, the frame's co2, temp and rh outside metrics.BOUNDS (300-1,600 ppm, 15-34 C,
    50-85 %) count x 0.25 h = the result's viol_hours_<c>, for a 1-day PAPER season.

    Known answers that make both sides of the CO2 check run:
    - rb default has viol_hours_co2 = 0.0: the golden 60-day PAPER season, which starts with this
      day, has 0.0 (tests/golden/values.json), so no step of it is out of bounds.
    - co2_day 2,500 doses CO2 toward 2,500 ppm by day, above the 1,600 ppm bound, so some steps
      violate: viol_hours_co2 > 0.
    """
    import greenleaf_control.evaluate as evaluate

    if kind == "rb":
        result, frame = evaluate.run_rb(params, PAPER, 1, record_steps=True)
    else:
        run_dir = request.getfixturevalue("residual_run")
        result, frame = evaluate.run_policy(run_dir, "final", PAPER, 1, record_steps=True)

    if kind == "rb" and params["co2_day"] == 800:
        assert result["viol_hours_co2"] == 0.0
    if kind == "rb" and params["co2_day"] == 2500:
        assert result["viol_hours_co2"] > 0
    _assert_frame_violations_are_the_metrics(frame, result)


# --- The seal (Done when; PROTOCOL §8) ---------------------------------------------------------


@pytest.mark.parametrize("split", TEST_SPLITS)
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_a_test_split_without_final(split, tmp_path, monkeypatch, capsys):
    """Done when: evaluate --split <a test split> without --final exits non-zero, naming the seal.

    --split takes every split in scenarios.SPLITS, so argparse doesn't refuse it as an unknown
    choice; evaluate refuses it, with SystemExit (code not 0) whose text says "sealed (PROTOCOL
    §8)" and names --final, before any env is built, any weather is read or any file is written:
    tmp_path stays empty."""
    import greenleaf_control.evaluate as evaluate

    _no_env_built(monkeypatch, evaluate)

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main(_argv("rb_default", split, tmp_path / "x.parquet"))
    text = _refusal(excinfo, capsys)
    assert "sealed (PROTOCOL §8)" in text, text
    assert "--final" in text, text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("split", ["paper", "amsterdam_2012"])
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_refuses_final_as_f1s_not_implemented(split, tmp_path, monkeypatch, capsys):
    """Any --final is refused: the final evaluation is F1's (Stage 3), not implemented here. On a
    non-test split and on a test split alike, with a well-formed final-registered row id, it is
    refused naming F1 and "not implemented", before any env, weather read or file."""
    import greenleaf_control.evaluate as evaluate

    _no_env_built(monkeypatch, evaluate)
    argv = _argv(
        "rb_default", split, tmp_path / "x.parquet", "--final", "final-registered-deadbeef"
    )

    with pytest.raises((SystemExit, NotImplementedError)) as excinfo:
        evaluate.main(argv)
    text = _refusal(excinfo, capsys)
    assert "F1" in text, text
    assert "not implemented" in text.lower(), text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("split", TEST_SPLITS)
@pytest.mark.usefixtures("no_2012_reads")
def test_run_rb_and_run_policy_refuse_a_test_scenario(split, residual_run):
    """run_rb and run_policy on a TEST scenario raise ValueError naming the seal (or allow_test,
    the only way past it), through make_raw_env's own refusal, before any 2012 file is read."""
    import greenleaf_control.evaluate as evaluate

    s = scenarios.TEST[split][0]

    with pytest.raises(ValueError, match=r"sealed|allow_test"):
        evaluate.run_rb(RB_DEFAULT, s, 1)
    with pytest.raises(ValueError, match=r"sealed|allow_test"):
        evaluate.run_policy(residual_run, "final", s, 1)


# --- The CLI -----------------------------------------------------------------------------------


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_scores_a_baseline_on_paper_into_one_parquet_row(tmp_path):
    """evaluate --controller zero_controls --split paper --out x.parquet returns 0 and writes one
    row (PAPER is one scenario): the identity columns (controller, checkpoint, split, location,
    growth_year, start_day, season_length) plus the result dict's keys, nothing else. Known
    answers: seasons are 60 days, so 5,760 steps; zero controls never fail the solver on PAPER
    (the golden zero_controls season), so valid. About 4 s."""
    import greenleaf_control.evaluate as evaluate

    out = tmp_path / "x.parquet"
    assert evaluate.main(_argv("zero_controls", "paper", out)) == 0

    df = pd.read_parquet(out)
    result_keys = set(_graded(envs.run_baseline("zero_controls", PAPER, 1), PAPER, 1))
    assert set(df.columns) == ROW_KEYS | result_keys
    [row] = df.to_dict("records")
    assert (row["controller"], row["split"], row["season_length"]) == ("zero_controls", "paper", 60)
    assert (row["location"], row["growth_year"], row["start_day"]) == ("Amsterdam", 2010, 59)
    assert row["steps"] == 60 * N_DAY
    assert bool(row["valid"]) is True


@pytest.mark.usefixtures("no_2012_reads")
def test_cli_scores_a_run_dir_with_its_final_checkpoint(residual_run, tmp_path):
    """--controller <run_dir> (a dir with manifest.json) is run_policy(run_dir, "final", s) over a
    60-day season: one row on PAPER with the controller as given, checkpoint "final", 5,760 steps,
    valid. About 4 s."""
    import greenleaf_control.evaluate as evaluate

    out = tmp_path / "x.parquet"
    assert evaluate.main(_argv(str(residual_run), "paper", out)) == 0

    [row] = pd.read_parquet(out).to_dict("records")
    assert (row["controller"], row["checkpoint"]) == (str(residual_run), "final")
    assert (row["split"], row["season_length"], row["steps"]) == ("paper", 60, 60 * N_DAY)
    assert bool(row["valid"]) is True


def test_cli_refuses_an_existing_out_before_evaluating(tmp_path, monkeypatch, capsys):
    """An existing --out is refused, naming it, and left as it was. The refusal comes before any
    env is built, so a 60-day season isn't played only to be thrown away."""
    import greenleaf_control.evaluate as evaluate

    out = tmp_path / "x.parquet"
    out.write_bytes(b"an earlier result\n")
    _no_env_built(monkeypatch, evaluate)

    with pytest.raises((SystemExit, FileExistsError)) as excinfo:
        evaluate.main(_argv("zero_controls", "paper", out))
    assert "x.parquet" in _refusal(excinfo, capsys)
    assert out.read_bytes() == b"an earlier result\n"


@pytest.mark.parametrize("make", ["missing", "no-manifest"])
def test_cli_refuses_a_controller_that_is_no_baseline_and_no_run_dir(
    make, tmp_path, monkeypatch, capsys
):
    """--controller must be a BASELINES name or a dir holding manifest.json. A path that doesn't
    exist, and a dir without manifest.json, are refused naming the controller, before any env and
    with no --out written."""
    import greenleaf_control.evaluate as evaluate

    controller = tmp_path / "not-a-run"
    if make == "no-manifest":
        controller.mkdir()
    out = tmp_path / "x.parquet"
    _no_env_built(monkeypatch, evaluate)

    with pytest.raises((SystemExit, ValueError, FileNotFoundError)) as excinfo:
        evaluate.main(_argv(str(controller), "paper", out))
    assert "not-a-run" in _refusal(excinfo, capsys)
    assert not out.exists()


@pytest.mark.slow
@pytest.mark.usefixtures("no_2012_reads")
def test_cli_rows_are_run_rb_and_run_policy_per_scenario(residual_run, tmp_path):
    """The CLI's numbers are the library's: --controller rb_default --split validation writes one
    row per VALIDATION scenario, in order, each == run_rb(default params, s) over 60 days; and
    --controller <run_dir> --split paper writes one row == run_policy(run_dir, "final", PAPER)
    over 60 days. Eight 60-day seasons, about 30 s."""
    import greenleaf_control.evaluate as evaluate

    out = tmp_path / "rb.parquet"
    assert evaluate.main(_argv("rb_default", "validation", out)) == 0
    rows = pd.read_parquet(out).to_dict("records")
    assert [(r["location"], r["growth_year"], r["start_day"]) for r in rows] == [
        (s["location"], s["growth_year"], s["start_day"]) for s in scenarios.VALIDATION
    ]
    for row, s in zip(rows, scenarios.VALIDATION, strict=True):
        want, _ = evaluate.run_rb(RB_DEFAULT, s)
        assert {k: row[k] for k in want} == want, s
        assert (row["controller"], row["split"], row["season_length"]) == (
            "rb_default",
            "validation",
            60,
        )

    out = tmp_path / "policy.parquet"
    assert evaluate.main(_argv(str(residual_run), "paper", out)) == 0
    [row] = pd.read_parquet(out).to_dict("records")
    want, _ = evaluate.run_policy(residual_run, "final", PAPER)
    assert {k: row[k] for k in want} == want


def test_cli_runs_as_a_module_and_lists_its_flags_and_every_split(tmp_path):
    """python -m greenleaf_control.evaluate --help exits 0 and lists --controller, --split, --out
    and --final, and every split in scenarios.SPLITS (the test splits too: refusing them is
    evaluate's seal, not an unknown choice)."""
    result = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.evaluate", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    missing = [word for word in (*CLI_FLAGS, *scenarios.SPLITS) if word not in result.stdout]
    assert not missing, f"--help doesn't list {missing}"


# --- From 13a's review --------------------------------------------------------------------------


@pytest.mark.parametrize(("scale", "expected"), [(0.3, 0.3), (None, 0.15)])
def test_build_vec_env_builds_the_residual_arm_at_the_recorded_scale(scale, expected):
    """build_vec_env passes the config's residual_scale to the co-pilot, so run_policy rebuilds a
    run at the scale it trained with: 0.3 gives ResidualRBWrapper.scale 0.3; None (no recorded
    scale) leaves make_env's default, 0.15 (PROTOCOL §3)."""
    config = {
        "method": "residual",
        "seed": 0,
        "season_length": 1,
        "penalty_weight": 1.0,
        "rb_params": RB_DEFAULT,
        "residual_scale": scale,
    }
    venv = train.build_vec_env(config, 1)
    try:
        assert _residual_wrapper(_chain(venv)[-1].envs[0]).scale == expected
    finally:
        venv.close()


def test_signatures_match_the_contract_and_no_test_wrote_the_real_ledger_or_runs():
    """run_rb(params, scenario, season_length=60, record_steps=False), run_policy(run_dir,
    checkpoint, scenario, season_length=60, record_steps=False) and main(argv=None).

    This test runs last in the module and checks the real ledger's size and mtime, and the
    listing of control/runs/, are as they were when the module was collected.
    """
    import greenleaf_control.evaluate as evaluate

    rb = inspect.signature(evaluate.run_rb).parameters
    assert list(rb) == ["params", "scenario", "season_length", "record_steps"]
    policy = inspect.signature(evaluate.run_policy).parameters
    assert list(policy) == ["run_dir", "checkpoint", "scenario", "season_length", "record_steps"]
    for params in (rb, policy):
        assert (params["season_length"].default, params["record_steps"].default) == (60, False)
    main = inspect.signature(evaluate.main).parameters
    assert list(main) == ["argv"]
    assert main["argv"].default is None
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE

"""Tests for greenleaf_control.envs (issue #9, core: a scorer file).

envs.py builds every method's env one way (docs/API-CONTRACT.md). make_raw_env is upstream's tomato
env (gl_gym @2d3febb) with our ConstraintCostReward and WEATHER_DIR, a season that ends after exactly
N steps, a flat Box observation, and season progress when asked. make_env adds the co-pilot
(ResidualRBWrapper) for the residual arms and always puts SolverFailureFlag on top. A TEST (2012)
scenario is refused unless allow_test=True, which make_env and run_baseline pass through to
make_raw_env (keyword-only, default False; issue #41).

Known answers used throughout:
- One day is 86,400 s / 900 s = 96 steps; 60 days is 5,760. Upstream ends `terminated` on step
  N + 1 (_terminalState checks timestep >= N before timestep += 1); Aarmen's decision B
  (2026-10-28) ends our seasons on step N.
- Upstream's 5 observation modules flatten to 22 dims (crop 3, controls 6, indoor climate 4,
  weather 5, time 4). Season progress adds 1 (23); the residual arms add the next RB action (29).
- Upstream reports the applied controls as info["controls"] and integrates with env.unwrapped.F,
  whose failures it swallows (bare except, prints "Error in ODE approximation", truncates).

The 2012 weather is sealed (PROTOCOL §8): tests only reset an env on it to check that it loads,
never step it, and reduce any check on it to a bool so no weather value is printed. The refusal
tests run under a tripwire on the file read itself (pandas.read_csv, which upstream's loader
calls): it fails before any 2012.csv is opened, including the next year's file upstream appends
when a season plus its 0.5-day lookahead runs past the end of the year (so a late 2011 season
reads 2012.csv although its scenario says 2011). The allow_test tests that step a TEST scenario
(issue #41) run on a synthetic weather dir in tmp_path whose 2012.csv files are byte copies of the
real Amsterdam/2010.csv, under a tripwire that fails on any 2012.csv outside tmp_path.

Issue #71 seals 2013 the same way (scenarios.SEALS = {2012: TEST, 2013: TEST_2013}; is_test is
growth_year in SEALS, so envs.py itself is untouched): the refusal tripwire fails before any 2013.csv
is opened too, and the one test that opens 2013 weather only resets on it with allow_test=True.
"""

import functools
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import gl_gym  # noqa: F401  (importing gl_gym registers its env ids with gymnasium)
import gl_gym.environments.greenlight_env as greenlight_env
import gymnasium as gym
import hypothesis.extra.numpy as hnp
import numpy as np
import pandas as pd
import pytest
from gl_gym.components.weather import FixedWeatherSampler, RandomWeatherSampler
from gl_gym.core.types import RewardContext
from gymnasium.utils.env_checker import check_env
from hypothesis import given, settings
from hypothesis import strategies as st

from greenleaf_control import rule_based, scenarios
from greenleaf_control.rewards import ConstraintCostReward

# greenleaf_control.envs is imported inside each test body (or a helper it calls), so while it is
# missing each test FAILS rather than ERRORs at collection.

ENV_ID = "gl_gym/GreenLightTomato-v0"
METHODS = ("ppo_paper", "ppo", "residual", "residual_lag")
BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned")
RESIDUAL = ("residual", "residual_lag")
LEARNING = ("ppo", "residual", "residual_lag")  # TRAIN sampler, plus season progress
N_DAY = 96  # 86,400 s / 900 s
UPSTREAM_DIM = 22  # 3 crop + 6 controls + 4 indoor climate + 5 weather + 4 time
DIMS = {"ppo_paper": 22, "ppo": 23, "residual": 29, "residual_lag": 29}
# configs/rule_based_tuned.yml (make_env's rb_params=None default) doesn't exist until E1.
RB_DEFAULT = rule_based.load_rb_params("default")
# A plain upstream env on our weather files, for step-for-step parity.
UP_KW = {"season_length": 1, "weather_data_dir": str(scenarios.WEATHER_DIR)}
TEST_NAMES = list(scenarios.TEST)
TEST_SCENARIOS = [s for name in TEST_NAMES for s in scenarios.TEST[name]]
_kw = scenarios.TRAIN_SAMPLER_KWARGS
TRAIN_SUPPORT = {
    (loc, year, day)
    for loc in _kw["locations"]
    for year in _kw["growth_years"]
    for day in _kw["start_days"]
}
ZERO = np.zeros(6, dtype=np.float32)
N_STATES = 28  # upstream's state vector x (GreenLightEnv nx)
# Day 340 + 60 days + the 0.5-day lookahead runs to day 400.5, past the year's 365 days, so
# upstream appends the next year's file: 2011.csv for LATE_2010 (allowed), 2012.csv for LATE_2011.
LATE_2010 = {"location": "Amsterdam", "growth_year": 2010, "start_day": 340}
LATE_2011 = {"location": "Amsterdam", "growth_year": 2011, "start_day": 340}
# "2012" != 2012, but upstream reads <location>/2012.csv for it all the same.
STRING_2012 = {"location": "Amsterdam", "growth_year": "2012", "start_day": 45}
# Issue #71: the 2013 seal's sites and splits, written out so collection can't ERROR while
# scenarios.TEST_2013 is missing; test_scenarios pins TEST_2013 to them.
SITES_2013 = {
    "amsterdam_2013": "Amsterdam",
    "london_2013": "London",
    "abbotsford_2013": "Abbotsford",
}


def _only(s: dict) -> dict:
    """sampler_kwargs whose only possible draw is scenario s."""
    return {
        "locations": [s["location"]],
        "growth_years": [s["growth_year"]],
        "start_days": [s["start_day"]],
    }


def _make(method: str, *, seed: int = 0, **kwargs) -> gym.Env:
    """make_env(method) for a 1-day season, with the default RB parameters."""
    import greenleaf_control.envs as envs

    return envs.make_env(method, seed=seed, season_length=1, rb_params=RB_DEFAULT, **kwargs)


def _wrappers(env) -> list[type]:
    """The wrapper classes from the outside in, down to the raw env."""
    chain = []
    while isinstance(env, gym.Wrapper):
        chain.append(type(env))
        env = env.env
    return chain


def _has(chain: list[type], cls: type) -> bool:
    return any(issubclass(t, cls) for t in chain)


def _finite(info: dict) -> bool:
    """Every number and array in a step's info is finite (bools and strings are skipped)."""
    return all(
        bool(np.all(np.isfinite(v)))
        for v in info.values()
        if isinstance(v, (int, float, np.number, np.ndarray))
        and not isinstance(v, (bool, np.bool_))
    )


def _controls(infos: list[dict]) -> np.ndarray:
    return np.array([info["controls"] for info in infos])


def _random_controls(start_day: int) -> np.ndarray:
    """The random baseline's actions: one generator seeded by the start day, one draw per step."""
    rng = np.random.default_rng(start_day)
    return np.array([rng.uniform(0, 1, 6).astype(np.float32) for _ in range(N_DAY)])


def _flaky(real_f, fail_on: int):
    """The real CasADi integrator, except that call number fail_on raises (a faked solver failure)."""
    calls = 0

    def f(**kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError("injected solver failure")
        return real_f(**kwargs)

    return f


def _record_flags(monkeypatch, envs) -> list[tuple[bool, bool]]:
    """Spy on SolverFailureFlag.step (the real one still runs): each step's (terminated, truncated)."""
    real_step = envs.SolverFailureFlag.step
    flags = []

    def step(self, action):
        out = real_step(self, action)
        flags.append((bool(out[2]), bool(out[3])))
        return out

    monkeypatch.setattr(envs.SolverFailureFlag, "step", step)
    return flags


@pytest.fixture
def seal_tripwire(monkeypatch) -> list[tuple[str, ...]]:
    """Fail loudly if the test opens any 2012 or 2013 (#71) weather file: a refusal must come before
    the read.

    It sits on pandas.read_csv, the only file read upstream's loader makes: load_weather_data reads
    <location>/<year>.csv, and expand_weather_data reads <location>/<year + 1>.csv when the season
    plus the 0.5-day lookahead runs past the end of the year file. So it also trips on a late 2011
    season, which WeatherRepository.load sees as 2011. The check runs before the real read; every
    other file goes through the real one. Returns the (location, file name) of each file read.
    """
    real_read_csv = pd.read_csv
    reads = []

    def read_csv(filepath_or_buffer, *args, **kwargs):
        path = Path(str(filepath_or_buffer))
        if path.name in ("2012.csv", "2013.csv"):
            raise AssertionError(f"the sealed {path.stem} test weather was read: {path}")
        reads.append(path.parts[-2:])
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    return reads


@pytest.fixture
def synthetic_2012(tmp_path, monkeypatch) -> SimpleNamespace:
    """A synthetic weather dir, tmp_path/"synthetic", where Amsterdam, London and Abbotsford each
    have a 2012.csv that is a byte copy of the real Amsterdam/2010.csv, under a pandas.read_csv
    tripwire that fails on any *2012.csv outside tmp_path (the real data dir, gl_gym's own copy).

    scenarios.WEATHER_DIR is not patched here: a test points it at .dir itself, after building
    anything it compares against on the real dir. .reads lists every 2012.csv read (resolved, so
    /var and /private/var compare equal on the Mac); .trips every one the tripwire refused, in case
    upstream swallows the AssertionError.
    """
    synthetic = tmp_path / "synthetic"
    for site in ("Amsterdam", "London", "Abbotsford"):
        (synthetic / site).mkdir(parents=True)
        shutil.copyfile(
            scenarios.WEATHER_DIR / "Amsterdam" / "2010.csv", synthetic / site / "2012.csv"
        )
    root = tmp_path.resolve()
    real_read_csv = pd.read_csv
    out = SimpleNamespace(dir=synthetic, reads=[], trips=[])

    def read_csv(filepath_or_buffer, *args, **kwargs):
        path = Path(str(filepath_or_buffer)).resolve()
        if path.name.endswith("2012.csv"):
            if not path.is_relative_to(root):
                out.trips.append(path)
                raise AssertionError(f"the sealed 2012 test weather was read: {path}")
            out.reads.append(path)
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    return out


def _assert_read_only_synthetic_2012(synthetic: SimpleNamespace, location: str) -> None:
    """The tripwire never fired, and the only 2012.csv read was location's synthetic copy (so the
    test did open TEST-year weather, and from tmp_path)."""
    assert synthetic.trips == [], f"a real 2012.csv was read: {synthetic.trips}"
    assert set(synthetic.reads) == {(synthetic.dir / location / "2012.csv").resolve()}


def _takes_allow_test(fn) -> bool:
    """fn has a keyword-only allow_test defaulting to False (issue #41)."""
    p = inspect.signature(fn).parameters.get("allow_test")
    return p is not None and p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False


# --- The contract surface ----------------------------------------------------------------------


def test_methods_and_baselines_are_the_contract_tuples():
    import greenleaf_control.envs as envs

    assert envs.METHODS == METHODS
    assert envs.BASELINES == BASELINES


def test_signatures_match_the_contract():
    """make_raw_env is keyword-only with the contract's defaults (normalize_actions has none);
    make_env and run_baseline take allow_test keyword-only, default False (issue #41), so with
    the default they can never open a TEST scenario. All three take price_regime keyword-only,
    default "sim" (issue #67), so every earlier call keeps upstream's registered prices."""
    import greenleaf_control.envs as envs

    empty = inspect.Parameter.empty
    raw = inspect.signature(envs.make_raw_env).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in raw.values())
    assert {name: p.default for name, p in raw.items()} == {
        "scenario": None,
        "sampler_kwargs": None,
        "season_length": 60,
        "normalize_actions": empty,
        "penalty_weight": 1.0,
        "season_progress": False,
        "allow_test": False,
        "price_regime": "sim",
    }
    make = inspect.signature(envs.make_env).parameters
    assert {name: p.default for name, p in make.items()} == {
        "method": empty,
        "seed": empty,
        "scenario": None,
        "season_length": 60,
        "penalty_weight": 1.0,
        "rb_params": None,
        "residual_scale": 0.15,
        "allow_test": False,
        "price_regime": "sim",
    }
    assert make["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert make["allow_test"].kind is inspect.Parameter.KEYWORD_ONLY
    assert make["price_regime"].kind is inspect.Parameter.KEYWORD_ONLY
    baseline = inspect.signature(envs.run_baseline).parameters
    assert {name: p.default for name, p in baseline.items()} == {
        "name": empty,
        "scenario": empty,
        "season_length": 60,
        "allow_test": False,
        "price_regime": "sim",
    }
    assert baseline["allow_test"].kind is inspect.Parameter.KEYWORD_ONLY
    assert baseline["price_regime"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("method", METHODS)
def test_check_env_passes(method):
    """Gymnasium 0.29's check_env passes on make_env(method) for a 1-day season.

    It checks the spaces, that reset(seed=123) twice gives equal observations inside the space, a
    reset with options, one step, and finally rebuilds the env from env.spec, so every wrapper in
    the chain must be recreatable from it (gymnasium.utils.RecordConstructorArgs). Its warnings
    (such as a Box(0, 1) action space) don't fail it.
    """
    check_env(_make(method))


@pytest.mark.parametrize("method", METHODS)
def test_observation_is_a_flat_float32_box_of_the_method_dim(method):
    """ppo_paper 22 (upstream's), ppo 23 (+ progress), residual and residual_lag 29 (+ 6 RB dims).

    Every method acts in Box(-1, 1, (6,)): normalized rate actions for ppo_paper and ppo, the
    residual for the co-pilot. Season progress is bounded [0, 1], and so are the 6 RB dims.
    """
    env = _make(method)
    dim = DIMS[method]
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (dim,)
    assert space.dtype == np.float32
    assert env.action_space == gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
    if method in LEARNING:
        assert (space.low[UPSTREAM_DIM], space.high[UPSTREAM_DIM]) == (0.0, 1.0)
    if method in RESIDUAL:
        np.testing.assert_array_equal(space.low[-6:], np.zeros(6))
        np.testing.assert_array_equal(space.high[-6:], np.ones(6))
    obs, _ = env.reset()
    assert obs.shape == (dim,)
    assert obs.dtype == np.float32
    assert obs in space


def test_residual_obs_is_ppo_obs_plus_6_and_ppo_is_ppo_paper_plus_progress():
    dims = {m: _make(m).observation_space.shape[0] for m in METHODS}
    assert dims["residual"] == dims["ppo"] + 6
    assert dims["residual_lag"] == dims["ppo"] + 6
    assert dims["ppo"] == dims["ppo_paper"] + 1 == UPSTREAM_DIM + 1


@pytest.mark.parametrize("method", LEARNING)
def test_learning_arms_carry_season_progress_after_upstreams_22_dims(method):
    """Dim 22 (the last for ppo, the 7th from last for the residual arms) is timestep / 96: 0 after
    reset, then 1/96, 2/96, ... after each step."""
    env = _make(method, scenario=scenarios.PAPER)
    obs, _ = env.reset(seed=0)
    seen = [float(obs[UPSTREAM_DIM])]
    for _ in range(10):
        obs, *_ = env.step(ZERO)
        seen.append(float(obs[UPSTREAM_DIM]))
    assert seen == pytest.approx([k / N_DAY for k in range(11)], abs=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_wrapper_chain_per_method(method):
    """SolverFailureFlag is outermost on every method; ResidualRBWrapper only on the residual arms;
    SeasonProgress on every arm but ppo_paper."""
    import greenleaf_control.envs as envs

    env = _make(method)
    chain = _wrappers(env)
    assert isinstance(env, envs.SolverFailureFlag)
    assert _has(chain, envs.ResidualRBWrapper) is (method in RESIDUAL)
    assert _has(chain, envs.SeasonProgress) is (method in LEARNING)
    assert not _has(chain, gym.wrappers.TimeLimit)  # nothing truncates for time


# --- Step-for-step parity with upstream ---------------------------------------------------------


def _assert_obs_matches_upstream(method, obs, up, up_obs, k, controller):
    """Our observation after k steps is upstream's flattened one, then progress k/96 (not on
    ppo_paper), then the RB action for upstream's current state (residual arms only)."""
    np.testing.assert_array_equal(
        obs[:UPSTREAM_DIM], gym.spaces.flatten(up.observation_space, up_obs)
    )
    if method == "ppo_paper":
        assert obs.shape == (UPSTREAM_DIM,)
        return
    assert float(obs[UPSTREAM_DIM]) == pytest.approx(k / N_DAY, abs=1e-6)
    if method in RESIDUAL:
        np.testing.assert_array_equal(obs[UPSTREAM_DIM + 1 :], rule_based.rb_action(up, controller))


@pytest.mark.parametrize("method", METHODS)
def test_each_method_matches_upstream_on_paper_step_for_step(method):
    """Each method on PAPER for one day, against a plain upstream env stepped with the same controls.

    Known answers, at every one of 96 steps: the same applied controls; info["paper_reward"] is
    upstream's reward (so the registered prices reached ConstraintCostReward); our reward is
    upstream's at w = 1 and scaled_profit for residual_lag (w forced to 0); our observation is
    upstream's flattened (ppo_paper keeps upstream's modules and nothing else) plus progress and
    the next RB action where the method adds them. Ours ends terminated on step 96, no solver
    failure; upstream's own env is still running then (it ends on step 97).
    """
    env = _make(method, scenario=scenarios.PAPER)
    residual = method in RESIDUAL
    up = gym.make(ENV_ID, normalize_actions=not residual, **UP_KW)
    controller = rule_based.make_controller(RB_DEFAULT)
    obs, info = env.reset(seed=0)
    up_obs, up_info = up.reset(seed=0)
    assert info["scenario"] == up_info["scenario"] == scenarios.PAPER
    actions = np.random.default_rng(7).uniform(-1, 1, (N_DAY, 6)).astype(np.float32)
    for k, a in enumerate(actions):
        _assert_obs_matches_upstream(method, obs, up, up_obs, k, controller)
        obs, reward, terminated, truncated, info = env.step(a)
        up_obs, up_reward, up_terminated, _, up_info = up.step(info["controls"] if residual else a)
        np.testing.assert_array_equal(info["controls"], up_info["controls"])
        assert info["paper_reward"] == up_reward
        assert reward == (info["scaled_profit"] if method == "residual_lag" else up_reward)
        assert (bool(terminated), bool(truncated)) == (k == N_DAY - 1, False), k
        assert info["solver_failure"] is False, k
    _assert_obs_matches_upstream(method, obs, up, up_obs, N_DAY, controller)
    assert up_terminated is False


# --- Season length and season progress (decision B) ---------------------------------------------


@pytest.mark.parametrize(("days", "steps"), [(1, 96), (2, 192)])
def test_raw_episode_is_exactly_n_steps_ending_terminated(days, steps):
    """A make_raw_env season of d days is exactly d x 96 steps: terminated on the last, never
    before, never truncated. The observation is a flat 22-dim Box without season progress."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(scenario=scenarios.PAPER, season_length=days, normalize_actions=False)
    assert env.unwrapped.N == steps
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (UPSTREAM_DIM,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (UPSTREAM_DIM,)
    flags = []
    for _ in range(steps + 2):
        _, _, terminated, truncated, _ = env.step(ZERO)
        flags.append((bool(terminated), bool(truncated)))
        if terminated or truncated:
            break
    assert flags == [(False, False)] * (steps - 1) + [(True, False)]


def test_season_progress_goes_from_0_to_n_minus_1_over_n_linearly():
    """The observations an action is taken from carry progress 0, 1/96, ..., 95/96: 96 decisions,
    evenly spaced by 1/96. The terminal observation, after step 96, carries 96/96 = 1."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(
        scenario=scenarios.PAPER, season_length=1, normalize_actions=True, season_progress=True
    )
    assert isinstance(env, envs.SeasonProgress)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (UPSTREAM_DIM + 1,)
    assert (space.low[-1], space.high[-1]) == (0.0, 1.0)
    obs, _ = env.reset(seed=0)
    decisions = []
    for _ in range(N_DAY + 2):
        decisions.append(float(obs[-1]))
        obs, _, terminated, truncated, _ = env.step(ZERO)
        assert not truncated
        if terminated:
            break
    assert len(decisions) == N_DAY
    assert decisions == pytest.approx([k / N_DAY for k in range(N_DAY)], abs=1e-6)
    assert np.diff(decisions) == pytest.approx(np.full(N_DAY - 1, 1 / N_DAY), abs=1e-6)
    assert float(obs[-1]) == pytest.approx(1.0, abs=1e-6)


# --- The co-pilot: ResidualRBWrapper ------------------------------------------------------------


def test_residual_wrapper_on_a_raw_env_appends_the_next_rb_action():
    """ResidualRBWrapper(env, controller, scale) on a raw PAPER env (no progress): acts in
    Box(-1, 1, (6,)); observes the inner 22 dims (same bounds) plus 6 in [0, 1], which equal the RB
    action for the state the agent now sees, at reset and after every step."""
    import greenleaf_control.envs as envs

    raw = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    env = envs.ResidualRBWrapper(raw, rule_based.make_controller(RB_DEFAULT), scale=0.15)
    check = rule_based.make_controller(RB_DEFAULT)
    assert env.action_space == gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
    space = env.observation_space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (UPSTREAM_DIM + 6,)
    np.testing.assert_array_equal(space.low[:-6], raw.observation_space.low)
    np.testing.assert_array_equal(space.high[:-6], raw.observation_space.high)
    np.testing.assert_array_equal(space.low[-6:], np.zeros(6))
    np.testing.assert_array_equal(space.high[-6:], np.ones(6))
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        assert obs.dtype == np.float32
        assert obs.shape == (UPSTREAM_DIM + 6,)
        np.testing.assert_array_equal(obs[-6:], rule_based.rb_action(env, check))
        obs, *_ = env.step(ZERO)


@pytest.mark.parametrize("method", RESIDUAL)
def test_zero_residual_applies_the_rule_based_action(method):
    """With a zero residual, all 96 steps apply exactly the RB action: info["controls"] equals
    info["u_rb"] (atol 1e-6), u_rb is the default rules' action for that step's state, and
    u_residual is 0. The RB action changes over the day, so this isn't a constant."""
    env = _make(method, scenario=scenarios.PAPER)
    check = rule_based.make_controller(RB_DEFAULT)
    env.reset(seed=0)
    u_rbs = []
    for k in range(N_DAY):
        expected = rule_based.rb_action(env, check)
        _, _, terminated, truncated, info = env.step(ZERO)
        np.testing.assert_allclose(info["controls"], info["u_rb"], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(info["u_rb"], expected)
        np.testing.assert_array_equal(info["u_residual"], np.zeros(6))
        assert not truncated, k
        u_rbs.append(info["u_rb"])
    assert terminated
    assert np.unique(np.array(u_rbs), axis=0).shape[0] > 1


@pytest.mark.parametrize("scale", [0.15, 0.5], ids=["default", "0.5"])
def test_residual_scale_sets_the_correction_size(scale):
    """u_residual = scale * a and controls = clip(u_rb + scale * a, 0, 1); 0.15 is the default.
    With a = (1, -1, 0.5, -0.5, 0.2, 0) and scale 0.5, the correction is (0.5, -0.5, 0.25, ...)."""
    kwargs = {} if scale == 0.15 else {"residual_scale": scale}
    env = _make("residual", scenario=scenarios.PAPER, **kwargs)
    obs, _ = env.reset(seed=0)
    u_rb = obs[-6:].astype(np.float64)
    a = np.array([1.0, -1.0, 0.5, -0.5, 0.2, 0.0], dtype=np.float32)
    _, _, _, _, info = env.step(a)
    correction = scale * a.astype(np.float64)
    np.testing.assert_allclose(info["u_residual"], correction, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        info["controls"], np.clip(u_rb + correction, 0, 1), rtol=0, atol=1e-6
    )


@functools.cache
def _residual_day() -> gym.Env:
    """One 1-day residual env (TRAIN sampler), built once and reset by every hypothesis example."""
    import greenleaf_control.envs as envs

    return envs.make_env("residual", seed=0, season_length=1, rb_params=RB_DEFAULT)


def _check_residual_step(a: np.ndarray, steps: int, seed: int) -> None:
    """From a random state (a seeded TRAIN scenario, then `steps` seeded random residual steps),
    one step with residual a applies clip(u_rb + 0.15 a, 0, 1), which stays in [0, 1]."""
    env = _residual_day()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        obs, *_ = env.step(rng.uniform(-1, 1, 6).astype(np.float32))
    u_rb = obs[-6:].copy()
    _, _, _, _, info = env.step(a)
    controls = np.asarray(info["controls"], dtype=np.float64)
    in_box = bool(np.all((controls >= 0.0) & (controls <= 1.0)))
    assert in_box, f"controls {controls} left [0, 1] for residual {a}"
    a64 = a.astype(np.float64)
    np.testing.assert_array_equal(info["u_rb"], u_rb)
    np.testing.assert_allclose(info["u_residual"], 0.15 * a64, rtol=1e-6, atol=1e-7)
    expected = np.clip(u_rb.astype(np.float64) + 0.15 * a64, 0.0, 1.0)
    np.testing.assert_allclose(controls, expected, rtol=0, atol=1e-6)


@settings(max_examples=15, deadline=None, derandomize=True, database=None)
@given(
    a=hnp.arrays(np.float32, 6, elements=st.floats(-1, 1, width=32)),
    steps=st.integers(0, N_DAY - 1),
    seed=st.integers(0, 2**16),
)
def test_residual_controls_stay_in_unit_box_for_any_action_in_bounds(a, steps, seed):
    """Property: for any residual in [-1, 1]^6 from a random state, controls stay in [0, 1]."""
    _check_residual_step(a, steps, seed)


@settings(max_examples=15, deadline=None, derandomize=True, database=None)
@given(
    a=hnp.arrays(np.float32, 6, elements=st.floats(-1e3, 1e3, width=32)),
    steps=st.integers(0, N_DAY - 1),
    seed=st.integers(0, 2**16),
)
def test_residual_controls_stay_in_unit_box_beyond_the_action_bounds(a, steps, seed):
    """The wrapper doesn't pre-clip a, so up to |a| = 1e3 (a correction of 150) the control is still
    clip(u_rb + 0.15 a, 0, 1), in [0, 1], and u_residual is the unclipped 0.15 a."""
    _check_residual_step(a, steps, seed)


# --- SolverFailureFlag --------------------------------------------------------------------------


@pytest.mark.parametrize("fail_on", [1, 50, N_DAY], ids=["first", "mid", "last"])
@pytest.mark.parametrize("method", METHODS)
def test_solver_failure_truncates_and_is_flagged(method, fail_on, monkeypatch, capsys):
    """env.unwrapped.F raising on step k (faked) makes upstream print "Error in ODE approximation"
    and truncate: that step is truncated with info["solver_failure"] True, every step before it is
    clean (False). A failure on step 96, the last, is flagged too (and also terminates)."""
    env = _make(method, scenario=scenarios.PAPER)
    env.reset(seed=0)
    monkeypatch.setattr(env.unwrapped, "F", _flaky(env.unwrapped.F, fail_on))
    for step in range(1, fail_on + 1):
        _, _, terminated, truncated, info = env.step(ZERO)
        if step < fail_on:
            assert not truncated, step
            assert info["solver_failure"] is False, step
    assert bool(truncated) is True
    assert info["solver_failure"] is True
    assert bool(terminated) is (fail_on == N_DAY)
    assert "Error in ODE approximation" in capsys.readouterr().out


def test_run_baseline_stops_at_a_solver_failure(monkeypatch, capsys):
    """A solver failing on step 10 (faked through upstream's define_model) ends run_baseline's
    episode there: 10 infos, only the last flagged."""
    import greenleaf_control.envs as envs

    real_define_model = greenlight_env.define_model
    monkeypatch.setattr(
        greenlight_env, "define_model", lambda **kw: _flaky(real_define_model(**kw), fail_on=10)
    )
    infos = envs.run_baseline("zero_controls", scenarios.PAPER, season_length=1)
    assert [info["solver_failure"] for info in infos] == [False] * 9 + [True]
    assert "Error in ODE approximation" in capsys.readouterr().out


# --- The raw states in every step's info --------------------------------------------------------


def _assert_has_states(info: dict) -> None:
    """info carries x_prev and x, each a float64 array of upstream's 28 states."""
    for key in ("x_prev", "x"):
        assert key in info, f"info has no {key!r}"
        assert isinstance(info[key], np.ndarray), key
        assert info[key].shape == (N_STATES,), key
        assert info[key].dtype == np.float64, key


def test_make_raw_env_info_carries_the_states_before_and_after_the_step():
    """Upstream's step sets x_prev = copy(x), then integrates x. So after each of 10 steps (random
    absolute controls on PAPER), info["x_prev"] equals env.unwrapped.x just before the step and
    info["x"] equals it right after. Both are copies: they share no memory with the env's arrays,
    and changing them leaves the env's state alone."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    env.reset(seed=0)
    raw = env.unwrapped
    for k, a in enumerate(np.random.default_rng(3).uniform(0, 1, (10, 6)).astype(np.float32)):
        before = raw.x.copy()
        _, _, _, _, info = env.step(a)
        _assert_has_states(info)
        np.testing.assert_array_equal(info["x_prev"], before, err_msg=f"step {k}")
        np.testing.assert_array_equal(info["x"], raw.x, err_msg=f"step {k}")
        assert not np.array_equal(info["x"], info["x_prev"]), k  # the step moved the state
        assert not np.shares_memory(info["x"], raw.x), k
        assert not np.shares_memory(info["x_prev"], raw.x_prev), k
        after = raw.x.copy()
        info["x"] += 1.0
        info["x_prev"] += 1.0
        np.testing.assert_array_equal(raw.x, after, err_msg=f"step {k}")
        np.testing.assert_array_equal(raw.x_prev, before, err_msg=f"step {k}")


@pytest.mark.parametrize("method", METHODS)
def test_every_method_info_carries_the_raw_states(method):
    """make_env puts the raw env's states in the info for every method (under the co-pilot and
    SolverFailureFlag too): after one step, x_prev is unwrapped.x before it and x is unwrapped.x
    after it."""
    env = _make(method, scenario=scenarios.PAPER)
    env.reset(seed=0)
    before = env.unwrapped.x.copy()
    _, _, _, _, info = env.step(ZERO)
    _assert_has_states(info)
    np.testing.assert_array_equal(info["x_prev"], before)
    np.testing.assert_array_equal(info["x"], env.unwrapped.x)


def test_run_baseline_infos_carry_each_steps_raw_states():
    """Every info of a 1-day rb_default run on PAPER carries x_prev, the 28 states the step started
    from, and x, the states it ended on. Known answers: step t starts exactly where step t - 1
    ended; step 1 starts from upstream's initial state on PAPER; and each x is the state of a plain
    upstream env after the same controls. They are copies: changing an info's x leaves the next
    info's x_prev alone."""
    import greenleaf_control.envs as envs

    infos = envs.run_baseline("rb_default", scenarios.PAPER, season_length=1)
    assert len(infos) == N_DAY
    for info in infos:
        _assert_has_states(info)
    for t in range(1, N_DAY):
        np.testing.assert_array_equal(infos[t]["x_prev"], infos[t - 1]["x"], err_msg=f"step {t}")
    up = gym.make(ENV_ID, normalize_actions=False, **UP_KW)
    up.reset(seed=0, options={"scenario": scenarios.PAPER})
    np.testing.assert_array_equal(infos[0]["x_prev"], up.unwrapped.x)
    for t, info in enumerate(infos):
        up.step(info["controls"])
        np.testing.assert_array_equal(info["x"], up.unwrapped.x, err_msg=f"step {t}")
    up.close()
    next_starts = [info["x_prev"].copy() for info in infos[1:]]
    for info in infos[:-1]:
        info["x"] += 1.0
    for t, start in enumerate(next_starts, start=1):
        np.testing.assert_array_equal(infos[t]["x_prev"], start, err_msg=f"step {t}")


# --- The TEST seal ------------------------------------------------------------------------------


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize("name", TEST_NAMES)
def test_make_raw_env_refuses_every_test_scenario(name):
    """Constructing make_raw_env on any of the split's 8 scenarios raises a ValueError naming
    allow_test, before any 2012 weather loads."""
    import greenleaf_control.envs as envs

    for s in scenarios.TEST[name]:
        with pytest.raises(ValueError, match="allow_test"):
            envs.make_raw_env(scenario=s, season_length=1, normalize_actions=False)


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize("method", METHODS)
def test_make_env_refuses_every_test_scenario(method):
    """With the default allow_test=False, all 24 TEST scenarios are refused for every method."""
    for s in TEST_SCENARIOS:
        with pytest.raises(ValueError, match="allow_test"):
            _make(method, scenario=s)


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize("name", ["zero_controls", "random", "rb_default"])
def test_run_baseline_refuses_every_test_scenario(name):
    """With the default allow_test=False, all 24 TEST scenarios are refused, before a single
    step."""
    import greenleaf_control.envs as envs

    for s in TEST_SCENARIOS:
        with pytest.raises(ValueError, match="allow_test"):
            envs.run_baseline(name, s, season_length=1)


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize(
    "sampler_kwargs",
    [
        {"locations": ["Amsterdam"], "growth_years": [2012], "start_days": [59]},
        {
            "locations": ["Amsterdam"],
            "growth_years": [2011, 2012],
            "start_days": list(range(45, 76)),
        },
        {"locations": ["London", "Abbotsford"], "growth_years": [2012], "start_days": [45, 73]},
    ],
    ids=["only_2012", "2012_among_train_years", "other_sites_2012"],
)
def test_sampler_kwargs_that_can_draw_2012_are_refused(sampler_kwargs):
    """A random sampler that could draw any 2012 scenario is refused at construction."""
    import greenleaf_control.envs as envs

    with pytest.raises(ValueError, match="allow_test"):
        envs.make_raw_env(sampler_kwargs=sampler_kwargs, season_length=1, normalize_actions=True)


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize("method", ["raw", *METHODS])
def test_reset_onto_a_test_scenario_is_refused(method):
    """reset(options={"scenario": s}) bypasses upstream's sampler, so it is checked too: a TEST s
    raises a ValueError naming allow_test before its weather loads. The env stays usable, and a
    VALIDATION scenario passed the same way resets onto that scenario."""
    import greenleaf_control.envs as envs

    if method == "raw":
        env = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    else:
        env = _make(method)
    for name in TEST_NAMES:
        with pytest.raises(ValueError, match="allow_test"):
            env.reset(options={"scenario": scenarios.TEST[name][0]})
    _, info = env.reset(options={"scenario": scenarios.VALIDATION[0]})
    assert info["scenario"] == scenarios.VALIDATION[0]


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize("how", ["scenario", "sampler", "reset"])
@pytest.mark.parametrize(
    ("s", "days"), [(LATE_2011, 60), (STRING_2012, 1)], ids=["late_2011", "string_2012"]
)
def test_a_season_that_would_read_2012_is_refused(s, days, how):
    """A season that would read 2012.csv is TEST weather, whatever its scenario says, and is
    refused (a ValueError naming allow_test) before the read, three ways in: make_raw_env(scenario=s)
    and a sampler whose only draw is s at construction, and reset(options={"scenario": s}) on a
    PAPER env of the same season length.

    - late_2011: 2011 day 340 for 60 days runs to day 400.5 (with the 0.5-day lookahead), past
      2011's 365, so upstream appends Amsterdam/2012.csv.
    - string_2012: growth_year "2012" (a str) isn't == 2012, but upstream reads 2012.csv for it.
    """
    import greenleaf_control.envs as envs

    if how == "reset":
        env = envs.make_raw_env(
            scenario=scenarios.PAPER, season_length=days, normalize_actions=False
        )
        with pytest.raises(ValueError, match="allow_test"):
            env.reset(options={"scenario": s})
    else:
        weather = {"scenario": s} if how == "scenario" else {"sampler_kwargs": _only(s)}
        with pytest.raises(ValueError, match="allow_test"):
            envs.make_raw_env(**weather, season_length=days, normalize_actions=False)


@pytest.mark.usefixtures("seal_tripwire")
def test_make_env_refuses_a_train_season_long_enough_to_reach_2012():
    """TRAIN draws 2011 with start days up to 75: a 300-day season from day 75 runs to day 375.5,
    past 2011's 365, so upstream would append 2012.csv. A sampler that can draw that season is
    refused before make_env's reset, whichever scenario seed 0 happens to draw first."""
    import greenleaf_control.envs as envs

    with pytest.raises(ValueError, match="allow_test"):
        envs.make_env("ppo", seed=0, season_length=300)


@pytest.mark.parametrize("how", ["scenario", "sampler", "reset"])
def test_a_late_2010_season_that_reads_2011_is_allowed(how, seal_tripwire):
    """Control for the refusal above: 2010 day 340 for 60 days also runs past its year, but into
    2011, which isn't sealed. It builds and resets onto that scenario all three ways in, reading
    exactly Amsterdam/2010.csv then 2011.csv (so the season does cross the year). Reset only."""
    import greenleaf_control.envs as envs

    if how == "reset":
        env = envs.make_raw_env(scenario=scenarios.PAPER, season_length=60, normalize_actions=False)
        _, info = env.reset(options={"scenario": LATE_2010})
    else:
        weather = (
            {"scenario": LATE_2010} if how == "scenario" else {"sampler_kwargs": _only(LATE_2010)}
        )
        env = envs.make_raw_env(**weather, season_length=60, normalize_actions=False)
        _, info = env.reset(seed=0)
    env.close()
    assert info["scenario"] == LATE_2010
    assert seal_tripwire == [("Amsterdam", "2010.csv"), ("Amsterdam", "2011.csv")]


@pytest.mark.parametrize("name", TEST_NAMES)
def test_allow_test_lets_a_test_scenario_reset(name):
    """With allow_test=True a TEST scenario resets (PROTOCOL §8 allows a reset, nothing more): the
    env is on that scenario, with a finite 22-dim observation. No step is taken, and the checks are
    reduced to bools so no weather value is printed."""
    import greenleaf_control.envs as envs

    s = scenarios.TEST[name][0]
    env = envs.make_raw_env(scenario=s, season_length=1, normalize_actions=False, allow_test=True)
    obs, info = env.reset(seed=0)
    env.close()
    assert info["scenario"] == s
    shape_ok = obs.shape == (UPSTREAM_DIM,)
    assert shape_ok, f"wrong observation shape after resetting on {s}"
    finite = bool(np.all(np.isfinite(obs)))
    assert finite, f"non-finite observation after resetting on {s}"


@pytest.mark.usefixtures("seal_tripwire")
@pytest.mark.parametrize("how", ["scenario", "sampler", "reset"])
@pytest.mark.parametrize("site", SITES_2013.values())
def test_2013_weather_is_refused_without_allow_test(site, how):
    """Issue #71: 2013 is the second seal, refused like 2012. Every 2013 scenario (start days 45 to
    73 step 4) at the site raises a ValueError naming allow_test before its weather loads, three
    ways in: make_raw_env(scenario=s) and a sampler whose only draw is s at construction, and
    reset(options={"scenario": s}) on a PAPER env. A sampler that mixes 2013 into the TRAIN years
    is refused too. The env stays usable: a VALIDATION reset after the refusals works."""
    import greenleaf_control.envs as envs

    days = range(45, 74, 4)
    paper = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    for day in days:
        s = {"location": site, "growth_year": 2013, "start_day": day}
        with pytest.raises(ValueError, match="allow_test"):
            if how == "reset":
                paper.reset(options={"scenario": s})
            else:
                weather = {"scenario": s} if how == "scenario" else {"sampler_kwargs": _only(s)}
                envs.make_raw_env(**weather, season_length=1, normalize_actions=False)
    mixed = {"locations": [site], "growth_years": [2011, 2013], "start_days": list(days)}
    with pytest.raises(ValueError, match="allow_test"):
        envs.make_raw_env(sampler_kwargs=mixed, season_length=1, normalize_actions=True)
    _, info = paper.reset(options={"scenario": scenarios.VALIDATION[0]})
    paper.close()
    assert info["scenario"] == scenarios.VALIDATION[0]


@pytest.mark.parametrize("name", SITES_2013)
def test_allow_test_lets_a_2013_scenario_reset(name):
    """Issue #71: with allow_test=True the 2013 split's first scenario (scenarios.TEST_2013) resets,
    as a 2012 one does: the env is on that scenario, with a finite 22-dim observation. No step is
    taken, and the checks are reduced to bools so no weather value is printed."""
    import greenleaf_control.envs as envs

    s = scenarios.TEST_2013[name][0]
    assert s == {"location": SITES_2013[name], "growth_year": 2013, "start_day": 45}
    env = envs.make_raw_env(scenario=s, season_length=1, normalize_actions=False, allow_test=True)
    obs, info = env.reset(seed=0)
    env.close()
    assert info["scenario"] == s
    shape_ok = obs.shape == (UPSTREAM_DIM,)
    assert shape_ok, f"wrong observation shape after resetting on {s}"
    finite = bool(np.all(np.isfinite(obs)))
    assert finite, f"non-finite observation after resetting on {s}"


@pytest.mark.parametrize("name", TEST_NAMES)
@pytest.mark.parametrize("method", METHODS)
def test_make_env_with_allow_test_plays_a_synthetic_test_day(
    method, name, synthetic_2012, monkeypatch
):
    """Issue #41: make_env(method, scenario=<the split's first TEST scenario>, season_length=1,
    allow_test=True) on the synthetic weather dir resets onto that scenario and plays a whole day
    of zero actions: exactly 96 steps, terminated on the last and never truncated, each
    observation the method's dim and finite. Its only 2012 read is the site's synthetic copy."""
    import greenleaf_control.envs as envs

    assert _takes_allow_test(envs.make_env), "make_env takes no keyword-only allow_test=False"
    s = scenarios.TEST[name][0]
    monkeypatch.setattr(scenarios, "WEATHER_DIR", synthetic_2012.dir)
    env = _make(method, scenario=s, allow_test=True)
    obs, info = env.reset(seed=0)
    assert info["scenario"] == s
    observations, flags = [obs], []
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step(ZERO)
        observations.append(obs)
        flags.append((bool(terminated), bool(truncated)))
    env.close()
    assert flags == [(False, False)] * (N_DAY - 1) + [(True, False)]
    observations = np.stack(observations)
    assert observations.shape == (N_DAY + 1, DIMS[method])
    assert bool(np.all(np.isfinite(observations)))
    _assert_read_only_synthetic_2012(synthetic_2012, s["location"])


@pytest.mark.parametrize("name", TEST_NAMES)
@pytest.mark.parametrize("baseline", BASELINES)
def test_run_baseline_with_allow_test_returns_a_synthetic_test_day(
    baseline, name, synthetic_2012, monkeypatch
):
    """Issue #41: run_baseline(baseline, <the split's first TEST scenario>, 1, allow_test=True) on
    the synthetic weather dir returns one day, 96 infos, each carrying the raw states. Its only
    2012 read is the site's synthetic copy."""
    import greenleaf_control.envs as envs

    assert _takes_allow_test(envs.run_baseline), "run_baseline takes no keyword-only allow_test"
    s = scenarios.TEST[name][0]
    monkeypatch.setattr(scenarios, "WEATHER_DIR", synthetic_2012.dir)
    infos = envs.run_baseline(baseline, s, 1, allow_test=True)
    assert len(infos) == N_DAY
    assert all("x_prev" in info and "x" in info for info in infos)
    _assert_read_only_synthetic_2012(synthetic_2012, s["location"])


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"scenario": scenarios.PAPER, "sampler_kwargs": scenarios.TRAIN_SAMPLER_KWARGS}],
    ids=["neither", "both"],
)
def test_make_raw_env_needs_exactly_one_of_scenario_or_sampler_kwargs(kwargs):
    import greenleaf_control.envs as envs

    with pytest.raises(ValueError):
        envs.make_raw_env(season_length=1, normalize_actions=False, **kwargs)


# --- Wiring -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "normalized", "weight"),
    [
        ("ppo_paper", True, 0.5),
        ("ppo", True, 0.5),
        ("residual", False, 0.5),
        ("residual_lag", False, 0.0),  # residual_lag forces w = 0; the Lagrangian prices the costs
    ],
)
def test_make_env_wires_reward_weather_and_actions(method, normalized, weight):
    """Asked for penalty_weight 0.5: the reward is a ConstraintCostReward at 0.5 (0.0 for
    residual_lag), the weather comes from WEATHER_DIR, rate actions are normalized for ppo_paper and
    ppo, and the residual arms' raw env takes absolute controls (the RB works below any
    normalization)."""
    env = _make(method, penalty_weight=0.5)
    raw = env.unwrapped
    assert isinstance(raw.reward_fn, ConstraintCostReward)
    assert raw.reward_fn.penalty_weight == weight
    assert raw.weather_repository.weather_data_dir == scenarios.WEATHER_DIR
    assert raw.action_scheme.normalize_actions is normalized
    assert raw.N == N_DAY


@pytest.mark.parametrize("method", METHODS)
def test_make_env_samples_paper_for_ppo_paper_and_train_for_the_rest(method):
    sampler = _make(method).unwrapped.weather_scenario_sampler
    if method == "ppo_paper":
        assert isinstance(sampler, FixedWeatherSampler)
        assert vars(sampler) == scenarios.PAPER
    else:
        assert isinstance(sampler, RandomWeatherSampler)
        assert {k: list(v) for k, v in vars(sampler).items()} == scenarios.TRAIN_SAMPLER_KWARGS


@pytest.mark.parametrize("method", METHODS)
def test_a_given_scenario_overrides_the_method_default(method):
    """For evaluation, make_env(scenario=s) fixes the weather on s (VALIDATION's day 45) for every
    method, whatever the reset seed."""
    s = scenarios.VALIDATION[0]
    env = _make(method, scenario=s)
    sampler = env.unwrapped.weather_scenario_sampler
    assert isinstance(sampler, FixedWeatherSampler)
    assert vars(sampler) == s
    for seed in (0, 1):
        _, info = env.reset(seed=seed)
        assert info["scenario"] == s


def test_make_raw_env_fixed_scenario_wiring():
    """A fixed scenario: the fixed sampler on it, the asked weight 0.25, absolute controls in
    Box(0, 1, (6,)), and 1 day = 96 steps."""
    import greenleaf_control.envs as envs

    s = scenarios.VALIDATION[2]
    env = envs.make_raw_env(
        scenario=s, season_length=1, normalize_actions=False, penalty_weight=0.25
    )
    raw = env.unwrapped
    assert isinstance(raw.reward_fn, ConstraintCostReward)
    assert raw.reward_fn.penalty_weight == 0.25
    assert raw.weather_repository.weather_data_dir == scenarios.WEATHER_DIR
    assert isinstance(raw.weather_scenario_sampler, FixedWeatherSampler)
    assert vars(raw.weather_scenario_sampler) == s
    assert raw.action_scheme.normalize_actions is False
    assert env.action_space == gym.spaces.Box(0.0, 1.0, (6,), np.float32)
    assert raw.N == N_DAY
    _, info = env.reset(seed=0)
    assert info["scenario"] == s


def test_make_raw_env_sampler_kwargs_wiring():
    """sampler_kwargs: the random sampler on them, the default weight 1.0, normalized rate actions,
    and the default 60-day season = 5,760 steps (built only; nothing is stepped)."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(sampler_kwargs=scenarios.TRAIN_SAMPLER_KWARGS, normalize_actions=True)
    raw = env.unwrapped
    assert isinstance(raw.reward_fn, ConstraintCostReward)
    assert raw.reward_fn.penalty_weight == 1.0
    assert isinstance(raw.weather_scenario_sampler, RandomWeatherSampler)
    sampler = {k: list(v) for k, v in vars(raw.weather_scenario_sampler).items()}
    assert sampler == scenarios.TRAIN_SAMPLER_KWARGS
    assert raw.action_scheme.normalize_actions is True
    assert raw.N == 5_760  # 60 days x 96 steps


def test_make_raw_env_reads_weather_dir_at_call_time(tmp_path, monkeypatch):
    """With scenarios.WEATHER_DIR pointed at a tmp dir (holding a copy of PAPER's Amsterdam 2010
    file), the env's repository reads from there."""
    import greenleaf_control.envs as envs

    (tmp_path / "Amsterdam").mkdir()
    shutil.copyfile(
        scenarios.WEATHER_DIR / "Amsterdam" / "2010.csv", tmp_path / "Amsterdam" / "2010.csv"
    )
    monkeypatch.setattr(scenarios, "WEATHER_DIR", tmp_path)
    env = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    assert env.unwrapped.weather_repository.weather_data_dir == tmp_path


# --- Price regimes (issue #67, PROTOCOL v3) -----------------------------------------------------

# Upstream's registered reward kwargs (gl_gym/__init__.py:33-40): elec 0.3 EUR/kWh, heating
# 0.09 EUR/kWh, CO2 0.3 EUR/kg, fruit 1.6 EUR/kg, dmfm 0.065, pen_lamp 0.1. sim keeps them all.
REGISTERED_REWARD_KW = dict(gym.spec(ENV_ID).kwargs["reward_kwargs"])
# agc2020 overrides only upstream's four price kwargs, with AGC2's 2020 tariffs (Hemming et al.
# 2020, Sensors 20(22):6430, §2): heat 0.03 EUR/kWh, electricity 0.06 EUR/kWh, CO2 0.08 EUR/kg,
# fruit 1.6 EUR/kg unchanged. dmfm and pen_lamp stay upstream's.
REGIME_PRICES = {
    "sim": {},
    "agc2020": {"fruit_price": 1.6, "heating_price": 0.03, "elec_price": 0.06, "co2_price": 0.08},
}
ZERO_U = (0.0,) * 6
FULL_COST_U = (1.0, 1.0, 0.0, 0.0, 1.0, 0.0)  # boiler, CO2 dosing and lamps at full
# p[154] = 0.328 mg/m2/s max fruit growth x 900 s = 295.2 mg DM/m2: the max-profit step.
MAX_GROWTH = 295.2


def _reward_kwargs(env) -> dict:
    """The reward kwargs gym.make built the raw env's ConstraintCostReward with."""
    return dict(env.unwrapped.spec.kwargs["reward_kwargs"])


def _ctx(co2, temp, rh, *, p, u=ZERO_U, growth=0.0):
    """A RewardContext at indoor (co2 ppm, temp C, rh %), with x[25] - x_prev[25] = growth, built
    as the env's step() builds one (as tests/test_rewards.py does)."""
    x_prev = np.zeros(N_STATES)
    x = np.zeros(N_STATES)
    x[25] = growth  # x[25] is fruit dry matter [mg/m2]
    return RewardContext(
        t=0,
        dt=900,
        Np=0,
        x_prev=x_prev,
        x=x,
        u=np.asarray(u, dtype=np.float64),
        p=p,
        d=np.zeros((1, 10)),
        hour_of_day=12.0,
        day_of_year=59.0,
        obs={"IndoorClimateObservations": np.array([co2, temp, rh, 40.0], dtype=np.float32)},
        constraints_low=np.array([300.0, 15.0, 50.0]),
        constraints_high=np.array([1600.0, 34.0, 85.0]),
    )


@pytest.mark.parametrize("weight", [1.0, 0.25])
@pytest.mark.parametrize("regime", ["sim", "agc2020"])
def test_make_raw_env_overrides_only_the_four_price_kwargs(regime, weight):
    """make_raw_env(price_regime=r) builds ConstraintCostReward with upstream's registered reward
    kwargs, r's four prices over them (none for sim), and the asked penalty_weight: under agc2020,
    elec 0.06, heating 0.03, co2 0.08 and fruit 1.6, with dmfm 0.065 and pen_lamp 0.1 kept. The
    reward's own price models hold the same four prices."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(
        scenario=scenarios.PAPER,
        season_length=1,
        normalize_actions=False,
        penalty_weight=weight,
        price_regime=regime,
    )
    want = {**REGISTERED_REWARD_KW, **REGIME_PRICES[regime], "penalty_weight": weight}
    assert _reward_kwargs(env) == want
    reward = env.unwrapped.reward_fn
    assert isinstance(reward, ConstraintCostReward)
    assert reward.penalty_weight == weight
    assert {
        "fruit_price": reward.fruit_price_model.price,
        "heating_price": reward.heating_price_model.price,
        "elec_price": reward.elec_price_model.price,
        "co2_price": reward.co2_price_model.price,
    } == {k: want[k] for k in ("fruit_price", "heating_price", "elec_price", "co2_price")}


@pytest.mark.parametrize("method", METHODS)
def test_make_env_threads_the_price_regime(method):
    """make_env(method, price_regime="agc2020") builds each method's raw env at agc2020's prices,
    keeping its own penalty weight (residual_lag's forced 0.0, the asked 1.0 otherwise)."""
    env = _make(method, price_regime="agc2020")
    weight = 0.0 if method == "residual_lag" else 1.0
    assert _reward_kwargs(env) == {
        **REGISTERED_REWARD_KW,
        **REGIME_PRICES["agc2020"],
        "penalty_weight": weight,
    }


AGC2020_STEP = [
    # Under agc2020, per full step: heat 130 W/m2 x 0.25 h = 0.0325 kWh/m2 x 0.03 EUR/kWh = 0.000975;
    # elec 116 W/m2 x 0.25 h = 0.029 kWh/m2 x 0.06 EUR/kWh = 0.00174; CO2 5 mg/m2/s x 900 s =
    # 0.0045 kg/m2 x 0.08 EUR/kg = 0.00036 EUR/m2. min_profit = -(0.000975 + 0.00174 + 0.00036) =
    # -0.003075; max_profit = 295.2 mg/m2 x 1e-6 / 0.065 x 1.6 EUR/kg = 0.0072665 (unchanged);
    # span 0.0103415. No controls, no growth: 0.003075 / 0.0103415 = 0.2973467
    pytest.param((800.0, 22.0, 70.0), ZERO_U, 0.0, 0.2973467, 0.2973467, id="zero"),
    # boiler only: (-0.000975 + 0.003075) / 0.0103415 = 0.2030661
    pytest.param((800.0, 22.0, 70.0), (1, 0, 0, 0, 0, 0), 0.0, 0.2030661, 0.2030661, id="boiler"),
    # lamps only: (-0.00174 + 0.003075) / 0.0103415 = 0.1290920
    pytest.param((800.0, 22.0, 70.0), (0, 0, 0, 0, 1, 0), 0.0, 0.1290920, 0.1290920, id="lamps"),
    # CO2 only: (-0.00036 + 0.003075) / 0.0103415 = 0.2625354
    pytest.param((800.0, 22.0, 70.0), (0, 1, 0, 0, 0, 0), 0.0, 0.2625354, 0.2625354, id="co2"),
    # boiler, CO2 and lamps at full, no growth: profit = min_profit, so 0
    pytest.param((800.0, 22.0, 70.0), FULL_COST_U, 0.0, 0.0, 0.0, id="full_cost"),
    # no controls, max growth: profit = max_profit, so 1
    pytest.param((800.0, 22.0, 70.0), ZERO_U, MAX_GROWTH, 1.0, 1.0, id="max_growth"),
    # full costs and max growth: (0.0072665 - 0.003075 + 0.003075) / 0.0103415 = 0.7026532
    pytest.param((800.0, 22.0, 70.0), FULL_COST_U, MAX_GROWTH, 0.7026532, 0.7026532, id="full_max"),
    # 36 C, no controls: scaled profit 0.2973467; the paper reward subtracts (36 - 34) / 15 C:
    # 0.2973467 - 0.1333333 = 0.1640134
    pytest.param((800.0, 36.0, 70.0), ZERO_U, 0.0, 0.2973467, 0.1640134, id="hot"),
]


@pytest.mark.parametrize(("obs", "u", "growth", "scaled", "paper"), AGC2020_STEP)
def test_agc2020_reward_on_one_hand_built_step_matches_hand_values(obs, u, growth, scaled, paper):
    """Done when (#67): under agc2020, one hand-built step's upstream scaled profit equals the
    hand-computed value. The reward is the one make_raw_env(price_regime="agc2020") built, fed a
    RewardContext built by hand on the env's own params: its scaled_profit, and its paper_reward
    (upstream's reward), at agc2020's prices. rel 1e-6 because p[154] = 0.328 is float32."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(
        scenario=scenarios.PAPER, season_length=1, normalize_actions=False, price_regime="agc2020"
    )
    p = np.asarray(env.unwrapped.p, dtype=np.float64)
    _, info = env.unwrapped.reward_fn.compute_reward(_ctx(*obs, p=p, u=u, growth=growth))
    assert info["scaled_profit"] == pytest.approx(scaled, rel=1e-6, abs=1e-12)
    assert info["paper_reward"] == pytest.approx(paper, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # 0.0325 kWh/m2 x 0.03 EUR/kWh
        ("heat_cost", 0.000975),
        # 0.029 kWh/m2 x 0.06 EUR/kWh
        ("elec_cost", 0.00174),
        # 0.0045 kg/m2 x 0.08 EUR/kg
        ("co2_cost", 0.00036),
        # 295.2 mg DM/m2 x 1e-6 / 0.065 = 0.0045415 kg FFW/m2 x 1.6 EUR/kg (fruit price unchanged)
        ("revenue", 0.0072664615),
        # 0.0072664615 revenue - (0.000975 + 0.00174 + 0.00036) costs = 0.0041914615
        ("profit", 0.0041914615),
    ],
)
def test_agc2020_money_keys_on_one_hand_built_step_match_hand_values(key, expected):
    """The full-cost, max-growth step's money keys at agc2020's prices, from the reward
    make_raw_env(price_regime="agc2020") built."""
    import greenleaf_control.envs as envs

    env = envs.make_raw_env(
        scenario=scenarios.PAPER, season_length=1, normalize_actions=False, price_regime="agc2020"
    )
    p = np.asarray(env.unwrapped.p, dtype=np.float64)
    ctx = _ctx(800.0, 22.0, 70.0, p=p, u=FULL_COST_U, growth=MAX_GROWTH)
    _, info = env.unwrapped.reward_fn.compute_reward(ctx)
    assert info[key] == pytest.approx(expected, rel=1e-6)


def test_price_regime_sim_is_todays_env_bit_for_bit():
    """With sim, everything scores as today: make_raw_env(price_regime="sim") has exactly the
    default env's reward kwargs, and over a 1-day PAPER season driven by one seeded action sequence
    (absolute controls, default_rng(7)) every step's observation, reward, flags and info are the
    default env's, compared exactly (arrays with array_equal)."""
    import greenleaf_control.envs as envs

    kw = {"scenario": scenarios.PAPER, "season_length": 1, "normalize_actions": False}
    today = envs.make_raw_env(**kw)
    sim = envs.make_raw_env(**kw, price_regime="sim")
    assert _reward_kwargs(sim) == _reward_kwargs(today)
    obs, _ = today.reset(seed=0)
    sim_obs, _ = sim.reset(seed=0)
    np.testing.assert_array_equal(sim_obs, obs)
    for k, a in enumerate(np.random.default_rng(7).uniform(0, 1, (N_DAY, 6)).astype(np.float32)):
        obs, reward, terminated, truncated, info = today.step(a)
        sim_obs, sim_reward, sim_terminated, sim_truncated, sim_info = sim.step(a)
        np.testing.assert_array_equal(sim_obs, obs)
        assert (sim_reward, sim_terminated, sim_truncated) == (reward, terminated, truncated), k
        assert sim_info.keys() == info.keys(), k
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(sim_info[key], value, err_msg=f"{key} at step {k}")
            else:
                assert sim_info[key] == value, (key, k)
    assert terminated


@pytest.mark.parametrize("build", ["make_raw_env", "make_env", "run_baseline"])
def test_an_unknown_price_regime_raises_valueerror(build):
    """An unknown regime name is refused, naming it, by each env builder: a typo can't silently
    fall back to sim's prices."""
    import greenleaf_control.envs as envs

    calls = {
        "make_raw_env": lambda: envs.make_raw_env(
            scenario=scenarios.PAPER,
            season_length=1,
            normalize_actions=False,
            price_regime="agc2021",
        ),
        "make_env": lambda: envs.make_env("ppo", seed=0, season_length=1, price_regime="agc2021"),
        "run_baseline": lambda: envs.run_baseline(
            "zero_controls", scenarios.PAPER, 1, price_regime="agc2021"
        ),
    }
    with pytest.raises(ValueError, match="agc2021"):
        calls[build]()


def test_the_reward_and_the_grader_price_each_regime_the_same():
    """The reward's prices (envs, in upstream's kwarg names) and the grader's (metrics, which never
    imports envs) are kept apart by design, so this cross-check holds them together: envs.REWARD_PRICES
    and metrics.PRICE_REGIMES name exactly sim and agc2020, and for each the env make_raw_env builds
    prices fruit, dmfm, heat, electricity and CO2 exactly as the grader does."""
    import greenleaf_control.envs as envs
    from greenleaf_control import metrics

    assert set(envs.REWARD_PRICES) == set(metrics.PRICE_REGIMES) == {"sim", "agc2020"}
    for regime, prices in metrics.PRICE_REGIMES.items():
        kw = _reward_kwargs(
            envs.make_raw_env(
                scenario=scenarios.PAPER,
                season_length=1,
                normalize_actions=False,
                price_regime=regime,
            )
        )
        assert {
            "fruit": kw["fruit_price"],
            "dmfm": kw["dmfm"],
            "heat": kw["heating_price"],
            "elec": kw["elec_price"],
            "co2": kw["co2_price"],
        } == prices, regime


# --- Determinism --------------------------------------------------------------------------------


def _trajectory(method: str, seed: int) -> tuple[str, tuple]:
    """sha256 of a 1-day episode of make_env(method, seed=seed), reset once more and driven by one
    fixed action sequence (default_rng(2026)): the scenario, every observation, reward, control and
    flag. Returns the hex digest and the scenario."""
    env = _make(method, seed=seed)
    obs, info = env.reset()
    s = info["scenario"]
    scenario = (str(s["location"]), int(s["growth_year"]), int(s["start_day"]))
    digest = hashlib.sha256(json.dumps(scenario).encode())
    digest.update(obs.tobytes())
    for a in np.random.default_rng(2026).uniform(-1, 1, (N_DAY, 6)).astype(np.float32):
        obs, reward, terminated, truncated, info = env.step(a)
        digest.update(obs.tobytes())
        digest.update(np.float64(reward).tobytes())
        digest.update(np.ascontiguousarray(info["controls"]).tobytes())
        digest.update(np.array([terminated, truncated], dtype=bool).tobytes())
    return digest.hexdigest(), scenario


@pytest.mark.parametrize("method", LEARNING)
def test_same_seed_same_trajectory_and_a_different_seed_differs(method):
    """make_env(seed=s) resets the env once with seed s, so its next reset and the whole day are a
    function of s: seed 0 twice gives the same hash.

    Known answer, from upstream's random sampler alone (gym.make on TRAIN_SAMPLER_KWARGS, then
    reset(seed=s), then reset()): the second reset draws Amsterdam 2009 day 53 for seed 0 and
    Amsterdam 2011 day 74 for seed 1. Different weather, so the two hashes differ.
    """
    first, s0 = _trajectory(method, 0)
    again, s0_again = _trajectory(method, 0)
    other, s1 = _trajectory(method, 1)
    assert (again, s0_again) == (first, s0)
    assert s0 == ("Amsterdam", 2009, 53)
    assert s1 == ("Amsterdam", 2011, 74)
    assert s0 in TRAIN_SUPPORT
    assert s1 in TRAIN_SUPPORT
    assert other != first


@pytest.mark.parametrize("method", METHODS)
def test_make_env_seeds_the_action_space(method):
    """make_env seeds the action space too: the same seed samples the same action, another seed
    doesn't."""
    a = _make(method, seed=3).action_space.sample()
    b = _make(method, seed=3).action_space.sample()
    c = _make(method, seed=4).action_space.sample()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --- run_baseline -------------------------------------------------------------------------------


def _expected_baseline_controls(name: str) -> np.ndarray:
    """Each baseline's 96 controls on PAPER, worked out without envs.py: zeros; the start-day-seeded
    draws (default_rng(59)); or the default rules on a plain upstream env with absolute controls."""
    if name == "zero_controls":
        return np.zeros((N_DAY, 6), dtype=np.float32)
    if name == "random":
        return _random_controls(scenarios.PAPER["start_day"])
    up = gym.make(ENV_ID, normalize_actions=False, **UP_KW)
    up.reset(seed=0, options={"scenario": scenarios.PAPER})
    controller = rule_based.make_controller(RB_DEFAULT)
    out = []
    for _ in range(N_DAY):
        u = rule_based.rb_action(up, controller)
        up.step(u)
        out.append(u)
    up.close()
    return np.array(out)


@pytest.mark.parametrize("name", ["zero_controls", "random", "rb_default"])
def test_run_baseline_one_paper_day(name, monkeypatch):
    """One PAPER day of each baseline is exactly 96 infos: the episode ends terminated on step 96,
    never truncated, no solver failure, every number finite, and the applied controls are the
    baseline's own (zeros; default_rng(59) draws; the default rules' actions)."""
    import greenleaf_control.envs as envs

    flags = _record_flags(monkeypatch, envs)
    infos = envs.run_baseline(name, scenarios.PAPER, season_length=1)
    assert len(infos) == N_DAY
    assert flags == [(False, False)] * (N_DAY - 1) + [(True, False)]
    assert [info["solver_failure"] for info in infos] == [False] * N_DAY
    assert [k for k, info in enumerate(infos) if not _finite(info)] == []
    np.testing.assert_array_equal(_controls(infos), _expected_baseline_controls(name))


def test_run_baseline_random_is_seeded_by_the_start_day():
    """Two random PAPER days are identical; VALIDATION's day-45 scenario draws from
    default_rng(45) instead, so its controls differ from PAPER's (default_rng(59))."""
    import greenleaf_control.envs as envs

    first = envs.run_baseline("random", scenarios.PAPER, season_length=1)
    again = envs.run_baseline("random", scenarios.PAPER, season_length=1)
    day45 = envs.run_baseline("random", scenarios.VALIDATION[0], season_length=1)
    np.testing.assert_array_equal(_controls(again), _controls(first))
    assert [i["paper_reward"] for i in again] == [i["paper_reward"] for i in first]
    np.testing.assert_array_equal(_controls(day45), _random_controls(45))
    assert not np.array_equal(_controls(day45), _controls(first))


@pytest.mark.slow
def test_full_paper_season_of_rb_default(monkeypatch):
    """A full 60-day rb_default season on PAPER: 5,760 steps (60 x 96), terminated on the last,
    never truncated, no solver failure, every number finite (about 3 s)."""
    import greenleaf_control.envs as envs

    flags = _record_flags(monkeypatch, envs)
    infos = envs.run_baseline("rb_default", scenarios.PAPER)
    assert len(infos) == 5_760
    assert flags == [(False, False)] * 5_759 + [(True, False)]
    assert not any(info["solver_failure"] for info in infos)
    assert [k for k, info in enumerate(infos) if not _finite(info)] == []

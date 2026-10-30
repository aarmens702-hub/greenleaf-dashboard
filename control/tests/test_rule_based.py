"""Tests for greenleaf_control.rule_based and the two configs copied from upstream (issue #8)."""

import hashlib
import inspect
import warnings
from pathlib import Path

import gl_gym  # noqa: F401  (importing gl_gym registers its env ids with gymnasium)
import gymnasium as gym
import numpy as np
import pytest
import yaml
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# PAPER per PROTOCOL §2; switch to scenarios.PAPER once #6 ships.
PAPER = {"location": "Amsterdam", "growth_year": 2010, "start_day": 59}

# Upstream configs/agents/rule_based.yml at GreenLight-Gym2@2d3febb, as yaml.safe_load parses it (29 keys).
RB_DEFAULT = {
    "lamps_on": 0,
    "lamps_off": 18,
    "lamps_day_start": -1,
    "lamps_day_stop": 366,
    "lamps_off_sun": 400,
    "lamp_rad_sum_limit": 10,
    "temp_setpoint_day": 19.5,
    "temp_setpoint_night": 16.5,
    "heat_correction": 0,
    "heat_deadzone": 5,
    "co2_day": 800,
    "vent_heat_Pband": 4,
    "rh_max": 85,
    "mech_dehumid_Pband": 2,
    "vent_rh_Pband": 5,
    "t_vent_off": 1,
    "vent_cold_Pband": -1,
    "thScrSpDay": 5,
    "thScrSpNight": 10,
    "thScrPband": -1,
    "thScrDeadZone": 4,
    "thScrRh": -2,
    "thScrRhPband": 2,
    "lampExtraHeat": 2,
    "blScrExtraRh": 100,
    "rhMax": 85,
    "tHeatBand": -1,
    "co2Band": -100,
    "useBlScr": 1,
}

# The PPO values issue #8's Done-when lists (upstream configs/agents/ppo.yml at 2d3febb).
PPO_EXPECTED = {
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 8,
    "learning_rate": 2e-5,
    "gamma": 0.9631,
    "gae_lambda": 0.9167,
    "ent_coef": 0.05434,
    "vf_coef": 0.8225,
    "max_grad_norm": 0.3,
}

# sha256 of upstream's exact bytes for configs/agents/<f>.yml at GreenLight-Gym2@2d3febb
# (2d3febb1ea002b24b452e32293e990beb78d3ce1), measured 2026-10-28 from the orchestrator's fetched
# copies (rule_based.yml: 559 bytes, ppo.yml: 874 bytes; LF endings, trailing newline).
# Our copy = a 3-line attribution comment header + these bytes, unmodified.
UPSTREAM_COPIES = [
    (
        "rule_based_default.yml",
        "rule_based.yml",
        "29d7c36080bce9c41cbddb9420b1024d8d65beb5f97230a3d178be9c80b9f8fc",
    ),
    (
        "ppo.yml",
        "ppo.yml",
        "4eed0cf650562a91b74c9098df7620af2af987f94a3be33686d543f418c56bd0",
    ),
]


def _paper_env() -> gym.Env:
    """The upstream tomato env, as gym.make wraps it, reset on PAPER for a one-day season.

    normalize_actions=False makes the action the absolute control in [0, 1], which is what the
    rule-based controller outputs.
    """
    env = gym.make("gl_gym/GreenLightTomato-v0", season_length=1, normalize_actions=False)
    _, info = env.reset(seed=0, options={"scenario": PAPER})
    assert info["scenario"] == PAPER
    assert env.unwrapped.N == 96  # 1 day = 86,400 s / 900 s per step = 96 steps
    return env


class _RecordingController:
    """Stands in for the controller (not the thing under test): records the context it gets and
    returns a fixed, deliberately out-of-range float64 action."""

    def __init__(self, out: np.ndarray):
        self.out = out
        self.ctx = None

    def predict(self, ctx):
        self.ctx = ctx
        return self.out.copy()


def test_load_rb_params_default_is_upstream_flat_dict():
    """load_rb_params("default") returns upstream's rule_based.yml unwrapped from its top-level
    `GreenLightEnv:` key: exactly the 29 parameters RuleBasedController.__init__ takes, at
    upstream's values."""
    import greenleaf_control.rule_based as rule_based

    params = rule_based.load_rb_params("default")

    assert isinstance(params, dict)
    assert set(params) == set(inspect.signature(RuleBasedController).parameters)
    assert len(params) == 29
    assert params == RB_DEFAULT


def test_make_controller_round_trips_params():
    """make_controller(params) is RuleBasedController(**params): every key lands on the attribute
    of the same name, including a value changed from the default."""
    import greenleaf_control.rule_based as rule_based

    params = rule_based.load_rb_params("default")
    controller = rule_based.make_controller(params)
    assert isinstance(controller, RuleBasedController)
    for key, value in params.items():
        assert getattr(controller, key) == value, key

    tweaked = {**params, "temp_setpoint_day": 21.0, "co2_day": 900}
    controller = rule_based.make_controller(tweaked)
    assert controller.temp_setpoint_day == 21.0
    assert controller.co2_day == 900


@pytest.mark.parametrize("name", ["best", "", "DEFAULT"])
def test_load_rb_params_rejects_unknown_name(name):
    """Only "default" and "tuned" are valid names; anything else raises ValueError. "DEFAULT" is
    in the table because a case-insensitive filesystem (macOS) would find rule_based_DEFAULT.yml."""
    import greenleaf_control.rule_based as rule_based

    with pytest.raises(ValueError):
        rule_based.load_rb_params(name)


def test_rb_action_on_paper_env_is_float32_unit_box_and_steps_a_day():
    """rb_action on a fresh PAPER env, passed as gym.make wraps it, returns a finite float32 (6,)
    action in [0, 1]; it reads raw_env.unwrapped (gymnasium 0.29 still forwards env.<attr>
    through wrappers, with a deprecation warning made an error here), and two calls with no step
    between them agree. Then the controller drives the whole 96-step day: every action stays in
    [0, 1], no step truncates (a solver failure), and the actions change over the day, so each
    one is computed from that step's state."""
    import greenleaf_control.rule_based as rule_based

    controller = rule_based.make_controller(rule_based.load_rb_params("default"))
    env = _paper_env()

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*from other wrappers is deprecated")
        first = rule_based.rb_action(env, controller)
        again = rule_based.rb_action(env, controller)

    assert isinstance(first, np.ndarray)
    assert first.dtype == np.float32
    assert first.shape == (6,)
    assert np.all(np.isfinite(first))
    assert np.all((first >= 0.0) & (first <= 1.0))
    np.testing.assert_array_equal(first, again)
    assert env.unwrapped.timestep == 0  # rb_action reads the env; it doesn't step it

    actions = []
    for _ in range(env.unwrapped.N):
        action = rule_based.rb_action(env, controller)
        assert action.dtype == np.float32 and action.shape == (6,)
        assert np.all((action >= 0.0) & (action <= 1.0)), action
        actions.append(action)
        _, _, _, truncated, _ = env.step(action)
        assert not truncated, f"solver failure at step {env.unwrapped.timestep}"
    env.close()

    assert np.unique(np.array(actions), axis=0).shape[0] > 1


def test_rb_action_builds_context_from_current_step_and_clips():
    """rb_action hands the controller a StepContext of the env's current step, read from
    raw_env.unwrapped, and returns its output clipped to [0, 1] as float32. Ten steps in, the
    context must carry t = 10 and the current x (not x_prev). The recorded output
    [-0.5, 1.5, 0.25, 0.0, 1.0, 7.0] clips to [0, 1, 0.25, 0, 1, 1]."""
    import greenleaf_control.rule_based as rule_based

    env = _paper_env()
    for _ in range(10):
        env.step(np.full(6, 0.5, dtype=np.float32))
    e = env.unwrapped
    assert e.timestep == 10
    assert not np.array_equal(e.x, e.x_prev)  # so passing x_prev for x would be caught

    recorder = _RecordingController(np.array([-0.5, 1.5, 0.25, 0.0, 1.0, 7.0]))
    action = rule_based.rb_action(env, recorder)
    env.close()

    ctx = recorder.ctx
    assert isinstance(ctx, StepContext)
    assert ctx.t == e.timestep == 10
    assert ctx.dt == e.dt
    assert ctx.Np == e.Np
    np.testing.assert_array_equal(ctx.x, e.x)
    np.testing.assert_array_equal(ctx.x_prev, e.x_prev)
    np.testing.assert_array_equal(ctx.u, e.u)
    np.testing.assert_array_equal(ctx.p, e.p)
    np.testing.assert_array_equal(ctx.d, e.weather_data)
    assert ctx.hour_of_day == e.hour_of_day  # 10 steps x 900 s = 2.5 h
    assert ctx.day_of_year == e.day_of_year

    assert action.dtype == np.float32
    assert action.shape == (6,)
    np.testing.assert_array_equal(
        action, np.array([0.0, 1.0, 0.25, 0.0, 1.0, 1.0], dtype=np.float32)
    )


def test_ppo_config_has_the_protocol_values():
    """configs/ppo.yml, under upstream's top-level `GreenLightEnv:` key, has the PPO
    hyperparameters issue #8 lists, the learning rate parsed as a float (upstream writes `2.e-5`),
    and the policy net pi 3x256 / value net 3x512."""
    ppo = yaml.safe_load((CONFIGS / "ppo.yml").read_text())["GreenLightEnv"]

    assert {key: ppo.get(key) for key in PPO_EXPECTED} == PPO_EXPECTED
    assert isinstance(ppo["learning_rate"], float)
    assert ppo["policy_kwargs"]["net_arch"] == {"pi": [256, 256, 256], "vf": [512, 512, 512]}


@pytest.mark.parametrize(
    ("ours", "upstream", "sha256"), UPSTREAM_COPIES, ids=[c[0] for c in UPSTREAM_COPIES]
)
def test_config_is_attributed_unmodified_upstream_copy(ours, upstream, sha256):
    """configs/<ours> opens with a 3-line comment header naming the source (GreenLight-Gym2 URL,
    configs/agents/<upstream>, commit 2d3febb) and the AGPL license, and every byte after that
    header is upstream's file, unmodified (sha256 pinned above)."""
    path = CONFIGS / ours
    assert path.is_file(), f"missing {path}"
    parts = path.read_bytes().split(b"\n", 3)
    assert len(parts) == 4, f"{ours} is shorter than its 3-line header"

    header, body = parts[:3], parts[3]
    assert all(line.startswith(b"#") for line in header), header
    joined = b"\n".join(header)
    assert b"https://github.com/BartvLaatum/GreenLight-Gym2" in joined
    assert f"configs/agents/{upstream}".encode() in joined
    assert b"2d3febb" in joined
    assert b"AGPL" in joined
    assert hashlib.sha256(body).hexdigest() == sha256

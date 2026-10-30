"""The grower's rules: upstream's RuleBasedController, its parameters and one action per step.

rb_action returns ABSOLUTE controls in [0, 1], so the rule-based baseline needs an env built with
normalize_actions=False. configs/rule_based_default.yml and configs/ppo.yml are unmodified
AGPL-3.0 copies of upstream GreenLight-Gym2's configs/agents/ files (see each file's header).
"""

from pathlib import Path

import numpy as np
import yaml
from gl_gym.components.rule_based import RuleBasedController
from gl_gym.core.types import StepContext

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NAMES = ("default", "tuned")


def load_rb_params(name: str) -> dict:
    """The controller's parameters from configs/rule_based_<name>.yml; name is "default" or "tuned"."""
    if name not in NAMES:
        raise ValueError(f"unknown rule-based params {name!r}; expected one of {NAMES}")
    return yaml.safe_load((CONFIGS / f"rule_based_{name}.yml").read_text())["GreenLightEnv"]


def make_controller(params: dict) -> RuleBasedController:
    """Upstream's RuleBasedController with these parameters."""
    return RuleBasedController(**params)


def rb_action(raw_env, controller) -> np.ndarray:
    """The controller's action for the env's current step: float32, shape (6,), clipped to [0, 1]."""
    e = raw_env.unwrapped
    ctx = StepContext(
        t=e.timestep,
        dt=e.dt,
        Np=e.Np,
        x_prev=e.x_prev,
        x=e.x,
        u=e.u,
        p=e.p,
        d=e.weather_data,
        hour_of_day=e.hour_of_day,
        day_of_year=e.day_of_year,
    )
    return np.clip(controller.predict(ctx), 0.0, 1.0).astype(np.float32)

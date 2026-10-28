"""Every method's env, built one way (docs/API-CONTRACT.md). This is a scorer file.

make_raw_env is upstream's tomato env (gl_gym @2d3febb) with our ConstraintCostReward and
scenarios.WEATHER_DIR, a flat Box observation, and season progress when asked. make_env adds the
co-pilot (ResidualRBWrapper) for the residual arms and always puts SolverFailureFlag on top.

- Decision B (Aarmen, 2026-10-28): a season is exactly N steps (96 a day, 5,760 for 60 days).
  Upstream ends `terminated` on step N + 1, so _SeasonEnd ends it on step N. Nothing truncates for
  time, so a truncation is always one of upstream's swallowed solver failures.
- The rule-based controller works on the raw env (absolute controls, raw state), below anything
  that normalizes observations or actions.
- A season that would read TEST (2012) weather is refused unless make_raw_env gets allow_test=True
  (PROTOCOL §8): its own year, or the next year's file, which upstream appends when the season
  plus its 0.5-day lookahead runs past the end of the year.
- Every step's info carries x_prev and x, the raw state before and after the step (for metrics).

metrics.py grades runs from raw states and controls and must never import this module (it
imports rewards).
"""

from itertools import product

import gl_gym  # noqa: F401  (importing gl_gym registers ENV_ID with gymnasium)
import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from greenleaf_control import rule_based, scenarios
from greenleaf_control.rewards import ConstraintCostReward

ENV_ID = "gl_gym/GreenLightTomato-v0"
METHODS = ("ppo_paper", "ppo", "residual", "residual_lag")
BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned")
_REFUSED = "TEST weather (sealed, PROTOCOL §8); only make_raw_env(allow_test=True) may open it"


def _refuse_test(s: dict, season_length: int) -> None:
    """Raises if a season on s would read a TEST year's weather file.

    Upstream int()-casts the year (so "2012" reads 2012.csv), and appends <year + 1>.csv when start
    day + season + the 0.5-day lookahead runs past the year's file. The + 1 day covers the lookahead
    and rounding, so the check is conservative.
    """
    year = int(s["growth_year"])
    if scenarios.is_test({**s, "growth_year": year}):
        raise ValueError(f"scenario {s}: its own year, {year}, is {_REFUSED}")
    if int(s["start_day"]) + season_length + 1 >= 365 and scenarios.is_test(
        {**s, "growth_year": year + 1}
    ):
        raise ValueError(
            f"scenario {s} over {season_length} days runs past its year, so upstream would read the"
            f" next year's file, {year + 1}.csv, which is {_REFUSED}"
        )


class _RawStates(gym.Wrapper, RecordConstructorArgs):
    """Puts float64 copies of the raw state before and after each step in info["x_prev"], info["x"].

    Upstream's step sets x_prev = copy(x) before integrating x, so after it they are the two ends.
    """

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped
        info["x_prev"] = np.array(raw.x_prev, dtype=np.float64, copy=True)
        info["x"] = np.array(raw.x, dtype=np.float64, copy=True)
        return obs, reward, terminated, truncated, info


class _SeasonEnd(gym.Wrapper, RecordConstructorArgs):
    """Ends the season `terminated` after exactly N steps (decision B); upstream runs N + 1."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw = self.env.unwrapped
        return obs, reward, bool(terminated or raw.timestep >= raw.N), truncated, info


class _RefuseTestReset(gym.Wrapper, RecordConstructorArgs):
    """Refuses reset(options={"scenario": s}) onto a season that would read TEST weather, which
    bypasses upstream's sampler (and so the check at construction)."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def reset(self, *, seed=None, options=None):
        if options is not None and "scenario" in options:
            _refuse_test(options["scenario"], self.env.unwrapped.season_length)
        return self.env.reset(seed=seed, options=options)


class SeasonProgress(gym.ObservationWrapper, RecordConstructorArgs):
    """Appends season progress, timestep / N in [0, 1], to a flat Box observation."""

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            np.append(space.low, 0.0).astype(np.float32),
            np.append(space.high, 1.0).astype(np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        raw = self.env.unwrapped
        return np.append(observation, np.float32(raw.timestep / raw.N)).astype(np.float32)


class ResidualRBWrapper(gym.Wrapper, RecordConstructorArgs):
    """The co-pilot: applies clip(u_rb + scale * a, 0, 1) and observes the next RB action.

    Wraps a raw env with absolute controls (normalize_actions=False), which upstream doesn't clip,
    so the clip here is the only guard. The action a is not pre-clipped.
    """

    def __init__(self, env: gym.Env, controller, scale: float):
        RecordConstructorArgs.__init__(self, controller=controller, scale=scale)
        gym.Wrapper.__init__(self, env)
        self.controller = controller
        self.scale = scale
        self.u_rb = None
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        space = env.observation_space
        self.observation_space = gym.spaces.Box(
            np.concatenate([space.low, np.zeros(6)]).astype(np.float32),
            np.concatenate([space.high, np.ones(6)]).astype(np.float32),
            dtype=np.float32,
        )

    def _observe(self, obs):
        """Computes the RB action for the state the agent now sees and appends it."""
        self.u_rb = rule_based.rb_action(self.env, self.controller)
        return np.concatenate([obs, self.u_rb]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._observe(obs), info

    def step(self, action):
        u_rb = self.u_rb
        u_residual = self.scale * np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(
            np.clip(u_rb + u_residual, 0.0, 1.0)
        )
        info["u_rb"] = u_rb
        info["u_residual"] = u_residual
        return self._observe(obs), reward, terminated, truncated, info


class SolverFailureFlag(gym.Wrapper, RecordConstructorArgs):
    """Sets info["solver_failure"] = bool(truncated).

    Nothing truncates for time (decision B), so upstream truncates only when its ODE solver fails,
    an error it swallows and prints. That includes a failure on the season's last step.
    """

    def __init__(self, env: gym.Env):
        RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["solver_failure"] = bool(truncated)
        return obs, reward, terminated, truncated, info


def make_raw_env(
    *,
    scenario: dict | None = None,
    sampler_kwargs: dict | None = None,
    season_length: int = 60,
    normalize_actions: bool,
    penalty_weight: float = 1.0,
    season_progress: bool = False,
    allow_test: bool = False,
) -> gym.Env:
    """Upstream's env on a fixed scenario or a random sampler, with ConstraintCostReward.

    Ends the season after exactly N steps (decision B) and returns a flat Box observation
    (FlattenObservation, then SeasonProgress if asked). Refuses TEST weather unless allow_test.
    """
    if (scenario is None) == (sampler_kwargs is None):
        raise ValueError("make_raw_env needs exactly one of scenario or sampler_kwargs")
    if scenario is not None:
        sampler, sampler_args, support = "fixed", scenario, [scenario]
    else:
        sampler, sampler_args = "random", sampler_kwargs
        support = [
            {"location": loc, "growth_year": year, "start_day": day}
            for loc, year, day in product(
                sampler_kwargs["locations"],
                sampler_kwargs["growth_years"],
                sampler_kwargs["start_days"],
            )
        ]
    if not allow_test:
        for s in support:
            _refuse_test(s, season_length)
    env = gym.make(
        ENV_ID,
        season_length=season_length,
        normalize_actions=normalize_actions,
        weather_data_dir=str(scenarios.WEATHER_DIR),
        reward_fn=ConstraintCostReward,
        # GreenhouseReward has no price defaults, so keep the registered ones.
        reward_kwargs={
            **gym.spec(ENV_ID).kwargs["reward_kwargs"],
            "penalty_weight": penalty_weight,
        },
        weather_scenario_sampler=sampler,
        weather_scenario_sampler_kwargs=sampler_args,
    )
    env = _SeasonEnd(_RawStates(env))
    if not allow_test:
        env = _RefuseTestReset(env)
    env = gym.wrappers.FlattenObservation(env)
    if season_progress:
        env = SeasonProgress(env)
    return env


def make_env(
    method: str,
    *,
    seed: int,
    scenario: dict | None = None,
    season_length: int = 60,
    penalty_weight: float = 1.0,
    rb_params: dict | None = None,
    residual_scale: float = 0.15,
) -> gym.Env:
    """The env a METHODS entry trains and is evaluated on, reset once with seed (and its action
    space seeded). A given scenario overrides the method's default weather (PAPER or TRAIN)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    residual = method in ("residual", "residual_lag")
    if scenario is not None:
        weather = {"scenario": scenario}
    elif method == "ppo_paper":
        weather = {"scenario": scenarios.PAPER}
    else:
        weather = {"sampler_kwargs": scenarios.TRAIN_SAMPLER_KWARGS}
    env = make_raw_env(
        **weather,
        season_length=season_length,
        normalize_actions=not residual,
        # residual_lag's costs are priced by the Lagrangian, not the reward.
        penalty_weight=0.0 if method == "residual_lag" else penalty_weight,
        season_progress=method != "ppo_paper",
    )
    if residual:
        params = rule_based.load_rb_params("tuned") if rb_params is None else rb_params
        env = ResidualRBWrapper(env, rule_based.make_controller(params), residual_scale)
    env = SolverFailureFlag(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def run_baseline(name: str, scenario: dict, season_length: int = 60) -> list[dict]:
    """A full episode's per-step info dicts for a BASELINES entry on a fixed scenario.

    random draws from one generator per episode, seeded by the scenario's start day.
    """
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINES}")
    env = SolverFailureFlag(
        make_raw_env(scenario=scenario, season_length=season_length, normalize_actions=False)
    )
    rng = np.random.default_rng(scenario["start_day"])
    if name in ("rb_default", "rb_tuned"):
        controller = rule_based.make_controller(rule_based.load_rb_params(name.removeprefix("rb_")))
    env.reset()
    infos = []
    terminated = truncated = False
    while not (terminated or truncated):
        if name == "zero_controls":
            action = np.zeros(6, dtype=np.float32)
        elif name == "random":
            action = rng.uniform(0, 1, 6).astype(np.float32)
        else:
            action = rule_based.rb_action(env, controller)
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
    env.close()
    return infos

"""Tests for greenleaf_control.lagrangian (issue #12): the PID-Lagrangian.

One PIDLagrangian per constraint (rewards.CONSTRAINTS order: co2, temp, rh) turns a rollout's mean
cost into a multiplier. lam, I and e_prev all start at 0.0, and each update does
    e = (cost - budget) / max(budget, 1e-8)
    I = clip(I + ki*e, 0, lambda_max)   (anti-windup above too: Aarmen's decision, 2026-10-28)
    d = max(0, e - e_prev)              (the derivative acts only on increases of the error)
    lam = clip(I + kp*e + kd*d, 0, lambda_max), then e_prev = e.
VecLagrangianReward turns each reward into (r - sum(lam_i * c_i)) / (1 + sum(lam_i)) and keeps the
per-step mean cost since the last pop. LagrangianCallback updates the PIDs from those means at each
rollout end, sets the wrapper's lambdas, records lag/* and appends a row to lambdas.csv.
Issue #38: a FixedMultiplier(value=1.0) stands in for a PID on a zero-budget constraint; its lam
and every update() are its value. At training start the callback sets the wrapper's lambdas to
the multipliers' lam, so a fixed one prices its constraint from the first step.

The env under the wrapper is a scripted toy (no gl_gym): every step reports fixed costs under
cost_co2, cost_temp and cost_rh, so every mean, lambda and reward below is computed by hand.
"""

import csv
import inspect
import itertools

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecMonitor, VecNormalize

# greenleaf_control.lagrangian is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

COST_KEYS = ("cost_co2", "cost_temp", "cost_rh")
HEADER = ["timesteps", "lambda_co2", "lambda_temp", "lambda_rh", "cost_co2", "cost_temp", "cost_rh"]
ACTIONS = np.zeros((2, 1), dtype=np.float32)  # one action per toy; the toys ignore it

# Two 2-step episodes, (cost_co2, cost_temp, cost_rh) per step. Over one step of each episode in
# both toys (4 env-steps) the means are co2 (1.0 + 0.5 + 0.5 + 0) / 4 = 0.5,
# temp (0 + 0 + 0.5 + 0) / 4 = 0.125 and rh (0.25 + 0 + 0 + 0.75) / 4 = 0.25.
MIX_A = [(1.0, 0.0, 0.25), (0.5, 0.0, 0.0)]
MIX_B = [(0.5, 0.5, 0.0), (0.0, 0.0, 0.75)]


class Toy(gym.Env):
    """Step k reports costs[k] and a fixed reward; the episode terminates after the last step."""

    def __init__(self, costs, reward=1.0, drop=None):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.costs = costs
        self.reward = reward
        self.drop = drop  # a cost key to leave out of info
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        info = dict(zip(COST_KEYS, self.costs[self.t], strict=True))
        info.pop(self.drop, None)
        self.t += 1
        return np.zeros(1, np.float32), self.reward, self.t == len(self.costs), False, info


def _venv(costs_a, costs_b, **toy_kwargs):
    return DummyVecEnv([lambda: Toy(costs_a, **toy_kwargs), lambda: Toy(costs_b, **toy_kwargs)])


def _ppo(env):
    """A PPO on env that never trains here, with a silent logger (SB3's default makes a tmp dir)."""
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=16, n_epochs=1, seed=0, device="cpu")
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def _lag_logged(model):
    return {k: v for k, v in model.logger.name_to_value.items() if k.startswith("lag/")}


# PIDLagrangian


def test_pid_signature_defaults_and_initial_state():
    import greenleaf_control.lagrangian as lagrangian

    params = inspect.signature(lagrangian.PIDLagrangian.__init__).parameters
    assert {name: p.default for name, p in params.items() if name != "self"} == {
        "budget": inspect.Parameter.empty,
        "kp": 0.5,
        "ki": 0.05,
        "kd": 0.1,
        "lambda_max": 100.0,
    }
    pid = lagrangian.PIDLagrangian(0.25)
    assert (pid.lam, pid.I, pid.e_prev) == (0.0, 0.0, 0.0)


def test_default_gains_five_update_sequence_matches_hand_values():
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(0.25)  # kp 0.5, ki 0.05, kd 0.1, lambda_max 100
    # 1: e = (0.5 - 0.25) / 0.25 = 1; I = 0.05; d = max(0, 1 - 0) = 1; lam = 0.05 + 0.5 + 0.1 = 0.65
    # 2: e = 1; I = 0.1; d = max(0, 1 - 1) = 0; lam = 0.1 + 0.5 = 0.6
    # 3: e = (0.125 - 0.25) / 0.25 = -0.5; I = 0.1 - 0.025 = 0.075; d = max(0, -0.5 - 1) = 0;
    #    lam = 0.075 - 0.25 = -0.175, clipped to 0
    # 4: e = (0 - 0.25) / 0.25 = -1; I = 0.075 - 0.05 = 0.025; d = max(0, -1 + 0.5) = 0;
    #    lam = 0.025 - 0.5 = -0.475, clipped to 0
    # 5: e = 0; I = 0.025; d = max(0, 0 + 1) = 1; lam = 0.025 + 0 + 0.1 = 0.125
    lams, integrals, e_prevs = [], [], []
    for cost in [0.5, 0.5, 0.125, 0.0, 0.25]:
        lam = pid.update(cost)
        assert isinstance(lam, float)
        assert lam == pid.lam  # update returns lam and stores it
        lams.append(lam)
        integrals.append(pid.I)
        e_prevs.append(pid.e_prev)
    assert lams == pytest.approx([0.65, 0.6, 0.0, 0.0, 0.125], abs=1e-12)
    assert integrals == pytest.approx([0.05, 0.1, 0.075, 0.025, 0.025], abs=1e-12)
    assert e_prevs == pytest.approx([1.0, 1.0, -0.5, -1.0, 0.0], abs=1e-12)


@pytest.mark.parametrize(
    ("costs", "expected"),
    [
        # budget 0.25, so e = (1, 1, -0.5); I = 1, 2, 1.5 and lam = I
        ([0.5, 0.5, 0.125], [1.0, 2.0, 1.5]),
        # e = (-1, -1, 1); I = max(0, -1) = 0, then 0, then 1
        # (an unclamped integral would go -1, -2, -1 and give lam 0, 0, 0)
        ([0.0, 0.0, 0.5], [0.0, 0.0, 1.0]),
    ],
    ids=["integral_accumulates", "integral_floored_at_0"],
)
def test_pure_integral_matches_hand_values(costs, expected):
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(0.25, kp=0.0, ki=1.0, kd=0.0)
    assert [pid.update(c) for c in costs] == expected  # exact: every e and I here is dyadic


@pytest.mark.parametrize(
    ("gains", "costs", "expected"),
    [
        # budget 0.25, e = (1, 0): d = max(0, 1 - 0) = 1, then max(0, 0 - 1) = 0
        # (a |e - e_prev| derivative would give 1 the second time)
        ({"kp": 0.0, "ki": 0.0, "kd": 1.0}, [0.5, 0.25], [1.0, 0.0]),
        # e = (3, 1): lam = 3 + max(0, 3 - 0) = 6, then 1 + max(0, 1 - 3) = 1
        # (a two-sided derivative would give 1 + (1 - 3) = -1, clipped to 0)
        ({"kp": 1.0, "ki": 0.0, "kd": 1.0}, [1.0, 0.5], [6.0, 1.0]),
        # e = (-1, -0.5): d = max(0, -1 - 0) = 0, then max(0, -0.5 + 1) = 0.5: a rising error
        # counts below the budget too
        ({"kp": 0.0, "ki": 0.0, "kd": 1.0}, [0.0, 0.125], [0.0, 0.5]),
        # e = (1, 1): d = 1, then 0; a steady error has no derivative
        ({"kp": 0.0, "ki": 0.0, "kd": 1.0}, [0.5, 0.5], [1.0, 0.0]),
    ],
    ids=["falls_to_budget", "falls_with_kp", "rises_below_budget", "steady"],
)
def test_the_derivative_acts_only_on_increases(gains, costs, expected):
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(0.25, **gains)
    assert [pid.update(c) for c in costs] == expected  # exact: every e and d here is dyadic


@pytest.mark.parametrize(
    ("budget", "cost", "e"),
    [
        (0.25, 0.5, 1.0),  # (0.5 - 0.25) / 0.25
        (0.5, 1.0, 1.0),  # (1.0 - 0.5) / 0.5: the same relative overshoot gives the same e
        (0.1, 0.15, 0.5),  # (0.15 - 0.1) / 0.1
        (0.25, 0.125, -0.5),  # (0.125 - 0.25) / 0.25, so lam = max(0, -0.5) = 0
        (0.0, 1e-8, 1.0),  # a zero budget divides by the 1e-8 floor: 1e-8 / 1e-8
        (0.0, 0.0, 0.0),  # (0 - 0) / 1e-8: no division by zero
        (1e-9, 2e-8, 1.9),  # the floor is in the denominator only: (2e-8 - 1e-9) / 1e-8
    ],
)
def test_the_error_is_relative_to_the_budget(budget, cost, e):
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(budget, kp=1.0, ki=0.0, kd=0.0)  # so lam = clip(e, 0, 100)
    lam = pid.update(cost)
    assert pid.e_prev == pytest.approx(e, rel=1e-12, abs=1e-12)
    assert lam == pytest.approx(max(0.0, e), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("cap_kwargs", "cap"),
    [({"lambda_max": 1.5}, 1.5), ({}, 100.0)],
    ids=["cap_1.5", "default_cap_100"],
)
def test_lambda_stays_in_0_to_lambda_max_over_1000_updates_at_a_huge_cost(cap_kwargs, cap):
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(0.25, **cap_kwargs)  # default gains
    # e = (1e6 - 0.25) / 0.25 = 3,999,999, so kp*e alone is ~2e6: every lam sits on the cap
    lams = [pid.update(1e6) for _ in range(1000)]
    assert all(0.0 <= lam <= cap for lam in lams)
    assert lams == [cap] * 1000


def test_integral_is_clamped_to_lambda_max_so_lambda_unwinds_soon_after_a_huge_cost():
    """Aarmen's anti-windup decision (2026-10-28): I = clip(I + ki*e, 0, lambda_max).

    Default gains, budget 0.25, lambda_max 1.5, then 1,000 updates at cost 1e6 (e = 3,999,999).
    Clamped, I = 1.5 from the first update on; unclamped it would reach
    1000 x 0.05 x 3,999,999 = 199,999,950. At cost 0, e = -1 and d = max(0, -1 - 3,999,999) = 0,
    so lam = I - 0.5 with I falling by 0.05 per update:
      clamped:   lam = 1.5 - 0.05k - 0.5 = 1.0 - 0.05k, so 0.95 after 1 update and 0 after 20;
      unclamped: lam stays at the 1.5 cap until k = 3,999,998,961 and reaches 0 at 3,999,998,990.
    """
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(0.25, lambda_max=1.5)
    for _ in range(1000):
        pid.update(1e6)
    assert pid.I <= 1.5
    assert pid.I == 1.5
    at_zero = [pid.update(0.0) for _ in range(30)]
    assert at_zero[0] == pytest.approx(0.95, rel=1e-12)
    # k = 19 leaves lam = 0.05; k = 20 leaves float noise at most
    assert next((k for k, lam in enumerate(at_zero, start=1) if lam < 1e-9), None) == 20


@pytest.mark.parametrize(
    ("gains", "lam_start", "lam_after_50"),
    [
        # 100 updates at e = 1: I = 100 x 0.05 = 5, lam = 5 + 0.5 x 1 = 5.5 (d = 0 after the first).
        # At cost 0, e = -1 and d = 0, so I falls 0.05 per update: after 50, I = 2.5 and
        # lam = 2.5 - 0.5 = 2.0; I reaches 0 after 100
        ({}, 5.5, 2.0),
        # integral only: lam = I = 5, then 5 - 50 x 0.05 = 2.5 after 50 updates at cost 0
        ({"kp": 0.0, "kd": 0.0}, 5.0, 2.5),
    ],
    ids=["default_gains", "integral_only"],
)
def test_cost_held_at_zero_makes_lambda_decay_to_zero(gains, lam_start, lam_after_50):
    import greenleaf_control.lagrangian as lagrangian

    pid = lagrangian.PIDLagrangian(0.25, **gains)
    for _ in range(100):
        pid.update(0.5)  # e = (0.5 - 0.25) / 0.25 = 1
    assert pid.lam == pytest.approx(lam_start, rel=1e-9)
    decay = [pid.update(0.0) for _ in range(200)]
    assert decay[49] == pytest.approx(lam_after_50, rel=1e-9)
    assert all(later <= earlier for earlier, later in itertools.pairwise(decay))
    assert decay[-1] == 0.0
    assert pid.I == 0.0


# FixedMultiplier (issue #38: a zero-budget constraint keeps a multiplier of 1 in residual_lag)


@pytest.mark.parametrize(
    ("args", "value"),
    [((), 1.0), ((2.5,), 2.5)],
    ids=["default_1", "value_2.5"],
)
def test_fixed_multiplier_signature_default_and_lam(args, value):
    import greenleaf_control.lagrangian as lagrangian

    params = inspect.signature(lagrangian.FixedMultiplier.__init__).parameters
    assert {name: p.default for name, p in params.items() if name != "self"} == {"value": 1.0}
    # lam is the value from construction, before any update (a PIDLagrangian's starts at 0.0)
    assert lagrangian.FixedMultiplier(*args).lam == value


@pytest.mark.parametrize("cost", [0.0, 1e-8, 0.5, 1e6])
def test_fixed_multiplier_update_returns_its_value_for_any_cost(cost):
    """Known answer: update returns the float value, 1.0 by default, whatever the cost. A
    PIDLagrangian(0.0) with the default gains gives 0.0 at cost 0 (e = 0), 0.65 at 1e-8
    (e = 1e-8 / 1e-8 = 1: 0.05 + 0.5 + 0.1) and the cap, 100, at 0.5 and 1e6."""
    import greenleaf_control.lagrangian as lagrangian

    fixed = lagrangian.FixedMultiplier()
    lam = fixed.update(cost)
    assert isinstance(lam, float)
    assert lam == 1.0
    assert fixed.lam == 1.0


def test_fixed_multiplier_never_moves_over_a_sequence_of_costs():
    """1,000 updates at cost 1e6 (a PID's lambda would sit on its cap), then 1,000 at 0 (a PID's
    would unwind): every update returns the value, 1.0 by default or 2.5 when given, and lam stays
    there."""
    import greenleaf_control.lagrangian as lagrangian

    costs = [1e6] * 1000 + [0.0] * 1000
    for value in (1.0, 2.5):
        fixed = lagrangian.FixedMultiplier(value)
        assert [fixed.update(c) for c in costs] == [value] * 2000
        assert fixed.lam == value


# VecLagrangianReward


def test_wrapper_and_callback_are_sb3_classes():
    import greenleaf_control.lagrangian as lagrangian

    assert issubclass(lagrangian.VecLagrangianReward, VecEnvWrapper)
    assert issubclass(lagrangian.LagrangianCallback, BaseCallback)


def test_wrapper_lambdas_start_at_zero_and_leave_rewards_unchanged():
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv([(0.1, 0.0, 0.2)], [(0.0, 0.0, 0.0)], reward=0.75))
    assert isinstance(w.lambdas, np.ndarray)
    assert w.lambdas.dtype == np.float64
    assert w.lambdas.tolist() == [0.0, 0.0, 0.0]
    w.reset()
    _, rewards, _, _ = w.step(ACTIONS)
    # (0.75 - 0) / (1 + 0) = 0.75 in both envs, with or without a cost
    assert rewards.tolist() == [0.75, 0.75]


@pytest.mark.parametrize(
    ("reward", "costs", "lambdas", "expected"),
    [
        # (1 - (0.1 x 1 + 0 x 0 + 0.2 x 2)) / (1 + 1 + 0 + 2) = 0.5 / 4
        (1.0, (0.1, 0.0, 0.2), (1.0, 0.0, 2.0), 0.125),
        # no violation: (1 - 0) / 4; the 1 + sum(lam) scale still applies
        (1.0, (0.0, 0.0, 0.0), (1.0, 0.0, 2.0), 0.25),
        # lam = 0: (1 - 0) / 1, the reward passes through
        (1.0, (0.1, 0.0, 0.2), (0.0, 0.0, 0.0), 1.0),
        # CO2 alone: (1 - 0.5 x 2) / (1 + 2) = 0
        (1.0, (0.5, 0.0, 0.0), (2.0, 0.0, 0.0), 0.0),
        # temperature alone: (1 - 0.5 x 4) / (1 + 4) = -0.2
        (1.0, (0.0, 0.5, 0.0), (0.0, 4.0, 0.0), -0.2),
        # RH alone: (1 - 0.25 x 2) / (1 + 2) = 1/6
        (1.0, (0.0, 0.0, 0.25), (0.0, 0.0, 2.0), 1 / 6),
        # RH's lam on a CO2-only cost: (1 - 0) / (1 + 4) = 0.2; each lam meets only its own cost
        (1.0, (0.5, 0.0, 0.0), (0.0, 0.0, 4.0), 0.2),
        # a negative reward: (-0.5 - (0.5 x 2 + 0.25 x 4 + 0 x 1)) / (1 + 7) = -2.5 / 8
        (-0.5, (0.5, 0.25, 0.0), (2.0, 4.0, 1.0), -0.3125),
    ],
    ids=[
        "done_when",
        "no_cost",
        "lambda_zero",
        "co2_only",
        "temp_only",
        "rh_only",
        "rh_lambda_on_co2_cost",
        "negative_reward",
    ],
)
def test_wrapper_reward_matches_hand_values(reward, costs, lambdas, expected):
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv([costs], [costs], reward=reward))
    w.lambdas = np.array(lambdas)
    w.reset()
    _, rewards, _, _ = w.step(ACTIONS)
    # rel 1e-6: DummyVecEnv keeps rewards in float32
    assert rewards.tolist() == pytest.approx([expected, expected], rel=1e-6, abs=1e-9)


def test_wrapper_applies_each_envs_own_costs():
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv([(0.1, 0.0, 0.2)], [(0.0, 0.0, 0.0)]))
    w.lambdas = np.array([1.0, 0.0, 2.0])
    w.reset()
    _, rewards, _, _ = w.step(ACTIONS)
    # env 0: (1 - 0.5) / 4 = 0.125; env 1 has no cost: (1 - 0) / 4 = 0.25
    assert rewards.tolist() == pytest.approx([0.125, 0.25], rel=1e-6)


def test_pop_mean_costs_returns_the_per_step_means_then_resets():
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    w.reset()
    w.step(ACTIONS)
    w.step(ACTIONS)
    means = w.pop_mean_costs()
    assert isinstance(means, np.ndarray)
    assert means.dtype == np.float64
    assert means.shape == (3,)
    # 4 env-steps: (1.0 + 0.5 + 0.5 + 0) / 4, (0 + 0 + 0.5 + 0) / 4, (0.25 + 0 + 0 + 0.75) / 4
    # (a sum over both envs divided by the 2 vec-steps would give 1.0, 0.25, 0.5)
    assert means.tolist() == [0.5, 0.125, 0.25]
    w.step(ACTIONS)  # both toys restarted: (1.0, 0, 0.25) and (0.5, 0.5, 0)
    # only this step: (1.0 + 0.5) / 2, (0 + 0.5) / 2, (0.25 + 0) / 2
    # (without the reset, CO2 would be (2.0 + 1.5) / 6 = 0.583)
    assert w.pop_mean_costs().tolist() == [0.75, 0.25, 0.125]


def test_pop_mean_costs_with_no_steps_since_the_last_pop_raises():
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    w.reset()
    with pytest.raises(RuntimeError):
        w.pop_mean_costs()
    w.step(ACTIONS)
    w.pop_mean_costs()
    with pytest.raises(RuntimeError):
        w.pop_mean_costs()


def test_reset_passes_through_and_keeps_the_cost_sums():
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    w.reset()
    w.step(ACTIONS)  # step 0 of both toys: (1.0, 0, 0.25) and (0.5, 0.5, 0)
    obs = w.reset()
    assert obs.shape == (2, 1)
    # the step before the reset still counts: (1.0 + 0.5) / 2, (0 + 0.5) / 2, (0.25 + 0) / 2
    assert w.pop_mean_costs().tolist() == [0.75, 0.25, 0.125]
    w.step(ACTIONS)
    # the reset reached the toys, so this is step 0 again, not step 1 ((0.5 + 0) / 2, 0, 0.75 / 2)
    assert w.pop_mean_costs().tolist() == [0.75, 0.25, 0.125]


@pytest.mark.parametrize("missing", COST_KEYS)
def test_a_missing_cost_key_raises_keyerror(missing):
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B, drop=missing))
    w.reset()
    with pytest.raises(KeyError, match=missing):
        w.step(ACTIONS)


# LagrangianCallback


@pytest.mark.parametrize("n_pids", [0, 2, 4])
def test_callback_refuses_anything_but_three_pids(n_pids, tmp_path):
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    pids = [lagrangian.PIDLagrangian(0.125) for _ in range(n_pids)]
    with pytest.raises(ValueError):
        lagrangian.LagrangianCallback(w, pids, tmp_path / "lambdas.csv")


def test_callback_on_step_returns_true(tmp_path):
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    pids = [lagrangian.PIDLagrangian(0.125) for _ in range(3)]
    cb = lagrangian.LagrangianCallback(w, pids, tmp_path / "lambdas.csv")
    cb.init_callback(_ppo(w))
    assert cb.on_step() is True


def test_callback_updates_lambdas_logs_and_appends_a_csv_row_at_each_rollout_end(tmp_path):
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    # proportional only, so lam = e = (mean - 0.125) / 0.125
    pids = [lagrangian.PIDLagrangian(0.125, kp=1.0, ki=0.0, kd=0.0) for _ in range(3)]
    log = tmp_path / "lambdas.csv"
    cb = lagrangian.LagrangianCallback(w, pids, log)
    model = _ppo(w)
    cb.init_callback(model)
    w.reset()
    w.step(ACTIONS)
    w.step(ACTIONS)
    cb.on_rollout_end()
    # means (0.5, 0.125, 0.25): lam = (0.375 / 0.125, 0 / 0.125, 0.125 / 0.125) = (3, 0, 1)
    assert w.lambdas.tolist() == [3.0, 0.0, 1.0]
    assert _lag_logged(model) == {
        "lag/lambda_co2": 3.0,
        "lag/lambda_temp": 0.0,
        "lag/lambda_rh": 1.0,
        "lag/cost_co2": 0.5,
        "lag/cost_temp": 0.125,
        "lag/cost_rh": 0.25,
    }
    w.step(ACTIONS)  # both toys restarted: (1.0, 0, 0.25) and (0.5, 0.5, 0)
    cb.on_rollout_end()
    # means (0.75, 0.25, 0.125): lam = (0.625 / 0.125, 0.125 / 0.125, 0 / 0.125) = (5, 1, 0)
    assert w.lambdas.tolist() == [5.0, 1.0, 0.0]
    assert _lag_logged(model) == {
        "lag/lambda_co2": 5.0,
        "lag/lambda_temp": 1.0,
        "lag/lambda_rh": 0.0,
        "lag/cost_co2": 0.75,
        "lag/cost_temp": 0.25,
        "lag/cost_rh": 0.125,
    }
    rows = list(csv.reader(log.read_text().splitlines()))
    assert rows[0] == HEADER
    # no training ran, so the callback's num_timesteps is still 0
    assert [int(row[0]) for row in rows[1:]] == [0, 0]
    assert [[float(v) for v in row[1:]] for row in rows[1:]] == [
        [3.0, 0.0, 1.0, 0.5, 0.125, 0.25],
        [5.0, 1.0, 0.0, 0.75, 0.25, 0.125],
    ]


def test_callback_with_a_fixed_multiplier_logs_its_constant_beside_the_pids(tmp_path):
    """Issue #38: LagrangianCallback takes a FixedMultiplier wherever it takes a PIDLagrangian.

    Known answer: pids [FixedMultiplier(), PID(0.125) x 2], the PIDs proportional only, so
    lam = e = (mean - 0.125) / 0.125, and co2's lam is 1.0 whatever its cost:
      rollout 1, means (0.5, 0.125, 0.25): lam = (1, 0 / 0.125, 0.125 / 0.125) = (1, 0, 1)
      rollout 2, means (0.75, 0.25, 0.125): lam = (1, 0.125 / 0.125, 0 / 0.125) = (1, 1, 0)
    (a PID on co2 with this budget would give 3 then 5, as in the test above).
    """
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    pids = [
        lagrangian.FixedMultiplier(),
        *(lagrangian.PIDLagrangian(0.125, kp=1.0, ki=0.0, kd=0.0) for _ in range(2)),
    ]
    log = tmp_path / "lambdas.csv"
    cb = lagrangian.LagrangianCallback(w, pids, log)
    model = _ppo(w)
    cb.init_callback(model)
    w.reset()
    w.step(ACTIONS)
    w.step(ACTIONS)
    cb.on_rollout_end()
    assert w.lambdas.tolist() == [1.0, 0.0, 1.0]
    assert _lag_logged(model) == {
        "lag/lambda_co2": 1.0,
        "lag/lambda_temp": 0.0,
        "lag/lambda_rh": 1.0,
        "lag/cost_co2": 0.5,
        "lag/cost_temp": 0.125,
        "lag/cost_rh": 0.25,
    }
    w.step(ACTIONS)  # both toys restarted: (1.0, 0, 0.25) and (0.5, 0.5, 0)
    cb.on_rollout_end()
    assert w.lambdas.tolist() == [1.0, 1.0, 0.0]
    rows = list(csv.reader(log.read_text().splitlines()))
    assert rows[0] == HEADER
    assert [[float(v) for v in row[1:]] for row in rows[1:]] == [
        [1.0, 0.0, 1.0, 0.5, 0.125, 0.25],
        [1.0, 1.0, 0.0, 0.75, 0.25, 0.125],
    ]


def test_callback_sets_the_wrapper_lambdas_from_the_multipliers_at_training_start(tmp_path):
    """Issue #38 (the orchestrator's addition): at training start the callback sets
    wrapper.lambdas to [pid.lam for pid in pids], so a fixed multiplier prices its constraint from
    the first step, not only after the first rollout end. PIDLagrangian's lam starts at 0.0.

    Known answer: pids [FixedMultiplier(), PID, PID] give lambdas (1, 0, 0). The first step's
    costs are (1.0, 0, 0.25) in env 0 and (0.5, 0.5, 0) in env 1, reward 1, so the rewards are
    (1 - 1 x 1.0) / (1 + 1) = 0 and (1 - 1 x 0.5) / (1 + 1) = 0.25 (with lambdas still at 0: 1, 1).
    """
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    pids = [lagrangian.FixedMultiplier(), *(lagrangian.PIDLagrangian(0.125) for _ in range(2))]
    cb = lagrangian.LagrangianCallback(w, pids, tmp_path / "lambdas.csv")
    cb.init_callback(_ppo(w))
    cb.on_training_start(locals(), globals())
    assert isinstance(w.lambdas, np.ndarray)
    assert w.lambdas.dtype == np.float64
    assert w.lambdas.tolist() == [1.0, 0.0, 0.0]
    w.reset()
    _, rewards, _, _ = w.step(ACTIONS)
    assert rewards.tolist() == [0.0, 0.25]  # exact: every value here is dyadic


def test_ppo_learn_prices_the_first_rollout_with_the_fixed_multiplier(tmp_path):
    """Issue #38, through PPO.learn: the first rollout's rewards are priced with lambda_co2 = 1.

    No VecNormalize, so the rollout buffer holds the wrapper's rewards. 2 envs x n_steps 8 = one
    16-step rollout; each toy plays its 2-step episode 4 times. With lambdas (1, 0, 0) from the
    start, reward 1, each reward is (1 - 1 x cost_co2) / (1 + 1):
      env 0 (MIX_A): step 0, co2 1.0: 0;    step 1, co2 0.5: 0.25
      env 1 (MIX_B): step 0, co2 0.5: 0.25; step 1, co2 0:   0.5
    so the buffer's rows alternate (0, 0.25) and (0.25, 0.5); with lambdas at 0 they would all be
    1. At the rollout end the means are (0.5, 0.125, 0.25) and the default-gain PIDs on budget
    0.125 give e = (0, 1): temp I = 0, d = 0, lam 0; rh I = 0.05, d = 1, lam 0.05 + 0.5 + 0.1 = 0.65.
    """
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    pids = [lagrangian.FixedMultiplier(), *(lagrangian.PIDLagrangian(0.125) for _ in range(2))]
    log = tmp_path / "lambdas.csv"
    cb = lagrangian.LagrangianCallback(w, pids, log)
    model = _ppo(w)
    model.learn(total_timesteps=16, callback=cb)
    rewards = model.rollout_buffer.rewards
    assert rewards.shape == (8, 2)
    assert rewards.tolist() == [[0.0, 0.25], [0.25, 0.5]] * 4
    rows = list(csv.reader(log.read_text().splitlines()))
    assert rows[0] == HEADER
    assert len(rows) == 2
    assert int(rows[1][0]) == 16
    assert [float(v) for v in rows[1][1:]] == pytest.approx(
        [1.0, 0.0, 0.65, 0.5, 0.125, 0.25], rel=1e-12, abs=1e-12
    )


def test_callback_refuses_to_overwrite_an_existing_log(tmp_path):
    import greenleaf_control.lagrangian as lagrangian

    log = tmp_path / "lambdas.csv"
    log.write_text("from an earlier run\n")
    w = lagrangian.VecLagrangianReward(_venv(MIX_A, MIX_B))
    pids = [lagrangian.PIDLagrangian(0.125) for _ in range(3)]
    cb = lagrangian.LagrangianCallback(w, pids, log)
    cb.init_callback(_ppo(w))
    w.reset()
    w.step(ACTIONS)
    with pytest.raises(FileExistsError):
        cb.on_rollout_end()
    assert log.read_text() == "from an earlier run\n"


def test_ppo_learn_with_the_callback_writes_one_csv_row_per_rollout(tmp_path):
    """The contract's stack minus SubprocVecEnv: DummyVecEnv, VecMonitor, the wrapper, VecNormalize.

    2 envs x n_steps 8 = 16 timesteps per rollout, so learning 32 ends rollouts at 16 and 32.
    Each toy plays its 2-step episode 4 times per rollout: means (0.5, 0.125, 0.25), and with
    budget 0.125, e = (3, 0, 1) both times. Default gains (kp 0.5, ki 0.05, kd 0.1):
      rollout 1: I = 0.05e = (0.15, 0, 0.05), d = e; lam = I + 0.5e + 0.1e = (1.95, 0, 0.65)
      rollout 2: I = (0.3, 0, 0.1), d = 0;   lam = I + 0.5e = (1.8, 0, 0.6)
    """
    import greenleaf_control.lagrangian as lagrangian

    w = lagrangian.VecLagrangianReward(VecMonitor(_venv(MIX_A, MIX_B)))
    pids = [lagrangian.PIDLagrangian(0.125) for _ in range(3)]
    log = tmp_path / "lambdas.csv"
    cb = lagrangian.LagrangianCallback(w, pids, log)
    model = _ppo(VecNormalize(w))
    model.learn(total_timesteps=32, callback=cb)
    rows = list(csv.reader(log.read_text().splitlines()))
    assert rows[0] == HEADER
    assert [int(row[0]) for row in rows[1:]] == [16, 32]
    lams = [[float(v) for v in row[1:4]] for row in rows[1:]]
    assert lams[0] == pytest.approx([1.95, 0.0, 0.65], rel=1e-6, abs=1e-12)
    assert lams[1] == pytest.approx([1.8, 0.0, 0.6], rel=1e-6, abs=1e-12)
    costs = [[float(v) for v in row[4:]] for row in rows[1:]]
    assert costs == [[0.5, 0.125, 0.25], [0.5, 0.125, 0.25]]
    assert w.lambdas.tolist() == pytest.approx([1.8, 0.0, 0.6], rel=1e-12)

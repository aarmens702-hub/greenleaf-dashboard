"""The probe ladder (PROTOCOL §7, rung 6; issue #14): our exact training plumbing learns toy problems
with known answers, so a later failure can't be the plumbing.

Each probe trains PPO on a toy env from toys.py (pytest puts this folder on sys.path) through
train.build_vec_env and train.build_model, untouched: FlattenObservation over the toy's Dict
observation, DummyVecEnv (one env), VecMonitor, VecNormalize, then configs/ppo.yml's PPO on cpu,
seeded. The seam is test-side: envs.make_env is patched with toys.make, which build_vec_env's
functools.partial looks up at call time. The config has no run_dir, so nothing is written to disk.
Each probe runs at seeds 0 and 1 within a stated step budget.

Reading the answers:
- Values: model.policy.predict_values on venv.normalize_obs(raw) gives V in VecNormalize's units;
  divided by venv.normalize_reward(1), it is in units of the reward PPO saw for a raw +1. Neither
  call updates the stats. Where the reward clip doesn't bind, that is raw units (V x sqrt(ret_rms.var
  + 1e-8)).
- Actions: model.predict(venv.normalize_obs(raw), deterministic=True), the policy's mean clipped to
  the action bounds, as evaluation reads a policy.
- Each probe's criterion is one function, shared by its trained test and by the untrained negative
  control, which shows that none of them passes before training.

Budgets: measured 2026-10-28 at d3844cb (M3, gl-gym 2d3febb, SB3 2.3.2, torch 2.14.1). The criterion
was checked after every 2,048-step update for 60 updates at seeds 0 and 1; each budget is 2x the
first update from which it held at both seeds, as a multiple of 2,048 (one rollout).
"""

import functools

import gymnasium as gym
import numpy as np
import pytest
import torch
import toys
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from greenleaf_control import envs, train

SEEDS = [0, 1]
GAMMA = 0.9631  # configs/ppo.yml, copied by hand


def _config(seed: int) -> dict:
    """What build_vec_env and build_model read for ppo; no run_dir, so no tb/ on disk."""
    return {"method": "ppo", "seed": seed, "season_length": 1, "penalty_weight": 1.0}


def _build(monkeypatch, toy, seed: int):
    """The real build_vec_env and build_model on toy, untrained."""
    monkeypatch.setattr(envs, "make_env", functools.partial(toys.make, toy))
    venv = train.build_vec_env(_config(seed), 1)
    return train.build_model(_config(seed), venv), venv


def _train(monkeypatch, toy, seed: int, steps: int):
    model, venv = _build(monkeypatch, toy, seed)
    model.learn(steps)
    return model, venv


def _raw(xs) -> np.ndarray:
    """Raw (flattened) observations, one row per x."""
    return np.array([[x] for x in xs], dtype=np.float32)


def _values(model, venv, xs) -> tuple[np.ndarray, np.ndarray]:
    """V at raw observations xs: (normalised units, units of the reward PPO saw for a raw +1)."""
    obs = obs_as_tensor(venv.normalize_obs(_raw(xs)), model.device)
    with torch.no_grad():
        v = model.policy.predict_values(obs).numpy().ravel()
    return v, v / venv.normalize_reward(np.array([1.0]))[0]


def _actions(model, venv, raw: np.ndarray) -> np.ndarray:
    """The deterministic action at each raw observation."""
    action, _ = model.predict(venv.normalize_obs(raw), deterministic=True)
    return action.ravel()


def _constant_reward(model, venv) -> tuple[bool, str]:
    """V(0) = 1 +- 0.1, in units of the reward PPO saw."""
    v_norm, v = _values(model, venv, [0.0])
    readout = f"V(0) = {v_norm[0]:.4f} normalised, {v[0]:.4f} in units of the reward PPO saw"
    return bool(abs(v[0] - 1.0) <= 0.1), readout


def _obs_dependent(model, venv) -> tuple[bool, str]:
    """V(+1) = +1 and V(-1) = -1, +- 0.2."""
    _, v = _values(model, venv, [1.0, -1.0])
    ok = abs(v[0] - 1.0) <= 0.2 and abs(v[1] + 1.0) <= 0.2
    return bool(ok), f"V(+1) = {v[0]:.4f}, V(-1) = {v[1]:.4f}"


def _discounting(model, venv) -> tuple[bool, str]:
    """V(s0) = gamma +- 0.1 and V(s0) / V(s1) = gamma +- 0.01."""
    _, v = _values(model, venv, [0.0, 1.0])
    ratio = v[0] / v[1]
    ok = abs(v[0] - GAMMA) <= 0.1 and abs(ratio - GAMMA) <= 0.01
    return bool(ok), f"V(s0) = {v[0]:.4f}, V(s1) = {v[1]:.4f}, ratio {ratio:.4f}"


def _action_dependent(model, venv) -> tuple[bool, str]:
    """The deterministic action at obs 0 is > 0.9."""
    [a] = _actions(model, venv, _raw([0.0]))
    return bool(a > 0.9), f"action = {a:.4f}"


def _action_times_obs(model, venv) -> tuple[bool, str]:
    """The deterministic action is > 0.9 at obs +1 and < -0.9 at obs -1."""
    a_plus, a_minus = _actions(model, venv, _raw([1.0, -1.0]))
    readout = f"action(+1) = {a_plus:.4f}, action(-1) = {a_minus:.4f}"
    return bool(a_plus > 0.9 and a_minus < -0.9), readout


CRITERIA = {
    toys.ConstantReward: _constant_reward,
    toys.ObsDependentReward: _obs_dependent,
    toys.Discounting: _discounting,
    toys.ActionDependentReward: _action_dependent,
    toys.ActionTimesObs: _action_times_obs,
}


def test_seam_builds_the_real_stack_on_the_toy(monkeypatch):
    """With envs.make_env patched, build_vec_env builds VecNormalize -> VecMonitor -> DummyVecEnv on
    the toy, whose Dict observation FlattenObservation makes a 1-dim Box. If train ever bound
    make_env at import time, this would build the greenhouse instead (23 dims for ppo)."""
    monkeypatch.setattr(envs, "make_env", functools.partial(toys.make, toys.ActionTimesObs))
    venv = train.build_vec_env(_config(0), 1)
    assert [type(venv), type(venv.venv), type(venv.venv.venv)] == [
        VecNormalize,
        VecMonitor,
        DummyVecEnv,
    ]
    [env] = venv.venv.venv.envs
    assert isinstance(env, gym.wrappers.FlattenObservation)
    assert isinstance(env.unwrapped, toys.ActionTimesObs)
    assert venv.observation_space.shape == (1,)
    venv.close()


@pytest.mark.parametrize("toy", list(CRITERIA), ids=lambda toy: toy.__name__)
def test_no_probe_passes_untrained(monkeypatch, toy):
    """Negative control: every probe's criterion fails on the untrained model at seed 0 (the stack
    is built and never learns or resets), so no probe can pass vacuously. Measured: V = 0.0 for
    the constant, V = -0.0144 / +0.0155 at +1 / -1, V(s0) = 0.0, actions ~1e-4."""
    model, venv = _build(monkeypatch, toy, 0)
    ok, readout = CRITERIA[toy](model, venv)
    print(f"untrained {toy.__name__}: {readout}")
    assert not ok, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_constant_reward_value_is_1(monkeypatch, seed):
    """Constant reward: +1 on a one-step episode. Known answer: V = 1 +- 0.1.

    PPO never sees the 1. VecNormalize divides by sqrt(ret_rms.var + 1e-8); every return is exactly
    1, so that var falls from 1 to ~2e-4 after one step and to ~2e-4 / N after N, and
    1 / sqrt(2e-4) = 70.7 is clipped to clip_reward = 10.0 on every step: the value net learns
    V = 10. So V is read in units of the reward PPO saw, V / normalize_reward(1) = V / 10.
    De-normalising by sqrt(ret_rms.var + 1e-8) instead is meaningless here: with the clip it gives
    10 x sqrt(2e-4 / N + 1e-8) = 0.00135 at N = 24,576, which depends only on N.

    Budget 24,576 steps (12 updates); passes from 12,288 at seeds 0 and 1. Untrained: V = 0.0.
    """
    model, venv = _train(monkeypatch, toys.ConstantReward, seed, 24_576)
    ok, readout = _constant_reward(model, venv)
    print(f"seed {seed}: {readout}")
    assert ok, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_obs_dependent_value_tracks_the_observation(monkeypatch, seed):
    """Observation-dependent reward: obs +1 or -1 at random, reward = obs, one step.

    Known answer: V(+1) = +1 and V(-1) = -1, +- 0.2, in raw units (R = +-1, so ret_rms.var ~ 1 and
    the clip doesn't bind).

    Budget 8,192 steps (4 updates); passes from 4,096 at seeds 0 and 1. Untrained: V ~ +-0.015.
    """
    model, venv = _train(monkeypatch, toys.ObsDependentReward, seed, 8_192)
    ok, readout = _obs_dependent(model, venv)
    print(f"seed {seed}: {readout}")
    assert ok, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_discounting_first_state_value_is_gamma(monkeypatch, seed):
    """Discounting: obs 0 then 1, reward 0 then +1, two steps.

    Known answer: V(s1) = 1 and V(s0) = gamma x 1 = 0.9631 in raw units (R alternates 0, 1, so
    ret_rms.var = 0.25 and the normalised +1 is 2: the clip doesn't bind). V(s0) must be
    0.9631 +- 0.1 (the Done-when), and V(s0) / V(s1) = 0.9631 +- 0.01: the ratio is unit-free, and
    +- 0.1 alone can't tell gamma from 1.

    Budget 20,480 steps (10 updates); passes from 10,240 at seeds 0 and 1. Untrained: V(s0) = 0.0.
    """
    model, venv = _train(monkeypatch, toys.Discounting, seed, 20_480)
    ok, readout = _discounting(model, venv)
    print(f"seed {seed}: {readout}")
    assert ok, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_action_dependent_mean_action_goes_positive(monkeypatch, seed):
    """Action-dependent reward: obs 0, reward +1 if action > 0 else -1, one step.

    Known answer: any action > 0 is optimal, so the policy's mean (its deterministic action) is
    pushed up while its Gaussian has mass below 0; it must end > 0.9.

    Budget 24,576 steps (12 updates); passes from 12,288 at seeds 0 and 1. Untrained: 0.0.
    """
    model, venv = _train(monkeypatch, toys.ActionDependentReward, seed, 24_576)
    ok, readout = _action_dependent(model, venv)
    print(f"seed {seed}: {readout}")
    assert ok, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_action_times_obs_picks_the_right_side_per_context(monkeypatch, seed):
    """Action x observation: obs x = +1 or -1 at random, reward +1 if the action's sign matches x,
    else -1, one step.

    Known answer: the optimal action is > 0 at +1 and <= 0 at -1. This is the deterministic reading,
    as evaluation reads a policy. With only two contexts, the Done-when's "right action per context
    > 90% of the time" means both contexts right, by a margin: as in the action-dependent probe,
    the mean is pushed to the bound, so the action must end > 0.9 at +1 and < -0.9 at -1.
    - The sign-only literal isn't enough: the untrained policy already has the right sign in both
      contexts at seeds 0, 1 and 5 (|a| ~ 1e-4), so it passes with no training.
    - The sampled-policy reading, P(right) = Phi(mu / sigma) per context from the unclipped
      Gaussian mean, is not claimed: the Gaussian mean isn't clipped (sigma stays ~1 at lr 2e-5),
      so that reading isn't ruled out either.

    Budget 20,480 steps (10 updates); passes from 10,240 at seeds 0 and 1. Untrained: |a| ~ 1e-4.
    """
    model, venv = _train(monkeypatch, toys.ActionTimesObs, seed, 20_480)
    ok, readout = _action_times_obs(model, venv)
    print(f"seed {seed}: {readout}")
    assert ok, readout

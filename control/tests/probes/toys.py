"""The probe ladder's toy envs (PROTOCOL §7, rung 6; issue #14), each with its known answer.

Each toy looks like upstream's env to the training plumbing: a Dict observation (one key, "x", a
float32 Box(-1, 1, (1,))) that FlattenObservation turns into a flat Box, and a Box(-1, 1, (1,))
float32 action. Every episode ends terminated=True, truncated=False, so PPO never bootstraps past
it. Random observations come from self.np_random, seeded by reset(seed=...).

make(toy, method, *, seed, **_) stands in for greenleaf_control.envs.make_env: the probes patch
functools.partial(make, Toy) over it, so train.build_vec_env builds its real stack on the toy.

The new-piece probes (issue #15) add two toys. CostEqualsReward is one more make() toy, for the
Lagrangian. ResidualTarget goes one level lower: make_raw(toy, **_) stands in for
envs.make_raw_env, so the real envs.make_env puts its ResidualRBWrapper and SolverFailureFlag on
it. Its action is the wrapper's absolute control, a Box(0, 1, (6,)).

Known answers are stated in raw units (gamma = 0.9631, configs/ppo.yml). The stack trains on
VecNormalize's rewards, r / sqrt(ret_rms.var + 1e-8) clipped to +-10, where ret_rms tracks the
running discounted return R_t = gamma * R_{t-1} + r_t, zeroed at each episode end; each docstring
also gives what that makes of the answer.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class _Toy(gym.Env):
    """Shared spaces: Dict({"x": Box(-1, 1, (1,))}) observations, Box(-1, 1, (1,)) actions."""

    def __init__(self):
        self.observation_space = spaces.Dict({"x": spaces.Box(-1.0, 1.0, (1,), np.float32)})
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)

    @staticmethod
    def _obs(x: float) -> dict:
        return {"x": np.array([x], dtype=np.float32)}


class ConstantReward(_Toy):
    """One step, observation 0, reward +1 whatever the action.

    Known answer: V(s) = 1 in raw units. Through VecNormalize: every return R is exactly 1, so
    ret_rms.var falls from its initial 1 (count 1e-4) to ~2e-4 after the first step and towards 0
    after that; 1 / sqrt(2e-4 + 1e-8) = 70.7 > 10 already, so PPO sees reward = clip_reward = 10.0
    on every step and its value net must learn V = 10 (normalised units): V = 1 in units of the
    reward it saw for the raw +1.
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(0.0), {}

    def step(self, action):
        return self._obs(0.0), 1.0, True, False, {}


class ObsDependentReward(_Toy):
    """One step, observation x = +1 or -1 at random (p = 0.5), reward x whatever the action.

    Known answer: V(+1) = +1 and V(-1) = -1 in raw units. Through VecNormalize: R = +-1, so
    ret_rms.var ~ 1 and the normalised value is ~ +-1 too; de-normalised, exactly +-1.
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.choice([-1.0, 1.0]))
        return self._obs(self.x), {}

    def step(self, action):
        return self._obs(self.x), self.x, True, False, {}


class Discounting(_Toy):
    """Two steps: observation 0 then 1; reward 0 on step 1 and +1 on step 2, whatever the action.

    Known answer: V(s1) = 1 and V(s0) = gamma * 1 = 0.9631 in raw units, so V(s0) / V(s1) = gamma.
    Through VecNormalize: R alternates 0, 1 (mean 0.5, var 0.25), so the normalised rewards are
    0 and 1 / sqrt(0.25) = 2: V(s1) = 2, V(s0) = 2 * 0.9631 = 1.926 (normalised units).
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(0.0), {}

    def step(self, action):
        self.t += 1
        if self.t == 1:
            return self._obs(1.0), 0.0, False, False, {}
        return self._obs(1.0), 1.0, True, False, {}


class ActionDependentReward(_Toy):
    """One step, observation 0, reward +1 if action > 0 else -1.

    Known answer: any action > 0 is optimal, so the policy's mean (its deterministic action) is
    pushed up for as long as its Gaussian puts mass below 0: it ends > 0.9 (clipped to <= 1).
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(0.0), {}

    def step(self, action):
        return self._obs(0.0), 1.0 if action[0] > 0 else -1.0, True, False, {}


class ActionTimesObs(_Toy):
    """One step, observation x = +1 or -1 at random (p = 0.5); reward +1 if the action's sign
    matches x (action > 0 when x = +1, action <= 0 when x = -1), else -1.

    Known answer: the optimal action is > 0 in context x = +1 and <= 0 in context x = -1; as in
    ActionDependentReward, the policy's mean is pushed to the bound in each context, so its
    deterministic action ends > 0.9 at +1 and < -0.9 at -1.
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = float(self.np_random.choice([-1.0, 1.0]))
        return self._obs(self.x), {}

    def step(self, action):
        right = (action[0] > 0) == (self.x > 0)
        return self._obs(self.x), 1.0 if right else -1.0, True, False, {}


class ResidualTarget(_Toy):
    """One step, observation 0; the action u is 6 absolute controls in [0, 1], and the reward is
    -sum_i (u_i - target)^2. info["controls"] is u as the toy received it.

    Known answer: under ResidualRBWrapper with a base controller fixed at 0.30 and scale 0.15, the
    toy gets u = clip(0.30 + 0.15 a, 0, 1) with a in [-1, 1] (SB3 clips it), so the reward peaks
    at u = target on every dimension: a* = (target - 0.30) / 0.15, which is 1.0 for target 0.45
    (on the action bound) and 2/3 for target 0.40 (interior).
    """

    def __init__(self, target: float):
        super().__init__()
        self.action_space = spaces.Box(0.0, 1.0, (6,), np.float32)
        self.target = target

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(0.0), {}

    def step(self, action):
        u = np.array(action, dtype=np.float64)
        reward = -float(np.sum((u - self.target) ** 2))
        return self._obs(0.0), reward, True, False, {"controls": u}


class CostEqualsReward(_Toy):
    """One step, observation 0; u = (a + 1) / 2 in [0, 1] for a in [-1, 1]; reward u, and info
    cost_co2 = u, cost_temp = cost_rh = 0 (all three keys, as constraint_costs reads each).

    Known answer, under VecLagrangianReward with budget 0.3 on cost_co2: the reward PPO is shaped
    with is (u - lambda u) / (1 + lambda) = u (1 - lambda) / (1 + lambda). It rises with u while
    lambda < 1 and falls while lambda > 1, so u is pushed up or down until the PID's integral
    stops moving, which needs the rollout's mean cost at the budget: u* = 0.3, and lambda_co2
    settles near 1.0, the one value at which the shaped reward is flat in u.
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(0.0), {}

    def step(self, action):
        u = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0
        info = {"cost_co2": u, "cost_temp": 0.0, "cost_rh": 0.0}
        return self._obs(0.0), u, True, False, info


def make_raw(toy, **_) -> gym.Env:
    """make_raw_env's stand-in: FlattenObservation(toy()), toy a zero-argument callable; the weather,
    season and penalty arguments are ignored. make_env resets and seeds what it wraps."""
    return gym.wrappers.FlattenObservation(toy())


def make(toy: type[_Toy], method: str, *, seed: int, **_) -> gym.Env:
    """make_env's stand-in: FlattenObservation(toy()), reset once with seed, action space seeded."""
    env = gym.wrappers.FlattenObservation(toy())
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

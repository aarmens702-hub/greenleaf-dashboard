"""The PID-Lagrangian: one multiplier per constraint, updated from each rollout's mean cost.

After Stooke, Achiam and Abbeel 2020, "Responsive Safety in Reinforcement Learning by PID Lagrangian
Methods" (arXiv 2007.03964). The error is relative to the budget, the derivative acts only on
increases of the error, and lambda is clamped to [0, lambda_max]. Anti-windup (Aarmen's decision,
2026-10-28): the integral is clamped to [0, lambda_max] too, so it never passes lambda_max and
lambda leaves the cap on the first update under budget. Costs are >= 0, so e >= -1 and the integral
falls by at most ki per update: with the integral at the cap, lambda takes (lambda_max - kp) / ki
updates to reach 0 at zero cost (20 at lambda_max 1.5; 1,990 at the defaults, far more than the
~244 updates in a run). Prompt recovery within a run needs a smaller lambda_max or a larger ki,
which are per-run settings.

FixedMultiplier holds a zero-budget constraint's lambda at 1, where a PID's would pin at its cap.

VecLagrangianReward sits between VecMonitor and VecNormalize and folds the multipliers into the
reward; LagrangianCallback updates them at each rollout end and logs them.
"""

import csv
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper

from greenleaf_control.rewards import CONSTRAINTS, constraint_costs


class PIDLagrangian:
    """One constraint's multiplier; update() takes a rollout's mean cost and returns lambda."""

    def __init__(
        self,
        budget: float,
        kp: float = 0.5,
        ki: float = 0.05,
        kd: float = 0.1,
        lambda_max: float = 100.0,
    ) -> None:
        self.budget = budget
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_max = lambda_max
        self.lam = 0.0
        self.I = 0.0
        self.e_prev = 0.0

    def update(self, mean_cost: float) -> float:
        """e = (cost - budget) / max(budget, 1e-8); lam = clip(I + kp*e + kd*max(0, e - e_prev))."""
        e = (mean_cost - self.budget) / max(self.budget, 1e-8)
        self.I = min(self.lambda_max, max(0.0, self.I + self.ki * e))
        d = max(0.0, e - self.e_prev)
        self.e_prev = e
        self.lam = float(min(self.lambda_max, max(0.0, self.I + self.kp * e + self.kd * d)))
        return self.lam


class FixedMultiplier:
    """A constant multiplier for a zero-budget constraint; update() ignores the cost."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value
        self.lam = float(value)

    def update(self, mean_cost: float) -> float:
        return self.lam


class VecLagrangianReward(VecEnvWrapper):
    """Each reward becomes (r - sum(lambda_i * c_i)) / (1 + sum(lambda_i)); costs are accumulated."""

    def __init__(self, venv) -> None:
        super().__init__(venv)
        self.lambdas = np.zeros(len(CONSTRAINTS))
        self._cost_sum = np.zeros(len(CONSTRAINTS))
        self._n = 0

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        costs = np.stack([constraint_costs(info) for info in infos])  # (n_envs, 3)
        self._cost_sum += costs.sum(axis=0)
        self._n += len(infos)
        rews = (rews - costs @ self.lambdas) / (1.0 + self.lambdas.sum())
        return obs, rews, dones, infos

    def pop_mean_costs(self) -> np.ndarray:
        """The mean cost per env-step since the last pop, shape (3,); then the sums restart."""
        if self._n == 0:
            raise RuntimeError("no env steps since the last pop_mean_costs()")
        means = self._cost_sum / self._n
        self._cost_sum = np.zeros(len(CONSTRAINTS))
        self._n = 0
        return means


class LagrangianCallback(BaseCallback):
    """At each rollout end: update the PIDs, set wrapper.lambdas, log lag/* and a lambdas.csv row."""

    def __init__(
        self,
        wrapper: VecLagrangianReward,
        pids: list[PIDLagrangian | FixedMultiplier],
        log_path: Path,
    ) -> None:
        if len(pids) != len(CONSTRAINTS):
            raise ValueError(
                f"need one PIDLagrangian per constraint {CONSTRAINTS}, got {len(pids)}"
            )
        super().__init__()
        self.wrapper = wrapper
        self.pids = list(pids)
        self.log_path = Path(log_path)
        self._wrote = False

    def _on_training_start(self) -> None:
        """A FixedMultiplier prices from the first rollout; a PID's lam starts at 0."""
        self.wrapper.lambdas = np.array([pid.lam for pid in self.pids])

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        means = [float(m) for m in self.wrapper.pop_mean_costs()]
        lams = [pid.update(m) for pid, m in zip(self.pids, means, strict=True)]
        self.wrapper.lambdas = np.array(lams)
        for c, lam, cost in zip(CONSTRAINTS, lams, means, strict=True):
            self.logger.record(f"lag/lambda_{c}", lam)
            self.logger.record(f"lag/cost_{c}", cost)
        first = not self._wrote
        # "x" on the first write: an existing log from another run raises and is left untouched.
        with self.log_path.open("x" if first else "a", newline="") as f:
            writer = csv.writer(f)
            if first:
                lam_cols = [f"lambda_{c}" for c in CONSTRAINTS]
                writer.writerow(["timesteps", *lam_cols, *(f"cost_{c}" for c in CONSTRAINTS)])
            writer.writerow([int(self.num_timesteps), *lams, *means])
        self._wrote = True

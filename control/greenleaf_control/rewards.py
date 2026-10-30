"""The training reward: upstream's profit-minus-violations, with true labels and a tunable weight.

ConstraintCostReward keeps GreenLight-Gym's GreenhouseReward (gl_gym @2d3febb) money terms, scaling
and summation, so at penalty_weight=1 its reward is upstream's to the bit. It exists to fix two
things. First, each scaled violation is reported under its true name (cost_co2, cost_temp,
cost_rh): upstream's info["temp_penalty"] holds CO2 and info["co2_penalty"] holds temperature.
Second, the penalty is weighted, so the Lagrangian can tune it. Upstream's lamp penalty never fires
(it returns a local 0), so there is no lamp term here; paper_reward keeps upstream's value as is.

metrics.py grades runs from raw states and controls and must never import this module.
"""

import numpy as np
from gl_gym.components.rewards import GreenhouseReward

CONSTRAINTS = ("co2", "temp", "rh")


class ConstraintCostReward(GreenhouseReward):
    """reward = scaled_profit - penalty_weight * (cost_co2 + cost_temp + cost_rh)."""

    def __init__(self, *, penalty_weight: float = 1.0, **greenhouse_reward_kwargs) -> None:
        self.penalty_weight = penalty_weight
        super().__init__(**greenhouse_reward_kwargs)

    def compute_reward(self, ctx) -> tuple[float, dict]:
        paper_reward, info = super().compute_reward(ctx)
        # Upstream's own scaling (rewards.py:208-209); _output_violations is in CONSTRAINTS order.
        costs = self.scale_reward(
            self._output_violations(ctx), self.min_state_violations, self.max_state_violations
        )
        cost_co2, cost_temp, cost_rh = (float(c) for c in costs)
        scaled_profit = float(self.scale_reward(info["profit"], self.min_profit, self.max_profit))
        # Summed left to right, as upstream's sum() does, so w=1 is bit-identical to upstream.
        reward = float(scaled_profit - self.penalty_weight * (cost_co2 + cost_temp + cost_rh))
        for key in ("temp_penalty", "co2_penalty", "rh_penalty"):  # upstream's swapped labels
            del info[key]
        info.update(
            reward=reward,
            scaled_profit=scaled_profit,
            cost_co2=cost_co2,
            cost_temp=cost_temp,
            cost_rh=cost_rh,
            paper_reward=float(paper_reward),
        )
        return reward, info


def constraint_costs(info: dict) -> np.ndarray:
    """The step's scaled constraint costs, shape (3,), in CONSTRAINTS order."""
    return np.array([info[f"cost_{c}"] for c in CONSTRAINTS], dtype=np.float64)

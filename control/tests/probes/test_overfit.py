"""Overfit a trivial case (issue #16, probe; PROTOCOL §7, ladder rung 7, amended 2026-10-27).

PPO with the shipped plumbing (train.build_vec_env and train.build_model: configs/ppo.yml as written,
4 envs, VecNormalize) trains on one fixed season, PAPER over 2 days (192 steps), with penalty weight
0, seed 0, for 196,608 steps: 24 rollouts of n_steps 2,048 x n_envs 4 = 8,192, under the issue's
200,000 cap (SB3 finishes whole rollouts, so asking for 200,000 would run 204,800). No early stop.

Known answer: PPO's return clearly beats both `random` and `zero_controls` on the same 2-day
season. Return is the season's sum of info["scaled_profit"], which at w = 0 is exactly the reward.
The baselines come from envs.run_baseline, which runs at w = 1, so their scaled_profit is summed, not
their reward. On d3844cb they were random 67.5762 (profit -1.123359 EUR/m2), zero_controls 126.8019
(0.075457) and rb_default 86.8200 (-0.733836). They are computed live here, not pinned, and each
baseline's season must be valid (192 steps, no solver failure), so a short one can't lower the gate.

- w = 0 reached the env: PPO's episode has nonzero summed costs (cost_co2 + cost_temp + cost_rh),
  yet its summed reward equals its summed scaled_profit. At w = 1 they would differ by those costs.

- Affine identity (exact at w = 0): return = 123.07 + 49.40 x profit_eur_m2 over this season, so
  one return unit is 1 / 49.40 = 0.0202 EUR/m2 of profit.
- Gate: ppo_return >= max(random, zero_controls) + 1.0. The +1.0 (0.0202 EUR/m2) is above PPO's
  start handicap of about 0.64 units: upstream starts every actuator at u = 0.5 and moves it at
  most 0.1 per step, so PPO pays for heat, CO2 and lamps while it closes them; zero_controls
  doesn't. The eval is one deterministic season of one seed, so this is a threshold, not a CI.
- The gap to rb_default is printed and put in the message, not asserted: a hand-tuned controller
  can win a 2-day season, so beating it is not a plumbing test. pytest shows the printed line only
  with -s (or -rP); run with -s to see the recorded line.
- If PPO can't beat random and zero_controls, that is a finding to report, not a margin to loosen.
"""

import copy

import numpy as np
import pytest

from greenleaf_control import envs, metrics, scenarios
from greenleaf_control.rewards import CONSTRAINTS, constraint_costs

# greenleaf_control.train is imported inside the test body, as tests/test_train_smoke.py does.

SEASON_DAYS = 2
SEASON_STEPS = 192  # 2 days x 96 steps (86,400 s / 900 s)
TIMESTEPS = 196_608  # 24 rollouts x (n_steps 2,048 x n_envs 4 = 8,192), under the 200,000 cap
N_ENVS = 4  # configs/ppo.yml's n_envs
MARGIN = 1.0  # return units; 1 / 49.40 = 0.0202 EUR/m2 of profit
BASELINES = ("random", "zero_controls", "rb_default")


def _ppo_season(train, model, trained, config) -> list[dict]:
    """One deterministic season with frozen copies of the trained VecNormalize stats, as
    train._final_eval freezes them; its per-step infos. Moves to evaluate.run_policy when 13b lands."""
    ev = train.build_vec_env(config, 1)
    try:
        ev.obs_rms = copy.deepcopy(trained.obs_rms)
        ev.ret_rms = copy.deepcopy(trained.ret_rms)
        ev.training = False
        ev.norm_reward = False
        obs = ev.reset()
        infos, done = [], False
        while not done:  # DummyVecEnv resets itself when the season ends
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, step_infos = ev.step(action)
            infos.append(step_infos[0])
            done = bool(dones[0])
    finally:
        ev.close()
    return infos


@pytest.mark.slow
def test_ppo_overfits_paper_two_day_season_beating_random_and_zero_controls():
    """PPO (ppo.yml, seed 0, 196,608 steps) on PAPER's 2-day season at w = 0 must return at least
    max(random, zero_controls) + 1.0. Every episode must be valid (192 steps, no solver failure),
    and PPO's costs must be nonzero while its reward equals its scaled_profit (so w = 0 reached the
    env). The gap to rb_default is printed (run with -s to see it), not gated."""
    import greenleaf_control.train as train

    config = {"method": "ppo_paper", "seed": 0, "season_length": SEASON_DAYS, "penalty_weight": 0.0}
    venv = train.build_vec_env(config, N_ENVS)
    try:
        model = train.build_model(config, venv)
        model.learn(TIMESTEPS)
        trained_steps = model.num_timesteps
        episodes = {"ppo": _ppo_season(train, model, venv, config)}
    finally:
        venv.close()
    for name in BASELINES:
        episodes[name] = envs.run_baseline(name, scenarios.PAPER, season_length=SEASON_DAYS)

    raw = envs.make_raw_env(scenario=scenarios.PAPER, normalize_actions=False)
    params = np.asarray(raw.unwrapped.p)
    raw.close()
    graded = {
        name: metrics.episode_metrics(
            *metrics.trace_from_infos(infos), params, season_steps=SEASON_STEPS
        )
        for name, infos in episodes.items()
    }
    returns = {
        name: sum(info["scaled_profit"] for info in infos) for name, infos in episodes.items()
    }
    ppo = episodes["ppo"]
    ppo_costs = np.sum([constraint_costs(info) for info in ppo], axis=0)  # co2, temp, rh
    line = (
        f"overfit PAPER {SEASON_DAYS}d w=0 seed 0, {trained_steps:,} steps: "
        + ", ".join(
            f"{name} return {returns[name]:.4f} profit {graded[name]['profit_eur_m2']:.6f}"
            for name in ("ppo", *BASELINES)
        )
        + f"; ppo - rb_default {returns['ppo'] - returns['rb_default']:+.4f} (recorded, not gated)"
        + "; ppo summed costs "
        + " ".join(f"{c} {v:.4f}" for c, v in zip(CONSTRAINTS, ppo_costs, strict=True))
    )
    print(line)

    assert trained_steps == TIMESTEPS, line
    assert len(ppo) == SEASON_STEPS, line
    assert not any(info["solver_failure"] for info in ppo), line
    assert graded["ppo"]["steps"] == SEASON_STEPS, line
    assert graded["ppo"]["solver_failures"] == 0, line
    assert graded["ppo"]["valid"] is True, line
    # The gate is built from the baselines, so a short baseline season would lower it.
    for name in BASELINES:
        assert graded[name]["valid"] is True, f"{name} season invalid; {line}"
    # Each step's reward is scaled_profit - w x costs. With nonzero costs, equal sums mean w = 0
    # reached the env; with zero costs they'd be equal at any w.
    assert ppo_costs.sum() > 0, line
    assert sum(info["reward"] for info in ppo) == pytest.approx(returns["ppo"], abs=1e-9), line
    gate = max(returns["random"], returns["zero_controls"]) + MARGIN
    assert returns["ppo"] >= gate, f"PPO return {returns['ppo']:.4f} < gate {gate:.4f}; {line}"

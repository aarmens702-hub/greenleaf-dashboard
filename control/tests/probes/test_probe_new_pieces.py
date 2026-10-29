"""The new pieces' probes (PROTOCOL §7, rung 6; issue #15): the co-pilot learns a known correction,
and the PID-Lagrangian holds a cost at its budget, through the shipped plumbing.

Both train PPO through train.build_vec_env and train.build_model, untouched: configs/ppo.yml on cpu
with log_std_init -1.0 (the residual arms), one env, seeded, at seeds 0 and 1. The seams are
test-side, and each is a module attribute the shipped code looks up at call time:
- Residual offset: the real envs.make_env("residual"), so ResidualRBWrapper at scale 0.15 and
  SolverFailureFlag, over toys.ResidualTarget. envs.make_raw_env is patched with toys.make_raw (the
  toy instead of the greenhouse) and rule_based.rb_action with a base controller fixed at 0.30 on
  all 6 controls. rb_params is the default RB's (configs/rule_based_tuned.yml doesn't exist yet);
  the stub never asks the controller anything.
- Lagrangian: envs.make_env is patched with toys.make on toys.CostEqualsReward, as in the probe
  ladder. build_vec_env puts VecLagrangianReward under VecNormalize for residual_lag; the PIDs (at
  their default gains) and LagrangianCallback are built as train._train builds them.
The configs have no run_dir, so the only file written is the Lagrangian's lambdas.csv, in tmp_path.

Each probe's criterion is one function, shared by its trained test and its fast negative control.

Budgets: measured 2026-10-28, first at ae94411, and the numbers reproduce at the base, 777c42a,
plus this change (M3, gl-gym 2d3febb, SB3 2.3.2, torch 2.14.1; 0.75 to 0.85 s per update at
moderate load, up to ~12 s at load 7 to 10). The criterion
was checked after every 2,048-step update of one 150-update run per seed (and target); each budget
is 2x the first update from which it held at every later update, at both seeds, as a multiple of
2,048 (one rollout), and for the Lagrangian a multiple of 5 updates. A seed-0 run repeated at a
different load reproduced its lambdas.csv to the printed digits.
"""

import csv
import functools

import gymnasium as gym
import numpy as np
import pytest
import toys
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from greenleaf_control import envs, lagrangian, rule_based, train

SEEDS = [0, 1]
BASE = 0.30  # the stubbed base controller's control, on all 6
SCALE = 0.15  # residual_scale, make_env's default (PROTOCOL §3), passed explicitly
TOL = 0.03  # the issue's band on each control
BUDGETS = [0.3, 0.3, 0.3]  # one per constraint, in CONSTRAINTS order (co2, temp, rh)
COST_TOL = 0.05  # the issue's band on the mean cost
W = 30  # updates per window, several of lambda's ~6-7-update cycles
DRIFT = 0.10  # |mean of the last W - mean of the W before| < 10% of the last W's mean
LAMBDA_STAR, LAMBDA_TOL = 1.0, 0.15  # the known answer for the last W's mean
SWING = 0.5  # every lambda of the last W within LAMBDA_STAR +- 0.5; chosen after the traces
RESIDUAL_STEPS = {0.45: 65_536, 0.40: 167_936}  # 32 and 82 updates
LAGRANGIAN_STEPS = 245_760  # 120 updates, a multiple of 5 and >= 2W


# Residual offset


def _patch_residual(monkeypatch, target: float) -> None:
    """make_raw_env builds the toy; rb_action is the base controller, BASE on all 6 controls."""
    toy = functools.partial(toys.ResidualTarget, target)
    monkeypatch.setattr(envs, "make_raw_env", functools.partial(toys.make_raw, toy))
    monkeypatch.setattr(
        rule_based, "rb_action", lambda raw_env, controller: np.full(6, BASE, np.float32)
    )


def _residual_config(seed: int) -> dict:
    """What build_vec_env and build_model read for residual; no run_dir, so no tb/ on disk."""
    return {
        "method": "residual",
        "seed": seed,
        "season_length": 1,
        "penalty_weight": 1.0,
        "rb_params": rule_based.load_rb_params("default"),
        "residual_scale": SCALE,
    }


def _build_residual(monkeypatch, target: float, seed: int):
    """The real build_vec_env and build_model on the patched make_env("residual"), untrained."""
    _patch_residual(monkeypatch, target)
    venv = train.build_vec_env(_residual_config(seed), 1)
    return train.build_model(_residual_config(seed), venv), venv


def _controls(model, venv, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The 6 controls the toy gets from the deterministic policy, and the sampled policy's mean.

    Deterministic: a fresh make_env("residual") under the same patches, reset, one step with
    model.predict(deterministic=True) on the normalised observation; info["controls"] is what the
    real wrapper sent, clip(0.30 + 0.15 clip(mu, -1, 1), 0, 1). Sampled, for the record only:
    E[u] = 0.30 + 0.15 E[clip(a, -1, 1)], from 10,000 draws at the same observation.
    """
    config = _residual_config(seed)
    env = envs.make_env(
        "residual",
        seed=seed,
        season_length=1,
        rb_params=config["rb_params"],
        residual_scale=config["residual_scale"],
    )
    obs, _ = env.reset()
    norm = venv.normalize_obs(obs[None])
    action, _ = model.predict(norm, deterministic=True)
    _, _, _, _, info = env.step(action[0])
    draws, _ = model.predict(np.repeat(norm, 10_000, axis=0), deterministic=False)
    return info["controls"], BASE + SCALE * draws.mean(axis=0)


def _residual_offset(model, venv, target: float, seed: int) -> tuple[bool, str]:
    """All 6 deterministic controls within target +- 0.03."""
    u, sampled = _controls(model, venv, seed)
    sigma = model.policy.log_std.detach().exp().numpy()
    readout = (
        f"controls {np.array2string(u, precision=4)}, sampled E[u]"
        f" {np.array2string(sampled, precision=4)}, sigma {np.array2string(sigma, precision=3)}"
    )
    return bool(np.all(np.abs(u - target) <= TOL)), readout


# Lagrangian cost at budget


def _lagrangian_config(seed: int) -> dict:
    """residual_lag trains at penalty weight 0 (the Lagrangian prices the costs); no run_dir."""
    return {"method": "residual_lag", "seed": seed, "season_length": 1, "penalty_weight": 0.0}


def _build_lagrangian(monkeypatch, seed: int):
    monkeypatch.setattr(envs, "make_env", functools.partial(toys.make, toys.CostEqualsReward))
    venv = train.build_vec_env(_lagrangian_config(seed), 1)
    return train.build_model(_lagrangian_config(seed), venv), venv


def _train_lagrangian(monkeypatch, tmp_path, seed: int, steps: int) -> dict[str, np.ndarray]:
    """Trains as train._train does for residual_lag; lambdas.csv's columns, one row per update."""
    model, venv = _build_lagrangian(monkeypatch, seed)
    pids = [lagrangian.PIDLagrangian(budget) for budget in BUDGETS]
    # The VecLagrangianReward itself, which build_vec_env puts right under VecNormalize.
    callback = lagrangian.LagrangianCallback(venv.venv, pids, tmp_path / "lambdas.csv")
    model.learn(steps, callback=callback)
    with (tmp_path / "lambdas.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def _cost_at_budget(log: dict[str, np.ndarray]) -> tuple[bool, str]:
    """1. Mean cost_co2 over the last fifth of the updates in 0.3 +- 0.05.
    2. Every lambda >= 0; lambda_temp and lambda_rh exactly 0.
    3. Settled, as no drift over whole cycles (there must be 2W updates): |mean(lambda_co2[-W:]) -
       mean(lambda_co2[-2W:-W])| < 0.10 x mean(lambda_co2[-W:]).
    4. The known answer: mean(lambda_co2[-W:]) = 1.0 +- 0.15.
    5. The swing bound: every lambda_co2 of the last W within 1.0 +- 0.5, i.e. [0.5, 1.5]; 3 and 4
       alone pass a lambda alternating 0, 2 (mean 1, no drift).
    Printed, not gated: the cycle's amplitude (max - min over the last W) and the issue's literal
    std / mean of the last 10. 3 to 5 are on lambda_co2 only; the zero-cost lambdas are 0 by 2."""
    cost, lam = log["cost_co2"], log["lambda_co2"]
    tail = cost[len(cost) * 4 // 5 :]  # the last fifth when there are 5k rows
    at_budget = abs(tail.mean() - BUDGETS[0]) <= COST_TOL
    nonneg = all(np.all(log[f"lambda_{c}"] >= 0.0) for c in ("co2", "temp", "rh"))
    zero = bool(np.all(log["lambda_temp"] == 0.0) and np.all(log["lambda_rh"] == 0.0))
    last = lam[-W:]
    mean = float(last.mean())
    low, high = float(last.min()), float(last.max())
    within = bool(np.all(np.abs(last - LAMBDA_STAR) <= SWING))
    std10 = float(np.std(lam[-10:]) / np.mean(lam[-10:])) if np.mean(lam[-10:]) > 0 else np.inf
    if len(lam) >= 2 * W:
        drift = abs(mean - float(lam[-2 * W : -W].mean()))
        settled = drift < DRIFT * mean  # False at mean 0: drift >= 0
        known = abs(mean - LAMBDA_STAR) <= LAMBDA_TOL
        share = f"{drift / mean:.4f} of the mean" if mean > 0 else "the mean is 0"
        drift_text = f"{drift:.4f} ({share})"
    else:
        settled = known = False
        drift_text = f"n/a (fewer than {2 * W} updates)"
    readout = (
        f"{len(cost)} updates; mean cost_co2 over the last {len(tail)} = {tail.mean():.4f}"
        f" (first {cost[0]:.4f}, last {cost[-1]:.4f}); lambda_co2 over the last {len(last)}:"
        f" mean {mean:.4f}, drift {drift_text}, range {low:.4f} to {high:.4f} (amplitude"
        f" {high - low:.4f}; all within {LAMBDA_STAR} +- {SWING}: {within}); std/mean of the last"
        f" 10 {std10:.4f} (not gated); min lambda"
        f" {min(log[f'lambda_{c}'].min() for c in ('co2', 'temp', 'rh')):.4f};"
        f" lambda_temp, lambda_rh all 0: {zero}"
    )
    return bool(at_budget and nonneg and zero and settled and known and within), readout


def _synthetic_log(lam: np.ndarray) -> dict[str, np.ndarray]:
    """A lambdas.csv as _train_lagrangian returns it, for this lambda_co2 trace: cost_co2 0.3 at
    every update (the cost criterion holds), the zero-cost constraints at 0."""
    n = len(lam)
    return {
        "timesteps": 2_048.0 * np.arange(1, n + 1),
        "lambda_co2": np.asarray(lam, dtype=np.float64),
        "lambda_temp": np.zeros(n),
        "lambda_rh": np.zeros(n),
        "cost_co2": np.full(n, 0.3),
        "cost_temp": np.zeros(n),
        "cost_rh": np.zeros(n),
    }


_K = np.arange(120)  # LAGRANGIAN_STEPS' 120 updates
SYNTHETIC = {  # lambda_co2 over 120 updates, and whether the criterion must hold
    "flat_at_1": (np.full(120, 1.0), True),  # mean 1.0, drift 0, all within [0.5, 1.5]
    # Like the measured cycle: 0.96 +- 0.24, period 6.5, so 0.72 to 1.20, window means ~0.96
    "measured_cycle": (0.96 + 0.24 * np.sin(2 * np.pi * _K / 6.5), True),
    # Last 30 mean 0.5 + 0.8 x 104.5 / 119 = 1.2025, the 30 before 1.0008: drift 0.2017 > 0.12
    # and |1.2025 - 1| > 0.15
    "climbing_0.5_to_1.3": (np.linspace(0.5, 1.3, 120), False),
    "at_the_cap_100": (np.full(120, 100.0), False),  # mean 100, not 1.0 +- 0.15; outside the swing
    "at_0": (np.zeros(120), False),  # drift 0 < 0.1 x 0 is False; mean 0; and no division by 0
    "flat_at_1.3": (np.full(120, 1.3), False),  # drift 0 and swing ok, but |1.3 - 1| = 0.3 > 0.15
    "alternating_0_2": (np.tile([0.0, 2.0], 60), False),  # mean 1.0, drift 0, but 0 and 2 swing 1
}


# Tests


def test_seams_build_the_real_stacks_on_the_toys(monkeypatch):
    """Residual: VecNormalize -> VecMonitor -> DummyVecEnv -> SolverFailureFlag ->
    ResidualRBWrapper (scale 0.15) -> FlattenObservation -> ResidualTarget, with the stub's 0.30
    on all 6 controls appended to the toy's x = 0: observation (7,), action (6,). Lagrangian:
    VecNormalize -> VecLagrangianReward -> VecMonitor -> DummyVecEnv -> CostEqualsReward. Both
    policies start at log_std -1.0. If make_env or the wrapper bound make_raw_env or rb_action at
    import time, the residual chain would be the greenhouse's (and the RB's controls)."""
    model, venv = _build_residual(monkeypatch, 0.40, 0)
    assert [type(venv), type(venv.venv), type(venv.venv.venv)] == [
        VecNormalize,
        VecMonitor,
        DummyVecEnv,
    ]
    [env] = venv.venv.venv.envs
    assert isinstance(env, envs.SolverFailureFlag)
    assert isinstance(env.env, envs.ResidualRBWrapper)
    assert env.env.scale == SCALE
    assert isinstance(env.env.env, gym.wrappers.FlattenObservation)
    assert isinstance(env.unwrapped, toys.ResidualTarget)
    assert env.unwrapped.target == 0.40
    assert venv.observation_space.shape == (7,)
    assert venv.action_space.shape == (6,)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs, np.array([0.0] + [BASE] * 6, dtype=np.float32))
    np.testing.assert_array_equal(model.policy.log_std.detach().numpy(), np.full(6, -1.0))
    venv.close()

    model, venv = _build_lagrangian(monkeypatch, 0)
    assert [type(venv), type(venv.venv), type(venv.venv.venv), type(venv.venv.venv.venv)] == [
        VecNormalize,
        lagrangian.VecLagrangianReward,
        VecMonitor,
        DummyVecEnv,
    ]
    [env] = venv.venv.venv.venv.envs
    assert isinstance(env.unwrapped, toys.CostEqualsReward)
    np.testing.assert_array_equal(model.policy.log_std.detach().numpy(), np.full(1, -1.0))
    venv.close()


@pytest.mark.parametrize("target", list(RESIDUAL_STEPS))
def test_residual_probe_fails_untrained(monkeypatch, target):
    """Negative control: untrained at seed 0, the policy's mean is ~0, so every control is ~0.300,
    0.10 or 0.15 from the target: the criterion fails for both targets."""
    model, venv = _build_residual(monkeypatch, target, 0)
    ok, readout = _residual_offset(model, venv, target, 0)
    print(f"untrained, target {target}: {readout}")
    assert not ok, readout
    np.testing.assert_allclose(_controls(model, venv, 0)[0], BASE, atol=0.005)


def test_lagrangian_plumbing_one_update(monkeypatch, tmp_path):
    """Plumbing and negative control: one 2,048-step rollout at seed 0 logs one lambdas.csv row.

    Untrained, a ~ N(0, 0.37), so u = (a + 1) / 2 averages 0.5 (+- 0.004 over 2,048 steps): the
    logged cost_co2 is 0.5 +- 0.02, outside 0.3 +- 0.05, so the criterion fails (as do the drift
    and known-answer checks, which need 2W = 60 updates). lambda_co2 is the
    PID's first update on that cost, e.g. at 0.5: e = (0.5 - 0.3) / 0.3 = 0.667, I = 0.05 x 0.667
    = 0.0333, lambda = 0.0333 + 0.5 x 0.667 + 0.1 x 0.667 = 0.4333. cost_temp = cost_rh = 0 give
    e = -1, I = max(0, 0 - 0.05) = 0, D = max(0, -1 - 0) = 0, lambda = max(0, 0 - 0.5) = 0.
    """
    log = _train_lagrangian(monkeypatch, tmp_path, 0, 2_048)
    ok, readout = _cost_at_budget(log)
    print(f"one update: {readout}")
    assert len(log["cost_co2"]) == 1
    [cost] = log["cost_co2"]
    assert abs(cost - 0.5) <= 0.02, readout
    assert abs(log["lambda_co2"][0] - lagrangian.PIDLagrangian(0.3).update(cost)) <= 1e-9
    assert log["lambda_temp"][0] == 0.0 and log["lambda_rh"][0] == 0.0
    assert log["cost_temp"][0] == 0.0 and log["cost_rh"][0] == 0.0
    assert not ok, readout


@pytest.mark.parametrize(("lam", "holds"), list(SYNTHETIC.values()), ids=list(SYNTHETIC))
def test_lagrangian_criterion_on_synthetic_logs(lam, holds):
    """Negative control for the Lagrangian criterion itself, on 120-update logs with cost_co2 0.3
    and the zero-cost lambdas at 0, so only lambda_co2 decides (each case's arithmetic is by
    SYNTHETIC). It must hold for lambda flat at 1.0 and for a cycle like the measured one (0.72 to
    1.20, period ~6.5); it must fail, without raising, for lambda climbing 0.5 to 1.3 (drift), at
    the cap of 100 and at 0 (known answer), flat at 1.3 (known answer) and alternating 0 / 2 (the
    swing, which neither drift nor the mean can see)."""
    ok, readout = _cost_at_budget(_synthetic_log(lam))
    print(readout)
    assert ok == holds, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("target", list(RESIDUAL_STEPS))
def test_residual_offset_is_learned(monkeypatch, target, seed):
    """Residual offset: base controller 0.30, scale 0.15, reward -sum_i (u_i - target)^2 on 6
    controls, one step. Known answer: a* = (target - 0.30) / 0.15 on every control, so the
    deterministic controls end at target +- 0.03.

    - Target 0.45 (the issue's): a* = 1.0 sits on SB3's action bound. model.predict clips the mean
      to 1, so the deterministic control caps at 0.45 and the band is one-sided: it holds iff the
      mean is >= 0.8 on every control. It can't tell "learned 0.45" from "pushed to the maximum".
    - Target 0.40: a* = 2/3 is interior, so this one locates the offset: all 6 controls in
      0.40 +- 0.03 iff the mean is within +- 0.2 of 2/3 on each. But 2/3 is the deterministic
      optimum. PPO optimises the sampled policy, whose actions SB3 clips to [-1, 1], and at the
      start sigma = 0.37 puts the clip only 0.9 sigma above 2/3: the sampled loss is then least at a
      mean of ~0.905 (Monte Carlo), a deterministic control of 0.436, out of the band. As PPO
      shrinks sigma, that optimum and the control fall back towards 2/3 and 0.40.

    Measured after every update, 150 updates, seeds 0 and 1 (at the budgets, the sampled reading,
    printed, is ~0.002 below the deterministic one at 0.45 and 0.006 to 0.008 below at 0.40):
    - 0.45: in the band from update 16 at both seeds, and exactly 0.4500 from update 20.
    - 0.40: in the band at updates 11 to 24; out at 25 to 40 (seed 0) and 29 to 31 (seed 1), up
      to 0.4319, as the mean peaks at ~0.85; in the band at every update from 41 (seed 0) and 32
      (seed 1) to 150, while sigma falls from 0.367 to 0.268 and the controls to 0.411 to 0.415.
      So it locates the offset within the band, biased towards the bound by +0.01 to +0.02.

    Budgets: 0.45, 65,536 steps (32 updates, 2 x 16); 0.40, 167,936 steps (82 updates, 2 x 41),
    where the largest control is 0.4240 (seed 0) and 0.4239 (seed 1), 0.006 inside the band.
    Untrained: 0.300 on all 6.
    """
    model, venv = _build_residual(monkeypatch, target, seed)
    model.learn(RESIDUAL_STEPS[target])
    ok, readout = _residual_offset(model, venv, target, seed)
    print(f"target {target}, seed {seed}: {readout}")
    assert ok, readout


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_lagrangian_holds_cost_at_budget(monkeypatch, tmp_path, seed):
    """Lagrangian cost at budget: reward = cost_co2 = u = (a + 1) / 2, budget 0.3 on each of the 3
    constraints, VecLagrangianReward and LagrangianCallback at the PID's default gains (kp 0.5,
    ki 0.05, kd 0.1, lambda_max 100). Known answer: u* = 0.3 and lambda_co2 settles near 1.0,
    where the shaped reward u (1 - lambda) / (1 + lambda) is flat in u; lambda_temp and lambda_rh
    are exactly 0 at every update (zero cost under budget 0.3: e = -1, I = 0, lambda = 0).

    Read from lambdas.csv, with W = 30 updates (_cost_at_budget):
    (1) mean cost_co2 over the last fifth of the updates in 0.3 +- 0.05;
    (2) lambda >= 0 throughout, and lambda_temp = lambda_rh = 0.0;
    (3) no drift over whole cycles: |mean of lambda_co2's last W - mean of the W before| < 10% of
        the last W's mean;
    (4) the known answer: that mean = 1.0 +- 0.15;
    (5) the swing bound: every lambda_co2 of the last W within 1.0 +- 0.5, i.e. [0.5, 1.5].

    Changed criterion. The issue's literal "settled", std of lambda_co2's last 10 updates < 10% of
    their mean, failed at seed 1 (0.1052 at 50 updates), because lambda settles into a limit cycle
    around the right value rather than onto it: from update 25 it cycles between 0.72 and 1.20
    with a period of ~6 to 7 updates, around a mean of 0.99 to 1.00, and the cost between 0.16
    and 0.43 around 0.30. The cycle doesn't shrink by update 150, and that std/mean crosses 0.1
    back and forth (0.08 to 0.15), so whether it is under 0.1 at a given budget is the cycle's
    phase, not a pass; it is printed, not gated. On 2026-10-28 the orchestrator took option (a)
    from the #15 comment under Aarmen's standing instruction to complete everything that costs no
    money; Aarmen didn't pick it himself. The issue's literal criterion is replaced by (3) drift,
    (4) the known-answer mean and (5) the swing bound; /ship shows him this swap explicitly. The
    swing's +- 0.5 was chosen by the orchestrator after seeing these traces: a value for Aarmen to
    confirm.

    Measured on 150-update runs (307,200 steps) at seeds 0 and 1, checked at every update count:
    (1) holds from 25 on (last-fifth means 0.286 to 0.346) and (5) from 33; (3) and (4) hold at
    every count from 60 (the first with 2W updates) to 150, with drift at most 0.072 (seed 0) and
    0.089 (seed 1) of the mean and the last W's mean 0.970 to 1.025; from 60 on, every last-W
    window lies within 0.717 to 1.197.

    Budget 245,760 steps (120 updates): 2 x 60. At 120, reproduced at 777c42a: seed 0, mean
    1.0018, drift 0.0158 of the mean, lambda 0.800 to 1.171, cost 0.3085; seed 1, mean 1.0011,
    drift 0.0093, lambda 0.834 to 1.165, cost 0.2926. Untrained (one update): cost 0.5037,
    lambda 0.4412.
    """
    log = _train_lagrangian(monkeypatch, tmp_path, seed, LAGRANGIAN_STEPS)
    ok, readout = _cost_at_budget(log)
    print(f"seed {seed}: {readout}")
    assert ok, readout

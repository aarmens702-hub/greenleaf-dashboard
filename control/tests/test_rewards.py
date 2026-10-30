"""Tests for greenleaf_control.rewards (issue #7, core: a scorer file).

ConstraintCostReward keeps upstream GreenhouseReward's money terms, scaling and summation, and
changes only two things: each scaled constraint violation is reported under its true name
(cost_co2, cost_temp, cost_rh), and the penalty is weighted, so
reward = scaled_profit - penalty_weight * (cost_co2 + cost_temp + cost_rh).

Upstream (gl_gym @2d3febb, gl_gym/components/rewards.py) is untrusted. Two of its bugs are pinned
at the bottom of this module on upstream's own class, so a pin bump that fixes either is noticed:
info["temp_penalty"] holds the CO2 violation and info["co2_penalty"] the temperature one
(:218-219), and the lamp penalty never fires (:188-192).

Most tests build a RewardContext directly, as the env's step() does (greenlight_env.py:262-276).
Constraint bounds are the registered env's: CO2 300-1600 ppm, temperature 15-34 C, RH 50-85 %.
Each violation is divided by upstream's normaliser: CO2 2500 ppm, temperature 15 C, RH 15 %.
"""

import inspect

import gl_gym  # noqa: F401  (importing gl_gym registers its env ids with gymnasium)
import gymnasium as gym
import numpy as np
import pytest
from gl_gym.components.rewards import GreenhouseReward
from gl_gym.configs.default_params import init_default_params
from gl_gym.core.types import RewardContext

# greenleaf_control.rewards is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

ENV_ID = "gl_gym/GreenLightTomato-v0"
DT = 900  # [s] one control step
# Upstream's registered reward kwargs (gl_gym/__init__.py:33-40): elec 0.3 EUR/kWh, heating
# 0.09 EUR/kWh, CO2 0.3 EUR/kg, fruit 1.6 EUR/kg, dmfm 0.065, pen_lamp 0.1.
UPSTREAM_KW = dict(gym.spec(ENV_ID).kwargs["reward_kwargs"])
# Default parameters as the env builds them (greenlight_env.py:121); they come from float32.
P = np.asarray(init_default_params(208), dtype=np.float64)
KW = {**UPSTREAM_KW, "dt": DT, "p": P}
LOW = np.array([300.0, 15.0, 50.0])  # co2 [ppm], temp [C], rh [%]
HIGH = np.array([1600.0, 34.0, 85.0])
MONEY_KEYS = ("profit", "revenue", "heat_cost", "elec_cost", "co2_cost", "fruit_growth_ffw")
COST_KEYS = ("cost_co2", "cost_temp", "cost_rh")

# u is (uBoil, uCO2, uThScr, uVent, uLamp, uBlScr) in [0, 1].
FULL_COST_U = (1.0, 1.0, 0.0, 0.0, 1.0, 0.0)  # boiler, CO2 dosing and lamps all at full
# p[154] = 0.328 mg/m2/s max fruit growth x 900 s = 295.2 mg DM/m2: the max-profit step.
MAX_GROWTH = 295.2


def _ctx(co2, temp, rh, *, pipe=40.0, u=(0.0,) * 6, hour=12.0, growth=0.0):
    """A RewardContext at indoor (co2 ppm, temp C, rh %), with x[25] - x_prev[25] = growth."""
    x_prev = np.zeros(28)
    x = np.zeros(28)
    x[25] = growth  # x[25] is fruit dry matter [mg/m2]
    return RewardContext(
        t=0,
        dt=DT,
        Np=0,
        x_prev=x_prev,
        x=x,
        u=np.asarray(u, dtype=np.float64),
        p=P,
        d=np.zeros((1, 10)),
        hour_of_day=hour,
        day_of_year=59.0,
        obs={"IndoorClimateObservations": np.array([co2, temp, rh, pipe], dtype=np.float32)},
        constraints_low=LOW,
        constraints_high=HIGH,
    )


def _upstream(ctx):
    return GreenhouseReward(**KW).compute_reward(ctx)


# Contexts for the parity checks: violating, controls on, fruit growth, lamps at night, all at once.
STATES = {
    "ok": dict(co2=800.0, temp=22.0, rh=70.0),
    "hot": dict(co2=800.0, temp=36.0, rh=70.0),
    "co2_high": dict(co2=1850.0, temp=22.0, rh=70.0),
    "humid": dict(co2=800.0, temp=22.0, rh=95.0),
    "cold_lowco2_dry": dict(co2=250.0, temp=13.0, rh=40.0, pipe=15.0),
    "controls_on": dict(co2=800.0, temp=22.0, rh=70.0, u=(0.3, 0.5, 0.2, 0.7, 1.0, 0.1)),
    "growth": dict(co2=800.0, temp=22.0, rh=70.0, growth=150.0),
    "night_lamp": dict(co2=800.0, temp=22.0, rh=70.0, u=(0, 0, 0, 0, 1.0, 0), hour=21.0),
    "everything": dict(
        co2=1850.0, temp=36.0, rh=95.0, pipe=60.0, u=(1.0,) * 6, hour=22.0, growth=MAX_GROWTH
    ),
    # costs 100/2500 = 0.04, 3/15 = 0.2, 3/15 = 0.2, whose float sum depends on the order of
    # addition: (0.04 + 0.2) + 0.2 = 0.44000000000000006, but 0.04 + (0.2 + 0.2) = 0.44.
    "order_sensitive": dict(co2=1700.0, temp=37.0, rh=88.0),
}
VIOLATING = ["hot", "co2_high", "humid", "cold_lowco2_dry", "everything", "order_sensitive"]


def test_constraints_are_co2_temp_rh_in_that_order():
    import greenleaf_control.rewards as rewards

    assert rewards.CONSTRAINTS == ("co2", "temp", "rh")


def test_reward_subclasses_upstream_and_takes_a_keyword_only_weight_defaulting_to_one():
    import greenleaf_control.rewards as rewards

    assert issubclass(rewards.ConstraintCostReward, GreenhouseReward)
    param = inspect.signature(rewards.ConstraintCostReward.__init__).parameters["penalty_weight"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default == 1.0


@pytest.mark.parametrize(
    ("obs", "expected"),
    [
        # 36 C is 2 C over the 34 C max; / 15 C = 2/15
        ((800.0, 36.0, 70.0), {"cost_co2": 0.0, "cost_temp": 2 / 15, "cost_rh": 0.0}),
        # 13 C is 2 C under the 15 C min; / 15 C = 2/15
        ((800.0, 13.0, 70.0), {"cost_co2": 0.0, "cost_temp": 2 / 15, "cost_rh": 0.0}),
        # 1850 ppm is 250 ppm over the 1600 ppm max; / 2500 ppm = 0.1
        ((1850.0, 22.0, 70.0), {"cost_co2": 250 / 2500, "cost_temp": 0.0, "cost_rh": 0.0}),
        # 250 ppm is 50 ppm under the 300 ppm min; / 2500 ppm = 0.02
        ((250.0, 22.0, 70.0), {"cost_co2": 50 / 2500, "cost_temp": 0.0, "cost_rh": 0.0}),
        # 95 % is 10 % over the 85 % max; / 15 % = 2/3
        ((800.0, 22.0, 95.0), {"cost_co2": 0.0, "cost_temp": 0.0, "cost_rh": 10 / 15}),
        # 40 % is 10 % under the 50 % min; / 15 % = 2/3
        ((800.0, 22.0, 40.0), {"cost_co2": 0.0, "cost_temp": 0.0, "cost_rh": 10 / 15}),
        # all three at once: 250/2500 + 2/15 + 10/15, each under its own name
        ((1850.0, 36.0, 95.0), {"cost_co2": 250 / 2500, "cost_temp": 2 / 15, "cost_rh": 10 / 15}),
    ],
    ids=["temp_36", "temp_13", "co2_1850", "co2_250", "rh_95", "rh_40", "all_three"],
)
def test_each_violation_is_costed_under_its_own_name(obs, expected):
    import greenleaf_control.rewards as rewards

    _, info = rewards.ConstraintCostReward(**KW).compute_reward(_ctx(*obs))
    assert {k: info[k] for k in COST_KEYS} == expected  # exact: these divisions are exact


@pytest.mark.parametrize(
    "obs",
    [(800.0, 22.0, 70.0), (300.0, 15.0, 50.0), (1600.0, 34.0, 85.0)],
    ids=["mid_band", "at_every_min", "at_every_max"],
)
def test_no_cost_at_a_non_violating_state(obs):
    import greenleaf_control.rewards as rewards

    _, info = rewards.ConstraintCostReward(**KW).compute_reward(_ctx(*obs))
    assert {k: info[k] for k in COST_KEYS} == {"cost_co2": 0.0, "cost_temp": 0.0, "cost_rh": 0.0}


def test_upstream_swapped_penalty_keys_are_not_in_info():
    import greenleaf_control.rewards as rewards

    _, info = rewards.ConstraintCostReward(**KW).compute_reward(_ctx(1850.0, 36.0, 95.0))
    assert not {"temp_penalty", "co2_penalty", "rh_penalty"} & info.keys()


@pytest.mark.parametrize("state", VIOLATING)
def test_penalty_weight_zero_gives_scaled_profit(state):
    import greenleaf_control.rewards as rewards

    reward, info = rewards.ConstraintCostReward(penalty_weight=0.0, **KW).compute_reward(
        _ctx(**STATES[state])
    )
    assert sum(info[k] for k in COST_KEYS) > 0  # the state violates, so w=0 is what removes it
    assert reward == info["scaled_profit"]


@pytest.mark.parametrize("weight", [0.5, 2.0, 10.0])
def test_penalty_weight_scales_the_summed_costs(weight):
    import greenleaf_control.rewards as rewards

    # costs 250/2500 + 2/15 + 10/15 = 0.9 at (1850 ppm, 36 C, 95 %)
    reward, info = rewards.ConstraintCostReward(penalty_weight=weight, **KW).compute_reward(
        _ctx(1850.0, 36.0, 95.0)
    )
    assert reward == pytest.approx(info["scaled_profit"] - weight * 0.9, rel=1e-12)


@pytest.mark.parametrize("state", list(STATES))
@pytest.mark.parametrize("weight_kwargs", [{"penalty_weight": 1.0}, {}], ids=["w1", "default"])
def test_weight_one_matches_upstream_bit_for_bit(state, weight_kwargs):
    import greenleaf_control.rewards as rewards

    ctx = _ctx(**STATES[state])
    up_reward, up_info = _upstream(ctx)
    reward, info = rewards.ConstraintCostReward(**weight_kwargs, **KW).compute_reward(ctx)
    assert reward == up_reward  # ==, not approx: same terms, same summation order
    assert info["paper_reward"] == up_reward
    # The same scaled violations, under their true names (upstream's labels are swapped).
    assert info["cost_co2"] == up_info["temp_penalty"]
    assert info["cost_temp"] == up_info["co2_penalty"]
    assert info["cost_rh"] == up_info["rh_penalty"]


def test_paper_reward_is_upstreams_and_info_reward_is_ours_at_any_weight():
    import greenleaf_control.rewards as rewards

    ctx = _ctx(1850.0, 36.0, 95.0)
    up_reward, _ = _upstream(ctx)
    reward, info = rewards.ConstraintCostReward(penalty_weight=0.5, **KW).compute_reward(ctx)
    assert info["paper_reward"] == up_reward
    # costs sum to 0.9, so w=0.5 puts ours 0.45 above upstream's (w=1)
    assert reward == pytest.approx(up_reward + 0.45, rel=1e-12)
    assert info["reward"] == reward


@pytest.mark.parametrize(
    ("u", "growth", "expected"),
    [
        # profit 0 EUR/m2. min_profit = -(0.002925 heat + 0.0087 elec + 0.00135 CO2) = -0.012975;
        # max_profit = 295.2 mg/m2 x 1e-6 / 0.065 x 1.6 EUR/kg = 0.0072665 EUR/m2;
        # (0 + 0.012975) / (0.0072665 + 0.012975) = 0.641011
        ((0.0,) * 6, 0.0, 0.6410110),
        # boiler, CO2 and lamps at full, no growth: profit = min_profit, so 0
        (FULL_COST_U, 0.0, 0.0),
        # no controls, max growth: profit = max_profit, so 1
        ((0.0,) * 6, MAX_GROWTH, 1.0),
        # full costs and max growth: profit = max - |min|; scaled = 0.0072665 / 0.0202415 = 0.358989
        (FULL_COST_U, MAX_GROWTH, 0.3589890),
    ],
    ids=["zero", "full_cost", "max_growth", "full_cost_max_growth"],
)
def test_scaled_profit_matches_hand_values(u, growth, expected):
    import greenleaf_control.rewards as rewards

    _, info = rewards.ConstraintCostReward(**KW).compute_reward(
        _ctx(800.0, 22.0, 70.0, u=u, growth=growth)
    )
    # rel 1e-6: p[154] = 0.328 is stored as float32 (0.328000009)
    assert info["scaled_profit"] == pytest.approx(expected, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        # 18720 W / 144 m2 = 130 W/m2 x 0.25 h = 0.0325 kWh/m2 x 0.09 EUR/kWh
        ("heat_cost", 0.002925),
        # 116 W/m2 x 0.25 h = 0.029 kWh/m2 x 0.3 EUR/kWh
        ("elec_cost", 0.0087),
        # 720 mg/s / 144 m2 = 5 mg/m2/s x 900 s = 0.0045 kg/m2 x 0.3 EUR/kg
        ("co2_cost", 0.00135),
        # 295.2 mg DM/m2 x 1e-6 / 0.065 DM per FM = 0.0045415 kg FFW/m2
        ("fruit_growth_ffw", 0.0045415385),
        # 0.0045415 kg/m2 x 1.6 EUR/kg
        ("revenue", 0.0072664615),
        # 0.0072665 revenue - (0.002925 + 0.0087 + 0.00135) costs
        ("profit", -0.0057085385),
    ],
)
def test_money_keys_match_hand_values_and_upstream(key, expected):
    import greenleaf_control.rewards as rewards

    ctx = _ctx(800.0, 22.0, 70.0, u=FULL_COST_U, growth=MAX_GROWTH)
    _, up_info = _upstream(ctx)
    _, info = rewards.ConstraintCostReward(**KW).compute_reward(ctx)
    assert info[key] == pytest.approx(expected, rel=1e-6)
    assert info[key] == up_info[key]


@pytest.mark.parametrize("state", ["ok", "everything"])
def test_reward_costs_scaled_profit_and_paper_reward_are_python_floats(state):
    import greenleaf_control.rewards as rewards

    reward, info = rewards.ConstraintCostReward(**KW).compute_reward(_ctx(**STATES[state]))
    values = {
        "reward": reward,
        **{k: info[k] for k in (*COST_KEYS, "scaled_profit", "paper_reward")},
    }
    assert {k: type(v) for k, v in values.items()} == {k: float for k in values}


def test_constraint_costs_returns_float64_in_constraints_order():
    import greenleaf_control.rewards as rewards

    info = {"profit": 0.5, "cost_rh": 0.3, "cost_temp": 0.2, "cost_co2": 0.1}
    costs = rewards.constraint_costs(info)
    assert isinstance(costs, np.ndarray)
    assert costs.dtype == np.float64
    assert costs.shape == (3,)
    assert costs.tolist() == [0.1, 0.2, 0.3]


def test_constraint_costs_of_a_real_step():
    import greenleaf_control.rewards as rewards

    _, info = rewards.ConstraintCostReward(**KW).compute_reward(_ctx(1850.0, 36.0, 95.0))
    # (1850 - 1600) / 2500, (36 - 34) / 15, (95 - 85) / 15
    assert rewards.constraint_costs(info).tolist() == [250 / 2500, 2 / 15, 10 / 15]


@pytest.mark.parametrize("missing", COST_KEYS)
def test_constraint_costs_raises_keyerror_on_a_missing_key(missing):
    import greenleaf_control.rewards as rewards

    info = {k: 0.1 for k in COST_KEYS if k != missing}
    with pytest.raises(KeyError):
        rewards.constraint_costs(info)


def test_one_day_episode_matches_upstream_env_step_for_step():
    """A one-day PAPER-weather (Amsterdam 2010, day 59) episode with a zero action, run on two envs.

    Ours (w=1) and upstream's reward see identical trajectories, so every step's reward must be
    bit-identical, paper_reward must be upstream's, and each cost_* must be upstream's swapped label.
    """
    import greenleaf_control.rewards as rewards

    ours = gym.make(
        ENV_ID,
        season_length=1,
        reward_fn=rewards.ConstraintCostReward,
        reward_kwargs={**UPSTREAM_KW, "penalty_weight": 1.0},
    )
    theirs = gym.make(ENV_ID, season_length=1)
    ours.reset(seed=0)
    theirs.reset(seed=0)
    action = np.zeros(ours.action_space.shape, dtype=ours.action_space.dtype)
    violating_steps = 0
    for _ in range(ours.unwrapped.N):  # 96 steps of 900 s
        obs, reward, _, _, info = ours.step(action)
        up_obs, up_reward, _, _, up_info = theirs.step(action)
        for key in obs:
            np.testing.assert_array_equal(obs[key], up_obs[key])
        assert reward == up_reward
        assert info["paper_reward"] == up_reward
        assert (info["cost_co2"], info["cost_temp"], info["cost_rh"]) == (
            up_info["temp_penalty"],
            up_info["co2_penalty"],
            up_info["rh_penalty"],
        )
        violating_steps += sum(info[k] for k in COST_KEYS) > 0
    ours.close()
    theirs.close()
    assert violating_steps > 0  # otherwise the cost comparison above checked only zeros


# Upstream bug pins: these use only upstream's class and pass today. A failure here means the
# pinned simulator changed, and the costs and paper_reward above must be re-derived.


@pytest.mark.parametrize(
    ("obs", "swapped_key", "expected"),
    [
        # 36 C: (36 - 34) / 15 lands in upstream's "co2_penalty"
        ((800.0, 36.0, 70.0), "co2_penalty", 2 / 15),
        # 1850 ppm: (1850 - 1600) / 2500 lands in upstream's "temp_penalty"
        ((1850.0, 22.0, 70.0), "temp_penalty", 250 / 2500),
    ],
    ids=["temp_in_co2_penalty", "co2_in_temp_penalty"],
)
def test_upstream_bug_co2_and_temp_penalty_labels_are_swapped(obs, swapped_key, expected):
    _, up_info = _upstream(_ctx(*obs))
    assert up_info[swapped_key] == expected
    other = ({"co2_penalty", "temp_penalty"} - {swapped_key}).pop()
    assert up_info[other] == 0.0


def test_upstream_bug_lamp_penalty_never_fires():
    up = GreenhouseReward(**KW)
    # Lamps fully on at 21:00, after upstream's 20:00 cutoff, with pen_lamp = 0.1.
    reward, up_info = up.compute_reward(_ctx(800.0, 22.0, 70.0, u=(0, 0, 0, 0, 1.0, 0), hour=21.0))
    assert up.lamp_violation == 1  # the night-lamp branch did fire ...
    assert up_info["lamp_penalty"] == 0.0  # ... but the penalty it returns is a local 0
    assert up_info["control_penalty"] == 0.0

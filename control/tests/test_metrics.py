"""Tests for greenleaf_control.metrics (issue #10, core: a scorer file, the independent grader).

metrics.episode_metrics grades an episode from its raw states (T + 1, 28) and controls (T, 6) alone,
never from the reward code, so a reward bug can't hide inside the score. trace_from_infos(infos)
collects those from envs' per-step infos: states = [infos[0]["x_prev"], *each info's "x"], controls =
each info's "controls". The cross-checks against upstream's own per-step info (gl_gym @2d3febb) are
a check, not shared code: the AST and sys.modules tests keep metrics.py off rewards, envs and gl_gym.

Known answers used throughout (upstream file:line, under gl_gym/):
- Bounds (__init__.py:24-31): CO2 300-1600 ppm, temperature 15-34 C, RH 50-85 %. A value on a bound
  is in bounds: upstream's violation is max(0, low - v) + max(0, v - high) (rewards.py:173-180).
- One step is dt = 900 s, so a violating step is 0.25 h; pct_in_bounds = 100 x (1 - violating / T).
- Step t is graded on states[t + 1], the state after it (greenlight_env.py:238-278); states[0], the
  initial state, is not graded. The indoor climate is x[0] CO2 [mg/m3] -> ppm by the ideal gas law,
  x[2] air temperature [C], and x[15] vapour pressure [Pa] -> RH (observations.py:53-55,
  environments/utils.py:336-362).
- Parameters (configs/default_params.py:61-217): aFlr p[46] = 144 m2, pBoil p[108] = 130 x 144 =
  18,720 W, phiExtCo2 p[109] = 5 x 144 = 720 mg/s, rgFruit p[154] = 0.328 mg/m2/s (stored as float32,
  0.328000009), thetaLampMax p[172] = 116 W/m2.
- Prices (__init__.py:33-40): fruit 1.6 EUR/kg, dmfm 0.065, heating 0.09 EUR/kWh, electricity
  0.3 EUR/kWh, CO2 0.3 EUR/kg. Costs are per step (rewards.py:134-139); yield is the fresh-weight
  change in fruit dry matter x[25] (rewards.py:155-158).
- paper_return sums upstream's reward: (profit - min_profit) / (max_profit - min_profit) minus each
  violation over its normaliser, 2500 ppm, 15 C, 15 % (rewards.py:79-110, 208-210). The lamp
  penalty never fires (an upstream bug), so it adds nothing.
- A swallowed solver failure leaves x unchanged (greenlight_env.py:243-250), so solver_failures
  counts the steps with states[t + 1] == states[t]. Every real step moves x[27], time in days.

Upstream grades violations on float32 observations and metrics on float64 states. The violation
counts must still agree exactly with upstream's; a split at a bound would be a finding, not a
tolerance to widen.
"""

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import gl_gym  # noqa: F401  (importing gl_gym registers its env ids with gymnasium)
import gymnasium as gym
import numpy as np
import pytest
from gl_gym.components.rewards import GreenhouseReward
from gl_gym.configs.default_params import init_default_params

import greenleaf_control
from greenleaf_control import envs, scenarios

# greenleaf_control.metrics is imported inside each test body (or a helper it calls), so while it
# is missing each test FAILS rather than ERRORs at collection.

ENV_ID = "gl_gym/GreenLightTomato-v0"
DT = 900  # [s] one control step
N_DAY = 96  # 86,400 s / 900 s
N_SEASON = 5_760  # 60 days x 96
NX = 28  # upstream's state vector x
CONSTRAINTS = ("co2", "temp", "rh")
UPSTREAM_KW = dict(gym.spec(ENV_ID).kwargs["reward_kwargs"])
UPSTREAM_BOUNDS = dict(gym.spec(ENV_ID).kwargs["constraints"])
# Default parameters as the env builds them (greenlight_env.py:121); they come from float32.
P = np.asarray(init_default_params(208), dtype=np.float64)
KEYS = {
    "profit_eur_m2",
    "yield_kg_m2",
    "heat_cost",
    "elec_cost",
    "co2_cost",
    "paper_return",
    "solver_failures",
    "steps",
    "valid",
    *(f"viol_hours_{c}" for c in CONSTRAINTS),
    *(f"pct_in_bounds_{c}" for c in CONSTRAINTS),
}
FORBIDDEN = ("greenleaf_control.rewards", "greenleaf_control.envs", "gl_gym.components.rewards")

# Indoor climates (CO2 ppm, temperature C, RH %).
IN_BAND = (800.0, 22.0, 70.0)
HOT = (800.0, 36.0, 70.0)  # 36 C is 2 C over the 34 C max
# u is (uBoil, uCO2, uThScr, uVent, uLamp, uBlScr) in [0, 1].
ZERO_U = (0.0,) * 6
FULL_COST_U = (1.0, 1.0, 0.0, 0.0, 1.0, 0.0)  # boiler, CO2 dosing and lamps at full
# p[154] = 0.328 mg/m2/s max fruit growth x 900 s = 295.2 mg DM/m2: the max-profit step.
MAX_GROWTH = 295.2


def _state(ppm: float, temp: float, rh: float, *, day: float = 0.0, fruit: float = 0.0):
    """A raw state x (28,) at indoor (ppm, temp, rh), by inverting upstream's conversions.

    x[0] = ppm x 101325 Pa x 44.01e-3 kg/mol / (8.3144598 J/mol/K x (temp + 273.15) K) [mg/m3];
    x[15] = rh / 100 x 610.78 x exp(17.2694 temp / (temp + 238.3)) [Pa]. x[25] is fruit dry matter
    [mg/m2]; x[27] is time [days], so the states of a trace all differ.
    """
    x = np.zeros(NX)
    x[0] = ppm * 101325 * 44.01e-3 / (8.3144598 * (temp + 273.15))
    x[2] = temp
    x[15] = rh / 100 * 610.78 * np.exp(17.2694 * temp / (temp + 238.3))
    x[25] = fruit
    x[27] = day
    return x


def _trace(climates, *, fruit=None, u=ZERO_U, dt=DT):
    """states through the given climates (one per state, T + 1 of them) and T steps of controls u."""
    fruit = np.zeros(len(climates)) if fruit is None else fruit
    states = np.stack(
        [
            _state(*c, day=t * dt / 86400, fruit=f)
            for t, (c, f) in enumerate(zip(climates, fruit, strict=True))
        ]
    )
    controls = np.tile(np.asarray(u, dtype=np.float64), (len(climates) - 1, 1))
    return states, controls


def _metrics(states, controls, params=P, **kwargs):
    """episode_metrics with the default prices and bounds; season_steps defaults to the trace's T."""
    import greenleaf_control.metrics as metrics

    kwargs.setdefault("season_steps", len(controls))
    return metrics.episode_metrics(states, controls, params, **kwargs)


def _hours(m: dict) -> dict:
    return {c: m[f"viol_hours_{c}"] for c in CONSTRAINTS}


def _pct(m: dict) -> dict:
    return {c: m[f"pct_in_bounds_{c}"] for c in CONSTRAINTS}


def _imports(source: str) -> set[str]:
    """Every module metrics.py imports, and every name it from-imports, relative imports resolved
    against greenleaf_control. Walks the whole tree, so imports inside functions count."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(filter(None, ["greenleaf_control", base]))
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def _paper_day(name: str) -> tuple[list[dict], np.ndarray]:
    """A 1-day PAPER run of a baseline, and the params of the env it ran on."""
    infos = envs.run_baseline(name, scenarios.PAPER, season_length=1)
    params = envs.make_raw_env(
        scenario=scenarios.PAPER, season_length=1, normalize_actions=False
    ).unwrapped.p
    return infos, np.asarray(params)


def _assert_matches_upstream(m: dict, infos: list[dict], season_steps: int) -> None:
    """The metrics of a clean run agree with the sums of upstream's per-step info."""
    up = {
        k: sum(info[k] for info in infos)
        for k in ("profit", "heat_cost", "elec_cost", "co2_cost", "fruit_growth_ffw")
    }
    assert m["profit_eur_m2"] == pytest.approx(up["profit"], abs=1e-6)
    for key in ("heat_cost", "elec_cost", "co2_cost"):
        assert m[key] == pytest.approx(up[key], rel=1e-9, abs=1e-12), key
    assert m["yield_kg_m2"] == pytest.approx(up["fruit_growth_ffw"], rel=1e-9, abs=1e-12)
    # paper_reward is upstream's reward; rel 1e-6 for upstream's float32 observations.
    paper = sum(info["paper_reward"] for info in infos)
    assert m["paper_return"] == pytest.approx(paper, rel=1e-6)
    counts = {c: sum(info[f"cost_{c}"] > 0 for info in infos) for c in CONSTRAINTS}
    assert sum(counts.values()) > 0  # otherwise the count check compares only zeros
    assert _hours(m) == {c: n * 0.25 for c, n in counts.items()}  # exact: n x 900 s / 3600
    for c, n in counts.items():
        assert m[f"pct_in_bounds_{c}"] == pytest.approx(100 * (1 - n / len(infos)), rel=1e-12), c
    assert (m["steps"], m["solver_failures"]) == (season_steps, 0)
    assert m["valid"] is True


def _flaky(real_f, fail_on: int):
    """The real CasADi integrator, except that call number fail_on raises (a faked solver failure)."""
    calls = 0

    def f(**kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_on:
            raise RuntimeError("injected solver failure")
        return real_f(**kwargs)

    return f


# --- Independence: the grader never imports the graded code ------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    ["greenleaf_control.rewards", "greenleaf_control.envs", "gl_gym"],
    ids=["rewards", "envs", "gl_gym_so_the_conversions_are_reimplemented"],
)
def test_metrics_imports_nothing_from_the_graded_code(forbidden):
    """metrics.py (parsed with ast) imports nothing from rewards, from envs (which imports rewards),
    or from upstream gl_gym, whose reward module comes with its conversion helpers: importing
    gl_gym.environments.utils loads gl_gym.components.rewards. Relative imports count."""
    path = Path(greenleaf_control.__file__).parent / "metrics.py"
    assert path.is_file(), f"{path} does not exist"
    names = _imports(path.read_text())
    # The parser itself: each spelling of a rewards import is caught.
    for spelling in (
        "import greenleaf_control.rewards",
        "from greenleaf_control import rewards",
        "from greenleaf_control.rewards import CONSTRAINTS",
        "from . import rewards",
        "from .rewards import CONSTRAINTS",
        "def f():\n    import greenleaf_control.rewards as r",
    ):
        assert "greenleaf_control.rewards" in _imports(spelling), spelling
    assert [n for n in sorted(names) if n == forbidden or n.startswith(forbidden + ".")] == []


def test_using_metrics_loads_neither_rewards_nor_envs():
    """In a fresh interpreter, importing metrics and calling trace_from_infos and episode_metrics
    leaves greenleaf_control.rewards, greenleaf_control.envs and gl_gym.components.rewards out of
    sys.modules, so no import, however indirect or lazy, reaches the reward code."""
    code = (
        "import json, sys\n"
        "import numpy as np\n"
        "import greenleaf_control.metrics as metrics\n"
        "x = np.arange(56, dtype=float).reshape(2, 28)\n"
        "info = {'x_prev': x[0], 'x': x[1], 'controls': np.full(6, 0.5, dtype=np.float32)}\n"
        "states, controls = metrics.trace_from_infos([info])\n"
        "metrics.episode_metrics(states, controls, np.ones(208), season_steps=1)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert sorted(loaded & set(FORBIDDEN)) == []


# --- The contract surface ----------------------------------------------------------------------


def test_signatures_match_the_contract():
    """episode_metrics(states, controls, params, *, dt=900.0, season_steps, prices=..., bounds=...)
    and trace_from_infos(infos): the unused raw_env argument is dropped (orchestrator default 1)."""
    import greenleaf_control.metrics as metrics

    params = inspect.signature(metrics.episode_metrics).parameters
    assert list(params) == [
        "states",
        "controls",
        "params",
        "dt",
        "season_steps",
        "prices",
        "bounds",
    ]
    kinds = {name: p.kind for name, p in params.items()}
    assert kinds == {
        **dict.fromkeys(["states", "controls", "params"], inspect.Parameter.POSITIONAL_OR_KEYWORD),
        **dict.fromkeys(["dt", "season_steps", "prices", "bounds"], inspect.Parameter.KEYWORD_ONLY),
    }
    assert params["dt"].default == 900.0
    assert params["season_steps"].default is inspect.Parameter.empty
    assert list(inspect.signature(metrics.trace_from_infos).parameters) == ["infos"]


def test_constants_equal_upstreams_registered_values():
    """CONSTRAINTS, PRICES, BOUNDS and PAPER_NORMALISERS are upstream's registered reward kwargs,
    constraints and GreenhouseReward's violation normalisers, read here from gl_gym:
    fruit 1.6, dmfm 0.065, heat 0.09, elec 0.3, co2 0.3; 300-1600 ppm, 15-34 C, 50-85 %; 2500, 15, 15.
    """
    import greenleaf_control.metrics as metrics

    assert metrics.CONSTRAINTS == CONSTRAINTS
    assert metrics.PRICES == {
        "fruit": UPSTREAM_KW["fruit_price"],
        "dmfm": UPSTREAM_KW["dmfm"],
        "heat": UPSTREAM_KW["heating_price"],
        "elec": UPSTREAM_KW["elec_price"],
        "co2": UPSTREAM_KW["co2_price"],
    }
    assert metrics.BOUNDS == {
        c: (UPSTREAM_BOUNDS[f"{c}_min"], UPSTREAM_BOUNDS[f"{c}_max"]) for c in CONSTRAINTS
    }
    upstream = GreenhouseReward(**UPSTREAM_KW, dt=DT, p=P)
    assert metrics.PAPER_NORMALISERS == tuple(float(v) for v in upstream.max_state_violations)


def test_returns_the_protocol_keys_with_json_ready_types():
    """The keys are PROTOCOL §4's plus steps and valid, nothing else. steps and solver_failures are
    ints and valid a bool (Python's, so a ledger row serializes); every other value is a float."""
    m = _metrics(*_trace([IN_BAND] + [HOT, IN_BAND] * 4))
    assert set(m) == KEYS
    assert type(m["valid"]) is bool
    assert type(m["steps"]) is int
    assert type(m["solver_failures"]) is int
    floats = {k: v for k, v in m.items() if k not in ("valid", "steps", "solver_failures")}
    assert {k: isinstance(v, float) for k, v in floats.items()} == dict.fromkeys(floats, True)


def test_prices_and_bounds_default_to_the_module_constants():
    """Omitting prices and bounds gives the same metrics as passing PRICES and BOUNDS."""
    import greenleaf_control.metrics as metrics

    states, controls = _trace(
        [IN_BAND] + [HOT, (1850.0, 22.0, 95.0)] * 4, fruit=np.linspace(0, 6.5e5, 9), u=FULL_COST_U
    )
    omitted = metrics.episode_metrics(states, controls, P, season_steps=8)
    given = metrics.episode_metrics(
        states, controls, P, season_steps=8, prices=metrics.PRICES, bounds=metrics.BOUNDS
    )
    assert omitted == given


def test_given_prices_and_bounds_are_the_ones_used():
    """Doubled prices (dmfm kept) double each cost and the revenue; a 15-40 C band clears 36 C."""
    import greenleaf_control.metrics as metrics

    states, controls = _trace(
        [IN_BAND] + [HOT, IN_BAND] * 4, fruit=np.linspace(0, 6.5e5, 9), u=FULL_COST_U
    )
    doubled = {k: v if k == "dmfm" else 2 * v for k, v in metrics.PRICES.items()}
    m = metrics.episode_metrics(states, controls, P, season_steps=8, prices=doubled)
    # 8 full steps: heat 8 x 0.002925 x 2 = 0.0468, elec 8 x 0.0087 x 2 = 0.1392,
    # co2 8 x 0.00135 x 2 = 0.0216 EUR/m2; revenue 10.0 kg/m2 x 3.2 EUR/kg = 32.0 EUR/m2
    assert m["heat_cost"] == pytest.approx(0.0468, rel=1e-9)
    assert m["elec_cost"] == pytest.approx(0.1392, rel=1e-9)
    assert m["co2_cost"] == pytest.approx(0.0216, rel=1e-9)
    assert m["yield_kg_m2"] == pytest.approx(10.0, rel=1e-12)
    # 32.0 - (0.0468 + 0.1392 + 0.0216) = 31.7924
    assert m["profit_eur_m2"] == pytest.approx(31.7924, rel=1e-9)
    wide = {**metrics.BOUNDS, "temp": (15.0, 40.0)}
    m = metrics.episode_metrics(states, controls, P, season_steps=8, bounds=wide)
    assert (m["viol_hours_temp"], m["pct_in_bounds_temp"]) == (0.0, 100.0)


# --- Violation-hours and percent in bounds -----------------------------------------------------


def test_four_of_eight_steps_at_36_c_is_one_hour_and_50_percent():
    """The Done-when trace: 8 steps, every other one ending at 36 C (CO2 and RH in band).
    4 steps x 900 s / 3600 = 1.0 h; 100 x (1 - 4/8) = 50.0 %. CO2 and RH never violate."""
    m = _metrics(*_trace([IN_BAND] + [HOT, IN_BAND] * 4))
    assert m["viol_hours_temp"] == 1.0
    assert m["pct_in_bounds_temp"] == 50.0
    assert _hours(m) == {"co2": 0.0, "temp": 1.0, "rh": 0.0}
    assert _pct(m) == {"co2": 100.0, "temp": 50.0, "rh": 100.0}


@pytest.mark.parametrize(
    ("climate", "hours"),
    [
        # 1850 ppm is over the 1600 ppm max
        ((1850.0, 22.0, 70.0), {"co2": 1.0, "temp": 0.0, "rh": 0.0}),
        # 250 ppm is under the 300 ppm min
        ((250.0, 22.0, 70.0), {"co2": 1.0, "temp": 0.0, "rh": 0.0}),
        # 36 C is over the 34 C max
        ((800.0, 36.0, 70.0), {"co2": 0.0, "temp": 1.0, "rh": 0.0}),
        # 13 C is under the 15 C min
        ((800.0, 13.0, 70.0), {"co2": 0.0, "temp": 1.0, "rh": 0.0}),
        # 95 % is over the 85 % max
        ((800.0, 22.0, 95.0), {"co2": 0.0, "temp": 0.0, "rh": 1.0}),
        # 40 % is under the 50 % min
        ((800.0, 22.0, 40.0), {"co2": 0.0, "temp": 0.0, "rh": 1.0}),
        # all three at once, each counted under its own name
        ((1850.0, 36.0, 95.0), {"co2": 1.0, "temp": 1.0, "rh": 1.0}),
        # in band: no term fires
        (IN_BAND, {"co2": 0.0, "temp": 0.0, "rh": 0.0}),
    ],
    ids=["co2_high", "co2_low", "temp_high", "temp_low", "rh_high", "rh_low", "all", "in_band"],
)
def test_each_violation_counts_under_its_own_name(climate, hours):
    """4 of 8 steps end at the climate: each violated constraint gets 4 x 0.25 h = 1.0 h and
    100 x (1 - 4/8) = 50.0 % in bounds; the others 0.0 h and 100.0 %."""
    m = _metrics(*_trace([IN_BAND] + [climate, IN_BAND] * 4))
    assert _hours(m) == hours
    assert _pct(m) == {c: 100.0 - 50.0 * h for c, h in hours.items()}


@pytest.mark.parametrize(
    ("climates", "hours", "pct"),
    [
        # only states[0] is hot: the initial state is not graded, so 0 h and 100 %
        ([HOT] + [IN_BAND] * 8, 0.0, 100.0),
        # only states[8] is hot: the last step is graded, 0.25 h and 100 x (1 - 1/8) = 87.5 %
        ([IN_BAND] * 8 + [HOT], 0.25, 87.5),
    ],
    ids=["initial_state_hot", "final_state_hot"],
)
def test_step_t_is_graded_on_the_state_after_it(climates, hours, pct):
    m = _metrics(*_trace(climates))
    assert (m["viol_hours_temp"], m["pct_in_bounds_temp"]) == (hours, pct)


@pytest.mark.parametrize(
    ("temp", "hours", "pct"),
    [
        (15.0, 0.0, 100.0),  # exactly on the min: in bounds
        (34.0, 0.0, 100.0),  # exactly on the max: in bounds
        (np.nextafter(15.0, -np.inf), 2.0, 0.0),  # one ulp under: 8 steps x 0.25 h
        (np.nextafter(34.0, np.inf), 2.0, 0.0),  # one ulp over
    ],
    ids=["on_min", "on_max", "ulp_under_min", "ulp_over_max"],
)
def test_a_value_on_a_bound_does_not_violate(temp, hours, pct):
    m = _metrics(*_trace([(800.0, temp, 70.0)] * 9))
    assert (m["viol_hours_temp"], m["pct_in_bounds_temp"]) == (hours, pct)


# --- Costs, yield and profit -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("u", "expected"),
    [
        # Per full step: heat 18720 W / 144 m2 = 130 W/m2 x 0.25 h = 0.0325 kWh/m2 x 0.09 EUR/kWh
        # = 0.002925; elec 116 W/m2 x 0.25 h = 0.029 kWh/m2 x 0.3 EUR/kWh = 0.0087;
        # co2 720 mg/s / 144 m2 = 5 mg/m2/s x 900 s = 0.0045 kg/m2 x 0.3 EUR/kg = 0.00135.
        # Two full steps: 2 x each.
        ([FULL_COST_U, FULL_COST_U], {"heat": 0.00585, "elec": 0.0174, "co2": 0.0027}),
        # A full step, then uBoil 0.5, uCO2 0.2, uLamp 0.25: heat 0.002925 x 1.5 = 0.0043875,
        # elec 0.0087 x 1.25 = 0.010875, co2 0.00135 x 1.2 = 0.00162 (each reads its own control)
        (
            [FULL_COST_U, (0.5, 0.2, 0.0, 0.0, 0.25, 0.0)],
            {"heat": 0.0043875, "elec": 0.010875, "co2": 0.00162},
        ),
        # Screens and vents fully on cost nothing
        ([(0.0, 0.0, 1.0, 1.0, 0.0, 1.0)] * 2, {"heat": 0.0, "elec": 0.0, "co2": 0.0}),
    ],
    ids=["two_full_steps", "full_then_part", "screens_and_vents"],
)
def test_costs_match_a_hand_calculation_on_a_two_step_trace(u, expected):
    states, _ = _trace([IN_BAND] * 3)
    m = _metrics(states, np.asarray(u, dtype=np.float64))
    costs = {c: m[f"{c}_cost"] for c in ("heat", "elec", "co2")}
    assert costs == pytest.approx(expected, rel=1e-9, abs=1e-15)
    # No fruit growth, so profit is minus the costs.
    assert m["profit_eur_m2"] == pytest.approx(-sum(expected.values()), rel=1e-9, abs=1e-15)


@pytest.mark.parametrize(
    ("fruit", "yield_kg", "profit"),
    [
        # 6.5e5 mg DM/m2 x 1e-6 = 0.65 kg DM / 0.065 DM per FM = 10.0 kg FFW/m2; x 1.6 EUR/kg
        # = 16.0 EUR/m2 revenue, minus 2 full steps' 0.02595 EUR/m2 of costs = 15.97405
        ((0.0, 3.0e5, 6.5e5), 10.0, 15.97405),
        # the same 6.5e5 mg of growth from a nonzero start: yield is x[T] - x[0], not x[T]
        ((1.0e6, 1.2e6, 1.65e6), 10.0, 15.97405),
        # no growth: no yield, profit is minus the costs
        ((2.0e5, 2.0e5, 2.0e5), 0.0, -0.02595),
    ],
    ids=["from_zero", "from_nonzero", "no_growth"],
)
def test_yield_is_fresh_weight_fruit_growth_and_profit_is_revenue_minus_costs(
    fruit, yield_kg, profit
):
    m = _metrics(*_trace([IN_BAND] * 3, fruit=fruit, u=FULL_COST_U))
    assert m["yield_kg_m2"] == pytest.approx(yield_kg, rel=1e-12, abs=1e-15)
    assert m["profit_eur_m2"] == pytest.approx(profit, rel=1e-9)


def test_dt_sets_the_hours_per_step_and_the_costs():
    """dt = 1800 s: 4 violating steps x 0.5 h = 2.0 h, still 50 % of steps. Per full step, heat
    130 W/m2 x 0.5 h = 0.065 kWh/m2 x 0.09 = 0.00585, elec 116 W/m2 x 0.5 h x 1e-3 x 0.3 = 0.0174,
    co2 5 mg/m2/s x 1800 s = 0.009 kg/m2 x 0.3 = 0.0027 EUR/m2; x 8 steps."""
    states, controls = _trace([IN_BAND] + [HOT, IN_BAND] * 4, u=FULL_COST_U, dt=1800)
    m = _metrics(states, controls, dt=1800.0)
    assert (m["viol_hours_temp"], m["pct_in_bounds_temp"]) == (2.0, 50.0)
    assert m["heat_cost"] == pytest.approx(0.0468, rel=1e-9)
    assert m["elec_cost"] == pytest.approx(0.1392, rel=1e-9)
    assert m["co2_cost"] == pytest.approx(0.0216, rel=1e-9)


@pytest.mark.parametrize(
    ("climates", "u", "growth", "expected"),
    [
        # profit 0. min_profit = -(0.002925 heat + 0.0087 elec + 0.00135 CO2) = -0.012975;
        # max_profit = 295.2 mg/m2 x 1e-6 / 0.065 x 1.6 EUR/kg = 0.0072665 EUR/m2;
        # (0 + 0.012975) / (0.0072665 + 0.012975) = 0.6410110
        ((IN_BAND, IN_BAND), ZERO_U, 0.0, 0.6410110),
        # full-cost controls, no growth: profit = min_profit, so 0
        ((IN_BAND, IN_BAND), FULL_COST_U, 0.0, 0.0),
        # no controls, max growth: profit = max_profit, so 1
        ((IN_BAND, IN_BAND), ZERO_U, MAX_GROWTH, 1.0),
        # 36 C: 0.6410110 - (36 - 34) / 15 = 0.5076777
        ((IN_BAND, HOT), ZERO_U, 0.0, 0.5076777),
        # 1850 ppm, 36 C, 95 %: 0.6410110 - (250/2500 + 2/15 + 10/15 = 0.9) = -0.2589890
        ((IN_BAND, (1850.0, 36.0, 95.0)), ZERO_U, 0.0, -0.2589890),
        # 250 ppm, 13 C, 40 %: 0.6410110 - (50/2500 + 2/15 + 10/15 = 0.82) = -0.1789890
        ((IN_BAND, (250.0, 13.0, 40.0)), ZERO_U, 0.0, -0.1789890),
        # a hot initial state is not graded: 0.6410110
        ((HOT, IN_BAND), ZERO_U, 0.0, 0.6410110),
        # supersaturated: x[15] is 1.1 x the saturation pressure at 22 C, and RH clips to 100 %
        # as upstream's does (environments/utils.py:362): 0.6410110 - (100 - 85)/15 = -0.3589890
        # (unclipped, 110 %: 0.6410110 - 25/15 = -1.0256556)
        ((IN_BAND, (800.0, 22.0, 110.0)), ZERO_U, 0.0, -0.3589890),
    ],
    ids=["zero", "full_cost", "max_growth", "hot", "all_high", "all_low", "hot_start", "rh_110"],
)
def test_paper_return_matches_hand_values_on_one_step(climates, u, growth, expected):
    """paper_return is upstream's reward, rebuilt from the state: rel 1e-6 because p[154] = 0.328
    is stored as float32 (0.328000009)."""
    m = _metrics(*_trace(list(climates), fruit=(0.0, growth), u=u))
    assert m["paper_return"] == pytest.approx(expected, rel=1e-6, abs=1e-12)
    # One step ending outside RH 50-85 % (40, 95 or 110 %) is 0.25 h and 0 % in bounds.
    rh_out = not 50.0 <= climates[-1][2] <= 85.0
    assert (m["viol_hours_rh"], m["pct_in_bounds_rh"]) == ((0.25, 0.0) if rh_out else (0.0, 100.0))


# --- Validity ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("repeats", "season_steps", "failures", "valid"),
    [
        ((), 8, 0, True),  # 8 of 8 steps, every state moved
        ((), 96, 0, False),  # 8 of 96 steps: a short episode
        ((), 4, 0, False),  # 8 steps of a 4-step season: steps must equal season_steps
        ((8,), 8, 1, False),  # states[8] == states[7]: the last step's solver failed
        ((3,), 8, 1, False),  # states[3] == states[2]
        ((3, 8), 8, 2, False),  # two unchanged steps
    ],
    ids=["clean", "short", "long", "repeat_last", "repeat_mid", "two_repeats"],
)
def test_validity(repeats, season_steps, failures, valid):
    """valid iff solver_failures == 0 and steps == season_steps; a solver failure is a step whose
    state didn't move. An invalid episode is still graded (reported, never dropped)."""
    states, controls = _trace([IN_BAND] * 9)
    for t in repeats:
        states[t] = states[t - 1]
    m = _metrics(states, controls, season_steps=season_steps)
    assert (m["steps"], m["solver_failures"]) == (8, failures)
    assert m["valid"] is valid
    assert set(m) == KEYS


@pytest.mark.parametrize("fail_on", [10, N_DAY], ids=["mid", "last"])
def test_a_faked_solver_failure_is_counted_and_makes_the_episode_invalid(
    fail_on, monkeypatch, capsys
):
    """env.unwrapped.F raising on step k (faked) makes upstream truncate there with x unchanged:
    k steps, solver_failures 1, valid False. On step 96 the episode has all its steps, so the
    failure alone makes it invalid."""
    import greenleaf_control.metrics as metrics

    env = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    env.reset(seed=0)
    monkeypatch.setattr(env.unwrapped, "F", _flaky(env.unwrapped.F, fail_on))
    infos = []
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        infos.append(info)
    params = np.asarray(env.unwrapped.p)
    env.close()
    assert "Error in ODE approximation" in capsys.readouterr().out
    states, controls = metrics.trace_from_infos(infos)
    m = metrics.episode_metrics(states, controls, params, season_steps=N_DAY)
    assert (m["steps"], m["solver_failures"]) == (fail_on, 1)
    assert m["valid"] is False


# --- trace_from_infos --------------------------------------------------------------------------


def test_trace_from_infos_on_a_one_day_run():
    """A 1-day rb_default run on PAPER: states (97, 28) are infos[0]["x_prev"] then each info's
    "x"; controls (96, 6) are each info's "controls"; both float64 (upstream's controls are
    float32)."""
    import greenleaf_control.metrics as metrics

    infos = envs.run_baseline("rb_default", scenarios.PAPER, season_length=1)
    states, controls = metrics.trace_from_infos(infos)
    assert (states.shape, states.dtype) == ((N_DAY + 1, NX), np.float64)
    assert (controls.shape, controls.dtype) == ((N_DAY, 6), np.float64)
    np.testing.assert_array_equal(states[0], infos[0]["x_prev"])
    np.testing.assert_array_equal(states[1:], np.stack([info["x"] for info in infos]))
    np.testing.assert_array_equal(controls, np.stack([info["controls"] for info in infos]))


@pytest.mark.parametrize(
    ("states_shape", "controls_shape"),
    [
        ((9, NX), (7, 6)),  # one control short of T = 8
        ((9, NX), (9, 6)),  # one control too many
        ((9, NX), (8, 5)),  # 5 controls a step, not 6
        ((9, NX), (8,)),  # controls not 2-D
        ((9 * NX,), (8, 6)),  # states not 2-D
    ],
    ids=["controls_short", "controls_long", "controls_narrow", "controls_1d", "states_1d"],
)
def test_bad_shapes_raise_valueerror(states_shape, controls_shape):
    with pytest.raises(ValueError):
        _metrics(np.ones(states_shape), np.zeros(controls_shape), season_steps=8)


# --- The cross-check against upstream ----------------------------------------------------------


@pytest.mark.parametrize("name", ["rb_default", "zero_controls", "random"])
def test_one_paper_day_matches_upstream(name):
    """On a 1-day PAPER run of each baseline, the metrics written from states and controls agree
    with upstream's per-step info: profit to 1e-6, each cost and the yield to rel 1e-9, the
    violation counts exactly (upstream's cost_<c> > 0), paper_return to rel 1e-6. 96 steps, valid."""
    import greenleaf_control.metrics as metrics

    infos, params = _paper_day(name)
    states, controls = metrics.trace_from_infos(infos)
    m = metrics.episode_metrics(states, controls, params, season_steps=N_DAY)
    _assert_matches_upstream(m, infos, N_DAY)


@pytest.mark.slow
def test_full_paper_season_of_rb_default_matches_upstream():
    """The Done-when golden season: a full 60-day rb_default run on PAPER (5,760 steps). profit_eur_m2
    equals the sum of upstream's info["profit"] to 1e-6, and every other check of the 1-day cross-check
    holds; valid (about 5 s)."""
    import greenleaf_control.metrics as metrics

    infos = envs.run_baseline("rb_default", scenarios.PAPER)
    params = envs.make_raw_env(scenario=scenarios.PAPER, normalize_actions=False).unwrapped.p
    states, controls = metrics.trace_from_infos(infos)
    m = metrics.episode_metrics(states, controls, np.asarray(params), season_steps=N_SEASON)
    assert m["profit_eur_m2"] == pytest.approx(sum(info["profit"] for info in infos), abs=1e-6)
    assert m["steps"] == N_SEASON
    assert m["valid"] is True
    _assert_matches_upstream(m, infos, N_SEASON)

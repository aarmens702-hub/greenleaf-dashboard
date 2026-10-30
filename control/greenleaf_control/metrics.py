"""The independent grader: per-episode metrics from raw states and controls (PROTOCOL §4).

This is a scorer file. It never imports the reward or env code (greenleaf_control.rewards,
greenleaf_control.envs) or anything from gl_gym, whose conversion helpers load its reward module,
so a reward bug can't hide inside the score. Every formula is written here from the model's units
and PROTOCOL §1's prices and bounds; upstream's per-step info (gl_gym @2d3febb) is a cross-check in
the tests, not shared code.

- Step t runs from states[t] to states[t + 1] under controls[t], and its indoor climate is read from
  states[t + 1]. states[0], the initial state, is not graded.
- yield_kg_m2 is the net change in standing fruit dry matter x[25] over the season, as fresh weight
  (upstream's fruit_growth_ffw, summed).
- A value on a bound is in bounds; a violation's amount is how far outside the bound it is.
- Upstream swallows a solver failure and leaves x unchanged, so a step whose state didn't move is a
  solver failure. An episode with one, or with steps != season_steps, is invalid: reported, never
  dropped.
"""

import numpy as np

CONSTRAINTS = ("co2", "temp", "rh")
# PROTOCOL §1: EUR/kg fresh fruit, dry matter per fresh matter, EUR/kWh heat and power, EUR/kg CO2.
PRICES = {"fruit": 1.6, "dmfm": 0.065, "heat": 0.09, "elec": 0.3, "co2": 0.3}
# PROTOCOL §1: CO2 [ppm], air temperature [C], relative humidity [%].
BOUNDS = {"co2": (300.0, 1600.0), "temp": (15.0, 34.0), "rh": (50.0, 85.0)}
# Upstream's violation scales for its reward (paper_return), in CONSTRAINTS order.
PAPER_NORMALISERS = (2500.0, 15.0, 15.0)


def episode_metrics(
    states: np.ndarray,
    controls: np.ndarray,
    params: np.ndarray,
    *,
    dt: float = 900.0,
    season_steps: int,
    prices: dict = PRICES,
    bounds: dict = BOUNDS,
) -> dict:
    """PROTOCOL §4's metrics plus steps and valid, for states (T + 1, nx) and controls (T, 6).

    params is the model's parameter vector: p[46] aFlr [m2], p[108] pBoil [W], p[109] phiExtCo2
    [mg/s], p[154] rgFruit [mg/m2/s], p[172] thetaLampMax [W/m2]. dt is one step [s].
    """
    states = np.asarray(states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if states.ndim != 2 or controls.ndim != 2 or controls.shape != (len(states) - 1, 6):
        raise ValueError(
            f"need states (T + 1, nx) and controls (T, 6); got {states.shape} and {controls.shape}"
        )
    p = np.asarray(params, dtype=np.float64)
    steps = len(controls)
    x_prev, x = states[:-1], states[1:]

    # One step at full control [EUR/m2]: boiler W over the floor -> kWh/m2, lamps W/m2 -> kWh/m2,
    # CO2 dosing mg/s over the floor -> kg/m2.
    heat_full = p[108] / p[46] * dt / 3600 * 1e-3 * prices["heat"]
    elec_full = p[172] * dt / 3600 * 1e-3 * prices["elec"]
    co2_full = p[109] / p[46] * dt * 1e-6 * prices["co2"]
    heat = controls[:, 0] * heat_full  # uBoil
    elec = controls[:, 4] * elec_full  # uLamp
    co2 = controls[:, 1] * co2_full  # uCO2
    # Fruit dry matter x[25] [mg/m2] -> fresh weight [kg/m2].
    growth = (x[:, 25] - x_prev[:, 25]) * 1e-6 / prices["dmfm"]
    yield_kg_m2 = (states[-1, 25] - states[0, 25]) * 1e-6 / prices["dmfm"]
    profit = yield_kg_m2 * prices["fruit"] - heat.sum() - elec.sum() - co2.sum()

    # Indoor climate: x[0] CO2 [mg/m3] -> ppm by the ideal gas law at 1 atm; x[2] air temperature
    # [C]; x[15] vapour pressure [Pa] -> RH [%] over the saturation pressure at that temperature.
    temp = x[:, 2]
    ppm = 1e6 * 8.3144598 * (temp + 273.15) * (x[:, 0] * 1e-6) / (101325 * 44.01e-3)
    rh = np.clip(100 * x[:, 15] / (610.78 * np.exp(17.2694 * temp / (temp + 238.3))), 0.0, 100.0)
    climate = {"co2": ppm, "temp": temp, "rh": rh}

    # paper_return: upstream's per-step reward, profit min-max scaled between full-control costs
    # and no costs with the maximum fruit growth, minus each violation over its normaliser.
    max_profit = p[154] * dt * 1e-6 / prices["dmfm"] * prices["fruit"]
    min_profit = -(heat_full + elec_full + co2_full)
    step_profit = growth * prices["fruit"] - heat - elec - co2
    paper = (step_profit - min_profit) / (max_profit - min_profit)

    out = {"profit_eur_m2": float(profit)}
    for c, norm in zip(CONSTRAINTS, PAPER_NORMALISERS, strict=True):
        low, high = bounds[c]
        v = climate[c]
        violating = int(np.count_nonzero((v < low) | (v > high)))
        out[f"viol_hours_{c}"] = float(violating * dt / 3600)
        out[f"pct_in_bounds_{c}"] = float(100 * (1 - violating / steps))
        paper = paper - (np.maximum(low - v, 0.0) + np.maximum(v - high, 0.0)) / norm
    solver_failures = int(np.count_nonzero(np.all(x == x_prev, axis=1)))
    return {
        **out,
        "yield_kg_m2": float(yield_kg_m2),
        "heat_cost": float(heat.sum()),
        "elec_cost": float(elec.sum()),
        "co2_cost": float(co2.sum()),
        "paper_return": float(paper.sum()),
        "solver_failures": solver_failures,
        "steps": steps,
        "valid": bool(solver_failures == 0 and steps == season_steps),
    }


def trace_from_infos(infos: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """An episode's states (T + 1, nx) and controls (T, 6), float64, from its T per-step infos.

    states are the first step's x_prev, then each step's x; controls are each step's controls.
    """
    states = np.stack([infos[0]["x_prev"], *(info["x"] for info in infos)]).astype(np.float64)
    controls = np.stack([info["controls"] for info in infos]).astype(np.float64)
    return states, controls

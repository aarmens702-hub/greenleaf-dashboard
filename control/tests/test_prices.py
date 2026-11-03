"""Tests for greenleaf_control.prices (issue #68, the price-regime study for #66): `python -m
greenleaf_control.prices [--out DIR]` re-prices every recorded season (validation and the three 2012
test splits) under other prices and writes, per split and controller, profit IQM [95% CI], paired
gains over rb_tuned and rb_default, the ranking, two price sweeps and the prices where key rankings
flip. Nothing is simulated: profit is linear in each price, so re-pricing the recorded yield and
costs is exact. Exploratory, post hoc: no season is played and the seal is not reopened.

The conventions pinned here (the scout's change map, with the orchestrator's defaults).
- REGIMES = {"sim": a copy of metrics.PRICES, "agc2020": {"fruit": 1.6, "dmfm": 0.065, "heat":
  0.03, "elec": 0.06, "co2": 0.08}} [EUR/kg fruit, dry/fresh, EUR/kWh heat and power, EUR/kg CO2].
- reprice(seasons, prices) -> one profit per row [EUR/m2], array-like: yield_kg_m2 * fruit -
  heat_cost * heat / 0.09 - elec_cost * elec / 0.30 - co2_cost * co2 / 0.30 (the recorded costs are
  at metrics.PRICES). A regime whose dmfm is not metrics.PRICES["dmfm"] is a ValueError: the
  recorded yield was converted from dry matter at 0.065.
- load_seasons(*, ledger_path=ledger.LEDGER) -> DataFrame, one row per recorded season, at least
  SEASON_COLUMNS. It reads and checks nothing: validation from
  control/results/analysis/validation/scores.parquet, only the 7 labels site_build._validation_arms
  maps (selected by label, never by arm), controller = the final's arm id; the 3 test splits from
  control/results/final/<final-registered row>/seasons.parquet, found from the ledger's 2012
  final-result row, random included. No final-result row: ValueError naming "final-result". #73:
  the 2012 final-result is the one whose final-registered row has ledger.seal_of(...) == 2012, so a
  2013 final-result in the ledger (#71's second seal), before or after it, changes nothing, and a
  ledger whose only final-result is 2013's is refused like one with none.
- tables(seasons) -> {"regimes", "sweep", "crossovers"}: DataFrames with exactly REGIMES_COLUMNS,
  SWEEP_COLUMNS and CROSSOVER_COLUMNS, for the splits present in seasons (already checked, random
  dropped). A controller's matrix is runs by seed (NaN -> -1) x scenarios in scenarios.SPLITS order,
  as report._matrix.
  - regimes: one row per split x regime x controller. profit_* = stats.iqm_ci(matrix);
    gain_<b>_* = stats.iqm_ci(matrix - b's 1 x S row) for b in rb_tuned, rb_default; rank 1 = the
    highest profit_iqm in its split and regime; viol_hours_iqm (each season's co2 + temp + rh hours)
    and yield_iqm are price-free, the same under every regime.
  - sweep: one row per split x sweep point x controller; profit_iqm is the point IQM (rliable's
    aggregate_iqm). sweep "lambda": x = lambda in linspace(0, 1, 41), heat, elec, co2 = agc2020 +
    lambda * (sim - agc2020); sweep "elec": x = 0.02, 0.03, ..., 0.40 EUR/kWh (39 points), heat and
    co2 at agc2020's. fruit and dmfm are fixed.
  - crossovers: for every split, sweep and (controller, versus) in PAIRS, one row per sign change of
    the pair's difference along the sweep, x linearly interpolated between the two grid points; none
    is one row with x empty (NaN) and note "none in range". The difference is the paired-gain IQM,
    IQM(controller - versus's same season), against a deterministic versus (rb_default, rb_tuned),
    and the difference of the point IQMs for (ppo, residual_w05).
- main(argv=None, *, ledger_path=ledger.LEDGER) -> int, --out (default
  control/results/analysis/prices): it calls load_seasons, then first checks every row it returned,
  random included: reprice at sim equals the recorded profit_eur_m2 within 1e-9, else ValueError.
  Then it drops random and refuses (ValueError) a kept season that is not valid. Everything is
  loaded, checked and computed before out is made; then out holds exactly regimes.csv, sweep.csv,
  crossovers.csv, summary.md and sweep.png, and main returns 0. summary.md's line 1 names `python -m
  greenleaf_control.prices` and says "exploratory, re-priced recorded seasons", with no clock time
  and no out path. sweep.png: profit IQM vs lambda per controller, one panel per test split, display
  names from site_build.CONTROLLERS, every text >= 9 pt, and it says "exploratory, re-priced
  recorded seasons" too.

Every check imports greenleaf_control.prices inside its test body, so while it is missing each test
FAILS on an assertion rather than ERRORs. The real data is the committed ledger, the validation
scores and the final's seasons.parquet (the seal is spent: #45); every output goes under tmp_path.
The tests that run main on all recorded seasons (about 170 bootstrap CIs, ~20 s) are slow.
"""

import importlib
import importlib.util
import inspect
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenleaf_control import ledger, metrics, scenarios, site_build, stats

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "control"
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
ANALYSIS = CONTROL / "results" / "analysis"
VALIDATION_SCORES = ANALYSIS / "validation" / "scores.parquet"
FINAL_SEASONS = CONTROL / "results" / "final" / "final-registered-4d0a4ba9" / "seasons.parquet"
FINAL_RESULT = "final-result-aee23915"  # #45's, the 2012 seal's
FINAL_REGISTERED = "final-registered-4d0a4ba9"
# #73: a synthetic 2013 final (#71's second seal), under ids no real row has; its files are nowhere.
FINAL_REGISTERED_2013 = "final-registered-13131313"
FINAL_RESULT_2013 = "final-result-13131313"

PHRASE = "exploratory, re-priced recorded seasons"
COMMAND = "python -m greenleaf_control.prices"
SPLITS = ["validation", "amsterdam_2012", "london_2012", "abbotsford_2012"]
TEST_SPLITS = SPLITS[1:]
# The 7 page controllers and their runs on every split: the final's (validation's are the same
# registrations: residual_w05 is #36/f2fd's 10 seeds, not #34's 5; lag_b100 #39/6d92's 10).
RUNS = {
    "rb_tuned": 1,
    "rb_default": 1,
    "residual_w05": 10,
    "lag_b100": 10,
    "ppo": 5,
    "ppo_paper": 5,
    "zero_controls": 1,
}
# Validation's labels for them (site_build._validation_arms on the committed ledger).
VALIDATION_ARMS = {
    "zero_controls": "zero_controls",
    "rb_default": "rb_default",
    "rb_tuned": "rb_tuned",
    "ppo_paper #31/2d02": "ppo_paper",
    "ppo #33/0221": "ppo",
    "residual_w05 #36/f2fd": "residual_w05",
    "lag_b100 #39/6d92": "lag_b100",
}
# Recorded seasons per split, every label and arm (random included).
RECORDED_ROWS = {
    "validation": 234,
    "amsterdam_2012": 272,
    "london_2012": 272,
    "abbotsford_2012": 272,
}
PAIRS = [
    ("ppo", "rb_default"),
    ("ppo", "residual_w05"),
    ("zero_controls", "rb_default"),
    ("rb_tuned", "rb_default"),
    ("residual_w05", "rb_tuned"),
]
LAMBDAS = np.linspace(0, 1, 41)
ELECS = np.arange(2, 41) / 100  # 0.02 .. 0.40 EUR/kWh
SWEEP_X = {"lambda": LAMBDAS, "elec": ELECS}

SEASON_COLUMNS = (
    "split",
    "controller",
    "seed",
    "location",
    "growth_year",
    "start_day",
    "yield_kg_m2",
    "heat_cost",
    "elec_cost",
    "co2_cost",
    "profit_eur_m2",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "valid",
)
REGIMES_COLUMNS = {
    "split",
    "regime",
    "controller",
    "n_runs",
    *(
        f"{m}_{s}"
        for m in ("profit", "gain_rb_tuned", "gain_rb_default")
        for s in ("iqm", "lo", "hi")
    ),
    "rank",
    "viol_hours_iqm",
    "yield_iqm",
}
SWEEP_COLUMNS = {"split", "sweep", "x", "heat", "elec", "co2", "controller", "profit_iqm"}
CROSSOVER_COLUMNS = {"split", "sweep", "controller", "versus", "x", "note"}
OUTPUTS = ["crossovers.csv", "regimes.csv", "summary.md", "sweep.csv", "sweep.png"]

# Loaded by a run in a fresh interpreter, they would mean it built an env, played a season or
# loaded a policy (selection pulls the env in).
FORBIDDEN = (
    "gl_gym",
    "greenleaf_control.envs",
    "greenleaf_control.evaluate",
    "greenleaf_control.report",
    "greenleaf_control.export_replay",
    "greenleaf_control.rewards",
    "greenleaf_control.selection",
    "stable_baselines3",
    "torch",
)


# --- Helpers ---------------------------------------------------------------------------------------


def _module():
    """greenleaf_control.prices; an assertion, not an ImportError, while it is missing."""
    found = importlib.util.find_spec("greenleaf_control.prices")
    assert found is not None, "greenleaf_control.prices doesn't exist yet (#68)"
    return importlib.import_module("greenleaf_control.prices")


def _tree(top: Path) -> dict[str, tuple[int, int, int]]:
    """Every file, dir and link under top -> (type, size, mtime_ns), so a refusal that made, wrote
    or touched anything changes it."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames + filenames:
            st = (Path(dirpath) / name).lstat()
            found[str((Path(dirpath) / name).relative_to(top))] = (
                stat.S_IFMT(st.st_mode),
                st.st_size,
                st.st_mtime_ns,
            )
    return found


def _with_a_2013_final(text: str, *, where: str) -> str:
    """The ledger text with a synthetic 2013 final's two rows added (#73): a final-registered row of
    seal 2013 at the 2012 final's checkpoints, with #72's rb_tuned_agc2020 among its baselines, and
    its final-result over the 2013 splits (per_split nested by price regime, #72's shape; empty
    placeholders). where="after" appends them, the ledger's real order; where="before" puts them
    just before the 2012 final-result row, so the first final-result in file order is 2013's."""
    lines = text.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    [registered_2012] = [r for r in rows if r["row_id"] == FINAL_REGISTERED]
    stamp = {k: registered_2012[k] for k in ("schema_version", "at", "fingerprints")}
    registered = {
        **registered_2012,
        "row_id": FINAL_REGISTERED_2013,
        "issue": 72,
        "baselines": [*registered_2012["baselines"], "rb_tuned_agc2020"],
        "seal": 2013,
    }
    result = {
        **stamp,
        "row_id": FINAL_RESULT_2013,
        "kind": "final-result",
        "issue": 72,
        "final-registered": FINAL_REGISTERED_2013,
        "per_split": {split: {"sim": {}, "agc2020": {}} for split in scenarios.SEALS[2013]},
    }
    new = [json.dumps(r, sort_keys=True) + "\n" for r in (registered, result)]
    if where == "after":
        return "".join(lines + new)
    at = next(i for i, r in enumerate(rows) if r["row_id"] == FINAL_RESULT)
    return "".join(lines[:at] + new + lines[at:])


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    """The one row of frame whose columns equal where's values."""
    hit = frame
    for column, value in where.items():
        hit = hit[hit[column] == value]
    assert len(hit) == 1, f"{len(hit)} rows where {where}, not 1"
    return hit.iloc[0]


def _season_keys(rows: pd.DataFrame) -> list[tuple]:
    """rows' seasons as sorted (seed or -1, location, year, start day, profit, yield, valid)."""
    return sorted(
        (
            -1 if pd.isna(seed) else int(seed),
            str(location),
            int(year),
            int(day),
            float(profit),
            float(yield_),
            bool(valid),
        )
        for seed, location, year, day, profit, yield_, valid in zip(
            rows["seed"],
            rows["location"],
            rows["growth_year"],
            rows["start_day"],
            rows["profit_eur_m2"],
            rows["yield_kg_m2"],
            rows["valid"],
            strict=True,
        )
    )


def _sign_changes(x: np.ndarray, d: np.ndarray) -> list[float]:
    """Where d changes sign along x, linearly interpolated between the two grid points."""
    return [
        x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1])
        for i in range(len(x) - 1)
        if d[i] * d[i + 1] < 0
    ]


def _patch_loader(monkeypatch, prices, change) -> None:
    """prices.load_seasons returns the real seasons with change(frame) applied."""
    real = prices.load_seasons

    def patched(*args, **kwargs):
        frame = real(*args, **kwargs).copy()
        change(frame)
        return frame

    monkeypatch.setattr(prices, "load_seasons", patched)


def _first(frame: pd.DataFrame, split: str, controller: str, *, invalid: bool = False) -> int:
    """The position of frame's first season of controller on split (an invalid one if asked)."""
    hit = (frame["split"] == split) & (frame["controller"] == controller)
    if invalid:
        hit &= ~frame["valid"].astype(bool)
    assert hit.any(), f"load_seasons returned no {controller} season on {split} (invalid={invalid})"
    return int(np.flatnonzero(hit.to_numpy())[0])


# --- The hand-built seasons ----------------------------------------------------------------------

# Validation seasons (Amsterdam 2010, start days 45, 59, 75) of every controller, written by hand:
# (seed, yield per start day [kg/m2], heat, elec, co2 cost per season [EUR/m2 at sim prices],
# violation hours per season (co2, temp, rh) [h]); seed None is a baseline's one run. Listed with
# ppo's seed 1 first and the start days reversed, so a table built in frame order is caught.
HAND = {
    "zero_controls": [(None, (3, 4, 5), 0.0, 0.0, 0.0, (100, 50, 25))],
    "rb_default": [(None, (8, 8, 8), 0.0, 9.0, 0.0, (0, 1, 2))],
    "rb_tuned": [(None, (9, 9, 9), 2.88, 0.0, 0.0, (0.5, 0.25, 0))],
    "ppo": [(1, (6, 8, 10), 0.0, 0.0, 0.0, (2, 2, 2)), (0, (5, 6, 7), 0.0, 0.0, 0.0, (1, 1, 1))],
    "ppo_paper": [(s, (3.5, 3.5, 3.5), 0.0, 0.0, 0.8, (0, 0, 0)) for s in (0, 1)],
    "residual_w05": [(s, (10, 10, 10), 5.4, 0.0, 0.0, (0, 0, 0)) for s in (101, 100)],
    "lag_b100": [(s, (9.25, 9.25, 9.25), 2.88, 0.0, 0.0, (0, 0, 0)) for s in (200, 201)],
}


def _hand_built() -> pd.DataFrame:
    """HAND as season rows with SEASON_COLUMNS; profit_eur_m2 is the recorded one at sim prices,
    yield * 1.6 - heat - elec - co2, so the frame passes main's check."""
    days = [s["start_day"] for s in scenarios.VALIDATION]
    assert days == [45, 59, 75]
    rows = []
    for controller, runs in HAND.items():
        for seed, yields, heat, elec, co2, (h_co2, h_temp, h_rh) in runs:
            for day, y in reversed(list(zip(days, yields, strict=True))):
                rows.append(
                    {
                        "split": "validation",
                        "controller": controller,
                        "seed": np.nan if seed is None else float(seed),
                        "location": "Amsterdam",
                        "growth_year": 2010,
                        "start_day": day,
                        "yield_kg_m2": float(y),
                        "heat_cost": heat,
                        "elec_cost": elec,
                        "co2_cost": co2,
                        "profit_eur_m2": y * 1.6 - heat - elec - co2,
                        "viol_hours_co2": float(h_co2),
                        "viol_hours_temp": float(h_temp),
                        "viol_hours_rh": float(h_rh),
                        "valid": True,
                    }
                )
    return pd.DataFrame(rows, columns=list(SEASON_COLUMNS))


@pytest.fixture(scope="module")
def hand():
    """A function returning prices.tables(the hand-built seasons), computed once. Called inside
    each test body, so a missing module FAILS the test."""
    cache = []

    def get() -> dict:
        if not cache:
            cache.append(_module().tables(_hand_built()))
        return cache[0]

    return get


# Every per-season profit is 1.6 * yield - cost * price / sim price. At agc2020 heat is x 0.03/0.09
# = 1/3, elec x 0.06/0.30 = 0.2 and co2 x 0.08/0.30 = 4/15 of its sim cost. Every controller here
# but ppo is one run or identical runs on seasons of equal profit (zero_controls: 3 seasons, 1 run).
# ppo: 1.6 x yields [5, 6, 7] (seed 0) and [6, 8, 10] (seed 1); 6 scores, so the IQM cuts 1 at each
# end: 1.6 x mean(6, 6, 7, 8) = 1.6 x 6.75 = 10.8 EUR/m2 at any price (it has no costs).
# (regime, controller, n_runs, profit, gain over rb_tuned, gain over rb_default, rank, Σ hours,
# yield), profits and gains in EUR/m2, hours in h, yield in kg/m2.
HAND_REGIMES = [
    # sim: rb_tuned 9 x 1.6 - 2.88 = 11.52, rb_default 8 x 1.6 - 9 = 3.8
    ("sim", "zero_controls", 1, 6.4, -5.12, 2.6, 5, 175.0, 4.0),  # mean(3, 4, 5) x 1.6; h 100+50+25
    ("sim", "rb_default", 1, 3.8, -7.72, 0.0, 7, 3.0, 8.0),  # 12.8 - 9; 3.8 - 11.52; h 0+1+2
    ("sim", "rb_tuned", 1, 11.52, 0.0, 7.72, 2, 0.75, 9.0),  # 14.4 - 2.88; 11.52 - 3.8
    ("sim", "ppo", 2, 10.8, -0.72, 7.0, 3, 4.5, 6.75),  # 10.8 - 11.52; hours mean(3, 3, 6, 6)
    ("sim", "ppo_paper", 2, 4.8, -6.72, 1.0, 6, 0.0, 3.5),  # 3.5 x 1.6 - 0.8
    ("sim", "residual_w05", 2, 10.6, -0.92, 6.8, 4, 0.0, 10.0),  # 16 - 5.4
    ("sim", "lag_b100", 2, 11.92, 0.4, 8.12, 1, 0.0, 9.25),  # 9.25 x 1.6 - 2.88
    # agc2020: rb_tuned 14.4 - 2.88/3 = 13.44, rb_default 12.8 - 9 x 0.2 = 11.0
    ("agc2020", "zero_controls", 1, 6.4, -7.04, -4.6, 6, 175.0, 4.0),  # 6.4 - 13.44; 6.4 - 11.0
    ("agc2020", "rb_default", 1, 11.0, -2.44, 0.0, 4, 3.0, 8.0),  # 11.0 - 13.44
    ("agc2020", "rb_tuned", 1, 13.44, 0.0, 2.44, 3, 0.75, 9.0),  # 13.44 - 11.0
    ("agc2020", "ppo", 2, 10.8, -2.64, -0.2, 5, 4.5, 6.75),  # 10.8 - 13.44; 10.8 - 11.0
    # ppo_paper: 5.6 - 0.8 x 4/15 = 5.3867; - 13.44; - 11.0
    (
        "agc2020",
        "ppo_paper",
        2,
        5.6 - 0.8 * 4 / 15,
        -7.84 - 0.8 * 4 / 15,
        -5.4 - 0.8 * 4 / 15,
        7,
        0.0,
        3.5,
    ),
    ("agc2020", "residual_w05", 2, 14.2, 0.76, 3.2, 1, 0.0, 10.0),  # 16 - 5.4/3; 14.2 - 13.44
    ("agc2020", "lag_b100", 2, 13.84, 0.4, 2.84, 2, 0.0, 9.25),  # 14.8 - 0.96; 13.84 - 11.0
]

# Each controller's profit along the two sweeps [EUR/m2]: lambda moves heat 0.03 -> 0.09, elec 0.06
# -> 0.30 and co2 0.08 -> 0.30 EUR/kWh or /kg; elec sweeps e with heat 0.03 and co2 0.08.
HAND_SWEEPS = {
    "zero_controls": (lambda lam: 6.4, lambda e: 6.4),  # no costs
    "rb_default": (
        lambda lam: 11.0 - 7.2 * lam,  # 12.8 - 9 x (0.06 + 0.24 lam) / 0.30
        lambda e: 12.8 - 30 * e,  # 12.8 - 9 x e / 0.30
    ),
    "rb_tuned": (
        lambda lam: 13.44 - 1.92 * lam,
        lambda e: 13.44,
    ),  # 14.4 - 2.88 (0.03+0.06 lam)/0.09
    "ppo": (lambda lam: 10.8, lambda e: 10.8),  # no costs
    "ppo_paper": (
        lambda lam: 5.6 - 0.8 * (0.08 + 0.22 * lam) / 0.30,
        lambda e: 5.6 - 0.8 * 0.08 / 0.30,
    ),
    "residual_w05": (lambda lam: 14.2 - 3.6 * lam, lambda e: 14.2),  # 16 - 5.4 (0.03+0.06 lam)/0.09
    "lag_b100": (lambda lam: 13.84 - 1.92 * lam, lambda e: 13.84),  # 14.8 - 2.88 (...)/0.09
}

# (sweep, controller, versus, crossings, note): every paired difference d = controller - versus
# here is the same in every season and run, so it is linear along each sweep and its zero is exact;
# d rising through 0 puts the controller ahead above x, d falling through 0 ahead below x.
ABOVE, BELOW, NONE = "controller ahead above x", "controller ahead below x", "none in range"
HAND_CROSSINGS = [
    # 10.8 - (11.0 - 7.2 lam) = -0.2 + 7.2 lam, rising through 0 at lam = 0.2/7.2
    ("lambda", "ppo", "rb_default", [1 / 36], ABOVE),
    # 10.8 - (14.2 - 3.6 lam) = -3.4 + 3.6 lam, rising through 0 at lam = 3.4/3.6
    ("lambda", "ppo", "residual_w05", [17 / 18], ABOVE),
    # 6.4 - (11.0 - 7.2 lam) = -4.6 + 7.2 lam, rising through 0 at lam = 4.6/7.2
    ("lambda", "zero_controls", "rb_default", [23 / 36], ABOVE),
    ("lambda", "rb_tuned", "rb_default", [], NONE),  # 2.44 + 5.28 lam > 0 on [0, 1]
    # (14.2 - 3.6 lam) - (13.44 - 1.92 lam) = 0.76 - 1.68 lam, falling through 0 at lam = 0.76/1.68
    ("lambda", "residual_w05", "rb_tuned", [19 / 42], BELOW),
    # 10.8 - (12.8 - 30 e) = -2 + 30 e, rising through 0 at e = 2/30 EUR/kWh
    ("elec", "ppo", "rb_default", [1 / 15], ABOVE),
    ("elec", "ppo", "residual_w05", [], NONE),  # 10.8 - 14.2 < 0 at every e
    # 6.4 - (12.8 - 30 e) = -6.4 + 30 e, rising through 0 at e = 6.4/30 EUR/kWh
    ("elec", "zero_controls", "rb_default", [16 / 75], ABOVE),
    ("elec", "rb_tuned", "rb_default", [], NONE),  # 0.64 + 30 e > 0
    ("elec", "residual_w05", "rb_tuned", [], NONE),  # 14.2 - 13.44 = 0.76 > 0
]


# --- The contract surface ----------------------------------------------------------------------


def test_signatures_match_the_contract():
    """load_seasons(*, ledger_path=ledger.LEDGER) and main(argv=None, *, ledger_path=ledger.LEDGER);
    reprice and tables are plain functions."""
    prices = _module()
    load = inspect.signature(prices.load_seasons).parameters
    assert list(load) == ["ledger_path"]
    assert load["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert load["ledger_path"].default == ledger.LEDGER
    main = inspect.signature(prices.main).parameters
    assert list(main) == ["argv", "ledger_path"]
    assert main["argv"].default is None
    assert main["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert main["ledger_path"].default == ledger.LEDGER
    assert inspect.isfunction(prices.reprice) and inspect.isfunction(prices.tables)


# --- The regimes and re-pricing ------------------------------------------------------------------


def test_regimes_are_the_simulators_prices_and_agc2020s():
    """sim is metrics.PRICES (a copy, so a change to it can't reach the grader); agc2020 is the
    issue's 2020 prices: fruit 1.6 EUR/kg, dmfm 0.065, heat 0.03, elec 0.06 EUR/kWh, CO2 0.08
    EUR/kg."""
    prices = _module()
    assert set(prices.REGIMES) == {"sim", "agc2020"}
    assert prices.REGIMES["sim"] == metrics.PRICES
    assert prices.REGIMES["sim"] is not metrics.PRICES
    assert prices.REGIMES["agc2020"] == {
        "fruit": 1.6,
        "dmfm": 0.065,
        "heat": 0.03,
        "elec": 0.06,
        "co2": 0.08,
    }


# Two seasons: (yield [kg/m2], heat, elec, co2 cost [EUR/m2 at sim prices]).
REPRICE_SEASONS = [(10.0, 0.9, 3.0, 0.6), (0.0, 0.09, 0.3, 0.3)]


@pytest.mark.parametrize(
    ("regime", "want"),
    [
        # 10 x 1.6 - 0.9 - 3 - 0.6 = 11.5; 0 - 0.09 - 0.3 - 0.3 = -0.69 EUR/m2
        pytest.param(
            {"fruit": 1.6, "dmfm": 0.065, "heat": 0.09, "elec": 0.3, "co2": 0.3},
            [11.5, -0.69],
            id="sim",
        ),
        # 16 - 0.9/3 - 3 x 0.2 - 0.6 x 4/15 = 16 - 0.3 - 0.6 - 0.16 = 14.94; -0.03 - 0.06 - 0.08 = -0.17
        pytest.param(
            {"fruit": 1.6, "dmfm": 0.065, "heat": 0.03, "elec": 0.06, "co2": 0.08},
            [14.94, -0.17],
            id="agc2020",
        ),
        # 10 x 2 - 0.9 x 0.5 - 3 x 0.5 - 0.6 x 2 = 20 - 0.45 - 1.5 - 1.2 = 16.85; -0.045 - 0.15 - 0.6
        pytest.param(
            {"fruit": 2.0, "dmfm": 0.065, "heat": 0.045, "elec": 0.15, "co2": 0.6},
            [16.85, -0.795],
            id="every-price-moved",
        ),
        # free energy and CO2: 10 x 1.6 = 16; 0
        pytest.param(
            {"fruit": 1.6, "dmfm": 0.065, "heat": 0.0, "elec": 0.0, "co2": 0.0},
            [16.0, 0.0],
            id="free-inputs",
        ),
    ],
)
def test_reprice_is_hand_computed_profit(regime, want):
    """profit = yield x fruit - heat_cost x heat/0.09 - elec_cost x elec/0.30 - co2_cost x co2/0.30
    per season, each cost term scaled from its sim price."""
    prices = _module()
    seasons = pd.DataFrame(
        REPRICE_SEASONS, columns=["yield_kg_m2", "heat_cost", "elec_cost", "co2_cost"]
    )
    got = np.asarray(prices.reprice(seasons, regime), dtype=np.float64)
    assert got.shape == (2,)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dmfm", [0.07, 0.06, 0.0])
def test_a_regime_with_another_dry_matter_ratio_is_refused(dmfm):
    """The recorded yield is fresh weight converted from dry matter at 0.065; a regime that moves
    dmfm would need the dry matter, so reprice refuses it (ValueError)."""
    prices = _module()
    seasons = pd.DataFrame(
        REPRICE_SEASONS, columns=["yield_kg_m2", "heat_cost", "elec_cost", "co2_cost"]
    )
    with pytest.raises(ValueError):
        prices.reprice(seasons, {**prices.REGIMES["sim"], "dmfm": dmfm})


@pytest.mark.parametrize("split", SPLITS)
def test_repricing_at_sim_reproduces_every_recorded_profit(split):
    """Every recorded season of the split (validation: all 234 rows of scores.parquet, every label,
    random included; a 2012 split: all 272 of the final's seasons.parquet rows, random included)
    re-priced at REGIMES["sim"] equals its recorded profit_eur_m2 within 1e-9 EUR/m2."""
    prices = _module()
    if split == "validation":
        recorded = pd.read_parquet(VALIDATION_SCORES)
    else:
        recorded = pd.read_parquet(FINAL_SEASONS)
        recorded = recorded[recorded["split"] == split]
    assert len(recorded) == RECORDED_ROWS[split]
    got = np.asarray(prices.reprice(recorded, prices.REGIMES["sim"]), dtype=np.float64)
    np.testing.assert_allclose(got, recorded["profit_eur_m2"].to_numpy(), rtol=0, atol=1e-9)


# --- Loading ---------------------------------------------------------------------------------------


def test_load_seasons_reads_validations_seven_controllers_by_label():
    """Validation's rows are the 7 labels site_build._validation_arms maps, as the final's arm ids:
    residual_w05 is #36/f2fd's 10 seeds (not #34's 5), lag_b100 #39/6d92's, ppo #33's, ppo_paper
    #31's, and the 3 baselines; each with its recorded seasons unchanged."""
    prices = _module()
    seasons = prices.load_seasons()
    assert set(SEASON_COLUMNS) <= set(seasons.columns), sorted(
        set(SEASON_COLUMNS) - set(seasons.columns)
    )
    val = seasons[seasons["split"] == "validation"]
    kept = val[val["controller"] != "random"]
    assert kept.groupby("controller").size().to_dict() == {c: 3 * n for c, n in RUNS.items()}
    recorded = pd.read_parquet(VALIDATION_SCORES)
    for label, controller in VALIDATION_ARMS.items():
        want = _season_keys(recorded[recorded["label"] == label])
        assert _season_keys(kept[kept["controller"] == controller]) == want, (label, controller)


@pytest.mark.parametrize("split", TEST_SPLITS)
def test_load_seasons_reads_every_final_season_random_included(split):
    """A 2012 split's rows are the final's seasons.parquet rows of that split, unchanged: the 7
    controllers and random (main checks random at sim prices before it drops it)."""
    prices = _module()
    seasons = prices.load_seasons()
    mine = seasons[seasons["split"] == split]
    assert set(mine["controller"]) == {*RUNS, "random"}
    recorded = pd.read_parquet(FINAL_SEASONS)
    recorded = recorded[recorded["split"] == split]
    for arm in [*RUNS, "random"]:
        want = _season_keys(recorded[recorded["arm"] == arm])
        assert len(want) == 8 * RUNS.get(arm, 1)
        assert _season_keys(mine[mine["controller"] == arm]) == want, arm


@pytest.mark.parametrize("where", ["after", "before"])
def test_load_seasons_reads_the_2012_final_with_a_2013_final_result_in_the_ledger(where, tmp_path):
    """Done when (#73 c): with a 2013 final added to a copy of the committed ledger
    (_with_a_2013_final, after the 2012 final-result or before it), load_seasons returns exactly
    what it returns on the committed ledger: validation and the 2012 final's seasons
    (final-registered-4d0a4ba9's seasons.parquet), never the 2013 final's (its files are
    nowhere)."""
    prices = _module()
    path = tmp_path / "ledger.jsonl"
    text = REAL_LEDGER.read_text(encoding="utf-8")
    path.write_text(_with_a_2013_final(text, where=where), encoding="utf-8")
    want = prices.load_seasons()
    try:
        got = prices.load_seasons(ledger_path=path)
    except (ValueError, OSError) as e:
        pytest.fail(f"a 2013 final-result in the ledger stopped load_seasons: {e!r}")
    pd.testing.assert_frame_equal(got, want)
    assert set(got["split"]) == set(SPLITS)


# --- The hand-built tables ---------------------------------------------------------------------------


def test_hand_built_tables_have_their_columns_and_only_validation(hand):
    """tables returns exactly regimes, sweep and crossovers with their columns; the seasons are
    validation's only, so every row is validation's; 7 controllers x 2 regimes; 80 sweep points x 7
    controllers; every PAIR on both sweeps."""
    got = hand()
    assert set(got) == {"regimes", "sweep", "crossovers"}
    assert set(got["regimes"].columns) == REGIMES_COLUMNS
    assert set(got["sweep"].columns) == SWEEP_COLUMNS
    assert set(got["crossovers"].columns) == CROSSOVER_COLUMNS
    for frame in got.values():
        assert set(frame["split"]) == {"validation"}
    assert len(got["regimes"]) == 7 * 2
    assert len(got["sweep"]) == (41 + 39) * 7
    pairs = set(
        zip(
            got["crossovers"]["sweep"],
            got["crossovers"]["controller"],
            got["crossovers"]["versus"],
            strict=True,
        )
    )
    assert pairs == {(s, a, b) for s in SWEEP_X for a, b in PAIRS}


@pytest.mark.parametrize(
    (
        "regime",
        "controller",
        "n",
        "profit",
        "gain_tuned",
        "gain_default",
        "rank",
        "hours",
        "yield_",
    ),
    [pytest.param(*row, id=f"{row[0]}-{row[1]}") for row in HAND_REGIMES],
)
def test_hand_built_regime_rows(
    hand, regime, controller, n, profit, gain_tuned, gain_default, rank, hours, yield_
):
    """Each controller's profit IQM, paired gains over rb_tuned's and rb_default's same season,
    rank, Σ violation-hour IQM and yield IQM under each regime, hand-computed (HAND_REGIMES). Every
    controller but ppo has zero-width CIs: one run, or identical runs, resample to itself."""
    row = _one(hand()["regimes"], split="validation", regime=regime, controller=controller)
    assert int(row["n_runs"]) == n
    assert row["profit_iqm"] == pytest.approx(profit, abs=1e-9)
    assert row["gain_rb_tuned_iqm"] == pytest.approx(gain_tuned, abs=1e-9)
    assert row["gain_rb_default_iqm"] == pytest.approx(gain_default, abs=1e-9)
    assert int(row["rank"]) == rank
    assert row["viol_hours_iqm"] == pytest.approx(hours, abs=1e-9)
    assert row["yield_iqm"] == pytest.approx(yield_, abs=1e-9)
    if controller != "ppo":
        for m in ("profit", "gain_rb_tuned", "gain_rb_default"):
            assert row[f"{m}_lo"] == pytest.approx(row[f"{m}_iqm"], abs=1e-9), m
            assert row[f"{m}_hi"] == pytest.approx(row[f"{m}_iqm"], abs=1e-9), m


@pytest.mark.parametrize(
    ("regime", "tuned", "default"), [("sim", 11.52, 3.8), ("agc2020", 13.44, 11.0)]
)
def test_hand_built_ppo_cis_are_iqm_ci_of_its_seed_ordered_matrix(hand, regime, tuned, default):
    """ppo's profit [CI] is stats.iqm_ci of its runs (seed 0, then 1) x start days (45, 59, 75)
    matrix 1.6 x [[5, 6, 7], [6, 8, 10]] EUR/m2, and its gains are stats.iqm_ci of that minus
    rb_tuned's (9 x 1.6 - 2.88 x heat price/0.09) and rb_default's (8 x 1.6 - 9 x elec price/0.30)
    same-season profit, a 1 x 3 row."""
    row = _one(hand()["regimes"], split="validation", regime=regime, controller="ppo")
    ppo = 1.6 * np.array([[5.0, 6.0, 7.0], [6.0, 8.0, 10.0]])
    for m, want in (
        ("profit", stats.iqm_ci(ppo)),
        ("gain_rb_tuned", stats.iqm_ci(ppo - np.full((1, 3), tuned))),
        ("gain_rb_default", stats.iqm_ci(ppo - np.full((1, 3), default))),
    ):
        got = (row[f"{m}_iqm"], row[f"{m}_lo"], row[f"{m}_hi"])
        assert got == pytest.approx(want, abs=1e-9), m
    assert row["profit_lo"] < row["profit_iqm"] < row["profit_hi"]


def test_hand_built_sweep_grids_and_prices(hand):
    """lambda: x = linspace(0, 1, 41) and heat, elec, co2 = 0.03 + 0.06 x, 0.06 + 0.24 x, 0.08 +
    0.22 x (agc2020 at 0, sim at 1); elec: x = 0.02 .. 0.40 EUR/kWh in 0.01 steps (39 points), elec
    = x, heat 0.03, co2 0.08. Every controller has every point once."""
    sweep = hand()["sweep"]
    assert set(sweep["sweep"]) == {"lambda", "elec"}
    for controller in HAND:
        lam = sweep[(sweep["sweep"] == "lambda") & (sweep["controller"] == controller)].sort_values(
            "x"
        )
        np.testing.assert_allclose(lam["x"], LAMBDAS, rtol=0, atol=1e-12)
        np.testing.assert_allclose(lam["heat"], 0.03 + 0.06 * LAMBDAS, rtol=0, atol=1e-12)
        np.testing.assert_allclose(lam["elec"], 0.06 + 0.24 * LAMBDAS, rtol=0, atol=1e-12)
        np.testing.assert_allclose(lam["co2"], 0.08 + 0.22 * LAMBDAS, rtol=0, atol=1e-12)
        elec = sweep[(sweep["sweep"] == "elec") & (sweep["controller"] == controller)].sort_values(
            "x"
        )
        np.testing.assert_allclose(elec["x"], ELECS, rtol=0, atol=1e-12)
        np.testing.assert_allclose(elec["elec"], ELECS, rtol=0, atol=1e-12)
        np.testing.assert_allclose(elec["heat"], np.full(39, 0.03), rtol=0, atol=1e-12)
        np.testing.assert_allclose(elec["co2"], np.full(39, 0.08), rtol=0, atol=1e-12)


@pytest.mark.parametrize("controller", list(HAND_SWEEPS))
def test_hand_built_sweep_profit_iqms(hand, controller):
    """At every sweep point the controller's profit IQM is HAND_SWEEPS' hand-derived line."""
    sweep = hand()["sweep"]
    on_lambda, on_elec = HAND_SWEEPS[controller]
    for name, line in (("lambda", on_lambda), ("elec", on_elec)):
        rows = sweep[(sweep["sweep"] == name) & (sweep["controller"] == controller)].sort_values(
            "x"
        )
        want = [line(x) for x in SWEEP_X[name]]
        np.testing.assert_allclose(rows["profit_iqm"], want, rtol=0, atol=1e-9, err_msg=name)


@pytest.mark.parametrize(
    ("sweep", "controller", "versus", "want", "note"),
    [pytest.param(*row, id=f"{row[0]}-{row[1]}-vs-{row[2]}") for row in HAND_CROSSINGS],
)
def test_hand_built_crossovers(hand, sweep, controller, versus, want, note):
    """The pair's crossings along the sweep are HAND_CROSSINGS' hand-computed zeros (lambda, or EUR/
    kWh on the elec sweep), each between two grid points, with the side the controller wins on as
    its note: "controller ahead above x" where controller - versus rises through 0, "controller
    ahead below x" where it falls (summary.md publishes it); a pair that never changes sign is one
    row, x empty and note "none in range"."""
    crossovers = hand()["crossovers"]
    rows = crossovers[
        (crossovers["split"] == "validation")
        & (crossovers["sweep"] == sweep)
        & (crossovers["controller"] == controller)
        & (crossovers["versus"] == versus)
    ]
    if want:
        assert sorted(rows["x"]) == pytest.approx(want, abs=1e-9)
        assert rows["note"].tolist() == [note] * len(want)
    else:
        assert len(rows) == 1
        assert pd.isna(rows.iloc[0]["x"])
        assert rows.iloc[0]["note"] == note


# --- Refusals, before anything is written ------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "controller", "column", "invalid"),
    [
        pytest.param(
            "validation", "residual_w05", "profit_eur_m2", False, id="validation-residual-profit"
        ),
        pytest.param("london_2012", "ppo", "heat_cost", False, id="london-ppo-heat-cost"),
        pytest.param(
            "abbotsford_2012", "rb_tuned", "yield_kg_m2", False, id="abbotsford-rb-tuned-yield"
        ),
        # random is dropped from the tables, but checked first, invalid seasons included
        pytest.param(
            "abbotsford_2012", "random", "profit_eur_m2", True, id="abbotsford-random-invalid"
        ),
    ],
)
def test_a_season_nudged_by_1e6_is_refused_before_anything_is_written(
    split, controller, column, invalid, tmp_path, monkeypatch
):
    """One recorded season with column + 1e-6 (yield: 1.6e-6 EUR/m2 of profit) no longer re-prices
    at sim to its recorded profit: main raises ValueError, out is never made and nothing under
    tmp_path changes."""
    prices = _module()

    def nudge(frame):
        i = _first(frame, split, controller, invalid=invalid)
        frame.iloc[i, frame.columns.get_loc(column)] += 1e-6

    _patch_loader(monkeypatch, prices, nudge)
    out = tmp_path / "out"
    before = _tree(tmp_path)
    with pytest.raises(ValueError) as refused:
        prices.main(["--out", str(out)])
    assert str(refused.value).strip()
    assert not out.exists()
    assert _tree(tmp_path) == before


def test_an_invalid_kept_season_is_refused_before_anything_is_written(tmp_path, monkeypatch):
    """rb_tuned's first Amsterdam 2012 season marked invalid (its numbers unchanged, so it passes
    the sim check): main raises ValueError and out is never made. random's 3 invalid Abbotsford
    seasons are dropped, never refused (the real run below)."""
    prices = _module()

    def invalidate(frame):
        i = _first(frame, "amsterdam_2012", "rb_tuned")
        frame.iloc[i, frame.columns.get_loc("valid")] = False

    _patch_loader(monkeypatch, prices, invalidate)
    out = tmp_path / "out"
    before = _tree(tmp_path)
    with pytest.raises(ValueError) as refused:
        prices.main(["--out", str(out)])
    assert str(refused.value).strip()
    assert not out.exists()
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("case", ["no-final-result", "only-a-2013-final-result"])
def test_a_ledger_without_a_final_result_row_is_refused(tmp_path, case):
    """The committed ledger minus its 2012 final-result line (by row id: #73, a 2013 one may follow
    it), alone or with a 2013 final added (#73: a final-result of another seal has no 2012
    seasons): no final to read the 2012 seasons (or map validation's labels) from. main raises
    ValueError naming "final-result" (not an error reading the 2013 final's missing files); out is
    never made."""
    prices = _module()
    lines = REAL_LEDGER.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line)["row_id"] != FINAL_RESULT]
    assert len(kept) == len(lines) - 1
    text = "".join(kept)
    if case == "only-a-2013-final-result":
        text = _with_a_2013_final(text, where="after")
    no_final = tmp_path / "ledger.jsonl"
    no_final.write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    try:
        prices.main(["--out", str(out)], ledger_path=no_final)
    except ValueError as e:
        said = str(e)
    except OSError as e:
        pytest.fail(f"main read a final's files instead of refusing: {e!r}")
    else:
        pytest.fail("main ran without a 2012 final-result row")
    assert "final-result" in said, said
    assert not out.exists()


# --- A real run on every recorded season (slow: ~170 bootstrap CIs) --------------------------------


def _figure_facts(fig, args, kwargs) -> dict:
    """What a saved figure shows: its path, every visible non-empty text with its size in pt, and
    each line's (x, y) data."""
    from matplotlib.text import Text

    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return {
        "path": str(args[0] if args else kwargs.get("fname")),
        "texts": [(t.get_text(), float(t.get_fontsize())) for t in texts],
        "lines_per_panel": [len(ax.get_lines()) for ax in fig.axes if ax.get_lines()],
        "curves": [
            (
                np.asarray(line.get_xdata(), dtype=np.float64),
                np.asarray(line.get_ydata(), dtype=np.float64),
            )
            for ax in fig.axes
            for line in ax.get_lines()
        ],
    }


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """A function returning (out, main's return value, the figures it saved): its first call runs
    main(["--out", out]) on the committed ledger and recorded seasons into a fresh, missing dir,
    with Figure.savefig wrapped to record each figure after the real save; later calls return the
    same. Called inside each test body, so a missing module FAILS the test."""
    cache = []

    def get() -> tuple[Path, int, list[dict]]:
        if not cache:
            from matplotlib.figure import Figure

            prices = _module()
            out = tmp_path_factory.mktemp("prices") / "out"
            figures = []
            real = Figure.savefig

            def spy(self, *args, **kwargs):
                done = real(self, *args, **kwargs)
                figures.append(_figure_facts(self, args, kwargs))
                return done

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(Figure, "savefig", spy)
                code = prices.main(["--out", str(out)])
            cache.append((out, code, figures))
        return cache[0]

    return get


def _csvs(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(out / "regimes.csv"),
        pd.read_csv(out / "sweep.csv"),
        pd.read_csv(out / "crossovers.csv"),
    )


@pytest.mark.slow
def test_main_writes_exactly_the_five_outputs_with_their_columns(ran):
    """main returns 0; out holds exactly regimes.csv (4 splits x 2 regimes x 7 controllers = 56
    rows), sweep.csv (4 x 80 points x 7 = 2,240), crossovers.csv (at least one row per split x
    sweep x pair: 40), summary.md and sweep.png (a PNG), with exactly the pinned columns, and no
    random anywhere."""
    out, code, _ = ran()
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    regimes, sweep, crossovers = _csvs(out)
    assert set(regimes.columns) == REGIMES_COLUMNS
    assert set(sweep.columns) == SWEEP_COLUMNS
    assert set(crossovers.columns) == CROSSOVER_COLUMNS
    assert len(regimes) == 4 * 2 * 7
    assert len(sweep) == 4 * (41 + 39) * 7
    assert len(crossovers) >= 4 * 2 * len(PAIRS)
    for frame in (regimes, sweep):
        assert set(frame["split"]) == set(SPLITS)
        assert set(frame["controller"]) == set(RUNS)
    assert (out / "sweep.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.slow
def test_summary_md_says_what_it_is_on_line_one_with_no_time_or_out_path(ran):
    """summary.md's first line names `python -m greenleaf_control.prices` and says "exploratory,
    re-priced recorded seasons"; no clock time and no out path anywhere, so a rerun into another
    dir gives the same bytes."""
    out, _, _ = ran()
    text = (out / "summary.md").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert COMMAND in first
    assert PHRASE in first
    assert str(out) not in text and str(out.resolve()) not in text
    assert re.search(r"\b\d{1,2}:\d{2}\b", text) is None


@pytest.mark.slow
@pytest.mark.parametrize("split", SPLITS)
def test_regimes_csv_has_the_seven_controllers_with_their_runs(ran, split):
    """Under each regime the split has exactly the 7 page controllers (no random), each with the
    final's run count: residual_w05 and lag_b100 10 (#36/f2fd's and #39/6d92's on validation, not
    #34's or #35's 5), ppo and ppo_paper 5, the baselines 1."""
    out, _, _ = ran()
    regimes, _, _ = _csvs(out)
    for regime in ("sim", "agc2020"):
        rows = regimes[(regimes["split"] == split) & (regimes["regime"] == regime)]
        assert len(rows) == len(RUNS), (split, regime, rows["controller"].tolist())
        assert dict(zip(rows["controller"], rows["n_runs"].astype(int), strict=True)) == RUNS


@pytest.mark.slow
@pytest.mark.parametrize("split", SPLITS)
def test_at_sim_prices_regimes_csv_reproduces_the_reports_summary(ran, split):
    """At sim prices, each controller's profit IQM [CI] and gain over rb_tuned [CI] equal the
    report's summary.csv row for it (validation: its #31/#33/#36/#39 label) within 1e-9: the same
    seasons, matrices and bootstrap. Σ hours and yield IQMs equal summary.csv's viol_hours_total and
    yield_kg_m2 IQMs under both regimes: price doesn't move them."""
    out, _, _ = ran()
    regimes, _, _ = _csvs(out)
    summary = pd.read_csv(ANALYSIS / split / "summary.csv")
    labels = VALIDATION_ARMS if split == "validation" else {c: c for c in RUNS}
    for label, controller in labels.items():
        want = _one(summary, label=label)
        sim = _one(regimes, split=split, regime="sim", controller=controller)
        for mine, theirs in (("profit", "profit_eur_m2"), ("gain_rb_tuned", "profit_gain")):
            for s in ("iqm", "lo", "hi"):
                assert sim[f"{mine}_{s}"] == pytest.approx(want[f"{theirs}_{s}"], abs=1e-9), (
                    label,
                    mine,
                    s,
                )
        assert int(sim["n_runs"]) == int(want["n_runs"])
        for regime in ("sim", "agc2020"):
            row = _one(regimes, split=split, regime=regime, controller=controller)
            assert row["viol_hours_iqm"] == pytest.approx(want["viol_hours_total_iqm"], abs=1e-9)
            assert row["yield_iqm"] == pytest.approx(want["yield_kg_m2_iqm"], abs=1e-9)


@pytest.mark.slow
def test_ranks_order_profit_within_each_split_and_regime(ran):
    """In every split and regime the ranks are 1..7, 1 the highest profit IQM."""
    out, _, _ = ran()
    regimes, _, _ = _csvs(out)
    for (split, regime), rows in regimes.groupby(["split", "regime"]):
        ordered = rows.sort_values("profit_iqm", ascending=False)
        assert ordered["rank"].astype(int).tolist() == list(range(1, 8)), (split, regime)


@pytest.mark.slow
def test_sweep_ends_equal_the_regime_rows(ran):
    """lambda = 0 is agc2020 and lambda = 1 is sim: their sweep.csv profit IQMs equal regimes.csv's
    for every split and controller; so does the elec sweep at 0.06 EUR/kWh (agc2020's elec, with
    its heat and CO2)."""
    out, _, _ = ran()
    regimes, sweep, _ = _csvs(out)
    for split in SPLITS:
        for controller in RUNS:
            mine = sweep[(sweep["split"] == split) & (sweep["controller"] == controller)]
            lam = mine[mine["sweep"] == "lambda"].sort_values("x")
            elec = mine[mine["sweep"] == "elec"].sort_values("x")
            np.testing.assert_allclose(lam["x"], LAMBDAS, rtol=0, atol=1e-12)
            np.testing.assert_allclose(elec["x"], ELECS, rtol=0, atol=1e-12)
            agc = _one(regimes, split=split, regime="agc2020", controller=controller)["profit_iqm"]
            sim = _one(regimes, split=split, regime="sim", controller=controller)["profit_iqm"]
            where = (split, controller)
            assert lam["profit_iqm"].iloc[0] == pytest.approx(agc, abs=1e-9), where
            assert lam["profit_iqm"].iloc[-1] == pytest.approx(sim, abs=1e-9), where
            assert elec["profit_iqm"].iloc[4] == pytest.approx(agc, abs=1e-9), where


@pytest.mark.slow
def test_sweep_profit_never_rises_as_prices_rise(ran):
    """Every price rises along both sweeps and fruit is fixed, so every season's profit, and so
    every order statistic and the IQM, never rises: non-increasing along x within 1e-12."""
    out, _, _ = ran()
    _, sweep, _ = _csvs(out)
    for (split, name, controller), rows in sweep.groupby(["split", "sweep", "controller"]):
        steps = np.diff(rows.sort_values("x")["profit_iqm"].to_numpy())
        assert (steps <= 1e-12).all(), (split, name, controller, steps.max())


@pytest.mark.slow
def test_crossovers_cover_every_split_sweep_and_pair(ran):
    """Each split x sweep x PAIR has its rows: crossings inside the sweep's range, or exactly one
    row, x empty and note "none in range". Only PAIRS appear."""
    out, _, _ = ran()
    _, _, crossovers = _csvs(out)
    seen = set(
        zip(
            crossovers["split"],
            crossovers["sweep"],
            crossovers["controller"],
            crossovers["versus"],
            strict=True,
        )
    )
    assert seen == {(s, w, a, b) for s in SPLITS for w in SWEEP_X for a, b in PAIRS}
    for (split, name, a, b), rows in crossovers.groupby(["split", "sweep", "controller", "versus"]):
        low, high = SWEEP_X[name][0], SWEEP_X[name][-1]
        if rows["x"].isna().any():
            assert len(rows) == 1 and rows.iloc[0]["note"] == "none in range", (split, name, a, b)
        else:
            assert rows["x"].between(low, high).all(), (split, name, a, b)


@pytest.mark.slow
def test_crossovers_of_iqm_differences_match_sweep_csv(ran):
    """(ppo, residual_w05)'s crossings are where sweep.csv's ppo minus residual_w05 profit IQM
    changes sign, interpolated, on every split and sweep; on validation (3 seasons, 1 run, nothing
    trimmed) so are (zero_controls, rb_default)'s and (rb_tuned, rb_default)'s paired gains."""
    out, _, _ = ran()
    _, sweep, crossovers = _csvs(out)
    checked = [(s, "ppo", "residual_w05") for s in SPLITS] + [
        ("validation", "zero_controls", "rb_default"),
        ("validation", "rb_tuned", "rb_default"),
    ]
    for split, a, b in checked:
        for name, x in SWEEP_X.items():
            on = sweep[(sweep["split"] == split) & (sweep["sweep"] == name)]
            ya = on[on["controller"] == a].sort_values("x")["profit_iqm"].to_numpy()
            yb = on[on["controller"] == b].sort_values("x")["profit_iqm"].to_numpy()
            want = _sign_changes(x, ya - yb)
            rows = crossovers[
                (crossovers["split"] == split)
                & (crossovers["sweep"] == name)
                & (crossovers["controller"] == a)
                & (crossovers["versus"] == b)
            ]
            got = sorted(rows["x"].dropna())
            assert got == pytest.approx(want, abs=1e-9), (split, name, a, b)


@pytest.mark.slow
def test_on_amsterdam_2012s_elec_sweep_ppo_overtakes_rb_default_above_its_crossover(ran):
    """On amsterdam_2012's elec sweep, ppo's paired gain over rb_default (IQM of ppo's 5 seeds x 8
    seasons minus rb_default's same season) rises through 0 once, from about -6.3 EUR/m2 at 0.02
    EUR/kWh to about +14.0 at 0.40, so its one crossover row says "controller ahead above x": PPO
    wins at electricity prices above it."""
    out, _, _ = ran()
    _, _, crossovers = _csvs(out)
    rows = crossovers[
        (crossovers["split"] == "amsterdam_2012")
        & (crossovers["sweep"] == "elec")
        & (crossovers["controller"] == "ppo")
        & (crossovers["versus"] == "rb_default")
    ]
    assert rows["x"].notna().all() and len(rows) == 1, rows
    assert rows.iloc[0]["note"] == ABOVE


@pytest.mark.slow
def test_sweep_png_draws_lambda_per_test_split_with_display_names(ran):
    """One figure, saved as out/sweep.png: 3 panels (one per test split, each city named), each with
    a line per controller; every controller's lambda-sweep profit IQMs from sweep.csv drawn against
    lambda; site_build.CONTROLLERS' display names shown, never a raw id; every text >= 9 pt; and
    "exploratory, re-priced recorded seasons" on it."""
    out, _, figures = ran()
    _, sweep, _ = _csvs(out)
    saved = [f for f in figures if Path(f["path"]).name == "sweep.png"]
    assert len(saved) == 1, [f["path"] for f in figures]
    fig = saved[0]
    assert len(fig["lines_per_panel"]) == 3
    assert all(n >= 7 for n in fig["lines_per_panel"]), fig["lines_per_panel"]
    shown = [" ".join(t.split()) for t, _ in fig["texts"]]
    assert set(site_build.CONTROLLERS.values()) <= set(shown)
    assert [t for t in shown if t in site_build.CONTROLLERS] == []
    assert [(t, size) for t, size in fig["texts"] if size < 9] == []
    joined = " ".join(shown)
    assert all(city in joined for city in ("Amsterdam", "London", "Abbotsford"))
    assert any(PHRASE in t for t in shown)
    for split in TEST_SPLITS:
        for controller in RUNS:
            rows = sweep[
                (sweep["split"] == split)
                & (sweep["sweep"] == "lambda")
                & (sweep["controller"] == controller)
            ]
            y = rows.sort_values("x")["profit_iqm"].to_numpy()
            drawn = any(
                len(cx) == len(LAMBDAS)
                and np.allclose(cx, LAMBDAS, atol=1e-9)
                and np.allclose(cy, y, atol=1e-9)
                for cx, cy in fig["curves"]
            )
            assert drawn, (split, controller)


# --- No env, no weather, no writes to the record ---------------------------------------------------

# Run in a fresh interpreter: an audit hook records every open of a weather file and every write,
# make, move or delete under control/ledger, control/results or control/runs, then main(--out
# argv[1]); it prints {"code", "modules": sorted(sys.modules), "seen": [...]} as its last line.
_NO_ENV = r"""
import json, os, sys

out, root = sys.argv[1], sys.argv[2]
weather = [os.path.join(root, "control", "data", "weather").casefold()]
guarded = [os.path.join(root, "control", name).casefold() for name in ("ledger", "results", "runs")]
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# Audit event -> the indexes of the args it writes, makes, moves or deletes.
edits = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1),
    "os.truncate": (0,), "os.chmod": (0,), "os.utime": (0,), "os.symlink": (1,), "os.link": (1,),
    "shutil.rmtree": (0,), "shutil.copyfile": (1,), "shutil.copytree": (1,), "shutil.move": (0, 1),
}
seen = []


def under(path, prefixes):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    p = os.path.abspath(os.fsdecode(path)).casefold()
    return any(p == q or p.startswith(q + os.sep) for q in prefixes)


def hook(event, args):
    if event == "open":
        path, mode, flags = args
        if under(path, weather):
            seen.append(["opened weather", os.fsdecode(path)])
        writes = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(flags & write_flags)
        if writes and under(path, guarded):
            seen.append(["opened to write", os.fsdecode(path)])
    elif event in edits:
        for i in edits[event]:
            if i < len(args) and under(args[i], guarded):
                seen.append([event, os.fsdecode(args[i])])


sys.addaudithook(hook)
from greenleaf_control import prices

code = prices.main(["--out", out])
print(json.dumps({"code": code, "modules": sorted(sys.modules), "seen": seen}))
"""


@pytest.mark.slow
def test_a_run_loads_no_env_reads_no_weather_and_writes_no_record(tmp_path):
    """In a fresh interpreter, importing prices and running main into tmp_path returns 0, leaves
    gl_gym, envs, evaluate, report, export_replay, rewards, selection, stable_baselines3 and torch
    out of sys.modules, opens nothing under control/data/weather and writes, makes, moves or
    deletes nothing under control/ledger, control/results or control/runs."""
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-c", _NO_ENV, str(out), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["code"] == 0
    loaded = [m for m in report["modules"] for f in FORBIDDEN if m == f or m.startswith(f + ".")]
    assert loaded == []
    assert report["seen"] == []
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS

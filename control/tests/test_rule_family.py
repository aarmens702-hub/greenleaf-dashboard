"""Tests for greenleaf_control.rule_family (issue #69, P3 of the price-regime study #66): two
deterministic analyses of the rule-controller family on VALIDATION weather (Amsterdam 2010, start
days 45, 59, 75), one command each. Nothing trains and nothing is selected; the 2012/2013 weather is
never opened.

    python -m greenleaf_control.rule_family frontier [--n N] [--seed S] [--season-length D]
        [--out DIR] [--jobs J]
    python -m greenleaf_control.rule_family horizon [--lengths L ...] [--out DIR] [--jobs J]

The conventions pinned here (the orchestrator's design, with the test-writer's defaults).
- OUT = control/results/analysis/rule_family (the default --out of both commands);
  VALIDATION_SCORES = control/results/analysis/validation/scores.parquet, read at call time.
- main(argv: list[str] | None = None) -> int, 0 on success.
- --jobs J: J worker processes. --jobs 1 plays every season in the calling process through
  evaluate.run_rb and envs.run_baseline, looked up on those modules at call time, so a test can
  wrap them. Any --jobs writes the same bytes.
- sample_settings(n, seed) -> list of n + 2 full params dicts: [rule_based.load_rb_params("default"),
  load_rb_params("tuned")], then n draws from one rng = np.random.default_rng(seed); draw i is the
  defaults overlaid with {k: rng.uniform(lo, hi) for k, (lo, hi) in tune_rb.TUNE_SPACE.items()},
  draws taken in that order (the feasibility probe's). A setting's id is its index in that list.
- Re-pricing [EUR/m2], as prices.reprice: profit = yield_kg_m2 x fruit - heat_cost x heat / 0.09 -
  elec_cost x elec / 0.30 - co2_cost x co2 / 0.30, the recorded costs being at the simulator's
  prices, with metrics.PRICE_REGIMES' sim and agc2020. Every season is played once, at sim.
- frontier: every setting of sample_settings(--n, --seed) on the 3 VALIDATION seasons of
  --season-length days via evaluate.run_rb(params, scenario, season_length). Writes into a fresh
  --out exactly frontier_seasons.csv, frontier.csv, frontier_summary.csv, frontier.png, frontier.md:
  - frontier_seasons.csv: one row per setting x season, sorted by setting then start_day, columns
    SEASON_COLUMNS (profit_eur_m2 at sim, as metrics records it). An invalid season is a row with
    valid False: flagged, never dropped.
  - frontier.csv: one row per setting, sorted by setting, columns FRONTIER_COLUMNS: means over its 3
    seasons of profit at each regime (profit_sim, profit_agc2020), yield_kg_m2, viol_hours (each
    season's co2 + temp + rh hours) and heat/elec/co2 cost; valid = every season valid. A setting
    with an invalid season stays in frontier.csv, is excluded from the comparison, and frontier.md
    has a line with "excluded" and how many settings were.
  - frontier_summary.csv: columns SUMMARY_COLUMNS, one row per LABELS entry x regime. The point:
    profit_iqm = the IQM (rliable's, = scipy's trim_mean(x, 0.25) over the label's runs x seasons)
    of its validation seasons re-priced at the regime, yield_iqm, viol_hours_iqm = the sum of the
    three per-constraint IQMs. Then compare's columns.
  - compare(frontier, points) -> points with best_setting, best_profit, gap, n_qualifying and note
    added per row: among frontier's valid settings with yield_kg_m2 >= the row's yield_iqm and
    viol_hours <= its viol_hours_iqm, both within 1e-9 (so a rule baseline's own setting qualifies
    against its own point), the one with the highest profit_<regime>; gap = profit_iqm -
    best_profit (positive: the agent earns more); n_qualifying their count. None: best_setting,
    best_profit and gap NaN, n_qualifying 0, note "no setting qualifies".
  - frontier.md line 1: the command with --n, --seed and --season-length as given, and PHRASE; it
    says the comparison is in-sample for the agents (VALIDATION selected their checkpoints). It
    also says, each on one line: best is in-sample too and optimistic for the rules (a maximum over
    the qualifying settings on the same 3 seasons); gap is descriptive, not an out-of-sample
    estimate; zero_controls and random are left out of the comparison, and why. Its tables name the
    hours column "Σ per-constraint IQMs", as validation's summary.md does.
  - A --season-length with a skip_reason for any VALIDATION season is a ValueError before any
    season is played or --out is made.
- horizon: rb_default and rb_tuned (evaluate.run_rb with load_rb_params) and zero_controls
  (envs.run_baseline, graded by metrics on the raw env's params and N, as evaluate._score) on each
  VALIDATION season at each of --lengths. Writes into a fresh --out exactly horizon.csv, horizon.png,
  horizon.md. horizon.csv: one row per controller x start day x length, columns HORIZON_COLUMNS. A
  cell with skip_reason(scenario, length) is never played: its row has the reason in note and NaN
  numbers; a played cell's note is empty.
  - skip_reason(scenario, season_length) -> str | None: a reason naming day 365 if start_day +
    season_length >= 365, else None. Upstream's loader reads the weather 0.5 day past a season's
    end, so a season ending on day 365 already opens the next year's file (2011 for 2010).
  - horizon.md line 1: the command with --lengths as given, and PHRASE.
- Refuse before write, as prices: every played season and every recorded agent season must re-price
  at sim to its recorded profit within 1e-9 EUR/m2, else ValueError before --out is made. So is a
  LABELS label with no row in VALIDATION_SCORES, or a LABELS season recorded with valid False.
- Figures: matplotlib Figure(layout="constrained"), saved at 200 dpi, every text >= 9 pt, display
  names from site_build.CONTROLLERS (no raw id shown), one panel per regime whose title names it
  ("sim", "agc2020"). frontier.png: valid settings as points at (yield_kg_m2, profit_<regime>) and
  each controller's (yield_iqm, profit_iqm) as a labelled marker, and a footer (a figure-level text)
  saying "in-sample" and "Σ hours not shown"; horizon.png: profit vs season length per controller.
- No clock time and no out path in the .md files, so a rerun into another dir gives the same bytes.

Every check imports greenleaf_control.rule_family inside its test body, so while it is missing each
test FAILS rather than ERRORs. Fast tests play 1-day seasons (96 steps, ~0.5 s each); every output
goes under tmp_path. The 60-day, --jobs > 1 and audit-hook runs are slow.
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
from scipy.stats import trim_mean

from greenleaf_control import envs, evaluate, metrics, rule_based, scenarios, site_build, tune_rb

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "control"
VALIDATION_SCORES = CONTROL / "results" / "analysis" / "validation" / "scores.parquet"
COMMAND = "python -m greenleaf_control.rule_family"
PHRASE = "VALIDATION weather; exploratory analysis; nothing trains or is selected"
DAYS = [45, 59, 75]
REGIMES = ("sim", "agc2020")
# The recorded validation labels compared with the frontier -> their controller ids.
LABELS = {
    "ppo #33/0221": "ppo",
    "ppo_paper #31/2d02": "ppo_paper",
    "residual_w05 #36/f2fd": "residual_w05",
    "lag_b100 #39/6d92": "lag_b100",
    "rb_default": "rb_default",
    "rb_tuned": "rb_tuned",
}
HORIZON_CONTROLLERS = ("rb_default", "rb_tuned", "zero_controls")
METRICS = (
    "profit_eur_m2",
    "yield_kg_m2",
    "heat_cost",
    "elec_cost",
    "co2_cost",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
)
SEASON_COLUMNS = {"setting", "start_day", *METRICS, "valid"}
FRONTIER_COLUMNS = {
    "setting",
    "valid",
    "profit_sim",
    "profit_agc2020",
    "yield_kg_m2",
    "viol_hours",
    "heat_cost",
    "elec_cost",
    "co2_cost",
}
COMPARE_COLUMNS = {"best_setting", "best_profit", "gap", "n_qualifying", "note"}
POINT_COLUMNS = {"controller", "label", "regime", "profit_iqm", "yield_iqm", "viol_hours_iqm"}
SUMMARY_COLUMNS = POINT_COLUMNS | COMPARE_COLUMNS
HORIZON_COLUMNS = {
    "controller",
    "start_day",
    "length",
    "profit_sim",
    "profit_agc2020",
    "yield_kg_m2",
    "viol_hours",
    "valid",
    "note",
}
FRONTIER_OUTPUTS = [
    "frontier.csv",
    "frontier.md",
    "frontier.png",
    "frontier_seasons.csv",
    "frontier_summary.csv",
]
HORIZON_OUTPUTS = ["horizon.csv", "horizon.md", "horizon.png"]
NONE = "no setting qualifies"


# --- Helpers ---------------------------------------------------------------------------------------


def _module():
    """greenleaf_control.rule_family; an assertion, not an ImportError, while it is missing."""
    found = importlib.util.find_spec("greenleaf_control.rule_family")
    assert found is not None, "greenleaf_control.rule_family doesn't exist yet (#69)"
    return importlib.import_module("greenleaf_control.rule_family")


# Re-pricing [EUR/m2] from a season's yield [kg/m2] and its costs at the simulator's prices
# [EUR/m2]: yield x 1.6 - heat x heat price / 0.09 - elec x elec price / 0.30 - co2 x co2 price /
# 0.30. At sim every ratio is 1; at agc2020 heat is x 0.03/0.09 = 1/3, elec x 0.06/0.30 = 0.2 and
# co2 x 0.08/0.30 = 4/15 of its sim cost.
def _at_sim(yield_, heat, elec, co2):
    return yield_ * 1.6 - heat - elec - co2


def _at_agc2020(yield_, heat, elec, co2):
    return yield_ * 1.6 - heat / 3 - elec * 0.2 - co2 * 4 / 15


HAND_PRICE = {"sim": _at_sim, "agc2020": _at_agc2020}


def _repriced(rows: pd.DataFrame, regime: str) -> np.ndarray:
    return HAND_PRICE[regime](
        rows["yield_kg_m2"].to_numpy(dtype=np.float64),
        rows["heat_cost"].to_numpy(dtype=np.float64),
        rows["elec_cost"].to_numpy(dtype=np.float64),
        rows["co2_cost"].to_numpy(dtype=np.float64),
    )


def _hand_draws(n: int, seed: int) -> list[dict]:
    """The n sampled settings written out by hand: defaults overlaid with one uniform draw per
    TUNE_SPACE key, in its order, from one default_rng(seed)."""
    defaults = rule_based.load_rb_params("default")
    rng = np.random.default_rng(seed)
    return [
        {
            **defaults,
            **{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in tune_rb.TUNE_SPACE.items()},
        }
        for _ in range(n)
    ]


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


def _one(frame: pd.DataFrame, **where) -> pd.Series:
    """The one row of frame whose columns equal where's values."""
    hit = frame
    for column, value in where.items():
        hit = hit[hit[column] == value]
    assert len(hit) == 1, f"{len(hit)} rows where {where}, not 1"
    return hit.iloc[0]


def _close(a, b, tol: float = 1e-9) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=tol, atol=tol))


def _same_params(got, want: dict) -> bool:
    """The same keys, each value equal within 1e-12 (a drawn setting may differ in the last bit)."""
    got = dict(got)
    return set(got) == set(want) and all(_close(got[k], want[k], 1e-12) for k in want)


def _patch_run_rb(monkeypatch, change) -> list[dict]:
    """evaluate.run_rb plays the real season, then change(params, scenario, result) may edit a copy
    of its result dict. Returns the list of calls (params, scenario, season_length, price_regime)."""
    real = evaluate.run_rb
    signature = inspect.signature(real)
    calls = []

    def patched(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        given = bound.arguments
        calls.append(
            {
                "params": dict(given["params"]),
                "scenario": dict(given["scenario"]),
                "season_length": given["season_length"],
                "price_regime": given["price_regime"],
            }
        )
        done = real(*args, **kwargs)
        result = dict(done[0])
        change(given["params"], given["scenario"], result)
        return (result, *done[1:])

    monkeypatch.setattr(evaluate, "run_rb", patched)
    return calls


def _figure_facts(fig, args, kwargs) -> dict:
    """What a saved figure shows: its path, its layout engine, every visible non-empty text with its
    size in pt, its footer (fig.text's figure-level texts, the suptitle and legend left out), and
    per panel its titles, every plotted (x, y) (scatter offsets and line data) and its lines."""
    from matplotlib.text import Text

    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    panels = []
    for ax in fig.axes:
        points = []
        for collection in ax.collections:
            offsets = np.asarray(collection.get_offsets(), dtype=np.float64)
            if offsets.ndim == 2 and offsets.shape[1] == 2:
                points += [tuple(p) for p in offsets]
        lines = [
            (
                np.asarray(line.get_xdata(), dtype=np.float64),
                np.asarray(line.get_ydata(), dtype=np.float64),
            )
            for line in ax.get_lines()
        ]
        for x, y in lines:
            points += list(zip(x.ravel(), y.ravel(), strict=True))
        panels.append(
            {
                "title": " ".join(ax.get_title(loc=loc) for loc in ("left", "center", "right")),
                "points": np.array(points, dtype=np.float64).reshape(-1, 2),
                "lines": lines,
            }
        )
    engine = fig.get_layout_engine()
    return {
        "path": str(args[0] if args else kwargs.get("fname")),
        "layout": type(engine).__name__ if engine is not None else "",
        "texts": [(t.get_text(), float(t.get_fontsize())) for t in texts],
        "footer": [
            " ".join(t.get_text().split())
            for t in fig.texts
            if t.get_visible() and t.get_text() != fig.get_suptitle()
        ],
        "panels": panels,
    }


def _run_with_spy(rule_family, argv: list[str]) -> tuple[int, list[dict]]:
    """main(argv) with Figure.savefig wrapped to record each figure after the real save."""
    from matplotlib.figure import Figure

    figures = []
    real = Figure.savefig

    def spy(self, *args, **kwargs):
        done = real(self, *args, **kwargs)
        figures.append(_figure_facts(self, args, kwargs))
        return done

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "savefig", spy)
        code = rule_family.main(argv)
    return code, figures


@pytest.fixture(scope="module")
def frontier_run(tmp_path_factory):
    """A function returning (out, main's return value, the figures it saved): its first call runs
    `frontier --n 3 --seed 0 --season-length 1 --jobs 1` into a fresh, missing dir (5 settings x 3
    one-day seasons); later calls return the same. Called inside each test body, so a missing
    module FAILS the test."""
    cache = []

    def get() -> tuple[Path, int, list[dict]]:
        if not cache:
            rule_family = _module()
            out = tmp_path_factory.mktemp("frontier") / "out"
            argv = ["frontier", "--n", "3", "--seed", "0", "--season-length", "1"]
            code, figures = _run_with_spy(rule_family, [*argv, "--jobs", "1", "--out", str(out)])
            cache.append((out, code, figures))
        return cache[0]

    return get


@pytest.fixture(scope="module")
def horizon_run(tmp_path_factory):
    """A function returning (out, main's return value, the figures it saved): its first call runs
    `horizon --lengths 1 400 --jobs 1` into a fresh, missing dir. Every 400-day cell runs past day
    365 (45 + 400 > 365), so it is skipped, never played; the 1-day cells are 3 controllers x 3
    seasons. Later calls return the same."""
    cache = []

    def get() -> tuple[Path, int, list[dict]]:
        if not cache:
            rule_family = _module()
            out = tmp_path_factory.mktemp("horizon") / "out"
            argv = ["horizon", "--lengths", "1", "400", "--jobs", "1", "--out", str(out)]
            code, figures = _run_with_spy(rule_family, argv)
            cache.append((out, code, figures))
        return cache[0]

    return get


def _frontier_csvs(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(out / "frontier_seasons.csv"),
        pd.read_csv(out / "frontier.csv"),
        pd.read_csv(out / "frontier_summary.csv"),
    )


def _validation_rows(label: str) -> pd.DataFrame:
    scores = pd.read_parquet(VALIDATION_SCORES)
    rows = scores[scores["label"] == label]
    assert len(rows) > 0, label
    return rows


# --- The surface -----------------------------------------------------------------------------------


def test_surface_and_paths_match_the_design():
    """main(argv=None); sample_settings(n, seed); compare(frontier, points); skip_reason(scenario,
    season_length); OUT and VALIDATION_SCORES are the analysis dir and validation's scores."""
    rule_family = _module()
    main = inspect.signature(rule_family.main).parameters
    assert list(main) == ["argv"]
    assert main["argv"].default is None
    assert list(inspect.signature(rule_family.sample_settings).parameters) == ["n", "seed"]
    assert list(inspect.signature(rule_family.compare).parameters) == ["frontier", "points"]
    skip = inspect.signature(rule_family.skip_reason).parameters
    assert list(skip) == ["scenario", "season_length"]
    assert Path(rule_family.OUT) == CONTROL / "results" / "analysis" / "rule_family"
    assert Path(rule_family.VALIDATION_SCORES) == VALIDATION_SCORES


# --- (a) The sampled settings --------------------------------------------------------------------


@pytest.mark.parametrize(("n", "seed"), [(0, 0), (3, 0), (5, 0), (5, 7)])
def test_sample_settings_are_default_tuned_then_seeded_uniform_draws(n, seed):
    """n + 2 full params dicts: the default and tuned rules, then n draws over TUNE_SPACE exactly as
    the probe made them (defaults overlaid, one uniform per key in its order, one default_rng(seed)).
    Every drawn value is inside its [lo, hi]; every other key stays at its default; a second call
    gives the same settings."""
    rule_family = _module()
    settings = rule_family.sample_settings(n, seed)
    defaults = rule_based.load_rb_params("default")
    assert len(settings) == n + 2
    assert settings[0] == defaults
    assert settings[1] == rule_based.load_rb_params("tuned")
    hand = _hand_draws(n, seed)
    for i, (got, want) in enumerate(zip(settings[2:], hand, strict=True), start=2):
        assert set(got) == set(defaults), i
        for key, value in got.items():
            if key in tune_rb.TUNE_SPACE:
                lo, hi = tune_rb.TUNE_SPACE[key]
                assert lo <= value <= hi, (i, key, value)
                assert _close(value, want[key], 1e-12), (i, key, value, want[key])
            else:
                assert value == defaults[key], (i, key)
    assert rule_family.sample_settings(n, seed) == settings


def test_sample_settings_differ_by_seed_and_share_their_first_draws():
    """Seeds 0 and 1 draw different settings; sample_settings(3, 0)'s draws are the first 3 of
    sample_settings(5, 0)'s (one stream, setting by setting)."""
    rule_family = _module()
    five = rule_family.sample_settings(5, 0)
    assert five[2:] != rule_family.sample_settings(5, 1)[2:]
    assert rule_family.sample_settings(3, 0) == five[:5]


# --- (b) frontier on 1-day seasons ---------------------------------------------------------------


def test_frontier_writes_exactly_its_five_outputs_with_the_pinned_columns(frontier_run):
    """main returns 0; out holds exactly frontier_seasons.csv (5 settings x 3 seasons = 15 rows,
    sorted by setting then start day), frontier.csv (5 rows, settings 0..4), frontier_summary.csv
    (6 labels x 2 regimes = 12 rows), frontier.png (a PNG) and frontier.md, each CSV with exactly
    its pinned columns."""
    out, code, _ = frontier_run()
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == FRONTIER_OUTPUTS
    seasons, frontier, summary = _frontier_csvs(out)
    assert set(seasons.columns) == SEASON_COLUMNS
    assert set(frontier.columns) == FRONTIER_COLUMNS
    assert set(summary.columns) == SUMMARY_COLUMNS
    keys = list(zip(seasons["setting"], seasons["start_day"], strict=True))
    assert keys == [(s, d) for s in range(5) for d in DAYS]
    assert frontier["setting"].tolist() == [0, 1, 2, 3, 4]
    assert len(summary) == len(LABELS) * len(REGIMES)
    got = sorted(zip(summary["label"], summary["controller"], summary["regime"], strict=True))
    assert got == sorted((label, c, r) for label, c in LABELS.items() for r in REGIMES)
    assert (out / "frontier.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    ("setting", "params"),
    [
        pytest.param(0, lambda: rule_based.load_rb_params("default"), id="0-default"),
        pytest.param(1, lambda: rule_based.load_rb_params("tuned"), id="1-tuned"),
        pytest.param(2, lambda: _hand_draws(3, 0)[0], id="2-first-draw"),
        pytest.param(4, lambda: _hand_draws(3, 0)[2], id="4-third-draw"),
    ],
)
def test_frontier_seasons_equal_run_rbs_for_the_same_setting_and_season(
    frontier_run, setting, params
):
    """Each of the setting's 3 rows in frontier_seasons.csv equals evaluate.run_rb(its params,
    scenario, 1) on that season: profit, yield, the three costs and the three violation hours
    (float round trip through the CSV), and valid."""
    out, _, _ = frontier_run()
    seasons, _, _ = _frontier_csvs(out)
    for s in scenarios.VALIDATION:
        want = evaluate.run_rb(params(), s, 1)[0]
        row = _one(seasons, setting=setting, start_day=s["start_day"])
        for m in METRICS:
            assert _close(row[m], want[m], 1e-12), (setting, s["start_day"], m, row[m], want[m])
        assert bool(row["valid"]) is want["valid"]


# --- (c) Means and re-pricing ----------------------------------------------------------------------


def test_frontier_means_and_repricing_are_hand_computed(frontier_run):
    """Every season re-priced at sim by hand (yield x 1.6 - heat - elec - co2) is its recorded
    profit within 1e-9. Each setting's frontier.csv row is the mean of its 3 seasons: profit_sim of
    the recorded profits, profit_agc2020 of yield x 1.6 - heat/3 - elec x 0.2 - co2 x 4/15,
    yield_kg_m2, viol_hours of co2 + temp + rh hours, each cost; valid = all 3 valid. Each cost term
    is nonzero in some season, so a wrong price on any term shows."""
    out, _, _ = frontier_run()
    seasons, frontier, _ = _frontier_csvs(out)
    for cost in ("heat_cost", "elec_cost", "co2_cost"):
        assert (seasons[cost] > 1e-6).any(), cost
    at_sim = _repriced(seasons, "sim")
    assert np.allclose(at_sim, seasons["profit_eur_m2"], rtol=0, atol=1e-9)
    for setting in range(5):
        mine = seasons[seasons["setting"] == setting]
        row = _one(frontier, setting=setting)
        hours = mine["viol_hours_co2"] + mine["viol_hours_temp"] + mine["viol_hours_rh"]
        assert _close(row["profit_sim"], mine["profit_eur_m2"].mean())
        assert _close(row["profit_agc2020"], _repriced(mine, "agc2020").mean())
        assert _close(row["yield_kg_m2"], mine["yield_kg_m2"].mean())
        assert _close(row["viol_hours"], hours.mean())
        for cost in ("heat_cost", "elec_cost", "co2_cost"):
            assert _close(row[cost], mine[cost].mean()), (setting, cost)
        assert bool(row["valid"]) is bool(mine["valid"].all())
    # agc2020's energy is cheaper, so every setting with any cost earns more there.
    assert (frontier["profit_agc2020"] - frontier["profit_sim"] > 1e-6).all()


def test_frontier_summary_points_are_iqms_of_the_recorded_validation_seasons(frontier_run):
    """Each label's point in frontier_summary.csv, per regime, from its recorded validation rows:
    profit_iqm = trim_mean(the rows re-priced by hand at the regime, 0.25), yield_iqm =
    trim_mean(yield), viol_hours_iqm = the sum of the 3 per-constraint trim_means. (rb_default at
    sim, IQM of 3 seasons = their mean: -7.4136 EUR/m2, 11.338 kg/m2, 743.17 h; no 1-day setting
    reaches the agents' fruit, so every row has no qualifying setting.)"""
    out, _, _ = frontier_run()
    _, _, summary = _frontier_csvs(out)
    for label, controller in LABELS.items():
        rows = _validation_rows(label)
        hours = sum(trim_mean(rows[f"viol_hours_{c}"], 0.25) for c in ("co2", "temp", "rh"))
        for regime in REGIMES:
            row = _one(summary, label=label, regime=regime)
            assert row["controller"] == controller
            assert _close(row["profit_iqm"], trim_mean(_repriced(rows, regime), 0.25)), (
                label,
                regime,
            )
            assert _close(row["yield_iqm"], trim_mean(rows["yield_kg_m2"], 0.25))
            assert _close(row["viol_hours_iqm"], hours)
            assert pd.isna(row["best_setting"]) and pd.isna(row["best_profit"])
            assert pd.isna(row["gap"])
            assert int(row["n_qualifying"]) == 0
            assert NONE in str(row["note"])
    default = _one(summary, label="rb_default", regime="sim")
    assert _close(default["profit_iqm"], -7.413559647638915)  # mean of its 3 recorded seasons


# --- (d) The comparison on a hand-built frame ------------------------------------------------------

# (setting, valid, fruit [kg/m2], Σ hours [h], profit at sim, profit at agc2020 [EUR/m2]), listed
# out of setting order so a positional pick is caught.
HAND_SETTINGS = [
    (3, True, 11.0, 100.5, 50.0, 50.0),  # 0.5 h over a's 100: never a's
    (5, True, 14.0, 90.0, 2.0, 11.5),
    (0, True, 9.9, 50.0, 100.0, 100.0),  # 0.1 kg under a's 10: never a's
    (4, False, 15.0, 20.0, 200.0, 200.0),  # dominates everything, but has an invalid season
    (1, True, 10.0, 100.0, 3.0, 11.0),  # exactly a's fruit and hours: qualifies
    (2, True, 12.0, 80.0, 4.0, 10.0),
]
# (controller, fruit IQM, Σ hours IQM, profit IQM at sim, at agc2020)
HAND_POINTS = [
    ("a", 10.0, 100.0, 5.0, 12.0),  # settings 1, 2, 5 qualify
    ("b", 20.0, 10.0, 1.0, 1.0),  # none (4 is invalid, and short of 20 kg anyway)
    ("c", 12.0, 80.0, 4.0, 9.0),  # only 2, exactly on both bounds
    ("d", 14.0 + 1e-12, 90.0 - 1e-12, 3.0, 3.0),  # only 5, 1e-12 off both bounds: within 1e-9
    ("e", 0.0, 1e6, -1.0, -1.0),  # every valid setting: 0, 1, 2, 3, 5
]
# (controller, regime, best setting, best profit, gap = profit IQM - best profit, n qualifying)
HAND_COMPARE = [
    ("a", "sim", 2, 4.0, 1.0, 3),  # 1, 2, 5 at sim: 3, 4, 2 -> 2; 5 - 4
    ("a", "agc2020", 5, 11.5, 0.5, 3),  # at agc2020: 11, 10, 11.5 -> 5; 12 - 11.5
    ("b", "sim", None, np.nan, np.nan, 0),
    ("b", "agc2020", None, np.nan, np.nan, 0),
    ("c", "sim", 2, 4.0, 0.0, 1),  # 4 - 4
    ("c", "agc2020", 2, 10.0, -1.0, 1),  # 9 - 10
    ("d", "sim", 5, 2.0, 1.0, 1),  # 3 - 2
    ("d", "agc2020", 5, 11.5, -8.5, 1),  # 3 - 11.5
    ("e", "sim", 0, 100.0, -101.0, 5),  # not 4 (200, invalid); -1 - 100
    ("e", "agc2020", 0, 100.0, -101.0, 5),
]


def _hand_frontier() -> pd.DataFrame:
    """HAND_SETTINGS as frontier.csv rows (costs 0: compare reads profit, fruit and hours)."""
    return pd.DataFrame(
        [
            {
                "setting": s,
                "valid": valid,
                "profit_sim": sim,
                "profit_agc2020": agc,
                "yield_kg_m2": fruit,
                "viol_hours": hours,
                "heat_cost": 0.0,
                "elec_cost": 0.0,
                "co2_cost": 0.0,
            }
            for s, valid, fruit, hours, sim, agc in HAND_SETTINGS
        ]
    )


def _hand_points() -> pd.DataFrame:
    """HAND_POINTS as one row per controller x regime, label the controller upper-cased."""
    return pd.DataFrame(
        [
            {
                "controller": c,
                "label": c.upper(),
                "regime": regime,
                "profit_iqm": profit,
                "yield_iqm": fruit,
                "viol_hours_iqm": hours,
            }
            for c, fruit, hours, sim, agc in HAND_POINTS
            for regime, profit in (("sim", sim), ("agc2020", agc))
        ]
    )


@pytest.mark.parametrize(
    ("controller", "regime", "best", "best_profit", "gap", "n"),
    HAND_COMPARE,
    ids=[f"{c}-{r}" for c, r, *_ in HAND_COMPARE],
)
def test_compare_picks_the_most_profitable_qualifying_valid_setting(
    controller, regime, best, best_profit, gap, n
):
    """compare(frontier, points) keeps each point row (label passed through) and adds the most
    profitable valid setting, at that row's regime, with fruit >= and Σ hours <= the point's
    (inclusive, within 1e-9), its profit, gap = profit IQM - its profit [EUR/m2], and the count; no
    setting qualifying gives NaN, 0 and "no setting qualifies"."""
    rule_family = _module()
    done = rule_family.compare(_hand_frontier(), _hand_points())
    assert COMPARE_COLUMNS | POINT_COLUMNS <= set(done.columns)
    assert len(done) == len(HAND_POINTS) * 2
    row = _one(done, controller=controller, regime=regime)
    assert row["label"] == controller.upper()
    assert int(row["n_qualifying"]) == n
    if best is None:
        assert pd.isna(row["best_setting"])
        assert pd.isna(row["best_profit"]) and pd.isna(row["gap"])
        assert NONE in str(row["note"])
    else:
        assert int(row["best_setting"]) == best
        assert _close(row["best_profit"], best_profit, 1e-12)
        assert _close(row["gap"], gap, 1e-12)


# --- (e) horizon -----------------------------------------------------------------------------------


def test_horizon_writes_a_row_per_cell_and_skips_the_cells_past_day_365(horizon_run):
    """main returns 0; out holds exactly horizon.csv, horizon.md and horizon.png; horizon.csv has
    exactly its columns and one row per controller x start day x length (3 x 3 x 2 = 18). The
    400-day cells (45 + 400 > 365) are rows with a reason naming 365 in note and NaN numbers; the
    1-day cells have an empty note and are valid."""
    out, code, _ = horizon_run()
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == HORIZON_OUTPUTS
    horizon = pd.read_csv(out / "horizon.csv")
    assert set(horizon.columns) == HORIZON_COLUMNS
    cells = sorted(zip(horizon["controller"], horizon["start_day"], horizon["length"], strict=True))
    assert cells == sorted((c, d, n) for c in HORIZON_CONTROLLERS for d in DAYS for n in (1, 400))
    numbers = ["profit_sim", "profit_agc2020", "yield_kg_m2", "viol_hours"]
    skipped = horizon[horizon["length"] == 400]
    assert skipped["note"].notna().all()
    assert skipped["note"].astype(str).str.contains("365").all()
    assert skipped[numbers].isna().all().all()
    played = horizon[horizon["length"] == 1]
    assert played["note"].isna().all() or (played["note"].fillna("") == "").all()
    assert played[numbers].notna().all().all()
    assert played["valid"].astype(bool).all()


def _reference(controller: str, scenario: dict, days: int) -> dict:
    """The season's metrics as evaluate grades it: run_rb for the rules; for zero_controls,
    envs.run_baseline's infos graded on the raw env's params and N, as evaluate._score."""
    if controller != "zero_controls":
        return evaluate.run_rb(
            rule_based.load_rb_params(controller.removeprefix("rb_")), scenario, days
        )[0]
    infos = envs.run_baseline("zero_controls", scenario, days)
    env = envs.make_raw_env(scenario=scenario, season_length=days, normalize_actions=False)
    graded = metrics.episode_metrics(
        *metrics.trace_from_infos(infos), np.asarray(env.unwrapped.p), season_steps=env.unwrapped.N
    )
    env.close()
    return graded


@pytest.mark.parametrize("controller", HORIZON_CONTROLLERS)
def test_horizon_one_day_cells_equal_the_graded_season(horizon_run, controller):
    """Each 1-day cell of horizon.csv is the controller's season as evaluate grades it: profit_sim
    its profit, profit_agc2020 its yield and costs re-priced by hand (yield x 1.6 - heat/3 - elec x
    0.2 - co2 x 4/15), its yield, viol_hours its co2 + temp + rh hours, valid."""
    out, _, _ = horizon_run()
    horizon = pd.read_csv(out / "horizon.csv")
    for s in scenarios.VALIDATION:
        want = _reference(controller, s, 1)
        row = _one(horizon, controller=controller, start_day=s["start_day"], length=1)
        agc = _at_agc2020(
            want["yield_kg_m2"], want["heat_cost"], want["elec_cost"], want["co2_cost"]
        )
        hours = want["viol_hours_co2"] + want["viol_hours_temp"] + want["viol_hours_rh"]
        assert _close(row["profit_sim"], want["profit_eur_m2"], 1e-12)
        assert _close(row["profit_agc2020"], agc)
        assert _close(row["yield_kg_m2"], want["yield_kg_m2"], 1e-12)
        assert _close(row["viol_hours"], hours, 1e-12)
        assert bool(row["valid"]) is want["valid"]


# (start day, season length [days], skipped): the full-size grid is 60..300 days from 45, 59, 75;
# only 75 + 300 = 375 runs past day 365 (59 + 300 = 359 and 75 + 240 = 315 fit). Upstream reads the
# weather 0.5 day past a season's end, so a season ending on day 365 (start + length = 365) already
# opens the next year's file: the year-end rows below are upstream's known answer (see the next
# test).
SKIPS = [
    (45, 300, False),
    (59, 300, False),
    (75, 240, False),
    (75, 300, True),
    (45, 400, True),  # 445
    (59, 1, False),
    (75, 289, False),  # ends on day 364: 364 + 0.5 day of lookahead fits in 2010
    (75, 290, True),  # ends on day 365: the 0.5 day of lookahead reads 2011
    (59, 306, True),  # 365, as above
]


@pytest.mark.parametrize(("day", "length", "skipped"), SKIPS)
def test_skip_reason_names_day_365_only_for_a_season_past_it(day, length, skipped):
    """skip_reason(the VALIDATION season starting on day, length) is a reason naming 365 when start
    day + length >= 365 (the season plus upstream's 0.5-day lookahead runs past the 2010 file), else
    None."""
    rule_family = _module()
    [scenario] = [s for s in scenarios.VALIDATION if s["start_day"] == day]
    reason = rule_family.skip_reason(scenario, length)
    if skipped:
        assert isinstance(reason, str) and "365" in reason, reason
    else:
        assert reason is None, reason


@pytest.mark.parametrize(("day", "length"), [(75, 289), (75, 290), (59, 306)])
def test_skip_reason_is_set_exactly_when_upstream_opens_the_next_years_weather(
    day, length, monkeypatch
):
    """Known answer from upstream's own loader: gl_gym's load_weather_data, called as the env's
    weather repository calls it (the registered env's pred_horizon 0.5 day, dt 900 s as h, nd 10),
    reads Amsterdam/2010.csv and, when the season plus its 0.5-day lookahead runs past that file,
    Amsterdam/2011.csv too. From day 75, 289 days (ending on day 364) reads only 2010; 290 days
    (ending on day 365) reads 2011; from day 59, 306 days (365) reads 2011. skip_reason must be set
    exactly for the seasons that read 2011. pandas.read_csv is wrapped, not replaced: 2011 is TRAIN
    weather, and any other file fails the test before it is opened."""
    import gymnasium as gym
    from gl_gym.environments.utils import load_weather_data

    rule_family = _module()
    allowed = {(scenarios.WEATHER_DIR / "Amsterdam" / f"{y}.csv").resolve() for y in (2010, 2011)}
    real = pd.read_csv
    opened = []

    def spy(path, *args, **kwargs):
        assert Path(path).resolve() in allowed, f"upstream tried to open {path}"
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    spec = gym.spec(envs.ENV_ID).kwargs
    [scenario] = [s for s in scenarios.VALIDATION if s["start_day"] == day]
    load_weather_data(
        str(scenarios.WEATHER_DIR),
        scenario["location"],
        scenario["growth_year"],
        day,
        length,
        spec["pred_horizon"],
        float(spec["dt"]),
        spec["nd"],
    )
    monkeypatch.undo()
    assert opened[0] == "2010.csv", opened
    reads_2011 = "2011.csv" in opened
    reason = rule_family.skip_reason(scenario, length)
    assert reads_2011 is (reason is not None), (day, length, opened, reason)


def test_the_full_size_horizon_skips_exactly_day_75_at_300_days():
    """Of the 5 x 3 full-size cells (60, 120, 180, 240, 300 days from 45, 59, 75), skip_reason
    skips exactly (75, 300): 300-day seasons fit in the 2010 weather year from 45 and 59."""
    rule_family = _module()
    skipped = [
        (s["start_day"], n)
        for s in scenarios.VALIDATION
        for n in (60, 120, 180, 240, 300)
        if rule_family.skip_reason(s, n) is not None
    ]
    assert skipped == [(75, 300)]


# --- (f) Invalid seasons and refusals --------------------------------------------------------------


def test_an_invalid_season_setting_is_flagged_counted_and_kept(tmp_path, monkeypatch):
    """With run_rb's day-59 season of setting 2 (the first draw) marked invalid (a solver failure,
    its numbers unchanged): main returns 0, every season is played once at sim (3 settings x 3 =
    9 run_rb calls of 1 day); the season stays in frontier_seasons.csv with valid False; setting 2
    stays in frontier.csv with valid False, the others True; frontier.md counts 1 setting
    excluded."""
    rule_family = _module()
    first_draw = _hand_draws(1, 0)[0]

    def invalidate(params, scenario, result):
        if _same_params(params, first_draw) and scenario["start_day"] == 59:
            result["valid"] = False
            result["solver_failures"] = 1

    calls = _patch_run_rb(monkeypatch, invalidate)
    out = tmp_path / "out"
    argv = ["frontier", "--n", "1", "--seed", "0", "--season-length", "1", "--jobs", "1"]
    assert rule_family.main([*argv, "--out", str(out)]) == 0
    assert len(calls) == 9
    assert all(c["season_length"] == 1 and c["price_regime"] == "sim" for c in calls)
    seasons, frontier, _ = _frontier_csvs(out)
    mine = seasons[seasons["setting"] == 2].set_index("start_day")["valid"].astype(bool)
    assert mine.to_dict() == {45: True, 59: False, 75: True}
    assert frontier.set_index("setting")["valid"].astype(bool).to_dict() == {
        0: True,
        1: True,
        2: False,
    }
    text = (out / "frontier.md").read_text(encoding="utf-8")
    counted = [line for line in text.splitlines() if "excluded" in line]
    assert any(re.search(r"\b1\b", line) for line in counted), counted


@pytest.mark.parametrize(
    ("argv", "rules", "day"),
    [
        pytest.param(
            ["frontier", "--n", "0", "--seed", "0", "--season-length", "1"],
            "tuned",
            75,
            id="frontier-tuned-day-75",
        ),
        pytest.param(["horizon", "--lengths", "1"], "default", 45, id="horizon-default-day-45"),
    ],
)
def test_a_played_season_that_does_not_reprice_is_refused_before_anything_is_written(
    argv, rules, day, tmp_path, monkeypatch
):
    """One played season's profit + 1e-6 EUR/m2 (its yield and costs unchanged) no longer re-prices
    at sim to its recorded profit: main raises ValueError, out is never made and nothing under
    tmp_path changes."""
    rule_family = _module()
    target = rule_based.load_rb_params(rules)

    def nudge(params, scenario, result):
        if _same_params(params, target) and scenario["start_day"] == day:
            result["profit_eur_m2"] += 1e-6

    calls = _patch_run_rb(monkeypatch, nudge)
    out = tmp_path / "out"
    before = _tree(tmp_path)
    with pytest.raises(ValueError) as refused:
        rule_family.main([*argv, "--jobs", "1", "--out", str(out)])
    assert str(refused.value).strip()
    assert calls, "evaluate.run_rb was never called in this process"
    assert not out.exists()
    assert _tree(tmp_path) == before


def test_a_recorded_agent_season_that_does_not_reprice_is_refused(tmp_path, monkeypatch):
    """validation's scores with one residual_w05 #36/f2fd season's profit + 1e-6 EUR/m2: frontier
    raises ValueError, out is never made and nothing under tmp_path changes."""
    rule_family = _module()
    scores = pd.read_parquet(VALIDATION_SCORES)
    i = int(np.flatnonzero((scores["label"] == "residual_w05 #36/f2fd").to_numpy())[0])
    scores.iloc[i, scores.columns.get_loc("profit_eur_m2")] += 1e-6
    nudged = tmp_path / "scores.parquet"
    scores.to_parquet(nudged)
    monkeypatch.setattr(rule_family, "VALIDATION_SCORES", nudged)
    out = tmp_path / "out"
    before = _tree(tmp_path)
    argv = ["frontier", "--n", "0", "--seed", "0", "--season-length", "1", "--jobs", "1"]
    with pytest.raises(ValueError) as refused:
        rule_family.main([*argv, "--out", str(out)])
    assert str(refused.value).strip()
    assert not out.exists()
    assert _tree(tmp_path) == before


def _nudged_scores(tmp_path: Path, change) -> Path:
    """validation's scores with change(scores) applied, as tmp_path/scores.parquet."""
    scores = pd.read_parquet(VALIDATION_SCORES)
    nudged = tmp_path / "scores.parquet"
    change(scores).to_parquet(nudged)
    return nudged


def _drop(label):
    return lambda scores: scores[scores["label"] != label]


def _invalidate_first(label):
    def change(scores):
        i = int(np.flatnonzero((scores["label"] == label).to_numpy())[0])
        scores.iloc[i, scores.columns.get_loc("valid")] = False
        return scores

    return change


@pytest.mark.parametrize("label", list(LABELS))
@pytest.mark.parametrize(
    "change", [_drop, _invalidate_first], ids=["label-missing", "season-invalid"]
)
def test_a_missing_label_or_an_invalid_recorded_season_is_refused(
    tmp_path, monkeypatch, change, label
):
    """validation's scores with every row of one LABELS label removed, or with that label's first
    season recorded valid False (its numbers unchanged, so it still re-prices): frontier raises a
    ValueError naming the label, out is never made and nothing under tmp_path changes."""
    rule_family = _module()
    monkeypatch.setattr(rule_family, "VALIDATION_SCORES", _nudged_scores(tmp_path, change(label)))
    out = tmp_path / "out"
    before = _tree(tmp_path)
    argv = ["frontier", "--n", "0", "--seed", "0", "--season-length", "1", "--jobs", "1"]
    with pytest.raises(ValueError) as refused:
        rule_family.main([*argv, "--out", str(out)])
    assert label in str(refused.value), str(refused.value)
    assert not out.exists()
    assert _tree(tmp_path) == before


def test_a_frontier_season_length_ending_on_day_365_is_refused_before_play(tmp_path, monkeypatch):
    """`frontier --season-length 290`: day 75 + 290 = day 365, whose 0.5-day lookahead reads 2011
    (see the skip_reason tests), so main raises a ValueError naming 365 before any season is played
    (evaluate.run_rb and envs.run_baseline are never called), out is never made and nothing under
    tmp_path changes."""
    rule_family = _module()

    def never(*args, **kwargs):
        raise AssertionError("a 290-day season was played instead of refused")

    monkeypatch.setattr(evaluate, "run_rb", never)
    monkeypatch.setattr(envs, "run_baseline", never)
    out = tmp_path / "out"
    before = _tree(tmp_path)
    argv = ["frontier", "--n", "0", "--seed", "0", "--season-length", "290", "--jobs", "1"]
    with pytest.raises(ValueError) as refused:
        rule_family.main([*argv, "--out", str(out)])
    assert "365" in str(refused.value), str(refused.value)
    assert not out.exists()
    assert _tree(tmp_path) == before


# --- The write-ups ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "flags"),
    [
        ("frontier", ["--n 3", "--seed 0", "--season-length 1"]),
        ("horizon", ["--lengths 1 400"]),
    ],
)
def test_md_line_one_is_the_command_and_what_it_is(frontier_run, horizon_run, command, flags):
    """<command>.md's first line names `python -m greenleaf_control.rule_family <command>` with its
    flags as given and says "VALIDATION weather; exploratory analysis; nothing trains or is
    selected"; no clock time and no out path anywhere. frontier.md says the agents' side is
    in-sample and counts 0 settings excluded; horizon.md gives the day-365 reason."""
    out, _, _ = (frontier_run if command == "frontier" else horizon_run)()
    text = (out / f"{command}.md").read_text(encoding="utf-8")
    first = text.splitlines()[0]
    assert f"{COMMAND} {command}" in first
    assert all(flag in first for flag in flags), first
    assert PHRASE in first
    assert str(out) not in text and str(out.resolve()) not in text
    assert re.search(r"\b\d{1,2}:\d{2}\b", text) is None
    if command == "frontier":
        assert "in-sample" in text
        counted = [line for line in text.splitlines() if "excluded" in line]
        assert any(re.search(r"\b0\b", line) for line in counted), counted
    else:
        assert "365" in text


# (what frontier.md must say, the words one of its lines must hold, case-insensitive; each inner
# tuple is alternatives). random has no display name; zero_controls may be named by its own.
FRONTIER_MD_PINS = [
    pytest.param(
        [("best",), ("in-sample",), ("optimistic",)],
        id="best-is-in-sample-and-optimistic-for-the-rules",
    ),
    pytest.param(
        [("gap",), ("descriptive",), ("not an out-of-sample",)],
        id="gap-is-descriptive-not-an-out-of-sample-estimate",
    ),
    pytest.param(
        [
            ("zero_controls", site_build.CONTROLLERS["zero_controls"].lower()),
            ("random",),
            ("left out",),
        ],
        id="zero-controls-and-random-are-left-out",
    ),
]


@pytest.mark.parametrize("words", FRONTIER_MD_PINS)
def test_frontier_md_says_what_best_and_gap_are_not(frontier_run, words):
    """frontier.md is honest about its own numbers, each on one line: best is in-sample too (a
    maximum over the qualifying settings on the same 3 VALIDATION seasons, optimistic for the
    rules, more so as n_qualifying grows); gap is descriptive, not an out-of-sample estimate;
    zero_controls and random are left out of the comparison (and why)."""
    out, _, _ = frontier_run()
    lines = (out / "frontier.md").read_text(encoding="utf-8").lower().splitlines()
    hits = [line for line in lines if all(any(w in line for w in alts) for alts in words)]
    assert hits, f"no line of frontier.md holds all of {words}"


def test_frontier_md_names_the_hours_column_sum_of_per_constraint_iqms(frontier_run):
    """Every frontier.md table header names the hours column "Σ per-constraint IQMs", as
    validation's summary.md does: it is the sum of 3 IQMs, not an IQM of Σ hours."""
    out, _, _ = frontier_run()
    text = (out / "frontier.md").read_text(encoding="utf-8")
    headers = [line for line in text.splitlines() if line.startswith("| controller")]
    assert len(headers) == len(REGIMES), headers
    assert all("Σ per-constraint IQMs" in h for h in headers), headers


# --- (g) Figures -----------------------------------------------------------------------------------


def _saved(figures: list[dict], name: str) -> dict:
    saved = [f for f in figures if Path(f["path"]).name == name]
    assert len(saved) == 1, [f["path"] for f in figures]
    return saved[0]


def _regime_panels(fig: dict) -> dict[str, dict]:
    """The panel whose title names agc2020 and the other whose title names sim."""
    agc = [p for p in fig["panels"] if "agc2020" in p["title"]]
    sim = [p for p in fig["panels"] if "sim" in p["title"] and "agc2020" not in p["title"]]
    assert len(agc) == 1 and len(sim) == 1, [p["title"] for p in fig["panels"]]
    return {"sim": sim[0], "agc2020": agc[0]}


def _drawn(panel: dict, x: float, y: float) -> bool:
    pts = panel["points"]
    hit = np.isclose(pts[:, 0], x, rtol=1e-9, atol=1e-9) & np.isclose(
        pts[:, 1], y, rtol=1e-9, atol=1e-9
    )
    return bool(hit.any())


def _readable(fig: dict, out: Path, name: str, shown_ids: list[str]) -> None:
    """Constrained layout, 200 dpi, every text >= 9 pt, each id's display name in some text and no
    raw controller id as a text."""
    from PIL import Image

    assert "Constrained" in fig["layout"], fig["layout"]
    with Image.open(out / name) as image:
        dpi = image.info.get("dpi")
    assert dpi is not None and round(dpi[0]) == 200 and round(dpi[1]) == 200, dpi
    assert [(t, size) for t, size in fig["texts"] if size < 9] == []
    shown = [" ".join(t.split()) for t, _ in fig["texts"]]
    missing = [c for c in shown_ids if not any(site_build.CONTROLLERS[c] in t for t in shown)]
    assert missing == [], (missing, shown)
    assert [t for t in shown if t in site_build.CONTROLLERS] == []


def test_frontier_png_draws_settings_and_labelled_points_per_regime(frontier_run):
    """frontier.png, saved once: a sim and an agc2020 panel; each draws every valid setting at
    (yield_kg_m2, profit_<regime>) from frontier.csv and each label's (yield_iqm, profit_iqm) from
    frontier_summary.csv; the 6 controllers by display name; readable."""
    out, _, figures = frontier_run()
    _, frontier, summary = _frontier_csvs(out)
    fig = _saved(figures, "frontier.png")
    _readable(fig, out, "frontier.png", list(LABELS.values()))
    for regime, panel in _regime_panels(fig).items():
        for _, row in frontier[frontier["valid"].astype(bool)].iterrows():
            assert _drawn(panel, row["yield_kg_m2"], row[f"profit_{regime}"]), (regime, row)
        for _, row in summary[summary["regime"] == regime].iterrows():
            assert _drawn(panel, row["yield_iqm"], row["profit_iqm"]), (regime, row["label"])


def test_frontier_png_footer_says_in_sample_and_that_hours_are_not_shown(frontier_run):
    """frontier.png's footer (a figure-level text) says "in-sample" and "Σ hours not shown": the
    panels plot fruit and profit only, so a grey setting above a marker may still have more hours
    and not qualify."""
    _, _, figures = frontier_run()
    footer = _saved(figures, "frontier.png")["footer"]
    said = [t for t in footer if "in-sample" in t and "Σ hours not shown" in t]
    assert said, footer


def test_horizon_png_draws_profit_against_length_per_controller_per_regime(horizon_run):
    """horizon.png, saved once: a sim and an agc2020 panel, each with a line per controller (at
    least 3); Tuned rules, Default rules and Do nothing by name; readable."""
    out, _, figures = horizon_run()
    fig = _saved(figures, "horizon.png")
    _readable(fig, out, "horizon.png", list(HORIZON_CONTROLLERS))
    for regime, panel in _regime_panels(fig).items():
        assert len(panel["lines"]) >= len(HORIZON_CONTROLLERS), regime


# --- Slow: 60-day seasons, worker processes, what a run opens --------------------------------------


@pytest.mark.slow
def test_60_day_frontier_reproduces_the_recorded_rule_seasons_and_run_rb(tmp_path):
    """`frontier --n 0 --season-length 60 --jobs 3`: settings 0 and 1's per-season numbers equal
    rb_default's and rb_tuned's recorded validation seasons (evaluate's 60-day run_rb, recorded in
    scores.parquet), and setting 0's day-45 season equals a live run_rb(default, day 45, 60). Each
    baseline's own setting qualifies against its own point (fruit and hours equal up to the sum
    order, within 1e-9), so rb_default's best is setting 0 and rb_tuned's setting 1, gap ~0."""
    rule_family = _module()
    out = tmp_path / "out"
    argv = ["frontier", "--n", "0", "--seed", "0", "--season-length", "60", "--jobs", "3"]
    assert rule_family.main([*argv, "--out", str(out)]) == 0
    seasons, _, summary = _frontier_csvs(out)
    for setting, label in ((0, "rb_default"), (1, "rb_tuned")):
        recorded = _validation_rows(label).set_index("start_day")
        for day in DAYS:
            row = _one(seasons, setting=setting, start_day=day)
            for m in METRICS:
                assert _close(row[m], recorded.loc[day, m], 1e-12), (label, day, m)
            assert bool(row["valid"])
        for regime in REGIMES:
            point = _one(summary, label=label, regime=regime)
            assert int(point["best_setting"]) == setting, (label, regime)
            assert abs(float(point["gap"])) <= 1e-9, (label, regime, point["gap"])
    live = evaluate.run_rb(rule_based.load_rb_params("default"), scenarios.VALIDATION[0], 60)[0]
    row = _one(seasons, setting=0, start_day=45)
    for m in METRICS:
        assert _close(row[m], live[m], 1e-12), m


@pytest.mark.slow
def test_worker_processes_write_the_same_bytes(frontier_run, tmp_path):
    """`frontier --n 3 --seed 0 --season-length 1 --jobs 2` writes frontier_seasons.csv,
    frontier.csv, frontier_summary.csv and frontier.md byte-identical to the --jobs 1 run's."""
    rule_family = _module()
    one, _, _ = frontier_run()
    out = tmp_path / "out"
    argv = ["frontier", "--n", "3", "--seed", "0", "--season-length", "1", "--jobs", "2"]
    assert rule_family.main([*argv, "--out", str(out)]) == 0
    for name in ("frontier_seasons.csv", "frontier.csv", "frontier_summary.csv", "frontier.md"):
        assert (out / name).read_bytes() == (one / name).read_bytes(), name


# Run in a fresh interpreter: an audit hook records every weather file opened and every write,
# make, move or delete under control/ledger, control/results or control/runs, then runs a 1-day
# frontier (--n 0) and horizon (--lengths 1) at --jobs 1 into argv[1]; it prints {"codes",
# "weather", "seen"} as its last line.
_AUDIT = r"""
import json, os, sys

out, root = sys.argv[1], sys.argv[2]
weather = os.path.join(root, "control", "data", "weather").casefold()
guarded = [os.path.join(root, "control", name).casefold() for name in ("ledger", "results", "runs")]
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
edits = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1),
    "os.truncate": (0,), "os.chmod": (0,), "os.utime": (0,), "os.symlink": (1,), "os.link": (1,),
    "shutil.rmtree": (0,), "shutil.copyfile": (1,), "shutil.copytree": (1,), "shutil.move": (0, 1),
}
opened, seen = set(), []


def under(path, prefixes):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    p = os.path.abspath(os.fsdecode(path)).casefold()
    return any(p == q or p.startswith(q + os.sep) for q in prefixes)


def hook(event, args):
    if event == "open":
        path, mode, flags = args
        if under(path, [weather]):
            opened.add(os.path.relpath(os.path.abspath(os.fsdecode(path)).casefold(), weather))
        writes = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(flags & write_flags)
        if writes and under(path, guarded):
            seen.append(["opened to write", os.fsdecode(path)])
    elif event in edits:
        for i in edits[event]:
            if i < len(args) and under(args[i], guarded):
                seen.append([event, os.fsdecode(args[i])])


sys.addaudithook(hook)
from greenleaf_control import rule_family

codes = [
    rule_family.main(["frontier", "--n", "0", "--seed", "0", "--season-length", "1", "--jobs", "1",
                      "--out", os.path.join(out, "frontier")]),
    rule_family.main(["horizon", "--lengths", "1", "--jobs", "1", "--out", os.path.join(out, "horizon")]),
]
print(json.dumps({"codes": codes, "weather": sorted(opened), "seen": seen}))
"""


@pytest.mark.slow
def test_a_run_opens_only_amsterdam_2010_weather_and_writes_no_record(tmp_path):
    """In a fresh interpreter, a 1-day frontier and horizon into tmp_path return 0, open no weather
    file but Amsterdam/2010.csv (never 2012 or 2013) and write, make, move or delete nothing under
    control/ledger, control/results or control/runs."""
    done = subprocess.run(
        [sys.executable, "-c", _AUDIT, str(tmp_path), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["codes"] == [0, 0]
    assert report["weather"] == [os.path.join("amsterdam", "2010.csv")]
    assert report["seen"] == []
    assert sorted(p.name for p in (tmp_path / "frontier").iterdir()) == FRONTIER_OUTPUTS
    assert sorted(p.name for p in (tmp_path / "horizon").iterdir()) == HORIZON_OUTPUTS

"""Tests for greenleaf_control.media (issue #21, S4-0): one command regenerates the README and
LinkedIn visuals from the simulator, each file labelled with its source and commit.

media plays the baselines on PAPER and draws them:
- play(controller, season_length=60) -> {"frame", "result", "states", "controls", "params"}: the
  season from envs.run_baseline(controller, PAPER, season_length); states, controls from
  metrics.trace_from_infos; params the PAPER model's parameter vector; result
  metrics.episode_metrics(states, controls, params, season_steps=96 x season_length); frame
  evaluate._frame(infos), the contract's per-step frame.
- running(states, controls, params, ends) -> DataFrame[step, profit_eur_m2, viol_hours_co2,
  viol_hours_temp, viol_hours_rh]: one row per k in ends (k >= 1), step = k, the metrics of the
  first k steps (states[:k + 1], controls[:k]).
- season_replay(frame, counters, out_gif, out_mp4=None, *, title, subtitle=None, footer, fps=15)
  -> Figure: a FuncAnimation on a plain Figure (no pyplot), saved as a GIF (and an MP4 when
  out_mp4 is given), returned in its final frame's state. Its climate lines are frame.iloc[::4]
  (hourly) co2, temp and rh against t, each over its metrics.BOUNDS band (axhspan); its actuator
  heat strip is the same hourly rows' six u_* columns as a 6-row array (row j = U_COLS[j], column
  i = hourly row i); its counters are counters' rows as text, the final frame showing the last
  row; the title, the subtitle (when given) and the footer are each one text, as given.
- baselines_card(seasons, out, *, footer) -> Figure: seasons maps rb_default, zero_controls and
  random to play()'s dicts; three columns left to right in that order, each with its own strip
  (the same 6-row hourly array), its result's numbers, and each day's share of steps outside each
  safe range (a 3 x days array, rows in metrics.CONSTRAINTS order). One text says what profit
  counts: fruit value minus heating, power and CO₂.
- footer(source, stamp, *, place=PAPER_NAME, note="A simulator run, not a registered result")
  -> str, and main(argv=None) -> int, the CLI. main refuses an invalid season (a solver failure,
  or the wrong length) before writing anything, and titles the replay as simulated, with the
  season's day count from the run and PAPER's location and year; its subtitle names the
  controller and the simulator.

The trade-off still (issue #50) draws recorded VALIDATION results, every number read from the
ledger when the card is drawn:
- TRADEOFF, (result_id, arm, status) entries (row ids and statuses, never numbers), and
  E1_RESULT, rb_tuned's E1 result row; both read at call time, so the tests point them at
  synthetic rows.
- tradeoff_points(path=ledger.LEDGER) -> list[dict], one per TRADEOFF entry, in order: arm,
  status, result, decision (the one decision row on that result), registered and seeds (the
  registered row's); profit and each viol_hours_<c> as (iqm, ci_low, ci_high) from per_arm[arm]
  and its guards; yield_kg_m2, the guard's iqm. ValueError naming the row for a row that is
  missing or isn't a result, no decision row (or two), scenarios other than ["validation"], an
  arm missing from per_arm, a null iqm, crashed or invalid_episodes > 0, or a missing guard.
- rb_tuned_reference(n_seeds, path=ledger.LEDGER, parquet=selection.RESULTS /
  "rb_tuned_validation.parquet") -> dict: first the parquet checked against E1's row (each
  viol_hours_<c>'s per-season mean equal to its guard's iqm exactly, paper_return's to the iqm
  within rel 1e-9; ValueError otherwise), then profit_eur_m2, each viol_hours_<c> and
  yield_kg_m2 as stats.iqm_ci(np.tile(seasons, (n_seeds, 1)))[0], n_seeds, and sha256 (the
  parquet file's).
- tradeoff_card(points, references, out, *, footer) -> Figure, saved at exactly 1200 x 627 px:
  three panels sharing the profit axis, x the hours outside each safe range in metrics.CONSTRAINTS
  order. Each point is an errorbar at (its hours' iqm, its profit iqm) with bars to both CIs; the
  dropped arm hollow and MUTED, residual_w05 LEAF, the other adopted arms filled. rb_tuned is a
  distinct hollow marker with no bars at the largest-n reference. A legend outside the panels has
  one line per point naming its result id; above the panels, the title (VALIDATION, Amsterdam
  2010), the CI line and a note with rb_tuned's profit at each smaller seed count.
- main also writes tradeoff.png and a media.json entry for it (split "validation", scenario
  VALIDATION, ledger [row ids], metrics {points, references}); the other entries are unchanged.

The card's tests read synthetic rows that ledger.register, run_start, run_end, record_result and
decide write to a tmp ledger, and a tmp parquet. No test reads the committed ledger or results: a
tripwire on ledger.rows and pandas.read_parquet fails first, and every test that runs main binds
its card to the tmp files.

The tests read what is drawn from the returned Figure (its lines, patches, images and texts),
never its pixels. Numbers on the figures are printed with 2 decimals, f"{v:.2f}" (hours are
multiples of 0.25 h, so 2 decimals are exact), each on one text line with its label: profit with
"€/m²", each constraint's hours with its viz label ("CO₂", "temp", "RH"), fruit growth with "fruit
growth". No figure prints a total of the three hour counts (an hour out of two bands would count
twice), says "yield" (yield_kg_m2 is the net change in standing fruit, not a harvest) or uses an em
dash.

Known answers: one day is 96 steps (86,400 s / 900 s), so a 1-day season has 96 frame rows and 24
hourly rows (t = 0, 4, ..., 92). A violating step is 900 s = 0.25 h, so a prefix's viol_hours_<c>
is its out-of-band frame rows (a value on a bound is in bounds) x 0.25.

The fast tests play 1-day PAPER seasons and write a GIF only. The slow one runs the CLI at its
defaults (three 60-day seasons, GIF and MP4 through ffmpeg) and is skipped without ffmpeg and
ffprobe. The 2012 weather is sealed (PROTOCOL §8): every test runs under a tripwire on
pandas.read_csv that fails before any *2012.csv is read.
"""

import ast
import functools
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.colors import to_hex, to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image

import greenleaf_control
from greenleaf_control import (
    envs,
    evaluate,
    ledger,
    manifest,
    metrics,
    rule_based,
    scenarios,
    selection,
    stats,
)

# greenleaf_control.media is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

PAPER = scenarios.PAPER
N_DAY = 96  # 86,400 s / 900 s
CONTROLLERS = ("rb_default", "zero_controls", "random")  # the card's columns, left to right
U_COLS = ["u_boil", "u_co2", "u_thscr", "u_vent", "u_lamp", "u_blscr"]  # upstream's order
RUNNING_COLS = ["step", "profit_eur_m2", "viol_hours_co2", "viol_hours_temp", "viol_hours_rh"]
LABELS = {"co2": "CO₂", "temp": "temp", "rh": "RH"}  # viz.CLIMATE_LABELS' words
RB_LABEL = "upstream's default rule controller (the grower's rules)"
ENDS = [24, 48, 72, 96]  # the replay test's counter rows: four frames, the last the whole day
TITLE = "The grower's rules on PAPER, one day"
SUBTITLE = "upstream's default rule controller (the grower's rules), simulated in GreenLight-Gym"
FOOTER = "source: test, PAPER (Amsterdam 2010), simulator run, not a registered result"
PLACE = f"{PAPER['location']} {PAPER['growth_year']}"  # "Amsterdam 2010"
RB_SUBTITLE = ("upstream's default rule controller", "GreenLight-Gym")  # what main's names
SHA = "0123456789abcdef0123456789abcdef01234567"  # a fixed commit for the footer tests
GOLDEN = json.loads((Path(__file__).parent / "golden" / "values.json").read_text())["ours"]
MEDIA_FILES = [
    "baselines.png",
    "media.json",
    "season_replay.gif",
    "season_replay.mp4",
    "tradeoff.png",
]
ENTRY_KEYS = {"file", "source", "scenario", "commit", "dirty", "gl_gym", "made_at", "metrics"}

# --- The trade-off still (issue #50) ---
VALIDATION = scenarios.VALIDATION
VAL_PLACE = f"{VALIDATION[0]['location']} {VALIDATION[0]['growth_year']}"  # "Amsterdam 2010"
# The card's footer place and note, as the issue words them.
FOOTER_PLACE = f"VALIDATION, {VAL_PLACE} days " + ", ".join(
    str(s["start_day"]) for s in VALIDATION
)  # "VALIDATION, Amsterdam 2010 days 45, 59, 75"
CARD_NOTE = "Recorded VALIDATION results; the 2012 test weather is still sealed"
PAPER_NOTE = "A simulator run, not a registered result"  # footer's default note, today's words
CI_LINE = "IQM over seeds × seasons, 95% stratified bootstrap CI"
CARD_FOOTER = "source: synthetic rows in a tmp ledger, commit 0123456"  # what the card tests pass
PARQUET = "rb_tuned_validation.parquet"
NEW_NAMES = ("TRADEOFF", "E1_RESULT", "tradeoff_points", "rb_tuned_reference", "tradeoff_card")
# The issue's table, (result_id, arm, status); E4 has one decision row, residual_w4 dropped in it.
ISSUE_TRADEOFF = [
    ("result-06364fa1", "ppo", "adopted"),
    ("result-d1f0b8e7", "residual_w05", "adopted"),
    ("result-2a890455", "residual_w1", "adopted"),
    ("result-2a890455", "residual_w4", "dropped"),
]
HOUR_KEYS = [f"viol_hours_{c}" for c in metrics.CONSTRAINTS]
GUARDS = [*HOUR_KEYS, "paper_return", "yield_kg_m2", "solver_failures"]  # E3, E4 and E6's
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}

# Synthetic ledger numbers, (iqm, ci_low, ci_high): profit [€/m²], hours [h] (multiples of
# 0.25 h), fruit growth [kg/m²]. Every point sits apart from the others in every panel, prints
# apart at 2 decimals, and has two or more nonzero hours whose total is printed nowhere.
E3 = {
    "ppo": {
        "profit": (2.4, 2.33, 2.46),
        "co2": (0.5, 0.25, 1.0),
        "temp": (153.25, 140.0, 166.5),
        "rh": (135.0, 120.5, 150.0),
        "yield": (7.13, 6.9, 7.4),
    }
}
E4 = {
    # Not on the card: E6's residual_w05 is. Its numbers differ, so reading the wrong row shows.
    "residual_w05": {
        "profit": (0.63, 0.48, 0.74),
        "co2": (2.25, 1.5, 3.0),
        "temp": (60.0, 52.5, 68.25),
        "rh": (220.0, 210.0, 231.5),
        "yield": (9.1, 8.8, 9.4),
    },
    "residual_w1": {
        "profit": (0.4, 0.32, 0.45),
        "co2": (3.0, 2.25, 4.0),
        "temp": (70.25, 61.0, 80.5),
        "rh": (230.5, 221.0, 240.25),
        "yield": (8.81, 8.6, 9.0),
    },
    "residual_w4": {
        "profit": (-0.37, -0.48, -0.26),
        "co2": (6.5, 5.0, 8.25),
        "temp": (110.0, 99.5, 121.75),
        "rh": (180.25, 170.0, 191.0),
        "yield": (6.42, 6.2, 6.7),
    },
}
E6 = {
    "residual_w05": {
        "profit": (0.77, 0.7, 0.84),
        "co2": (1.75, 1.0, 2.5),
        "temp": (48.5, 42.0, 55.75),
        "rh": (214.0, 205.25, 223.5),
        "yield": (9.63, 9.4, 9.9),
    }
}
# experiment -> (its arms' numbers, its registered seeds): 5 seeds for E3 and E4, 10 for E6.
EXPERIMENTS = {
    "E3": (E3, list(range(5))),
    "E4": (E4, list(range(5))),
    "E6": (E6, list(range(100, 110))),
}
# The synthetic TRADEOFF, in the issue's order: (experiment, arm, status).
TRADEOFF_SPEC = (
    ("E3", "ppo", "adopted"),
    ("E6", "residual_w05", "adopted"),
    ("E4", "residual_w1", "adopted"),
    ("E4", "residual_w4", "dropped"),
)
DROPPED = "residual_w4"

# rb_tuned's three VALIDATION seasons (days 45, 59, 75), as evaluate's parquet holds them.
RB_SEASONS = {
    "profit_eur_m2": (-3.0, 0.6, 1.2),
    "viol_hours_co2": (0.0, 1.0, 5.0),
    "viol_hours_temp": (120.0, 90.0, 75.25),
    "viol_hours_rh": (300.0, 250.5, 222.75),
    "yield_kg_m2": (8.0, 9.0, 11.0),
    "paper_return": (3000.0, 3600.0, 4000.0),
}
# rb_tuned tiled to n seeds is n x 3 scores; the IQM drops int(0.25 x 3n) from each end:
# 7 of 30 for n = 10, leaving 3 x low + 10 x middle + 3 x high over 16; 3 of 15 for n = 5,
# leaving 2 x low + 5 x middle + 2 x high over 9.
REF = {
    10: {
        "profit_eur_m2": 0.0375,  # (3 x -3.0 + 10 x 0.6 + 3 x 1.2) / 16 = 0.6 / 16 €/m²
        "viol_hours_co2": 1.5625,  # (3 x 0 + 10 x 1.0 + 3 x 5.0) / 16 = 25 / 16 h
        "viol_hours_temp": 92.859375,  # (3 x 75.25 + 10 x 90 + 3 x 120) / 16 = 1485.75 / 16 h
        "viol_hours_rh": 254.578125,  # (3 x 222.75 + 10 x 250.5 + 3 x 300) / 16 = 4073.25 / 16 h
        "yield_kg_m2": 9.1875,  # (3 x 8 + 10 x 9 + 3 x 11) / 16 = 147 / 16 kg/m²
    },
    5: {
        "profit_eur_m2": -0.6 / 9,  # (2 x -3.0 + 5 x 0.6 + 2 x 1.2) / 9 = -0.0667 €/m²
        "viol_hours_co2": 15 / 9,  # (2 x 0 + 5 x 1.0 + 2 x 5.0) / 9 = 1.6667 h
        "viol_hours_temp": 840.5 / 9,  # (2 x 75.25 + 5 x 90 + 2 x 120) / 9 = 93.3889 h
        "viol_hours_rh": 2298 / 9,  # (2 x 222.75 + 5 x 250.5 + 2 x 300) / 9 = 255.3333 h
        "yield_kg_m2": 83 / 9,  # (2 x 8 + 5 x 9 + 2 x 11) / 9 = 9.2222 kg/m²
    },
}
# Parquets that disagree with E1's row, which holds the per-season means of the seasons above.
PARQUET_MISMATCHES = {
    # One season 0.25 h (one 900 s step) more outside the temp band: the mean is 0.0833 h off.
    "temp_hours_off_by_one_step": {**RB_SEASONS, "viol_hours_temp": (120.25, 90.0, 75.25)},
    # Every season's paper_return 1e-6 too high: the mean is off by rel 1e-6 > 1e-9.
    "paper_return_off_by_rel_1e-6": {
        **RB_SEASONS,
        "paper_return": tuple(v * (1 + 1e-6) for v in RB_SEASONS["paper_return"]),
    },
}
# The rows tradeoff_points refuses; _refusal builds each.
REFUSALS = (
    "missing",
    "not_a_result",
    "undecided",
    "two_decisions",
    "not_validation",
    "arm_absent",
    "iqm_null",
    "crashed",
    "invalid_episodes",
    "no_rh_guard",
    "no_yield_guard",
)


@pytest.fixture(autouse=True)
def no_2012_reads(monkeypatch) -> None:
    """Fail loudly if the test reads any 2012 weather file. pandas.read_csv is the only file read
    upstream's weather loader makes (its own year, and the next year's file past the year's end);
    the check runs before the real read, and every other file goes through the real one."""
    real_read_csv = pd.read_csv

    def read_csv(filepath_or_buffer, *args, **kwargs):
        if Path(str(filepath_or_buffer)).name.endswith("2012.csv"):
            raise AssertionError(f"the sealed 2012 test weather was read: {filepath_or_buffer}")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)


@pytest.fixture(autouse=True)
def no_committed_reads(monkeypatch) -> None:
    """Fail loudly if the test reads the committed ledger or results (AGENTS.md: a test touches
    control/ledger/ and control/results/ only through tmp_path). ledger.rows and
    pandas.read_parquet are the card's reads; the check runs before the real read, and every tmp
    file goes through the real one."""
    real_rows, real_read_parquet = ledger.rows, pd.read_parquet
    committed = (ledger.LEDGER.parent.resolve(), selection.RESULTS.parent.resolve())

    def check(path) -> None:
        p = Path(path).resolve()
        if any(p == d or d in p.parents for d in committed):
            raise AssertionError(f"a committed ledger or result was read: {p}")

    def rows(kind=None, issue=None, path=ledger.LEDGER):
        check(path)
        return real_rows(kind, issue, path)

    def read_parquet(path, *args, **kwargs):
        check(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(ledger, "rows", rows)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory) -> Path:
    """A tmp dir the cached figures below are written into (makes no media call, so never ERRORs)."""
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="module")
def sources(media_dir) -> dict:
    """The synthetic ledger and parquet the card reads, built once with the ledger's own API
    (makes no media call, so never ERRORs)."""
    return _sources(media_dir / "sources")


@functools.cache
def _baseline(controller: str) -> dict:
    """A 1-day PAPER season of a baseline, built here without media: run_baseline's infos graded
    by trace_from_infos + episode_metrics (the PAPER model's params, 96 season steps), and
    evaluate's frame of the same infos. Cached: the season is deterministic (random's generator is
    seeded by the start day)."""
    infos = envs.run_baseline(controller, PAPER, 1)
    states, controls = metrics.trace_from_infos(infos)
    env = envs.make_raw_env(scenario=PAPER, season_length=1, normalize_actions=False)
    params = np.asarray(env.unwrapped.p, dtype=np.float64)
    env.close()
    result = metrics.episode_metrics(states, controls, params, season_steps=N_DAY)
    return {
        "frame": evaluate._frame(infos),
        "result": result,
        "states": states,
        "controls": controls,
        "params": params,
    }


@functools.cache
def _played(controller: str) -> dict:
    import greenleaf_control.media as media

    return media.play(controller, 1)


def _counters(b: dict, ends: list[int]) -> pd.DataFrame:
    """running()'s table built here from metrics: the first k steps' numbers for each k."""
    rows = []
    for k in ends:
        m = metrics.episode_metrics(
            b["states"][: k + 1], b["controls"][:k], b["params"], season_steps=k
        )
        rows.append({"step": k, **{col: m[col] for col in RUNNING_COLS[1:]}})
    return pd.DataFrame(rows, columns=RUNNING_COLS)


@functools.cache
def _replay(controller: str, out_dir: Path) -> tuple[Figure, Path]:
    """season_replay of a baseline's 1-day frame with _counters at ENDS, GIF only; cached so the
    replay tests render it once per controller."""
    import greenleaf_control.media as media

    b = _baseline(controller)
    out = out_dir / f"replay_{controller}" / "season_replay.gif"
    out.parent.mkdir(exist_ok=True)
    fig = media.season_replay(b["frame"], _counters(b, ENDS), out, title=TITLE, footer=FOOTER)
    return fig, out


@functools.cache
def _card(out_dir: Path) -> tuple[Figure, Path]:
    """baselines_card of the three baselines' 1-day seasons; cached so it renders once."""
    import greenleaf_control.media as media

    out = out_dir / "card" / "baselines.png"
    out.parent.mkdir(exist_ok=True)
    fig = media.baselines_card({c: _baseline(c) for c in CONTROLLERS}, out, footer=FOOTER)
    return fig, out


def _hourly(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[::4]


def _xy(line) -> tuple[list[float], list[float]]:
    return (
        np.asarray(line.get_xdata(), dtype=float).tolist(),
        np.asarray(line.get_ydata(), dtype=float).tolist(),
    )


def _texts(fig: Figure) -> list[Text]:
    """Every visible, non-empty text artist on fig (titles, labels, ticks, figure texts)."""
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def _lines_of(texts: list[Text]) -> list[str]:
    return [line for t in texts for line in t.get_text().splitlines()]


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _has(line: str, v: float) -> bool:
    """line prints v at 2 decimals as a whole number: not a piece of a longer one (0.00 in 10.00)
    nor the unsigned tail of its negative (2.89 in -2.89)."""
    return re.search(rf"(?<![\d.\-]){re.escape(_fmt(v))}(?!\d)", line) is not None


def _assert_numbers(lines: list[str], result: dict, where: str) -> None:
    """profit on a line with "€/m²", each constraint's hours on a line with its label."""
    profit = result["profit_eur_m2"]
    assert any("€/m²" in ln and _has(ln, profit) for ln in lines), (where, _fmt(profit), lines)
    for c in metrics.CONSTRAINTS:
        hours = result[f"viol_hours_{c}"]
        assert any(LABELS[c] in ln and _has(ln, hours) for ln in lines), (where, c, _fmt(hours))


def _strips(fig: Figure, n_cols: int) -> list:
    """Every image or mesh on fig holding a 6-row x n_cols array (a colorbar's mesh is not one)."""
    found = fig.findobj(lambda a: isinstance(a, (AxesImage, QuadMesh)))
    return [a for a in found if np.shape(a.get_array()) == (6, n_cols)]


def _strip_array(artist) -> np.ndarray:
    """The strip's values, with anything masked (a frame not yet shown) as NaN."""
    return np.ma.filled(np.ma.asarray(artist.get_array(), dtype=np.float64), np.nan)


def _columns(fig: Figure, strips: list) -> tuple[list, list[list[str]]]:
    """The strips left to right, and the text lines whose rendered centre falls in each strip's
    column: the figure split at the midpoints between the strips' axes centres."""
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def centre(bbox) -> float:
        return (bbox.x0 + bbox.x1) / 2

    strips = sorted(strips, key=lambda a: centre(a.axes.get_window_extent(renderer)))
    xs = [centre(a.axes.get_window_extent(renderer)) for a in strips]
    cuts = [(a + b) / 2 for a, b in zip(xs, xs[1:], strict=False)]
    columns = [[] for _ in strips]
    for t in _texts(fig):
        k = int(np.searchsorted(cuts, centre(t.get_window_extent(renderer))))
        columns[k].extend(t.get_text().splitlines())
    return strips, columns


def _no_season_played(monkeypatch, media) -> None:
    """Every way media could play a season fails: envs.run_baseline and make_raw_env, evaluate's
    run_rb, and media's own play (and those names on media, in case it imported them by name)."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any season")

        return refuse

    for module, name in [(envs, "run_baseline"), (envs, "make_raw_env"), (evaluate, "run_rb")]:
        monkeypatch.setattr(module, name, trip(f"{module.__name__}.{name}"))
    for name in ("play", "run_baseline", "make_raw_env", "run_rb"):
        if hasattr(media, name):
            monkeypatch.setattr(media, name, trip(f"media.{name}"))


def _imported_modules(path: Path) -> set[str]:
    """Every module path imported anywhere in path's source (as in test_viz.py)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(filter(None, ["greenleaf_control", base]))
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


# --- The trade-off still's synthetic sources (issue #50) ---------------------------------------


def _media_card():
    """media, asserting the trade-off card's names exist first, so until they do a test FAILS on
    this assertion rather than an AttributeError or a TypeError."""
    import greenleaf_control.media as media

    missing = [name for name in NEW_NAMES if not hasattr(media, name)]
    assert not missing, f"media has no {', '.join(missing)} yet (issue #50)"
    return media


def _ci(t) -> dict:
    return dict(zip(("iqm", "ci_low", "ci_high"), t, strict=True))


def _numbers(arm: dict) -> dict:
    """A learning arm's per_arm entry as selection writes it: profit the primary, then the guards
    (crashed and spend_usd are the ledger's to add)."""
    return {
        **_ci(arm["profit"]),
        "guards": {
            **{f"viol_hours_{c}": _ci(arm[c]) for c in metrics.CONSTRAINTS},
            "paper_return": _ci((3500.0, 3400.0, 3600.0)),
            "yield_kg_m2": _ci(arm["yield"]),
            "solver_failures": _ci((0.0, 0.0, 0.0)),
        },
        "invalid_episodes": 0,
    }


def _e1_numbers() -> dict:
    """rb_tuned's E1 entry as selection writes it: iqm_ci of its 1 run x 3 VALIDATION seasons, a
    3-score IQM being the plain mean (int(0.25 x 3) = 0 trimmed). paper_return
    (3000 + 3600 + 4000) / 3 = 3533.33; hours co2 (0 + 1 + 5) / 3 = 2.0 h, temp
    (120 + 90 + 75.25) / 3 = 95.08 h, rh (300 + 250.5 + 222.75) / 3 = 257.75 h. No fruit growth
    guard, as E1 has none."""

    def ci(key: str) -> dict:
        return _ci(stats.iqm_ci(np.array([RB_SEASONS[key]])))

    return {
        **ci("paper_return"),
        "guards": {key: ci(key) for key in HOUR_KEYS},
        "invalid_episodes": 0,
    }


def _experiment(
    path: Path,
    *,
    issue: int,
    arms: dict,
    seeds: list[int],
    method: str = "ppo",
    on: tuple = ("validation",),
    primary: str = "profit_eur_m2",
    guards: list[str] = GUARDS,
    crashed: bool = False,
    decided: bool = True,
) -> dict:
    """Register an experiment on path, start and end a run per arm x seed (the first seed's
    crashed when crashed), record arms ({name: per_arm entry}) and, when decided, adopt it.
    Returns its registered, result and decision row ids."""
    reg = ledger.register(
        issue=issue,
        hypothesis="synthetic rows for the trade-off card's tests",
        arms=[{"name": name, "method": method, "config": {}} for name in arms],
        seeds=list(seeds),
        timesteps=1000,
        scenarios=list(on),
        primary=primary,
        guards=list(guards),
        rule="synthetic",
        budget={"cpu_hours": 0.0, "usd": 0.0, "backend": "local"},
        path=path,
    )
    selected = {}
    for name in arms:
        for i, seed in enumerate(seeds):
            run = ledger.run_start(
                registered=reg["row_id"],
                arm=name,
                seed=seed,
                run_dir=f"runs/{name}-{seed}",
                path=path,
            )
            ledger.run_end(
                run=run["row_id"],
                status="crashed" if crashed and i == 0 else "finished",
                wall_s=1.0,
                timesteps=1000,
                solver_failures=0,
                cost_usd=None,
                diagnostics=DIAGNOSTICS,
                path=path,
            )
            selected[run["row_id"]] = "final"
    res = ledger.record_result(registered=reg["row_id"], per_arm=arms, selected=selected, path=path)
    ids = {"registered": reg["row_id"], "result": res["row_id"]}
    if decided:
        dec = ledger.decide(result=res["row_id"], decision="adopted", note="synthetic", path=path)
        ids["decision"] = dec["row_id"]
    return ids


def _write_parquet(path: Path, seasons: dict) -> None:
    """rb_tuned's VALIDATION parquet as evaluate's CLI writes it: one row per season, in order."""
    rows = [
        {
            "controller": "rb_tuned",
            "checkpoint": None,
            "split": "validation",
            **s,
            "season_length": 60,
            **{key: values[i] for key, values in seasons.items()},
            "valid": True,
            "solver_failures": 0,
            "steps": 60 * N_DAY,
        }
        for i, s in enumerate(VALIDATION)
    ]
    pd.DataFrame(rows).to_parquet(path)


@functools.cache
def _sources(root: Path) -> dict:
    """The synthetic ledger and parquet, built once: an E1-like rb_tuned row (1 seed, primary
    paper_return, hour guards only), E3-like ppo (5 seeds), E4-like residual_w05, residual_w1 and
    residual_w4 (5 seeds, one decision) and E6-like residual_w05 (10 seeds), each decided; and
    rb_tuned's parquet of RB_SEASONS. tradeoff is TRADEOFF_SPEC with the rows' ids."""
    root.mkdir(exist_ok=True)
    path = root / "ledger.jsonl"
    path.unlink(missing_ok=True)
    rows = {
        "E1": _experiment(
            path,
            issue=101,
            arms={"rb_tuned": _e1_numbers()},
            seeds=[0],
            method="rb_tuned",
            primary="paper_return",
            guards=HOUR_KEYS,
        )
    }
    for k, (name, (numbers, seeds)) in enumerate(EXPERIMENTS.items()):
        arms = {arm: _numbers(n) for arm, n in numbers.items()}
        rows[name] = _experiment(path, issue=102 + k, arms=arms, seeds=seeds)
    parquet = root / PARQUET
    _write_parquet(parquet, RB_SEASONS)
    tradeoff = [(rows[exp]["result"], arm, status) for exp, arm, status in TRADEOFF_SPEC]
    return {
        "ledger": path,
        "parquet": parquet,
        "rows": rows,
        "tradeoff": tradeoff,
        "e1": rows["E1"]["result"],
    }


def _want_points(src: dict) -> list[dict]:
    """tradeoff_points' answer on the synthetic ledger, in TRADEOFF order, written from the
    numbers the rows were made with."""
    want = []
    for exp, arm, status in TRADEOFF_SPEC:
        numbers, seeds = EXPERIMENTS[exp]
        n, ids = numbers[arm], src["rows"][exp]
        want.append(
            {
                "arm": arm,
                "status": status,
                "result": ids["result"],
                "decision": ids["decision"],
                "registered": ids["registered"],
                "seeds": seeds,
                "profit": n["profit"],
                **{f"viol_hours_{c}": n[c] for c in metrics.CONSTRAINTS},
                "yield_kg_m2": n["yield"][0],
            }
        )
    return want


def _tiled(key: str, n: int) -> float:
    """stats.iqm_ci of rb_tuned's seasons tiled to n seeds: the IQM the reference draws."""
    return stats.iqm_ci(np.tile(np.array(RB_SEASONS[key]), (n, 1)))[0]


def _json(value):
    """value as media.json holds it (tuples become lists)."""
    return json.loads(json.dumps(value))


def _card_reads(monkeypatch, media, src: dict, **over) -> None:
    """main's card reads the synthetic rows, never the committed ones: TRADEOFF and E1_RESULT name
    them, and tradeoff_points and rb_tuned_reference are bound to the tmp ledger and parquet
    (their defaults bind the committed files at import). over replaces src's tradeoff, ledger or
    parquet. Before #50 media has no card, so there is nothing to bind and main reads neither."""
    src = {**src, **over}
    monkeypatch.setattr(media, "TRADEOFF", src["tradeoff"], raising=False)
    monkeypatch.setattr(media, "E1_RESULT", src["e1"], raising=False)
    binds = {
        "tradeoff_points": {"path": src["ledger"]},
        "rb_tuned_reference": {"path": src["ledger"], "parquet": src["parquet"]},
    }
    for name, kwargs in binds.items():
        real = getattr(media, name, None)
        if real is not None:
            monkeypatch.setattr(media, name, functools.partial(real, **kwargs))


def _refusal(case: str, src: dict, path: Path) -> tuple[list, set[str]]:
    """src's ledger copied to path with case's flaw added. Returns the TRADEOFF to draw (src's
    entries, plus the flawed one last) and the row ids the error may name."""
    shutil.copyfile(src["ledger"], path)
    e3 = src["rows"]["E3"]
    entries = list(src["tradeoff"])
    if case == "missing":
        return [*entries, ("result-0badc0de", "ppo", "adopted")], {"result-0badc0de"}
    if case == "not_a_result":
        return [*entries, (e3["decision"], "ppo", "adopted")], {e3["decision"]}
    if case == "arm_absent":
        return [*entries, (e3["result"], "residual_w9", "adopted")], {e3["result"]}
    if case == "two_decisions":
        ledger.decide(result=e3["result"], decision="dropped", note="a second call", path=path)
        return entries, {e3["result"]}
    numbers, kwargs = _numbers(E3["ppo"]), {}
    if case == "undecided":
        kwargs["decided"] = False
    elif case == "not_validation":
        kwargs["on"] = ("train_eval",)
    elif case == "crashed":
        kwargs["crashed"] = True
    elif case == "iqm_null":
        numbers.update(iqm=None, ci_low=None, ci_high=None)
    elif case == "invalid_episodes":
        numbers["invalid_episodes"] = 1
    else:
        guard = {"no_rh_guard": "viol_hours_rh", "no_yield_guard": "yield_kg_m2"}[case]
        del numbers["guards"][guard]
    ids = _experiment(path, issue=200, arms={"ppo": numbers}, seeds=[0, 1], **kwargs)
    named = {ids["result"], ids["registered"]} if case == "not_validation" else {ids["result"]}
    return [*entries, (ids["result"], "ppo", "adopted")], named


def _edited(src: Path, dst: Path, result_id: str, arm: str, **iqms: float) -> None:
    """A copy of the ledger at src with result_id's per_arm[arm] iqms replaced: profit= sets the
    primary's, viol_hours_<c>= that guard's."""
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    for row in rows:
        if row["row_id"] == result_id:
            numbers = row["per_arm"][arm]
            for key, v in iqms.items():
                (numbers if key == "profit" else numbers["guards"][key])["iqm"] = v
    dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _draw_tradeoff(media, src: dict, out: Path, ledger_path: Path | None = None) -> Figure:
    """The card of src's rows (or ledger_path's), with TRADEOFF and E1_RESULT pointed at them:
    tradeoff_points, rb_tuned_reference at 5 and 10 seeds (the points' seed counts), then
    tradeoff_card with CARD_FOOTER."""
    path = ledger_path or src["ledger"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(media, "TRADEOFF", src["tradeoff"])
        mp.setattr(media, "E1_RESULT", src["e1"])
        points = media.tradeoff_points(path)
        refs = [media.rb_tuned_reference(n, path, src["parquet"]) for n in (5, 10)]
        return media.tradeoff_card(points, refs, out, footer=CARD_FOOTER)


@functools.cache
def _tradeoff(out_dir: Path) -> tuple[Figure, Path]:
    """The card of the synthetic sources; cached so it renders once."""
    media = _media_card()
    out = out_dir / "tradeoff" / "tradeoff.png"
    out.parent.mkdir(exist_ok=True)
    return _draw_tradeoff(media, _sources(out_dir / "sources"), out), out


def _errorbars(ax) -> list[ErrorbarContainer]:
    return [c for c in ax.containers if isinstance(c, ErrorbarContainer)]


def _segments(container) -> list:
    """An errorbar's bar segments, each [[x0, y0], [x1, y1]]."""
    return [
        np.asarray(s, dtype=float).tolist()
        for col in container.lines[2]
        for s in col.get_segments()
    ]


def _point(line) -> list[list[float]]:
    return np.asarray(line.get_xydata(), dtype=float).tolist()


def _errorbar_at(ax, x: float, y: float) -> list:
    """ax's errorbars whose marker is the one point (x, y), exactly."""
    return [c for c in _errorbars(ax) if c.lines[0] is not None and _point(c.lines[0]) == [[x, y]]]


def _lines_near(ax, x: float, y: float) -> list[Line2D]:
    """ax's lines drawing the one point (x, y), to rel 1e-9."""
    return [
        line
        for line in ax.get_lines()
        if len(_point(line)) == 1 and _point(line)[0] == pytest.approx([x, y], rel=1e-9)
    ]


def _panels(fig: Figure) -> list:
    """The axes holding errorbars with bars, left to right."""
    found = [ax for ax in fig.axes if any(_segments(c) for c in _errorbars(ax))]
    return sorted(found, key=lambda ax: ax.get_position().x0)


def _legend_texts(fig: Figure) -> tuple[list, list[str]]:
    """Every legend on fig (the axes' and the figure's), and each legend entry's text, whitespace
    joined (an entry may wrap)."""
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + fig.legends
    return legends, [" ".join(t.get_text().split()) for lg in legends for t in lg.get_texts()]


# --- play: the season the pictures draw --------------------------------------------------------


def test_play_rb_default_is_run_rbs_season():
    """play("rb_default", 1) is evaluate.run_rb(default params, PAPER, 1, record_steps=True)'s
    season: the frame equal exactly, and the result run_rb's minus paper_reward_sum (media grades
    with metrics only). 96 steps, valid."""
    want, want_frame = evaluate.run_rb(rule_based.load_rb_params("default"), PAPER, 1, True)

    got = _played("rb_default")

    assert set(got) == {"frame", "result", "states", "controls", "params"}
    pd.testing.assert_frame_equal(got["frame"], want_frame, check_exact=True)
    assert got["result"] == {k: v for k, v in want.items() if k != "paper_reward_sum"}
    assert (got["result"]["steps"], got["result"]["valid"]) == (N_DAY, True)


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_play_is_the_baselines_season_graded_by_metrics(controller):
    """play(c, 1) for each baseline: states and controls are trace_from_infos of
    envs.run_baseline(c, PAPER, 1), params the PAPER model's parameter vector, result
    episode_metrics of them over 96 season steps, and frame evaluate's frame of the same infos,
    all exactly."""
    want = _baseline(controller)

    got = _played(controller)

    np.testing.assert_array_equal(got["states"], want["states"])
    np.testing.assert_array_equal(got["controls"], want["controls"])
    np.testing.assert_array_equal(np.asarray(got["params"], dtype=np.float64), want["params"])
    assert got["result"] == want["result"]
    pd.testing.assert_frame_equal(got["frame"], want["frame"], check_exact=True)


# --- running: the counters ---------------------------------------------------------------------


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_running_counts_each_prefix_and_ends_on_the_seasons_metrics(controller):
    """running(states, controls, params, ends) on a 1-day season with hourly ends 4, 8, ..., 96:
    the columns step, profit_eur_m2 and viol_hours_<c>, one row per end with step = k, each row the
    metrics of the first k steps, and the last row (k = 96, the whole day) equal to the season's
    result exactly.

    Hand check of the hours: row k's viol_hours_<c> is the frame's out-of-band rows t = 0..k-1
    (metrics.BOUNDS, a value on a bound in bounds) x 0.25 h (900 s / 3,600 s per step).
    Non-vacuous: the profit changes over the day, and zero_controls and random leave at least one
    band during it, so their hours grow.
    """
    import greenleaf_control.media as media

    b = _baseline(controller)
    ends = list(range(4, N_DAY + 1, 4))

    got = media.running(b["states"], b["controls"], b["params"], ends)

    assert list(got.columns) == RUNNING_COLS
    assert got["step"].tolist() == ends
    for row, k in zip(got.to_dict("records"), ends, strict=True):
        m = metrics.episode_metrics(
            b["states"][: k + 1], b["controls"][:k], b["params"], season_steps=k
        )
        assert {col: row[col] for col in RUNNING_COLS[1:]} == {
            col: m[col] for col in RUNNING_COLS[1:]
        }, k
        for c in metrics.CONSTRAINTS:
            low, high = metrics.BOUNDS[c]
            v = b["frame"][c].to_numpy(dtype=np.float64)[:k]
            # each out-of-band step is 900 s = 0.25 h
            assert row[f"viol_hours_{c}"] == np.count_nonzero((v < low) | (v > high)) * 0.25, (k, c)
    last = got.iloc[-1]
    assert {col: last[col] for col in RUNNING_COLS[1:]} == {
        col: b["result"][col] for col in RUNNING_COLS[1:]
    }
    assert got["profit_eur_m2"].nunique() > 1
    if controller != "rb_default":
        assert any(got[f"viol_hours_{c}"].iloc[-1] > 0 for c in metrics.CONSTRAINTS)


# --- season_replay: the animation --------------------------------------------------------------


@pytest.mark.parametrize("controller", ["rb_default", "zero_controls"])
def test_replay_draws_the_hourly_climate_over_the_bounds_bands(controller, media_dir):
    """Done when: the animation's series equal the run's frame and the bands equal metrics.BOUNDS.

    In the final frame, for each of co2, temp and rh, a line holds frame.iloc[::4]'s values against
    its t (24 hourly points for one day, t = 0, 4, ..., 92) exactly, on axes whose only patch is
    one band (axhspan) from metrics.BOUNDS[c]: co2 300-1600 ppm, temp 15-34 °C, rh 50-85 %."""
    fig, _ = _replay(controller, media_dir)

    assert isinstance(fig, Figure)
    hourly = _hourly(_baseline(controller)["frame"])
    lines = fig.findobj(Line2D)
    for c in metrics.CONSTRAINTS:
        want = (hourly["t"].astype(float).tolist(), hourly[c].tolist())
        matches = [line for line in lines if _xy(line) == want]
        assert matches, f"no line draws frame.iloc[::4][{c!r}] against t"
        for line in matches:
            patches = line.axes.patches
            assert all(isinstance(p, Rectangle) for p in patches), c  # axhspan's patch
            bands = [(p.get_y(), p.get_y() + p.get_height()) for p in patches]
            assert bands == [metrics.BOUNDS[c]], c


def test_replay_strip_is_the_hourly_actuators(media_dir):
    """rb_default's final frame has one 6-row heat strip: an image or mesh whose array is
    frame.iloc[::4]'s u_boil, u_co2, u_thscr, u_vent, u_lamp, u_blscr as rows (6 x 24 for one
    day), every value shown (nothing masked or NaN), exactly. (zero_controls' strip is all 0, so
    it couldn't show a row-order slip; rb_default's rows read differently upside down.)"""
    fig, _ = _replay("rb_default", media_dir)

    u = _hourly(_baseline("rb_default")["frame"])[U_COLS].to_numpy(dtype=np.float64).T
    assert not np.array_equal(u, u[::-1])  # non-vacuous: a flipped strip would differ
    [strip] = _strips(fig, u.shape[1])
    np.testing.assert_array_equal(_strip_array(strip), u)


@pytest.mark.parametrize("controller", ["rb_default", "zero_controls"])
def test_replay_counters_end_on_the_seasons_metrics(controller, media_dir):
    """Done when: the counters' final values equal metrics' profit_eur_m2 and viol_hours.

    The final frame's text shows the counters' last row, here the whole day (k = 96), which is the
    season's metrics: profit on a line with "€/m²", each constraint's hours on a line with its
    label, all at 2 decimals. No total of the three hours (one hour out of two bands would count
    twice; non-vacuous: at least two of them are nonzero, so the total is none of them)."""
    fig, _ = _replay(controller, media_dir)

    result = _baseline(controller)["result"]
    lines = _lines_of(_texts(fig))
    _assert_numbers(lines, result, "replay")
    hours = [result[f"viol_hours_{c}"] for c in metrics.CONSTRAINTS]
    assert sum(h > 0 for h in hours) >= 2
    total = sum(hours)
    assert total not in hours
    assert not any(_has(ln, total) for ln in lines), f"a total of the hours, {_fmt(total)}"


@pytest.mark.parametrize("controller", ["rb_default", "zero_controls"])
def test_replay_carries_its_title_and_footer_and_writes_only_a_gif(controller, media_dir):
    """The title and footer passed in are each one text on the figure, as given; no text has an
    em dash; the GIF is written (GIF8 header) and, with out_mp4=None, no MP4."""
    fig, out = _replay(controller, media_dir)

    texts = [t.get_text() for t in _texts(fig)]
    assert TITLE in texts
    assert FOOTER in texts
    assert not [t for t in texts if "—" in t]
    assert out.read_bytes()[:4] == b"GIF8"
    assert sorted(p.name for p in out.parent.iterdir()) == ["season_replay.gif"]


def test_replay_draws_its_subtitle(tmp_path):
    """season_replay takes a keyword subtitle (None by default) and draws it as one text, as
    given: the line under the title that names the controller and the simulator."""
    import greenleaf_control.media as media

    params = inspect.signature(media.season_replay).parameters
    assert "subtitle" in params, "season_replay has no subtitle keyword"
    b = _baseline("rb_default")

    fig = media.season_replay(
        b["frame"],
        _counters(b, ENDS),
        tmp_path / "season_replay.gif",
        title=TITLE,
        subtitle=SUBTITLE,
        footer=FOOTER,
    )

    texts = [t.get_text() for t in _texts(fig)]
    assert SUBTITLE in texts
    assert TITLE in texts


# --- baselines_card ----------------------------------------------------------------------------


def test_card_has_one_column_per_baseline_with_its_own_numbers_and_strip(media_dir):
    """Three strips side by side, left to right rb_default, zero_controls, random: each the
    controller's own frame.iloc[::4] actuators (6 x 24 for one day). The texts in each strip's
    column print that controller's result: profit on a line with "€/m²", each constraint's hours
    on a line with its label, fruit growth (yield_kg_m2, kg/m²) on a line with "fruit growth", all
    at 2 decimals. Non-vacuous: the three profits differ at 2 decimals."""
    fig, out = _card(media_dir)

    assert isinstance(fig, Figure)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    seasons = [_baseline(c) for c in CONTROLLERS]
    assert len({_fmt(s["result"]["profit_eur_m2"]) for s in seasons}) == 3
    strips, columns = _columns(fig, _strips(fig, len(_hourly(seasons[0]["frame"]))))
    assert len(strips) == 3
    for c, s, strip, lines in zip(CONTROLLERS, seasons, strips, columns, strict=True):
        hourly = _hourly(s["frame"])
        np.testing.assert_array_equal(
            _strip_array(strip), hourly[U_COLS].to_numpy(dtype=np.float64).T, err_msg=c
        )
        _assert_numbers(lines, s["result"], c)
        growth = s["result"]["yield_kg_m2"]
        assert any("fruit growth" in ln and _has(ln, growth) for ln in lines), (c, _fmt(growth))


def test_card_labels_say_what_the_numbers_are(media_dir):
    """rb_default's column names it "upstream's default rule controller (the grower's rules)"
    (line breaks allowed); the footer passed in is one text, as given; no text says "yield" (it is
    fruit growth, not a harvest) or has an em dash."""
    fig, _ = _card(media_dir)

    seasons = [_baseline(c) for c in CONTROLLERS]
    strips, columns = _columns(fig, _strips(fig, len(_hourly(seasons[0]["frame"]))))
    assert len(strips) == 3
    assert RB_LABEL in " ".join(" ".join(columns[0]).split())
    texts = [t.get_text() for t in _texts(fig)]
    assert FOOTER in texts
    assert not [t for t in texts if "yield" in t.lower()]
    assert not [t for t in texts if "—" in t]


def test_card_says_what_profit_counts(media_dir):
    """One text on the card says what profit is (metrics' profit_eur_m2: fruit growth x 1.6 €/kg
    minus the heat, lamp power and CO₂ dosing costs): it contains "fruit value minus" and names
    heat, power and CO₂ (line breaks allowed)."""
    fig, _ = _card(media_dir)

    texts = [" ".join(t.get_text().split()) for t in _texts(fig)]
    says = [t for t in texts if "fruit value minus" in t]
    assert says, "no text says profit is fruit value minus the costs"
    assert any(all(word in t for word in ("heat", "power", "CO₂")) for t in says), says


def _share_images(fig: Figure, n_days: int) -> list:
    """Every image or mesh on fig holding a 3-row (constraints) x n_days array, left to right."""
    found = fig.findobj(lambda a: isinstance(a, (AxesImage, QuadMesh)))
    shares = [a for a in found if np.shape(a.get_array()) == (3, n_days)]
    return sorted(shares, key=lambda a: a.axes.get_position().x0)


def test_card_share_of_each_day_outside_the_band_is_the_graders_hours(media_dir):
    """The card's per-day share of steps outside each safe range agrees with the grader: on a
    1-day season each column's 3 x 1 array (rows co2, temp, rh), times 24 h, is that controller's
    viol_hours_<c>. E.g. a day with 51 of its 96 steps outside the RH band: 51 / 96 x 24 h =
    12.75 h = 51 x 0.25 h. Columns left to right: rb_default, zero_controls, random.
    Non-vacuous: some share is strictly between 0 and 1."""
    fig, _ = _card(media_dir)

    shares = _share_images(fig, 1)
    assert len(shares) == 3
    seen = []
    for c, image in zip(CONTROLLERS, shares, strict=True):
        share = _strip_array(image)
        result = _baseline(c)["result"]
        for j, k in enumerate(metrics.CONSTRAINTS):
            # share is the day's out-of-band steps / 96; x 24 h = out-of-band steps x 0.25 h
            want = result[f"viol_hours_{k}"]
            assert share[j, 0] * 24 == pytest.approx(want, abs=1e-9), (c, k)
            seen.append(share[j, 0])
    assert any(0 < s < 1 for s in seen)


# --- footer ------------------------------------------------------------------------------------


@pytest.mark.parametrize("dirty", [True, False], ids=["dirty", "clean"])
def test_footer_names_source_scenario_simulator_and_commit(dirty):
    """Done when: every footer names its source and commit.

    footer("rb_default", stamp) for a stamp of commit SHA: the source, "PAPER (Amsterdam 2010)",
    "GreenLight-Gym @2d3febb" (the installed gl-gym's commit, from manifest._versions(), to 7
    characters), "commit 0123456", "(dirty)" exactly when the stamp is dirty, and "simulator run,
    not a registered result". No em dash."""
    import greenleaf_control.media as media

    stamp = {"commit": SHA, "dirty": dirty, "fingerprints": {}, "weather_hashes": {}}

    text = media.footer("rb_default", stamp)

    gl7 = manifest._versions()["gl-gym"][:7]
    assert gl7 == "2d3febb"  # the pinned simulator (AGENTS.md, gl_gym @2d3febb)
    for part in (
        "rb_default",
        "PAPER (Amsterdam 2010)",
        f"GreenLight-Gym @{gl7}",
        f"commit {SHA[:7]}",
        "simulator run, not a registered result",
    ):
        assert part in text, (part, text)
    assert ("(dirty)" in text) is dirty, text
    assert "—" not in text


def test_footer_reads_the_simulator_commit_from_the_install(monkeypatch):
    """The simulator's commit comes from manifest._versions() at call time, not a hard-coded
    2d3febb: with _versions patched to report gl-gym feedfacecafe..., the footer says
    "GreenLight-Gym @feedfac" and not 2d3febb."""
    import greenleaf_control.media as media

    real = manifest._versions()
    monkeypatch.setattr(
        manifest, "_versions", lambda: {**real, "gl-gym": "feedfacecafebeef" + "0" * 24}
    )
    stamp = {"commit": SHA, "dirty": False, "fingerprints": {}, "weather_hashes": {}}

    text = media.footer("zero_controls", stamp)

    assert "GreenLight-Gym @feedfac" in text, text
    assert "2d3febb" not in text, text


def test_footer_takes_a_place_and_a_note():
    """footer(source, stamp, *, place, note): the card's footer swaps PAPER for its place and the
    simulator-run note for its own, keeping the source and both commits. Called with neither, it
    is today's PAPER footer: footer(s, stamp) == footer(s, stamp, place=PAPER_NAME, note="A
    simulator run, not a registered result")."""
    import greenleaf_control.media as media

    params = inspect.signature(media.footer).parameters
    assert {"place", "note"} <= set(params), f"footer has no place and note keywords: {params}"
    stamp = {"commit": SHA, "dirty": False, "fingerprints": {}, "weather_hashes": {}}
    source = "result-0123abcd, decision-4567cdef and rb_tuned_validation.parquet"

    text = " ".join(media.footer(source, stamp, place=FOOTER_PLACE, note=CARD_NOTE).split())

    for part in (source, FOOTER_PLACE, CARD_NOTE, f"commit {SHA[:7]}", "GreenLight-Gym @"):
        assert part in text, (part, text)
    assert PAPER_NOTE not in text and "PAPER" not in text, text
    assert "—" not in text
    assert media.footer("rb_default", stamp) == media.footer(
        "rb_default", stamp, place=media.PAPER_NAME, note=PAPER_NOTE
    )


# --- The trade-off still: TRADEOFF, tradeoff_points and rb_tuned_reference (issue #50) ----------


def test_tradeoff_names_the_issues_rows_never_numbers():
    """TRADEOFF holds the issue's four (result_id, arm, status) entries (E7's lag_b100 may join
    them later) and only ids and statuses: each a result-<8 hex> id, an arm name and adopted or
    dropped. E1_RESULT is rb_tuned's E1 row, result-525a6e90."""
    media = _media_card()

    entries = [tuple(e) for e in media.TRADEOFF]

    assert [e for e in ISSUE_TRADEOFF if e not in entries] == [], entries
    for result, arm, status in entries:
        assert re.fullmatch(r"result-[0-9a-f]{8}", result), result
        assert isinstance(arm, str) and arm, arm
        assert status in ledger.DECISIONS, status
    assert media.E1_RESULT == "result-525a6e90"


def test_tradeoff_points_read_each_entrys_rows(sources):
    """Done when: numbers are read from the ledger when the card is drawn.

    On the synthetic ledger, tradeoff_points(path) gives one point per TRADEOFF entry, in order,
    holding the entry's arm and status, its result, decision and registered row ids, the
    registered row's seeds, and from the result's per_arm[arm]: profit and each viol_hours_<c> as
    (iqm, ci_low, ci_high) and yield_kg_m2 as the guard's iqm, all exactly. E.g. ppo's profit
    (2.40, 2.33, 2.46) €/m² over seeds 0-4. Non-vacuous: E4 also holds a residual_w05 (profit
    0.63), but the card's residual_w05 is E6's (0.77, seeds 100-109); and residual_w4's status
    comes from TRADEOFF (dropped), not from E4's one decision row (adopted)."""
    media = _media_card()
    src = sources

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(media, "TRADEOFF", src["tradeoff"])
        got = media.tradeoff_points(src["ledger"])

    want = _want_points(src)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert _json({key: g.get(key) for key in w}) == _json(w), w["arm"]


@pytest.mark.parametrize("case", REFUSALS)
def test_tradeoff_points_refuse_a_row_that_cant_be_drawn(case, sources, tmp_path, monkeypatch):
    """Done when: each refusal in step 2 raises.

    tradeoff_points raises ValueError naming the row when a TRADEOFF entry's row is missing
    (missing) or isn't a result (not_a_result: a decision id); its result has no decision row
    (undecided) or two (two_decisions); its registration's scenarios aren't exactly
    ["validation"] (not_validation: ["train_eval"]; the error may name the registered row); the
    arm isn't in per_arm (arm_absent); its iqm is null (iqm_null); a run crashed (crashed: 1);
    invalid_episodes is 1; or a guard the card draws is missing (no_rh_guard, no_yield_guard).
    The flawed entry comes after the four good ones, so every entry is checked."""
    media = _media_card()
    path = tmp_path / "ledger.jsonl"
    tradeoff, named = _refusal(case, sources, path)
    monkeypatch.setattr(media, "TRADEOFF", tradeoff)

    with pytest.raises(ValueError) as excinfo:
        media.tradeoff_points(path)

    assert any(row in str(excinfo.value) for row in named), (named, str(excinfo.value))


@pytest.mark.parametrize("n", [5, 10])
def test_rb_tuned_reference_is_the_tiled_iqm_of_the_parquet_seasons(n, sources, monkeypatch):
    """Done when: rb_tuned's reference equals stats.iqm_ci of the tiled parquet seasons.

    rb_tuned_reference(n, path, parquet) gives profit_eur_m2, each viol_hours_<c> and
    yield_kg_m2 as stats.iqm_ci(np.tile(seasons, (n, 1)))[0], exactly, which are the REF values
    (e.g. n = 10: profit (3 x -3.0 + 10 x 0.6 + 3 x 1.2) / 16 = 0.0375 €/m²), plus n_seeds = n
    and sha256, the parquet file's. Like-for-like, not the plain mean: (-3.0 + 0.6 + 1.2) / 3 =
    -0.40 €/m² differs at 2 decimals."""
    media = _media_card()
    monkeypatch.setattr(media, "E1_RESULT", sources["e1"])

    ref = media.rb_tuned_reference(n, sources["ledger"], sources["parquet"])

    for key, want in REF[n].items():
        assert ref[key] == _tiled(key, n), key
        assert ref[key] == pytest.approx(want, rel=1e-12), key
    assert ref["n_seeds"] == n
    assert ref["sha256"] == hashlib.sha256(sources["parquet"].read_bytes()).hexdigest()
    assert _fmt(ref["profit_eur_m2"]) != _fmt(np.mean(RB_SEASONS["profit_eur_m2"]))


@pytest.mark.parametrize("case", list(PARQUET_MISMATCHES))
def test_rb_tuned_reference_refuses_a_parquet_that_disagrees_with_e1(
    case, sources, tmp_path, monkeypatch
):
    """Done when: a parquet that disagrees with the E1 row raises.

    The parquet's per-season means must equal E1's row: each viol_hours_<c> exactly (one season
    0.25 h off moves the temp mean by 0.0833 h), paper_return within rel 1e-9 (every season
    1e-6 high is rel 1e-6 off). Either raises ValueError naming E1's row or the parquet."""
    media = _media_card()
    monkeypatch.setattr(media, "E1_RESULT", sources["e1"])
    parquet = tmp_path / PARQUET
    _write_parquet(parquet, PARQUET_MISMATCHES[case])

    with pytest.raises(ValueError) as excinfo:
        media.rb_tuned_reference(10, sources["ledger"], parquet)

    assert sources["e1"] in str(excinfo.value) or PARQUET in str(excinfo.value), excinfo.value


def test_rb_tuned_reference_takes_paper_return_within_rel_1e9(sources, tmp_path, monkeypatch):
    """paper_return is checked to rel 1e-9, not exactly: a parquet whose every paper_return is
    rel 1e-11 high (float noise, as a different summation order gives) still matches E1, and its
    reference is the exact parquet's."""
    media = _media_card()
    monkeypatch.setattr(media, "E1_RESULT", sources["e1"])
    parquet = tmp_path / PARQUET
    noisy = tuple(v * (1 + 1e-11) for v in RB_SEASONS["paper_return"])
    _write_parquet(parquet, {**RB_SEASONS, "paper_return": noisy})

    ref = media.rb_tuned_reference(10, sources["ledger"], parquet)

    assert ref["profit_eur_m2"] == _tiled("profit_eur_m2", 10)


# --- The trade-off still: tradeoff_card (issue #50) ---------------------------------------------


def test_card_draws_each_point_at_its_ledger_numbers_with_bars_to_its_cis(media_dir, sources):
    """Done when: in each panel, each marker sits at (guard iqm, profit iqm) and its bars end at
    the ledger's ci_low and ci_high.

    Three panels, left to right viol_hours_co2, _temp and _rh (metrics.CONSTRAINTS), each x label
    saying hours outside the safe range and naming its constraint, sharing one y axis labelled
    profit [€/m²]. In each, every point is one errorbar whose marker is (its hours iqm, its profit
    iqm), with one horizontal bar from the hours' ci_low to ci_high at the profit iqm and one
    vertical bar from the profit's ci_low to ci_high at the hours iqm. E.g. ppo in the temp panel:
    marker (153.25 h, 2.40 €/m²), bars x 140.00-166.50 and y 2.33-2.46. Four errorbars with bars
    per panel: rb_tuned has none."""
    fig, _ = _tradeoff(media_dir)

    panels = _panels(fig)
    assert len(panels) == 3, f"{len(panels)} panels hold errorbars"
    words = {"co2": ("CO₂",), "temp": ("temp",), "rh": ("RH", "humidity")}
    for c, ax in zip(metrics.CONSTRAINTS, panels, strict=True):
        titles = [ax.get_title(loc) for loc in ("left", "center", "right")]
        label = " ".join(" ".join([ax.get_xlabel(), *titles]).split())
        assert "hours outside" in label.lower() and any(w in label for w in words[c]), (c, label)
        assert ax is panels[0] or panels[0].get_shared_y_axes().joined(panels[0], ax), c
        assert sum(bool(_segments(e)) for e in _errorbars(ax)) == 4, c
        for p in _want_points(sources):
            hours, profit = p[f"viol_hours_{c}"], p["profit"]
            where = (c, p["arm"])
            found = _errorbar_at(ax, hours[0], profit[0])
            assert len(found) == 1, where
            segments = _segments(found[0])
            assert len(segments) == 2, where
            [[(x0, y0), (x1, y1)]] = [s for s in segments if s[0][1] == s[1][1]]
            assert (sorted([x0, x1]), y0) == (pytest.approx(list(hours[1:])), profit[0]), where
            [[(x0, y0), (x1, y1)]] = [s for s in segments if s[0][0] == s[1][0]]
            assert (x0, sorted([y0, y1])) == (hours[0], pytest.approx(list(profit[1:]))), where
    ylabel = " ".join(panels[0].get_ylabel().split())
    assert "profit" in ylabel and "€/m²" in ylabel, ylabel


def test_card_marks_the_dropped_arm_and_rb_tuned_apart(media_dir, sources):
    """In every panel: the adopted arms' markers are filled, residual_w05's LEAF; the dropped
    arm's (residual_w4) is hollow with a MUTED edge. rb_tuned is one hollow marker, of a shape
    no arm uses, at its largest-n (10-seed) reference (hours and profit to rel 1e-9; profit
    0.0375 €/m²), with no bars; nothing is drawn at the 5-seed reference."""
    media = _media_card()
    fig, _ = _tradeoff(media_dir)

    panels = _panels(fig)
    assert len(panels) == 3
    for c, ax in zip(metrics.CONSTRAINTS, panels, strict=True):
        markers = {}
        for p in _want_points(sources):
            found = _errorbar_at(ax, p[f"viol_hours_{c}"][0], p["profit"][0])
            assert len(found) == 1, (c, p["arm"])
            markers[p["arm"]] = found[0].lines[0]
        for arm, m in markers.items():
            hollow = to_rgba(m.get_markerfacecolor())[3] == 0
            assert hollow is (arm == DROPPED), (c, arm, m.get_markerfacecolor())
        assert to_hex(markers[DROPPED].get_markeredgecolor()) == media.MUTED, c
        assert to_hex(markers["residual_w05"].get_markerfacecolor()) == media.LEAF, c

        rb = _lines_near(ax, _tiled(f"viol_hours_{c}", 10), _tiled("profit_eur_m2", 10))
        assert len(rb) == 1, (c, "rb_tuned's 10-seed marker")
        assert to_rgba(rb[0].get_markerfacecolor())[3] == 0, (c, "rb_tuned is hollow")
        shape = rb[0].get_marker()
        assert shape not in (None, "", " ", "None", "none"), c
        assert shape not in {m.get_marker() for m in markers.values()}, (c, shape)
        assert not [e for e in _errorbars(ax) if e.lines[0] is rb[0] and _segments(e)], c
        assert not _lines_near(ax, _tiled(f"viol_hours_{c}", 5), _tiled("profit_eur_m2", 5)), c


def test_card_reads_the_ledger_when_drawn(sources, tmp_path):
    """Done when: editing a number in the tmp ledger moves the drawn point.

    With E6's residual_w05 profit iqm edited from 0.77 to 0.81 €/m² and its viol_hours_temp iqm
    from 48.50 to 51.25 h (each still inside its CI, 0.70-0.84 and 42.00-55.75), the temp panel's
    residual_w05 marker sits at (51.25, 0.81) and none at (48.50, 0.77)."""
    media = _media_card()
    path = tmp_path / "ledger.jsonl"
    e6 = sources["rows"]["E6"]["result"]
    _edited(sources["ledger"], path, e6, "residual_w05", profit=0.81, viol_hours_temp=51.25)

    fig = _draw_tradeoff(media, sources, tmp_path / "tradeoff.png", ledger_path=path)

    panels = _panels(fig)
    assert len(panels) == 3
    temp = panels[metrics.CONSTRAINTS.index("temp")]
    assert len(_errorbar_at(temp, 51.25, 0.81)) == 1
    assert _errorbar_at(temp, 48.5, 0.77) == []


def test_card_png_is_linkedins_1200_by_627(media_dir):
    """Done when: the PNG is exactly 1200 x 627 px (PIL): 12 x 6.27 in at 100 dpi."""
    _, out = _tradeoff(media_dir)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(out) as im:
        assert im.size == (1200, 627)


def test_card_legend_names_each_points_result_outside_the_panels(media_dir, sources):
    """Done when: every legend line names its result id.

    One legend entry per point plus rb_tuned's. Each point's names its arm and result id, "<n>
    seeds × 3 VALIDATION seasons" (5 for ppo, residual_w1 and residual_w4; 10 for residual_w05),
    its profit IQM and CI at 2 decimals, and "fruit growth" with its yield_kg_m2; "dropped" is on
    residual_w4's only. rb_tuned's names it, E1's result id and its 10-seed profit (0.04). No
    legend overlaps a panel."""
    fig, _ = _tradeoff(media_dir)

    legends, texts = _legend_texts(fig)
    points = _want_points(sources)
    assert len(texts) == len(points) + 1, texts
    for p in points:
        mine = [t for t in texts if p["arm"] in t and p["result"] in t]
        assert len(mine) == 1, (p["arm"], texts)
        line = mine[0]
        assert f"{len(p['seeds'])} seeds × 3 VALIDATION seasons" in line, line
        assert all(_has(line, v) for v in p["profit"]), (p["profit"], line)
        assert "fruit growth" in line and _has(line, p["yield_kg_m2"]), line
        assert ("dropped" in line) is (p["status"] == "dropped"), line
    rb = [t for t in texts if "rb_tuned" in t]
    assert len(rb) == 1, texts
    assert sources["e1"] in rb[0] and _has(rb[0], _tiled("profit_eur_m2", 10)), rb[0]
    assert "dropped" not in rb[0]

    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for lg in legends:
        box = lg.get_window_extent(renderer)
        assert not [ax for ax in _panels(fig) if box.overlaps(ax.get_window_extent(renderer))]


def test_card_says_validation_and_how_it_was_scored(media_dir):
    """Above the panels: a title saying VALIDATION and Amsterdam 2010 (scenarios.VALIDATION's
    place); the line "IQM over seeds × seasons, 95% stratified bootstrap CI"; and a note giving
    rb_tuned's profit at the points' other seed count, 5: "5-seed" with -0.07 €/m² (REF[5],
    -0.0667) on one text."""
    fig, _ = _tradeoff(media_dir)

    texts = [" ".join(t.get_text().split()) for t in _texts(fig)]

    assert [t for t in texts if "VALIDATION" in t and VAL_PLACE in t], texts
    assert [t for t in texts if CI_LINE in t], texts
    ref5 = _tiled("profit_eur_m2", 5)
    assert [t for t in texts if "5-seed" in t and _has(t, ref5) and "€/m²" in t], texts


def test_card_keeps_medias_text_rules(media_dir, sources):
    """Done when: no em dash, no "yield" and no summed hours appear in any text.

    The footer passed in is one text, as given. No text has an em dash or says "yield" (it is
    fruit growth, not a harvest), and no line prints a point's three hour IQMs summed: ppo
    0.5 + 153.25 + 135 = 288.75 h, residual_w05 264.25, residual_w1 303.75, residual_w4 296.75,
    rb_tuned at 10 seeds 1.5625 + 92.859375 + 254.578125 = 349.00 and at 5 seeds 350.39.
    Non-vacuous: each has two or more nonzero hours, so its total is none of them."""
    fig, _ = _tradeoff(media_dir)

    texts = [t.get_text() for t in _texts(fig)]
    lines = _lines_of(_texts(fig))
    assert CARD_FOOTER in texts
    assert not [t for t in texts if "—" in t]
    assert not [t for t in texts if "yield" in t.lower()]
    hour_sets = [[p[key][0] for key in HOUR_KEYS] for p in _want_points(sources)]
    hour_sets += [[_tiled(key, n) for key in HOUR_KEYS] for n in (5, 10)]
    for hours in hour_sets:
        total = sum(hours)
        assert sum(h > 0 for h in hours) >= 2 and total not in hours
        assert not [ln for ln in lines if _has(ln, total)], f"a total of the hours, {_fmt(total)}"


# --- the CLI -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["no_out", "no_ffmpeg"])
def test_main_refuses_before_any_season(case, sources, tmp_path, monkeypatch, capsys):
    """--out is required; and without ffmpeg (FFMpegWriter.isAvailable() false) main refuses with a
    nonzero SystemExit naming ffmpeg. Both come before any season is played, and nothing is
    written: --out isn't created. The card, if main reads it first, reads the synthetic rows."""
    import greenleaf_control.media as media

    _no_season_played(monkeypatch, media)
    _card_reads(monkeypatch, media, sources)
    out = tmp_path / "media"
    if case == "no_out":
        argv = []
    else:
        monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
        argv = ["--out", str(out)]

    with pytest.raises(SystemExit) as excinfo:
        media.main(argv)

    assert excinfo.value.code not in (0, None)
    err = capsys.readouterr().err
    assert ("--out" if case == "no_out" else "ffmpeg") in err.lower(), err
    assert not out.exists()


def _whole(text: str, n: int) -> bool:
    """text prints n as a whole number (2 in "2 days", not in "2010" or "12")."""
    return re.search(rf"(?<!\d){n}(?!\d)", text) is not None


def _drawing_recorded(monkeypatch, media) -> dict:
    """season_replay, baselines_card and tradeoff_card (once it exists) replaced by recorders of
    their arguments that draw and write nothing, so main's own choices (its title, subtitle and
    footers) are read fast."""
    calls = {}

    def record(name: str):
        def drawn(*args, **kwargs):
            calls[name] = kwargs
            return Figure()

        return drawn

    monkeypatch.setattr(media, "season_replay", record("replay"))
    monkeypatch.setattr(media, "baselines_card", record("card"))
    monkeypatch.setattr(media, "tradeoff_card", record("tradeoff"), raising=False)
    return calls


def test_main_titles_the_replay_as_simulated_with_the_runs_days_place_and_controller(
    sources, tmp_path, monkeypatch
):
    """The replay's title says it is simulated, with the season's day count taken from the run and
    PAPER's location and year (Amsterdam 2010, from scenarios.PAPER); its subtitle names
    "upstream's default rule controller" and "GreenLight-Gym". The 8 pt footer alone is unreadable
    on a phone, so a real greenhouse must not be what the title reads as.

    Known answer: with play shortened to real 2-day seasons (192 steps), the title says 2 (days),
    never 60, so the count can't be typed in. Drawing is recorded, not done (the real
    season_replay is tested above), ffmpeg is reported available, and the card reads the
    synthetic rows, so this runs anywhere.
    """
    import greenleaf_control.media as media

    real_play = media.play
    monkeypatch.setattr(
        media, "play", lambda controller, season_length=60: real_play(controller, 2)
    )
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    _card_reads(monkeypatch, media, sources)
    calls = _drawing_recorded(monkeypatch, media)

    assert media.main(["--out", str(tmp_path / "media")]) == 0

    title = calls["replay"]["title"]
    assert "simulated" in title.lower(), title
    assert "day" in title.lower() and _whole(title, 2), title  # 192 steps / 96 a day
    assert not _whole(title, 60), title
    assert PLACE in " ".join(title.split()), title
    subtitle = calls["replay"].get("subtitle") or ""
    assert all(part in " ".join(subtitle.split()) for part in RB_SUBTITLE), subtitle
    assert "—" not in title + subtitle


@pytest.mark.parametrize("invalid", CONTROLLERS)
def test_main_refuses_an_invalid_season_before_writing(
    invalid, sources, tmp_path, monkeypatch, capsys
):
    """A season whose result isn't valid (metrics: a solver failure, or steps != season_steps) is
    never drawn: main exits nonzero naming that controller, and only that one, before anything is
    drawn or written (--out stays absent or empty). play returns the real 1-day seasons, the one
    under test marked invalid with one solver failure; ffmpeg is reported available; the card
    reads the synthetic rows."""
    import greenleaf_control.media as media

    _card_reads(monkeypatch, media, sources)

    def play(controller, season_length=60):
        b = _baseline(controller)
        if controller != invalid:
            return b
        return {**b, "result": {**b["result"], "valid": False, "solver_failures": 1}}

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(
                f"media.{name} was called: an invalid season must be refused first"
            )

        return refuse

    monkeypatch.setattr(media, "play", play)
    monkeypatch.setattr(media, "season_replay", trip("season_replay"))
    monkeypatch.setattr(media, "baselines_card", trip("baselines_card"))
    monkeypatch.setattr(media, "tradeoff_card", trip("tradeoff_card"), raising=False)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    out = tmp_path / "media"

    try:
        code = media.main(["--out", str(out)])
    except SystemExit as e:
        code = e.code
    text = f"{code} {capsys.readouterr().err}"

    assert code not in (0, None), text
    assert invalid in text, text
    assert not [c for c in CONTROLLERS if c != invalid and c in text], text
    assert not out.exists() or not list(out.iterdir())


@pytest.mark.parametrize("case", [*REFUSALS, *PARQUET_MISMATCHES])
def test_main_refuses_a_bad_row_or_parquet_before_any_season(
    case, sources, tmp_path, monkeypatch, capsys
):
    """Done when: each refusal, and a parquet that disagrees with the E1 row, raises before any
    file exists.

    main builds the card's points and references first, so a row tradeoff_points refuses (each of
    REFUSALS, as in its test) or a parquet rb_tuned_reference refuses (each of
    PARQUET_MISMATCHES) stops it before any season is played (every way to play one trips) or
    anything is written: it exits nonzero or raises ValueError, naming the row (or E1's row or
    the parquet), and --out isn't created. ffmpeg is reported available."""
    media = _media_card()
    _no_season_played(monkeypatch, media)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    if case in REFUSALS:
        path = tmp_path / "ledger.jsonl"
        tradeoff, named = _refusal(case, sources, path)
        _card_reads(monkeypatch, media, sources, tradeoff=tradeoff, ledger=path)
    else:
        parquet = tmp_path / PARQUET
        _write_parquet(parquet, PARQUET_MISMATCHES[case])
        named = {sources["e1"], PARQUET}
        _card_reads(monkeypatch, media, sources, parquet=parquet)
    out = tmp_path / "media"

    raised = ""
    try:
        code = media.main(["--out", str(out)])
    except SystemExit as e:
        code = e.code
    except ValueError as e:
        code, raised = 1, str(e)
    text = f"{code} {raised} {capsys.readouterr().err}"

    assert code not in (0, None), text
    assert any(row in text for row in named), (named, text)
    assert not out.exists()


def test_main_writes_the_tradeoff_card_and_its_media_json_entry(sources, tmp_path, monkeypatch):
    """main writes tradeoff.png (1200 x 627 px) and adds its media.json entry; the other entries
    are unchanged.

    play returns the real 1-day seasons; the replay and the baselines card are recorded (they
    write nothing); the real tradeoff_card draws, spied on; the card reads the synthetic rows.
    - The card gets tradeoff_points' points (every number as the rows were written) and one
      reference per distinct seed count among them (5 and 10), and its footer names every
      point's result and decision row, E1's result row, rb_tuned_validation.parquet,
      "VALIDATION, Amsterdam 2010 days 45, 59, 75", "commit <sha7>" and "Recorded VALIDATION
      results; the 2012 test weather is still sealed", and not the simulator-run note.
    - --out holds media.json and tradeoff.png only.
    - media.json's tradeoff.png entry has today's keys plus split "validation" and ledger (every
      point's result and decision id and E1's result id), scenario scenarios.VALIDATION, this
      checkout's commit, and metrics {points, references}, what the card drew.
    - The replay's and the baselines card's entries keep exactly today's keys, scenario PAPER
      and their seasons' numbers."""
    media = _media_card()
    monkeypatch.setattr(media, "play", lambda controller, season_length=60: _baseline(controller))
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    _card_reads(monkeypatch, media, sources)
    real_card = media.tradeoff_card
    calls = _drawing_recorded(monkeypatch, media)

    def card(*args, **kwargs):
        calls["tradeoff"] = inspect.signature(real_card).bind(*args, **kwargs).arguments
        return real_card(*args, **kwargs)

    monkeypatch.setattr(media, "tradeoff_card", card)
    stamp = manifest.code_stamp(smoke=True)
    out = tmp_path / "media"

    assert media.main(["--out", str(out)]) == 0

    drawn = calls["tradeoff"]
    want = _want_points(sources)
    assert _json(
        [{key: p.get(key) for key in w} for p, w in zip(drawn["points"], want, strict=True)]
    ) == _json(want)
    assert sorted(r["n_seeds"] for r in drawn["references"]) == [5, 10]
    foot = " ".join(drawn["footer"].split())
    ids = [p[key] for p in want for key in ("result", "decision")] + [sources["e1"]]
    for part in (*ids, PARQUET, FOOTER_PLACE, f"commit {stamp['commit'][:7]}", CARD_NOTE):
        assert part in foot, (part, foot)
    assert PAPER_NOTE not in foot, foot

    assert sorted(p.name for p in out.iterdir()) == ["media.json", "tradeoff.png"]
    with Image.open(out / "tradeoff.png") as im:
        assert im.size == (1200, 627)

    by_file = {e["file"]: e for e in json.loads((out / "media.json").read_text())}
    assert sorted(by_file) == [
        "baselines.png",
        "season_replay.gif",
        "season_replay.mp4",
        "tradeoff.png",
    ]
    e = by_file["tradeoff.png"]
    assert ENTRY_KEYS | {"split", "ledger"} <= set(e), sorted(e)
    assert (e["split"], e["scenario"]) == ("validation", VALIDATION)
    assert e["commit"] == stamp["commit"] and e["dirty"] is stamp["dirty"]
    assert not [i for i in ids if i not in e["ledger"]], e["ledger"]
    assert e["metrics"] == _json({"points": drawn["points"], "references": drawn["references"]})
    rb = _baseline("rb_default")["result"]
    for name in ("season_replay.gif", "season_replay.mp4", "baselines.png"):
        assert set(by_file[name]) == ENTRY_KEYS, name
        assert by_file[name]["scenario"] == PAPER, name
    assert by_file["season_replay.gif"]["metrics"] == _json(rb)
    assert by_file["baselines.png"]["metrics"] == _json(
        {c: _baseline(c)["result"] for c in CONTROLLERS}
    )


def test_cli_runs_as_a_module_and_lists_out(tmp_path):
    """python -m greenleaf_control.media --help exits 0 and lists --out (the issue's one command
    is python -m greenleaf_control.media --out docs/media)."""
    result = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.media", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout


@pytest.mark.parametrize("module", ["matplotlib.pyplot"])
def test_media_never_imports_pyplot(module):
    """media builds its Figures and animation directly, never through pyplot (headless, no global
    state), as viz does."""
    path = Path(greenleaf_control.__file__).parent / "media.py"
    assert path.is_file(), f"{path} doesn't exist"
    imported = _imported_modules(path)
    assert not [name for name in imported if name == module or name.startswith(f"{module}.")]


def test_signatures():
    """Done when: test_signatures covers the new functions and footer's keywords.

    play(controller, season_length=60), running(states, controls, params, ends),
    season_replay(frame, counters, out_gif, out_mp4=None, *, title, subtitle=None, footer, fps=15),
    baselines_card(seasons, out, *, footer), footer(source, stamp, *, place=PAPER_NAME, note="A
    simulator run, not a registered result"), main(argv=None), and issue #50's
    tradeoff_points(path=ledger.LEDGER), rb_tuned_reference(n_seeds, path=ledger.LEDGER,
    parquet=selection.RESULTS / "rb_tuned_validation.parquet") and tradeoff_card(points,
    references, out, *, footer)."""
    import greenleaf_control.media as media

    def shape(fn) -> list[tuple[str, str, object]]:
        return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]

    empty = inspect.Parameter.empty
    pos, kw = "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"
    assert shape(media.play) == [("controller", pos, empty), ("season_length", pos, 60)]
    assert shape(media.running) == [
        (name, pos, empty) for name in ("states", "controls", "params", "ends")
    ]
    assert shape(media.season_replay) == [
        ("frame", pos, empty),
        ("counters", pos, empty),
        ("out_gif", pos, empty),
        ("out_mp4", pos, None),
        ("title", kw, empty),
        ("subtitle", kw, None),
        ("footer", kw, empty),
        ("fps", kw, 15),
    ]
    assert shape(media.baselines_card) == [
        ("seasons", pos, empty),
        ("out", pos, empty),
        ("footer", kw, empty),
    ]
    assert shape(media.footer) == [
        ("source", pos, empty),
        ("stamp", pos, empty),
        ("place", kw, media.PAPER_NAME),
        ("note", kw, PAPER_NOTE),
    ]
    assert shape(media.main) == [("argv", pos, None)]
    _media_card()
    assert shape(media.tradeoff_points) == [("path", pos, ledger.LEDGER)]
    assert shape(media.rb_tuned_reference) == [
        ("n_seeds", pos, empty),
        ("path", pos, ledger.LEDGER),
        ("parquet", pos, selection.RESULTS / PARQUET),
    ]
    assert shape(media.tradeoff_card) == [
        ("points", pos, empty),
        ("references", pos, empty),
        ("out", pos, empty),
        ("footer", kw, empty),
    ]


# --- The one command, end to end ---------------------------------------------------------------


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "stream=codec_name,width:format=duration"),
            *("-of", "json", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _gif_seconds(path: Path) -> float:
    with Image.open(path) as im:
        total_ms = 0
        for i in range(im.n_frames):
            im.seek(i)
            total_ms += im.info["duration"]
    return total_ms / 1000


def _assert_golden(got: dict, want: dict, name: str) -> None:
    """The card's numbers against tests/golden/values.json "ours": hours exactly (multiples of
    0.25 h), profit and fruit growth at rel 1e-6, as test_golden does."""
    for c in metrics.CONSTRAINTS:
        assert got[f"viol_hours_{c}"] == want[f"viol_hours_{c}"], (name, c)
    for key in ("profit_eur_m2", "yield_kg_m2"):
        assert got[key] == pytest.approx(want[key], rel=1e-6), (name, key)


@pytest.mark.slow  # three 60-day seasons, then the GIF, the MP4 and the card
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)
def test_one_command_writes_every_file_labelled_with_its_source_and_commit(
    sources, tmp_path, monkeypatch
):
    """Done when: one command writes the GIF, the MP4, the baselines PNG, the trade-off PNG and
    media.json into --out.

    main(["--out", out]) at its defaults (60-day PAPER seasons) returns 0 and writes exactly
    season_replay.gif, season_replay.mp4, baselines.png, tradeoff.png and media.json:
    - the GIF under 8 MB and 8-20 s long (the README's ~10-15 s);
    - the MP4 H.264, 1080 px wide and 8-20 s long by ffprobe (LinkedIn);
    - the PNG under 500 KB;
    - tradeoff.png 1200 x 627 px (PIL) and under 1 MB, drawn from the synthetic rows (a test
      never reads the committed ledger or results);
    - media.json a list of one entry per picture, each with file, source, scenario, commit
      and dirty (this checkout's code_stamp, taken before writing), gl_gym (the installed gl-gym's
      commit, 7 characters or more), made_at (ISO 8601) and metrics. The replay's entries: source
      rb_default, metrics its season's; the card's: source naming all three baselines, metrics
      keyed by baseline; both scenario PAPER. Every number equals tests/golden/values.json "ours"
      (hours exactly, the rest at rel 1e-6). The trade-off's: scenario scenarios.VALIDATION,
      split "validation".
    - The footers drawn (season_replay's and baselines_card's footer argument, spied on; the real
      functions run) name their sources and "commit <sha7>".
    - The replay's title says "simulated", 60 (days) and Amsterdam 2010 (from scenarios.PAPER);
      its subtitle, drawn on the final figure, names "upstream's default rule controller" and
      "GreenLight-Gym".
    """
    media = _media_card()
    _card_reads(monkeypatch, media, sources)

    stamp = manifest.code_stamp(smoke=True)
    sha7 = stamp["commit"][:7]
    footers, replays = {}, {}
    real_replay, real_card = media.season_replay, media.baselines_card

    def replay(*args, **kwargs):
        footers["replay"] = kwargs["footer"]
        replays["kwargs"] = kwargs
        replays["fig"] = real_replay(*args, **kwargs)
        return replays["fig"]

    def card(*args, **kwargs):
        footers["card"] = kwargs["footer"]
        return real_card(*args, **kwargs)

    monkeypatch.setattr(media, "season_replay", replay)
    monkeypatch.setattr(media, "baselines_card", card)
    out = tmp_path / "media"

    assert media.main(["--out", str(out)]) == 0

    assert sorted(p.name for p in out.iterdir()) == MEDIA_FILES
    gif, mp4, png = out / "season_replay.gif", out / "season_replay.mp4", out / "baselines.png"
    assert gif.read_bytes()[:4] == b"GIF8"
    assert gif.stat().st_size < 8_000_000
    assert 8 <= _gif_seconds(gif) <= 20
    probe = _ffprobe(mp4)
    [stream] = probe["streams"]
    assert (stream["codec_name"], stream["width"]) == ("h264", 1080)
    assert 8 <= float(probe["format"]["duration"]) <= 20
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.stat().st_size < 500_000
    tradeoff = out / "tradeoff.png"
    with Image.open(tradeoff) as im:
        assert im.size == (1200, 627)
    assert tradeoff.stat().st_size < 1_000_000

    assert "rb_default" in footers["replay"] and f"commit {sha7}" in footers["replay"]
    assert all(c in footers["card"] for c in CONTROLLERS) and f"commit {sha7}" in footers["card"]

    title = replays["kwargs"]["title"]
    assert "simulated" in title.lower(), title
    assert _whole(title, 60), title  # the 60-day season, 5,760 steps / 96 a day
    assert PLACE in " ".join(title.split()), title
    subtitles = [
        " ".join(t.get_text().split())
        for t in _texts(replays["fig"])
        if all(part in " ".join(t.get_text().split()) for part in RB_SUBTITLE)
    ]
    assert subtitles, "the final replay figure has no subtitle naming the controller"

    entries = json.loads((out / "media.json").read_text())
    assert isinstance(entries, list)
    by_file = {e["file"]: e for e in entries}
    assert sorted(by_file) == [name for name in MEDIA_FILES if name != "media.json"]
    gl_gym = manifest._versions()["gl-gym"]
    for name, e in by_file.items():
        assert ENTRY_KEYS <= set(e), name
        assert e["scenario"] == (VALIDATION if name == "tradeoff.png" else PAPER), name
        assert len(e["commit"]) >= 7 and stamp["commit"].startswith(e["commit"]), name
        assert e["dirty"] is stamp["dirty"], name
        assert len(e["gl_gym"]) >= 7 and gl_gym.startswith(e["gl_gym"]), name
        datetime.fromisoformat(e["made_at"])
    for name in ("season_replay.gif", "season_replay.mp4"):
        assert "rb_default" in by_file[name]["source"], name
        _assert_golden(by_file[name]["metrics"], GOLDEN["rb_default"], name)
    card_entry = by_file["baselines.png"]
    assert all(c in card_entry["source"] for c in CONTROLLERS)
    for c in CONTROLLERS:
        _assert_golden(card_entry["metrics"][c], GOLDEN[c], c)
    assert by_file["tradeoff.png"]["split"] == "validation"

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
- footer(source, stamp) -> str, and main(argv=None) -> int, the CLI. main refuses an invalid
  season (a solver failure, or the wrong length) before writing anything, and titles the replay
  as simulated, with the season's day count from the run and PAPER's location and year; its
  subtitle names the controller and the simulator.

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
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image

import greenleaf_control
from greenleaf_control import envs, evaluate, manifest, metrics, rule_based, scenarios

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
MEDIA_FILES = ["baselines.png", "media.json", "season_replay.gif", "season_replay.mp4"]
ENTRY_KEYS = {"file", "source", "scenario", "commit", "dirty", "gl_gym", "made_at", "metrics"}


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


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory) -> Path:
    """A tmp dir the cached figures below are written into (makes no media call, so never ERRORs)."""
    return tmp_path_factory.mktemp("media")


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


# --- the CLI -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["no_out", "no_ffmpeg"])
def test_main_refuses_before_any_season(case, tmp_path, monkeypatch, capsys):
    """--out is required; and without ffmpeg (FFMpegWriter.isAvailable() false) main refuses with a
    nonzero SystemExit naming ffmpeg. Both come before any season is played, and nothing is
    written: --out isn't created."""
    import greenleaf_control.media as media

    _no_season_played(monkeypatch, media)
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
    """season_replay and baselines_card replaced by recorders of their arguments that draw and
    write nothing, so main's own choices (its title, subtitle and footers) are read fast."""
    calls = {}

    def record(name: str):
        def drawn(*args, **kwargs):
            calls[name] = kwargs
            return Figure()

        return drawn

    monkeypatch.setattr(media, "season_replay", record("replay"))
    monkeypatch.setattr(media, "baselines_card", record("card"))
    return calls


def test_main_titles_the_replay_as_simulated_with_the_runs_days_place_and_controller(
    tmp_path, monkeypatch
):
    """The replay's title says it is simulated, with the season's day count taken from the run and
    PAPER's location and year (Amsterdam 2010, from scenarios.PAPER); its subtitle names
    "upstream's default rule controller" and "GreenLight-Gym". The 8 pt footer alone is unreadable
    on a phone, so a real greenhouse must not be what the title reads as.

    Known answer: with play shortened to real 2-day seasons (192 steps), the title says 2 (days),
    never 60, so the count can't be typed in. Drawing is recorded, not done (the real
    season_replay is tested above), and ffmpeg is reported available, so this runs anywhere.
    """
    import greenleaf_control.media as media

    real_play = media.play
    monkeypatch.setattr(
        media, "play", lambda controller, season_length=60: real_play(controller, 2)
    )
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
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
def test_main_refuses_an_invalid_season_before_writing(invalid, tmp_path, monkeypatch, capsys):
    """A season whose result isn't valid (metrics: a solver failure, or steps != season_steps) is
    never drawn: main exits nonzero naming that controller, and only that one, before anything is
    drawn or written (--out stays absent or empty). play returns the real 1-day seasons, the one
    under test marked invalid with one solver failure; ffmpeg is reported available."""
    import greenleaf_control.media as media

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
    """play(controller, season_length=60), running(states, controls, params, ends),
    season_replay(frame, counters, out_gif, out_mp4=None, *, title, subtitle=None, footer, fps=15),
    baselines_card(seasons, out, *, footer), footer(source, stamp) and main(argv=None)."""
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
    assert shape(media.footer) == [("source", pos, empty), ("stamp", pos, empty)]
    assert shape(media.main) == [("argv", pos, None)]


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
def test_one_command_writes_every_file_labelled_with_its_source_and_commit(tmp_path, monkeypatch):
    """Done when: one command writes the GIF, the MP4, the baselines PNG and media.json into --out.

    main(["--out", out]) at its defaults (60-day PAPER seasons) returns 0 and writes exactly
    season_replay.gif, season_replay.mp4, baselines.png and media.json:
    - the GIF under 8 MB and 8-20 s long (the README's ~10-15 s);
    - the MP4 H.264, 1080 px wide and 8-20 s long by ffprobe (LinkedIn);
    - the PNG under 500 KB;
    - media.json a list of one entry per picture, each with file, source, scenario (PAPER), commit
      and dirty (this checkout's code_stamp, taken before writing), gl_gym (the installed gl-gym's
      commit, 7 characters or more), made_at (ISO 8601) and metrics. The replay's entries: source
      rb_default, metrics its season's; the card's: source naming all three baselines, metrics
      keyed by baseline. Every number equals tests/golden/values.json "ours" (hours exactly, the
      rest at rel 1e-6).
    - The footers drawn (season_replay's and baselines_card's footer argument, spied on; the real
      functions run) name their sources and "commit <sha7>".
    - The replay's title says "simulated", 60 (days) and Amsterdam 2010 (from scenarios.PAPER);
      its subtitle, drawn on the final figure, names "upstream's default rule controller" and
      "GreenLight-Gym".
    """
    import greenleaf_control.media as media

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
    assert sorted(by_file) == ["baselines.png", "season_replay.gif", "season_replay.mp4"]
    gl_gym = manifest._versions()["gl-gym"]
    for name, e in by_file.items():
        assert ENTRY_KEYS <= set(e), name
        assert e["scenario"] == PAPER, name
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

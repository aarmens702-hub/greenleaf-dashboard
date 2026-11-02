"""Tests for greenleaf_control.viz (issue #18, S1-13): the standard plots of a run.

Each function draws on a matplotlib Figure(facecolor="white") (no pyplot, so headless), saves `out`
as a PNG and returns the Figure. The tests read what each plot draws from the returned Figure (its
axes, lines, patches and errorbar containers), never its pixels:
- season_strip(frame, out): 9 axes sharing x. co2, temp and rh against frame["t"], each over its
  metrics.BOUNDS band (axhspan), then the 6 actuators u_boil, u_co2, u_thscr, u_vent, u_lamp,
  u_blscr in frame order, each labelled [0–1] on a ylim padded past 0 and 1, so an actuator held
  at 0 or 1 stays visible. A residual frame's u_rb_* columns aren't actuators.
- learning_curves(run_dirs, out, reference=None): one line of rollout/ep_rew_mean per run dir, read
  from its one tb/**/events.out.tfevents.* and labelled by the dir's name; with a reference, one
  horizontal line at it, labelled "reference".
- lambda_traces(lambdas_csv, out): lambda_<c> against timesteps, one line per metrics.CONSTRAINTS
  entry, labelled c.
- tradeoff(points, out, *, xlabel="violation hours [h]", ylabel="profit [€/m²]", footer=None):
  one errorbar per point at (viol_hours IQM, profit IQM), its bars reaching each CI's lo and hi,
  each errorbar labelled p["label"].

Issue #47 (A1b) adds the report's two other figures and the tradeoff keywords:
- per_scenario(panels, scenario_labels, out, *, footer=None): one axes per panel (x label its key),
  one row per label (a y tick, dict order, the first on top), every value of the label's runs x
  scenarios array a scatter dot in its row, the scenarios told apart and named.
- profiles(panels, out, *, highlight=(), footer=None): one axes per panel (x label its key), one
  ax.ecdf(values.ravel(), complementary=True, label=f"{label} (n={values.size})") per label, a
  dashed vertical line at 0, y label "fraction of seeds × seasons above τ".

Aarmen's "Instrument" style for public figures (the orchestrator's defaults on #47) changes the
issue's "white facecolor" and "colours from tab20" for these three report figures. These are
requirement changes, not loosened checks:
- facecolor GROUND #F2F3F0 (season_strip, learning_curves and lambda_traces stay white);
- monochrome with direct labels: every point or highlighted line carries its own text in ax.texts,
  in MONO; a legend is never the only way to identify one. profiles' highlighted labels are INK
  #111311 and labelled, the others HAIRLINE #D5D8D2 and unlabelled;
- the tradeoff's limits fit the learning arms (labels not in envs.BASELINES, which viz tells by
  name) with their CIs; a point outside them (on VALIDATION zero_controls, random, rb_default) is
  a triangle on the axes edge pointing outward, plus a text with its label and true value(s);
- footer=None is a keyword: the report passes its source line.

The fix round on #47 (the orchestrator's rules) adds:
- a point's optional "status" ("adopted", "dropped" or None): only an adopted learning arm gets
  ink (a filled INK dot, an INK label); a dropped or undecided one is hollow and MUTED #6A6F68;
- a one-line MONO key on the tradeoff and on profiles naming the encoding (it says "adopted",
  "dropped" and "rb_tuned");
- per_scenario's far baseline row: its values outside the learning arms' x range are a triangle
  on that row's edge plus a text giving their true lo and hi, never dots;
- viz.BASELINES, its own copy of envs.BASELINES (viz can't import envs), kept equal to it.

#53 (A1c, the orchestrator's scope note) adds one keyword to the three report figures: title,
keyword-only, defaulting to today's exact title (per_scenario "Every run on every season",
profiles "Performance profiles", tradeoff "Trade-off: IQM and 95% CI per label"), so a caller
that passes none (the validation report) draws what it drew before; given, it is the figure's
suptitle in its place (the report names a test split in it).

A refused input (a frame without 6 actuators, a run dir without exactly one event file, a
per_scenario array whose columns aren't the scenarios, or nothing to draw: no run dirs, no points,
no panels, a header-only lambdas.csv) is a ValueError, and no PNG is written.

viz reads only metrics' constants: it never imports rewards, envs, lagrangian or train.

The fixtures are hand-built (a synthetic per-step frame, TensorBoard logs written by SB3's own
logger, a lambdas.csv under LagrangianCallback's header), plus one smoke run of train as a drift
guard. Nothing here touches control/runs/ or the real ledger.
"""

import ast
import csv
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from stable_baselines3.common.logger import configure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import greenleaf_control
from greenleaf_control import envs, metrics

# greenleaf_control.viz is imported inside each test body, so while it is missing each test FAILS
# rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"

PNG = b"\x89PNG\r\n\x1a\n"  # the 8-byte PNG signature
CLIMATE = ("co2", "temp", "rh")  # the strip's first three axes, in metrics.CONSTRAINTS order
# The per-step frame's actuators (docs/API-CONTRACT.md), in upstream's control order
U_COLS = ["u_boil", "u_co2", "u_thscr", "u_vent", "u_lamp", "u_blscr"]
U_RB_COLS = [f"u_rb_{c.removeprefix('u_')}" for c in U_COLS]  # the residual frame's RB actions
COST_COLS = [f"cost_{c}" for c in metrics.CONSTRAINTS]
N = 8  # steps in the synthetic frame

# LagrangianCallback's lambdas.csv header, as in test_lagrangian.py
HEADER = ["timesteps", "lambda_co2", "lambda_temp", "lambda_rh", "cost_co2", "cost_temp", "cost_rh"]
# Three rollout ends; no lambda column equals a cost column, so a misread column shows
LAMBDA_ROWS = [
    [2048, 1.95, 0.0, 0.65, 0.5, 0.125, 0.25],
    [4096, 1.8, 0.25, 0.6, 0.375, 0.5, 0.0],
    [6144, 1.5, 0.5, 0.0, 0.25, 0.75, 0.125],
]

# Each seed's (step, rollout/ep_rew_mean) dumps; float32-exact, as TensorBoard stores float32.
# seed2 logs fewer points at other steps, so a shared x axis across seeds would show.
CURVES = {
    "seed0": [(2048, 0.5), (4096, -1.25), (6144, 3.0)],
    "seed1": [(2048, -2.5), (4096, 0.75), (6144, 1.0)],
    "seed2": [(4096, 4.0), (8192, -0.25)],
}
DECOY = 99.0  # train/approx_kl, logged at every dump; it must never be drawn

# (iqm, lo, hi) as stats.iqm_ci returns them; every bar is asymmetric (iqm - lo != hi - iqm). All
# three are learning arms (no envs.BASELINES label), so the axes fit them all (#47: a baseline
# outside the learning arms' limits is an edge marker, not an errorbar).
POINTS = [
    {"label": "ppo", "profit": (12.5, 10.0, 14.0), "viol_hours": (40.0, 30.0, 52.0)},
    {"label": "residual", "profit": (15.25, 14.5, 16.0), "viol_hours": (60.5, 48.0, 64.0)},
    {"label": "residual_lag", "profit": (13.0, 11.75, 13.5), "viol_hours": (8.0, 6.5, 12.25)},
]

# The Instrument style (#47, the orchestrator's defaults), as matplotlib's to_hex gives them
GROUND = "#f2f3f0"  # the report figures' facecolor
INK = "#111311"  # profiles' highlighted (adopted) lines
HAIRLINE = "#d5d8d2"  # profiles' other lines
MUTED = "#6a6f68"  # the tradeoff's dropped and undecided arms
MONO = ["Menlo", "DejaVu Sans Mono"]  # the requested family of direct labels, installed ones only
# The VALIDATION scenarios as report labels them, `<location> day <start_day>`
SCENARIO_LABELS = ["Amsterdam day 45", "Amsterdam day 59", "Amsterdam day 75"]
FOOTER = (
    "source: python -m greenleaf_control.report --split validation; derived from"
    " summary.csv / scores.parquet"
)
# The key's words: the tradeoff's and profiles' encoding names these three
KEY_WORDS = ("adopted", "dropped", "rb_tuned")

# per_scenario's panels: label -> runs x scenarios. Every value is distinct within its row, so a
# dot's x names its scenario column; rb_tuned's values sit inside the learning arms' range
# (profit -2.5..4.0, hours 0..30), so no dot is an edge marker.
PROFIT_KEY, HOURS_KEY = "profit [€/m²]", "total violation hours [h]"
PANELS = {
    PROFIT_KEY: {
        "rb_tuned": np.array([[-2.0, 1.0, 2.0]]),
        "ppo #33/0221": np.array([[1.5, 2.5, 3.5], [0.25, 4.0, 2.75]]),
        "lag_b100 #39/6d92": np.array([[-1.0, 0.5, 1.25], [3.0, -2.5, 0.75], [2.0, -0.5, 1.75]]),
    },
    HOURS_KEY: {
        "rb_tuned": np.array([[3.0, 9.0, 21.0]]),
        "ppo #33/0221": np.array([[12.0, 4.0, 30.0], [7.0, 18.0, 25.0]]),
        "lag_b100 #39/6d92": np.array([[0.0, 6.0, 15.0], [2.0, 11.0, 27.0], [5.0, 8.0, 19.0]]),
    },
}
# random, a far baseline row for per_scenario (#46's VALIDATION random sits near -30 €/m²
# and 1,300 h): in each panel two values far outside the learning arms' range and one inside it.
FAR_ROW = {
    PROFIT_KEY: np.array([[-30.0, -28.0, 1.5]]),
    HOURS_KEY: np.array([[2600.0, 2400.0, 10.0]]),
}
# panel: (the side its two far values are off, their true lo and hi, the one value inside)
FAR_SIDES = {PROFIT_KEY: ("left", -30.0, -28.0, 1.5), HOURS_KEY: ("right", 2400.0, 2600.0, 10.0)}

# profiles' panels: label -> runs x scenarios of paired changes against rb_tuned. ppo's profit
# gains are the issue's (-1, 0.5, 2, 3): 0.75 of them above tau just above -1, 0.5 at tau = 1.
GAIN_KEY = "profit gain over rb_tuned's same season [€/m²]"
SAVED_KEY = "violation hours saved vs rb_tuned's same season [h]"
PROFILES = {
    GAIN_KEY: {
        "zero_controls": np.array([[4.0, -2.0]]),
        "ppo #33/0221": np.array([[-1.0, 2.0], [0.5, 3.0]]),
        "lag_b50 #35/976c": np.array([[0.25, -0.75], [1.5, -1.25], [2.5, 0.75]]),
    },
    SAVED_KEY: {
        "zero_controls": np.array([[-2500.0, -2400.0]]),
        "ppo #33/0221": np.array([[60.0, -10.0], [35.0, 90.0]]),
        "lag_b50 #35/976c": np.array([[180.0, 150.0], [210.0, -5.0], [120.0, 195.0]]),
    },
}
PROFILES_YLABEL = "fraction of seeds × seasons above τ"

# The report's trade-off: x violation hours saved vs rb_tuned [h], y profit gain over rb_tuned
# [€/m²], (iqm, lo, hi). The three learning arms span x -18..310 and y -0.7..2.5, CIs included;
# rb_tuned's origin is inside. #46's VALIDATION table puts the other baselines far outside
# (zero_controls -2587 h, random -29.8 €/m², rb_default -388 h and -6.9 €/m²); here zero_controls
# is off on the left only, random below only, rb_default both.
SPREAD = [
    {"label": "ppo #33/0221", "profit": (2.4, 2.3, 2.5), "viol_hours": (60.0, 40.0, 80.0)},
    {
        "label": "residual_w4 #34/4d20",
        "profit": (-0.6, -0.7, -0.5),
        "viol_hours": (300.0, 295.0, 310.0),
    },
    {"label": "lag_b100 #39/6d92", "profit": (0.6, 0.5, 0.7), "viol_hours": (-10.0, -18.0, -2.0)},
    {"label": "rb_tuned", "profit": (0.0, 0.0, 0.0), "viol_hours": (0.0, 0.0, 0.0)},
    {"label": "zero_controls", "profit": (1.0, 1.0, 1.0), "viol_hours": (-2500.0,) * 3},
    {"label": "random", "profit": (-30.0,) * 3, "viol_hours": (100.0,) * 3},
    {"label": "rb_default", "profit": (-7.0,) * 3, "viol_hours": (-400.0,) * 3},
]
# label: (the sides of the axes it's off, the true values its text must give)
OFF_LIMITS = {
    "zero_controls": ({"left"}, [-2500.0]),
    "random": ({"bottom"}, [-30.0]),
    "rb_default": ({"left", "bottom"}, [-400.0, -7.0]),
}
# An edge marker's triangle and the side it points to: a str marker, or matplotlib's CARET and
# CARETBASE ints
TRIANGLES = {
    "<": "left",
    ">": "right",
    "^": "top",
    "v": "bottom",
    **{k: side for k, side in zip((4, 5, 6, 7), ("left", "right", "top", "bottom"), strict=True)},
    **{k: side for k, side in zip((8, 9, 10, 11), ("left", "right", "top", "bottom"), strict=True)},
}


def _frame(residual: bool = False) -> pd.DataFrame:
    """The contract's per-step frame for 8 steps (t = 0..7), in BOUNDS' units.

    Every climate series leaves its band on both sides. Actuator k is (k + 1) / 10 + t / 100, so
    u_boil runs 0.10..0.17 and u_blscr 0.60..0.67: all in [0, 1] and all distinct. A residual
    frame adds u_rb_<a> = 1 - u_<a>, falling where every u_ rises, so no u_rb_ equals a u_.
    """
    t = np.arange(N)
    cols = {
        "t": t,
        "co2": [250.0, 300.0, 800.0, 1600.0, 1700.0, 900.0, 400.0, 299.0],  # ppm; band 300-1600
        "temp": [14.0, 15.0, 20.0, 34.0, 35.5, 25.0, 18.0, 16.0],  # °C; band 15-34
        "rh": [45.0, 50.0, 70.0, 85.0, 90.0, 80.0, 60.0, 55.0],  # %; band 50-85
    }
    for k, c in enumerate(U_COLS):
        cols[c] = (k + 1) / 10 + t / 100
    if residual:
        for c, rb in zip(U_COLS, U_RB_COLS, strict=True):
            cols[rb] = 1 - cols[c]
    cols["profit"] = 0.01 * t - 0.02
    for k, c in enumerate(COST_COLS):
        cols[c] = (k + 1) * 0.1 * (t % 2)
    return pd.DataFrame(cols)


def _tb_run(run_dir: Path, curve: list[tuple[int, float]]) -> Path:
    """run_dir/tb/PPO_1/events.out.tfevents.*, written by SB3's logger (train's own writer).

    A dump of the decoy alone at step 1024 (no episode has ended yet, so no ep_rew_mean), then at
    each (step, value) of curve both rollout/ep_rew_mean and the decoy.
    """
    logger = configure(str(run_dir / "tb" / "PPO_1"), ["tensorboard"])
    logger.record("train/approx_kl", DECOY)
    logger.dump(1024)
    for step, value in curve:
        logger.record("rollout/ep_rew_mean", value)
        logger.record("train/approx_kl", DECOY)
        logger.dump(step)
    logger.close()
    return run_dir


def _seed_dirs(tmp_path: Path) -> list[Path]:
    return [_tb_run(tmp_path / name, curve) for name, curve in CURVES.items()]


def _lambdas_csv(path: Path, rows: list[list] = LAMBDA_ROWS) -> Path:
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([HEADER, *rows])
    return path


def _assert_png(out: Path) -> None:
    assert out.is_file(), f"{out} wasn't written"
    assert out.read_bytes()[:8] == PNG


def _lines(fig: Figure) -> list:
    """Every line drawn on every axes of fig."""
    return [line for ax in fig.axes for line in ax.get_lines()]


def _xy(line) -> tuple[list[float], list[float]]:
    return (
        np.asarray(line.get_xdata(), dtype=float).tolist(),
        np.asarray(line.get_ydata(), dtype=float).tolist(),
    )


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


def _viz(name: str):
    """viz.<name>; an assertion, not an AttributeError, while it is missing (#47 adds
    per_scenario and profiles)."""
    import greenleaf_control.viz as viz

    assert hasattr(viz, name), f"greenleaf_control.viz.{name} doesn't exist yet (#47)"
    return getattr(viz, name)


def _args(plot: str, tmp_path: Path) -> list:
    """The arguments before `out` of a well-formed call of viz.<plot>."""
    return {
        "season_strip": lambda: [_frame()],
        "learning_curves": lambda: [_seed_dirs(tmp_path)],
        "lambda_traces": lambda: [_lambdas_csv(tmp_path / "lambdas.csv")],
        "tradeoff": lambda: [POINTS],
        "per_scenario": lambda: [PANELS, SCENARIO_LABELS],
        "profiles": lambda: [PROFILES],
    }[plot]()


def _texts(fig: Figure) -> list[str]:
    """Every text drawn anywhere on fig (ax.texts, legends, fig.texts, tick labels), its
    whitespace collapsed."""
    return [" ".join(t.get_text().split()) for t in fig.findobj(Text)]


def _rows(fig: Figure) -> dict[str, float]:
    """{label: its row's y}, from the first panel's y tick labels (a shared y hides the others')."""
    return {
        t.get_text(): t.get_position()[1] for t in fig.axes[0].get_yticklabels() if t.get_text()
    }


def _top_down(ax, rows: dict[str, float]) -> list[str]:
    """rows' labels from the top of ax to the bottom, as drawn (an inverted y axis included)."""
    return sorted(rows, key=lambda label: -ax.transData.transform([(0.0, rows[label])])[0][1])


def _rgba(colors, i: int) -> tuple:
    return tuple(np.round(colors[i % len(colors)], 3)) if len(colors) else ("none",)


def _dots(ax) -> list[tuple[float, float, tuple]]:
    """(x, y, look) of every dot scattered on ax (each offset of each PathCollection); look is its
    marker path, face and edge colours and size, so dots that look alike share it."""
    out = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        paths, sizes = coll.get_paths(), coll.get_sizes()
        faces, edges = coll.get_facecolors(), coll.get_edgecolors()
        for i, (x, y) in enumerate(np.asarray(coll.get_offsets(), dtype=float)):
            path = paths[i % len(paths)] if paths else None
            look = (
                None if path is None else tuple(np.round(path.vertices, 3).ravel()),
                _rgba(faces, i),
                _rgba(edges, i),
                float(sizes[i % len(sizes)]) if len(sizes) else None,
            )
            out.append((float(x), float(y), look))
    return out


def _by_row(ax, rows: dict[str, float]) -> dict[str, list[tuple[float, float, tuple]]]:
    """ax's dots grouped by the row whose tick is nearest, as (x, y offset from the row, look);
    every dot must lie within half a row spacing of its row."""
    ys = sorted(rows.values())
    half = min(np.diff(ys)) / 2 if len(ys) > 1 else 0.5
    out = {label: [] for label in rows}
    for x, y, look in _dots(ax):
        label = min(rows, key=lambda r: abs(rows[r] - y))
        assert abs(rows[label] - y) < half, f"a dot at y {y} is in no row of {rows}"
        out[label].append((x, round(y - rows[label], 3), look))
    return out


def _row_at(ax, rows: dict[str, float], xy) -> str:
    """The label of the row nearest a position given in ax's axes fractions."""
    y = ax.transData.inverted().transform(ax.transAxes.transform(xy))[1]
    return min(rows, key=lambda label: abs(rows[label] - y))


def _triangle_side(path) -> str | None:
    """The side a scatter's marker path points to, if it is one of the four triangles."""
    for marker, side in (("<", "left"), (">", "right"), ("^", "top"), ("v", "bottom")):
        style = MarkerStyle(marker)
        ref = style.get_path().transformed(style.get_transform())
        if ref.vertices.shape == path.vertices.shape and np.allclose(ref.vertices, path.vertices):
            return side
    return None


def _triangles(ax) -> list[tuple[str, np.ndarray]]:
    """(side it points to, position in axes fractions) of every triangle marker on ax, drawn by
    plot (a Line2D marker) or scatter (a PathCollection), in data or axes coordinates."""
    to_axes = ax.transAxes.inverted()
    out = []
    for line in ax.get_lines():
        marker = line.get_marker()
        side = TRIANGLES.get(marker) if isinstance(marker, str | int) else None
        if side is not None:
            at = to_axes.transform(line.get_transform().transform(line.get_xydata()))
            out += [(side, xy) for xy in at]
    for coll in ax.collections:
        paths = coll.get_paths() if isinstance(coll, PathCollection) else []
        side = _triangle_side(paths[0]) if len(paths) == 1 else None
        if side is not None:
            at = coll.get_offset_transform().transform(np.asarray(coll.get_offsets(), dtype=float))
            out += [(side, xy) for xy in to_axes.transform(at)]
    return out


def _numbers(text: str) -> list[float]:
    """Every number written in text: a Unicode minus read as -, a thousands comma dropped."""
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text.replace("−", "-"))
    return [float(n) for n in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def _names(text: str, label: str) -> bool:
    """text names label as a whole: "residual" isn't named by "residual_lag"."""
    return re.search(rf"(?<!\w){re.escape(label)}(?!\w)", text) is not None


def _gives(text: str, value: float) -> bool:
    """text writes value, to 1% (or 0.05 near zero)."""
    return any(abs(n - value) <= max(0.01 * abs(value), 0.05) for n in _numbers(text))


def _above(line, tau: float) -> float:
    """A complementary-ECDF line's y at tau. ax.ecdf(complementary=True) draws steps-pre: on
    x[i - 1] < tau <= x[i] the line reads y[i]."""
    x, y = (np.asarray(d, dtype=float) for d in (line.get_xdata(), line.get_ydata()))
    assert x[0] < tau <= x[-1], (tau, x)
    return float(y[np.searchsorted(x, tau, side="left")])


# Every function


FACECOLOR = {
    "season_strip": "#ffffff",
    "learning_curves": "#ffffff",
    "lambda_traces": "#ffffff",
    # The report's figures (#47): GROUND, Aarmen's Instrument style for public visuals
    "tradeoff": GROUND,
    "per_scenario": GROUND,
    "profiles": GROUND,
}


@pytest.mark.parametrize("plot", list(FACECOLOR))
def test_every_plot_returns_its_figure_on_an_opaque_facecolor_and_saves_it_as_png(plot, tmp_path):
    """An opaque facecolor (not transparent) keeps each PNG readable in both light and dark README
    themes; the function returns the Figure it saved.

    A run's plots stay white. The report's three figures (tradeoff, per_scenario, profiles) are on
    GROUND #F2F3F0: a requirement change (the Instrument style Aarmen chose for every public
    visual, the orchestrator's default on #47), not a loosened check."""
    plot_fn = _viz(plot)
    args = _args(plot, tmp_path)
    out = tmp_path / f"{plot}.png"
    fig = plot_fn(*args, out)
    assert isinstance(fig, Figure)
    assert to_hex(fig.get_facecolor()) == FACECOLOR[plot]
    _assert_png(out)


@pytest.mark.parametrize(
    "plot",
    ["learning_curves", "lambda_traces", "tradeoff", "per_scenario", "profiles"],
    ids=["no_run_dirs", "header_only_csv", "no_points", "no_scenario_panels", "no_profile_panels"],
)
def test_an_empty_input_raises_and_writes_no_png(plot, tmp_path):
    """Nothing to draw is a ValueError, never a blank PNG that reads as a run with no data."""
    plot_fn = _viz(plot)
    empty = {
        "learning_curves": lambda: [[]],
        "lambda_traces": lambda: [_lambdas_csv(tmp_path / "lambdas.csv", rows=[])],
        "tradeoff": lambda: [[]],
        "per_scenario": lambda: [{}, SCENARIO_LABELS],
        "profiles": lambda: [{}],
    }[plot]()
    out = tmp_path / f"{plot}.png"
    with pytest.raises(ValueError):
        plot_fn(*empty, out)
    assert not out.exists()


@pytest.mark.parametrize("plot", ["per_scenario", "profiles", "tradeoff"])
def test_the_report_figures_print_the_footer_they_are_given(plot, tmp_path):
    """footer is a keyword-only argument, default None; given, it is a text of the figure (here
    the report's source line), so a PNG names the command that made it."""
    plot_fn = _viz(plot)
    footer = inspect.signature(plot_fn).parameters.get("footer")
    assert footer is not None, f"viz.{plot} takes no footer yet (#47)"
    assert (footer.kind, footer.default) == (inspect.Parameter.KEYWORD_ONLY, None)
    out = tmp_path / f"{plot}.png"
    fig = plot_fn(*_args(plot, tmp_path), out, footer=FOOTER)
    _assert_png(out)
    assert any(FOOTER in text for text in _texts(fig)), _texts(fig)


# The report figures' titles before #53, their title= defaults
TITLES = {
    "per_scenario": "Every run on every season",
    "profiles": "Performance profiles",
    "tradeoff": "Trade-off: IQM and 95% CI per label",
}


@pytest.mark.parametrize("plot", list(TITLES))
def test_the_report_figures_take_a_title_keyword_defaulting_to_todays(plot, tmp_path):
    """title is keyword-only and defaults to today's exact title: with none given, the figure's
    suptitle is today's (validation's figures don't change). Given, it is the suptitle instead,
    drawn once, and today's isn't drawn beside it (the report's test-split title starts with
    today's, so "instead" is checked on the whole text)."""
    plot_fn = _viz(plot)
    title = inspect.signature(plot_fn).parameters.get("title")
    assert title is not None, f"viz.{plot} takes no title keyword yet (#53)"
    assert (title.kind, title.default) == (inspect.Parameter.KEYWORD_ONLY, TITLES[plot])
    plain = plot_fn(*_args(plot, tmp_path), tmp_path / "plain.png")
    assert plain.get_suptitle() == TITLES[plot]

    named = f"{TITLES[plot]} · london_2012, 2012 test weather, scored once"
    out = tmp_path / f"{plot}.png"
    fig = plot_fn(*_args(plot, tmp_path), out, title=named)
    _assert_png(out)
    assert fig.get_suptitle() == named
    texts = [t.get_text() for t in fig.findobj(Text)]
    assert texts.count(named) == 1, texts
    assert TITLES[plot] not in texts, texts


def _imported_modules(path: Path) -> set[str]:
    """Every module path imported anywhere in path's source. `import a.b` gives a.b; `from a import
    b` gives a and a.b (b may be a module); a relative import resolves inside greenleaf_control."""
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


@pytest.mark.parametrize(
    "module",
    [
        "greenleaf_control.rewards",
        "greenleaf_control.envs",
        "greenleaf_control.lagrangian",
        "greenleaf_control.train",
        "matplotlib.pyplot",
    ],
)
def test_viz_never_imports_the_graded_code_or_pyplot(module):
    """viz draws from raw data and metrics' constants only, so a plot can't be written from the
    graded code; and it builds Figures directly, never through pyplot (headless, no global state)."""
    path = Path(greenleaf_control.__file__).parent / "viz.py"
    assert path.is_file(), f"{path} doesn't exist"
    imported = _imported_modules(path)
    assert "greenleaf_control.metrics" in imported  # where BOUNDS and CONSTRAINTS come from
    assert not [name for name in imported if name == module or name.startswith(f"{module}.")]


def test_viz_holds_envs_baselines():
    """A drift guard: viz can't import envs (above), so it keeps its own copy of the baseline
    names, which tell a baseline from a learning arm in every report figure. It must stay
    envs.BASELINES: a baseline renamed or added there and not here would be fitted and inked as a
    learning arm."""
    import greenleaf_control.viz as viz

    assert hasattr(viz, "BASELINES"), "viz.BASELINES doesn't exist"
    assert tuple(viz.BASELINES) == envs.BASELINES


# season_strip


@pytest.mark.parametrize("residual", [False, True], ids=["rb_frame", "residual_frame"])
def test_season_strip_has_nine_axes_sharing_x(residual, tmp_path):
    """co2, temp, rh, then the 6 actuators: 9 axes either way (u_rb_* columns get none)."""
    import greenleaf_control.viz as viz

    out = tmp_path / "strip.png"
    fig = viz.season_strip(_frame(residual), out)
    _assert_png(out)
    assert len(fig.axes) == 9
    first = fig.axes[0]
    assert all(first.get_shared_x_axes().joined(first, ax) for ax in fig.axes)


@pytest.mark.parametrize("residual", [False, True], ids=["rb_frame", "residual_frame"])
def test_season_strip_draws_each_climate_series_over_its_bounds_band(residual, tmp_path):
    """Axes 0, 1, 2 are co2, temp, rh: one line of frame[c] against frame["t"], over one band from
    metrics.BOUNDS[c] (co2 300-1600 ppm, temp 15-34 °C, rh 50-85 %)."""
    import greenleaf_control.viz as viz

    frame = _frame(residual)
    out = tmp_path / "strip.png"
    fig = viz.season_strip(frame, out)
    _assert_png(out)
    for ax, c in zip(fig.axes[:3], CLIMATE, strict=True):
        [line] = ax.get_lines()
        assert _xy(line) == (frame["t"].tolist(), frame[c].tolist()), c
        assert all(isinstance(p, Rectangle) for p in ax.patches), c  # axhspan's patch
        bands = [(p.get_y(), p.get_y() + p.get_height()) for p in ax.patches]
        assert bands == [metrics.BOUNDS[c]], c


def _saturated_frame() -> pd.DataFrame:
    """_frame() with u_boil held at 0 and u_vent at 1 all season: the pattern a reward-hacking
    policy leaves, which the strip must show."""
    return _frame().assign(u_boil=0.0, u_vent=1.0)


@pytest.mark.parametrize(
    "make_frame",
    [_frame, lambda: _frame(residual=True), _saturated_frame],
    ids=["rb_frame", "residual_frame", "u_boil_0_u_vent_1"],
)
def test_season_strip_draws_each_actuator_inside_a_padded_0_to_1_axes(make_frame, tmp_path):
    """Axes 3..8 are the 6 actuators in frame order, one line each, equal to its own column (in a
    residual frame, the u_ column, never its u_rb_ twin), labelled [0–1].

    Each ylim is padded past 0 and 1 (lo < 0, hi > 1), so the line lies strictly inside it. On
    ylim (0, 1) exactly, an actuator held at 0 or 1 sits on the axes frame: matplotlib clips the
    line in half and draws the spine over it, and the box looks empty (the reviewer's render of
    u_boil = 0 and u_vent = 1).
    """
    import greenleaf_control.viz as viz

    frame = make_frame()
    out = tmp_path / "strip.png"
    fig = viz.season_strip(frame, out)
    _assert_png(out)
    assert sum(len(ax.get_lines()) for ax in fig.axes[3:]) == 6
    for ax, c in zip(fig.axes[3:], U_COLS, strict=True):
        [line] = ax.get_lines()
        x, y = _xy(line)
        assert (x, y) == (frame["t"].tolist(), frame[c].tolist()), c
        lo, hi = ax.get_ylim()
        assert lo < 0.0 and hi > 1.0, (c, lo, hi)
        assert lo < min(y) and max(y) < hi, (c, lo, hi)
        assert "[0–1]" in ax.get_ylabel(), c


@pytest.mark.parametrize(
    ("residual", "dropped"),
    [(False, "u_blscr"), (True, "u_boil")],
    # the residual frame still has 11 u_ columns (5 actuators + 6 u_rb_), but only 5 actuators
    ids=["rb_frame_without_u_blscr", "residual_frame_without_u_boil"],
)
def test_season_strip_refuses_a_frame_with_five_actuators(residual, dropped, tmp_path):
    import greenleaf_control.viz as viz

    out = tmp_path / "strip.png"
    with pytest.raises(ValueError):
        viz.season_strip(_frame(residual).drop(columns=dropped), out)
    assert not out.exists()


# learning_curves


@pytest.mark.parametrize("kwargs", [{}, {"reference": -1.5}], ids=["no_reference", "reference"])
def test_learning_curves_draws_every_seed_by_name(kwargs, tmp_path):
    """One line per run dir, labelled by its name: x the steps rollout/ep_rew_mean was logged at,
    y its values (never the decoy's, and not the decoy-only dump at step 1024). With a reference,
    one more line, horizontal at it and labelled "reference"; without one, no other line."""
    import greenleaf_control.viz as viz

    out = tmp_path / "curves.png"
    fig = viz.learning_curves(_seed_dirs(tmp_path), out, **kwargs)
    _assert_png(out)
    lines = _lines(fig)
    extra = ["reference"] if kwargs else []
    assert sorted(line.get_label() for line in lines) == sorted([*CURVES, *extra])
    by_label = {line.get_label(): line for line in lines}
    for name, curve in CURVES.items():
        steps, values = zip(*curve, strict=True)
        assert _xy(by_label[name]) == (list(steps), list(values)), name
    if kwargs:
        _, y = _xy(by_label["reference"])
        assert set(y) == {-1.5}


@pytest.mark.parametrize("n_event_files", [0, 2])
def test_learning_curves_refuses_a_run_dir_without_exactly_one_event_file(n_event_files, tmp_path):
    """A run dir must hold exactly one tb/**/events.out.tfevents.*; otherwise a ValueError names it
    (no silent skip of a seed, no silent pick between two logs)."""
    import greenleaf_control.viz as viz

    good = _tb_run(tmp_path / "seed0", CURVES["seed0"])
    bad = tmp_path / "seed7"
    (bad / "tb").mkdir(parents=True)
    if n_event_files == 2:
        _tb_run(bad, CURVES["seed1"])
        logger = configure(str(bad / "tb" / "PPO_2"), ["tensorboard"])
        logger.record("rollout/ep_rew_mean", 0.5)
        logger.dump(2048)
        logger.close()
    assert len(list(bad.glob("tb/**/events.out.tfevents.*"))) == n_event_files
    out = tmp_path / "curves.png"
    with pytest.raises(ValueError, match="seed7"):
        viz.learning_curves([good, bad], out)
    assert not out.exists()


# lambda_traces


def test_lambda_traces_draws_one_line_per_constraint(tmp_path):
    """3 lines, lambda_<c> against timesteps for c in metrics.CONSTRAINTS, labelled c; the cost_
    columns aren't drawn."""
    import greenleaf_control.viz as viz

    out = tmp_path / "lambdas.png"
    fig = viz.lambda_traces(_lambdas_csv(tmp_path / "lambdas.csv"), out)
    _assert_png(out)
    lines = _lines(fig)
    assert sorted(line.get_label() for line in lines) == sorted(metrics.CONSTRAINTS)
    by_label = {line.get_label(): line for line in lines}
    steps = [row[0] for row in LAMBDA_ROWS]
    for k, c in enumerate(metrics.CONSTRAINTS):
        assert _xy(by_label[c]) == (steps, [row[1 + k] for row in LAMBDA_ROWS]), c


# tradeoff


def test_tradeoff_draws_each_point_at_its_iqms_with_bars_to_its_cis(tmp_path):
    """One errorbar per point, labelled: the marker at (viol_hours IQM, profit IQM), a horizontal
    bar from viol_hours' lo to hi and a vertical bar from profit's lo to hi, both at the marker
    (e.g. ppo: marker (40, 12.5), bars x 30..52 at y 12.5 and y 10..14 at x 40).

    Each label is a direct text on the axes (ax.texts), not only a legend entry: the Instrument
    style's direct labels (#47), a requirement change from "the legend holds the three labels". A
    legend may still exist. "residual" must be its own text, not found inside "residual_lag"."""
    import greenleaf_control.viz as viz

    out = tmp_path / "tradeoff.png"
    fig = viz.tradeoff(POINTS, out)
    _assert_png(out)
    containers = [c for ax in fig.axes for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert sorted(c.get_label() for c in containers) == sorted(p["label"] for p in POINTS)
    by_label = {c.get_label(): c for c in containers}
    for p in POINTS:
        (profit, p_lo, p_hi), (viol, v_lo, v_hi) = p["profit"], p["viol_hours"]
        marker, _caps, bars = by_label[p["label"]].lines
        assert np.asarray(marker.get_xydata()).tolist() == [[viol, profit]], p["label"]
        segments = [np.asarray(s).tolist() for col in bars for s in col.get_segments()]
        horizontal = [s for s in segments if s[0][1] == s[1][1]]
        vertical = [s for s in segments if s[0][0] == s[1][0]]
        assert len(segments) == 2, p["label"]
        [[(x0, y0), (x1, y1)]] = horizontal
        assert (sorted([x0, x1]), y0) == (pytest.approx([v_lo, v_hi]), profit), p["label"]
        [[(x0, y0), (x1, y1)]] = vertical
        assert (x0, sorted([y0, y1])) == (viol, pytest.approx([p_lo, p_hi])), p["label"]
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    for p in POINTS:
        assert any(_names(text, p["label"]) for text in texts), (p["label"], texts)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "xlabel": "violation hours saved vs rb_tuned [h]",
            "ylabel": "profit gain over rb_tuned [€/m²]",
        },
    ],
    ids=["defaults", "the_reports_labels"],
)
def test_tradeoff_axis_labels_are_keywords_defaulting_to_todays(kwargs, tmp_path):
    """xlabel and ylabel are keyword-only, defaulting to today's "violation hours [h]" and "profit
    [€/m²]" (so a caller that passes neither draws what it drew before #47); the report passes its
    paired-change labels."""
    tradeoff = _viz("tradeoff")
    params = inspect.signature(tradeoff).parameters
    defaults = {"xlabel": "violation hours [h]", "ylabel": "profit [€/m²]"}
    for name, default in defaults.items():
        assert name in params, f"viz.tradeoff takes no {name} keyword yet (#47)"
        assert (params[name].kind, params[name].default) == (
            inspect.Parameter.KEYWORD_ONLY,
            default,
        )
    out = tmp_path / "tradeoff.png"
    fig = tradeoff(POINTS, out, **kwargs)
    _assert_png(out)
    want = {**defaults, **kwargs}
    assert (fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()) == (want["xlabel"], want["ylabel"])


def test_tradeoff_fits_its_axes_to_the_learning_arms(tmp_path):
    """The limits fit the learning arms (labels not in envs.BASELINES) with a margin: each arm's
    marker and both CI bars lie strictly inside them, and so does rb_tuned's origin. They don't
    stretch to the far baselines: zero_controls (x -2500) is off on the left only, random (y -30)
    below only, rb_default (-400, -7) on both. Without this, VALIDATION's zero_controls at -2587 h
    squeezes every learning arm into one pixel column (the Instrument style's scale rule, #47)."""
    tradeoff = _viz("tradeoff")
    out = tmp_path / "tradeoff.png"
    fig = tradeoff(SPREAD, out)
    _assert_png(out)
    ax = fig.axes[0]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for p in SPREAD:
        if p["label"] in envs.BASELINES:
            continue
        (_, v_lo, v_hi), (_, p_lo, p_hi) = p["viol_hours"], p["profit"]
        assert x0 < v_lo and v_hi < x1, (p["label"], (x0, x1))
        assert y0 < p_lo and p_hi < y1, (p["label"], (y0, y1))
    assert x0 < 0.0 < x1 and y0 < 0.0 < y1, "rb_tuned's origin is off the axes"
    for p in SPREAD:
        if p["label"] in OFF_LIMITS:
            x, y = p["viol_hours"][0], p["profit"][0]
            off = {"left": x < x0, "right": x > x1, "bottom": y < y0, "top": y > y1}
            sides = {side for side, is_off in off.items() if is_off}
            assert sides == OFF_LIMITS[p["label"]][0], (p["label"], (x0, x1), (y0, y1))


def test_tradeoff_labels_every_point_directly_in_mono(tmp_path):
    """Every point, the off-limit baselines and rb_tuned included, has its own text on the axes
    naming its label, in the requested MONO family list (the Instrument style's direct labels,
    #47); no label is dropped."""
    tradeoff = _viz("tradeoff")
    out = tmp_path / "tradeoff.png"
    fig = tradeoff(SPREAD, out)
    _assert_png(out)
    texts = list(fig.axes[0].texts)
    for p in SPREAD:
        mine = [t for t in texts if _names(t.get_text(), p["label"])]
        assert mine, (p["label"], [t.get_text() for t in texts])
        assert any(list(t.get_fontfamily()) == MONO for t in mine), (
            p["label"],
            [t.get_fontfamily() for t in mine],
        )


@pytest.mark.parametrize("label", list(OFF_LIMITS))
def test_tradeoff_draws_an_off_limit_point_as_an_edge_marker_with_its_true_values(label, tmp_path):
    """A point outside the limits isn't dropped or clipped away: a triangle sits on the axes edge
    where the point would be (its in-range coordinate kept, its off coordinate clamped to the edge,
    within 5% of the axes), pointing outward on a side the point is off (rb_default, off two
    sides, sits at the lower-left corner pointing left or down), drawn by plot or scatter. One text
    on the axes gives its label and the true value of each coordinate it's off: zero_controls
    -2500 (h saved), random -30 (€/m²), rb_default -400 and -7 (a Unicode minus or a thousands
    comma is fine)."""
    tradeoff = _viz("tradeoff")
    out = tmp_path / "tradeoff.png"
    fig = tradeoff(SPREAD, out)
    _assert_png(out)
    ax = fig.axes[0]
    sides, values = OFF_LIMITS[label]
    [p] = [p for p in SPREAD if p["label"] == label]
    at = ax.transAxes.inverted().transform(
        ax.transData.transform([(p["viol_hours"][0], p["profit"][0])])
    )[0]
    edge = np.clip(at, 0.0, 1.0)
    marks = [(side, xy) for side, xy in _triangles(ax) if side in sides]
    assert any(np.all(np.abs(xy - edge) <= 0.05) for _, xy in marks), (
        label,
        edge,
        _triangles(ax),
    )
    texts = [t.get_text() for t in ax.texts if _names(t.get_text(), label)]
    assert any(all(_gives(text, v) for v in values) for text in texts), (label, values, texts)


def _hollow(face) -> bool:
    """A marker face that shows the ground through it: "none", fully transparent, or GROUND."""
    if isinstance(face, str) and face.lower() == "none":
        return True
    return to_rgba(face)[3] == 0 or to_hex(face) == GROUND


def test_tradeoff_inks_only_adopted_arms(tmp_path):
    """The orchestrator's rule on #47: only an adopted learning arm gets ink. A point's "status"
    is "adopted", "dropped" or None (undecided: no decision row yet). ppo (adopted) is a filled INK
    dot (face and edge INK) with an INK direct label; residual_w4 (dropped) and lag_b100
    (undecided) are hollow dots (face GROUND or none) with a MUTED #6A6F68 edge and MUTED direct
    labels, so an undecided arm never reads as adopted."""
    tradeoff = _viz("tradeoff")
    statuses = {"ppo #33/0221": "adopted", "residual_w4 #34/4d20": "dropped"}
    statuses["lag_b100 #39/6d92"] = None
    points = [{**p, "status": statuses[p["label"]]} for p in SPREAD[:3]] + [SPREAD[3]]
    out = tmp_path / "tradeoff.png"
    fig = tradeoff(points, out)
    _assert_png(out)
    ax = fig.axes[0]
    bars = {c.get_label(): c for c in ax.containers if isinstance(c, ErrorbarContainer)}
    for label, status in statuses.items():
        dot = bars[label].lines[0]
        face, edge = dot.get_markerfacecolor(), to_hex(dot.get_markeredgecolor())
        colours = {to_hex(t.get_color()) for t in ax.texts if _names(t.get_text(), label)}
        if status == "adopted":
            assert (to_hex(face), edge, colours) == (INK, INK, {INK}), (label, status)
        else:
            assert (_hollow(face), edge, colours) == (True, MUTED, {MUTED}), (
                label,
                status,
                face,
                edge,
                colours,
            )


# The report's trade-off on london_2012 and abbotsford_2012 (#53's review): every learning arm
# gains a little profit and saves hours, so the y axis bottoms out at -0.096 (0.06 x 1.6 below 0)
# and x at -12, and rb_tuned's origin sits near the lower-left corner, where random and
# rb_default (-312 h, -6.47 €/m²), both off-scale left and below, are clamped; zero_controls is
# off on the left only.
CORNER = [
    {
        "label": "ppo_paper",
        "profit": (1.2, 0.9, 1.6),
        "viol_hours": (150.0, 120.0, 200.0),
        "status": "adopted",
        "n": 5,
    },
    {
        "label": "residual_w05",
        "profit": (0.5, 0.2, 0.8),
        "viol_hours": (60.0, 40.0, 90.0),
        "status": "adopted",
        "n": 10,
    },
    {
        "label": "lag_b100",
        "profit": (0.3, 0.05, 0.6),
        "viol_hours": (20.0, 5.0, 35.0),
        "status": None,
        "n": 10,
    },
    {"label": "rb_tuned", "profit": (0.0,) * 3, "viol_hours": (0.0,) * 3},
    {"label": "zero_controls", "profit": (0.4,) * 3, "viol_hours": (-2500.0,) * 3},
    {"label": "random", "profit": (-28.0,) * 3, "viol_hours": (-1400.0,) * 3},
    {"label": "rb_default", "profit": (-6.47,) * 3, "viol_hours": (-312.0,) * 3},
]


def _drawn(fig: Figure):
    """fig drawn on an Agg canvas, its layout run as for the PNG; returns the renderer, so every
    extent below is in the PNG's pixels."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def _text_box(text: Text, renderer) -> Bbox:
    """A text's own drawn box: for an annotation, without its leader (a hairline)."""
    return Text.get_window_extent(text, renderer)


@pytest.mark.parametrize(
    ("points", "off_scale"),
    [
        pytest.param(CORNER, ("zero_controls", "random", "rb_default"), id="london-shaped"),
        pytest.param(SPREAD, tuple(OFF_LIMITS), id="validation-shaped"),
    ],
)
def test_tradeoff_off_scale_labels_never_cover_the_rb_tuned_marker(points, off_scale, tmp_path):
    """Regression for #53 (the reviewer's FIX): on london_2012 and abbotsford_2012 the y axis
    bottoms out near -0.1, so rb_tuned's diamond at the origin sits next to the lower-left corner
    where random's and rb_default's off-scale labels are stacked, and they run across it. However
    the labels are placed, no text naming an off-scale baseline may overlap the rb_tuned marker's
    drawn box (its marker size included), in the drawn figure's pixels. Each off-scale baseline
    keeps a text naming it (test_tradeoff_draws_an_off_limit_point_as_an_edge_marker_with_its_true_
    values pins its true values). The validation-shaped case (SPREAD: y down to about -0.9)
    doesn't collide today and must not start to."""
    tradeoff = _viz("tradeoff")
    out = tmp_path / "tradeoff.png"
    fig = tradeoff(points, out)
    _assert_png(out)
    renderer = _drawn(fig)
    ax = fig.axes[0]
    [rb] = [
        c for c in ax.containers if isinstance(c, ErrorbarContainer) and c.get_label() == "rb_tuned"
    ]
    diamond = rb.lines[0].get_window_extent(renderer)
    for label in off_scale:
        texts = [t for t in fig.findobj(Text) if _names(t.get_text(), label)]
        assert texts, (label, _texts(fig))
        for t in texts:
            box = _text_box(t, renderer)
            assert not box.overlaps(diamond), (
                f"{t.get_text()!r} at {box} covers rb_tuned's marker at {diamond}"
            )


@pytest.mark.parametrize("plot", ["tradeoff", "profiles"])
def test_tradeoff_and_profiles_carry_a_one_line_mono_key_to_their_encoding(plot, tmp_path):
    """With direct labels and no colour legend, one line of MONO text names the encoding (exact
    wording free): it says "adopted", "dropped" and "rb_tuned" (e.g. ink = adopted, hollow =
    dropped or undecided, gains paired with rb_tuned's same season)."""
    plot_fn = _viz(plot)
    out = tmp_path / f"{plot}.png"
    fig = plot_fn(*_args(plot, tmp_path), out)
    _assert_png(out)
    keys = [t for t in fig.findobj(Text) if all(w in t.get_text() for w in KEY_WORDS)]
    assert keys, (plot, _texts(fig))
    assert any("\n" not in t.get_text() and list(t.get_fontfamily()) == MONO for t in keys), [
        (t.get_text(), t.get_fontfamily()) for t in keys
    ]


# per_scenario


def test_per_scenario_draws_every_value_as_a_dot_in_its_labels_row(tmp_path):
    """One axes per panel, its x label the panel's key. Each label is a y tick (every label, none
    dropped), in dict order from the top (the first on top), and every value of its runs x
    scenarios array is one scatter dot in its row: the x offsets in the row are the array's values
    (rb_tuned's profit row is -2, 1, 2; ppo's six, lag_b100's nine), nothing more. A dot may sit a
    little above or below its row's tick (under half a row), never in another row."""
    per_scenario = _viz("per_scenario")
    out = tmp_path / "per_scenario.png"
    fig = per_scenario(PANELS, SCENARIO_LABELS, out)
    _assert_png(out)
    assert [ax.get_xlabel() for ax in fig.axes] == list(PANELS)
    rows = _rows(fig)
    labels = list(PANELS[PROFIT_KEY])
    assert sorted(rows) == sorted(labels), rows
    assert _top_down(fig.axes[0], rows) == labels
    for ax, (key, arrays) in zip(fig.axes, PANELS.items(), strict=True):
        got = _by_row(ax, rows)
        for label in labels:
            xs = sorted(x for x, _, _ in got[label])
            assert xs == sorted(arrays[label].ravel().tolist()), (key, label)


def test_per_scenario_tells_the_scenarios_apart_in_every_row_and_names_them(tmp_path):
    """The issue's "coloured by scenario", monochrome under the Instrument style (#47): in each
    row, every dot of one scenario column looks alike (marker, fill, edge, size and offset from the
    row tick) and differs from the other two scenarios' dots, however that's drawn (shape, shade or
    a small offset). Each scenario label ("Amsterdam day 45" ...) is a text of the figure (a
    legend entry or a direct label), so a reader can name them."""
    per_scenario = _viz("per_scenario")
    out = tmp_path / "per_scenario.png"
    fig = per_scenario(PANELS, SCENARIO_LABELS, out)
    _assert_png(out)
    rows = _rows(fig)
    for ax, (key, arrays) in zip(fig.axes, PANELS.items(), strict=True):
        got = _by_row(ax, rows)
        for label, values in arrays.items():
            column = {float(v): j for (_, j), v in np.ndenumerate(values)}
            looks = {j: set() for j in range(len(SCENARIO_LABELS))}
            for x, offset, look in got[label]:
                assert x in column, (key, label, x)
                looks[column[x]].add((offset, look))
            assert all(len(s) == 1 for s in looks.values()), (key, label, looks)
            assert len({next(iter(s)) for s in looks.values()}) == len(SCENARIO_LABELS), (
                key,
                label,
                looks,
            )
    texts = _texts(fig)
    for scenario in SCENARIO_LABELS:
        assert any(scenario in text for text in texts), (scenario, texts)


def test_per_scenario_draws_a_far_baselines_values_as_a_triangle_on_its_rows_edge(tmp_path):
    """random's row (first) holds two values per panel far outside the learning arms' range
    (profit -30 and -28 €/m², left of about -2.5; hours 2,400 and 2,600, right of about 30) and one
    inside it (1.5, 10). The x limits still fit the learning arms and rb_tuned, so the far values
    lie beyond the limit on their side. In random's row the dots are exactly the inside value; on
    that row's edge a triangle points out (< in profit, > in hours, within 5% of the axes edge);
    one text on the axes gives the far values' true lo and hi (-30 and -28; 2,400 and 2,600). The
    other rows' dots are unchanged."""
    per_scenario = _viz("per_scenario")
    panels = {key: {"random": FAR_ROW[key], **arrays} for key, arrays in PANELS.items()}
    out = tmp_path / "per_scenario.png"
    fig = per_scenario(panels, SCENARIO_LABELS, out)
    _assert_png(out)
    rows = _rows(fig)
    assert _top_down(fig.axes[0], rows) == ["random", *PANELS[PROFIT_KEY]]
    for ax, key in zip(fig.axes, panels, strict=True):
        side, lo, hi, inside = FAR_SIDES[key]
        x0, x1 = ax.get_xlim()
        assert (hi < x0) if side == "left" else (lo > x1), (key, (x0, x1))
        got = _by_row(ax, rows)
        assert [x for x, _, _ in got["random"]] == [inside], (key, got["random"])
        for label, values in PANELS[key].items():
            assert sorted(x for x, _, _ in got[label]) == sorted(values.ravel().tolist()), label
        on_row = [xy for s, xy in _triangles(ax) if s == side and _row_at(ax, rows, xy) == "random"]
        edge = 0.0 if side == "left" else 1.0
        assert any(abs(xy[0] - edge) <= 0.05 for xy in on_row), (key, _triangles(ax))
        texts = [t.get_text() for t in ax.texts]
        assert any(_gives(t, lo) and _gives(t, hi) for t in texts), (key, lo, hi, texts)


# amsterdam_2012's per_scenario (#53's review): 8 scenarios, and zero_controls (a far baseline)
# has five profit seasons in scale near the right edge (2.33 to 3.47) and three off it (3.80 to
# 3.90). The learning arms and rb_tuned span -0.5..3.5, so the profit x limits are -0.74..3.74
# (0.06 x 4 each side); in hours zero_controls is wholly off the right edge.
AMSTERDAM_LABELS = [f"Amsterdam day {d}" for d in (45, 49, 53, 57, 61, 65, 69, 73)]
NEAR_EDGE = {
    PROFIT_KEY: {
        "zero_controls": np.array([[2.33, 2.68, 2.99, 3.27, 3.47, 3.80, 3.85, 3.90]]),
        "rb_tuned": np.array([[0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2]]),
        "ppo_paper": np.array(
            [
                [-0.5, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25],
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            ]
        ),
    },
    HOURS_KEY: {
        "zero_controls": np.array(
            [[2600.0, 2650.0, 2700.0, 2750.0, 2800.0, 2850.0, 2900.0, 2950.0]]
        ),
        "rb_tuned": np.array([[20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0]]),
        "ppo_paper": np.array(
            [
                [10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                [5.0, 12.0, 18.0, 26.0, 34.0, 42.0, 51.0, 66.0],
            ]
        ),
    },
}


def _dot_boxes(ax) -> list[Bbox]:
    """The drawn box of every scatter dot on ax (the in-scale values), in pixels: its centre padded
    by its marker path's reach at its size (s is in points², so a path unit is sqrt(s) points)."""
    dpi = ax.get_figure(root=True).dpi
    boxes = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        paths, sizes = coll.get_paths(), coll.get_sizes()
        offsets = np.asarray(coll.get_offsets(), dtype=float).reshape(-1, 2)
        for i, (x, y) in enumerate(coll.get_offset_transform().transform(offsets)):
            reach = np.abs(paths[i % len(paths)].vertices).max(axis=0)
            dx, dy = reach * np.sqrt(sizes[i % len(sizes)]) * dpi / 72
            boxes.append(Bbox.from_extents(x - dx, y - dy, x + dx, y + dy))
    return boxes


@pytest.mark.parametrize(
    ("panels", "scenario_labels", "ranges"),
    [
        pytest.param(
            NEAR_EDGE,
            AMSTERDAM_LABELS,
            {PROFIT_KEY: (3.80, 3.90), HOURS_KEY: (2600.0, 2950.0)},
            id="amsterdam-shaped",
        ),
        pytest.param(
            {key: {"random": FAR_ROW[key], **arrays} for key, arrays in PANELS.items()},
            SCENARIO_LABELS,
            {key: FAR_SIDES[key][1:3] for key in PANELS},
            id="validation-shaped",
        ),
    ],
)
def test_per_scenario_off_scale_range_text_never_covers_an_in_scale_dot(
    panels, scenario_labels, ranges, tmp_path
):
    """Regression for #53 (the reviewer's FIX): on amsterdam_2012, zero_controls' off-scale range
    text "3.80 to 3.90" is right-aligned from its row's right edge across x 2.3 to 3.5, and its
    own in-scale dots (2.33 ... 3.47) are drawn on top of it. However the text is placed, the
    text giving a panel's off-scale range (its true lo and hi) may not overlap any in-scale dot's
    drawn box on that panel (in practice its row's), in the drawn figure's pixels. The
    validation-shaped case (random's far values with one in-scale value mid-axes) doesn't collide
    today and must not start to."""
    per_scenario = _viz("per_scenario")
    out = tmp_path / "per_scenario.png"
    fig = per_scenario(panels, scenario_labels, out)
    _assert_png(out)
    renderer = _drawn(fig)
    for ax in fig.axes:
        lo, hi = ranges[ax.get_xlabel()]
        texts = [
            t for t in fig.findobj(Text) if _gives(t.get_text(), lo) and _gives(t.get_text(), hi)
        ]
        assert texts, (ax.get_xlabel(), lo, hi, _texts(fig))
        dots = _dot_boxes(ax)
        for t in texts:
            box = _text_box(t, renderer)
            hits = [d for d in dots if box.overlaps(d)]
            assert not hits, f"{t.get_text()!r} at {box} covers {len(hits)} in-scale dot(s): {hits}"


@pytest.mark.parametrize("columns", [2, 4])
def test_per_scenario_refuses_an_array_whose_columns_arent_the_scenarios(columns, tmp_path):
    """Every array is runs x scenarios: one with 2 or 4 columns against 3 scenario labels (here the
    hours panel's last label) is a ValueError, and no PNG is written."""
    per_scenario = _viz("per_scenario")
    panels = {key: dict(arrays) for key, arrays in PANELS.items()}
    panels[HOURS_KEY]["lag_b100 #39/6d92"] = np.arange(2.0 * columns).reshape(2, columns)
    out = tmp_path / "per_scenario.png"
    with pytest.raises(ValueError):
        per_scenario(panels, SCENARIO_LABELS, out)
    assert not out.exists()


# profiles


def test_profiles_draws_each_labels_fraction_above_tau(tmp_path):
    """One complementary ECDF per label per panel, labelled "<label> (n=<values.size>)", drawn
    steps-pre: at each tau between its smallest and largest value the line reads the fraction of
    the label's values above tau. Known answers, ppo's profit gains (-1, 0.5, 2, 3): 3 of 4 above
    tau = -0.999 (0.75), 3 of 4 above 0, 2 of 4 above 1 (0.5), 1 of 4 above 2.5."""
    profiles = _viz("profiles")
    out = tmp_path / "profiles.png"
    fig = profiles(PROFILES, out)
    _assert_png(out)
    assert [ax.get_xlabel() for ax in fig.axes] == list(PROFILES)
    for ax, (key, arrays) in zip(fig.axes, PROFILES.items(), strict=True):
        lines = {line.get_label(): line for line in ax.get_lines()}
        for label, values in arrays.items():
            name = f"{label} (n={values.size})"
            assert name in lines, (key, name, list(lines))
            line = lines[name]
            assert line.get_drawstyle() == "steps-pre", (key, name)
            v = np.sort(values.ravel())
            taus = [v[0] + 1e-3, *((v[:-1] + v[1:]) / 2)]
            for tau in taus:
                assert _above(line, tau) == pytest.approx(np.mean(v > tau), abs=1e-12), (key, tau)
    ppo = {line.get_label(): line for line in fig.axes[0].get_lines()}["ppo #33/0221 (n=4)"]
    for tau, fraction in [(-0.999, 0.75), (0.0, 0.75), (1.0, 0.5), (2.5, 0.25)]:
        assert _above(ppo, tau) == pytest.approx(fraction, abs=1e-12), tau


def test_profiles_marks_tau_zero_with_a_dashed_line_and_labels_the_y_axis(tmp_path):
    """Each panel has a dashed vertical line at tau = 0 (where the profit profile reads the
    same-season win fraction), and the y label "fraction of seeds × seasons above τ" (on every
    panel, or on the first when the y axis is shared)."""
    profiles = _viz("profiles")
    out = tmp_path / "profiles.png"
    fig = profiles(PROFILES, out)
    _assert_png(out)
    assert fig.axes[0].get_ylabel() == PROFILES_YLABEL
    for ax in fig.axes:
        assert ax.get_ylabel() in (PROFILES_YLABEL, ""), ax.get_ylabel()
        zero = [
            line
            for line in ax.get_lines()
            if line.get_linestyle() == "--"
            and set(np.asarray(line.get_xdata(), dtype=float).tolist()) == {0.0}
        ]
        assert zero, (
            ax.get_xlabel(),
            [(ln.get_label(), ln.get_linestyle()) for ln in ax.get_lines()],
        )


def test_profiles_draws_highlighted_labels_in_ink_with_a_direct_label_and_the_rest_in_hairline(
    tmp_path,
):
    """highlight names the adopted arms: each such line is INK #111311 with a direct text naming
    its label on the axes; every other line is HAIRLINE #D5D8D2 with no text naming it, though its
    ECDF is still drawn and labelled "<label> (n=...)" (the Instrument style, #47, in place of the
    issue's tab20 colours)."""
    profiles = _viz("profiles")
    highlight = inspect.signature(profiles).parameters.get("highlight")
    assert highlight is not None, "viz.profiles takes no highlight yet (#47)"
    assert (highlight.kind, tuple(highlight.default)) == (inspect.Parameter.KEYWORD_ONLY, ())
    out = tmp_path / "profiles.png"
    fig = profiles(PROFILES, out, highlight=("ppo #33/0221",))
    _assert_png(out)
    for ax, (key, arrays) in zip(fig.axes, PROFILES.items(), strict=True):
        lines = {line.get_label(): line for line in ax.get_lines()}
        texts = [t.get_text() for t in ax.texts]
        for label, values in arrays.items():
            line = lines[f"{label} (n={values.size})"]
            named = any(_names(text, label) for text in texts)
            if label == "ppo #33/0221":
                assert (to_hex(line.get_color()), named) == (INK, True), (key, label, texts)
            else:
                assert (to_hex(line.get_color()), named) == (HAIRLINE, False), (key, label, texts)


# Drift guard against train's real output


@pytest.mark.slow  # 13-29 s on the Mac: residual_lag's budgets (9 RB seasons), 4,096 PPO steps
def test_curves_and_lambda_traces_read_a_real_residual_lag_smoke_run(tmp_path):
    """train --smoke for residual_lag, seed 0, one env, 4,096 steps on 1-day seasons, rb default,
    budget_frac 0.5; its runs dir and ledger are under tmp_path.

    Known answers: with one env a rollout is n_steps = 2,048 steps, so 4,096 steps is two rollouts.
    PPO dumps its logger after each rollout (log_interval 1), when 96-step episodes have ended, so
    tb/ holds rollout/ep_rew_mean at steps 2,048 and 4,096; LagrangianCallback appends one
    lambdas.csv row per rollout end, at 2,048 and 4,096.
    """
    import greenleaf_control.train as train

    real_ledger, real_runs = _stat(REAL_LEDGER), _listing(REAL_RUNS)
    runs = tmp_path / "runs"
    argv = [
        *("--method", "residual_lag"),
        *("--seed", "0"),
        "--smoke",
        *("--timesteps", "4096"),
        *("--season-length", "1"),
        *("--n-envs", "1"),
        *("--rb", "default"),
        *("--budget-frac", "0.5"),
        *("--runs-dir", str(runs)),
    ]
    assert train.main(argv, ledger_path=tmp_path / "ledger.jsonl") == 0
    assert (_stat(REAL_LEDGER), _listing(REAL_RUNS)) == (real_ledger, real_runs)
    [run_dir] = list((runs / "smoke").iterdir())
    [events] = list(run_dir.glob("tb/**/events.out.tfevents.*"))
    acc = EventAccumulator(str(events.parent))
    acc.Reload()
    logged = [(e.step, e.value) for e in acc.Scalars("rollout/ep_rew_mean")]
    assert [step for step, _ in logged] == [2048, 4096]
    with (run_dir / "lambdas.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["timesteps"]) for row in rows] == [2048, 4096]

    import greenleaf_control.viz as viz  # after the run: while viz is missing, this test fails here

    out = tmp_path / "curves.png"
    [curve] = _lines(viz.learning_curves([run_dir], out))
    _assert_png(out)
    assert curve.get_label() == run_dir.name
    assert _xy(curve) == ([2048.0, 4096.0], [value for _, value in logged])

    out = tmp_path / "lambdas.png"
    lines = _lines(viz.lambda_traces(run_dir / "lambdas.csv", out))
    _assert_png(out)
    assert sorted(line.get_label() for line in lines) == sorted(metrics.CONSTRAINTS)
    for line in lines:
        c = line.get_label()
        x, y = _xy(line)
        assert x == [2048.0, 4096.0], c
        # rel 1e-12: the callback writes repr floats, which pandas' default parser can miss by 1 ulp
        assert y == pytest.approx([float(row[f"lambda_{c}"]) for row in rows], rel=1e-12), c

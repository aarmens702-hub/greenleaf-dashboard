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
- tradeoff(points, out): one errorbar per point at (viol_hours IQM, profit IQM), its bars reaching
  each CI's lo and hi, labelled in a legend.

A refused input (a frame without 6 actuators, a run dir without exactly one event file, or nothing
to draw: no run dirs, no points, a header-only lambdas.csv) is a ValueError, and no PNG is written.

viz reads only metrics' constants: it never imports rewards, envs, lagrangian or train.

The fixtures are hand-built (a synthetic per-step frame, TensorBoard logs written by SB3's own
logger, a lambdas.csv under LagrangianCallback's header), plus one smoke run of train as a drift
guard. Nothing here touches control/runs/ or the real ledger.
"""

import ast
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from stable_baselines3.common.logger import configure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import greenleaf_control
from greenleaf_control import metrics

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

# (iqm, lo, hi) as stats.iqm_ci returns them; every bar is asymmetric (iqm - lo != hi - iqm)
POINTS = [
    {"label": "rb_default", "profit": (12.5, 10.0, 14.0), "viol_hours": (40.0, 30.0, 52.0)},
    {"label": "residual", "profit": (15.25, 14.5, 16.0), "viol_hours": (60.5, 48.0, 64.0)},
    {"label": "residual_lag", "profit": (13.0, 11.75, 13.5), "viol_hours": (8.0, 6.5, 12.25)},
]


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


# Every function


@pytest.mark.parametrize("plot", ["season_strip", "learning_curves", "lambda_traces", "tradeoff"])
def test_every_plot_returns_a_white_figure_and_saves_it_as_png(plot, tmp_path):
    """A white facecolor (not transparent) keeps each PNG readable in both light and dark README
    themes; the function returns the Figure it saved."""
    import greenleaf_control.viz as viz

    first = {
        "season_strip": lambda: _frame(),
        "learning_curves": lambda: _seed_dirs(tmp_path),
        "lambda_traces": lambda: _lambdas_csv(tmp_path / "lambdas.csv"),
        "tradeoff": lambda: POINTS,
    }[plot]()
    out = tmp_path / f"{plot}.png"
    fig = getattr(viz, plot)(first, out)
    assert isinstance(fig, Figure)
    assert to_hex(fig.get_facecolor()) == "#ffffff"
    _assert_png(out)


@pytest.mark.parametrize(
    "plot",
    ["learning_curves", "lambda_traces", "tradeoff"],
    ids=["no_run_dirs", "header_only_csv", "no_points"],
)
def test_an_empty_input_raises_and_writes_no_png(plot, tmp_path):
    """Nothing to draw is a ValueError, never a blank PNG that reads as a run with no data."""
    import greenleaf_control.viz as viz

    empty = {
        "learning_curves": lambda: [],
        "lambda_traces": lambda: _lambdas_csv(tmp_path / "lambdas.csv", rows=[]),
        "tradeoff": lambda: [],
    }[plot]()
    out = tmp_path / f"{plot}.png"
    with pytest.raises(ValueError):
        getattr(viz, plot)(empty, out)
    assert not out.exists()


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
    (e.g. rb_default: marker (40, 12.5), bars x 30..52 at y 12.5 and y 10..14 at x 40). The legend
    holds the three labels."""
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
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + fig.legends
    texts = [t.get_text() for legend in legends for t in legend.get_texts()]
    assert sorted(texts) == sorted(p["label"] for p in POINTS)


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

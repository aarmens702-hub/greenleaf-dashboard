"""The standard plots of a run: what a reviewer looks at to catch reward hacking (S1-13).

Each function draws on a matplotlib Figure built without pyplot (headless, no global state), saves
it to `out` as a PNG and returns it. A run's plots are on white with default text colours, so a PNG
reads as a white card in both light and dark README themes. The report's three figures (tradeoff,
per_scenario, profiles; #47) use the Instrument style: an opaque GROUND card, ink and hairlines,
MONO direct labels instead of a colour legend, and a scale fitted to the learning arms and rb_tuned,
a far baseline drawn as a triangle on the edge with its true value. viz draws raw data and metrics'
constants only: it never imports the reward, env, Lagrangian or training code, and it draws the CIs
it's given without computing any.
"""

import csv
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from greenleaf_control import metrics

CLIMATE_LABELS = {"co2": "CO₂\n[ppm]", "temp": "temp\n[°C]", "rh": "RH\n[%]"}

# The Instrument style (#47): the report figures' palette and fonts. Installed families only, as in
# media.py (#51): matplotlib logs a warning per glyph for each missing family it tries. The lists
# fall back in order, so a machine without the Mac fonts draws matplotlib's own DejaVu.
GROUND, INK, MUTED, HAIRLINE, GREY = "#F2F3F0", "#111311", "#6A6F68", "#D5D8D2", "#8A8F87"
SANS = ["Helvetica Neue", "Arial", "DejaVu Sans"]
MONO = ["Menlo", "DejaVu Sans Mono"]
AXIS = {"fontfamily": MONO, "fontsize": 8, "color": MUTED}  # axis labels
# envs.BASELINES, held here because viz can't import envs. rb_tuned is the reference every gain is
# paired with; the scale rule fits it and the learning arms, never a FAR baseline.
BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned")
REFERENCE = "rb_tuned"
FAR = tuple(b for b in BASELINES if b != REFERENCE)
SHAPES = "o^sDv<>ph*"  # per_scenario's scenarios, one hollow shape each
CLAIM_RUNS = 5  # PROTOCOL: no claim on fewer runs (selection.NO_CLAIM); tradeoff flags them
# One-line MONO keys: with direct labels and no colour legend, they name the encoding
TRADEOFF_KEY = (
    "filled ink: adopted · hollow: dropped or undecided · ◇ rb_tuned · ■ baseline"
    " · ▸ off scale, true (x, y)"
)
PROFILES_KEY = (
    "ink: adopted · pale: dropped, undecided or baseline · each value paired with rb_tuned's"
    " same season · dashed: τ = 0"
)


def _ylabel(ax, text: str) -> None:
    """A horizontal label left of a strip axes, so short stacked axes keep it readable."""
    ax.set_ylabel(text, rotation=0, ha="right", va="center")


def season_strip(frame, out: Path) -> Figure:
    """Indoor co2, temp and rh against frame["t"] over their metrics.BOUNDS bands, then each of the
    6 actuators (the u_ columns that aren't u_rb_, in frame order) on 0-1: 9 axes sharing x.

    frame is the contract's per-step frame (or any mapping of column to sequence). The actuator
    axes are padded past 0 and 1, so an actuator held at a bound stays visible instead of hiding
    under the axes frame.
    """
    actuators = [c for c in frame if c.startswith("u_") and not c.startswith("u_rb_")]
    if len(actuators) != 6:
        raise ValueError(f"season_strip needs 6 actuator columns u_*, got {actuators}")
    fig = Figure(figsize=(10, 12), facecolor="white", layout="constrained")
    axes = fig.subplots(9, 1, sharex=True)
    for ax, c in zip(axes[:3], metrics.CONSTRAINTS, strict=True):
        ax.axhspan(*metrics.BOUNDS[c], color="tab:green", alpha=0.2, lw=0)
        ax.plot(frame["t"], frame[c], color="tab:blue")
        _ylabel(ax, CLIMATE_LABELS[c])
    for ax, c in zip(axes[3:], actuators, strict=True):
        ax.plot(frame["t"], frame[c], color="tab:orange")
        ax.set_ylim(-0.05, 1.05)
        _ylabel(ax, f"{c}\n[0–1]")
    axes[-1].set_xlabel("step (15 min)")
    fig.savefig(out)
    return fig


def learning_curves(run_dirs, out: Path, reference: float | None = None) -> Figure:
    """rollout/ep_rew_mean against timesteps, one line per run dir (every seed, never just the
    best), labelled by the dir's name and read from its one tb/**/events.out.tfevents.*; with a
    reference, a dashed horizontal line at it.

    - ep_rew_mean is the env reward VecMonitor records below VecLagrangianReward: scaled_profit
      minus w times the summed costs, at the run's penalty weight w. residual_lag runs at w = 0, so
      its curve is profit only and can rise while violations grow.
    - reference must be the grower's return under the same reward and season length.
    - ep_rew_mean is SB3's mean over the last 100 episodes, so it lags training.
    - One arm per plot: registered run dirs are all named seed<N>, so labels collide across arms.
    """
    run_dirs = [Path(d) for d in run_dirs]
    if not run_dirs:
        raise ValueError("learning_curves needs at least one run dir")
    fig = Figure(figsize=(8, 5), facecolor="white", layout="constrained")
    ax = fig.subplots()
    for run_dir in run_dirs:
        events = sorted(run_dir.glob("tb/**/events.out.tfevents.*"))
        if len(events) != 1:
            raise ValueError(
                f"{run_dir}: need exactly one tb/**/events.out.tfevents.*, got {len(events)}"
            )
        acc = EventAccumulator(str(events[0]), size_guidance={"scalars": 0})
        scalars = acc.Reload().Scalars("rollout/ep_rew_mean")
        ax.plot([s.step for s in scalars], [s.value for s in scalars], label=run_dir.name)
    if reference is not None:
        ax.axhline(reference, ls="--", color="black", label="reference")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("episode return (env reward at the run's w)")
    ax.legend()
    fig.savefig(out)
    return fig


def lambda_traces(lambdas_csv: Path, out: Path) -> Figure:
    """Each constraint's multiplier lambda_<c> against timesteps, from LagrangianCallback's
    lambdas.csv, one line per metrics.CONSTRAINTS entry labelled c."""
    with Path(lambdas_csv).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{lambdas_csv} has no data rows")
    fig = Figure(figsize=(8, 4.5), facecolor="white", layout="constrained")
    ax = fig.subplots()
    steps = [int(row["timesteps"]) for row in rows]
    for c in metrics.CONSTRAINTS:
        ax.plot(steps, [float(row[f"lambda_{c}"]) for row in rows], label=c)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("λ")
    ax.legend()
    fig.savefig(out)
    return fig


def _ground(
    figsize: tuple[float, float], title: str, footer: str | None, key: str | None = None
) -> Figure:
    """A report figure: GROUND, constrained layout, a SANS title top left and, along the bottom,
    the footer and above it the key (each if given) in MONO lines, the layout kept clear of them."""
    fig = Figure(figsize=figsize, facecolor=GROUND, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontfamily=SANS, fontsize=12, color=INK)
    pt = 1 / (72 * figsize[1])  # one point in figure fractions
    bottom = 3 * pt
    for text, size in ((footer, 7), (key, 7.5)):
        if text is not None:
            fig.text(0.01, bottom, text, fontfamily=MONO, fontsize=size, color=MUTED, va="bottom")
            bottom += (size + 7) * pt
    if bottom > 3 * pt:
        fig.get_layout_engine().set(rect=(0, bottom, 1, 1 - bottom))
    return fig


def _instrument(ax) -> None:
    """The Instrument look of one axes: GROUND, no top or right spine, HAIRLINE spines and grid
    under the data, MONO tick labels in MUTED."""
    ax.set_facecolor(GROUND)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(HAIRLINE)
    ax.grid(color=HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelfontfamily=MONO, colors=MUTED, labelsize=8, length=0)


def _fit(values) -> tuple[float, float]:
    """Limits holding every value and 0, with a 6% margin each side (1 when they're all 0)."""
    lo, hi = min([0.0, *values]), max([0.0, *values])
    pad = 0.06 * (hi - lo) or 1.0
    return lo - pad, hi + pad


def _num(v: float) -> str:
    """v in a direct label: whole and thousands-separated from 100 up, else 2 decimals."""
    return (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}").replace("-", "−")


def _direct(ax, text: str, xy, at, color: str, **kwargs):
    """A MONO direct label beside xy (data coordinates unless kwargs give xycoords); at is xy in
    axes fractions. It sits up and right of xy, flipped left in the axes' right third and below in
    its top tenth, so it stays inside; kwargs override any of that. A GROUND halo keeps it legible
    over lines, and a leader ties it to xy once _spread moves it. It's out of the layout, so
    _spread can move it."""
    left, below = at[0] > 0.7, at[1] > 0.9
    leader = {"arrowstyle": "-", "color": MUTED, "lw": 0.5, "shrinkA": 0, "shrinkB": 4}
    style = {
        "xytext": (-4 if left else 4, -3 if below else 3),
        "textcoords": "offset points",
        "ha": "right" if left else "left",
        "va": "top" if below else "bottom",
        "color": color,
        "fontfamily": MONO,
        "fontsize": 7.5,
        "path_effects": [patheffects.withStroke(linewidth=2.5, foreground=GROUND)],
        "arrowprops": {**leader, "relpos": (1 if left else 0, 1 if below else 0)},
        "in_layout": False,
        "annotation_clip": False,
    }
    return ax.annotate(text, xy, **{**style, **kwargs})


def _spread(ax, texts: list) -> None:
    """Moves ax's direct labels apart vertically until no two overlap, each keeping its x: those in
    the lower half of the axes, lowest first, move up past every label already placed; the rest,
    highest first, move down."""
    if not texts:
        return
    fig = ax.get_figure(root=True)
    fig.draw_without_rendering()  # the layout, so the boxes are where the PNG puts them
    start = {t: t.get_window_extent() for t in texts}
    mid = (ax.bbox.y0 + ax.bbox.y1) / 2
    low = sorted(
        (t for t in texts if start[t].y0 + start[t].y1 < 2 * mid), key=lambda t: start[t].y0
    )
    high = sorted((t for t in texts if t not in low), key=lambda t: -start[t].y1)
    placed = []
    for t in low + high:
        box = start[t]
        while hits := [b for b in placed if box.overlaps(b)]:
            if t in low:
                box = box.translated(0, max(b.y1 for b in hits) + 1 - box.y0)
            else:
                box = box.translated(0, min(b.y0 for b in hits) - 1 - box.y1)
        placed.append(box)
        dx, dy = t.xyann
        t.xyann = (dx, dy + (box.y0 - start[t].y0) * 72 / fig.dpi)


def tradeoff(
    points: list[dict],
    out: Path,
    *,
    xlabel: str = "violation hours [h]",
    ylabel: str = "profit [€/m²]",
    footer: str | None = None,
) -> Figure:
    """Profit against violation-hours: one errorbar per point at its (viol_hours, profit) IQMs, its
    bars reaching each CI's lo and hi. A point is {"label", "profit": (iqm, lo, hi), "viol_hours":
    (iqm, lo, hi)}, as stats.iqm_ci returns them, plus an optional "status" ("adopted",
    "dropped" or None: undecided) and "n" (its runs).

    x is whatever violation-hours the caller passes: PROTOCOL §4 defines viol_hours per constraint,
    so the caller states whether it's one constraint's or the sum (and xlabel, ylabel say so).

    Instrument style: only an adopted learning arm is a filled ink dot; a dropped or undecided one
    is hollow and muted, a baseline a grey square, rb_tuned a hollow diamond. Each has a MONO
    direct label (a learning arm under CLAIM_RUNS runs adds "(n=<k>, no claim)"), and one MONO
    line keys the encoding; no legend. The limits fit every learning arm's marker and CIs,
    rb_tuned's and the origin; a point outside them is a triangle on the edge pointing out,
    labelled with its true (x, y).
    """
    if not points:
        raise ValueError("tradeoff needs at least one point")
    fit = [p for p in points if p["label"] not in FAR] or points
    (x0, x1), (y0, y1) = (_fit([v for p in fit for v in p[k]]) for k in ("viol_hours", "profit"))
    fig = _ground((8, 5.8), "Trade-off: IQM and 95% CI per label", footer, TRADEOFF_KEY)
    ax = fig.subplots()
    _instrument(ax)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.axvline(0.0, color=MUTED, lw=0.6)
    labels = []
    for p in points:
        (x, x_lo, x_hi), (y, y_lo, y_hi) = p["viol_hours"], p["profit"]
        fx, fy = (x - x0) / (x1 - x0), (y - y0) / (y1 - y0)
        beyond = {"<": -fx, ">": fx - 1, "v": -fy, "^": fy - 1}  # how far off each side
        mark = max(beyond, key=beyond.get)
        name = p["label"]
        if name not in BASELINES and p.get("n", CLAIM_RUNS) < CLAIM_RUNS:
            name += f" (n={p['n']}, no claim)"
        if beyond[mark] > 0:
            at = (min(max(fx, 0.015), 0.985), min(max(fy, 0.015), 0.985))
            ax.plot(*at, marker=mark, ms=7, color=GREY, transform=ax.transAxes, clip_on=False)
            text = f"{name} ({_num(x)}, {_num(y)})"
            labels.append(_direct(ax, text, at, at, MUTED, xycoords="axes fraction"))
            continue
        if p["label"] == REFERENCE:
            look = {"fmt": "D", "color": INK, "mfc": GROUND, "ms": 6.5}
        elif p["label"] in BASELINES:
            look = {"fmt": "s", "color": GREY, "ms": 6}
        elif p.get("status") == "adopted":
            look = {"fmt": "o", "color": INK, "ms": 6}
        else:  # dropped or undecided: only an adopted arm gets ink
            look = {"fmt": "o", "color": MUTED, "mfc": GROUND, "ms": 6}
        xerr, yerr = [[x - x_lo], [x_hi - x]], [[y - y_lo], [y_hi - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, elinewidth=0.8, label=p["label"], **look)
        ink = INK if look["color"] == INK else MUTED
        labels.append(_direct(ax, name, (x, y), (fx, fy), ink))
    ax.set_xlabel(xlabel, **AXIS)
    ax.set_ylabel(ylabel, **AXIS)
    _spread(ax, labels)
    fig.savefig(out)
    return fig


def per_scenario(
    panels: dict[str, dict[str, np.ndarray]],
    scenario_labels: list[str],
    out: Path,
    *,
    footer: str | None = None,
) -> Figure:
    """Every value as a dot: one axes per panel (its key the x label), one row per label (the
    panels' shared labels, in dict order, the first on top), and each label's runs x scenarios
    array drawn as dots at (value, row), one hollow shape per scenario, named in a key.

    The x limits fit every label but a FAR baseline, and 0; a FAR baseline's values outside them
    are a triangle on the row's edge with their true range. A panel with other labels than the
    first's, or an array whose column count isn't len(scenario_labels), raises a ValueError.
    """
    if not panels:
        raise ValueError("per_scenario needs at least one panel")
    if len(scenario_labels) > len(SHAPES):
        raise ValueError(f"per_scenario tells {len(SHAPES)} scenarios apart, not {scenario_labels}")
    rows = list(next(iter(panels.values())))
    for key, arrays in panels.items():
        if list(arrays) != rows:
            raise ValueError(f"panel {key!r} has labels {list(arrays)}, not the first's {rows}")
        for label, values in arrays.items():
            if np.ndim(values) != 2 or np.shape(values)[1] != len(scenario_labels):
                raise ValueError(
                    f"panel {key!r}: {label}'s array is {np.shape(values)}, not runs x"
                    f" {len(scenario_labels)} scenarios"
                )
    fig = _ground((4.8 * len(panels), 1.6 + 0.3 * len(rows)), "Every run on every season", footer)
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for ax, (key, arrays) in zip(axes, panels.items(), strict=True):
        _instrument(ax)
        x0, x1 = _fit([v for label in rows if label not in FAR for v in np.ravel(arrays[label])])
        ax.set_xlim(x0, x1)
        for j, (scenario, shape) in enumerate(zip(scenario_labels, SHAPES, strict=False)):
            xs, ys = [], []
            for row, values in enumerate(arrays.values()):
                column = np.asarray(values, dtype=float)[:, j]
                inside = column[(x0 <= column) & (column <= x1)]
                xs += inside.tolist()
                ys += [row] * inside.size
            ax.scatter(
                xs,
                ys,
                s=24,
                marker=shape,
                facecolors="none",
                edgecolors=INK,
                lw=0.8,
                label=scenario,
            )
        xy = ax.get_yaxis_transform()  # x in axes fractions, y in rows
        for row, values in enumerate(arrays.values()):
            v = np.ravel(values)
            for off, edge, mark in ((v[v < x0], 0.012, "<"), (v[v > x1], 0.988, ">")):
                if off.size:
                    lo, hi = _num(off.min()), _num(off.max())
                    text = lo if lo == hi else f"{lo} to {hi}"
                    ax.plot(edge, row, marker=mark, ms=6, color=GREY, transform=xy, clip_on=False)
                    beside = {"xytext": (8 if edge < 0.5 else -8, 0), "va": "center"}
                    _direct(ax, text, (edge, row), (edge, 0.5), MUTED, xycoords=xy, **beside)
        ax.set_xlabel(key, **AXIS)
    axes[0].set_yticks(range(len(rows)), rows)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)  # inverted: the first label on top
    axes[0].tick_params(axis="y", labelcolor=INK)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper right",
        ncols=len(scenario_labels),
        frameon=False,
        prop={"family": MONO, "size": 7.5},
        labelcolor=MUTED,
    )
    fig.savefig(out)
    return fig


def profiles(
    panels: dict[str, dict[str, np.ndarray]],
    out: Path,
    *,
    highlight=(),
    footer: str | None = None,
) -> Figure:
    """Performance profiles: one axes per panel (its key the x label), and per label one
    complementary ECDF of its runs x scenarios values, `ax.ecdf(values.ravel(),
    complementary=True)` labelled "<label> (n=<values.size>)": at each τ, the fraction of the
    label's values above τ. A dashed line marks τ = 0, and one MONO line keys the encoding (it
    reads the values as paired with rb_tuned's same season, as the report passes them).

    highlight's labels are INK, each named by a direct label at its line's end; the others are
    HAIRLINE and unnamed. The x limits fit every label but a FAR baseline, and 0: a FAR baseline's
    line stays exact inside them.
    """
    if not panels:
        raise ValueError("profiles needs at least one panel")
    fig = _ground((5.4 * len(panels), 4.8), "Performance profiles", footer, PROFILES_KEY)
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for ax, (key, arrays) in zip(axes, panels.items(), strict=True):
        _instrument(ax)
        ax.set_xlim(
            *_fit([v for label in arrays if label not in FAR for v in np.ravel(arrays[label])])
        )
        ax.set_ylim(-0.03, 1.03)
        ax.axvline(0.0, ls="--", color=MUTED, lw=0.8)
        labels = []
        for label, values in arrays.items():
            v = np.ravel(values)
            on = label in highlight
            line = ax.ecdf(
                v,
                complementary=True,
                label=f"{label} (n={v.size})",
                color=INK if on else HAIRLINE,
                lw=1.4 if on else 1.6,  # context lines heavier than the grid
                zorder=3 if on else 2,
            )
            if on:
                end = (float(line.get_xdata()[-1]), 0.0)
                labels.append(_direct(ax, label, end, ax.transLimits.transform(end), INK))
        ax.set_xlabel(key, **AXIS)
        _spread(ax, labels)
    axes[0].set_ylabel("fraction of seeds × seasons above τ", **AXIS)
    fig.savefig(out)
    return fig

"""The standard plots of a run: what a reviewer looks at to catch reward hacking (S1-13).

Each function draws on a matplotlib Figure built without pyplot (headless, no global state), saves
it to `out` as a PNG and returns it. The facecolor is white with default text colours, so a PNG
reads as a white card in both light and dark README themes. viz draws raw data and metrics'
constants only: it never imports the reward, env, Lagrangian or training code, and it draws the CIs
it's given without computing any.
"""

import csv
from pathlib import Path

from matplotlib.figure import Figure
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from greenleaf_control import metrics

CLIMATE_LABELS = {"co2": "CO₂\n[ppm]", "temp": "temp\n[°C]", "rh": "RH\n[%]"}


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


def tradeoff(points: list[dict], out: Path) -> Figure:
    """Profit against violation-hours: one errorbar per point at its (viol_hours, profit) IQMs, its
    bars reaching each CI's lo and hi. A point is {"label", "profit": (iqm, lo, hi), "viol_hours":
    (iqm, lo, hi)}, as stats.iqm_ci returns them.

    x is whatever violation-hours the caller passes: PROTOCOL §4 defines viol_hours per constraint,
    so the caller states whether it's one constraint's or the sum.
    """
    if not points:
        raise ValueError("tradeoff needs at least one point")
    fig = Figure(figsize=(7, 5), facecolor="white", layout="constrained")
    ax = fig.subplots()
    for p in points:
        (x, x_lo, x_hi), (y, y_lo, y_hi) = p["viol_hours"], p["profit"]
        xerr, yerr = [[x - x_lo], [x_hi - x]], [[y - y_lo], [y_hi - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4, label=p["label"])
    ax.set_xlabel("violation hours [h]")
    ax.set_ylabel("profit [€/m²]")
    ax.legend()
    fig.savefig(out)
    return fig

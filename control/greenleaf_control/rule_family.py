"""The rule-controller family on VALIDATION weather (#69, for #66): two deterministic analyses, one
command each:

    python -m greenleaf_control.rule_family frontier [--n N] [--seed S] [--season-length D]
        [--out DIR] [--jobs J]
    python -m greenleaf_control.rule_family horizon [--lengths L ...] [--out DIR] [--jobs J]

frontier plays the default rules, the tuned rules and N settings drawn uniformly over
tune_rb.TUNE_SPACE on the 3 VALIDATION seasons, and sets each recorded controller's validation IQM
point against them on (profit, fruit, Σ hours) at both price regimes. horizon plays rb_default,
rb_tuned and zero_controls on the VALIDATION seasons at several season lengths, skipping any season
that would reach day 365 of its 2010 weather year. Every season is played once, at sim prices, and re-priced
at agc2020 as prices.reprice does. Nothing trains and nothing is selected; VALIDATION selected the
agents' checkpoints, so the comparison is in-sample for them. Only Amsterdam/2010.csv is read.

Everything is played, checked and computed before --out is made: every played season and every
recorded agent season must re-price at sim to its recorded profit within 1e-9 EUR/m2.
"""

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from rliable.metrics import aggregate_iqm

from greenleaf_control import (
    envs,
    evaluate,
    metrics,
    prices,
    rule_based,
    scenarios,
    site_build,
    tune_rb,
)

CONTROL = Path(__file__).resolve().parents[1]
VALIDATION_SCORES = CONTROL / "results" / "analysis" / "validation" / "scores.parquet"
OUT = CONTROL / "results" / "analysis" / "rule_family"
COMMAND = "python -m greenleaf_control.rule_family"
PHRASE = "VALIDATION weather; exploratory analysis; nothing trains or is selected"
REGIMES = ("sim", "agc2020")
REGIME_TITLES = {
    "sim": "sim prices, the simulator's",
    "agc2020": "agc2020 prices, AGC 2020 tariffs",
}
# The recorded validation labels (scores.parquet) set against the frontier -> their controller ids.
LABELS = {
    "ppo #33/0221": "ppo",
    "ppo_paper #31/2d02": "ppo_paper",
    "residual_w05 #36/f2fd": "residual_w05",
    "lag_b100 #39/6d92": "lag_b100",
    "rb_default": "rb_default",
    "rb_tuned": "rb_tuned",
}
HORIZON_CONTROLLERS = ("rb_default", "rb_tuned", "zero_controls")
LENGTHS = [60, 120, 180, 240, 300]  # days
DAYS = [s["start_day"] for s in scenarios.VALIDATION]
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
TOL = 1e-9  # EUR/m2 for the re-pricing check; kg/m2 and h for a setting's qualifying bounds
NONE = "no setting qualifies"
YEAR_END = 365  # a season may end before this day of its weather year, not on it
MARKERS = {
    "rb_tuned": "s",
    "rb_default": "D",
    "residual_w05": "o",
    "lag_b100": "^",
    "ppo": "v",
    "ppo_paper": "P",
}
DAY_MARKERS = {45: "o", 59: "s", 75: "^"}
SETTING = "#B4B8B0"  # the sampled settings' grey, lighter than any controller's


def sample_settings(n: int, seed: int) -> list[dict]:
    """[default rules, tuned rules, then n draws]: draw i is the defaults overlaid with one uniform
    draw per tune_rb.TUNE_SPACE key, in its order, from one default_rng(seed). A setting's id is its
    index here."""
    defaults = rule_based.load_rb_params("default")
    rng = np.random.default_rng(seed)
    draws = [
        {
            **defaults,
            **{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in tune_rb.TUNE_SPACE.items()},
        }
        for _ in range(n)
    ]
    return [defaults, rule_based.load_rb_params("tuned"), *draws]


def skip_reason(scenario: dict, season_length: int) -> str | None:
    """Why a season of season_length days from scenario can't be played, or None: a season ending
    on day 365 or later runs off its weather year, as upstream's loader reads the weather 0.5 day
    past a season's end, so it would open the next year's file."""
    end = scenario["start_day"] + season_length
    if end >= YEAR_END:
        return (
            f"start day {scenario['start_day']} + {season_length} days = day {end}, on or past day"
            f" {YEAR_END}: with upstream's 0.5-day weather lookahead it would read the"
            f" {scenario['growth_year'] + 1} file, past its {scenario['growth_year']} weather year"
        )
    return None


def compare(frontier: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """points with best_setting, best_profit, gap, n_qualifying and note per row: among frontier's
    valid settings with yield_kg_m2 >= the row's yield_iqm and viol_hours <= its viol_hours_iqm,
    both within TOL, the one with the highest profit_<regime>; gap = profit_iqm - its profit."""
    valid = frontier[frontier["valid"].astype(bool)]
    found = []
    for point in points.itertuples(index=False):
        ok = valid[
            (valid["yield_kg_m2"] >= point.yield_iqm - TOL)
            & (valid["viol_hours"] <= point.viol_hours_iqm + TOL)
        ]
        if ok.empty:
            found.append((pd.NA, np.nan, np.nan, 0, NONE))
            continue
        best = ok.loc[ok[f"profit_{point.regime}"].idxmax()]
        profit = float(best[f"profit_{point.regime}"])
        found.append((int(best["setting"]), profit, point.profit_iqm - profit, len(ok), ""))
    columns = ["best_setting", "best_profit", "gap", "n_qualifying", "note"]
    added = pd.DataFrame(found, columns=columns, index=points.index)
    return pd.concat([points, added.astype({"best_setting": "Int64"})], axis=1)


def _play(task: tuple) -> dict:
    """One season graded at sim. (params, scenario, days): the rules with params through
    evaluate.run_rb; params None: zero_controls through envs.run_baseline, graded by metrics on the
    raw env's params and N, as evaluate._score."""
    params, scenario, days = task
    if params is not None:
        return evaluate.run_rb(params, scenario, days)[0]
    infos = envs.run_baseline("zero_controls", scenario, days)
    env = envs.make_raw_env(scenario=scenario, season_length=days, normalize_actions=False)
    graded = metrics.episode_metrics(
        *metrics.trace_from_infos(infos), np.asarray(env.unwrapped.p), season_steps=env.unwrapped.N
    )
    env.close()
    return graded


def _play_all(tasks: list[tuple], jobs: int) -> list[dict]:
    """Each task's graded season, in task order: in this process for jobs 1, else in jobs spawned
    worker processes."""
    if jobs == 1:
        return [_play(t) for t in tasks]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        return pool.map(_play, tasks, chunksize=1)


def _check_reprices(seasons: pd.DataFrame, what: str, keys: list[str]) -> None:
    """ValueError unless every season re-prices at sim to its recorded profit within TOL."""
    got = prices.reprice(seasons, metrics.PRICE_REGIMES["sim"])
    recorded = seasons["profit_eur_m2"].to_numpy(dtype=np.float64)
    off = np.flatnonzero(~(np.abs(got - recorded) <= TOL))
    if off.size:
        first = seasons.iloc[off[0]][keys].to_dict()
        raise ValueError(
            f"{off.size} {what} don't re-price at sim to their recorded profit within {TOL} EUR/m2;"
            f" the first, {first}: re-priced {got[off[0]]!r}, recorded {recorded[off[0]]!r}"
        )


def _hours(seasons: pd.DataFrame) -> pd.Series:
    """Each season's Σ hours: its CO2 + temperature + RH violation hours."""
    return seasons["viol_hours_co2"] + seasons["viol_hours_temp"] + seasons["viol_hours_rh"]


def _points(scores: pd.DataFrame) -> pd.DataFrame:
    """Each LABELS entry's point per regime: rliable's IQM over its runs x seasons of its profit
    re-priced at the regime and of its yield, and the sum of its three per-constraint hours IQMs."""
    rows = []
    for label, controller in LABELS.items():
        mine = scores[scores["label"] == label]
        hours = sum(
            float(aggregate_iqm(mine[f"viol_hours_{c}"].to_numpy())) for c in metrics.CONSTRAINTS
        )
        for regime in REGIMES:
            profit = prices.reprice(mine, metrics.PRICE_REGIMES[regime])
            rows.append(
                {
                    "controller": controller,
                    "label": label,
                    "regime": regime,
                    "profit_iqm": float(aggregate_iqm(profit)),
                    "yield_iqm": float(aggregate_iqm(mine["yield_kg_m2"].to_numpy())),
                    "viol_hours_iqm": hours,
                }
            )
    return pd.DataFrame(rows)


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(prices.GROUND)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(prices.HAIRLINE)
    ax.grid(color=prices.HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelfontfamily=prices.MONO, colors=prices.MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlabel, fontfamily=prices.MONO, fontsize=9, color=prices.MUTED)
    ax.set_ylabel(ylabel, fontfamily=prices.MONO, fontsize=9, color=prices.MUTED)


def _figure(title: str, footer: str) -> tuple[Figure, list]:
    """A two-panel figure, one panel per regime, titled and footed."""
    fig = Figure(figsize=(13, 5.5), facecolor=prices.GROUND, layout="constrained")
    fig.suptitle(title, x=0.01, ha="left", fontfamily=prices.SANS, fontsize=12, color=prices.INK)
    fig.text(0.01, 0.01, footer, fontfamily=prices.MONO, fontsize=9, color=prices.MUTED)
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    axes = fig.subplots(1, len(REGIMES))
    for ax, regime in zip(axes, REGIMES, strict=True):
        ax.set_title(
            REGIME_TITLES[regime], loc="left", fontfamily=prices.SANS, fontsize=11, color=prices.INK
        )
    return fig, list(axes)


def _legend(fig: Figure, handles: list) -> None:
    fig.legend(
        handles=handles,
        loc="outside right upper",
        frameon=False,
        prop={"family": prices.SANS, "size": 9},
        labelcolor=prices.INK,
    )


def _frontier_figure(frontier: pd.DataFrame, summary: pd.DataFrame) -> Figure:
    """frontier.png: per regime, the valid settings at (fruit, profit) as small grey points and each
    controller's IQM point as a labelled marker."""
    valid = frontier[frontier["valid"]]
    fig, axes = _figure(
        "Sampled rule settings and the recorded controllers, VALIDATION weather (Amsterdam 2010)",
        f"source: {COMMAND} frontier; {PHRASE}\nin-sample for the agents and the rules; Σ hours not"
        " shown; frontier.md has the qualifying comparison",
    )
    for ax, regime in zip(axes, REGIMES, strict=True):
        handles = [
            ax.scatter(
                valid["yield_kg_m2"],
                valid[f"profit_{regime}"],
                s=7,
                color=SETTING,
                linewidths=0,
                label=f"rule settings ({len(valid)}), mean of 3 seasons",
            )
        ]
        for point in summary[summary["regime"] == regime].itertuples(index=False):
            color = prices.LOOK[point.controller][0]
            handles.append(
                ax.scatter(
                    [point.yield_iqm],
                    [point.profit_iqm],
                    s=70,
                    marker=MARKERS[point.controller],
                    color=color,
                    edgecolors=prices.INK,
                    linewidths=0.8,
                    zorder=3,
                    label=f"{site_build.CONTROLLERS[point.controller]}, IQM",
                )
            )
        _style(ax, "fruit [kg/m²]", "profit [€/m²]")
    _legend(fig, handles)
    return fig


def _horizon_figure(horizon: pd.DataFrame) -> Figure:
    """horizon.png: per regime, each controller's profit against season length, a line per start
    day."""
    played = horizon[horizon["note"] == ""]
    fig, axes = _figure(
        "Profit against season length, VALIDATION weather (Amsterdam 2010)",
        f"source: {COMMAND} horizon; {PHRASE}",
    )
    for ax, regime in zip(axes, REGIMES, strict=True):
        handles = []
        for c in HORIZON_CONTROLLERS:
            color, style, width = prices.LOOK[c]
            for day in DAYS:
                rows = played[(played["controller"] == c) & (played["start_day"] == day)]
                rows = rows.sort_values("length")
                ax.plot(
                    rows["length"],
                    rows[f"profit_{regime}"],
                    color=color,
                    ls=style,
                    lw=width,
                    marker=DAY_MARKERS[day],
                    ms=5,
                )
            handles.append(Line2D([], [], color=color, ls=style, lw=width))
            handles[-1].set_label(site_build.CONTROLLERS[c])
        _style(ax, "season length [days]", "profit [€/m²]")
    for day in DAYS:
        handles.append(Line2D([], [], color=prices.MUTED, ls="none", marker=DAY_MARKERS[day], ms=5))
        handles[-1].set_label(f"start day {day}")
    _legend(fig, handles)
    return fig


def _write(out: Path, outputs: dict) -> None:
    """Makes out and writes each name -> DataFrame (CSV), str (text) or Figure (PNG, 200 dpi)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, thing in outputs.items():
        if isinstance(thing, pd.DataFrame):
            thing.to_csv(out / name, index=False)
        elif isinstance(thing, str):
            (out / name).write_text(thing, encoding="utf-8")
        else:
            thing.savefig(out / name, dpi=200, facecolor=prices.GROUND)
        print(f"wrote {out / name}")


def _prices_line() -> str:
    regimes = "; ".join(
        f"{r}: " + ", ".join(f"{k} {metrics.PRICE_REGIMES[r][k]:g}" for k in prices.ENERGY)
        for r in REGIMES
    )
    return (
        "- Prices (heat and elec EUR/kWh, CO₂ EUR/kg; fruit 1.6 EUR/kg in both),"
        f" metrics.PRICE_REGIMES: {regimes}."
    )


def _frontier_md(args, frontier: pd.DataFrame, summary: pd.DataFrame, n: dict) -> str:
    excluded = int((~frontier["valid"]).sum())
    lines = [
        f"Made by `uv run {COMMAND} frontier --n {args.n} --seed {args.seed} --season-length"
        f" {args.season_length}` (in control/): {PHRASE}.",
        "",
        f"{len(frontier)} rule settings, each played once at sim prices on the 3 VALIDATION seasons"
        f" (Amsterdam 2010, start days 45, 59, 75) of {args.season_length} days and re-priced at"
        f" agc2020: setting 0 is the default rules, 1 the tuned rules, and the {args.n} after them"
        f" uniform draws over tune_rb.TUNE_SPACE from default_rng({args.seed}) (setting i is"
        f" sample_settings({args.n}, {args.seed})[i]). A setting's numbers are means over its 3"
        " seasons; frontier_seasons.csv has each season.",
        "",
        "- The comparison is in-sample for the agents: VALIDATION selected their checkpoints."
        " Nothing is selected or tuned from this frontier.",
        f"- {excluded} of {len(frontier)} settings excluded from the comparison for an invalid"
        " season (a solver failure or a short season); they stay in frontier.csv with valid False.",
        "- A point is a controller's IQM (rliable's) over its recorded validation runs x seasons"
        " (validation's scores.parquet, by label; n seasons): its profit re-priced at the regime,"
        " its fruit, and Σ hours as the sum of its CO₂, temperature and RH hours IQMs (Σ"
        " per-constraint IQMs).",
        "- best: among the valid settings with fruit ≥ the point's and Σ hours ≤ the point's"
        " (within 1e-9), the one with the highest profit at the regime; qualifying counts them;"
        " gap = the point's profit − best's profit (positive: the controller earns more).",
        "- best is in-sample too: a maximum over the qualifying settings on the same 3 VALIDATION"
        " seasons, so it is optimistic for the rules, more so as qualifying grows.",
        "- gap is descriptive, not an out-of-sample estimate of what an agent earns over the rules.",
        "- Do nothing (zero_controls) and random are left out of the comparison: random is a single"
        " stochastic run, and Do nothing grows ~2 kg/m² of fruit in the recorded validation seasons"
        " (1.843, validation's summary.md), far below every agent, so it is no comparison point.",
        _prices_line(),
    ]
    header = [
        "controller",
        "n",
        "profit IQM [EUR/m²]",
        "fruit IQM [kg/m²]",
        "Σ per-constraint IQMs [h]",
        "best setting",
        "best profit [EUR/m²]",
        "gap [EUR/m²]",
        "qualifying",
    ]
    for regime in REGIMES:
        lines += ["", f"## {regime}", "", "| " + " | ".join(header) + " |"]
        lines.append("|" + "---|" * len(header))
        for row in summary[summary["regime"] == regime].itertuples(index=False):
            if pd.isna(row.best_setting):
                best = [row.note, "", ""]
            else:
                best = [str(row.best_setting), f"{row.best_profit:.3f}", f"{row.gap:.3f}"]
            cells = [
                f"{site_build.CONTROLLERS[row.controller]} ({row.label})",
                str(n[row.label]),
                f"{row.profit_iqm:.3f}",
                f"{row.yield_iqm:.3f}",
                f"{row.viol_hours_iqm:.2f}",
                *best,
                str(row.n_qualifying),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _horizon_md(args, horizon: pd.DataFrame) -> str:
    skipped = horizon[horizon["note"] != ""]
    reasons = list(dict.fromkeys(skipped["note"])) or ["none here"]
    lines = [
        f"Made by `uv run {COMMAND} horizon --lengths {' '.join(str(n) for n in args.lengths)}` (in"
        f" control/): {PHRASE}.",
        "",
        "rb_default and rb_tuned (evaluate.run_rb) and zero_controls (envs.run_baseline, graded by"
        " metrics) on each VALIDATION season (Amsterdam 2010, start days 45, 59, 75) at each season"
        " length, played once at sim prices and re-priced at agc2020. All three are deterministic:"
        " one run per cell, so no interval.",
        "",
        "- fruit is metrics' yield_kg_m2, the net change in standing fruit over the season as fresh"
        " weight; Σ hours is the season's CO₂ + temperature + RH violation hours.",
        f"- A cell whose season would end on or past day {YEAR_END} is skipped, never played:"
        f" {'; '.join(reasons)}.",
        _prices_line(),
    ]
    tables = (
        ("profit at sim [EUR/m²]", "profit_sim", ".2f"),
        ("profit at agc2020 [EUR/m²]", "profit_agc2020", ".2f"),
        ("fruit [kg/m²]", "yield_kg_m2", ".3f"),
        ("Σ hours [h]", "viol_hours", ".2f"),
    )
    header = ["controller", "start day", *(f"{n} days" for n in args.lengths)]
    for title, column, spec in tables:
        lines += ["", f"## {title}", "", "| " + " | ".join(header) + " |"]
        lines.append("|" + "---|" * len(header))
        for c in HORIZON_CONTROLLERS:
            for day in DAYS:
                cells = [site_build.CONTROLLERS[c], str(day)]
                for n in args.lengths:
                    cell = horizon[
                        (horizon["controller"] == c)
                        & (horizon["start_day"] == day)
                        & (horizon["length"] == n)
                    ].iloc[0]
                    if cell["note"]:
                        cells.append("skipped")
                    else:
                        shown = format(cell[column], spec)
                        cells.append(shown if cell["valid"] else f"{shown} (invalid)")
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _frontier(args) -> int:
    for s in scenarios.VALIDATION:
        reason = skip_reason(s, args.season_length)
        if reason is not None:
            raise ValueError(f"--season-length {args.season_length}: {reason}")
    scores = pd.read_parquet(VALIDATION_SCORES)
    scores = scores[scores["label"].isin(LABELS)]
    missing = [label for label in LABELS if label not in set(scores["label"])]
    if missing:
        raise ValueError(f"{VALIDATION_SCORES} has no rows labelled {missing}")
    _check_reprices(scores, "recorded validation season(s)", ["label", "seed", "start_day"])
    invalid = scores[~scores["valid"].astype(bool)]
    if not invalid.empty:
        which = invalid[["label", "seed", "start_day"]].to_dict("records")
        raise ValueError(f"{len(invalid)} invalid recorded validation season(s): {which}")

    settings = sample_settings(args.n, args.seed)
    cells = [(i, s) for i in range(len(settings)) for s in scenarios.VALIDATION]
    tasks = [(settings[i], s, args.season_length) for i, s in cells]
    played = _play_all(tasks, args.jobs)
    seasons = pd.DataFrame(
        [
            {"setting": i, "start_day": s["start_day"], **{m: r[m] for m in METRICS}}
            | {"valid": r["valid"]}
            for (i, s), r in zip(cells, played, strict=True)
        ]
    )
    _check_reprices(seasons, "played season(s)", ["setting", "start_day"])
    per = seasons.assign(
        profit_sim=seasons["profit_eur_m2"],
        profit_agc2020=prices.reprice(seasons, metrics.PRICE_REGIMES["agc2020"]),
        viol_hours=_hours(seasons),
    ).groupby("setting", sort=True)
    means = ["profit_sim", "profit_agc2020", "yield_kg_m2", "viol_hours"]
    frontier = per[[*means, "heat_cost", "elec_cost", "co2_cost"]].mean()
    frontier.insert(0, "valid", per["valid"].all())
    frontier = frontier.reset_index()
    summary = compare(frontier, _points(scores))
    n = scores["label"].value_counts().to_dict()
    _write(
        args.out,
        {
            "frontier_seasons.csv": seasons,
            "frontier.csv": frontier,
            "frontier_summary.csv": summary,
            "frontier.md": _frontier_md(args, frontier, summary, n),
            "frontier.png": _frontier_figure(frontier, summary),
        },
    )
    return 0


def _horizon(args) -> int:
    cells = [
        (c, s, n) for c in HORIZON_CONTROLLERS for s in scenarios.VALIDATION for n in args.lengths
    ]
    reasons = [skip_reason(s, n) for _, s, n in cells]
    tasks = [
        (None if c == "zero_controls" else rule_based.load_rb_params(c.removeprefix("rb_")), s, n)
        for (c, s, n), reason in zip(cells, reasons, strict=True)
        if reason is None
    ]
    played = iter(_play_all(tasks, args.jobs))
    found = []
    for (c, s, n), reason in zip(cells, reasons, strict=True):
        row = {"controller": c, "start_day": s["start_day"], "length": n}
        if reason is None:  # a played cell: its graded season, in task order
            r = next(played)
            row |= {m: r[m] for m in METRICS} | {"valid": r["valid"], "note": ""}
        else:  # skipped: NaN numbers, the reason in note
            row |= {m: np.nan for m in METRICS} | {"valid": None, "note": reason}
        found.append(row)
    rows = pd.DataFrame(found)
    _check_reprices(
        rows[rows["note"] == ""], "played season(s)", ["controller", "start_day", "length"]
    )
    horizon = pd.DataFrame(
        {
            "controller": rows["controller"],
            "start_day": rows["start_day"],
            "length": rows["length"],
            "profit_sim": rows["profit_eur_m2"],
            "profit_agc2020": prices.reprice(rows, metrics.PRICE_REGIMES["agc2020"]),
            "yield_kg_m2": rows["yield_kg_m2"],
            "viol_hours": _hours(rows),
            "valid": rows["valid"],
            "note": rows["note"],
        }
    )
    _write(
        args.out,
        {
            "horizon.csv": horizon,
            "horizon.md": _horizon_md(args, horizon),
            "horizon.png": _horizon_figure(horizon),
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog=COMMAND)
    sub = p.add_subparsers(dest="command", required=True)
    frontier = sub.add_parser("frontier", help="the rule family against the recorded controllers")
    frontier.add_argument("--n", type=int, default=1000, help="sampled settings (default 1000)")
    frontier.add_argument("--seed", type=int, default=0, help="the draws' seed (default 0)")
    frontier.add_argument(
        "--season-length", type=int, default=evaluate.SEASON_LENGTH, help="days (default 60)"
    )
    horizon = sub.add_parser("horizon", help="three controllers at several season lengths")
    horizon.add_argument(
        "--lengths", type=int, nargs="+", default=LENGTHS, help="days (default 60 ... 300)"
    )
    for command in (frontier, horizon):
        command.add_argument(
            "--out", type=Path, default=OUT, help="default control/results/analysis/rule_family"
        )
        command.add_argument("--jobs", type=int, default=1, help="worker processes (default 1)")
    args = p.parse_args(argv)
    return _frontier(args) if args.command == "frontier" else _horizon(args)


if __name__ == "__main__":
    raise SystemExit(main())

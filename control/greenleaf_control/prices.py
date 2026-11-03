"""The price-regime study (#68, for #66): every recorded season re-priced, one command:

    python -m greenleaf_control.prices [--out DIR]

Profit is linear in each price, so re-pricing a season's recorded yield and costs is exact: profit =
yield x fruit - heat_cost x heat / 0.09 - elec_cost x elec / 0.30 - co2_cost x co2 / 0.30, the
recorded costs being at metrics.PRICES. Nothing is simulated: no season is played, no weather file
is opened and the 2012 seal is not reopened (the test seasons are the final's recorded ones).
Exploratory and post hoc, and every output says so.

Writes <out>/{regimes, sweep, crossovers}.csv, summary.md and sweep.png, only after every recorded
season, re-priced at sim, matches its recorded profit within 1e-9 EUR/m2. It imports no env,
evaluate, report or selection: the matrix order is report._matrix's, written out here.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from rliable.metrics import aggregate_iqm

from greenleaf_control import ledger, metrics, scenarios, site_build, stats

CONTROL = Path(__file__).resolve().parents[1]
VALIDATION_SCORES = CONTROL / "results" / "analysis" / "validation" / "scores.parquet"
FINAL_RESULTS = CONTROL / "results" / "final"
OUT = CONTROL / "results" / "analysis" / "prices"
COMMAND = "python -m greenleaf_control.prices"
PHRASE = "exploratory, re-priced recorded seasons"
SCORED_ONCE = "2012 test weather, scored once"  # report.SCORED_ONCE's words (report imports envs)

# The numbers #67 puts in metrics.PRICE_REGIMES and PROTOCOL v3, defined here so this analysis ships
# independently of it; a follow-up swaps in metrics.PRICE_REGIMES. sim is the simulator's (every
# recorded season's). agc2020: heat 0.03 €/kWh and CO₂ 0.08 €/kg (its ≤ 12 kg/m² tier) from the 2nd
# Autonomous Greenhouse Challenge's 2020 net-profit scoring (Hemming et al. 2020, Sensors
# 20(22):6430); electricity 0.06 €/kWh, the unweighted mean of its 0.08 on-peak (07:00-23:00) and
# 0.04 off-peak tariffs, flat, not time-of-use; fruit 1.6 €/kg and dmfm 0.065 are the simulator's.
REGIMES = {
    "sim": dict(metrics.PRICES),
    "agc2020": {"fruit": 1.6, "dmfm": 0.065, "heat": 0.03, "elec": 0.06, "co2": 0.08},
}
ENERGY = ("heat", "elec", "co2")
# Eurostat nrg_pc_205, non-household electricity, 500-1999 MWh band, excluding taxes [EUR/kWh]
# (retrieved 2026-11-03): an annotation beside the elec sweep's crossovers in summary.md, no input.
REFERENCE_ELEC = {
    "NL 2020-S1": 0.0739,
    "NL 2022-S2": 0.151,
    "EU27 2020-S1": 0.0818,
    "EU27 2022-S2": 0.1987,
}
GAINS = ("rb_tuned", "rb_default")  # deterministic, one run each: every gain pairs with its season
PAIRS = (
    ("ppo", "rb_default"),
    ("ppo", "residual_w05"),
    ("zero_controls", "rb_default"),
    ("rb_tuned", "rb_default"),
    ("residual_w05", "rb_tuned"),
)
COLUMNS = (
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
    "run",
    "source",
)
# viz's Instrument palette and fonts (#47), held here because viz imports tensorboard.
GROUND, INK, MUTED, HAIRLINE, GREY = "#F2F3F0", "#111311", "#6A6F68", "#D5D8D2", "#8A8F87"
SANS = ["Helvetica Neue", "Arial", "DejaVu Sans"]
MONO = ["Menlo", "DejaVu Sans Mono"]
# sweep.png's line per controller: (colour, line style, width), in site_build.CONTROLLERS order.
LOOK = {
    "rb_tuned": (INK, "--", 1.6),
    "rb_default": (GREY, "--", 1.6),
    "residual_w05": (INK, "-", 2.4),
    "lag_b100": (INK, ":", 2.0),
    "ppo": (MUTED, "-", 1.6),
    "ppo_paper": (MUTED, "-.", 1.4),
    "zero_controls": (GREY, ":", 1.6),
}


def _along(lam: float) -> dict:
    """agc2020's prices with heat, elec and co2 moved lam of the way to sim's."""
    agc, sim = REGIMES["agc2020"], REGIMES["sim"]
    return {**agc, **{k: agc[k] + lam * (sim[k] - agc[k]) for k in ENERGY}}


# (x, prices) along each sweep; fruit and dmfm stay fixed.
SWEEPS = {
    "lambda": [(float(x), _along(float(x))) for x in np.linspace(0, 1, 41)],
    "elec": [(float(x), {**REGIMES["agc2020"], "elec": float(x)}) for x in np.arange(2, 41) / 100],
}


def reprice(seasons: pd.DataFrame, prices: dict) -> np.ndarray:
    """Each season's profit [EUR/m2] at prices: yield x fruit minus each recorded cost scaled by its
    price over metrics.PRICES'. A dmfm other than metrics.PRICES' is refused: the recorded yield was
    converted from dry matter at it."""
    sim = metrics.PRICES
    if prices["dmfm"] != sim["dmfm"]:
        raise ValueError(
            f"dmfm {prices['dmfm']} is not {sim['dmfm']}: the recorded yield is fresh weight converted"
            " from dry matter at that ratio, so another needs the dry matter, which isn't recorded"
        )
    profit = seasons["yield_kg_m2"].to_numpy(dtype=np.float64) * prices["fruit"]
    for k in ENERGY:
        profit = profit - seasons[f"{k}_cost"].to_numpy(dtype=np.float64) * (prices[k] / sim[k])
    return profit


def load_seasons(*, ledger_path: Path = ledger.LEDGER) -> pd.DataFrame:
    """One row per recorded season with COLUMNS, controller = the final's arm id: validation's
    scores.parquet rows of the labels site_build._validation_arms maps, and every row of the final's
    seasons.parquet (the three 2012 splits, random included). Reads; checks nothing."""
    existing = ledger.rows(path=ledger_path)
    finals = [r for r in existing if r["kind"] == "final-result"]
    if not finals:
        raise ValueError(
            f"{ledger_path} has no final-result row: no recorded 2012 seasons to re-price, and no"
            " final to map validation's labels to its arms"
        )
    [final] = finals
    registered = ledger._find(existing, final["final-registered"], "final-registered")
    arms = site_build._validation_arms(registered, existing)
    validation = pd.read_parquet(VALIDATION_SCORES)
    validation = validation[validation["label"].isin(arms)]
    test_path = FINAL_RESULTS / registered["row_id"] / "seasons.parquet"
    test = pd.read_parquet(test_path)
    root = CONTROL.parent  # source: the file's repo path
    frames = [
        validation.assign(
            controller=validation["label"].map(arms),
            source=VALIDATION_SCORES.relative_to(root).as_posix(),
        ),
        test.assign(controller=test["arm"], source=test_path.relative_to(root).as_posix()),
    ]
    return pd.concat(frames, ignore_index=True)[list(COLUMNS)].astype({"seed": "float64"})


def _sorted(rows: pd.DataFrame) -> pd.DataFrame:
    """One controller's rows of one split in runs (by seed, a baseline's one run -1) x scenarios
    (scenarios.SPLITS order) order, as report._matrix: the bootstrap depends on row order."""
    split = scenarios.SPLITS[rows["split"].iloc[0]]
    column = {(s["location"], s["growth_year"], s["start_day"]): i for i, s in enumerate(split)}
    keys = zip(rows["location"], rows["growth_year"], rows["start_day"], strict=True)
    at = [column[k] for k in keys]
    seed = rows["seed"].fillna(-1).to_numpy(dtype=np.int64)
    return rows.iloc[np.lexsort((at, seed))]


def _profits(runs: dict[str, pd.DataFrame], prices: dict, width: int) -> dict[str, np.ndarray]:
    """Each controller's runs x scenarios profit matrix at prices."""
    return {c: reprice(rows, prices).reshape(-1, width) for c, rows in runs.items()}


def _crossings(x: np.ndarray, d: np.ndarray, **ids) -> list[dict]:
    """A row per sign change of d along x, x linearly interpolated between the two grid points;
    one row with x NaN and note "none in range" if d never changes sign."""
    found = [
        {
            **ids,
            "x": x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1]),
            "note": "controller ahead below x" if d[i] > 0 else "controller ahead above x",
        }
        for i in range(len(x) - 1)
        if d[i] * d[i + 1] < 0
    ]
    return found or [{**ids, "x": np.nan, "note": "none in range"}]


def tables(seasons: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{"regimes", "sweep", "crossovers"} for each split in seasons (checked, random dropped):
    profit IQM [CI] and paired gains over GAINS [CI] per regime with rank, Σ hours and yield IQMs;
    each sweep point's profit IQM; and each PAIR's sign changes along each sweep."""
    regimes, sweep, crossovers = [], [], []
    present = set(seasons["split"])
    for split in (s for s in scenarios.SPLITS if s in present):
        on = seasons[seasons["split"] == split]
        width = len(scenarios.SPLITS[split])
        runs = {c: _sorted(on[on["controller"] == c]) for c in dict.fromkeys(on["controller"])}
        for b in GAINS:
            if len(runs[b]) != width:
                raise ValueError(f"{split}: {b} has {len(runs[b]) // width} runs; a gain needs 1")
        for regime, prices in REGIMES.items():
            profit = _profits(runs, prices, width)
            rows = []
            for c, m in profit.items():
                r = runs[c]
                hours = r["viol_hours_co2"] + r["viol_hours_temp"] + r["viol_hours_rh"]
                row = {"split": split, "regime": regime, "rank": 0, "controller": c}
                row["n_runs"] = len(m)
                for name, x in (("profit", m), *((f"gain_{b}", m - profit[b]) for b in GAINS)):
                    ci = stats.iqm_ci(x)
                    row |= {f"{name}_{s}": v for s, v in zip(("iqm", "lo", "hi"), ci, strict=True)}
                row["viol_hours_iqm"] = float(aggregate_iqm(hours.to_numpy().reshape(-1, width)))
                row["yield_iqm"] = float(
                    aggregate_iqm(r["yield_kg_m2"].to_numpy().reshape(-1, width))
                )
                rows.append(row)
            for rank, row in enumerate(sorted(rows, key=lambda one: -one["profit_iqm"]), start=1):
                row["rank"] = rank
            regimes += rows
        for name, points in SWEEPS.items():
            curves = [_profits(runs, prices, width) for _, prices in points]
            for (x, prices), profit in zip(points, curves, strict=True):
                sweep += [
                    {
                        "split": split,
                        "sweep": name,
                        "x": x,
                        **{k: prices[k] for k in ENERGY},
                        "controller": c,
                        "profit_iqm": float(aggregate_iqm(m)),
                    }
                    for c, m in profit.items()
                ]
            xs = np.array([x for x, _ in points])
            for a, b in PAIRS:
                if b in GAINS:  # paired with b's same season
                    d = [float(aggregate_iqm(p[a] - p[b])) for p in curves]
                else:  # different runs: the difference of the point IQMs
                    d = [float(aggregate_iqm(p[a])) - float(aggregate_iqm(p[b])) for p in curves]
                ids = {"split": split, "sweep": name, "controller": a, "versus": b}
                crossovers += _crossings(xs, np.array(d), **ids)
    return {
        "regimes": pd.DataFrame(regimes),
        "sweep": pd.DataFrame(sweep),
        "crossovers": pd.DataFrame(crossovers),
    }


def _ci(row, prefix: str) -> str:
    return f"{row[f'{prefix}_iqm']:.3f} [{row[f'{prefix}_lo']:.3f}, {row[f'{prefix}_hi']:.3f}]"


def _summary(done: dict[str, pd.DataFrame], sources: list[str]) -> str:
    """summary.md: what it is on line 1, the sources, prices and legend, a table per split, then
    the crossovers with Eurostat's electricity prices beside the elec sweep's."""
    regimes, crossovers = done["regimes"], done["crossovers"]
    prices = {
        "sim": "the simulator's (PROTOCOL §1), at which every season was recorded",
        "agc2020": "heat 0.03 €/kWh and CO₂ 0.08 €/kg (its ≤ 12 kg/m² tier) from the 2nd Autonomous"
        " Greenhouse Challenge's 2020 net-profit scoring (Hemming et al. 2020, Sensors"
        " 20(22):6430); electricity 0.06 €/kWh, the unweighted mean of its 0.08 on-peak"
        " (07-23 h) and 0.04 off-peak tariffs, flat, not time-of-use; fruit 1.6 €/kg and dmfm"
        " 0.065 are the simulator's",
    }
    measures = ("profit", *(f"gain_{b}" for b in GAINS))
    header = ["controller", "n", "Σ hours IQM", "yield IQM"]
    header += [f"{r} {h}" for r in REGIMES for h in ("rank", *measures)]
    lines = [
        f"Made by `uv run {COMMAND}` (in control/): {PHRASE}, post hoc. No season is played and the"
        " 2012 seal is not reopened; every number is in regimes.csv, sweep.csv and crossovers.csv at"
        " full precision.",
        "",
        f"Sources: {'; '.join(sources)}. The 2012 seasons are the final's ({SCORED_ONCE}).",
        "",
        "Prices (fruit EUR/kg, heat and elec EUR/kWh, CO₂ EUR/kg; dry matter 0.065 throughout):",
        *(
            f"- {r}: " + ", ".join(f"{k} {p[k]:g}" for k in ("fruit", *ENERGY)) + f"; {prices[r]}."
            for r, p in REGIMES.items()
        ),
        "",
        "- IQM [95% stratified-bootstrap CI] over each controller's runs x seasons; profit and gains"
        " in EUR/m², hours in h, yield in kg/m². rank 1 is the highest profit IQM in its split and"
        " regime.",
        "- A gain pairs each season with rb_tuned's or rb_default's same season. rb_tuned,"
        " rb_default and zero_controls are deterministic, one run each, so a baseline-vs-baseline"
        " gain has a zero-width CI: one run, not certainty.",
        "- Σ hours (each season's CO₂ + temp + RH violation hours) and yield don't move with price.",
        "- random is left out: noise, not a controller (3 invalid Abbotsford seasons).",
    ]
    for split in dict.fromkeys(regimes["split"]):
        on = regimes[regimes["split"] == split]
        by = {r: on[on["regime"] == r].set_index("controller") for r in REGIMES}
        note = SCORED_ONCE if split in scenarios.TEST else "checkpoints were selected on these"
        lines += ["", f"## {split} ({note})", "", "| " + " | ".join(header) + " |"]
        lines.append("|" + "---|" * len(header))
        for c in by["sim"].sort_values("rank").index:
            sim = by["sim"].loc[c]
            cells = [
                c,
                str(sim["n_runs"]),
                f"{sim['viol_hours_iqm']:.2f}",
                f"{sim['yield_iqm']:.3f}",
            ]
            for r in REGIMES:
                cells += [str(by[r].loc[c, "rank"]), *(_ci(by[r].loc[c], m) for m in measures)]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Crossovers",
        "",
        "Where a pair's difference changes sign along a sweep, linearly interpolated between grid"
        " points: the paired-gain IQM against rb_default or rb_tuned, the difference of profit IQMs"
        " for ppo vs residual_w05 (different runs). λ moves heat, elec and CO₂ together from agc2020"
        " (λ = 0) to sim (λ = 1) in 41 points; the elec sweep moves electricity alone, 0.02 to 0.40"
        " EUR/kWh, with heat and CO₂ at agc2020's. Fruit stays 1.6 EUR/kg.",
        "",
        "| split | controller vs versus | λ | elec [EUR/kWh] |",
        "|---|---|---|---|",
    ]
    for (split, a, b), rows in crossovers.groupby(["split", "controller", "versus"], sort=False):
        cells = [split, f"{a} vs {b}"]
        for name in SWEEPS:
            found = rows[rows["sweep"] == name]
            cells.append(
                ", ".join(
                    "none in range" if pd.isna(x) else f"{x:.3f} ({note})"
                    for x, note in zip(found["x"], found["note"], strict=True)
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Reference non-household electricity prices beside the elec column (Eurostat nrg_pc_205,"
        " 500-1999 MWh band, excluding taxes, retrieved 2026-11-03; an annotation, no input): "
        + ", ".join(f"{k} {v} EUR/kWh" for k, v in REFERENCE_ELEC.items())
        + ".",
    ]
    return "\n".join(lines) + "\n"


def _figure(sweep: pd.DataFrame) -> Figure:
    """sweep.png: each controller's profit IQM along lambda, one panel per test split."""
    fig = Figure(figsize=(13, 5), facecolor=GROUND, layout="constrained")
    fig.suptitle(
        "Profit IQM as energy prices move from agc2020 (λ = 0) to the simulator's (λ = 1)",
        x=0.01,
        ha="left",
        fontfamily=SANS,
        fontsize=12,
        color=INK,
    )
    footer = f"source: {COMMAND}; {PHRASE}; {SCORED_ONCE}"
    fig.text(0.01, 0.01, footer, fontfamily=MONO, fontsize=9, color=MUTED, va="bottom")
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    axes = fig.subplots(1, len(scenarios.TEST), sharey=True)
    for ax, split in zip(axes, scenarios.TEST, strict=True):
        on = sweep[(sweep["split"] == split) & (sweep["sweep"] == "lambda")]
        for c, name in site_build.CONTROLLERS.items():
            color, style, width = LOOK[c]
            rows = on[on["controller"] == c].sort_values("x")
            ax.plot(rows["x"], rows["profit_iqm"], color=color, ls=style, lw=width, label=name)
        place = scenarios.SPLITS[split][0]
        title = f"{place['location']} {place['growth_year']}"
        ax.set_title(title, loc="left", fontfamily=SANS, fontsize=11, color=INK)
        ax.set_facecolor(GROUND)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(HAIRLINE)
        ax.grid(color=HAIRLINE, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelfontfamily=MONO, colors=MUTED, labelsize=9, length=0)
        ax.set_xlabel("λ, energy prices", fontfamily=MONO, fontsize=9, color=MUTED)
    axes[0].set_ylabel("profit IQM [€/m²]", fontfamily=MONO, fontsize=9, color=MUTED)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside right upper",
        frameon=False,
        prop={"family": SANS, "size": 9},
        labelcolor=INK,
    )
    return fig


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    p = argparse.ArgumentParser(prog=COMMAND)
    p.add_argument("--out", type=Path, default=OUT, help="default control/results/analysis/prices")
    args = p.parse_args(argv)
    # Everything is loaded, checked and computed before a dir is made or a file written.
    seasons = load_seasons(ledger_path=ledger_path)
    got = reprice(seasons, REGIMES["sim"])
    recorded = seasons["profit_eur_m2"].to_numpy(dtype=np.float64)
    off = np.flatnonzero(~(np.abs(got - recorded) <= 1e-9))
    if off.size:
        r = seasons.iloc[off[0]]
        raise ValueError(
            f"{off.size} recorded season(s) don't re-price at sim to their recorded profit within"
            f" 1e-9 EUR/m2; the first: {r['split']} {r['controller']} run {r['run']} seed"
            f" {r['seed']} start day {r['start_day']}: re-priced {got[off[0]]!r}, recorded"
            f" {recorded[off[0]]!r} ({r['source']})"
        )
    kept = seasons[seasons["controller"] != "random"]
    invalid = kept[~kept["valid"].astype(bool)]
    if not invalid.empty:
        which = invalid[["split", "controller", "seed", "start_day"]].to_dict("records")
        raise ValueError(f"{len(invalid)} invalid season(s) of a kept controller: {which}")
    done = tables(kept)
    text = _summary(done, list(dict.fromkeys(seasons["source"])))
    fig = _figure(done["sweep"])

    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in done.items():
        frame.to_csv(args.out / f"{name}.csv", index=False)
        print(f"wrote {args.out / f'{name}.csv'} ({len(frame)} rows)")
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")
    fig.savefig(args.out / "sweep.png", dpi=200, facecolor=GROUND)
    print(f"wrote {args.out / 'sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

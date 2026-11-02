"""Per-split stats tables from the ledger and the runs (A1a, PROTOCOL §6), one command:

    python -m greenleaf_control.report --split <validation | a TEST split> [--out DIR]

rebuilds every arm's per-season scores from the ledger, the runs' selected.json files and
evaluate's baseline parquets, checks them against the ledger's result rows, and only then writes
<out>/<split>/{scores.parquet, summary.csv, summary.md} and the figures (A1b) {per_scenario,
profiles, tradeoff}.png, replacing any previous version (a derived view; the ledger is the record).
Every figure is derived from scores.parquet and summary.csv. It plays no season, trains nothing
and writes no ledger row.

A 2012 TEST split (A1c) is sealed (PROTOCOL §8) until the ledger holds the final-result row; then
its scores come only from the final's seasons.parquet, and are checked against that row's
per_split before anything is written.

Every statistic is stats' on a label's runs (by seed) x scenarios (split order) array, the order
selection scored in. rb_tuned is 1 deterministic run x S seasons, so its like-for-like bar is its
row tiled to the arm's n runs before the IQM; gains are paired with its same season.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from greenleaf_control import envs, evaluate, ledger, metrics, scenarios, selection, stats, viz

ANALYSIS = Path(__file__).resolve().parents[1] / "results" / "analysis"
BASELINE = "rb_tuned"
METRICS = (
    "profit_eur_m2",
    "viol_hours_total",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "yield_kg_m2",
    "paper_return",
)
VIOL = tuple(f"viol_hours_{c}" for c in metrics.CONSTRAINTS)
# PROTOCOL §6: deterministic arms (the RBs, zero_controls) need one run.
DETERMINISTIC = ("zero_controls", "rb_default", "rb_tuned")
# A decision row calls a whole registration, but its note can call one arm otherwise; the figures
# draw (decision row id, arm) -> that arm's own call, as media.TRADEOFF does. E4 (#34) has one
# decision row, decision-b159ac9a, "adopted", and its note drops residual_w4.
NOTE_CALLS = {("decision-b159ac9a", "residual_w4"): "dropped"}
# A TEST split's footer, figure titles and summary.md line 1 say so (PROTOCOL §8)
SCORED_ONCE = "2012 test weather, scored once"
IDS = ("label", "arm", "issue", "registered", "result", "decision", "protocol", "run", "seed")
COLUMNS = (
    *IDS,
    "crashed",
    "split",
    "location",
    "growth_year",
    "start_day",
    *ledger.RESULT_KEYS,
    "valid",
    "viol_hours_total",
)
HEADER = (
    "label",
    "decision",
    "PROTOCOL",
    "n",
    "crashed",
    "invalid",
    "profit IQM [95% CI]",
    "rb_tuned like-for-like",
    "gain [CI]",
    "P(profit > rb_tuned)",
    "total violation hours IQM [CI]",
    "Σ per-constraint IQMs (rb_tuned like-for-like)",
    "hours saved [CI]",
    "P(fewer hours)",
    "yield IQM [CI]",
    "flag",
)


def _key(s: dict) -> tuple:
    return (s["location"], s["growth_year"], s["start_day"])


def _record(season: dict, scenario: dict, split: str, **ids) -> dict:
    """One score row: ids (null unless given), the scenario, the season's numbers and its total
    violation hours."""
    return {
        **dict.fromkeys(IDS),
        "crashed": None,
        **ids,
        "split": split,
        **{k: scenario[k] for k in ("location", "growth_year", "start_day")},
        **{k: season[k] for k in (*ledger.RESULT_KEYS, "valid")},
        "viol_hours_total": season["viol_hours_co2"]
        + season["viol_hours_temp"]
        + season["viol_hours_rh"],
    }


def _baselines(split: str) -> dict[str, list[dict]]:
    """{baseline: its score rows in split order}, envs.BASELINES order, from every evaluate parquet
    in selection.RESULTS (rows of other splits ignored)."""
    seen = {}
    for path in sorted(selection.RESULTS.glob("*.parquet")):
        frame = pd.read_parquet(path)
        for row in frame[frame["split"] == split].to_dict("records"):
            controller = row["controller"]
            if controller not in envs.BASELINES:
                raise ValueError(
                    f"{path}: controller {controller!r} is not one of {envs.BASELINES}; a learning"
                    " run's scores come from its selected.json, never a parquet"
                )
            if (controller, _key(row)) in seen:
                raise ValueError(
                    f"{path}: {controller} on {_key(row)} is already in another parquet in"
                    f" {selection.RESULTS}; one baseline scores each season once"
                )
            seen[controller, _key(row)] = row
    order = scenarios.SPLITS[split]
    out = {}
    for b in envs.BASELINES:
        have = {key for c, key in seen if c == b}
        if have and have != {_key(s) for s in order}:
            raise ValueError(f"{b}'s {split} parquet rows are {sorted(have)}, not the split's")
        if have:
            out[b] = [_record(seen[b, _key(s)], s, split, label=b, arm=b) for s in order]
    return out


def _ids(reg: dict, existing: list[dict]) -> dict | None:
    """The ids reg's arms carry: its issue, registered, result, decision and protocol; None while
    it has no result row. Two result rows, or two decision rows on its result, are refused."""
    reg_id = reg["row_id"]
    results = [r for r in existing if r["kind"] == "result" and r["registered"] == reg_id]
    if not results:
        return None
    if len(results) > 1:
        raise ValueError(f"{reg_id} has {len(results)} result rows; the table shows one")
    result = results[0]["row_id"]
    decisions = [r for r in existing if r["kind"] == "decision" and r["result"] == result]
    if len(decisions) > 1:
        raise ValueError(
            f"result {result} has {len(decisions)} decision rows"
            f" ({', '.join(d['row_id'] for d in decisions)}); the table shows one"
        )
    return {
        "issue": reg["issue"],
        "registered": reg_id,
        "result": result,
        "decision": f"{decisions[0]['decision']} {decisions[0]['row_id']}" if decisions else None,
        "protocol": reg["fingerprints"]["docs/PROTOCOL.md"][:8],
    }


def _final(split: str, existing: list[dict]) -> tuple[dict, dict]:
    """(the final-result row, the final-registered row it names). A TEST split is sealed (PROTOCOL
    §8) until the ledger holds the final-result: a ValueError naming the split says so."""
    finals = [r for r in existing if r["kind"] == "final-result"]
    if not finals:
        raise ValueError(
            f"split {split!r} is 2012 test weather, sealed (PROTOCOL §8): the report reads it only"
            " from the final's files, once the ledger holds the final-result row, and it has none"
        )
    return finals[0], ledger._find(existing, finals[0]["final-registered"], "final-registered")


def _ordered(mine: list[dict], split: str, who: str, seasons: Path) -> list[dict]:
    """mine, one run's rows of the final's seasons on split, in scenarios.SPLITS[split] order;
    refused unless it holds each of the split's scenarios once."""
    order = [_key(s) for s in scenarios.SPLITS[split]]
    if sorted(_key(r) for r in mine) != sorted(order):
        raise ValueError(
            f"{seasons}: {who} has {len(mine)} rows on {split}, not its {len(order)} scenarios"
            " once each"
        )
    at = {_key(r): r for r in mine}
    return [at[k] for k in order]


def _final_scores(split: str, existing: list[dict]) -> list[dict]:
    """A TEST split's score rows, from the final's seasons.parquet only: the final's baselines in
    envs.BASELINES order, then its learning arms by registration (ledger order) and arm
    (registration order), runs by seed. label = arm = the name; a learning arm's ids are its
    registration's, a baseline's null; crashed null (the final scores every season)."""
    final, registered = _final(split, existing)
    seasons = evaluate.FINAL_RESULTS / registered["row_id"] / "seasons.parquet"
    if not seasons.is_file():
        raise ValueError(
            f"no {seasons}: a test split's scores come only from the final's files"
            f" ({final['row_id']})"
        )
    frame = pd.read_parquet(seasons)
    rows = frame[frame["split"] == split].to_dict("records")
    order = scenarios.SPLITS[split]
    out = []
    for b in envs.BASELINES:
        if b in registered["baselines"]:
            mine = _ordered([r for r in rows if r["arm"] == b], split, b, seasons)
            out += [
                _record(season, s, split, label=b, arm=b)
                for season, s in zip(mine, order, strict=True)
            ]
    arms = ledger.final_arms(registered, existing)
    runs = {r["row_id"]: r for r in existing if r["kind"] == "run"}
    for reg in existing:
        if reg["kind"] != "registered":
            continue
        for name in (a["name"] for a in reg["arms"]):
            mine = [runs[r] for r in arms.get(name, {}) if runs[r]["registered"] == reg["row_id"]]
            if not mine:
                continue
            ids = _ids(reg, existing)
            for run in sorted(mine, key=lambda r: r["seed"]):
                who = f"{name} run {run['row_id']} (seed {run['seed']})"
                got = _ordered([r for r in rows if r["run"] == run["row_id"]], split, who, seasons)
                out += [
                    _record(
                        season,
                        s,
                        split,
                        label=name,
                        arm=name,
                        **ids,
                        run=run["row_id"],
                        seed=run["seed"],
                    )
                    for season, s in zip(got, order, strict=True)
                ]
    return out


def _frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=list(COLUMNS))
    return frame.astype({"issue": "Int64", "seed": "Int64", "crashed": "Int64"})


def load_scores(split: str, *, ledger_path: Path = ledger.LEDGER) -> pd.DataFrame:
    """One row per label x run x scenario: baselines in envs.BASELINES order, then each registration
    on [split] with a result row, in ledger order, arms in registration order, runs by seed,
    scenarios in scenarios.SPLITS[split] order. Plays nothing: a run without selected.json is
    refused. A TEST split reads the final's seasons.parquet (_final_scores), once the ledger holds
    the final-result row."""
    if split in scenarios.TEST:
        return _frame(_final_scores(split, ledger.rows(path=ledger_path)))
    if split != "validation":
        raise ValueError(
            f"split {split!r}: the report reads validation, and the TEST splits"
            f" ({', '.join(scenarios.TEST)}) from the final's files"
        )
    baselines = _baselines(split)
    existing = ledger.rows(path=ledger_path)
    arms, folded = [], None
    for reg in existing:
        if reg["kind"] != "registered" or reg["scenarios"] != [split]:
            continue
        reg_id = reg["row_id"]
        ids = _ids(reg, existing)
        if ids is None:
            print(f"skipped {reg_id}: no result row yet")
            continue
        result = ids["result"]
        _, runs = selection._registration(reg_id, ledger_path)
        ends = ledger._run_ends(existing, reg_id)
        for arm in reg["arms"]:
            name, method = arm["name"], arm["method"]
            if method == BASELINE:  # E1: its numbers are the rb_tuned baseline row's
                if folded is not None:
                    raise ValueError(
                        f"{reg_id}: a second {BASELINE} arm; {folded}'s already annotates the"
                        f" {BASELINE} baseline row"
                    )
                if not runs[name]:
                    raise ValueError(
                        f"{reg_id} (result {result}): its {BASELINE} arm {name} has no run that"
                        " didn't crash, so it can't annotate the baseline row"
                    )
                if BASELINE not in baselines:
                    raise ValueError(
                        f"{reg_id}: no {BASELINE} parquet in {selection.RESULTS} to annotate;"
                        f" run evaluate --controller {BASELINE} --split {split}"
                    )
                folded = reg_id
                for row in baselines[BASELINE]:
                    row.update(arm=name, **{k: v for k, v in ids.items() if k != "issue"})
                continue
            if not runs[name]:
                raise ValueError(f"{reg_id}: arm {name} has no run that didn't crash to report")
            crashed = ledger._arm_tally(ends, name)["crashed"]
            label = f"{name} #{reg['issue']}/{reg_id.removeprefix('registered-')[:4]}"
            for run in runs[name]:
                read = selection._read(method, Path(run["run_dir"]))
                if read is None:
                    raise ValueError(
                        f"run {run['row_id']} ({run['run_dir']}) has no selected.json; run"
                        f" selection first (python -m greenleaf_control.selection --registered"
                        f" {reg_id})"
                    )
                seasons = read["candidates"][read["checkpoint"]]["per_scenario"]
                arms += [
                    _record(
                        season,
                        s,
                        split,
                        label=label,
                        arm=name,
                        **ids,
                        run=run["row_id"],
                        seed=run["seed"],
                        crashed=crashed,
                    )
                    for season, s in zip(seasons, scenarios.SPLITS[split], strict=True)
                ]
    return _frame([row for rows in baselines.values() for row in rows] + arms)


def _matrix(rows: pd.DataFrame, key: str) -> np.ndarray:
    """key's runs (by seed) x scenarios (split order) array of one label's or arm's rows, whatever
    their order in the frame: the bootstrap depends on row order."""
    split = scenarios.SPLITS[rows["split"].iloc[0]]
    column = {_key(s): i for i, s in enumerate(split)}
    keys = zip(rows["location"], rows["growth_year"], rows["start_day"], strict=True)
    at = [column[k] for k in keys]
    seed = rows["seed"].fillna(-1).to_numpy(dtype=np.int64)  # a baseline's one run has none
    order = np.lexsort((at, seed))
    return rows[key].to_numpy(dtype=np.float64)[order].reshape(-1, len(split))


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per label, in the scores' order: counts, IQM [CI] per metric, rb_tuned like-for-like,
    paired gains over rb_tuned's same season, P(improvement) and the flag."""
    rb_rows = scores[scores["label"] == BASELINE]
    if rb_rows.empty:
        raise ValueError(
            f"no {BASELINE} rows in the scores: every like-for-like bar, gain and P(improvement)"
            f" is against {BASELINE}; run evaluate --controller {BASELINE} --split validation"
        )
    rb = {m: _matrix(rb_rows, m) for m in ("profit_eur_m2", "viol_hours_total", *VIOL)}
    out = []
    for label in dict.fromkeys(scores["label"]):
        rows = scores[scores["label"] == label]
        x = {m: _matrix(rows, m) for m in METRICS}
        n = len(x["profit_eur_m2"])
        first = rows.iloc[0]
        row = {
            "label": label,
            "decision": first["decision"],
            "protocol": first["protocol"],
            "n_runs": n,
            "crashed": first["crashed"],
            "invalid_episodes": int((~rows["valid"].astype(bool)).sum()),
        }
        for m in METRICS:
            row[f"{m}_iqm"], row[f"{m}_lo"], row[f"{m}_hi"] = stats.iqm_ci(x[m])
        row["viol_hours_sum_iqm"] = sum(row[f"{v}_iqm"] for v in VIOL)
        row["rb_tuned_profit_iqm"] = stats.iqm_ci(np.tile(rb["profit_eur_m2"], (n, 1)))[0]
        row["rb_tuned_viol_sum_iqm"] = sum(stats.iqm_ci(np.tile(rb[v], (n, 1)))[0] for v in VIOL)
        gain = stats.iqm_ci(x["profit_eur_m2"] - rb["profit_eur_m2"])
        row["profit_gain_iqm"], row["profit_gain_lo"], row["profit_gain_hi"] = gain
        saved = stats.iqm_ci(rb["viol_hours_total"] - x["viol_hours_total"])
        row["viol_saved_iqm"], row["viol_saved_lo"], row["viol_saved_hi"] = saved
        row["p_improve_profit"] = stats.prob_improvement(x["profit_eur_m2"], rb["profit_eur_m2"])
        row["p_improve_viol"] = stats.prob_improvement(
            -x["viol_hours_total"], -rb["viol_hours_total"]
        )
        if label in DETERMINISTIC:
            row["flag"] = "deterministic, 1 run"
        else:
            row["flag"] = selection.NO_CLAIM if n < 5 else ""
        out.append(row)
    return pd.DataFrame(out).astype({"crashed": "Int64"})


def _rebuilt(rows: pd.DataFrame, numbers: dict, keys: tuple, where: str) -> int:
    """Each key's recorded (iqm, ci_low, ci_high), the first key's at numbers' top level and the
    others' in numbers["guards"], equal stats.iqm_ci on rows within 1e-12, or ValueError starting
    with where. Returns the number of keys checked."""
    for key in keys:
        recorded = numbers if key == keys[0] else numbers["guards"][key]
        got = stats.iqm_ci(_matrix(rows, key))
        for field, value in zip(("iqm", "ci_low", "ci_high"), got, strict=True):
            if recorded[field] is None or abs(recorded[field] - value) > 1e-12:
                raise ValueError(
                    f"{where} {key} {field} is {recorded[field]!r} in the ledger but {value!r}"
                    " from the scores"
                )
    return len(keys)


def check_ledger(scores: pd.DataFrame, *, ledger_path: Path) -> int:
    """Every result row of a registration in the scores, rebuilt: its per_arm primary and each guard
    (iqm, ci_low, ci_high) equal stats.iqm_ci on the scores within 1e-12, or ValueError naming the
    result row, arm and key. A TEST split's scores rebuild the final-result row's per_split[split]
    instead: every arm's evaluate.FINAL_PRIMARY and each guard, or ValueError naming the
    final-result row, split, arm and key. Returns the number of (arm, key) pairs checked."""
    existing = ledger.rows(path=ledger_path)
    split = scores["split"].iloc[0]
    checked = 0
    if split in scenarios.TEST:
        final, _ = _final(split, existing)
        for arm, numbers in final["per_split"][split].items():
            where = f"{final['row_id']}: per_split[{split!r}][{arm!r}]"
            rows = scores[scores["arm"] == arm]
            if rows.empty:
                raise ValueError(f"{where} has no score rows to rebuild it from")
            checked += _rebuilt(rows, numbers, (evaluate.FINAL_PRIMARY, *numbers["guards"]), where)
        return checked
    regs = {r["row_id"]: r for r in existing if r["kind"] == "registered"}
    mine = set(scores["registered"].dropna())
    for result in existing:
        if result["kind"] != "result" or result["registered"] not in mine:
            continue
        reg = regs[result["registered"]]
        for arm, numbers in result["per_arm"].items():
            rows = scores[(scores["registered"] == reg["row_id"]) & (scores["arm"] == arm)]
            where = f"result {result['row_id']}: per_arm[{arm!r}]"
            checked += _rebuilt(rows, numbers, (reg["primary"], *reg["guards"]), where)
    return checked


def _text(v) -> str:
    return "" if pd.isna(v) else str(v)


def _ci(row: dict, prefix: str, digits: int) -> str:
    """row's prefix_{iqm,lo,hi} as "iqm [lo, hi]"."""
    iqm, lo, hi = (row[f"{prefix}_{s}"] for s in ("iqm", "lo", "hi"))
    return f"{iqm:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _markdown(summary: pd.DataFrame, split: str, final: tuple[dict, dict] | None = None) -> str:
    """summary.md: the command and its source (a TEST split's: the final's seasons.parquet, by its
    repo path, and the final-result row, from final = _final's pair), a legend, then a pipe table
    with one row per label."""
    source = "the ledger, the runs' selected.json and evaluate's baseline parquets"
    if final is not None:
        result, registered = final
        source = (
            f"control/results/final/{registered['row_id']}/seasons.parquet, checked against"
            f" {result['row_id']} ({SCORED_ONCE})"
        )
    lines = [
        f"Made by `uv run python -m greenleaf_control.report --split {split}` (in control/) from"
        f" {source}; every number is in summary.csv, at full precision.",
        "",
        "IQM [95% stratified-bootstrap CI] over each row's runs x seasons. Profit and gain in"
        " EUR/m², hours in h, yield in kg/m². rb_tuned like-for-like: rb_tuned's one-run row"
        " repeated to the row's n runs, then its IQM. Gain and hours saved pair each season with"
        " rb_tuned's same season; P is the share of same-season wins over rb_tuned, ties half.",
        "",
        "- Total violation hours: the IQM of each season's CO₂ + temp + RH hours, with a CI.",
        "- Σ per-constraint IQMs: the CO₂, temp and RH IQMs added up, the number the registered"
        " rule (b) of E2 to E7 (#31 to #39) uses. It has no CI; rb_tuned's like-for-like Σ is in parentheses.",
        "- Gain and hours saved are IQMs of per-season differences, not differences of IQMs:"
        " hours saved is not rb_tuned's hours minus the row's, in either hours column.",
        "- decision: the result's single decision row, shown on every arm of its registration, so"
        ' an arm the decision note drops can still read "adopted". Read the note.',
        "- A 1-run row's CI is zero-width because there is one run, not because the value is"
        " certain. That matters for random, which is stochastic.",
        "- PROTOCOL: the first 8 hex of the docs/PROTOCOL.md hash at registration. Rows under"
        " different hashes aren't fingerprint-comparable (#38).",
        "- rb_tuned's own row compares it with itself (gain 0, P 0.5).",
        "",
        "| " + " | ".join(HEADER) + " |",
        "|" + "---|" * len(HEADER),
    ]
    for r in summary.to_dict("records"):
        cells = [
            r["label"],
            _text(r["decision"]),
            _text(r["protocol"]),
            str(r["n_runs"]),
            _text(r["crashed"]),
            str(r["invalid_episodes"]),
            _ci(r, "profit_eur_m2", 3),
            f"{r['rb_tuned_profit_iqm']:.3f}",
            _ci(r, "profit_gain", 3),
            f"{r['p_improve_profit']:.3f}",
            _ci(r, "viol_hours_total", 2),
            f"{r['viol_hours_sum_iqm']:.2f} ({r['rb_tuned_viol_sum_iqm']:.2f})",
            _ci(r, "viol_saved", 2),
            f"{r['p_improve_viol']:.3f}",
            _ci(r, "yield_kg_m2", 3),
            _text(r["flag"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _title(split: str, title: str) -> dict:
    """viz's title= for a TEST split's figure: title, the split and SCORED_ONCE. Nothing on
    validation, so viz draws its own."""
    return {"title": f"{title} · {split}, {SCORED_ONCE}"} if split in scenarios.TEST else {}


def figures(scores: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    """out_dir/{per_scenario, profiles, tradeoff}.png, drawn by viz from the scores and the summary
    (PROTOCOL §6), each with a footer naming the command (on a TEST split, SCORED_ONCE on a second
    footer line, and in each title with the split):
    - per_scenario: every label's raw runs x scenarios profit and total violation hours, the
      scenarios labelled `<location> day <start_day>`;
    - profiles: every label but rb_tuned, paired with rb_tuned's same season: profit gain (arm -
      rb_tuned) and violation hours saved (rb_tuned - arm); the adopted labels highlighted;
    - tradeoff: one point per summary row at (viol_saved, profit_gain) IQM [CI] with its status
      and n_runs, so rb_tuned sits at the origin and up and to the right beats it on both.

    A label's status is its decision's word ("adopted", "dropped"), NOTE_CALLS' call for its
    arm overriding it; None for a baseline (rb_tuned's E1 fold included) or an undecided arm.
    """
    split = scores["split"].iloc[0]
    seasons = [f"{s['location']} day {s['start_day']}" for s in scenarios.SPLITS[split]]
    footer = (
        f"source: python -m greenleaf_control.report --split {split}; derived from"
        " summary.csv / scores.parquet"
    )
    if split in scenarios.TEST:  # its own line: on the first, the 8 in wide tradeoff cuts it off
        footer += f"\n{SCORED_ONCE}"
    arm = dict(zip(scores["label"], scores["arm"], strict=True))
    status = {}
    for label, decision in zip(summary["label"], summary["decision"].fillna(""), strict=True):
        word, _, row_id = decision.partition(" ")
        call = NOTE_CALLS.get((row_id, arm[label]), word) or None
        status[label] = None if label in envs.BASELINES else call
    rows = {label: scores[scores["label"] == label] for label in summary["label"]}
    profit = {label: _matrix(r, "profit_eur_m2") for label, r in rows.items()}
    hours = {label: _matrix(r, "viol_hours_total") for label, r in rows.items()}
    panels = {"profit [€/m²]": profit, "total violation hours [h]": hours}
    viz.per_scenario(
        panels,
        seasons,
        out_dir / "per_scenario.png",
        footer=footer,
        **_title(split, viz.PER_SCENARIO_TITLE),
    )
    arms = [label for label in rows if label != BASELINE]
    gains = {
        "profit gain over rb_tuned's same season [€/m²]": {
            label: profit[label] - profit[BASELINE] for label in arms
        },
        "violation hours saved vs rb_tuned's same season [h]": {
            label: hours[BASELINE] - hours[label] for label in arms
        },
    }
    adopted = tuple(label for label in arms if status[label] == "adopted")
    viz.profiles(
        gains,
        out_dir / "profiles.png",
        highlight=adopted,
        footer=footer,
        **_title(split, viz.PROFILES_TITLE),
    )
    points = [
        {
            "label": r["label"],
            "profit": tuple(r[f"profit_gain_{s}"] for s in ("iqm", "lo", "hi")),
            "viol_hours": tuple(r[f"viol_saved_{s}"] for s in ("iqm", "lo", "hi")),
            "status": status[r["label"]],
            "n": r["n_runs"],
        }
        for r in summary.to_dict("records")
    ]
    viz.tradeoff(
        points,
        out_dir / "tradeoff.png",
        xlabel="violation hours saved vs rb_tuned [h]",
        ylabel="profit gain over rb_tuned [€/m²]",
        footer=footer,
        **_title(split, viz.TRADEOFF_TITLE),
    )


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.report")
    p.add_argument(
        "--split",
        choices=["validation", *scenarios.TEST],
        required=True,
        help="validation, or a 2012 test split once the ledger holds the final-result row",
    )
    p.add_argument("--out", type=Path, default=ANALYSIS, help="default control/results/analysis")
    args = p.parse_args(argv)
    # Everything is loaded, checked and summarized before a dir is made or a file written.
    scores = load_scores(args.split, ledger_path=ledger_path)
    pairs = check_ledger(scores, ledger_path=ledger_path)
    summary = summarize(scores)
    final = None
    if args.split in scenarios.TEST:
        final = _final(args.split, ledger.rows(path=ledger_path))
    table = _markdown(summary, args.split, final)

    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out / "scores.parquet")
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(table)
    figures(scores, summary, out)
    print(f"matches the ledger: {pairs} (arm, key) pairs")
    for r in summary.to_dict("records"):
        print(
            f"{r['label']}  n {r['n_runs']}  profit {_ci(r, 'profit_eur_m2', 3)} vs rb_tuned"
            f" {r['rb_tuned_profit_iqm']:.3f}  P {r['p_improve_profit']:.3f}  Σ violation-hour"
            f" IQMs {r['viol_hours_sum_iqm']:.2f} vs rb_tuned {r['rb_tuned_viol_sum_iqm']:.2f}"
            f"  {r['flag']}".rstrip()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

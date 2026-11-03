"""The registered 2013 hypotheses H1-H4, graded from the 2013 final's recorded seasons (#73), one
command:

    python -m greenleaf_control.hypotheses [--out DIR]

The 2013 final is found from the ledger, never typed: the final-result whose final-registered row
has ledger.seal_of(...) == 2013, then FINAL_RESULTS/<that final-registered row>/seasons.parquet (one
row per season and price regime). Each 2013 split is graded alone and each regime from its own
rows: nothing is pooled, no season is played, no weather file is opened and the 2012 final is never
read. It imports no env, evaluate, report or selection: the matrix order is report._matrix's,
written out here.

The conventions (the issue's): an arm's matrix is its runs (by seed) x the split's seasons
(scenarios.SEALS[2013][split] order: the bootstrap depends on it), a baseline's is 1 run. A paired
gain is stats.iqm_ci(arm - the baseline's 1 x 8 row). A baseline's like-for-like profit IQM is
stats.iqm_ci(its row tiled to the arm's run count)[0]. "CI above X" is ci_low > X, "below X"
ci_high < X. Σ is the sum of the three per-constraint violation-hour IQMs (rliable's point IQM,
stats.iqm_ci's first value without its bootstrap), rb_tuned's tiled the same way.

Writes <out>/hypotheses.csv and summary.md, only after everything is loaded and graded.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rliable.metrics import aggregate_iqm

from greenleaf_control import ledger, metrics, scenarios, stats

CONTROL = Path(__file__).resolve().parents[1]
FINAL_RESULTS = CONTROL / "results" / "final"  # read at call time, so tests can point it elsewhere
OUT = CONTROL / "results" / "analysis" / "final_2013"
COMMAND = "python -m greenleaf_control.hypotheses"
SEAL = 2013
COLUMNS = ("split", "hypothesis", "arm", "regime", "part", "value", "bar", "pass")
VIOL = tuple(f"viol_hours_{c}" for c in metrics.CONSTRAINTS)
# H1, H2: (arm, baseline, ((regime, side), ...)): the arm's paired profit gain over the baseline has
# its CI entirely above or below 0 at each regime. Each part is named by its regime.
GAINS = {
    "H1": ("ppo", "rb_default", (("agc2020", "below"), ("sim", "above"))),
    "H2": ("residual_w05", "rb_tuned", (("sim", "above"), ("agc2020", "above"))),
}
# H3, H4: (arms, regime, the baselines whose highest like-for-like profit IQM is part a's bar).
# Part b's bar is rb_tuned's like-for-like Σ; part c's is 0 invalid seasons and solver failures.
RULES = {
    "H3": (("ppo_agc", "residual_agc"), "agc2020", ("rb_default", "rb_tuned", "rb_tuned_agc2020")),
    "H4": (("residual_w05", "lag_b100"), "sim", ("rb_tuned",)),
}
# The registered wording, word for word from issue #73's section (d): its Conventions line, then
# each hypothesis bullet without its "**" bold markers. summary.md states it as written.
REGISTERED = {
    "Conventions": "Conventions: like-for-like baselines = the baseline's 8 seasons tiled to the"
    " arm's seed count through stats.iqm_ci; a paired gain = stats.iqm_ci(arm n×8 − baseline 1×8),"
    ' same seasons; "CI above/below X" = the 95% CI\'s low end > X / high end < X; Σ = the sum of'
    " the three per-constraint hour IQMs (hours don't depend on prices).",
    "H1": "H1 (the price regime reverses the ranking): ppo's paired profit gain over rb_default has"
    " its CI entirely below 0 at agc2020 AND entirely above 0 at sim.",
    "H2": "H2 (the residual agent's edge is robust to prices): residual_w05's paired profit gain over"
    " rb_tuned has its CI entirely above 0 at sim AND at agc2020.",
    "H3": "H3 (price-matched RL beats the price-matched rules): for each of ppo_agc and residual_agc,"
    " at agc2020: (a) its profit IQM CI lies entirely above max(rb_default, rb_tuned,"
    " rb_tuned_agc2020)'s like-for-like profit IQM; (b) its Σ ≤ rb_tuned's like-for-like Σ (no"
    " slack); (c) 0 invalid, 0 solver failures.",
    "H4": "H4 (replication of #45 on fresh weather): #45's rule at sim for residual_w05 and lag_b100:"
    " (a) profit CI above rb_tuned's like-for-like profit IQM; (b) Σ ≤ rb_tuned's like-for-like Σ;"
    " (c) 0 invalid, 0 solver failures.",
}
SOURCE = "registered as issue #73's text (plan approved 2026-11-02)"
NO_CLAIM = "n<5: no claim"  # evaluate.NO_CLAIM, copied: evaluate imports the env and SB3


def _final(existing: list[dict], ledger_path: Path) -> tuple[dict, dict]:
    """(the 2013 final-result row, the final-registered row it names); a ValueError without one."""
    registered = {r["row_id"]: r for r in existing if r["kind"] == "final-registered"}
    finals = [
        r
        for r in existing
        if r["kind"] == "final-result" and ledger.seal_of(registered[r["final-registered"]]) == SEAL
    ]
    if not finals:
        raise ValueError(
            f"{ledger_path} has no final-result row of the 2013 seal: the 2013 final isn't recorded,"
            " so there are no 2013 seasons to grade"
        )
    [final] = finals
    return final, registered[final["final-registered"]]


def _matrix(rows: pd.DataFrame, key: str, split: str, who: str) -> np.ndarray:
    """key's runs (by seed, a baseline's one run -1) x seasons (scenarios.SEALS[2013][split] order)
    array of one arm's rows at one regime, whatever their order in the frame: the bootstrap depends
    on it. Refused (ValueError naming who) unless every run holds each of the split's seasons once,
    so a missing arm is refused too."""
    seasons = scenarios.SEALS[SEAL][split]
    column = {(s["location"], s["growth_year"], s["start_day"]): i for i, s in enumerate(seasons)}
    keys = zip(rows["location"], rows["growth_year"], rows["start_day"], strict=True)
    at = np.array([column[k] for k in keys], dtype=np.int64)
    seed = rows["seed"].fillna(-1).to_numpy(dtype=np.int64)
    order = np.lexsort((at, seed))
    whole = np.tile(np.arange(len(seasons)), len(rows) // len(seasons))
    if rows.empty or not np.array_equal(at[order], whole):
        raise ValueError(
            f"{who} has {len(rows)} rows on {split}, not each of its {len(seasons)} seasons once per"
            " run: the hypotheses can't be graded without them"
        )
    return rows[key].to_numpy(dtype=np.float64)[order].reshape(-1, len(seasons))


def _grade(frame: pd.DataFrame, split: str) -> list[dict]:
    """One dict per hypothesis x arm x part on split: COLUMNS, plus the part's relation (value
    relation bar is what passes) and the arm's run count, for summary.md."""
    on = frame[frame["split"] == split]
    graded = []

    def mine(arm: str, regime: str) -> pd.DataFrame:
        return on[(on["arm"] == arm) & (on["price_regime"] == regime)]

    def matrix(arm: str, regime: str, key: str = "profit_eur_m2") -> np.ndarray:
        return _matrix(mine(arm, regime), key, split, f"{arm} at {regime}")

    def add(hyp, arm, regime, part, value, bar, ok, relation, n, **shown) -> None:
        values = (split, hyp, arm, regime, part, float(value), float(bar), bool(ok))
        graded.append(
            {**dict(zip(COLUMNS, values, strict=True)), "relation": relation, "n": n, **shown}
        )

    for hyp, (arm, baseline, sides) in GAINS.items():
        for regime, side in sides:
            m = matrix(arm, regime)
            _, low, high = stats.iqm_ci(m - matrix(baseline, regime))
            if side == "below":
                add(hyp, arm, regime, regime, high, 0.0, high < 0, "<", len(m))
            else:
                add(hyp, arm, regime, regime, low, 0.0, low > 0, ">", len(m))
    for hyp, (arms, regime, baselines) in RULES.items():
        for arm in arms:
            profit = matrix(arm, regime)
            n = len(profit)
            low = stats.iqm_ci(profit)[1]
            bar = max(stats.iqm_ci(np.tile(matrix(b, regime), (n, 1)))[0] for b in baselines)
            sigma = sum(float(aggregate_iqm(matrix(arm, regime, v))) for v in VIOL)
            tuned = sum(
                float(aggregate_iqm(np.tile(matrix("rb_tuned", regime, v), (n, 1)))) for v in VIOL
            )
            rows = mine(arm, regime)
            invalid = int((~rows["valid"].astype(bool)).sum())
            failures = int(rows["solver_failures"].sum())
            bad = invalid + failures
            add(hyp, arm, regime, "a", low, bar, low > bar, ">", n)
            add(hyp, arm, regime, "b", sigma, tuned, sigma <= tuned, "≤", n)
            add(
                hyp,
                arm,
                regime,
                "c",
                bad,
                0.0,
                bad == 0,
                "=",
                n,
                invalid=invalid,
                failures=failures,
            )
    return graded


def _verdicts(graded: list[dict]) -> dict[tuple[str, str, str], tuple[str, list[dict]]]:
    """(split, hypothesis, arm) -> (its verdict, its parts), in graded's order: "verified" when every
    part passes, else "fails (<the parts that don't>)"; followed by NO_CLAIM on fewer than 5 runs
    (PROTOCOL: no claim on n < 5)."""
    groups = {}
    for p in graded:
        groups.setdefault((p["split"], p["hypothesis"], p["arm"]), []).append(p)
    verdicts = {}
    for key, parts in groups.items():
        failed = [p["part"] for p in parts if not p["pass"]]
        verdict = f"fails ({', '.join(failed)})" if failed else "verified"
        verdicts[key] = (f"{verdict}; {NO_CLAIM}" if parts[0]["n"] < 5 else verdict, parts)
    return verdicts


def _summary(graded: list[dict], final: dict, registered: dict) -> str:
    """summary.md: the command and its source (the 2013 final's seasons.parquet by its repo path,
    and its final-result row), the hypotheses, a legend, then a pipe table with one row per
    split x hypothesis x arm and its verdict."""
    lines = [
        f"Made by `uv run {COMMAND}` (in control/) from"
        f" control/results/final/{registered['row_id']}/seasons.parquet, the 2013 final"
        f" ({final['row_id']}; 2013 test weather, scored once). Each 2013 split is graded alone;"
        " nothing is pooled. Every number is in hypotheses.csv at full precision.",
        "",
        f"The hypotheses, {SOURCE}, word for word (graded per split; verified only if every part"
        " holds on that split):",
        "",
        REGISTERED["Conventions"],
        "",
        *(f"- {REGISTERED[h]}" for h in ("H1", "H2", "H3", "H4")),
        "",
        "- A part reads value, relation, bar: it passes when the relation holds. H1's and H2's"
        " parts are named by their price regime: the value is the paired profit gain's 95% CI high"
        " end (below 0) or low end (above 0), EUR/m².",
        "- a: the arm's profit IQM 95% CI low end vs the like-for-like baseline profit IQM, EUR/m²."
        " H3(a)'s bar is the largest of the three baselines' like-for-like profit IQMs, each"
        " through stats.iqm_ci(its one-run row tiled to the arm's n runs)[0]; H4(a)'s is"
        " rb_tuned's, the same way.",
        "- b: Σ of the CO₂, temp and RH violation-hour IQMs vs rb_tuned's like-for-like Σ, h.",
        "- c: the arm's invalid seasons and its solver failures at the regime graded; it passes"
        " when both are 0.",
        "- 95% stratified-bootstrap CIs over the arm's n runs x 8 seasons. A baseline is"
        f" deterministic, one run. A verdict on fewer than 5 runs is flagged {NO_CLAIM}.",
        "",
        "| split | hypothesis | arm | n | verdict | parts |",
        "|---|---|---|---|---|---|",
    ]
    for (split, hyp, arm), (verdict, parts) in _verdicts(graded).items():
        cells = []
        for p in parts:
            ok = "pass" if p["pass"] else "fail"
            if p["part"] == "c":
                cells.append(f"c: invalid {p['invalid']}, solver failures {p['failures']} ({ok})")
            else:
                cells.append(f"{p['part']}: {p['value']:.3f} {p['relation']} {p['bar']:.3f} ({ok})")
        row = [split, hyp, arm, str(parts[0]["n"]), verdict, "; ".join(cells)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    p = argparse.ArgumentParser(prog=COMMAND)
    p.add_argument(
        "--out", type=Path, default=OUT, help="default control/results/analysis/final_2013"
    )
    args = p.parse_args(argv)
    # Everything is loaded and graded before a dir is made or a file written.
    final, registered = _final(ledger.rows(path=ledger_path), ledger_path)
    seasons = FINAL_RESULTS / registered["row_id"] / "seasons.parquet"
    if not seasons.is_file():
        raise ValueError(
            f"no {seasons}: the 2013 final ({final['row_id']}) is graded from its files"
        )
    frame = pd.read_parquet(seasons)
    graded = [row for split in scenarios.SEALS[SEAL] for row in _grade(frame, split)]
    table = pd.DataFrame(graded)[list(COLUMNS)]
    text = _summary(graded, final, registered)

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "hypotheses.csv", index=False)
    print(f"wrote {args.out / 'hypotheses.csv'} ({len(table)} rows)")
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(f"wrote {args.out / 'summary.md'}")
    for (split, hyp, arm), (verdict, _) in _verdicts(graded).items():
        print(f"{split} {hyp} {arm}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for greenleaf_control.report (issue #46, A1a): `python -m greenleaf_control.report --split
validation [--out DIR]` rebuilds every arm's per-season scores from the ledger, the runs'
selected.json files and evaluate's baseline parquets, checks them against the ledger's result rows,
and only then writes <out>/validation/{scores.parquet, summary.csv, summary.md}.

With the orchestrator's defaults on the issue: A1a is VALIDATION only (a test split is an argparse
error; the 2012 path is a follow-up after F1), and load_scores carries a per-label crashed column,
null for a baseline. As these tests pin it:
- ANALYSIS = control/results/analysis, BASELINE = "rb_tuned", METRICS = (profit_eur_m2,
  viol_hours_total, viol_hours_co2, viol_hours_temp, viol_hours_rh, yield_kg_m2, paper_return).
- load_scores(split, *, ledger_path=ledger.LEDGER): one row per label x run x scenario.
  - Columns: label, arm, issue, registered, result, decision, protocol, run, seed, crashed, split,
    location, growth_year, start_day, every ledger.RESULT_KEYS key, valid, and viol_hours_total
    (co2 + temp + rh).
  - Order: baselines first in envs.BASELINES order, then registrations in ledger order, arms in
    registration order, runs by seed, scenarios in scenarios.SPLITS order.
  - Baselines: every *.parquet in selection.RESULTS, rows with split validation; labelled by name;
    issue, registered, result, decision, protocol, run, seed and crashed null. A controller not in
    envs.BASELINES, or a (controller, scenario) seen twice, is refused.
  - Arms: each registered row with scenarios ["validation"] and a result row; its runs are
    selection._registration's (no crashed run), each scored from selection._read's selected.json,
    candidates[checkpoint]["per_scenario"]. A run with no selected.json is refused ("run selection
    first"). A registration with no result row prints `skipped <row>: no result row yet` and none
    of its files are read.
  - label `<arm> #<issue>/<first 4 hex of the registered id>`; decision `<value> <decision row id>`
    or null (two decision rows on one result are refused); protocol the first 8 hex of the
    registration's fingerprints["docs/PROTOCOL.md"]; crashed ledger._arm_tally's count.
  - An arm of method rb_tuned (E1) is not a second row: its registered, result, decision and
    protocol go on the rb_tuned baseline row. A second such arm is refused, as is one whose run
    crashed.
  - A test split is refused with a ValueError before any parquet is read.
- summarize(scores): one row per label (a `label` column), in the scores' order, each statistic on
  the label's runs (by seed) x scenarios array: n_runs, crashed, invalid_episodes; <m>_{iqm,lo,hi}
  from stats.iqm_ci for m in METRICS; viol_hours_sum_iqm (the sum of the 3 per-constraint IQMs);
  rb_tuned_profit_iqm and rb_tuned_viol_sum_iqm (rb_tuned's 1 x S rows tiled to n_runs x S);
  profit_gain_* = iqm_ci(arm - rb_tuned) on profit, viol_saved_* = iqm_ci(rb_tuned - arm) on
  viol_hours_total; p_improve_profit = prob_improvement(arm, rb_tuned), p_improve_viol the same on
  negated viol_hours_total; flag "deterministic, 1 run" (rb_default, rb_tuned, zero_controls),
  "n<5: no claim" under 5 runs (random included), else empty. No rb_tuned rows raises.
- check_ledger(scores, *, ledger_path) -> int: every result row's per_arm primary and guards
  (iqm, ci_low, ci_high) equal stats.iqm_ci on the scores within 1e-12, else a ValueError naming
  the result row, arm and key; returns the number of (arm, key) pairs.
- main(argv=None, *, ledger_path=ledger.LEDGER) -> int: loads, summarizes and checks before it
  writes anything (a refusal makes no dir), prints `matches the ledger: <k> (arm, key) pairs` and a
  line per label; summary.csv at full precision, summary.md a pipe table (one row per label, 3
  decimals, hours at 2) under a first line naming the command and summary.csv; no timestamp or out
  path, so two --out dirs get identical bytes; a rerun replaces the files.
- figures(scores, summary, out_dir) (#47, A1b), called by main after the tables: three PNGs next
  to them, each drawn by viz (called as viz.<name>, so a spy that wraps the real function sees
  it) with footer `source: python -m greenleaf_control.report --split validation; ...`:
  - per_scenario.png: viz.per_scenario, two panels (profit, then total violation hours), every
    label's raw runs x scenarios array, scenario labels `<location> day <start_day>`;
  - profiles.png: viz.profiles, panels "profit gain over rb_tuned's same season [€/m²]" (arm -
    rb_tuned) and "violation hours saved vs rb_tuned's same season [h]" (rb_tuned - arm), every
    label but rb_tuned; highlight = the labels whose decision starts "adopted" (the Instrument
    style's default on #47);
  - tradeoff.png: viz.tradeoff, one point per summary row at (viol_saved_*, profit_gain_*),
    rb_tuned at (0, 0, 0), x label "violation hours saved vs rb_tuned [h]", y label "profit gain
    over rb_tuned [€/m²]".
  The fix round on #47 adds: the footer says "derived from summary.csv / scores.parquet"; each
  tradeoff point carries a "status" (its decision's call, "adopted" or "dropped", None for a
  baseline or an undecided arm), with report.NOTE_CALLS[(decision row id, arm)] overriding it for
  an arm a decision's note calls on its own (#34's decision-b159ac9a drops residual_w4); highlight
  is the labels whose status is "adopted"; a learning label under 5 runs has "(n=<k>, no claim)"
  in its drawn tradeoff label.

Offline: every ledger, run dir, parquet and output is under tmp_path; selection.RESULTS is patched
to tmp_path in every test that loads; every --out is under tmp_path. Nothing plays a season:
evaluate.run_policy and run_rb, envs.run_baseline, make_raw_env and make_env,
train.build_vec_env, rule_based.load_rb_params and selection.select_checkpoint raise if called. The
last test checks the real ledger and control/results/ are as they were when this module was
collected. No 2012 weather is opened.
"""

import importlib
import importlib.util
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from greenleaf_control import (
    envs,
    evaluate,
    ledger,
    metrics,
    rule_based,
    scenarios,
    selection,
    stats,
    train,
)

# greenleaf_control.report is imported inside each test body (by _module), so while it is missing
# each test FAILS on an assertion rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RESULTS = CONTROL / "results"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    """Every path under path, relative and sorted; None if path doesn't exist."""
    return sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else None


# The real ledger and results, read when this module is collected, before any test runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RESULTS_BEFORE = _listing(REAL_RESULTS)

VALIDATION = scenarios.SPLITS["validation"]
SEASON_STEPS = 60 * 96
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
GUARDS = ["viol_hours_temp", "yield_kg_m2"]
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}
NO_CLAIM = "n<5: no claim"
DETERMINISTIC = "deterministic, 1 run"
# load_scores' identity columns, null on a baseline's rows; then every column it must have.
IDS = ("label", "issue", "registered", "result", "decision", "protocol", "run", "seed", "crashed")
SCORE_COLUMNS = {
    *IDS,
    *ledger.RESULT_KEYS,
    *"arm split location growth_year start_day valid viol_hours_total".split(),
}
SUMMARY_COLUMNS = {
    *"label n_runs crashed invalid_episodes viol_hours_sum_iqm flag".split(),
    *"rb_tuned_profit_iqm rb_tuned_viol_sum_iqm p_improve_profit p_improve_viol".split(),
    *(f"{m}_{s}" for m in (*METRICS, "profit_gain", "viol_saved") for s in ("iqm", "lo", "hi")),
}
# The figures (#47): viz's names, each written as <name>.png next to the tables
FIGURES = ("per_scenario", "profiles", "tradeoff")
PNG = b"\x89PNG\r\n\x1a\n"  # the 8-byte PNG signature


# --- Seasons, parquets, ledgers --------------------------------------------------------------


def _season(valid: bool = True, **values) -> dict:
    """One season's result dict as run_policy and run_rb return it (metrics.episode_metrics' keys
    plus paper_reward_sum), each at a neutral value unless given."""
    return {
        "profit_eur_m2": 0.0,
        **{v: 0.0 for v in VIOL},
        **{f"pct_in_bounds_{c}": 100.0 for c in metrics.CONSTRAINTS},
        "yield_kg_m2": 30.0,
        "heat_cost": 1.0,
        "elec_cost": 1.0,
        "co2_cost": 1.0,
        "paper_return": 0.0,
        "solver_failures": 0 if valid else 1,
        "steps": SEASON_STEPS,
        "valid": valid,
        "paper_reward_sum": 0.0,
        **values,
    }


def _seasons(
    profit=(0, 0, 0),
    *,
    co2=(0, 0, 0),
    temp=(0, 0, 0),
    rh=(0, 0, 0),
    yld=(30, 30, 30),
    paper_return=(0, 0, 0),
    valid=(True, True, True),
) -> list[dict]:
    """The three VALIDATION seasons, in order, at these per-season values."""
    return [
        _season(
            valid=valid[i],
            profit_eur_m2=float(profit[i]),
            viol_hours_co2=float(co2[i]),
            viol_hours_temp=float(temp[i]),
            viol_hours_rh=float(rh[i]),
            yield_kg_m2=float(yld[i]),
            paper_return=float(paper_return[i]),
        )
        for i in range(3)
    ]


# rb_tuned's VALIDATION seasons. Its profits (-5, 1, 2) are the issue's like-for-like example;
# viol_hours_total is (0 + 0 + 3, 0 + 6 + 3, 0 + 9 + 12) = (3, 9, 21).
RB = _seasons((-5, 1, 2), temp=(0, 6, 9), rh=(3, 3, 12), yld=(20, 30, 40), paper_return=(1, 2, 6))
# Five seeds' scores, seed -> (VALIDATION start day 45, 59, 75): PROFIT's 15 values are 1..15, so
# its IQM (int(0.25 x 15) = 3 trimmed from each end) is the mean of 4..12 = 8.0.
PROFIT = {0: (15, 1, 8), 1: (2, 14, 9), 2: (7, 3, 13), 3: (12, 6, 4), 4: (5, 11, 10)}
# The ledger's run order; the bootstrap depends on row order, so only seed order gives its CI.
LEDGER_ORDER = (3, 0, 4, 1, 2)
# _world's arms, seed -> seasons.
ALPHA = {
    s: _seasons(
        (10 * s + 1, 10 * s + 2, 10 * s + 4), temp=(s, 2 * s, 3), rh=(1, 0, s), yld=(25, 26, s)
    )
    for s in range(5)
}
BRAVO = {
    s: _seasons((10 * s + 3, 5, 10 * s), temp=(2, s, 1), rh=(s, s, 0), yld=(20, 21 + s, 22))
    for s in range(5)
}


def _decoy() -> list[dict]:
    """A checkpoint no number may read: 1000 in every metric, every season invalid."""
    k = (1000,) * 3
    return _seasons(k, co2=k, temp=k, rh=k, yld=k, paper_return=k, valid=(False,) * 3)


def _selected_json(method: str, seasons: list[dict]) -> dict:
    """selected.json as select_checkpoint writes it: checkpoint 500000 holds seasons, 250000 and
    final the decoy."""
    candidates = {"250000": _decoy(), "500000": seasons, "final": _decoy()}
    return {
        "checkpoint": "500000",
        "method": method,
        "rule": "max_mean_paper_return",
        "scenarios": scenarios.VALIDATION,
        "season_length": 60,
        "budgets": None,
        "tolerance": None,
        "candidates": {
            name: {
                "per_scenario": per,
                "mean_paper_return": float(np.mean([s["paper_return"] for s in per])),
                "mean_profit_eur_m2": float(np.mean([s["profit_eur_m2"] for s in per])),
                "mean_cost": None,
                "qualifies": None,
            }
            for name, per in candidates.items()
        },
    }


def _write_selected(run_dir: Path, method: str, seasons: list[dict], **stale) -> None:
    selected = {**_selected_json(method, seasons), **stale}
    (run_dir / "selected.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")


def _baseline(
    results: Path, controller: str, seasons: list[dict], *, split="validation", file=None
) -> Path:
    """An evaluate --controller <controller> --split <split> parquet in results: one row per
    scenario in split order, evaluate's identity columns then the result keys."""
    rows = [
        {
            "controller": controller,
            "checkpoint": None,
            "split": split,
            **s,
            "season_length": 60,
            **season,
        }
        for s, season in zip(scenarios.SPLITS[split], seasons, strict=False)
    ]
    out = results / (file or f"{controller}_{split}.parquet")
    pd.DataFrame(rows).to_parquet(out)
    return out


def _register(path: Path, *, issue: int, arms, seeds, dry: bool = False, **over) -> dict:
    """A registration on the tmp ledger: arms is ((name, method), ...); an rb_tuned arm's config
    is {"trials": 2}, as tune_rb records it. over replaces any other field."""
    kwargs = {
        "issue": issue,
        "hypothesis": "the arms differ in validation profit",
        "arms": [
            {"name": n, "method": m, "config": {"trials": 2} if m == "rb_tuned" else {}}
            for n, m in arms
        ],
        "seeds": list(seeds),
        "timesteps": 20_000,
        "scenarios": ["validation"],
        "primary": "profit_eur_m2",
        "guards": list(GUARDS),
        "rule": "adopt if the profit IQM CI excludes rb_tuned's",
        "budget": {"cpu_hours": 1.0, "usd": 0.0, "backend": "local"},
        **over,
    }
    return ledger.register(**kwargs, dry=dry, path=path)


def _register_e1(path: Path, issue: int = 22) -> dict:
    """E1's shape (#22): one arm rb_tuned of method rb_tuned, seed 0, timesteps 0, primary
    paper_return, guards the three viol_hours_*."""
    return _register(
        path,
        issue=issue,
        arms=(("rb_tuned", "rb_tuned"),),
        seeds=(0,),
        timesteps=0,
        primary="paper_return",
        guards=list(VIOL),
    )


def _run(
    path: Path,
    root: Path,
    reg: dict,
    arm: str,
    seed: int,
    *,
    method: str = "ppo",
    status: str = "finished",
    seasons: list[dict] | None = None,
    end: bool = True,
) -> tuple[dict, Path]:
    """One run of reg's arm at seed: its dir under root/runs (selected.json from seasons if
    given), its run row and, unless end is False, its run-end. Returns (run row, run dir)."""
    run_dir = root / "runs" / reg["row_id"] / arm / f"seed{seed}"
    run_dir.mkdir(parents=True)
    if seasons is not None:
        _write_selected(run_dir, method, seasons)
    run = ledger.run_start(
        registered=reg["row_id"], arm=arm, seed=seed, run_dir=str(run_dir), path=path
    )
    if end:
        ledger.run_end(
            run=run["row_id"],
            status=status,
            wall_s=1.0,
            timesteps=reg["timesteps"],
            solver_failures=0,
            cost_usd=None,
            diagnostics=DIAGNOSTICS,
            path=path,
        )
    return run, run_dir


def _rb_tuned_run(path: Path, root: Path, reg: dict, *, status: str = "finished") -> dict:
    """E1's run as tune_rb records it: a dir holding rule_based_tuned.yml, a run row and a run-end
    (timesteps 0, diagnostics null)."""
    run_dir = root / "runs" / reg["row_id"] / "rb_tuned" / "seed0"
    run_dir.mkdir(parents=True)
    text = yaml.safe_dump({"GreenLightEnv": {"co2_day": 937.5}})
    (run_dir / "rule_based_tuned.yml").write_text(text)
    run = ledger.run_start(
        registered=reg["row_id"], arm="rb_tuned", seed=0, run_dir=str(run_dir), path=path
    )
    ledger.run_end(
        run=run["row_id"],
        status=status,
        wall_s=1.0,
        timesteps=0,
        solver_failures=0,
        cost_usd=None,
        diagnostics=dict.fromkeys(ledger.DIAGNOSTIC_KEYS),
        path=path,
    )
    return run


def _ci(by_run: list[list[dict]], key: str) -> dict:
    """stats.iqm_ci of key's runs x VALIDATION array, as selection records it; null with no run."""
    if not by_run:
        return {"iqm": None, "ci_low": None, "ci_high": None}
    iqm, low, high = stats.iqm_ci(np.array([[s[key] for s in run] for run in by_run]))
    return {"iqm": iqm, "ci_low": low, "ci_high": high}


def _result(path: Path, reg: dict, scored: dict, off: tuple | None = None) -> dict:
    """reg's result row as selection's numbers and `ledger result` record it: per_arm[arm] is
    _ci of the primary and each guard on scored[arm] (its runs' seasons, in seed order), with
    invalid_episodes the count of valid False. off = (arm, key, field) adds 1e-9 to one number."""
    per_arm = {
        arm: {
            **_ci(by_run, reg["primary"]),
            "guards": {g: _ci(by_run, g) for g in reg["guards"]},
            "invalid_episodes": sum(not s["valid"] for run in by_run for s in run),
        }
        for arm, by_run in scored.items()
    }
    if off is not None:
        arm, key, field = off
        numbers = per_arm[arm] if key == reg["primary"] else per_arm[arm]["guards"][key]
        numbers[field] += 1e-9
    return ledger.record_result(registered=reg["row_id"], per_arm=per_arm, selected={}, path=path)


def _arm(path: Path, root: Path, reg: dict, arm: str, by_seed: dict, *, method="ppo", order=None):
    """A finished run per seed (in order, default by seed) with by_seed[seed] at its checkpoint.
    Returns {seed: run row}."""
    return {
        s: _run(path, root, reg, arm, s, method=method, seasons=by_seed[s])[0]
        for s in (order or sorted(by_seed))
    }


def _world(
    path: Path,
    root: Path,
    results: Path,
    *,
    arms=("alpha",),
    seeds=(0, 1),
    e1: str | None = "finished",
    off: tuple | None = None,
) -> dict:
    """The ledger and files most tests start from:
    - rb_tuned's parquet (RB) in results;
    - unless e1 is None, E1 (#22, _register_e1) with its run (status e1), its result (RB's
      numbers, null if the run crashed) and an adopted decision;
    - R (#36): arms alpha (ppo, ALPHA) and bravo (residual, BRAVO), the first len(arms), at
      seeds, runs appended in reverse seed order, each with selected.json; its result and a
      dropped decision.
    off = ("e1" | "r", arm, key, field) puts 1e-9 on one number of that result.
    Returns the row ids and run dirs."""
    _baseline(results, "rb_tuned", RB)
    ids = {}
    if e1 is not None:
        reg = _register_e1(path)
        _rb_tuned_run(path, root, reg, status=e1)
        res = _result(
            path,
            reg,
            {"rb_tuned": [RB] if e1 != "crashed" else []},
            off=off[1:] if off and off[0] == "e1" else None,
        )
        dec = ledger.decide(result=res["row_id"], decision="adopted", note="E1", path=path)
        ids.update(e1=reg, e1_result=res["row_id"], e1_decision=dec["row_id"])
    methods = {"alpha": ("ppo", ALPHA), "bravo": ("residual", BRAVO)}
    reg = _register(path, issue=36, arms=[(a, methods[a][0]) for a in arms], seeds=seeds)
    runs, dirs = {}, {}
    for s in sorted(seeds, reverse=True):
        for a in arms:
            method, table = methods[a]
            runs[a, s], dirs[a, s] = _run(path, root, reg, a, s, method=method, seasons=table[s])
    scored = {a: [methods[a][1][s] for s in sorted(seeds)] for a in arms}
    res = _result(path, reg, scored, off=off[1:] if off and off[0] == "r" else None)
    dec = ledger.decide(result=res["row_id"], decision="dropped", note="R", path=path)
    ids.update(r=reg, r_result=res["row_id"], r_decision=dec["row_id"], runs=runs, dirs=dirs)
    return ids


# --- The module, tripwires, small helpers --------------------------------------------------------


def _module():
    """greenleaf_control.report; an assertion, not an ImportError, while it is missing."""
    found = importlib.util.find_spec("greenleaf_control.report")
    assert found is not None, "greenleaf_control.report doesn't exist yet (#46)"
    return importlib.import_module("greenleaf_control.report")


def _tripwire(name: str):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{name} was called")

    return refuse


def _trip(monkeypatch) -> None:
    """Every way to play a season, select_checkpoint and reading configs/rule_based_*.yml raise."""
    for module, name in [
        (evaluate, "run_policy"),
        (evaluate, "run_rb"),
        (train, "build_vec_env"),
        (envs, "make_raw_env"),
        (envs, "make_env"),
        (envs, "run_baseline"),
        (rule_based, "load_rb_params"),
        (selection, "select_checkpoint"),
    ]:
        monkeypatch.setattr(module, name, _tripwire(f"{module.__name__}.{name}"))


def _report(monkeypatch, tmp_path: Path):
    """(report, results): selection.RESULTS patched to tmp_path/results/validation (made here),
    and every season-playing function tripped."""
    report = _module()
    results = tmp_path / "results" / "validation"
    results.mkdir(parents=True)
    monkeypatch.setattr(selection, "RESULTS", results)
    _trip(monkeypatch)
    return report, results


def _refused(report, argv: list[str], path: Path, capsys) -> str:
    """main refuses argv: it raises ValueError or a parser error (SystemExit, non-zero), or
    returns non-zero. Returns everything it said: the error, stdout and stderr."""
    capsys.readouterr()
    try:
        code = report.main(argv, ledger_path=path)
    except (ValueError, SystemExit) as e:
        if isinstance(e, SystemExit):
            assert e.code not in (0, None), f"main exited {e.code!r}, which is not a refusal"
        said = capsys.readouterr()
        return f"{e}\n{said.out}\n{said.err}"
    said = capsys.readouterr()
    assert code != 0, f"main returned 0 instead of refusing; it printed {said.out!r}"
    return f"{said.out}\n{said.err}"


def _tree(root: Path) -> dict:
    """Every file (its bytes) and dir (None) under root."""
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


def _null(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


def _vals(series: pd.Series) -> list:
    """series as a list, every null (None, NaN, pd.NA) as None."""
    return [None if _null(v) else v for v in series.tolist()]


def _blank(v) -> bool:
    return _null(v) or v == ""


def _label(arm: str, reg: dict) -> str:
    return f"{arm} #{reg['issue']}/{reg['row_id'].removeprefix('registered-')[:4]}"


def _array(by_seed: dict, key: str, order=None) -> np.ndarray:
    """key's runs x VALIDATION array, rows in order (default by seed)."""
    rows = order or sorted(by_seed)
    return np.array([[s[key] for s in by_seed[seed]] for seed in rows], dtype=np.float64)


def _row(summary: pd.DataFrame, label: str) -> pd.Series:
    [i] = np.flatnonzero(summary["label"].to_numpy() == label)
    return summary.iloc[i]


def _triple(row: pd.Series, prefix: str) -> tuple:
    return (row[f"{prefix}_iqm"], row[f"{prefix}_lo"], row[f"{prefix}_hi"])


# --- load_scores: rows, ids and order ----------------------------------------------------------


def test_scores_have_one_row_per_label_run_and_scenario_in_selections_order(tmp_path, monkeypatch):
    """load_scores("validation"), its columns and order:
    - the four baselines first, in envs.BASELINES order (zero_controls, random, rb_default,
      rb_tuned), though glob gives their files alphabetically; issue, registered, result,
      decision, protocol, run, seed and crashed null;
    - then R1 (#33, arms alpha then bravo, seeds 0 and 1, runs appended seed 1 first and bravo
      first, a dropped decision), each arm's runs by seed and scenarios in VALIDATION order;
    - then R2 (#35, arm charlie), its registration planted with PROTOCOL.md's fingerprint
      0123abcd..., so protocol is read per registration; no decision, so decision is null.
    Ignored: a rb_default parquet on the paper split, a numbers .json in RESULTS, and R3 (#34),
    whose scenarios are ["paper"] (its run dir has no selected.json, so reading it would refuse).
    Each row carries its season's numbers as stored (the chosen 500000 checkpoint's, never the
    decoys), viol_hours_total = co2 + temp + rh, and valid (alpha seed 1's last season False).
    27 rows: 4 baselines x 3 + R1's 2 arms x 2 seeds x 3 + R2's 1 x 1 x 3."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    base = {
        b: _seasons(
            (i + 0.25, i + 0.5, i + 0.75),
            co2=(i, 0, 0),
            temp=(0, i, 0),
            rh=(0, 0, i),
            valid=(True, b != "rb_default", True),
        )
        for i, b in enumerate(envs.BASELINES)
    }
    for b in reversed(envs.BASELINES):
        _baseline(results, b, base[b])
    _baseline(results, "rb_default", base["rb_default"], split="paper")
    (results / "registered-00000000.json").write_text('{"per_arm": {}, "selected": {}}\n')

    def table(a: int, s: int, invalid_last: bool = False) -> list[dict]:
        last = (True, True, not invalid_last)
        return _seasons(
            (100 * a + 10 * s + 1, 100 * a + 10 * s + 2, 100 * a + 10 * s + 3),
            co2=(a, s, 1),
            temp=(1, a, s),
            rh=(s, 1, a),
            valid=last,
        )

    r1 = _register(path, issue=33, arms=(("alpha", "ppo"), ("bravo", "residual")), seeds=(0, 1))
    seasons = {
        (arm, s): table(a, s, (arm, s) == ("alpha", 1))
        for a, arm in enumerate(("alpha", "bravo"))
        for s in (0, 1)
    }
    runs = {}
    for s in (1, 0):
        for arm, method in (("bravo", "residual"), ("alpha", "ppo")):
            run, _ = _run(path, tmp_path, r1, arm, s, method=method, seasons=seasons[arm, s])
            runs[arm, s] = run
    res1 = _result(path, r1, {arm: [seasons[arm, s] for s in (0, 1)] for arm in ("alpha", "bravo")})
    dec1 = ledger.decide(result=res1["row_id"], decision="dropped", note="R1", path=path)

    r2 = _register(path, issue=35, arms=(("charlie", "residual_lag"),), seeds=(0,), dry=True)
    fake = "0123abcd" + "e" * 56
    r2 = {**r2, "fingerprints": {**r2["fingerprints"], "docs/PROTOCOL.md": fake}}
    with path.open("a") as f:
        f.write(json.dumps(r2, sort_keys=True) + "\n")
    seasons["charlie", 0] = table(2, 0)
    runs["charlie", 0] = _run(
        path, tmp_path, r2, "charlie", 0, method="residual_lag", seasons=seasons["charlie", 0]
    )[0]
    res2 = _result(path, r2, {"charlie": [seasons["charlie", 0]]})

    r3 = _register(path, issue=34, arms=(("echo", "ppo"),), seeds=(0,), scenarios=["paper"])
    _run(path, tmp_path, r3, "echo", 0)
    one = {"iqm": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    per_arm = {"echo": {**one, "guards": {g: one for g in GUARDS}, "invalid_episodes": 0}}
    ledger.record_result(registered=r3["row_id"], per_arm=per_arm, selected={}, path=path)

    scores = report.load_scores("validation", ledger_path=path)

    # Per row: its IDS values, arm (None: not pinned for a baseline) and season, in order.
    want_ids = [(b, *[None] * 8) for b in envs.BASELINES for _ in range(3)]
    want_arms, want_seasons = [None] * 12, [x for b in envs.BASELINES for x in base[b]]
    proto1 = r1["fingerprints"]["docs/PROTOCOL.md"][:8]
    for reg, res, decision, protocol, arms in (
        (r1, res1, f"dropped {dec1['row_id']}", proto1, ("alpha", "bravo")),
        (r2, res2, None, "0123abcd", ("charlie",)),
    ):
        for arm in arms:
            for s in sorted(s for a, s in seasons if a == arm):
                ids = (reg["issue"], reg["row_id"], res["row_id"], decision, protocol)
                want_ids += [(_label(arm, reg), *ids, runs[arm, s]["row_id"], s, 0)] * 3
                want_arms += [arm] * 3
                want_seasons += seasons[arm, s]
    assert SCORE_COLUMNS <= set(scores.columns), SCORE_COLUMNS - set(scores.columns)
    assert len(scores) == len(want_ids) == 27
    assert list(zip(*(_vals(scores[c]) for c in IDS), strict=True)) == want_ids
    assert _vals(scores["arm"])[12:] == want_arms[12:]
    assert scores["split"].tolist() == ["validation"] * 27
    for col in ("location", "growth_year", "start_day"):
        assert scores[col].tolist() == [s[col] for s in VALIDATION] * 9, col
    for key in (*ledger.RESULT_KEYS, "valid"):
        assert scores[key].tolist() == [x[key] for x in want_seasons], key
    totals = [sum(x[v] for v in VIOL) for x in want_seasons]
    assert scores["viol_hours_total"].tolist() == totals
    assert r1["fingerprints"]["docs/PROTOCOL.md"][:8] != "0123abcd"
    assert re.fullmatch(r"alpha #33/[0-9a-f]{4}", _label("alpha", r1))


def test_an_rb_tuned_arm_folds_onto_the_rb_tuned_baseline_row(tmp_path, monkeypatch):
    """E1 (an arm of method rb_tuned) is not a second row: rb_tuned's 3 rows (from its parquet)
    carry E1's registered, result, decision ("adopted <id>") and protocol, and no label starts
    "rb_tuned #". rb_default stays unannotated. check_ledger checks E1's result against the
    rb_tuned rows: E1's paper_return and 3 guards plus alpha's profit and 2 guards = 7 pairs."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "rb_default", RB)
    w = _world(path, tmp_path, results, seeds=(0,))

    scores = report.load_scores("validation", ledger_path=path)

    assert list(dict.fromkeys(scores["label"])) == [
        "rb_default",
        "rb_tuned",
        _label("alpha", w["r"]),
    ]
    assert not any(label.startswith("rb_tuned #") for label in scores["label"])
    rb = scores[scores["label"] == "rb_tuned"]
    assert len(rb) == 3
    assert _vals(rb["registered"]) == [w["e1"]["row_id"]] * 3
    assert _vals(rb["result"]) == [w["e1_result"]] * 3
    assert _vals(rb["decision"]) == [f"adopted {w['e1_decision']}"] * 3
    assert _vals(rb["protocol"]) == [w["e1"]["fingerprints"]["docs/PROTOCOL.md"][:8]] * 3
    assert rb["profit_eur_m2"].tolist() == [-5.0, 1.0, 2.0]
    default = scores[scores["label"] == "rb_default"]
    for col in ("registered", "result", "decision", "protocol"):
        assert _vals(default[col]) == [None] * 3, col
    assert report.check_ledger(scores, ledger_path=path) == 7


def test_a_registration_with_no_result_is_skipped_and_none_of_its_files_read(
    tmp_path, monkeypatch, capsys
):
    """E7's state (#39 is open): a registration with no result row prints `skipped <row>: no
    result row yet`, has no rows, and nothing of it is read: selection._registration never sees
    its id (it has an open run, which _registration refuses) and selection._read never sees its
    run dirs (one holds a selected.json that isn't JSON)."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    w = _world(path, tmp_path, results, seeds=(0,), e1=None)
    e7 = _register(path, issue=39, arms=(("delta", "ppo"),), seeds=(0, 1))
    _, done = _run(path, tmp_path, e7, "delta", 0)
    (done / "selected.json").write_text("not json: reading it raises\n")
    _, running = _run(path, tmp_path, e7, "delta", 1, end=False)
    registrations, reads = [], []
    real_registration, real_read = selection._registration, selection._read

    def registration(row_id, ledger_path):
        registrations.append(row_id)
        return real_registration(row_id, ledger_path)

    def read(method, run_dir):
        reads.append(Path(run_dir).resolve())
        return real_read(method, run_dir)

    monkeypatch.setattr(selection, "_registration", registration)
    monkeypatch.setattr(selection, "_read", read)
    capsys.readouterr()

    scores = report.load_scores("validation", ledger_path=path)

    assert f"skipped {e7['row_id']}: no result row yet" in capsys.readouterr().out
    assert list(dict.fromkeys(scores["label"])) == ["rb_tuned", _label("alpha", w["r"])]
    assert e7["row_id"] not in _vals(scores["registered"])
    assert e7["row_id"] not in registrations
    assert done.resolve() not in reads and running.resolve() not in reads


# --- summarize: like-for-like, gains, P(improvement), flags --------------------------------------


@pytest.mark.parametrize(
    ("n", "profit", "viol_sum"),
    [
        # 15 values, 3 trimmed from each end: (2 x -5 + 5 x 1 + 2 x 2) / 9 = -1/9 EUR/m2; temp
        # (2 x 0 + 5 x 6 + 2 x 9) / 9 = 16/3 h, rh (7 x 3 + 2 x 12) / 9 = 5 h: 31/3 h.
        (5, -1 / 9, 31 / 3),
        # 30 values, 7 trimmed: (3 x -5 + 10 x 1 + 3 x 2) / 16 = 1/16 EUR/m2; temp
        # (3 x 0 + 10 x 6 + 3 x 9) / 16 = 87/16 h, rh (13 x 3 + 3 x 12) / 16 = 75/16 h: 81/8 h.
        (10, 1 / 16, 81 / 8),
    ],
    ids=["5-runs", "10-runs"],
)
def test_rb_tuned_like_for_like_is_its_row_tiled_to_the_arms_run_count(
    n, profit, viol_sum, tmp_path, monkeypatch
):
    """rb_tuned is 1 deterministic run x 3 seasons; an arm is n runs x 3. rb_tuned_profit_iqm on
    the arm's row is stats.iqm_ci of rb_tuned's row tiled n times, and rb_tuned_viol_sum_iqm the
    same per constraint, summed. Known answer (RB): profits (-5, 1, 2), temp (0, 6, 9), rh
    (3, 3, 12), co2 0; the values are in the parametrize comments. rb_tuned's own row is its plain
    mean: profit (-5 + 1 + 2) / 3 = -2/3, total violation hours (3 + 9 + 21) / 3 = 11, and the
    per-constraint sum 0 + 5 + 6 = 11."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "rb_tuned", RB)
    reg = _register(path, issue=36, arms=(("alpha", "ppo"),), seeds=range(n))
    zeros = {s: _seasons() for s in range(n)}
    _arm(path, tmp_path, reg, "alpha", zeros)
    _result(path, reg, {"alpha": [zeros[s] for s in range(n)]})

    summary = report.summarize(report.load_scores("validation", ledger_path=path))

    alpha = _row(summary, _label("alpha", reg))
    assert alpha["n_runs"] == n
    assert alpha["rb_tuned_profit_iqm"] == pytest.approx(profit, abs=1e-12)
    assert alpha["rb_tuned_viol_sum_iqm"] == pytest.approx(viol_sum, abs=1e-12)
    rb = _row(summary, "rb_tuned")
    assert rb["profit_eur_m2_iqm"] == pytest.approx(-2 / 3, abs=1e-12)
    assert rb["viol_hours_total_iqm"] == pytest.approx(11.0, abs=1e-12)
    assert rb["viol_hours_sum_iqm"] == pytest.approx(11.0, abs=1e-12)


# Paired-gain arm, seeds 0-4. D is profit gain over rb_tuned's same season (EUR/m2), S the
# violation hours saved (rb_tuned's total (3, 9, 21) minus the arm's). The arm's own profit is
# RB + D; its total RB - S, split temp and rh as below (co2 0).
D = {0: (1, 2, 3), 1: (4, 5, 6), 2: (7, 8, 9), 3: (10, 11, 12), 4: (-1, -2, -3)}
S = {0: (-5, -2, 4), 1: (-4, -1, 5), 2: (-3, 0, 6), 3: (0, 2, 7), 4: (1, 3, 8)}
GAIN_RH = {s: (0, 0, 7 - s) for s in range(5)}


def _gain_arm() -> dict:
    rb_profit = [x["profit_eur_m2"] for x in RB]
    rb_total = [sum(x[v] for v in VIOL) for x in RB]
    out = {}
    for s in range(5):
        total = [rb_total[i] - S[s][i] for i in range(3)]
        out[s] = _seasons(
            [rb_profit[i] + D[s][i] for i in range(3)],
            temp=[total[i] - GAIN_RH[s][i] for i in range(3)],
            rh=GAIN_RH[s],
        )
    return out


def test_paired_gains_and_p_improvement_over_rb_tuneds_same_season(tmp_path, monkeypatch):
    """Against a 1-run baseline, P(improvement) is the fraction of same-season wins, ties half.

    Known answers (D, S and the arm above; 15 values each, IQM trims 3 from each end):
    - profit gain D is 1..12 and -1, -2, -3: 12 wins of 15, p_improve_profit = 0.8; IQM of 1..9
      = 5.0 EUR/m2; profit_gain_{iqm,lo,hi} = stats.iqm_ci(D by seed).
    - hours saved S: 8 positive, 2 zero, 5 negative: p_improve_viol = (8 + 2/2) / 15 = 0.6; S
      trimmed of -5, -4, -3 and 6, 7, 8 leaves -2..5 with 0 twice, 12 / 9 = 4/3 h.
    - viol_hours_total IQM: totals 2, 3, 6, 6, 7, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17 trimmed to
      6..14, 85 / 9 h; viol_hours_sum_iqm: temp 77/9 + rh 7/9 + co2 0 = 84/9 = 28/3 h, so the
      two "total" columns differ.
    - 5 runs: no flag; like-for-like profit -1/9."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "rb_tuned", RB)
    arm = _gain_arm()
    reg = _register(path, issue=36, arms=(("alpha", "ppo"),), seeds=range(5))
    _arm(path, tmp_path, reg, "alpha", arm)
    _result(path, reg, {"alpha": [arm[s] for s in range(5)]})
    d = np.array([D[s] for s in range(5)], dtype=np.float64)
    saved = np.array([S[s] for s in range(5)], dtype=np.float64)

    summary = report.summarize(report.load_scores("validation", ledger_path=path))

    assert SUMMARY_COLUMNS <= set(summary.columns), SUMMARY_COLUMNS - set(summary.columns)
    alpha = _row(summary, _label("alpha", reg))
    assert alpha["profit_gain_iqm"] == pytest.approx(5.0, abs=1e-12)
    assert _triple(alpha, "profit_gain") == stats.iqm_ci(d)
    assert alpha["p_improve_profit"] == pytest.approx(0.8, abs=1e-12)
    assert alpha["viol_saved_iqm"] == pytest.approx(4 / 3, abs=1e-12)
    assert _triple(alpha, "viol_saved") == stats.iqm_ci(saved)
    assert alpha["p_improve_viol"] == pytest.approx(0.6, abs=1e-12)
    assert alpha["viol_hours_total_iqm"] == pytest.approx(85 / 9, abs=1e-12)
    assert alpha["viol_hours_temp_iqm"] == pytest.approx(77 / 9, abs=1e-12)
    assert alpha["viol_hours_rh_iqm"] == pytest.approx(7 / 9, abs=1e-12)
    assert alpha["viol_hours_sum_iqm"] == pytest.approx(28 / 3, abs=1e-12)
    assert _triple(alpha, "profit_eur_m2") == stats.iqm_ci(_array(arm, "profit_eur_m2"))
    assert alpha["rb_tuned_profit_iqm"] == pytest.approx(-1 / 9, abs=1e-12)
    assert alpha["n_runs"] == 5
    assert _blank(alpha["flag"]), alpha["flag"]


def test_a_crashed_run_is_counted_never_scored_and_n4_is_flagged(tmp_path, monkeypatch):
    """Arm alpha, seeds 0-4: seed 2 crashed, its dir holding a decoy selected.json (1000s, every
    season invalid). It is in no score row, crashed is 1 (scores and summary), n_runs 4, and the
    flag is "n<5: no claim". Seed 0's middle season is invalid: scored, and counted.

    Known answers: profits of seeds 0, 1, 3, 4: (15, 1, 8), (2, 14, 9), (12, 6, 4), (5, 11, 10);
    12 values, 3 trimmed from each end (1, 2, 4 and 12, 14, 15): 49 / 6 EUR/m2. invalid_episodes
    1 (the decoy's 3 aren't read). Like-for-like on 4 runs: profit (-5 + 4 x 1 + 2) / 6 = 1/6
    EUR/m2; temp (0 + 4 x 6 + 9) / 6 = 5.5 h, rh (5 x 3 + 12) / 6 = 4.5 h: 10.0 h."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "rb_tuned", RB)
    reg = _register(path, issue=35, arms=(("alpha", "ppo"),), seeds=range(5))
    table = {s: _seasons(PROFIT[s], valid=(True, s != 0, True)) for s in range(5)}
    runs = {}
    for s in range(5):
        status, seasons = ("crashed", _decoy()) if s == 2 else ("finished", table[s])
        runs[s] = _run(path, tmp_path, reg, "alpha", s, status=status, seasons=seasons)[0]
    scored = (0, 1, 3, 4)
    _result(path, reg, {"alpha": [table[s] for s in scored]})

    scores = report.load_scores("validation", ledger_path=path)
    summary = report.summarize(scores)

    mine = scores[scores["label"] == _label("alpha", reg)]
    assert _vals(mine["run"]) == [runs[s]["row_id"] for s in scored for _ in range(3)]
    assert runs[2]["row_id"] not in _vals(scores["run"])
    assert _vals(mine["crashed"]) == [1] * 12
    alpha = _row(summary, _label("alpha", reg))
    assert (alpha["n_runs"], alpha["crashed"], alpha["invalid_episodes"]) == (4, 1, 1)
    assert alpha["flag"] == NO_CLAIM
    assert alpha["profit_eur_m2_iqm"] == pytest.approx(49 / 6, abs=1e-12)
    want = stats.iqm_ci(_array(table, "profit_eur_m2", order=scored))
    assert _triple(alpha, "profit_eur_m2") == want
    assert alpha["rb_tuned_profit_iqm"] == pytest.approx(1 / 6, abs=1e-12)
    assert alpha["rb_tuned_viol_sum_iqm"] == pytest.approx(10.0, abs=1e-12)
    assert report.check_ledger(scores, ledger_path=path) == 3


def test_baselines_are_flagged_deterministic_or_no_claim(tmp_path, monkeypatch):
    """PROTOCOL §6: rb_default, rb_tuned and zero_controls are deterministic, so 1 run is all they
    need ("deterministic, 1 run"); random isn't, so its 1 run gets "n<5: no claim". Each has
    n_runs 1 and crashed null; invalid_episodes counts valid False (rb_default's middle season).

    Known answer for random, profits (1, 2, 1) against rb_tuned's (-5, 1, 2): 1 run, so its IQM
    is the mean 4/3 EUR/m2, CI [4/3, 4/3]; it wins 2 of 3 seasons, p_improve_profit 2/3; gains
    (6, 1, -1), mean 2.0; like-for-like (n = 1) is rb_tuned's mean, -2/3."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "zero_controls", _seasons((-1, -1, -1)))
    _baseline(results, "random", _seasons((1, 2, 1)))
    _baseline(results, "rb_default", _seasons((0, 0, 0), valid=(True, False, True)))
    _baseline(results, "rb_tuned", RB)

    summary = report.summarize(report.load_scores("validation", ledger_path=path))

    assert summary["label"].tolist() == list(envs.BASELINES)
    flags = dict(zip(summary["label"], summary["flag"], strict=True))
    assert flags == {
        "zero_controls": DETERMINISTIC,
        "random": NO_CLAIM,
        "rb_default": DETERMINISTIC,
        "rb_tuned": DETERMINISTIC,
    }
    assert summary["n_runs"].tolist() == [1, 1, 1, 1]
    assert _vals(summary["crashed"]) == [None] * 4
    assert summary["invalid_episodes"].tolist() == [0, 0, 1, 0]
    random = _row(summary, "random")
    assert _triple(random, "profit_eur_m2") == pytest.approx((4 / 3,) * 3, abs=1e-12)
    assert random["p_improve_profit"] == pytest.approx(2 / 3, abs=1e-12)
    assert random["profit_gain_iqm"] == pytest.approx(2.0, abs=1e-12)
    assert random["rb_tuned_profit_iqm"] == pytest.approx(-2 / 3, abs=1e-12)


def test_shuffled_score_rows_give_the_same_summary(tmp_path, monkeypatch):
    """The bootstrap depends on row order, so the array is always runs by seed x scenarios in
    VALIDATION order. The ledger has alpha's runs in order 3, 0, 4, 1, 2: load_scores gives them
    by seed, and summarize of the rows shuffled is the summary unshuffled, number for number.

    Known answer: PROFIT is 1..15, IQM 8.0; its CI is stats.iqm_ci's on the seed-ordered array,
    which differs from the ledger-ordered one's (ci_high 10.6667 vs 10.7778)."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "rb_tuned", RB)
    reg = _register(path, issue=36, arms=(("alpha", "ppo"),), seeds=range(5))
    table = {s: _seasons(PROFIT[s], temp=(s, 1, 2 * s), yld=(s, s, 1)) for s in range(5)}
    _arm(path, tmp_path, reg, "alpha", table, order=LEDGER_ORDER)
    _result(path, reg, {"alpha": [table[s] for s in range(5)]})
    seed_order = stats.iqm_ci(_array(table, "profit_eur_m2"))
    assert seed_order != stats.iqm_ci(_array(table, "profit_eur_m2", order=LEDGER_ORDER))

    scores = report.load_scores("validation", ledger_path=path)
    shuffled = scores.sample(frac=1, random_state=0).reset_index(drop=True)
    plain = report.summarize(scores).set_index("label")
    mixed = report.summarize(shuffled).set_index("label").loc[plain.index]

    label = _label("alpha", reg)
    assert _vals(scores[scores["label"] == label]["seed"]) == [
        s for s in range(5) for _ in range(3)
    ]
    assert not shuffled["start_day"].equals(scores["start_day"])
    assert plain.loc[label, "profit_eur_m2_iqm"] == pytest.approx(8.0, abs=1e-12)
    assert _triple(plain.loc[label], "profit_eur_m2") == seed_order
    pd.testing.assert_frame_equal(plain, mixed, check_exact=True)


# --- check_ledger ------------------------------------------------------------------------------


def test_check_ledger_passes_on_matching_results_and_counts_pairs(tmp_path, monkeypatch):
    """Every result row whose registration is in the scores is rebuilt from them: E1's
    paper_return and 3 guards (against rb_tuned's parquet rows) and R's alpha and bravo, each
    profit_eur_m2 and 2 guards: 4 + 2 x 3 = 10 (arm, key) pairs, all exactly equal."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _world(path, tmp_path, results, arms=("alpha", "bravo"), seeds=(0, 1))

    scores = report.load_scores("validation", ledger_path=path)

    assert report.check_ledger(scores, ledger_path=path) == 10


@pytest.mark.parametrize(
    "off",
    [
        ("e1", "rb_tuned", "viol_hours_rh", "ci_low"),
        ("r", "alpha", "profit_eur_m2", "iqm"),
        ("r", "bravo", "yield_kg_m2", "ci_high"),
    ],
    ids=["e1-guard-ci_low", "primary-iqm", "guard-ci_high"],
)
def test_check_ledger_refuses_a_result_off_by_1e9_naming_row_arm_and_key(
    off, tmp_path, monkeypatch
):
    """One number of one result row is 1e-9 off what stats.iqm_ci gives on the scores (a primary,
    a guard, or E1's guard against the rb_tuned rows): check_ledger raises a ValueError naming
    that result row, arm and key."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    w = _world(path, tmp_path, results, arms=("alpha", "bravo"), seeds=(0, 1), off=off)
    scores = report.load_scores("validation", ledger_path=path)

    with pytest.raises(ValueError) as e:
        report.check_ledger(scores, ledger_path=path)

    said = str(e.value)
    row, arm, key, _ = off
    assert w[f"{row}_result"] in said, said
    assert arm in said, said
    assert key in said, said


# --- main ------------------------------------------------------------------------------------------


def _md_table(text: str) -> list[tuple[str, str]]:
    """summary.md's pipe-table body rows: (first cell, line), header and separator dropped."""
    lines = [line for line in text.splitlines() if line.lstrip().startswith("|")]
    body = [line for line in lines[1:] if not re.fullmatch(r"\s*\|[\s:|-]+\|\s*", line)]
    return [(line.strip().strip("|").split("|")[0].strip(), line) for line in body]


def test_main_writes_six_files_identical_tables_across_out_dirs(tmp_path, monkeypatch, capsys):
    """main --split validation --out DIR writes DIR/validation/{scores.parquet, summary.csv,
    summary.md} and, since #47, the three figures {per_scenario, profiles, tradeoff}.png (PNGs),
    six files in all; it prints `matches the ledger: 7 (arm, key) pairs` (E1's 4 + alpha's 3) and
    a line per label. A second --out, already holding a previous version of the three tables, gets
    them replaced, with summary.csv and summary.md byte-identical to the first's (no timestamp, no
    out path).

    - scores.parquet is load_scores' frame; summary.csv is summarize's at full precision (alpha's
      profit triple is stats.iqm_ci's exactly, read back with float_precision="round_trip").
    - summary.md: line 1 names `python -m greenleaf_control.report` and summary.csv; then a pipe
      table with one row per label, first cell the label, in summary order. Known answers for
      alpha (seeds 0, 1; 6 values, 1 trimmed from each end), 3 decimals and hours 2:
      profit (1, 2, 4), (11, 12, 14) -> (2 + 4 + 11 + 12) / 4 = 7.250 EUR/m2; like-for-like
      (-5, 1, 2) x 2 -> (-5 + 1 + 1 + 2) / 4 = -0.250; Σ per-constraint IQMs temp (0, 0, 3),
      (1, 2, 3) -> 1.5 h + rh (1, 0, 0), (1, 0, 1) -> 0.5 h = 2.00 h; rb_tuned's on 2 runs:
      temp (0 + 6 + 6 + 9) / 4 = 5.25 + rh (3 + 3 + 3 + 12) / 4 = 5.25 = 10.50 h."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    w = _world(path, tmp_path, results, seeds=(0, 1))
    alpha = _label("alpha", w["r"])
    out1, out2 = tmp_path / "out1", tmp_path / "elsewhere" / "out2"
    d1, d2 = out1 / "validation", out2 / "validation"
    d2.mkdir(parents=True)
    for name in ("scores.parquet", "summary.csv", "summary.md"):
        (d2 / name).write_text("a previous version\n")
    capsys.readouterr()

    assert report.main(["--split", "validation", "--out", str(out1)], ledger_path=path) == 0
    printed = capsys.readouterr().out
    assert report.main(["--split", "validation", "--out", str(out2)], ledger_path=path) == 0

    assert sorted(p.name for p in d1.iterdir()) == [
        "per_scenario.png",
        "profiles.png",
        "scores.parquet",
        "summary.csv",
        "summary.md",
        "tradeoff.png",
    ]
    for name in FIGURES:
        assert (d1 / f"{name}.png").read_bytes()[:8] == PNG, name
    for name in ("summary.csv", "summary.md"):
        text = (d1 / name).read_text()
        assert (d2 / name).read_bytes() == (d1 / name).read_bytes(), name
        assert str(tmp_path) not in text, name
        assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}", text), name

    lines = printed.splitlines()
    [at] = [i for i, line in enumerate(lines) if "matches the ledger: 7 (arm, key) pairs" in line]
    for label in ("rb_tuned", alpha):
        assert any(label in line for line in lines[at + 1 :]), (label, printed)

    scores = pd.read_parquet(d1 / "scores.parquet").reset_index(drop=True)
    want = report.load_scores("validation", ledger_path=path).reset_index(drop=True)
    pd.testing.assert_frame_equal(scores, want, check_dtype=False)
    replaced = pd.read_parquet(d2 / "scores.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(replaced, want, check_dtype=False)

    # pandas' default float parser isn't exact; round_trip reads back what to_csv wrote.
    summary = pd.read_csv(d1 / "summary.csv", float_precision="round_trip")
    assert summary["label"].tolist() == ["rb_tuned", alpha]
    row = _row(summary, alpha)
    assert _triple(row, "profit_eur_m2") == stats.iqm_ci(
        _array(ALPHA, "profit_eur_m2", order=(0, 1))
    )

    md = (d1 / "summary.md").read_text()
    first = md.splitlines()[0]
    assert "python -m greenleaf_control.report" in first and "summary.csv" in first, first
    table = _md_table(md)
    assert [cell for cell, _ in table] == ["rb_tuned", alpha], md
    line = dict(table)[alpha]
    for number in ("7.250", "-0.250", "2.00", "10.50"):
        assert re.search(rf"(?<![\d.]){re.escape(number)}(?!\d)", line), (number, line)


# (case, names): case breaks one thing in _world's base (rb_tuned's parquet, E1, R's alpha at
# seeds 0 and 1); the refusal must name, for each group in names, one of its strings ("{key}" is
# an id _broken returns).
REFUSALS = [
    pytest.param(
        "no-selected-json", [["run selection first"], ["{run}", "{run_dir}"]], id="no-selected-json"
    ),
    pytest.param("stale-selected-json", [["selected.json"]], id="stale-selected-json"),
    pytest.param("unknown-controller", [["rb_fancy"]], id="unknown-baseline-controller"),
    pytest.param("run-dir-controller", [["elsewhere"]], id="run-dir-controller"),
    pytest.param("duplicate", [["rb_tuned"]], id="duplicate-controller-and-scenario"),
    pytest.param("two-decisions", [["{r_result}", "{r_decision}", "{second}"]], id="two-decisions"),
    pytest.param("second-rb_tuned-arm", [["{e1}", "{e1b}"]], id="second-rb_tuned-arm"),
    pytest.param("rb_tuned-crashed", [["{e1}", "{e1_result}"]], id="rb_tuned-arm-all-crashed"),
    pytest.param("no-rb_tuned", [["rb_tuned"]], id="no-rb_tuned-rows"),
    pytest.param("off", [["{r_result}"], ["alpha"], ["viol_hours_temp"]], id="result-off-by-1e-9"),
    pytest.param("two-results", [["{r}"]], id="two-results-on-a-registration"),
    pytest.param("arm-all-crashed", [["{dead}"], ["charlie"]], id="learning-arm-all-crashed"),
    pytest.param("baseline-short", [["rb_default"]], id="baseline-missing-a-scenario"),
]


def _broken(case: str, path: Path, root: Path, results: Path) -> dict:
    """_world's base with case's one thing wrong; the ids a refusal may name."""
    kwargs = {
        "rb_tuned-crashed": {"e1": "crashed"},
        "no-rb_tuned": {"e1": None},
        "off": {"off": ("r", "alpha", "viol_hours_temp", "ci_high")},
    }.get(case, {})
    w = _world(path, root, results, seeds=(0, 1), **kwargs)
    ids = {
        "r": w["r"]["row_id"],
        "r_result": w["r_result"],
        "r_decision": w["r_decision"],
        "run": w["runs"]["alpha", 1]["row_id"],
        "run_dir": str(w["dirs"]["alpha", 1]),
    }
    if "e1" in w:
        ids.update(e1=w["e1"]["row_id"], e1_result=w["e1_result"])
    seed1 = w["dirs"]["alpha", 1]
    if case == "no-selected-json":
        (seed1 / "selected.json").unlink()
    elif case == "stale-selected-json":
        (seed1 / "selected.json").unlink()
        _write_selected(seed1, "ppo", ALPHA[1], season_length=2)
    elif case == "unknown-controller":
        _baseline(results, "rb_fancy", RB)
    elif case == "run-dir-controller":
        _baseline(results, str(root / "elsewhere" / "seed0"), RB, file="run_validation.parquet")
    elif case == "duplicate":
        _baseline(results, "rb_tuned", RB, file="rb_tuned_validation_again.parquet")
    elif case == "two-decisions":
        second = ledger.decide(result=w["r_result"], decision="adopted", note="2", path=path)
        ids["second"] = second["row_id"]
    elif case == "second-rb_tuned-arm":
        e1b = _register_e1(path, issue=23)
        _rb_tuned_run(path, root, e1b)
        _result(path, e1b, {"rb_tuned": [RB]})
        ids["e1b"] = e1b["row_id"]
    elif case == "no-rb_tuned":
        (results / "rb_tuned_validation.parquet").unlink()
        _baseline(results, "rb_default", RB)
    elif case == "two-results":
        # The ledger doesn't refuse a second result row on R; the same numbers, so every result
        # row still matches the scores and only the one-result check can refuse.
        _result(path, w["r"], {"alpha": [ALPHA[0], ALPHA[1]]})
    elif case == "arm-all-crashed":
        # A registration (#37) whose one arm crashed at both seeds: no selected.json, per_arm null.
        dead = _register(path, issue=37, arms=(("charlie", "ppo"),), seeds=(0, 1))
        for s in (0, 1):
            _run(path, root, dead, "charlie", s, status="crashed")
        _result(path, dead, {"charlie": []})
        ids["dead"] = dead["row_id"]
    elif case == "baseline-short":
        # rb_default's parquet holds VALIDATION's first two seasons, not the third.
        _baseline(results, "rb_default", RB[:2])
    return ids


@pytest.mark.parametrize(("case", "names"), REFUSALS)
def test_main_refuses_before_writing_anything(case, names, tmp_path, monkeypatch, capsys):
    """Each case is _world's base with one thing wrong: a run with no selected.json, a stale one
    (season_length 2), a baseline controller that isn't in envs.BASELINES (a name, or a run dir
    evaluate scored), a (controller, scenario) in two parquets, two decision rows on one result,
    a second rb_tuned-method arm, an rb_tuned arm whose run crashed, no rb_tuned rows, a result
    1e-9 off, two result rows on one registration (the ledger allows it), a learning arm whose runs
    all crashed, a baseline parquet missing one of the split's seasons. main refuses (ValueError,
    or a non-zero exit) naming it, never prints "matches the ledger", plays nothing, and writes
    nothing: --out isn't made and every file under tmp_path is as it was. Any other exception (a
    bare KeyError or IndexError) isn't a refusal: _refused lets it through and the test fails."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    ids = _broken(case, path, tmp_path, results)
    out = tmp_path / "analysis"
    before = _tree(tmp_path)

    said = _refused(report, ["--split", "validation", "--out", str(out)], path, capsys)

    for group in names:
        group = [ids[n.strip("{}")] if n.startswith("{") else n for n in group]
        assert any(n in said for n in group), f"the refusal names none of {group}: {said!r}"
    assert "matches the ledger" not in said
    assert not out.exists()
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("split", list(scenarios.TEST))
def test_a_test_split_is_refused_before_any_parquet_is_read(split, tmp_path, monkeypatch):
    """A1a is VALIDATION only (the 2012 path follows F1): `--split <test split>` is an argparse
    error (SystemExit 2), and load_scores(<test split>) a ValueError naming it. Neither reads a
    parquet (a spy on pd.read_parquet) or makes --out."""
    report, results = _report(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    _world(path, tmp_path, results, seeds=(0,))
    reads = []
    real = pd.read_parquet

    def read_parquet(*args, **kwargs):
        reads.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    out = tmp_path / "analysis"

    with pytest.raises(SystemExit) as e:
        report.main(["--split", split, "--out", str(out)], ledger_path=path)
    assert e.value.code == 2
    with pytest.raises(ValueError, match=split):
        report.load_scores(split, ledger_path=path)
    assert reads == []
    assert not out.exists()


# --- figures (#47) -----------------------------------------------------------------------------

GAIN_KEY = "profit gain over rb_tuned's same season [€/m²]"
SAVED_KEY = "violation hours saved vs rb_tuned's same season [h]"
COMMAND = "python -m greenleaf_control.report --split validation"
# zero_controls: profits -1, so gains (4, -2, -3) over RB's (-5, 1, 2); no violation hours, so
# hours saved (3, 9, 21).
ZERO = _seasons((-1, -1, -1))


def _spies(monkeypatch) -> dict[str, list[dict]]:
    """Wrap viz.per_scenario, viz.profiles and viz.tradeoff: each still draws and returns its
    Figure, and each call is recorded as its arguments bound by name (defaults applied), the
    Figure ("fig"), and whether the three tables were already in the PNG's dir ("after_tables")."""
    from greenleaf_control import viz

    calls = {}
    for name in FIGURES:
        assert hasattr(viz, name), f"greenleaf_control.viz.{name} doesn't exist yet (#47)"
        calls[name] = []
        real = getattr(viz, name)

        def spy(*args, _real=real, _calls=calls[name], **kwargs):
            bound = inspect.signature(_real).bind(*args, **kwargs)
            bound.apply_defaults()
            here = Path(bound.arguments["out"]).parent
            tables = ("scores.parquet", "summary.csv", "summary.md")
            fig = _real(*args, **kwargs)
            _calls.append(
                {
                    **bound.arguments,
                    "fig": fig,
                    "after_tables": all((here / t).is_file() for t in tables),
                }
            )
            return fig

        monkeypatch.setattr(viz, name, spy)
    return calls


def _figure_world(path: Path, root: Path, results: Path) -> dict[str, dict]:
    """The figures' ledger and files, tie-free in profit against rb_tuned, four labels (summarize
    bootstraps each, about 1.3 s a label on the Mac, so no more):
    - parquets: zero_controls (ZERO) and rb_tuned (RB); no random or rb_default;
    - #36: alpha (ppo), seeds 0-4, _gain_arm's seasons (profit gains D: 1..12 and -1, -2, -3;
      hours saved S), its result and an adopted decision;
    - #37: bravo (residual), seeds 0 and 1, BRAVO's seasons, its result and a dropped decision.
    Returns {label: seed -> seasons} in the summary's order (baselines, then #36, then #37)."""
    _baseline(results, "zero_controls", ZERO)
    _baseline(results, "rb_tuned", RB)
    tables = {"zero_controls": {0: ZERO}, "rb_tuned": {0: RB}}
    for issue, arm, method, table, decision in (
        (36, "alpha", "ppo", _gain_arm(), "adopted"),
        (37, "bravo", "residual", {s: BRAVO[s] for s in (0, 1)}, "dropped"),
    ):
        reg = _register(path, issue=issue, arms=((arm, method),), seeds=sorted(table))
        _arm(path, root, reg, arm, table, method=method)
        res = _result(path, reg, {arm: [table[s] for s in sorted(table)]})
        ledger.decide(result=res["row_id"], decision=decision, note=arm, path=path)
        tables[_label(arm, reg)] = table
    return tables


def _totals(by_seed: dict) -> np.ndarray:
    """viol_hours_total's runs x VALIDATION array (co2 + temp + rh), rows by seed."""
    return sum(_array(by_seed, v) for v in VIOL)


def _above(line, tau: float) -> float:
    """A complementary-ECDF line's y at tau. ax.ecdf(complementary=True) draws steps-pre: on
    x[i - 1] < tau <= x[i] the line reads y[i]."""
    assert line.get_drawstyle() == "steps-pre", line.get_drawstyle()
    x, y = (np.asarray(d, dtype=float) for d in (line.get_xdata(), line.get_ydata()))
    assert x[0] < tau <= x[-1], (tau, x)
    return float(y[np.searchsorted(x, tau, side="left")])


def _draw(report, monkeypatch, tmp_path: Path) -> tuple[dict, dict, pd.DataFrame, Path]:
    """_figure_world, then main --split validation --out tmp_path/analysis under the spies.
    Returns (calls, tables, the summary.csv main wrote, read back exactly, out)."""
    calls = _spies(monkeypatch)
    path = tmp_path / "ledger.jsonl"
    tables = _figure_world(path, tmp_path, selection.RESULTS)
    out = tmp_path / "analysis"
    assert report.main(["--split", "validation", "--out", str(out)], ledger_path=path) == 0
    for name in FIGURES:
        assert len(calls[name]) == 1, f"main drew {name} {len(calls[name])} times, not once"
    summary = pd.read_csv(out / "validation" / "summary.csv", float_precision="round_trip")
    return calls, tables, summary, out


def test_per_scenario_draws_raw_profit_and_total_hours_for_every_label(tmp_path, monkeypatch):
    """per_scenario.png: two panels, profit then total violation hours, each holding every label
    in the summary's order (zero_controls, rb_tuned, the adopted alpha, the dropped bravo) with
    its raw runs (by seed) x scenarios array from the scores: rb_tuned's profit row is (-5, 1, 2)
    and its hours (3, 9, 21); alpha's are RB + D and RB's totals - S. The scenario labels are
    `<location> day <start_day>` in VALIDATION order."""
    report, _ = _report(monkeypatch, tmp_path)
    calls, tables, _, out = _draw(report, monkeypatch, tmp_path)

    [call] = calls["per_scenario"]
    assert list(call["scenario_labels"]) == [
        "Amsterdam day 45",
        "Amsterdam day 59",
        "Amsterdam day 75",
    ]
    keys = list(call["panels"])
    assert len(keys) == 2 and "profit" in keys[0] and "violation" in keys[1], keys
    profit, hours = (call["panels"][k] for k in keys)
    assert list(profit) == list(tables) and list(hours) == list(tables)
    for label, table in tables.items():
        np.testing.assert_array_equal(profit[label], _array(table, "profit_eur_m2"), label)
        np.testing.assert_array_equal(hours[label], _totals(table), label)
    np.testing.assert_array_equal(profit["rb_tuned"], [[-5.0, 1.0, 2.0]])
    np.testing.assert_array_equal(hours["rb_tuned"], [[3.0, 9.0, 21.0]])
    assert Path(call["out"]) == out / "validation" / "per_scenario.png"


def test_profiles_draw_every_label_but_rb_tuned_and_read_p_improve_just_above_zero(
    tmp_path, monkeypatch
):
    """profiles.png: panels GAIN_KEY (arm - rb_tuned's same season) and SAVED_KEY (rb_tuned's same
    season - arm), every label but rb_tuned, as runs x scenarios arrays; highlight is the adopted
    labels (alpha), never the dropped bravo.

    Known answers (alpha, 15 values each): its profit gains are D, 1..12 and -1, -2, -3, so just
    above tau = 0 its profile reads 12 / 15 = 0.8, which is p_improve_profit (no ties). Its hours
    saved S has 8 positive values, 2 zeros and 5 negative: the profile reads 8 / 15 just above 0,
    while P(fewer hours) counts the ties half, 0.6."""
    report, _ = _report(monkeypatch, tmp_path)
    calls, tables, summary, _ = _draw(report, monkeypatch, tmp_path)
    alpha, bravo = list(tables)[2:]

    [call] = calls["profiles"]
    assert list(call["panels"]) == [GAIN_KEY, SAVED_KEY]
    gains, saved = call["panels"][GAIN_KEY], call["panels"][SAVED_KEY]
    drawn = [label for label in tables if label != "rb_tuned"]
    assert list(gains) == drawn and list(saved) == drawn
    rb_profit, rb_total = _array({0: RB}, "profit_eur_m2"), _totals({0: RB})
    for label in drawn:
        want = _array(tables[label], "profit_eur_m2") - rb_profit
        np.testing.assert_array_equal(gains[label], want, label)
        np.testing.assert_array_equal(saved[label], rb_total - _totals(tables[label]), label)
    np.testing.assert_array_equal(gains[alpha], [D[s] for s in range(5)])
    np.testing.assert_array_equal(saved[alpha], [S[s] for s in range(5)])
    np.testing.assert_array_equal(gains["zero_controls"], [[4.0, -2.0, -3.0]])
    assert list(call["highlight"]) == [alpha], (call["highlight"], bravo)

    axes = {ax.get_xlabel(): ax for ax in call["fig"].axes}
    lines = {k: {ln.get_label(): ln for ln in axes[k].get_lines()} for k in (GAIN_KEY, SAVED_KEY)}
    above = _above(lines[GAIN_KEY][f"{alpha} (n=15)"], 1e-9)
    assert above == pytest.approx(0.8, abs=1e-12)
    assert _row(summary, alpha)["p_improve_profit"] == pytest.approx(above, abs=1e-12)
    assert _above(lines[SAVED_KEY][f"{alpha} (n=15)"], 1e-9) == pytest.approx(8 / 15, abs=1e-12)
    assert _row(summary, alpha)["p_improve_viol"] == pytest.approx(0.6, abs=1e-12)


def test_tradeoff_points_are_summarys_gain_and_saved_columns_with_rb_tuned_at_the_origin(
    tmp_path, monkeypatch
):
    """tradeoff.png: one point per row of the summary.csv main wrote, in its order, the dropped arm
    and the baselines included: "viol_hours" is the row's viol_saved_{iqm,lo,hi} and "profit" its
    profit_gain_{iqm,lo,hi} (viz.tradeoff's point keys), with the report's axis labels. rb_tuned
    is exactly (0, 0, 0) on both, the origin. Known answers for alpha (as in
    test_paired_gains_and_p_improvement_over_rb_tuneds_same_season): profit gain IQM 5.0 €/m²,
    hours saved IQM 4/3 h.

    Each point's "status" is its decision's call: alpha "adopted", bravo "dropped", the baselines
    None. A learning label under 5 runs carries the n<5 flag in its drawn direct label (the text on
    the axes starting with the label): bravo's 2 runs read "(n=2, no claim)"; alpha's 5 runs and
    the baselines carry no "no claim"."""
    report, _ = _report(monkeypatch, tmp_path)
    calls, tables, summary, out = _draw(report, monkeypatch, tmp_path)
    alpha, bravo = list(tables)[2:]

    [call] = calls["tradeoff"]
    points = call["points"]
    assert [p["label"] for p in points] == summary["label"].tolist() == list(tables)
    for p, row in zip(points, summary.to_dict("records"), strict=True):
        profit, saved = _triple(row, "profit_gain"), _triple(row, "viol_saved")
        assert tuple(p["profit"]) == pytest.approx(profit, abs=1e-12), p
        assert tuple(p["viol_hours"]) == pytest.approx(saved, abs=1e-12), p
    by_label = {p["label"]: p for p in points}
    rb = by_label["rb_tuned"]
    assert (tuple(rb["profit"]), tuple(rb["viol_hours"])) == ((0.0,) * 3, (0.0,) * 3)
    assert by_label[alpha]["profit"][0] == pytest.approx(5.0, abs=1e-12)
    assert by_label[alpha]["viol_hours"][0] == pytest.approx(4 / 3, abs=1e-12)
    assert (call["xlabel"], call["ylabel"]) == (
        "violation hours saved vs rb_tuned [h]",
        "profit gain over rb_tuned [€/m²]",
    )
    assert Path(call["out"]) == out / "validation" / "tradeoff.png"
    status = {label: p.get("status") for label, p in by_label.items()}
    assert status == {"zero_controls": None, "rb_tuned": None, alpha: "adopted", bravo: "dropped"}
    texts = [t.get_text() for ax in call["fig"].axes for t in ax.texts]
    drawn = {label: [t for t in texts if t.startswith(label)] for label in tables}
    assert any("(n=2, no claim)" in t for t in drawn[bravo]), (bravo, texts)
    for label in ("zero_controls", "rb_tuned", alpha):
        assert drawn[label], (label, texts)
        assert not any("no claim" in t for t in drawn[label]), (label, drawn[label])


def test_every_figure_is_drawn_after_the_tables_with_the_commands_footer(tmp_path, monkeypatch):
    """Each of the three figures is drawn once, after scores.parquet, summary.csv and summary.md
    are written (so only once every check passed), into <out>/validation/<name>.png (a PNG). Its
    footer names `python -m greenleaf_control.report --split validation` and no path under
    tmp_path (the --out dir), so two --out dirs get the same footer. It says the figure is
    "derived from summary.csv / scores.parquet", not that "every point is in" them (the
    orchestrator's wording on #47: a profile line or a gain array is computed from the scores, not
    stored in them)."""
    report, _ = _report(monkeypatch, tmp_path)
    calls, _, _, out = _draw(report, monkeypatch, tmp_path)

    for name in FIGURES:
        [call] = calls[name]
        assert call["after_tables"], f"{name} was drawn before the tables were written"
        footer = call["footer"]
        assert isinstance(footer, str) and COMMAND in footer, (name, footer)
        assert str(tmp_path) not in footer, (name, footer)
        assert "derived from summary.csv / scores.parquet" in footer, (name, footer)
        assert "every point is in" not in footer, (name, footer)
        png = out / "validation" / f"{name}.png"
        assert Path(call["out"]) == png, name
        assert png.read_bytes()[:8] == PNG, name


def test_note_calls_hold_34s_dropped_residual_w4():
    """A decision row is one call on a whole registration, but its note can call arms one by one:
    #34's decision-b159ac9a adopts the registration (residual_w05, residual_w1, residual_w4) and
    its note drops residual_w4. report.NOTE_CALLS maps (decision row id, arm) to that arm's own
    call, so the figures don't ink residual_w4 as adopted. Pinned from the orchestrator's reading
    of the note; the real ledger isn't read here."""
    report = _module()
    assert hasattr(report, "NOTE_CALLS"), "report.NOTE_CALLS doesn't exist yet (#47)"
    assert report.NOTE_CALLS.get(("decision-b159ac9a", "residual_w4")) == "dropped"


def test_a_note_call_drops_one_arm_of_an_adopted_registration(tmp_path, monkeypatch):
    """One registration (#34) with arms alpha (ppo) and bravo (residual), seeds 0-4 (ALPHA, BRAVO),
    one result and one "adopted" decision; NOTE_CALLS patched to {(that decision, "bravo"):
    "dropped"}, as its note would call it. Then the tradeoff's statuses are alpha "adopted", bravo
    "dropped" (its note call beats the registration's), the baselines (zero_controls, rb_tuned)
    None; profiles highlights alpha only."""
    report, results = _report(monkeypatch, tmp_path)
    assert hasattr(report, "NOTE_CALLS"), "report.NOTE_CALLS doesn't exist yet (#47)"
    calls = _spies(monkeypatch)
    path = tmp_path / "ledger.jsonl"
    _baseline(results, "zero_controls", ZERO)
    _baseline(results, "rb_tuned", RB)
    reg = _register(path, issue=34, arms=(("alpha", "ppo"), ("bravo", "residual")), seeds=range(5))
    _arm(path, tmp_path, reg, "alpha", ALPHA)
    _arm(path, tmp_path, reg, "bravo", BRAVO, method="residual")
    scored = {"alpha": [ALPHA[s] for s in range(5)], "bravo": [BRAVO[s] for s in range(5)]}
    res = _result(path, reg, scored)
    note = "adopt alpha; bravo dropped"
    dec = ledger.decide(result=res["row_id"], decision="adopted", note=note, path=path)
    monkeypatch.setattr(report, "NOTE_CALLS", {(dec["row_id"], "bravo"): "dropped"})
    out = tmp_path / "analysis"

    assert report.main(["--split", "validation", "--out", str(out)], ledger_path=path) == 0

    alpha, bravo = _label("alpha", reg), _label("bravo", reg)
    assert [len(calls[name]) for name in ("tradeoff", "profiles")] == [1, 1]
    status = {p["label"]: p.get("status") for p in calls["tradeoff"][0]["points"]}
    assert status == {"zero_controls": None, "rb_tuned": None, alpha: "adopted", bravo: "dropped"}
    assert list(calls["profiles"][0]["highlight"]) == [alpha]


# --- The contract ------------------------------------------------------------------------------


def test_contract_and_no_test_wrote_the_real_ledger_or_results():
    """ANALYSIS is control/results/analysis, BASELINE "rb_tuned", METRICS the issue's seven, and
    the signatures are the issue's: load_scores(split, *, ledger_path=ledger.LEDGER),
    summarize(scores), check_ledger(scores, *, ledger_path), main(argv=None, *,
    ledger_path=ledger.LEDGER), and #47's figures(scores, summary, out_dir).

    This test runs last in the module: the real ledger's size and mtime, and the listing of
    control/results/, are as they were when the module was collected. figures' signature is
    checked after them, so that check runs while figures is missing."""
    report = _module()

    assert report.ANALYSIS == CONTROL / "results" / "analysis"
    assert report.BASELINE == "rb_tuned"
    assert tuple(report.METRICS) == METRICS
    for fn, names, keyword in [
        (report.load_scores, ["split", "ledger_path"], "ledger_path"),
        (report.summarize, ["scores"], None),
        (report.check_ledger, ["scores", "ledger_path"], "ledger_path"),
        (report.main, ["argv", "ledger_path"], "ledger_path"),
    ]:
        params = inspect.signature(fn).parameters
        assert list(params) == names, fn.__name__
        if keyword:
            assert params[keyword].kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
    for fn in (report.load_scores, report.main):
        assert inspect.signature(fn).parameters["ledger_path"].default == ledger.LEDGER
    assert inspect.signature(report.main).parameters["argv"].default is None
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RESULTS) == REAL_RESULTS_BEFORE
    assert hasattr(report, "figures"), "report.figures doesn't exist yet (#47)"
    assert list(inspect.signature(report.figures).parameters) == ["scores", "summary", "out_dir"]

"""Checkpoint selection (PROTOCOL §5, docs/API-CONTRACT.md): the rule on VALIDATION only.

select_checkpoint scores each of a run's checkpoints, ckpt_<steps>.zip in numeric order of steps
then ckpt_final.zip, with evaluate.run_policy (deterministic, frozen stats) on scenarios.VALIDATION
in 60-day seasons, and picks one:
- residual_lag: the highest mean profit_eur_m2 among checkpoints whose mean per-step cost_<c>,
  pooled over the seasons, is <= budget x 1.10 on every constraint with a nonzero budget; if none
  qualifies, the lowest total cost (the plain sum over constraints of those means).
- every other arm: the highest mean paper_return (metrics', not paper_reward_sum).
A tie goes to the earliest checkpoint; an invalid season is scored, never dropped (PROTOCOL §4); a
non-finite score raises. run_dir/selected.json is written once.

`python -m greenleaf_control.selection --registered ROW [--out FILE]` turns a registration's runs
into the numbers `ledger result --numbers` records (PROTOCOL §6), written once to --out (default
results/validation/<row>.json):
- It refuses, before anything is scored or written, a row it can't score whole: an unknown row,
  scenarios other than ["validation"], an arm that isn't a METHODS entry or rb_tuned, an open run,
  an arm x seed with no run or with two that didn't crash, an existing --out, a primary or guard
  that isn't a RESULT_KEYS entry, and a selected.json not made on VALIDATION in 60-day seasons.
- Each run that didn't crash is scored on VALIDATION: a learning run from its chosen checkpoint's
  per_scenario results (its selected.json, or select_checkpoint's), rb_tuned with evaluate.run_rb
  on its run dir's rule_based_tuned.yml.
- per_arm[arm] is stats.iqm_ci of the primary and each guard on the runs (by seed) x VALIDATION
  array, null with no run to score, and invalid_episodes; selected maps each learning run to its
  checkpoint. crashed and spend_usd are the ledger's to add.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from greenleaf_control import envs, evaluate, ledger, metrics, scenarios, stats

TOLERANCE = 1.10  # PROTOCOL §5: a mean cost qualifies up to budget x 1.10, inclusive
RESULTS = Path(__file__).resolve().parents[1] / "results" / "validation"
# A scored season's numeric keys, what a primary or guard may name; the ledger's, which register
# checks against.
RESULT_KEYS = ledger.RESULT_KEYS
NO_CLAIM = "n<5: no claim"


def _candidates(run_dir: Path) -> list[str]:
    """The checkpoints as run_policy takes them: "<steps>" in numeric order, then "final"."""
    stems = [p.stem.removeprefix("ckpt_") for p in run_dir.glob("ckpt_*.zip")]
    names = sorted((s for s in stems if s.isdecimal()), key=int)
    if "final" in stems:
        names.append("final")
    if not names:
        raise FileNotFoundError(f"no ckpt_<steps>.zip or ckpt_final.zip in {run_dir}")
    return names


def _mean(values, what: str) -> float:
    """The plain mean (NaN propagates, unlike pandas'); non-finite raises, since a NaN silently
    loses every comparison and an inf silently wins."""
    mean = float(np.asarray(values, dtype=np.float64).mean())
    if not np.isfinite(mean):
        raise ValueError(f"{what} is {mean}: a non-finite score can't be ranked")
    return mean


def _score(run_dir: Path, checkpoint: str, budgets: list[float] | None) -> dict:
    """One checkpoint's VALIDATION seasons and means; with budgets (residual_lag), its pooled mean
    per-step cost on each constraint and whether each with a nonzero budget is within budget x
    TOLERANCE."""
    lag = budgets is not None
    results, frames = [], []
    for s in scenarios.VALIDATION:
        result, frame = evaluate.run_policy(
            run_dir, checkpoint, s, evaluate.SEASON_LENGTH, record_steps=lag
        )
        results.append(result)
        frames.append(frame)
    scored = {
        "per_scenario": results,
        "mean_paper_return": _mean(
            [r["paper_return"] for r in results], f"{checkpoint}'s mean paper_return"
        ),
        "mean_profit_eur_m2": _mean(
            [r["profit_eur_m2"] for r in results], f"{checkpoint}'s mean profit_eur_m2"
        ),
        "mean_cost": None,
        "qualifies": None,
    }
    if lag:
        pooled = pd.concat(frames, ignore_index=True)
        mean_cost = {
            c: _mean(pooled[f"cost_{c}"].to_numpy(), f"{checkpoint}'s mean cost_{c}")
            for c in metrics.CONSTRAINTS
        }
        scored["mean_cost"] = mean_cost
        scored["qualifies"] = all(
            mean_cost[c] <= b * TOLERANCE
            for c, b in zip(metrics.CONSTRAINTS, budgets, strict=True)
            if b > 0
        )
    return scored


def select_checkpoint(run_dir: Path) -> str:
    """The PROTOCOL §5 rule on VALIDATION; writes run_dir/selected.json once and returns the
    checkpoint as run_policy takes it ("<steps>" or "final")."""
    run_dir = Path(run_dir)
    out = run_dir / "selected.json"
    if out.exists():  # before scoring: selection plays 3 seasons of 60 days per checkpoint
        raise FileExistsError(f"{out} already exists; a selection is written once")
    config = json.loads((run_dir / "manifest.json").read_text())["config"]
    lag = config["method"] == "residual_lag"
    budgets = config["budgets"] if lag else None
    candidates = {name: _score(run_dir, name, budgets) for name in _candidates(run_dir)}

    # max and min return the first of equal keys, so a tie goes to the earliest checkpoint.
    qualifying = [name for name, c in candidates.items() if c["qualifies"]]
    if not lag:
        rule = "max_mean_paper_return"
        checkpoint = max(candidates, key=lambda name: candidates[name]["mean_paper_return"])
    elif qualifying:
        rule = "max_mean_profit_within_budget"
        checkpoint = max(qualifying, key=lambda name: candidates[name]["mean_profit_eur_m2"])
    else:
        rule = "min_total_cost"
        checkpoint = min(candidates, key=lambda name: sum(candidates[name]["mean_cost"].values()))

    selected = {
        "checkpoint": checkpoint,
        "method": config["method"],
        "rule": rule,
        "scenarios": scenarios.VALIDATION,
        "season_length": evaluate.SEASON_LENGTH,
        "budgets": budgets,
        "tolerance": TOLERANCE if lag else None,
        "candidates": candidates,
    }
    # allow_nan=False: a NaN or inf anywhere raises here, before the file is opened.
    text = json.dumps(selected, indent=2, sort_keys=True, allow_nan=False)
    with out.open("x") as f:
        f.write(text + "\n")
    return checkpoint


def _registration(row_id: str, ledger_path: Path) -> tuple[dict, dict[str, list[dict]]]:
    """The registered row and, per arm, the run rows to score (every run that didn't crash),
    sorted by seed; refuses a row that can't be scored whole."""
    existing = ledger.rows(path=ledger_path)
    found = [r for r in existing if r["kind"] == "registered" and r["row_id"] == row_id]
    if not found:
        raise ValueError(f"--registered {row_id}: no registered row with that id in {ledger_path}")
    [reg] = found
    if reg["scenarios"] != ["validation"]:
        raise ValueError(
            f"{row_id}: results are scored on exactly ['validation'] (PROTOCOL §2, §5); its"
            f" scenarios are {reg['scenarios']}"
        )
    for arm in reg["arms"]:
        if arm["method"] not in (*envs.METHODS, "rb_tuned"):
            raise ValueError(
                f"{row_id}: arm {arm['name']}'s method {arm['method']!r} has no runs to select;"
                f" selection scores {envs.METHODS} and rb_tuned (evaluate --controller scores"
                " the other baselines)"
            )
    for key in [reg["primary"], *reg["guards"]]:
        if key not in RESULT_KEYS:
            raise ValueError(
                f"{row_id}: its primary or guard {key!r} is not a numeric result key;"
                f" expected one of {RESULT_KEYS}"
            )

    runs = [r for r in existing if r["kind"] == "run" and r["registered"] == row_id]
    ends = {r["run"]: r for r in existing if r["kind"] == "run-end"}
    unclosed = [
        f"run {r['row_id']} (arm {r['arm']}, seed {r['seed']})"
        for r in runs
        if r["row_id"] not in ends
    ]
    if unclosed:
        raise ValueError(
            f"{row_id}: no run-end for {', '.join(unclosed)}; close each before selection"
        )
    scored = {}
    for arm in reg["arms"]:
        name = arm["name"]
        scored[name] = []
        for seed in sorted(reg["seeds"]):
            mine = [r for r in runs if (r["arm"], r["seed"]) == (name, seed)]
            if not mine:
                raise ValueError(f"{row_id}: arm {name} seed {seed} has no run")
            done = [r for r in mine if ends[r["row_id"]]["status"] != "crashed"]
            if len(done) > 1:
                raise ValueError(
                    f"{row_id}: arm {name} seed {seed} has {len(done)} runs that didn't crash"
                    f" ({', '.join(r['row_id'] for r in done)}); one arm x seed scores one run"
                )
            scored[name] += done
    return reg, scored


def _read(method: str, run_dir: Path) -> dict | None:
    """What scoring a run reads, read before anything is scored: rb_tuned's params from its run
    dir's rule_based_tuned.yml; a learning run's selected.json, or None if it has none yet."""
    if method == "rb_tuned":
        return yaml.safe_load((run_dir / "rule_based_tuned.yml").read_text())["GreenLightEnv"]
    path = run_dir / "selected.json"
    if not path.exists():
        return None
    selected = json.loads(path.read_text())
    if (
        selected["scenarios"] != scenarios.VALIDATION
        or selected["season_length"] != evaluate.SEASON_LENGTH
    ):
        raise ValueError(
            f"{path} was selected on {selected['scenarios']} in {selected['season_length']}-day"
            f" seasons, not VALIDATION in {evaluate.SEASON_LENGTH}-day ones (PROTOCOL §5)"
        )
    return selected


def _seasons(method: str, run_dir: Path, read: dict | None) -> tuple[list[dict], str | None]:
    """A run's VALIDATION seasons, in order, and its checkpoint (None for rb_tuned)."""
    if method == "rb_tuned":
        seasons = [
            evaluate.run_rb(read, s, evaluate.SEASON_LENGTH)[0] for s in scenarios.VALIDATION
        ]
        return seasons, None
    if read is None:
        select_checkpoint(run_dir)
        read = json.loads((run_dir / "selected.json").read_text())
    checkpoint = read["checkpoint"]
    return read["candidates"][checkpoint]["per_scenario"], checkpoint


def _ci(by_run: list[list[dict]], key: str) -> dict:
    """stats.iqm_ci of key's runs x VALIDATION array; null with no run to score."""
    if not by_run:
        return {"iqm": None, "ci_low": None, "ci_high": None}
    scores = np.array([[s[key] for s in seasons] for seasons in by_run], dtype=np.float64)
    iqm, low, high = stats.iqm_ci(scores)
    return {"iqm": iqm, "ci_low": low, "ci_high": high}


def _shown(ci: dict) -> str:
    if ci["iqm"] is None:
        return "null"
    return f"{ci['iqm']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.selection")
    p.add_argument("--registered", metavar="ROW", required=True, help="the registered row")
    p.add_argument("--out", type=Path, help="default control/results/validation/<row>.json")
    args = p.parse_args(argv)
    out = args.out or RESULTS / f"{args.registered}.json"
    if out.exists():
        raise FileExistsError(f"{out} already exists; a numbers file is written once")
    reg, runs = _registration(args.registered, ledger_path)
    ends = ledger._run_ends(ledger.rows(path=ledger_path), reg["row_id"])  # the result's crashed
    methods = {arm["name"]: arm["method"] for arm in reg["arms"]}
    # Every selected.json and rb_tuned's params, checked before a season is played or a file written.
    read = {
        run["row_id"]: _read(methods[name], Path(run["run_dir"]))
        for name, arm_runs in runs.items()
        for run in arm_runs
    }

    per_arm, selected, lines = {}, {}, []
    for name, arm_runs in runs.items():
        by_run = []
        for run in arm_runs:
            seasons, checkpoint = _seasons(methods[name], Path(run["run_dir"]), read[run["row_id"]])
            by_run.append(seasons)
            if checkpoint is not None:
                selected[run["row_id"]] = checkpoint
        numbers = {
            **_ci(by_run, reg["primary"]),
            "guards": {g: _ci(by_run, g) for g in reg["guards"]},
            "invalid_episodes": sum(not s["valid"] for seasons in by_run for s in seasons),
        }
        per_arm[name] = numbers
        guards = "  ".join(f"{g} {_shown(numbers['guards'][g])}" for g in reg["guards"])
        line = (
            f"{name}  {reg['primary']} {_shown(numbers)}  {guards}  n_runs {len(by_run)}"
            f"  invalid_episodes {numbers['invalid_episodes']}"
            f"  crashed {ledger._arm_tally(ends, name)['crashed']}"
        )
        if methods[name] == "rb_tuned" and len(by_run) == 1:
            line += "  deterministic, 1 run"  # PROTOCOL §6: a deterministic arm needs one run
        elif len(by_run) < 5:
            line += f"  {NO_CLAIM}"
        lines.append(line)

    # allow_nan=False: a NaN or inf raises here, before a dir is made or the file opened.
    text = json.dumps(
        {"per_arm": per_arm, "selected": selected}, indent=2, sort_keys=True, allow_nan=False
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        f.write(text + "\n")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

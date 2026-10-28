"""Checkpoint selection (PROTOCOL §5, docs/API-CONTRACT.md): the rule on VALIDATION only.

select_checkpoint scores each of a run's checkpoints, ckpt_<steps>.zip in numeric order of steps
then ckpt_final.zip, with evaluate.run_policy (deterministic, frozen stats) on scenarios.VALIDATION
in 60-day seasons, and picks one:
- residual_lag: the highest mean profit_eur_m2 among checkpoints whose mean per-step cost_<c>,
  pooled over the seasons, is <= budget x 1.10 on every constraint; if none qualifies, the lowest
  total cost (the plain sum over constraints of those means).
- every other arm: the highest mean paper_return (metrics', not paper_reward_sum).
A tie goes to the earliest checkpoint; an invalid season is scored, never dropped (PROTOCOL §4); a
non-finite score raises. run_dir/selected.json is written once.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from greenleaf_control import evaluate, metrics, scenarios

TOLERANCE = 1.10  # PROTOCOL §5: a mean cost qualifies up to budget x 1.10, inclusive


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
    per-step cost on each constraint and whether all are within budget x TOLERANCE."""
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
            mean_cost[c] <= b * TOLERANCE for c, b in zip(metrics.CONSTRAINTS, budgets, strict=True)
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

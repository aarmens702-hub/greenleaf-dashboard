"""Tests for greenleaf_control.selection (issue #13, part 13c): select_checkpoint, the PROTOCOL §5
checkpoint rule, on VALIDATION only.

select_checkpoint(run_dir) reads run_dir/manifest.json's config: its method, and for residual_lag
its budgets (a list in metrics.CONSTRAINTS order, co2, temp, rh, as train writes it). The
candidates are ckpt_<steps>.zip in numeric order of steps, then ckpt_final.zip; no other file is
one (not ckpt_zero.zip, the zeroed policy test_evaluate saves beside a run's checkpoints). Each is
scored with evaluate.run_policy(run_dir, "<steps>" | "final", s, 60, record_steps=<residual_lag>)
for s in scenarios.VALIDATION, in order: 60-day seasons (PROTOCOL §1), not the manifest's. Then:
- residual_lag: a checkpoint qualifies when, on every constraint c, its mean per-step cost_<c> (the
  frames' column, pooled over the three seasons) is <= budgets[c] x 1.10; the qualifying one with
  the highest mean profit_eur_m2 wins; if none qualifies, the lowest total cost wins, the plain sum
  over c of those means.
- every other arm: the highest mean paper_return (metrics', not paper_reward_sum).
A tie goes to the earliest checkpoint; an invalid season is scored, never dropped (PROTOCOL §4); a
non-finite score raises. It writes run_dir/selected.json once (exclusive create) and returns the
checkpoint as run_policy takes it: "<steps>" or "final".

run_policy is faked: the rule is under test here, and test_evaluate covers the evaluation itself.
The fake looks up a table of each candidate's three VALIDATION seasons (paper_return,
profit_eur_m2, valid, and each constraint's mean per-step cost) and records every call. Every other
way to play a season (train.build_vec_env, envs.make_raw_env, make_env, run_baseline,
evaluate.run_rb) fails if called, so selection must go through evaluate.run_policy, the module
attribute.

Known answers used throughout (each test's docstring has its own):
- A fake season is 5,760 steps (60 days x 96). Its cost_<c> column alternates m/2 and 3m/2, so its
  mean is m while its first, last and largest steps are not. Every m is dyadic (a sum of powers of
  2), so every mean here is exact in floating point, however it is summed or pooled.
- The fake's paper_reward_sum is -paper_return, and its frame's (upstream) profit column sums to
  -profit_eur_m2: both rank the candidates the other way, so a rule that reads them picks wrong.
- The boundary: 1.25 x 1.10 == 1.375 exactly in IEEE doubles (and 1.375 / 1.25 == 1.10), so a
  budget of 1.25 and a mean cost of 1.375 sit exactly on the line. Budget 1.0 and cost 1.1 don't:
  the mean of 5,760 steps of 1.1 is 1.0999999999999999, under the line, so a strict < passes too.

Nothing here may write the real ledger or control/runs/; the last test checks both are as they
were when this module was collected.
"""

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenleaf_control import envs, evaluate, metrics, scenarios, train

# greenleaf_control.selection is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger and runs dir, read when this module is collected, before any test in it runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)

CONSTRAINTS = metrics.CONSTRAINTS  # ("co2", "temp", "rh")
SEASON_STEPS = 60 * 96  # 60 days x 96 steps of 900 s = 5,760
ALTERNATING = np.tile([0.5, 1.5], SEASON_STEPS // 2)  # mean 1, so m x ALTERNATING has mean m
NON_LAG = [m for m in envs.METHODS if m != "residual_lag"]  # ppo_paper, ppo, residual
LAG_BUDGETS = [0.5, 1.0, 2.0]  # co2, temp, rh: distinct, so a permuted list fails some check
BOUNDARY_BUDGET = 1.25
BOUNDARY_COST = 1.375  # 1.25 x 1.10 == 1.375 exactly in IEEE doubles
JUST_OVER = 1.375 + 2**-8  # 1.37890625 = 1.25 x 1.103125: over 1.10, under any tolerance >= 1.1032


def _three(x) -> tuple:
    """A number is the same in all three VALIDATION seasons; a tuple is per season, in order."""
    return tuple(x) if isinstance(x, tuple | list) else (x, x, x)


def _cand(paper_return=0.0, profit=0.0, cost=(0.25, 0.25, 0.25), valid=True) -> dict:
    """One candidate's three VALIDATION seasons. cost is each constraint's season-mean per-step
    cost_<c>, in CONSTRAINTS order (each a number or a 3-tuple)."""
    return {
        "paper_return": _three(paper_return),
        "profit": _three(profit),
        "cost": {c: _three(v) for c, v in zip(CONSTRAINTS, cost, strict=True)},
        "valid": _three(valid),
    }


def _result(spec: dict, i: int) -> dict:
    """Season i's result dict, with metrics.episode_metrics' keys plus paper_reward_sum. Its
    paper_reward_sum is -paper_return, so it ranks the candidates the other way."""
    valid = spec["valid"][i]
    return {
        "profit_eur_m2": spec["profit"][i],
        **{f"viol_hours_{c}": 0.0 for c in CONSTRAINTS},
        **{f"pct_in_bounds_{c}": 100.0 for c in CONSTRAINTS},
        "yield_kg_m2": 30.0,
        "heat_cost": 1.0,
        "elec_cost": 1.0,
        "co2_cost": 1.0,
        "paper_return": spec["paper_return"][i],
        "solver_failures": 0 if valid else 1,
        "steps": SEASON_STEPS,
        "valid": valid,
        "paper_reward_sum": -spec["paper_return"][i],
    }


def _frame(spec: dict, i: int) -> pd.DataFrame:
    """Season i's per-step frame: cost_<c> alternates m/2, 3m/2 (mean m, exact); the upstream
    profit column sums to -profit_eur_m2, so it ranks the candidates the other way."""
    return pd.DataFrame(
        {
            "t": np.arange(SEASON_STEPS),
            "profit": np.full(SEASON_STEPS, -spec["profit"][i] / SEASON_STEPS),
            **{f"cost_{c}": spec["cost"][c][i] * ALTERNATING for c in CONSTRAINTS},
        }
    )


def _no_other_evaluation(monkeypatch) -> None:
    """Every other way to play a season fails, so selection must score through
    evaluate.run_policy, the module attribute the fake replaces."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: selection scores through run_policy only")

        return refuse

    for module, name in [
        (train, "build_vec_env"),
        (envs, "make_raw_env"),
        (envs, "make_env"),
        (envs, "run_baseline"),
        (evaluate, "run_rb"),
    ]:
        monkeypatch.setattr(module, name, trip(f"{module.__name__}.{name}"))


def _fake_run_policy(monkeypatch, table: dict) -> list[dict]:
    """Replace evaluate.run_policy with a lookup in table, keyed by (checkpoint, scenario); every
    call is recorded. A frame comes back only with record_steps, as from the real one."""
    calls: list[dict] = []

    def run_policy(run_dir, checkpoint, scenario, season_length=60, record_steps=False):
        calls.append(
            {
                "run_dir": Path(run_dir),
                "checkpoint": checkpoint,
                "scenario": dict(scenario),
                "season_length": season_length,
                "record_steps": record_steps,
            }
        )
        if scenarios.is_test(scenario):
            raise AssertionError(f"selection scored the sealed 2012 test weather: {scenario}")
        if scenario not in scenarios.VALIDATION:
            raise AssertionError(f"selection scored {scenario}, not a VALIDATION scenario")
        if checkpoint not in table:
            raise AssertionError(
                f"run_policy got checkpoint {checkpoint!r}; the candidates are {list(table)}"
            )
        spec, i = table[checkpoint], scenarios.VALIDATION.index(scenario)
        return _result(spec, i), _frame(spec, i) if record_steps else None

    monkeypatch.setattr(evaluate, "run_policy", run_policy)
    return calls


def _setup(tmp_path, monkeypatch, method, table, budgets=None, extra=()) -> tuple[Path, list]:
    """A run dir holding manifest.json {"config": ...} (a 2-day training season, as a smoke run's)
    and an empty ckpt_<name>.zip + vecnorm_<name>.pkl per table key, in the table's order, then per
    extra name; with run_policy faked and every other evaluation tripped."""
    run_dir = tmp_path / f"{method}-seed0"
    run_dir.mkdir()
    lag = method == "residual_lag"
    config = {
        "method": method,
        "seed": 0,
        "season_length": 2,
        "penalty_weight": 0.0 if lag else 1.0,
        "budget_frac": 1.0 if lag else None,
        "budgets": budgets,
    }
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}, indent=2) + "\n")
    for name in [*table, *extra]:
        (run_dir / f"ckpt_{name}.zip").write_bytes(b"")
        (run_dir / f"vecnorm_{name}.pkl").write_bytes(b"")
    calls = _fake_run_policy(monkeypatch, table)
    _no_other_evaluation(monkeypatch)
    return run_dir, calls


def _selected(run_dir: Path) -> dict:
    return json.loads((run_dir / "selected.json").read_text())


def _paper_return_table() -> dict:
    """250000: paper_return (8, 8, 8), mean 8, median 8, profit 300; 500000: (0, 30, 0), mean 10,
    median 0, profit 100; final: (5, 5, 5), mean 5, profit 200."""
    return {
        "250000": _cand(paper_return=8.0, profit=300.0),
        "500000": _cand(paper_return=(0.0, 30.0, 0.0), profit=100.0),
        "final": _cand(paper_return=5.0, profit=200.0),
    }


def _one_over_table(over: str) -> dict:
    """Budgets LAG_BUDGETS (0.5, 1.0, 2.0), so the lines are 0.55, 1.1, 2.2.
    - 250000: profit 400, paper_return 50 (both the best); cost 0.5 x budget on two constraints
      and 1.5 x budget on `over`, above its line: out.
    - 500000: profit (0, 600, 0), mean 200, median 0; paper_return 1; cost exactly 1.0 x budget
      on each, (0.5, 1.0, 2.0): in.
    - final: profit 150 (mean and median 150); paper_return 20; cost 0.25 x budget on each: in."""
    return {
        "250000": _cand(
            paper_return=50.0,
            profit=400.0,
            cost=tuple(
                (1.5 if c == over else 0.5) * b
                for c, b in zip(CONSTRAINTS, LAG_BUDGETS, strict=True)
            ),
        ),
        "500000": _cand(paper_return=1.0, profit=(0.0, 600.0, 0.0), cost=tuple(LAG_BUDGETS)),
        "final": _cand(paper_return=20.0, profit=150.0, cost=tuple(0.25 * b for b in LAG_BUDGETS)),
    }


# --- The rule (Done when: PROTOCOL §5, on fake checkpoint metrics) ------------------------------


@pytest.mark.parametrize("method", NON_LAG)
def test_a_non_lag_arm_picks_the_highest_mean_paper_return(method, tmp_path, monkeypatch):
    """Done when (PROTOCOL §5, every other learning arm): the highest MEAN paper_return over the
    three VALIDATION seasons.

    Known answer (_paper_return_table): 500000, mean (0 + 30 + 0) / 3 = 10, beats 250000 (mean 8)
    and final (mean 5). It is not what the median picks (8 vs 0: 250000), nor the highest mean
    profit_eur_m2 (300: 250000), nor paper_reward_sum (-paper_return here: final), nor the first or
    the last checkpoint. The rule is named max_mean_paper_return.
    """
    import greenleaf_control.selection as selection

    run_dir, _ = _setup(tmp_path, monkeypatch, method, _paper_return_table())

    assert selection.select_checkpoint(run_dir) == "500000"
    assert _selected(run_dir)["rule"] == "max_mean_paper_return"


@pytest.mark.parametrize("over", CONSTRAINTS)
def test_lag_excludes_the_best_profit_checkpoint_over_budget_on_one_constraint(
    over, tmp_path, monkeypatch
):
    """Done when (PROTOCOL §5, residual_lag): the highest mean profit among checkpoints within
    budget x 1.10 on EVERY constraint; one constraint over is enough to exclude one.

    Known answer (_one_over_table): 250000 has the best profit (400) but is at 1.5 x budget on
    `over` alone: out. 500000 (at exactly 1.0 x budget on each) and final (0.25 x) qualify, and
    500000's mean profit, (0 + 600 + 0) / 3 = 200, beats final's 150. So 500000, under the rule
    max_mean_profit_within_budget. Profit alone or paper_return picks 250000; the median profit
    (0 vs 150) picks final; so does any permutation of the budgets, which puts one of 500000's
    costs at 2 x or more of another constraint's budget.
    """
    import greenleaf_control.selection as selection

    run_dir, _ = _setup(
        tmp_path, monkeypatch, "residual_lag", _one_over_table(over), budgets=LAG_BUDGETS
    )

    assert selection.select_checkpoint(run_dir) == "500000"
    selected = _selected(run_dir)
    assert selected["rule"] == "max_mean_profit_within_budget"
    assert {name: c["qualifies"] for name, c in selected["candidates"].items()} == {
        "250000": False,
        "500000": True,
        "final": True,
    }


def test_lag_qualifies_on_the_mean_pooled_over_the_three_seasons(tmp_path, monkeypatch):
    """A checkpoint qualifies on its mean per-step cost pooled over the three VALIDATION seasons,
    not on each season, nor on the median season. Budgets 1.0 each, so the line is 1.1.

    Known answer (co2's season means; temp and rh are 0.25 everywhere):
    - 250000: (0.5, 0.5, 2.5): mean 3.5 / 3 = 1.167 > 1.1, out, though two seasons of three (and
      the median, 0.5) are under. Profit 300.
    - 500000: (1.5, 0.75, 0.75): mean 3.0 / 3 = 1.0 <= 1.1, in, though its first season is over.
      Profit 200.
    - final: 0.25: in. Profit 100.
    So 500000. Requiring every season under the line picks final; the median picks 250000.
    """
    import greenleaf_control.selection as selection

    table = {
        "250000": _cand(profit=300.0, cost=((0.5, 0.5, 2.5), 0.25, 0.25)),
        "500000": _cand(profit=200.0, cost=((1.5, 0.75, 0.75), 0.25, 0.25)),
        "final": _cand(profit=100.0),
    }
    run_dir, _ = _setup(tmp_path, monkeypatch, "residual_lag", table, budgets=[1.0, 1.0, 1.0])

    assert selection.select_checkpoint(run_dir) == "500000"
    assert {name: c["qualifies"] for name, c in _selected(run_dir)["candidates"].items()} == {
        "250000": False,
        "500000": True,
        "final": True,
    }


@pytest.mark.parametrize("at", CONSTRAINTS)
def test_lag_a_mean_cost_of_exactly_budget_x_1_10_qualifies(at, tmp_path, monkeypatch):
    """ "<= budget x 1.10" is inclusive, and the tolerance is 1.10. Budgets 1.25 each, so the line
    is 1.25 x 1.10 = 1.375, exact in floating point (asserted), as is every mean here.

    Known answer (the other constraints are 0.25 everywhere):
    - 250000: mean cost_<at> exactly 1.375, on the line: in. Profit 200.
    - 500000: mean cost_<at> 1.37890625 (1.25 x 1.103125), just over: out. Profit 300.
    - final: 0.25: in. Profit 100.
    So 250000. A strict < picks final; a tolerance of 1.1032 or more picks 500000.
    """
    import greenleaf_control.selection as selection

    assert BOUNDARY_BUDGET * 1.10 == BOUNDARY_COST
    assert BOUNDARY_COST / BOUNDARY_BUDGET == 1.10
    assert (BOUNDARY_COST * ALTERNATING).mean() == BOUNDARY_COST

    def cost(value: float) -> tuple:
        return tuple(value if c == at else 0.25 for c in CONSTRAINTS)

    table = {
        "250000": _cand(profit=200.0, cost=cost(BOUNDARY_COST)),
        "500000": _cand(profit=300.0, cost=cost(JUST_OVER)),
        "final": _cand(profit=100.0),
    }
    run_dir, _ = _setup(tmp_path, monkeypatch, "residual_lag", table, budgets=[BOUNDARY_BUDGET] * 3)

    assert selection.select_checkpoint(run_dir) == "250000"
    assert {name: c["qualifies"] for name, c in _selected(run_dir)["candidates"].items()} == {
        "250000": True,
        "500000": False,
        "final": True,
    }


def test_lag_with_none_qualifying_picks_the_lowest_plain_total_cost(tmp_path, monkeypatch):
    """Done when (PROTOCOL §5, residual_lag): if none qualifies, the lowest total cost, the plain
    sum over constraints of the mean per-step costs, whatever the profit. Budgets (0.5, 1.0, 1.0),
    so the lines are 0.55, 1.1, 1.1, and every co2 cost below is over 0.55: none qualifies.

    Known answer (mean costs co2, temp, rh):
    - 250000: (1.0, 2.0, 2.0): total 5.0; worst single 2.0; sum of cost / budget 6.0. Profit 300.
    - 500000: (3.0, 0.5, 0.5): total 4.0; worst single 3.0; sum of cost / budget 7.0. Profit 100.
    - final: (0.75, 1.25, 2.5): total 4.5; worst single 2.5; sum of cost / budget 5.25. Profit 200.
    So 500000, with the lowest profit, under the rule min_total_cost. The smallest worst
    constraint (or worst cost / budget, 2.0) and the highest profit pick 250000; the
    budget-normalised sum picks final.
    """
    import greenleaf_control.selection as selection

    table = {
        "250000": _cand(profit=300.0, cost=(1.0, 2.0, 2.0)),
        "500000": _cand(profit=100.0, cost=(3.0, 0.5, 0.5)),
        "final": _cand(profit=200.0, cost=(0.75, 1.25, 2.5)),
    }
    run_dir, _ = _setup(tmp_path, monkeypatch, "residual_lag", table, budgets=[0.5, 1.0, 1.0])

    assert selection.select_checkpoint(run_dir) == "500000"
    selected = _selected(run_dir)
    assert selected["rule"] == "min_total_cost"
    assert all(c["qualifies"] is False for c in selected["candidates"].values())


# --- The candidates ----------------------------------------------------------------------------


def test_every_checkpoint_is_a_candidate_in_numeric_step_order_then_final(tmp_path, monkeypatch):
    """The candidates are ckpt_<steps>.zip in numeric order of steps (1,000,000 after 750,000,
    though "1000000" sorts first as a string), then ckpt_final.zip, each scored on the three
    VALIDATION seasons before the next. ckpt_zero.zip is not one, nor is any other file. The files
    are made out of order.

    Known answer: 15 calls, (250000, 500000, 750000, 1000000, final) x VALIDATION; never "zero";
    1000000 has the highest mean paper_return (9) and is returned; selected.json lists exactly
    these five candidates.
    """
    import greenleaf_control.selection as selection

    table = {
        "final": _cand(paper_return=4.0),
        "1000000": _cand(paper_return=9.0),
        "500000": _cand(paper_return=2.0),
        "250000": _cand(paper_return=1.0),
        "750000": _cand(paper_return=3.0),
    }
    run_dir, calls = _setup(tmp_path, monkeypatch, "ppo", table, extra=("zero",))
    (run_dir / "final_eval.json").write_text("{}\n")
    (run_dir / "tb").mkdir()
    order = ["250000", "500000", "750000", "1000000", "final"]

    assert selection.select_checkpoint(run_dir) == "1000000"
    assert [(c["checkpoint"], c["scenario"]) for c in calls] == [
        (name, s) for name in order for s in scenarios.VALIDATION
    ]
    assert set(_selected(run_dir)["candidates"]) == set(order)


@pytest.mark.parametrize("method", envs.METHODS)
def test_selection_scores_on_validation_only_in_60_day_seasons(method, tmp_path, monkeypatch):
    """The seal (PROTOCOL §5, §8): every candidate is scored on exactly scenarios.VALIDATION, in
    order, so never on a 2012 test scenario; in 60-day seasons (PROTOCOL §1), though the manifest
    trained on 2-day ones; through run_policy on this run dir; with frames (record_steps True) for
    residual_lag only, the one arm whose rule reads per-step costs. Nothing else plays a season
    (the other evaluation paths are tripped).
    """
    import greenleaf_control.selection as selection

    lag = method == "residual_lag"
    table = {
        "250000": _cand(paper_return=1.0, profit=1.0),
        "final": _cand(paper_return=2.0, profit=2.0),
    }
    budgets = [1.0, 1.0, 1.0] if lag else None
    run_dir, calls = _setup(tmp_path, monkeypatch, method, table, budgets=budgets)

    selection.select_checkpoint(run_dir)

    for name in table:
        assert [c["scenario"] for c in calls if c["checkpoint"] == name] == scenarios.VALIDATION
    assert len(calls) == 2 * len(scenarios.VALIDATION)
    assert not any(scenarios.is_test(c["scenario"]) for c in calls)
    assert [c["season_length"] for c in calls] == [60] * len(calls)
    assert [c["record_steps"] for c in calls] == [lag] * len(calls)
    assert {c["run_dir"].resolve() for c in calls} == {run_dir.resolve()}


# --- selected.json -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["ppo", "residual_lag"])
def test_selected_json_records_the_choice_and_every_candidate(method, tmp_path, monkeypatch):
    """run_dir/selected.json: checkpoint (== the returned value), method, rule, scenarios
    (VALIDATION), season_length 60, budgets and tolerance 1.10 (residual_lag; else null), and per
    candidate its per_scenario result dicts (as run_policy returned them, valid included),
    mean_paper_return, mean_profit_eur_m2, and for residual_lag mean_cost {co2, temp, rh} and
    qualifies (else null).

    Known answers: ppo is _paper_return_table, so 500000 with mean paper_return 8, 10, 5 and mean
    profit 300, 100, 200 for 250000, 500000, final. residual_lag is _one_over_table("temp"), so
    500000; mean profit 400, 200, 150; mean paper_return 50, 1, 20; mean cost (0.25, 1.5, 1.0),
    (0.5, 1.0, 2.0), (0.125, 0.25, 0.5); qualifies False, True, True.
    """
    import greenleaf_control.selection as selection

    lag = method == "residual_lag"
    table = _one_over_table("temp") if lag else _paper_return_table()
    budgets = LAG_BUDGETS if lag else None
    run_dir, _ = _setup(tmp_path, monkeypatch, method, table, budgets=budgets)

    got = selection.select_checkpoint(run_dir)

    selected = _selected(run_dir)
    assert selected["checkpoint"] == got == "500000"
    assert selected["method"] == method
    assert selected["rule"] == ("max_mean_profit_within_budget" if lag else "max_mean_paper_return")
    assert selected["scenarios"] == scenarios.VALIDATION
    assert selected["season_length"] == 60
    assert selected["budgets"] == budgets
    assert selected["tolerance"] == (pytest.approx(1.10) if lag else None)
    candidates = selected["candidates"]
    assert set(candidates) == set(table)
    for name, spec in table.items():
        per_scenario = candidates[name]["per_scenario"]
        assert len(per_scenario) == len(scenarios.VALIDATION), name
        for i, recorded in enumerate(per_scenario):
            want = _result(spec, i)
            assert {k: recorded.get(k) for k in want} == want, (name, i)
    if lag:
        want_means = {
            "250000": (50.0, 400.0, {"co2": 0.25, "temp": 1.5, "rh": 1.0}, False),
            "500000": (1.0, 200.0, {"co2": 0.5, "temp": 1.0, "rh": 2.0}, True),
            "final": (20.0, 150.0, {"co2": 0.125, "temp": 0.25, "rh": 0.5}, True),
        }
    else:
        want_means = {
            "250000": (8.0, 300.0, None, None),
            "500000": (10.0, 100.0, None, None),
            "final": (5.0, 200.0, None, None),
        }
    for name, (paper_return, profit, cost, qualifies) in want_means.items():
        c = candidates[name]
        assert c["mean_paper_return"] == pytest.approx(paper_return), name
        assert c["mean_profit_eur_m2"] == pytest.approx(profit), name
        assert c["mean_cost"] == (None if cost is None else pytest.approx(cost)), name
        assert c["qualifies"] is qualifies, name


def test_an_existing_selected_json_is_refused_and_left_unchanged(tmp_path, monkeypatch):
    """selected.json is written once (exclusive create): a run dir that already has one is refused
    with FileExistsError naming it, and the earlier file keeps its bytes."""
    import greenleaf_control.selection as selection

    run_dir, _ = _setup(tmp_path, monkeypatch, "ppo", _paper_return_table())
    earlier = b'{"checkpoint": "250000"}\n'
    (run_dir / "selected.json").write_bytes(earlier)

    with pytest.raises(FileExistsError, match="selected.json"):
        selection.select_checkpoint(run_dir)
    assert (run_dir / "selected.json").read_bytes() == earlier


# --- Ties, invalid seasons, and what raises -----------------------------------------------------

TIES = [
    pytest.param(
        "ppo",
        None,
        {
            "250000": _cand(paper_return=4.0),
            "500000": _cand(paper_return=(6.0, 12.0, 12.0)),  # (6 + 12 + 12) / 3 = 10
            "750000": _cand(paper_return=10.0),
            "final": _cand(paper_return=10.0),
        },
        id="mean-paper-return",
    ),
    pytest.param(
        "residual_lag",
        [1.0, 1.0, 1.0],
        {
            "250000": _cand(profit=100.0),
            "500000": _cand(profit=(200.0, 400.0, 300.0)),  # (200 + 400 + 300) / 3 = 300
            "750000": _cand(profit=500.0, cost=(2.0, 0.25, 0.25)),  # co2 over 1.1: out
            "final": _cand(profit=300.0),
        },
        id="mean-profit-within-budget",
    ),
    pytest.param(
        "residual_lag",
        [1.0, 1.0, 1.0],
        {
            "250000": _cand(profit=100.0, cost=(2.0, 2.0, 2.0)),  # total 6.0
            "500000": _cand(profit=100.0, cost=(3.0, 0.75, 0.75)),  # total 4.5
            "750000": _cand(profit=100.0, cost=(1.5, 1.5, 1.5)),  # total 4.5
            "final": _cand(profit=100.0, cost=(2.5, 1.25, 1.25)),  # total 5.0
        },
        id="min-total-cost",
    ),
]


@pytest.mark.parametrize(("method", "budgets", "table"), TIES)
def test_a_tie_goes_to_the_earliest_checkpoint(method, budgets, table, tmp_path, monkeypatch):
    """Each rule's best score is shared, exactly (every value is exact in floating point): by 500000
    and final on the mean paper_return (10) and on the qualifying mean profit (300); by 500000 and
    750000 on the lowest total cost (4.5, none qualifying). 250000 scores worse every time. The
    earliest of the tied, 500000, wins; a last-wins comparison (>=) picks the later one."""
    import greenleaf_control.selection as selection

    run_dir, _ = _setup(tmp_path, monkeypatch, method, table, budgets=budgets)

    assert selection.select_checkpoint(run_dir) == "500000"


def test_an_invalid_season_is_scored_not_dropped(tmp_path, monkeypatch):
    """PROTOCOL §4: an invalid episode is reported, never dropped. 500000's second season is
    invalid (valid False, one solver failure) and holds its best paper_return.

    Known answer (_paper_return_table): 500000's mean is (0 + 30 + 0) / 3 = 10 with that season and
    0 without it; 250000's is 8. So 500000, and selected.json records valid False on its second
    season.
    """
    import greenleaf_control.selection as selection

    table = _paper_return_table()
    table["500000"] = _cand(paper_return=(0.0, 30.0, 0.0), profit=100.0, valid=(True, False, True))
    run_dir, _ = _setup(tmp_path, monkeypatch, "ppo", table)

    assert selection.select_checkpoint(run_dir) == "500000"
    per_scenario = _selected(run_dir)["candidates"]["500000"]["per_scenario"]
    assert [r.get("valid") for r in per_scenario] == [True, False, True]


@pytest.mark.parametrize("extra", [(), ("zero",)], ids=["manifest-only", "ckpt-zero-only"])
def test_a_run_dir_without_candidate_checkpoints_raises(extra, tmp_path, monkeypatch):
    """A run dir with no ckpt_<steps>.zip and no ckpt_final.zip (just manifest.json, or only
    ckpt_zero.zip, which is not a candidate) raises FileNotFoundError or ValueError, scores
    nothing and writes no selected.json."""
    import greenleaf_control.selection as selection

    run_dir, calls = _setup(tmp_path, monkeypatch, "ppo", {}, extra=extra)

    with pytest.raises((FileNotFoundError, ValueError)):
        selection.select_checkpoint(run_dir)
    assert calls == []
    assert not (run_dir / "selected.json").exists()


NON_FINITE = [
    pytest.param("ppo", {"paper_return": (8.0, math.nan, 8.0)}, id="ppo-nan-paper-return"),
    pytest.param("ppo", {"paper_return": (8.0, math.inf, 8.0)}, id="ppo-inf-paper-return"),
    pytest.param("residual_lag", {"profit": (100.0, math.nan, 100.0)}, id="lag-nan-profit"),
    pytest.param(
        "residual_lag", {"cost": (0.25, (0.25, math.nan, 0.25), 0.25)}, id="lag-nan-cost-temp"
    ),
]


@pytest.mark.parametrize(("method", "bad"), NON_FINITE)
def test_a_non_finite_score_raises_and_writes_nothing(method, bad, tmp_path, monkeypatch):
    """A non-finite score in any candidate raises (ValueError or ArithmeticError) and leaves no
    selected.json, even a partial one. NaN compares False, so it would silently lose (or a NaN
    cost would silently fail its budget check), and +inf would silently win. The bad value is in
    500000, the middle candidate; budgets 1.0 each, every other cost 0.25, every other score
    finite."""
    import greenleaf_control.selection as selection

    good = {"paper_return": 8.0, "profit": 100.0, "cost": (0.25, 0.25, 0.25)}
    table = {
        "250000": _cand(paper_return=4.0, profit=50.0),
        "500000": _cand(**{**good, **bad}),
        "final": _cand(paper_return=2.0, profit=25.0),
    }
    budgets = [1.0, 1.0, 1.0] if method == "residual_lag" else None
    run_dir, _ = _setup(tmp_path, monkeypatch, method, table, budgets=budgets)

    with pytest.raises((ValueError, ArithmeticError)):
        selection.select_checkpoint(run_dir)
    assert not (run_dir / "selected.json").exists()


# --- The contract ------------------------------------------------------------------------------


def test_signature_matches_the_contract_and_no_test_wrote_the_real_ledger_or_runs():
    """select_checkpoint(run_dir) (docs/API-CONTRACT.md), nothing else.

    This test runs last in the module and checks the real ledger's size and mtime, and the
    listing of control/runs/, are as they were when the module was collected.
    """
    import greenleaf_control.selection as selection

    assert list(inspect.signature(selection.select_checkpoint).parameters) == ["run_dir"]
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE

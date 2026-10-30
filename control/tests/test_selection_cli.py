"""Tests for the selection CLI (issue #20): `python -m greenleaf_control.selection --registered ROW
[--out FILE]` turns a registration's runs into the per-arm numbers `ledger result --numbers`
records.

selection.main(argv=None, *, ledger_path=ledger.LEDGER) -> int, as these tests pin it:
- It reads the registered row and its run and run-end rows from ledger_path; a run's dir is its run
  row's run_dir (train and tune_rb record <runs-dir>/<row>/<arm>/seed<S> there).
- It refuses before anything is scored or written, with a message naming the problem:
  - an unknown row;
  - scenarios other than exactly ["validation"] (train_eval, any TEST split, validation + paper);
  - an arm whose method is neither in envs.METHODS nor rb_tuned (rb_default, zero_controls and
    random have no run rows: they are scored by evaluate --controller);
  - a run with no run-end (an open run), even beside a finished rerun;
  - a registered arm x seed with no run;
  - two finished runs of one arm x seed;
  - an existing --out;
  - a primary or guard that isn't a numeric key of the scored result dicts
    (metrics.episode_metrics' keys plus paper_reward_sum; not valid). ledger.register refuses
    such a row too (#25), so these tests plant it in the tmp ledger by hand;
  - an existing selected.json whose scenarios aren't VALIDATION or whose season_length isn't 60.
- Every run that didn't crash (finished or budget-stop) is scored on VALIDATION:
  - a learning arm (envs.METHODS) from run_dir/selected.json, or, when there is none, through
    selection.select_checkpoint(run_dir), which writes it: the chosen checkpoint's per_scenario
    results, with no second evaluation;
  - rb_tuned with evaluate.run_rb(<run_dir/rule_based_tuned.yml's GreenLightEnv params>, s, 60) for
    s in VALIDATION: the run dir's copy, never configs/.
- The numbers file is exactly {"per_arm", "selected"}:
  - per_arm[arm] is exactly {iqm, ci_low, ci_high, guards: {g: {iqm, ci_low, ci_high}},
    invalid_episodes}: stats.iqm_ci (its defaults) on the runs (sorted by seed) x VALIDATION array
    of each metric, and the count of valid False among the scored seasons. No crashed or spend_usd:
    the ledger adds them from the run-ends. An arm with no run to score has null numbers.
  - selected maps each scored learning run's row_id to its checkpoint ("<steps>" or "final").
- --out defaults to selection.RESULTS / "<row>.json" (RESULTS is control/results/validation) and
  is created exclusively. main prints one line per arm and nothing else, flagged "n<5: no claim"
  under 5 runs, except an rb_tuned arm's one run, flagged "deterministic, 1 run" (PROTOCOL §6).
  Each line shows "crashed N": the arm's crashed run-ends, the count the ledger's result records.
- selection.RESULT_KEYS is exactly a real metrics.episode_metrics output's numeric keys (all but
  valid) plus paper_reward_sum.

Nothing here plays a season except the RESULT_KEYS pin, which grades one real PAPER day of zero
controls: run_policy and run_rb are faked or tripped, as are select_checkpoint
(except where a test spies on it), every env builder and rule_based.load_rb_params. Every ledger,
run dir and numbers file is under tmp_path, and selection.RESULTS is patched to a tmp dir in every
test that calls main; the last test checks the real ledger, control/runs/ and control/results/ are
as they were when this module was collected.
"""

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from greenleaf_control import envs, evaluate, ledger, metrics, rule_based, scenarios, stats, train

# greenleaf_control.selection is imported inside each test body (by _selection), so while its main
# and RESULTS are missing each test FAILS rather than ERRORs at collection.

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"
REAL_RESULTS = CONTROL / "results"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    """Every path under path, relative and sorted; None if path doesn't exist."""
    return sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else None


# The real ledger, runs and results, read when this module is collected, before any test runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)
REAL_RESULTS_BEFORE = _listing(REAL_RESULTS)

PRIMARY = "profit_eur_m2"
GUARDS = ["viol_hours_temp", "yield_kg_m2"]
SEASON_STEPS = 60 * 96  # 60 days x 96 steps of 900 s
CANDIDATES = ("250000", "500000", "final")
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}
NUMBERS_KEYS = {"iqm", "ci_low", "ci_high", "guards", "invalid_episodes"}
CI_KEYS = {"iqm", "ci_low", "ci_high"}
NO_CLAIM = "n<5: no claim"

# Five seeds' scores, seed -> (VALIDATION start day 45, 59, 75). PROFIT's 15 values are 1..15, each
# once, so its IQM (int(0.25 x 15) = 3 trimmed from each end) is the mean of 4..12 = 8.0. VIOL's
# seed s is 10(s + 1) + (0, 1, 2); trimming 10, 11, 12 and 50, 51, 52 leaves 20..22, 30..32,
# 40..42, mean 279 / 9 = 31.0. YIELD is PROFIT / 4: IQM 2.0.
PROFIT = {0: (15, 1, 8), 1: (2, 14, 9), 2: (7, 3, 13), 3: (12, 6, 4), 4: (5, 11, 10)}
VIOL = {s: (10 * (s + 1), 10 * (s + 1) + 1, 10 * (s + 1) + 2) for s in range(5)}
YIELD = {s: tuple(v / 4 for v in PROFIT[s]) for s in range(5)}
# The ledger's run order. stats.iqm_ci's bootstrap depends on the row order (checked: PROFIT's
# ci_high is 10.6667 in seed order, 10.7778 in this one), so only seed order gives its numbers.
LEDGER_ORDER = (3, 0, 4, 1, 2)


def _season(valid: bool = True, **values) -> dict:
    """One season's result dict as run_policy and run_rb return it: metrics.episode_metrics' keys
    plus paper_reward_sum, each at a neutral value unless given."""
    return {
        "profit_eur_m2": 0.0,
        **{f"viol_hours_{c}": 0.0 for c in metrics.CONSTRAINTS},
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
    profit, viol=(0, 0, 0), yld=(0, 0, 0), valid=(True, True, True), paper_return=(0, 0, 0)
) -> list[dict]:
    """The three VALIDATION seasons, in order, with PRIMARY (profit_eur_m2), the GUARDS
    (viol_hours_temp, yield_kg_m2) and paper_return at these per-season values."""
    return [
        _season(
            valid=valid[i],
            profit_eur_m2=float(profit[i]),
            viol_hours_temp=float(viol[i]),
            yield_kg_m2=float(yld[i]),
            paper_return=float(paper_return[i]),
        )
        for i in range(3)
    ]


def _decoy() -> list[dict]:
    """A checkpoint no number may read: 1000 in every metric, every season invalid."""
    return _seasons((1000,) * 3, (1000,) * 3, (1000,) * 3, valid=(False,) * 3)


def _spec(seasons: list[dict], chosen: str = "final") -> dict:
    """A run's selection: chosen holds seasons, every other candidate the decoy."""
    return {
        "chosen": chosen,
        "candidates": {name: seasons if name == chosen else _decoy() for name in CANDIDATES},
    }


def _selected_json(method: str, spec: dict) -> dict:
    """selected.json as select_checkpoint writes it (docs/API-CONTRACT.md), for a non-lag arm."""
    return {
        "checkpoint": spec["chosen"],
        "method": method,
        "rule": "max_mean_paper_return",
        "scenarios": scenarios.VALIDATION,
        "season_length": 60,
        "budgets": None,
        "tolerance": None,
        "candidates": {
            name: {
                "per_scenario": seasons,
                "mean_paper_return": float(np.mean([s["paper_return"] for s in seasons])),
                "mean_profit_eur_m2": float(np.mean([s["profit_eur_m2"] for s in seasons])),
                "mean_cost": None,
                "qualifies": None,
            }
            for name, seasons in spec["candidates"].items()
        },
    }


def _register(path: Path, *, arms, seeds, **over) -> dict:
    """A registration on the tmp ledger: arms is ((name, method), ...); an rb_tuned arm's config is
    {"trials": 2}, as tune_rb requires. over replaces any other field."""
    kwargs = {
        "issue": 20,
        "hypothesis": "the arms differ in validation profit",
        "arms": [
            {
                "name": name,
                "method": method,
                "config": {"trials": 2} if method == "rb_tuned" else {},
            }
            for name, method in arms
        ],
        "seeds": list(seeds),
        "timesteps": 20_000,
        "scenarios": ["validation"],
        "primary": PRIMARY,
        "guards": list(GUARDS),
        "rule": "adopt if the profit IQM CI excludes rb_tuned's",
        "budget": {"cpu_hours": 1.0, "usd": 0.0, "backend": "local"},
        **over,
    }
    return ledger.register(**kwargs, path=path)


def _run(
    path: Path,
    root: Path,
    reg: dict,
    arm: str,
    seed: int,
    *,
    method: str = "ppo",
    status: str = "finished",
    cost_usd: float | None = None,
    spec: dict | None = None,
    end: bool = True,
    stale: dict | None = None,
    tag: str = "",
) -> tuple[dict, Path]:
    """One learning run of reg's arm at seed, as train records it: a run dir under root/runs
    (manifest.json, an empty ckpt_ and vecnorm_ file per candidate, and selected.json from spec if
    given, with stale's keys over it), a run row naming that dir, and unless end is False its
    run-end row. Returns (run row, run dir)."""
    run_dir = root / "runs" / reg["row_id"] / arm / f"seed{seed}{tag}"
    run_dir.mkdir(parents=True)
    config = {
        "method": method,
        "seed": seed,
        "timesteps": reg["timesteps"],
        "season_length": 60,
        "penalty_weight": 1.0,
        "budget_frac": None,
        "budgets": None,
        "run_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}, indent=2) + "\n")
    for name in CANDIDATES:
        (run_dir / f"ckpt_{name}.zip").write_bytes(b"")
        (run_dir / f"vecnorm_{name}.pkl").write_bytes(b"")
    if spec is not None:
        selected = {**_selected_json(method, spec), **(stale or {})}
        text = json.dumps(selected, indent=2, sort_keys=True)
        (run_dir / "selected.json").write_text(text + "\n")
    run = ledger.run_start(
        registered=reg["row_id"], arm=arm, seed=seed, run_dir=str(run_dir), path=path
    )
    if end:
        ledger.run_end(
            run=run["row_id"],
            status=status,
            wall_s=12.5,
            timesteps=reg["timesteps"],
            solver_failures=0,
            cost_usd=cost_usd,
            diagnostics=DIAGNOSTICS,
            path=path,
        )
    return run, run_dir


def _selection(monkeypatch, tmp_path: Path):
    """greenleaf_control.selection with RESULTS at a tmp dir that doesn't exist yet (nor does
    control/results/), so no default --out can reach the real one."""
    import greenleaf_control.selection as selection

    monkeypatch.setattr(selection, "RESULTS", tmp_path / "results" / "validation")
    return selection


def _tripwire(name: str):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{name} was called")

    return refuse


def _trip(monkeypatch, selection) -> None:
    """Every way to play a season, select_checkpoint, and reading configs/rule_based_*.yml fail if
    called. A test installs its fake or spy after this."""
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


def _numbers(selection, path: Path, row_id: str, out: Path) -> dict:
    """main --registered row_id --out out on the tmp ledger; the numbers file it wrote."""
    assert selection.main(["--registered", row_id, "--out", str(out)], ledger_path=path) == 0
    return json.loads(out.read_text())


def _record(path: Path, row_id: str, out: Path) -> dict:
    """ledger result --registered row_id --numbers out on the tmp ledger; the result row."""
    assert ledger.main(["result", "--registered", row_id, "--numbers", str(out)], path=path) == 0
    [res] = ledger.rows(kind="result", path=path)
    return res


def _refused(selection, argv: list[str], path: Path, capsys) -> str:
    """main refuses argv: it raises ValueError, FileExistsError or a parser error (SystemExit,
    non-zero), or returns non-zero. Returns everything it said: the error, stdout and stderr."""
    capsys.readouterr()
    try:
        code = selection.main(argv, ledger_path=path)
    except (ValueError, FileExistsError, SystemExit) as e:
        if isinstance(e, SystemExit):
            assert e.code not in (0, None), f"main exited {e.code!r}, which is not a refusal"
        said = capsys.readouterr()
        return f"{e}\n{said.out}\n{said.err}"
    said = capsys.readouterr()
    assert code != 0, f"main returned 0 instead of refusing; it printed {said.out!r}"
    return f"{said.out}\n{said.err}"


def _by_seed(table: dict, shift: float = 0.0) -> np.ndarray:
    """table's runs x VALIDATION array, rows in seed order."""
    return np.array([[v + shift for v in table[s]] for s in sorted(table)], dtype=np.float64)


def _ci(numbers: dict) -> tuple:
    return (numbers["iqm"], numbers["ci_low"], numbers["ci_high"])


def _tree(root: Path) -> dict:
    """Every file (its bytes) and dir (None) under root."""
    return {
        str(p.relative_to(root)): p.read_bytes() if p.is_file() else None
        for p in sorted(root.rglob("*"))
    }


# --- The numbers (Done when: per_arm equals hand-computed stats.iqm_ci values) ------------------


def test_per_arm_is_stats_iqm_ci_on_runs_sorted_by_seed_by_validation(tmp_path, monkeypatch):
    """Each arm's primary and each guard: stats.iqm_ci (reps 2000, seed 0) on the runs x VALIDATION
    array, rows sorted by seed (the ledger has them in order 3, 0, 4, 1, 2), columns in VALIDATION
    order, from each run's chosen checkpoint only. Arms are never pooled.

    Known answer: alpha is PROFIT, VIOL, YIELD, so IQMs 8.0, 31.0 and 2.0 (see their comment);
    bravo is each + 100, so 108.0, 131.0 and 102.0. Each run chose a different checkpoint
    (CANDIDATES[seed % 3]); the others hold 1000s.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    arms = (("alpha", "ppo", 0), ("bravo", "residual", 100))
    reg = _register(path, arms=[(a, m) for a, m, _ in arms], seeds=range(5))
    for seed in LEDGER_ORDER:
        for arm, method, shift in arms:
            seasons = _seasons(
                [v + shift for v in PROFIT[seed]],
                [v + shift for v in VIOL[seed]],
                [v + shift for v in YIELD[seed]],
            )
            spec = _spec(seasons, chosen=CANDIDATES[seed % 3])
            _run(path, tmp_path, reg, arm, seed, method=method, spec=spec)
    _trip(monkeypatch, selection)

    numbers = _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")

    for arm, shift, want in (("alpha", 0, (8.0, 31.0, 2.0)), ("bravo", 100, (108.0, 131.0, 102.0))):
        got = numbers["per_arm"][arm]
        assert _ci(got) == stats.iqm_ci(_by_seed(PROFIT, shift)), arm
        assert got["iqm"] == pytest.approx(want[0]), arm
        for g, table, iqm in zip(GUARDS, (VIOL, YIELD), want[1:], strict=True):
            assert _ci(got["guards"][g]) == stats.iqm_ci(_by_seed(table, shift)), (arm, g)
            assert got["guards"][g]["iqm"] == pytest.approx(iqm), (arm, g)
        assert got["invalid_episodes"] == 0, arm


def test_one_run_scores_its_plain_mean_with_a_collapsed_ci(tmp_path, monkeypatch):
    """Known answer: one run whose profits are (1, 2, 6). The IQM trims int(0.25 x 3) = 0 scores
    from each end, so it is the mean, (1 + 2 + 6) / 3 = 3.0 (not the median, 2.0); every bootstrap
    resample draws that one run, so the CI is [3.0, 3.0]. Likewise viol_hours_temp (4, 5, 9):
    18 / 3 = 6.0; yield_kg_m2 (0.5, 0.5, 2.0): 3 / 3 = 1.0."""
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"),), seeds=(0,))
    _run(path, tmp_path, reg, "alpha", 0, spec=_spec(_seasons((1, 2, 6), (4, 5, 9), (0.5, 0.5, 2))))
    _trip(monkeypatch, selection)

    alpha = _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")["per_arm"]["alpha"]

    assert _ci(alpha) == pytest.approx((3.0, 3.0, 3.0))
    assert _ci(alpha["guards"]["viol_hours_temp"]) == pytest.approx((6.0, 6.0, 6.0))
    assert _ci(alpha["guards"]["yield_kg_m2"]) == pytest.approx((1.0, 1.0, 1.0))


# --- Crashed runs and invalid episodes (Done when: each is counted) ----------------------------


@pytest.mark.parametrize("decoy", [False, True], ids=["bare-dir", "decoy-selected-json"])
def test_a_crashed_run_is_counted_and_never_scored(decoy, tmp_path, monkeypatch):
    """Arm alpha, seeds 0-4: seed 2 crashed and seed 4 stopped on its budget (budget-stop is not a
    crash: it is scored). The crashed run's dir has checkpoints and, bare, no selected.json, so
    scoring it would call the tripped select_checkpoint; or a decoy selected.json whose chosen
    seasons are 1000s and invalid. Either way it is not scored, not in selected, and the ledger's
    result counts crashed 1.

    Known answer: the four scored runs' profits, seeds 0, 1, 3, 4: (15, 1, 8), (2, 14, 9),
    (12, 6, 4), (5, 11, 10). The IQM trims int(0.25 x 12) = 3 from each end (1, 2, 4 and 12, 14,
    15), leaving (5 + 6 + 8 + 9 + 10 + 11) / 6 = 49 / 6 = 8.1667. invalid_episodes 0.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"),), seeds=range(5))
    runs = {}
    for seed in range(5):
        status = {2: "crashed", 4: "budget-stop"}.get(seed, "finished")
        spec = _spec(_seasons(PROFIT[seed], VIOL[seed], YIELD[seed]))
        if status == "crashed":
            spec = _spec(_decoy()) if decoy else None
        runs[seed], _ = _run(path, tmp_path, reg, "alpha", seed, status=status, spec=spec)
    _trip(monkeypatch, selection)
    out = tmp_path / "numbers.json"

    numbers = _numbers(selection, path, reg["row_id"], out)

    scored = (0, 1, 3, 4)
    alpha = numbers["per_arm"]["alpha"]
    assert alpha["iqm"] == pytest.approx(49 / 6)
    assert _ci(alpha) == stats.iqm_ci(_by_seed({s: PROFIT[s] for s in scored}))
    for g, table in zip(GUARDS, (VIOL, YIELD), strict=True):
        assert _ci(alpha["guards"][g]) == stats.iqm_ci(_by_seed({s: table[s] for s in scored})), g
    assert alpha["invalid_episodes"] == 0
    assert numbers["selected"] == {runs[s]["row_id"]: "final" for s in scored}
    assert _record(path, reg["row_id"], out)["per_arm"]["alpha"]["crashed"] == 1


def test_a_crashed_run_and_its_finished_rerun_score_the_rerun(tmp_path, monkeypatch):
    """Arm alpha, seeds 0-4: seed 2's first run crashed, its dir holding a decoy selected.json
    (chosen "final", 1000 in every metric, every season invalid), and a rerun of seed 2, appended
    last in the ledger, finished and chose "500000". The rerun is scored in seed 2's place.

    Known answer: all five seeds' PROFIT, VIOL and YIELD, so IQMs 8.0, 31.0 and 2.0 (see their
    comment), each CI stats.iqm_ci's on the rows in seed order, though the rerun is the ledger's
    last run; invalid_episodes 0 (the decoy's 3 invalid seasons aren't read). selected maps the
    rerun to "500000" and every other finished run to "final", and never names the crashed run.
    The ledger's result counts crashed 1.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"),), seeds=range(5))
    runs = {}
    for seed in range(5):
        if seed == 2:
            crashed, _ = _run(
                path, tmp_path, reg, "alpha", 2, status="crashed", spec=_spec(_decoy())
            )
        else:
            spec = _spec(_seasons(PROFIT[seed], VIOL[seed], YIELD[seed]))
            runs[seed], _ = _run(path, tmp_path, reg, "alpha", seed, spec=spec)
    rerun_spec = _spec(_seasons(PROFIT[2], VIOL[2], YIELD[2]), chosen="500000")
    runs[2], _ = _run(path, tmp_path, reg, "alpha", 2, spec=rerun_spec, tag="-rerun")
    _trip(monkeypatch, selection)
    out = tmp_path / "numbers.json"

    numbers = _numbers(selection, path, reg["row_id"], out)

    alpha = numbers["per_arm"]["alpha"]
    assert _ci(alpha) == stats.iqm_ci(_by_seed(PROFIT))
    assert alpha["iqm"] == pytest.approx(8.0)
    for g, table, iqm in zip(GUARDS, (VIOL, YIELD), (31.0, 2.0), strict=True):
        assert _ci(alpha["guards"][g]) == stats.iqm_ci(_by_seed(table)), g
        assert alpha["guards"][g]["iqm"] == pytest.approx(iqm), g
    assert alpha["invalid_episodes"] == 0
    want = {runs[s]["row_id"]: "500000" if s == 2 else "final" for s in range(5)}
    assert numbers["selected"] == want
    assert crashed["row_id"] not in numbers["selected"]
    assert _record(path, reg["row_id"], out)["per_arm"]["alpha"]["crashed"] == 1


def test_an_arm_whose_runs_all_crashed_has_null_numbers_the_ledger_accepts(tmp_path, monkeypatch):
    """An arm with no run to score (bravo: both seeds crashed) has iqm, ci_low and ci_high null and
    each guard's null too (null, not NaN, which the ledger refuses), and invalid_episodes 0. The
    ledger's result accepts the file and counts bravo's crashed 2, spend 0.0 (cost_usd None)."""
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"), ("bravo", "residual")), seeds=(0, 1))
    alpha_runs = [
        _run(path, tmp_path, reg, "alpha", seed, spec=_spec(_seasons((1, 2, 3))))[0]
        for seed in (0, 1)
    ]
    for seed in (0, 1):
        _run(path, tmp_path, reg, "bravo", seed, method="residual", status="crashed")
    _trip(monkeypatch, selection)
    out = tmp_path / "numbers.json"

    numbers = _numbers(selection, path, reg["row_id"], out)

    null = {"iqm": None, "ci_low": None, "ci_high": None}
    assert numbers["per_arm"]["bravo"] == {
        **null,
        "guards": {g: null for g in GUARDS},
        "invalid_episodes": 0,
    }
    assert numbers["selected"] == {run["row_id"]: "final" for run in alpha_runs}
    assert _record(path, reg["row_id"], out)["per_arm"]["bravo"] == {
        **numbers["per_arm"]["bravo"],
        "crashed": 2,
        "spend_usd": 0.0,
    }


def test_invalid_episodes_counts_the_chosen_seasons_and_they_are_scored(tmp_path, monkeypatch):
    """invalid_episodes is the number of valid False among the seasons the numbers come from: each
    run's chosen checkpoint's. An invalid season is scored, never dropped (PROTOCOL §4).

    Known answer: seed 0 chose 500000, valid (True, False, True); seed 1 chose final, (False, True,
    False): 1 + 2 = 3. Their other two candidates are invalid in every season (12 more: counting
    every candidate gives 15). Profits (1, 30, 2) and (40, 3, 50), the 30, 40 and 50 invalid: the
    IQM trims int(0.25 x 6) = 1 from each end (1 and 50), (2 + 3 + 30 + 40) / 4 = 18.75; without
    the invalid seasons it would be 2.0.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"),), seeds=(0, 1))
    chosen = {
        0: ("500000", (1, 30, 2), (True, False, True)),
        1: ("final", (40, 3, 50), (False, True, False)),
    }
    for seed, (name, profit, valid) in chosen.items():
        spec = _spec(_seasons(profit, valid=valid), chosen=name)
        _run(path, tmp_path, reg, "alpha", seed, spec=spec)
    _trip(monkeypatch, selection)

    alpha = _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")["per_arm"]["alpha"]

    assert alpha["invalid_episodes"] == 3
    assert alpha["iqm"] == pytest.approx(18.75)
    assert _ci(alpha) == stats.iqm_ci(np.array([[1.0, 30.0, 2.0], [40.0, 3.0, 50.0]]))


# --- Where a run's scores come from ------------------------------------------------------------


def test_an_existing_selected_json_is_used_as_is(tmp_path, monkeypatch):
    """A run whose dir has selected.json is scored from it as it is: nothing plays a season
    (run_policy, select_checkpoint and the rest are tripped), each file keeps its bytes, and
    selected maps each run's row_id to its file's checkpoint as run_policy takes it ("500000",
    "final"), not a file name.

    Known answer: alpha's chosen profits are (1, 2, 3) and (4, 5, 6): the IQM trims
    int(0.25 x 6) = 1 from each end, (2 + 3 + 4 + 5) / 4 = 3.5. bravo's are 10 x those: 35.0.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"), ("bravo", "residual")), seeds=(0, 1))
    chosen = {
        ("alpha", "ppo", 0): ("500000", (1, 2, 3)),
        ("alpha", "ppo", 1): ("final", (4, 5, 6)),
        ("bravo", "residual", 0): ("250000", (10, 20, 30)),
        ("bravo", "residual", 1): ("500000", (40, 50, 60)),
    }
    want_selected, before = {}, {}
    for (arm, method, seed), (name, profit) in chosen.items():
        spec = _spec(_seasons(profit), chosen=name)
        run, run_dir = _run(path, tmp_path, reg, arm, seed, method=method, spec=spec)
        want_selected[run["row_id"]] = name
        before[run_dir] = (run_dir / "selected.json").read_bytes()
    _trip(monkeypatch, selection)

    numbers = _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")

    assert numbers["selected"] == want_selected
    assert numbers["per_arm"]["alpha"]["iqm"] == pytest.approx(3.5)
    assert numbers["per_arm"]["bravo"]["iqm"] == pytest.approx(35.0)
    for run_dir, data in before.items():
        assert (run_dir / "selected.json").read_bytes() == data, run_dir


def test_a_run_without_selected_json_goes_through_select_checkpoint_once(tmp_path, monkeypatch):
    """A run dir with no selected.json is selected through selection.select_checkpoint(run_dir),
    the module function, once; it writes selected.json, and its chosen checkpoint's per_scenario
    results are the run's scores, with no second evaluation: run_policy plays each candidate's
    three VALIDATION seasons once, for the selection, and never again.

    Known answer (run_policy faked): mean paper_return 1 (250000), 0 (500000), 2 (final), so
    select_checkpoint picks final, and selected maps the run to "final". One run, so the IQM is the
    mean of final's profits, (5 + 6 + 10) / 3 = 7.0, CI [7.0, 7.0]; viol_hours_temp (1, 1, 1):
    1.0; yield_kg_m2 (2, 2, 2): 2.0. run_policy is called 3 checkpoints x 3 scenarios = 9 times.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"),), seeds=(0,))
    run, run_dir = _run(path, tmp_path, reg, "alpha", 0)
    table = {
        "250000": _seasons((100, 200, 300), paper_return=(1, 1, 1)),
        "500000": _seasons((400, 500, 600), paper_return=(0, 0, 0)),
        "final": _seasons((5, 6, 10), (1, 1, 1), (2, 2, 2), paper_return=(2, 2, 2)),
    }
    real = selection.select_checkpoint
    _trip(monkeypatch, selection)
    spied, played = [], []

    def select_checkpoint(run_dir):
        spied.append(Path(run_dir))
        return real(run_dir)

    def run_policy(run_dir, checkpoint, scenario, season_length=60, record_steps=False):
        played.append((checkpoint, scenario["start_day"], season_length))
        if scenario not in scenarios.VALIDATION:
            raise AssertionError(f"run_policy played {scenario}, not a VALIDATION scenario")
        return dict(table[checkpoint][scenarios.VALIDATION.index(scenario)]), None

    monkeypatch.setattr(selection, "select_checkpoint", select_checkpoint)
    monkeypatch.setattr(evaluate, "run_policy", run_policy)

    numbers = _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")

    assert [p.resolve() for p in spied] == [run_dir.resolve()]
    assert sorted(played) == sorted(
        (name, s["start_day"], 60) for name in CANDIDATES for s in scenarios.VALIDATION
    )
    assert json.loads((run_dir / "selected.json").read_text())["checkpoint"] == "final"
    assert numbers["selected"] == {run["row_id"]: "final"}
    alpha = numbers["per_arm"]["alpha"]
    assert _ci(alpha) == pytest.approx((7.0, 7.0, 7.0))
    assert _ci(alpha["guards"]["viol_hours_temp"]) == pytest.approx((1.0, 1.0, 1.0))
    assert _ci(alpha["guards"]["yield_kg_m2"]) == pytest.approx((2.0, 2.0, 2.0))


def test_rb_tuned_is_scored_with_its_run_dirs_params_on_validation(tmp_path, monkeypatch):
    """An rb_tuned run (as tune_rb records one: seeds [0], timesteps 0, config {"trials": N}) is
    scored with evaluate.run_rb(params, s, 60) for s in VALIDATION, in order, where params is its run
    dir's rule_based_tuned.yml's GreenLightEnv block: the run's own copy, never configs/
    (load_rb_params is tripped). It has no checkpoint, so it is not in selected; the ledger accepts
    the file.

    Known answer (run_rb faked): profit (5, 6, 10), viol_hours_temp (1, 2, 3), yield_kg_m2 (2, 2,
    8), the second season invalid. One run, so each IQM is the mean and its CI collapses on it:
    21 / 3 = 7.0, 6 / 3 = 2.0, 12 / 3 = 4.0; invalid_episodes 1.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("rb", "rb_tuned"),), seeds=(0,), timesteps=0)
    defaults = yaml.safe_load((rule_based.CONFIGS / "rule_based_default.yml").read_text())
    tuned = {**defaults["GreenLightEnv"], "temp_setpoint_day": 21.25, "co2_day": 937.5}
    tuned["lamps_off"] = 16.5
    assert tuned != defaults["GreenLightEnv"]
    run_dir = tmp_path / "runs" / reg["row_id"] / "rb" / "seed0"
    run_dir.mkdir(parents=True)
    config = {"method": "rb_tuned", "seed": 0, "trials": 2, "jobs": 1, "season_length": 60}
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}, indent=2) + "\n")
    text = yaml.safe_dump({"GreenLightEnv": tuned}, sort_keys=False)
    (run_dir / "rule_based_tuned.yml").write_text(text)
    run = ledger.run_start(
        registered=reg["row_id"], arm="rb", seed=0, run_dir=str(run_dir), path=path
    )
    ledger.run_end(
        run=run["row_id"],
        status="finished",
        wall_s=3.0,
        timesteps=0,
        solver_failures=0,
        cost_usd=None,
        diagnostics=dict.fromkeys(ledger.DIAGNOSTIC_KEYS),
        path=path,
    )
    seasons = _seasons((5, 6, 10), (1, 2, 3), (2, 2, 8), valid=(True, False, True))
    _trip(monkeypatch, selection)
    played = []

    def run_rb(params, scenario, season_length=60, record_steps=False):
        played.append((dict(params), dict(scenario), season_length))
        if scenario not in scenarios.VALIDATION:
            raise AssertionError(f"run_rb played {scenario}, not a VALIDATION scenario")
        return dict(seasons[scenarios.VALIDATION.index(scenario)]), None

    monkeypatch.setattr(evaluate, "run_rb", run_rb)
    out = tmp_path / "numbers.json"

    numbers = _numbers(selection, path, reg["row_id"], out)

    assert played == [(tuned, s, 60) for s in scenarios.VALIDATION]
    rb = numbers["per_arm"]["rb"]
    assert _ci(rb) == pytest.approx((7.0, 7.0, 7.0))
    assert _ci(rb["guards"]["viol_hours_temp"]) == pytest.approx((2.0, 2.0, 2.0))
    assert _ci(rb["guards"]["yield_kg_m2"]) == pytest.approx((4.0, 4.0, 4.0))
    assert rb["invalid_episodes"] == 1
    assert numbers["selected"] == {}
    assert _record(path, reg["row_id"], out)["per_arm"]["rb"] == {
        **rb,
        "crashed": 0,
        "spend_usd": 0.0,
    }


# --- The ledger round trip (Done when: ledger result accepts the file) --------------------------


def test_the_numbers_file_round_trips_through_ledger_result(tmp_path, monkeypatch):
    """`ledger result --registered <row> --numbers <file>` accepts the file. It is exactly
    {per_arm, selected}; per_arm[arm] exactly {iqm, ci_low, ci_high, guards, invalid_episodes},
    guards exactly the registration's, each {iqm, ci_low, ci_high}; no crashed or spend_usd, which
    the ledger refuses from a caller and adds from the run-ends. selected names the 5 scored runs.

    Known answer (the ledger's tally): alpha's seeds 0, 1, 2 cost 0.5, 0.25 and None USD and seed 2
    crashed: crashed 1, spend 0.5 + 0.25 + 0 = 0.75 USD. bravo's cost 1.0, None, None, none
    crashed: crashed 0, spend 1.0 USD.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    arms = (("alpha", "ppo"), ("bravo", "residual"))
    reg = _register(path, arms=arms, seeds=range(3))
    costs = {("alpha", 0): 0.5, ("alpha", 1): 0.25, ("bravo", 0): 1.0}
    want_selected = {}
    for seed in range(3):
        for arm, method in arms:
            crashed = (arm, seed) == ("alpha", 2)
            run, _ = _run(
                path,
                tmp_path,
                reg,
                arm,
                seed,
                method=method,
                status="crashed" if crashed else "finished",
                cost_usd=costs.get((arm, seed)),
                spec=None if crashed else _spec(_seasons(PROFIT[seed], VIOL[seed], YIELD[seed])),
            )
            if not crashed:
                want_selected[run["row_id"]] = "final"
    _trip(monkeypatch, selection)
    out = tmp_path / "numbers.json"

    numbers = _numbers(selection, path, reg["row_id"], out)

    assert set(numbers) == {"per_arm", "selected"}
    assert set(numbers["per_arm"]) == {"alpha", "bravo"}
    for arm, given in numbers["per_arm"].items():
        assert set(given) == NUMBERS_KEYS, arm
        assert set(given["guards"]) == set(GUARDS), arm
        for g, ci in given["guards"].items():
            assert set(ci) == CI_KEYS, (arm, g)
    assert numbers["selected"] == want_selected
    res = _record(path, reg["row_id"], out)
    assert res["per_arm"] == {
        "alpha": {**numbers["per_arm"]["alpha"], "crashed": 1, "spend_usd": 0.75},
        "bravo": {**numbers["per_arm"]["bravo"], "crashed": 0, "spend_usd": 1.0},
    }
    assert res["selected"] == numbers["selected"]


# --- Refusals (Done when: refused before any scoring) ------------------------------------------

# (case, names): case changes one thing in _refusal_setup's base; the refusal must name one of
# names ("{key}" is the run id _refusal_setup returns under key).
REFUSALS = [
    pytest.param({"row": "registered-00000000"}, ["registered-00000000"], id="unknown-row"),
    pytest.param({"scenarios": ["train_eval"]}, ["train_eval"], id="train-eval"),
    *[pytest.param({"scenarios": [split]}, [split], id=split) for split in scenarios.TEST],
    pytest.param(
        {"scenarios": ["validation", "amsterdam_2012"]},
        ["amsterdam_2012"],
        id="validation-and-a-test-split",
    ),
    pytest.param({"scenarios": ["validation", "paper"]}, ["paper"], id="validation-and-paper"),
    *[
        pytest.param({"methods": {"bravo": m}}, [m], id=f"{m}-arm")
        for m in ("rb_default", "zero_controls", "random")
    ],
    pytest.param({"open": ("alpha", 1)}, ["run-end", "{open}"], id="open-run"),
    pytest.param(
        {"open": ("alpha", 1), "rerun": ("alpha", 1)},
        ["run-end", "{open}"],
        id="open-run-beside-a-finished-rerun",
    ),
    pytest.param({"missing": ("alpha", 1)}, ["alpha"], id="arm-seed-never-run"),
    pytest.param({"rerun": ("bravo", 0)}, ["bravo", "{first}", "{rerun}"], id="two-finished-runs"),
    pytest.param({"out_exists": True}, ["numbers.json"], id="existing-out"),
    pytest.param({"primary": "profit"}, ["profit"], id="primary-not-a-metric"),
    pytest.param({"primary": "valid"}, ["valid"], id="primary-is-the-valid-flag"),
    pytest.param(
        {"guards": ["viol_hours_temp", "violation_hours"]},
        ["violation_hours"],
        id="guard-not-a-metric",
    ),
    pytest.param({"stale": {"season_length": 2}}, ["selected.json"], id="stale-season-length"),
    pytest.param(
        {"stale": {"scenarios": scenarios.TRAIN_EVAL[:3]}},
        ["selected.json"],
        id="stale-scenarios",
    ),
]


def _refusal_setup(path: Path, root: Path, case: dict) -> tuple[str, dict]:
    """The base, which main would score: arms alpha (ppo) and bravo (residual), seeds 0 and 1, every
    run finished, scenarios ["validation"], primary PRIMARY, guards GUARDS. alpha seed 0's dir has
    no selected.json, so scoring it calls the tripped select_checkpoint; every other run's has one
    (bravo seed 1's with case's stale keys). case changes one thing. Returns the row to pass and the
    run ids a message may name: "open", "first" and "rerun"."""
    methods = {"alpha": "ppo", "bravo": "residual", **case.get("methods", {})}
    over = {key: case[key] for key in ("scenarios",) if key in case}
    unscorable = {key: case[key] for key in ("primary", "guards") if key in case}
    reg = _register(path, arms=tuple(methods.items()), seeds=(0, 1), dry=bool(unscorable), **over)
    if unscorable:
        # ledger.register refuses a primary or guard that isn't a result key (#25), so plant the
        # row as one written before that check would sit in the ledger: a valid row, then edited.
        reg = {**reg, **unscorable}
        with path.open("a") as f:
            f.write(json.dumps(reg, sort_keys=True, allow_nan=False) + "\n")
    ids = {}
    for arm, method in methods.items():
        for seed in (0, 1):
            if (arm, seed) == case.get("missing"):
                continue
            run, _ = _run(
                path,
                root,
                reg,
                arm,
                seed,
                method=method,
                spec=None if (arm, seed) == ("alpha", 0) else _spec(_seasons((1, 2, 3))),
                end=(arm, seed) != case.get("open"),
                stale=case.get("stale") if (arm, seed) == ("bravo", 1) else None,
            )
            ids[(arm, seed)] = run["row_id"]
    if "open" in case:
        ids["open"] = ids[case["open"]]
    if "rerun" in case:
        arm, seed = case["rerun"]
        spec = _spec(_seasons((1, 2, 3)))
        run, _ = _run(path, root, reg, arm, seed, method=methods[arm], spec=spec, tag="-rerun")
        ids["first"], ids["rerun"] = ids[(arm, seed)], run["row_id"]
    return case.get("row", reg["row_id"]), ids


@pytest.mark.parametrize(("case", "names"), REFUSALS)
def test_refuses_before_scoring_or_writing_anything(case, names, tmp_path, monkeypatch, capsys):
    """Each case is _refusal_setup's base with one thing wrong (see the module docstring). main
    refuses with a message naming it (one of names), plays no season, calls neither
    select_checkpoint nor stats.iqm_ci, and writes nothing: every file and dir under tmp_path (the
    ledger, the run dirs and their selected.json, --out, selection.RESULTS) is as it was."""
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    row, ids = _refusal_setup(path, tmp_path, case)
    out = tmp_path / "numbers.json"
    if case.get("out_exists"):
        out.write_text('{"per_arm": {}, "selected": {}}\n')
    _trip(monkeypatch, selection)
    monkeypatch.setattr(stats, "iqm_ci", _tripwire("stats.iqm_ci"))
    before = _tree(tmp_path)

    said = _refused(selection, ["--registered", row, "--out", str(out)], path, capsys)

    names = [ids[n.strip("{}")] if n.startswith("{") else n for n in names]
    assert any(n in said for n in names), f"the refusal names none of {names}: {said!r}"
    assert _tree(tmp_path) == before


# --- What it prints, and where it writes ---------------------------------------------------------


@pytest.mark.parametrize("n", [1, 5])
def test_prints_one_line_per_arm_flagged_under_five_runs(n, tmp_path, monkeypatch, capsys):
    """main prints one line per arm and nothing else: the arm with its primary IQM [CI], each guard,
    its run count and invalid episodes. Under 5 runs the line says "n<5: no claim" (no claim on
    n < 5); at 5 runs no line has "n<5"."""
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    arms = (("alpha", "ppo"), ("bravo", "residual"))
    reg = _register(path, arms=arms, seeds=range(n))
    for seed in range(n):
        for arm, method in arms:
            spec = _spec(_seasons(PROFIT[seed], VIOL[seed], YIELD[seed]))
            _run(path, tmp_path, reg, arm, seed, method=method, spec=spec)
    _trip(monkeypatch, selection)
    capsys.readouterr()

    _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == len(arms), lines
    for arm, _ in arms:
        mine = [line for line in lines if arm in line]
        assert len(mine) == 1, (arm, lines)
        assert all(g in mine[0] for g in GUARDS), mine[0]
        assert (NO_CLAIM in mine[0]) is (n < 5), mine[0]
    if n >= 5:
        assert not any("n<5" in line for line in lines), lines


def _rb_tuned_run(path: Path, root: Path, reg: dict, arm: str, seed: int = 0) -> dict:
    """An rb_tuned run as tune_rb records one: a run dir holding manifest.json and
    rule_based_tuned.yml (the default params here), a run row naming it and a finished run-end
    (timesteps 0, diagnostics null). Returns the run row."""
    run_dir = root / "runs" / reg["row_id"] / arm / f"seed{seed}"
    run_dir.mkdir(parents=True)
    config = {"method": "rb_tuned", "seed": seed, "trials": 2, "jobs": 1, "season_length": 60}
    (run_dir / "manifest.json").write_text(json.dumps({"config": config}, indent=2) + "\n")
    defaults = yaml.safe_load((rule_based.CONFIGS / "rule_based_default.yml").read_text())
    (run_dir / "rule_based_tuned.yml").write_text(yaml.safe_dump(defaults, sort_keys=False))
    run = ledger.run_start(
        registered=reg["row_id"], arm=arm, seed=seed, run_dir=str(run_dir), path=path
    )
    ledger.run_end(
        run=run["row_id"],
        status="finished",
        wall_s=3.0,
        timesteps=0,
        solver_failures=0,
        cost_usd=None,
        diagnostics=dict.fromkeys(ledger.DIAGNOSTIC_KEYS),
        path=path,
    )
    return run


def _fake_run_rb(monkeypatch, seasons: list[dict]) -> None:
    """evaluate.run_rb returns seasons[i] for VALIDATION[i] and refuses any other scenario."""

    def run_rb(params, scenario, season_length=60, record_steps=False):
        if scenario not in scenarios.VALIDATION:
            raise AssertionError(f"run_rb played {scenario}, not a VALIDATION scenario")
        return dict(seasons[scenarios.VALIDATION.index(scenario)]), None

    monkeypatch.setattr(evaluate, "run_rb", run_rb)


def _arm_lines(out: str, arms) -> dict[str, str]:
    """The printed lines, one per arm and nothing else, keyed by the arm each starts with."""
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == len(arms), lines
    by_arm = {line.split()[0]: line for line in lines}
    assert set(by_arm) == set(arms), lines
    return by_arm


def test_rb_tuneds_one_run_prints_deterministic_not_no_claim(tmp_path, monkeypatch, capsys):
    """E1's shape (#22): an rb_tuned arm with one seed, primary paper_return, guards
    viol_hours_co2, viol_hours_temp and viol_hours_rh. A deterministic arm needs one run (PROTOCOL
    §6), so its line says "deterministic, 1 run" and has no "n<5". A ppo arm's one run in the same
    registration still says "n<5: no claim" and not "deterministic": the flag follows the method,
    not the run count alone."""
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    guards = [f"viol_hours_{c}" for c in metrics.CONSTRAINTS]
    arms = (("rb", "rb_tuned"), ("alpha", "ppo"))
    reg = _register(path, arms=arms, seeds=(0,), primary="paper_return", guards=guards)
    _rb_tuned_run(path, tmp_path, reg, "rb")
    _run(path, tmp_path, reg, "alpha", 0, spec=_spec(_seasons((1, 2, 3), paper_return=(1, 2, 3))))
    _trip(monkeypatch, selection)
    _fake_run_rb(monkeypatch, _seasons((5, 6, 10), paper_return=(4, 5, 6)))
    capsys.readouterr()

    _numbers(selection, path, reg["row_id"], tmp_path / "numbers.json")

    lines = _arm_lines(capsys.readouterr().out, ("rb", "alpha"))
    assert "deterministic, 1 run" in lines["rb"], lines["rb"]
    assert "n<5" not in lines["rb"], lines["rb"]
    assert NO_CLAIM in lines["alpha"], lines["alpha"]
    assert "deterministic" not in lines["alpha"], lines["alpha"]


def test_each_arms_line_shows_its_crashed_count(tmp_path, monkeypatch, capsys):
    """Each arm's line shows "crashed N", N the arm's crashed run-ends: the count the ledger's
    result records, a rerun's crash included.

    Known answer, seeds 0 and 1: alpha's seed 0 crashed and its rerun finished, and seed 1 crashed
    with no rerun: crashed 2. bravo's two runs finished: crashed 0.
    """
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"), ("bravo", "residual")), seeds=(0, 1))
    for seed in (0, 1):
        _run(path, tmp_path, reg, "alpha", seed, status="crashed")
        spec = _spec(_seasons(PROFIT[seed], VIOL[seed], YIELD[seed]))
        _run(path, tmp_path, reg, "bravo", seed, method="residual", spec=spec)
    _run(path, tmp_path, reg, "alpha", 0, spec=_spec(_seasons((1, 2, 3))), tag="-rerun")
    _trip(monkeypatch, selection)
    out = tmp_path / "numbers.json"
    capsys.readouterr()

    _numbers(selection, path, reg["row_id"], out)

    lines = _arm_lines(capsys.readouterr().out, ("alpha", "bravo"))
    res = _record(path, reg["row_id"], out)
    for arm, want in (("alpha", 2), ("bravo", 0)):
        assert re.findall(r"\bcrashed (\d+)\b", lines[arm]) == [str(want)], lines[arm]
        assert res["per_arm"][arm]["crashed"] == want, arm


def test_out_defaults_to_results_row_json_and_is_written_once(tmp_path, monkeypatch, capsys):
    """Without --out the file is selection.RESULTS / "<row>.json" (RESULTS is control/results/
    validation, patched here to a tmp dir that doesn't exist yet, as control/results/ doesn't). A
    second run refuses and leaves the first file's bytes, alone in RESULTS."""
    selection = _selection(monkeypatch, tmp_path)
    path = tmp_path / "ledger.jsonl"
    reg = _register(path, arms=(("alpha", "ppo"),), seeds=(0,))
    _run(path, tmp_path, reg, "alpha", 0, spec=_spec(_seasons((1, 2, 6))))
    _trip(monkeypatch, selection)
    out = tmp_path / "results" / "validation" / f"{reg['row_id']}.json"

    assert selection.main(["--registered", reg["row_id"]], ledger_path=path) == 0

    assert set(json.loads(out.read_text())) == {"per_arm", "selected"}
    written = out.read_bytes()
    _refused(selection, ["--registered", reg["row_id"]], path, capsys)
    assert out.read_bytes() == written
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]


# --- The contract ------------------------------------------------------------------------------


def test_result_keys_are_a_real_seasons_numeric_keys_plus_paper_reward_sum():
    """selection.RESULT_KEYS, what a primary or guard may name, is exactly the numeric keys of a
    real metrics.episode_metrics output (every key but its bool valid) plus evaluate's
    paper_reward_sum, with no duplicates. So a metric added to or renamed in metrics fails here,
    not after seasons are scored. The output grades one real PAPER day (96 steps, not TEST weather)
    of zero controls as evaluate grades a season: trace_from_infos, the model's params and N."""
    import greenleaf_control.selection as selection

    infos = envs.run_baseline("zero_controls", scenarios.PAPER, 1)
    raw = envs.make_raw_env(scenario=scenarios.PAPER, season_length=1, normalize_actions=False)
    params, season_steps = np.asarray(raw.unwrapped.p), raw.unwrapped.N
    raw.close()
    states, controls = metrics.trace_from_infos(infos)
    graded = metrics.episode_metrics(states, controls, params, season_steps=season_steps)

    numeric = {
        k for k, v in graded.items() if isinstance(v, int | float) and not isinstance(v, bool)
    }
    assert set(graded) - numeric == {"valid"}
    assert len(selection.RESULT_KEYS) == len(set(selection.RESULT_KEYS))
    assert set(selection.RESULT_KEYS) == numeric | {"paper_reward_sum"}


def test_main_matches_the_contract_and_no_test_wrote_the_real_ledger_runs_or_results():
    """main(argv=None, *, ledger_path=ledger.LEDGER), as train and tune_rb; RESULTS is
    control/results/validation.

    This test runs last in the module and checks the real ledger's size and mtime, and the listings
    of control/runs/ and control/results/, are as they were when the module was collected.
    """
    import greenleaf_control.selection as selection

    assert selection.RESULTS == CONTROL / "results" / "validation"
    params = inspect.signature(selection.main).parameters
    assert list(params) == ["argv", "ledger_path"]
    assert params["argv"].default is None
    assert params["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["ledger_path"].default == ledger.LEDGER
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE
    assert _listing(REAL_RESULTS) == REAL_RESULTS_BEFORE

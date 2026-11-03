"""Tests for evaluate --final (issue #43, F1a-3): score every registered run at its registered
checkpoint and every registered baseline once on the three test splits, after refusing anything that
could break the seal (PROTOCOL §8). Issue #44 (F1a-4): the same --final writes export_replay's
hourly JSON for every season of each learning arm's lowest registered seed and of every baseline,
from the seasons it scores (no second season), into FINAL_RESULTS/<row>/replays/. #44's fix round:
a replay that fails, or a traced season cut short, costs no scored file; a short season isn't
replayed, the rest are, and one RuntimeError names it.

The test rule (F1a): no test opens the real 2012 weather. One module fixture, under a single
pytest.MonkeyPatch.context() for the whole module:
- puts a pandas.read_csv tripwire on any *2012.csv outside pytest's tmp dirs, recording each read;
- replaces manifest.code_stamp with a fixed stamp, so no test runs git or hashes the real weather;
- trains two residual smoke runs (seeds 0 and 1, one env, 2,048 steps, 1-day seasons, rb default)
  on the real 2008-2011 weather;
- registers seed 0 at "final" and seed 1 at "1024", a checkpoint whose policy differs from its
  "final": ckpt_1024.zip is its ckpt_final.zip with action_net's weight and bias zeroed (so it
  plays the RB exactly, test_evaluate's known answer) and vecnorm_1024.pkl a copy of its
  vecnorm_final.pkl; each run's selected.json names its checkpoint (CHECKPOINTS);
- writes a tmp ledger chain with ledger's own functions: registered -> a run and a run-end per seed
  -> result (selected: seed 0 at "final", seed 1 at "1024") -> final-registered (the same
  checkpoints, the four BASELINES: zero_controls, random, rb_default, rb_tuned);
- then points scenarios.WEATHER_DIR at tmp "synthetic", where Amsterdam, London and Abbotsford each
  have a 2012.csv that is a byte copy of the real Amsterdam/2010.csv; cuts scenarios.TEST to start
  days 45 and 49 per split; sets evaluate.SEASON_LENGTH to 1 day (96 steps) and
  evaluate.FINAL_RESULTS to tmp "final_out".

One full --final runs once (final_run) and is shared; pass-through spies record its calls to
evaluate.run_policy, evaluate.run_rb, envs.run_baseline and manifest.code_stamp. A test that
changes the ledger, a run dir or FINAL_RESULTS works on its own copies (_copy_world).
evaluate.main runs in-process, since guard_runs blocks `evaluate --final` on the command line. No
fixture calls the --final, so while it is missing every test FAILS in its own body rather than
ERRORs.

Issue #71, the second seal: scenarios.SEALS = {2012: TEST, 2013: TEST_2013}, each spent once. The
module fixture cuts SEALS[2012] with TEST, and its tripwire fails on any real *2013.csv too (2013
reads from tmp are listed apart, in .reads_2013). The 2013 tests build their world in their own
body (_world_2013): a copied world where 2012 is spent (its files in FINAL_RESULTS, its
final-result recorded), synthetic 2013.csv files in tmp that are byte copies of the real
Amsterdam/2010.csv (never the real 2013 weather), SEALS[2013] cut to days 45 and 49, and a seal-2013
final-registered row of seed 0 at "final" with rb_default and rb_tuned.

Issue #72: the 2013 final plays each (run or baseline, scenario) once and grades it at both price
regimes, sim and agc2020: two seasons.parquet rows per season (price_regime after season_length),
a per_split nested {split: {regime: {arm: entry}}} with each regime's stats on its rows alone, and
rb_tuned_agc2020 registrable as a baseline. Most #72 tests share one 2013 final (final_run_2013) in
a dual world: both seeds registered, the four BASELINES, and seed 0's copied manifest set to
price_regime agc2020, so the traced run plays at agc2020 and seed 1 at sim. The 2012 path is
unchanged: the tests above pin it as it was.
"""

import ast
import contextlib
import inspect
import io
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from stable_baselines3 import PPO

from greenleaf_control import (
    envs,
    evaluate,
    export_replay,
    ledger,
    manifest,
    metrics,
    rule_based,
    scenarios,
    stats,
    train,
)

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"
REAL_RESULTS = CONTROL / "results"  # its top level only: never list inside results/final/


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger, runs dir and results dir, read when this module is collected.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)
REAL_RESULTS_BEFORE = _listing(REAL_RESULTS)

N_DAY = 96  # 86,400 s / 900 s
ARM = "residual"  # the one learning arm
SITES = {"amsterdam_2012": "Amsterdam", "london_2012": "London", "abbotsford_2012": "Abbotsford"}
TEST_SPLITS = tuple(SITES)
# The four baselines the module's rows register, in the row's order: ledger.FINAL_BASELINES as it
# was before #72, written out because #72 adds rb_tuned_agc2020 to it and the 2012 world must not
# change (rb_tuned_agc2020 has its own 2013 test).
BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned")
PRIMARY = "profit_eur_m2"
FINAL_GUARDS = (
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "paper_return",
    "yield_kg_m2",
    "solver_failures",
)
IDENTITY = [
    "arm",
    "run",
    "seed",
    "checkpoint",
    "split",
    "location",
    "growth_year",
    "start_day",
    "season_length",
]
RB_DEFAULT = rule_based.load_rb_params("default")
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}
CHECKPOINTS = ("final", "1024")  # seed 0's and seed 1's registered (and selected) checkpoints
DAYS = (45, 49)  # the start days of each test split's two scenarios (scenarios.TEST is cut to them)
# #44: a replay's file prefix: the residual arm's lowest registered seed (seed 0), then each baseline.
TRACED = (ARM, *BASELINES)
REPLAY_META = {
    "controller",
    "scenario",
    "split",
    "run",
    "commit",
    "dirty",
    "gl_gym",
    "note",
    "bounds",
    "units",
}
MAX_BYTES = 250_000  # export_replay's ceiling, 250 KB, in decimal kilobytes


@pytest.fixture(scope="module", autouse=True)
def world(tmp_path_factory):
    """The module's runs, tmp ledger chain and patches (see the module docstring). About 15 s."""
    root = tmp_path_factory.mktemp("final")
    synthetic = root / "synthetic"
    for site in SITES.values():
        (synthetic / site).mkdir(parents=True)
        real_2010 = scenarios.WEATHER_DIR / "Amsterdam" / "2010.csv"
        shutil.copyfile(real_2010, synthetic / site / "2012.csv")
    w = SimpleNamespace(
        synthetic=synthetic,
        runs=root / "runs",
        ledger=root / "ledger.jsonl",
        results=root / "final_out",
        reads=[],
        reads_2013=[],
        trips=[],
        final_results_default=getattr(evaluate, "FINAL_RESULTS", None),
        stamp={
            "commit": "0" * 40,
            "dirty": False,
            "fingerprints": manifest.fingerprints(),
            "weather_hashes": {},
        },
    )
    tmp_root = tmp_path_factory.getbasetemp().resolve()
    real_read_csv = pd.read_csv

    def read_csv(filepath_or_buffer, *args, **kwargs):
        path = Path(str(filepath_or_buffer)).resolve()
        if path.name.endswith(("2012.csv", "2013.csv")):  # 2013: the second seal (#71)
            if not path.is_relative_to(tmp_root):
                w.trips.append(path)
                raise AssertionError(f"the sealed {path.stem} test weather was read: {path}")
            (w.reads if path.name.endswith("2012.csv") else w.reads_2013).append(path)
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pd, "read_csv", read_csv)
        mp.setattr(manifest, "code_stamp", lambda *, smoke: dict(w.stamp))
        for seed in (0, 1):
            argv = [
                *("--method", "residual"),
                *("--seed", str(seed)),
                "--smoke",
                *("--runs-dir", str(w.runs)),
                *("--n-envs", "1"),
                *("--timesteps", "2048"),
                *("--season-length", "1"),
                *("--rb", "default"),
            ]
            assert train.main(argv, ledger_path=root / "smoke_ledger.jsonl") == 0
        w.run_dirs = sorted((w.runs / "smoke").iterdir())  # residual-seed0-..., residual-seed1-...
        assert [d.name.split("-")[1] for d in w.run_dirs] == ["seed0", "seed1"]
        seed1 = w.run_dirs[1]
        model = PPO.load(seed1 / "ckpt_final.zip", device="cpu")
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
        model.save(seed1 / "ckpt_1024.zip")
        shutil.copyfile(seed1 / "vecnorm_final.pkl", seed1 / "vecnorm_1024.pkl")
        for run_dir, c in zip(w.run_dirs, CHECKPOINTS, strict=True):
            (run_dir / "selected.json").write_text(json.dumps({"checkpoint": c}))

        reg = ledger.register(
            issue=43,
            hypothesis="two residual smoke runs, scored once on synthetic test weather",
            arms=[{"name": ARM, "method": "residual", "config": {}}],
            seeds=[0, 1],
            timesteps=2048,
            scenarios=["validation"],
            primary=PRIMARY,
            guards=["viol_hours_temp"],
            rule="a test fixture",
            budget={"cpu_hours": 0.0, "usd": 0.0, "backend": "local"},
            path=w.ledger,
        )
        w.reg_id, w.run_ids = reg["row_id"], []
        for seed, run_dir in enumerate(w.run_dirs):
            run = ledger.run_start(
                registered=w.reg_id, arm=ARM, seed=seed, run_dir=str(run_dir), path=w.ledger
            )
            ledger.run_end(
                run=run["row_id"],
                status="finished",
                wall_s=6.0,
                timesteps=2048,
                solver_failures=0,
                cost_usd=None,
                diagnostics=DIAGNOSTICS,
                path=w.ledger,
            )
            w.run_ids.append(run["row_id"])
        w.checkpoints = dict(zip(w.run_ids, CHECKPOINTS, strict=True))
        numbers = {"iqm": 1.0, "ci_low": 0.0, "ci_high": 2.0, "guards": {}, "invalid_episodes": 0}
        ledger.record_result(
            registered=w.reg_id, per_arm={ARM: numbers}, selected=w.checkpoints, path=w.ledger
        )
        w.freg_id = ledger.final_register(
            issue=43,
            checkpoints=w.checkpoints,
            baselines=list(BASELINES),
            approved_by="Aarmen",
            path=w.ledger,
        )["row_id"]
        w.ledger_bytes = w.ledger.read_bytes()

        mp.setattr(scenarios, "WEATHER_DIR", synthetic)
        test = {
            split: [{"location": site, "growth_year": 2012, "start_day": d} for d in (45, 49)]
            for split, site in SITES.items()
        }
        mp.setattr(scenarios, "TEST", test)
        # #71: SEALS[2012] is TEST, so it is cut the same way. Set as a fresh dict (and created
        # while scenarios has no SEALS) so the fixture never ERRORs before #71 lands.
        seals = {**getattr(scenarios, "SEALS", {}), 2012: test}
        mp.setattr(scenarios, "SEALS", seals, raising=False)
        mp.setattr(evaluate, "SEASON_LENGTH", 1)
        mp.setattr(evaluate, "FINAL_RESULTS", w.results, raising=False)
        yield w


def _main(argv: list[str], ledger_path: Path) -> int:
    """evaluate.main(argv, ledger_path=ledger_path), in-process."""
    assert "ledger_path" in inspect.signature(evaluate.main).parameters, (
        "evaluate.main takes no ledger_path: main(argv=None, *, ledger_path=ledger.LEDGER)"
    )
    return evaluate.main(argv, ledger_path=ledger_path)


def _spy(real, calls: list[dict]):
    """real, passed through, recording each call's bound arguments (defaults applied) in calls."""
    signature = inspect.signature(real)

    def spy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return real(*args, **kwargs)

    return spy


@pytest.fixture(scope="module")
def final_run(world):
    """A function that runs one full evaluate --final <the world's row> into world.results on its
    first call and returns {code, stdout, dir, calls}. calls maps run_policy, run_rb, run_baseline
    and code_stamp to the run's calls of evaluate.run_policy, evaluate.run_rb, envs.run_baseline
    and manifest.code_stamp, each call's bound arguments (pass-through spies: the real function
    still plays the season). A failed attempt isn't cached, so each test that needs it fails in
    its own body."""
    done = {}

    def run() -> SimpleNamespace:
        if not done:
            out, calls = io.StringIO(), {}
            with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
                mp.setattr(evaluate, "FINAL_RESULTS", world.results)
                for module, name in [
                    (evaluate, "run_policy"),
                    (evaluate, "run_rb"),
                    (envs, "run_baseline"),
                    (manifest, "code_stamp"),
                ]:
                    calls[name] = []
                    mp.setattr(module, name, _spy(getattr(module, name), calls[name]))
                code = _main(["--final", world.freg_id], world.ledger)
            done.update(
                code=code,
                stdout=out.getvalue(),
                dir=world.results / world.freg_id,
                calls=calls,
            )
        return SimpleNamespace(**done)

    return run


def _copy_world(world, tmp_path, monkeypatch) -> SimpleNamespace:
    """world with its ledger and run dirs copied into tmp_path (the ledger's run_dir fields point
    at the copies) and FINAL_RESULTS at tmp_path/"final_out", which doesn't exist yet."""
    runs = tmp_path / "runs"
    shutil.copytree(world.runs, runs)
    text = world.ledger.read_text()
    assert text.count(str(world.runs)) == 2, "the ledger should name each run dir once"
    copy = tmp_path / "ledger.jsonl"
    copy.write_text(text.replace(str(world.runs), str(runs)))
    results = tmp_path / "final_out"
    monkeypatch.setattr(evaluate, "FINAL_RESULTS", results)
    run_dirs = [runs / d.relative_to(world.runs) for d in world.run_dirs]
    return SimpleNamespace(
        **{**vars(world), "ledger": copy, "results": results, "run_dirs": run_dirs}
    )


def _tree(path: Path) -> dict[str, bytes] | None:
    """Every file under path by relative path, with its bytes; None if path doesn't exist."""
    if not path.exists():
        return None
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def _null(value):
    """None for a parquet null (pandas 3 reads a null str or int as NaN), else the value."""
    return None if pd.isna(value) else value


def _no_env_built(monkeypatch) -> None:
    """Every way evaluate could build an env or play a season fails: envs.make_raw_env, make_env,
    run_baseline, train.build_vec_env, and the same names (plus run_rb, run_policy) on evaluate
    itself, in case it imported them by name."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any env")

        return refuse

    for module, name in [
        (envs, "make_raw_env"),
        (envs, "make_env"),
        (envs, "run_baseline"),
        (train, "build_vec_env"),
    ]:
        monkeypatch.setattr(module, name, trip(f"{module.__name__}.{name}"))
    for name in (
        "make_raw_env",
        "make_env",
        "run_baseline",
        "build_vec_env",
        "run_rb",
        "run_policy",
    ):
        if hasattr(evaluate, name):
            monkeypatch.setattr(evaluate, name, trip(f"evaluate.{name}"))


def _refusal(excinfo, capsys) -> str:
    """A refusal's text: the exception's message plus argparse's error from stderr, without the
    usage line before it (which names every flag, so would match any flag)."""
    return str(excinfo.value) + capsys.readouterr().err.split(" error: ", 1)[-1]


def _scores(df: pd.DataFrame, split: str, arm: str, key: str) -> np.ndarray:
    """runs x scenarios of key on one split for one arm, from seasons.parquet: the learning arm's
    runs by ascending seed, a baseline as one run; scenarios in the parquet's (the split's) order."""
    rows = df[(df["split"] == split) & (df["arm"] == arm)]
    if arm != ARM:
        return rows[key].to_numpy(dtype=np.float64)[None, :]
    return np.stack([rows[rows["seed"] == s][key].to_numpy(dtype=np.float64) for s in (0, 1)])


def test_final_names_match_the_contract(world):
    """Issue #43's new names: FINAL_RESULTS is control/results/final/ (the dir guard_runs seals
    until a final-result row exists), FINAL_PRIMARY is profit_eur_m2, FINAL_GUARDS the six guards
    in the issue's order, and the primary and every guard are scored result keys."""
    assert world.final_results_default == CONTROL / "results" / "final"
    assert getattr(evaluate, "FINAL_PRIMARY", None) == PRIMARY
    assert getattr(evaluate, "FINAL_GUARDS", None) == FINAL_GUARDS
    assert {PRIMARY, *FINAL_GUARDS} <= set(ledger.RESULT_KEYS)


# --- The seal: every refusal comes before any env, weather read or file -------------------------


def _spent_seal(w, final_run, monkeypatch):
    """A full --final, then ledger final-result on its per_split.json: the seal is spent."""
    shutil.copytree(final_run().dir, w.results / w.freg_id)
    numbers = w.results / w.freg_id / "per_split.json"
    argv = ["final-result", "--final-registered", w.freg_id, "--numbers", str(numbers)]
    assert ledger.main(argv, path=w.ledger) == 0
    return ["--final", w.freg_id], "seal is spent"


def _rerun_before_the_ledger_row(w, final_run, monkeypatch):
    """A full --final's files in FINAL_RESULTS, no final-result row yet."""
    shutil.copytree(final_run().dir, w.results / w.freg_id)
    return ["--final", w.freg_id], str(w.results)


def _an_interrupted_final_then_a_newer_row(w, final_run, monkeypatch):
    """An earlier --final interrupted after its first season, so FINAL_RESULTS holds only its
    stamp.json; then a newer final-registered row of the same checkpoints, whose --final passes
    every other refusal and would write FINAL_RESULTS/<newer>/ beside it: only the non-empty
    FINAL_RESULTS stops the test weather being opened a second time."""
    real, calls = evaluate.run_policy, []

    def interrupted(*args, **kwargs):
        calls.append(args)
        if len(calls) > 1:
            raise KeyboardInterrupt("interrupted after the first season")
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(evaluate, "run_policy", interrupted)
        with pytest.raises(KeyboardInterrupt):
            _main(["--final", w.freg_id], w.ledger)
    assert len(calls) == 2, "the earlier --final should play one season, then be interrupted"
    assert list(_tree(w.results)) == [f"{w.freg_id}/stamp.json"]
    newer = ledger.final_register(
        issue=43,
        checkpoints=w.checkpoints,
        baselines=list(BASELINES),
        approved_by="Aarmen",
        path=w.ledger,
    )
    return ["--final", newer["row_id"]], f"{w.results} is not empty"


def _an_older_final_registered_row(w, final_run, monkeypatch):
    """A newer final-registered row of the same checkpoints; --final names the older one."""
    newer = ledger.final_register(
        issue=43,
        checkpoints=w.checkpoints,
        baselines=list(BASELINES),
        approved_by="Aarmen",
        path=w.ledger,
    )
    return ["--final", w.freg_id], newer["row_id"]


def _selected_json_says_250000(w, final_run, monkeypatch):
    """Seed 0's selected.json says "250000"; the row registers it at "final"."""
    (w.run_dirs[0] / "selected.json").write_text(json.dumps({"checkpoint": "250000"}))
    return ["--final", w.freg_id], w.run_ids[0]


def _no_vecnorm_final(w, final_run, monkeypatch):
    """Seed 0, registered at "final", without vecnorm_final.pkl."""
    (w.run_dirs[0] / "vecnorm_final.pkl").unlink()
    return ["--final", w.freg_id], w.run_ids[0]


def _no_vecnorm_1024(w, final_run, monkeypatch):
    """Seed 1, registered at "1024", without vecnorm_1024.pkl (its vecnorm_final.pkl is there)."""
    (w.run_dirs[1] / "vecnorm_1024.pkl").unlink()
    return ["--final", w.freg_id], w.run_ids[1]


def _no_selected_json(w, final_run, monkeypatch):
    (w.run_dirs[0] / "selected.json").unlink()
    return ["--final", w.freg_id], w.run_ids[0]


def _fingerprints_unlike_the_rows(w, final_run, monkeypatch):
    changed = {**manifest.fingerprints(), "control/greenleaf_control/rewards.py": "0" * 64}
    monkeypatch.setattr(manifest, "fingerprints", lambda root=None: changed)
    return ["--final", w.freg_id], "rewards.py"


def _a_dirty_tree(w, final_run, monkeypatch):
    """code_stamp refuses a dirty tree unless smoke, as the real one does."""

    def code_stamp(*, smoke):
        if not smoke:
            raise RuntimeError("git tree is dirty; commit first or pass smoke:\n M control/x.py")
        return dict(w.stamp)

    monkeypatch.setattr(manifest, "code_stamp", code_stamp)
    return ["--final", w.freg_id], "dirty"


def _an_unknown_row(w, final_run, monkeypatch):
    return ["--final", "final-registered-deadbeef"], "final-registered-deadbeef"


def _a_registered_rows_id(w, final_run, monkeypatch):
    return ["--final", w.reg_id], w.reg_id


def _final_with(flag: str, value: str):
    def setup(w, final_run, monkeypatch):
        return ["--final", w.freg_id, flag, value], flag

    return setup


SEAL_CASES = [
    pytest.param(_spent_seal, SystemExit, id="a-second-final-after-final-result"),
    pytest.param(_rerun_before_the_ledger_row, SystemExit, id="a-rerun-before-the-ledger-row"),
    pytest.param(
        _an_interrupted_final_then_a_newer_row,
        SystemExit,
        id="an-interrupted-final-then-a-newer-row",
    ),
    pytest.param(_an_older_final_registered_row, SystemExit, id="an-older-final-registered-row"),
    pytest.param(_selected_json_says_250000, SystemExit, id="selected-json-says-250000"),
    pytest.param(_no_vecnorm_final, SystemExit, id="no-vecnorm-final-pkl"),
    pytest.param(_no_vecnorm_1024, SystemExit, id="no-vecnorm-1024-pkl"),
    pytest.param(_no_selected_json, SystemExit, id="no-selected-json"),
    pytest.param(_fingerprints_unlike_the_rows, SystemExit, id="fingerprints-unlike-the-rows"),
    pytest.param(_a_dirty_tree, RuntimeError, id="a-dirty-tree"),
    pytest.param(_an_unknown_row, SystemExit, id="an-unknown-row"),
    pytest.param(_a_registered_rows_id, SystemExit, id="a-registered-rows-id"),
    pytest.param(_final_with("--controller", "rb_default"), SystemExit, id="final-with-controller"),
    pytest.param(_final_with("--split", "paper"), SystemExit, id="final-with-split"),
    pytest.param(_final_with("--out", "x.parquet"), SystemExit, id="final-with-out"),
]


@pytest.mark.parametrize(("setup", "raised"), SEAL_CASES)
def test_final_refuses_what_could_break_the_seal_before_any_env(
    setup, raised, world, final_run, tmp_path, monkeypatch, capsys
):
    """Done when (the seal): each case is refused before any env is built (_no_env_built), so
    before any weather is read, naming its cause, with FINAL_RESULTS and the ledger as they were:
    - a second --final after a full --final and ledger final-result, naming the spent seal;
    - a rerun while FINAL_RESULTS holds the first's files and no final-result exists, naming
      FINAL_RESULTS;
    - an earlier --final interrupted after one season (FINAL_RESULTS holds only its stamp.json),
      then --final on a newer final-registered row, naming FINAL_RESULTS as not empty: the case
      refusal 4 exists for, since the newer row's own dir doesn't exist yet;
    - a final-registered row older than the ledger's newest, naming the newer;
    - seed 0's selected.json saying "250000" while the row says "final", seed 0 without
      vecnorm_final.pkl, seed 1 (registered at "1024") without vecnorm_1024.pkl, and seed 0
      without selected.json, naming the run's row id;
    - manifest.fingerprints() unlike the row's (rewards.py patched), naming the key;
    - a dirty tree (code_stamp(smoke=False) raising);
    - an unknown row id, and a registered row's id, naming it;
    - --final with --controller, --split or --out, naming the flag in the error itself.
    Every refusal but the dirty tree is parser.error's SystemExit with code 2; the dirty tree is
    code_stamp's RuntimeError. Nothing else counts: a FileExistsError from mkdir, a
    FileNotFoundError from a missing file or an AssertionError from _no_env_built is a failure.
    """
    w = _copy_world(world, tmp_path, monkeypatch)
    argv, named = setup(w, final_run, monkeypatch)
    results, ledger_bytes = _tree(w.results), w.ledger.read_bytes()
    _no_env_built(monkeypatch)
    capsys.readouterr()

    with pytest.raises(raised) as excinfo:
        _main(argv, w.ledger)

    if raised is SystemExit:
        assert excinfo.value.code == 2, "a refusal is parser.error's exit 2"
    text = _refusal(excinfo, capsys)
    assert named in text, text
    assert _tree(w.results) == results
    assert w.ledger.read_bytes() == ledger_bytes


# --- A full --final ----------------------------------------------------------------------------


def test_a_full_final_writes_three_files_and_replays_with_36_seasons_in_order(world, final_run):
    """Done when: a full --final returns 0 and writes exactly FINAL_RESULTS/<row>/ holding
    per_split.json, replays/ (#44; its files are tested below), seasons.parquet and stamp.json (the
    code stamp), nothing else in FINAL_RESULTS.

    seasons.parquet has the 9 identity columns in order (arm, run, seed, checkpoint, split,
    location, growth_year, start_day, season_length), then the result keys (RESULT_KEYS plus
    valid), and (2 runs + 4 baselines) x 3 splits x 2 scenarios = 36 rows: TEST split order, then
    scenario order (days 45, 49), then the residual runs by ascending seed at their registered
    checkpoints (seed 0 "final", seed 1 "1024"), then the baselines in the row's order with a null
    checkpoint. Every season is 1 day, 96 steps. The last line printed is the ledger command that
    records per_split.json.
    """
    final = final_run()

    assert final.code == 0
    assert _listing(world.results) == [world.freg_id]
    assert _listing(final.dir) == ["per_split.json", "replays", "seasons.parquet", "stamp.json"]
    assert json.loads((final.dir / "stamp.json").read_text()) == world.stamp
    df = pd.read_parquet(final.dir / "seasons.parquet")
    assert list(df.columns[: len(IDENTITY)]) == IDENTITY
    assert set(df.columns[len(IDENTITY) :]) == {*ledger.RESULT_KEYS, "valid"}
    records = df.to_dict("records")
    assert len(records) == 36  # (2 runs + 4 baselines) x 3 splits x 2 scenarios
    learning = [
        (ARM, run_id, seed, c) for seed, (run_id, c) in enumerate(world.checkpoints.items())
    ]
    want = [
        (split, site, 2012, day, *arm)
        for split, site in SITES.items()
        for day in (45, 49)
        for arm in [*learning, *((b, None, None, None) for b in BASELINES)]
    ]
    got = [
        (r["split"], r["location"], r["growth_year"], r["start_day"], r["arm"])
        + ((r["run"], r["seed"]) if r["arm"] == ARM else (None, None))
        + (_null(r["checkpoint"]),)
        for r in records
    ]
    assert got == want
    assert {(r["season_length"], r["steps"]) for r in records} == {(1, N_DAY)}
    numbers = final.dir / "per_split.json"
    assert final.stdout.strip().splitlines()[-1] == (
        "next: python -m greenleaf_control.ledger final-result"
        f" --final-registered {world.freg_id} --numbers {numbers}"
    )


def test_per_split_json_is_what_ledger_final_result_records(world, final_run, tmp_path):
    """Done when (the file records): per_split.json is {"per_split": ...} written with sorted keys,
    indent 2 and no NaN, and ledger.main(["final-result", "--final-registered", <row>, "--numbers",
    <it>]) on a copy of the tmp ledger returns 0 (F1a-2's shape check passes), recording exactly
    its per_split."""
    final = final_run()
    numbers = final.dir / "per_split.json"
    text = numbers.read_text()
    assert text.rstrip("\n") == json.dumps(
        json.loads(text), sort_keys=True, indent=2, allow_nan=False
    )
    path = tmp_path / "ledger.jsonl"
    shutil.copyfile(world.ledger, path)

    argv = ["final-result", "--final-registered", world.freg_id, "--numbers", str(numbers)]
    assert ledger.main(argv, path=path) == 0

    [row] = ledger.rows(kind="final-result", path=path)
    assert row["per_split"] == json.loads(text)["per_split"]


def test_per_split_is_stats_on_the_seasons(world, final_run):
    """Done when (known answers): per split and arm, from seasons.parquet's runs x 2 scenarios
    array (the residual arm 2 x 2, seeds ascending; a baseline 1 x 2):
    - iqm, ci_low, ci_high are stats.iqm_ci (2,000 reps, seed 0) on profit_eur_m2, exactly;
    - guards holds {iqm, ci_low, ci_high} for exactly FINAL_GUARDS; for a baseline (one run) each
      is the mean of its 2 seasons (see the comment below);
    - invalid_episodes counts seasons with valid False;
    - prob_improvement is stats.prob_improvement(arm, rb_tuned) on the profit, null for rb_tuned;
    - scored is the registered {run row_id: checkpoint} for the residual arm, {} for a baseline.
    """
    final = final_run()
    df = pd.read_parquet(final.dir / "seasons.parquet")
    per_split = json.loads((final.dir / "per_split.json").read_text())["per_split"]

    assert list(per_split) == sorted(TEST_SPLITS)
    for split in TEST_SPLITS:
        assert set(per_split[split]) == {ARM, *BASELINES}, split
        rb_tuned = _scores(df, split, "rb_tuned", PRIMARY)
        for arm, entry in per_split[split].items():
            where = f"{split} {arm}"
            profit = _scores(df, split, arm, PRIMARY)
            assert (entry["iqm"], entry["ci_low"], entry["ci_high"]) == stats.iqm_ci(profit), where
            assert set(entry["guards"]) == set(FINAL_GUARDS), where
            assert all(set(g) == {"iqm", "ci_low", "ci_high"} for g in entry["guards"].values())
            valid = df[(df["split"] == split) & (df["arm"] == arm)]["valid"].to_numpy(dtype=bool)
            assert entry["invalid_episodes"] == int(np.count_nonzero(~valid)), where
            assert entry["scored"] == (world.checkpoints if arm == ARM else {}), where
            if arm == "rb_tuned":
                assert entry["prob_improvement"] is None, where
            else:
                want = stats.prob_improvement(profit, rb_tuned)
                assert entry["prob_improvement"] == want, where
            if arm == ARM:
                continue
            # One run x 2 scenarios: the 25% trimmed mean of 2 scores cuts int(0.25 x 2) = 0 from
            # each end, so the IQM is their mean; every bootstrap rep redraws the one run, so the
            # CI is [mean, mean].
            for g in FINAL_GUARDS:
                m = pytest.approx(float(_scores(df, split, arm, g).mean()), rel=1e-12)
                assert entry["guards"][g] == {"iqm": m, "ci_low": m, "ci_high": m}, (where, g)


def test_a_final_season_is_run_policy_and_run_rb_on_its_scenario(world, final_run):
    """Done when (one evaluation path): the parquet's london_2012 day-49 seasons are the library's
    on that scenario: seed 0's run (registered at "final") is run_policy(run_dir, "final", s, 1,
    allow_test=True), and rb_default's is run_rb(default params, s, 1, allow_test=True), key for
    key. (Day 49 of the second split, so a season filed under the wrong split or day would differ;
    the synthetic weather is the 2010 copy, so day 45 and day 49 are different days.)"""
    final = final_run()
    df = pd.read_parquet(final.dir / "seasons.parquet")
    s = {"location": "London", "growth_year": 2012, "start_day": 49}
    rows = df[(df["split"] == "london_2012") & (df["start_day"] == 49)]
    [run_row] = rows[rows["run"] == world.run_ids[0]].to_dict("records")
    [rb_row] = rows[rows["arm"] == "rb_default"].to_dict("records")

    want_run, _ = evaluate.run_policy(world.run_dirs[0], "final", s, 1, allow_test=True)
    want_rb, _ = evaluate.run_rb(RB_DEFAULT, s, 1, allow_test=True)

    assert {k: run_row[k] for k in want_run} == want_run
    assert {k: rb_row[k] for k in want_rb} == want_rb
    assert want_run != want_rb, "the run plays exactly the RB, so the pairing above proves less"


def test_each_run_is_scored_at_its_registered_checkpoint(world, final_run):
    """Done when (each run's checkpoint is the registered one): seed 1 is registered at "1024",
    whose policy differs from its "final" (action_net zeroed). Every parquet row of seed 0 says
    checkpoint "final" and every row of seed 1 says "1024"; and on abbotsford_2012 day 45, seed
    1's season is run_policy(run_dir, "1024", s, 1, allow_test=True), key for key, and differs from
    run_policy(run_dir, "final", s, 1, allow_test=True), so a --final that scores every run at
    "final" whatever the row registers fails here."""
    final = final_run()
    df = pd.read_parquet(final.dir / "seasons.parquet")
    runs = df[df["arm"] == ARM]
    assert sorted({(int(r["seed"]), r["checkpoint"]) for r in runs.to_dict("records")}) == [
        (0, "final"),
        (1, "1024"),
    ]
    s = {"location": "Abbotsford", "growth_year": 2012, "start_day": 45}
    where = (runs["split"] == "abbotsford_2012") & (runs["start_day"] == 45)
    [row] = runs[where & (runs["run"] == world.run_ids[1])].to_dict("records")

    at_1024, _ = evaluate.run_policy(world.run_dirs[1], "1024", s, 1, allow_test=True)
    at_final, _ = evaluate.run_policy(world.run_dirs[1], "final", s, 1, allow_test=True)

    assert {k: row[k] for k in at_1024} == at_1024
    assert at_1024 != at_final, "seed 1's 1024 and final play one season, so this proves nothing"


# --- #44: the replays of the traced seasons ----------------------------------------------------


def _reject_constant(name: str):
    raise AssertionError(f"{name} in a replay: json.dumps(allow_nan=False) can't have written it")


def _replay_names() -> dict[str, tuple[str, str, str, int]]:
    """Every replay a full --final writes, by file name: (prefix, split, location, start_day)."""
    return {
        f"{prefix}_{site}_2012_{day}.json": (prefix, split, site, day)
        for prefix in TRACED
        for split, site in SITES.items()
        for day in DAYS
    }


def test_a_full_final_writes_one_replay_per_traced_season(world, final_run):
    """Done when (which replays, #44): FINAL_RESULTS/<row>/replays/ holds exactly the 30 files
    <prefix>_<location>_2012_<start_day>.json, prefix residual or a baseline, on the 3 test sites x
    days 45 and 49: (1 traced run + 4 baselines) x 3 splits x 2 scenarios. The residual arm's
    traced run is its lowest registered seed, seed 0 (at "final"), not seed 1 (the next test tells
    their seasons apart by totals). Each file:
    - is under 250,000 bytes, and is json.dumps(payload, allow_nan=False, separators=(",", ":"))
      byte for byte, as export_replay writes it: compact, no NaN or infinity, no trailing newline;
    - is export's {meta, hourly, totals}, with 24 hourly points (a 1-day season);
    - has export's 10 meta keys: controller is the file's prefix; scenario the season's, with
      season_length 1; split its test split; run, for the residual arm, export's provenance of
      seed 0's run (its run row id, registration, result row, arm, seed 0, checkpoint "final" and
      its manifest's commit) and None for a baseline; commit and dirty the --final's code stamp
      and gl_gym manifest's gl-gym commit; note "registered final <row>: residual seed 0 at
      checkpoint final" or "registered final <row>: <baseline>", naming the row; bounds and units
      export's.
    """
    final = final_run()
    replays = final.dir / "replays"
    want = _replay_names()
    assert len(want) == 30  # (1 traced run + 4 baselines) x 3 splits x 2 scenarios

    assert _listing(replays) == sorted(want)
    [result_row] = ledger.rows(kind="result", path=world.ledger)
    seed0 = {
        "id": world.run_ids[0],
        "registered": world.reg_id,
        "result": result_row["row_id"],
        "arm": ARM,
        "seed": 0,
        "checkpoint": "final",
        "commit": json.loads((world.run_dirs[0] / "manifest.json").read_text())["commit"],
    }
    gl_gym = manifest._versions()["gl-gym"]
    for name, (prefix, split, site, day) in want.items():
        path = replays / name
        size = path.stat().st_size
        assert size < MAX_BYTES, f"{name}: {size:,} bytes"
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text, parse_constant=_reject_constant)
        assert text == json.dumps(payload, allow_nan=False, separators=(",", ":")), name
        assert set(payload) == {"meta", "hourly", "totals"}, name
        assert payload["hourly"]["t_hour"] == list(range(24)), name
        meta = payload["meta"]
        assert set(meta) == REPLAY_META, name
        named = "residual seed 0 at checkpoint final" if prefix == ARM else prefix
        assert meta == {
            "controller": prefix,
            "scenario": {
                "location": site,
                "growth_year": 2012,
                "start_day": day,
                "season_length": 1,
            },
            "split": split,
            "run": seed0 if prefix == ARM else None,
            "commit": world.stamp["commit"],
            "dirty": world.stamp["dirty"],
            "gl_gym": gl_gym,
            "note": f"registered final {world.freg_id}: {named}",
            "bounds": {c: list(bounds) for c, bounds in metrics.BOUNDS.items()},
            "units": export_replay.UNITS,
        }, name


def test_each_replays_totals_are_its_seasons_row_in_seasons_parquet(world, final_run):
    """Done when (same season, #44): each replay's totals is its season's seasons.parquet row, key
    for key and exactly: the parquet's result columns (RESULT_KEYS plus valid), no more and no
    fewer; the residual replay's row is seed 0's on its split and day, a baseline's its own.
    Non-vacuous: on every scenario the residual replay's totals differ from seed 1's row (seed 1
    plays its "1024", which is the RB exactly), so a replay of seed 1 fails here, and so does one of
    the other day's season (days 45 and 49 differ). The three sites' 2012.csv are one copy, so their
    seasons are equal: the first replay test ties each file to its site through meta.scenario."""
    final = final_run()
    df = pd.read_parquet(final.dir / "seasons.parquet")
    keys = set(df.columns[len(IDENTITY) :])
    assert keys == {*ledger.RESULT_KEYS, "valid"}

    for name, (prefix, split, _, day) in _replay_names().items():
        path = final.dir / "replays" / name
        assert path.is_file(), f"no replay {name}"
        totals = json.loads(path.read_text(encoding="utf-8"))["totals"]
        rows = df[(df["split"] == split) & (df["start_day"] == day) & (df["arm"] == prefix)]
        if prefix == ARM:
            [seed1] = rows[rows["run"] == world.run_ids[1]].to_dict("records")
            rows = rows[rows["run"] == world.run_ids[0]]
        [row] = rows.to_dict("records")
        assert set(totals) == keys, name
        assert {k: row[k] for k in keys} == totals, name
        if prefix == ARM:
            assert {k: seed1[k] for k in keys} != totals, f"{name}: seed 1's season is seed 0's"


def test_each_traced_season_is_its_scored_season_played_once(world, final_run):
    """Done when (no second season, #44): the replays come from the seasons --final scores, so the
    full --final plays each scored season once (final_run's pass-through spies):
    - seasons.parquet still has 36 rows;
    - evaluate.run_policy is called 12 times, once per run x scenario (seed 0's run dir at
      "final", seed 1's at "1024"); evaluate.run_rb 12 times (rb_default's and rb_tuned's params,
      from load_rb_params, once per scenario); envs.run_baseline 12 times (zero_controls and random
      once per scenario); every call a 1-day season with allow_test=True;
    - the traced calls ask for the trace: seed 0's 6 run_policy calls (its lowest seed) and all 12
      run_rb calls (rb_default and rb_tuned are baselines, all traced) have record_steps=True and
      return_trace=True;
    - manifest.code_stamp is called exactly once, with smoke=False: the replays reuse the stamp
      taken at refusal 6.
    """
    final = final_run()
    df = pd.read_parquet(final.dir / "seasons.parquet")
    calls = final.calls
    scored = [(site, day) for site in SITES.values() for day in DAYS]

    assert len(df) == 36  # (2 runs + 4 baselines) x 3 splits x 2 scenarios
    played = sorted(
        (str(c["run_dir"]), c["checkpoint"], c["scenario"]["location"], c["scenario"]["start_day"])
        for c in calls["run_policy"]
    )
    assert played == sorted(
        (str(d), cp, site, day)
        for d, cp in zip(world.run_dirs, CHECKPOINTS, strict=True)
        for site, day in scored
    )
    rb_params = {"rb_default": RB_DEFAULT, "rb_tuned": rule_based.load_rb_params("tuned")}
    assert rb_params["rb_default"] != rb_params["rb_tuned"]
    rb = sorted(  # a call with neither's params is dropped, so the count below fails
        (b, c["scenario"]["location"], c["scenario"]["start_day"])
        for c in calls["run_rb"]
        for b, params in rb_params.items()
        if c["params"] == params
    )
    assert rb == sorted((b, *s) for b in ("rb_default", "rb_tuned") for s in scored)
    baselines = sorted(
        (c["name"], c["scenario"]["location"], c["scenario"]["start_day"])
        for c in calls["run_baseline"]
    )
    assert baselines == sorted((b, *s) for b in ("zero_controls", "random") for s in scored)
    every = [*calls["run_policy"], *calls["run_rb"], *calls["run_baseline"]]
    assert {(c["season_length"], c["allow_test"]) for c in every} == {(1, True)}
    traced = [c for c in calls["run_policy"] if Path(c["run_dir"]) == world.run_dirs[0]]
    traced += calls["run_rb"]
    flags = [(c["record_steps"], c.get("return_trace")) for c in traced]
    assert flags == [(True, True)] * 18  # 6 seed 0 seasons + 12 rb_default and rb_tuned seasons
    assert calls["code_stamp"] == [{"smoke": False}]


# --- #44's fix round: a failed or short replay costs no scored file ----------------------------

# The short season of the test below: seed 0's (the traced run's) london_2012 day 49, mid-loop, so
# replays come both before it and after it.
SHORT_SPLIT = "london_2012"
SHORT = {"location": "London", "growth_year": 2012, "start_day": 49}


def _outcome(argv: list[str], ledger_path: Path) -> int | RuntimeError | ValueError:
    """--final's return code, or the RuntimeError or ValueError it raised. A ValueError is what
    export_replay.export's hourly reshape raises on a frame that isn't whole hours."""
    try:
        return _main(argv, ledger_path)
    except (RuntimeError, ValueError) as error:
        return error


def _same_scored_files(out: Path, shared: Path) -> None:
    """out's stamp.json and per_split.json are the shared full run's byte for byte, and its
    seasons.parquet (36 rows) is the shared run's frame."""
    for name in ("stamp.json", "per_split.json", "seasons.parquet"):
        assert (out / name).is_file(), f"no {name}: the scored files come before any replay"
    for name in ("stamp.json", "per_split.json"):
        assert (out / name).read_bytes() == (shared / name).read_bytes(), name
    df = pd.read_parquet(out / "seasons.parquet")
    assert len(df) == 36  # (2 runs + 4 baselines) x 3 splits x 2 scenarios
    pd.testing.assert_frame_equal(df, pd.read_parquet(shared / "seasons.parquet"))


def test_a_failing_replay_export_leaves_seasons_and_per_split_complete(
    world, final_run, tmp_path, monkeypatch
):
    """Done when (#44 fix round): seasons.parquet and per_split.json are always written before any
    replay work, so a replay that fails can't cost the scored numbers. With export_replay.export
    patched to raise RuntimeError("boom"), --final on a copied world raises it, and
    FINAL_RESULTS/<row>/ still holds stamp.json and per_split.json byte for byte the shared full
    run's and seasons.parquet equal to its (36 rows), so `ledger final-result` can record them."""
    final = final_run()  # first: it plays the real run_policy and export
    w = _copy_world(world, tmp_path, monkeypatch)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(export_replay, "export", boom)

    with pytest.raises(RuntimeError, match="boom"):
        _main(["--final", w.freg_id], w.ledger)

    _same_scored_files(w.results / w.freg_id, final.dir)


def test_a_failing_replay_stamp_fails_before_final_results_is_made(world, tmp_path, monkeypatch):
    """Done when (#44 fix round, the reviewer's note): the replays' stamp (gl_gym, from
    manifest._versions) and each traced run's meta.run are built before FINAL_RESULTS/<row>/ is
    made and before any env, so a failure there costs neither the seal nor a season. With
    manifest._versions patched to raise, --final on a copied world raises that error, no env is
    built (_no_env_built), and FINAL_RESULTS doesn't exist afterwards. (The fixture's code_stamp
    and manifest.fingerprints don't call _versions, so the replay stamp is the call that raises.)"""
    w = _copy_world(world, tmp_path, monkeypatch)

    def no_versions():
        raise RuntimeError("no gl-gym install record")

    monkeypatch.setattr(manifest, "_versions", no_versions)
    _no_env_built(monkeypatch)

    with pytest.raises(RuntimeError, match="no gl-gym install record"):
        _main(["--final", w.freg_id], w.ledger)

    assert not w.results.exists(), _tree(w.results)


@pytest.mark.parametrize("steps", [94, 92])
def test_a_short_traced_season_is_named_not_replayed_and_the_rest_are(
    steps, world, final_run, tmp_path, monkeypatch
):
    """Done when (#44 fix round): a traced season whose frame isn't SEASON_LENGTH x 96 steps long
    (a solver failure ends a season early) is not exported, every full-length one still is, and
    after the replay loop --final raises one RuntimeError naming the short season.

    evaluate.run_policy is wrapped so seed 0's (the traced run's) london_2012 day-49 season returns
    its frame and trace cut to `steps` steps (frame.iloc[:steps], states[:steps + 1],
    controls[:steps]) with its real result, so the scoring is the full run's. 94 steps isn't whole
    hours (export's hourly reshape raises ValueError today); 92 is 23 whole hours (export writes a
    0-day replay today, and --final returns 0). Then:
    - stamp.json, per_split.json and seasons.parquet are the shared full run's;
    - replays/ holds exactly the other 29 files, each byte for byte the shared run's, and no
      residual_London_2012_49.json;
    - --final raises RuntimeError (it doesn't return 0 or raise export's ValueError) whose message
      names the season: "residual", "london_2012", start day 49 and the step count, each found
      with the tmp paths and row ids taken out (so no number comes from a path); names
      seasons.parquet and per_split.json; and gives `final-result --final-registered <row>
      --numbers <per_split.json>`.
    """
    final = final_run()  # first: its spies must wrap the real run_policy, not the cut below
    w = _copy_world(world, tmp_path, monkeypatch)
    real, cuts = evaluate.run_policy, []

    def cut(run_dir, checkpoint, scenario, *args, **kwargs):
        played = real(run_dir, checkpoint, scenario, *args, **kwargs)
        if Path(run_dir) != w.run_dirs[0] or scenario != SHORT:
            return played
        result, frame, (states, controls, params) = played
        cuts.append(len(frame))
        return result, frame.iloc[:steps], (states[: steps + 1], controls[:steps], params)

    monkeypatch.setattr(evaluate, "run_policy", cut)

    outcome = _outcome(["--final", w.freg_id], w.ledger)

    assert cuts == [N_DAY], "seed 0's london_2012 day-49 season should be played, and cut, once"
    out = w.results / w.freg_id
    _same_scored_files(out, final.dir)
    short = f"{ARM}_{SHORT['location']}_2012_{SHORT['start_day']}.json"
    want = {name: data for name, data in _tree(final.dir / "replays").items() if name != short}
    assert len(want) == 29  # 30 traced seasons, less the short one
    got = _tree(out / "replays") or {}
    assert sorted(got) == sorted(want), f"the 29 full seasons' replays, and no {short}"
    for name, data in want.items():
        assert got[name] == data, name
    assert type(outcome) is RuntimeError, (
        f"--final should raise one RuntimeError naming the short season, not {outcome!r}"
    )
    text = str(outcome)
    numbers = out / "per_split.json"
    assert f"final-result --final-registered {w.freg_id} --numbers {numbers}" in text, text
    rest = re.sub(r"-[0-9a-f]{8}\b", "-<id>", text.replace(str(tmp_path), "<tmp>"))
    for named in (ARM, SHORT_SPLIT, "seasons.parquet", "per_split.json"):
        assert named in rest, (named, text)
    for number in (SHORT["start_day"], steps):
        assert re.search(rf"(?<![\d.]){number}(?!\d)", rest), (number, text)


def test_the_traced_run_is_the_lowest_seed_not_the_first_registered(
    world, final_run, tmp_path, monkeypatch
):
    """Done when (#44, the reviewer's note): the residual arm's traced run is its lowest registered
    seed, whatever the ledger's order. A fresh tmp ledger chain over the copied run dirs puts seed 1
    first everywhere, as stored: its run row before seed 0's, and first in the stored result's
    selected and the stored final-registered row's checkpoints, {seed 1: "1024", seed 0: "final"}.
    The ledger writes each row with sort_keys=True, so a stored {run row_id: checkpoint} is in row
    id order, and row ids are random; here ledger's secrets.token_hex is patched to count up
    (00000000, 00000001, ...), so seed 1's run row, created first, has the smaller id and sorts
    first every time. --final returns 0 and each of the 6 residual replays is seed 0's: meta.run is
    seed 0's run row at checkpoint "final", the note names seed 0, and its totals are the shared
    run's seed-0 replay's on that scenario (seed 1 at "1024" plays the RB exactly, so its totals
    differ; the totals test above shows it). A --final that traced the first registered run would
    trace seed 1 here."""
    final = final_run()
    w = _copy_world(world, tmp_path, monkeypatch)
    ids = iter(range(1 << 32))
    monkeypatch.setattr(ledger.secrets, "token_hex", lambda nbytes: f"{next(ids):0{2 * nbytes}x}")
    path = tmp_path / "seed1_first.jsonl"
    reg = ledger.register(
        issue=44,
        hypothesis="the same two residual smoke runs, seed 1 recorded first",
        arms=[{"name": ARM, "method": "residual", "config": {}}],
        seeds=[0, 1],
        timesteps=2048,
        scenarios=["validation"],
        primary=PRIMARY,
        guards=["viol_hours_temp"],
        rule="a test fixture",
        budget={"cpu_hours": 0.0, "usd": 0.0, "backend": "local"},
        path=path,
    )["row_id"]
    run_ids = {}
    for seed in (1, 0):
        run = ledger.run_start(
            registered=reg, arm=ARM, seed=seed, run_dir=str(w.run_dirs[seed]), path=path
        )
        ledger.run_end(
            run=run["row_id"],
            status="finished",
            wall_s=6.0,
            timesteps=2048,
            solver_failures=0,
            cost_usd=None,
            diagnostics=DIAGNOSTICS,
            path=path,
        )
        run_ids[seed] = run["row_id"]
    checkpoints = {run_ids[1]: "1024", run_ids[0]: "final"}
    numbers = {"iqm": 1.0, "ci_low": 0.0, "ci_high": 2.0, "guards": {}, "invalid_episodes": 0}
    ledger.record_result(registered=reg, per_arm={ARM: numbers}, selected=checkpoints, path=path)
    freg = ledger.final_register(
        issue=44,
        checkpoints=checkpoints,
        baselines=list(BASELINES),
        approved_by="Aarmen",
        path=path,
    )
    assert [r["seed"] for r in ledger.rows(kind="run", path=path)] == [1, 0]
    [result] = ledger.rows(kind="result", path=path)
    [stored] = ledger.rows(kind="final-registered", path=path)
    assert list(result["selected"]) == [run_ids[1], run_ids[0]]
    assert list(stored["checkpoints"]) == [run_ids[1], run_ids[0]]

    assert _main(["--final", freg["row_id"]], path) == 0

    replays = w.results / freg["row_id"] / "replays"
    for site, day in [(site, day) for site in SITES.values() for day in DAYS]:
        name = f"{ARM}_{site}_2012_{day}.json"
        assert (replays / name).is_file(), f"no replay {name}"
        payload = json.loads((replays / name).read_text(encoding="utf-8"))
        run = payload["meta"]["run"]
        assert (run["id"], run["seed"], run["checkpoint"]) == (run_ids[0], 0, "final"), name
        note = f"registered final {freg['row_id']}: residual seed 0 at checkpoint final"
        assert payload["meta"]["note"] == note, name
        shared = json.loads((final.dir / "replays" / name).read_text(encoding="utf-8"))
        assert payload["totals"] == shared["totals"], name


def test_the_only_allow_test_true_in_evaluate_is_inside_final():
    """Done when (source check): every allow_test=True in evaluate.py, as a call's keyword or a
    parameter's default, lies inside _final, the --final path, and there is at least one.
    Non-vacuous: the walk finds both forms in a snippet."""

    def allow_test_true(node) -> list[ast.AST]:
        """Every allow_test=True under node: a call keyword's value or a parameter's default."""
        found = []
        for n in ast.walk(node):
            if isinstance(n, ast.keyword) and n.arg == "allow_test":
                found.append(n.value)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                a = n.args
                pos = a.posonlyargs + a.args
                pairs = [*zip(pos[len(pos) - len(a.defaults) :], a.defaults, strict=True)]
                pairs += [*zip(a.kwonlyargs, a.kw_defaults, strict=True)]
                found += [d for p, d in pairs if p.arg == "allow_test"]
        return [v for v in found if isinstance(v, ast.Constant) and v.value is True]

    snippet = ast.parse("f(allow_test=True)\ndef g(*, allow_test=True): pass")
    assert len(allow_test_true(snippet)) == 2
    tree = ast.parse(Path(evaluate.__file__).read_text())
    finals = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_final"]
    assert len(finals) == 1, "evaluate.py has no top-level _final, the --final path"
    inside = {id(n) for n in allow_test_true(finals[0])}
    outside = [n.lineno for n in allow_test_true(tree) if id(n) not in inside]
    assert outside == [], f"allow_test=True outside _final at evaluate.py lines {outside}"
    assert inside, "_final passes no allow_test=True, so it can't score the test splits"


# --- #71: the 2013 seal, finalized once beside the spent 2012 seal ------------------------------

SITES_2013 = {
    "amsterdam_2013": "Amsterdam",
    "london_2013": "London",
    "abbotsford_2013": "Abbotsford",
}
REAL_2010 = CONTROL / "data" / "weather" / "Amsterdam" / "2010.csv"  # not test weather


def _world_2013(
    world, final_run, tmp_path, monkeypatch, *, seeds=(0,), baselines=("rb_default", "rb_tuned")
) -> SimpleNamespace:
    """A copied world (_copy_world) where the 2012 seal is spent (_spent_seal: a full --final's
    files in FINAL_RESULTS/<2012 row>/, then ledger final-result on them), with:
    - scenarios.WEATHER_DIR at tmp "synthetic_2013" (.synthetic13), where Amsterdam, London and
      Abbotsford each have a 2013.csv that is a byte copy of the real Amsterdam/2010.csv;
    - scenarios.SEALS[2013] (and TEST_2013) cut to start days 45 and 49 per 2013 split;
    - a seal-2013 final-registered row (.freg13), issue 71, of seeds' runs at their CHECKPOINTS
      (.checkpoints13; by default seed 0 at "final") with baselines (by default rb_default and
      rb_tuned).
    Called from a test's body, so a missing seal= fails that test, not a fixture."""
    w = _copy_world(world, tmp_path, monkeypatch)
    _spent_seal(w, final_run, monkeypatch)
    w.synthetic13 = tmp_path / "synthetic_2013"
    for site in SITES_2013.values():
        (w.synthetic13 / site).mkdir(parents=True)
        shutil.copyfile(REAL_2010, w.synthetic13 / site / "2013.csv")
    monkeypatch.setattr(scenarios, "WEATHER_DIR", w.synthetic13)
    test_2013 = {
        split: [{"location": site, "growth_year": 2013, "start_day": d} for d in DAYS]
        for split, site in SITES_2013.items()
    }
    monkeypatch.setattr(scenarios, "TEST_2013", test_2013, raising=False)
    monkeypatch.setitem(scenarios.SEALS, 2013, test_2013)
    w.checkpoints13 = {w.run_ids[seed]: CHECKPOINTS[seed] for seed in seeds}
    w.freg13 = ledger.final_register(
        issue=71,
        checkpoints=w.checkpoints13,
        baselines=list(baselines),
        approved_by="Aarmen",
        seal=2013,
        path=w.ledger,
    )["row_id"]
    return w


def _the_spent_2012_row(w):
    """--final on the 2012 row, a 2013 row now newer: 2012's seal is spent, and that is the
    refusal (not the newer row)."""
    return ["--final", w.freg_id], "seal is spent", SystemExit


def _a_stray_dir_in_final_results(w):
    """FINAL_RESULTS holds the spent 2012 row's dir and one that no final-registered row names."""
    stray = w.results / "final-registered-0badc0de"
    stray.mkdir()
    (stray / "stamp.json").write_text("{}\n")
    return ["--final", w.freg13], f"{w.results} is not empty", SystemExit


def _an_interrupted_2013_final(w):
    """FINAL_RESULTS holds the 2013 row's own dir with only a stamp.json, as an interrupted 2013
    --final leaves it: the 2013 seal isn't spent, so FINAL_RESULTS may not hold its dir."""
    (w.results / w.freg13).mkdir()
    (w.results / w.freg13 / "stamp.json").write_text("{}\n")
    return ["--final", w.freg13], f"{w.results} is not empty", SystemExit


def _a_missing_2013_weather_file(w):
    """london_2013's weather file is gone; scenarios.require_weather names it."""
    missing = w.synthetic13 / "London" / "2013.csv"
    missing.unlink()
    return ["--final", w.freg13], str(missing), (SystemExit, FileNotFoundError)


SEAL_2013_CASES = [
    pytest.param(_the_spent_2012_row, id="the-spent-2012-row"),
    pytest.param(_a_stray_dir_in_final_results, id="a-stray-dir-in-final-results"),
    pytest.param(_an_interrupted_2013_final, id="an-interrupted-2013-final"),
    pytest.param(_a_missing_2013_weather_file, id="a-missing-2013-weather-file"),
]


@pytest.mark.parametrize("setup", SEAL_2013_CASES)
def test_the_2013_world_refuses_what_could_break_a_seal_before_any_env(
    setup, world, final_run, tmp_path, monkeypatch, capsys
):
    """Done when (#71): with 2012 spent and a seal-2013 row registered, each case is refused before
    any env is built (_no_env_built), naming its cause, with FINAL_RESULTS and the ledger as they
    were:
    - --final on the 2012 row: "seal is spent" (2012 stays spent);
    - a dir in FINAL_RESULTS that no spent seal's row names, or the unspent 2013 row's own (an
      interrupted 2013 --final): "<FINAL_RESULTS> is not empty". FINAL_RESULTS may hold only spent
      seals' dirs, so the 2012 row's dir alone doesn't stop the 2013 final (the next test);
    - a missing 2013 weather file: the refusal names its path, before anything is written.
    A refusal is parser.error's SystemExit with code 2 (the missing file may also be
    require_weather's FileNotFoundError)."""
    w = _world_2013(world, final_run, tmp_path, monkeypatch)
    argv, named, raised = setup(w)
    results, ledger_bytes = _tree(w.results), w.ledger.read_bytes()
    _no_env_built(monkeypatch)
    capsys.readouterr()

    with pytest.raises(raised) as excinfo:
        _main(argv, w.ledger)

    if isinstance(excinfo.value, SystemExit):
        assert excinfo.value.code == 2, "a refusal is parser.error's exit 2"
    text = _refusal(excinfo, capsys)
    assert named in text, text
    assert _tree(w.results) == results
    assert w.ledger.read_bytes() == ledger_bytes


def test_a_2013_final_plays_only_its_seal_and_records_once(
    world, final_run, tmp_path, monkeypatch, capsys
):
    """Done when (#71): with 2012 spent and its files in FINAL_RESULTS, --final <the 2013 row>
    returns 0 and, through pass-through spies on evaluate.run_policy, evaluate.run_rb and
    envs.run_baseline:
    - plays exactly the 2013 seal's 6 scenarios (3 sites x days 45 and 49), each 3 times: seed 0's
      run dir at "final" (run_policy), rb_default and rb_tuned (run_rb); no run_baseline (no
      zero_controls or random registered); every call a 1-day season with allow_test=True; no 2012
      scenario;
    - reads only the three synthetic 2013 files and no 2012 file;
    - writes FINAL_RESULTS/<2013 row>/ beside the 2012 row's dir, which is byte for byte as it was;
    - seasons.parquet has (1 run + 2 baselines) x 3 splits x 2 scenarios x 2 price regimes = 36
      rows (#72: each season graded at sim and at agc2020), split by split in the seal's order, all
      growth year 2013;
    - per_split.json is keyed by exactly the 2013 splits, and `ledger final-result` records it
      (the ledger checks its shape, nested by price regime since #72), spending 2013.
    Then a second --final <the 2013 row> is refused before any env, "seal is spent", with
    FINAL_RESULTS and the ledger as they were."""
    w = _world_2013(world, final_run, tmp_path, monkeypatch)
    spent_2012 = _tree(w.results / w.freg_id)
    n_2012, n_2013 = len(world.reads), len(world.reads_2013)
    calls = {}
    for module, name in [(evaluate, "run_policy"), (evaluate, "run_rb"), (envs, "run_baseline")]:
        calls[name] = []
        monkeypatch.setattr(module, name, _spy(getattr(module, name), calls[name]))

    assert _main(["--final", w.freg13], w.ledger) == 0

    seal = [(site, 2013, day) for site in SITES_2013.values() for day in DAYS]
    every = [*calls["run_policy"], *calls["run_rb"], *calls["run_baseline"]]
    played = sorted(
        (c["scenario"]["location"], c["scenario"]["growth_year"], c["scenario"]["start_day"])
        for c in every
    )
    assert played == sorted(seal * 3)  # seed 0, rb_default and rb_tuned on each
    assert calls["run_baseline"] == []
    assert {(Path(c["run_dir"]), c["checkpoint"]) for c in calls["run_policy"]} == {
        (w.run_dirs[0], "final")
    }
    assert {(c["season_length"], c["allow_test"]) for c in every} == {(1, True)}
    assert world.reads[n_2012:] == [], "the 2013 final read 2012 weather"
    assert set(world.reads_2013[n_2013:]) == {
        (w.synthetic13 / site / "2013.csv").resolve() for site in SITES_2013.values()
    }
    assert _listing(w.results) == sorted([w.freg_id, w.freg13])
    assert _tree(w.results / w.freg_id) == spent_2012
    out = w.results / w.freg13
    df = pd.read_parquet(out / "seasons.parquet")
    assert len(df) == 36  # (1 run + 2 baselines) x 3 splits x 2 scenarios x 2 price regimes
    assert list(dict.fromkeys(df["split"])) == list(SITES_2013)
    assert set(df["growth_year"]) == {2013}
    numbers = out / "per_split.json"
    assert set(json.loads(numbers.read_text())["per_split"]) == set(SITES_2013)
    argv = ["final-result", "--final-registered", w.freg13, "--numbers", str(numbers)]
    assert ledger.main(argv, path=w.ledger) == 0
    assert ledger.seal_spent(ledger.rows(path=w.ledger), 2013) is True

    results, ledger_bytes = _tree(w.results), w.ledger.read_bytes()
    _no_env_built(monkeypatch)
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        _main(["--final", w.freg13], w.ledger)
    assert excinfo.value.code == 2
    text = _refusal(excinfo, capsys)
    assert "seal is spent" in text, text
    assert _tree(w.results) == results
    assert w.ledger.read_bytes() == ledger_bytes


# --- #72: the 2013 final plays each season once and grades it at both price regimes -------------

# PROTOCOL v3's price regimes, written out (as test_evaluate does) so the regrading checks don't read
# metrics' own table: EUR/kg fresh fruit, dry matter per fresh matter, EUR/kWh heat, EUR/kWh
# electricity, EUR/kg CO2.
REGIMES = ("sim", "agc2020")
PRICES = {
    "sim": {"fruit": 1.6, "dmfm": 0.065, "heat": 0.09, "elec": 0.3, "co2": 0.3},
    "agc2020": {"fruit": 1.6, "dmfm": 0.065, "heat": 0.03, "elec": 0.06, "co2": 0.08},
}
# The result keys prices move; every other column of a season's two rows is equal.
PRICED = ("profit_eur_m2", "heat_cost", "elec_cost", "co2_cost", "paper_return")
UNPRICED = [*IDENTITY, *(k for k in (*ledger.RESULT_KEYS, "valid") if k not in PRICED)]
# The dual world's runs play at: seed 0 (at "final", the traced run) with its copied manifest's
# price_regime set to agc2020; seed 1 (at "1024") left at sim, as trained.
PLAYED_AT = {0: "agc2020", 1: "sim"}


def _set_price_regime(run_dir: Path, regime: str) -> None:
    """run_dir's manifest config with price_regime set to regime (as test_evaluate does), so
    run_policy plays and grades its seasons at regime; the policy never sees a price."""
    path = run_dir / "manifest.json"
    m = json.loads(path.read_text())
    m["config"]["price_regime"] = regime
    path.write_text(json.dumps(m, indent=2) + "\n")


@pytest.fixture(scope="module")
def final_run_2013(world, final_run, tmp_path_factory):
    """A function that, on its first call, builds the dual world and runs one full --final <its
    2013 row> in it, returning {code, stdout, dir, calls, w}:
    - w is _world_2013 in a tmp dir of its own, registering both seeds (seed 0 at "final", seed 1
      at "1024") and the four BASELINES in order, with seed 0's copied manifest at
      price_regime agc2020 (PLAYED_AT);
    - calls holds pass-through spies' calls of evaluate.run_policy, evaluate.run_rb and
      envs.run_baseline, as final_run's; stdout is what --final printed; dir its FINAL_RESULTS dir.
    Its patches (WEATHER_DIR, SEALS[2013], FINAL_RESULTS, the spies) end with the call; a test that
    plays a 2013 season again points WEATHER_DIR at w.synthetic13 itself. (2 runs + 4 baselines) x
    6 scenarios = 36 one-day seasons, about 25 s. A failed attempt is kept and raised again in each
    test that needs it, so each fails in its own body without replaying."""
    done = {}

    def run() -> SimpleNamespace:
        if "error" in done:
            raise done["error"]
        if not done:
            out, calls = io.StringIO(), {}
            root = tmp_path_factory.mktemp("final_2013")
            with pytest.MonkeyPatch.context() as mp:
                try:
                    w = _world_2013(world, final_run, root, mp, seeds=(0, 1), baselines=BASELINES)
                    _set_price_regime(w.run_dirs[0], PLAYED_AT[0])
                    for module, name in [
                        (evaluate, "run_policy"),
                        (evaluate, "run_rb"),
                        (envs, "run_baseline"),
                    ]:
                        calls[name] = []
                        mp.setattr(module, name, _spy(getattr(module, name), calls[name]))
                    with contextlib.redirect_stdout(out):
                        code = _main(["--final", w.freg13], w.ledger)
                except Exception as error:
                    done["error"] = error
                    raise
            done.update(
                code=code, stdout=out.getvalue(), dir=w.results / w.freg13, calls=calls, w=w
            )
        return SimpleNamespace(**done)

    return run


def _graded(trace: tuple, regime: str) -> dict:
    """metrics.episode_metrics on a 1-day season's (states, controls, params) at regime's PRICES."""
    states, controls, params = trace
    return metrics.episode_metrics(
        states, controls, params, season_steps=N_DAY, prices=PRICES[regime]
    )


def _baseline_trace(name: str, s: dict) -> tuple:
    """A zero_controls or random 1-day season on s (allow_test=True) as (states, controls,
    params): envs.run_baseline's infos and the raw env's params, as test_evaluate builds it."""
    infos = envs.run_baseline(name, s, 1, allow_test=True)
    env = envs.make_raw_env(scenario=s, season_length=1, normalize_actions=False, allow_test=True)
    params = np.asarray(env.unwrapped.p, dtype=np.float64)
    env.close()
    return (*metrics.trace_from_infos(infos), params)


def _two_rows(df: pd.DataFrame, split: str, day: int, arm: str, run=None) -> dict[str, dict]:
    """One 2013 season's seasons.parquet rows (of run, for the learning arm) by price_regime,
    asserting there are exactly two, one per regime."""
    rows = df[(df["split"] == split) & (df["start_day"] == day) & (df["arm"] == arm)]
    if run is not None:
        rows = rows[rows["run"] == run]
    records = rows.to_dict("records")
    regimes = sorted(str(r.get("price_regime")) for r in records)
    assert regimes == sorted(REGIMES), f"{arm} {run} {split} day {day}: rows at {regimes}"
    return {r["price_regime"]: r for r in records}


def _scores_2013(df: pd.DataFrame, split: str, arm: str, key: str, regime: str) -> np.ndarray:
    """_scores on regime's rows alone: runs x scenarios of key (seeds ascending; a baseline one
    run)."""
    return _scores(df[df["price_regime"] == regime], split, arm, key)


def test_a_2013_final_plays_each_season_once_and_grades_it_at_both_regimes(final_run_2013):
    """Done when (#72, played once, two rows per season): --final <the dual world's 2013 row>
    returns 0, and
    - its spies saw each (run or baseline, scenario) played exactly once, on the 2013 seal's 6
      scenarios only (3 sites x days 45 and 49), each a 1-day season with allow_test=True:
      run_policy for seed 0 at "final" and seed 1 at "1024", run_rb for rb_default's and
      rb_tuned's params, run_baseline for zero_controls and random, 12 calls each; every run_rb
      and run_baseline call at price_regime sim (a baseline plays at sim);
    - seasons.parquet has the 9 identity columns, then price_regime, then the result keys
      (RESULT_KEYS plus valid), and exactly two rows per season, one at sim and one at agc2020:
      (2 runs + 4 baselines) x 3 splits x 2 scenarios x 2 regimes = 72 rows;
    - a season's two rows are equal in every column prices don't move (identity, yield, violation
      hours, in-bounds %, solver_failures, steps, valid, and paper_reward_sum, the play's own);
    - its agc2020 row's costs are its sim row's at agc2020's prices, and its profit follows (the
      arithmetic below), within 1e-9 EUR/m2; its sim row's profit is yield x 1.6 less its costs.
    Non-vacuous: some season heats (sim heat_cost > 0), and some season's agc2020 profit differs
    from its sim profit by more than 1e-6 EUR/m2."""
    final = final_run_2013()
    w, calls = final.w, final.calls

    assert final.code == 0
    seal = [(site, day) for site in SITES_2013.values() for day in DAYS]
    every = [*calls["run_policy"], *calls["run_rb"], *calls["run_baseline"]]
    assert {c["scenario"]["growth_year"] for c in every} == {2013}
    assert {(c["season_length"], c["allow_test"]) for c in every} == {(1, True)}
    policy = sorted(
        (str(c["run_dir"]), c["checkpoint"], c["scenario"]["location"], c["scenario"]["start_day"])
        for c in calls["run_policy"]
    )
    assert policy == sorted(
        (str(w.run_dirs[seed]), CHECKPOINTS[seed], *s) for seed in (0, 1) for s in seal
    )
    rb_params = {"rb_default": RB_DEFAULT, "rb_tuned": rule_based.load_rb_params("tuned")}
    rb = sorted(  # a call with neither's params is dropped, so the comparison below fails
        (b, c["scenario"]["location"], c["scenario"]["start_day"])
        for c in calls["run_rb"]
        for b, params in rb_params.items()
        if c["params"] == params
    )
    assert len(calls["run_rb"]) == 12
    assert rb == sorted((b, *s) for b in rb_params for s in seal)
    baselines = sorted(
        (c["name"], c["scenario"]["location"], c["scenario"]["start_day"])
        for c in calls["run_baseline"]
    )
    assert baselines == sorted((b, *s) for b in ("zero_controls", "random") for s in seal)
    # The rules never see a price: every baseline plays at sim (the spies bind price_regime's default).
    assert {c["price_regime"] for c in calls["run_rb"] + calls["run_baseline"]} == {"sim"}

    df = pd.read_parquet(final.dir / "seasons.parquet")
    assert list(df.columns[: len(IDENTITY) + 1]) == [*IDENTITY, "price_regime"]
    assert set(df.columns[len(IDENTITY) + 1 :]) == {*ledger.RESULT_KEYS, "valid"}
    assert len(df) == 72  # (2 runs + 4 baselines) x 3 splits x 2 scenarios x 2 price regimes
    seasons = {}
    for row in df.to_dict("records"):
        seasons.setdefault(tuple(_null(row[c]) for c in IDENTITY), []).append(row)
    assert len(seasons) == 36
    heats, moved = [], []
    for season, rows in seasons.items():
        assert sorted(r["price_regime"] for r in rows) == sorted(REGIMES), season
        sim, agc = (next(r for r in rows if r["price_regime"] == g) for g in REGIMES)
        assert {c: _null(agc[c]) for c in UNPRICED} == {c: _null(sim[c]) for c in UNPRICED}, season
        # agc2020 over sim: heat 0.03 / 0.09 = 1/3 and electricity 0.06 / 0.30 = 1/5 (EUR/kWh), CO2
        # 0.08 / 0.30 = 4/15 (EUR/kg); fruit 1.6 EUR/kg in both. So, in EUR/m2, from the sim row:
        # agc2020 profit = yield x 1.6 - heat_cost / 3 - elec_cost / 5 - co2_cost x 4/15.
        assert agc["heat_cost"] == pytest.approx(sim["heat_cost"] / 3, rel=0, abs=1e-9), season
        assert agc["elec_cost"] == pytest.approx(sim["elec_cost"] / 5, rel=0, abs=1e-9), season
        assert agc["co2_cost"] == pytest.approx(sim["co2_cost"] * 4 / 15, rel=0, abs=1e-9), season
        want = (
            sim["yield_kg_m2"] * 1.6
            - sim["heat_cost"] / 3
            - sim["elec_cost"] / 5
            - sim["co2_cost"] * 4 / 15
        )
        assert agc["profit_eur_m2"] == pytest.approx(want, rel=0, abs=1e-9), season
        # sim: yield x 1.6 EUR/kg less its own heat, electricity and CO2 costs [EUR/m2].
        costs = sim["heat_cost"] + sim["elec_cost"] + sim["co2_cost"]
        own = sim["yield_kg_m2"] * 1.6 - costs
        assert sim["profit_eur_m2"] == pytest.approx(own, rel=0, abs=1e-9), season
        heats.append(sim["heat_cost"] > 0)
        moved.append(abs(agc["profit_eur_m2"] - sim["profit_eur_m2"]) > 1e-6)
    assert any(heats), "no season heats, so the heat price is never tested"
    assert any(moved), "no season's profit moves with the prices"


def test_a_2013_seasons_rows_are_its_play_and_the_play_regraded(final_run_2013, monkeypatch):
    """Done when (#72, one play graded twice): a season's row at the regime it was played at is
    the library's play of it, key for key and exactly; its other row is the same raw trace graded
    by metrics.episode_metrics at the other regime's PRICES (written out; 96 season steps),
    carrying the play's own paper_reward_sum:
    - seed 0 (manifest agc2020) on london_2013 day 49: its agc2020 row is
      run_policy(run_dir, "final", s, 1, allow_test=True)'s result; its sim row is that play's
      trace at sim prices plus the play's paper_reward_sum;
    - seed 1 (manifest sim) on abbotsford_2013 day 45: its sim row is run_policy(run_dir, "1024",
      s, 1, allow_test=True)'s result; its agc2020 row the trace at agc2020 prices plus the play's
      paper_reward_sum;
    - rb_default (run_rb) and random (run_baseline) on amsterdam_2013 day 45: each row is the
      play's trace graded at its row's regime, and both rows carry one paper_reward_sum, the
      play's own reward summed: within rel 1e-6 of one row's paper_return (a baseline's play
      regime isn't pinned).
    A --final that took each regime's row from a separate play, or graded the learners at sim
    whatever their manifest, fails here. Non-vacuous: each learner's two rows differ in profit."""
    final = final_run_2013()
    w = final.w
    df = pd.read_parquet(final.dir / "seasons.parquet")
    assert "price_regime" in df.columns, "seasons.parquet has no price_regime column"
    monkeypatch.setattr(scenarios, "WEATHER_DIR", w.synthetic13)

    for seed, split, day in [(0, "london_2013", 49), (1, "abbotsford_2013", 45)]:
        s = {"location": SITES_2013[split], "growth_year": 2013, "start_day": day}
        played, _, trace = evaluate.run_policy(
            w.run_dirs[seed], CHECKPOINTS[seed], s, 1, return_trace=True, allow_test=True
        )
        rows = _two_rows(df, split, day, ARM, w.run_ids[seed])
        play = PLAYED_AT[seed]
        [other] = [g for g in REGIMES if g != play]
        assert {k: rows[play][k] for k in played} == played, (seed, play)
        want = {**_graded(trace, other), "paper_reward_sum": played["paper_reward_sum"]}
        assert {k: rows[other][k] for k in want} == want, (seed, other)
        assert rows["sim"]["profit_eur_m2"] != rows["agc2020"]["profit_eur_m2"], seed

    s = {"location": "Amsterdam", "growth_year": 2013, "start_day": 45}
    _, _, rb_trace = evaluate.run_rb(RB_DEFAULT, s, 1, return_trace=True, allow_test=True)
    for arm, trace in [("rb_default", rb_trace), ("random", _baseline_trace("random", s))]:
        rows = _two_rows(df, "amsterdam_2013", 45, arm)
        for regime in REGIMES:
            want = _graded(trace, regime)
            assert {k: rows[regime][k] for k in want} == want, (arm, regime)
        reward = rows["sim"]["paper_reward_sum"]
        assert rows["agc2020"]["paper_reward_sum"] == reward, arm
        played_at = [
            g for g in REGIMES if reward == pytest.approx(rows[g]["paper_return"], rel=1e-6)
        ]
        assert len(played_at) == 1, (arm, reward, played_at)


def test_a_2013_per_split_is_stats_on_each_regimes_rows_alone(final_run_2013, tmp_path):
    """Done when (#72, per_split per regime): per_split.json is {split: {regime: {arm: entry}}},
    each 2013 split holding exactly sim and agc2020, each of those exactly the residual arm and the
    four baselines; and each entry is computed on that regime's seasons.parquet rows alone, as
    test_per_split_is_stats_on_the_seasons computes a 2012 entry, exactly:
    - iqm, ci_low, ci_high: stats.iqm_ci (2,000 reps, seed 0) on the runs x scenarios profit
      (the residual arm 2 x 2, seeds ascending; a baseline 1 x 2);
    - guards: {iqm, ci_low, ci_high} of stats.iqm_ci on each FINAL_GUARDS key's array;
    - invalid_episodes: that regime's rows with valid False;
    - prob_improvement: stats.prob_improvement(arm, rb_tuned) on that regime's profit, null for
      rb_tuned; scored: the registered checkpoints, {} for a baseline.
    `ledger final-result` on a copy of the dual world's ledger then records it as written.
    Non-vacuous: some arm's sim and agc2020 profit IQMs differ, so stats on both regimes' rows
    together, or on one regime's for both, give other numbers."""
    final = final_run_2013()
    w = final.w
    df = pd.read_parquet(final.dir / "seasons.parquet")
    numbers = final.dir / "per_split.json"
    per_split = json.loads(numbers.read_text())["per_split"]

    assert set(per_split) == set(SITES_2013)
    for split in SITES_2013:
        assert set(per_split[split]) == set(REGIMES), (split, sorted(per_split[split]))
    differ = []
    for split in SITES_2013:
        for regime in REGIMES:
            assert set(per_split[split][regime]) == {ARM, *BASELINES}, (split, regime)
            rb_tuned = _scores_2013(df, split, "rb_tuned", PRIMARY, regime)
            for arm, entry in per_split[split][regime].items():
                where = (split, regime, arm)
                profit = _scores_2013(df, split, arm, PRIMARY, regime)
                assert profit.shape == ((2, 2) if arm == ARM else (1, 2)), where
                assert (entry["iqm"], entry["ci_low"], entry["ci_high"]) == stats.iqm_ci(profit)
                guards = {
                    g: dict(
                        zip(
                            ("iqm", "ci_low", "ci_high"),
                            stats.iqm_ci(_scores_2013(df, split, arm, g, regime)),
                            strict=True,
                        )
                    )
                    for g in FINAL_GUARDS
                }
                assert entry["guards"] == guards, where
                mine = df[
                    (df["split"] == split) & (df["arm"] == arm) & (df["price_regime"] == regime)
                ]
                invalid = int(np.count_nonzero(~mine["valid"].to_numpy(dtype=bool)))
                assert entry["invalid_episodes"] == invalid, where
                want = None if arm == "rb_tuned" else stats.prob_improvement(profit, rb_tuned)
                assert entry["prob_improvement"] == want, where
                assert entry["scored"] == (w.checkpoints13 if arm == ARM else {}), where
        for arm in (ARM, *BASELINES):
            differ.append(
                per_split[split]["sim"][arm]["iqm"] != per_split[split]["agc2020"][arm]["iqm"]
            )
    assert any(differ), "every arm's sim and agc2020 profit IQMs are equal"

    path = tmp_path / "ledger.jsonl"
    shutil.copyfile(w.ledger, path)
    argv = ["final-result", "--final-registered", w.freg13, "--numbers", str(numbers)]
    assert ledger.main(argv, path=path) == 0
    [row] = [
        r for r in ledger.rows(kind="final-result", path=path) if r["final-registered"] == w.freg13
    ]
    assert row["per_split"] == per_split


def test_a_2013_final_prints_one_line_per_split_regime_and_arm(final_run_2013):
    """Done when (#72, printed lines name the regime): before its last line (the `next:` ledger
    command), --final prints exactly one line per 2013 split x price regime x arm (the residual arm
    and the four baselines), 3 x 2 x 5 = 30, each naming its split, its regime and its arm as words
    (so rb_tuned isn't found in another arm's name) and not the other regime."""
    final = final_run_2013()
    lines = final.stdout.strip().splitlines()

    assert lines[-1].startswith("next: python -m greenleaf_control.ledger final-result"), lines[-1]
    shown = lines[:-1]
    assert len(shown) == 30, "\n".join(shown)  # 3 splits x 2 regimes x (1 arm + 4 baselines)

    def word(name: str) -> re.Pattern:
        return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)")

    for split in SITES_2013:
        for regime in REGIMES:
            [other] = [g for g in REGIMES if g != regime]
            for arm in (ARM, *BASELINES):
                found = [
                    line
                    for line in shown
                    if all(word(n).search(line) for n in (split, regime, arm))
                    and not word(other).search(line)
                ]
                assert len(found) == 1, (split, regime, arm, found)


def _hourly_profit(trace: tuple, regime: str) -> list[float]:
    """export_replay's hourly profit by hand for a 1-day season's (states, controls, params): hour
    h is metrics.episode_metrics on the season's first 4h + 4 steps at regime's PRICES (written
    out), profit_eur_m2 to 4 decimals, 24 hours."""
    states, controls, params = trace
    return [
        round(
            metrics.episode_metrics(
                states[: k + 1], controls[:k], params, season_steps=k, prices=PRICES[regime]
            )["profit_eur_m2"],
            4,
        )
        for k in range(4, N_DAY + 1, 4)
    ]


def test_a_2013_final_writes_one_replay_per_traced_play_naming_its_regime(
    final_run_2013, monkeypatch
):
    """Done when (#72, replays): replays/ holds one file per traced play, not one per graded row:
    <prefix>_<location>_2013_<start_day>.json for the residual arm's lowest seed (seed 0) and each
    of the four baselines on the 6 scenarios, 30 files. Each meta.note is "registered final <row>:
    ..." and names, as a word, the regime its season was played at and not the other; seed 0's
    (manifest agc2020) names agc2020. Each totals is its season's seasons.parquet row at that
    regime, key for key and exactly: the play's own result.
    #72 fix round: the hourly running profit is graded at the play's regime too, so a replay's last
    hour is its totals: hourly.profit[-1] == round(totals.profit_eur_m2, 4) for every replay, seed
    0's at agc2020 included; and hourly.profit is the prefix grading at the play regime's PRICES
    (_hourly_profit), checked on seed 0's london_2013 day 49 (agc2020, replayed with run_policy)
    and rb_default's amsterdam_2013 day 45 (sim, replayed with run_rb). Non-vacuous: seed 0's
    prefixes at sim and at agc2020 differ."""
    final = final_run_2013()
    w = final.w
    df = pd.read_parquet(final.dir / "seasons.parquet")
    want = {
        f"{prefix}_{site}_2013_{day}.json": (prefix, split, day)
        for prefix in TRACED
        for split, site in SITES_2013.items()
        for day in DAYS
    }
    assert len(want) == 30  # (1 traced run + 4 baselines) x 3 splits x 2 scenarios

    assert _listing(final.dir / "replays") == sorted(want)
    for name, (prefix, split, day) in want.items():
        payload = json.loads((final.dir / "replays" / name).read_text(encoding="utf-8"))
        note = payload["meta"]["note"]
        assert note.startswith(f"registered final {w.freg13}: "), (name, note)
        named = [g for g in REGIMES if re.search(rf"(?<!\w){g}(?!\w)", note)]
        assert len(named) == 1, (name, note)
        if prefix == ARM:
            assert named == [PLAYED_AT[0]], (name, note)
        rows = _two_rows(df, split, day, prefix, w.run_ids[0] if prefix == ARM else None)
        totals = payload["totals"]
        assert {k: rows[named[0]][k] for k in totals} == totals, name
        # export_replay rounds each hour's profit to 4 decimals; its last hour is the whole season.
        last = payload["hourly"]["profit"][-1]
        assert last == round(totals["profit_eur_m2"], 4), (name, named[0], last)

    monkeypatch.setattr(scenarios, "WEATHER_DIR", w.synthetic13)
    s = {"location": "London", "growth_year": 2013, "start_day": 49}
    _, _, seed0 = evaluate.run_policy(
        w.run_dirs[0], CHECKPOINTS[0], s, 1, return_trace=True, allow_test=True
    )
    s = {"location": "Amsterdam", "growth_year": 2013, "start_day": 45}
    _, _, rb = evaluate.run_rb(RB_DEFAULT, s, 1, return_trace=True, allow_test=True)
    for name, trace, regime in [
        (f"{ARM}_London_2013_49.json", seed0, PLAYED_AT[0]),
        ("rb_default_Amsterdam_2013_45.json", rb, "sim"),
    ]:
        payload = json.loads((final.dir / "replays" / name).read_text(encoding="utf-8"))
        assert payload["hourly"]["profit"] == _hourly_profit(trace, regime), (name, regime)
    assert _hourly_profit(seed0, "sim") != _hourly_profit(seed0, "agc2020")


def test_a_2013_final_scores_rb_tuned_agc2020_like_the_other_baselines(
    world, final_run, tmp_path, monkeypatch
):
    """Done when (#72, rb_tuned_agc2020 among the 2013 final's baselines): a 2013 row registers
    rb_tuned_agc2020 beside rb_default and rb_tuned (_world_2013, seed 0 at "final"), and --final
    scores it like them. rule_based.CONFIGS points at a tmp copy of configs/ whose
    rule_based_tuned_agc2020.yml holds the default params with three setpoints moved (as
    test_evaluate's rb_tuned_agc2020 test writes it); SEALS[2013] is cut to day 49 per split, so 3
    scenarios. --final returns 0, and:
    - evaluate.run_rb (a pass-through spy) plays those params once per scenario, allow_test=True;
    - seasons.parquet holds its two rows per season, sim and agc2020, each the play's trace
      (run_rb with those params on london_2013 day 49; the three sites' 2013.csv are one copy)
      graded at that row's regime, exactly;
    - per_split.json holds it in every (split, regime), scored {} and prob_improvement a number,
      and `ledger final-result` records the file.
    Non-vacuous: its seasons aren't rb_default's (the moved setpoints are what it plays)."""
    baselines = ("rb_default", "rb_tuned", "rb_tuned_agc2020")
    w = _world_2013(world, final_run, tmp_path, monkeypatch, baselines=baselines)
    configs = tmp_path / "configs"
    shutil.copytree(rule_based.CONFIGS, configs)
    tuned = {**RB_DEFAULT, "temp_setpoint_day": 21.25, "co2_day": 937.5, "lamps_off": 16.5}
    text = yaml.safe_dump({"GreenLightEnv": tuned}, sort_keys=False)
    (configs / "rule_based_tuned_agc2020.yml").write_text(text)
    monkeypatch.setattr(rule_based, "CONFIGS", configs)
    cut = {
        split: [{"location": site, "growth_year": 2013, "start_day": 49}]
        for split, site in SITES_2013.items()
    }
    monkeypatch.setattr(scenarios, "TEST_2013", cut)
    monkeypatch.setitem(scenarios.SEALS, 2013, cut)
    play, calls = evaluate.run_rb, []
    monkeypatch.setattr(evaluate, "run_rb", _spy(play, calls))

    assert _main(["--final", w.freg13], w.ledger) == 0

    mine = [c for c in calls if c["params"] == tuned]
    assert sorted(c["scenario"]["location"] for c in mine) == sorted(SITES_2013.values())
    assert all(c["allow_test"] for c in mine)
    out = w.results / w.freg13
    df = pd.read_parquet(out / "seasons.parquet")
    _, _, trace = play(tuned, cut["london_2013"][0], 1, return_trace=True, allow_test=True)
    moved = []
    for split in SITES_2013:
        rows = _two_rows(df, split, 49, "rb_tuned_agc2020")
        for regime in REGIMES:
            want = _graded(trace, regime)
            assert {k: rows[regime][k] for k in want} == want, (split, regime)
        rb_default = _two_rows(df, split, 49, "rb_default")["sim"]
        moved.append(any(rows["sim"][k] != rb_default[k] for k in ("yield_kg_m2", *PRICED)))
    assert any(moved), "rb_tuned_agc2020 plays every season as rb_default does"
    per_split = json.loads((out / "per_split.json").read_text())["per_split"]
    for split in SITES_2013:
        for regime in REGIMES:
            entry = per_split[split][regime]["rb_tuned_agc2020"]
            assert entry["scored"] == {}, (split, regime)
            assert isinstance(entry["prob_improvement"], float), (split, regime)
    argv = [
        "final-result",
        "--final-registered",
        w.freg13,
        "--numbers",
        str(out / "per_split.json"),
    ]
    assert ledger.main(argv, path=w.ledger) == 0


# A row's rb_* baselines whose config yml is missing: (the row's seal, the baseline whose yml is
# removed from the tmp configs/ copy). The 2013 rows register rb_default, rb_tuned and
# rb_tuned_agc2020; the 2012 row is the module's (the four BASELINES).
MISSING_RB_CONFIG = [
    pytest.param(2013, "rb_tuned_agc2020", id="2013-rb-tuned-agc2020"),
    pytest.param(2013, "rb_default", id="2013-rb-default"),
    pytest.param(2013, "rb_tuned", id="2013-rb-tuned"),
    pytest.param(2012, "rb_tuned", id="2012-rb-tuned"),
]


@pytest.mark.parametrize(("seal", "missing"), MISSING_RB_CONFIG)
def test_a_final_refuses_a_registered_rb_baseline_without_its_config_before_any_env(
    seal, missing, world, final_run, tmp_path, monkeypatch, capsys
):
    """Done when (#72 fix round, a registered rb_* baseline needs its params on disk):
    rule_based.CONFIGS points at a tmp copy of configs/ holding rule_based_tuned_agc2020.yml (the
    default params with three setpoints moved) but without the missing baseline's
    rule_based_<name>.yml. --final <the row> is refused before any env is built (_no_env_built) and before any weather is read (no new
    2012 or 2013 read, the tripwire silent), as parser.error's SystemExit with code 2 naming the
    missing file's path (and no other config's), with no FINAL_RESULTS/<row>/ made (FINAL_RESULTS
    as it was) and the ledger byte-identical. Otherwise the final opens the seal, plays seasons and
    only then fails on load_rb_params, leaving an interrupted final that can never be rerun.
    - 2013, rb_tuned_agc2020: the case the fix round names (_world_2013, seed 0 at "final");
    - 2013, rb_default and rb_tuned: any registered rb_* baseline, the same way;
    - 2012, rb_tuned: the 2012 row (_copy_world) too."""
    if seal == 2013:
        baselines = ("rb_default", "rb_tuned", "rb_tuned_agc2020")
        w = _world_2013(world, final_run, tmp_path, monkeypatch, baselines=baselines)
        row = w.freg13
    else:
        w = _copy_world(world, tmp_path, monkeypatch)
        row = w.freg_id
    configs = tmp_path / "configs"
    shutil.copytree(rule_based.CONFIGS, configs)
    tuned = {**RB_DEFAULT, "temp_setpoint_day": 21.25, "co2_day": 937.5, "lamps_off": 16.5}
    text = yaml.safe_dump({"GreenLightEnv": tuned}, sort_keys=False)
    (configs / "rule_based_tuned_agc2020.yml").write_text(text)
    gone = configs / f"rule_based_{missing.removeprefix('rb_')}.yml"
    gone.unlink()
    monkeypatch.setattr(rule_based, "CONFIGS", configs)
    present = sorted(configs.glob("rule_based_*.yml"))
    assert present and gone not in present, present
    results, ledger_bytes = _tree(w.results), w.ledger.read_bytes()
    n_2012, n_2013, n_trips = len(world.reads), len(world.reads_2013), len(world.trips)
    _no_env_built(monkeypatch)
    capsys.readouterr()

    with pytest.raises(SystemExit) as excinfo:
        _main(["--final", row], w.ledger)

    assert excinfo.value.code == 2, "a refusal is parser.error's exit 2"
    text = _refusal(excinfo, capsys)
    assert str(gone) in text, text
    assert not any(str(p) in text for p in present), text
    assert not (w.results / row).exists()
    assert _tree(w.results) == results
    assert w.ledger.read_bytes() == ledger_bytes
    assert world.reads[n_2012:] == [] and world.reads_2013[n_2013:] == []
    assert world.trips[n_trips:] == []


def test_the_full_final_touches_nothing_real_and_writes_no_ledger_row(world, final_run):
    """Done when (nothing real touched): after the full --final, the tripwire never fired and the
    only 2012 files read were the three synthetic copies; the tmp ledger is byte-identical (the
    --final writes no ledger row); and the real ledger, control/runs/ and control/results/'s top
    level are as they were when this module was collected. Runs last in the module."""
    final_run()

    assert world.trips == [], f"a real 2012.csv was read: {world.trips}"
    assert set(world.reads) == {
        (world.synthetic / s / "2012.csv").resolve() for s in SITES.values()
    }
    assert world.ledger.read_bytes() == world.ledger_bytes
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE
    assert _listing(REAL_RESULTS) == REAL_RESULTS_BEFORE

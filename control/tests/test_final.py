"""Tests for evaluate --final (issue #43, F1a-3): score every registered run at its registered
checkpoint and every registered baseline once on the three test splits, after refusing anything that
could break the seal (PROTOCOL §8).

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
  checkpoints, all four baselines, in FINAL_BASELINES order);
- then points scenarios.WEATHER_DIR at tmp "synthetic", where Amsterdam, London and Abbotsford each
  have a 2012.csv that is a byte copy of the real Amsterdam/2010.csv; cuts scenarios.TEST to start
  days 45 and 49 per split; sets evaluate.SEASON_LENGTH to 1 day (96 steps) and
  evaluate.FINAL_RESULTS to tmp "final_out".

One full --final runs once (final_run) and is shared. A test that changes the ledger, a run dir or
FINAL_RESULTS works on its own copies (_copy_world). evaluate.main runs in-process, since
guard_runs blocks `evaluate --final` on the command line. No fixture calls the --final, so while it
is missing every test FAILS in its own body rather than ERRORs.
"""

import ast
import contextlib
import inspect
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO

from greenleaf_control import envs, evaluate, ledger, manifest, rule_based, scenarios, stats, train

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
BASELINES = ledger.FINAL_BASELINES  # zero_controls, random, rb_default, rb_tuned: the row's order
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
        if path.name.endswith("2012.csv"):
            if not path.is_relative_to(tmp_root):
                w.trips.append(path)
                raise AssertionError(f"the sealed 2012 test weather was read: {path}")
            w.reads.append(path)
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
        mp.setattr(evaluate, "SEASON_LENGTH", 1)
        mp.setattr(evaluate, "FINAL_RESULTS", w.results, raising=False)
        yield w


def _main(argv: list[str], ledger_path: Path) -> int:
    """evaluate.main(argv, ledger_path=ledger_path), in-process."""
    assert "ledger_path" in inspect.signature(evaluate.main).parameters, (
        "evaluate.main takes no ledger_path: main(argv=None, *, ledger_path=ledger.LEDGER)"
    )
    return evaluate.main(argv, ledger_path=ledger_path)


@pytest.fixture(scope="module")
def final_run(world):
    """A function that runs one full evaluate --final <the world's row> into world.results on its
    first call and returns {code, stdout, dir}. A failed attempt isn't cached, so each test that
    needs it fails in its own body."""
    done = {}

    def run() -> SimpleNamespace:
        if not done:
            out = io.StringIO()
            with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
                mp.setattr(evaluate, "FINAL_RESULTS", world.results)
                code = _main(["--final", world.freg_id], world.ledger)
            done.update(code=code, stdout=out.getvalue(), dir=world.results / world.freg_id)
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


def test_a_full_final_writes_three_files_with_36_seasons_in_order(world, final_run):
    """Done when: a full --final returns 0 and writes exactly FINAL_RESULTS/<row>/ holding
    per_split.json, seasons.parquet and stamp.json (the code stamp), nothing else in FINAL_RESULTS.

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
    assert _listing(final.dir) == ["per_split.json", "seasons.parquet", "stamp.json"]
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

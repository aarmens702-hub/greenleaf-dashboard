"""Tests for greenleaf_control.run_grid and greenleaf_control.modal_app (issue #17, part 17a; issue
#19, part 19b).

run_grid (docs/API-CONTRACT.md):
- --dry takes the Measurement's shape as flags, before any registration exists, in ledger
  register's CLI shapes: --issue N --arms <JSON list of {name, method, config}> --seeds S [S ...]
  --timesteps T --dry [--parallel P]. It prints the run count, the worst-case CPU-hours, the
  worst-case dollars on Modal and a wall-clock estimate, and writes nothing.
- --registered <row> --backend local [--parallel 2] checks every cell of the registration, then
  hands one train argv per arm x seed (what train.main(argv) takes) to the module-level seam
  _run_local(argvs, parallel), which returns each argv's exit code in argvs order. Tests replace
  that seam, so nothing trains here.
- --registered <row> --backend modal [--parallel 2] (#19b) makes the same checks, then refuses a
  dirty tree (manifest.code_stamp(smoke=False)) and any cell whose local run dir
  run_grid.RUNS/<row>/<arm>/seed<S> already exists, before any call. Then, inside one session from
  the module-level seam _run_remote() (a context manager yielding call(argv, meta) -> the finished
  manifest, as a dict), --parallel cells at a time: a run row, call(argv, {"stamp": code_stamp,
  "registered": row}), then a run-end, crashed if the call raised. The Mac is the only ledger
  writer, and a run-end's cost_usd is its wall_s x the hard limits' price per second. Tests
  replace _run_remote with a fake session, and tests/conftest.py's autouse fixture makes the real
  one raise, so nothing here can reach Modal or spend.
- Before any call it also refuses a cell that already has a run row for the registration (its
  run dir may be only on the volume, not yet downloaded). call(argv, meta, run_rel) gets
  run_rel = "<row>/<arm>/seed<S>", the run dir under the volume's /runs. Ctrl-C mid-grid cancels
  the cells not yet launched (they get no rows), names them, closes the in-flight runs and
  re-raises.

Print format pinned here (loosely, by regex on stdout):
- "<n> runs"; "<h> CPU-h" (or CPU-hours); "$<usd>" to the cent;
- the wall-clock estimate in hours, as "<x> h" (or hours): the per-run time and the local total;
- "no measured run" when the ledger has no finished run-end to measure a rate from.

Known answers (Modal prices as of 2026-10-28: $0.0000131 per physical core-second, $0.00000222
per GiB-second; a run is capped at 4 cores, 8,192 MiB = 8 GiB, 4 h = 14,400 s):
- worst case per run = 14,400 s x (4 x 0.0000131 + 8 x 0.00000222) $/s = 14,400 x 0.00007016 =
  $1.010304; 3 arms x 5 seeds = 15 runs: 15 x 4 h x 4 cores = 240 CPU-h and $15.15456 -> $15.15.
- one remote run at its hard limits costs 4 x 0.0000131 + 8 x 0.00000222 = 0.0000524 + 0.00001776
  = $0.00007016 per second of wall clock, so a run-end's cost_usd = wall_s x 0.00007016.

modal_app (and the SDK) is imported only in a subprocess with the network blocked and the SDK's
config pointed at an absent file, so ~/.modal.toml is never read and nothing can reach Modal; the
pytest process never imports either.

greenleaf_control.run_grid and modal_app are imported inside each test body, so while they are
missing each test FAILS rather than ERRORs at collection. Nothing here may write the real ledger or
control/runs/; the last test checks both are as they were when this module was collected.
"""

import contextlib
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path, PurePosixPath

import pytest

import greenleaf_control
from greenleaf_control import ledger, manifest, rule_based, scenarios, train

CONTROL = Path(__file__).resolve().parents[1]
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
REAL_RUNS = CONTROL / "runs"
SCRIPT = CONTROL / "scripts" / "check_locks.py"
# One remote run at its hard limits, $ per second of wall clock: 4 cores x $0.0000131 = $0.0000524,
# plus 8 GiB x $0.00000222 = $0.00001776, so $0.00007016.
USD_PER_S = 0.00007016
# CI has no git identity and may sign commits globally; no hook may run in a temp repo.
COMMIT_CONFIG = (
    *("-c", "user.name=t", "-c", "user.email=t@t"),
    *("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"),
)
CONTAINER_WALL_S = 12_345.0  # a fake manifest's wall_s (the container's), never the Mac's
PARTNER_WAIT_S = 2.0  # how long a fake call waits for --parallel calls to be in flight
STOP_WAIT_S = 10.0  # how long an interrupted fake call waits for its session to close


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _listing(path: Path) -> list[str] | None:
    return sorted(p.name for p in path.iterdir()) if path.exists() else None


# The real ledger and runs dir, read when this module is collected, before any test in it runs.
REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)
REAL_RUNS_BEFORE = _listing(REAL_RUNS)

# Two arms whose configs name run-shaping train flags, so their argvs carry them.
ARMS = [
    {
        "name": "ppo_low_w",
        "method": "ppo",
        "config": {"penalty_weight": 0.5, "season_length": 1, "n_envs": 1},
    },
    {
        "name": "lag_half",
        "method": "residual_lag",
        "config": {"budget_frac": 0.5, "rb": "default", "season_length": 1, "n_envs": 1},
    },
]
# Each arm's train flags, written out by hand: its method, then its config as --kebab-case flags.
ARM_FLAGS = {
    "ppo_low_w": ["--method", "ppo", "--penalty-weight", "0.5", "--season-length", "1"]
    + ["--n-envs", "1"],
    "lag_half": ["--method", "residual_lag", "--budget-frac", "0.5", "--rb", "default"]
    + ["--season-length", "1", "--n-envs", "1"],
}
SEEDS = [3, 7]
METHOD = {arm["name"]: arm["method"] for arm in ARMS}
CELLS = [(arm["method"], seed) for arm in ARMS for seed in SEEDS]  # (method, seed), as _cell reads
REGISTRATION = {
    "issue": 17,
    "hypothesis": "plumbing: run_grid launches every registered cell",
    "arms": ARMS,
    "seeds": SEEDS,
    "timesteps": 2048,
    "scenarios": ["validation"],
    "primary": "profit_eur_m2",
    "guards": ["viol_hours_temp"],
    "rule": "none: a test registration",
    "budget": {"cpu_hours": 0.01, "usd": 0.0, "backend": "local"},
}
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}


@pytest.fixture(autouse=True)
def _never_the_real_ledger(monkeypatch):
    """A ledger call that forgets path= would append to the real ledger; this refuses it."""
    real_append = ledger.append

    def append(row: dict, path: Path = ledger.LEDGER) -> None:
        if Path(path).resolve() == REAL_LEDGER.resolve():
            raise AssertionError(f"a test tried to append a {row['kind']} row to the real ledger")
        real_append(row, path)

    monkeypatch.setattr(ledger, "append", append)


def _no_launch(*args, **kwargs):
    raise AssertionError("_run_local was called: this command must launch nothing")


def _files(root: Path) -> dict[str, bytes]:
    """Every file under root, by relative path, with its bytes."""
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _registered(tmp_path: Path, **over) -> tuple[Path, dict]:
    """A tmp ledger holding REGISTRATION with over replacing its fields: (ledger path, row)."""
    path = tmp_path / "ledger.jsonl"
    return path, ledger.register(**{**REGISTRATION, **over}, path=path)


def _run_end(path: Path, reg: dict, arm: str, seed: int, status: str, wall_s, timesteps) -> None:
    """One run of reg and its run-end, through the ledger API."""
    run = ledger.run_start(
        registered=reg["row_id"], arm=arm, seed=seed, run_dir=f"runs/{arm}/seed{seed}", path=path
    )
    ledger.run_end(
        run=run["row_id"],
        status=status,
        wall_s=wall_s,
        timesteps=timesteps,
        solver_failures=0,
        cost_usd=None,
        diagnostics=DIAGNOSTICS,
        path=path,
    )


def _dry_argv(n_arms: int, seeds: list[int], *extra: str, timesteps: int = 1_000_000) -> list[str]:
    arms = [
        {"name": f"arm{i}", "method": "ppo", "config": {"penalty_weight": 0.5 * (i + 1)}}
        for i in range(n_arms)
    ]
    return [
        *("--issue", "17"),
        *("--arms", json.dumps(arms)),
        *("--seeds", *map(str, seeds)),
        *("--timesteps", str(timesteps)),
        "--dry",
        *extra,
    ]


def _dry(run_grid, argv: list[str], tmp_path: Path, path: Path, monkeypatch, capsys) -> str:
    """main(argv) with --dry, run from tmp_path: it returns 0, launches nothing and leaves every
    file under tmp_path (the tmp ledger included) as it was. Returns its stdout."""
    monkeypatch.setattr(run_grid, "_run_local", _no_launch, raising=False)
    monkeypatch.chdir(tmp_path)
    before = _files(tmp_path)
    capsys.readouterr()
    assert run_grid.main(argv, ledger_path=path) == 0
    assert _files(tmp_path) == before
    return capsys.readouterr().out


def _train_args(argv) -> dict:
    """argv as train's own parser reads it, minus --runs-dir (where runs go isn't pinned here)."""
    args = vars(train._parser().parse_args(list(argv)))
    del args["runs_dir"]
    return args


def _cell(args: dict) -> tuple[str, int]:
    return args["method"], args["seed"]


def _expected_cells(row_id: str) -> list[dict]:
    """The four cells, 2 arms x seeds 3 and 7, as train would parse them."""
    return sorted(
        (
            _train_args(
                [*flags, "--seed", str(seed), "--timesteps", "2048", "--registered", row_id]
            )
            for flags in ARM_FLAGS.values()
            for seed in SEEDS
        ),
        key=_cell,
    )


def _fake_launch(monkeypatch, run_grid, fails=lambda args: False) -> list:
    """Replace _run_local with a recorder: each argv "exits" 1 if fails(its train args), else 0.

    Returns the list of (argvs, parallel) calls it received.
    """
    calls = []

    def run_local(argvs, parallel):
        calls.append((argvs, parallel))
        return [1 if fails(_train_args(argv)) else 0 for argv in argvs]

    monkeypatch.setattr(run_grid, "_run_local", run_local)
    return calls


def _refused(run_grid, argv: list[str], path: Path, capsys) -> str:
    """main(argv) must refuse: a ValueError, a RuntimeError (code_stamp's dirty tree), a
    FileExistsError, a nonzero SystemExit or a nonzero return.

    Returns the refusal's text: the exception's message (or exit code) plus stdout and stderr.
    """
    capsys.readouterr()
    try:
        code = run_grid.main(argv, ledger_path=path)
    except SystemExit as exc:
        assert exc.code not in (0, None), "a refusal must exit nonzero"
        text = str(exc.code)
    except (ValueError, RuntimeError, FileExistsError) as exc:
        text = str(exc)
    else:
        assert code not in (0, None), f"main returned {code!r}; expected a refusal"
        text = ""
    captured = capsys.readouterr()
    return text + captured.out + captured.err


def _main(run_grid, argv: list[str], path: Path, capsys) -> tuple[int | None, str]:
    """main(argv): (its return value, or a SystemExit's code; its stdout plus stderr)."""
    capsys.readouterr()
    try:
        code = run_grid.main(argv, ledger_path=path)
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], check=True, cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, *COMMIT_CONFIG, "commit", "-q", "-m", message)


def _clean_repo(tmp_path: Path, monkeypatch) -> Path:
    """A temp git repo laid out like this one (docs/PROTOCOL.md and control/ledger/.gitkeep
    committed) as manifest.REPO_ROOT, so code_stamp(smoke=False) gives its HEAD, clean."""
    # A git hook's environment would point these git calls at the real repo.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    (repo / "control" / "ledger").mkdir(parents=True)
    (repo / "control" / "ledger" / ".gitkeep").write_text("")
    (repo / "docs").mkdir()
    (repo / "docs" / "PROTOCOL.md").write_text("protocol\n")
    _git(repo, "init", "-q")
    _commit(repo, "init")
    monkeypatch.setattr(manifest, "REPO_ROOT", repo)
    return repo


def _runs_at(monkeypatch, run_grid, runs: Path) -> None:
    """runs (not made) becomes run_grid.RUNS and train.RUNS, where a remote run's dir lands."""
    monkeypatch.setattr(run_grid, "RUNS", runs, raising=False)
    monkeypatch.setattr(train, "RUNS", runs)


def _remote_grid(tmp_path: Path, monkeypatch, run_grid, **over) -> tuple[Path, dict, Path]:
    """A remote grid's Mac side: a clean temp repo (tmp_path/repo) as REPO_ROOT, tmp_path/runs as
    the runs dir, and a tmp ledger outside the repo holding REGISTRATION with over replacing its
    fields. Returns (ledger path, row, runs dir)."""
    _clean_repo(tmp_path, monkeypatch)
    runs = tmp_path / "runs"
    _runs_at(monkeypatch, run_grid, runs)
    path, row = _registered(tmp_path, **over)
    return path, row, runs


def _cell_diagnostics(seed: int) -> dict:
    return {"min_approx_kl": seed / 1000, "final_entropy": -1.5, "final_explained_variance": 0.8}


def _finished_manifest(args: dict, meta: dict) -> dict:
    """What train_remote returns for a finished run: its manifest.json, as a dict.

    Its numbers are the cell's own (solver_failures = its seed, min_approx_kl = seed / 1000), so a
    run-end written from another cell's manifest shows. Its wall_s is the container's
    (CONTAINER_WALL_S) and its cost_usd None: neither is the Mac's.
    """
    seed = args["seed"]
    return {
        **meta["stamp"],
        "config": {"method": args["method"], "seed": seed, "timesteps": args["timesteps"]},
        "registered": meta["registered"]["row_id"],
        "issue": meta["registered"]["issue"],
        "smoke": False,
        "status": "finished",
        "wall_s": CONTAINER_WALL_S,
        "timesteps": args["timesteps"],
        "solver_failures": seed,
        "cost_usd": None,
        "diagnostics": _cell_diagnostics(seed),
    }


def _fake_session(
    monkeypatch, run_grid, *, error=None, fails=lambda args: False, parallel=2, total=4
) -> dict:
    """Replace run_grid._run_remote with a fake session: a context manager yielding
    call(argv, meta, run_rel), which stands in for train_remote.remote and contacts nothing.

    Each call records (argv, meta, run_rel) and a ("call", (method, seed)) event, waits until
    `parallel` calls are in flight (or all `total` have started; PARTNER_WAIT_S at most), sleeps
    0.01 s, then raises error if fails(its train args), else returns the cell's finished manifest.

    Returns the fake's state: opened and closed (session entries and exits), calls, events, peak
    (the most calls in flight at once), outside (calls made with no session open) and lock.
    """
    lock = threading.Condition()
    state = {
        **{"opened": 0, "closed": 0, "calls": [], "events": []},
        **{"in_flight": 0, "peak": 0, "outside": 0, "lock": lock},
    }

    def started_enough() -> bool:
        return state["in_flight"] >= parallel or len(state["calls"]) >= total

    def call(argv, meta, run_rel):
        args = _train_args(argv)
        with lock:
            if state["opened"] <= state["closed"]:
                state["outside"] += 1
            state["calls"].append((list(argv), meta, run_rel))
            state["events"].append(("call", _cell(args)))
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
            lock.notify_all()
            lock.wait_for(started_enough, timeout=PARTNER_WAIT_S)
        try:
            time.sleep(0.01)
            if fails(args):
                raise error
            return _finished_manifest(args, meta)
        finally:
            with lock:
                state["in_flight"] -= 1

    @contextlib.contextmanager
    def run_remote():
        state["opened"] += 1
        try:
            yield call
        finally:
            state["closed"] += 1

    monkeypatch.setattr(run_grid, "_run_remote", run_remote, raising=False)
    return state


def _record_appends(monkeypatch, state: dict) -> None:
    """ledger.append also logs ("append", row) to the fake session's events once the row is
    written, so each cell's order of run row, call and run-end row can be read off."""
    inner = ledger.append

    def append(row: dict, path: Path = ledger.LEDGER) -> None:
        inner(row, path)
        with state["lock"]:
            state["events"].append(("append", row))

    monkeypatch.setattr(ledger, "append", append)


def _cell_order(events: list) -> dict:
    """{(method, seed): what happened to that cell, in order}: "run", "call" and "run-end"."""
    runs, order = {}, {}
    for event, what in events:
        if event == "call":
            cell, kind = what, "call"
        elif what["kind"] in ("run", "run-end"):
            kind = what["kind"]
            run = what if kind == "run" else runs[what["run"]]
            runs[run["row_id"]] = run
            cell = (METHOD[run["arm"]], run["seed"])
        else:
            continue
        order.setdefault(cell, []).append(kind)
    return order


def _ends_by_cell(path: Path) -> dict:
    """{(arm, seed): its run-end} from the ledger at path, through each run-end's run row."""
    runs = {r["row_id"]: r for r in ledger.rows(kind="run", path=path)}
    return {
        (runs[end["run"]]["arm"], runs[end["run"]]["seed"]): end
        for end in ledger.rows(kind="run-end", path=path)
    }


def test_remote_resources_and_prices_are_module_constants():
    """One remote run's caps and Modal's prices are module constants, read by --dry and by
    modal_app: 4.0 physical cores, 8,192 MiB, 4 h = 14,400 s; $0.0000131 per core-second and
    $0.00000222 per GiB-second, as of 2026-10-28 (modal.com/pricing; one core is 2 vCPU). And
    run_grid.RUNS, where a remote run's dir lands, is train.RUNS (control/runs): run_grid states it
    rather than import train (torch, gl_gym)."""
    from greenleaf_control import run_grid

    assert (run_grid.CPU, run_grid.MEMORY_MB, run_grid.TIMEOUT_S) == (4.0, 8192, 14_400)
    assert (run_grid.USD_PER_CORE_S, run_grid.USD_PER_GIB_S) == (0.0000131, 0.00000222)
    assert run_grid.PRICES_AS_OF == "2026-10-28"
    assert run_grid.RUNS == train.RUNS == REAL_RUNS


DRY_CASES = [
    # 3 x 5 = 15 runs; 15 x 4 h x 4 cores = 240 CPU-h; 15 x $1.010304 = $15.15456 -> $15.15
    pytest.param(3, [0, 1, 2, 3, 4], "15", "240", "15.15", id="3-arms-5-seeds"),
    # 1 x 2 = 2 runs; 2 x 4 h x 4 cores = 32 CPU-h; 2 x $1.010304 = $2.020608 -> $2.02
    pytest.param(1, [0, 1], "2", "32", "2.02", id="1-arm-2-seeds"),
]


@pytest.mark.parametrize(("n_arms", "seeds", "runs", "cpu_hours", "usd"), DRY_CASES)
def test_dry_prints_the_worst_case(
    n_arms, seeds, runs, cpu_hours, usd, tmp_path, monkeypatch, capsys
):
    """Done when: run_grid --dry prints a worst case for a fake registration.

    runs = arms x seeds; worst-case CPU-hours = runs x 4 h x 4 cores; worst-case dollars = runs x
    14,400 s x (4 cores x $0.0000131 + 8 GiB x $0.00000222). No ledger exists, and none is made.
    """
    from greenleaf_control import run_grid

    path = tmp_path / "ledger.jsonl"
    out = _dry(run_grid, _dry_argv(n_arms, seeds), tmp_path, path, monkeypatch, capsys)

    assert re.search(rf"(?<![\d.]){runs} runs\b", out), out
    assert re.search(rf"(?<![\d.]){cpu_hours}(\.0+)?\s*CPU-h", out), out
    assert re.search(rf"\$\s?{re.escape(usd)}(?!\d)", out), out
    assert not path.exists()


@pytest.mark.parametrize(
    ("extra", "total"),
    [
        ([], r"12(\.0+)?"),  # --parallel 2 (default): ceil(15 / 2) = 8 waves x 1.5 h = 12.0 h
        (["--parallel", "3"], r"7\.50*"),  # ceil(15 / 3) = 5 waves x 1.5 h = 7.5 h
        (["--parallel", "4"], r"6(\.0+)?"),  # ceil(15 / 4) = 4 waves x 1.5 h = 6.0 h
    ],
    ids=["parallel-default-2", "parallel-3", "parallel-4"],
)
def test_dry_estimates_wall_clock_from_the_newest_finished_run(
    extra, total, tmp_path, monkeypatch, capsys
):
    """The wall-clock estimate comes from the newest finished run-end in the ledger:
    rate = wall_s / timesteps, per run = rate x T, local total = ceil(runs / parallel) x per run.

    The ledger holds, oldest first: a finished run at 999 s / 1,000 steps (0.999 s/step), a finished
    run at 108 s / 20,000 steps (0.0054 s/step) and a crashed run at 7 s / 10 steps (0.7 s/step).
    Known answer: the newest finished is the second, so per run = 0.0054 s/step x 1,000,000 steps =
    5,400 s = 1.5 h, and 3 arms x 5 seeds = 15 runs. The older run would give 277.5 h per run and
    the crashed one 194.4 h. The ledger is byte-identical afterwards.
    """
    from greenleaf_control import run_grid

    path, reg = _registered(tmp_path)
    _run_end(path, reg, "ppo_low_w", 3, "finished", wall_s=999.0, timesteps=1_000)
    _run_end(path, reg, "ppo_low_w", 7, "finished", wall_s=108.0, timesteps=20_000)
    _run_end(path, reg, "lag_half", 3, "crashed", wall_s=7.0, timesteps=10)

    out = _dry(run_grid, _dry_argv(3, [0, 1, 2, 3, 4], *extra), tmp_path, path, monkeypatch, capsys)

    assert re.search(r"(?<![\d.])1\.50*\s*h", out), out  # per run
    assert re.search(rf"(?<![\d.]){total}\s*h", out), out  # local total
    assert "no measured run" not in out


@pytest.mark.parametrize("state", ["no-ledger-file", "only-a-crashed-run"])
def test_dry_says_no_measured_run_without_a_finished_run_end(state, tmp_path, monkeypatch, capsys):
    """With no ledger, or a ledger whose only run-end crashed, there is no rate to measure: the
    wall-clock line says "no measured run", and the worst case is still printed."""
    from greenleaf_control import run_grid

    path = tmp_path / "ledger.jsonl"
    if state == "only-a-crashed-run":
        path, reg = _registered(tmp_path)
        _run_end(path, reg, "lag_half", 3, "crashed", wall_s=7.0, timesteps=10)

    out = _dry(run_grid, _dry_argv(3, [0, 1, 2, 3, 4]), tmp_path, path, monkeypatch, capsys)

    assert "no measured run" in out
    assert re.search(r"(?<![\d.])15 runs\b", out), out


def _tune_end(path: Path) -> None:
    """A finished tune_rb run-end, appended through the ledger API: its registration (one rb_tuned
    arm of 200 trials, seed 0, timesteps 0), its run, and the run-end tune_rb.main records when it
    finishes: 5,200 s of wall clock, timesteps 0 (nothing trains), solver_failures 0, cost None
    and every diagnostic None (no PPO update)."""
    arm = {"name": "rb_tuned", "method": "rb_tuned", "config": {"trials": 200}}
    reg = ledger.register(
        **{**REGISTRATION, "arms": [arm], "seeds": [0], "timesteps": 0}, path=path
    )
    run = ledger.run_start(
        registered=reg["row_id"], arm="rb_tuned", seed=0, run_dir="runs/rb_tuned/seed0", path=path
    )
    ledger.run_end(
        run=run["row_id"],
        status="finished",
        wall_s=5200.0,
        timesteps=0,
        solver_failures=0,
        cost_usd=None,
        diagnostics=dict.fromkeys(ledger.DIAGNOSTIC_KEYS),
        path=path,
    )


def test_dry_estimates_from_a_training_run_not_a_newer_tune_with_no_timesteps(
    tmp_path, monkeypatch, capsys
):
    """Regression for #17 (17b review): tune_rb records a finished run-end with timesteps 0, and
    --dry divided its wall_s by those 0 timesteps (ZeroDivisionError). The wall-clock estimate uses
    only finished run-ends with timesteps > 0.

    The ledger holds, oldest first: a finished training run at 108 s / 20,000 steps (0.0054
    s/step), then a NEWER finished tune_rb run-end (5,200 s, timesteps 0). Known answer: the
    estimate is the training run's, 0.0054 s/step x 1,000,000 steps = 5,400 s = 1.5 h per run, and
    3 arms x 5 seeds = 15 runs at --parallel 2 is ceil(15 / 2) = 8 waves x 1.5 h = 12 h locally. It
    exits 0 and the ledger is byte-identical afterwards.
    """
    from greenleaf_control import run_grid

    path, reg = _registered(tmp_path)
    _run_end(path, reg, "ppo_low_w", 3, "finished", wall_s=108.0, timesteps=20_000)
    _tune_end(path)

    out = _dry(run_grid, _dry_argv(3, [0, 1, 2, 3, 4]), tmp_path, path, monkeypatch, capsys)

    assert re.search(r"(?<![\d.])1\.50*\s*h", out), out  # per run
    assert re.search(r"(?<![\d.])12(\.0+)?\s*h", out), out  # local total
    assert "no measured run" not in out


def test_dry_says_no_measured_run_when_the_only_finished_run_end_has_no_timesteps(
    tmp_path, monkeypatch, capsys
):
    """Regression for #17 (17b review): a ledger whose only finished run-end is a tune's
    (timesteps 0) has no rate to measure: --dry exits 0, the wall-clock line says "no measured
    run", and the worst case (15 runs) is still printed."""
    from greenleaf_control import run_grid

    path = tmp_path / "ledger.jsonl"
    _tune_end(path)

    out = _dry(run_grid, _dry_argv(3, [0, 1, 2, 3, 4]), tmp_path, path, monkeypatch, capsys)

    assert "no measured run" in out
    assert re.search(r"(?<![\d.])15 runs\b", out), out


@pytest.mark.parametrize(("extra", "parallel"), [([], 2), (["--parallel", "3"], 3)])
def test_registered_local_launches_one_train_run_per_arm_and_seed(
    extra, parallel, tmp_path, monkeypatch
):
    """--registered <row> --backend local hands _run_local, once, one train argv per arm x seed,
    with --parallel (default 2).

    Known answer: 2 arms x seeds 3 and 7 = 4 argvs. Read by train's own parser, each is the arm's
    method and config as flags (penalty_weight 0.5 -> --penalty-weight 0.5, ...), the seed, the
    registered 2,048 timesteps and --registered <row>; every flag the arm doesn't name is at train's
    default. Every argv element is a str. run_grid itself writes no ledger row.
    """
    from greenleaf_control import run_grid

    path, row = _registered(tmp_path)
    calls = _fake_launch(monkeypatch, run_grid)
    before = path.read_bytes()

    argv = ["--registered", row["row_id"], "--backend", "local", *extra]
    assert run_grid.main(argv, ledger_path=path) == 0

    [(argvs, got_parallel)] = calls
    assert got_parallel == parallel
    assert all(isinstance(arg, str) for argv in argvs for arg in argv)
    assert sorted(map(_train_args, argvs), key=_cell) == _expected_cells(row["row_id"])
    assert path.read_bytes() == before


def test_registered_local_reports_a_failed_run_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    """One of the four runs (lag_half, seed 7) exits 1. Every cell is still handed to _run_local,
    main exits nonzero, and its output has a line naming the failure with that run's arm (or
    method) and seed; no failure line names the arm that succeeded on both seeds."""
    from greenleaf_control import run_grid

    path, row = _registered(tmp_path)
    calls = _fake_launch(monkeypatch, run_grid, fails=lambda a: _cell(a) == ("residual_lag", 7))
    capsys.readouterr()

    try:
        code = run_grid.main(
            ["--registered", row["row_id"], "--backend", "local"], ledger_path=path
        )
    except SystemExit as exc:
        code = exc.code
    captured = capsys.readouterr()
    text = captured.out + captured.err

    assert code not in (0, None)
    [(argvs, _)] = calls
    assert len(argvs) == 4
    failures = [line for line in text.splitlines() if "fail" in line.lower()]
    assert any(
        ("lag_half" in line or "residual_lag" in line) and re.search(r"(?<![\d.])7\b", line)
        for line in failures
    ), text
    assert not any("ppo_low_w" in line for line in failures), text


REGISTERED_REFUSALS = [
    pytest.param({}, "registered-deadbeef", "registered-deadbeef", id="no-such-row"),
    pytest.param(
        {"arms": [ARMS[0], {"name": "dqn_arm", "method": "dqn", "config": {}}]},
        None,
        "dqn",
        id="method-not-in-envs-METHODS",
    ),
    pytest.param(
        {"arms": [ARMS[0], {"name": "lr_arm", "method": "ppo", "config": {"learning_rate": 3e-4}}]},
        None,
        "learning_rate",
        id="config-key-is-no-train-flag",
    ),
    # train's flags are --kebab-case but a config's keys are their snake_case dests, as train's
    # registration check reads them; "penalty-weight" would launch a run train then refuses.
    pytest.param(
        {"arms": [ARMS[0], {"name": "kebab", "method": "ppo", "config": {"penalty-weight": 0.5}}]},
        None,
        "penalty-weight",
        id="config-key-in-kebab-case",
    ),
    *[
        pytest.param({"scenarios": ["validation", split]}, None, split, id=f"test-split-{split}")
        for split in scenarios.TEST
    ],
]


@pytest.mark.parametrize("backend", ["local", "modal"], ids=["local", "remote"])
@pytest.mark.parametrize(("over", "row_id", "named"), REGISTERED_REFUSALS)
def test_registered_refuses_before_launching_anything(
    over, row_id, named, backend, tmp_path, monkeypatch, capsys
):
    """run_grid checks every cell before it launches any, on either backend: an unknown row, an
    arm whose method isn't in envs.METHODS, an arm config key that isn't a train flag, or a 2012
    TEST split among the registration's scenarios (PROTOCOL §8) each refuse naming what's wrong.
    The bad arm is the second, after a good one. _run_local is never called, no remote session is
    opened and the ledger is byte-identical. The remote grid has a clean temp repo and an empty
    runs dir, so only the registration can be what's refused."""
    from greenleaf_control import run_grid

    if backend == "modal":
        _clean_repo(tmp_path, monkeypatch)
        _runs_at(monkeypatch, run_grid, tmp_path / "runs")
    path, row = _registered(tmp_path, **over)
    monkeypatch.setattr(run_grid, "_run_local", _no_launch, raising=False)
    session = _fake_session(monkeypatch, run_grid)
    before = path.read_bytes()

    argv = ["--registered", row_id or row["row_id"], "--backend", backend]
    assert named in _refused(run_grid, argv, path, capsys)
    assert path.read_bytes() == before
    assert (session["opened"], session["calls"]) == (0, [])


def test_registered_remote_records_a_run_and_run_end_per_cell_with_its_cost(
    tmp_path, monkeypatch, capsys
):
    """Done when (#19): with _run_remote patched to a fake, run_grid --registered <row> --backend
    modal writes one run and one run-end per arm x seed to a tmp ledger, each run-end with
    cost_usd > 0.

    The Mac is the only ledger writer. Inside one session, --parallel (default 2) cells at a time,
    each cell gets a run row, then call(argv, meta, run_rel), then a run-end row. The argv is
    exactly what the local backend hands train for that cell; meta is {"stamp":
    manifest.code_stamp(smoke=False) of the clean temp repo, "registered": the row}, and survives
    JSON as it is (train_remote passes it as --remote); run_rel is the str "<row>/<arm>/seed<S>",
    the cell's run dir under the volume's /runs, so train_remote reads that manifest and no other.
    A run row's run_dir is the absolute run_grid.RUNS/run_rel, where the volume's copy lands from
    control/; run_grid writes nothing there itself. run_grid.RUNS, unpatched, is train's
    control/runs.

    Known answers: the fake returns each cell's finished manifest, solver_failures = its seed (3 or
    7) and min_approx_kl = seed / 1000, so a run-end written from another cell's manifest shows.
    A run-end's wall_s is the Mac's monotonic wall around its call (at least the fake's 0.01 s
    sleep), not the manifest's 12,345 s, and its cost_usd = wall_s x $0.00007016 per s.
    """
    from greenleaf_control import run_grid

    default_runs = getattr(run_grid, "RUNS", None)
    path, row, runs = _remote_grid(tmp_path, monkeypatch, run_grid)
    local = _fake_launch(monkeypatch, run_grid)
    assert (
        run_grid.main(["--registered", row["row_id"], "--backend", "local"], ledger_path=path) == 0
    )
    [(local_argvs, _)] = local
    stamp = manifest.code_stamp(smoke=False)
    session = _fake_session(monkeypatch, run_grid)
    _record_appends(monkeypatch, session)

    argv = ["--registered", row["row_id"], "--backend", "modal"]
    code, text = _main(run_grid, argv, path, capsys)

    assert code == 0, text
    argvs = [call_argv for call_argv, _, _ in session["calls"]]
    assert sorted(argvs) == sorted(local_argvs)
    assert sorted(map(_train_args, argvs), key=_cell) == _expected_cells(row["row_id"])
    for _, meta, _ in session["calls"]:
        assert meta == {"stamp": stamp, "registered": row}
        assert json.loads(json.dumps(meta)) == meta
    run_rels = {_cell(_train_args(a)): run_rel for a, _, run_rel in session["calls"]}
    assert run_rels == {
        (arm["method"], seed): f"{row['row_id']}/{arm['name']}/seed{seed}"
        for arm in ARMS
        for seed in SEEDS
    }
    assert all(type(run_rel) is str for run_rel in run_rels.values())
    # one session, every call inside it, --parallel 2 calls at a time
    assert (session["opened"], session["closed"], session["outside"]) == (1, 1, 0)
    assert session["peak"] == 2
    assert _cell_order(session["events"]) == {cell: ["run", "call", "run-end"] for cell in CELLS}

    reg, *rest = ledger.rows(path=path)
    assert reg == row
    assert sorted(r["kind"] for r in rest) == ["run"] * 4 + ["run-end"] * 4
    for run in ledger.rows(kind="run", path=path):
        assert (run["registered"], run["status"]) == (row["row_id"], "running")
        assert Path(run["run_dir"]).is_absolute()
        assert Path(run["run_dir"]) == runs / row["row_id"] / run["arm"] / f"seed{run['seed']}"
    ends = _ends_by_cell(path)
    assert sorted(ends) == sorted((arm["name"], seed) for arm in ARMS for seed in SEEDS)
    for (_, seed), end in ends.items():
        assert (end["status"], end["timesteps"], end["solver_failures"]) == ("finished", 2048, seed)
        assert end["diagnostics"] == _cell_diagnostics(seed)
        assert 0.009 <= end["wall_s"] < 60, end  # >= the 0.01 s sleep; not the container's 12,345 s
        assert end["cost_usd"] == pytest.approx(end["wall_s"] * USD_PER_S, rel=1e-9)
        assert end["cost_usd"] > 0
    assert not runs.exists()
    assert default_runs == REAL_RUNS


class _Cancelled(BaseException):  # noqa: N818  (named for what it stands in for)
    """Like a remote input's cancellation: a BaseException, so `except Exception` misses it."""


FAILED_CALLS = [
    pytest.param(RuntimeError("remote train crashed"), id="crash"),
    pytest.param(TimeoutError("remote call timed out"), id="timeout"),
    pytest.param(_Cancelled("remote input cancelled"), id="cancelled"),
]


@pytest.mark.parametrize("error", FAILED_CALLS)
def test_registered_remote_records_a_failed_call_as_crashed(error, tmp_path, monkeypatch, capsys):
    """Done when (#19): a fake remote crash yields a run-end with status crashed.

    lag_half seed 7's call raises: an error, a timeout, or a BaseException like a cancellation,
    which `except Exception` would miss. Every cell is still called. main exits nonzero (it may not
    let the exception escape), with a line naming the failure's arm (or method) and seed 7, and no
    failure line naming ppo_low_w. That run-end is crashed with timesteps, solver_failures and
    every diagnostic null (no manifest came back); its wall_s is the Mac's wall around the call
    and cost_usd = wall_s x $0.00007016 > 0, since a failed call still bills. The other three
    run-ends are finished.
    """
    from greenleaf_control import run_grid

    path, row, _ = _remote_grid(tmp_path, monkeypatch, run_grid)
    session = _fake_session(
        monkeypatch, run_grid, error=error, fails=lambda a: _cell(a) == ("residual_lag", 7)
    )

    argv = ["--registered", row["row_id"], "--backend", "modal"]
    try:
        code, text = _main(run_grid, argv, path, capsys)
    except type(error) as exc:
        pytest.fail(f"main let {exc!r} escape instead of recording a crashed run-end and exiting 1")

    assert code not in (0, None), text
    assert len(session["calls"]) == 4
    failures = [line for line in text.splitlines() if "fail" in line.lower()]
    assert any(
        ("lag_half" in line or "residual_lag" in line) and re.search(r"(?<![\d.])7\b", line)
        for line in failures
    ), text
    assert not any("ppo_low_w" in line for line in failures), text
    ends = _ends_by_cell(path)
    assert sorted(ends) == sorted((arm["name"], seed) for arm in ARMS for seed in SEEDS)
    crashed = ends.pop(("lag_half", 7))
    assert crashed["status"] == "crashed"
    assert (crashed["timesteps"], crashed["solver_failures"]) == (None, None)
    assert crashed["diagnostics"] == dict.fromkeys(ledger.DIAGNOSTIC_KEYS)
    assert crashed["wall_s"] >= 0.009  # the fake sleeps 0.01 s before it raises
    assert crashed["cost_usd"] == pytest.approx(crashed["wall_s"] * USD_PER_S, rel=1e-9)
    assert crashed["cost_usd"] > 0
    assert [end["status"] for end in ends.values()] == ["finished"] * 3


def test_a_remote_grid_with_a_crash_passes_check_locks(tmp_path, monkeypatch):
    """Done when (#19): a fake remote crash yields a run-end with status crashed, and the ledger
    passes check_locks.

    Laid out like this repo: the ledger is control/ledger/ledger.jsonl in a temp repo, its
    registration committed. lag_half seed 7's call raises RuntimeError; the grid exits nonzero
    with every run closed (one crashed run-end, three finished); its rows are committed; then
    scripts/check_locks.py --repo <the temp repo> exits 0 with one line starting "check_locks:
    clean": every run has its run-end and the ledger only grew.
    """
    from greenleaf_control import run_grid

    repo = _clean_repo(tmp_path, monkeypatch)
    path = repo / "control" / "ledger" / "ledger.jsonl"
    row = ledger.register(**REGISTRATION, path=path)
    _commit(repo, "register")
    _runs_at(monkeypatch, run_grid, tmp_path / "runs")
    _fake_session(
        monkeypatch,
        run_grid,
        error=RuntimeError("remote train crashed"),
        fails=lambda a: _cell(a) == ("residual_lag", 7),
    )

    try:
        code = run_grid.main(
            ["--registered", row["row_id"], "--backend", "modal"], ledger_path=path
        )
    except SystemExit as exc:
        code = exc.code
    assert code not in (0, None)
    runs = ledger.rows(kind="run", path=path)
    ends = ledger.rows(kind="run-end", path=path)
    assert len(runs) == 4
    assert sorted(end["run"] for end in ends) == sorted(run["row_id"] for run in runs)
    assert sorted(end["status"] for end in ends) == ["crashed"] + ["finished"] * 3
    _commit(repo, "runs and run-ends")

    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("check_locks: clean"), out.stdout


def test_registered_remote_refuses_a_dirty_tree_before_any_call(tmp_path, monkeypatch, capsys):
    """A remote run's manifest records the Mac's stamp, so the tree must be clean before anything
    starts: with an untracked control/new.py in the temp repo, --backend modal refuses (code_stamp's
    RuntimeError or its own) saying dirty and naming the file, before any session opens or any
    call; the ledger is byte-identical and nothing is written under the runs dir."""
    from greenleaf_control import run_grid

    path, row, runs = _remote_grid(tmp_path, monkeypatch, run_grid)
    (manifest.REPO_ROOT / "control" / "new.py").write_text("written after the commit\n")
    session = _fake_session(monkeypatch, run_grid)
    before = path.read_bytes()

    text = _refused(run_grid, ["--registered", row["row_id"], "--backend", "modal"], path, capsys)
    assert "dirty" in text, text
    assert "control/new.py" in text, text
    assert (session["opened"], session["calls"]) == (0, [])
    assert path.read_bytes() == before
    assert not runs.exists()


def test_registered_remote_refuses_an_existing_local_run_dir_before_any_call(
    tmp_path, monkeypatch, capsys
):
    """A remote run's dir comes back to run_grid.RUNS/<row>/<arm>/seed<S> (PROTOCOL §5: written
    once, never resumed), so an existing one refuses the whole grid before any session opens or
    any call. Here the last cell's, lag_half seed 7, exists: the refusal names it
    (<row>/lag_half/seed7), the ledger is byte-identical and the dir is as it was."""
    from greenleaf_control import run_grid

    path, row, runs = _remote_grid(tmp_path, monkeypatch, run_grid)
    existing = runs / row["row_id"] / "lag_half" / "seed7"
    existing.mkdir(parents=True)
    (existing / "notes.txt").write_text("an earlier run\n")
    session = _fake_session(monkeypatch, run_grid)
    before = path.read_bytes()

    text = _refused(run_grid, ["--registered", row["row_id"], "--backend", "modal"], path, capsys)
    assert f"{row['row_id']}/lag_half/seed7" in text, text
    assert (session["opened"], session["calls"]) == (0, [])
    assert path.read_bytes() == before
    assert _files(runs) == {f"{row['row_id']}/lag_half/seed7/notes.txt": b"an earlier run\n"}


def test_registered_remote_refuses_cells_that_already_ran_before_any_call(
    tmp_path, monkeypatch, capsys
):
    """Regression for #19 (19b review): a remote run's dir is only on the volume until `modal
    volume get` copies it into control/runs, so the local run-dir check misses it, and a second
    --backend modal on the same registration relaunched every cell into its existing dir on the
    volume (train refuses it) and recorded each as crashed.

    run_grid refuses any cell that already has a run row for this registration in the ledger,
    before any session opens or any call, naming the cell. Here lag_half seed 7 ran (its run and
    run-end rows are in the ledger) and nothing is in the local runs dir: the refusal has a line
    naming lag_half and seed 7, and the ledger is byte-identical.
    """
    from greenleaf_control import run_grid

    path, row, runs = _remote_grid(tmp_path, monkeypatch, run_grid)
    _run_end(path, row, "lag_half", 7, "finished", wall_s=99.0, timesteps=2048)
    session = _fake_session(monkeypatch, run_grid)
    before = path.read_bytes()

    text = _refused(run_grid, ["--registered", row["row_id"], "--backend", "modal"], path, capsys)
    # first: a grid whose calls all failed also exits nonzero naming lag_half seed 7
    assert (session["opened"], session["calls"]) == (0, []), text
    assert any(
        "lag_half" in line and re.search(r"seed\s?7(?!\d)", line) for line in text.splitlines()
    ), text
    assert path.read_bytes() == before
    assert not runs.exists()


def _interrupting_session(monkeypatch, run_grid, *, swallows: bool) -> dict:
    """A fake session whose first call is where Ctrl-C lands mid-grid.

    That call sends SIGINT to the main thread, as Ctrl-C does: a real signal, so a main thread
    blocked waiting on the cells wakes at once (_thread.interrupt_main() wouldn't wake it until a
    cell finished, by when the pool has started the next). It then stays in flight until the
    session closes (STOP_WAIT_S at most) and raises, as a real call does when its app stops. A
    later call raises at once.

    With swallows, a KeyboardInterrupt reaching the session's exit is caught there and the session
    returns normally, as the real one does (the SDK's _run_app catches the first KeyboardInterrupt,
    disconnects and returns, so `with app.run()` suppresses it); otherwise it propagates. Returns
    its state: run_rel of each call, in order, sigints, and swallowed (interrupts it caught).
    """
    closed = threading.Event()
    state = {"calls": [], "sigints": 0, "swallowed": 0}
    lock = threading.Lock()

    def call(argv, meta, run_rel):
        with lock:
            state["calls"].append(run_rel)
            first = len(state["calls"]) == 1
        if first:
            state["sigints"] += 1
            signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
            closed.wait(timeout=STOP_WAIT_S)
        raise RuntimeError("the remote app stopped: input cancelled")

    @contextlib.contextmanager
    def run_remote():
        try:
            yield call
        except KeyboardInterrupt:
            if not swallows:
                raise
            state["swallowed"] += 1  # disconnect and return normally, as _run_app does
        finally:
            closed.set()

    monkeypatch.setattr(run_grid, "_run_remote", run_remote)
    return state


@pytest.mark.parametrize("swallows", [True, False], ids=["session-swallows", "session-reraises"])
def test_ctrl_c_mid_grid_launches_no_queued_cell_and_closes_the_one_in_flight(
    swallows, tmp_path, monkeypatch, capsys
):
    """Regression for #19 (19b review): on Ctrl-C mid-grid, the pool still ran every queued cell
    after shutdown began, so each wrote a run row, failed at once against the stopped app and got
    a crashed run-end: runs that never trained, recorded as crashes. And the real session swallows
    the interrupt (the SDK's _run_app catches it, disconnects and returns), after which asking a
    cancelled cell's future for its exception raised CancelledError.

    --parallel 1 over the 4 cells: the first cell's call is in flight when Ctrl-C (a SIGINT to the
    main thread) lands, and the other 3 are queued. The outcome is the same whether the session
    swallows the interrupt or lets it through. Pinned: KeyboardInterrupt propagates out of main
    (run_grid re-raises it after the session and the pool close; never a CancelledError, never an
    exit code). Known answers: exactly one call and one run row, closed by a crashed run-end (no
    open run); the 3 queued cells get no row and are never called; the output has a line saying
    "not launched" (any case) naming each queued cell's arm and seed.
    """
    from greenleaf_control import run_grid

    path, row, _ = _remote_grid(tmp_path, monkeypatch, run_grid)
    session = _interrupting_session(monkeypatch, run_grid, swallows=swallows)
    argv = ["--registered", row["row_id"], "--backend", "modal", "--parallel", "1"]
    threads_before = set(threading.enumerate())
    capsys.readouterr()

    old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)  # SIGINT is Ctrl-C
    try:
        outcome = f"main returned {run_grid.main(argv, ledger_path=path)!r}"
    except KeyboardInterrupt:
        outcome = "KeyboardInterrupt"
    except CancelledError as exc:  # a cancelled cell's future asked for its result or exception
        outcome = f"CancelledError {exc!r}"
    finally:
        signal.signal(signal.SIGINT, old_handler)
    # A main that waits for its pool leaves no thread; one that doesn't may leave workers still
    # writing rows, joined here where they're still listed (best effort, for a wrong build only).
    for thread in set(threading.enumerate()) - threads_before:
        thread.join(timeout=STOP_WAIT_S)
    captured = capsys.readouterr()
    text = captured.out + captured.err

    assert session["sigints"] == 1
    assert outcome == "KeyboardInterrupt", outcome
    runs = ledger.rows(kind="run", path=path)
    assert len(runs) == 1, [(r["arm"], r["seed"]) for r in runs]
    [run] = runs
    assert len(session["calls"]) == 1, session["calls"]
    ends = _ends_by_cell(path)
    assert list(ends) == [(run["arm"], run["seed"])]
    assert ends[(run["arm"], run["seed"])]["status"] == "crashed"
    queued = sorted({(arm["name"], seed) for arm in ARMS for seed in SEEDS} - set(ends))
    assert len(queued) == 3
    lines = [line for line in text.splitlines() if "not launched" in line.lower()]
    for arm, seed in queued:
        assert any(arm in line and re.search(rf"(?<![\d.]){seed}\b", line) for line in lines), (
            arm,
            seed,
            text,
        )


def test_remote_grid_with_a_faked_session_never_imports_the_remote_app(
    tmp_path, monkeypatch, capsys
):
    """run_grid imports the remote app (and with it the SDK) only inside the real _run_remote, so
    neither importing run_grid nor a grid whose session is faked imports greenleaf_control.modal_app
    or modal: the pytest process never holds the SDK. run_grid is imported afresh here and its own
    _run_remote replaced by the fake; the grid calls all 4 cells and exits 0."""
    assert "modal" not in sys.modules, "an earlier test imported the SDK into the pytest process"
    for name in ("run_grid", "modal_app"):
        monkeypatch.delitem(sys.modules, f"greenleaf_control.{name}", raising=False)
        monkeypatch.delattr(greenleaf_control, name, raising=False)
    import greenleaf_control.run_grid as run_grid

    path, row, _ = _remote_grid(tmp_path, monkeypatch, run_grid)
    session = _fake_session(monkeypatch, run_grid)

    code, text = _main(
        run_grid, ["--registered", row["row_id"], "--backend", "modal"], path, capsys
    )

    assert code == 0, text
    assert len(session["calls"]) == 4
    assert "greenleaf_control.modal_app" not in sys.modules
    assert "modal" not in sys.modules


CLI_REFUSALS = [
    pytest.param([], None, id="neither-dry-nor-registered"),
    pytest.param(["--registered", "{row}"], "--backend", id="registered-without-backend"),
    pytest.param(["--registered", "{row}", "--backend", "aws"], "aws", id="unknown-backend"),
    pytest.param(
        ["--issue", "17", "--seeds", "0", "--timesteps", "1000", "--dry"],
        "--arms",
        id="dry-without-arms",
    ),
]


@pytest.mark.parametrize(("flags", "named"), CLI_REFUSALS)
def test_command_line_refusals(flags, named, tmp_path, monkeypatch, capsys):
    """No mode, --registered without --backend, a backend other than local or modal, and --dry
    without the Measurement's arms each exit nonzero (naming the flag or value where there is
    one), launch nothing and leave the ledger byte-identical."""
    from greenleaf_control import run_grid

    path, row = _registered(tmp_path)
    monkeypatch.setattr(run_grid, "_run_local", _no_launch, raising=False)
    before = path.read_bytes()

    text = _refused(run_grid, [f.replace("{row}", row["row_id"]) for f in flags], path, capsys)
    assert named is None or named in text
    assert path.read_bytes() == before


def test_help_lists_the_contract_flags(capsys):
    """main(["--help"]) exits 0 and lists every flag: --registered, --backend (local, modal),
    --parallel, and --dry's --issue, --arms, --seeds, --timesteps."""
    from greenleaf_control import run_grid

    with pytest.raises(SystemExit) as excinfo:
        run_grid.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    flags = ["--registered", "--backend", "--parallel", "--dry", "--issue", "--arms", "--seeds"]
    missing = [flag for flag in [*flags, "--timesteps", "local", "modal"] if flag not in out]
    assert not missing, f"--help doesn't list {missing}"


# Run in a fresh interpreter: block every socket call (recording any attempt), spy on the Image
# builder methods (each spy records its bound arguments, then calls the real method), import the
# remote app, then print one JSON line: what the SDK holds locally for train_remote, no network.
_OFFLINE_IMPORT = textwrap.dedent(
    """
    import inspect
    import json
    import pathlib
    import socket

    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(repr(args)[:200])
        raise OSError("the network is blocked in this test")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked

    import modal
    import modal.config

    def plain(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, pathlib.PurePath):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        return repr(value)

    calls = []

    def spy(name, raw):
        func = raw.__func__ if isinstance(raw, staticmethod) else raw
        signature = inspect.signature(func)

        def record(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = {k: plain(v) for k, v in bound.arguments.items() if k != "self"}
            calls.append({"name": name, "args": arguments})
            return func(*args, **kwargs)

        return staticmethod(record) if isinstance(raw, staticmethod) else record

    for name in ("debian_slim", "apt_install", "uv_sync", "add_local_python_source", "add_local_dir"):
        setattr(modal.Image, name, spy(name, inspect.getattr_static(modal.Image, name)))

    from greenleaf_control import modal_app

    info = modal_app.train_remote.info()
    print(json.dumps({
        "app_is_app": isinstance(modal_app.app, modal.App),
        "config_path": modal.config.user_config_path,
        "cpu": info.cpu,
        "memory_mib": info.memory_mib,
        "timeout": info.timeout,
        "max_retries": info.max_retries,
        "single_use_containers": info.single_use_containers,
        "nonpreemptible": info.nonpreemptible,
        "volumes": {path: mount.name for path, mount in info.volumes.items()},
        "calls": calls,
        "attempts": attempts,
    }))
    """
)


@pytest.fixture(scope="module")
def remote_import(tmp_path_factory) -> tuple[Path, subprocess.CompletedProcess]:
    """One offline import of the remote app in a subprocess, shared by the tests that check it.

    The SDK's own env vars are dropped and MODAL_CONFIG_PATH names a file that doesn't exist, so no
    token is read and ~/.modal.toml is never opened. Asserts nothing: the tests do.
    """
    tmp = tmp_path_factory.mktemp("remote")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("MODAL_")}
    env["MODAL_CONFIG_PATH"] = str(tmp / "absent.toml")
    result = subprocess.run(
        [sys.executable, "-c", _OFFLINE_IMPORT],
        env=env,
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    return tmp, result


def _report(result: subprocess.CompletedProcess) -> dict:
    """The subprocess exited 0 with no traceback; its last stdout line, parsed."""
    assert result.returncode == 0, result.stderr[-4000:]
    assert "Traceback" not in result.stderr, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_remote_app_imports_offline_with_the_contract_resources(remote_import):
    """Done when: modal_app imports without contacting Modal.

    In a subprocess whose sockets all raise, importing greenleaf_control.modal_app makes no network
    attempt and reads its SDK config from the absent file, not ~/.modal.toml. app is a modal.App,
    and train_remote's resources, read offline from the SDK (Function.info(), modal 1.6.3), are the
    contract's: a cpu request of 4.0, a memory request of 8,192 MiB, timeout 4 h = 14,400 s, and
    the volume greenleaf-runs mounted at /runs. (The hard limits are the next test's.)
    """
    tmp, result = remote_import
    report = _report(result)

    assert report["attempts"] == []
    assert report["config_path"] == str(tmp / "absent.toml")
    assert not (tmp / "absent.toml").exists()
    assert report["app_is_app"] is True
    requests = (_request(report["cpu"]), _request(report["memory_mib"]))
    assert (*requests, report["timeout"]) == (4.0, 8192, 14_400)
    assert report["volumes"] == {"/runs": "greenleaf-runs"}


def _request(value):
    """A resource as Function.info() holds it: a request, or a (request, limit) pair (a JSON list)."""
    return value[0] if isinstance(value, list) else value


def test_remote_function_caps_cpu_and_memory_at_hard_limits(remote_import):
    """Modal bills CPU and memory at max(request, usage); a (request, limit) pair also sets a hard
    limit (CPU throttled above it, memory OOM-killed). So train_remote is given
    cpu=(run_grid.CPU, run_grid.CPU) and memory=(run_grid.MEMORY_MB, run_grid.MEMORY_MB), and
    --dry's worst case (4 cores, 8 GiB, 4 h) is a true bound on what one run can bill.

    Known answer: modal 1.6.3's Function.info() holds, offline, the pair as given, so cpu is
    (4.0, 4.0) and memory_mib (8192, 8192), JSON lists here. A bare 4.0 or 8192 sets no limit.
    """
    _, result = remote_import
    report = _report(result)

    assert report["cpu"] == [4.0, 4.0]
    assert report["memory_mib"] == [8192, 8192]


def test_remote_function_runs_each_input_once_in_its_own_container(remote_import):
    """A remote run is attempted once: train_remote has retries=0, so a failed or preempted input
    isn't rerun (a rerun would meet train's refusal of its existing run dir on the volume, or bill
    twice) and the Mac records it as crashed. single_use_containers=True: a container takes one
    input and shuts down, so no run inherits another's process.

    Known answer: modal 1.6.3's Function.info() holds, offline, max_retries = the retry policy's
    retries (0 for retries=0, None when retries is unset) and single_use_containers as given.
    nonpreemptible is reported, not pinned: it stays the SDK's default (False) until Aarmen's call
    on spend.
    """
    _, result = remote_import
    report = _report(result)

    shown = {k: report[k] for k in ("max_retries", "single_use_containers", "nonpreemptible")}
    assert (report["max_retries"], report["single_use_containers"]) == (0, True), shown


def _flat(values) -> list:
    """apt_install and add_local_python_source take strings or lists of strings."""
    return [v for item in values for v in (item if isinstance(item, list) else [item])]


def test_remote_image_is_the_frozen_uv_lock_with_cpu_torch_and_the_source(remote_import):
    """The image is defined, not built: Debian slim on Python 3.12, git installed (gl-gym is a git
    dependency), then uv sync of control/'s uv.lock, frozen, from the absolute control dir (so the
    import works from any cwd), plus the greenleaf_control source (uv sync installs dependencies
    only). On linux, the image's platform, uv.lock takes torch from PyTorch's CPU index."""
    _, result = remote_import
    calls = _report(result)["calls"]

    def args_of(name: str) -> list[dict]:
        return [call["args"] for call in calls if call["name"] == name]

    [slim] = args_of("debian_slim")
    assert str(slim["python_version"]).startswith("3.12")
    assert any("git" in _flat(args["packages"]) for args in args_of("apt_install")), calls
    [uv] = args_of("uv_sync")
    assert Path(uv["uv_project_dir"]).is_absolute(), uv
    assert Path(uv["uv_project_dir"]).resolve() == CONTROL.resolve()
    assert uv["frozen"] is True
    assert (CONTROL / "uv.lock").is_file()
    sources = [
        args
        for args in args_of("add_local_python_source")
        if "greenleaf_control" in _flat(args["modules"])
    ] + [
        args
        for args in args_of("add_local_dir")
        if Path(args["local_path"]).resolve()
        in (CONTROL.resolve(), (CONTROL / "greenleaf_control").resolve())
    ]
    assert sources, calls
    cpu_torch = (
        r'\{ name = "torch", version = "[^"]+\+cpu", '
        r'source = \{ registry = "https://download\.pytorch\.org/whl/cpu" \}, '
        r"marker = \"sys_platform == 'linux'\" \}"
    )
    assert re.search(cpu_torch, (CONTROL / "uv.lock").read_text())


def test_remote_image_carries_the_configs_and_the_weather(remote_import):
    """train reads configs/ (ppo.yml, the rule-based YAMLs) and data/weather/ (hashed into the
    manifest, which must match the Mac's stamp) next to the package: rule_based.CONFIGS is
    control/configs and scenarios.WEATHER_DIR control/data/weather. The image puts the
    greenleaf_control source at /root/greenleaf_control, so the image adds control/configs at
    /root/configs and control/data/weather at /root/data/weather with add_local_dir, and train
    finds both there. (uv sync installs dependencies only, so neither comes with it.)"""
    assert rule_based.CONFIGS == CONTROL / "configs"
    assert scenarios.WEATHER_DIR == CONTROL / "data" / "weather"
    _, result = remote_import
    calls = _report(result)["calls"]

    added = {
        Path(call["args"]["local_path"]).resolve(): PurePosixPath(call["args"]["remote_path"])
        for call in calls
        if call["name"] == "add_local_dir"
    }
    assert added.get((CONTROL / "configs").resolve()) == PurePosixPath("/root/configs"), calls
    weather = (CONTROL / "data" / "weather").resolve()
    assert added.get(weather) == PurePosixPath("/root/data/weather"), calls


# train_remote's body, run offline in a fresh interpreter (sockets blocked, the SDK's config an
# absent file) through the SDK's Function.local(), which runs the decorated function here (1.6.3
# has no get_raw_f). modal_app.RUNS is a tmp dir already holding an OLDER run's manifest;
# modal_app.vol is a fake whose commit() writes ANOTHER concurrent run's manifest (what the
# volume's reload on commit exposes); greenleaf_control.train is a fake module whose main writes
# THIS run's manifest under --runs-dir, then returns or raises. Prints, per scenario, what
# train_remote returned or raised, the order of train and commit, and the argv train got.
BODY = {
    "argv": ["--method", "ppo", "--seed", "0", "--timesteps", "2048"]
    + ["--registered", "registered-0badc0de"],
    "meta": {
        "stamp": {"commit": "0" * 40, "dirty": False},
        "registered": {"row_id": "registered-0badc0de"},
    },
    "this": "registered-0badc0de/a/seed0",  # this run's run_rel
    "older": "registered-00000000/a/seed0",  # a run already on the volume
    "other": "registered-0badc0de/b/seed1",  # a concurrent run's, exposed by commit's reload
}
_OFFLINE_BODY = textwrap.dedent(
    """
    import json
    import pathlib
    import socket
    import sys
    import types

    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(repr(args)[:200])
        raise OSError("the network is blocked in this test")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked

    import greenleaf_control
    from greenleaf_control import modal_app

    tmp, body = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2])

    def write(runs, rel, who, status):
        path = runs / rel / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"who": who, "status": status}))

    def scenario(name, crash):
        runs = tmp / name
        write(runs, body["older"], "older", "finished")
        seen = {"events": [], "argv": None}

        class Volume:
            def commit(self):
                seen["events"].append("commit")
                write(runs, body["other"], "other", "finished")

        def main(argv):
            seen["events"].append("train")
            seen["argv"] = list(argv)
            runs_dir = pathlib.Path(argv[argv.index("--runs-dir") + 1])
            write(runs_dir, body["this"], "this", "crashed" if crash else "finished")
            if crash:
                raise RuntimeError("train crashed")
            return 0

        train = types.ModuleType("greenleaf_control.train")
        train.main = main
        sys.modules["greenleaf_control.train"] = train
        greenleaf_control.train = train
        modal_app.RUNS = runs
        modal_app.vol = Volume()
        try:
            # .local() runs the decorated function here, as calling it directly would
            returned = modal_app.train_remote.local(body["argv"], body["meta"], body["this"])
            error = None
        except Exception as exc:
            returned, error = None, f"{type(exc).__name__}: {exc}"
        return {**seen, "returned": returned, "error": error, "runs": str(runs)}

    print(json.dumps({
        "finished": scenario("finished", crash=False),
        "crashed": scenario("crashed", crash=True),
        "attempts": attempts,
    }))
    """
)


@pytest.fixture(scope="module")
def remote_body(tmp_path_factory) -> subprocess.CompletedProcess:
    """One offline run of train_remote's body in a subprocess, both scenarios; asserts nothing."""
    tmp = tmp_path_factory.mktemp("remote-body")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("MODAL_")}
    env["MODAL_CONFIG_PATH"] = str(tmp / "absent.toml")
    return subprocess.run(
        [sys.executable, "-c", _OFFLINE_BODY, str(tmp), json.dumps(BODY)],
        env=env,
        cwd=tmp,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_train_remote_returns_its_own_runs_manifest_after_one_commit(remote_body):
    """Regression for #19 (19b review): vol.commit() reloads the volume, so after it the volume
    shows other containers' new manifests too, and train_remote's "the one new manifest.json"
    glob raised on a finished, paid run.

    train_remote(argv, meta, run_rel) runs train with argv + --runs-dir <RUNS> + --remote
    <json of meta>, commits the volume once, after train, and returns the manifest at
    RUNS/run_rel/manifest.json, found by its path. Known answer: {"who": "this", "status":
    "finished"}, not the older run's or the concurrent one the commit exposed. Nothing reaches the
    network.
    """
    report = _report(remote_body)
    finished = report["finished"]

    assert report["attempts"] == []
    assert finished["error"] is None, finished["error"]
    assert finished["returned"] == {"who": "this", "status": "finished"}
    assert finished["events"] == ["train", "commit"]
    argv = finished["argv"]
    assert argv[: len(BODY["argv"])] == BODY["argv"]
    assert argv[argv.index("--runs-dir") + 1] == finished["runs"]
    assert json.loads(argv[argv.index("--remote") + 1]) == BODY["meta"]


def test_train_remote_commits_once_and_reraises_when_train_crashes(remote_body):
    """A crashed run's manifest and checkpoints must reach the volume too: when train raises
    RuntimeError("train crashed"), train_remote commits the volume once, after train, and the
    error propagates as it was, so the Mac's call raises and records the run as crashed."""
    report = _report(remote_body)
    crashed = report["crashed"]

    assert crashed["error"] == "RuntimeError: train crashed", crashed
    assert crashed["events"] == ["train", "commit"], crashed


@pytest.mark.slow
def test_run_local_returns_each_runs_exit_code_in_argv_order(tmp_path, monkeypatch):
    """The real _run_local, at parallel 2, runs `python -m greenleaf_control.train <argv>` for
    each argv and returns their exit codes in argvs order.

    Known answer: ["--help"] exits 0 (argparse prints help and stops) and
    ["--method", "nope", "--seed", "0", "--smoke"] exits 2 (argparse: invalid choice "nope"),
    so [0, 2]. Neither trains: train's own parser is checked first to exit 2 on the second argv,
    so a parser that accepted it fails here before anything launches. Slow: each run is a fresh
    interpreter importing train (torch, SB3).
    """
    from greenleaf_control import run_grid

    bad = ["--method", "nope", "--seed", "0", "--smoke"]
    with pytest.raises(SystemExit) as excinfo:
        train._parser().parse_args(bad)
    assert excinfo.value.code == 2
    monkeypatch.chdir(tmp_path)

    assert run_grid._run_local([["--help"], bad], 2) == [0, 2]
    assert list(tmp_path.iterdir()) == []


def test_main_takes_ledger_path_and_no_test_wrote_the_real_ledger_or_runs():
    """main(argv=None, *, ledger_path=ledger.LEDGER), as train.main: every test here passes a tmp
    ledger_path.

    This test runs last in the module and checks the real ledger's size and mtime, and the
    listing of control/runs/, are as they were when the module was collected.
    """
    from greenleaf_control import run_grid

    params = inspect.signature(run_grid.main).parameters
    assert list(params) == ["argv", "ledger_path"]
    assert params["argv"].default is None
    assert params["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["ledger_path"].default == ledger.LEDGER
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE
    assert _listing(REAL_RUNS) == REAL_RUNS_BEFORE

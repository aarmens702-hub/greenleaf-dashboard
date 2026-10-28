"""Tests for greenleaf_control.run_grid and greenleaf_control.modal_app (issue #17, part 17a).

run_grid (docs/API-CONTRACT.md):
- --dry takes the Measurement's shape as flags, before any registration exists, in ledger
  register's CLI shapes: --issue N --arms <JSON list of {name, method, config}> --seeds S [S ...]
  --timesteps T --dry [--parallel P]. It prints the run count, the worst-case CPU-hours, the
  worst-case dollars on Modal and a wall-clock estimate, and writes nothing.
- --registered <row> --backend local [--parallel 2] checks every cell of the registration, then
  hands one train argv per arm x seed (what train.main(argv) takes) to the module-level seam
  _run_local(argvs, parallel), which returns each argv's exit code in argvs order. Tests replace
  that seam, so nothing trains here.
- --backend modal refuses until issue #19 (Record Modal runs): remote runs can't record their
  ledger rows yet.

Print format pinned here (loosely, by regex on stdout):
- "<n> runs"; "<h> CPU-h" (or CPU-hours); "$<usd>" to the cent;
- the wall-clock estimate in hours, as "<x> h" (or hours): the per-run time and the local total;
- "no measured run" when the ledger has no finished run-end to measure a rate from.

Known answers (Modal prices as of 2026-10-28: $0.0000131 per physical core-second, $0.00000222
per GiB-second; a run is capped at 4 cores, 8,192 MiB = 8 GiB, 4 h = 14,400 s):
- worst case per run = 14,400 s x (4 x 0.0000131 + 8 x 0.00000222) $/s = 14,400 x 0.00007016 =
  $1.010304; 3 arms x 5 seeds = 15 runs: 15 x 4 h x 4 cores = 240 CPU-h and $15.15456 -> $15.15.

modal_app is imported only in a subprocess with the network blocked and the SDK's config pointed
at an absent file, so ~/.modal.toml is never read and nothing can reach Modal.

greenleaf_control.run_grid and modal_app are imported inside each test body, so while they are
missing each test FAILS rather than ERRORs at collection. Nothing here may write the real ledger or
control/runs/; the last test checks both are as they were when this module was collected.
"""

import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import greenleaf_control
from greenleaf_control import ledger, scenarios, train

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
    """main(argv) must refuse: a ValueError, a nonzero SystemExit or a nonzero return.

    Returns the refusal's text: the exception's message (or exit code) plus stdout and stderr.
    """
    capsys.readouterr()
    try:
        code = run_grid.main(argv, ledger_path=path)
    except SystemExit as exc:
        assert exc.code not in (0, None), "a refusal must exit nonzero"
        text = str(exc.code)
    except ValueError as exc:
        text = str(exc)
    else:
        assert code not in (0, None), f"main returned {code!r}; expected a refusal"
        text = ""
    captured = capsys.readouterr()
    return text + captured.out + captured.err


def test_remote_resources_and_prices_are_module_constants():
    """One remote run's caps and Modal's prices are module constants, read by --dry and by
    modal_app: 4.0 physical cores, 8,192 MiB, 4 h = 14,400 s; $0.0000131 per core-second and
    $0.00000222 per GiB-second, as of 2026-10-28 (modal.com/pricing; one core is 2 vCPU)."""
    from greenleaf_control import run_grid

    assert (run_grid.CPU, run_grid.MEMORY_MB, run_grid.TIMEOUT_S) == (4.0, 8192, 14_400)
    assert (run_grid.USD_PER_CORE_S, run_grid.USD_PER_GIB_S) == (0.0000131, 0.00000222)
    assert run_grid.PRICES_AS_OF == "2026-10-28"


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


@pytest.mark.parametrize(("over", "row_id", "named"), REGISTERED_REFUSALS)
def test_registered_refuses_before_launching_anything(
    over, row_id, named, tmp_path, monkeypatch, capsys
):
    """run_grid checks every cell before it launches any: an unknown row, an arm whose method isn't
    in envs.METHODS, an arm config key that isn't a train flag, or a 2012 TEST split among the
    registration's scenarios (PROTOCOL §8) each refuse naming what's wrong. The bad arm is the
    second, after a good one. _run_local is never called and the ledger is byte-identical."""
    from greenleaf_control import run_grid

    path, row = _registered(tmp_path, **over)
    monkeypatch.setattr(run_grid, "_run_local", _no_launch, raising=False)
    before = path.read_bytes()

    argv = ["--registered", row_id or row["row_id"], "--backend", "local"]
    assert named in _refused(run_grid, argv, path, capsys)
    assert path.read_bytes() == before


def test_remote_backend_refuses_until_issue_19_without_importing_the_remote_app(
    tmp_path, monkeypatch, capsys
):
    """--backend modal refuses, nonzero, naming issue #19 (Record Modal runs): a remote run can't
    record its run and run-end rows yet. It refuses before building anything: run_grid is imported
    afresh here, and neither that import nor the refusal imports greenleaf_control.modal_app.
    Nothing is launched and the ledger is byte-identical."""
    path, row = _registered(tmp_path)
    for name in ("run_grid", "modal_app"):
        monkeypatch.delitem(sys.modules, f"greenleaf_control.{name}", raising=False)
        monkeypatch.delattr(greenleaf_control, name, raising=False)
    import greenleaf_control.run_grid as run_grid

    monkeypatch.setattr(run_grid, "_run_local", _no_launch, raising=False)
    before = path.read_bytes()

    argv = ["--registered", row["row_id"], "--backend", "modal"]
    assert "#19" in _refused(run_grid, argv, path, capsys)
    assert "greenleaf_control.modal_app" not in sys.modules
    assert path.read_bytes() == before


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

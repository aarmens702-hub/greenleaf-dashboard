"""The grid runner: every cell of one registration, run locally or on Modal, and the worst case
before a registration exists (docs/API-CONTRACT.md, PROTOCOL §3).

- --dry takes the Measurement's shape, in ledger register's CLI shapes: --issue N --arms <JSON list
  of {name, method, config}> --seeds S [S ...] --timesteps T [--parallel P]. It prints the run
  count, the worst case (every run at its hard limits, 4 cores and 8 GiB, for the full 4 h timeout:
  CPU-hours, and dollars on Modal) and a wall-clock estimate from the ledger's newest finished
  run-end. It writes nothing.
- --registered <row> --backend local [--parallel 2] checks every cell of the registration, then
  runs `python -m greenleaf_control.train` once per arm x seed, parallel at a time. Each train run
  records its own run and run-end rows; run_grid writes no ledger row.
- --registered <row> --backend modal [--parallel 2] makes the same checks, stamps the code
  (refusing a dirty tree) and refuses any cell whose local run dir or run row exists, all before
  any call. Then, in one remote session, parallel cells at a time, it records each cell's run row,
  calls modal_app.train_remote and records its run-end (crashed if the call raised anything): the
  Mac is the only ledger writer, and a run-end's cost is its wall clock at the hard limits' price.
  On a Ctrl-C, the queued cells are never launched (no rows; listed), the in-flight ones end with
  the session as crashed, and run_grid re-raises the KeyboardInterrupt (the real session
  suppresses it).
"""

import argparse
import contextlib
import json
import math
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from greenleaf_control import ledger, manifest, scenarios

# One remote run's hard limits (4 cores, 8 GiB) for the full 4 h timeout: modal_app's train_remote
# sets CPU and memory as (request, limit) pairs. Modal bills max(request, usage), and the limits
# make this the true worst case. Modal's prices: modal.com/pricing, fetched 2026-10-28; one core =
# 2 vCPU.
CPU = 4.0
MEMORY_MB = 8192
TIMEOUT_S = 4 * 3600
USD_PER_CORE_S = 0.0000131
USD_PER_GIB_S = 0.00000222
PRICES_AS_OF = "2026-10-28"
# One remote run at its hard limits, $ per second of wall clock: $0.00007016.
USD_PER_S = CPU * USD_PER_CORE_S + MEMORY_MB / 1024 * USD_PER_GIB_S
# train.RUNS, where `modal volume get greenleaf-runs <row> runs` from control/ lands a remote run;
# stated here, not imported, because train imports torch and gl_gym.
RUNS = Path(__file__).resolve().parents[1] / "runs"
# The train flags an arm's config may set, by their snake_case dests, as train's registration check
# reads the config; the arm gives --method, the registration --seed and --timesteps.
CONFIG_KEYS = ("penalty_weight", "budget_frac", "season_length", "n_envs", "rb")


def _dry(args: argparse.Namespace, ledger_path: Path) -> int:
    """Print the run count, the worst case and the wall-clock estimate; write nothing."""
    runs = len(args.arms) * len(args.seeds)
    hours = TIMEOUT_S / 3600
    cpu_hours = runs * hours * CPU
    usd = runs * TIMEOUT_S * USD_PER_S
    print(
        f"issue #{args.issue}: arms x seeds = {len(args.arms)} x {len(args.seeds)} = {runs} runs"
        f" of {args.timesteps:,} timesteps"
    )
    print(
        f"worst case, hard limits ({CPU:g} cores, {MEMORY_MB / 1024:g} GiB) for the full"
        f" {hours:g} h timeout: {cpu_hours:,.1f} CPU-h, ${usd:,.2f} on Modal"
    )
    print(
        "  Modal bills max(request, usage), and the limits make this the true worst case"
        f" (prices as of {PRICES_AS_OF})"
    )
    ends = ledger.rows(kind="run-end", path=ledger_path)
    # A tune_rb run-end trains nothing (timesteps 0), so it has no per-step rate.
    finished = [end for end in ends if end["status"] == "finished" and end["timesteps"] > 0]
    if not finished:
        print(
            f"wall clock: no measured run (no finished run-end with timesteps > 0 in {ledger_path})"
        )
        return 0
    end = finished[-1]  # the ledger is append-only, so the last is the newest
    rate = end["wall_s"] / end["timesteps"]
    per_run = rate * args.timesteps / 3600
    waves = math.ceil(runs / args.parallel)
    print(
        f"wall clock: {per_run:.2f} h per run ({rate:.4g} s/step, run-end {end['row_id']});"
        f" {waves * per_run:.2f} h locally, {waves} waves at --parallel {args.parallel}"
    )
    return 0


def _cells(row_id: str, ledger_path: Path) -> tuple[dict, list[tuple[dict, int, list[str]]]]:
    """The registered row, and (arm, seed, train argv) for every cell of it, arms x seeds; refuses
    the whole registration before any cell is launched."""
    from greenleaf_control import envs  # here, not at the top: envs imports gl_gym

    found = [r for r in ledger.rows(kind="registered", path=ledger_path) if r["row_id"] == row_id]
    if not found:
        raise ValueError(f"--registered {row_id}: no registered row with that id in {ledger_path}")
    [row] = found
    sealed = [s for s in row["scenarios"] if s in scenarios.TEST]
    if sealed:
        raise ValueError(
            f"{row_id}: scenarios {sealed} are the sealed 2012 TEST weather (PROTOCOL §8)"
        )
    for arm in row["arms"]:
        if arm["method"] not in envs.METHODS:
            raise ValueError(
                f"{row_id}: arm {arm['name']}'s method {arm['method']!r} is not one of {envs.METHODS}"
            )
        unknown = sorted(set(arm["config"]) - set(CONFIG_KEYS))
        if unknown:
            raise ValueError(
                f"{row_id}: arm {arm['name']}'s config keys {unknown} are not train flags; a config"
                f" names {CONFIG_KEYS} (train's snake_case dests)"
            )
    return row, [
        (
            arm,
            seed,
            [
                *("--method", arm["method"]),
                *(
                    x
                    for k, v in arm["config"].items()
                    for x in (f"--{k.replace('_', '-')}", str(v))
                ),
                *("--seed", str(seed)),
                *("--timesteps", str(row["timesteps"])),
                *("--registered", row["row_id"]),
            ],
        )
        for arm in row["arms"]
        for seed in row["seeds"]
    ]


def _run_local(argvs: list[list[str]], parallel: int) -> list[int]:
    """Run `python -m greenleaf_control.train <argv>` for each argv, parallel at a time; returns
    each run's exit code, in argvs order."""

    def run(argv: list[str]) -> int:
        command = [sys.executable, "-m", "greenleaf_control.train", *argv]
        print("launch:", " ".join(command[1:]), flush=True)
        return subprocess.run(command, check=False).returncode

    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(run, argvs))


@contextlib.contextmanager
def _run_remote():
    """One remote session: yields call(argv, meta, run_rel) -> the finished manifest, as a dict.

    On a KeyboardInterrupt the SDK disconnects and returns, so this suppresses it; _remote
    re-raises it."""
    from greenleaf_control import modal_app  # here, so only a real session imports the SDK

    with modal_app.app.run():
        yield modal_app.train_remote.remote


def _remote(
    row: dict, cells: list[tuple[dict, int, list[str]]], parallel: int, ledger_path: Path
) -> list[str | None]:
    """Run each cell remotely, parallel at a time, in one session, writing its run row before the
    call and its run-end after it to ledger_path; returns each cell's error (None if it finished),
    in cells order. Refuses a dirty tree, any existing local run dir and any cell that already has
    a run row before any call. On a Ctrl-C (or any error) while it waits, the queued cells never
    launch, the in-flight ones end with the session, and the error re-raises: a KeyboardInterrupt
    by run_grid itself, once the session and the pool close, since the real session suppresses
    it."""
    stamp = manifest.code_stamp(smoke=False)
    # Each cell's run dir under the volume's /runs, and under RUNS once downloaded.
    run_rels = [f"{row['row_id']}/{arm['name']}/seed{seed}" for arm, seed, _ in cells]
    existing = [str(RUNS / rel) for rel in run_rels if (RUNS / rel).exists()]
    if existing:
        raise FileExistsError(
            f"{row['row_id']}: run dirs {existing} already exist; a run dir is written once, never"
            " resumed (PROTOCOL §5)"
        )
    # A remote run's dir is only on the volume until it's downloaded, so its run row is the check.
    ran = {
        (r["arm"], r["seed"])
        for r in ledger.rows(kind="run", path=ledger_path)
        if r["registered"] == row["row_id"]
    }
    again = [
        f"arm {arm['name']} seed {seed}" for arm, seed, _ in cells if (arm["name"], seed) in ran
    ]
    if again:
        raise ValueError(
            f"{row['row_id']}: already has run rows in {ledger_path} for: {', '.join(again)}; a"
            " run is launched once, never resumed (PROTOCOL §5)"
        )
    meta = {"stamp": stamp, "registered": row}
    lock = threading.Lock()  # the ledger's checks read the file, then append

    def launch(cell: tuple[dict, int, list[str]], run_rel: str, call) -> None:
        arm, seed, argv = cell
        with lock:
            run = ledger.run_start(
                registered=row["row_id"],
                arm=arm["name"],
                seed=seed,
                run_dir=str(RUNS / run_rel),
                path=ledger_path,
            )
        print("remote:", " ".join(argv), flush=True)
        end = {
            "status": "crashed",
            "timesteps": None,
            "solver_failures": None,
            "diagnostics": dict.fromkeys(ledger.DIAGNOSTIC_KEYS),
        }
        start = time.monotonic()
        try:
            result = call(argv, meta, run_rel)
            end = {k: result[k] for k in ("status", "timesteps", "solver_failures", "diagnostics")}
        finally:  # anything the call raises, BaseException included, still closes the run
            wall_s = time.monotonic() - start
            with lock:
                ledger.run_end(
                    run=run["row_id"],
                    **end,
                    wall_s=wall_s,
                    cost_usd=wall_s * USD_PER_S,
                    path=ledger_path,
                )

    # The pool outlives the session: closing the session ends the in-flight calls, whose cells
    # then get their crashed run-ends before the pool shuts down. The real session swallows a
    # KeyboardInterrupt (the SDK disconnects and returns), so it's noted here and re-raised below.
    interrupted = False
    with ThreadPoolExecutor(parallel) as pool, _run_remote() as call:
        futures = []
        try:
            for cell, run_rel in zip(cells, run_rels, strict=True):
                futures.append(pool.submit(launch, cell, run_rel, call))
            wait(futures)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:  # after a Ctrl-C (or any error) here, no queued cell launches: it gets no rows
            for i, (arm, seed, _) in enumerate(cells):
                if i >= len(futures) or futures[i].cancel():  # unsubmitted, or still queued
                    print(
                        f"not launched: arm {arm['name']} ({arm['method']}) seed {seed}",
                        file=sys.stderr,
                    )
    if interrupted:  # the session and the pool are closed; a cancelled future has no exception
        raise KeyboardInterrupt
    return [None if f.exception() is None else repr(f.exception()) for f in futures]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.run_grid")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--registered", metavar="ROW", help="run every cell of this registered row")
    mode.add_argument(
        "--dry", action="store_true", help="print the worst case and wall clock; writes nothing"
    )
    p.add_argument(
        "--backend", choices=["local", "modal"], help="--registered: where the runs train"
    )
    p.add_argument("--parallel", type=int, default=2, help="runs at a time (default 2)")
    p.add_argument("--issue", type=int, help="--dry: the issue")
    p.add_argument("--arms", type=json.loads, help="--dry: JSON list of {name, method, config}")
    p.add_argument("--seeds", type=int, nargs="+", help="--dry: the seeds")
    p.add_argument("--timesteps", type=int, help="--dry: timesteps per run")
    return p


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    """ledger_path is where the registration (or, with --dry, the newest run-end) is read, and
    where --backend modal writes each run's rows; local train runs record theirs in train's own
    default ledger."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.parallel < 1:
        parser.error(f"--parallel must be at least 1, got {args.parallel}")
    shape = {
        "--issue": args.issue,
        "--arms": args.arms,
        "--seeds": args.seeds,
        "--timesteps": args.timesteps,
    }
    if args.dry:
        missing = [flag for flag, value in shape.items() if value is None]
        if missing:
            parser.error(f"--dry needs the Measurement's shape; missing {', '.join(missing)}")
        return _dry(args, ledger_path)
    given = [flag for flag, value in shape.items() if value is not None]
    if given:
        parser.error(f"{', '.join(given)}: --registered reads these from its row")
    if args.backend is None:
        parser.error("--registered needs --backend {local,modal}")
    row, cells = _cells(args.registered, ledger_path)
    if args.backend == "local":
        codes = _run_local([cell_argv for _, _, cell_argv in cells], args.parallel)
        errors = [None if code == 0 else f"exit {code}" for code in codes]
    else:
        errors = _remote(row, cells, args.parallel, ledger_path)
    failed = [
        (arm, seed, error)
        for (arm, seed, _), error in zip(cells, errors, strict=True)
        if error is not None
    ]
    for arm, seed, error in failed:
        print(f"failed: arm {arm['name']} ({arm['method']}) seed {seed}, {error}", file=sys.stderr)
    print(f"{len(cells) - len(failed)} of {len(cells)} runs finished")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

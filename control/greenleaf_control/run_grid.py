"""The grid runner: every cell of one registration, run locally, and the worst case before a
registration exists (docs/API-CONTRACT.md, PROTOCOL §3).

- --dry takes the Measurement's shape, in ledger register's CLI shapes: --issue N --arms <JSON list
  of {name, method, config}> --seeds S [S ...] --timesteps T [--parallel P]. It prints the run
  count, the worst case (every run at its hard limits, 4 cores and 8 GiB, for the full 4 h timeout:
  CPU-hours, and dollars on Modal) and a wall-clock estimate from the ledger's newest finished
  run-end. It writes nothing.
- --registered <row> --backend local [--parallel 2] checks every cell of the registration, then
  runs `python -m greenleaf_control.train` once per arm x seed, parallel at a time. Each train run
  records its own run and run-end rows; run_grid writes no ledger row.
- --backend modal refuses until issue #19 (Record Modal runs): a remote run can't record its rows
  yet.
"""

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from greenleaf_control import ledger, scenarios

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
# The train flags an arm's config may set, by their snake_case dests, as train's registration check
# reads the config; the arm gives --method, the registration --seed and --timesteps.
CONFIG_KEYS = ("penalty_weight", "budget_frac", "season_length", "n_envs", "rb")


def _dry(args: argparse.Namespace, ledger_path: Path) -> int:
    """Print the run count, the worst case and the wall-clock estimate; write nothing."""
    runs = len(args.arms) * len(args.seeds)
    hours = TIMEOUT_S / 3600
    cpu_hours = runs * hours * CPU
    usd = runs * TIMEOUT_S * (CPU * USD_PER_CORE_S + MEMORY_MB / 1024 * USD_PER_GIB_S)
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


def _cells(row_id: str, ledger_path: Path) -> list[tuple[dict, int, list[str]]]:
    """(arm, seed, train argv) for every cell of the registration, arms x seeds; refuses the whole
    registration before any cell is launched."""
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
    return [
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m greenleaf_control.run_grid")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--registered", metavar="ROW", help="run every cell of this registered row")
    mode.add_argument(
        "--dry", action="store_true", help="print the worst case and wall clock; writes nothing"
    )
    p.add_argument(
        "--backend", choices=["local", "modal"], help="--registered: where (modal waits for #19)"
    )
    p.add_argument("--parallel", type=int, default=2, help="local runs at a time (default 2)")
    p.add_argument("--issue", type=int, help="--dry: the issue")
    p.add_argument("--arms", type=json.loads, help="--dry: JSON list of {name, method, config}")
    p.add_argument("--seeds", type=int, nargs="+", help="--dry: the seeds")
    p.add_argument("--timesteps", type=int, help="--dry: timesteps per run")
    return p


def main(argv: list[str] | None = None, *, ledger_path: Path = ledger.LEDGER) -> int:
    """ledger_path is where the registration (or, with --dry, the newest run-end) is read; the
    train runs record their rows in train's own default ledger."""
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
    if args.backend == "modal":
        parser.error(
            "--backend modal waits for issue #19 (Record Modal runs: ledger rows, fingerprints and"
            " cost from the Mac): a remote run can't record its run and run-end rows yet"
        )
    cells = _cells(args.registered, ledger_path)
    codes = _run_local([cell_argv for _, _, cell_argv in cells], args.parallel)
    failed = [
        (arm, seed, code) for (arm, seed, _), code in zip(cells, codes, strict=True) if code != 0
    ]
    for arm, seed, code in failed:
        print(
            f"failed: arm {arm['name']} ({arm['method']}) seed {seed}, exit {code}", file=sys.stderr
        )
    print(f"{len(cells) - len(failed)} of {len(cells)} runs finished")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

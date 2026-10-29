"""The experiment ledger: the only writer of ledger/ledger.jsonl, one JSON row per line, appended.

Every check runs before a byte is written, so a refused row leaves the file as it was.
"""

import argparse
import json
import os
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path

from greenleaf_control import manifest

# Built from this file, not manifest.REPO_ROOT, so patching REPO_ROOT can't redirect writes.
LEDGER = Path(__file__).resolve().parents[1] / "ledger" / "ledger.jsonl"
SCHEMA_VERSION = 1

BASE_FIELDS = {"schema_version", "row_id", "kind", "at", "issue", "fingerprints"}
KIND_FIELDS = {
    "registered": {
        "hypothesis",
        "arms",
        "seeds",
        "timesteps",
        "scenarios",
        "primary",
        "guards",
        "rule",
        "budget",
        "owner_ok",
    },
    "run": {"registered", "arm", "seed", "run_dir", "status"},
    "run-end": {
        "run",
        "status",
        "wall_s",
        "timesteps",
        "solver_failures",
        "cost_usd",
        "diagnostics",
    },
    "result": {"registered", "per_arm", "selected"},
    "decision": {"result", "decision", "note"},
    "final-registered": {"checkpoints", "baselines", "approved_by", "approved_at"},
    "final-result": {"final-registered", "per_split"},
}
# A linked kind -> the field naming its parent, which is also the parent's kind.
LINKS = {
    "run": "registered",
    "run-end": "run",
    "result": "registered",
    "decision": "result",
    "final-result": "final-registered",
}
ARM_KEYS = {"name", "method", "config"}
BUDGET_KEYS = {"cpu_hours", "usd", "backend"}
PER_ARM_KEYS = {"iqm", "ci_low", "ci_high", "guards", "crashed", "invalid_episodes", "spend_usd"}
DIAGNOSTIC_KEYS = {"min_approx_kl", "final_entropy", "final_explained_variance"}
DECISIONS = {"adopted", "dropped"}
# A scored season's numeric keys, what a primary or guard may name: metrics.episode_metrics' keys
# but valid, plus evaluate's paper_reward_sum. Written out so the ledger stays stdlib-only; a test
# pins it to a real episode_metrics output, and selection scores with it.
RESULT_KEYS = (
    "profit_eur_m2",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "pct_in_bounds_co2",
    "pct_in_bounds_temp",
    "pct_in_bounds_rh",
    "yield_kg_m2",
    "heat_cost",
    "elec_cost",
    "co2_cost",
    "paper_return",
    "solver_failures",
    "steps",
    "paper_reward_sum",
)

# One short detail per kind for show, naming the parent row first.
SUMMARY = {
    "registered": lambda r: (
        f"arms {','.join(a['name'] for a in r['arms'])} seeds {r['seeds']}: {r['hypothesis']}"
    ),
    "run": lambda r: f"{r['registered']} {r['arm']} seed {r['seed']} {r['status']}",
    "run-end": lambda r: f"{r['run']} {r['status']} {r['wall_s']}s cost_usd {r['cost_usd']}",
    "result": lambda r: (
        f"{r['registered']} iqm " + " ".join(f"{n}={v['iqm']}" for n, v in r["per_arm"].items())
    ),
    "decision": lambda r: f"{r['result']} {r['decision']}: {r['note']}",
    "final-registered": lambda r: f"approved_by {r['approved_by']} at {r['approved_at']}",
    "final-result": lambda r: f"{r['final-registered']} splits {','.join(r['per_split'])}",
}


def _new_row(kind: str, issue: int, **fields) -> dict:
    """Stamp a row; fingerprints are read now, so a row records the scorer as it was written."""
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": f"{kind}-{secrets.token_hex(4)}",
        "kind": kind,
        "at": datetime.now(UTC).isoformat(),
        "issue": issue,
        "fingerprints": manifest.fingerprints(),
        **fields,
    }


def _need_keys(what: str, value: object, keys: set[str]) -> None:
    """Raise unless value is a dict with exactly keys, naming what's missing or extra."""
    have = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or have != keys:
        raise ValueError(
            f"{what} needs exactly {sorted(keys)}: "
            f"missing {sorted(keys - have)}, extra {sorted(have - keys)}"
        )


def _find(existing: list[dict], row_id: str, kind: str) -> dict:
    """The row a link names; it must exist and be of the right kind."""
    for row in existing:
        if row["row_id"] == row_id and row["kind"] == kind:
            return row
    raise ValueError(f"{row_id!r} names no {kind} row in the ledger")


def _run_ends(existing: list[dict], registered: str) -> list[tuple[dict, dict]]:
    """(run, run-end) pairs for one registration's runs; another registration's don't count."""
    runs = {
        r["row_id"]: r for r in existing if r["kind"] == "run" and r["registered"] == registered
    }
    return [(runs[r["run"]], r) for r in existing if r["kind"] == "run-end" and r["run"] in runs]


def _arm_tally(ends: list[tuple[dict, dict]], arm: str) -> dict:
    """One arm's crashed count and spend_usd from its registration's run-ends.

    record_result fills a result with it and _check verifies one with it: the same sum, same order.
    """
    mine = [end for run, end in ends if run["arm"] == arm]
    return {
        "crashed": sum(end["status"] == "crashed" for end in mine),
        "spend_usd": sum([end["cost_usd"] or 0.0 for end in mine], 0.0),
    }


def _check(row: dict, path: Path) -> str:
    """Validate row and run the ledger's refusals against path; return its JSON line."""
    kind = row.get("kind")
    if kind not in KIND_FIELDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {list(KIND_FIELDS)}")
    _need_keys(f"a {kind} row", row, BASE_FIELDS | KIND_FIELDS[kind])
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}, got {row['schema_version']!r}")
    if type(row["issue"]) is not int:
        raise ValueError(f"issue must be an int, got {row['issue']!r}")
    row_id = row["row_id"]
    if not isinstance(row_id, str) or not re.fullmatch(rf"{re.escape(kind)}-[0-9a-f]{{8}}", row_id):
        raise ValueError(f"row_id must be {kind}-<8 lowercase hex>, got {row_id!r}")
    existing = rows(path=path)
    if any(r["row_id"] == row_id for r in existing):
        raise ValueError(f"row_id {row_id} is already in the ledger")
    parent = _find(existing, row[LINKS[kind]], LINKS[kind]) if kind in LINKS else None

    if kind == "registered":
        for arm in row["arms"]:
            _need_keys("an arm", arm, ARM_KEYS)
        _need_keys("budget", row["budget"], BUDGET_KEYS)
        for key in [row["primary"], *row["guards"]]:
            if key not in RESULT_KEYS:
                raise ValueError(
                    f"primary or guard {key!r} is not a scored result key;"
                    f" expected one of {RESULT_KEYS}"
                )
        earlier = [r for r in existing if r["kind"] == "registered" and r["issue"] == row["issue"]]
        if len(earlier) >= 2 and row["owner_ok"] is not True:
            raise ValueError(
                f"a third registration on issue {row['issue']} needs owner_ok=True (--owner-ok)"
            )
    elif kind == "run" and row["status"] != "running":
        raise ValueError(f"a run row's status is 'running', got {row['status']!r}")
    elif kind == "run-end":
        if row["status"] not in manifest.STATUSES:
            raise ValueError(
                f"status must be one of {sorted(manifest.STATUSES)}, got {row['status']!r}"
            )
        _need_keys("diagnostics", row["diagnostics"], DIAGNOSTIC_KEYS)
    elif kind == "result":
        names = [arm["name"] for arm in parent["arms"]]
        _need_keys("per_arm", row["per_arm"], set(names))
        for name, numbers in row["per_arm"].items():
            _need_keys(f"per_arm[{name!r}]", numbers, PER_ARM_KEYS)
        ends = _run_ends(existing, parent["row_id"])
        closed = {run["row_id"] for run, _ in ends}
        unclosed = [
            f"run {r['row_id']} (arm {r['arm']}, seed {r['seed']})"
            for r in existing
            if r["kind"] == "run"
            and r["registered"] == parent["row_id"]
            and r["row_id"] not in closed
        ]
        if unclosed:
            raise ValueError(
                f"no run-end for {', '.join(unclosed)}; close each with run_end(status='crashed')"
            )
        ended = {(run["arm"], run["seed"]) for run, _ in ends}
        missing = [f"{n}/seed {s}" for n in names for s in parent["seeds"] if (n, s) not in ended]
        if missing:
            raise ValueError(f"no run-end yet for {', '.join(missing)}; a result needs them all")
        for name, numbers in row["per_arm"].items():
            tally = _arm_tally(ends, name)
            claimed = {key: numbers[key] for key in tally}
            if claimed != tally:
                raise ValueError(f"per_arm[{name!r}] claims {claimed}; its run-ends give {tally}")
    elif kind == "decision" and row["decision"] not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}, got {row['decision']!r}")
    elif kind == "final-registered":
        approved_by = row["approved_by"]
        if not isinstance(approved_by, str) or not approved_by.strip():
            raise ValueError("final-registered needs approved_by: the owner who approved the seal")
    elif kind == "final-result" and any(r["kind"] == "final-result" for r in existing):
        raise ValueError("a final-result already exists: the seal is spent")
    # allow_nan=False: a NaN or inf raises ValueError here, before the file is opened.
    return json.dumps(row, sort_keys=True, allow_nan=False)


def append(row: dict, path: Path = LEDGER) -> None:
    """Validate, refuse, then write one line; never rewrite, truncate or rename the file."""
    line = _check(row, path)
    with path.open("a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def rows(kind: str | None = None, issue: int | None = None, path: Path = LEDGER) -> list[dict]:
    """Every row in file order, filtered by kind and issue; a missing file has none."""
    if not path.exists():
        return []
    found = [json.loads(line) for line in path.read_text().splitlines()]
    return [
        r
        for r in found
        if (kind is None or r["kind"] == kind) and (issue is None or r["issue"] == issue)
    ]


def _append_linked(kind: str, link: str, path: Path, **fields) -> dict:
    """Stamp and append a row naming its parent; it inherits the parent's issue."""
    issue = _find(rows(path=path), link, LINKS[kind])["issue"]
    row = _new_row(kind, issue, **{LINKS[kind]: link}, **fields)
    append(row, path)
    return row


def register(
    *,
    issue: int,
    hypothesis: str,
    arms: list[dict],
    seeds: list[int],
    timesteps: int,
    scenarios: list[str],
    primary: str,
    guards: list[str],
    rule: str,
    budget: dict,
    owner_ok: bool = False,
    dry: bool = False,
    path: Path = LEDGER,
) -> dict:
    """Pre-register an experiment before its first run; dry checks it all and writes nothing."""
    row = _new_row(
        "registered",
        issue,
        hypothesis=hypothesis,
        arms=arms,
        seeds=seeds,
        timesteps=timesteps,
        scenarios=scenarios,
        primary=primary,
        guards=guards,
        rule=rule,
        budget=budget,
        owner_ok=owner_ok,
    )
    if dry:
        _check(row, path)
    else:
        append(row, path)
    return row


def run_start(*, registered: str, arm: str, seed: int, run_dir: str, path: Path = LEDGER) -> dict:
    return _append_linked(
        "run", registered, path, arm=arm, seed=seed, run_dir=run_dir, status="running"
    )


def run_end(
    *,
    run: str,
    status: str,
    wall_s: float,
    timesteps: int,
    solver_failures: int,
    cost_usd: float | None,
    diagnostics: dict,
    path: Path = LEDGER,
) -> dict:
    return _append_linked(
        "run-end",
        run,
        path,
        status=status,
        wall_s=wall_s,
        timesteps=timesteps,
        solver_failures=solver_failures,
        cost_usd=cost_usd,
        diagnostics=diagnostics,
    )


def record_result(*, registered: str, per_arm: dict, selected: dict, path: Path = LEDGER) -> dict:
    """Record per-arm numbers; crashed and spend_usd come from run-end rows, so none can hide."""
    if any({"crashed", "spend_usd"} & set(numbers) for numbers in per_arm.values()):
        raise ValueError("per_arm can't carry crashed or spend_usd: the ledger counts them itself")
    ends = _run_ends(rows(path=path), registered)
    per_arm = {name: {**numbers, **_arm_tally(ends, name)} for name, numbers in per_arm.items()}
    return _append_linked("result", registered, path, per_arm=per_arm, selected=selected)


def decide(*, result: str, decision: str, note: str, path: Path = LEDGER) -> dict:
    return _append_linked("decision", result, path, decision=decision, note=note)


def final_register(
    *, issue: int, checkpoints: dict, baselines: list[str], approved_by: str, path: Path = LEDGER
) -> dict:
    """Register the one-time final evaluation; it needs the owner's name."""
    row = _new_row(
        "final-registered",
        issue,
        checkpoints=checkpoints,
        baselines=baselines,
        approved_by=approved_by,
        approved_at=datetime.now(UTC).isoformat(),
    )
    append(row, path)
    return row


def final_result(*, final_registered: str, per_split: dict, path: Path = LEDGER) -> dict:
    return _append_linked("final-result", final_registered, path, per_split=per_split)


def show(*, issue: int | None = None, spend: bool = False, path: Path = LEDGER) -> None:
    """Print one line per row, or with spend the USD the run-end rows recorded."""
    if spend:
        total = sum([r["cost_usd"] or 0.0 for r in rows(kind="run-end", issue=issue, path=path)])
        print(f"spend_usd {total:.2f}")
        return
    for r in rows(issue=issue, path=path):
        summary = " ".join(SUMMARY[r["kind"]](r).split())  # a multi-line note stays on one line
        print(f"{r['row_id']}  #{r['issue']}  {r['kind']}  {summary}")


def main(argv: list[str] | None = None, path: Path = LEDGER) -> int:
    parser = argparse.ArgumentParser(prog="python -m greenleaf_control.ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("register", help="pre-register an experiment")
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument(
        "--arms", type=json.loads, required=True, help="JSON list of {name, method, config}"
    )
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--timesteps", type=int, required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--primary", required=True)
    p.add_argument("--guards", nargs="+", required=True)
    p.add_argument("--rule", required=True)
    p.add_argument("--cpu-hours", type=float, required=True)
    p.add_argument("--usd", type=float, required=True)
    p.add_argument("--backend", choices=["local", "modal"], required=True)
    p.add_argument("--owner-ok", action="store_true")
    p.add_argument("--dry", action="store_true")

    p = sub.add_parser("result", help="record a registration's numbers")
    p.add_argument("--registered", required=True)
    p.add_argument("--numbers", type=Path, required=True, help="JSON file: {per_arm, selected}")

    p = sub.add_parser("decide", help="adopt or drop a result")
    p.add_argument("--result", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--adopted", dest="decision", action="store_const", const="adopted")
    group.add_argument("--dropped", dest="decision", action="store_const", const="dropped")
    p.add_argument("--note", required=True)

    p = sub.add_parser("show", help="print the rows, or the spend")
    p.add_argument("--issue", type=int)
    p.add_argument("--spend", action="store_true")

    p = sub.add_parser("final-register", help="register the one-time final evaluation")
    p.add_argument("--issue", type=int, required=True)
    p.add_argument("--checkpoints", type=json.loads, required=True)
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument("--approved-by")  # not required, so the ledger's own refusal fires

    # Each flag's dest is the API's parameter name, so the CLI passes through the same checks.
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "register":
        budget = {key: args.pop(key) for key in ("cpu_hours", "usd", "backend")}
        row = register(**args, budget=budget, path=path)
        print(json.dumps(row, indent=2, sort_keys=True) if args["dry"] else row["row_id"])
    elif command == "result":
        numbers = json.loads(args["numbers"].read_text())  # {"per_arm": ..., "selected": ...}
        print(record_result(registered=args["registered"], **numbers, path=path)["row_id"])
    elif command == "decide":
        print(decide(**args, path=path)["row_id"])
    elif command == "show":
        show(**args, path=path)
    else:
        print(final_register(**args, path=path)["row_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for greenleaf_control.ledger (issue #3): the append-only experiment ledger and its refusals.

Every test passes path= under tmp_path. The defaults bind to the real control/ledger/ledger.jsonl,
which no test may write; the last test in this module checks that.

Issue #71 makes each test year its own seal: a final-registered row may carry an int seal (absent
means 2012, as every row before #71), each seal is spent once by a final-result under one of its
final-registered rows, and a final-result's per_split holds exactly its seal's splits.

Issue #72 grades the 2013 final at both price regimes: a 2013 final-result's per_split is {split:
{regime: {arm: entry}}}, each split keyed by exactly ledger.PRICE_REGIMES ("sim", "agc2020"), the
arm checks run per (split, regime), and a flat 2013 per_split (#71's shape) is refused. The 2012
per_split stays flat. A final may register rb_tuned_agc2020 (the agc2020 tune's rules) as a baseline.
"""

import copy
import inspect
import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# manifest already exists (issue #2), so importing it here can't ERROR collection. ledger is
# imported inside each test body, so while it is missing each test FAILS rather than ERRORs.
from greenleaf_control import manifest

# The real ledger, stat-ed when this module is collected, before any test in it runs.
REAL_LEDGER = Path(__file__).resolve().parents[1] / "ledger" / "ledger.jsonl"


def _stat(path: Path) -> tuple[int, int] | None:
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


REAL_LEDGER_BEFORE = _stat(REAL_LEDGER)

# docs/API-CONTRACT.md: the fields every row has, and each kind's extra fields, in chain order.
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
# Every public function that takes path=, all defaulting to the real ledger.
PATH_FUNCTIONS = [
    "register",
    "run_start",
    "run_end",
    "record_result",
    "decide",
    "final_register",
    "final_result",
    "rows",
    "append",
    "show",
    "main",
]
# A registration's fields other than issue, arms and seeds. "validation" is a split name; the
# primary and guards are scored result keys (selection.RESULT_KEYS), which register checks.
REGISTRATION = {
    "hypothesis": "residual PPO beats plain PPO on validation profit",
    "timesteps": 20_000,
    "scenarios": ["validation"],
    "primary": "profit_eur_m2",
    "guards": ["viol_hours_temp"],
    "rule": "adopt if the profit IQM CI excludes 0",
    "budget": {"cpu_hours": 1.0, "usd": 0.0, "backend": "local"},
}
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}
# The final's test splits (scenarios.TEST's keys, in order) and the two baselines every final
# registers. Written out, not read from ledger, so collection can't ERROR; a test pins them.
SPLITS = ("amsterdam_2012", "london_2012", "abbotsford_2012")
BASELINES = ("rb_default", "rb_tuned")
# Issue #71: the 2013 seal's splits (scenarios.TEST_2013's keys), and every seal's, written out
# the same way and pinned below.
SPLITS_2013 = ("amsterdam_2013", "london_2013", "abbotsford_2013")
SEALS = {2012: SPLITS, 2013: SPLITS_2013}
# Issue #72: the price regimes a 2013 split is graded at (metrics.PRICE_REGIMES' keys, in order),
# written out the same way and pinned below.
REGIMES = ("sim", "agc2020")
# The CLI's register arguments, and the row fields they must parse to.
CLI_ARMS = [
    {"name": "a", "method": "ppo", "config": {"learning_rate": 0.0003}},
    {"name": "b", "method": "residual", "config": {}},
]
CLI_REGISTERED = {
    "hypothesis": "residual PPO beats plain PPO on validation profit",
    "arms": CLI_ARMS,
    "seeds": [0, 1],
    "timesteps": 20_000,
    "scenarios": ["validation"],
    "primary": "profit_eur_m2",
    "guards": ["viol_hours_temp", "elec_cost"],
    "rule": "adopt if the profit IQM CI excludes 0",
    "budget": {"cpu_hours": 2.5, "usd": 0.0, "backend": "local"},
    "owner_ok": False,
}


def _register_argv(*extra: str, issue: int = 3) -> list[str]:
    return [
        "register",
        *("--issue", str(issue)),
        *("--hypothesis", CLI_REGISTERED["hypothesis"]),
        *("--arms", json.dumps(CLI_ARMS)),
        *("--seeds", "0", "1"),
        *("--timesteps", "20000"),
        *("--scenarios", "validation"),
        *("--primary", "profit_eur_m2"),
        *("--guards", "viol_hours_temp", "elec_cost"),
        *("--rule", CLI_REGISTERED["rule"]),
        *("--cpu-hours", "2.5"),
        *("--usd", "0"),
        *("--backend", "local"),
        *extra,
    ]


def _arms(names) -> list[dict]:
    return [{"name": name, "method": "ppo", "config": {}} for name in names]


def _register(ledger, path, *, issue=3, arms=("a", "b"), seeds=(0,), **over) -> dict:
    """A registration with sensible defaults; over replaces or adds keyword arguments."""
    kwargs = {**REGISTRATION, "issue": issue, "arms": _arms(arms), "seeds": list(seeds), **over}
    return ledger.register(**kwargs, path=path)


def _run(ledger, path, registered, arm, seed, *, status="finished", cost_usd=None):
    """Start and end one run of a registration; return (run row, run-end row)."""
    run = ledger.run_start(
        registered=registered["row_id"], arm=arm, seed=seed, run_dir=f"runs/{arm}-{seed}", path=path
    )
    end = ledger.run_end(
        run=run["row_id"],
        status=status,
        wall_s=12.5,
        timesteps=20_000,
        solver_failures=0,
        cost_usd=cost_usd,
        diagnostics=DIAGNOSTICS,
        path=path,
    )
    return run, end


def _per_arm(arms, iqm=1.5) -> dict:
    """The caller's per-arm numbers: no crashed or spend_usd, which the ledger computes itself."""
    return {
        name: {
            "iqm": iqm,
            "ci_low": 1.0,
            "ci_high": 2.0,
            "guards": {"viol_hours_temp": 3.0},
            "invalid_episodes": 0,
        }
        for name in arms
    }


def _final_register(
    ledger, path, checkpoints, *, issue=9, baselines=BASELINES, approved_by="Aarmen", **seal
) -> dict:
    """final_register with these defaults; seal, when given (#71), is passed as seal=."""
    return ledger.final_register(
        issue=issue,
        checkpoints=checkpoints,
        baselines=list(baselines),
        approved_by=approved_by,
        **seal,
        path=path,
    )


def _entry(arm: str, scored: dict) -> dict:
    """One arm's numbers on one test split, scored at scored ({run row_id: checkpoint})."""
    return {
        "iqm": 31.5,
        "ci_low": 30.0,
        "ci_high": 33.0,
        "guards": {"viol_hours_temp": {"iqm": 3.0, "ci_low": 2.0, "ci_high": 4.0}},
        "invalid_episodes": 0,
        "prob_improvement": None if arm == "rb_tuned" else 0.75,  # P(arm > rb_tuned)
        "scored": dict(scored),
    }


def _per_split(scored: dict, baselines=BASELINES, splits=SPLITS, regimes=None) -> dict:
    """A final-result's per_split: each test split (the 2012 seal's unless splits says otherwise)
    holds every learning arm, scored at its registered checkpoints ({arm: {run row_id:
    checkpoint}}), and every baseline, scored at none. With regimes (#72, a 2013 final), each split
    holds each regime, and each regime those arms."""
    arms = {**scored, **{name: {} for name in baselines}}

    def graded() -> dict:
        return {arm: _entry(arm, at) for arm, at in arms.items()}

    if regimes is None:
        return {split: graded() for split in splits}
    return {split: {regime: graded() for regime in regimes} for split in splits}


def _experiment(ledger, path) -> dict:
    """Issue 7's experiment, keyed by kind: arm a's one run, validation-selected at "final"."""
    reg = _register(ledger, path, issue=7, arms=("a",))
    run, end = _run(ledger, path, reg, "a", 0, cost_usd=0.5)
    res = ledger.record_result(
        registered=reg["row_id"],
        per_arm=_per_arm(["a"]),
        selected={run["row_id"]: "final"},
        path=path,
    )
    dec = ledger.decide(result=res["row_id"], decision="adopted", note="CI excludes 0", path=path)
    return {"registered": reg, "run": run, "run-end": end, "result": res, "decision": dec}


def _chain(ledger, path, *, final_result=True) -> dict:
    """One row of every kind, keyed by kind: issue 7's experiment, then issue 9's final evaluation
    of its selected checkpoint."""
    chain = _experiment(ledger, path)
    checkpoints = chain["result"]["selected"]
    freg = _final_register(ledger, path, checkpoints, issue=9)
    chain["final-registered"] = freg
    if final_result:
        chain["final-result"] = ledger.final_result(
            final_registered=freg["row_id"], per_split=_per_split({"a": checkpoints}), path=path
        )
    return chain


def _spend_2013(ledger, path, checkpoints: dict) -> tuple[dict, dict]:
    """#71: register issue 71's seal-2013 final of arm a at checkpoints and spend it with a 2013
    per_split, nested by price regime (#72); return (final-registered, final-result)."""
    freg = _final_register(ledger, path, checkpoints, issue=71, seal=2013)
    fres = ledger.final_result(
        final_registered=freg["row_id"],
        per_split=_per_split({"a": checkpoints}, splits=SPLITS_2013, regimes=REGIMES),
        path=path,
    )
    return freg, fres


def _two_seeds_registered(ledger, path) -> tuple[dict, dict]:
    """Issue 7's arm residual_w05 over seeds 0 and 1, validation-selected at "1750000" and "final",
    and issue 9's final registration of exactly those; return (final-registered row, checkpoints)."""
    reg = _register(ledger, path, issue=7, arms=("residual_w05",), seeds=(0, 1))
    run0, run1 = (_run(ledger, path, reg, "residual_w05", s)[0]["row_id"] for s in (0, 1))
    selected = {run0: "1750000", run1: "final"}
    ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["residual_w05"]), selected=selected, path=path
    )
    return _final_register(ledger, path, selected), selected


def _show(ledger, path, via, *, issue=None, spend=False) -> None:
    """show through the Python API or through main's argv."""
    if via == "api":
        ledger.show(issue=issue, spend=spend, path=path)
        return
    argv = ["show", *(["--issue", str(issue)] if issue is not None else [])]
    assert ledger.main([*argv, *(["--spend"] if spend else [])], path=path) == 0


def _line(row: dict) -> bytes:
    """The exact bytes append writes for one row."""
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


def _utc(stamp: str) -> datetime:
    """Parse an ISO timestamp and check it is tz-aware UTC."""
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() == timedelta(0), stamp
    return parsed


def _check_rejected(ledger, path, broken: dict, intact: dict) -> None:
    """append refuses broken and leaves the file as it was; it then accepts intact.

    Accepting intact shows the one change made to broken is what append refused.
    """
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ledger.append(broken, path=path)
    assert path.read_bytes() == before
    ledger.append(intact, path=path)
    assert path.read_bytes() == before + _line(intact)


def test_every_kind_is_stamped_and_written_as_one_sorted_line(tmp_path):
    """Each kind's row has exactly the contract's fields, stamped by the ledger.

    schema_version 1; row_id <kind>-<8 lowercase hex>, all distinct; at in UTC, within the calls;
    issue an int, taken from the row each link names (7 for the experiment, 9 for the final);
    fingerprints equal to manifest.fingerprints(). The file holds the returned rows in call order,
    each as one json.dumps(sort_keys=True, allow_nan=False) line.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    t0 = datetime.now(UTC).replace(microsecond=0)  # a stamp may be truncated to whole seconds
    chain = _chain(ledger, path)
    t1 = datetime.now(UTC)

    assert ledger.SCHEMA_VERSION == 1
    assert list(chain) == list(KIND_FIELDS)
    for kind, row in chain.items():
        assert set(row) == BASE_FIELDS | KIND_FIELDS[kind], kind
        assert row["kind"] == kind
        assert row["schema_version"] == 1
        assert re.fullmatch(rf"{re.escape(kind)}-[0-9a-f]{{8}}", row["row_id"]), row["row_id"]
        assert t0 <= _utc(row["at"]) <= t1, kind
        assert type(row["issue"]) is int, kind
        assert row["fingerprints"] == manifest.fingerprints(), kind
    assert [row["issue"] for row in chain.values()] == [7, 7, 7, 7, 7, 9, 9]
    assert len({row["row_id"] for row in chain.values()}) == len(chain)
    assert path.read_bytes() == b"".join(_line(row) for row in chain.values())


def test_rows_link_their_parents_and_keep_the_callers_values(tmp_path):
    """Each row records what its caller passed and names the row it links to.

    run → registered with status "running"; run-end → run; result → registered; decision → result;
    final-result → final-registered under the key "final-registered". approved_at is stamped now.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    t0 = datetime.now(UTC).replace(microsecond=0)
    reg, run, end, res, dec, freg, fres = _chain(ledger, path).values()
    t1 = datetime.now(UTC)

    assert {k: reg[k] for k in KIND_FIELDS["registered"]} == {
        **REGISTRATION,
        "arms": _arms(["a"]),
        "seeds": [0],
        "owner_ok": False,
    }
    assert {k: run[k] for k in KIND_FIELDS["run"]} == {
        "registered": reg["row_id"],
        "arm": "a",
        "seed": 0,
        "run_dir": "runs/a-0",
        "status": "running",
    }
    assert {k: end[k] for k in KIND_FIELDS["run-end"]} == {
        "run": run["row_id"],
        "status": "finished",
        "wall_s": 12.5,
        "timesteps": 20_000,
        "solver_failures": 0,
        "cost_usd": 0.5,
        "diagnostics": DIAGNOSTICS,
    }
    assert res["registered"] == reg["row_id"]
    assert res["selected"] == {run["row_id"]: "final"}
    assert {k: dec[k] for k in KIND_FIELDS["decision"]} == {
        "result": res["row_id"],
        "decision": "adopted",
        "note": "CI excludes 0",
    }
    assert freg["checkpoints"] == {run["row_id"]: "final"}
    assert freg["baselines"] == ["rb_default", "rb_tuned"]
    assert freg["approved_by"] == "Aarmen"
    assert t0 <= _utc(freg["approved_at"]) <= t1
    assert fres["final-registered"] == freg["row_id"]
    assert fres["per_split"] == _per_split({"a": {run["row_id"]: "final"}})


def test_rows_carry_manifest_fingerprints_computed_at_call_time(tmp_path, monkeypatch):
    """A row's fingerprints are manifest.fingerprints() at the moment it is written, not cached.

    Known answer: the temp tree's docs/PROTOCOL.md holds b"abc", whose sha256 is the FIPS 180-2
    test vector ba7816bf...f20015ad. Editing it between two rows changes the second row's hash.
    """
    import greenleaf_control.ledger as ledger

    tree = tmp_path / "repo"
    protocol = tree / "docs" / "PROTOCOL.md"
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(b"abc")
    monkeypatch.setattr(manifest, "REPO_ROOT", tree)
    path = tmp_path / "ledger.jsonl"

    reg = _register(ledger, path, arms=("a",))
    assert reg["fingerprints"] == manifest.fingerprints(tree)
    assert (
        reg["fingerprints"]["docs/PROTOCOL.md"]
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    protocol.write_bytes(b"abc, edited")
    run = ledger.run_start(registered=reg["row_id"], arm="a", seed=0, run_dir="runs/a-0", path=path)
    assert run["fingerprints"] == manifest.fingerprints(tree)
    assert run["fingerprints"]["docs/PROTOCOL.md"] != reg["fingerprints"]["docs/PROTOCOL.md"]
    assert [r["fingerprints"] for r in ledger.rows(path=path)] == [
        reg["fingerprints"],
        run["fingerprints"],
    ]


def test_appends_never_change_earlier_bytes(tmp_path):
    """Append-only: every write adds exactly one line after the old bytes, in the same file.

    A refused call and a dry registration leave the bytes as they were. The inode check catches a
    rewrite through a temp file and a rename, which would keep the bytes but not the file.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",))
    inode = path.stat().st_ino
    seen = [path.read_bytes()]

    run = ledger.run_start(registered=reg["row_id"], arm="a", seed=0, run_dir="runs/a", path=path)
    seen.append(path.read_bytes())
    ledger.run_end(
        run=run["row_id"],
        status="finished",
        wall_s=3.0,
        timesteps=96,
        solver_failures=2,
        cost_usd=None,
        diagnostics=DIAGNOSTICS,
        path=path,
    )
    seen.append(path.read_bytes())
    selected = {run["row_id"]: "final"}  # the final registers this validation-selected checkpoint
    res = ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["a"]), selected=selected, path=path
    )
    seen.append(path.read_bytes())
    ledger.decide(result=res["row_id"], decision="dropped", note="no gain", path=path)
    seen.append(path.read_bytes())
    _register(ledger, path, arms=("a",))  # the issue's second registration
    seen.append(path.read_bytes())

    with pytest.raises(ValueError):
        _register(ledger, path, arms=("a",))  # a third, without owner_ok
    _register(ledger, path, issue=4, arms=("a",), dry=True)
    assert path.read_bytes() == seen[-1]

    freg = _final_register(ledger, path, selected)
    seen.append(path.read_bytes())
    ledger.final_result(
        final_registered=freg["row_id"], per_split=_per_split({"a": selected}), path=path
    )
    seen.append(path.read_bytes())

    for old, new in zip(seen, seen[1:], strict=False):
        assert new.startswith(old)
        assert new.count(b"\n") == old.count(b"\n") + 1
    assert path.stat().st_ino == inode


@pytest.mark.parametrize("field", sorted(BASE_FIELDS | KIND_FIELDS["registered"]))
def test_append_rejects_a_row_missing_a_required_field(tmp_path, field):
    """A valid registered row (from register(dry=True)) minus one required field raises ValueError.

    The file is byte-identical afterwards, and the intact row is then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    _register(ledger, path)
    row = _register(ledger, path, dry=True)
    broken = {k: v for k, v in row.items() if k != field}

    _check_rejected(ledger, path, broken, row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "experiment"),  # not a contract kind
        ("schema_version", 2),
        ("schema_version", "1"),
        ("issue", "3"),  # a string, not an int
        ("issue", 3.5),
        ("issue", None),
        ("row_id", "registered-0123abc"),  # 7 hex digits
        ("row_id", "registered-0123abcde"),  # 9 hex digits
        ("row_id", "registered-0123ABCD"),  # uppercase
        ("row_id", "registered-0123abcg"),  # g isn't hex
        ("row_id", "run-0123abcd"),  # another kind's prefix
        ("row_id", "0123abcd"),  # no prefix
        # not a scored result key (selection.RESULT_KEYS)
        pytest.param("primary", "profit", id="primary-not-a-result-key"),
        pytest.param("guards", ["viol_hours_temp", "violation_hours"], id="guard-not-a-result-key"),
    ],
)
def test_append_rejects_a_bad_value(tmp_path, field, value):
    """An unknown kind, a wrong schema_version, a non-int issue, a malformed row_id, or a primary or
    guard that isn't a scored result key raises ValueError with the file byte-identical; the intact
    row is then accepted. So a hand-built registered row can't skip register's metric-name check."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    _register(ledger, path)
    row = _register(ledger, path, dry=True)

    _check_rejected(ledger, path, {**row, field: value}, row)


def test_append_rejects_a_duplicate_row_id(tmp_path):
    """A row reusing a row_id already in the file raises ValueError and writes nothing."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    first = _register(ledger, path)
    row = _register(ledger, path, dry=True)

    _check_rejected(ledger, path, {**row, "row_id": first["row_id"]}, row)


def test_a_nan_or_inf_is_rejected_and_nothing_written(tmp_path):
    """allow_nan=False: a NaN wall_s or an infinite IQM raises ValueError and writes nothing."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",))
    run = ledger.run_start(registered=reg["row_id"], arm="a", seed=0, run_dir="runs/a", path=path)
    end_kwargs = {
        "run": run["row_id"],
        "status": "finished",
        "timesteps": 96,
        "solver_failures": 0,
        "cost_usd": None,
        "diagnostics": DIAGNOSTICS,
        "path": path,
    }
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ledger.run_end(wall_s=math.nan, **end_kwargs)
    assert path.read_bytes() == before

    ledger.run_end(wall_s=12.5, **end_kwargs)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ledger.record_result(
            registered=reg["row_id"], per_arm=_per_arm(["a"], iqm=math.inf), selected={}, path=path
        )
    assert path.read_bytes() == before


# (function, its link argument, the kind of row the link must name)
LINKS = [
    ("run_start", "registered", "registered"),
    ("run_end", "run", "run"),
    ("record_result", "registered", "registered"),
    ("decide", "result", "result"),
    ("final_result", "final_registered", "final-registered"),
]
# For each linked kind, a real row of the nearest other kind.
WRONG_KIND = {
    "registered": "final-registered",
    "run": "run-end",
    "result": "decision",
    "final-registered": "registered",
}


def _link_kwargs(func: str, chain: dict) -> dict:
    """Valid arguments for func besides its link."""
    checkpoints = chain["final-registered"]["checkpoints"]
    return {
        "run_start": {"arm": "a", "seed": 0, "run_dir": "runs/a-0-again"},
        "run_end": {
            "status": "finished",
            "wall_s": 1.0,
            "timesteps": 96,
            "solver_failures": 0,
            "cost_usd": None,
            "diagnostics": DIAGNOSTICS,
        },
        "record_result": {"per_arm": _per_arm(["a"]), "selected": {}},
        "decide": {"decision": "dropped", "note": "no gain"},
        "final_result": {"per_split": _per_split({"a": checkpoints})},
    }[func]


@pytest.mark.parametrize("target", ["missing", "wrong-kind"])
@pytest.mark.parametrize(("func", "arg", "kind"), LINKS)
def test_a_link_must_name_an_existing_row_of_its_kind(tmp_path, func, arg, kind, target):
    """A link naming no row, or a row of another kind, raises ValueError and writes nothing.

    "missing" is a well-formed id no row has. "wrong-kind" is a real row of the nearest other kind
    (final-registered for registered, run-end for run, decision for result, registered for
    final-registered). The chain stops before final_result, so the spent seal isn't the reason.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    chain = _chain(ledger, path, final_result=False)
    link = f"{kind}-00000000" if target == "missing" else chain[WRONG_KIND[kind]]["row_id"]
    before = path.read_bytes()

    with pytest.raises(ValueError):
        getattr(ledger, func)(**{arg: link}, **_link_kwargs(func, chain), path=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("dry", [False, True])
def test_a_third_registration_on_one_issue_needs_owner_ok(tmp_path, dry):
    """Two registrations per issue are free; a third, dry or not, raises ValueError naming owner_ok.

    Another issue's registrations don't count toward the two. With owner_ok=True the third is
    written and records owner_ok True; a fourth without it is refused again.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    first = _register(ledger, path, issue=3)
    _register(ledger, path, issue=4)
    _register(ledger, path, issue=4)
    _register(ledger, path, issue=3)  # issue 3's second: issue 4's two don't count
    before = path.read_bytes()

    with pytest.raises(ValueError, match="owner_ok"):
        _register(ledger, path, issue=3, dry=dry)
    assert path.read_bytes() == before

    third = _register(ledger, path, issue=3, owner_ok=True)
    assert first["owner_ok"] is False
    assert third["owner_ok"] is True
    assert ledger.rows(kind="registered", issue=3, path=path)[-1] == third

    before = path.read_bytes()
    with pytest.raises(ValueError, match="owner_ok"):
        _register(ledger, path, issue=3)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("missing", "started"),
    [
        (("arm_beta", 23), False),  # never started
        (("arm_beta", 23), True),  # started, no run-end yet
        (("arm_alpha", 11), True),
    ],
)
def test_record_result_refuses_until_every_arm_seed_has_a_run_end(tmp_path, missing, started):
    """2 arms × 2 seeds with one pair lacking a run-end: ValueError naming run-end and that pair.

    Nothing is written.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    arms, seeds = ("arm_alpha", "arm_beta"), (11, 23)
    reg = _register(ledger, path, arms=arms, seeds=seeds)
    for arm in arms:
        for seed in seeds:
            if (arm, seed) != missing:
                _run(ledger, path, reg, arm, seed)
            elif started:
                ledger.run_start(
                    registered=reg["row_id"], arm=arm, seed=seed, run_dir="runs/open", path=path
                )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="run-end") as excinfo:
        ledger.record_result(
            registered=reg["row_id"], per_arm=_per_arm(arms), selected={}, path=path
        )
    assert missing[0] in str(excinfo.value)
    assert str(missing[1]) in str(excinfo.value)
    assert path.read_bytes() == before


def test_record_result_refuses_while_a_finished_rerun_hides_an_open_run(tmp_path):
    """Every run needs its own run-end; one finished run of an arm/seed doesn't cover another.

    a/0 is started twice and only the second run ends (finished). record_result raises ValueError
    naming run-end and the first run's row_id, and writes nothing. Once the first run ends crashed,
    the result is written and arm a's crashed count is 1.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",), seeds=(0,))
    start = {"registered": reg["row_id"], "arm": "a", "seed": 0, "path": path}
    run1 = ledger.run_start(run_dir="runs/a-0", **start)
    run2 = ledger.run_start(run_dir="runs/a-0-rerun", **start)
    end = {
        "wall_s": 12.5,
        "timesteps": 20_000,
        "solver_failures": 0,
        "cost_usd": None,
        "diagnostics": DIAGNOSTICS,
        "path": path,
    }
    ledger.run_end(run=run2["row_id"], status="finished", **end)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="run-end") as excinfo:
        ledger.record_result(
            registered=reg["row_id"], per_arm=_per_arm(["a"]), selected={}, path=path
        )
    assert run1["row_id"] in str(excinfo.value)
    assert path.read_bytes() == before

    ledger.run_end(run=run1["row_id"], status="crashed", **end)
    res = ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["a"]), selected={}, path=path
    )
    assert res["per_arm"]["a"]["crashed"] == 1  # run1 crashed, run2 finished: 1 crash
    assert ledger.rows(kind="result", path=path) == [res]


def test_record_result_ignores_run_ends_of_another_registration(tmp_path):
    """A run-end only covers the registration its run names, even on the same issue.

    The second registration's arm b has no run of its own, so its result is refused, while the
    first registration, fully run, gets its result.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    first = _register(ledger, path)
    second = _register(ledger, path)
    for arm in ("a", "b"):
        _run(ledger, path, first, arm, 0)
    _run(ledger, path, second, "a", 0)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="run-end"):
        ledger.record_result(
            registered=second["row_id"], per_arm=_per_arm(["a", "b"]), selected={}, path=path
        )
    assert path.read_bytes() == before
    res = ledger.record_result(
        registered=first["row_id"], per_arm=_per_arm(["a", "b"]), selected={}, path=path
    )
    assert ledger.rows(kind="result", path=path) == [res]


def test_result_counts_crashed_runs_and_sums_spend_per_arm(tmp_path):
    """The ledger adds each arm's crashed count and spend_usd from that registration's run-ends.

    Arm a: seed 0 finished at 0.5 USD, seed 1 crashed at 0.25 USD, so crashed 1 and
    spend 0.5 + 0.25 = 0.75 USD. Arm b: seed 0 budget-stop at 1.0 USD (not a crash), seed 1 finished
    with cost None (counts as 0), so crashed 0 and spend 1.0 + 0 = 1.0 USD. A crashed 9.0 USD run of
    arm a under another registration on the same issue counts toward neither.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a", "b"), seeds=(0, 1))
    run_a0, _ = _run(ledger, path, reg, "a", 0, status="finished", cost_usd=0.5)
    _run(ledger, path, reg, "a", 1, status="crashed", cost_usd=0.25)
    _run(ledger, path, reg, "b", 0, status="budget-stop", cost_usd=1.0)
    run_b1, _ = _run(ledger, path, reg, "b", 1, status="finished", cost_usd=None)
    decoy = _register(ledger, path, arms=("a",))
    _run(ledger, path, decoy, "a", 0, status="crashed", cost_usd=9.0)
    per_arm = {**_per_arm(["a"], iqm=1.5), **_per_arm(["b"], iqm=2.5)}
    selected = {run_a0["row_id"]: "ckpt_final.zip", run_b1["row_id"]: "ckpt_250000.zip"}

    res = ledger.record_result(
        registered=reg["row_id"], per_arm=per_arm, selected=selected, path=path
    )

    caller = {"ci_low": 1.0, "ci_high": 2.0, "guards": {"viol_hours_temp": 3.0}}
    assert res["per_arm"] == {
        "a": {**caller, "iqm": 1.5, "invalid_episodes": 0, "crashed": 1, "spend_usd": 0.75},
        "b": {**caller, "iqm": 2.5, "invalid_episodes": 0, "crashed": 0, "spend_usd": 1.0},
    }
    assert res["selected"] == selected
    assert res["issue"] == 3
    assert ledger.rows(kind="result", path=path) == [res]


@pytest.mark.parametrize("field", ["crashed", "spend_usd"])
def test_record_result_refuses_caller_supplied_crashed_or_spend(tmp_path, field):
    """crashed and spend_usd come from the ledger's own run-end rows; a caller's value raises
    ValueError and nothing is written."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",))
    _run(ledger, path, reg, "a", 0, status="crashed")
    per_arm = _per_arm(["a"])
    per_arm["a"][field] = 0
    before = path.read_bytes()

    with pytest.raises(ValueError):
        ledger.record_result(registered=reg["row_id"], per_arm=per_arm, selected={}, path=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("crashed", "spend_usd"), [(0, 0.25), (1, 0.0)], ids=["crash hidden", "spend hidden"]
)
def test_append_checks_a_hand_built_result_against_the_run_ends(tmp_path, crashed, spend_usd):
    """append itself checks a result's crashed and spend_usd against the registration's run-ends.

    Arm a's one run crashed at 0.25 USD, so its result must carry crashed 1 and spend_usd 0.25.
    A hand-built row (not from record_result) claiming crashed 0, or spend 0.0 USD, raises
    ValueError with the file byte-identical; the honest row is then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",), seeds=(0,))
    _run(ledger, path, reg, "a", 0, status="crashed", cost_usd=0.25)
    numbers = _per_arm(["a"])["a"]
    honest = {
        "schema_version": 1,
        "row_id": "result-0123abcd",
        "kind": "result",
        "at": datetime.now(UTC).isoformat(),
        "issue": reg["issue"],
        "fingerprints": manifest.fingerprints(),
        "registered": reg["row_id"],
        # 1 crashed run of arm a; its cost 0.25 USD is the arm's whole spend
        "per_arm": {"a": {**numbers, "crashed": 1, "spend_usd": 0.25}},
        "selected": {},
    }
    broken = {**honest, "per_arm": {"a": {**numbers, "crashed": crashed, "spend_usd": spend_usd}}}

    _check_rejected(ledger, path, broken, honest)
    assert ledger.rows(kind="result", path=path) == [honest]


@pytest.mark.parametrize(
    "arms", [["a"], ["a", "b", "c"]], ids=["a registered arm missing", "an unregistered arm"]
)
def test_record_result_needs_numbers_for_exactly_the_registered_arms(tmp_path, arms):
    """per_arm is keyed by the registered arm names; a missing or extra arm raises ValueError.

    A result that drops an arm would hide that arm's crashes. Nothing is written.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a", "b"))
    for arm in ("a", "b"):
        _run(ledger, path, reg, arm, 0)
    before = path.read_bytes()

    with pytest.raises(ValueError):
        ledger.record_result(
            registered=reg["row_id"], per_arm=_per_arm(arms), selected={}, path=path
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("target", ["missing", "a registered row"])
def test_decide_refuses_without_an_existing_result(tmp_path, target):
    """decide on an id that is not a result row raises ValueError naming result; nothing written."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path)
    result = "result-00000000" if target == "missing" else reg["row_id"]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="result"):
        ledger.decide(result=result, decision="adopted", note="looks good", path=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("decision", ["adopt", "Adopted", "maybe", ""])
def test_decide_rejects_anything_but_adopted_or_dropped(tmp_path, decision):
    """decision is "adopted" or "dropped"; anything else raises ValueError and writes nothing."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",))
    _run(ledger, path, reg, "a", 0)
    res = ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["a"]), selected={}, path=path
    )
    before = path.read_bytes()

    with pytest.raises(ValueError):
        ledger.decide(result=res["row_id"], decision=decision, note="n", path=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("status", ["running", "done", "Finished", ""])
def test_run_end_rejects_an_unknown_status(tmp_path, status):
    """A run-end's status is finished, crashed or budget-stop; "running" is a run's, never an end's.

    Anything else raises ValueError and writes nothing.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, arms=("a",))
    run = ledger.run_start(registered=reg["row_id"], arm="a", seed=0, run_dir="runs/a", path=path)
    before = path.read_bytes()

    with pytest.raises(ValueError):
        ledger.run_end(
            run=run["row_id"],
            status=status,
            wall_s=1.0,
            timesteps=96,
            solver_failures=0,
            cost_usd=None,
            diagnostics=DIAGNOSTICS,
            path=path,
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("approved_by", [None, "", "   ", "\t\n"])
def test_final_register_refuses_without_approved_by(tmp_path, approved_by):
    """final_register with approved_by None, empty or blank raises ValueError naming approved_by.

    Nothing is written; the file isn't even created. approved_by is checked first, so on this
    empty ledger it is what's named, not the checkpoints no result has selected yet.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"

    with pytest.raises(ValueError, match="approved_by"):
        _final_register(ledger, path, {"run-0123abcd": "final"}, approved_by=approved_by)
    assert not path.exists()


@pytest.mark.parametrize("second_on", ["the same final-registered", "another issue's"])
def test_a_second_final_result_is_refused_on_any_issue(tmp_path, second_on):
    """The seal is spent once: after one final-result, any other raises ValueError naming
    final-result, whichever final-registered row it names. Nothing is written.

    Issue 10's final-registered row, of the same checkpoints, is written before the seal is spent:
    after it, final_register itself is refused.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    per_split = _per_split({"a": checkpoints})
    freg = _final_register(ledger, path, checkpoints, issue=9)
    other = _final_register(ledger, path, checkpoints, issue=10)
    ledger.final_result(final_registered=freg["row_id"], per_split=per_split, path=path)
    target = freg if second_on == "the same final-registered" else other
    before = path.read_bytes()

    with pytest.raises(ValueError, match="final-result"):
        ledger.final_result(final_registered=target["row_id"], per_split=per_split, path=path)
    assert path.read_bytes() == before
    assert len(ledger.rows(kind="final-result", path=path)) == 1


def test_final_register_is_refused_once_the_seal_is_spent(tmp_path):
    """After a final-result, a new final_register raises ValueError "the seal is spent", even
    another issue's naming the same validation-selected checkpoint. The file is byte-identical."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    chain = _chain(ledger, path)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="the seal is spent"):
        _final_register(ledger, path, chain["result"]["selected"], issue=10)
    assert path.read_bytes() == before
    assert ledger.rows(kind="final-registered", path=path) == [chain["final-registered"]]


@pytest.mark.parametrize("checkpoints", [{}, "a list"], ids=["empty", "not a dict"])
def test_final_register_refuses_checkpoints_that_are_empty_or_not_a_dict(tmp_path, checkpoints):
    """checkpoints is a non-empty {run row_id: checkpoint}. {} or a bare list of the selected run
    ids raises ValueError naming checkpoints, the file byte-identical; the selected dict is then
    accepted."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    selected = _experiment(ledger, path)["result"]["selected"]
    broken = list(selected) if checkpoints == "a list" else checkpoints
    before = path.read_bytes()

    with pytest.raises(ValueError, match="checkpoints"):
        _final_register(ledger, path, broken)
    assert path.read_bytes() == before
    assert _final_register(ledger, path, selected)["checkpoints"] == selected


@pytest.mark.parametrize(
    "case", ['"final" where selected says "1750000"', "a run no result selected", "a crashed run"]
)
def test_final_register_refuses_a_checkpoint_no_result_selected(tmp_path, case):
    """Each registered (run, checkpoint) pair must be exactly a value of some result's selected.

    Arm residual_w05's seed 0 was validation-selected at "1750000"; seed 1 crashed, so, as
    selection builds it, the result's selected leaves it out. Registering seed 0 at "final", a
    run id no result selected (run-0badc0de), or the crashed seed 1 raises ValueError naming
    selected and that run, the file byte-identical (registered vs selected). The selected
    checkpoints are then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, issue=7, arms=("residual_w05",), seeds=(0, 1))
    run0 = _run(ledger, path, reg, "residual_w05", 0)[0]["row_id"]
    run1 = _run(ledger, path, reg, "residual_w05", 1, status="crashed")[0]["row_id"]
    selected = {run0: "1750000"}  # the crashed run1 is absent, as selection builds it
    ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["residual_w05"]), selected=selected, path=path
    )
    broken, named = {
        '"final" where selected says "1750000"': ({run0: "final"}, run0),
        "a run no result selected": ({**selected, "run-0badc0de": "final"}, "run-0badc0de"),
        "a crashed run": ({**selected, run1: "final"}, run1),
    }[case]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="selected") as excinfo:
        _final_register(ledger, path, broken)
    assert named in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before
    assert _final_register(ledger, path, selected)["checkpoints"] == selected


def test_final_register_refuses_one_arm_from_two_registrations(tmp_path):
    """A learning arm, named by its run rows' arm, comes from one registration.

    Issues 7 and 8 each ran an arm named residual_w05 (as E4 and E6 did), and each result selected
    its run. Registering both runs raises ValueError naming residual_w05 and both registered
    row_ids, the file byte-identical; issue 8's run alone is then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    picked = {}  # registered row_id -> that result's selected
    for issue, checkpoint in [(7, "final"), (8, "1500000")]:
        reg = _register(ledger, path, issue=issue, arms=("residual_w05",))
        run = _run(ledger, path, reg, "residual_w05", 0)[0]["row_id"]
        res = ledger.record_result(
            registered=reg["row_id"],
            per_arm=_per_arm(["residual_w05"]),
            selected={run: checkpoint},
            path=path,
        )
        picked[reg["row_id"]] = res["selected"]
    (reg7, at7), (reg8, at8) = picked.items()
    before = path.read_bytes()

    with pytest.raises(ValueError, match="residual_w05") as excinfo:
        _final_register(ledger, path, {**at7, **at8})
    assert reg7 in str(excinfo.value), str(excinfo.value)
    assert reg8 in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before
    assert _final_register(ledger, path, at8)["checkpoints"] == at8


@pytest.mark.parametrize(
    ("baselines", "named"),
    [
        pytest.param(["rb_default"], "rb_tuned", id="missing rb_tuned"),
        pytest.param(["rb_tuned"], "rb_default", id="missing rb_default"),
        pytest.param(["rb_default", "rb_tuned", "rb_magic"], "rb_magic", id="rb_magic"),
        pytest.param(["rb_default", "rb_tuned", "rb_tuned"], "rb_tuned", id="a duplicate"),
    ],
)
def test_final_register_refuses_bad_baselines(tmp_path, baselines, named):
    """baselines are drawn from FINAL_BASELINES (envs.BASELINES) without a duplicate, and hold both
    rb_default and rb_tuned. Otherwise ValueError naming baselines and the name at fault, the file
    byte-identical. All four of envs.BASELINES are then accepted."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="baselines") as excinfo:
        _final_register(ledger, path, checkpoints, baselines=baselines)
    assert named in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before
    every = ["zero_controls", "random", "rb_default", "rb_tuned"]
    assert _final_register(ledger, path, checkpoints, baselines=every)["baselines"] == every


@pytest.mark.parametrize(
    ("arm", "baselines"),
    [
        pytest.param("rb_tuned", ["rb_default", "rb_tuned"], id="rb_tuned"),
        pytest.param("random", ["rb_default", "rb_tuned", "random"], id="random"),
    ],
)
def test_final_register_refuses_a_learning_arm_named_like_a_baseline(tmp_path, arm, baselines):
    """per_split keys learning arms and baselines by name in one dict, so no learning arm may share
    a name with one of the final's baselines.

    Issue 7 ran arms <arm> (method ppo) and ppo, and its result selected both runs at "final".
    Registering both, with <arm> among the baselines, raises ValueError naming the arm and saying
    the learning arm shares a baseline's name, the file byte-identical; ppo's run alone, with the
    same baselines, is then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, issue=7, arms=(arm, "ppo"))
    runs = {name: _run(ledger, path, reg, name, 0)[0]["row_id"] for name in (arm, "ppo")}
    selected = {run: "final" for run in runs.values()}
    ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm([arm, "ppo"]), selected=selected, path=path
    )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="baseline") as excinfo:
        _final_register(ledger, path, selected, baselines=baselines)
    assert "arm" in str(excinfo.value), str(excinfo.value)
    assert arm in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before
    ppo = {runs["ppo"]: "final"}
    assert _final_register(ledger, path, ppo, baselines=baselines)["checkpoints"] == ppo


def test_final_arms_maps_each_registered_arm_to_its_checkpoints(tmp_path):
    """final_arms(final_registered, rows) is {arm: {run row_id: checkpoint}}, each arm named by its
    run rows' arm.

    Issue 7 ran arm ppo; issue 8 ran residual_w05 and residual_w4, seeds 0 and 1. The final
    registers ppo's run and residual_w05's two, not residual_w4's, so final_arms holds exactly
    ppo and residual_w05. A final-result holding those two arms and the baselines is accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg7 = _register(ledger, path, issue=7, arms=("ppo",))
    ppo = _run(ledger, path, reg7, "ppo", 0)[0]["row_id"]
    ledger.record_result(
        registered=reg7["row_id"], per_arm=_per_arm(["ppo"]), selected={ppo: "final"}, path=path
    )
    arms8 = ("residual_w05", "residual_w4")
    reg8 = _register(ledger, path, issue=8, arms=arms8, seeds=(0, 1))
    run = {(a, s): _run(ledger, path, reg8, a, s)[0]["row_id"] for a in arms8 for s in (0, 1)}
    w05 = {run["residual_w05", 0]: "1750000", run["residual_w05", 1]: "final"}
    w4 = {run["residual_w4", 0]: "final", run["residual_w4", 1]: "500000"}
    ledger.record_result(
        registered=reg8["row_id"], per_arm=_per_arm(arms8), selected={**w05, **w4}, path=path
    )
    freg = _final_register(ledger, path, {ppo: "final", **w05})

    arms = ledger.final_arms(freg, ledger.rows(path=path))

    assert arms == {"ppo": {ppo: "final"}, "residual_w05": w05}
    fres = ledger.final_result(
        final_registered=freg["row_id"], per_split=_per_split(arms), path=path
    )
    assert ledger.rows(kind="final-result", path=path) == [fres]


# A per_split fault, laid into the last test split so a check of the first split alone misses it,
# and the names the refusal must give.
PER_SPLIT_FAULTS = [
    pytest.param("a fourth split", ["validation"], id="a fourth split"),
    pytest.param("a missing arm", ["residual_w05"], id="a missing arm"),
    # a FINAL_BASELINE, but not one this final registered
    pytest.param("an extra arm", ["zero_controls"], id="an extra arm"),
    pytest.param("scored drops a run", ["scored", "residual_w05"], id="scored drops a run"),
    pytest.param("scored adds a run", ["scored", "residual_w05"], id="scored adds a run"),
    pytest.param("scored names 250000", ["scored", "residual_w05"], id="scored names 250000"),
    pytest.param("a baseline scored", ["scored", "rb_tuned"], id="a baseline's scored not {}"),
    pytest.param("no prob_improvement", ["prob_improvement"], id="no prob_improvement"),
]


@pytest.mark.parametrize(("fault", "named"), PER_SPLIT_FAULTS)
def test_final_result_refuses_a_per_split_that_breaks_the_registration(tmp_path, fault, named):
    """per_split is exactly the three test splits; each holds exactly the registered arms plus the
    baselines; each entry has exactly FINAL_ARM_KEYS; and scored is the arm's registered
    checkpoints, {} for a baseline (registered vs scored).

    residual_w05 is registered at seed 0's "1750000" and seed 1's "final". Each fault raises
    ValueError naming its cause, the file byte-identical; the faithful per_split is then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    freg, registered = _two_seeds_registered(ledger, path)
    run0, run1 = registered
    per_split = _per_split({"residual_w05": registered})
    broken = copy.deepcopy(per_split)
    last = broken["abbotsford_2012"]
    w05 = last["residual_w05"]
    {
        "a fourth split": lambda: broken.update(validation=copy.deepcopy(last)),
        "a missing arm": lambda: last.pop("residual_w05"),
        "an extra arm": lambda: last.update(zero_controls=copy.deepcopy(last["rb_default"])),
        "scored drops a run": lambda: w05.update(scored={run0: "1750000"}),
        "scored adds a run": lambda: w05.update(scored={**registered, "run-0badc0de": "final"}),
        "scored names 250000": lambda: w05.update(scored={run0: "250000", run1: "final"}),
        "a baseline scored": lambda: last["rb_tuned"].update(scored={run1: "final"}),
        "no prob_improvement": lambda: last["rb_default"].pop("prob_improvement"),
    }[fault]()
    before = path.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        ledger.final_result(final_registered=freg["row_id"], per_split=broken, path=path)
    for name in named:
        assert name in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before

    fres = ledger.final_result(final_registered=freg["row_id"], per_split=per_split, path=path)
    assert ledger.rows(kind="final-result", path=path) == [fres]


# A prob_improvement fault, laid into the last test split only: the arm and the value it is given.
PROB_IMPROVEMENT_FAULTS = [
    pytest.param("rb_tuned", 0.5, id="rb_tuned with a number"),
    pytest.param("residual_w05", None, id="a learning arm with null"),
    pytest.param("rb_default", None, id="another baseline with null"),
]


@pytest.mark.parametrize(("arm", "value"), PROB_IMPROVEMENT_FAULTS)
def test_final_result_refuses_prob_improvement_unless_null_exactly_for_rb_tuned(
    tmp_path, arm, value
):
    """prob_improvement is P(arm > rb_tuned) on the primary: null for rb_tuned itself, a number for
    every other arm, learning or baseline.

    In abbotsford_2012 alone, rb_tuned given 0.5, or residual_w05 or rb_default given null, raises
    ValueError naming prob_improvement and that arm, the file byte-identical; the faithful
    per_split is then accepted.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    freg, registered = _two_seeds_registered(ledger, path)
    per_split = _per_split({"residual_w05": registered})
    broken = copy.deepcopy(per_split)
    broken["abbotsford_2012"][arm]["prob_improvement"] = value
    before = path.read_bytes()

    with pytest.raises(ValueError, match="prob_improvement") as excinfo:
        ledger.final_result(final_registered=freg["row_id"], per_split=broken, path=path)
    assert arm in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before

    fres = ledger.final_result(final_registered=freg["row_id"], per_split=per_split, path=path)
    assert ledger.rows(kind="final-result", path=path) == [fres]


def test_final_names_are_pinned_to_scenarios_and_envs():
    """The ledger writes these out to stay stdlib-only, so they are pinned here: TEST_SPLITS is
    scenarios.TEST's keys in order, FINAL_BASELINES is envs.BASELINES plus rb_tuned_agc2020 (#72:
    evaluate's BASELINES, the agc2020 tune's rules a 2013 final grades), and FINAL_ARM_KEYS is the
    contract's per-split arm entry. This module's SPLITS copy is pinned too."""
    import greenleaf_control.ledger as ledger
    from greenleaf_control import envs, scenarios

    assert ledger.TEST_SPLITS == tuple(scenarios.TEST)
    assert ledger.FINAL_BASELINES == (*envs.BASELINES, "rb_tuned_agc2020")
    assert ledger.FINAL_ARM_KEYS == {
        "iqm",
        "ci_low",
        "ci_high",
        "guards",
        "invalid_episodes",
        "prob_improvement",
        "scored",
    }
    assert tuple(scenarios.TEST) == SPLITS


# --- Issue #71: seal groups, one per test year ----------------------------------------------------


def test_ledger_seals_are_pinned_to_scenarios_seals():
    """ledger.SEALS, written out so the ledger stays stdlib-only, is {seal year: its split names}:
    {2012: the 2012 splits, 2013: the 2013 splits}, each seal's names in scenarios.SEALS's order.
    The 2012 seal's names are TEST_SPLITS. This module's SEALS copy is pinned too."""
    import greenleaf_control.ledger as ledger
    from greenleaf_control import scenarios

    assert list(ledger.SEALS) == [2012, 2013]
    assert {seal: tuple(names) for seal, names in ledger.SEALS.items()} == SEALS
    assert {seal: tuple(splits) for seal, splits in scenarios.SEALS.items()} == SEALS
    assert tuple(ledger.SEALS[2012]) == ledger.TEST_SPLITS


def test_a_final_without_a_seal_is_the_2012_seal(tmp_path):
    """A final-registered row written without seal (every row before #71) carries no seal key and
    is the 2012 seal's: seal_of gives 2012. seal_spent(rows, 2012) is False until its final-result
    is recorded and True after; seal_spent(rows, 2013) stays False throughout."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    freg = _chain(ledger, path, final_result=False)["final-registered"]

    assert "seal" not in freg
    assert ledger.seal_of(freg) == 2012
    assert ledger.seal_spent(ledger.rows(path=path), 2012) is False
    assert ledger.seal_spent(ledger.rows(path=path), 2013) is False
    ledger.final_result(
        final_registered=freg["row_id"], per_split=_per_split({"a": freg["checkpoints"]}), path=path
    )
    assert ledger.seal_spent(ledger.rows(path=path), 2012) is True
    assert ledger.seal_spent(ledger.rows(path=path), 2013) is False


def test_a_2013_final_registers_and_records_once_2012_is_spent(tmp_path):
    """With the 2012 seal spent (a full chain), final_register(seal=2013) is accepted: its row has
    exactly the final-registered fields plus seal, the int 2013, and seal_of gives 2013. Its
    final-result, a per_split of exactly the 2013 splits, is accepted and has exactly the
    final-result fields (the seal is its parent's). Both seals are then spent; each line is the
    row append wrote."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _chain(ledger, path)["result"]["selected"]
    before = path.read_bytes()

    freg, fres = _spend_2013(ledger, path, checkpoints)

    assert set(freg) == BASE_FIELDS | KIND_FIELDS["final-registered"] | {"seal"}
    assert type(freg["seal"]) is int and freg["seal"] == 2013
    assert ledger.seal_of(freg) == 2013
    assert set(fres) == BASE_FIELDS | KIND_FIELDS["final-result"]
    assert list(fres["per_split"]) == list(SPLITS_2013)
    assert all(set(by_regime) == set(REGIMES) for by_regime in fres["per_split"].values())
    assert path.read_bytes() == before + _line(freg) + _line(fres)
    rows = ledger.rows(path=path)
    assert ledger.seal_spent(rows, 2012) is True
    assert ledger.seal_spent(rows, 2013) is True


@pytest.mark.parametrize(
    ("spent", "seal"),
    [
        pytest.param((2012,), 2012, id="seal=2012 once 2012 is spent"),
        pytest.param((2012, 2013), 2013, id="seal=2013 once 2013 is spent"),
    ],
)
def test_final_register_is_refused_once_its_seal_is_spent(tmp_path, spent, seal):
    """A spent seal takes no new final: final_register(seal=<a spent seal>) raises ValueError "the
    seal is spent", the file byte-identical. (Without seal, on a spent 2012,
    test_final_register_is_refused_once_the_seal_is_spent shows the same.)"""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _chain(ledger, path)["result"]["selected"]
    if 2013 in spent:
        _spend_2013(ledger, path, checkpoints)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="the seal is spent"):
        _final_register(ledger, path, checkpoints, issue=72, seal=seal)
    assert path.read_bytes() == before


@pytest.mark.parametrize("seal", [2014, "2013", True], ids=["2014", "the string 2013", "True"])
@pytest.mark.parametrize("via", ["final_register", "append"])
def test_an_unknown_seal_is_refused(tmp_path, seal, via):
    """A seal is an int key of SEALS. 2014 (no such seal), the string "2013" and True (a bool,
    though True == 1) raise ValueError naming seal, the file byte-identical, through
    final_register(seal=...) and through append of a hand-built final-registered row (the 2012
    row's copy, with the seal added). The same row with seal 2013 is then accepted."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    chain = _chain(ledger, path)
    checkpoints = chain["result"]["selected"]
    if via == "final_register":
        before = path.read_bytes()
        with pytest.raises(ValueError, match="seal"):
            _final_register(ledger, path, checkpoints, issue=71, seal=seal)
        assert path.read_bytes() == before
        assert _final_register(ledger, path, checkpoints, issue=71, seal=2013)["seal"] == 2013
    else:
        row = {**chain["final-registered"], "row_id": "final-registered-0badc0de", "issue": 71}
        _check_rejected(ledger, path, {**row, "seal": seal}, {**row, "seal": 2013})


PER_SPLIT_SEAL_FAULTS = [
    pytest.param(2013, SPLITS, "amsterdam_2013", id="a 2013 final with the 2012 splits"),
    pytest.param(
        2013, (*SPLITS_2013, "london_2012"), "london_2012", id="a 2013 final with a 2012 split too"
    ),
    pytest.param(2012, SPLITS_2013, "amsterdam_2012", id="a 2012 final with the 2013 splits"),
]


@pytest.mark.parametrize(("seal", "splits", "named"), PER_SPLIT_SEAL_FAULTS)
def test_a_final_results_per_split_holds_exactly_its_seals_splits(tmp_path, seal, splits, named):
    """per_split is keyed by exactly the seal's splits (ledger.SEALS[seal]), the seal its
    final-registered row's (absent: 2012). A 2013 final's per_split of the 2012 splits, or of the
    2013 splits plus london_2012, and a 2012 final's of the 2013 splits raise ValueError naming the
    split at fault, the file byte-identical; its own seal's splits are then accepted. (#72: a 2013
    final's per_split, wrong or right, nests each split by price regime; a 2012 one stays flat.)"""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    freg = _final_register(ledger, path, checkpoints, **({"seal": 2013} if seal == 2013 else {}))
    regimes = REGIMES if seal == 2013 else None
    before = path.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        ledger.final_result(
            final_registered=freg["row_id"],
            per_split=_per_split({"a": checkpoints}, splits=splits, regimes=regimes),
            path=path,
        )
    assert named in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before

    good = _per_split({"a": checkpoints}, splits=SEALS[seal], regimes=regimes)
    fres = ledger.final_result(final_registered=freg["row_id"], per_split=good, path=path)
    assert ledger.rows(kind="final-result", path=path) == [fres]


@pytest.mark.parametrize("second_on", ["the same 2013 row", "another 2013 row"])
def test_a_second_2013_final_result_is_refused(tmp_path, second_on):
    """The 2013 seal is spent once, like 2012's: with 2012 spent and two 2013 final-registered rows
    written, a 2013 final-result under one of them makes any further final-result under either
    raise ValueError naming final-result, the file byte-identical: two final-results in all."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _chain(ledger, path)["result"]["selected"]
    first = _final_register(ledger, path, checkpoints, issue=71, seal=2013)
    other = _final_register(ledger, path, checkpoints, issue=72, seal=2013)
    per_split = _per_split({"a": checkpoints}, splits=SPLITS_2013, regimes=REGIMES)
    ledger.final_result(final_registered=first["row_id"], per_split=per_split, path=path)
    target = first if second_on == "the same 2013 row" else other
    before = path.read_bytes()

    with pytest.raises(ValueError, match="final-result"):
        ledger.final_result(final_registered=target["row_id"], per_split=per_split, path=path)
    assert path.read_bytes() == before
    assert len(ledger.rows(kind="final-result", path=path)) == 2


def test_spending_2013_leaves_an_unspent_2012_open(tmp_path):
    """Seals are independent: on a ledger where no final has run, a 2013 final-registered and its
    final-result spend 2013 only. seal_spent(rows, 2012) stays False, and a 2012 final (no seal)
    then registers and records its final-result of the 2012 splits."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    _spend_2013(ledger, path, checkpoints)
    assert ledger.seal_spent(ledger.rows(path=path), 2012) is False

    freg = _final_register(ledger, path, checkpoints, issue=9)
    fres = ledger.final_result(
        final_registered=freg["row_id"], per_split=_per_split({"a": checkpoints}), path=path
    )
    assert ledger.rows(kind="final-result", path=path)[-1] == fres
    assert ledger.seal_spent(ledger.rows(path=path), 2012) is True


def test_spending_2013_leaves_a_spent_2012_spent(tmp_path):
    """2012 stays spent whatever 2013 does: after 2012's final-result and then 2013's, a final
    without seal raises ValueError "the seal is spent", the file byte-identical, and both seals
    read as spent."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _chain(ledger, path)["result"]["selected"]
    _spend_2013(ledger, path, checkpoints)
    before = path.read_bytes()

    with pytest.raises(ValueError, match="the seal is spent"):
        _final_register(ledger, path, checkpoints, issue=10)
    assert path.read_bytes() == before
    rows = ledger.rows(path=path)
    assert ledger.seal_spent(rows, 2012) is True
    assert ledger.seal_spent(rows, 2013) is True


def test_cli_final_register_takes_a_seal(tmp_path, capsys):
    """final-register --seal <year> passes an int seal through the ledger's own checks: on a ledger
    where 2012 is spent, --seal 2014 raises ValueError naming seal, the file byte-identical; --seal
    2013 prints the new row's id, and that row's seal is the int 2013."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _chain(ledger, path)["result"]["selected"]
    argv = [
        "final-register",
        *("--issue", "71"),
        *("--checkpoints", json.dumps(checkpoints)),
        *("--baselines", "rb_default", "rb_tuned"),
        *("--approved-by", "Aarmen"),
    ]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="seal"):
        ledger.main([*argv, "--seal", "2014"], path=path)
    assert path.read_bytes() == before

    capsys.readouterr()
    assert ledger.main([*argv, "--seal", "2013"], path=path) == 0
    row_id = capsys.readouterr().out.strip()
    row = ledger.rows(kind="final-registered", path=path)[-1]
    assert row["row_id"] == row_id
    assert type(row["seal"]) is int and row["seal"] == 2013


@pytest.mark.parametrize("via", ["api", "cli"])
def test_show_names_a_final_registered_rows_seal(tmp_path, capsys, via):
    """show prints a final-registered row's seal on its line: "seal 2013" for a row written with
    seal=2013, and "seal 2012" for a row without one (every row before #71; seal_of gives 2012).
    Issue 7's experiment, then issue 9's final without seal and issue 71's with seal 2013, neither
    spent: each final's one line names its own seal and not the other's."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    old = _final_register(ledger, path, checkpoints, issue=9)
    new = _final_register(ledger, path, checkpoints, issue=71, seal=2013)
    capsys.readouterr()

    _show(ledger, path, via)
    lines = capsys.readouterr().out.splitlines()

    for row, seal, other in ((old, 2012, 2013), (new, 2013, 2012)):
        [line] = [line for line in lines if row["row_id"] in line]
        assert re.search(rf"\bseal {seal}\b", line), line
        assert f"seal {other}" not in line, line


# --- Issue #72: a 2013 final is graded at both price regimes -------------------------------------


def test_ledger_price_regimes_are_pinned_to_metrics():
    """ledger.PRICE_REGIMES, written out so the ledger stays stdlib-only, is
    tuple(metrics.PRICE_REGIMES): ("sim", "agc2020"), the keys of each split of a 2013 final's
    per_split. This module's REGIMES copy is pinned too."""
    import greenleaf_control.ledger as ledger
    from greenleaf_control import metrics

    assert getattr(ledger, "PRICE_REGIMES", None) == tuple(metrics.PRICE_REGIMES)
    assert tuple(metrics.PRICE_REGIMES) == REGIMES


def _flat(per_split: dict, split: str | None = None) -> None:
    """Flatten per_split in place to #71's shape, {split: {arm: entry}} (each split's sim entries),
    for every split or only split."""
    for name in [split] if split else list(per_split):
        per_split[name] = per_split[name]["sim"]


# A 2013 per_split fault, laid into the last split and, where it is one regime's, its agc2020 (so a
# check of the first split, or of sim alone, misses it), and the names the refusal must give.
PER_SPLIT_REGIME_FAULTS = [
    pytest.param("a flat per_split", ["agc2020"], id="a flat per_split (#71's shape)"),
    pytest.param("a flat last split", ["abbotsford_2013", "agc2020"], id="a flat last split"),
    pytest.param("a missing regime", ["abbotsford_2013", "agc2020"], id="a missing regime"),
    pytest.param("an extra regime", ["abbotsford_2013", "agc2024"], id="an extra regime"),
    pytest.param(
        "a missing arm in agc2020",
        ["abbotsford_2013", "agc2020", "rb_default"],
        id="a missing arm in one regime",
    ),
    pytest.param(
        "rb_tuned's prob_improvement in agc2020",
        ["abbotsford_2013", "agc2020", "prob_improvement", "rb_tuned"],
        id="prob_improvement in one regime",
    ),
    pytest.param(
        "a baseline scored in agc2020",
        ["abbotsford_2013", "agc2020", "scored", "rb_default"],
        id="a baseline's scored not {} in one regime",
    ),
]


@pytest.mark.parametrize(("fault", "named"), PER_SPLIT_REGIME_FAULTS)
def test_a_2013_per_split_needs_both_regimes_in_every_split(tmp_path, fault, named):
    """A 2013 final's per_split is {split: {regime: {arm: entry}}}: each split holds exactly
    PRICE_REGIMES, sim and agc2020, and the arm checks (exactly the registered arms plus the
    baselines, FINAL_ARM_KEYS, prob_improvement null exactly for rb_tuned, scored as registered)
    run in every (split, regime).

    A seal-2013 final of arm a (issue 7's run at "final") with rb_default and rb_tuned is
    registered. Each fault raises ValueError naming its cause (the split at fault and the regime,
    or the arm and key), the file byte-identical: the whole per_split flat (#71's shape), the last
    split flat, the last split without agc2020, the last split with a third regime agc2024, and in
    the last split's agc2020 alone, rb_default missing, rb_tuned's prob_improvement 0.5, or
    rb_default scored at arm a's checkpoint. The faithful nested per_split is then accepted."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    freg = _final_register(ledger, path, checkpoints, issue=72, seal=2013)
    per_split = _per_split({"a": checkpoints}, splits=SPLITS_2013, regimes=REGIMES)
    broken = copy.deepcopy(per_split)
    last = broken["abbotsford_2013"]
    {
        "a flat per_split": lambda: _flat(broken),
        "a flat last split": lambda: _flat(broken, "abbotsford_2013"),
        "a missing regime": lambda: last.pop("agc2020"),
        "an extra regime": lambda: last.update(agc2024=copy.deepcopy(last["agc2020"])),
        "a missing arm in agc2020": lambda: last["agc2020"].pop("rb_default"),
        "rb_tuned's prob_improvement in agc2020": lambda: last["agc2020"]["rb_tuned"].update(
            prob_improvement=0.5
        ),
        "a baseline scored in agc2020": lambda: last["agc2020"]["rb_default"].update(
            scored=dict(checkpoints)
        ),
    }[fault]()
    before = path.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        ledger.final_result(final_registered=freg["row_id"], per_split=broken, path=path)
    for name in named:
        assert name in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before

    fres = ledger.final_result(final_registered=freg["row_id"], per_split=per_split, path=path)
    assert ledger.rows(kind="final-result", path=path) == [fres]


def test_a_2012_per_split_stays_flat(tmp_path):
    """The 2012 path is unchanged (#72): a 2012 final's per_split nested by price regime, as a 2013
    one is, raises ValueError naming its first split, the file byte-identical; the flat per_split is
    then accepted. (A guard: it passes before #72 too, and must after.)"""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    freg = _final_register(ledger, path, checkpoints)
    before = path.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        ledger.final_result(
            final_registered=freg["row_id"],
            per_split=_per_split({"a": checkpoints}, regimes=REGIMES),
            path=path,
        )
    assert "amsterdam_2012" in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before

    fres = ledger.final_result(
        final_registered=freg["row_id"], per_split=_per_split({"a": checkpoints}), path=path
    )
    assert ledger.rows(kind="final-result", path=path) == [fres]


def test_a_2013_final_registers_rb_tuned_agc2020_and_grades_it_in_each_regime(tmp_path):
    """A 2013 final may register rb_tuned_agc2020, the rules tune_rb tuned at agc2020's prices
    (configs/rule_based_tuned_agc2020.yml), beside rb_default and rb_tuned: with 2012 spent,
    final_register(seal=2013, baselines=[rb_default, rb_tuned, rb_tuned_agc2020]) is accepted with
    those baselines in that order, and its final-result, rb_tuned_agc2020 a baseline in every
    (split, regime) (scored {}, prob_improvement a number), is accepted as given."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _chain(ledger, path)["result"]["selected"]
    baselines = ("rb_default", "rb_tuned", "rb_tuned_agc2020")

    freg = _final_register(ledger, path, checkpoints, issue=72, baselines=baselines, seal=2013)

    assert freg["baselines"] == list(baselines)
    per_split = _per_split(
        {"a": checkpoints}, baselines=baselines, splits=SPLITS_2013, regimes=REGIMES
    )
    fres = ledger.final_result(final_registered=freg["row_id"], per_split=per_split, path=path)
    assert fres["per_split"] == per_split
    assert ledger.seal_spent(ledger.rows(path=path), 2013) is True


def test_register_dry_returns_the_row_and_writes_nothing(tmp_path):
    """register(dry=True) returns a complete, stamped registered row and creates no file."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    row = _register(ledger, path, dry=True)

    assert not path.exists()
    assert ledger.rows(path=path) == []
    assert set(row) == BASE_FIELDS | KIND_FIELDS["registered"]
    assert row["kind"] == "registered"
    assert re.fullmatch(r"registered-[0-9a-f]{8}", row["row_id"])
    assert row["issue"] == 3
    assert row["owner_ok"] is False


# A primary or guard selection can't score: (field, value, the name the refusal must give).
NOT_RESULT_KEYS = [
    pytest.param("primary", "profit", "profit", id="primary-profit"),
    pytest.param("primary", "valid", "valid", id="primary-the-valid-flag"),
    pytest.param("primary", "paper_reward", "paper_reward", id="primary-an-info-key"),
    pytest.param("guards", ["violation_hours"], "violation_hours", id="guard-violation-hours"),
    pytest.param(
        "guards", ["viol_hours_temp", "lamp_hours"], "lamp_hours", id="one-bad-guard-of-two"
    ),
]


@pytest.mark.parametrize("dry", [False, True])
@pytest.mark.parametrize(("field", "value", "name"), NOT_RESULT_KEYS)
def test_register_refuses_a_primary_or_guard_that_is_not_a_result_key(
    tmp_path, field, value, name, dry
):
    """register, dry or not, raises ValueError when the primary or any guard isn't one of a scored
    season's numeric keys (selection.RESULT_KEYS), naming the bad key by its repr ('profit', as the
    ledger's other refusals quote a value). A typo is caught at register time, not after the
    seasons are played. "valid" is episode_metrics' flag, not a number; "paper_reward" is the
    per-step info key, not the season's paper_reward_sum.

    The ledger, which already holds one registration, is byte-identical afterwards.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    _register(ledger, path, issue=4)
    before = path.read_bytes()

    with pytest.raises(ValueError) as excinfo:
        _register(ledger, path, dry=dry, **{field: value})
    assert repr(name) in str(excinfo.value), str(excinfo.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("flag", "values"),
    [("--primary", ["profit"]), ("--guards", ["viol_hours_temp", "violation_hours"])],
    ids=["primary-profit", "guard-violation-hours"],
)
def test_cli_register_refuses_a_primary_or_guard_that_is_not_a_result_key(
    tmp_path, capsys, flag, values
):
    """Through main, register --primary profit (or a guard violation_hours) is refused: ValueError,
    or an argparse error (SystemExit, non-zero). The ledger, holding one registration, is
    byte-identical and nothing names a new row."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    assert ledger.main(_register_argv(issue=4), path=path) == 0
    before = path.read_bytes()
    capsys.readouterr()
    argv = _register_argv(flag, *values)  # argparse keeps a repeated option's last value

    try:
        code = ledger.main(argv, path=path)
    except (ValueError, SystemExit) as e:
        if isinstance(e, SystemExit):
            assert e.code not in (0, None), f"main exited {e.code!r}, which is not a refusal"
    else:
        assert code != 0, "main accepted a primary or guard that isn't a result key"
    assert path.read_bytes() == before
    assert "registered-" not in capsys.readouterr().out


def test_register_accepts_e1s_registration(tmp_path):
    """E1's real registration (registered-b89a61b1, issue #22) still validates: arm rb_tuned
    (method rb_tuned, {"trials": 200}), seed 0, 0 timesteps, scenarios ["validation"], primary
    paper_return, guards viol_hours_co2, viol_hours_temp and viol_hours_rh, local at 0 USD with a
    3 CPU-hour cap. It is written as given."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    row = ledger.register(
        issue=22,
        hypothesis="tuning the RB's setpoints raises its validation paper_return",
        arms=[{"name": "rb_tuned", "method": "rb_tuned", "config": {"trials": 200}}],
        seeds=[0],
        timesteps=0,
        scenarios=["validation"],
        primary="paper_return",
        guards=["viol_hours_co2", "viol_hours_temp", "viol_hours_rh"],
        rule="adopt if (a)-(d) of #22 hold",
        budget={"cpu_hours": 3.0, "usd": 0.0, "backend": "local"},
        path=path,
    )

    assert ledger.rows(path=path) == [row]
    assert row["primary"] == "paper_return"
    assert row["guards"] == ["viol_hours_co2", "viol_hours_temp", "viol_hours_rh"]


def test_register_accepts_every_result_key(tmp_path):
    """register refuses only what selection can't score: each of selection.RESULT_KEYS is accepted
    as the primary, with all of them as guards (one registration per issue, so none needs
    owner_ok)."""
    import greenleaf_control.ledger as ledger
    import greenleaf_control.selection as selection

    path = tmp_path / "ledger.jsonl"
    keys = list(selection.RESULT_KEYS)
    for issue, key in enumerate(keys, start=100):
        _register(ledger, path, issue=issue, primary=key, guards=keys)

    written = ledger.rows(kind="registered", path=path)
    assert [r["primary"] for r in written] == keys
    assert all(r["guards"] == keys for r in written)


def test_rows_filters_by_kind_and_issue(tmp_path):
    """rows() is every row in file order (a missing file gives []); kind= and issue= filter."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    assert ledger.rows(path=path) == []

    a = _register(ledger, path, issue=3, arms=("a",))
    b = _register(ledger, path, issue=4, arms=("a",))
    run = ledger.run_start(registered=a["row_id"], arm="a", seed=0, run_dir="runs/a", path=path)

    assert ledger.rows(path=path) == [a, b, run]
    assert ledger.rows(kind="registered", path=path) == [a, b]
    assert ledger.rows(issue=3, path=path) == [a, run]
    assert ledger.rows(kind="run", issue=3, path=path) == [run]
    assert ledger.rows(kind="registered", issue=4, path=path) == [b]
    assert ledger.rows(kind="run", issue=4, path=path) == []


@pytest.mark.parametrize("via", ["api", "cli"])
def test_show_prints_one_line_per_row_even_for_a_multiline_note(tmp_path, capsys, via):
    """show prints exactly one line per row, holding its row_id, kind and issue.

    The decision's note spans lines (\\n, \\r\\n, a tab); whitespace is collapsed, so it still
    prints on one line.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    reg = _register(ledger, path, issue=3, arms=("a",))
    run, end = _run(ledger, path, reg, "a", 0)
    res = ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["a"]), selected={}, path=path
    )
    note = "line one\nline two\r\n\tline three\n"
    dec = ledger.decide(result=res["row_id"], decision="dropped", note=note, path=path)
    other = _register(ledger, path, issue=812, arms=("a",))
    written = [reg, run, end, res, dec, other]
    capsys.readouterr()

    _show(ledger, path, via)
    lines = capsys.readouterr().out.splitlines()

    assert len(lines) == len(written)
    for row in written:
        # A line may also name the row it links to, so look for one holding all three.
        assert any(
            row["row_id"] in line and row["kind"] in line and str(row["issue"]) in line
            for line in lines
        ), row["row_id"]


@pytest.mark.parametrize("via", ["api", "cli"])
def test_show_filters_by_issue(tmp_path, capsys, via):
    """show(issue=N) prints only issue N's rows, one line each."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    a = _register(ledger, path, issue=3)
    b = _register(ledger, path, issue=812)
    c = _register(ledger, path, issue=3)
    capsys.readouterr()

    _show(ledger, path, via, issue=3)
    out = capsys.readouterr().out

    assert len(out.splitlines()) == 2
    assert a["row_id"] in out
    assert c["row_id"] in out
    assert b["row_id"] not in out


@pytest.mark.parametrize(
    ("issue", "expected"),
    [(None, "spend_usd 3.75"), (3, "spend_usd 1.75"), (812, "spend_usd 2.00")],
    ids=["all issues", "issue 3", "issue 812"],
)
@pytest.mark.parametrize("via", ["api", "cli"])
def test_show_spend_sums_cost_usd_over_run_end_rows(tmp_path, capsys, via, issue, expected):
    """--spend prints one line, spend_usd <total:.2f>: the sum of cost_usd over run-end rows.

    Issue 3's run-ends cost 0.5, None and 1.25 USD: 0.5 + 0 + 1.25 = 1.75 USD. Issue 812's one
    run-end costs 2.0 USD, so all issues give 1.75 + 2.0 = 3.75 USD. The registrations' budget usd
    (5.0 each) and issue 3's result, whose per-arm spend_usd is 1.75, are not spend.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    budget = {"cpu_hours": 1.0, "usd": 5.0, "backend": "modal"}
    reg3 = _register(ledger, path, issue=3, arms=("a",), seeds=(0, 1, 2), budget=budget)
    for seed, cost in [(0, 0.5), (1, None), (2, 1.25)]:
        _run(ledger, path, reg3, "a", seed, cost_usd=cost)
    ledger.record_result(registered=reg3["row_id"], per_arm=_per_arm(["a"]), selected={}, path=path)
    reg812 = _register(ledger, path, issue=812, arms=("a",), budget=budget)
    _run(ledger, path, reg812, "a", 0, cost_usd=2.0)
    capsys.readouterr()

    _show(ledger, path, via, issue=issue, spend=True)

    assert capsys.readouterr().out.splitlines() == [expected]


@pytest.mark.parametrize(("flag", "decision"), [("--adopted", "adopted"), ("--dropped", "dropped")])
def test_cli_register_result_decide_round_trip(tmp_path, capsys, flag, decision):
    """register, result and decide through main each print a row_id that is then in rows().

    register parses its flags into the contract's types (JSON arms, int seeds and timesteps, a
    budget dict); runs go through the Python API; result reads its numbers from a JSON file.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    assert ledger.main(_register_argv(), path=path) == 0
    reg_id = capsys.readouterr().out.strip()
    assert re.fullmatch(r"registered-[0-9a-f]{8}", reg_id), reg_id
    [reg] = ledger.rows(kind="registered", path=path)
    assert reg["row_id"] == reg_id
    assert reg["issue"] == 3
    assert {k: reg[k] for k in KIND_FIELDS["registered"]} == CLI_REGISTERED
    assert type(reg["timesteps"]) is int
    assert all(type(seed) is int for seed in reg["seeds"])

    runs = [_run(ledger, path, reg, arm, seed)[0] for arm in ("a", "b") for seed in (0, 1)]
    numbers = tmp_path / "numbers.json"
    selected = {run["row_id"]: "ckpt_final.zip" for run in runs}
    numbers.write_text(json.dumps({"per_arm": _per_arm(["a", "b"]), "selected": selected}))
    capsys.readouterr()
    assert (
        ledger.main(["result", "--registered", reg_id, "--numbers", str(numbers)], path=path) == 0
    )
    res_id = capsys.readouterr().out.strip()
    [res] = ledger.rows(kind="result", path=path)
    assert res["row_id"] == res_id
    assert res["registered"] == reg_id
    assert res["selected"] == selected
    assert res["per_arm"] == {
        name: {**given, "crashed": 0, "spend_usd": 0.0}
        for name, given in _per_arm(["a", "b"]).items()
    }

    argv = ["decide", "--result", res_id, flag, "--note", "IQM CI excludes 0"]
    assert ledger.main(argv, path=path) == 0
    dec_id = capsys.readouterr().out.strip()
    [dec] = ledger.rows(kind="decision", path=path)
    assert dec["row_id"] == dec_id
    assert {k: dec[k] for k in KIND_FIELDS["decision"]} == {
        "result": res_id,
        "decision": decision,
        "note": "IQM CI excludes 0",
    }
    assert dec["issue"] == 3


def test_cli_register_dry_prints_the_row_and_writes_nothing(tmp_path, capsys):
    """register --dry prints the row as JSON and creates no file."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    assert ledger.main([*_register_argv(), "--dry"], path=path) == 0
    row = json.loads(capsys.readouterr().out)

    assert not path.exists()
    assert row["kind"] == "registered"
    assert row["issue"] == 3
    assert re.fullmatch(r"registered-[0-9a-f]{8}", row["row_id"])
    assert {k: row[k] for k in KIND_FIELDS["registered"]} == CLI_REGISTERED


def test_cli_third_register_needs_the_owner_ok_flag(tmp_path, capsys):
    """Through main, a third registration on one issue raises ValueError naming owner_ok;
    with --owner-ok it is written with owner_ok True."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    assert ledger.main(_register_argv(), path=path) == 0
    assert ledger.main(_register_argv(), path=path) == 0
    before = path.read_bytes()

    with pytest.raises(ValueError, match="owner_ok"):
        ledger.main(_register_argv(), path=path)
    assert path.read_bytes() == before

    capsys.readouterr()
    assert ledger.main(_register_argv("--owner-ok"), path=path) == 0
    row_id = capsys.readouterr().out.strip()
    third = ledger.rows(kind="registered", path=path)[-1]
    assert third["row_id"] == row_id
    assert third["owner_ok"] is True


def test_cli_final_register_needs_approved_by(tmp_path, capsys):
    """final-register without --approved-by reaches the ledger's own refusal (ValueError naming
    approved_by, the file byte-identical); with --approved-by Aarmen it prints the new row's id.
    It registers issue 7's validation-selected checkpoint."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    argv = [
        "final-register",
        *("--issue", "9"),
        *("--checkpoints", json.dumps(checkpoints)),
        *("--baselines", "rb_default", "rb_tuned"),
    ]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="approved_by"):
        ledger.main(argv, path=path)
    assert path.read_bytes() == before

    capsys.readouterr()
    assert ledger.main([*argv, "--approved-by", "Aarmen"], path=path) == 0
    row_id = capsys.readouterr().out.strip()
    [row] = ledger.rows(kind="final-registered", path=path)
    assert row["row_id"] == row_id
    assert row["issue"] == 9
    assert row["checkpoints"] == checkpoints
    assert row["baselines"] == ["rb_default", "rb_tuned"]
    assert row["approved_by"] == "Aarmen"
    _utc(row["approved_at"])


def test_cli_final_result_records_the_numbers_file(tmp_path, capsys):
    """A valid chain records both rows through main. final-register prints its row id; then
    final-result --final-registered <row> --numbers <file>, the file exactly {"per_split": ...},
    prints the final-result's row id. That row links the final-registered row, takes its issue 9,
    and its per_split equals the file's."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    checkpoints = _experiment(ledger, path)["result"]["selected"]
    capsys.readouterr()
    argv = [
        "final-register",
        *("--issue", "9"),
        *("--checkpoints", json.dumps(checkpoints)),
        *("--baselines", "rb_default", "rb_tuned"),
        *("--approved-by", "Aarmen"),
    ]
    assert ledger.main(argv, path=path) == 0
    freg_id = capsys.readouterr().out.strip()
    numbers = tmp_path / "numbers.json"
    numbers.write_text(json.dumps({"per_split": _per_split({"a": checkpoints})}))

    argv = ["final-result", "--final-registered", freg_id, "--numbers", str(numbers)]
    assert ledger.main(argv, path=path) == 0
    fres_id = capsys.readouterr().out.strip()

    [freg] = ledger.rows(kind="final-registered", path=path)
    [fres] = ledger.rows(kind="final-result", path=path)
    assert freg["row_id"] == freg_id
    assert fres["row_id"] == fres_id
    assert fres["final-registered"] == freg_id
    assert fres["issue"] == 9
    assert fres["per_split"] == json.loads(numbers.read_text())["per_split"]


def test_cli_final_result_refuses_a_numbers_file_with_extra_keys(tmp_path, capsys):
    """The numbers file is exactly {"per_split": ...}. One that also carries selected, as a
    result's numbers file does, raises ValueError naming selected, the file byte-identical; the
    exact file is then accepted and its row id printed."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    freg = _chain(ledger, path, final_result=False)["final-registered"]
    per_split = _per_split({"a": freg["checkpoints"]})
    numbers = tmp_path / "numbers.json"
    numbers.write_text(json.dumps({"per_split": per_split, "selected": {}}))
    argv = ["final-result", "--final-registered", freg["row_id"], "--numbers", str(numbers)]
    before = path.read_bytes()

    with pytest.raises(ValueError, match="selected"):
        ledger.main(argv, path=path)
    assert path.read_bytes() == before

    numbers.write_text(json.dumps({"per_split": per_split}))
    capsys.readouterr()
    assert ledger.main(argv, path=path) == 0
    [fres] = ledger.rows(kind="final-result", path=path)
    assert capsys.readouterr().out.strip() == fres["row_id"]


def test_default_path_is_the_real_ledger_and_no_test_wrote_it():
    """LEDGER is <control>/ledger/ledger.jsonl, and every path= argument defaults to it.

    So every test here must pass path= under tmp_path. This test runs last in the module and
    checks the real ledger's size and mtime are as they were when the module was collected.
    """
    import greenleaf_control.ledger as ledger

    assert ledger.LEDGER == REAL_LEDGER
    assert ledger.LEDGER == Path(ledger.__file__).resolve().parents[1] / "ledger" / "ledger.jsonl"
    for name in PATH_FUNCTIONS:
        default = inspect.signature(getattr(ledger, name)).parameters["path"].default
        assert default == ledger.LEDGER, name
    assert _stat(REAL_LEDGER) == REAL_LEDGER_BEFORE

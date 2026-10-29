"""Tests for greenleaf_control.ledger (issue #3): the append-only experiment ledger and its refusals.

Every test passes path= under tmp_path. The defaults bind to the real control/ledger/ledger.jsonl,
which no test may write; the last test in this module checks that.
"""

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
PER_SPLIT = {"heldout": {"iqm": 1.0, "ci_low": 0.5, "ci_high": 1.5}}
CHECKPOINTS = {"run-0123abcd": "ckpt_final.zip"}
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


def _final_register(ledger, path, *, issue=9, approved_by="Aarmen") -> dict:
    return ledger.final_register(
        issue=issue,
        checkpoints=CHECKPOINTS,
        baselines=["rb_default", "rb_tuned"],
        approved_by=approved_by,
        path=path,
    )


def _chain(ledger, path, *, final_result=True) -> dict:
    """One row of every kind, keyed by kind: issue 7's experiment, then issue 9's final evaluation."""
    reg = _register(ledger, path, issue=7, arms=("a",))
    run, end = _run(ledger, path, reg, "a", 0, cost_usd=0.5)
    res = ledger.record_result(
        registered=reg["row_id"],
        per_arm=_per_arm(["a"]),
        selected={run["row_id"]: "ckpt_final.zip"},
        path=path,
    )
    dec = ledger.decide(result=res["row_id"], decision="adopted", note="CI excludes 0", path=path)
    freg = _final_register(ledger, path, issue=9)
    chain = {
        "registered": reg,
        "run": run,
        "run-end": end,
        "result": res,
        "decision": dec,
        "final-registered": freg,
    }
    if final_result:
        chain["final-result"] = ledger.final_result(
            final_registered=freg["row_id"], per_split=PER_SPLIT, path=path
        )
    return chain


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
    assert res["selected"] == {run["row_id"]: "ckpt_final.zip"}
    assert {k: dec[k] for k in KIND_FIELDS["decision"]} == {
        "result": res["row_id"],
        "decision": "adopted",
        "note": "CI excludes 0",
    }
    assert freg["checkpoints"] == CHECKPOINTS
    assert freg["baselines"] == ["rb_default", "rb_tuned"]
    assert freg["approved_by"] == "Aarmen"
    assert t0 <= _utc(freg["approved_at"]) <= t1
    assert fres["final-registered"] == freg["row_id"]
    assert fres["per_split"] == PER_SPLIT


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
        status="crashed",
        wall_s=3.0,
        timesteps=96,
        solver_failures=2,
        cost_usd=None,
        diagnostics=DIAGNOSTICS,
        path=path,
    )
    seen.append(path.read_bytes())
    res = ledger.record_result(
        registered=reg["row_id"], per_arm=_per_arm(["a"]), selected={}, path=path
    )
    seen.append(path.read_bytes())
    ledger.decide(result=res["row_id"], decision="dropped", note="it crashed", path=path)
    seen.append(path.read_bytes())
    _register(ledger, path, arms=("a",))  # the issue's second registration
    seen.append(path.read_bytes())

    with pytest.raises(ValueError):
        _register(ledger, path, arms=("a",))  # a third, without owner_ok
    _register(ledger, path, issue=4, arms=("a",), dry=True)
    assert path.read_bytes() == seen[-1]

    freg = _final_register(ledger, path)
    seen.append(path.read_bytes())
    ledger.final_result(final_registered=freg["row_id"], per_split=PER_SPLIT, path=path)
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


def _link_kwargs(func: str) -> dict:
    """Valid arguments for func besides its link."""
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
        "final_result": {"per_split": PER_SPLIT},
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
        getattr(ledger, func)(**{arg: link}, **_link_kwargs(func), path=path)
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

    Nothing is written; the file isn't even created.
    """
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"

    with pytest.raises(ValueError, match="approved_by"):
        _final_register(ledger, path, approved_by=approved_by)
    assert not path.exists()


@pytest.mark.parametrize("second_on", ["the same final-registered", "another issue's"])
def test_a_second_final_result_is_refused_on_any_issue(tmp_path, second_on):
    """The seal is spent once: after one final-result, any other raises ValueError naming
    final-result, whichever final-registered row it names. Nothing is written."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    freg = _final_register(ledger, path, issue=9)
    ledger.final_result(final_registered=freg["row_id"], per_split=PER_SPLIT, path=path)
    target = (
        freg
        if second_on == "the same final-registered"
        else _final_register(ledger, path, issue=10)
    )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="final-result"):
        ledger.final_result(final_registered=target["row_id"], per_split=PER_SPLIT, path=path)
    assert path.read_bytes() == before
    assert len(ledger.rows(kind="final-result", path=path)) == 1


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
    approved_by, no file); with --approved-by Aarmen it prints the new row's id."""
    import greenleaf_control.ledger as ledger

    path = tmp_path / "ledger.jsonl"
    argv = [
        "final-register",
        *("--issue", "9"),
        *("--checkpoints", json.dumps(CHECKPOINTS)),
        *("--baselines", "rb_default", "rb_tuned"),
    ]

    with pytest.raises(ValueError, match="approved_by"):
        ledger.main(argv, path=path)
    assert not path.exists()

    capsys.readouterr()
    assert ledger.main([*argv, "--approved-by", "Aarmen"], path=path) == 0
    row_id = capsys.readouterr().out.strip()
    [row] = ledger.rows(kind="final-registered", path=path)
    assert row["row_id"] == row_id
    assert row["issue"] == 9
    assert row["checkpoints"] == CHECKPOINTS
    assert row["baselines"] == ["rb_default", "rb_tuned"]
    assert row["approved_by"] == "Aarmen"
    _utc(row["approved_at"])


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

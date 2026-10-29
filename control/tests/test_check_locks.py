"""Tests for scripts/check_locks.py (issue #4): the ledger's history lock.

Each test builds a temp git repo under tmp_path, writes its ledger through the real ledger API
(path= under the temp repo), commits, and runs the script on it as a subprocess with --repo.
No test runs it on this repo (CI clones at depth 1), and none writes the real ledger.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from greenleaf_control import ledger, manifest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_locks.py"
LEDGER = "control/ledger/ledger.jsonl"  # repo-relative, as check_locks reads it from git
# CI has no git identity and may sign commits globally; no hook may run in a temp repo.
COMMIT_CONFIG = (
    *("-c", "user.name=t", "-c", "user.email=t@t"),
    *("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"),
)
REGISTRATION = {
    "issue": 4,
    "hypothesis": "residual PPO beats plain PPO on validation profit",
    "arms": [{"name": "a", "method": "ppo", "config": {}}],
    "seeds": [0],
    "timesteps": 20_000,
    "scenarios": ["validation"],
    "primary": "profit_eur_m2",
    "guards": ["viol_hours_temp"],
    "rule": "adopt if the profit IQM CI excludes 0",
    "budget": {"cpu_hours": 1.0, "usd": 0.0, "backend": "local"},
}
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}
PER_SPLIT = {"heldout": {"iqm": 1.0, "ci_low": 0.5, "ci_high": 1.5}}
SECOND_FINAL = "final-result-0badc0de"


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], check=True, cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    """Stage every change, deletions included, and commit; return the new commit's full SHA."""
    _git(repo, "add", "-A")
    _git(repo, *COMMIT_CONFIG, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_repo(tmp_path: Path, monkeypatch) -> Path:
    """A temp git repo with docs/PROTOCOL.md and control/ledger/.gitkeep committed, no ledger yet.

    manifest.REPO_ROOT points at it, so each row's fingerprints hash the temp repo's files.
    """
    # A git hook's environment would point these git calls, and the script's, at the real repo.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    (repo / "control" / "ledger").mkdir(parents=True)  # ledger.append doesn't mkdir
    (repo / "control" / "ledger" / ".gitkeep").write_text("")
    (repo / "docs").mkdir()
    (repo / "docs" / "PROTOCOL.md").write_text("protocol v1\n")
    _git(repo, "init", "-q")
    _commit(repo, "init")
    monkeypatch.setattr(manifest, "REPO_ROOT", repo)
    return repo


def _check_locks(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_clean(out: subprocess.CompletedProcess) -> None:
    """Exit 0 and exactly one stdout line, starting "check_locks: clean"."""
    assert out.returncode == 0, out.stdout + out.stderr
    lines = out.stdout.splitlines()
    assert len(lines) == 1, out.stdout
    assert lines[0].startswith("check_locks: clean"), out.stdout


def _violations(out: subprocess.CompletedProcess) -> list[str]:
    """Assert exit 1 and return the stdout lines, one per violation."""
    assert out.returncode == 1, out.stdout + out.stderr
    return out.stdout.splitlines()


def _line(row: dict) -> str:
    """The exact text ledger.append writes for one row."""
    return json.dumps(row, sort_keys=True, allow_nan=False) + "\n"


def _run(path: Path, registered: dict) -> tuple[dict, dict]:
    """Start and end arm a's seed-0 run of a registration; return (run row, run-end row)."""
    run = ledger.run_start(
        registered=registered["row_id"], arm="a", seed=0, run_dir="runs/a-0", path=path
    )
    end = ledger.run_end(
        run=run["row_id"],
        status="finished",
        wall_s=12.5,
        timesteps=20_000,
        solver_failures=0,
        cost_usd=None,
        diagnostics=DIAGNOSTICS,
        path=path,
    )
    return run, end


def _result(path: Path, registered: dict, run: dict) -> dict:
    per_arm = {
        "a": {
            "iqm": 1.5,
            "ci_low": 1.0,
            "ci_high": 2.0,
            "guards": {"viol_hours_temp": 3.0},
            "invalid_episodes": 0,
        }
    }
    return ledger.record_result(
        registered=registered["row_id"],
        per_arm=per_arm,
        selected={run["row_id"]: "ckpt_final.zip"},
        path=path,
    )


def _final(path: Path) -> tuple[dict, dict]:
    """Register the final evaluation and spend the seal once; return (final-registered, final-result)."""
    freg = ledger.final_register(
        issue=9,
        checkpoints={"run-0123abcd": "ckpt_final.zip"},
        baselines=["rb_default", "rb_tuned"],
        approved_by="Aarmen",
        path=path,
    )
    fres = ledger.final_result(final_registered=freg["row_id"], per_split=PER_SPLIT, path=path)
    return freg, fres


def _clean_history(repo: Path, path: Path) -> list[str]:
    """register → run + run-end → result → decide, one commit each; return the 4 commit SHAs."""
    reg = ledger.register(**REGISTRATION, path=path)
    shas = [_commit(repo, "register")]
    run, _ = _run(path, reg)
    shas.append(_commit(repo, "run and run-end"))
    res = _result(path, reg, run)
    shas.append(_commit(repo, "result"))
    ledger.decide(result=res["row_id"], decision="adopted", note="CI excludes 0", path=path)
    shas.append(_commit(repo, "decide"))
    return shas


@pytest.mark.parametrize("final", [False, True], ids=["experiment only", "seal spent once"])
def test_a_clean_history_exits_0_with_one_summary_line(tmp_path, monkeypatch, final):
    """register → run → run-end → result → decide, committed step by step, exits 0.

    Every commit only appends; the result names the earlier registration with equal fingerprints
    (docs/PROTOCOL.md never changes); the run has its run-end. With final, one final-result names
    the earlier final-registered. stdout is one line, starting "check_locks: clean".
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    _clean_history(repo, path)
    if final:
        _final(path)
        _commit(repo, "final-register and final-result")

    _assert_clean(_check_locks(repo))


def test_an_edited_line_in_a_past_commit_exits_1_naming_that_commit(tmp_path, monkeypatch):
    """Commit X rewrites the registered line in place; X is named even after a clean commit Y.

    X changes only the hypothesis (row_id and fingerprints untouched), so every HEAD check passes:
    the one violation is X's file not starting with its parent's bytes. Y then only appends a run
    and its run-end. stdout is one line naming X's 12-hex short SHA, and not Y's: blaming the
    next commit is the off-by-one this catches. (A line may also name X's parent.)
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    reg = ledger.register(**REGISTRATION, path=path)
    _commit(repo, "register")
    path.write_text(_line({**reg, "hypothesis": "plain PPO beats residual PPO"}))
    edited = _commit(repo, "edit the registration in place")
    _run(path, reg)
    appended = _commit(repo, "run and run-end")

    out = _check_locks(repo)

    lines = _violations(out)
    assert len(lines) == 1, out.stdout
    assert edited[:12] in lines[0]
    assert appended[:12] not in out.stdout


def test_a_deleted_line_exits_1_naming_that_commit(tmp_path, monkeypatch):
    """Commit Z drops the last row (the decision) from a clean history; Z is named.

    Z's file is a prefix of its parent's, the reverse of append-only, so a check comparing the
    wrong way round would pass it. No HEAD check needs a decision, so Z is the one violation, and
    the clean commits before Z's parent are not named. (A line may also name Z's parent.)
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    clean = _clean_history(repo, path)
    rows = path.read_bytes().splitlines(keepends=True)
    assert json.loads(rows[-1])["kind"] == "decision"
    path.write_bytes(b"".join(rows[:-1]))
    dropped = _commit(repo, "drop the decision")

    out = _check_locks(repo)

    lines = _violations(out)
    assert len(lines) == 1, out.stdout
    assert dropped[:12] in lines[0]
    for sha in clean[:-1]:
        assert sha[:12] not in out.stdout


def test_a_result_with_mismatched_fingerprints_exits_1_naming_it(tmp_path, monkeypatch):
    """A result whose fingerprints differ from its registration's is named by its row_id.

    docs/PROTOCOL.md is edited and committed after the run ends, so the result recorded next
    hashes another protocol than its registration did. ledger.py writes it (it doesn't compare
    fingerprints); check_locks must catch it. Every commit appends, so it is the one violation.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    reg = ledger.register(**REGISTRATION, path=path)
    run, _ = _run(path, reg)
    _commit(repo, "register, run and run-end")
    (repo / "docs" / "PROTOCOL.md").write_text("protocol v2: the primary metric changed\n")
    _commit(repo, "edit the protocol")
    res = _result(path, reg, run)
    _commit(repo, "result")
    assert res["fingerprints"]["docs/PROTOCOL.md"] != reg["fingerprints"]["docs/PROTOCOL.md"]

    out = _check_locks(repo)

    lines = _violations(out)
    assert len(lines) == 1, out.stdout
    assert res["row_id"] in lines[0]


def test_two_final_results_exit_1_naming_the_second(tmp_path, monkeypatch):
    """The seal is spent once: a second final-result row is named by its row_id.

    ledger.py refuses a second final-result, so the copy (row_id final-result-0badc0de, naming
    the same earlier final-registered) is appended as raw text, as a hand edit would be. The file
    only grows, so history is clean and the second final-result is the violation.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    _, fres = _final(path)
    _commit(repo, "final-register and final-result")
    with path.open("a") as f:
        f.write(_line({**fres, "row_id": SECOND_FINAL}))
    _commit(repo, "a second final-result, by hand")

    out = _check_locks(repo)

    _violations(out)
    assert SECOND_FINAL in out.stdout


@pytest.mark.parametrize(
    "case",
    [
        "result above its registered",
        "final-result with no final-registered",
        "final-result above its final-registered",
    ],
)
def test_a_row_whose_parent_is_not_earlier_exits_1_naming_it(tmp_path, monkeypatch, case):
    """A result or final-result must name an EARLIER parent row; a later or missing one is named.

    The rows are written through the ledger API to a scratch file outside the repo (so the
    fingerprints are real and equal), then laid into the repo's ledger as raw lines in the wrong
    order and committed once. The file is created in one commit, so history is clean; the result's
    fingerprints equal its registration's, and the run has its run-end, so order is the one
    violation. Today's check_locks already flags these: this pins "earlier", which an
    implementation that looks the parent up anywhere in the file would break.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    scratch = tmp_path / "scratch.jsonl"
    reg = ledger.register(**REGISTRATION, path=scratch)
    run, end = _run(scratch, reg)
    res = _result(scratch, reg, run)
    freg, fres = _final(scratch)
    assert res["fingerprints"] == reg["fingerprints"]
    order, named = {
        "result above its registered": ([res, reg, run, end], res),
        "final-result with no final-registered": ([fres], fres),
        "final-result above its final-registered": ([fres, freg], fres),
    }[case]
    (repo / LEDGER).write_text("".join(_line(r) for r in order))
    _commit(repo, "rows out of order, by hand")

    out = _check_locks(repo)

    lines = _violations(out)
    assert len(lines) == 1, out.stdout
    assert named["row_id"] in lines[0]


def test_a_repo_that_never_had_a_ledger_is_clean(tmp_path, monkeypatch):
    """With only control/ledger/.gitkeep committed, as in this repo before its first registration,
    there is nothing to check: exit 0, one line starting "check_locks: clean"."""
    repo = _make_repo(tmp_path, monkeypatch)
    assert not (repo / LEDGER).exists()

    _assert_clean(_check_locks(repo))


def test_a_shallow_clone_exits_1_naming_shallow(tmp_path, monkeypatch):
    """A depth-1 clone lacks the history the lock walks, so it exits 1 with "shallow" in stdout.

    The source repo's 3 commits (init, register, run + run-end) are clean, and check_locks passes
    them in the full repo, so depth is the only reason to fail. file:// makes git honour --depth
    on a local clone.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    reg = ledger.register(**REGISTRATION, path=path)
    _commit(repo, "register")
    _run(path, reg)
    _commit(repo, "run and run-end")
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", repo.as_uri(), str(shallow))
    assert _git(shallow, "rev-parse", "--is-shallow-repository") == "true"

    out = _check_locks(shallow)

    _violations(out)
    assert "shallow" in out.stdout
    _assert_clean(_check_locks(repo))


def test_a_run_without_a_run_end_exits_1_naming_the_run(tmp_path, monkeypatch):
    """A committed run row that no run-end names is an open run, named by its row_id.

    History only appends and there is no result, so the open run is the one violation.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    reg = ledger.register(**REGISTRATION, path=path)
    run = ledger.run_start(registered=reg["row_id"], arm="a", seed=0, run_dir="runs/a-0", path=path)
    _commit(repo, "register and run")

    out = _check_locks(repo)

    lines = _violations(out)
    assert len(lines) == 1, out.stdout
    assert run["row_id"] in lines[0]


def test_deleting_the_ledger_file_exits_1_naming_that_commit(tmp_path, monkeypatch):
    """Commit D deletes a ledger that had rows; D is named.

    A missing file reads as empty bytes, which don't start with the parent's rows. HEAD then has
    no rows, so no HEAD check fires: D is the one violation, and the commit that created the
    ledger is not named. (A line may also name D's parent.)
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    reg = ledger.register(**REGISTRATION, path=path)
    created = _commit(repo, "register")
    _run(path, reg)
    _commit(repo, "run and run-end")
    path.unlink()
    deleted = _commit(repo, "delete the ledger")

    out = _check_locks(repo)

    lines = _violations(out)
    assert len(lines) == 1, out.stdout
    assert deleted[:12] in lines[0]
    assert created[:12] not in out.stdout


@pytest.mark.parametrize("lost", ["older version", "HEAD's version"])
def test_a_ledger_version_git_cannot_read_fails_closed(tmp_path, monkeypatch, lost):
    """A ledger blob missing from the object store is not a missing file: it never reads clean.

    Commit P creates the ledger with a registration; commit C appends a run and its run-end. The
    repo checks clean intact. Then one version's blob object is deleted (P's, or C's at HEAD).
    The script must not exit 0 and must not print "check_locks: clean"; crashing on the git
    error is fine. Reading any `cat-file` failure as "the file doesn't exist" turns the lost P
    into a ledger created at C, which is clean, and the lost C into "commit C deletes the
    ledger", which blames C for a deletion that never happened: no line may say "deletes".
    The fresh repo is unpacked, so the blob is a loose object; if it isn't, the test fails
    rather than skips.
    """
    repo = _make_repo(tmp_path, monkeypatch)
    path = repo / LEDGER
    reg = ledger.register(**REGISTRATION, path=path)
    older = _commit(repo, "register")
    _run(path, reg)
    head = _commit(repo, "run and run-end")
    _assert_clean(_check_locks(repo))
    sha = _git(repo, "rev-parse", f"{older if lost == 'older version' else head}:{LEDGER}")
    blob = repo / ".git" / "objects" / sha[:2] / sha[2:]
    assert blob.is_file(), f"blob {sha} is not a loose object (packed?), so it can't be deleted"
    blob.unlink()

    out = _check_locks(repo)

    assert out.returncode != 0, out.stdout + out.stderr
    assert "clean" not in out.stdout, out.stdout
    assert "deletes" not in out.stdout, out.stdout

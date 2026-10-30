"""Tests for greenleaf_control.manifest (issue #2): the run receipt and its two refusals."""

import hashlib
import json
import platform
import re
import subprocess
import sys
import types
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path

import pytest

# The fields docs/API-CONTRACT.md lists for manifest.json.
MANIFEST_FIELDS = {
    "commit",
    "dirty",
    "fingerprints",
    "weather_hashes",
    "versions",
    "config",
    "seeds",
    "host",
    "started_at",
    "issue",
    "registered",
    "smoke",
}
# The scorer files and the PPO config, keyed by repo-relative POSIX path.
SCORER_KEYS = [
    "control/greenleaf_control/scenarios.py",
    "control/greenleaf_control/rewards.py",
    "control/greenleaf_control/envs.py",
    "control/greenleaf_control/metrics.py",
    "docs/PROTOCOL.md",
    "control/configs/ppo.yml",
]
LIBRARIES = ["stable-baselines3", "torch", "gymnasium", "casadi", "numpy"]
CONFIG = {"method": "ppo", "seed": 7, "timesteps": 20_000, "season_length": 2}
LEDGER_FILE = "control/ledger/ledger.jsonl"  # the ledger, repo-relative


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], check=True, cwd=repo, capture_output=True, text=True)
    return out.stdout.strip()


def _commit(repo: Path) -> None:
    """Commit everything staged. CI has no git identity and may sign commits globally."""
    _git(
        repo,
        *("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"),
        *("commit", "-q", "-m", "commit"),
    )


def _make_repo(tmp_path: Path, monkeypatch, manifest) -> Path:
    """A temp git repo with one committed file, control/README.md, set as manifest.REPO_ROOT."""
    # A git hook's environment would point these git calls at the real repo.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    repo = tmp_path / "repo"
    (repo / "control").mkdir(parents=True)
    (repo / "control" / "README.md").write_text("temp repo\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _commit(repo)
    monkeypatch.setattr(manifest, "REPO_ROOT", repo)
    return repo


def _repo_with_a_new_ledger_row(tmp_path: Path, monkeypatch, manifest, *, tracked: bool) -> Path:
    """The temp repo laid out like this one: control/ledger/.gitkeep committed, and a ledger row
    appended to control/ledger/ledger.jsonl since HEAD.

    tracked: ledger.jsonl was committed with one row, so git shows it " M". Otherwise it was never
    added, so git shows "?? control/ledger/ledger.jsonl" (the directory itself is tracked).
    """
    repo = _make_repo(tmp_path, monkeypatch, manifest)
    (repo / "control" / "ledger").mkdir()
    (repo / "control" / "ledger" / ".gitkeep").write_text("")
    if tracked:
        (repo / LEDGER_FILE).write_text('{"row": 1}\n')
    _git(repo, "add", ".")
    _commit(repo)
    with (repo / LEDGER_FILE).open("a") as f:
        f.write('{"row": 2}\n')
    return repo


def _scorer_tree(root: Path) -> dict[str, bytes]:
    """Write the 6 fingerprinted files under root, each with distinct bytes; return path -> bytes."""
    contents = {key: f"contents of {key}\n".encode() for key in SCORER_KEYS}
    contents["docs/PROTOCOL.md"] = b"abc"
    for key, data in contents.items():
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return contents


def _utc(stamp: str) -> datetime:
    """Parse an ISO timestamp and check it is tz-aware UTC."""
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() == timedelta(0), stamp
    return parsed


def test_write_manifest_on_clean_tree_records_commit_and_every_field(tmp_path, monkeypatch):
    """A clean tree gets a manifest with HEAD's SHA, dirty False and every contract field.

    The file on disk equals the returned dict. The temp repo holds none of the 6 scorer files,
    so through the REPO_ROOT seam every fingerprint is None.
    """
    import greenleaf_control.manifest as manifest

    repo = _make_repo(tmp_path, monkeypatch, manifest)
    run_dir = tmp_path / "run-clean"
    m = manifest.write_manifest(
        run_dir, config=CONFIG, issue=2, registered="registered-0badc0de", smoke=False
    )

    assert json.loads((run_dir / "manifest.json").read_text()) == m
    missing = MANIFEST_FIELDS - m.keys()
    assert not missing, f"missing contract fields: {sorted(missing)}"
    assert m["commit"] == _git(repo, "rev-parse", "HEAD")
    assert m["dirty"] is False
    assert m["smoke"] is False
    assert m["issue"] == 2
    assert m["registered"] == "registered-0badc0de"
    assert m["config"] == CONFIG
    assert m["seeds"] == {"model": 7, "env": 7}
    assert isinstance(m["host"], str) and m["host"]
    _utc(m["started_at"])
    assert m["fingerprints"] == dict.fromkeys(SCORER_KEYS)


def test_weather_hashes_empty_while_scenarios_module_is_absent(tmp_path, monkeypatch):
    """Until greenleaf_control.scenarios exists (issue #6), weather_hashes is {}.

    None in sys.modules makes the import raise ModuleNotFoundError naming that module,
    exactly as a missing scenarios.py does.
    """
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    monkeypatch.setitem(sys.modules, "greenleaf_control.scenarios", None)
    run_dir = tmp_path / "run"
    m = manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=False)

    assert m["weather_hashes"] == {}
    assert json.loads((run_dir / "manifest.json").read_text())["weather_hashes"] == {}


def test_weather_hashes_recorded_from_scenarios_module(tmp_path, monkeypatch):
    """When greenleaf_control.scenarios exists, its weather_hashes() is recorded exactly."""
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    scenarios = types.ModuleType("greenleaf_control.scenarios")
    scenarios.weather_hashes = lambda: {"amsterdam_2011": "abc123"}
    monkeypatch.setitem(sys.modules, "greenleaf_control.scenarios", scenarios)
    run_dir = tmp_path / "run"
    m = manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=False)

    assert m["weather_hashes"] == {"amsterdam_2011": "abc123"}
    assert json.loads((run_dir / "manifest.json").read_text())["weather_hashes"] == {
        "amsterdam_2011": "abc123"
    }


def test_weather_hashes_propagates_a_different_missing_module(tmp_path, monkeypatch):
    """A scenarios.py that imports a missing module raises; it is not recorded as {}.

    Only the absence of greenleaf_control.scenarios itself means "no weather yet". A broken
    scenarios module must fail the run, naming the module it lacks, and make no run_dir.
    """
    import greenleaf_control
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    fake_pkg = tmp_path / "fake_pkg"  # outside the temp repo, so the tree stays clean
    fake_pkg.mkdir()
    (fake_pkg / "scenarios.py").write_text("import no_such_mod_xyz_greenleaf\n")
    # Searched first, so this scenarios.py shadows a real one; the copied cache drops its finder.
    monkeypatch.setattr(greenleaf_control, "__path__", [str(fake_pkg), *greenleaf_control.__path__])
    monkeypatch.setattr(sys, "path_importer_cache", dict(sys.path_importer_cache))
    monkeypatch.delitem(sys.modules, "greenleaf_control.scenarios", raising=False)
    run_dir = tmp_path / "run"

    with pytest.raises(ModuleNotFoundError) as excinfo:
        manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=False)
    assert excinfo.value.name == "no_such_mod_xyz_greenleaf"
    assert not run_dir.exists()


@pytest.mark.parametrize(
    "path",
    [
        "stray_notes.txt",  # an untracked file
        "control/README.md",  # an edit to the committed file
        "control/runs_notes.txt",  # not under control/runs/, so the filter must not hide it
    ],
)
def test_write_manifest_refuses_dirty_tree(tmp_path, monkeypatch, path):
    """A dirty tree without smoke raises RuntimeError naming the offending path, and makes no run_dir."""
    import greenleaf_control.manifest as manifest

    repo = _make_repo(tmp_path, monkeypatch, manifest)
    (repo / path).write_text("uncommitted\n")
    run_dir = tmp_path / "run-dirty"

    with pytest.raises(RuntimeError, match=re.escape(path)):
        manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=False)
    assert not run_dir.exists()


def test_write_manifest_sees_untracked_files_despite_git_config(tmp_path, monkeypatch):
    """status.showUntrackedFiles=no in the repo's git config doesn't hide an untracked file.

    A non-smoke run still raises RuntimeError naming the file, and makes no run_dir.
    """
    import greenleaf_control.manifest as manifest

    repo = _make_repo(tmp_path, monkeypatch, manifest)
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "stray_notes.txt").write_text("uncommitted\n")
    assert _git(repo, "status", "--porcelain") == ""  # plain git status now hides the file
    run_dir = tmp_path / "run-hidden-untracked"

    with pytest.raises(RuntimeError, match="stray_notes.txt"):
        manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=False)
    assert not run_dir.exists()


def test_write_manifest_refuses_unserializable_config_and_makes_no_run_dir(tmp_path, monkeypatch):
    """A config json can't serialize raises TypeError and leaves no run_dir behind.

    Path("runs") isn't JSON. A half-made run_dir would block the rerun with FileExistsError.
    """
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    run_dir = tmp_path / "run-unserializable"
    config = {"seed": 0, "runs_dir": Path("runs")}

    with pytest.raises(TypeError, match="JSON serializable"):
        manifest.write_manifest(run_dir, config=config, issue=2, registered=None, smoke=False)
    assert not run_dir.exists()


def test_write_manifest_smoke_on_dirty_tree_records_dirty(tmp_path, monkeypatch):
    """With smoke=True a dirty tree is allowed, and the manifest records dirty True."""
    import greenleaf_control.manifest as manifest

    repo = _make_repo(tmp_path, monkeypatch, manifest)
    (repo / "stray_notes.txt").write_text("uncommitted\n")
    run_dir = tmp_path / "run-smoke"
    m = manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=True)

    assert json.loads((run_dir / "manifest.json").read_text()) == m
    assert m["dirty"] is True
    assert m["smoke"] is True
    assert m["registered"] is None
    assert m["commit"] == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    ("dirty", "smoke"),
    [
        (False, False),
        (True, False),  # the run_dir refusal comes before the dirty check
        (True, True),  # smoke does not excuse a reused run_dir
    ],
)
def test_write_manifest_refuses_existing_run_dir(tmp_path, monkeypatch, dirty, smoke):
    """An existing run_dir raises FileExistsError naming it, and its old manifest is left untouched."""
    import greenleaf_control.manifest as manifest

    repo = _make_repo(tmp_path, monkeypatch, manifest)
    if dirty:
        (repo / "stray_notes.txt").write_text("uncommitted\n")
    run_dir = tmp_path / "run-reused"
    run_dir.mkdir()
    old = run_dir / "manifest.json"
    old.write_text('{"commit": "old"}\n')

    with pytest.raises(FileExistsError, match=re.escape(str(run_dir))):
        manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=smoke)
    assert old.read_text() == '{"commit": "old"}\n'


def test_dirty_check_ignores_control_runs(tmp_path, monkeypatch):
    """Run dirs under <repo>/control/runs/ don't make the tree dirty for the next non-smoke run.

    The temp repo has no .gitignore, so git status does list control/runs/ and only the
    filter can hide it. A stray file outside control/runs/ still counts as dirty.
    """
    import greenleaf_control.manifest as manifest

    repo = _make_repo(tmp_path, monkeypatch, manifest)
    runs = repo / "control" / "runs"
    runs.mkdir()
    first = manifest.write_manifest(
        runs / "run-a", config=CONFIG, issue=2, registered=None, smoke=False
    )
    assert "control/runs/" in _git(repo, "status", "--porcelain")  # git sees the first run
    second = manifest.write_manifest(
        runs / "run-b", config=CONFIG, issue=2, registered=None, smoke=False
    )

    assert first["dirty"] is False
    assert second["dirty"] is False
    (repo / "stray_notes.txt").write_text("uncommitted\n")
    with pytest.raises(RuntimeError, match="stray_notes.txt"):
        manifest.write_manifest(
            runs / "run-c", config=CONFIG, issue=2, registered=None, smoke=False
        )


@pytest.mark.parametrize(
    ("tracked", "porcelain"),
    [(True, f" M {LEDGER_FILE}"), (False, f"?? {LEDGER_FILE}")],
    ids=["tracked-and-modified", "untracked"],
)
def test_dirty_check_ignores_the_ledger_file(tmp_path, monkeypatch, tracked, porcelain):
    """A row appended to <repo>/control/ledger/ledger.jsonl doesn't make the tree dirty (#13).

    Each registered run appends its run and run-end rows to the ledger inside the repo, so without
    this the second run of a grid would be refused. git status does list the file, so only the
    filter can hide it; a non-smoke manifest is written with dirty False.
    """
    import greenleaf_control.manifest as manifest

    repo = _repo_with_a_new_ledger_row(tmp_path, monkeypatch, manifest, tracked=tracked)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert status == [porcelain]
    run_dir = tmp_path / "run"
    m = manifest.write_manifest(run_dir, config=CONFIG, issue=13, registered=None, smoke=False)

    assert m["dirty"] is False
    assert json.loads((run_dir / "manifest.json").read_text())["dirty"] is False


@pytest.mark.parametrize(
    "neighbour",
    [
        "control/ledger/notes.txt",  # next to the ledger
        "control/ledger/ledger.jsonl.bak",  # the ledger's path is a prefix of it
    ],
)
def test_dirty_check_exempts_only_the_ledger_file_itself(tmp_path, monkeypatch, neighbour):
    """Only control/ledger/ledger.jsonl is exempt: another uncommitted file in control/ledger/
    still makes the tree dirty. With the ledger modified too, a non-smoke run raises RuntimeError
    naming the neighbour, and makes no run_dir."""
    import greenleaf_control.manifest as manifest

    repo = _repo_with_a_new_ledger_row(tmp_path, monkeypatch, manifest, tracked=True)
    (repo / neighbour).write_text("uncommitted\n")
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match=re.escape(neighbour)):
        manifest.write_manifest(run_dir, config=CONFIG, issue=13, registered=None, smoke=False)
    assert not run_dir.exists()


def test_fingerprints_are_sha256_of_each_file(tmp_path):
    """fingerprints(root) maps exactly the 6 repo-relative paths to the sha256 hex of their bytes.

    Known answer: docs/PROTOCOL.md holds b"abc", whose sha256 is the FIPS 180-2 test vector
    ba7816bf...f20015ad.
    """
    import greenleaf_control.manifest as manifest

    contents = _scorer_tree(tmp_path)
    fp = manifest.fingerprints(tmp_path)

    assert set(fp) == set(SCORER_KEYS)
    assert (
        fp["docs/PROTOCOL.md"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert fp == {key: hashlib.sha256(data).hexdigest() for key, data in contents.items()}


@pytest.mark.parametrize("edited", SCORER_KEYS)
def test_fingerprints_change_for_the_edited_file_only(tmp_path, edited):
    """Changing one file's bytes changes its fingerprint and no other."""
    import greenleaf_control.manifest as manifest

    _scorer_tree(tmp_path)
    before = manifest.fingerprints(tmp_path)
    (tmp_path / edited).write_bytes((tmp_path / edited).read_bytes() + b"one more line\n")
    after = manifest.fingerprints(tmp_path)

    assert after[edited] != before[edited]
    assert {k: v for k, v in after.items() if k != edited} == {
        k: v for k, v in before.items() if k != edited
    }


@pytest.mark.parametrize("missing", SCORER_KEYS)
def test_fingerprints_record_a_missing_file_as_none(tmp_path, missing):
    """A file that doesn't exist yet is fingerprinted as None; the others still hash."""
    import greenleaf_control.manifest as manifest

    contents = _scorer_tree(tmp_path)
    (tmp_path / missing).unlink()
    fp = manifest.fingerprints(tmp_path)

    assert fp == {
        key: None if key == missing else hashlib.sha256(data).hexdigest()
        for key, data in contents.items()
    }


def test_fingerprints_default_root_is_repo_root(tmp_path, monkeypatch):
    """With no argument, fingerprints() hashes under manifest.REPO_ROOT, read at call time.

    REPO_ROOT defaults to this repo's root, where docs/PROTOCOL.md already exists.
    """
    import greenleaf_control.manifest as manifest

    repo_root = Path(__file__).resolve().parents[2]
    assert manifest.REPO_ROOT == repo_root
    protocol = (repo_root / "docs" / "PROTOCOL.md").read_bytes()
    assert manifest.fingerprints()["docs/PROTOCOL.md"] == hashlib.sha256(protocol).hexdigest()

    contents = _scorer_tree(tmp_path)
    monkeypatch.setattr(manifest, "REPO_ROOT", tmp_path)
    assert manifest.fingerprints() == {
        key: hashlib.sha256(data).hexdigest() for key, data in contents.items()
    }


@pytest.mark.parametrize(
    ("status", "cost_usd"),
    [("finished", 0.42), ("crashed", None), ("budget-stop", 3.0)],
)
def test_finish_manifest_records_status_and_keeps_fields(tmp_path, monkeypatch, status, cost_usd):
    """finish_manifest merges the end-of-run fields into manifest.json and keeps every original one.

    Each allowed status is accepted; cost_usd may be None (a local run). ended_at is UTC and
    not before started_at.
    """
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    run_dir = tmp_path / "run"
    started = manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=True)
    done = manifest.finish_manifest(
        run_dir, status=status, wall_s=12.5, timesteps=96, solver_failures=1, cost_usd=cost_usd
    )

    assert json.loads((run_dir / "manifest.json").read_text()) == done
    assert done["status"] == status
    assert done["wall_s"] == 12.5
    assert done["timesteps"] == 96
    assert done["solver_failures"] == 1
    assert done["cost_usd"] == cost_usd
    assert _utc(done["ended_at"]) >= _utc(started["started_at"])
    assert {k: v for k, v in done.items() if k in started} == started


@pytest.mark.parametrize("status", ["done", "Finished", "running", ""])
def test_finish_manifest_rejects_unknown_status(tmp_path, monkeypatch, status):
    """An unknown status raises ValueError and leaves manifest.json byte-identical.

    "running" is the ledger's in-progress status, never an end status.
    """
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    run_dir = tmp_path / "run"
    manifest.write_manifest(run_dir, config=CONFIG, issue=2, registered=None, smoke=True)
    before = (run_dir / "manifest.json").read_bytes()

    with pytest.raises(ValueError):
        manifest.finish_manifest(
            run_dir, status=status, wall_s=12.5, timesteps=96, solver_failures=0, cost_usd=None
        )
    assert (run_dir / "manifest.json").read_bytes() == before


def test_versions_record_gl_gym_commit_and_library_versions(tmp_path, monkeypatch):
    """versions names python, the pinned gl-gym commit and the installed library versions.

    Known answer: pyproject pins gl-gym at 2d3febb, so the installed distribution's
    direct_url.json records a full 40-hex commit starting with it.
    """
    import greenleaf_control.manifest as manifest

    _make_repo(tmp_path, monkeypatch, manifest)
    m = manifest.write_manifest(
        tmp_path / "run", config=CONFIG, issue=2, registered=None, smoke=True
    )
    v = m["versions"]

    assert set(v) == {"python", "gl-gym", *LIBRARIES}
    assert re.fullmatch(r"[0-9a-f]{40}", v["gl-gym"]), v["gl-gym"]
    assert v["gl-gym"].startswith("2d3febb")
    assert v["python"].startswith(platform.python_version())
    assert {name: v[name] for name in LIBRARIES} == {name: version(name) for name in LIBRARIES}

"""The run receipt: manifest.json ties a run to its exact code, config, data and versions."""

import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# The repo root. Functions read it at call time, so tests can point it at a temp repo.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The files that decide how results are scored, plus the PPO config, as repo-relative paths.
FINGERPRINTED = [
    "control/greenleaf_control/scenarios.py",
    "control/greenleaf_control/rewards.py",
    "control/greenleaf_control/envs.py",
    "control/greenleaf_control/metrics.py",
    "docs/PROTOCOL.md",
    "control/configs/ppo.yml",
]
LIBRARIES = ["stable-baselines3", "torch", "gymnasium", "casadi", "numpy"]
STATUSES = {"finished", "crashed", "budget-stop"}


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
    )
    return out.stdout


def _dumps(data: dict) -> str:
    # No default=str: a config that won't serialize must fail loudly.
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def fingerprints(root: Path | None = None) -> dict:
    """sha256 of each scorer file and configs/ppo.yml, or None if it doesn't exist yet.

    Hashes bytes only: importing the graded code would let it influence its own receipt.
    """
    root = REPO_ROOT if root is None else root
    return {
        key: hashlib.sha256((root / key).read_bytes()).hexdigest()
        if (root / key).is_file()
        else None
        for key in FINGERPRINTED
    }


def _versions() -> dict:
    """Python, the gl-gym commit from its install record, and the library versions."""
    direct_url = importlib.metadata.distribution("gl-gym").read_text("direct_url.json")
    if direct_url is None:
        raise RuntimeError("gl-gym has no direct_url.json, so its commit is unknown")
    versions = {
        "python": platform.python_version(),
        "gl-gym": json.loads(direct_url)["vcs_info"]["commit_id"],
    }
    versions.update({name: importlib.metadata.version(name) for name in LIBRARIES})
    return versions


def write_manifest(
    run_dir: Path, *, config: dict, issue: int, registered: str | None, smoke: bool
) -> dict:
    """Write run_dir/manifest.json; refuse a reused run_dir, and a dirty tree unless smoke."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"run_dir already exists: {run_dir}")
    # Porcelain v1 lines are "XY path"; run dirs under control/runs/ don't count as dirty.
    dirty_lines = [
        line
        for line in _git("status", "--porcelain", "--untracked-files=normal").splitlines()
        if not line[3:].startswith("control/runs/")
    ]
    if dirty_lines and not smoke:
        raise RuntimeError(
            "git tree is dirty; commit first or pass smoke:\n" + "\n".join(dirty_lines)
        )

    # S1-1 provides scenarios.weather_hashes and removes this fallback.
    try:
        from greenleaf_control.scenarios import weather_hashes
    except ModuleNotFoundError as e:
        if e.name != "greenleaf_control.scenarios":
            raise
        hashes = {}
    else:
        hashes = weather_hashes()

    manifest = {
        "commit": _git("rev-parse", "HEAD").strip(),
        "dirty": bool(dirty_lines),
        "fingerprints": fingerprints(),
        "weather_hashes": hashes,
        "versions": _versions(),
        "config": config,
        "seeds": {"model": config["seed"], "env": config["seed"]},
        "host": socket.gethostname(),
        "started_at": datetime.now(UTC).isoformat(),
        "issue": issue,
        "registered": registered,
        "smoke": smoke,
    }
    # Serialize before mkdir, so a bad config leaves no run_dir to block the rerun.
    text = _dumps(manifest)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "manifest.json").write_text(text)
    return manifest


def finish_manifest(
    run_dir: Path,
    *,
    status: str,
    wall_s: float,
    timesteps: int,
    solver_failures: int,
    cost_usd: float | None,
) -> dict:
    """Merge the end-of-run fields into run_dir/manifest.json and return the result."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}, got {status!r}")
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        status=status,
        wall_s=wall_s,
        timesteps=timesteps,
        solver_failures=solver_failures,
        cost_usd=cost_usd,
        ended_at=datetime.now(UTC).isoformat(),
    )
    path.write_text(_dumps(manifest))
    return manifest

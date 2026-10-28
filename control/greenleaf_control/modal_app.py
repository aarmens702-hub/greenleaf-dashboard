"""The remote app (docs/API-CONTRACT.md): train_remote runs one train argv on Modal, with its runs
on the volume greenleaf-runs mounted at /runs, at run_grid's hard limits (4 cores, 8 GiB) for the
full 4 h timeout. CPU and memory are (request, limit) pairs: Modal bills max(request, usage), and
the limits make run_grid --dry's worst case the true worst case. Above the limits CPU is throttled
and memory is OOM-killed (#19 records that run as crashed).

Importing this module contacts nothing: the app, the image, the volume and the function are only
defined here; Modal builds, looks up and runs them when something deploys or calls them, and
nothing in this repo does yet. Remote runs aren't recordable until issue #19 (Record Modal runs:
ledger rows, fingerprints and cost from the Mac): the configs, the weather, the git fingerprints
and the ledger rows are handled there, so run_grid refuses --backend modal until then.

The image: Debian slim on Python 3.12, git (gl-gym is a git dependency), control/'s uv.lock synced
frozen (on linux it takes torch from PyTorch's CPU index), then the greenleaf_control source, which
uv sync doesn't install.
"""

from pathlib import Path

import modal

from greenleaf_control import run_grid

CONTROL = Path(__file__).resolve().parents[1]  # absolute, so the image is the same from any cwd
RUNS = Path("/runs")

app = modal.App("greenleaf")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(str(CONTROL), frozen=True)
    .add_local_python_source("greenleaf_control")
)
vol = modal.Volume.from_name("greenleaf-runs", create_if_missing=True)


@app.function(
    image=image,
    cpu=(run_grid.CPU, run_grid.CPU),  # (request, limit): a hard limit
    memory=(run_grid.MEMORY_MB, run_grid.MEMORY_MB),
    timeout=run_grid.TIMEOUT_S,
    volumes={str(RUNS): vol},
)
def train_remote(argv: list[str]) -> str:
    """Run train.main(argv) with --runs-dir /runs; returns the run dir it made. Not called before #19."""
    from greenleaf_control import train  # here, so importing the app doesn't import torch

    before = set(RUNS.glob("**/manifest.json"))
    train.main([*argv, "--runs-dir", str(RUNS)])
    [manifest] = set(RUNS.glob("**/manifest.json")) - before  # train refuses an existing run dir
    return str(manifest.parent)

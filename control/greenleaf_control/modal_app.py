"""The remote app (docs/API-CONTRACT.md): train_remote runs one registered train argv on Modal, with
its runs on the volume greenleaf-runs mounted at /runs, at run_grid's hard limits (4 cores, 8 GiB)
for the full 4 h timeout. CPU and memory are (request, limit) pairs: Modal bills max(request,
usage), and the limits make run_grid --dry's worst case the true worst case. Above the limits CPU is
throttled and memory is OOM-killed; the call then raises, and run_grid records the run as crashed.

Importing this module contacts nothing: the app, the image, the volume and the function are only
defined here. Only run_grid's _run_remote imports it, to call train_remote inside one app.run()
session. The Mac writes every ledger row (#19): the container has no git checkout and no ledger, so
train_remote runs train with --remote, carrying the Mac's code stamp and the registered row, and
returns the manifest at the run dir run_grid names (run_rel).

Each input is attempted once (retries=0) in its own container (single_use_containers=True). A
failed input isn't rerun, and a preemption rerun meets train's refusal of its existing run dir on
the volume; either way the Mac records the run as crashed. nonpreemptible stays the SDK's default
(False), pending Aarmen's call on spend.

The image: Debian slim on Python 3.12, git (gl-gym is a git dependency), control/'s uv.lock synced
frozen (on linux it takes torch from PyTorch's CPU index), then the greenleaf_control source, which
uv sync doesn't install, and configs/ and data/weather/ beside it under /root, where train reads
them (rule_based.CONFIGS, scenarios.WEATHER_DIR). Every weather file ships, the sealed 2012
included: the container hashes them against the stamp and never trains on them.
"""

import json
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
    .add_local_dir(str(CONTROL / "configs"), "/root/configs")
    .add_local_dir(str(CONTROL / "data" / "weather"), "/root/data/weather")
)
vol = modal.Volume.from_name("greenleaf-runs", create_if_missing=True)


@app.function(
    image=image,
    cpu=(run_grid.CPU, run_grid.CPU),  # (request, limit): a hard limit
    memory=(run_grid.MEMORY_MB, run_grid.MEMORY_MB),
    timeout=run_grid.TIMEOUT_S,
    volumes={str(RUNS): vol},
    retries=0,  # a rerun would meet its existing run dir on the volume, or bill twice
    single_use_containers=True,  # one input per container: no run inherits another's process
)
def train_remote(argv: list[str], meta: dict, run_rel: str) -> dict:
    """Run train.main(argv) into /runs as a remote run (--remote meta, where meta is {"stamp":
    the Mac's code_stamp, "registered": the row}), commit the volume whether or not it finished,
    and return the run's manifest.json, at /runs/<run_rel> ("<row>/<arm>/seed<S>"), as a dict.

    The manifest is read by its own path: the commit reloads the volume, so other containers'
    new manifests are visible after it."""
    from greenleaf_control import train  # here, so importing the app doesn't import torch

    try:
        train.main([*argv, "--runs-dir", str(RUNS), "--remote", json.dumps(meta)])
    finally:  # a crashed run's manifest and checkpoints reach the volume too
        vol.commit()
    return json.loads((RUNS / run_rel / "manifest.json").read_text())

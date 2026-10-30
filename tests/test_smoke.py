"""Fresh-clone smoke test: the dashboard starts from tracked files alone and every tab renders.

A fresh clone has no data/derived/ (gitignored, built by build_derived.py), no RAG index
(app/rag/index_data/, gitignored), no .env and no PYTHONPATH. The documented command,
`streamlit run app/streamlit_app/main.py`, puts only app/streamlit_app/ on sys.path.

The app runs in a subprocess on a copy of the tracked inputs, so nothing this pytest process
has already imported (the worktree's `app` package, the main checkout's data/derived/) can
leak in, and so the copy's data/derived/ is written under tmp_path, never in the repo.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# main.py: st.tabs([...]); each page's top heading, as written in its render_*().
TAB_HEADINGS = {
    "Overview": "At a Glance",  # overview_page.py
    "Operations": "Stress-Risk Operations",  # risk_page.py
    "Financials": "The Season in Dollars",  # finance_page.py
    "Ask GreenLeaf": "Ask GreenLeaf",  # chat_page.py (gl-headline)
}
# build_derived.main() writes these six.
DERIVED = ["plot_meta", "season", "daily", "actions", "scouting", "bc_benchmarks"]

RUNNER = '''\
import json
import sys

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(sys.argv[1]).run(timeout=300)
assert not at.exception, "\\n\\n".join(
    e.message + "\\n" + "\\n".join(e.stack_trace) for e in at.exception
)
out = {t.label: " ".join(m.value for m in t.markdown) for t in at.tabs}
with open(sys.argv[2], "w") as fh:
    json.dump(out, fh)
'''


def _copy_tracked_inputs(repo):
    """What a fresh clone has: code, models, raw + external data. No derived data, no index."""
    skip = shutil.ignore_patterns("__pycache__", "index_data")
    for d in ["app", "analysis", "models"]:
        shutil.copytree(ROOT / d, repo / d, ignore=skip)
    (repo / "data" / "raw").mkdir(parents=True)
    for csv in (ROOT / "data" / "raw").glob("*.csv"):
        shutil.copy2(csv, repo / "data" / "raw" / csv.name)
    if (ROOT / "data" / "external").exists():
        shutil.copytree(ROOT / "data" / "external", repo / "data" / "external")
    shutil.copy2(ROOT / "build_derived.py", repo / "build_derived.py")


def test_fresh_clone_dashboard_starts_and_every_tab_renders(tmp_path):
    """Regression for #23. A fresh clone can't start the dashboard: main.py dies on
    `from app.streamlit_app...` (ModuleNotFoundError: 'app'); with PYTHONPATH=. it dies reading
    data/derived/season.parquet at import, through chat_page -> answer -> agent -> data_queries.
    After the fix the app starts with no PYTHONPATH, no derived data, no index and no API key,
    renders all four tabs with their headings, and has built data/derived/ itself."""
    repo = tmp_path / "repo"
    _copy_tracked_inputs(repo)
    assert not (repo / "data" / "derived").exists()
    assert not (repo / "app" / "rag" / "index_data").exists()

    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "run.py").write_text(RUNNER)
    out = tmp_path / "rendered.json"

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH" and not k.endswith("_API_KEY")}
    env["HF_HUB_OFFLINE"] = "1"  # no network: a model download at import fails instead of hanging
    proc = subprocess.run(
        [sys.executable, str(runner / "run.py"), str(repo / "app" / "streamlit_app" / "main.py"), str(out)],
        cwd=runner,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"dashboard failed to start from a fresh clone:\n{proc.stderr[-6000:]}"

    rendered = json.loads(out.read_text())
    assert list(rendered) == list(TAB_HEADINGS)
    for tab, heading in TAB_HEADINGS.items():
        assert heading in rendered[tab], f"tab {tab!r} is missing its heading {heading!r}"

    for name in DERIVED:
        assert (repo / "data" / "derived" / f"{name}.parquet").exists(), name

"""Tests for the GitHub Pages deploy (issue #61, S4): .github/workflows/pages.yml builds the demo site
with greenleaf_control.site_build (S1, issue #58) and publishes it at
https://aarmens702-hub.github.io/greenleaf-dashboard/, and the README links it.

The conventions pinned here.
- Pages deploys from master only (master is what the public sees, AGENTS.md), on a push or by hand
  (workflow_dispatch). Never from dev.
- The workflow's token reads the repo and may write Pages and mint an OIDC id-token, nothing more.
  Deploys share the concurrency group "pages" and never cancel one in progress.
- One job builds: checkout, setup-uv with control-ci.yml's pins (the same uv and Python the fast
  tests pass on), `uv sync --locked` in control/, `uv run python -m greenleaf_control.site_build
  --out ../_site` in control/, configure-pages, then upload-pages-artifact with `path: _site`. The
  site is written to the repo root's _site/, the folder the upload reads.
- One job deploys: it needs the build job, runs in the `github-pages` environment whose url is the
  deploy-pages step's page_url output, and runs actions/deploy-pages.
- Every `uses:` names one exact release (vX.Y.Z), not a branch or a floating major, so a moved tag
  can't change what runs with a token that can write Pages.
- The build needs no secrets: the site's data is committed.
- The README has one "Live demo:" line under the hero, before its first section, linking the Pages
  URL and saying it's a simulated replay. The README uses no em or en dashes.

PyYAML reads the workflow's `on:` key as True (YAML 1.1); the helpers accept either spelling. Every
check reads its files inside the test body, so while pages.yml is missing each test FAILS on an
assertion rather than ERRORs. Nothing here imports site_build, builds an env or touches the network.
"""

import posixpath
import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / ".github/workflows/pages.yml"
CI = ROOT / ".github/workflows/control-ci.yml"
README = ROOT / "README.md"

PAGES_URL = "https://aarmens702-hub.github.io/greenleaf-dashboard/"
LIVE_DEMO = f"Live demo: {PAGES_URL}"
PERMISSIONS = {"contents": "read", "pages": "write", "id-token": "write"}
EXACT_RELEASE = re.compile(r"@v\d+\.\d+\.\d+$")
PAGE_URL_OUTPUT = re.compile(r"\$\{\{\s*steps\.([\w-]+)\.outputs\.page_url\s*\}\}")
CHECKOUT = "actions/checkout@"
SETUP_UV = "astral-sh/setup-uv@"
CONFIGURE_PAGES = "actions/configure-pages@"
UPLOAD = "actions/upload-pages-artifact@"
DEPLOY = "actions/deploy-pages@"


def _pages_text() -> str:
    assert PAGES.is_file(), f"{PAGES.relative_to(ROOT)} is missing (issue #61 creates it)"
    return PAGES.read_text()


def _pages() -> dict:
    """The workflow, parsed."""
    wf = yaml.safe_load(_pages_text())
    assert isinstance(wf, dict) and isinstance(wf.get("jobs"), dict) and wf["jobs"], wf
    return wf


def _on(wf: dict) -> dict:
    """The workflow's triggers, whether PyYAML read `on:` as True or as "on"."""
    on = wf[True] if True in wf else wf.get("on")
    assert isinstance(on, dict), on
    return on


def _uses(step: dict) -> str:
    return str(step.get("uses", ""))


def _job_using(wf: dict, action: str) -> tuple[str, dict]:
    """The one job with a step that uses `action` (an "owner/name@" prefix), and its name."""
    found = [
        (name, job)
        for name, job in wf["jobs"].items()
        if any(_uses(step).startswith(action) for step in job.get("steps") or [])
    ]
    assert len(found) == 1, f"want one job using {action}, got {[name for name, _ in found]}"
    return found[0]


def _index(steps: list[dict], match) -> int:
    """The index of the one step that matches."""
    found = [i for i, step in enumerate(steps) if match(step)]
    assert len(found) == 1, (found, steps)
    return found[0]


def _workdir(wf: dict, job: dict, step: dict) -> str:
    """Where a run step runs: its own working-directory, else the job's, else the workflow's
    defaults, else the repo root.
    """
    job_default = ((job.get("defaults") or {}).get("run") or {}).get("working-directory")
    wf_default = ((wf.get("defaults") or {}).get("run") or {}).get("working-directory")
    return posixpath.normpath(step.get("working-directory") or job_default or wf_default or ".")


def _out(tokens: list[str]) -> str | None:
    """The value of `--out X` or `--out=X`."""
    for i, token in enumerate(tokens):
        if token == "--out" and i + 1 < len(tokens):
            return tokens[i + 1]
        if token.startswith("--out="):
            return token.removeprefix("--out=")
    return None


def test_pages_runs_on_master_pushes_and_by_hand():
    """The triggers are a push to master and workflow_dispatch, nothing else: no dev, no pull
    requests, no schedule.
    """
    on = _on(_pages())
    assert set(on) == {"push", "workflow_dispatch"}, on
    assert on["push"] == {"branches": ["master"]}, on["push"]


def test_pages_token_reads_the_repo_and_writes_pages_only():
    """Every job's token is contents: read, pages: write and id-token: write, set at the top of the
    workflow or on the job, and nothing more.
    """
    wf = _pages()
    for name, job in wf["jobs"].items():
        assert job.get("permissions", wf.get("permissions")) == PERMISSIONS, name


def test_pages_deploys_queue_and_never_cancel():
    """Deploys share one concurrency group and a new push waits for the one in progress instead of
    cancelling it halfway.
    """
    assert _pages().get("concurrency") == {"group": "pages", "cancel-in-progress": False}


def test_build_job_installs_with_the_ci_pins():
    """The build job installs with `uv sync --locked` in control/ (a stale lockfile fails, as in
    control-ci.yml) and its setup-uv step has control-ci.yml's action release, uv release (x.y.z)
    and Python, so the site builds on the toolchain the fast tests pass on.
    """
    wf = _pages()
    _, job = _job_using(wf, UPLOAD)
    steps = job.get("steps") or []
    syncs = [step for step in steps if "uv sync" in step.get("run", "")]
    assert len(syncs) == 1, syncs
    assert "--locked" in syncs[0]["run"].split(), syncs[0]["run"]
    assert "--frozen" not in syncs[0]["run"].split(), syncs[0]["run"]
    assert _workdir(wf, job, syncs[0]) == "control", syncs[0]

    setup = [step for step in steps if _uses(step).startswith(SETUP_UV)]
    assert len(setup) == 1, setup
    ci_setup = [
        step
        for ci_job in yaml.safe_load(CI.read_text())["jobs"].values()
        for step in ci_job.get("steps") or []
        if _uses(step).startswith(SETUP_UV)
    ]
    assert len(ci_setup) == 1, ci_setup
    assert _uses(setup[0]) == _uses(ci_setup[0]), (setup[0], ci_setup[0])
    with_, ci_with = setup[0].get("with") or {}, ci_setup[0].get("with") or {}
    version = with_.get("version")
    assert isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version), version
    assert version == ci_with.get("version"), (version, ci_with)
    assert with_.get("python-version") == ci_with.get("python-version"), (with_, ci_with)


def test_build_job_builds_the_site_into_the_uploaded_folder():
    """The build job checks out, installs, runs `uv run python -m greenleaf_control.site_build` in
    control/ with --out resolving to the repo root's _site/, configures Pages, and only then uploads
    _site with upload-pages-artifact. control/ + ../_site is _site, the folder the upload reads.
    """
    wf = _pages()
    _, job = _job_using(wf, UPLOAD)
    steps = job.get("steps") or []

    i_checkout = _index(steps, lambda step: _uses(step).startswith(CHECKOUT))
    i_sync = _index(steps, lambda step: "uv sync" in step.get("run", ""))
    i_build = _index(steps, lambda step: "greenleaf_control.site_build" in step.get("run", ""))
    i_configure = _index(steps, lambda step: _uses(step).startswith(CONFIGURE_PAGES))
    i_upload = _index(steps, lambda step: _uses(step).startswith(UPLOAD))
    assert i_checkout < i_sync < i_build < i_upload, (i_checkout, i_sync, i_build, i_upload)
    assert i_configure < i_upload, (i_configure, i_upload)

    build = steps[i_build]
    run = build["run"]
    tokens = shlex.split(run)
    assert "uv run" in run, run
    assert any(
        tokens[i : i + 2] == ["-m", "greenleaf_control.site_build"] for i in range(len(tokens))
    ), run
    workdir = _workdir(wf, job, build)
    assert workdir == "control", build
    out = _out(tokens)
    assert out is not None, run
    assert posixpath.normpath(posixpath.join(workdir, out)) == "_site", (workdir, out)

    path = (steps[i_upload].get("with") or {}).get("path")
    assert isinstance(path, str) and posixpath.normpath(path) == "_site", steps[i_upload]


def test_deploy_job_publishes_the_build_to_github_pages():
    """A second job needs the build job, runs in the github-pages environment with the deploy
    step's page_url as its url, and deploys with actions/deploy-pages.
    """
    wf = _pages()
    build_name, _ = _job_using(wf, UPLOAD)
    deploy_name, deploy = _job_using(wf, DEPLOY)
    assert deploy_name != build_name, deploy_name

    needs = deploy.get("needs")
    needs = [needs] if isinstance(needs, str) else list(needs or [])
    assert build_name in needs, (deploy_name, needs)

    environment = deploy.get("environment")
    assert isinstance(environment, dict), environment
    assert environment.get("name") == "github-pages", environment
    steps = deploy.get("steps") or []
    step = steps[_index(steps, lambda step: _uses(step).startswith(DEPLOY))]
    url = PAGE_URL_OUTPUT.fullmatch(str(environment.get("url", "")).strip())
    assert url is not None, environment
    assert step.get("id") == url.group(1), (step, environment)


def test_every_action_is_pinned_to_an_exact_release():
    """Every `uses:`, in every job, ends in an exact release tag (@vX.Y.Z): no @main, no floating
    @v4, so a moved tag can't change what runs with a token that can write Pages.
    """
    wf = _pages()
    uses = [
        str(item["uses"])
        for job in wf["jobs"].values()
        for item in [job, *(job.get("steps") or [])]
        if "uses" in item
    ]
    for action in (CHECKOUT, SETUP_UV, CONFIGURE_PAGES, UPLOAD, DEPLOY):
        assert any(use.startswith(action) for use in uses), (action, uses)
    unpinned = [use for use in uses if not EXACT_RELEASE.search(use)]
    assert unpinned == [], unpinned


def test_pages_workflow_needs_no_secrets():
    """The site's data is committed, so the build reads no secret: the workflow never names one."""
    text = _pages_text()
    assert "secrets." not in text, [line for line in text.splitlines() if "secrets." in line]


def test_readme_links_the_live_demo():
    """One README line, under the hero and before its first section, reads "Live demo: <the Pages
    URL>" and says the demo is a simulated replay. The README keeps its punctuation: no em dash
    (U+2014) or en dash (U+2013).
    """
    text = README.read_text(encoding="utf-8")
    lines = text.splitlines()
    demo = [i for i, line in enumerate(lines) if LIVE_DEMO in line]
    assert len(demo) == 1, f"want one README line containing {LIVE_DEMO!r}, got {len(demo)}"
    first_section = next(i for i, line in enumerate(lines) if line.startswith("## "))
    assert demo[0] < first_section, (demo[0], first_section)
    line = lines[demo[0]].lower()
    assert "simulat" in line and "replay" in line, lines[demo[0]]
    dashes = [line for line in lines if "—" in line or "–" in line]
    assert dashes == [], dashes

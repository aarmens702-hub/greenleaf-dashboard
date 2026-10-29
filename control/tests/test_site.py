"""Tests for the demo site's preview (issue #28, D3a): site/, a static page that replays one
simulated 60-day VALIDATION season, the grower's default rules (rb_default) against the tuned rules
(rb_tuned, E1, adopted), from JSON written by greenleaf_control.export_replay.

The conventions pinned here.
- The page is site/index.html, site/style.css and site/app.js at the repo root, plain files with no
  build step. Its data is site/data/, written by the CLI, never by hand:
  export_replay --controller {rb_default|rb_tuned} --split validation --out ../site/data.
- VALIDATION is Amsterdam 2010 from start days 45, 59 and 75, so site/data/ holds exactly six
  files, <controller>_<location>_<growth_year>_<start_day>.json, each export_replay's
  {"meta", "hourly", "totals"} for a 60-day season: 60 days x 24 h = 1,440 points in each of the
  14 hourly series.
- Sizes are decimal: 250 KB is 250,000 bytes, 1 MB is 1,000,000 bytes.
- The scoreboard's numbers are E1's: each file's totals equal, to 1e-9, the row of
  control/results/validation/<controller>_validation.parquet for its season (E1's
  result-525a6e90, recorded with the CLIs). Those parquets are only read here, never written.
- meta.dirty isn't checked: site/ is untracked when the CLI runs, so it is honestly true.

Every check reads files inside its test body, so while site/ is missing each test FAILS on an
assertion rather than ERRORs. Nothing here builds an env, reads weather or trains.
"""

import json
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest

from greenleaf_control import scenarios

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
DATA = SITE / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "validation"

CONTROLLERS = ("rb_default", "rb_tuned")
SEASON_DAYS = 60
N_HOURS = SEASON_DAYS * 24  # 1,440 hourly points
MAX_FILE_BYTES = 250_000  # 250 KB, decimal
MAX_SITE_BYTES = 1_000_000  # 1 MB, decimal
TOLERANCE = 1e-9
PAGE_FILES = ("index.html", "style.css", "app.js")
HOURLY_KEYS = [
    "t_hour",
    "co2",
    "temp",
    "rh",
    "u_boil",
    "u_co2",
    "u_thscr",
    "u_vent",
    "u_lamp",
    "u_blscr",
    "profit",
    "viol_co2",
    "viol_temp",
    "viol_rh",
]
TOTAL_KEYS = [
    "profit_eur_m2",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "yield_kg_m2",
    "paper_return",
]


def _file_name(controller: str, scenario: dict) -> str:
    return (
        f"{controller}_{scenario['location']}_{scenario['growth_year']}_"
        f"{scenario['start_day']}.json"
    )


# (controller, scenario, file name) for each of the six files: 2 controllers x 3 VALIDATION seasons.
SEASONS = [(c, s, _file_name(c, s)) for c in CONTROLLERS for s in scenarios.VALIDATION]
SEASON_CASES = [pytest.param(c, s, name, id=name.removesuffix(".json")) for c, s, name in SEASONS]
FILE_NAMES = sorted(name for _, _, name in SEASONS)


def _reject_constant(name: str):
    raise AssertionError(f"{name} in the JSON: the payload must be finite numbers only")


def _load(name: str) -> dict:
    path = DATA / name
    assert path.is_file(), f"{path} is missing"
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


class _Page(HTMLParser):
    """index.html's <script src> values, <link> (rel, href) pairs and <title> text."""

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "script" and a.get("src") is not None:
            self.scripts.append(a["src"])
        if tag == "link" and a.get("href") is not None:
            self.links.append(((a.get("rel") or "").lower(), a["href"]))
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def _is_remote(url: str) -> bool:
    """A URL another host serves: a scheme (http://, https://, ...) or protocol-relative (//)."""
    url = url.strip()
    return "://" in url or url.startswith("//")


# --- site/data/: the six files (Done when) ------------------------------------------------------


def test_site_data_holds_exactly_the_six_validation_files():
    """Done when: site/data/ holds exactly the six JSON files the CLI writes for VALIDATION, one per
    controller and season: rb_default_Amsterdam_2010_{45,59,75}.json and the same three for
    rb_tuned (names from scenarios.VALIDATION, so a seventh season or a missing one fails)."""
    assert DATA.is_dir(), f"{DATA} is missing"
    assert len(FILE_NAMES) == 6  # 2 controllers x 3 start days
    assert sorted(p.name for p in DATA.glob("*.json")) == FILE_NAMES


@pytest.mark.parametrize(("controller", "scenario", "name"), SEASON_CASES)
def test_site_data_file_is_under_250_kb(controller, scenario, name):
    """Done when: each of the six files is under 250,000 bytes (250 KB, decimal)."""
    path = DATA / name
    assert path.is_file(), f"{path} is missing"
    size = path.stat().st_size
    assert size < MAX_FILE_BYTES, f"{name}: {size:,} bytes"


@pytest.mark.parametrize(("controller", "scenario", "name"), SEASON_CASES)
def test_site_data_file_is_the_60_day_season_it_is_named_for(controller, scenario, name):
    """Done when: each file parses (finite numbers only) to exactly {"meta", "hourly", "totals"};
    hourly holds exactly the 14 series, each 1,440 points long (60 days x 24 h) with t_hour
    0..1,439; meta names the controller and the season in the file name, 60 days long; and totals
    holds the six scoreboard keys."""
    payload = _load(name)

    assert set(payload) == {"meta", "hourly", "totals"}
    hourly = payload["hourly"]
    assert set(hourly) == set(HOURLY_KEYS)
    assert {k: len(v) for k, v in hourly.items()} == dict.fromkeys(HOURLY_KEYS, N_HOURS)
    assert hourly["t_hour"] == list(range(N_HOURS))
    meta = payload["meta"]
    assert meta["controller"] == controller
    assert meta["scenario"] == {**scenario, "season_length": SEASON_DAYS}
    missing = [k for k in TOTAL_KEYS if k not in payload["totals"]]
    assert not missing, f"{name}: totals lacks {missing}"


@pytest.mark.parametrize(("controller", "scenario", "name"), SEASON_CASES)
def test_site_totals_are_e1s_recorded_numbers(controller, scenario, name):
    """Done when: each file's totals equal its season's row in E1's recorded parquet,
    control/results/validation/<controller>_validation.parquet, to 1e-9 on profit_eur_m2,
    viol_hours_co2, viol_hours_temp, viol_hours_rh, yield_kg_m2 and paper_return. The row is the
    one with this controller and start day, and it is Amsterdam 2010, 60 days, and unique.

    Non-vacuous: rb_default's and rb_tuned's rows differ in every season (start day 59: profit
    -4.697726 vs 1.309663 €/m², viol_hours_rh 629.75 h vs 251.0 h), so a file holding the other
    controller's season, or another start day's, fails."""
    payload = _load(name)
    totals = payload["totals"]

    recorded = pd.read_parquet(RESULTS / f"{controller}_validation.parquet")
    rows = recorded[
        (recorded["controller"] == controller) & (recorded["start_day"] == scenario["start_day"])
    ]
    assert len(rows) == 1, f"{len(rows)} rows for {controller} day {scenario['start_day']}"
    row = rows.iloc[0]
    assert row["location"] == scenario["location"] == "Amsterdam"
    assert int(row["growth_year"]) == scenario["growth_year"] == 2010
    assert int(row["season_length"]) == SEASON_DAYS
    off = {
        k: (totals.get(k), float(row[k]))
        for k in TOTAL_KEYS
        if not (
            isinstance(totals.get(k), int | float) and abs(totals[k] - float(row[k])) <= TOLERANCE
        )
    }
    assert not off, f"{name}: (file, parquet) differ by more than 1e-9 on {off}"


# --- site/index.html (Done when) ----------------------------------------------------------------


def test_site_index_loads_only_local_scripts_and_stylesheets_and_says_simulated():
    """Done when: site/index.html, parsed with html.parser,
    - loads at least one script (<script src>) and one stylesheet (<link rel="stylesheet">), so
      the check below isn't passed by an empty page;
    - has no <script src> or <link href> served by another host: none contains "://" (http://,
      https://) or starts with "//" (protocol-relative). An <a href> to the repo is fine;
    - each script and stylesheet is a relative path to a file inside site/ (a leading "/" or a
      missing file fails), ignoring any ?query or #fragment;
    - its <title> contains "simulated" (any case): a simulator number is never presented as
      real-world evidence (AGENTS.md)."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    page = _Page()
    page.feed(index.read_text(encoding="utf-8"))
    page.close()

    stylesheets = [href for rel, href in page.links if "stylesheet" in rel.split()]
    assert page.scripts, "index.html loads no script"
    assert stylesheets, "index.html loads no stylesheet"
    remote = [u for u in (*page.scripts, *(href for _, href in page.links)) if _is_remote(u)]
    assert not remote, f"index.html loads from another host: {remote}"
    site = SITE.resolve()
    for url in (*page.scripts, *stylesheets):
        target = (SITE / url.strip().split("#")[0].split("?")[0]).resolve()
        assert target.is_relative_to(site), f"{url} is not a relative path inside site/"
        assert target.is_file(), f"{url} is not a file in site/"
    assert "simulated" in page.title.lower(), f"<title> is {page.title!r}"


# --- site/: the whole page (Done when) ----------------------------------------------------------


def test_site_totals_under_1_mb():
    """Done when: every file under site/, recursively (the page and its data), totals under
    1,000,000 bytes (1 MB, decimal). The page's three files and the six data files must be among
    them, so a missing or empty site/ doesn't pass."""
    assert SITE.is_dir(), f"{SITE} is missing"
    files = [p for p in SITE.rglob("*") if p.is_file()]
    present = {p.relative_to(SITE).as_posix() for p in files}
    required = {*PAGE_FILES, *(f"data/{name}" for name in FILE_NAMES)}
    assert required <= present, f"site/ lacks {sorted(required - present)}"
    total = sum(p.stat().st_size for p in files)
    assert total < MAX_SITE_BYTES, f"site/ is {total:,} bytes"

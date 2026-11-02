"""Tests for the demo site (issue #28, D3a; #52, D3b-2a; #49, D3b-2b; #59, S2; #60, S3): site/, a
static page that replays one simulated 60-day season, the grower's tuned rules (rb_tuned, E1, adopted) on
the left against the same rules with the AI co-pilot on top on the right (residual_w05_seed100:
E6's arm residual_w05, seed 100, at its selected checkpoint) by default, from JSON written by
greenleaf_control.export_replay. Since #59 the page picks the season and both sides from
site_build's data/index.json (see the last convention).

The conventions pinned here.
- The page is site/index.html, site/style.css and site/app.js at the repo root, plain files with no
  build step. Its data is site/data/, written by the CLI, never by hand:
  export_replay --controller rb_tuned --split validation --out ../site/data, then the same with
  --controller <seed 100's run dir>, whose files the CLI names residual_w05_seed100_... (#49:
  rb_default left the page).
- VALIDATION is Amsterdam 2010 from start days 45, 59 and 75, so site/data/ holds exactly six
  files, <controller>_<location>_<growth_year>_<start_day>.json, each export_replay's
  {"meta", "hourly", "totals"} for a 60-day season: 60 days x 24 h = 1,440 points in each of the
  14 hourly series.
- Every file says what it is: meta.split is "validation" and meta.note is export_replay's "a
  simulator replay of a validation season, not the final result" (D3b-1). Until #59, index.html's
  static text said it too: "validation" and "not the final result" (#59 moves the label to
  results.json's note; see the last convention).
- Sizes are decimal: 250 KB is 250,000 bytes, 1 MB is 1,000,000 bytes.
- The rules' numbers are E1's: each rb_tuned file's totals equal, to 1e-9, its season's row of
  control/results/validation/rb_tuned_validation.parquet (E1's result-525a6e90, recorded with the
  CLIs), and its meta.run is null (a baseline).
- The co-pilot's provenance is the ledger's: meta.run names E6's registration registered-f2fd71e9,
  its run row for seed 100 (run-5de00091), the checkpoint the registration's one result row
  (result-d1f0b8e7) selected for it ("1500000", the same in
  control/results/validation/registered-f2fd71e9.json), and that result row, which Aarmen adopted
  (decision-eab26197). Its totals aren't pinned to 1e-9 here: seed 100's run dir and its
  selected.json aren't in git, so the orchestrator's export check pins them.
- The ledger and the results/validation files are committed, and only read here, never written.
- meta.dirty isn't pinned: the committed data is exported into a scratch dir from a clean commit
  (dirty false) and then copied in, but a test can't tell how a file was made.
- Issue #52 (D3b-2a, the Instrument restyle): one Google Fonts stylesheet becomes the page's only
  remote load (Aarmen's design comment on #49), and index.html and app.js carry the Instrument
  layout's hooks (nav.side, numbered section.sec blocks, drawGrid).
- Issue #59 (S2, the comparison view; Aarmen, 2026-11-02: "yes build it and use github pages"):
  the page plays any season site_build's data/index.json lists (the three 2012 test cities and
  Amsterdam 2010 validation) with any two of its controllers side by side, and shows what the
  right side changed. That changes some of #28's and #49's pinned requirements. These are
  requirement changes Aarmen asked for, not loosened checks:
  - index.html no longer says "not the final result" or "validation-season" (true of validation
    views only); app.js fills the label from results.json's note, and index.html's static text
    still says "simulated";
  - #49's `const SIDES` (rb_tuned, residual_w05_seed100) gives way to `const DEFAULTS` (left
    rb_tuned, right residual_w05: index.json's controller ids, not file prefixes);
  - the "Co-pilot minus rules" column becomes "Right − left", true of any pair;
  - a failed load clears the difference block (ui.diff) too.
  New: app.js reads data/index.json and data/results.json at relative URLs; index.html has the
  city, start-day (id season, kept), left and right selects and the "What the right side changed"
  block (id diff); and the final-result row id is read from index.json's final_result, never
  typed. site/data/ still holds exactly the six validation files: site_build adds the 2012 replays
  in its output dir, never in site/.
- Issue #60 (S3, the Results section; the same yes from Aarmen): a new section.sec, id results,
  "02 Results", sits between Replay and Scoreboard, which become 03 and 04 with Source, in the
  gutter and in nav.side alike. It charts each split's results.json rows (profit against hours
  outside the safe range, or fruit growth: IQM and 95% CI) from a new site/results.js, loaded after
  app.js, which reads results.json's fields by name, never types a result number (a 2-decimal
  literal equal to a summary.csv number; the scanner below) and never reads `decision` (numbers,
  not verdicts). The readout (id results-readout) and the split's note (id results-note) sit inside
  the section, and site/ stays under 1 MB with results.js among the page's files. The four
  control/results/analysis/<split>/summary.csv files are committed and only read here. The chart's
  positions against tradeoff.png, the axis switch, the readouts (hover, tap, keyboard) and 390 px
  are checked in Chrome (the issue's QA item).
- Issue #63 (#60's review, and the issue's scope comment, 2026-11-02, matching Aarmen's wording in
  his posts): requirement changes, not loosened checks. The readout's hours line adds the sum the
  README and report quote, read from the row's viol_hours_sum_iqm, never typed; the Results
  table's caption says Gain is paired (each season against the tuned rules' same season); and the
  page no longer says "co-pilot": index.html's lede and meta description use residual RL agent
  wording, and the residual arms' names ("Residual agent, ...", site_build's) come from
  index.json, never typed in app.js or results.js.

Every check reads files inside its test body, so while site/ is missing each test FAILS on an
assertion rather than ERRORs. Nothing here builds an env, reads weather or trains.
"""

import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest

from greenleaf_control import ledger, scenarios, site_build

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
DATA = SITE / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "validation"

RULES, COPILOT = CONTROLLERS = ("rb_tuned", "residual_w05_seed100")  # left side, right side (#49)
SPLIT = "validation"
NOTE = (
    "a simulator replay of a validation season, not the final result"  # export_replay's meta.note
)
E6_REGISTERED = "registered-f2fd71e9"  # E6: residual_w05 on 10 fresh seeds (issue #36)
E6_ARM, E6_SEED = "residual_w05", 100  # the experiment's first registered seed, by order
E6_SEED_100 = ("run-5de00091", "1500000")  # seed 100's run row and its selected checkpoint
RUN_KEYS = {"id", "registered", "result", "arm", "seed", "checkpoint", "commit"}  # API-CONTRACT
SEASON_DAYS = 60
N_HOURS = SEASON_DAYS * 24  # 1,440 hourly points
MAX_FILE_BYTES = 250_000  # 250 KB, decimal
MAX_SITE_BYTES = 1_000_000  # 1 MB, decimal
TOLERANCE = 1e-9
PAGE_FILES = ("index.html", "style.css", "app.js", "results.js")  # results.js since #60
FONTS_CSS2 = "https://fonts.googleapis.com/css2?"  # Google Fonts' css2 API (#52)
FONT_FAMILIES = ("Geist Mono", "Schibsted Grotesk", "Source Serif 4")  # decoded family= names
DEFAULTS = {"left": "rb_tuned", "right": "residual_w05"}  # index.json controller ids (#59)
SELECT_IDS = {"city", "season", "left", "right"}  # city, start day, left, right (#59)
DIFF_TITLE = "what the right side changed"  # the difference block's heading (#59), lowercased
FINAL_RESULT_ID = re.compile(r"final-result-[0-9a-f]{8}")  # a ledger final-result row id
# "Right − left": U+2212, its JS or HTML escape, a hyphen or "minus", on lowercased text (#59).
RIGHT_MINUS_LEFT = re.compile(r"right\s*(?:−|\\u2212|&minus;|-|minus)\s*left")
SECTIONS = ("replay", "results", "scoreboard", "source")  # section.sec ids, in page order (#60)
SCRIPTS = ("app.js", "results.js")  # index.html's scripts, in load order (#60)
RESULTS_IDS = {"results-readout", "results-note"}  # inside <section id="results"> (#60)
# "co-pilot", "copilot" or "co pilot" (any Unicode hyphen), on lowercased, collapsed text (#63).
CO_PILOT = re.compile(r"co[-‐‑ ]?pilot")
ANALYSIS = Path(__file__).resolve().parents[1] / "results" / "analysis"
RESULT_SPLITS = ("amsterdam_2012", "london_2012", "abbotsford_2012", "validation")  # its dirs
# summary.csv's metrics whose _iqm, _lo and _hi results.json carries (site_build's, #58).
RESULT_METRICS = (
    "profit_eur_m2",
    "profit_gain",
    "viol_hours_total",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "yield_kg_m2",
)
# The results.json fields results.js must name, each as a whole word (#60 steps 2 and 3).
READ_NAMES = (
    "profit_eur_m2",
    "profit_gain",
    "viol_hours_total",
    "yield_kg_m2",
    "n_runs",
    "invalid_episodes",
    "controller",
    "note",
    "iqm",
    "lo",
    "hi",
)
# A number literal with 2 or more decimals, not part of a name, a longer number or a member access.
TYPED = re.compile(r"(?<![\w$.])\d+\.\d{2,}(?![\w$])")
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
RULES_CASES = [p for p in SEASON_CASES if p.values[0] == RULES]
COPILOT_CASES = [p for p in SEASON_CASES if p.values[0] == COPILOT]
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


class _Layout(HTMLParser):
    """index.html's Instrument hooks: whether it has a <nav class="side">, and for each
    <section class="sec">, in page order, the stripped text of every <span class="num"> inside it
    (a span belongs to its innermost enclosing section.sec)."""

    def __init__(self) -> None:
        super().__init__()
        self.nav_side = False
        self.sec_nums: list[list[str]] = []
        self._sections: list[int | None] = []  # open <section>s: index into sec_nums, or None
        self._num_owner: int | None = None  # the open span.num's section.sec, if any
        self._num_text = ""

    def handle_starttag(self, tag, attrs):
        classes = (dict(attrs).get("class") or "").split()
        if tag == "nav" and "side" in classes:
            self.nav_side = True
        if tag == "section":
            if "sec" in classes:
                self.sec_nums.append([])
                self._sections.append(len(self.sec_nums) - 1)
            else:
                self._sections.append(None)
        if tag == "span" and "num" in classes:
            self._num_owner = next((i for i in reversed(self._sections) if i is not None), None)
            self._num_text = ""

    def handle_endtag(self, tag):
        if tag == "section" and self._sections:
            self._sections.pop()
        if tag == "span" and self._num_owner is not None:
            self.sec_nums[self._num_owner].append(self._num_text.strip())
            self._num_owner = None

    def handle_data(self, data):
        if self._num_owner is not None:
            self._num_text += data


def _is_remote(url: str) -> bool:
    """A URL another host serves: a scheme (http://, https://, ...) or protocol-relative (//)."""
    url = url.strip()
    return "://" in url or url.startswith("//")


class _Text(HTMLParser):
    """index.html's static text: the character data inside <body>, outside <script>, <style> and
    <template>, joined. Attribute values (a meta description, an aria-label) aren't text."""

    SKIP = ("script", "style", "template")

    def __init__(self) -> None:
        super().__init__()
        self.text = ""
        self._in_body = False
        self._skipping = 0  # open <script>/<style>/<template> elements

    def handle_starttag(self, tag, attrs):
        if tag == "body":
            self._in_body = True
        if tag in self.SKIP:
            self._skipping += 1

    def handle_endtag(self, tag):
        if tag == "body":
            self._in_body = False
        if tag in self.SKIP and self._skipping:
            self._skipping -= 1

    def handle_data(self, data):
        if self._in_body and not self._skipping:
            self.text += data


class _Ids(HTMLParser):
    """index.html's elements that carry an id: (tag, id) for each, in page order."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        found = dict(attrs).get("id")
        if found is not None:
            self.ids.append((tag, found.strip()))


def _close(text: str, start: int) -> int:
    """The index just past the bracket that closes the "(" or "{" at text[start]."""
    opener = text[start]
    closer = {"(": ")", "{": "}"}[opener]
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == opener) - (text[i] == closer)
        if depth == 0:
            return i + 1
    raise AssertionError(f"site/app.js: the {opener!r} at offset {start} is never closed")


def _catch_handlers(js: str) -> list[str]:
    """The source of every catch handler in app.js: a promise's .catch(...) argument, or a try's
    catch (err) { ... } / catch { ... } block, each matched to its closing bracket."""
    found = re.finditer(r"\.catch\s*\(|\bcatch\s*(?:\(\s*\w*\s*\)\s*)?\{", js)
    return [js[m.end() - 1 : _close(js, m.end() - 1)] for m in found]


class _Nav(HTMLParser):
    """index.html's order: the id of each <section class="sec"> in page order (None if it has
    none), and for each <a> inside <nav class="side">, (its href, the stripped text of its
    <span>)."""

    def __init__(self) -> None:
        super().__init__()
        self.sections: list[str | None] = []
        self.links: list[tuple[str, str]] = []
        self._in_nav = False
        self._href: str | None = None  # the open nav link's href
        self._in_span = False
        self._span = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        classes = (a.get("class") or "").split()
        if tag == "section" and "sec" in classes:
            self.sections.append(a.get("id"))
        if tag == "nav" and "side" in classes:
            self._in_nav = True
        if self._in_nav and tag == "a":
            self._href, self._span = (a.get("href") or "").strip(), ""
        if self._href is not None and tag == "span":
            self._in_span = True

    def handle_endtag(self, tag):
        if tag == "nav":
            self._in_nav = False
        if tag == "span":
            self._in_span = False
        if tag == "a" and self._href is not None:
            self.links.append((self._href, self._span.strip()))
            self._href = None

    def handle_data(self, data):
        if self._in_span:
            self._span += data


class _ResultsIds(HTMLParser):
    """Whether index.html has a <section id="results">, and the ids of the elements inside it (its
    own excluded; nested <section>s are counted, so its own </section> ends it)."""

    def __init__(self) -> None:
        super().__init__()
        self.found = False
        self.ids: set[str] = set()
        self._depth = 0  # <section>s open since #results opened, itself included

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self._depth:
            if a.get("id") is not None:
                self.ids.add(a["id"].strip())
            if tag == "section":
                self._depth += 1
        elif tag == "section" and (a.get("id") or "").strip() == "results":
            self.found = True
            self._depth = 1

    def handle_endtag(self, tag):
        if tag == "section" and self._depth:
            self._depth -= 1


class _ResultsCaptions(HTMLParser):
    """Whether index.html has a <section id="results">, and the text of each <caption> inside it
    (nested <section>s counted, as in _ResultsIds)."""

    def __init__(self) -> None:
        super().__init__()
        self.found = False
        self.captions: list[str] = []
        self._depth = 0  # <section>s open since #results opened, itself included
        self._in_caption = False

    def handle_starttag(self, tag, attrs):
        if self._depth:
            if tag == "section":
                self._depth += 1
            if tag == "caption":
                self._in_caption = True
                self.captions.append("")
        elif tag == "section" and (dict(attrs).get("id") or "").strip() == "results":
            self.found = True
            self._depth = 1

    def handle_endtag(self, tag):
        if tag == "section" and self._depth:
            self._depth -= 1
        if tag == "caption":
            self._in_caption = False

    def handle_data(self, data):
        if self._in_caption:
            self.captions[-1] += data


def _strip_comments(js: str) -> str:
    """js without its /* ... */ blocks and its // comments, a // that starts a line or follows
    whitespace or ";" (so a URL's "://" in a string stays)."""
    js = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    return re.sub(r"(^|[\s;])//[^\n]*", r"\1", js, flags=re.M)


def _summary_values() -> set[float]:
    """The result numbers results.json carries, from the four committed summary.csv files (only
    read; random's row included though site_build drops it): every <metric>_{iqm,lo,hi},
    viol_hours_sum_iqm and rb_tuned_* cell, each as round(abs(x), 2), dropping those that round
    below 0.10 (so a small layout constant such as a 6% pad, 0.06, can't collide)."""
    columns = {f"{m}_{s}" for m in RESULT_METRICS for s in ("iqm", "lo", "hi")}
    columns.add("viol_hours_sum_iqm")
    values = set()
    for split in RESULT_SPLITS:
        path = ANALYSIS / split / "summary.csv"
        assert path.is_file(), f"{path} is missing"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cells = [v for k, v in row.items() if k in columns or k.startswith("rb_tuned_")]
                values |= {round(abs(float(v)), 2) for v in cells}
    return {v for v in values if v >= 0.10}


def _typed_values(js: str, values: set[float]) -> list[str]:
    """The number literals in js, comments stripped, with 2 or more decimals whose absolute value
    rounded to 2 decimals is one of values: a result typed by hand."""
    found = TYPED.findall(_strip_comments(js))
    return [lit for lit in found if round(abs(float(lit)), 2) in values]


# --- site/data/: the six files (Done when) ------------------------------------------------------


def test_site_data_holds_exactly_the_six_validation_files():
    """Done when (#49 step 1): site/data/ holds exactly the six JSON files the CLI writes for
    VALIDATION, one per controller and season: rb_tuned_Amsterdam_2010_{45,59,75}.json and
    residual_w05_seed100_Amsterdam_2010_{45,59,75}.json (names from scenarios.VALIDATION, so a
    seventh season or a missing one fails). #28's rb_default files have left the page."""
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
    0..1,439; meta names the controller and the season in the file name, 60 days long; totals
    holds the six scoreboard keys; and (#49) meta says what the file is: split "validation" and the
    note "a simulator replay of a validation season, not the final result", exactly. #28's files
    said "a simulator run, not a registered result" and had no split."""
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
    assert meta.get("split") == SPLIT, f"{name}: meta.split is {meta.get('split')!r}"
    assert meta.get("note") == NOTE, f"{name}: meta.note is {meta.get('note')!r}"


@pytest.mark.parametrize(("controller", "scenario", "name"), RULES_CASES)
def test_site_totals_are_e1s_recorded_numbers(controller, scenario, name):
    """Done when: each rb_tuned file's totals equal its season's row in E1's recorded parquet,
    control/results/validation/rb_tuned_validation.parquet, to 1e-9 on profit_eur_m2,
    viol_hours_co2, viol_hours_temp, viol_hours_rh, yield_kg_m2 and paper_return. The row is the
    one with this controller and start day, and it is Amsterdam 2010, 60 days, and unique. And
    (#49) its meta.run is present and null: rb_tuned is a baseline, not a registered run.

    Non-vacuous: rb_default's and rb_tuned's rows differ in every season (start day 59: profit
    -4.697726 vs 1.309663 €/m², viol_hours_rh 629.75 h vs 251.0 h), so a file holding another
    controller's season, or another start day's, fails."""
    payload = _load(name)
    totals = payload["totals"]
    meta = payload["meta"]

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
    assert "run" in meta and meta["run"] is None, (
        f"{name}: meta.run is {meta.get('run', '<absent>')!r}, not null (a baseline)"
    )


@pytest.mark.parametrize(("controller", "scenario", "name"), COPILOT_CASES)
def test_site_copilot_file_is_e6s_seed_100_at_its_selected_checkpoint(controller, scenario, name):
    """Done when (#49): each residual_w05_seed100 file's meta.run is the provenance export_replay
    reads from the ledger (API-CONTRACT: exactly {id, registered, result, arm, seed, checkpoint,
    commit}), checked against the committed ledger and E6's selection record, both only read:
    - registered "registered-f2fd71e9", arm "residual_w05", seed 100, and the ledger's run row
      named run.id is that registration's row for that arm and seed;
    - (run.id, run.checkpoint) is an entry of
      control/results/validation/registered-f2fd71e9.json's "selected", and it is
      ("run-5de00091", "1500000"), the issue's known answer;
    - run.result is the ledger's one result row for registered-f2fd71e9 (result-d1f0b8e7), whose
      own "selected" gives run.id the same checkpoint, and the ledger has an "adopted" decision row
      for that result (decision-eab26197);
    - run.commit is a full 40-hex git commit (the footer shows its first 7 characters).
    Non-vacuous: totals.profit_eur_m2 differs from the rb_tuned file's for the same season, so a
    copy of the rules' season under the co-pilot's name fails. Which side is higher isn't checked:
    that is E6's result, an experiment, not a test."""
    payload = _load(name)
    run = payload["meta"].get("run")
    assert isinstance(run, dict), f"{name}: meta.run is {run!r}, not a registered run's provenance"
    assert set(run) == RUN_KEYS, f"{name}: meta.run has the keys {sorted(run)}"
    assert (run["registered"], run["arm"], run["seed"]) == (E6_REGISTERED, E6_ARM, E6_SEED)

    run_rows = [r for r in ledger.rows(kind="run", path=ledger.LEDGER) if r["row_id"] == run["id"]]
    assert [(r["registered"], r["arm"], r["seed"]) for r in run_rows] == [
        (E6_REGISTERED, E6_ARM, E6_SEED)
    ], f"{name}: the ledger's run rows named {run['id']!r} are {run_rows}"
    record = json.loads((RESULTS / f"{E6_REGISTERED}.json").read_text(encoding="utf-8"))
    pair = (run["id"], run["checkpoint"])
    assert pair in record["selected"].items(), (
        f"{name}: {pair} is not in {E6_REGISTERED}.json's selected {record['selected']}"
    )
    assert pair == E6_SEED_100, f"{name}: (run.id, run.checkpoint) is {pair}"

    results = [
        r
        for r in ledger.rows(kind="result", path=ledger.LEDGER)
        if r["registered"] == E6_REGISTERED
    ]
    assert [r["row_id"] for r in results] == [run["result"]], (
        f"{name}: run.result is {run['result']!r}; the ledger's result rows for {E6_REGISTERED}"
        f" are {[r['row_id'] for r in results]}"
    )
    assert results[0]["selected"].get(run["id"]) == run["checkpoint"]
    decisions = [
        r["decision"]
        for r in ledger.rows(kind="decision", path=ledger.LEDGER)
        if r["result"] == run["result"]
    ]
    assert "adopted" in decisions, f"{name}: the decisions on {run['result']} are {decisions}"
    commit = run["commit"]
    assert isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), (
        f"{name}: run.commit is {commit!r}"
    )

    rules = _load(_file_name(RULES, scenario))
    profit = payload["totals"]["profit_eur_m2"]
    assert profit != rules["totals"]["profit_eur_m2"], (
        f"{name}: profit {profit} €/m² is the rb_tuned file's for the same season"
    )


# --- site/index.html (Done when) ----------------------------------------------------------------


def test_site_index_loads_local_files_plus_one_google_fonts_stylesheet_and_says_simulated():
    """Done when (#28, changed by #52 step 4): site/index.html, parsed with html.parser,
    - loads at least one script (<script src>) and one local stylesheet (<link rel="stylesheet">),
      so the checks below aren't passed by an empty page;
    - loads exactly one thing from another host (a <script src> or <link href> that contains "://"
      or starts with "//"), and it is a <link rel="stylesheet"> to Google Fonts' css2 API: its
      href starts "https://fonts.googleapis.com/css2?", its family= values name exactly Geist
      Mono, Schibsted Grotesk and Source Serif 4 (Geist+Mono, Schibsted+Grotesk, Source+Serif+4
      in the URL, any :axis@weights after a name), and it has display=swap. No remote script and
      no other remote <link> (a preconnect included). An <a href> to the repo is fine.
      A requirement change, not a loosened check: Aarmen's Instrument design comment on #49 sets
      these three Google Fonts, and #52 (out of scope: self-hosting fonts) loads them from Google.
      Until #52 any remote <link href> failed here;
    - every script and every other stylesheet is a relative path to a file inside site/ (a
      leading "/" or a missing file fails), ignoring any ?query or #fragment;
    - its <title> contains "simulated" (any case): a simulator number is never presented as
      real-world evidence (AGENTS.md)."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    page = _Page()
    page.feed(index.read_text(encoding="utf-8"))
    page.close()

    stylesheets = [href for rel, href in page.links if "stylesheet" in rel.split()]
    local_sheets = [href for href in stylesheets if not _is_remote(href)]
    assert page.scripts, "index.html loads no script"
    assert local_sheets, "index.html loads no local stylesheet"
    remote = [u for u in (*page.scripts, *(href for _, href in page.links)) if _is_remote(u)]
    fonts = [href for href in stylesheets if _is_remote(href)]
    assert len(fonts) == 1 and remote == fonts, (
        f"index.html must load exactly one thing from another host, the Google Fonts stylesheet; "
        f"it loads {remote} (remote stylesheets: {fonts})"
    )
    href = fonts[0].strip()
    assert href.startswith(FONTS_CSS2), f"{href} is not Google Fonts' css2 API"
    query = parse_qs(urlsplit(href).query)
    families = sorted(v.split(":")[0] for v in query.get("family", []))
    assert families == sorted(FONT_FAMILIES), f"{href} names the families {families}"
    assert query.get("display") == ["swap"], f"{href} lacks display=swap"
    site = SITE.resolve()
    for url in (*page.scripts, *local_sheets):
        target = (SITE / url.strip().split("#")[0].split("?")[0]).resolve()
        assert target.is_relative_to(site), f"{url} is not a relative path inside site/"
        assert target.is_file(), f"{url} is not a file in site/"
    assert "simulated" in page.title.lower(), f"<title> is {page.title!r}"


def test_site_index_has_the_instrument_layout_hooks():
    """Done when (#52 step 1, the Instrument layout from Aarmen's design comment on #49):
    site/index.html, parsed with html.parser, has
    - a <nav class="side"> (the sidebar nav);
    - at least three <section class="sec"> blocks (Replay, Scoreboard, Source; #60 adds Results
      second, so four, whose order the Results-section tests pin);
    - in each, a <span class="num"> gutter number; the first one in each section, in page order,
      reads 01, 02, 03, ... (two digits, consecutive from 01).
    The grid's <canvas data-g> isn't required here: app.js builds it."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    layout = _Layout()
    layout.feed(index.read_text(encoding="utf-8"))
    layout.close()

    assert layout.nav_side, 'index.html has no <nav class="side">'
    n = len(layout.sec_nums)
    assert n >= 3, f'index.html has {n} <section class="sec">, not at least 3'
    bare = [i + 1 for i, nums in enumerate(layout.sec_nums) if not nums]
    assert not bare, f'section.sec number(s) {bare}, in page order, hold no <span class="num">'
    gutter = [nums[0] for nums in layout.sec_nums]
    assert gutter == [f"{i:02d}" for i in range(1, n + 1)], f"the gutter numbers are {gutter}"


def test_site_index_header_is_true_of_test_and_validation_views_and_says_simulated():
    """Done when (#59 step 5; a requirement change from #49 step 3, Aarmen 2026-11-02, not a
    loosened check): the header is true of a 2012 test view and a validation view alike, so
    site/index.html, all of it (the meta description and other attributes included), whitespace
    collapsed, any case, no longer contains "not the final result" or "validation-season": a 2012
    test view is the final result. app.js fills the label from results.json's note instead. And
    index.html's static text, the character data inside <body> outside <script>, <style> and
    <template>, still contains "simulated", so every view says it before app.js runs (AGENTS.md).
    That the label switches with the city is checked in Chrome (the issue's QA item). #49's page
    says "Validation-season replay, not the final result"."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    html = index.read_text(encoding="utf-8")
    page = _Text()
    page.feed(html)
    page.close()

    whole = " ".join(html.split()).lower()
    stale = [p for p in ("not the final result", "validation-season") if p in whole]
    assert not stale, f"site/index.html still says {stale}, untrue of a 2012 test view"
    text = " ".join(page.text.split()).lower()
    assert "simulated" in text, "index.html's static text doesn't say simulated"


def test_site_index_has_city_start_day_left_and_right_selects():
    """Done when (#59 step 1): site/index.html, parsed with html.parser, has a <select> with each
    of the ids city (Amsterdam, London, Abbotsford 2012 and Amsterdam 2010 validation), season (the
    start day: #49's id, kept, since id="day" is the scrub's output), left and right (the two
    controllers). Other selects (speed) are free. Their options come from index.json, filled by
    app.js, and the defaults, the hash and the same-controller rule are checked in Chrome (the
    issue's QA item). #49's page has the season and speed selects only."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    page = _Ids()
    page.feed(index.read_text(encoding="utf-8"))
    page.close()

    selects = {found for tag, found in page.ids if tag == "select"}
    missing = sorted(SELECT_IDS - selects)
    assert not missing, f"index.html has no <select> with the id(s) {missing}; its: {selects}"


def test_site_index_has_the_what_the_right_side_changed_block():
    """Done when (#59 step 3): site/index.html has an element with id="diff", where app.js draws
    the six controls' right-minus-left strips, and its static text (as above, whitespace
    collapsed, any case) contains the block's heading, "What the right side changed". The strips,
    their legend and their end values (one hour checked by hand against the two replay files) are
    checked in Chrome (the issue's QA item). #49's page has neither."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    html = index.read_text(encoding="utf-8")
    ids = _Ids()
    ids.feed(html)
    ids.close()
    page = _Text()
    page.feed(html)
    page.close()

    assert "diff" in {found for _, found in ids.ids}, 'index.html has no element with id="diff"'
    text = " ".join(page.text.split()).lower()
    assert DIFF_TITLE in text, f'index.html\'s static text lacks "{DIFF_TITLE}"'


def test_site_app_js_defines_draw_grid():
    """Done when (#52 step 3): site/app.js defines a function named drawGrid, the hour-by-hour dot
    grid (`function drawGrid(` or `const|let|var drawGrid =`). A plain text check: the grid's
    drawing is checked by eye in the browser (the issue's QA item)."""
    app = SITE / "app.js"
    assert app.is_file(), f"{app} is missing"
    text = app.read_text(encoding="utf-8")
    defined = re.search(r"\bfunction\s+drawGrid\s*\(|\b(?:const|let|var)\s+drawGrid\s*=", text)
    assert defined, "site/app.js defines no drawGrid function"


def test_site_app_js_defaults_are_the_tuned_rules_left_and_the_copilot_right():
    """Done when (#59 step 1; a requirement change from #49 step 2, Aarmen 2026-11-02, not a
    loosened check): app.js's `const DEFAULTS = {...}` sets left "rb_tuned" and right
    "residual_w05" (keys bare or quoted, any quote; other keys free): the tuned rules on the left
    and the co-pilot, fixed penalty, on the right, named by index.json's controller ids, not by
    file prefix (residual_w05_seed100 is the validation files' prefix, read from index.json).
    Non-vacuous: both are ids site_build writes, and validation, which offers only its two,
    offers both, so the default pair plays in every city. The default city and start day are
    index.json's first, read, not typed; that and the hash fallback are checked in Chrome (the
    issue's QA item). #49's app.js has `const SIDES` and no DEFAULTS."""
    ids = set(DEFAULTS.values())
    assert ids <= set(site_build.VALIDATION) <= set(site_build.CONTROLLERS)
    app = SITE / "app.js"
    assert app.is_file(), f"{app} is missing"
    js = app.read_text(encoding="utf-8")
    found = re.search(r"\bconst\s+DEFAULTS\s*=\s*(?:Object\.freeze\(\s*)?\{(.*?)\}", js, re.S)
    assert found, "site/app.js defines no `const DEFAULTS = {...}`"

    pairs = dict(re.findall(r"[\"'`]?(\w+)[\"'`]?\s*:\s*[\"'`]([\w-]+)[\"'`]", found.group(1)))
    got = {k: pairs.get(k) for k in DEFAULTS}
    assert got == DEFAULTS, f"DEFAULTS sets {got}"


def test_site_app_js_reads_index_and_results_json_at_relative_urls():
    """Done when (#59 steps 1 and 5): app.js loads site_build's data/index.json (the splits, their
    start days, controllers and file prefixes, and the final-result row) and data/results.json
    (each split's note, and n_runs for "the lowest of its k"), so it names both paths; and no
    string in app.js starts "/data/": GitHub Pages serves the site under /greenleaf-dashboard/,
    where a root-relative URL misses (S4). That both load and every city plays is checked in
    Chrome on site_build's output (the issue's QA item). #49's app.js types its file names and
    reads neither."""
    app = SITE / "app.js"
    assert app.is_file(), f"{app} is missing"
    js = app.read_text(encoding="utf-8")

    missing = [p for p in ("data/index.json", "data/results.json") if p not in js]
    assert not missing, f"site/app.js never names {missing}"
    rooted = re.findall(r"[\"'`]/data/[^\"'`]*", js)
    assert not rooted, f"site/app.js has root-relative data URLs {rooted}"


def test_site_app_js_failed_load_clears_the_panels_scoreboard_and_diff():
    """Done when (#49 step 6, #28's reviewer carry; extended by #59 step 3): a failed season load
    doesn't leave the old season's panels, scoreboard or (#59) right-minus-left strips under the
    new season's name. A light text check: app.js's catch handlers (a promise's .catch(...)
    argument or a try's catch block, where a failed fetch in load or start lands), taken together,
    reference ui.panels, ui.score and ui.diff themselves, not only through a helper they call. How
    they clear or hide them is free; that nothing stale shows is checked in Chrome with a data
    file removed (the issue's QA item). #28's only handler, start's .catch, touches ui.status
    alone; #49's touches ui.panels and ui.score but there is no ui.diff yet."""
    app = SITE / "app.js"
    assert app.is_file(), f"{app} is missing"
    handlers = _catch_handlers(app.read_text(encoding="utf-8"))
    assert handlers, "site/app.js has no catch handler"

    failure = "\n".join(handlers)
    missing = [el for el in ("ui.panels", "ui.score", "ui.diff") if el not in failure]
    assert not missing, (
        f"no catch handler in site/app.js touches {missing}: a failed load leaves the old season"
    )


def test_site_scoreboard_has_a_right_minus_left_column():
    """Done when (#59 step 4; a requirement change from #49 step 5, Aarmen 2026-11-02, not a
    loosened check): the scoreboard's third column is #49's generalised to any pair, headed
    "Right − left" (U+2212, its JS or HTML escape, a hyphen or "minus"; any case, whitespace
    collapsed), in app.js, which builds the table, or in index.html; and neither says "Co-pilot
    minus rules" any more, untrue of most pairs. A text check: its signed values, computed in JS
    from the two files' totals, are checked in Chrome (the issue's QA item; #49's day 45: profit
    -5.03 vs -4.44, +0.59). #49's column is "Co-pilot minus rules"."""
    sources = [SITE / "app.js", SITE / "index.html"]
    missing = [p for p in sources if not p.is_file()]
    assert not missing, f"{missing} missing"

    texts = [" ".join(p.read_text(encoding="utf-8").split()).lower() for p in sources]
    assert any(RIGHT_MINUS_LEFT.search(t) for t in texts), (
        'neither site/app.js nor site/index.html has the "Right − left" column'
    )
    stale = [p.name for p, t in zip(sources, texts, strict=True) if "co-pilot minus rules" in t]
    assert not stale, f'{stale} still head the column "Co-pilot minus rules"'


def test_site_footer_provenance_is_read_from_meta_run_not_typed():
    """Done when (#49 step 4): the footer's provenance for the AI side comes from the AI file's
    meta.run, never typed by hand:
    - app.js names each meta.run field the footer shows, arm, seed, checkpoint, result and commit,
      as a whole word (however it reads them: run.arm, a destructuring, run["arm"]);
    - neither app.js nor index.html contains the run's id, its result row id, its checkpoint
      ("1500000", or "1,500,000" for a digit checkpoint) or its commit's first 7 characters, all
      read from residual_w05_seed100_Amsterdam_2010_45.json.
    The E1 line for the tuned rules (result-525a6e90) stays typed: rb_tuned's meta.run is null.
    #28's app.js names none of the five fields."""
    app = SITE / "app.js"
    index = SITE / "index.html"
    assert app.is_file() and index.is_file(), f"{app} or {index} is missing"
    js = app.read_text(encoding="utf-8")
    html = index.read_text(encoding="utf-8")
    fields = ("arm", "seed", "checkpoint", "result", "commit")
    unread = [k for k in fields if not re.search(rf"\b{k}\b", js)]
    assert not unread, f"site/app.js never names meta.run's {unread}"

    run = _load(_file_name(COPILOT, scenarios.VALIDATION[0]))["meta"].get("run")
    assert isinstance(run, dict), f"the co-pilot file's meta.run is {run!r}"
    checkpoint = str(run["checkpoint"])
    literals = [run["id"], run["result"], checkpoint, str(run["commit"])[:7]]
    if checkpoint.isdigit():
        literals.append(f"{int(checkpoint):,}")
    typed = [v for v in literals if v in js or v in html]
    assert not typed, f"site/ types meta.run's {typed} by hand"


def test_site_final_result_row_is_read_from_index_json_not_typed():
    """Done when (#59 Context, the honesty text; step 6): a 2012 test view names the final-result
    row, read from index.json, never typed:
    - app.js names final_result as a whole word, index.json's key (however it reads it:
      index.final_result, a destructuring, index["final_result"]);
    - neither app.js nor index.html contains a final-result row id, "final-result-" and 8 hex
      digits. Non-vacuous: the ledger's one final-result row (only read) has that form, so typing
      it fails.
    That a test view shows "2012 test weather, scored once in the final" with the row id, and a
    validation view doesn't, is checked in Chrome (the issue's QA item). #49's app.js names
    neither."""
    finals = [r["row_id"] for r in ledger.rows(kind="final-result", path=ledger.LEDGER)]
    assert len(finals) == 1 and FINAL_RESULT_ID.fullmatch(finals[0]), (
        f"the ledger's final-result rows are {finals}"
    )
    app = SITE / "app.js"
    index = SITE / "index.html"
    assert app.is_file() and index.is_file(), f"{app} or {index} is missing"
    js = app.read_text(encoding="utf-8")
    html = index.read_text(encoding="utf-8")

    assert re.search(r"\bfinal_result\b", js), "site/app.js never names index.json's final_result"
    typed = sorted({m for t in (js, html) for m in FINAL_RESULT_ID.findall(t)})
    assert not typed, f"site/ types the final-result row id(s) {typed} by hand"


# --- the Results section: site/index.html and site/results.js (#60, Done when) ------------------


def test_site_results_section_is_second_in_page_and_nav_order():
    """Done when (#60 step 1): site/index.html, parsed with html.parser, has its
    <section class="sec"> blocks in page order with exactly the ids replay, results, scoreboard,
    source, and its <nav class="side"> links them in the same order, each <a>'s href "#<id>" and its
    <span> reading 01, 02, 03, 04: "02 Results" is new, Scoreboard and Source move to 03 and 04.
    (The gutter's span.num numbering is the Instrument hooks test's.) #59's page has replay,
    scoreboard and source, linked 01 to 03."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    nav = _Nav()
    nav.feed(index.read_text(encoding="utf-8"))
    nav.close()

    assert nav.sections == list(SECTIONS), f"the section.sec ids, in page order, are {nav.sections}"
    want = [(f"#{s}", f"{i:02d}") for i, s in enumerate(SECTIONS, 1)]
    assert nav.links == want, f"nav.side's (href, span) links are {nav.links}"


def test_site_index_loads_app_js_then_results_js():
    """Done when (#60 step 2, the map's file plan): index.html's <script src> values, any ?query
    or #fragment dropped, are exactly app.js then results.js: the chart lives in its own file,
    loaded after app.js, whose loads it reuses. #59's page loads app.js alone."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    page = _Page()
    page.feed(index.read_text(encoding="utf-8"))
    page.close()

    scripts = [s.strip().split("#")[0].split("?")[0] for s in page.scripts]
    assert scripts == list(SCRIPTS), f"index.html loads the scripts {scripts}"


def test_site_results_section_holds_the_readout_and_the_split_note():
    """Done when (#60 steps 3 and 5): index.html has a <section id="results">, and inside it an
    element with id="results-readout" (the fixed readout below the chart that hover, tap and focus
    fill) and one with id="results-note" (where the split's results.json note is shown under the
    chart). That the readout follows the pointer and keyboard focus and the note switches with the
    city is checked in Chrome (the issue's QA item). #59's page has neither."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    inside = _ResultsIds()
    inside.feed(index.read_text(encoding="utf-8"))
    inside.close()

    assert inside.found, 'index.html has no <section id="results">'
    missing = sorted(RESULTS_IDS - inside.ids)
    assert not missing, f"<section id=results> holds no element with the id(s) {missing}"


def test_site_results_js_reads_the_summary_fields_and_never_decision():
    """Done when (#60 steps 2, 3 and 5; Out of scope: verdicts): site/results.js, comments
    stripped, names each results.json field the chart, readout and note read as a whole word:
    profit_eur_m2, profit_gain, viol_hours_total, yield_kg_m2 (the metrics), iqm, lo and hi (their
    statistics; so a metric's keys are built from the metric and the statistic, e.g.
    `${m}_${s}`), n_runs, invalid_episodes, controller (the page id that picks a row) and note
    (the split's note). And it never names `decision`, the rows' adopt-or-drop verdict: the section
    shows numbers, not verdicts, and colours no controller pass or fail. How it reads them is free.
    #59 has no results.js."""
    path = SITE / "results.js"
    assert path.is_file(), f"{path} is missing"
    js = _strip_comments(path.read_text(encoding="utf-8"))

    unread = [k for k in READ_NAMES if not re.search(rf"\b{k}\b", js)]
    assert not unread, f"site/results.js never names {unread} as a whole word"
    assert not re.search(r"\bdecision\b", js), "site/results.js names `decision`: a verdict"


def test_site_results_js_types_no_result_number():
    """Done when (#60 step 5): the chart draws from results.json's values, never from numbers typed
    in JS. Every number literal in site/results.js with 2 or more decimals, comments stripped
    (TYPED: not part of a name, a longer number or a member access), is compared, as
    round(abs(x), 2), with every result number results.json carries from the four committed
    summary.csv files: each <metric>_{iqm,lo,hi}, viol_hours_sum_iqm and rb_tuned_* cell (random's
    row too), as round(abs(x), 2), those below 0.10 dropped. A literal in that set fails. app.js
    isn't scanned: it draws no result, and its replay constants 0.25 and 0.75 equal result values.

    Non-vacuous, both guards checked first:
    - the set holds over 200 numbers (517 in the committed files on 2026-11-02);
    - the scanner flags a planted "x = -1.92;": the tuned rules' Amsterdam 2012 profit IQM,
      -1.9175741968487416 €/m², is -1.92 to 2 decimals (abs 1.92), the obvious number to type."""
    values = _summary_values()
    assert len(values) > 200, f"only {len(values)} result numbers: the scanner would test nothing"
    planted = _typed_values("x = -1.92;", values)
    assert planted == ["1.92"], f"the scanner finds {planted} in a planted -1.92, not ['1.92']"

    path = SITE / "results.js"
    assert path.is_file(), f"{path} is missing"
    typed = _typed_values(path.read_text(encoding="utf-8"), values)
    assert not typed, f"site/results.js types result numbers by hand: {typed}"


# --- #63: the Results wording and the residual agent's name (Done when) ------------------------


def test_site_results_js_reads_the_readmes_hours_sum_from_the_row():
    """Done when (#63 step 1, #60's review): the readout's hours line says which statistic the
    README quotes. Its total is the IQM of each season's total (viol_hours_total_iqm); the README
    and report quote the sum of the three per-constraint IQMs (summary.csv's viol_hours_sum_iqm),
    and the two differ. So site/results.js, comments stripped, names viol_hours_sum_iqm as a whole
    word: the clause "these three add to <sum> h" reads it from the row, never types it (the
    typed-number scanner above still runs over the file). The clause's wording and its 365.2 h
    for Amsterdam 2012's tuned rules are checked in Chrome (the issue's QA item). #60's
    results.js never names it.

    Non-vacuous: in the committed amsterdam_2012 summary.csv (only read), the tuned rules' sum and
    total differ, so the readout's total can't stand in for the README's number."""
    with (ANALYSIS / "amsterdam_2012" / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = {row["label"]: row for row in csv.DictReader(f)}
    rules = rows["rb_tuned"]
    # CO2 0.0 h + temperature 83.125 h + humidity 282.0625 h = 365.1875 h (365.2 to 1 decimal);
    # the IQM of each season's total is 366.25 h.
    assert float(rules["viol_hours_sum_iqm"]) == 0.0 + 83.125 + 282.0625 == 365.1875
    assert float(rules["viol_hours_total_iqm"]) == 366.25

    path = SITE / "results.js"
    assert path.is_file(), f"{path} is missing"
    js = _strip_comments(path.read_text(encoding="utf-8"))
    assert re.search(r"\bviol_hours_sum_iqm\b", js), (
        "site/results.js never names viol_hours_sum_iqm: the README's hours sum isn't in the readout"
    )


def test_site_results_table_caption_says_gain_is_paired():
    """Done when (#63 step 2, #60's review): the Results table's caption, the one <caption> inside
    <section id="results">, whitespace collapsed, any case, contains "paired": Gain is the IQM of
    each season's difference from the tuned rules on the same season (like-for-like), not the
    distance from a controller's dot to the tuned rules' diamond, which compares two IQMs. #60's
    caption says only "its gain over the tuned rules"."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    inside = _ResultsCaptions()
    inside.feed(index.read_text(encoding="utf-8"))
    inside.close()

    assert inside.found, 'index.html has no <section id="results">'
    assert len(inside.captions) == 1, (
        f'<section id="results"> holds {len(inside.captions)} <caption>s, not the table\'s one'
    )
    caption = " ".join(inside.captions[0].split()).lower()
    assert "paired" in caption, f"the Results table's caption doesn't say paired: {caption!r}"


def test_site_says_residual_agent_not_co_pilot():
    """Done when (#63's scope comment, 2026-11-02, Aarmen's wording in his posts and resume; a
    requirement change, not a loosened check):
    - site/index.html, all of it (the meta description and other attributes included), whitespace
      collapsed, any case, no longer says "co-pilot" (nor "copilot" or "co pilot"), and says
      "residual" (the lede's and meta description's residual RL agent wording). #60's lede and
      meta description say "the AI co-pilot";
    - its static text (inside <body>, outside <script>, <style> and <template>) still says
      "simulated" (AGENTS.md);
    - neither site/app.js nor site/results.js contains the literal "Co-pilot": the residual arms'
      names, "Residual agent, fixed penalty" and "Residual agent, constrained", come from
      index.json (site_build's, pinned in test_site_build.py), never typed. #60's scripts type no
      name; this part guards the rename."""
    index = SITE / "index.html"
    assert index.is_file(), f"{index} is missing"
    html = index.read_text(encoding="utf-8")
    page = _Text()
    page.feed(html)
    page.close()

    whole = " ".join(html.split()).lower()
    stale = sorted({m.group(0) for m in CO_PILOT.finditer(whole)})
    assert not stale, f"site/index.html still says {stale}: the page's word is residual RL agent"
    assert "residual" in whole, "site/index.html never says residual: the lede lost the agent"
    text = " ".join(page.text.split()).lower()
    assert "simulated" in text, "index.html's static text doesn't say simulated"
    scripts = [SITE / name for name in SCRIPTS]
    missing = [p for p in scripts if not p.is_file()]
    assert not missing, f"{missing} missing"
    typed = [p.name for p in scripts if "Co-pilot" in p.read_text(encoding="utf-8")]
    assert not typed, f"{typed} type the old name Co-pilot: names come from index.json"


# --- site/: the whole page (Done when) ----------------------------------------------------------


def test_site_totals_under_1_mb():
    """Done when: every file under site/, recursively (the page and its data), totals under
    1,000,000 bytes (1 MB, decimal). The page's files (four since #60: results.js joins
    index.html, style.css and app.js) and the six data files must be among them, so a missing or
    empty site/ doesn't pass."""
    assert SITE.is_dir(), f"{SITE} is missing"
    files = [p for p in SITE.rglob("*") if p.is_file()]
    present = {p.relative_to(SITE).as_posix() for p in files}
    required = {*PAGE_FILES, *(f"data/{name}" for name in FILE_NAMES)}
    assert required <= present, f"site/ lacks {sorted(required - present)}"
    total = sum(p.stat().st_size for p in files)
    assert total < MAX_SITE_BYTES, f"site/ is {total:,} bytes"

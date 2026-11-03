"""Tests for greenleaf_control.site_build (issue #58, S1): `python -m greenleaf_control.site_build
--out DIR [--force]` assembles the deployable site in DIR from recorded files only: the static page
in site/, its validation replays, the final's recorded 2012 replays and two small JSON indexes the
page reads. It never plays a season, never reads weather, never writes the ledger, results or runs.

The conventions pinned here (the issue's steps, with the orchestrator's defaults on its open points).
- build(out, *, root=manifest.REPO_ROOT, ledger_path=ledger.LEDGER, force=False) -> dict with
  exactly {"final_result", "replays", "files", "bytes"}: the final-result row id, the number of
  replays copied into out/data/final (an int), the number of files the build wrote (an int: every
  file under a fresh out) and their total size in bytes (an int).
- main(argv=None, *, ledger_path=ledger.LEDGER) -> int: --out (required) and --force; it prints a
  line per file group (one naming "168" and "replays", one naming index.json, one results.json) and
  a line with the total bytes (the integer, with or without thousands commas, and the word
  "bytes"), and returns 0.
- Refusals are SystemExit with a message (a non-empty str), raised before anything is made or
  written: (1) the ledger has no final-result row (the message names "final-result"); (2) out,
  resolved and casefolded (macOS's /tmp is /private/tmp and APFS ignores case), is inside or equal
  to root/site, root/control/results, root/control/runs or root/control/ledger; (3) out is root or
  an ancestor of it; (4) out exists and is non-empty without force. --force overwrites same-named
  files and never deletes.
- The final is found from the ledger, never typed: its 2012 final-result row (#45's
  final-result-aee23915) names its final-registered row (final-registered-4d0a4ba9), and the
  replays are root/control/results/final/<that row id>/replays. #73: the 2012 final-result is the
  one whose final-registered row has ledger.seal_of(...) == 2012, so a 2013 final-result in the
  ledger (#71's second seal), before or after it, changes nothing, and a ledger whose only
  final-result is 2013's is refused like one with none.
- out/ holds site/'s plain files (index.html, style.css, app.js) and site/data/*.json at the same
  relative paths, byte for byte; the final's replays of exactly 7 controllers (rb_tuned,
  rb_default, residual_w05, lag_b100, ppo, ppo_paper, zero_controls; not random) x 3 cities x 8
  start days = 168 files in out/data/final/ under their own names, byte for byte; and
  out/data/index.json and out/data/results.json. Nothing else: 179 files with today's site/.
- index.json: {"final_result": <row id>, "splits": [...]}, splits amsterdam_2012, london_2012,
  abbotsford_2012, validation in that order, each exactly {"id", "location", "year", "label",
  "dir", "start_days", "controllers"}. A replay is
  f"{dir}/{file_prefix}_{location}_{year}_{day}.json" under out: dir "data/final" for the 2012
  splits, "data" for validation. A controller is exactly {"id", "name", "file_prefix", "seed"}, in
  the issue's order; seed is its replays' meta.run.seed (null for a baseline): residual_w05 100,
  lag_b100 200, ppo 0 and ppo_paper 0 (0, not null: 0 is falsy). Validation lists rb_tuned and
  residual_w05 (file prefix residual_w05_seed100, seed 100) on start days 45, 59, 75.
- results.json: {split id: {"final_result": <row id> (2012 splits only), "note": str, "rows":
  {label: row}}}, splits in index.json's order; rows are summary.csv's labels in its order without
  random. A row is exactly "controller" plus n_runs and invalid_episodes (ints), decision (a blank
  is null), the _iqm/_lo/_hi triples of profit_eur_m2, profit_gain, viol_hours_total,
  viol_hours_co2, viol_hours_temp, viol_hours_rh, yield_kg_m2, and viol_hours_sum_iqm,
  rb_tuned_profit_iqm, rb_tuned_viol_sum_iqm, each equal to float(summary.csv's text) exactly.
  controller is the label on a 2012 split; on validation it is the arm id for the 3 baselines and
  the 4 registrations the final scored (VALIDATION_ARMS below), null for every other label.
- Neither JSON holds NaN or Infinity.

Every check imports greenleaf_control.site_build inside its test body, so while it is missing each
test FAILS on an assertion rather than ERRORs. The real build reads the committed ledger,
control/results/analysis and control/results/final (the seal is spent: #45) and writes only under
tmp_path; refusals are tried against a fake checkout under tmp_path, so a broken refusal can't
write into this one. A subprocess checks that a build loads no env, simulator or policy code,
opens no weather file and writes nothing under control/ledger, control/results, control/runs or
site/. Nothing here plays a season.
"""

import csv
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from greenleaf_control import ledger, manifest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
CONTROL = ROOT / "control"
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
ANALYSIS = CONTROL / "results" / "analysis"
FINAL_RESULT = "final-result-aee23915"  # #45's final-result row
FINAL_REGISTERED = "final-registered-4d0a4ba9"  # the final-registered row it names
REPLAYS = CONTROL / "results" / "final" / FINAL_REGISTERED / "replays"
# A fake checkout's final, under other ids, so a builder that typed the real ones is caught.
FAKE_RESULT = "final-result-1e1e1e1e"
FAKE_REGISTERED = "final-registered-0f0f0f0f"
# #73: a synthetic 2013 final (#71's second seal), under ids no real row has. Its files are nowhere,
# so a build that took it has no replays to copy.
FINAL_REGISTERED_2013 = "final-registered-13131313"
FINAL_RESULT_2013 = "final-result-13131313"

# (id, display name, meta.run.seed of its final replays: the lowest seed the final scored; None
# for a baseline), in the issue's order. #63 (a requirement change, the issue's scope comment,
# 2026-11-02): the two residual arms' names are "Residual agent, ..." (#58's were "Co-pilot, ...");
# their ids, and every other name, are unchanged.
CONTROLLERS = [
    ("rb_tuned", "Tuned rules", None),
    ("rb_default", "Default rules", None),
    ("residual_w05", "Residual agent, fixed penalty", 100),
    ("lag_b100", "Residual agent, constrained", 200),
    ("ppo", "PPO, no rules", 0),
    ("ppo_paper", "PPO, paper setup", 0),
    ("zero_controls", "Do nothing", None),
]
TEST_CONTROLLERS = [{"id": c, "name": n, "file_prefix": c, "seed": s} for c, n, s in CONTROLLERS]
VALIDATION_CONTROLLERS = [
    {"id": "rb_tuned", "name": "Tuned rules", "file_prefix": "rb_tuned", "seed": None},
    {
        "id": "residual_w05",
        "name": "Residual agent, fixed penalty",
        "file_prefix": "residual_w05_seed100",
        "seed": 100,
    },
]
TEST_DAYS = [45, 49, 53, 57, 61, 65, 69, 73]  # scenarios.TEST's start days
VALIDATION_DAYS = [45, 59, 75]  # scenarios.VALIDATION's start days
# (id, location, year, label, dir, start_days, controllers), in index.json's order.
SPLITS = [
    ("amsterdam_2012", "Amsterdam", 2012, "Amsterdam 2012", "data/final", TEST_DAYS),
    ("london_2012", "London", 2012, "London 2012", "data/final", TEST_DAYS),
    ("abbotsford_2012", "Abbotsford", 2012, "Abbotsford 2012", "data/final", TEST_DAYS),
    ("validation", "Amsterdam", 2010, "Amsterdam 2010 (validation)", "data", VALIDATION_DAYS),
]
SPLIT_IDS = [s[0] for s in SPLITS]
TEST_SPLIT_IDS = SPLIT_IDS[:3]
SPLIT_CASES = [pytest.param(*s, id=s[0]) for s in SPLITS]
# 7 controllers x 3 cities x 8 start days = 168 replays, under the final's own file names.
REPLAYS_OF = {
    c: [f"{c}_{location}_2012_{day}.json" for _, location, *_ in SPLITS[:3] for day in TEST_DAYS]
    for c, _, _ in CONTROLLERS
}
REPLAY_NAMES = sorted(name for names in REPLAYS_OF.values() for name in names)
N_REPLAYS = 168

# Validation's summary.csv labels that name a controller on the page: the 3 baselines and the
# registrations whose runs the final scored (ppo_paper #31, ppo #33, residual_w05 #36, lag_b100
# #39). Every other validation label's controller is null.
VALIDATION_ARMS = {
    "zero_controls": "zero_controls",
    "rb_default": "rb_default",
    "rb_tuned": "rb_tuned",
    "ppo_paper #31/2d02": "ppo_paper",
    "ppo #33/0221": "ppo",
    "residual_w05 #36/f2fd": "residual_w05",
    "lag_b100 #39/6d92": "lag_b100",
}
METRICS = (
    "profit_eur_m2",
    "profit_gain",
    "viol_hours_total",
    "viol_hours_co2",
    "viol_hours_temp",
    "viol_hours_rh",
    "yield_kg_m2",
)
INT_COLUMNS = ("n_runs", "invalid_episodes")
FLOAT_COLUMNS = (
    *(f"{m}_{s}" for m in METRICS for s in ("iqm", "lo", "hi")),
    "viol_hours_sum_iqm",
    "rb_tuned_profit_iqm",
    "rb_tuned_viol_sum_iqm",
)
ROW_KEYS = {"controller", "decision", *INT_COLUMNS, *FLOAT_COLUMNS}
TEST_NOTE = "2012 test weather, scored once in the final ({}). Simulated, not a real greenhouse."
VALIDATION_NOTE = "Validation seasons: checkpoints were selected on these. Simulated."
SUMMARY_KEYS = {"final_result", "replays", "files", "bytes"}

# Loaded by a build in a fresh interpreter, they would mean it built an env, played a season or
# loaded a policy.
FORBIDDEN = (
    "gl_gym",
    "greenleaf_control.envs",
    "greenleaf_control.evaluate",
    "greenleaf_control.report",
    "greenleaf_control.export_replay",
    "greenleaf_control.rewards",
    "stable_baselines3",
    "torch",
)


# --- Helpers ---------------------------------------------------------------------------------------


def _module():
    """greenleaf_control.site_build; an assertion, not an ImportError, while it is missing."""
    found = importlib.util.find_spec("greenleaf_control.site_build")
    assert found is not None, "greenleaf_control.site_build doesn't exist yet (#58)"
    return importlib.import_module("greenleaf_control.site_build")


def _reject_constant(name: str):
    raise AssertionError(f"{name} in the JSON: the payload must be finite numbers only")


def _load(path: Path) -> dict:
    assert path.is_file(), f"{path} is missing"
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _summary_csv(split: str, root: Path = ROOT) -> list[dict]:
    """summary.csv's rows as text, read with the stdlib, so float(text) is the number it holds."""
    path = root / "control" / "results" / "analysis" / split / "summary.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _files(top: Path) -> dict[str, int]:
    """Every file under top, as its path relative to top -> its size in bytes."""
    return {p.relative_to(top).as_posix(): p.stat().st_size for p in top.rglob("*") if p.is_file()}


def _tree(top: Path) -> dict[str, tuple[int, int, int]]:
    """Every file, dir and link under top (links not followed) -> (type, size, mtime_ns), so a
    refusal that made, wrote or touched anything changes it."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            st = path.lstat()
            found[str(path.relative_to(top))] = (
                stat.S_IFMT(st.st_mode),
                st.st_size,
                st.st_mtime_ns,
            )
    return found


def _site_files() -> list[str]:
    """What the build copies from site/: its plain files, then data/*.json, relative paths."""
    plain = sorted(p.name for p in SITE.iterdir() if p.is_file())
    data = sorted(f"data/{p.name}" for p in (SITE / "data").glob("*.json"))
    return plain + data


def _replay_path(split: dict, controller: dict, day: int) -> str:
    """index.json's rule for a replay's path under out."""
    return (
        f"{split['dir']}/{controller['file_prefix']}_{split['location']}_{split['year']}_{day}.json"
    )


def _fake_checkout(base: Path, *, replays: str) -> tuple[Path, Path]:
    """(root, ledger path) of a fake checkout at base/ancestor/repo: a copy of site/, of the four
    summary.csv files, an empty control/runs/, and the real ledger with the final's two row ids
    renamed to FAKE_RESULT and FAKE_REGISTERED, at control/ledger/ledger.jsonl. Its final's
    replays, at control/results/final/FAKE_REGISTERED/replays, are a link to the real replays dir
    (replays="link") or copies of all 189, each one newline longer (replays="copy"), so a build
    that read the real dir instead is caught."""
    root = base / "ancestor" / "repo"
    shutil.copytree(SITE, root / "site")
    for split in SPLIT_IDS:
        dest = root / "control" / "results" / "analysis" / split
        dest.mkdir(parents=True)
        shutil.copyfile(ANALYSIS / split / "summary.csv", dest / "summary.csv")
    (root / "control" / "runs").mkdir()
    final = root / "control" / "results" / "final" / FAKE_REGISTERED
    final.mkdir(parents=True)
    if replays == "link":
        (final / "replays").symlink_to(REPLAYS, target_is_directory=True)
    else:
        (final / "replays").mkdir()
        for src in REPLAYS.iterdir():
            (final / "replays" / src.name).write_bytes(src.read_bytes() + b"\n")
    text = REAL_LEDGER.read_text(encoding="utf-8")
    renamed = text.replace(FINAL_REGISTERED, FAKE_REGISTERED).replace(FINAL_RESULT, FAKE_RESULT)
    ledger_path = root / "control" / "ledger" / "ledger.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(renamed, encoding="utf-8")
    return root, ledger_path


def _with_a_2013_final(text: str, *, where: str) -> str:
    """The ledger text with a synthetic 2013 final's two rows added (#73): a final-registered row of
    seal 2013 at the 2012 final's checkpoints, with #72's rb_tuned_agc2020 among its baselines, and
    its final-result over the 2013 splits (per_split nested by price regime, #72's shape; empty
    placeholders, which no reader of 2012 opens). where="after" appends them, the ledger's real
    order (2013 is spent after 2012); where="before" puts them just before the 2012 final-result
    row, so the first final-result in file order is 2013's."""
    lines = text.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    [registered_2012] = [r for r in rows if r["row_id"] == FINAL_REGISTERED]
    stamp = {k: registered_2012[k] for k in ("schema_version", "at", "fingerprints")}
    registered = {
        **registered_2012,
        "row_id": FINAL_REGISTERED_2013,
        "issue": 72,
        "baselines": [*registered_2012["baselines"], "rb_tuned_agc2020"],
        "seal": 2013,
    }
    result = {
        **stamp,
        "row_id": FINAL_RESULT_2013,
        "kind": "final-result",
        "issue": 72,
        "final-registered": FINAL_REGISTERED_2013,
        "per_split": {split: {"sim": {}, "agc2020": {}} for split in ledger.SEALS[2013]},
    }
    new = [json.dumps(r, sort_keys=True) + "\n" for r in (registered, result)]
    if where == "after":
        return "".join(lines + new)
    at = next(i for i, r in enumerate(rows) if r["row_id"] == FINAL_RESULT)
    return "".join(lines[:at] + new + lines[at:])


def _assert_refused_with_a_message(refused: pytest.ExceptionInfo) -> str:
    message = refused.value.code
    assert isinstance(message, str) and message.strip(), (
        f"a refusal is SystemExit with a message, got SystemExit({message!r})"
    )
    return message


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A function returning (out, summary): its first call builds the site from this checkout's
    root and ledger (build's defaults) into a fresh, missing dir under a module tmp dir; later
    calls return the same. Called inside each test body, so a missing module FAILS the test."""
    cache = []

    def get() -> tuple[Path, dict]:
        if not cache:
            out = tmp_path_factory.mktemp("built") / "out"
            cache.append((out, _module().build(out)))
        return cache[0]

    return get


# --- The contract surface ----------------------------------------------------------------------


def test_signatures_match_the_contract():
    """build(out, *, root=manifest.REPO_ROOT, ledger_path=ledger.LEDGER, force=False) and
    main(argv=None, *, ledger_path=ledger.LEDGER)."""
    site_build = _module()
    build = inspect.signature(site_build.build).parameters
    assert list(build) == ["out", "root", "ledger_path", "force"]
    assert build["out"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(build[n].kind is inspect.Parameter.KEYWORD_ONLY for n in list(build)[1:])
    assert build["root"].default == manifest.REPO_ROOT
    assert build["ledger_path"].default == ledger.LEDGER
    assert build["force"].default is False
    main = inspect.signature(site_build.main).parameters
    assert list(main) == ["argv", "ledger_path"]
    assert main["argv"].default is None
    assert main["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert main["ledger_path"].default == ledger.LEDGER


# --- What the build writes ---------------------------------------------------------------------


def test_out_holds_exactly_the_site_the_replays_and_two_indexes(built):
    """site/'s 3 page files and 6 validation replays, the 168 final replays and data/index.json and
    data/results.json, nothing else (179 files with today's site/)."""
    out, _ = built()
    expected = {
        *_site_files(),
        *(f"data/final/{name}" for name in REPLAY_NAMES),
        "data/index.json",
        "data/results.json",
    }
    assert {"index.html", "style.css", "app.js"} <= set(_site_files())
    assert len(expected) == len(_site_files()) + N_REPLAYS + 2
    assert sorted(_files(out)) == sorted(expected)


def test_build_returns_what_it_wrote(built):
    """{"final_result": "final-result-aee23915", "replays": 168, "files": every file under
    the fresh out (179 today), "bytes": their total size}."""
    out, summary = built()
    on_disk = _files(out)
    assert set(summary) == SUMMARY_KEYS
    assert summary["final_result"] == FINAL_RESULT
    assert summary["replays"] == N_REPLAYS and type(summary["replays"]) is int
    assert summary["files"] == len(on_disk) and type(summary["files"]) is int
    assert summary["bytes"] == sum(on_disk.values()) and type(summary["bytes"]) is int


def test_site_files_are_copied_byte_for_byte(built):
    """index.html, style.css, app.js (and any other plain file directly in site/) and every
    site/data/*.json, at the same relative path under out, with identical bytes."""
    out, _ = built()
    differ = [
        rel
        for rel in _site_files()
        if not (out / rel).is_file() or (out / rel).read_bytes() != (SITE / rel).read_bytes()
    ]
    assert differ == []


def test_data_final_holds_exactly_the_seven_controllers_replays(built):
    """out/data/final/ holds the 168 files <controller>_<City>_2012_<day>.json of the 7 page
    controllers, under the final's own names, and nothing else (no subdirectory)."""
    out, _ = built()
    final = out / "data" / "final"
    assert final.is_dir(), f"{final} is missing"
    assert sorted(p.name for p in final.iterdir()) == REPLAY_NAMES
    assert all((final / name).is_file() for name in REPLAY_NAMES)


@pytest.mark.parametrize("controller", [c for c, _, _ in CONTROLLERS])
def test_final_replays_are_byte_identical_to_the_finals(built, controller):
    """Each of the controller's 24 replays (3 cities x 8 start days) is a byte-for-byte copy of
    control/results/final/final-registered-4d0a4ba9/replays/<same name>."""
    out, _ = built()
    assert len(REPLAYS_OF[controller]) == 3 * len(TEST_DAYS)
    differ = [
        name
        for name in REPLAYS_OF[controller]
        if not (out / "data" / "final" / name).is_file()
        or (out / "data" / "final" / name).read_bytes() != (REPLAYS / name).read_bytes()
    ]
    assert differ == []


def test_random_is_left_out(built):
    """random (3 invalid Abbotsford seasons; noise is not a controller) has no replay under out, no
    index.json entry and no results.json row, on any split."""
    out, _ = built()
    assert [p.name for p in out.rglob("random_*")] == []
    index = _load(out / "data" / "index.json")
    results = _load(out / "data" / "results.json")
    assert [
        (s.get("id"), c.get("id"))
        for s in index["splits"]
        for c in s.get("controllers", [])
        if c.get("id") == "random"
    ] == []
    assert [
        (split, label)
        for split, entry in results.items()
        for label, row in entry.get("rows", {}).items()
        if label == "random" or row.get("controller") == "random"
    ] == []


# --- index.json ------------------------------------------------------------------------------


def test_index_names_the_final_and_the_four_splits_in_order(built):
    out, _ = built()
    index = _load(out / "data" / "index.json")
    assert set(index) == {"final_result", "splits"}
    assert index["final_result"] == FINAL_RESULT
    assert [s.get("id") for s in index["splits"]] == SPLIT_IDS


@pytest.mark.parametrize(("split_id", "location", "year", "label", "folder", "days"), SPLIT_CASES)
def test_index_split_fields(built, split_id, location, year, label, folder, days):
    """Each split is exactly {"id", "location", "year", "label", "dir", "start_days",
    "controllers"} with the issue's values; start days are the days whose files are present."""
    out, _ = built()
    split = {s.get("id"): s for s in _load(out / "data" / "index.json")["splits"]}[split_id]
    assert set(split) == {"id", "location", "year", "label", "dir", "start_days", "controllers"}
    assert (split["location"], split["year"], split["label"]) == (location, year, label)
    assert type(split["year"]) is int
    assert split["dir"] == folder
    assert split["start_days"] == days


@pytest.mark.parametrize("split_id", SPLIT_IDS)
def test_index_controllers_names_prefixes_and_seeds(built, split_id):
    """The 2012 splits list the 7 controllers in the issue's order, file_prefix = id, seeds 100,
    200, 0, 0 for residual_w05, lag_b100, ppo, ppo_paper and null for the baselines; validation
    lists rb_tuned and residual_w05 (file prefix residual_w05_seed100, seed 100). ppo's and
    ppo_paper's seed is the int 0, not null or false. Names are CONTROLLERS': since #63 (a
    requirement change, not a loosened check) residual_w05 is "Residual agent, fixed penalty" and
    lag_b100 "Residual agent, constrained" on every split; #58's names were "Co-pilot, ..."."""
    out, _ = built()
    split = {s.get("id"): s for s in _load(out / "data" / "index.json")["splits"]}[split_id]
    expected = VALIDATION_CONTROLLERS if split_id == "validation" else TEST_CONTROLLERS
    assert split["controllers"] == expected
    for got, want in zip(split["controllers"], expected, strict=True):
        assert set(got) == {"id", "name", "file_prefix", "seed"}
        assert type(got["seed"]) is type(want["seed"]), (got["id"], got["seed"])


@pytest.mark.parametrize("split_id", SPLIT_IDS)
def test_index_seeds_are_the_replays_meta_run_seed(built, split_id):
    """Every listed file's meta.run.seed (null when meta.run is null) equals its controller's seed
    in index.json."""
    out, _ = built()
    split = {s.get("id"): s for s in _load(out / "data" / "index.json")["splits"]}[split_id]
    differ = []
    for controller in split["controllers"]:
        for day in split["start_days"]:
            run = _load(out / _replay_path(split, controller, day))["meta"]["run"]
            seed = None if run is None else run["seed"]
            if seed != controller["seed"] or type(seed) is not type(controller["seed"]):
                differ.append((controller["id"], day, seed, controller["seed"]))
    assert differ == []


def test_every_listed_file_exists_and_nothing_unlisted_is_under_data_final(built):
    """Each (split, controller, start day) index.json lists is a file under out; the 2012 splits'
    168 listed files are exactly what out/data/final/ holds; validation's 6 are site/data's."""
    out, _ = built()
    index = _load(out / "data" / "index.json")
    listed = [
        _replay_path(s, c, d)
        for s in index["splits"]
        for c in s["controllers"]
        for d in s["start_days"]
    ]
    assert [p for p in listed if not (out / p).is_file()] == []
    assert len(listed) == len(set(listed)) == N_REPLAYS + 6
    in_final = sorted(f"data/final/{rel}" for rel in _files(out / "data" / "final"))
    assert sorted(p for p in listed if p.startswith("data/final/")) == in_final
    assert sorted(p for p in listed if not p.startswith("data/final/")) == sorted(
        rel for rel in _site_files() if rel.startswith("data/")
    )


# --- results.json ----------------------------------------------------------------------------


def test_results_splits_notes_and_final_result(built):
    """Splits in index.json's order; a 2012 split is {"final_result", "note", "rows"} naming
    final-result-aee23915 in both, validation {"note", "rows"}."""
    out, _ = built()
    results = _load(out / "data" / "results.json")
    assert list(results) == SPLIT_IDS
    for split in TEST_SPLIT_IDS:
        assert set(results[split]) == {"final_result", "note", "rows"}
        assert results[split]["final_result"] == FINAL_RESULT
        assert results[split]["note"] == TEST_NOTE.format(FINAL_RESULT)
    assert set(results["validation"]) == {"note", "rows"}
    assert results["validation"]["note"] == VALIDATION_NOTE


@pytest.mark.parametrize("split_id", SPLIT_IDS)
def test_results_rows_are_the_summary_labels_and_their_controllers(built, split_id):
    """rows: summary.csv's labels in its order, random dropped. controller: the label on a 2012
    split; on validation VALIDATION_ARMS' arm id, null for every other label."""
    out, _ = built()
    rows = _load(out / "data" / "results.json")[split_id]["rows"]
    labels = [r["label"] for r in _summary_csv(split_id) if r["label"] != "random"]
    assert list(rows) == labels
    if split_id == "validation":
        assert set(VALIDATION_ARMS) <= set(labels)
        want = {label: VALIDATION_ARMS.get(label) for label in labels}
    else:
        assert sorted(labels) == sorted(c for c, _, _ in CONTROLLERS)
        want = {label: label for label in labels}
    assert {label: row.get("controller") for label, row in rows.items()} == want


@pytest.mark.parametrize("split_id", SPLIT_IDS)
def test_results_numbers_equal_summary_csv_exactly(built, split_id):
    """Each row is exactly ROW_KEYS; every float column equals float(summary.csv's text) with ==
    (copied, never recomputed or rounded), n_runs and invalid_episodes are its ints, and decision
    is its text, a blank one null."""
    out, _ = built()
    rows = _load(out / "data" / "results.json")[split_id]["rows"]
    differ = []
    for csv_row in _summary_csv(split_id):
        label = csv_row["label"]
        if label == "random":
            continue
        got = rows[label]
        assert set(got) == ROW_KEYS, (label, sorted(set(got) ^ ROW_KEYS))
        for column in FLOAT_COLUMNS:
            value, want = got[column], float(csv_row[column])
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != want:
                differ.append((label, column, value, want))
        for column in INT_COLUMNS:
            value, want = got[column], int(csv_row[column])
            if type(value) is not int or value != want:
                differ.append((label, column, value, want))
        if got["decision"] != (csv_row["decision"] or None):
            differ.append((label, "decision", got["decision"], csv_row["decision"] or None))
    assert differ == []


# --- The final is read from the ledger -------------------------------------------------------


def test_the_finals_dir_comes_from_its_final_registered_row(tmp_path):
    """In a fake checkout whose ledger's final-result row is final-result-1e1e1e1e naming
    final-registered-0f0f0f0f, the replays copied are control/results/final/
    final-registered-0f0f0f0f/replays' own bytes (each one newline longer than the real ones), and
    index.json, results.json and the summary name final-result-1e1e1e1e."""
    site_build = _module()
    root, ledger_path = _fake_checkout(tmp_path, replays="copy")
    fake = root / "control" / "results" / "final" / FAKE_REGISTERED / "replays"
    assert not (root / "control" / "results" / "final" / FINAL_REGISTERED).exists()
    out = tmp_path / "out"
    summary = site_build.build(out, root=root, ledger_path=ledger_path)
    final = out / "data" / "final"
    assert sorted(p.name for p in final.iterdir()) == REPLAY_NAMES
    assert [n for n in REPLAY_NAMES if (final / n).read_bytes() != (fake / n).read_bytes()] == []
    assert summary["final_result"] == FAKE_RESULT
    assert _load(out / "data" / "index.json")["final_result"] == FAKE_RESULT
    results = _load(out / "data" / "results.json")
    for split in TEST_SPLIT_IDS:
        assert results[split]["final_result"] == FAKE_RESULT
        assert results[split]["note"] == TEST_NOTE.format(FAKE_RESULT)


@pytest.mark.parametrize("where", ["after", "before"])
def test_a_2013_final_result_in_the_ledger_leaves_the_build_on_2012s(built, tmp_path, where):
    """Done when (#73 c): the committed ledger with a 2013 final added (_with_a_2013_final, after
    the 2012 final-result or before it) builds exactly the site the committed ledger builds: the
    same summary (final_result-aee23915, 168 replays) and the same files with the same bytes
    (index.json and results.json naming the 2012 final-result, its replays). The 2013 final's files
    are nowhere, so a build that took it can't finish."""
    site_build = _module()
    real_out, real_summary = built()
    path = tmp_path / "ledger.jsonl"
    text = REAL_LEDGER.read_text(encoding="utf-8")
    path.write_text(_with_a_2013_final(text, where=where), encoding="utf-8")
    out = tmp_path / "out"
    try:
        summary = site_build.build(out, ledger_path=path)
    except (ValueError, SystemExit) as e:
        pytest.fail(f"a 2013 final-result in the ledger stopped the 2012 site's build: {e!r}")
    assert summary == real_summary
    assert summary["final_result"] == FINAL_RESULT
    on_disk = _files(out)
    assert on_disk == _files(real_out)
    assert [r for r in on_disk if (out / r).read_bytes() != (real_out / r).read_bytes()] == []


# --- Refusals --------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["no-final-result", "only-a-2013-final-result"])
def test_a_ledger_without_a_final_result_row_is_refused(tmp_path, case):
    """The real ledger minus its 2012 final-result line (by row id: #73, a 2013 one may follow it),
    alone or with a 2013 final added (#73: a final-result of another seal has no 2012 seasons):
    SystemExit naming "final-result", and out is not made."""
    site_build = _module()
    lines = REAL_LEDGER.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if json.loads(line)["row_id"] != FINAL_RESULT]
    assert len(kept) == len(lines) - 1
    text = "".join(kept)
    if case == "only-a-2013-final-result":
        text = _with_a_2013_final(text, where="after")
    no_final = tmp_path / "ledger.jsonl"
    no_final.write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as refused:
        site_build.build(out, root=ROOT, ledger_path=no_final)
    assert "final-result" in _assert_refused_with_a_message(refused)
    assert not out.exists()


# (case id, out relative to the fake checkout's root, force). Cases where out already exists are
# tried with force, so only the place can refuse them, not their being non-empty.
PROTECTED = [
    ("site", "site/build", False),
    ("site-itself", "site", True),
    ("site-in-other-case", "SITE/build", False),
    ("site-through-dotdot", "control/../site/build", False),
    ("results", "control/results/site", False),
    ("results-itself", "control/results", True),
    ("results-in-other-case", "Control/Results/site", False),
    ("results-final", "control/results/final/site", False),
    ("runs", "control/runs/site", False),
    ("ledger", "control/ledger/site", False),
    ("repo-root", ".", True),
    ("an-ancestor-of-the-root", "..", True),
]


@pytest.mark.parametrize(
    ("where", "force"), [pytest.param(w, f, id=case) for case, w, f in PROTECTED]
)
def test_out_in_a_protected_place_is_refused(tmp_path, where, force):
    """out inside or equal to site/, control/results/, control/runs/ or control/ledger/ (resolved,
    casefolded), or the repo root or an ancestor of it: SystemExit with a message, and nothing
    under tmp_path is made, written or touched."""
    site_build = _module()
    root, ledger_path = _fake_checkout(tmp_path, replays="link")
    before = _tree(tmp_path)
    with pytest.raises(SystemExit) as refused:
        site_build.build(root / where, root=root, ledger_path=ledger_path, force=force)
    _assert_refused_with_a_message(refused)
    assert _tree(tmp_path) == before


def test_out_through_a_link_into_site_is_refused(tmp_path):
    """out = <a link to the checkout's site/>/build resolves inside site/: refused, nothing made."""
    site_build = _module()
    root, ledger_path = _fake_checkout(tmp_path, replays="link")
    link = tmp_path / "link"
    link.symlink_to(root / "site", target_is_directory=True)
    before = _tree(tmp_path)
    with pytest.raises(SystemExit) as refused:
        site_build.build(link / "build", root=root, ledger_path=ledger_path)
    _assert_refused_with_a_message(refused)
    assert _tree(tmp_path) == before


def test_a_non_empty_out_is_refused_without_force(tmp_path):
    """out holding one file: SystemExit with a message, and out still holds just that file."""
    site_build = _module()
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    before = _tree(tmp_path)
    with pytest.raises(SystemExit) as refused:
        site_build.build(out, root=ROOT, ledger_path=REAL_LEDGER)
    _assert_refused_with_a_message(refused)
    assert _tree(tmp_path) == before


# --- The CLI ---------------------------------------------------------------------------------


def test_force_overwrites_same_named_files_and_deletes_nothing(tmp_path):
    """main(["--out", out, "--force"]) into an out holding keep.txt, an old index.html and an old
    data/index.json: returns 0, keep.txt is untouched, the other two are replaced by the build's,
    and the 168 replays are there."""
    site_build = _module()
    out = tmp_path / "out"
    (out / "data").mkdir(parents=True)
    (out / "keep.txt").write_text("mine", encoding="utf-8")
    (out / "index.html").write_text("old", encoding="utf-8")
    (out / "data" / "index.json").write_text('{"old": true}', encoding="utf-8")
    assert site_build.main(["--out", str(out), "--force"]) == 0
    assert (out / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert (out / "index.html").read_bytes() == (SITE / "index.html").read_bytes()
    assert _load(out / "data" / "index.json").get("final_result") == FINAL_RESULT
    assert sorted(p.name for p in (out / "data" / "final").iterdir()) == REPLAY_NAMES


def test_main_builds_prints_and_returns_0(tmp_path, capsys):
    """main(["--out", out]) returns 0 and prints a line naming 168 replays, one naming index.json,
    one naming results.json, and one with the total bytes written and the word "bytes"."""
    site_build = _module()
    out = tmp_path / "out"
    assert site_build.main(["--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    total = sum(_files(out).values())
    assert any("168" in line and "replays" in line for line in lines), lines
    assert any("index.json" in line for line in lines), lines
    assert any("results.json" in line for line in lines), lines
    assert any(
        "bytes" in line and (str(total) in line or f"{total:,}" in line) for line in lines
    ), (total, lines)


def test_main_requires_out(capsys):
    """No --out is argparse's usage error (exit code 2)."""
    site_build = _module()
    with pytest.raises(SystemExit) as refused:
        site_build.main([])
    assert refused.value.code == 2
    assert "--out" in capsys.readouterr().err


# --- No env, no weather, no writes to the record ---------------------------------------------

# Run in a fresh interpreter: an audit hook records every open of a weather file and every write,
# make, move or delete under control/ledger, control/results, control/runs or site/, then a build
# into argv[1]; it prints {"modules": sorted(sys.modules), "seen": [...]} as its last line.
_NO_ENV = r"""
import json, os, sys
from pathlib import Path

out, root = sys.argv[1], sys.argv[2]
weather = [os.path.join(root, "control", "data", "weather").casefold()]
guarded = [
    os.path.join(root, *parts).casefold()
    for parts in (("control", "ledger"), ("control", "results"), ("control", "runs"), ("site",))
]
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
# Audit event -> the indexes of the args it writes, makes, moves or deletes.
edits = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1),
    "os.truncate": (0,), "os.chmod": (0,), "os.utime": (0,), "os.symlink": (1,), "os.link": (1,),
    "shutil.rmtree": (0,), "shutil.copyfile": (1,), "shutil.copytree": (1,), "shutil.move": (0, 1),
}
seen = []


def under(path, prefixes):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    p = os.path.abspath(os.fsdecode(path)).casefold()
    return any(p == q or p.startswith(q + os.sep) for q in prefixes)


def hook(event, args):
    if event == "open":
        path, mode, flags = args
        if under(path, weather):
            seen.append(["opened weather", os.fsdecode(path)])
        writes = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(flags & write_flags)
        if writes and under(path, guarded):
            seen.append(["opened to write", os.fsdecode(path)])
    elif event in edits:
        for i in edits[event]:
            if i < len(args) and under(args[i], guarded):
                seen.append([event, os.fsdecode(args[i])])


sys.addaudithook(hook)
from greenleaf_control import site_build

site_build.build(Path(out))
print(json.dumps({"modules": sorted(sys.modules), "seen": seen}))
"""


def test_a_build_loads_no_env_reads_no_weather_and_writes_no_record(tmp_path):
    """In a fresh interpreter, importing site_build and building the real site into tmp_path
    leaves gl_gym, envs, evaluate, report, export_replay, rewards, stable_baselines3 and torch out
    of sys.modules, opens nothing under control/data/weather and writes, makes, moves or deletes
    nothing under control/ledger, control/results, control/runs or site/."""
    out = tmp_path / "out"
    done = subprocess.run(
        [sys.executable, "-c", _NO_ENV, str(out), str(ROOT)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    loaded = [m for m in report["modules"] for f in FORBIDDEN if m == f or m.startswith(f + ".")]
    assert loaded == []
    assert report["seen"] == []
    assert len(list((out / "data" / "final").iterdir())) == N_REPLAYS

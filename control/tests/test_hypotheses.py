"""Tests for greenleaf_control.hypotheses (issue #73, P6d): `python -m greenleaf_control.hypotheses
[--out DIR]` grades the registered 2013 hypotheses H1-H4 from the 2013 final's seasons.parquet, per
2013 split, so no hypothesis is graded by hand. Nothing is pooled and nothing is played.

The conventions pinned here (the issue's wording, with the orchestrator's defaults on its open points).
- main(argv=None, *, ledger_path=ledger.LEDGER) -> int; --out defaults to OUT =
  control/results/analysis/final_2013. FINAL_RESULTS (control/results/final) is read at call time,
  so the tests point it at a tmp dir.
- The 2013 final is found from the ledger, never typed: the final-result row whose final-registered
  row has ledger.seal_of(...) == 2013, then FINAL_RESULTS/<that final-registered row id>/
  seasons.parquet (#72's shape: arm, run, seed, checkpoint, split, location, growth_year,
  start_day, season_length, price_regime, then the result keys; one row per season and price
  regime, "sim" and "agc2020"). Without a 2013 final-result row main refuses (ValueError, or
  SystemExit with a message) naming "2013" and "final-result", before any parquet is read or out is
  made. It never reads the 2012 final's files, the 2012 analysis or any weather.
- Grading, on each 2013 split alone (scenarios.SEALS[2013] order), each regime's rows alone:
  - a run x seasons matrix is runs by seed ascending x the split's seasons in
    scenarios.SEALS[2013][split] order (report._matrix's order: the bootstrap depends on it); a
    baseline (rb_default, rb_tuned, rb_tuned_agc2020) is 1 run;
  - a paired gain is stats.iqm_ci(arm matrix - baseline's 1 x 8 row), same seasons;
  - a baseline's like-for-like IQM is stats.iqm_ci(its row tiled to the arm's run count)[0];
  - "CI above X" is ci_low > X and "CI below X" is ci_high < X, strictly;
  - Σ is the sum of the three per-constraint violation-hour IQMs (co2, temp, rh) on the regime's
    rows; rb_tuned's like-for-like Σ tiles its rows the same way.
  - H1 (ppo): its paired profit gain over rb_default: CI below 0 at agc2020 AND above 0 at sim.
  - H2 (residual_w05): its paired profit gain over rb_tuned: CI above 0 at sim AND at agc2020.
  - H3 (ppo_agc, residual_agc; at agc2020): (a) profit ci_low > the max like-for-like profit IQM
    of rb_default, rb_tuned and rb_tuned_agc2020; (b) Σ <= rb_tuned's like-for-like Σ (no slack);
    (c) 0 invalid seasons, 0 solver failures.
  - H4 (residual_w05, lag_b100; at sim): #45's rule: (a) profit ci_low > rb_tuned's like-for-like
    profit IQM; (b) Σ <= rb_tuned's like-for-like Σ; (c) 0 invalid seasons, 0 solver failures.
- out/hypotheses.csv: exactly the columns split, hypothesis, arm, regime, part, value, bar, pass,
  one row per split x hypothesis x arm x part. H1's and H2's parts are named by their regime
  (agc2020, sim); H3's and H4's are a, b, c. regime is the regime the part is graded at. value and
  bar: a gain CI's ci_high (below) or ci_low (above) and 0; H3a/H4a the arm's profit ci_low and the
  bar; b the arm's Σ and rb_tuned's like-for-like Σ; c a count of the arm's invalid seasons and
  solver failures (0 when clean) and 0. pass is a boolean (True/False).
- out/summary.md: one line per split x hypothesis x arm naming all three (e.g. a table row
  `| london_2013 | H3 | ppo_agc | fails (a) |`) with its verdict: "verified" when every part
  passes, else "fails (<part>)". It names the 2013 final-result row it graded, never 2012's.
  - The registration of H1-H4 is issue #73's text plus this module, committed before the 2013
    final-registered row (as #45's rule lived on its issue). So summary.md states the registered
    wording word for word (REGISTERED: the Conventions line as written, and each hypothesis bullet
    with its "- " and its "**" bold markers stripped, e.g. "H1 (the price regime reverses the
    ranking): ppo's paired ..."), each on one line, and a line naming its source: "registered as
    issue #73's text (plan approved 2026-11-02)".
  - A legend line says how H3(a)'s bar is read: the largest of the three baselines' like-for-like
    profit IQMs, each through stats.iqm_ci(tiled)[0] (pinned: "H3", "largest", "like-for-like",
    "profit", "stats.iqm_ci" on one line).
  - No claim on n < 5 (PROTOCOL): an arm with fewer than 5 runs has evaluate.NO_CLAIM's
    "n<5: no claim" after its verdict on its line; an arm with 5 or more has no "n<5" there.
  - Part c shows invalid seasons and solver failures apart, e.g. "c: invalid 1, solver failures 3
    (fail)": "invalid" then the count of invalid seasons, "solver failures" then their sum, each
    once on the arm's line, at the regime graded (hypotheses.csv's c value may stay their sum).
- main writes out/hypotheses.csv and out/summary.md and nothing else, only after everything is
  loaded and graded; a refusal leaves out unmade.

The world (_world): a tmp ledger that is the committed one (its 2012 final included) plus a
synthetic 2013 final (registrations, runs, a seal-2013 final-registered row and its final-result),
and a tmp FINAL_RESULTS holding that final's hand-built seasons.parquet and a decoy at the 2012
final's dir. Each split is built so every hypothesis passes in one case and fails in each part in
another (EXPECTED). Every check imports greenleaf_control.hypotheses inside its test body, so while
it is missing each test FAILS on an assertion rather than ERRORs. The committed ledger is only
read; nothing writes under control/ledger, control/results or control/runs; no weather is read.
"""

import importlib
import importlib.util
import inspect
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from greenleaf_control import ledger, scenarios, stats

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "control"
REAL_LEDGER = CONTROL / "ledger" / "ledger.jsonl"
FINAL_RESULT_2012 = "final-result-aee23915"  # #45's final-result row, the 2012 seal's
FINAL_REGISTERED_2012 = "final-registered-4d0a4ba9"  # the final-registered row it names
FINAL_REGISTERED_2013 = "final-registered-73730013"  # the synthetic 2013 final's rows
FINAL_RESULT_2013 = "final-result-73730013"

COLUMNS = "split,hypothesis,arm,regime,part,value,bar,pass"
OUTPUTS = ["hypotheses.csv", "summary.md"]
NO_CLAIM = "n<5: no claim"  # evaluate.NO_CLAIM: no claim on fewer than 5 runs
# The registered wording, word for word from issue #73's section (d): its Conventions line as
# written, then each hypothesis bullet without its leading "- " and its "**" bold markers.
REGISTERED = {
    "Conventions": "Conventions: like-for-like baselines = the baseline's 8 seasons tiled to the"
    " arm's seed count through stats.iqm_ci; a paired gain = stats.iqm_ci(arm n×8 − baseline 1×8),"
    ' same seasons; "CI above/below X" = the 95% CI\'s low end > X / high end < X; Σ = the sum of'
    " the three per-constraint hour IQMs (hours don't depend on prices).",
    "H1": "H1 (the price regime reverses the ranking): ppo's paired profit gain over rb_default has"
    " its CI entirely below 0 at agc2020 AND entirely above 0 at sim.",
    "H2": "H2 (the residual agent's edge is robust to prices): residual_w05's paired profit gain over"
    " rb_tuned has its CI entirely above 0 at sim AND at agc2020.",
    "H3": "H3 (price-matched RL beats the price-matched rules): for each of ppo_agc and residual_agc,"
    " at agc2020: (a) its profit IQM CI lies entirely above max(rb_default, rb_tuned,"
    " rb_tuned_agc2020)'s like-for-like profit IQM; (b) its Σ ≤ rb_tuned's like-for-like Σ (no"
    " slack); (c) 0 invalid, 0 solver failures.",
    "H4": "H4 (replication of #45 on fresh weather): #45's rule at sim for residual_w05 and lag_b100:"
    " (a) profit CI above rb_tuned's like-for-like profit IQM; (b) Σ ≤ rb_tuned's like-for-like Σ;"
    " (c) 0 invalid, 0 solver failures.",
}
SOURCE = "registered as issue #73's text (plan approved 2026-11-02)"
SPLITS = tuple(scenarios.SEALS[2013])  # amsterdam_2013, london_2013, abbotsford_2013
REGIMES = ("sim", "agc2020")
STEPS = 60 * 96  # a whole 60-day season's steps

# The final's learning arms (runs by seed) and baselines (one run each). zero_controls and random
# are in every final and graded by nothing here. residual_w05 has 5 runs (identical, so its numbers
# are those of 1 run) and the other learning arms 3, so the n<5 flag is checked both ways.
SEEDS = {
    "ppo": (0, 1, 2),
    "residual_w05": (100, 101, 102, 103, 104),
    "lag_b100": (200, 201, 202),
    "ppo_agc": (300, 301, 302),
    "residual_agc": (300, 301, 302),
}
BASELINES = ("zero_controls", "random", "rb_default", "rb_tuned", "rb_tuned_agc2020")
ARMS = (*SEEDS, *BASELINES)
# Each arm's registration in the synthetic ledger (E8's two arms share one, as in #70).
REGISTRATIONS = {
    "registered-73000001": ("ppo",),
    "registered-73000002": ("residual_w05",),
    "registered-73000003": ("lag_b100",),
    "registered-73000004": ("ppo_agc", "residual_agc"),
}
RUN_IDS = {
    (arm, seed): f"run-{0x73A00000 + k:08x}"
    for k, (arm, seed) in enumerate((a, s) for a, seeds in SEEDS.items() for s in seeds)
}

# A run's season j earns c + P[j] EUR/m2 at a regime whose level is c, so its IQM is c + 2: the
# 25% trimmed mean of 8 values (any number of identical runs) keeps the sorted middle four, 0, 1,
# 2, 5. P's mean is 3 and its median 1.5, so a mean or a median is caught. Every run of a split
# shares P's order, so a same-season gain is the levels' difference in every season.
P = (5.0, -40.0, 9.0, 0.0, 50.0, 2.0, -3.0, 1.0)
# Season j has h + Q[j] hours of a constraint whose level is h: its IQM is h + 4.5 (sorted middle
# four 2, 3, 4, 9), so Σ = h_co2 + h_temp + h_rh + 13.5 h. Hours are the same at both regimes.
Q = (9.0, 0.0, 100.0, 3.0, 1.0, 12.0, 2.0, 4.0)
# (sim, agc2020) profit levels [EUR/m2]. zero_controls earns 1000 at agc2020, so an H3 bar over
# every baseline is caught; each learning arm's level at the regime its hypotheses don't grade is
# set so that grading at the wrong regime flips a verdict.
LEVEL = {
    "amsterdam_2013": {
        "rb_default": (10, 20),
        "rb_tuned": (12, 18),
        "rb_tuned_agc2020": (8, 22),
        "zero_controls": (-5, 1000),
        "random": (-50, -40),
        "ppo": (15, 17),
        "residual_w05": (14, 19),
        "lag_b100": (13, 10),
        "ppo_agc": (0, 25),
        "residual_agc": (1, 23),
    },
    "london_2013": {
        "rb_default": (30, 40),
        "rb_tuned": (31, 38),
        "rb_tuned_agc2020": (25, 45),
        "zero_controls": (-5, 1000),
        "random": (-50, -40),
        "ppo": (33, 40),  # plus JITTER at both regimes
        "residual_w05": (34, 38),
        "lag_b100": (35, 30),  # plus JITTER at sim
        "ppo_agc": (50, 43),
        "residual_agc": (0, 50),
    },
    "abbotsford_2013": {
        "rb_default": (50, 60),
        "rb_tuned": (52, 55),
        "rb_tuned_agc2020": (45, 58),
        "zero_controls": (-5, 1000),
        "random": (-50, -40),
        "ppo": (50, 57),  # plus JITTER at sim
        "residual_w05": (51, 56),
        "lag_b100": (52, 40),
        "ppo_agc": (0, 70),
        "residual_agc": (10, 61),
    },
}
# (co2, temp, rh) hour levels [h]. rb_tuned's Σ is 5 + 10 + 20 + 13.5 = 48.5 on every split;
# rb_default's and rb_tuned_agc2020's differ from it, so a Σ bar from the wrong baseline is caught.
HOURS = {
    "amsterdam_2013": {
        "rb_default": (0, 0, 0),
        "rb_tuned": (5, 10, 20),
        "rb_tuned_agc2020": (1, 1, 1),
        "zero_controls": (50, 50, 50),
        "random": (60, 60, 60),
        "ppo": (20, 20, 20),
        "residual_w05": (5, 10, 19),
        "lag_b100": (0, 5, 10),
        "ppo_agc": (4, 10, 20),
        "residual_agc": (5, 10, 20),
    },
    "london_2013": {
        "rb_default": (10, 20, 30),
        "rb_tuned": (5, 10, 20),
        "rb_tuned_agc2020": (10, 20, 25),
        "zero_controls": (50, 50, 50),
        "random": (60, 60, 60),
        "ppo": (20, 20, 20),
        "residual_w05": (5, 10, 21),
        "lag_b100": (0, 0, 0),
        "ppo_agc": (0, 0, 0),
        "residual_agc": (6, 10, 20),
    },
    "abbotsford_2013": {
        "rb_default": (10, 20, 30),
        "rb_tuned": (5, 10, 20),
        "rb_tuned_agc2020": (10, 20, 25),
        "zero_controls": (50, 50, 50),
        "random": (60, 60, 60),
        "ppo": (20, 20, 20),
        "residual_w05": (0, 0, 0),
        "lag_b100": (5, 10, 20),
        "ppo_agc": (0, 0, 0),
        "residual_agc": (1, 2, 3),
    },
}
# Seed-by-season noise [EUR/m2] (runs by seed ascending x seasons) added to a few runs' profit, so
# their CIs aren't points and depend on the seed and season order: those parts' values are
# stats.iqm_ci on the ordered matrices, computed in the test (VALUE = None in EXPECTED).
E = np.array(
    [
        [0.25, -0.5, 0.0, 0.5, -0.25, 0.125, -0.125, 0.375],
        [-0.375, 0.25, 0.5, -0.5, 0.0, 0.25, 0.125, -0.25],
        [0.5, 0.0, -0.25, 0.25, 0.375, -0.5, 0.0, -0.125],
    ]
)
JITTER = {
    ("london_2013", "ppo", "agc2020"): E,  # gain over rb_default is E: its CI straddles 0
    ("london_2013", "ppo", "sim"): E[::-1],  # gain 3 + E: its CI is near [2.9, 3.2]
    ("abbotsford_2013", "ppo", "sim"): np.roll(E, 3, axis=1),  # gain straddles 0
    ("london_2013", "lag_b100", "sim"): np.roll(E, 1, axis=0),  # profit IQM CI near [36.8, 37.2]
}
# (split, arm) -> (seed, season index, solver_failures, steps) of its one invalid season.
INVALID = {
    ("london_2013", "lag_b100"): (201, 5, 0, STEPS - 96),  # cut a day short, no solver failure
    ("abbotsford_2013", "ppo_agc"): (302, 2, 3, STEPS),  # 3 solver failures
}
# (split, arm) -> (invalid seasons, solver failures) of an H3 or H4 arm at the regime graded, as
# summary.md's part c shows them; every other H3/H4 arm is (0, 0). The bad season is bad at both
# regimes, so counting both would double these.
C_COUNTS = {
    ("london_2013", "lag_b100"): (1, 0),  # 1 season a day short: invalid, 0 solver failures
    ("abbotsford_2013", "ppo_agc"): (1, 3),  # 1 season with 3 solver failures: invalid; 3, not 4
}

VALUE = None  # a part whose value is stats.iqm_ci on the jittered matrices, computed in the test
# (split, hypothesis, arm, verdict, [(part, regime, value, bar, pass), ...]), hand-checked. A gain
# below is (arm level - baseline level) EUR/m2 in every season, so its CI is that one point; an
# IQM is level + 2; Σ is the hour levels' sum + 13.5 h. c's value is 0 when clean (a count > 0
# otherwise, checked as > 0).
EXPECTED = [
    # amsterdam_2013: every hypothesis verified.
    (
        "amsterdam_2013",
        "H1",
        "ppo",
        "verified",
        [
            ("agc2020", "agc2020", -3.0, 0.0, True),  # 17 - 20 = -3: ci_high -3 < 0
            ("sim", "sim", 5.0, 0.0, True),  # 15 - 10 = 5: ci_low 5 > 0
        ],
    ),
    (
        "amsterdam_2013",
        "H2",
        "residual_w05",
        "verified",
        [
            ("sim", "sim", 2.0, 0.0, True),  # 14 - 12 = 2 > 0
            ("agc2020", "agc2020", 1.0, 0.0, True),  # 19 - 18 = 1 > 0
        ],
    ),
    (
        "amsterdam_2013",
        "H3",
        "ppo_agc",
        "verified",
        [
            ("a", "agc2020", 27.0, 24.0, True),  # 25 + 2 > max(20, 18, 22) + 2 = 24
            ("b", "agc2020", 47.5, 48.5, True),  # 4 + 10 + 20 + 13.5 <= 48.5 (rb_default 13.5)
            ("c", "agc2020", 0.0, 0.0, True),
        ],
    ),
    (
        "amsterdam_2013",
        "H3",
        "residual_agc",
        "verified",
        [
            ("a", "agc2020", 25.0, 24.0, True),  # 23 + 2 > 24
            ("b", "agc2020", 48.5, 48.5, True),  # equal to rb_tuned's: <= holds with no slack
            ("c", "agc2020", 0.0, 0.0, True),
        ],
    ),
    (
        "amsterdam_2013",
        "H4",
        "residual_w05",
        "verified",
        [
            ("a", "sim", 16.0, 14.0, True),  # 14 + 2 > 12 + 2
            ("b", "sim", 47.5, 48.5, True),  # 5 + 10 + 19 + 13.5
            ("c", "sim", 0.0, 0.0, True),
        ],
    ),
    (
        "amsterdam_2013",
        "H4",
        "lag_b100",
        "verified",
        [
            ("a", "sim", 15.0, 14.0, True),  # 13 + 2 > 14 (at agc2020 it would be 12 < 20)
            ("b", "sim", 28.5, 48.5, True),  # 0 + 5 + 10 + 13.5
            ("c", "sim", 0.0, 0.0, True),
        ],
    ),
    # london_2013
    (
        "london_2013",
        "H1",
        "ppo",
        "fails (agc2020)",
        [
            ("agc2020", "agc2020", VALUE, 0.0, False),  # gain E: CI ~[-0.13, 0.19], high >= 0
            ("sim", "sim", VALUE, 0.0, True),  # gain 3 + E: low ~2.9 > 0
        ],
    ),
    (
        "london_2013",
        "H2",
        "residual_w05",
        "fails (agc2020)",
        [
            ("sim", "sim", 3.0, 0.0, True),  # 34 - 31 = 3
            ("agc2020", "agc2020", 0.0, 0.0, False),  # 38 - 38 = 0: not strictly above 0
        ],
    ),
    (
        "london_2013",
        "H3",
        "ppo_agc",
        "fails (a)",
        [
            # 43 + 2 = 45 < max(40, 38, 45) + 2 = 47: rb_tuned_agc2020 sets the bar (without it,
            # 42, it would pass)
            ("a", "agc2020", 45.0, 47.0, False),
            ("b", "agc2020", 13.5, 48.5, True),
            ("c", "agc2020", 0.0, 0.0, True),
        ],
    ),
    (
        "london_2013",
        "H3",
        "residual_agc",
        "fails (b)",
        [
            ("a", "agc2020", 52.0, 47.0, True),  # 50 + 2 > 47
            # 6 + 10 + 20 + 13.5 = 49.5 > rb_tuned's 48.5 (but <= rb_tuned_agc2020's 68.5)
            ("b", "agc2020", 49.5, 48.5, False),
            ("c", "agc2020", 0.0, 0.0, True),
        ],
    ),
    (
        "london_2013",
        "H4",
        "residual_w05",
        "fails (b)",
        [
            ("a", "sim", 36.0, 33.0, True),  # 34 + 2 > 31 + 2
            ("b", "sim", 49.5, 48.5, False),  # 5 + 10 + 21 + 13.5 = 49.5 > 48.5
            ("c", "sim", 0.0, 0.0, True),
        ],
    ),
    (
        "london_2013",
        "H4",
        "lag_b100",
        "fails (c)",
        [
            ("a", "sim", VALUE, 33.0, True),  # 35 + 2 + noise: ci_low ~36.8 > 33
            ("b", "sim", 13.5, 48.5, True),
            ("c", "sim", 1.0, 0.0, False),  # seed 201's 6th season ran a day short: invalid
        ],
    ),
    # abbotsford_2013
    (
        "abbotsford_2013",
        "H1",
        "ppo",
        "fails (sim)",
        [
            ("agc2020", "agc2020", -3.0, 0.0, True),  # 57 - 60 = -3 < 0
            ("sim", "sim", VALUE, 0.0, False),  # gain = noise: CI ~[-0.14, 0.19], low <= 0
        ],
    ),
    (
        "abbotsford_2013",
        "H2",
        "residual_w05",
        "fails (sim)",
        [
            ("sim", "sim", -1.0, 0.0, False),  # 51 - 52 = -1
            ("agc2020", "agc2020", 1.0, 0.0, True),  # 56 - 55 = 1
        ],
    ),
    (
        "abbotsford_2013",
        "H3",
        "ppo_agc",
        "fails (c)",
        [
            ("a", "agc2020", 72.0, 62.0, True),  # 70 + 2 > max(60, 55, 58) + 2 = 62
            ("b", "agc2020", 13.5, 48.5, True),
            ("c", "agc2020", 1.0, 0.0, False),  # seed 302's 3rd season: 3 solver failures
        ],
    ),
    (
        "abbotsford_2013",
        "H3",
        "residual_agc",
        "verified",
        [
            ("a", "agc2020", 63.0, 62.0, True),  # 61 + 2 > 62: rb_default sets the bar here
            ("b", "agc2020", 19.5, 48.5, True),  # 1 + 2 + 3 + 13.5
            ("c", "agc2020", 0.0, 0.0, True),
        ],
    ),
    (
        "abbotsford_2013",
        "H4",
        "residual_w05",
        "fails (a)",
        [
            ("a", "sim", 53.0, 54.0, False),  # 51 + 2 < 52 + 2
            ("b", "sim", 13.5, 48.5, True),
            ("c", "sim", 0.0, 0.0, True),
        ],
    ),
    (
        "abbotsford_2013",
        "H4",
        "lag_b100",
        "fails (a)",
        [
            ("a", "sim", 54.0, 54.0, False),  # 52 + 2 = 54: not strictly above rb_tuned's 54
            ("b", "sim", 48.5, 48.5, True),  # equal: <= holds
            ("c", "sim", 0.0, 0.0, True),
        ],
    ),
]
CASES = [
    pytest.param(split, hyp, arm, verdict, parts, id=f"{split}-{hyp}-{arm}-{verdict}")
    for split, hyp, arm, verdict, parts in EXPECTED
]


# --- Helpers ---------------------------------------------------------------------------------------


def _module():
    """greenleaf_control.hypotheses; an assertion, not an ImportError, while it is missing."""
    found = importlib.util.find_spec("greenleaf_control.hypotheses")
    assert found is not None, "greenleaf_control.hypotheses doesn't exist yet (#73)"
    return importlib.import_module("greenleaf_control.hypotheses")


def _runs(arm: str) -> tuple:
    """arm's runs: its seeds ascending, or (None,) for a baseline's one run."""
    return SEEDS.get(arm, (None,))


def _profit(split: str, arm: str, regime: str) -> np.ndarray:
    """arm's runs (by seed) x seasons (SEALS[2013][split] order) profit at regime [EUR/m2]."""
    level = LEVEL[split][arm][REGIMES.index(regime)]
    matrix = np.tile(level + np.array(P), (len(_runs(arm)), 1))
    return matrix + JITTER.get((split, arm, regime), 0.0)


def _frame(*, learning_shift: float = 0.0, drop: tuple = ()) -> pd.DataFrame:
    """The 2013 final's seasons.parquet as #72 writes it: one row per (arm, run, season, regime),
    shuffled (fixed seed) so a grader must order runs and seasons itself. learning_shift is added
    to every learning arm's profit (the 2012 decoy); arms in drop are left out."""
    rows = []
    for split in SPLITS:
        for arm in (a for a in ARMS if a not in drop):
            hours = HOURS[split][arm]
            for regime in REGIMES:
                profit = _profit(split, arm, regime)
                for i, seed in enumerate(_runs(arm)):
                    for j, s in enumerate(scenarios.SEALS[2013][split]):
                        bad = INVALID.get((split, arm))
                        failures, steps = bad[2:] if bad and bad[:2] == (seed, j) else (0, STEPS)
                        if arm == "random" and j < 2:
                            steps = STEPS - 96  # random's invalid seasons: graded by nothing
                        constraints = zip(("co2", "temp", "rh"), hours, strict=True)
                        viol = {c: h + Q[j] for c, h in constraints}
                        rows.append(
                            {
                                "arm": arm,
                                "run": RUN_IDS.get((arm, seed)),
                                "seed": np.nan if seed is None else float(seed),
                                "checkpoint": None if seed is None else "final",
                                "split": split,
                                **s,
                                "season_length": 60,
                                "price_regime": regime,
                                "profit_eur_m2": float(profit[i, j])
                                + (learning_shift if arm in SEEDS else 0.0),
                                "viol_hours_co2": viol["co2"],
                                "pct_in_bounds_co2": 100.0 * (1 - viol["co2"] / 1440),
                                "viol_hours_temp": viol["temp"],
                                "pct_in_bounds_temp": 100.0 * (1 - viol["temp"] / 1440),
                                "viol_hours_rh": viol["rh"],
                                "pct_in_bounds_rh": 100.0 * (1 - viol["rh"] / 1440),
                                # placeholders: the grader reads each regime's profit_eur_m2,
                                # never re-prices
                                "yield_kg_m2": 10.0,
                                "heat_cost": 5.0,
                                "elec_cost": 3.0,
                                "co2_cost": 1.0,
                                "paper_return": 0.0,
                                "solver_failures": failures,
                                "steps": steps,
                                "valid": failures == 0 and steps == STEPS,
                                "paper_reward_sum": 0.0,
                            }
                        )
    frame = pd.DataFrame(rows)
    order = np.random.RandomState(73).permutation(len(frame))
    return frame.iloc[order].reset_index(drop=True)


def _placeholder(arm: str) -> dict:
    zero = {"iqm": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    return {
        **zero,
        "guards": {},
        "invalid_episodes": 0,
        "prob_improvement": None if arm == "rb_tuned" else 0.5,
        "scored": {},
    }


def _ledger_text(*, registered_2013: bool = True, result_2013: bool = True) -> str:
    """The committed ledger (its 2012 final included) without any 2013 final rows it may hold once
    the real 2013 final is recorded, then a synthetic 2013 final: a registration per REGISTRATIONS
    entry, a run row per learning-arm seed, a seal-2013 final-registered row at those runs and,
    unless result_2013 is False, its final-result (per_split nested by regime, #72's shape,
    placeholder numbers the grader must not read)."""
    committed = [json.loads(line) for line in REAL_LEDGER.read_text(encoding="utf-8").splitlines()]
    sealed_2013 = {
        r["row_id"]
        for r in committed
        if r["kind"] == "final-registered" and ledger.seal_of(r) == 2013
    }
    lines = [
        json.dumps(r, sort_keys=True)
        for r in committed
        if r["row_id"] not in sealed_2013 and r.get("final-registered") not in sealed_2013
    ]
    [reg_2012] = [r for r in committed if r["row_id"] == FINAL_REGISTERED_2012]
    stamp = {
        "schema_version": ledger.SCHEMA_VERSION,
        "at": "2026-11-03T00:00:00+00:00",
        "issue": 72,
        "fingerprints": reg_2012["fingerprints"],
    }
    new = []
    for reg_id, arms in REGISTRATIONS.items():
        new.append(
            {
                **stamp,
                "row_id": reg_id,
                "kind": "registered",
                "hypothesis": "synthetic, for test_hypotheses",
                "arms": [{"name": a, "method": "ppo", "config": f"configs/{a}.yml"} for a in arms],
                "seeds": list(SEEDS[arms[0]]),
                "timesteps": 2_000_000,
                "scenarios": ["validation"],
                "primary": "profit_eur_m2",
                "guards": ["viol_hours_co2", "viol_hours_temp", "viol_hours_rh"],
                "rule": "synthetic",
                "budget": {"cpu_hours": 1.0, "usd": 0.0, "backend": "local"},
                "owner_ok": False,
            }
        )
        for arm in arms:
            for seed in SEEDS[arm]:
                new.append(
                    {
                        **stamp,
                        "row_id": RUN_IDS[arm, seed],
                        "kind": "run",
                        "registered": reg_id,
                        "arm": arm,
                        "seed": seed,
                        "run_dir": f"runs/synthetic/{arm}_seed{seed}",
                        "status": "running",
                    }
                )
    if registered_2013:
        new.append(
            {
                **stamp,
                "row_id": FINAL_REGISTERED_2013,
                "kind": "final-registered",
                "checkpoints": {run: "final" for run in RUN_IDS.values()},
                "baselines": list(BASELINES),
                "approved_by": "Aarmen",
                "approved_at": stamp["at"],
                "seal": 2013,
            }
        )
    if registered_2013 and result_2013:
        new.append(
            {
                **stamp,
                "row_id": FINAL_RESULT_2013,
                "kind": "final-result",
                "final-registered": FINAL_REGISTERED_2013,
                "per_split": {
                    split: {regime: {arm: _placeholder(arm) for arm in ARMS} for regime in REGIMES}
                    for split in SPLITS
                },
            }
        )
    return "\n".join([*lines, *(json.dumps(r, sort_keys=True) for r in new)]) + "\n"


def _world(base: Path, **ledger_kwargs) -> dict:
    """base/ledger.jsonl (_ledger_text) and base/results/final holding the 2013 final's
    seasons.parquet (_frame) and, at the 2012 final's dir, a decoy whose learning arms all earn
    1000 EUR/m2 less (every hypothesis would fail on it). Returns the paths."""
    final_results = base / "results" / "final"
    seasons = final_results / FINAL_REGISTERED_2013 / "seasons.parquet"
    decoy = final_results / FINAL_REGISTERED_2012 / "seasons.parquet"
    for path in (seasons, decoy):
        path.parent.mkdir(parents=True)
    _frame().to_parquet(seasons)
    _frame(learning_shift=-1000.0).to_parquet(decoy)
    path = base / "ledger.jsonl"
    path.write_text(_ledger_text(**ledger_kwargs), encoding="utf-8")
    return {"base": base, "ledger": path, "final_results": final_results, "seasons": seasons}


def _tree(top: Path, skip: Path | None = None) -> dict[str, tuple[int, int, int]]:
    """Every file, dir and link under top (but under skip) -> (type, size, mtime_ns)."""
    found = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if skip is not None and (path == skip or path.is_relative_to(skip)):
                continue
            st = path.lstat()
            found[str(path.relative_to(top))] = (
                stat.S_IFMT(st.st_mode),
                st.st_size,
                st.st_mtime_ns,
            )
    return found


def _read_spy(mp) -> list:
    """The path of every pd.read_parquet call from now on (it still reads)."""
    reads = []
    real = pd.read_parquet

    def read_parquet(*args, **kwargs):
        reads.append(Path(args[0] if args else kwargs.get("path")).resolve())
        return real(*args, **kwargs)

    mp.setattr(pd, "read_parquet", read_parquet)
    return reads


def _point_to(mp, hypotheses, final_results: Path) -> None:
    assert hasattr(hypotheses, "FINAL_RESULTS"), "hypotheses has no FINAL_RESULTS to read the final"
    mp.setattr(hypotheses, "FINAL_RESULTS", final_results)


def _refusal(call) -> str:
    """call() refuses: a ValueError, or a SystemExit with a message (not 0 or None). Returns the
    message; returning instead fails the test."""
    try:
        code = call()
    except ValueError as e:
        return str(e)
    except SystemExit as e:
        assert e.code not in (0, None), f"main exited {e.code!r}, which is not a refusal"
        return str(e.code)
    pytest.fail(f"main returned {code!r} instead of refusing")


def _computed(split: str, hyp: str, arm: str, part: str) -> float:
    """The value of a jittered part (VALUE): stats.iqm_ci on runs by seed x seasons in
    SEALS[2013][split] order. H1's: ppo's gain over rb_default at the part's regime, ci_high at
    agc2020 (below) and ci_low at sim (above); H4a's: the arm's profit ci_low at sim."""
    if hyp == "H1":
        regime = part
        gain = _profit(split, arm, regime) - _profit(split, "rb_default", regime)
        _, low, high = stats.iqm_ci(gain)
        return high if regime == "agc2020" else low
    assert (hyp, part) == ("H4", "a"), (hyp, part)
    return stats.iqm_ci(_profit(split, arm, "sim"))[1]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """A function returning the in-process run: its first call builds a world under a module tmp
    dir, points hypotheses.FINAL_RESULTS at it and runs main(["--out", <a missing dir>],
    ledger_path=<its ledger>), spying on pd.read_parquet; later calls return the same. Called inside
    each test body, so a missing module FAILS the test. Returns {"code", "out", "csv", "text",
    "header", "summary", "reads", "world", "changed"} (text: hypotheses.csv's text; changed: what
    main made, wrote or touched outside out)."""
    cache = []

    def get() -> dict:
        if not cache:
            hypotheses = _module()
            world = _world(tmp_path_factory.mktemp("hypotheses_world"))
            out = world["base"] / "final_2013"
            before = _tree(world["base"])
            with pytest.MonkeyPatch.context() as mp:
                _point_to(mp, hypotheses, world["final_results"])
                reads = _read_spy(mp)
                code = hypotheses.main(["--out", str(out)], ledger_path=world["ledger"])
            after = _tree(world["base"], skip=out)
            changed = sorted(
                k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
            )
            text = (out / "hypotheses.csv").read_text(encoding="utf-8")
            cache.append(
                {
                    "code": code,
                    "out": out,
                    "csv": pd.read_csv(out / "hypotheses.csv"),
                    "text": text,
                    "header": text.splitlines()[0],
                    "summary": (out / "summary.md").read_text(encoding="utf-8"),
                    "reads": reads,
                    "world": world,
                    "changed": changed,
                }
            )
        return cache[0]

    return get


# --- The world itself (no module needed: checks this file's hand-built data) ---------------------


def test_the_hand_built_numbers_are_what_the_comments_say():
    """The world's arithmetic, so a typo in LEVEL or HOURS can't pass as a module bug: P's IQM is 2
    (mean 3, median 1.5), Q's 4.5; a degenerate run's profit IQM is level + 2 with a point CI;
    rb_tuned's Σ is 48.5 h; the jittered gains straddle 0 where EXPECTED says they do; and the 2013
    frame has one row per arm run x season x regime."""
    from rliable.metrics import aggregate_iqm

    assert aggregate_iqm(np.array([P])) == 2.0 and np.mean(P) == 3.0 and np.median(P) == 1.5
    assert aggregate_iqm(np.array([Q])) == 4.5
    assert stats.iqm_ci(_profit("amsterdam_2013", "ppo_agc", "agc2020")) == (27.0, 27.0, 27.0)
    sigma = sum(
        aggregate_iqm(np.array([[h + q for q in Q]])) for h in HOURS["london_2013"]["rb_tuned"]
    )
    assert sigma == 48.5
    for split, regime in (("london_2013", "agc2020"), ("abbotsford_2013", "sim")):
        gain = _profit(split, "ppo", regime) - _profit(split, "rb_default", regime)
        _, low, high = stats.iqm_ci(gain)
        assert low < 0 < high, (split, regime, low, high)
    _, low, _ = stats.iqm_ci(
        _profit("london_2013", "ppo", "sim") - _profit("london_2013", "rb_default", "sim")
    )
    assert low > 2.5
    assert 33.0 < _computed("london_2013", "H4", "lag_b100", "a") < 37.0
    frame = _frame()
    runs = sum(len(_runs(a)) for a in ARMS)
    assert len(frame) == runs * 8 * len(REGIMES) * len(SPLITS)


# --- The contract surface ----------------------------------------------------------------------


def test_signature_and_default_out():
    """main(argv=None, *, ledger_path=ledger.LEDGER); OUT is control/results/analysis/final_2013 and
    FINAL_RESULTS control/results/final."""
    hypotheses = _module()
    main = inspect.signature(hypotheses.main).parameters
    assert list(main) == ["argv", "ledger_path"]
    assert main["argv"].default is None
    assert main["ledger_path"].kind is inspect.Parameter.KEYWORD_ONLY
    assert main["ledger_path"].default == ledger.LEDGER
    assert Path(hypotheses.OUT) == CONTROL / "results" / "analysis" / "final_2013"
    assert Path(hypotheses.FINAL_RESULTS) == CONTROL / "results" / "final"


def test_cli_runs_as_a_module_and_lists_out(tmp_path):
    """python -m greenleaf_control.hypotheses --help exits 0 and lists --out."""
    _module()
    done = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.hypotheses", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "--out" in done.stdout


# --- What main writes --------------------------------------------------------------------------


def test_main_returns_0_and_writes_only_the_two_files_under_out(ran):
    """main returns 0; out (missing before) holds exactly hypotheses.csv and summary.md; nothing
    else under the world (its ledger, its 2013 final, the 2012 decoy) is made, written or touched."""
    r = ran()
    assert r["code"] == 0
    assert sorted(p.name for p in r["out"].iterdir()) == OUTPUTS
    assert r["changed"] == []


def test_main_reads_only_the_2013_finals_seasons(ran):
    """Every pd.read_parquet call reads FINAL_RESULTS/final-registered-73730013/seasons.parquet:
    never the 2012 final's (the decoy at final-registered-4d0a4ba9)."""
    r = ran()
    assert r["reads"], "main read no parquet through pd.read_parquet"
    assert set(r["reads"]) == {r["world"]["seasons"].resolve()}, r["reads"]


def test_hypotheses_csv_columns_are_pinned(ran):
    """hypotheses.csv's header is exactly split,hypothesis,arm,regime,part,value,bar,pass; pass is a
    boolean column."""
    r = ran()
    assert r["header"] == COLUMNS
    assert r["csv"]["pass"].dtype == bool, r["csv"]["pass"].dtype


def test_hypotheses_csv_has_one_row_per_split_hypothesis_arm_and_part(ran):
    """3 splits x 16 rows: H1 ppo (agc2020, sim), H2 residual_w05 (sim, agc2020), H3 ppo_agc and
    residual_agc (a, b, c at agc2020), H4 residual_w05 and lag_b100 (a, b, c at sim); each row's
    regime as listed, and no other row."""
    csv = ran()["csv"]
    columns = ("split", "hypothesis", "arm", "part", "regime")
    got = list(zip(*(csv[c] for c in columns), strict=True))
    want = [
        (split, hyp, arm, part, regime)
        for split, hyp, arm, _, parts in EXPECTED
        for part, regime, *_ in parts
    ]
    assert len(want) == 48
    assert sorted(got) == sorted(want)


@pytest.mark.parametrize(("split", "hyp", "arm", "verdict", "parts"), CASES)
def test_each_parts_value_bar_and_pass(ran, split, hyp, arm, verdict, parts):
    """Each part's row: value and bar as EXPECTED hand-computes them (VALUE: stats.iqm_ci on the
    ordered jittered matrices) within 1e-9, and pass. c's value is 0 when clean, > 0 otherwise."""
    csv = ran()["csv"]
    mine = csv[(csv["split"] == split) & (csv["hypothesis"] == hyp) & (csv["arm"] == arm)]
    for part, regime, value, bar, ok in parts:
        [row] = mine[mine["part"] == part].to_dict("records")
        assert row["regime"] == regime, (part, row)
        if value is VALUE:
            value = _computed(split, hyp, arm, part)
        if part == "c" and not ok:
            assert row["value"] > 0, (part, row)
        else:
            assert row["value"] == pytest.approx(value, abs=1e-9), (part, row, value)
        assert row["bar"] == pytest.approx(bar, abs=1e-9), (part, row, bar)
        assert bool(row["pass"]) is ok, (part, row)


@pytest.mark.parametrize(("split", "hyp", "arm", "verdict", "parts"), CASES)
def test_summary_md_gives_each_verdict(ran, split, hyp, arm, verdict, parts):
    """Exactly one line of summary.md names the split, the hypothesis and the arm (whole words), and
    it carries the verdict: "verified" (and no "fails"), or "fails (<part>)" (and no "verified")."""
    summary = ran()["summary"]
    words = [re.compile(rf"\b{re.escape(w)}\b") for w in (split, hyp, arm)]
    lines = [line for line in summary.splitlines() if all(w.search(line) for w in words)]
    assert len(lines) == 1, (split, hyp, arm, lines)
    [line] = lines
    if verdict == "verified":
        assert "verified" in line and "fails" not in line, line
    else:
        assert verdict in line and "verified" not in line, (verdict, line)


def test_summary_md_names_the_2013_final_result_and_never_2012s(ran):
    """summary.md names the final-result row it graded (final-result-73730013) and neither of the
    2012 final's rows."""
    summary = ran()["summary"]
    assert FINAL_RESULT_2013 in summary
    assert FINAL_RESULT_2012 not in summary and FINAL_REGISTERED_2012 not in summary


def _verdict_lines(summary: str, split: str, hyp: str, arm: str) -> list[str]:
    """The lines of summary.md naming split, hyp and arm as whole words."""
    words = [re.compile(rf"\b{re.escape(w)}\b") for w in (split, hyp, arm)]
    return [line for line in summary.splitlines() if all(w.search(line) for w in words)]


@pytest.mark.parametrize("sentence", list(REGISTERED.values()), ids=list(REGISTERED))
def test_summary_md_states_the_registered_wording_word_for_word(ran, sentence):
    """The Conventions line and each of H1-H4 as issue #73 registers them, word for word (a bullet's
    "- " and "**" markers stripped), each whole on one line of summary.md."""
    lines = ran()["summary"].splitlines()
    assert any(sentence in line for line in lines), sentence


def test_summary_md_names_where_the_wording_is_registered(ran):
    """summary.md names the wording's source: "registered as issue #73's text (plan approved
    2026-11-02)"."""
    assert SOURCE in ran()["summary"]


def test_summary_md_legend_says_how_h3as_bar_is_read(ran):
    """A legend line says H3(a)'s bar is the largest of the three baselines' like-for-like profit
    IQMs, each through stats.iqm_ci(tiled)[0]: one line has "H3", "largest", "like-for-like",
    "profit" and "stats.iqm_ci" (the registered H3 line says max(...), not "largest")."""
    summary = ran()["summary"]
    words = ("H3", "largest", "like-for-like", "profit", "stats.iqm_ci")
    assert [line for line in summary.splitlines() if all(w in line for w in words)], summary


def test_summary_md_flags_every_verdict_on_fewer_than_5_runs(ran):
    """No claim on n < 5: on its summary.md line, an arm with 3 runs (ppo, lag_b100, ppo_agc,
    residual_agc) has "n<5: no claim" after its verdict; residual_w05 (5 runs) has no "n<5"."""
    summary = ran()["summary"]
    wrong = []
    for split, hyp, arm, verdict, _ in EXPECTED:
        [line] = _verdict_lines(summary, split, hyp, arm)
        at = line.find(verdict)
        if len(SEEDS[arm]) < 5:
            flagged = at >= 0 and NO_CLAIM in line[at + len(verdict) :]
        else:
            flagged = "n<5" not in line
        if not flagged:
            wrong.append((len(SEEDS[arm]), line))
    assert wrong == [], wrong


@pytest.mark.parametrize(
    ("split", "hyp", "arm"),
    [
        pytest.param(split, hyp, arm, id=f"{split}-{hyp}-{arm}")
        for split, hyp, arm, _, _ in EXPECTED
        if hyp in ("H3", "H4")
    ],
)
def test_summary_md_shows_part_cs_invalid_seasons_and_solver_failures_apart(ran, split, hyp, arm):
    """Part c on the arm's summary.md line shows its invalid seasons and its solver failures apart
    (e.g. "c: invalid 1, solver failures 3 (fail)"): "invalid" then C_COUNTS' first count, and
    "solver failures" then its second, each once. Clean arms show 0 and 0."""
    [line] = _verdict_lines(ran()["summary"], split, hyp, arm)
    invalid, failures = C_COUNTS.get((split, arm), (0, 0))
    assert re.findall(r"\binvalid\D{0,12}?(\d+)", line) == [str(invalid)], line
    assert re.findall(r"\bsolver failures\D{0,3}?(\d+)", line) == [str(failures)], line


# --- Refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ledger_kwargs",
    [
        pytest.param({"registered_2013": False}, id="only-the-2012-final"),
        pytest.param({"result_2013": False}, id="a-2013-final-registered-without-its-result"),
    ],
)
def test_no_2013_final_result_is_refused_before_anything_is_read(ledger_kwargs, tmp_path):
    """A ledger whose only final-result is 2012's (with or without a 2013 final-registered row, as
    between `evaluate --final` and `ledger final-result`): main refuses naming "2013" and
    "final-result", reads no parquet (not the 2012 final's), and makes or writes nothing."""
    hypotheses = _module()
    world = _world(tmp_path / "world", **ledger_kwargs)
    out = tmp_path / "out"
    before = _tree(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        _point_to(mp, hypotheses, world["final_results"])
        reads = _read_spy(mp)
        said = _refusal(lambda: hypotheses.main(["--out", str(out)], ledger_path=world["ledger"]))
    assert "2013" in said and "final-result" in said, said
    assert reads == []
    assert not out.exists()
    assert _tree(tmp_path) == before


def test_a_missing_baseline_is_refused_before_anything_is_written(tmp_path):
    """The 2013 seasons.parquet without rb_tuned_agc2020's rows (H3's bar needs them): main refuses
    naming rb_tuned_agc2020 rather than grade without it, and out is never made."""
    hypotheses = _module()
    world = _world(tmp_path / "world")
    _frame(drop=("rb_tuned_agc2020",)).to_parquet(world["seasons"])
    out = tmp_path / "out"
    with pytest.MonkeyPatch.context() as mp:
        _point_to(mp, hypotheses, world["final_results"])
        said = _refusal(lambda: hypotheses.main(["--out", str(out)], ledger_path=world["ledger"]))
    assert "rb_tuned_agc2020" in said, said
    assert not out.exists()


def test_a_2013_final_without_its_seasons_is_refused_before_anything_is_written(tmp_path):
    """The 2013 final-result is recorded but its seasons.parquet is missing: main raises (a
    ValueError, SystemExit or FileNotFoundError) and out is never made."""
    hypotheses = _module()
    world = _world(tmp_path / "world")
    world["seasons"].unlink()
    out = tmp_path / "out"
    with pytest.MonkeyPatch.context() as mp:
        _point_to(mp, hypotheses, world["final_results"])
        with pytest.raises((ValueError, SystemExit, FileNotFoundError)):
            hypotheses.main(["--out", str(out)], ledger_path=world["ledger"])
    assert not out.exists()


# --- Never 2012, writes only under --out (a fresh interpreter, audited) -------------------------

# argv: root, world base, FINAL_RESULTS, ledger, out. An audit hook records every open, listing or
# edit of the 2012 final's dir (the decoy), the checkout's control/results, control/ledger and
# control/data/weather, and every write, make, move or delete under the checkout or the world but
# out; then main runs with FINAL_RESULTS pointed at the world. Prints {"code", "seen"} last.
_AUDITED = r"""
import json, os, sys
from pathlib import Path

root, base, final_results, ledger_path, out = sys.argv[1:6]


def key(path):
    return os.path.abspath(os.fsdecode(path)).casefold()


def under(path, prefixes):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    p = key(path)
    return any(p == q or p.startswith(q + os.sep) for q in prefixes)


untouchable = [
    key(os.path.join(final_results, "final-registered-4d0a4ba9")),
    *(key(os.path.join(root, "control", *p)) for p in (("results",), ("ledger",), ("data", "weather"))),
]
guarded = [key(root), key(base)]
mine = [key(out)]
write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
edits = {
    "os.mkdir": (0,), "os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1),
    "os.replace": (0, 1), "os.truncate": (0,), "os.chmod": (0,), "os.utime": (0,),
    "os.symlink": (1,), "os.link": (1,), "shutil.rmtree": (0,), "shutil.copyfile": (1,),
    "shutil.copytree": (1,), "shutil.move": (0, 1),
}
seen = []


def hook(event, args):
    if event == "open":
        path, mode, flags = args
        if under(path, untouchable):
            seen.append(["opened", os.fsdecode(path)])
        writes = any(c in mode for c in "wax+") if isinstance(mode, str) else bool(flags & write_flags)
        if writes and under(path, guarded) and not under(path, mine):
            seen.append(["opened to write", os.fsdecode(path)])
    elif event in ("os.listdir", "os.scandir"):
        if args and under(args[0], untouchable):
            seen.append([event, os.fsdecode(args[0])])
    elif event in edits:
        for i in edits[event]:
            if i < len(args) and under(args[i], untouchable + guarded) and not under(args[i], mine):
                seen.append([event, os.fsdecode(args[i])])


sys.addaudithook(hook)
from greenleaf_control import hypotheses

hypotheses.FINAL_RESULTS = Path(final_results)
code = hypotheses.main(["--out", out], ledger_path=Path(ledger_path))
print(json.dumps({"code": code, "seen": seen}))
"""


@pytest.mark.slow
def test_a_run_never_touches_2012_and_writes_only_under_out(ran, tmp_path):
    """In a fresh interpreter (no bytecode written), main on a fresh world into a missing out:
    opens, lists or edits nothing in the 2012 final's dir (the decoy), the checkout's
    control/results, control/ledger or control/data/weather; writes, makes, moves or deletes nothing
    under the checkout or the world but out; returns 0; and writes the same hypotheses.csv as the
    in-process run (deterministic)."""
    _module()
    world = _world(tmp_path / "world")
    out = tmp_path / "world" / "final_2013"
    argv = [
        str(ROOT),
        str(world["base"]),
        str(world["final_results"]),
        str(world["ledger"]),
        str(out),
    ]
    done = subprocess.run(
        [sys.executable, "-c", _AUDITED, *argv],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["code"] == 0
    assert report["seen"] == []
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    assert (out / "hypotheses.csv").read_text(encoding="utf-8") == ran()["text"]

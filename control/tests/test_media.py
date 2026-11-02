"""Tests for greenleaf_control.media (issue #21, S4-0): one command regenerates the README and
LinkedIn visuals from the simulator, each file labelled with its source and commit.

media plays the baselines on PAPER and draws them:
- play(controller, season_length=60) -> {"frame", "result", "states", "controls", "params"}: the
  season from envs.run_baseline(controller, PAPER, season_length); states, controls from
  metrics.trace_from_infos; params the PAPER model's parameter vector; result
  metrics.episode_metrics(states, controls, params, season_steps=96 x season_length); frame
  evaluate._frame(infos), the contract's per-step frame.
- running(states, controls, params, ends) -> DataFrame[step, profit_eur_m2, viol_hours_co2,
  viol_hours_temp, viol_hours_rh]: one row per k in ends (k >= 1), step = k, the metrics of the
  first k steps (states[:k + 1], controls[:k]).
- season_replay(frame, counters, out_gif, out_mp4=None, *, title, subtitle=None, footer, fps=15)
  -> Figure: a FuncAnimation on a plain Figure (no pyplot), saved as a GIF (and an MP4 when
  out_mp4 is given), returned in its final frame's state. Its climate lines are _hourly(frame)'s
  co2, temp and rh against t, each over its metrics.BOUNDS band (axhspan); its actuator heat
  strip is _hourly(frame)'s six u_* columns as a 6-row array (row j = U_COLS[j], column i = hour
  i); after a counters row's k steps it shows the k // 4 complete hours; its counters are
  counters' rows as text, the final frame showing the last row; the title, the subtitle (when
  given) and the footer are each one text, as given.
- baselines_card(seasons, out, *, footer) -> Figure: seasons maps rb_default, zero_controls and
  random to play()'s dicts; three columns left to right in that order, each with its own strip
  (the same 6-row _hourly array), its result's numbers, and each day's share of steps outside each
  safe range (a 3 x days array, rows in metrics.CONSTRAINTS order). One text says what profit
  counts: fruit value minus heating, power and CO₂.
- _hourly(frame) (issue #51) is export_replay's hourly convention, so the README and the site draw
  the same series: hour h's t, co2, temp and rh are frame row 4h + 3 (the end of the hour), and
  each u_* is the mean of rows 4h..4h+3. Before #51 media drew frame.iloc[::4] (15 min into the
  hour, actuators as instants); the tests pinning that changed with it, deliberately.
- footer(source, stamp, *, place=PAPER_NAME, note="A simulator run, not a registered result")
  -> str, and main(argv=None) -> int, the CLI. main refuses an invalid season (a solver failure,
  or the wrong length) before writing anything, and titles the replay as simulated, with the
  season's day count from the run and PAPER's location and year; its subtitle names the
  controller and the simulator.

The trade-off still (issue #50) draws recorded VALIDATION results, every number read from the
ledger when the card is drawn:
- TRADEOFF, (result_id, arm, status) entries (row ids and statuses, never numbers), and
  E1_RESULT, rb_tuned's E1 result row; both read at call time, so the tests point them at
  synthetic rows.
- tradeoff_points(path=ledger.LEDGER) -> list[dict], one per TRADEOFF entry, in order: arm,
  status, result, decision (the one decision row on that result), registered and seeds (the
  registered row's); profit and each viol_hours_<c> as (iqm, ci_low, ci_high) from per_arm[arm]
  and its guards; yield_kg_m2, the guard's iqm. ValueError naming the row for a row that is
  missing or isn't a result, no decision row (or two), scenarios other than ["validation"], an
  arm missing from per_arm, a null iqm, crashed or invalid_episodes > 0, or a missing guard.
- rb_tuned_reference(n_seeds, path=ledger.LEDGER, parquet=selection.RESULTS /
  "rb_tuned_validation.parquet") -> dict: first the parquet checked against E1's row (each
  viol_hours_<c>'s per-season mean equal to its guard's iqm exactly, paper_return's to the iqm
  within rel 1e-9; ValueError otherwise), then profit_eur_m2, each viol_hours_<c> and
  yield_kg_m2 as stats.iqm_ci(np.tile(seasons, (n_seeds, 1)))[0], n_seeds, and sha256 (the
  parquet file's).
- tradeoff_card(points, references, out, *, footer) -> Figure, saved at exactly 1200 x 627 px:
  three panels sharing the profit axis, x the hours outside each safe range in metrics.CONSTRAINTS
  order. Each point is an errorbar at (its hours' iqm, its profit iqm) with bars to both CIs; the
  dropped arm hollow and MUTED, residual_w05 LEAF, the other adopted arms filled. rb_tuned is a
  distinct hollow marker with no bars at the largest-n reference. A legend outside the panels has
  one line per point naming its result id; above the panels, the title (VALIDATION, Amsterdam
  2010), the CI line and a note with rb_tuned's profit at each smaller seed count.
- main also writes tradeoff.png and a media.json entry for it (split "validation", scenario
  VALIDATION, ledger [row ids], metrics {points, references}); the other entries are unchanged.

The card's tests read synthetic rows that ledger.register, run_start, run_end, record_result and
decide write to a tmp ledger, and a tmp parquet. No test reads the committed ledger or results: a
tripwire on ledger.rows and pandas.read_parquet fails first, and every test that runs main binds
its card to the tmp files.

The tests read what is drawn from the returned Figure (its lines, patches, images and texts),
never its pixels. Numbers on the figures are printed with 2 decimals, f"{v:.2f}" (hours are
multiples of 0.25 h, so 2 decimals are exact), each on one text line with its label: profit with
"€/m²", each constraint's hours with its viz label ("CO₂", "temp", "RH"), fruit growth with "fruit
growth". No figure prints a total of the three hour counts (an hour out of two bands would count
twice), says "yield" (yield_kg_m2 is the net change in standing fruit, not a harvest) or uses an em
dash.

Known answers: one day is 96 steps (86,400 s / 900 s), so a 1-day season has 96 frame rows and 24
hourly rows (t = 3, 7, ..., 95, each hour's last row). A violating step is 900 s = 0.25 h, so a
prefix's viol_hours_<c> is its out-of-band frame rows (a value on a bound is in bounds) x 0.25.

The climate stripes (issue #51; Aarmen's scope update replaced the side-by-side line replay) replay
one recorded VALIDATION result: rb_tuned above one E6 residual_w05 run at its selected checkpoint,
on SEASON = scenarios.VALIDATION[1] (Amsterdam 2010 from day 59, upstream's PAPER season), in the
site's "Instrument" style (SITE_BG #F2F3F0, SITE_INK #111311, SITE_MUTED #6A6F68; a GROTESK title,
MONO captions and totals, each set per text with fontfamily=, never in rcParams).
- replay_pick(result_id, path=ledger.LEDGER, parquet=selection.RESULTS / RB_PARQUET) -> dict: the
  run to replay by PROTOCOL §6's rule (no best seed): the result row's scored runs sorted by
  (recorded SEASON profit, seed), index (n - 1) // 2, the lower median, ties to the lower seed. A
  run's record is per_scenario[1] of its selected.json candidate at its selected checkpoint. It
  returns export_replay._registered_run's keys (id, registered, result, arm, seed, checkpoint from
  the result row, commit from the run's manifest) plus run_dir, decision (the decision row's id),
  record and rb_record (the parquet's SEASON row). ValueError naming the row, before any season:
  a row that isn't a result; a registration not on exactly ["validation"]; no decision row, two,
  or one that isn't adopted; more than one arm; a scored run's run_dir missing; any run's
  selected.json not on VALIDATION in 60-day seasons, or its checkpoint not the result row's; a
  parquet with no SEASON row.
- _stripes(panels, figsize, *, title, caption, footer) -> (Figure, draw): panels are
  [{heading, frame, result, counters}], "Tuned rules" (rb_tuned) above "Rules + co-pilot" (the
  run). Each panel is one image: the 24 x n_days grid _hourly(frame)["temp"] reshaped (n_days, 24)
  and transposed (row = hour 00..23 from the top, column = day), on TEMP_CMAP with clim
  TEMP_SCALE = (10, 36) °C in both panels. TEMP_CMAP has 26 colours, one per 1 °C, none of them
  the page colour. draw(d) shows days 1..d (later columns masked), "day d / n", and each panel's
  totals from its counters' row d - 1 (counters are running() at the end of each day): profit
  with "€/m²" and each constraint's hours with its label, at 2 decimals. The figure is figsize on
  SITE_BG; title, caption and footer are each one text, as given.
- duel_replay(panels, out_gif, out_mp4, out_png, *, title, caption, footer, fps=REPLAY_FPS) ->
  Figure: each day d = 1..n drawn DAY_FRAMES times, then a 2 s hold; the GIF at 12.8 x 7.2 in and
  100 dpi (1280 x 720), the MP4 (none when out_mp4 is None) at 150 dpi (1920 x 1080), and the
  still at 12 x 6.27 in and 100 dpi to out_png (hero.png, 1200 x 627).
- main --replay RESULT_ROW also writes hero.png, duel_replay.gif and duel_replay.mp4: rb_tuned
  through export_replay._season("rb_tuned", SEASON, None), the run through evaluate.run_policy at
  its selected checkpoint, each checked against its record on every ledger.RESULT_KEYS key (hours
  and steps exactly, the rest to rel 1e-9) before --out is created; a mismatch refuses. Each new
  media.json entry has today's keys plus split, ledger [run row, result, decision, E1], run {seed,
  checkpoint, run_dir, manifest_commit} and metrics {"rb_tuned", <arm>}. Without --replay main is
  as it was, so it still runs on a fresh clone (E6's run dirs aren't in git).
The replay's tests build an E6-shaped tmp ledger (10 seeds, fake run dirs holding manifest.json
and selected.json) and a tmp parquet, and replace evaluate.run_policy with a spy returning a real
season, so no checkpoint loads; the fast ones patch export_replay.SEASON_LENGTH to 1 day.

The notes once the final is recorded (issue #56): while the ledger main reads has no final-result
row, the trade-off footer's note is CARD_NOTE and the replay's REPLAY_NOTE, as before. Once it has
one, they read "Recorded VALIDATION results; the 2012 test results are in docs/REPORT.md §5
(<its row id>)" and "A replay of a recorded VALIDATION result; the 2012 test results are in
docs/REPORT.md §5 (<its row id>)", and neither says "sealed"; nothing else in the cards changes.
main reads that row through ledger.rows; _card_reads sends any read of the committed ledger to the
tmp one, and the tests spend the seal on a copy of the synthetic ledger with ledger.final_register
and ledger.final_result.

The fast tests play 1- or 2-day seasons and write a GIF only. The slow ones run the CLI (60-day
seasons, GIF and MP4 through ffmpeg) and are skipped without ffmpeg and ffprobe. The 2012 weather
is sealed (PROTOCOL §8): every test runs under a tripwire on pandas.read_csv that fails before any
*2012.csv is read.
"""

import ast
import functools
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FFMpegWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize, to_hex, to_rgb, to_rgba
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image

import greenleaf_control
from greenleaf_control import (
    envs,
    evaluate,
    ledger,
    manifest,
    metrics,
    rule_based,
    scenarios,
    selection,
    stats,
)

# greenleaf_control.media is imported inside each test body, so while it is missing each test
# FAILS rather than ERRORs at collection.

PAPER = scenarios.PAPER
N_DAY = 96  # 86,400 s / 900 s
CONTROLLERS = ("rb_default", "zero_controls", "random")  # the card's columns, left to right
U_COLS = ["u_boil", "u_co2", "u_thscr", "u_vent", "u_lamp", "u_blscr"]  # upstream's order
RUNNING_COLS = ["step", "profit_eur_m2", "viol_hours_co2", "viol_hours_temp", "viol_hours_rh"]
LABELS = {"co2": "CO₂", "temp": "temp", "rh": "RH"}  # viz.CLIMATE_LABELS' words
RB_LABEL = "upstream's default rule controller (the grower's rules)"
ENDS = [24, 48, 72, 96]  # the replay test's counter rows: four frames, the last the whole day
TITLE = "The grower's rules on PAPER, one day"
SUBTITLE = "upstream's default rule controller (the grower's rules), simulated in GreenLight-Gym"
FOOTER = "source: test, PAPER (Amsterdam 2010), simulator run, not a registered result"
PLACE = f"{PAPER['location']} {PAPER['growth_year']}"  # "Amsterdam 2010"
RB_SUBTITLE = ("upstream's default rule controller", "GreenLight-Gym")  # what main's names
SHA = "0123456789abcdef0123456789abcdef01234567"  # a fixed commit for the footer tests
GOLDEN = json.loads((Path(__file__).parent / "golden" / "values.json").read_text())["ours"]
MEDIA_FILES = [
    "baselines.png",
    "media.json",
    "season_replay.gif",
    "season_replay.mp4",
    "tradeoff.png",
]
ENTRY_KEYS = {"file", "source", "scenario", "commit", "dirty", "gl_gym", "made_at", "metrics"}

# --- The trade-off still (issue #50) ---
VALIDATION = scenarios.VALIDATION
VAL_PLACE = f"{VALIDATION[0]['location']} {VALIDATION[0]['growth_year']}"  # "Amsterdam 2010"
# The card's footer place and note, as the issue words them.
FOOTER_PLACE = f"VALIDATION, {VAL_PLACE} days " + ", ".join(
    str(s["start_day"]) for s in VALIDATION
)  # "VALIDATION, Amsterdam 2010 days 45, 59, 75"
CARD_NOTE = "Recorded VALIDATION results; the 2012 test weather is still sealed"
# Its note once the ledger holds a final-result row (issue #56); {} is that row's id.
CARD_FINAL_NOTE = "Recorded VALIDATION results; the 2012 test results are in docs/REPORT.md §5 ({})"
PAPER_NOTE = "A simulator run, not a registered result"  # footer's default note, today's words
CI_LINE = "IQM over seeds × seasons, 95% stratified bootstrap CI"
CARD_FOOTER = "source: synthetic rows in a tmp ledger, commit 0123456"  # what the card tests pass
PARQUET = "rb_tuned_validation.parquet"
NEW_NAMES = ("TRADEOFF", "E1_RESULT", "tradeoff_points", "rb_tuned_reference", "tradeoff_card")
# The issue's table, (result_id, arm, status); E4 has one decision row, residual_w4 dropped in it.
ISSUE_TRADEOFF = [
    ("result-06364fa1", "ppo", "adopted"),
    ("result-d1f0b8e7", "residual_w05", "adopted"),
    ("result-2a890455", "residual_w1", "adopted"),
    ("result-2a890455", "residual_w4", "dropped"),
]
HOUR_KEYS = [f"viol_hours_{c}" for c in metrics.CONSTRAINTS]
GUARDS = [*HOUR_KEYS, "paper_return", "yield_kg_m2", "solver_failures"]  # E3, E4 and E6's
DIAGNOSTICS = {"min_approx_kl": 0.002, "final_entropy": -1.5, "final_explained_variance": 0.8}

# Synthetic ledger numbers, (iqm, ci_low, ci_high): profit [€/m²], hours [h] (multiples of
# 0.25 h), fruit growth [kg/m²]. Every point sits apart from the others in every panel, prints
# apart at 2 decimals, and has two or more nonzero hours whose total is printed nowhere.
E3 = {
    "ppo": {
        "profit": (2.4, 2.33, 2.46),
        "co2": (0.5, 0.25, 1.0),
        "temp": (153.25, 140.0, 166.5),
        "rh": (135.0, 120.5, 150.0),
        "yield": (7.13, 6.9, 7.4),
    }
}
E4 = {
    # Not on the card: E6's residual_w05 is. Its numbers differ, so reading the wrong row shows.
    "residual_w05": {
        "profit": (0.63, 0.48, 0.74),
        "co2": (2.25, 1.5, 3.0),
        "temp": (60.0, 52.5, 68.25),
        "rh": (220.0, 210.0, 231.5),
        "yield": (9.1, 8.8, 9.4),
    },
    "residual_w1": {
        "profit": (0.4, 0.32, 0.45),
        "co2": (3.0, 2.25, 4.0),
        "temp": (70.25, 61.0, 80.5),
        "rh": (230.5, 221.0, 240.25),
        "yield": (8.81, 8.6, 9.0),
    },
    "residual_w4": {
        "profit": (-0.37, -0.48, -0.26),
        "co2": (6.5, 5.0, 8.25),
        "temp": (110.0, 99.5, 121.75),
        "rh": (180.25, 170.0, 191.0),
        "yield": (6.42, 6.2, 6.7),
    },
}
E6 = {
    "residual_w05": {
        "profit": (0.77, 0.7, 0.84),
        "co2": (1.75, 1.0, 2.5),
        "temp": (48.5, 42.0, 55.75),
        "rh": (214.0, 205.25, 223.5),
        "yield": (9.63, 9.4, 9.9),
    }
}
# experiment -> (its arms' numbers, its registered seeds): 5 seeds for E3 and E4, 10 for E6.
EXPERIMENTS = {
    "E3": (E3, list(range(5))),
    "E4": (E4, list(range(5))),
    "E6": (E6, list(range(100, 110))),
}
# The synthetic TRADEOFF, in the issue's order: (experiment, arm, status).
TRADEOFF_SPEC = (
    ("E3", "ppo", "adopted"),
    ("E6", "residual_w05", "adopted"),
    ("E4", "residual_w1", "adopted"),
    ("E4", "residual_w4", "dropped"),
)
DROPPED = "residual_w4"

# rb_tuned's three VALIDATION seasons (days 45, 59, 75), as evaluate's parquet holds them.
RB_SEASONS = {
    "profit_eur_m2": (-3.0, 0.6, 1.2),
    "viol_hours_co2": (0.0, 1.0, 5.0),
    "viol_hours_temp": (120.0, 90.0, 75.25),
    "viol_hours_rh": (300.0, 250.5, 222.75),
    "yield_kg_m2": (8.0, 9.0, 11.0),
    "paper_return": (3000.0, 3600.0, 4000.0),
}
# rb_tuned tiled to n seeds is n x 3 scores; the IQM drops int(0.25 x 3n) from each end:
# 7 of 30 for n = 10, leaving 3 x low + 10 x middle + 3 x high over 16; 3 of 15 for n = 5,
# leaving 2 x low + 5 x middle + 2 x high over 9.
REF = {
    10: {
        "profit_eur_m2": 0.0375,  # (3 x -3.0 + 10 x 0.6 + 3 x 1.2) / 16 = 0.6 / 16 €/m²
        "viol_hours_co2": 1.5625,  # (3 x 0 + 10 x 1.0 + 3 x 5.0) / 16 = 25 / 16 h
        "viol_hours_temp": 92.859375,  # (3 x 75.25 + 10 x 90 + 3 x 120) / 16 = 1485.75 / 16 h
        "viol_hours_rh": 254.578125,  # (3 x 222.75 + 10 x 250.5 + 3 x 300) / 16 = 4073.25 / 16 h
        "yield_kg_m2": 9.1875,  # (3 x 8 + 10 x 9 + 3 x 11) / 16 = 147 / 16 kg/m²
    },
    5: {
        "profit_eur_m2": -0.6 / 9,  # (2 x -3.0 + 5 x 0.6 + 2 x 1.2) / 9 = -0.0667 €/m²
        "viol_hours_co2": 15 / 9,  # (2 x 0 + 5 x 1.0 + 2 x 5.0) / 9 = 1.6667 h
        "viol_hours_temp": 840.5 / 9,  # (2 x 75.25 + 5 x 90 + 2 x 120) / 9 = 93.3889 h
        "viol_hours_rh": 2298 / 9,  # (2 x 222.75 + 5 x 250.5 + 2 x 300) / 9 = 255.3333 h
        "yield_kg_m2": 83 / 9,  # (2 x 8 + 5 x 9 + 2 x 11) / 9 = 9.2222 kg/m²
    },
}
# Parquets that disagree with E1's row, which holds the per-season means of the seasons above.
PARQUET_MISMATCHES = {
    # One season 0.25 h (one 900 s step) more outside the temp band: the mean is 0.0833 h off.
    "temp_hours_off_by_one_step": {**RB_SEASONS, "viol_hours_temp": (120.25, 90.0, 75.25)},
    # Every season's paper_return 1e-6 too high: the mean is off by rel 1e-6 > 1e-9.
    "paper_return_off_by_rel_1e-6": {
        **RB_SEASONS,
        "paper_return": tuple(v * (1 + 1e-6) for v in RB_SEASONS["paper_return"]),
    },
}
# The rows tradeoff_points refuses; _refusal builds each.
REFUSALS = (
    "missing",
    "not_a_result",
    "undecided",
    "two_decisions",
    "not_validation",
    "arm_absent",
    "iqm_null",
    "crashed",
    "invalid_episodes",
    "no_rh_guard",
    "no_yield_guard",
)

# --- The climate stripes (issue #51) ---
SEASON = VALIDATION[1]  # Amsterdam 2010 from day 59, upstream's PAPER season
REPLAY_NAMES = (
    "SEASON",
    "SITE_BG",
    "SITE_INK",
    "SITE_MUTED",
    "TEMP_SCALE",
    "TEMP_CMAP",
    "REPLAY_FPS",
    "DAY_FRAMES",
    "GROTESK",
    "MONO",
    "_hourly",
    "replay_pick",
    "_stripes",
    "duel_replay",
)
REPLAY_FILES = ("duel_replay.gif", "duel_replay.mp4", "hero.png")
TOP, BOTTOM = "Tuned rules", "Rules + co-pilot"  # the panels' headings, top to bottom
STRIPES_TITLE = "2 days of greenhouse air, hour by hour"
# A caption and footer the renderer tests pass in: no decimal number, so none is mistaken for a
# total.
CAPTION = (
    "air temperature, 10 to 36 °C; safe range 15 to 34 °C. VALIDATION season, Amsterdam 2010"
    " from day 59, not the final. run-0000abcd, seed 107, checkpoint 1750000"
)
REPLAY_FOOTER = "source: run-0000abcd, VALIDATION, Amsterdam 2010 day 59, commit 0123456"
REPLAY_PLACE = (
    "VALIDATION, Amsterdam 2010 day 59"  # the replay's footer place, as the issue words it
)
REPLAY_NOTE = "A replay of a recorded VALIDATION result; the 2012 test weather is still sealed"
# Its note once the ledger holds a final-result row (issue #56's scope note); {} is that row's id.
REPLAY_FINAL_NOTE = (
    "A replay of a recorded VALIDATION result; the 2012 test results are in docs/REPORT.md §5 ({})"
)
ARM = "residual_w05"
RUN_COMMIT = "fedcba9876543210fedcba9876543210fedcba98"  # the fake runs' training commit, made up
REPLAY_SEEDS = list(range(100, 110))  # E6's ten seeds
# Each seed's recorded SEASON profit as an offset from the record's [€/m²]. Sorted by (profit,
# seed): 100, 104, 102, 106, 107, 103, 108, 101, 105, 109. The lower median of 10 is index
# (10 - 1) // 2 = 4, seed 107; the upper median (index 5) is 103 and the best is 109.
OFFSETS = {
    100: -0.4,
    101: 0.3,
    102: -0.2,
    103: 0.1,
    104: -0.3,
    105: 0.4,
    106: -0.1,
    107: 0.0,
    108: 0.2,
    109: 0.5,
}
# Seeds 103 and 107 tie at the median: sorted by (profit, seed) 100, 104, 102, 106, 103, 107, ...;
# index 4 is 103, the lower seed of the tie (ties to the higher seed, or the upper median, give 107).
TIED = {**OFFSETS, 103: 0.0}
# Each run's selected checkpoint (the result row's, and its selected.json's).
CHECKPOINTS = {
    100: "1500000",
    101: "final",
    102: "1250000",
    103: "500000",
    104: "2000000",
    105: "1000000",
    106: "1500000",
    107: "1750000",
    108: "final",
    109: "1250000",
}
# A recorded 60-day season with every ledger.RESULT_KEYS key: made-up numbers, hours multiples of
# 0.25 h, steps 60 x 96 = 5,760. RB_RECORD is rb_tuned's SEASON row.
RECORD = {
    "profit_eur_m2": 1.5,
    "viol_hours_co2": 0.25,
    "viol_hours_temp": 50.0,
    "viol_hours_rh": 180.0,
    "pct_in_bounds_co2": 99.75,
    "pct_in_bounds_temp": 96.5,
    "pct_in_bounds_rh": 87.5,
    "yield_kg_m2": 9.5,
    "heat_cost": 1.25,
    "elec_cost": 3.5,
    "co2_cost": 0.5,
    "paper_return": 3500.0,
    "solver_failures": 0,
    "steps": 5760,
    "paper_reward_sum": 3400.0,
}
RB_RECORD = {
    **RECORD,
    "profit_eur_m2": 1.25,
    "viol_hours_co2": 0.0,
    "viol_hours_temp": 98.0,
    "viol_hours_rh": 251.0,
}
PER_ARM = {
    "iqm": 0.8,
    "ci_low": 0.7,
    "ci_high": 0.9,
    "guards": {key: {"iqm": 1.0, "ci_low": 0.5, "ci_high": 1.5} for key in HOUR_KEYS},
    "invalid_episodes": 0,
}
# The flaws replay_pick refuses; _replay_sources builds each.
PICK_REFUSALS = (
    "not_a_result",
    "not_validation",
    "undecided",
    "dropped",
    "two_decisions",
    "two_arms",
    "run_dir_missing",
    "selected_not_validation",
    "selected_not_60_days",
    "checkpoint_disagrees",
    "no_season_row",
)
# A replayed season that differs from its record: (side, key). One 900 s step more outside the temp
# band is 0.25 h; a profit rel 1e-6 off is far above the rel 1e-9 the check allows.
RECORD_MISMATCHES = [
    ("rb_tuned", "viol_hours_temp"),
    ("rb_tuned", "profit_eur_m2"),
    ("run", "viol_hours_temp"),
    ("run", "profit_eur_m2"),
]


@pytest.fixture(autouse=True)
def no_2012_reads(monkeypatch) -> None:
    """Fail loudly if the test reads any 2012 weather file. pandas.read_csv is the only file read
    upstream's weather loader makes (its own year, and the next year's file past the year's end);
    the check runs before the real read, and every other file goes through the real one."""
    real_read_csv = pd.read_csv

    def read_csv(filepath_or_buffer, *args, **kwargs):
        if Path(str(filepath_or_buffer)).name.endswith("2012.csv"):
            raise AssertionError(f"the sealed 2012 test weather was read: {filepath_or_buffer}")
        return real_read_csv(filepath_or_buffer, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_csv)


@pytest.fixture(autouse=True)
def no_committed_reads(monkeypatch) -> None:
    """Fail loudly if the test reads the committed ledger or results (AGENTS.md: a test touches
    control/ledger/ and control/results/ only through tmp_path). ledger.rows and
    pandas.read_parquet are the card's reads; the check runs before the real read, and every tmp
    file goes through the real one."""
    real_rows, real_read_parquet = ledger.rows, pd.read_parquet
    committed = (ledger.LEDGER.parent.resolve(), selection.RESULTS.parent.resolve())

    def check(path) -> None:
        p = Path(path).resolve()
        if any(p == d or d in p.parents for d in committed):
            raise AssertionError(f"a committed ledger or result was read: {p}")

    def rows(kind=None, issue=None, path=ledger.LEDGER):
        check(path)
        return real_rows(kind, issue, path)

    def read_parquet(path, *args, **kwargs):
        check(path)
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(ledger, "rows", rows)
    monkeypatch.setattr(pd, "read_parquet", read_parquet)


@pytest.fixture(scope="module")
def media_dir(tmp_path_factory) -> Path:
    """A tmp dir the cached figures below are written into (makes no media call, so never ERRORs)."""
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="module")
def sources(media_dir) -> dict:
    """The synthetic ledger and parquet the card reads, built once with the ledger's own API
    (makes no media call, so never ERRORs)."""
    return _sources(media_dir / "sources")


@functools.cache
def _baseline(controller: str) -> dict:
    """A 1-day PAPER season of a baseline, built here without media: run_baseline's infos graded
    by trace_from_infos + episode_metrics (the PAPER model's params, 96 season steps), and
    evaluate's frame of the same infos. Cached: the season is deterministic (random's generator is
    seeded by the start day)."""
    infos = envs.run_baseline(controller, PAPER, 1)
    states, controls = metrics.trace_from_infos(infos)
    env = envs.make_raw_env(scenario=PAPER, season_length=1, normalize_actions=False)
    params = np.asarray(env.unwrapped.p, dtype=np.float64)
    env.close()
    result = metrics.episode_metrics(states, controls, params, season_steps=N_DAY)
    return {
        "frame": evaluate._frame(infos),
        "result": result,
        "states": states,
        "controls": controls,
        "params": params,
    }


@functools.cache
def _played(controller: str) -> dict:
    import greenleaf_control.media as media

    return media.play(controller, 1)


def _counters(b: dict, ends: list[int]) -> pd.DataFrame:
    """running()'s table built here from metrics: the first k steps' numbers for each k."""
    rows = []
    for k in ends:
        m = metrics.episode_metrics(
            b["states"][: k + 1], b["controls"][:k], b["params"], season_steps=k
        )
        rows.append({"step": k, **{col: m[col] for col in RUNNING_COLS[1:]}})
    return pd.DataFrame(rows, columns=RUNNING_COLS)


@functools.cache
def _replay(controller: str, out_dir: Path) -> tuple[Figure, Path]:
    """season_replay of a baseline's 1-day frame with _counters at ENDS, GIF only; cached so the
    replay tests render it once per controller."""
    import greenleaf_control.media as media

    b = _baseline(controller)
    out = out_dir / f"replay_{controller}" / "season_replay.gif"
    out.parent.mkdir(exist_ok=True)
    fig = media.season_replay(b["frame"], _counters(b, ENDS), out, title=TITLE, footer=FOOTER)
    return fig, out


@functools.cache
def _card(out_dir: Path) -> tuple[Figure, Path]:
    """baselines_card of the three baselines' 1-day seasons; cached so it renders once."""
    import greenleaf_control.media as media

    out = out_dir / "card" / "baselines.png"
    out.parent.mkdir(exist_ok=True)
    fig = media.baselines_card({c: _baseline(c) for c in CONTROLLERS}, out, footer=FOOTER)
    return fig, out


def _hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """export_replay's hourly convention (issue #51), computed here as export_replay computes it:
    hour h's t, co2, temp and rh are frame row 4h + 3 (the end of the hour), each u_* the mean of
    rows 4h..4h+3."""
    n_hours = len(frame) // 4
    u = frame[U_COLS].to_numpy(dtype=np.float64).reshape(n_hours, 4, len(U_COLS)).mean(axis=1)
    end = frame.iloc[3::4].reset_index(drop=True)
    return end.assign(**{col: u[:, j] for j, col in enumerate(U_COLS)})


def _xy(line) -> tuple[list[float], list[float]]:
    return (
        np.asarray(line.get_xdata(), dtype=float).tolist(),
        np.asarray(line.get_ydata(), dtype=float).tolist(),
    )


def _texts(fig: Figure) -> list[Text]:
    """Every visible, non-empty text artist on fig (titles, labels, ticks, figure texts)."""
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]


def _lines_of(texts: list[Text]) -> list[str]:
    return [line for t in texts for line in t.get_text().splitlines()]


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _has(line: str, v: float) -> bool:
    """line prints v at 2 decimals as a whole number: not a piece of a longer one (0.00 in 10.00)
    nor the unsigned tail of its negative (2.89 in -2.89)."""
    return re.search(rf"(?<![\d.\-]){re.escape(_fmt(v))}(?!\d)", line) is not None


def _assert_numbers(lines: list[str], result: dict, where: str) -> None:
    """profit on a line with "€/m²", each constraint's hours on a line with its label."""
    profit = result["profit_eur_m2"]
    assert any("€/m²" in ln and _has(ln, profit) for ln in lines), (where, _fmt(profit), lines)
    for c in metrics.CONSTRAINTS:
        hours = result[f"viol_hours_{c}"]
        assert any(LABELS[c] in ln and _has(ln, hours) for ln in lines), (where, c, _fmt(hours))


def _strips(fig: Figure, n_cols: int) -> list:
    """Every image or mesh on fig holding a 6-row x n_cols array (a colorbar's mesh is not one)."""
    found = fig.findobj(lambda a: isinstance(a, (AxesImage, QuadMesh)))
    return [a for a in found if np.shape(a.get_array()) == (6, n_cols)]


def _strip_array(artist) -> np.ndarray:
    """The strip's values, with anything masked (a frame not yet shown) as NaN."""
    return np.ma.filled(np.ma.asarray(artist.get_array(), dtype=np.float64), np.nan)


def _columns(fig: Figure, strips: list) -> tuple[list, list[list[str]]]:
    """The strips left to right, and the text lines whose rendered centre falls in each strip's
    column: the figure split at the midpoints between the strips' axes centres."""
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def centre(bbox) -> float:
        return (bbox.x0 + bbox.x1) / 2

    strips = sorted(strips, key=lambda a: centre(a.axes.get_window_extent(renderer)))
    xs = [centre(a.axes.get_window_extent(renderer)) for a in strips]
    cuts = [(a + b) / 2 for a, b in zip(xs, xs[1:], strict=False)]
    columns = [[] for _ in strips]
    for t in _texts(fig):
        k = int(np.searchsorted(cuts, centre(t.get_window_extent(renderer))))
        columns[k].extend(t.get_text().splitlines())
    return strips, columns


def _no_season_played(monkeypatch, media) -> None:
    """Every way media could play a season fails: envs.run_baseline and make_raw_env, evaluate's
    run_rb, and media's own play (and those names on media, in case it imported them by name)."""

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any season")

        return refuse

    for module, name in [(envs, "run_baseline"), (envs, "make_raw_env"), (evaluate, "run_rb")]:
        monkeypatch.setattr(module, name, trip(f"{module.__name__}.{name}"))
    for name in ("play", "run_baseline", "make_raw_env", "run_rb"):
        if hasattr(media, name):
            monkeypatch.setattr(media, name, trip(f"media.{name}"))


def _imported_modules(path: Path) -> set[str]:
    """Every module path imported anywhere in path's source (as in test_viz.py)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join(filter(None, ["greenleaf_control", base]))
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


# --- The trade-off still's synthetic sources (issue #50) ---------------------------------------


def _media_card():
    """media, asserting the trade-off card's names exist first, so until they do a test FAILS on
    this assertion rather than an AttributeError or a TypeError."""
    import greenleaf_control.media as media

    missing = [name for name in NEW_NAMES if not hasattr(media, name)]
    assert not missing, f"media has no {', '.join(missing)} yet (issue #50)"
    return media


def _ci(t) -> dict:
    return dict(zip(("iqm", "ci_low", "ci_high"), t, strict=True))


def _numbers(arm: dict) -> dict:
    """A learning arm's per_arm entry as selection writes it: profit the primary, then the guards
    (crashed and spend_usd are the ledger's to add)."""
    return {
        **_ci(arm["profit"]),
        "guards": {
            **{f"viol_hours_{c}": _ci(arm[c]) for c in metrics.CONSTRAINTS},
            "paper_return": _ci((3500.0, 3400.0, 3600.0)),
            "yield_kg_m2": _ci(arm["yield"]),
            "solver_failures": _ci((0.0, 0.0, 0.0)),
        },
        "invalid_episodes": 0,
    }


def _e1_numbers() -> dict:
    """rb_tuned's E1 entry as selection writes it: iqm_ci of its 1 run x 3 VALIDATION seasons, a
    3-score IQM being the plain mean (int(0.25 x 3) = 0 trimmed). paper_return
    (3000 + 3600 + 4000) / 3 = 3533.33; hours co2 (0 + 1 + 5) / 3 = 2.0 h, temp
    (120 + 90 + 75.25) / 3 = 95.08 h, rh (300 + 250.5 + 222.75) / 3 = 257.75 h. No fruit growth
    guard, as E1 has none."""

    def ci(key: str) -> dict:
        return _ci(stats.iqm_ci(np.array([RB_SEASONS[key]])))

    return {
        **ci("paper_return"),
        "guards": {key: ci(key) for key in HOUR_KEYS},
        "invalid_episodes": 0,
    }


def _experiment(
    path: Path,
    *,
    issue: int,
    arms: dict,
    seeds: list[int],
    method: str = "ppo",
    on: tuple = ("validation",),
    primary: str = "profit_eur_m2",
    guards: list[str] = GUARDS,
    crashed: bool = False,
    decided: bool = True,
) -> dict:
    """Register an experiment on path, start and end a run per arm x seed (the first seed's
    crashed when crashed), record arms ({name: per_arm entry}) and, when decided, adopt it.
    Returns its registered, result and decision row ids."""
    reg = ledger.register(
        issue=issue,
        hypothesis="synthetic rows for the trade-off card's tests",
        arms=[{"name": name, "method": method, "config": {}} for name in arms],
        seeds=list(seeds),
        timesteps=1000,
        scenarios=list(on),
        primary=primary,
        guards=list(guards),
        rule="synthetic",
        budget={"cpu_hours": 0.0, "usd": 0.0, "backend": "local"},
        path=path,
    )
    selected = {}
    for name in arms:
        for i, seed in enumerate(seeds):
            run = ledger.run_start(
                registered=reg["row_id"],
                arm=name,
                seed=seed,
                run_dir=f"runs/{name}-{seed}",
                path=path,
            )
            ledger.run_end(
                run=run["row_id"],
                status="crashed" if crashed and i == 0 else "finished",
                wall_s=1.0,
                timesteps=1000,
                solver_failures=0,
                cost_usd=None,
                diagnostics=DIAGNOSTICS,
                path=path,
            )
            selected[run["row_id"]] = "final"
    res = ledger.record_result(registered=reg["row_id"], per_arm=arms, selected=selected, path=path)
    ids = {"registered": reg["row_id"], "result": res["row_id"]}
    if decided:
        dec = ledger.decide(result=res["row_id"], decision="adopted", note="synthetic", path=path)
        ids["decision"] = dec["row_id"]
    return ids


def _write_parquet(path: Path, seasons: dict) -> None:
    """rb_tuned's VALIDATION parquet as evaluate's CLI writes it: one row per season, in order."""
    rows = [
        {
            "controller": "rb_tuned",
            "checkpoint": None,
            "split": "validation",
            **s,
            "season_length": 60,
            **{key: values[i] for key, values in seasons.items()},
            "valid": True,
            "solver_failures": 0,
            "steps": 60 * N_DAY,
        }
        for i, s in enumerate(VALIDATION)
    ]
    pd.DataFrame(rows).to_parquet(path)


@functools.cache
def _sources(root: Path) -> dict:
    """The synthetic ledger and parquet, built once: an E1-like rb_tuned row (1 seed, primary
    paper_return, hour guards only), E3-like ppo (5 seeds), E4-like residual_w05, residual_w1 and
    residual_w4 (5 seeds, one decision) and E6-like residual_w05 (10 seeds), each decided; and
    rb_tuned's parquet of RB_SEASONS. tradeoff is TRADEOFF_SPEC with the rows' ids."""
    root.mkdir(exist_ok=True)
    path = root / "ledger.jsonl"
    path.unlink(missing_ok=True)
    rows = {
        "E1": _experiment(
            path,
            issue=101,
            arms={"rb_tuned": _e1_numbers()},
            seeds=[0],
            method="rb_tuned",
            primary="paper_return",
            guards=HOUR_KEYS,
        )
    }
    for k, (name, (numbers, seeds)) in enumerate(EXPERIMENTS.items()):
        arms = {arm: _numbers(n) for arm, n in numbers.items()}
        rows[name] = _experiment(path, issue=102 + k, arms=arms, seeds=seeds)
    parquet = root / PARQUET
    _write_parquet(parquet, RB_SEASONS)
    tradeoff = [(rows[exp]["result"], arm, status) for exp, arm, status in TRADEOFF_SPEC]
    return {
        "ledger": path,
        "parquet": parquet,
        "rows": rows,
        "tradeoff": tradeoff,
        "e1": rows["E1"]["result"],
    }


def _want_points(src: dict) -> list[dict]:
    """tradeoff_points' answer on the synthetic ledger, in TRADEOFF order, written from the
    numbers the rows were made with."""
    want = []
    for exp, arm, status in TRADEOFF_SPEC:
        numbers, seeds = EXPERIMENTS[exp]
        n, ids = numbers[arm], src["rows"][exp]
        want.append(
            {
                "arm": arm,
                "status": status,
                "result": ids["result"],
                "decision": ids["decision"],
                "registered": ids["registered"],
                "seeds": seeds,
                "profit": n["profit"],
                **{f"viol_hours_{c}": n[c] for c in metrics.CONSTRAINTS},
                "yield_kg_m2": n["yield"][0],
            }
        )
    return want


def _tiled(key: str, n: int) -> float:
    """stats.iqm_ci of rb_tuned's seasons tiled to n seeds: the IQM the reference draws."""
    return stats.iqm_ci(np.tile(np.array(RB_SEASONS[key]), (n, 1)))[0]


def _json(value):
    """value as media.json holds it (tuples become lists)."""
    return json.loads(json.dumps(value))


def _card_reads(monkeypatch, media, src: dict, **over) -> None:
    """main's card reads the synthetic rows, never the committed ones: TRADEOFF and E1_RESULT name
    them, and tradeoff_points and rb_tuned_reference are bound to the tmp ledger and parquet
    (their defaults bind the committed files at import). over replaces src's tradeoff, ledger or
    parquet. Before #50 media has no card, so there is nothing to bind and main reads neither.
    Any other ledger.rows read of the committed ledger (issue #56's: is a final-result row in it?)
    reads src's ledger instead, so the notes follow src's rows and the committed file is never
    opened."""
    src = {**src, **over}
    monkeypatch.setattr(media, "TRADEOFF", src["tradeoff"], raising=False)
    monkeypatch.setattr(media, "E1_RESULT", src["e1"], raising=False)
    binds = {
        "tradeoff_points": {"path": src["ledger"]},
        "rb_tuned_reference": {"path": src["ledger"], "parquet": src["parquet"]},
    }
    for name, kwargs in binds.items():
        real = getattr(media, name, None)
        if real is not None:
            monkeypatch.setattr(media, name, functools.partial(real, **kwargs))
    committed, inner = ledger.LEDGER.resolve(), ledger.rows

    def rows(kind=None, issue=None, path=ledger.LEDGER):
        return inner(kind, issue, src["ledger"] if Path(path).resolve() == committed else path)

    monkeypatch.setattr(ledger, "rows", rows)


def _refusal(case: str, src: dict, path: Path) -> tuple[list, set[str]]:
    """src's ledger copied to path with case's flaw added. Returns the TRADEOFF to draw (src's
    entries, plus the flawed one last) and the row ids the error may name."""
    shutil.copyfile(src["ledger"], path)
    e3 = src["rows"]["E3"]
    entries = list(src["tradeoff"])
    if case == "missing":
        return [*entries, ("result-0badc0de", "ppo", "adopted")], {"result-0badc0de"}
    if case == "not_a_result":
        return [*entries, (e3["decision"], "ppo", "adopted")], {e3["decision"]}
    if case == "arm_absent":
        return [*entries, (e3["result"], "residual_w9", "adopted")], {e3["result"]}
    if case == "two_decisions":
        ledger.decide(result=e3["result"], decision="dropped", note="a second call", path=path)
        return entries, {e3["result"]}
    numbers, kwargs = _numbers(E3["ppo"]), {}
    if case == "undecided":
        kwargs["decided"] = False
    elif case == "not_validation":
        kwargs["on"] = ("train_eval",)
    elif case == "crashed":
        kwargs["crashed"] = True
    elif case == "iqm_null":
        numbers.update(iqm=None, ci_low=None, ci_high=None)
    elif case == "invalid_episodes":
        numbers["invalid_episodes"] = 1
    else:
        guard = {"no_rh_guard": "viol_hours_rh", "no_yield_guard": "yield_kg_m2"}[case]
        del numbers["guards"][guard]
    ids = _experiment(path, issue=200, arms={"ppo": numbers}, seeds=[0, 1], **kwargs)
    named = {ids["result"], ids["registered"]} if case == "not_validation" else {ids["result"]}
    return [*entries, (ids["result"], "ppo", "adopted")], named


def _edited(src: Path, dst: Path, result_id: str, arm: str, **iqms: float) -> None:
    """A copy of the ledger at src with result_id's per_arm[arm] iqms replaced: profit= sets the
    primary's, viol_hours_<c>= that guard's."""
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    for row in rows:
        if row["row_id"] == result_id:
            numbers = row["per_arm"][arm]
            for key, v in iqms.items():
                (numbers if key == "profit" else numbers["guards"][key])["iqm"] = v
    dst.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def _draw_tradeoff(media, src: dict, out: Path, ledger_path: Path | None = None) -> Figure:
    """The card of src's rows (or ledger_path's), with TRADEOFF and E1_RESULT pointed at them:
    tradeoff_points, rb_tuned_reference at 5 and 10 seeds (the points' seed counts), then
    tradeoff_card with CARD_FOOTER."""
    path = ledger_path or src["ledger"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(media, "TRADEOFF", src["tradeoff"])
        mp.setattr(media, "E1_RESULT", src["e1"])
        points = media.tradeoff_points(path)
        refs = [media.rb_tuned_reference(n, path, src["parquet"]) for n in (5, 10)]
        return media.tradeoff_card(points, refs, out, footer=CARD_FOOTER)


@functools.cache
def _tradeoff(out_dir: Path) -> tuple[Figure, Path]:
    """The card of the synthetic sources; cached so it renders once."""
    media = _media_card()
    out = out_dir / "tradeoff" / "tradeoff.png"
    out.parent.mkdir(exist_ok=True)
    return _draw_tradeoff(media, _sources(out_dir / "sources"), out), out


def _errorbars(ax) -> list[ErrorbarContainer]:
    return [c for c in ax.containers if isinstance(c, ErrorbarContainer)]


def _segments(container) -> list:
    """An errorbar's bar segments, each [[x0, y0], [x1, y1]]."""
    return [
        np.asarray(s, dtype=float).tolist()
        for col in container.lines[2]
        for s in col.get_segments()
    ]


def _point(line) -> list[list[float]]:
    return np.asarray(line.get_xydata(), dtype=float).tolist()


def _errorbar_at(ax, x: float, y: float) -> list:
    """ax's errorbars whose marker is the one point (x, y), exactly."""
    return [c for c in _errorbars(ax) if c.lines[0] is not None and _point(c.lines[0]) == [[x, y]]]


def _lines_near(ax, x: float, y: float) -> list[Line2D]:
    """ax's lines drawing the one point (x, y), to rel 1e-9."""
    return [
        line
        for line in ax.get_lines()
        if len(_point(line)) == 1 and _point(line)[0] == pytest.approx([x, y], rel=1e-9)
    ]


def _panels(fig: Figure) -> list:
    """The axes holding errorbars with bars, left to right."""
    found = [ax for ax in fig.axes if any(_segments(c) for c in _errorbars(ax))]
    return sorted(found, key=lambda ax: ax.get_position().x0)


def _legend_texts(fig: Figure) -> tuple[list, list[str]]:
    """Every legend on fig (the axes' and the figure's), and each legend entry's text, whitespace
    joined (an entry may wrap)."""
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + fig.legends
    return legends, [" ".join(t.get_text().split()) for lg in legends for t in lg.get_texts()]


# --- The climate stripes' helpers (issue #51) ---------------------------------------------------


def _media_has(*names: str):
    """media, asserting the climate stripes' names (all of REPLAY_NAMES, or those given) exist
    first, so until they do a test FAILS on this assertion rather than an AttributeError."""
    import greenleaf_control.media as media

    missing = [name for name in names or REPLAY_NAMES if not hasattr(media, name)]
    assert not missing, f"media has no {', '.join(missing)} yet (issue #51)"
    return media


@functools.cache
def _graded(controller: str, days: int) -> dict:
    """A baseline's days-long SEASON as export_replay._season grades it, built here without media:
    run_baseline's infos, evaluate._grade with the model's params and N (so every
    ledger.RESULT_KEYS key, paper_reward_sum too), evaluate's frame and trace_from_infos. Cached:
    the seasons are deterministic."""
    infos = envs.run_baseline(controller, SEASON, days)
    env = envs.make_raw_env(scenario=SEASON, season_length=days, normalize_actions=False)
    result = evaluate._grade(infos, env.unwrapped)
    params = np.asarray(env.unwrapped.p, dtype=np.float64)
    env.close()
    states, controls = metrics.trace_from_infos(infos)
    return {
        "frame": evaluate._frame(infos),
        "result": result,
        "states": states,
        "controls": controls,
        "params": params,
    }


def _plain(v):
    """A number as JSON and a ledger hold it: an int stays an int, anything else a float."""
    return int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else float(v)


def _record(result: dict) -> dict:
    """A result's ledger.RESULT_KEYS numbers, as a selected.json or the parquet records them."""
    return {key: _plain(result[key]) for key in ledger.RESULT_KEYS}


def _keys(numbers, keys=ledger.RESULT_KEYS) -> dict:
    """numbers' keys (a dict or a pandas row), each as a plain number."""
    return {key: _plain(numbers[key]) for key in keys}


def _stripe_panels(days: int) -> list[dict]:
    """Two SEASON panels as main builds them, rb_tuned ("Tuned rules") above rb_default standing
    in for the run ("Rules + co-pilot"; a checkpoint can't load here), each with its frame, result
    and counters at the end of each day (96, 192, ... steps)."""
    panels = []
    for heading, controller in ((TOP, "rb_tuned"), (BOTTOM, "rb_default")):
        s = _graded(controller, days)
        panels.append(
            {
                "heading": heading,
                "frame": s["frame"],
                "result": s["result"],
                "counters": _counters(s, [N_DAY * d for d in range(1, days + 1)]),
            }
        )
    return panels


def _grid(frame: pd.DataFrame) -> np.ndarray:
    """The panel's 24 x n_days grid: _hourly(frame)["temp"], row = hour 00..23, column = day."""
    temp = _hourly(frame)["temp"].to_numpy(dtype=np.float64)
    return temp.reshape(len(frame) // N_DAY, 24).T


def _temp_images(fig: Figure, n_days: int) -> list:
    """Every image or mesh on fig holding a 24 x n_days array, top to bottom."""
    found = fig.findobj(lambda a: isinstance(a, (AxesImage, QuadMesh)))
    images = [a for a in found if np.shape(a.get_array()) == (24, n_days)]
    return sorted(images, key=lambda a: -a.axes.get_position().y0)


def _rendered(fig: Figure):
    """fig drawn once on an Agg canvas; returns its renderer, for the texts' rendered places."""
    FigureCanvasAgg(fig)
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _y(text: Text, renderer) -> float:
    box = text.get_window_extent(renderer)
    return (box.y0 + box.y1) / 2


def _printing(fig: Figure, label: str, v: float) -> list[Text]:
    """fig's texts with a line holding label and v at 2 decimals."""
    return [
        t
        for t in _texts(fig)
        if any(label in ln and _has(ln, v) for ln in t.get_text().splitlines())
    ]


def _family(text: Text) -> list[str]:
    return list(text.get_fontfamily())


def _selected_json(
    record: dict, seed: int, offset: float, *, checkpoint: str, scenarios_=None, days=60
) -> dict:
    """A run's selected.json as selection.select_checkpoint writes it, two candidates:
    - its checkpoint's per_scenario: days 45 and 75 at the record's profit - offset, day 59
      (SEASON) at + offset, so ranking on another season or on the mean gives another seed;
    - another checkpoint's, every season at the record's profit + 0.05 x (seed - 100), so ranking
      on it gives seed 104."""
    p = record["profit_eur_m2"]

    def candidate(profits: list[float]) -> dict:
        return {
            "per_scenario": [{**record, "profit_eur_m2": v, "valid": True} for v in profits],
            "mean_paper_return": record["paper_return"],
            "mean_profit_eur_m2": sum(profits) / 3,
            "mean_cost": None,
            "qualifies": None,
        }

    other = "1000000" if checkpoint == "final" else "final"
    alt = p + 0.05 * (seed - 100)
    return {
        "checkpoint": checkpoint,
        "method": "residual",
        "rule": "max_mean_paper_return",
        "scenarios": VALIDATION if scenarios_ is None else scenarios_,
        "season_length": days,
        "budgets": None,
        "tolerance": None,
        "candidates": {
            checkpoint: candidate([p - offset, p + offset, p - offset]),
            other: candidate([alt, alt, alt]),
        },
    }


def _write_rb_parquet(path: Path, season: dict, days=(45, 59, 75)) -> None:
    """rb_tuned's VALIDATION parquet as evaluate's CLI writes it, one row per season of days: day
    59's (SEASON's) holds season's numbers, day 45's and 75's its profit 0.5 €/m² higher and lower,
    so reading another season's row shows."""
    shift = {45: 0.5, 59: 0.0, 75: -0.5}
    rows = [
        {
            "controller": "rb_tuned",
            "checkpoint": None,
            "split": "validation",
            **s,
            "season_length": 60,
            **season,
            "profit_eur_m2": season["profit_eur_m2"] + shift[s["start_day"]],
            "valid": True,
        }
        for s in VALIDATION
        if s["start_day"] in days
    ]
    pd.DataFrame(rows).to_parquet(path)


def _replay_sources(
    root: Path,
    case: str | None = None,
    *,
    offsets: dict = OFFSETS,
    record: dict = RECORD,
    rb_record: dict = RB_RECORD,
) -> dict:
    """An E6-shaped tmp ledger and parquet, built with the ledger's own API: one registration
    (issue 36) of residual_w05 over seeds 100-109 on ["validation"]; per seed a fake run dir
    root/runs/<registered>/residual_w05/seed<S> holding manifest.json (commit RUN_COMMIT) and
    selected.json (_selected_json at CHECKPOINTS[S], its SEASON profit record's + offsets[S]); a
    run row naming the dir and a finished run-end; one result row selecting every run at
    CHECKPOINTS; one adopted decision; and rb_tuned's parquet with rb_record as SEASON's row.
    case adds one of PICK_REFUSALS' flaws (on seed 100, never the pick, for a run's flaw). Returns
    the ledger and parquet paths, the row ids, {seed: (run row, run dir)}, the id to pass
    (replay) and the names a refusal may give (named)."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ledger.jsonl"
    arms = [ARM, "residual_w1"] if case == "two_arms" else [ARM]
    reg = ledger.register(
        issue=36,
        hypothesis="synthetic rows for the climate stripes' tests",
        arms=[{"name": arm, "method": "residual", "config": {}} for arm in arms],
        seeds=REPLAY_SEEDS,
        timesteps=2_000_000,
        scenarios=["train_eval"] if case == "not_validation" else ["validation"],
        primary="profit_eur_m2",
        guards=HOUR_KEYS,
        rule="synthetic",
        budget={"cpu_hours": 0.0, "usd": 0.0, "backend": "local"},
        path=path,
    )
    runs, selected = {}, {}
    for arm in arms:
        for seed in REPLAY_SEEDS:
            run_dir = root / "runs" / reg["row_id"] / arm / f"seed{seed}"
            run_dir.mkdir(parents=True)
            manifest_json = {"commit": RUN_COMMIT, "config": {"method": "residual"}}
            (run_dir / "manifest.json").write_text(json.dumps(manifest_json))
            chosen = _selected_json(record, seed, offsets[seed], checkpoint=CHECKPOINTS[seed])
            (run_dir / "selected.json").write_text(json.dumps(chosen))
            run = ledger.run_start(
                registered=reg["row_id"], arm=arm, seed=seed, run_dir=str(run_dir), path=path
            )
            ledger.run_end(
                run=run["row_id"],
                status="finished",
                wall_s=1.0,
                timesteps=2_000_000,
                solver_failures=0,
                cost_usd=None,
                diagnostics=DIAGNOSTICS,
                path=path,
            )
            selected[run["row_id"]] = CHECKPOINTS[seed]
            if arm == ARM:
                runs[seed] = (run, run_dir)
    res = ledger.record_result(
        registered=reg["row_id"],
        per_arm={arm: PER_ARM for arm in arms},
        selected=selected,
        path=path,
    )
    calls = {"undecided": [], "dropped": ["dropped"], "two_decisions": ["adopted", "dropped"]}
    decisions = [
        ledger.decide(result=res["row_id"], decision=d, note="synthetic", path=path)["row_id"]
        for d in calls.get(case, ["adopted"])
    ]
    parquet = root / PARQUET
    _write_rb_parquet(parquet, rb_record, (45, 75) if case == "no_season_row" else (45, 59, 75))

    run100, dir100 = runs[100]
    flawed = json.loads((dir100 / "selected.json").read_text())
    if case == "run_dir_missing":
        shutil.rmtree(dir100)
    elif case == "selected_not_validation":
        flawed["scenarios"] = [PAPER]  # selected on PAPER alone, as ppo_paper's runs were
    elif case == "selected_not_60_days":
        flawed["season_length"] = 30
    elif case == "checkpoint_disagrees":
        flawed["checkpoint"] = "final"  # the result row selected "1500000"
    if case in ("selected_not_validation", "selected_not_60_days", "checkpoint_disagrees"):
        (dir100 / "selected.json").write_text(json.dumps(flawed))

    named = {
        "not_a_result": set(decisions),
        "not_validation": {res["row_id"], reg["row_id"]},
        "undecided": {res["row_id"]},
        "dropped": {res["row_id"], *decisions},
        "two_decisions": {res["row_id"], *decisions},
        "two_arms": {res["row_id"], reg["row_id"]},
        "no_season_row": {PARQUET},
    }.get(case, {run100["row_id"], str(dir100), str(dir100.resolve())})
    return {
        "ledger": path,
        "parquet": parquet,
        "registered": reg["row_id"],
        "result": res["row_id"],
        "decision": decisions[0] if decisions else None,
        "runs": runs,
        "replay": decisions[0] if case == "not_a_result" else res["row_id"],
        "named": named,
    }


def _replay_reads(monkeypatch, media, src: dict) -> None:
    """main's replay_pick reads src's tmp ledger and parquet (its defaults bind the committed files
    at import). Before #51 media has no replay_pick, so there is nothing to bind."""
    real = getattr(media, "replay_pick", None)
    if real is not None:
        bound = functools.partial(real, path=src["ledger"], parquet=src["parquet"])
        monkeypatch.setattr(media, "replay_pick", bound)


def _no_replay_played(monkeypatch, media) -> None:
    """_no_season_played, and evaluate.run_policy and export_replay's routes (run_policy, _season)
    too, so no replay can play either."""
    import greenleaf_control.export_replay as export_replay

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(f"{name} was called: the refusal must come before any season")

        return refuse

    _no_season_played(monkeypatch, media)
    monkeypatch.setattr(evaluate, "run_policy", trip("evaluate.run_policy"))
    for module in (export_replay, media):
        for name in ("run_policy", "_season"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, trip(f"{module.__name__}.{name}"))


def _replay_world(
    monkeypatch, media, sources: dict, src: dict, run: dict, *, days: int = 1, drawing="record"
) -> dict:
    """main --replay on fake data: play returns the cached 1-day PAPER seasons, ffmpeg is reported
    available, the card reads the synthetic rows and replay_pick src's, export_replay plays
    days-long seasons (rb_tuned for real), and evaluate.run_policy is a spy that records its
    arguments and returns run's season as run_policy(..., return_trace=True) does, so no
    checkpoint loads. drawing: "record" records every drawing (_drawing_recorded); "today" records
    today's three and leaves duel_replay real; "trip" fails on any drawing. Returns the calls."""
    import greenleaf_control.export_replay as export_replay

    monkeypatch.setattr(media, "play", lambda controller, season_length=60: _baseline(controller))
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    monkeypatch.setattr(export_replay, "SEASON_LENGTH", days)
    _card_reads(monkeypatch, media, sources)
    _replay_reads(monkeypatch, media, src)
    if drawing == "trip":
        calls = {}

        def trip(name: str):
            def refuse(*args, **kwargs):
                raise AssertionError(f"media.{name} was called: the refusal must come first")

            return refuse

        for name in ("season_replay", "baselines_card", "tradeoff_card", "duel_replay"):
            monkeypatch.setattr(media, name, trip(name))
    else:
        calls = _drawing_recorded(monkeypatch, media, duel=drawing == "record")

    signature = inspect.signature(evaluate.run_policy)
    calls["run_policy"] = []

    def run_policy(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls["run_policy"].append(dict(bound.arguments))
        trace = (run["states"], run["controls"], run["params"])
        return dict(run["result"]), run["frame"].copy(), trace

    monkeypatch.setattr(evaluate, "run_policy", run_policy)
    for module in (export_replay, media):
        if hasattr(module, "run_policy"):
            monkeypatch.setattr(module, "run_policy", run_policy)
    return calls


def _refused(media, argv: list[str], capsys) -> tuple[object, str]:
    """main(argv)'s exit code and its refusal's text (a ValueError's message, or stderr)."""
    raised = ""
    try:
        code = media.main(argv)
    except SystemExit as e:
        code = e.code
    except ValueError as e:
        code, raised = 1, str(e)
    return code, f"{code} {raised} {capsys.readouterr().err}"


@functools.cache
def _duel(out_dir: Path) -> tuple[Figure, Path]:
    """duel_replay of the 2-day panels with out_mp4=None (no ffmpeg here) at its default fps;
    cached so it renders once. Returns the Figure and the dir it wrote into."""
    media = _media_has()
    out = out_dir / "duel"
    out.mkdir(exist_ok=True)
    fig = media.duel_replay(
        _stripe_panels(2),
        out / "duel_replay.gif",
        None,
        out / "hero.png",
        title=STRIPES_TITLE,
        caption=CAPTION,
        footer=REPLAY_FOOTER,
    )
    return fig, out


# --- play: the season the pictures draw --------------------------------------------------------


def test_play_rb_default_is_run_rbs_season():
    """play("rb_default", 1) is evaluate.run_rb(default params, PAPER, 1, record_steps=True)'s
    season: the frame equal exactly, and the result run_rb's minus paper_reward_sum (media grades
    with metrics only). 96 steps, valid."""
    want, want_frame = evaluate.run_rb(rule_based.load_rb_params("default"), PAPER, 1, True)

    got = _played("rb_default")

    assert set(got) == {"frame", "result", "states", "controls", "params"}
    pd.testing.assert_frame_equal(got["frame"], want_frame, check_exact=True)
    assert got["result"] == {k: v for k, v in want.items() if k != "paper_reward_sum"}
    assert (got["result"]["steps"], got["result"]["valid"]) == (N_DAY, True)


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_play_is_the_baselines_season_graded_by_metrics(controller):
    """play(c, 1) for each baseline: states and controls are trace_from_infos of
    envs.run_baseline(c, PAPER, 1), params the PAPER model's parameter vector, result
    episode_metrics of them over 96 season steps, and frame evaluate's frame of the same infos,
    all exactly."""
    want = _baseline(controller)

    got = _played(controller)

    np.testing.assert_array_equal(got["states"], want["states"])
    np.testing.assert_array_equal(got["controls"], want["controls"])
    np.testing.assert_array_equal(np.asarray(got["params"], dtype=np.float64), want["params"])
    assert got["result"] == want["result"]
    pd.testing.assert_frame_equal(got["frame"], want["frame"], check_exact=True)


# --- running: the counters ---------------------------------------------------------------------


@pytest.mark.parametrize("controller", CONTROLLERS)
def test_running_counts_each_prefix_and_ends_on_the_seasons_metrics(controller):
    """running(states, controls, params, ends) on a 1-day season with hourly ends 4, 8, ..., 96:
    the columns step, profit_eur_m2 and viol_hours_<c>, one row per end with step = k, each row the
    metrics of the first k steps, and the last row (k = 96, the whole day) equal to the season's
    result exactly.

    Hand check of the hours: row k's viol_hours_<c> is the frame's out-of-band rows t = 0..k-1
    (metrics.BOUNDS, a value on a bound in bounds) x 0.25 h (900 s / 3,600 s per step).
    Non-vacuous: the profit changes over the day, and zero_controls and random leave at least one
    band during it, so their hours grow.
    """
    import greenleaf_control.media as media

    b = _baseline(controller)
    ends = list(range(4, N_DAY + 1, 4))

    got = media.running(b["states"], b["controls"], b["params"], ends)

    assert list(got.columns) == RUNNING_COLS
    assert got["step"].tolist() == ends
    for row, k in zip(got.to_dict("records"), ends, strict=True):
        m = metrics.episode_metrics(
            b["states"][: k + 1], b["controls"][:k], b["params"], season_steps=k
        )
        assert {col: row[col] for col in RUNNING_COLS[1:]} == {
            col: m[col] for col in RUNNING_COLS[1:]
        }, k
        for c in metrics.CONSTRAINTS:
            low, high = metrics.BOUNDS[c]
            v = b["frame"][c].to_numpy(dtype=np.float64)[:k]
            # each out-of-band step is 900 s = 0.25 h
            assert row[f"viol_hours_{c}"] == np.count_nonzero((v < low) | (v > high)) * 0.25, (k, c)
    last = got.iloc[-1]
    assert {col: last[col] for col in RUNNING_COLS[1:]} == {
        col: b["result"][col] for col in RUNNING_COLS[1:]
    }
    assert got["profit_eur_m2"].nunique() > 1
    if controller != "rb_default":
        assert any(got[f"viol_hours_{c}"].iloc[-1] > 0 for c in metrics.CONSTRAINTS)


# --- season_replay: the animation --------------------------------------------------------------


@pytest.mark.parametrize("controller", ["rb_default", "zero_controls"])
def test_replay_draws_the_hourly_climate_over_the_bounds_bands(controller, media_dir):
    """Done when: the animation's series equal the run's frame and the bands equal metrics.BOUNDS.

    In the final frame, for each of co2, temp and rh, a line holds _hourly(frame)'s values against
    its t (24 hourly points for one day, each hour's last row, t = 3, 7, ..., 95) exactly, on axes
    whose only patch is one band (axhspan) from metrics.BOUNDS[c]: co2 300-1600 ppm, temp
    15-34 °C, rh 50-85 %. Issue #51 moved this from frame.iloc[::4] (t = 0, 4, ..., 92) to
    export_replay's convention; non-vacuous: the two differ."""
    fig, _ = _replay(controller, media_dir)

    assert isinstance(fig, Figure)
    frame = _baseline(controller)["frame"]
    hourly = _hourly(frame)
    lines = fig.findobj(Line2D)
    for c in metrics.CONSTRAINTS:
        assert hourly[c].tolist() != frame.iloc[::4][c].tolist(), c
        want = (hourly["t"].astype(float).tolist(), hourly[c].tolist())
        matches = [line for line in lines if _xy(line) == want]
        assert matches, f"no line draws frame row 4h + 3's {c!r} against its t"
        for line in matches:
            patches = line.axes.patches
            assert all(isinstance(p, Rectangle) for p in patches), c  # axhspan's patch
            bands = [(p.get_y(), p.get_y() + p.get_height()) for p in patches]
            assert bands == [metrics.BOUNDS[c]], c


def test_replay_strip_is_the_hourly_actuators(media_dir):
    """rb_default's final frame has one 6-row heat strip: an image or mesh whose array is each
    hour's mean of u_boil, u_co2, u_thscr, u_vent, u_lamp, u_blscr (frame rows 4h..4h+3, as
    _hourly) as rows (6 x 24 for one day), every value shown (nothing masked or NaN), exactly.
    (zero_controls' strip is all 0, so it couldn't show a row-order slip; rb_default's rows read
    differently upside down, and its hourly means differ from frame.iloc[::4]'s instants.)"""
    fig, _ = _replay("rb_default", media_dir)

    frame = _baseline("rb_default")["frame"]
    u = _hourly(frame)[U_COLS].to_numpy(dtype=np.float64).T
    assert not np.array_equal(u, u[::-1])  # non-vacuous: a flipped strip would differ
    assert not np.array_equal(u, frame.iloc[::4][U_COLS].to_numpy(dtype=np.float64).T)
    [strip] = _strips(fig, u.shape[1])
    np.testing.assert_array_equal(_strip_array(strip), u)


def test_replay_shows_only_the_hours_complete_after_k_steps(tmp_path):
    """Issue #51: after k steps the replay shows k // 4 complete hours. With counters of one row at
    k = 26 steps (6.5 hours of steps), the final frame's climate lines hold hours 0-5 only
    (t = 3, 7, ..., 23, as _hourly) and the strip shows its first 6 columns, the rest masked.
    ceil(26 / 4) = 7 would draw hour 6, whose end (row 27) comes after step 26. fps=1 keeps the
    GIF to 3 frames."""
    import greenleaf_control.media as media

    b = _baseline("rb_default")
    hourly = _hourly(b["frame"])

    fig = media.season_replay(
        b["frame"],
        _counters(b, [26]),
        tmp_path / "season_replay.gif",
        title=TITLE,
        footer=FOOTER,
        fps=1,
    )

    lines = fig.findobj(Line2D)
    for c in metrics.CONSTRAINTS:
        want = (hourly["t"].astype(float).tolist()[:6], hourly[c].tolist()[:6])
        assert [line for line in lines if _xy(line) == want], f"{c}: not hours 0-5 of _hourly"
    [strip] = _strips(fig, 24)
    shown = _strip_array(strip)
    np.testing.assert_array_equal(shown[:, :6], hourly[U_COLS].to_numpy(dtype=np.float64).T[:, :6])
    assert np.isnan(shown[:, 6:]).all(), "hours 6-23 are shown after 26 steps"


@pytest.mark.parametrize("controller", ["rb_default", "zero_controls"])
def test_replay_counters_end_on_the_seasons_metrics(controller, media_dir):
    """Done when: the counters' final values equal metrics' profit_eur_m2 and viol_hours.

    The final frame's text shows the counters' last row, here the whole day (k = 96), which is the
    season's metrics: profit on a line with "€/m²", each constraint's hours on a line with its
    label, all at 2 decimals. No total of the three hours (one hour out of two bands would count
    twice; non-vacuous: at least two of them are nonzero, so the total is none of them)."""
    fig, _ = _replay(controller, media_dir)

    result = _baseline(controller)["result"]
    lines = _lines_of(_texts(fig))
    _assert_numbers(lines, result, "replay")
    hours = [result[f"viol_hours_{c}"] for c in metrics.CONSTRAINTS]
    assert sum(h > 0 for h in hours) >= 2
    total = sum(hours)
    assert total not in hours
    assert not any(_has(ln, total) for ln in lines), f"a total of the hours, {_fmt(total)}"


@pytest.mark.parametrize("controller", ["rb_default", "zero_controls"])
def test_replay_carries_its_title_and_footer_and_writes_only_a_gif(controller, media_dir):
    """The title and footer passed in are each one text on the figure, as given; no text has an
    em dash; the GIF is written (GIF8 header) and, with out_mp4=None, no MP4."""
    fig, out = _replay(controller, media_dir)

    texts = [t.get_text() for t in _texts(fig)]
    assert TITLE in texts
    assert FOOTER in texts
    assert not [t for t in texts if "—" in t]
    assert out.read_bytes()[:4] == b"GIF8"
    assert sorted(p.name for p in out.parent.iterdir()) == ["season_replay.gif"]


def test_replay_draws_its_subtitle(tmp_path):
    """season_replay takes a keyword subtitle (None by default) and draws it as one text, as
    given: the line under the title that names the controller and the simulator."""
    import greenleaf_control.media as media

    params = inspect.signature(media.season_replay).parameters
    assert "subtitle" in params, "season_replay has no subtitle keyword"
    b = _baseline("rb_default")

    fig = media.season_replay(
        b["frame"],
        _counters(b, ENDS),
        tmp_path / "season_replay.gif",
        title=TITLE,
        subtitle=SUBTITLE,
        footer=FOOTER,
    )

    texts = [t.get_text() for t in _texts(fig)]
    assert SUBTITLE in texts
    assert TITLE in texts


# --- baselines_card ----------------------------------------------------------------------------


def test_card_has_one_column_per_baseline_with_its_own_numbers_and_strip(media_dir):
    """Three strips side by side, left to right rb_default, zero_controls, random: each the
    controller's own hourly mean actuators, as _hourly (6 x 24 for one day; issue #51 moved them
    from frame.iloc[::4]'s instants, which differ for rb_default and random). The texts in each
    strip's column print that controller's result: profit on a line with "€/m²", each
    constraint's hours on a line with its label, fruit growth (yield_kg_m2, kg/m²) on a line with
    "fruit growth", all at 2 decimals. Non-vacuous: the three profits differ at 2 decimals."""
    fig, out = _card(media_dir)

    assert isinstance(fig, Figure)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    seasons = [_baseline(c) for c in CONTROLLERS]
    assert len({_fmt(s["result"]["profit_eur_m2"]) for s in seasons}) == 3
    strips, columns = _columns(fig, _strips(fig, len(_hourly(seasons[0]["frame"]))))
    assert len(strips) == 3
    for c, s, strip, lines in zip(CONTROLLERS, seasons, strips, columns, strict=True):
        hourly = _hourly(s["frame"])
        np.testing.assert_array_equal(
            _strip_array(strip), hourly[U_COLS].to_numpy(dtype=np.float64).T, err_msg=c
        )
        _assert_numbers(lines, s["result"], c)
        growth = s["result"]["yield_kg_m2"]
        assert any("fruit growth" in ln and _has(ln, growth) for ln in lines), (c, _fmt(growth))


def test_card_labels_say_what_the_numbers_are(media_dir):
    """rb_default's column names it "upstream's default rule controller (the grower's rules)"
    (line breaks allowed); the footer passed in is one text, as given; no text says "yield" (it is
    fruit growth, not a harvest) or has an em dash."""
    fig, _ = _card(media_dir)

    seasons = [_baseline(c) for c in CONTROLLERS]
    strips, columns = _columns(fig, _strips(fig, len(_hourly(seasons[0]["frame"]))))
    assert len(strips) == 3
    assert RB_LABEL in " ".join(" ".join(columns[0]).split())
    texts = [t.get_text() for t in _texts(fig)]
    assert FOOTER in texts
    assert not [t for t in texts if "yield" in t.lower()]
    assert not [t for t in texts if "—" in t]


def test_card_says_what_profit_counts(media_dir):
    """One text on the card says what profit is (metrics' profit_eur_m2: fruit growth x 1.6 €/kg
    minus the heat, lamp power and CO₂ dosing costs): it contains "fruit value minus" and names
    heat, power and CO₂ (line breaks allowed)."""
    fig, _ = _card(media_dir)

    texts = [" ".join(t.get_text().split()) for t in _texts(fig)]
    says = [t for t in texts if "fruit value minus" in t]
    assert says, "no text says profit is fruit value minus the costs"
    assert any(all(word in t for word in ("heat", "power", "CO₂")) for t in says), says


def _share_images(fig: Figure, n_days: int) -> list:
    """Every image or mesh on fig holding a 3-row (constraints) x n_days array, left to right."""
    found = fig.findobj(lambda a: isinstance(a, (AxesImage, QuadMesh)))
    shares = [a for a in found if np.shape(a.get_array()) == (3, n_days)]
    return sorted(shares, key=lambda a: a.axes.get_position().x0)


def test_card_share_of_each_day_outside_the_band_is_the_graders_hours(media_dir):
    """The card's per-day share of steps outside each safe range agrees with the grader: on a
    1-day season each column's 3 x 1 array (rows co2, temp, rh), times 24 h, is that controller's
    viol_hours_<c>. E.g. a day with 51 of its 96 steps outside the RH band: 51 / 96 x 24 h =
    12.75 h = 51 x 0.25 h. Columns left to right: rb_default, zero_controls, random.
    Non-vacuous: some share is strictly between 0 and 1."""
    fig, _ = _card(media_dir)

    shares = _share_images(fig, 1)
    assert len(shares) == 3
    seen = []
    for c, image in zip(CONTROLLERS, shares, strict=True):
        share = _strip_array(image)
        result = _baseline(c)["result"]
        for j, k in enumerate(metrics.CONSTRAINTS):
            # share is the day's out-of-band steps / 96; x 24 h = out-of-band steps x 0.25 h
            want = result[f"viol_hours_{k}"]
            assert share[j, 0] * 24 == pytest.approx(want, abs=1e-9), (c, k)
            seen.append(share[j, 0])
    assert any(0 < s < 1 for s in seen)


# --- footer ------------------------------------------------------------------------------------


@pytest.mark.parametrize("dirty", [True, False], ids=["dirty", "clean"])
def test_footer_names_source_scenario_simulator_and_commit(dirty):
    """Done when: every footer names its source and commit.

    footer("rb_default", stamp) for a stamp of commit SHA: the source, "PAPER (Amsterdam 2010)",
    "GreenLight-Gym @2d3febb" (the installed gl-gym's commit, from manifest._versions(), to 7
    characters), "commit 0123456", "(dirty)" exactly when the stamp is dirty, and "simulator run,
    not a registered result". No em dash."""
    import greenleaf_control.media as media

    stamp = {"commit": SHA, "dirty": dirty, "fingerprints": {}, "weather_hashes": {}}

    text = media.footer("rb_default", stamp)

    gl7 = manifest._versions()["gl-gym"][:7]
    assert gl7 == "2d3febb"  # the pinned simulator (AGENTS.md, gl_gym @2d3febb)
    for part in (
        "rb_default",
        "PAPER (Amsterdam 2010)",
        f"GreenLight-Gym @{gl7}",
        f"commit {SHA[:7]}",
        "simulator run, not a registered result",
    ):
        assert part in text, (part, text)
    assert ("(dirty)" in text) is dirty, text
    assert "—" not in text


def test_footer_reads_the_simulator_commit_from_the_install(monkeypatch):
    """The simulator's commit comes from manifest._versions() at call time, not a hard-coded
    2d3febb: with _versions patched to report gl-gym feedfacecafe..., the footer says
    "GreenLight-Gym @feedfac" and not 2d3febb."""
    import greenleaf_control.media as media

    real = manifest._versions()
    monkeypatch.setattr(
        manifest, "_versions", lambda: {**real, "gl-gym": "feedfacecafebeef" + "0" * 24}
    )
    stamp = {"commit": SHA, "dirty": False, "fingerprints": {}, "weather_hashes": {}}

    text = media.footer("zero_controls", stamp)

    assert "GreenLight-Gym @feedfac" in text, text
    assert "2d3febb" not in text, text


def test_footer_takes_a_place_and_a_note():
    """footer(source, stamp, *, place, note): the card's footer swaps PAPER for its place and the
    simulator-run note for its own, keeping the source and both commits. Called with neither, it
    is today's PAPER footer: footer(s, stamp) == footer(s, stamp, place=PAPER_NAME, note="A
    simulator run, not a registered result")."""
    import greenleaf_control.media as media

    params = inspect.signature(media.footer).parameters
    assert {"place", "note"} <= set(params), f"footer has no place and note keywords: {params}"
    stamp = {"commit": SHA, "dirty": False, "fingerprints": {}, "weather_hashes": {}}
    source = "result-0123abcd, decision-4567cdef and rb_tuned_validation.parquet"

    text = " ".join(media.footer(source, stamp, place=FOOTER_PLACE, note=CARD_NOTE).split())

    for part in (source, FOOTER_PLACE, CARD_NOTE, f"commit {SHA[:7]}", "GreenLight-Gym @"):
        assert part in text, (part, text)
    assert PAPER_NOTE not in text and "PAPER" not in text, text
    assert "—" not in text
    assert media.footer("rb_default", stamp) == media.footer(
        "rb_default", stamp, place=media.PAPER_NAME, note=PAPER_NOTE
    )


# --- The trade-off still: TRADEOFF, tradeoff_points and rb_tuned_reference (issue #50) ----------


def test_tradeoff_names_the_issues_rows_never_numbers():
    """TRADEOFF holds the issue's four (result_id, arm, status) entries (E7's lag_b100 may join
    them later) and only ids and statuses: each a result-<8 hex> id, an arm name and adopted or
    dropped. E1_RESULT is rb_tuned's E1 row, result-525a6e90."""
    media = _media_card()

    entries = [tuple(e) for e in media.TRADEOFF]

    assert [e for e in ISSUE_TRADEOFF if e not in entries] == [], entries
    for result, arm, status in entries:
        assert re.fullmatch(r"result-[0-9a-f]{8}", result), result
        assert isinstance(arm, str) and arm, arm
        assert status in ledger.DECISIONS, status
    assert media.E1_RESULT == "result-525a6e90"


def test_tradeoff_points_read_each_entrys_rows(sources):
    """Done when: numbers are read from the ledger when the card is drawn.

    On the synthetic ledger, tradeoff_points(path) gives one point per TRADEOFF entry, in order,
    holding the entry's arm and status, its result, decision and registered row ids, the
    registered row's seeds, and from the result's per_arm[arm]: profit and each viol_hours_<c> as
    (iqm, ci_low, ci_high) and yield_kg_m2 as the guard's iqm, all exactly. E.g. ppo's profit
    (2.40, 2.33, 2.46) €/m² over seeds 0-4. Non-vacuous: E4 also holds a residual_w05 (profit
    0.63), but the card's residual_w05 is E6's (0.77, seeds 100-109); and residual_w4's status
    comes from TRADEOFF (dropped), not from E4's one decision row (adopted)."""
    media = _media_card()
    src = sources

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(media, "TRADEOFF", src["tradeoff"])
        got = media.tradeoff_points(src["ledger"])

    want = _want_points(src)
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert _json({key: g.get(key) for key in w}) == _json(w), w["arm"]


@pytest.mark.parametrize("case", REFUSALS)
def test_tradeoff_points_refuse_a_row_that_cant_be_drawn(case, sources, tmp_path, monkeypatch):
    """Done when: each refusal in step 2 raises.

    tradeoff_points raises ValueError naming the row when a TRADEOFF entry's row is missing
    (missing) or isn't a result (not_a_result: a decision id); its result has no decision row
    (undecided) or two (two_decisions); its registration's scenarios aren't exactly
    ["validation"] (not_validation: ["train_eval"]; the error may name the registered row); the
    arm isn't in per_arm (arm_absent); its iqm is null (iqm_null); a run crashed (crashed: 1);
    invalid_episodes is 1; or a guard the card draws is missing (no_rh_guard, no_yield_guard).
    The flawed entry comes after the four good ones, so every entry is checked."""
    media = _media_card()
    path = tmp_path / "ledger.jsonl"
    tradeoff, named = _refusal(case, sources, path)
    monkeypatch.setattr(media, "TRADEOFF", tradeoff)

    with pytest.raises(ValueError) as excinfo:
        media.tradeoff_points(path)

    assert any(row in str(excinfo.value) for row in named), (named, str(excinfo.value))


@pytest.mark.parametrize("n", [5, 10])
def test_rb_tuned_reference_is_the_tiled_iqm_of_the_parquet_seasons(n, sources, monkeypatch):
    """Done when: rb_tuned's reference equals stats.iqm_ci of the tiled parquet seasons.

    rb_tuned_reference(n, path, parquet) gives profit_eur_m2, each viol_hours_<c> and
    yield_kg_m2 as stats.iqm_ci(np.tile(seasons, (n, 1)))[0], exactly, which are the REF values
    (e.g. n = 10: profit (3 x -3.0 + 10 x 0.6 + 3 x 1.2) / 16 = 0.0375 €/m²), plus n_seeds = n
    and sha256, the parquet file's. Like-for-like, not the plain mean: (-3.0 + 0.6 + 1.2) / 3 =
    -0.40 €/m² differs at 2 decimals."""
    media = _media_card()
    monkeypatch.setattr(media, "E1_RESULT", sources["e1"])

    ref = media.rb_tuned_reference(n, sources["ledger"], sources["parquet"])

    for key, want in REF[n].items():
        assert ref[key] == _tiled(key, n), key
        assert ref[key] == pytest.approx(want, rel=1e-12), key
    assert ref["n_seeds"] == n
    assert ref["sha256"] == hashlib.sha256(sources["parquet"].read_bytes()).hexdigest()
    assert _fmt(ref["profit_eur_m2"]) != _fmt(np.mean(RB_SEASONS["profit_eur_m2"]))


@pytest.mark.parametrize("case", list(PARQUET_MISMATCHES))
def test_rb_tuned_reference_refuses_a_parquet_that_disagrees_with_e1(
    case, sources, tmp_path, monkeypatch
):
    """Done when: a parquet that disagrees with the E1 row raises.

    The parquet's per-season means must equal E1's row: each viol_hours_<c> exactly (one season
    0.25 h off moves the temp mean by 0.0833 h), paper_return within rel 1e-9 (every season
    1e-6 high is rel 1e-6 off). Either raises ValueError naming E1's row or the parquet."""
    media = _media_card()
    monkeypatch.setattr(media, "E1_RESULT", sources["e1"])
    parquet = tmp_path / PARQUET
    _write_parquet(parquet, PARQUET_MISMATCHES[case])

    with pytest.raises(ValueError) as excinfo:
        media.rb_tuned_reference(10, sources["ledger"], parquet)

    assert sources["e1"] in str(excinfo.value) or PARQUET in str(excinfo.value), excinfo.value


def test_rb_tuned_reference_takes_paper_return_within_rel_1e9(sources, tmp_path, monkeypatch):
    """paper_return is checked to rel 1e-9, not exactly: a parquet whose every paper_return is
    rel 1e-11 high (float noise, as a different summation order gives) still matches E1, and its
    reference is the exact parquet's."""
    media = _media_card()
    monkeypatch.setattr(media, "E1_RESULT", sources["e1"])
    parquet = tmp_path / PARQUET
    noisy = tuple(v * (1 + 1e-11) for v in RB_SEASONS["paper_return"])
    _write_parquet(parquet, {**RB_SEASONS, "paper_return": noisy})

    ref = media.rb_tuned_reference(10, sources["ledger"], parquet)

    assert ref["profit_eur_m2"] == _tiled("profit_eur_m2", 10)


# --- The trade-off still: tradeoff_card (issue #50) ---------------------------------------------


def test_card_draws_each_point_at_its_ledger_numbers_with_bars_to_its_cis(media_dir, sources):
    """Done when: in each panel, each marker sits at (guard iqm, profit iqm) and its bars end at
    the ledger's ci_low and ci_high.

    Three panels, left to right viol_hours_co2, _temp and _rh (metrics.CONSTRAINTS), each x label
    saying hours outside the safe range and naming its constraint, sharing one y axis labelled
    profit [€/m²]. In each, every point is one errorbar whose marker is (its hours iqm, its profit
    iqm), with one horizontal bar from the hours' ci_low to ci_high at the profit iqm and one
    vertical bar from the profit's ci_low to ci_high at the hours iqm. E.g. ppo in the temp panel:
    marker (153.25 h, 2.40 €/m²), bars x 140.00-166.50 and y 2.33-2.46. Four errorbars with bars
    per panel: rb_tuned has none."""
    fig, _ = _tradeoff(media_dir)

    panels = _panels(fig)
    assert len(panels) == 3, f"{len(panels)} panels hold errorbars"
    words = {"co2": ("CO₂",), "temp": ("temp",), "rh": ("RH", "humidity")}
    for c, ax in zip(metrics.CONSTRAINTS, panels, strict=True):
        titles = [ax.get_title(loc) for loc in ("left", "center", "right")]
        label = " ".join(" ".join([ax.get_xlabel(), *titles]).split())
        assert "hours outside" in label.lower() and any(w in label for w in words[c]), (c, label)
        assert ax is panels[0] or panels[0].get_shared_y_axes().joined(panels[0], ax), c
        assert sum(bool(_segments(e)) for e in _errorbars(ax)) == 4, c
        for p in _want_points(sources):
            hours, profit = p[f"viol_hours_{c}"], p["profit"]
            where = (c, p["arm"])
            found = _errorbar_at(ax, hours[0], profit[0])
            assert len(found) == 1, where
            segments = _segments(found[0])
            assert len(segments) == 2, where
            [[(x0, y0), (x1, y1)]] = [s for s in segments if s[0][1] == s[1][1]]
            assert (sorted([x0, x1]), y0) == (pytest.approx(list(hours[1:])), profit[0]), where
            [[(x0, y0), (x1, y1)]] = [s for s in segments if s[0][0] == s[1][0]]
            assert (x0, sorted([y0, y1])) == (hours[0], pytest.approx(list(profit[1:]))), where
    ylabel = " ".join(panels[0].get_ylabel().split())
    assert "profit" in ylabel and "€/m²" in ylabel, ylabel


def test_card_marks_the_dropped_arm_and_rb_tuned_apart(media_dir, sources):
    """In every panel: the adopted arms' markers are filled, residual_w05's LEAF; the dropped
    arm's (residual_w4) is hollow with a MUTED edge. rb_tuned is one hollow marker, of a shape
    no arm uses, at its largest-n (10-seed) reference (hours and profit to rel 1e-9; profit
    0.0375 €/m²), with no bars; nothing is drawn at the 5-seed reference."""
    media = _media_card()
    fig, _ = _tradeoff(media_dir)

    panels = _panels(fig)
    assert len(panels) == 3
    for c, ax in zip(metrics.CONSTRAINTS, panels, strict=True):
        markers = {}
        for p in _want_points(sources):
            found = _errorbar_at(ax, p[f"viol_hours_{c}"][0], p["profit"][0])
            assert len(found) == 1, (c, p["arm"])
            markers[p["arm"]] = found[0].lines[0]
        for arm, m in markers.items():
            hollow = to_rgba(m.get_markerfacecolor())[3] == 0
            assert hollow is (arm == DROPPED), (c, arm, m.get_markerfacecolor())
        assert to_hex(markers[DROPPED].get_markeredgecolor()) == media.MUTED, c
        assert to_hex(markers["residual_w05"].get_markerfacecolor()) == media.LEAF, c

        rb = _lines_near(ax, _tiled(f"viol_hours_{c}", 10), _tiled("profit_eur_m2", 10))
        assert len(rb) == 1, (c, "rb_tuned's 10-seed marker")
        assert to_rgba(rb[0].get_markerfacecolor())[3] == 0, (c, "rb_tuned is hollow")
        shape = rb[0].get_marker()
        assert shape not in (None, "", " ", "None", "none"), c
        assert shape not in {m.get_marker() for m in markers.values()}, (c, shape)
        assert not [e for e in _errorbars(ax) if e.lines[0] is rb[0] and _segments(e)], c
        assert not _lines_near(ax, _tiled(f"viol_hours_{c}", 5), _tiled("profit_eur_m2", 5)), c


def test_card_reads_the_ledger_when_drawn(sources, tmp_path):
    """Done when: editing a number in the tmp ledger moves the drawn point.

    With E6's residual_w05 profit iqm edited from 0.77 to 0.81 €/m² and its viol_hours_temp iqm
    from 48.50 to 51.25 h (each still inside its CI, 0.70-0.84 and 42.00-55.75), the temp panel's
    residual_w05 marker sits at (51.25, 0.81) and none at (48.50, 0.77)."""
    media = _media_card()
    path = tmp_path / "ledger.jsonl"
    e6 = sources["rows"]["E6"]["result"]
    _edited(sources["ledger"], path, e6, "residual_w05", profit=0.81, viol_hours_temp=51.25)

    fig = _draw_tradeoff(media, sources, tmp_path / "tradeoff.png", ledger_path=path)

    panels = _panels(fig)
    assert len(panels) == 3
    temp = panels[metrics.CONSTRAINTS.index("temp")]
    assert len(_errorbar_at(temp, 51.25, 0.81)) == 1
    assert _errorbar_at(temp, 48.5, 0.77) == []


def test_card_png_is_linkedins_1200_by_627(media_dir):
    """Done when: the PNG is exactly 1200 x 627 px (PIL): 12 x 6.27 in at 100 dpi."""
    _, out = _tradeoff(media_dir)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(out) as im:
        assert im.size == (1200, 627)


def test_card_legend_names_each_points_result_outside_the_panels(media_dir, sources):
    """Done when: every legend line names its result id.

    One legend entry per point plus rb_tuned's. Each point's names its arm and result id, "<n>
    seeds × 3 VALIDATION seasons" (5 for ppo, residual_w1 and residual_w4; 10 for residual_w05),
    its profit IQM and CI at 2 decimals, and "fruit growth" with its yield_kg_m2; "dropped" is on
    residual_w4's only. rb_tuned's names it, E1's result id and its 10-seed profit (0.04). No
    legend overlaps a panel."""
    fig, _ = _tradeoff(media_dir)

    legends, texts = _legend_texts(fig)
    points = _want_points(sources)
    assert len(texts) == len(points) + 1, texts
    for p in points:
        mine = [t for t in texts if p["arm"] in t and p["result"] in t]
        assert len(mine) == 1, (p["arm"], texts)
        line = mine[0]
        assert f"{len(p['seeds'])} seeds × 3 VALIDATION seasons" in line, line
        assert all(_has(line, v) for v in p["profit"]), (p["profit"], line)
        assert "fruit growth" in line and _has(line, p["yield_kg_m2"]), line
        assert ("dropped" in line) is (p["status"] == "dropped"), line
    rb = [t for t in texts if "rb_tuned" in t]
    assert len(rb) == 1, texts
    assert sources["e1"] in rb[0] and _has(rb[0], _tiled("profit_eur_m2", 10)), rb[0]
    assert "dropped" not in rb[0]

    FigureCanvasAgg(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for lg in legends:
        box = lg.get_window_extent(renderer)
        assert not [ax for ax in _panels(fig) if box.overlaps(ax.get_window_extent(renderer))]


def test_card_says_validation_and_how_it_was_scored(media_dir):
    """Above the panels: a title saying VALIDATION and Amsterdam 2010 (scenarios.VALIDATION's
    place); the line "IQM over seeds × seasons, 95% stratified bootstrap CI"; and a note giving
    rb_tuned's profit at the points' other seed count, 5: "5-seed" with -0.07 €/m² (REF[5],
    -0.0667) on one text."""
    fig, _ = _tradeoff(media_dir)

    texts = [" ".join(t.get_text().split()) for t in _texts(fig)]

    assert [t for t in texts if "VALIDATION" in t and VAL_PLACE in t], texts
    assert [t for t in texts if CI_LINE in t], texts
    ref5 = _tiled("profit_eur_m2", 5)
    assert [t for t in texts if "5-seed" in t and _has(t, ref5) and "€/m²" in t], texts


def test_card_keeps_medias_text_rules(media_dir, sources):
    """Done when: no em dash, no "yield" and no summed hours appear in any text.

    The footer passed in is one text, as given. No text has an em dash or says "yield" (it is
    fruit growth, not a harvest), and no line prints a point's three hour IQMs summed: ppo
    0.5 + 153.25 + 135 = 288.75 h, residual_w05 264.25, residual_w1 303.75, residual_w4 296.75,
    rb_tuned at 10 seeds 1.5625 + 92.859375 + 254.578125 = 349.00 and at 5 seeds 350.39.
    Non-vacuous: each has two or more nonzero hours, so its total is none of them."""
    fig, _ = _tradeoff(media_dir)

    texts = [t.get_text() for t in _texts(fig)]
    lines = _lines_of(_texts(fig))
    assert CARD_FOOTER in texts
    assert not [t for t in texts if "—" in t]
    assert not [t for t in texts if "yield" in t.lower()]
    hour_sets = [[p[key][0] for key in HOUR_KEYS] for p in _want_points(sources)]
    hour_sets += [[_tiled(key, n) for key in HOUR_KEYS] for n in (5, 10)]
    for hours in hour_sets:
        total = sum(hours)
        assert sum(h > 0 for h in hours) >= 2 and total not in hours
        assert not [ln for ln in lines if _has(ln, total)], f"a total of the hours, {_fmt(total)}"


# --- _hourly: export_replay's hourly convention (issue #51) -------------------------------------


@pytest.mark.parametrize("controller", ["rb_default", "random"])
def test_hourly_is_the_end_of_hour_climate_and_the_hours_mean_actuators(controller):
    """Done when: each column draws the frame's row 4h + 3 climate, and its strip is the hourly
    means.

    media._hourly(frame) of a 1-day PAPER season has 24 rows: t, co2, temp and rh are the frame's
    rows 3, 7, ..., 95 (frame.iloc[3::4]) exactly, and each u_* is the mean of rows 4h..4h+3
    (reshape(24, 4, 6).mean(axis=1), as export_replay computes it) exactly. Non-vacuous: those rows
    differ from frame.iloc[::4], and rb_default's and random's means from their instants."""
    media = _media_has("_hourly")
    frame = _baseline(controller)["frame"]
    want = _hourly(frame)

    got = media._hourly(frame)

    assert len(got) == 24
    for c in ("t", *metrics.CONSTRAINTS):
        np.testing.assert_array_equal(
            np.asarray(got[c], dtype=np.float64), frame[c].to_numpy(dtype=np.float64)[3::4], c
        )
    for col in U_COLS:
        np.testing.assert_array_equal(
            np.asarray(got[col], dtype=np.float64), want[col].to_numpy(), col
        )
    assert not np.array_equal(frame["temp"].to_numpy()[3::4], frame["temp"].to_numpy()[::4])
    instants = frame.iloc[::4][U_COLS].to_numpy(dtype=np.float64)
    assert not np.array_equal(want[U_COLS].to_numpy(), instants)


def test_hourly_rounded_as_export_replay_rounds_is_its_hourly_json():
    """Done when: those series, rounded as export_replay rounds, equal
    export_replay.export(...)["hourly"] for the same season.

    On rb_default's 1-day PAPER season, media._hourly's co2, temp and rh to 3 significant digits
    (float(f"{v:.3g}")) and each u_* to 3 decimals equal export's hourly lists exactly, so the
    README's pictures and the site draw the same series."""
    import greenleaf_control.export_replay as export_replay

    media = _media_has("_hourly")
    b = _baseline("rb_default")
    stamp = {"commit": SHA, "dirty": False, "gl_gym": manifest._versions()["gl-gym"]}
    payload = export_replay.export(
        b["frame"],
        b["result"],
        b["states"],
        b["controls"],
        b["params"],
        controller="rb_default",
        scenario=PAPER,
        stamp=stamp,
        split="paper",
        run=None,
    )

    got = media._hourly(b["frame"])

    for c in metrics.CONSTRAINTS:
        rounded = [float(f"{v:.3g}") for v in np.asarray(got[c], dtype=np.float64)]
        assert rounded == payload["hourly"][c], c
    for col in U_COLS:
        rounded = [round(float(v), 3) for v in np.asarray(got[col], dtype=np.float64)]
        assert rounded == payload["hourly"][col], col


# --- The climate stripes: constants and the colour scale (issue #51) ---------------------------


def test_replay_constants_are_the_issues():
    """SEASON is scenarios.VALIDATION[1] (Amsterdam 2010 from day 59, the same dict as PAPER: the
    issue fixed it as the middle season, not by its outcome). The Instrument colours are Aarmen's:
    SITE_BG #F2F3F0, SITE_INK #111311, SITE_MUTED #6A6F68. TEMP_SCALE is (10, 36) °C. GROTESK and
    MONO end on matplotlib's bundled DejaVu Sans and DejaVu Sans Mono, so they render anywhere
    (which font loads is the machine's; only the requested list is checked)."""
    media = _media_has()

    assert media.SEASON == VALIDATION[1] == PAPER
    assert media.SEASON == {"location": "Amsterdam", "growth_year": 2010, "start_day": 59}
    colours = (media.SITE_BG, media.SITE_INK, media.SITE_MUTED)
    assert tuple(to_hex(c) for c in colours) == ("#f2f3f0", "#111311", "#6a6f68")
    assert tuple(media.TEMP_SCALE) == (10.0, 36.0)
    assert list(media.GROTESK)[-1] == "DejaVu Sans"
    assert list(media.MONO)[-1] == "DejaVu Sans Mono"


def test_temp_cmap_is_26_one_degree_steps_from_blue_to_rust_and_never_the_page():
    """TEMP_CMAP is quantised to 1 °C steps so the GIF's palette doesn't shimmer: N = 26 over
    TEMP_SCALE's 10-36 °C (26 °C / 26 = 1 °C a colour), 1,000 samples across it give exactly 26
    colours, and on Normalize(10, 36) every T + 0.25 °C shares T + 0.75 °C's colour while
    T + 0.5 and T + 1.5 differ (integer bin edges). It runs blue at 10 °C (blue > red) to rust at
    36 °C (red > blue). No colour in it is within 8/255 of SITE_BG in every channel: its neutral
    middle is a distinct light grey, not the page (a cell at 23 °C must not vanish)."""
    media = _media_has()
    cmap = media.TEMP_CMAP
    norm = Normalize(*media.TEMP_SCALE)

    assert cmap.N == 26
    assert len({to_hex(cmap(x)) for x in np.linspace(0.0, 1.0, 1000)}) == 26
    for t in range(10, 36):
        assert to_hex(cmap(norm(t + 0.25))) == to_hex(cmap(norm(t + 0.75))), t
        if t < 35:
            assert to_hex(cmap(norm(t + 0.5))) != to_hex(cmap(norm(t + 1.5))), t
    low, high = to_rgb(cmap(0.0)), to_rgb(cmap(1.0))
    assert low[2] > low[0], f"10 °C isn't blue: {to_hex(low)}"
    assert high[0] > high[2], f"36 °C isn't rust: {to_hex(high)}"
    page = np.array(to_rgb(media.SITE_BG))
    for i in range(cmap.N):
        colour = np.array(to_rgb(cmap(i)))
        assert np.abs(colour - page).max() >= 8 / 255, f"colour {i} {to_hex(colour)} is the page"


# --- replay_pick: the run the replay plays (issue #51) ------------------------------------------


@pytest.mark.parametrize(
    ("offsets", "seed"), [(OFFSETS, 107), (TIED, 103)], ids=["lower_median", "tie_to_lower_seed"]
)
def test_replay_pick_is_the_lower_median_seed_at_its_selected_checkpoint(
    offsets, seed, tmp_path, monkeypatch
):
    """Done when: main plays the lower-median seed, ties going to the lower seed, at the ledger's
    selected checkpoint (not final).

    On the E6-shaped tmp ledger (10 seeds), replay_pick(result, path, parquet) sorts the scored
    runs by (recorded SEASON profit, seed) and takes index (10 - 1) // 2 = 4:
    - lower_median: seed 107 (OFFSETS; the upper median is 103, the best 109; ranking on day 45 or
      75, on the mean, or on the other checkpoint's candidate gives 103 or 104);
    - tie_to_lower_seed: seeds 103 and 107 tie at index 4 and 5, so 103.
    It returns the run's row id, registration, result, arm, seed, checkpoint (CHECKPOINTS: 1750000
    for 107, 500000 for 103, not "final"), commit (its manifest's RUN_COMMIT), run_dir, the
    decision row's id, record (per_scenario[1] of its selected checkpoint: RECORD at its offset)
    and rb_record (the parquet's day-59 row, RB_RECORD), and plays no season."""
    media = _media_has()
    src = _replay_sources(tmp_path / "replay", offsets=offsets)
    _no_replay_played(monkeypatch, media)
    run, run_dir = src["runs"][seed]

    pick = media.replay_pick(src["result"], src["ledger"], src["parquet"])

    assert (pick["id"], pick["seed"], pick["checkpoint"]) == (
        run["row_id"],
        seed,
        CHECKPOINTS[seed],
    )
    assert pick["checkpoint"] != "final"
    assert (pick["registered"], pick["result"]) == (src["registered"], src["result"])
    assert (pick["arm"], pick["commit"]) == (ARM, RUN_COMMIT)
    assert pick["decision"] == src["decision"]
    assert Path(pick["run_dir"]).resolve() == run_dir.resolve()
    want = {**RECORD, "profit_eur_m2": RECORD["profit_eur_m2"] + offsets[seed]}
    assert _keys(pick["record"]) == want
    assert _keys(pick["rb_record"]) == RB_RECORD


@pytest.mark.parametrize("case", PICK_REFUSALS)
def test_replay_pick_refuses_a_row_it_cant_replay_before_any_season(case, tmp_path, monkeypatch):
    """Done when: every refusal in step 3 happens before any season plays.

    replay_pick raises ValueError naming the row when: the id isn't a result row (not_a_result: a
    decision id); the registration isn't on exactly ["validation"] (not_validation:
    ["train_eval"]); there is no decision row (undecided), it dropped the result (dropped), or
    there are two (two_decisions); the registration has two arms (two_arms); a scored run's dir is
    gone (run_dir_missing); or a run's selected.json was made on [PAPER] (selected_not_validation)
    or in 30-day seasons (selected_not_60_days), or names "final" where the result row selected
    "1500000" (checkpoint_disagrees); or the parquet has no day-59 row (no_season_row). A run's
    flaw is seed 100's, never the pick's (107), so every run is checked. The error names the row
    (or the run's dir, or the parquet); every way to play a season fails if called."""
    media = _media_has()
    src = _replay_sources(tmp_path / "replay", case)
    _no_replay_played(monkeypatch, media)

    with pytest.raises(ValueError) as excinfo:
        media.replay_pick(src["replay"], src["ledger"], src["parquet"])

    assert any(name in str(excinfo.value) for name in src["named"]), (src["named"], excinfo.value)


# --- _stripes and duel_replay: the climate stripes (issue #51) -----------------------------------


def test_stripes_draw_each_panels_hourly_temperature_grid_day_by_day():
    """Done when: each panel is the season's hourly air temperature as a colour grid, filling in
    day by day.

    _stripes(panels, (12.8, 7.2), ...) of two 2-day SEASON panels returns (Figure, draw). The
    Figure is 12.8 x 7.2 in on SITE_BG. After draw(1), the two 24 x 2 images, top to bottom, are
    rb_tuned's ("Tuned rules") and the other's grids (_grid: _hourly temp, row = hour, column =
    day) with day 1 shown and day 2 masked; after draw(2), both grids exactly, nothing masked. Both
    are on TEMP_CMAP with clim (10, 36) °C (the same scale in both) and nearest-neighbour cells.
    Non-vacuous: the two grids differ, each varies over the day, and it isn't frame.iloc[::4]'s."""
    media = _media_has()
    panels = _stripe_panels(2)
    grids = [_grid(p["frame"]) for p in panels]
    assert not np.array_equal(grids[0], grids[1])
    assert all(np.ptp(g[:, 0]) > 1.0 for g in grids)  # the day's swing, over 1 °C
    start = panels[0]["frame"]["temp"].to_numpy(dtype=np.float64)[::4].reshape(2, 24).T
    assert not np.array_equal(grids[0], start)

    fig, draw = media._stripes(
        panels, (12.8, 7.2), title=STRIPES_TITLE, caption=CAPTION, footer=REPLAY_FOOTER
    )

    assert isinstance(fig, Figure)
    assert tuple(fig.get_size_inches()) == pytest.approx((12.8, 7.2))
    assert to_hex(fig.get_facecolor()) == to_hex(media.SITE_BG)
    draw(1)
    images = _temp_images(fig, 2)
    assert len(images) == 2, f"{len(images)} images hold a 24 x 2 grid"
    for image, grid, p in zip(images, grids, panels, strict=True):
        shown = _strip_array(image)
        np.testing.assert_array_equal(shown[:, 0], grid[:, 0], err_msg=p["heading"])
        assert np.isnan(shown[:, 1]).all(), f"{p['heading']}: day 2 shows after draw(1)"
    draw(2)
    for image, grid, p in zip(_temp_images(fig, 2), grids, panels, strict=True):
        np.testing.assert_array_equal(_strip_array(image), grid, err_msg=p["heading"])
        assert tuple(image.get_clim()) == (10.0, 36.0), p["heading"]
        assert image.get_cmap().name == media.TEMP_CMAP.name, p["heading"]
        assert image.get_cmap().N == 26, p["heading"]
        assert image.get_interpolation() in ("nearest", "none"), p["heading"]


@pytest.mark.parametrize("day", [1, 2])
def test_stripes_totals_and_day_counter_tick_with_the_days(day):
    """Done when: each panel's counters end on its season's metrics, and the day and counters
    advance together.

    After draw(d), each panel prints its counters' row d - 1 (the first d days): profit on a line
    with "€/m²" and each constraint's hours on a line with its label, at 2 decimals; and one text
    says "day d / 2". Each total sits under its own panel: where the two panels' values differ at
    2 decimals, every text printing the top panel's is above every text printing the bottom
    one's, and "Tuned rules" is above "Rules + co-pilot". After draw(1) neither season's final
    profit shows. No line prints a panel's three hours summed (e.g. rb_tuned day 2:
    0 + 3.25 + 3.50 = 6.75 h). Day 1: rb_tuned -0.12 €/m² (temp 2.50 h, RH 1.50 h), rb_default
    -0.44 €/m² (temp 2.00 h, RH 12.25 h); day 2: -0.17 and -0.73 €/m² (the counters' rows,
    computed by metrics)."""
    media = _media_has()
    panels = _stripe_panels(2)
    fig, draw = media._stripes(
        panels, (12.8, 7.2), title=STRIPES_TITLE, caption=CAPTION, footer=REPLAY_FOOTER
    )

    draw(day)

    lines = _lines_of(_texts(fig))
    rows = [p["counters"].iloc[day - 1].to_dict() for p in panels]
    for p, row in zip(panels, rows, strict=True):
        _assert_numbers(lines, row, p["heading"])
        hours = [row[f"viol_hours_{c}"] for c in metrics.CONSTRAINTS]
        assert sum(h > 0 for h in hours) >= 2 and sum(hours) not in hours
        assert not [ln for ln in lines if _has(ln, sum(hours))], f"a total, {_fmt(sum(hours))}"
    assert [ln for ln in lines if re.search(rf"day\s*{day}\s*/\s*2(?!\d)", ln)], lines
    if day == 1:
        for p in panels:
            final = p["counters"].iloc[-1]["profit_eur_m2"]
            assert not [ln for ln in lines if _has(ln, final)], (p["heading"], _fmt(final))

    renderer = _rendered(fig)
    labels = {"profit_eur_m2": "€/m²", **{f"viol_hours_{c}": LABELS[c] for c in LABELS}}
    for key, label in labels.items():
        top, bottom = rows[0][key], rows[1][key]
        if _fmt(top) == _fmt(bottom):
            continue
        ys = [[_y(t, renderer) for t in _printing(fig, label, v)] for v in (top, bottom)]
        assert ys[0] and ys[1], (key, ys)
        assert min(ys[0]) > max(ys[1]), f"{key}: the top panel's total isn't above the bottom's"
    heads = [
        [_y(t, renderer) for t in _texts(fig) if p["heading"].lower() in t.get_text().lower()]
        for p in panels
    ]
    assert heads[0] and heads[1] and min(heads[0]) > max(heads[1]), heads


def test_stripes_texts_their_fonts_and_the_hours_down_the_side():
    """The title, caption and footer passed in are each one text, as given. The title is in
    GROTESK; the caption, each total (the texts with "€/m²") and the hour labels in MONO, each set
    on its own text (Text.get_fontfamily() is the requested list). Each panel's y axis is labelled
    by hour, "00" (the top row, drawn above) to "23". No text has an em dash or says "yield".
    Fonts are never set globally: drawing leaves every rcParams value as it was, and media.py
    names no rcParams, rc, rc_context or rcdefaults."""
    media = _media_has()
    before = dict(matplotlib.rcParams)
    fig, draw = media._stripes(
        _stripe_panels(2), (12.8, 7.2), title=STRIPES_TITLE, caption=CAPTION, footer=REPLAY_FOOTER
    )
    draw(2)
    renderer = _rendered(fig)
    changed = [k for k, v in matplotlib.rcParams.items() if before.get(k) != v]
    assert not changed, f"drawing changed rcParams {changed}"
    tree = ast.parse((Path(greenleaf_control.__file__).parent / "media.py").read_text())
    global_rc = {"rcParams", "rc", "rc_context", "rcdefaults"}
    named = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    named |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    named |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not named & global_rc, f"media.py names {sorted(named & global_rc)}"

    texts = _texts(fig)
    by_text = {t.get_text(): t for t in texts}
    for given in (STRIPES_TITLE, CAPTION, REPLAY_FOOTER):
        assert given in by_text, given
    assert _family(by_text[STRIPES_TITLE]) == list(media.GROTESK)
    assert _family(by_text[CAPTION]) == list(media.MONO)
    totals = [t for t in texts if "€/m²" in t.get_text()]
    assert totals and all(_family(t) == list(media.MONO) for t in totals), totals
    images = _temp_images(fig, 2)
    assert len(images) == 2
    for image in images:
        ticks = {t.get_text(): t for t in image.axes.get_yticklabels() if t.get_text()}
        assert {"00", "23"} <= set(ticks), sorted(ticks)
        assert all(_family(t) == list(media.MONO) for t in ticks.values()), sorted(ticks)
        assert _y(ticks["00"], renderer) > _y(ticks["23"], renderer)
    assert not [t.get_text() for t in texts if "—" in t.get_text()]
    assert not [t.get_text() for t in texts if "yield" in t.get_text().lower()]


def test_duel_replay_writes_a_1280x720_gif_and_a_1200x627_hero(media_dir):
    """Done when: hero.png is 1200 x 627 (LinkedIn's size; Aarmen's scope update) and the GIF
    1280 x 720.

    duel_replay(panels, out_gif, None, out_png, ...) writes the GIF (GIF8 header) at 12.8 x 7.2 in
    and 100 dpi, 1280 x 720 px, and the still to out_png, a PNG of exactly 1200 x 627 px (12 x
    6.27 in at 100 dpi; no bbox_inches="tight", which would change it). With out_mp4 None, no
    MP4: the dir holds the GIF and hero.png only."""
    _, out = _duel(media_dir)

    gif, png = out / "duel_replay.gif", out / "hero.png"
    assert gif.read_bytes()[:4] == b"GIF8"
    with Image.open(gif) as im:
        assert im.size == (1280, 720)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(png) as im:
        assert im.size == (1200, 627)
    assert sorted(p.name for p in out.iterdir()) == ["duel_replay.gif", "hero.png"]


def test_duel_replay_fills_in_day_by_day_then_holds_2_s(media_dir):
    """The GIF draws each day d = 1..n DAY_FRAMES times, then holds the last frame 2 s, at
    REPLAY_FPS: its frame durations sum to (n x DAY_FRAMES + 2 x fps) / fps seconds; for the 2-day
    panels at 2 frames a day and 12 fps, (4 + 24) / 12 = 2.33 s. GIF delays are whole centiseconds
    (Pillow may also merge identical frames, summing their delays), so the sum may be off by up to
    10 ms a frame. duel_replay returns a Figure in its final state: both grids whole, nothing
    masked."""
    media = _media_has()
    fig, out = _duel(media_dir)

    fps, n_frames = media.REPLAY_FPS, 2 * media.DAY_FRAMES + 2 * media.REPLAY_FPS
    want = n_frames / fps
    assert _gif_seconds(out / "duel_replay.gif") == pytest.approx(want, abs=0.01 * n_frames)
    assert isinstance(fig, Figure)
    images = _temp_images(fig, 2)
    assert len(images) == 2
    for image, p in zip(images, _stripe_panels(2), strict=True):
        np.testing.assert_array_equal(_strip_array(image), _grid(p["frame"]), err_msg=p["heading"])


# --- the CLI -----------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["no_out", "no_ffmpeg"])
def test_main_refuses_before_any_season(case, sources, tmp_path, monkeypatch, capsys):
    """--out is required; and without ffmpeg (FFMpegWriter.isAvailable() false) main refuses with a
    nonzero SystemExit naming ffmpeg. Both come before any season is played, and nothing is
    written: --out isn't created. The card, if main reads it first, reads the synthetic rows."""
    import greenleaf_control.media as media

    _no_season_played(monkeypatch, media)
    _card_reads(monkeypatch, media, sources)
    out = tmp_path / "media"
    if case == "no_out":
        argv = []
    else:
        monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
        argv = ["--out", str(out)]

    with pytest.raises(SystemExit) as excinfo:
        media.main(argv)

    assert excinfo.value.code not in (0, None)
    err = capsys.readouterr().err
    assert ("--out" if case == "no_out" else "ffmpeg") in err.lower(), err
    assert not out.exists()


def _whole(text: str, n: int) -> bool:
    """text prints n as a whole number (2 in "2 days", not in "2010" or "12")."""
    return re.search(rf"(?<!\d){n}(?!\d)", text) is not None


def _drawing_recorded(monkeypatch, media, *, duel: bool = True) -> dict:
    """season_replay, baselines_card and tradeoff_card (once it exists) replaced by recorders of
    their arguments that draw and write nothing, so main's own choices (its title, subtitle and
    footers) are read fast. With duel, duel_replay (issue #51, once it exists) too, its arguments
    bound to its real signature by name (panels, out_gif, out_mp4, out_png, title, ...)."""
    calls = {}

    def record(name: str):
        def drawn(*args, **kwargs):
            calls[name] = kwargs
            return Figure()

        return drawn

    monkeypatch.setattr(media, "season_replay", record("replay"))
    monkeypatch.setattr(media, "baselines_card", record("card"))
    monkeypatch.setattr(media, "tradeoff_card", record("tradeoff"), raising=False)
    real_duel = getattr(media, "duel_replay", None)
    if duel and real_duel is not None:

        def duel_drawn(*args, **kwargs):
            calls["duel"] = inspect.signature(real_duel).bind(*args, **kwargs).arguments
            return Figure()

        monkeypatch.setattr(media, "duel_replay", duel_drawn)
    return calls


def test_main_titles_the_replay_as_simulated_with_the_runs_days_place_and_controller(
    sources, tmp_path, monkeypatch
):
    """The replay's title says it is simulated, with the season's day count taken from the run and
    PAPER's location and year (Amsterdam 2010, from scenarios.PAPER); its subtitle names
    "upstream's default rule controller" and "GreenLight-Gym". The 8 pt footer alone is unreadable
    on a phone, so a real greenhouse must not be what the title reads as.

    Known answer: with play shortened to real 2-day seasons (192 steps), the title says 2 (days),
    never 60, so the count can't be typed in. Drawing is recorded, not done (the real
    season_replay is tested above), ffmpeg is reported available, and the card reads the
    synthetic rows, so this runs anywhere.
    """
    import greenleaf_control.media as media

    real_play = media.play
    monkeypatch.setattr(
        media, "play", lambda controller, season_length=60: real_play(controller, 2)
    )
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    _card_reads(monkeypatch, media, sources)
    calls = _drawing_recorded(monkeypatch, media)

    assert media.main(["--out", str(tmp_path / "media")]) == 0

    title = calls["replay"]["title"]
    assert "simulated" in title.lower(), title
    assert "day" in title.lower() and _whole(title, 2), title  # 192 steps / 96 a day
    assert not _whole(title, 60), title
    assert PLACE in " ".join(title.split()), title
    subtitle = calls["replay"].get("subtitle") or ""
    assert all(part in " ".join(subtitle.split()) for part in RB_SUBTITLE), subtitle
    assert "—" not in title + subtitle


@pytest.mark.parametrize("invalid", CONTROLLERS)
def test_main_refuses_an_invalid_season_before_writing(
    invalid, sources, tmp_path, monkeypatch, capsys
):
    """A season whose result isn't valid (metrics: a solver failure, or steps != season_steps) is
    never drawn: main exits nonzero naming that controller, and only that one, before anything is
    drawn or written (--out stays absent or empty). play returns the real 1-day seasons, the one
    under test marked invalid with one solver failure; ffmpeg is reported available; the card
    reads the synthetic rows."""
    import greenleaf_control.media as media

    _card_reads(monkeypatch, media, sources)

    def play(controller, season_length=60):
        b = _baseline(controller)
        if controller != invalid:
            return b
        return {**b, "result": {**b["result"], "valid": False, "solver_failures": 1}}

    def trip(name: str):
        def refuse(*args, **kwargs):
            raise AssertionError(
                f"media.{name} was called: an invalid season must be refused first"
            )

        return refuse

    monkeypatch.setattr(media, "play", play)
    monkeypatch.setattr(media, "season_replay", trip("season_replay"))
    monkeypatch.setattr(media, "baselines_card", trip("baselines_card"))
    monkeypatch.setattr(media, "tradeoff_card", trip("tradeoff_card"), raising=False)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    out = tmp_path / "media"

    try:
        code = media.main(["--out", str(out)])
    except SystemExit as e:
        code = e.code
    text = f"{code} {capsys.readouterr().err}"

    assert code not in (0, None), text
    assert invalid in text, text
    assert not [c for c in CONTROLLERS if c != invalid and c in text], text
    assert not out.exists() or not list(out.iterdir())


@pytest.mark.parametrize("case", [*REFUSALS, *PARQUET_MISMATCHES])
def test_main_refuses_a_bad_row_or_parquet_before_any_season(
    case, sources, tmp_path, monkeypatch, capsys
):
    """Done when: each refusal, and a parquet that disagrees with the E1 row, raises before any
    file exists.

    main builds the card's points and references first, so a row tradeoff_points refuses (each of
    REFUSALS, as in its test) or a parquet rb_tuned_reference refuses (each of
    PARQUET_MISMATCHES) stops it before any season is played (every way to play one trips) or
    anything is written: it exits nonzero or raises ValueError, naming the row (or E1's row or
    the parquet), and --out isn't created. ffmpeg is reported available."""
    media = _media_card()
    _no_season_played(monkeypatch, media)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    if case in REFUSALS:
        path = tmp_path / "ledger.jsonl"
        tradeoff, named = _refusal(case, sources, path)
        _card_reads(monkeypatch, media, sources, tradeoff=tradeoff, ledger=path)
    else:
        parquet = tmp_path / PARQUET
        _write_parquet(parquet, PARQUET_MISMATCHES[case])
        named = {sources["e1"], PARQUET}
        _card_reads(monkeypatch, media, sources, parquet=parquet)
    out = tmp_path / "media"

    raised = ""
    try:
        code = media.main(["--out", str(out)])
    except SystemExit as e:
        code = e.code
    except ValueError as e:
        code, raised = 1, str(e)
    text = f"{code} {raised} {capsys.readouterr().err}"

    assert code not in (0, None), text
    assert any(row in text for row in named), (named, text)
    assert not out.exists()


def test_main_writes_the_tradeoff_card_and_its_media_json_entry(sources, tmp_path, monkeypatch):
    """main writes tradeoff.png (1200 x 627 px) and adds its media.json entry; the other entries
    are unchanged.

    play returns the real 1-day seasons; the replay and the baselines card are recorded (they
    write nothing); the real tradeoff_card draws, spied on; the card reads the synthetic rows.
    - The card gets tradeoff_points' points (every number as the rows were written) and one
      reference per distinct seed count among them (5 and 10), and its footer names every
      point's result and decision row, E1's result row, rb_tuned_validation.parquet,
      "VALIDATION, Amsterdam 2010 days 45, 59, 75", "commit <sha7>" and "Recorded VALIDATION
      results; the 2012 test weather is still sealed", and not the simulator-run note.
    - --out holds media.json and tradeoff.png only.
    - media.json's tradeoff.png entry has today's keys plus split "validation" and ledger (every
      point's result and decision id and E1's result id), scenario scenarios.VALIDATION, this
      checkout's commit, and metrics {points, references}, what the card drew.
    - The replay's and the baselines card's entries keep exactly today's keys, scenario PAPER
      and their seasons' numbers."""
    media = _media_card()
    monkeypatch.setattr(media, "play", lambda controller, season_length=60: _baseline(controller))
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    _card_reads(monkeypatch, media, sources)
    real_card = media.tradeoff_card
    calls = _drawing_recorded(monkeypatch, media)

    def card(*args, **kwargs):
        calls["tradeoff"] = inspect.signature(real_card).bind(*args, **kwargs).arguments
        return real_card(*args, **kwargs)

    monkeypatch.setattr(media, "tradeoff_card", card)
    stamp = manifest.code_stamp(smoke=True)
    out = tmp_path / "media"

    assert media.main(["--out", str(out)]) == 0

    drawn = calls["tradeoff"]
    want = _want_points(sources)
    assert _json(
        [{key: p.get(key) for key in w} for p, w in zip(drawn["points"], want, strict=True)]
    ) == _json(want)
    assert sorted(r["n_seeds"] for r in drawn["references"]) == [5, 10]
    foot = " ".join(drawn["footer"].split())
    ids = [p[key] for p in want for key in ("result", "decision")] + [sources["e1"]]
    for part in (*ids, PARQUET, FOOTER_PLACE, f"commit {stamp['commit'][:7]}", CARD_NOTE):
        assert part in foot, (part, foot)
    assert PAPER_NOTE not in foot, foot

    assert sorted(p.name for p in out.iterdir()) == ["media.json", "tradeoff.png"]
    with Image.open(out / "tradeoff.png") as im:
        assert im.size == (1200, 627)

    by_file = {e["file"]: e for e in json.loads((out / "media.json").read_text())}
    assert sorted(by_file) == [
        "baselines.png",
        "season_replay.gif",
        "season_replay.mp4",
        "tradeoff.png",
    ]
    e = by_file["tradeoff.png"]
    assert ENTRY_KEYS | {"split", "ledger"} <= set(e), sorted(e)
    assert (e["split"], e["scenario"]) == ("validation", VALIDATION)
    assert e["commit"] == stamp["commit"] and e["dirty"] is stamp["dirty"]
    assert not [i for i in ids if i not in e["ledger"]], e["ledger"]
    assert e["metrics"] == _json({"points": drawn["points"], "references": drawn["references"]})
    rb = _baseline("rb_default")["result"]
    for name in ("season_replay.gif", "season_replay.mp4", "baselines.png"):
        assert set(by_file[name]) == ENTRY_KEYS, name
        assert by_file[name]["scenario"] == PAPER, name
    assert by_file["season_replay.gif"]["metrics"] == _json(rb)
    assert by_file["baselines.png"]["metrics"] == _json(
        {c: _baseline(c)["result"] for c in CONTROLLERS}
    )


def test_main_replay_plays_the_picked_run_and_names_it_everywhere(sources, tmp_path, monkeypatch):
    """Done when: main plays the lower-median seed at the ledger's selected checkpoint on
    VALIDATION[1], its counters end on each season's metrics, and the footer and media.json name
    the run row, its manifest commit, the result and decision rows, E1, VALIDATION and the commit
    the image was made from.

    main --out OUT --replay <result> on fake data (_replay_world: 1-day seasons, rb_tuned played
    for real, run_policy a spy returning rb_default's 1-day SEASON; every drawing recorded). The
    tmp E6-shaped ledger records seed 107's SEASON as that season and the parquet's day-59 row as
    rb_tuned's 1-day SEASON, so the records match. main returns 0, and:
    - run_policy is called once: seed 107's dir, checkpoint "1750000" (not "final"), SEASON,
      record_steps and return_trace True, allow_test False;
    - duel_replay gets OUT/duel_replay.gif, OUT/duel_replay.mp4 and OUT/hero.png, and two panels:
      "Tuned rules" with rb_tuned's frame and result, then the "co-pilot" one with the spy's; each
      panel's counters are running() at the end of each day (step 96) and end on its result;
    - the title counts the season's days from the run (1, never 60); the title or the caption
      says VALIDATION, Amsterdam 2010, day 59 and "not the final" (how large it is drawn is the
      next test's); the caption names the run row, seed 107, checkpoint 1750000, the scale
      (10, 36) and the safe range (15, 34);
    - the footer names the run row, RUN_COMMIT[:7], the result and decision rows, E1's row,
      "VALIDATION, Amsterdam 2010 day 59", "commit <sha7>" (this checkout's) and the note "A
      replay of a recorded VALIDATION result; the 2012 test weather is still sealed";
    - media.json holds today's 4 entries plus one per REPLAY_FILES file, each with today's keys and
      split "validation", scenario SEASON, ledger {run row, result, decision, E1}, run {seed 107,
      checkpoint "1750000", run_dir, manifest_commit RUN_COMMIT} and metrics {"rb_tuned": its
      result, "residual_w05": the run's}, every RESULT_KEYS number as replayed."""
    media = _media_has()
    rb, run = _graded("rb_tuned", 1), _graded("rb_default", 1)
    src = _replay_sources(
        tmp_path / "replay", record=_record(run["result"]), rb_record=_record(rb["result"])
    )
    calls = _replay_world(monkeypatch, media, sources, src, run)
    stamp = manifest.code_stamp(smoke=True)
    out = tmp_path / "media"
    run_row, run_dir = src["runs"][107]
    rid, e1 = run_row["row_id"], media.E1_RESULT

    assert media.main(["--out", str(out), "--replay", src["result"]]) == 0

    [call] = calls["run_policy"]
    assert Path(call["run_dir"]).resolve() == run_dir.resolve()
    assert (call["checkpoint"], call["scenario"]) == ("1750000", VALIDATION[1])
    assert (call["record_steps"], call["return_trace"], call["allow_test"]) == (True, True, False)

    duel = calls["duel"]
    assert [Path(duel[k]) for k in ("out_gif", "out_mp4", "out_png")] == [
        out / name for name in REPLAY_FILES
    ]
    top, bottom = duel["panels"]
    assert "tuned rules" in top["heading"].lower() and "co-pilot" in bottom["heading"].lower()
    for panel, season in ((top, rb), (bottom, run)):
        pd.testing.assert_frame_equal(panel["frame"], season["frame"], check_exact=True)
        assert _keys(panel["result"]) == _record(season["result"]), panel["heading"]
        counters = panel["counters"]
        assert counters["step"].tolist() == [N_DAY], panel["heading"]
        last = counters.iloc[-1]
        assert {k: last[k] for k in RUNNING_COLS[1:]} == {
            k: season["result"][k] for k in RUNNING_COLS[1:]
        }, panel["heading"]
    title, caption = duel["title"], " ".join(duel["caption"].split())
    assert _whole(title, 1) and not _whole(title, 60), title
    said = " ".join(f"{title} {caption}".split())  # the split may sit in the title or the caption
    for part in ("VALIDATION", "Amsterdam 2010", "not the final"):
        assert part in said, (part, said)
    assert _whole(said, 59), said
    for part in (rid, "1750000"):
        assert part in caption, (part, caption)
    for n in (107, 10, 36, 15, 34):
        assert _whole(caption, n), (n, caption)
    foot = " ".join(duel["footer"].split())
    for part in (
        rid,
        RUN_COMMIT[:7],
        src["result"],
        src["decision"],
        e1,
        REPLAY_PLACE,
        f"commit {stamp['commit'][:7]}",
        REPLAY_NOTE,
    ):
        assert part in foot, (part, foot)
    assert not [t for t in (title, caption, foot) if "—" in t]

    by_file = {e["file"]: e for e in json.loads((out / "media.json").read_text())}
    today = ["baselines.png", "season_replay.gif", "season_replay.mp4", "tradeoff.png"]
    assert sorted(by_file) == sorted([*today, *REPLAY_FILES])
    for name in REPLAY_FILES:
        e = by_file[name]
        assert ENTRY_KEYS | {"split", "ledger", "run"} <= set(e), (name, sorted(e))
        assert (e["split"], e["scenario"]) == ("validation", SEASON), name
        assert e["commit"] == stamp["commit"] and e["dirty"] is stamp["dirty"], name
        assert sorted(e["ledger"]) == sorted([rid, src["result"], src["decision"], e1]), name
        assert {k: e["run"][k] for k in ("seed", "checkpoint", "manifest_commit")} == {
            "seed": 107,
            "checkpoint": "1750000",
            "manifest_commit": RUN_COMMIT,
        }, name
        assert Path(e["run"]["run_dir"]).resolve() == run_dir.resolve(), name
        assert sorted(e["metrics"]) == ["rb_tuned", ARM], name
        assert _keys(e["metrics"]["rb_tuned"]) == _record(rb["result"]), name
        assert _keys(e["metrics"][ARM]) == _record(run["result"]), name


def test_main_replay_says_validation_not_the_final_as_large_as_the_totals(
    sources, tmp_path, monkeypatch
):
    """Review fix: the split is said at title level, not only in the caption and footer. At
    LinkedIn's 50% size a 9 pt caption is unreadable while the 13 pt totals are not, so the
    numbers would outlast their label.

    main --replay on fake data (as above), with duel_replay drawing for real (out_mp4 None: no
    ffmpeg here) and _stripes spied on, so both figures it draws are read: the GIF's (12.8 x 7.2
    in) and the still's (hero.png, 12 x 6.27 in), each in its final state (the season's last day).
    On each, some one text contains both "VALIDATION" and "not the final", and its fontsize is at
    least the largest fontsize of the per-panel totals, read from the same figure: the texts
    printing rb_tuned's and the run's profit with "€/m²". Before the fix the phrase is only in the
    9 pt caption, under the 13 pt totals."""
    media = _media_has()
    rb, run = _graded("rb_tuned", 1), _graded("rb_default", 1)
    src = _replay_sources(
        tmp_path / "replay", record=_record(run["result"]), rb_record=_record(rb["result"])
    )
    _replay_world(monkeypatch, media, sources, src, run, drawing="today")
    real_stripes, real_duel = media._stripes, media.duel_replay
    figures = []

    def stripes(*args, **kwargs):
        fig, draw = real_stripes(*args, **kwargs)
        figures.append(fig)
        return fig, draw

    def duel(panels, out_gif, out_mp4, out_png, **kwargs):
        return real_duel(panels, out_gif, None, out_png, **kwargs)

    monkeypatch.setattr(media, "_stripes", stripes)
    monkeypatch.setattr(media, "duel_replay", duel)

    assert media.main(["--out", str(tmp_path / "media"), "--replay", src["result"]]) == 0

    sizes = sorted(tuple(np.round(fig.get_size_inches(), 2)) for fig in figures)
    assert sizes == [(12.0, 6.27), (12.8, 7.2)], sizes
    profits = [rb["result"]["profit_eur_m2"], run["result"]["profit_eur_m2"]]
    for fig in figures:
        where = tuple(np.round(fig.get_size_inches(), 2))
        totals = [t for v in profits for t in _printing(fig, "€/m²", v)]
        assert totals, (where, "no per-panel profit total")
        floor = max(t.get_fontsize() for t in totals)
        said = [
            (t.get_fontsize(), t.get_text())
            for t in _texts(fig)
            if "VALIDATION" in t.get_text() and "not the final" in " ".join(t.get_text().split())
        ]
        assert said, (where, "no text says VALIDATION and not the final")
        assert max(size for size, _ in said) >= floor, (
            where,
            f"VALIDATION, not the final is at most {max(s for s, _ in said)} pt, under the"
            f" {floor} pt totals",
            said,
        )


@pytest.mark.parametrize("case", PICK_REFUSALS)
def test_main_replay_refuses_a_bad_pick_before_any_season_or_file(
    case, sources, tmp_path, monkeypatch, capsys
):
    """Done when: every refusal in step 3 happens before any file is written.

    main --replay on each of PICK_REFUSALS' flawed tmp ledgers (as in replay_pick's test) exits
    nonzero or raises ValueError, naming the row (or the run's dir, or the parquet), before any
    season is played (every way to play one, a replay included, fails if called) and before
    anything is written: --out isn't created. The card reads the synthetic rows; ffmpeg is
    reported available."""
    media = _media_has()
    src = _replay_sources(tmp_path / "replay", case)
    _no_replay_played(monkeypatch, media)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
    _card_reads(monkeypatch, media, sources)
    _replay_reads(monkeypatch, media, src)
    out = tmp_path / "media"

    code, text = _refused(media, ["--out", str(out), "--replay", src["replay"]], capsys)

    assert code not in (0, None), text
    assert any(name in text for name in src["named"]), (src["named"], text)
    assert not out.exists()


@pytest.mark.parametrize(("side", "key"), RECORD_MISMATCHES)
def test_main_replay_refuses_a_season_that_differs_from_its_record(
    side, key, sources, tmp_path, monkeypatch, capsys
):
    """Done when: a replayed result that differs from its record is refused before any file is
    written.

    Each replay must equal its record on every ledger.RESULT_KEYS key, hours and steps exactly and
    the rest to rel 1e-9: rb_tuned against the parquet's day-59 row, the run against its
    selected.json's per_scenario[1]. With one record off (viol_hours_temp 0.25 h high: rb_tuned's
    1-day 3.25 h recorded as 3.50, the run's 2.00 as 2.25; or profit_eur_m2 rel 1e-6 high), main
    exits nonzero or raises ValueError, naming the key and the side ("rb_tuned", or the run's arm
    or row), and nothing is drawn or written: every drawing fails if called and --out isn't
    created. The other side's record matches."""
    media = _media_has()
    rb, run = _graded("rb_tuned", 1), _graded("rb_default", 1)
    rb_record, run_record = _record(rb["result"]), _record(run["result"])
    record = rb_record if side == "rb_tuned" else run_record
    record[key] = record[key] + 0.25 if key.startswith("viol_hours") else record[key] * (1 + 1e-6)
    src = _replay_sources(tmp_path / "replay", record=run_record, rb_record=rb_record)
    _replay_world(monkeypatch, media, sources, src, run, drawing="trip")
    out = tmp_path / "media"
    names = {"rb_tuned"} if side == "rb_tuned" else {ARM, src["runs"][107][0]["row_id"]}

    code, text = _refused(media, ["--out", str(out), "--replay", src["result"]], capsys)

    assert code not in (0, None), text
    assert key in text, text
    assert any(name in text for name in names), (names, text)
    assert not out.exists()


def test_main_replay_takes_profit_within_rel_1e9(sources, tmp_path, monkeypatch):
    """Float noise is not a mismatch: with both records' profit_eur_m2 rel 1e-11 high (as a
    different summation order gives), main --replay still returns 0 and draws the replay."""
    media = _media_has()
    rb, run = _graded("rb_tuned", 1), _graded("rb_default", 1)
    rb_record, run_record = _record(rb["result"]), _record(run["result"])
    for record in (rb_record, run_record):
        record["profit_eur_m2"] *= 1 + 1e-11
    src = _replay_sources(tmp_path / "replay", record=run_record, rb_record=rb_record)
    calls = _replay_world(monkeypatch, media, sources, src, run)

    assert media.main(["--out", str(tmp_path / "media"), "--replay", src["result"]]) == 0

    assert "duel" in calls


# --- The notes once the final is recorded (issue #56) -------------------------------------------


def _spend_seal(path: Path, result_id: str) -> str:
    """Spend the seal once on the tmp ledger at path through the ledger's own API: a
    final-registered row at result_id's selected checkpoints (baselines rb_default and rb_tuned),
    then a final-result row over ledger.TEST_SPLITS. Returns the final-result's row id. Its numbers
    are made up and never drawn (the notes name the row, never a number); no 2012 weather is read."""
    [checkpoints] = [
        r["selected"] for r in ledger.rows("result", path=path) if r["row_id"] == result_id
    ]
    freg = ledger.final_register(
        issue=45,
        checkpoints=checkpoints,
        baselines=["rb_default", "rb_tuned"],
        approved_by="Aarmen",
        path=path,
    )

    def entry(arm: str, scored: dict) -> dict:
        return {
            "iqm": 1.0,
            "ci_low": 0.5,
            "ci_high": 1.5,
            "guards": {key: {"iqm": 1.0, "ci_low": 0.5, "ci_high": 1.5} for key in HOUR_KEYS},
            "invalid_episodes": 0,
            "prob_improvement": None if arm == "rb_tuned" else 0.5,  # P(arm > rb_tuned)
            "scored": dict(scored),
        }

    [arm] = {r["arm"] for r in ledger.rows("run", path=path) if r["row_id"] in checkpoints}
    arms = {arm: checkpoints, "rb_default": {}, "rb_tuned": {}}
    per_split = {s: {a: entry(a, at) for a, at in arms.items()} for s in ledger.TEST_SPLITS}
    fres = ledger.final_result(final_registered=freg["row_id"], per_split=per_split, path=path)
    return fres["row_id"]


def _main_drawn(media, src: dict, out: Path, replay: dict | None = None) -> dict:
    """main --out out on fake data in its own patch context, every drawing recorded
    (_drawing_recorded) and the card, and main's other ledger reads, on src's ledger
    (_card_reads). Without replay, play returns the cached 1-day seasons and ffmpeg is reported
    available; with replay ({"src", "run"}), main --replay <src's result> runs in _replay_world.
    Returns the recorded calls."""
    with pytest.MonkeyPatch.context() as mp:
        if replay is None:
            mp.setattr(media, "play", lambda controller, season_length=60: _baseline(controller))
            mp.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: True))
            _card_reads(mp, media, src)
            calls = _drawing_recorded(mp, media)
            argv = ["--out", str(out)]
        else:
            calls = _replay_world(mp, media, src, replay["src"], replay["run"])
            argv = ["--out", str(out), "--replay", replay["src"]["result"]]
        assert media.main(argv) == 0
    return calls


def test_main_tradeoff_note_says_sealed_until_a_final_result_row_then_names_it(sources, tmp_path):
    """Done when (#56): with a tmp ledger without a final-result row the note says sealed; with
    one, it names the row and never says "sealed".

    main --out OUT runs twice on fake data (_main_drawn: 1-day seasons, every drawing recorded):
    first on the synthetic ledger, which has no final-result row, then on a copy of it with the
    seal spent (_spend_seal: a final-registered row at E3's checkpoints, then a final-result row,
    written by ledger.final_register and ledger.final_result).
    - The first trade-off footer carries today's note, CARD_NOTE.
    - The second carries "Recorded VALIDATION results; the 2012 test results are in
      docs/REPORT.md §5 (<that final-result row id>)" and never "sealed".
    - Nothing else in the cards changes: the second footer is the first with only the note
      swapped (line breaks aside), and the replay's and the baselines card's footers are the same
      in both runs."""
    media = _media_card()
    path = tmp_path / "spent" / "ledger.jsonl"
    path.parent.mkdir()
    shutil.copyfile(sources["ledger"], path)
    fres = _spend_seal(path, sources["rows"]["E3"]["result"])

    sealed = _main_drawn(media, sources, tmp_path / "sealed" / "media")
    spent = _main_drawn(media, {**sources, "ledger": path}, tmp_path / "spent" / "media")

    before, after = (" ".join(c["tradeoff"]["footer"].split()) for c in (sealed, spent))
    note = CARD_FINAL_NOTE.format(fres)
    assert CARD_NOTE in before, before
    assert note in after, (note, after)
    assert "sealed" not in after.lower(), after
    assert after == before.replace(CARD_NOTE, note), (before, after)
    for name in ("replay", "card"):
        assert spent[name]["footer"] == sealed[name]["footer"], name


def test_main_replay_note_says_sealed_until_a_final_result_row_then_names_it(sources, tmp_path):
    """Done when (#56's scope note): the replay's note (hero.png, duel_replay.gif and .mp4)
    follows the trade-off card's rule.

    main --out OUT --replay <result> runs twice on fake data (_replay_world: 1-day seasons,
    rb_tuned played for real, run_policy a spy returning rb_default's 1-day SEASON, every drawing
    recorded). Each run reads one tmp ledger for the card, the pick and the rest, as the CLI reads
    one: first a copy of the synthetic ledger with the E6-shaped replay rows appended
    (_replay_sources), which has no final-result row; then a copy of that one with the seal spent
    after them (_spend_seal), so both name the same rows and run dirs.
    - Without a final-result row duel_replay's footer carries today's note, REPLAY_NOTE.
    - With one it carries "A replay of a recorded VALIDATION result; the 2012 test results are in
      docs/REPORT.md §5 (<that final-result row id>)" and never "sealed".
    - Nothing else changes: the second footer is the first with only the note swapped (line
      breaks aside), and the title and caption are the same in both runs."""
    media = _media_has()
    rb, run = _graded("rb_tuned", 1), _graded("rb_default", 1)
    root = tmp_path / "sealed"
    root.mkdir()
    shutil.copyfile(sources["ledger"], root / "ledger.jsonl")
    src = _replay_sources(root, record=_record(run["result"]), rb_record=_record(rb["result"]))
    spent = tmp_path / "spent" / "ledger.jsonl"
    spent.parent.mkdir()
    shutil.copyfile(src["ledger"], spent)
    fres = _spend_seal(spent, sources["rows"]["E3"]["result"])
    duels = {}
    for case, path in (("sealed", src["ledger"]), ("spent", spent)):
        replay = {"src": {**src, "ledger": path}, "run": run}
        calls = _main_drawn(media, {**sources, "ledger": path}, path.parent / "media", replay)
        duels[case] = calls["duel"]

    before, after = (" ".join(duels[c]["footer"].split()) for c in ("sealed", "spent"))
    note = REPLAY_FINAL_NOTE.format(fres)
    assert REPLAY_NOTE in before, before
    assert note in after, (note, after)
    assert "sealed" not in after.lower(), after
    assert after == before.replace(REPLAY_NOTE, note), (before, after)
    for key in ("title", "caption"):
        assert duels["spent"][key] == duels["sealed"][key], key


def test_cli_runs_as_a_module_and_lists_out(tmp_path):
    """python -m greenleaf_control.media --help exits 0 and lists --out (the issue's one command
    is python -m greenleaf_control.media --out docs/media) and --replay (issue #51's opt-in
    replay of a result row)."""
    result = subprocess.run(
        [sys.executable, "-m", "greenleaf_control.media", "--help"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout
    assert "--replay" in result.stdout, result.stdout


@pytest.mark.parametrize("module", ["matplotlib.pyplot"])
def test_media_never_imports_pyplot(module):
    """media builds its Figures and animation directly, never through pyplot (headless, no global
    state), as viz does."""
    path = Path(greenleaf_control.__file__).parent / "media.py"
    assert path.is_file(), f"{path} doesn't exist"
    imported = _imported_modules(path)
    assert not [name for name in imported if name == module or name.startswith(f"{module}.")]


def test_signatures():
    """Done when: test_signatures covers the new functions and footer's keywords.

    play(controller, season_length=60), running(states, controls, params, ends),
    season_replay(frame, counters, out_gif, out_mp4=None, *, title, subtitle=None, footer, fps=15),
    baselines_card(seasons, out, *, footer), footer(source, stamp, *, place=PAPER_NAME, note="A
    simulator run, not a registered result"), main(argv=None), issue #50's
    tradeoff_points(path=ledger.LEDGER), rb_tuned_reference(n_seeds, path=ledger.LEDGER,
    parquet=selection.RESULTS / "rb_tuned_validation.parquet") and tradeoff_card(points,
    references, out, *, footer), and issue #51's _hourly(frame), replay_pick(result_id,
    path=ledger.LEDGER, parquet=selection.RESULTS / "rb_tuned_validation.parquet"),
    _stripes(panels, figsize, *, title, caption, footer) and duel_replay(panels, out_gif, out_mp4,
    out_png, *, title, caption, footer, fps=REPLAY_FPS)."""
    import greenleaf_control.media as media

    def shape(fn) -> list[tuple[str, str, object]]:
        return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]

    empty = inspect.Parameter.empty
    pos, kw = "POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"
    assert shape(media.play) == [("controller", pos, empty), ("season_length", pos, 60)]
    assert shape(media.running) == [
        (name, pos, empty) for name in ("states", "controls", "params", "ends")
    ]
    assert shape(media.season_replay) == [
        ("frame", pos, empty),
        ("counters", pos, empty),
        ("out_gif", pos, empty),
        ("out_mp4", pos, None),
        ("title", kw, empty),
        ("subtitle", kw, None),
        ("footer", kw, empty),
        ("fps", kw, 15),
    ]
    assert shape(media.baselines_card) == [
        ("seasons", pos, empty),
        ("out", pos, empty),
        ("footer", kw, empty),
    ]
    assert shape(media.footer) == [
        ("source", pos, empty),
        ("stamp", pos, empty),
        ("place", kw, media.PAPER_NAME),
        ("note", kw, PAPER_NOTE),
    ]
    assert shape(media.main) == [("argv", pos, None)]
    _media_card()
    assert shape(media.tradeoff_points) == [("path", pos, ledger.LEDGER)]
    assert shape(media.rb_tuned_reference) == [
        ("n_seeds", pos, empty),
        ("path", pos, ledger.LEDGER),
        ("parquet", pos, selection.RESULTS / PARQUET),
    ]
    assert shape(media.tradeoff_card) == [
        ("points", pos, empty),
        ("references", pos, empty),
        ("out", pos, empty),
        ("footer", kw, empty),
    ]
    _media_has()
    assert shape(media._hourly) == [("frame", pos, empty)]
    assert shape(media.replay_pick) == [
        ("result_id", pos, empty),
        ("path", pos, ledger.LEDGER),
        ("parquet", pos, selection.RESULTS / PARQUET),
    ]
    assert shape(media._stripes) == [
        ("panels", pos, empty),
        ("figsize", pos, empty),
        ("title", kw, empty),
        ("caption", kw, empty),
        ("footer", kw, empty),
    ]
    assert shape(media.duel_replay) == [
        ("panels", pos, empty),
        ("out_gif", pos, empty),
        ("out_mp4", pos, empty),
        ("out_png", pos, empty),
        ("title", kw, empty),
        ("caption", kw, empty),
        ("footer", kw, empty),
        ("fps", kw, media.REPLAY_FPS),
    ]


# --- The one command, end to end ---------------------------------------------------------------


def _ffprobe(path: Path) -> dict:
    out = subprocess.run(
        [
            *("ffprobe", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", "stream=codec_name,width,height:format=duration"),
            *("-of", "json", str(path)),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _gif_seconds(path: Path) -> float:
    with Image.open(path) as im:
        total_ms = 0
        for i in range(im.n_frames):
            im.seek(i)
            total_ms += im.info["duration"]
    return total_ms / 1000


def _assert_golden(got: dict, want: dict, name: str) -> None:
    """The card's numbers against tests/golden/values.json "ours": hours exactly (multiples of
    0.25 h), profit and fruit growth at rel 1e-6, as test_golden does."""
    for c in metrics.CONSTRAINTS:
        assert got[f"viol_hours_{c}"] == want[f"viol_hours_{c}"], (name, c)
    for key in ("profit_eur_m2", "yield_kg_m2"):
        assert got[key] == pytest.approx(want[key], rel=1e-6), (name, key)


@pytest.mark.slow  # three 60-day seasons, then the GIF, the MP4 and the card
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)
def test_one_command_writes_every_file_labelled_with_its_source_and_commit(
    sources, tmp_path, monkeypatch
):
    """Done when: one command writes the GIF, the MP4, the baselines PNG, the trade-off PNG and
    media.json into --out.

    main(["--out", out]) at its defaults (60-day PAPER seasons) returns 0 and writes exactly
    season_replay.gif, season_replay.mp4, baselines.png, tradeoff.png and media.json:
    - the GIF under 8 MB and 8-20 s long (the README's ~10-15 s);
    - the MP4 H.264, 1080 px wide and 8-20 s long by ffprobe (LinkedIn);
    - the PNG under 500 KB;
    - tradeoff.png 1200 x 627 px (PIL) and under 1 MB, drawn from the synthetic rows (a test
      never reads the committed ledger or results);
    - media.json a list of one entry per picture, each with file, source, scenario, commit
      and dirty (this checkout's code_stamp, taken before writing), gl_gym (the installed gl-gym's
      commit, 7 characters or more), made_at (ISO 8601) and metrics. The replay's entries: source
      rb_default, metrics its season's; the card's: source naming all three baselines, metrics
      keyed by baseline; both scenario PAPER. Every number equals tests/golden/values.json "ours"
      (hours exactly, the rest at rel 1e-6). The trade-off's: scenario scenarios.VALIDATION,
      split "validation".
    - The footers drawn (season_replay's and baselines_card's footer argument, spied on; the real
      functions run) name their sources and "commit <sha7>".
    - The replay's title says "simulated", 60 (days) and Amsterdam 2010 (from scenarios.PAPER);
      its subtitle, drawn on the final figure, names "upstream's default rule controller" and
      "GreenLight-Gym".
    """
    media = _media_card()
    _card_reads(monkeypatch, media, sources)

    stamp = manifest.code_stamp(smoke=True)
    sha7 = stamp["commit"][:7]
    footers, replays = {}, {}
    real_replay, real_card = media.season_replay, media.baselines_card

    def replay(*args, **kwargs):
        footers["replay"] = kwargs["footer"]
        replays["kwargs"] = kwargs
        replays["fig"] = real_replay(*args, **kwargs)
        return replays["fig"]

    def card(*args, **kwargs):
        footers["card"] = kwargs["footer"]
        return real_card(*args, **kwargs)

    monkeypatch.setattr(media, "season_replay", replay)
    monkeypatch.setattr(media, "baselines_card", card)
    out = tmp_path / "media"

    assert media.main(["--out", str(out)]) == 0

    assert sorted(p.name for p in out.iterdir()) == MEDIA_FILES
    gif, mp4, png = out / "season_replay.gif", out / "season_replay.mp4", out / "baselines.png"
    assert gif.read_bytes()[:4] == b"GIF8"
    assert gif.stat().st_size < 8_000_000
    assert 8 <= _gif_seconds(gif) <= 20
    probe = _ffprobe(mp4)
    [stream] = probe["streams"]
    assert (stream["codec_name"], stream["width"]) == ("h264", 1080)
    assert 8 <= float(probe["format"]["duration"]) <= 20
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.stat().st_size < 500_000
    tradeoff = out / "tradeoff.png"
    with Image.open(tradeoff) as im:
        assert im.size == (1200, 627)
    assert tradeoff.stat().st_size < 1_000_000

    assert "rb_default" in footers["replay"] and f"commit {sha7}" in footers["replay"]
    assert all(c in footers["card"] for c in CONTROLLERS) and f"commit {sha7}" in footers["card"]

    title = replays["kwargs"]["title"]
    assert "simulated" in title.lower(), title
    assert _whole(title, 60), title  # the 60-day season, 5,760 steps / 96 a day
    assert PLACE in " ".join(title.split()), title
    subtitles = [
        " ".join(t.get_text().split())
        for t in _texts(replays["fig"])
        if all(part in " ".join(t.get_text().split()) for part in RB_SUBTITLE)
    ]
    assert subtitles, "the final replay figure has no subtitle naming the controller"

    entries = json.loads((out / "media.json").read_text())
    assert isinstance(entries, list)
    by_file = {e["file"]: e for e in entries}
    assert sorted(by_file) == [name for name in MEDIA_FILES if name != "media.json"]
    gl_gym = manifest._versions()["gl-gym"]
    for name, e in by_file.items():
        assert ENTRY_KEYS <= set(e), name
        assert e["scenario"] == (VALIDATION if name == "tradeoff.png" else PAPER), name
        assert len(e["commit"]) >= 7 and stamp["commit"].startswith(e["commit"]), name
        assert e["dirty"] is stamp["dirty"], name
        assert len(e["gl_gym"]) >= 7 and gl_gym.startswith(e["gl_gym"]), name
        datetime.fromisoformat(e["made_at"])
    for name in ("season_replay.gif", "season_replay.mp4"):
        assert "rb_default" in by_file[name]["source"], name
        _assert_golden(by_file[name]["metrics"], GOLDEN["rb_default"], name)
    card_entry = by_file["baselines.png"]
    assert all(c in card_entry["source"] for c in CONTROLLERS)
    for c in CONTROLLERS:
        _assert_golden(card_entry["metrics"][c], GOLDEN[c], c)
    assert by_file["tradeoff.png"]["split"] == "validation"


@pytest.mark.slow  # two 60-day seasons here, rb_tuned's again in main, then the GIF and the MP4
@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)
def test_replay_command_writes_the_gif_mp4_and_hero(sources, tmp_path, monkeypatch):
    """Done when: --replay on fake run dirs writes a GIF under 8 MB and 8 to 20 s long, and an MP4
    that is h264, 1920 wide and 8 to 20 s long (ffprobe).

    main --out OUT --replay <result> on the E6-shaped tmp ledger with real 60-day seasons: rb_tuned
    played by main through export_replay at its 60 days, and run_policy a spy returning
    rb_default's 60-day SEASON, each recorded as itself, so the records match. Today's three
    pictures are recorded (the other slow test draws them); duel_replay draws for real. It returns
    0 and writes:
    - duel_replay.gif: GIF8, 1280 x 720, under 8 MB, 8-20 s (60 days x DAY_FRAMES + a 2 s hold);
    - duel_replay.mp4: h264, 1920 x 1080, 8-20 s (ffprobe);
    - hero.png: 1200 x 627 (PIL)."""
    media = _media_has()
    rb, run = _graded("rb_tuned", 60), _graded("rb_default", 60)
    src = _replay_sources(
        tmp_path / "replay", record=_record(run["result"]), rb_record=_record(rb["result"])
    )
    _replay_world(monkeypatch, media, sources, src, run, days=60, drawing="today")
    out = tmp_path / "media"

    assert media.main(["--out", str(out), "--replay", src["result"]]) == 0

    gif, mp4, png = (out / name for name in REPLAY_FILES)
    assert gif.read_bytes()[:4] == b"GIF8"
    with Image.open(gif) as im:
        assert im.size == (1280, 720)
    assert gif.stat().st_size < 8_000_000
    assert 8 <= _gif_seconds(gif) <= 20
    probe = _ffprobe(mp4)
    [stream] = probe["streams"]
    assert (stream["codec_name"], stream["width"], stream["height"]) == ("h264", 1920, 1080)
    assert 8 <= float(probe["format"]["duration"]) <= 20
    with Image.open(png) as im:
        assert im.size == (1200, 627)

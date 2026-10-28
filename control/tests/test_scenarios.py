"""Tests for greenleaf_control.scenarios and the weather data (issue #6).

The splits are docs/PROTOCOL.md §2, the interface is docs/API-CONTRACT.md. The 2012 files are the
sealed test weather: tests only reset an env on them, hash them, and check their format and
ranges in code, and an assertion on them never prints a weather value.
"""

import base64
import functools
import hashlib
import importlib.util
import itertools
import re
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

# Upstream and library imports live here; greenleaf_control.scenarios is imported inside each
# test body, so its absence FAILS the test instead of ERRORing at collection.
import gl_gym  # noqa: F401  (importing gl_gym registers its env ids with gymnasium)
import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import PchipInterpolator

CONTROL = Path(__file__).resolve().parents[1]
WEATHER = CONTROL / "data" / "weather"
FETCHER = CONTROL / "scripts" / "fetch_weather.py"
HEADER = (
    "time,global radiation,wind speed,air temperature,sky temperature,??,"
    "CO2 concentration,day number,RH"
)
SPLIT_NAMES = [
    "train_eval",
    "validation",
    "paper",
    "amsterdam_2012",
    "london_2012",
    "abbotsford_2012",
]
TEST_NAMES = ["amsterdam_2012", "london_2012", "abbotsford_2012"]
TEST_DAYS = (45, 49, 53, 57, 61, 65, 69, 73)
# Every weather file, keyed as weather_hashes() keys it.
WEATHER_KEYS = {
    "Amsterdam/2008.csv",
    "Amsterdam/2009.csv",
    "Amsterdam/2010.csv",
    "Amsterdam/2011.csv",
    "Amsterdam/2012.csv",
    "London/2012.csv",
    "Abbotsford/2012.csv",
}


def _scen(location: str, year: int, days) -> list[dict]:
    return [{"location": location, "growth_year": year, "start_day": d} for d in days]


# PROTOCOL §2, ordered by year, then start day.
EXPECTED = {
    "train_eval": [s for y in (2008, 2009, 2011) for s in _scen("Amsterdam", y, (45, 60, 75))],
    "validation": _scen("Amsterdam", 2010, (45, 59, 75)),
    "paper": _scen("Amsterdam", 2010, (59,)),
    "amsterdam_2012": _scen("Amsterdam", 2012, TEST_DAYS),
    "london_2012": _scen("London", 2012, TEST_DAYS),
    "abbotsford_2012": _scen("Abbotsford", 2012, TEST_DAYS),
}
SIZES = {
    "train_eval": 9,
    "validation": 3,
    "paper": 1,
    "amsterdam_2012": 8,
    "london_2012": 8,
    "abbotsford_2012": 8,
}


def _key(scenario: dict) -> tuple:
    return (scenario["location"], scenario["growth_year"], scenario["start_day"])


def _own_split(scenarios, name: str) -> list[dict]:
    """The split read from its own module attribute, not from SPLITS."""
    if name == "train_eval":
        return scenarios.TRAIN_EVAL
    if name == "validation":
        return scenarios.VALIDATION
    if name == "paper":
        return [scenarios.PAPER]
    return scenarios.TEST[name]


def _train_support(scenarios) -> set[tuple]:
    """Every (location, year, start day) the TRAIN sampler can draw."""
    kw = scenarios.TRAIN_SAMPLER_KWARGS
    return set(itertools.product(kw["locations"], kw["growth_years"], kw["start_days"]))


@functools.cache
def _one_day_env():
    """One 1-day tomato env on our WEATHER_DIR, built once and reused by the reset tests."""
    import greenleaf_control.scenarios as scenarios

    return gym.make(
        "gl_gym/GreenLightTomato-v0",
        season_length=1,
        weather_data_dir=str(scenarios.WEATHER_DIR),
    )


def _fetcher():
    """Import scripts/fetch_weather.py by path (scripts/ is not a package)."""
    assert FETCHER.is_file(), f"missing {FETCHER}"
    spec = importlib.util.spec_from_file_location("fetch_weather", FETCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weather_dir_is_control_data_weather():
    """WEATHER_DIR is the Path control/data/weather, next to the package, and it exists."""
    import greenleaf_control.scenarios as scenarios

    assert isinstance(scenarios.WEATHER_DIR, Path)
    assert (
        scenarios.WEATHER_DIR == Path(scenarios.__file__).resolve().parents[1] / "data" / "weather"
    )
    assert scenarios.WEATHER_DIR == WEATHER
    assert scenarios.WEATHER_DIR.is_dir()


@pytest.mark.parametrize(
    "name",
    ["train_eval", "validation", "paper", "amsterdam_2012", "london_2012", "abbotsford_2012"],
)
def test_split_has_protocol_scenarios(name):
    """Each split holds exactly PROTOCOL §2's scenarios, ordered by year then start day.

    Sizes: TRAIN_EVAL 9 (Amsterdam {2008, 2009, 2011} × {45, 60, 75}), VALIDATION 3 (Amsterdam
    2010 × {45, 59, 75}), PAPER 1 (Amsterdam 2010, day 59), and 8 per TEST split (2012 ×
    {45, 49, ..., 73}). The split is read both from its own attribute and from SPLITS, and its
    values are plain str and int (they go into JSON manifests and ledger rows).
    """
    import greenleaf_control.scenarios as scenarios

    own = _own_split(scenarios, name)
    assert len(own) == SIZES[name]
    assert own == EXPECTED[name]
    assert scenarios.SPLITS[name] == EXPECTED[name]
    for s in own:
        assert type(s["location"]) is str
        assert type(s["growth_year"]) is int
        assert type(s["start_day"]) is int


def test_splits_has_exactly_the_six_named_splits():
    """SPLITS has the 6 split names; TEST has the 3 test names; SPLITS["paper"] is [PAPER]."""
    import greenleaf_control.scenarios as scenarios

    assert set(scenarios.SPLITS) == set(SPLIT_NAMES)
    assert set(scenarios.TEST) == set(TEST_NAMES)
    assert scenarios.SPLITS["paper"] == [scenarios.PAPER]


def test_paper_is_upstream_default_and_inside_validation():
    """PAPER is Amsterdam 2010, day 59: gl-gym's own default scenario, and one of VALIDATION's.

    VALIDATION contains PAPER by design (PROTOCOL §2).
    """
    import greenleaf_control.scenarios as scenarios

    assert scenarios.PAPER == {"location": "Amsterdam", "growth_year": 2010, "start_day": 59}
    upstream = gym.spec("gl_gym/GreenLightTomato-v0").kwargs["weather_scenario_sampler_kwargs"]
    assert scenarios.PAPER == upstream
    assert scenarios.PAPER in scenarios.VALIDATION


def test_train_sampler_kwargs_match_contract_and_exclude_held_out_years():
    """TRAIN_SAMPLER_KWARGS is the contract literal; it holds no 2010 (VALIDATION) or 2012 (TEST)."""
    import greenleaf_control.scenarios as scenarios

    assert scenarios.TRAIN_SAMPLER_KWARGS == {
        "locations": ["Amsterdam"],
        "growth_years": [2008, 2009, 2011],
        "start_days": list(range(45, 76)),
    }
    assert 2010 not in scenarios.TRAIN_SAMPLER_KWARGS["growth_years"]
    assert 2012 not in scenarios.TRAIN_SAMPLER_KWARGS["growth_years"]


def test_splits_are_disjoint():
    """TRAIN_EVAL, VALIDATION and the 3 TEST splits share no scenario, and none repeats one.

    TRAIN's support (every year × start day its sampler can draw) misses VALIDATION, PAPER and
    every TEST split. TRAIN_EVAL is inside TRAIN's support by design, so it isn't checked here.
    """
    import greenleaf_control.scenarios as scenarios

    held = ["train_eval", "validation", *TEST_NAMES]
    keys = {name: [_key(s) for s in scenarios.SPLITS[name]] for name in SPLIT_NAMES}
    for name in SPLIT_NAMES:
        assert len(set(keys[name])) == len(keys[name]), f"{name} repeats a scenario"
    for a, b in itertools.combinations(held, 2):
        assert not set(keys[a]) & set(keys[b]), f"{a} and {b} overlap"
    train = _train_support(scenarios)
    for name in ["validation", "paper", *TEST_NAMES]:
        assert not train & set(keys[name]), f"TRAIN can draw a {name} scenario"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("train_eval", False),
        ("validation", False),
        ("paper", False),
        ("amsterdam_2012", True),
        ("london_2012", True),
        ("abbotsford_2012", True),
    ],
)
def test_is_test_flags_exactly_the_test_splits(name, expected):
    """is_test is True on every TEST scenario and False on every scenario of the other splits."""
    import greenleaf_control.scenarios as scenarios

    for s in scenarios.SPLITS[name]:
        assert scenarios.is_test(s) is expected, s


def test_is_test_on_scenarios_outside_the_splits():
    """Any 2012 scenario is a test one, on or off a split; nothing TRAIN can draw is.

    Amsterdam 2012 day 100 and London 2012 day 1 are in no split, but they are still the
    sealed test weather.
    """
    import greenleaf_control.scenarios as scenarios

    for s in [
        {"location": "Amsterdam", "growth_year": 2012, "start_day": 100},
        {"location": "London", "growth_year": 2012, "start_day": 1},
    ]:
        assert scenarios.is_test(s) is True, s
    for location, year, day in sorted(_train_support(scenarios)):
        s = {"location": location, "growth_year": year, "start_day": day}
        assert scenarios.is_test(s) is False, s


def test_no_training_or_validation_season_reaches_the_next_years_file():
    """No non-test season runs past the end of its year's weather file.

    A 60-day season plus the 0.5-day lookahead (rounded up to 1 day) needs
    start_day + 60 + 1 <= 365, the shortest year. Past that, upstream's loader appends the
    next year's file, and for 2011 that is 2012, the test weather. Known answer: the latest
    start, day 75, ends at 75 + 61 = 136 <= 365.
    """
    import greenleaf_control.scenarios as scenarios

    days = [
        s["start_day"]
        for name in ["train_eval", "validation", "paper"]
        for s in scenarios.SPLITS[name]
    ]
    days += [day for _, _, day in _train_support(scenarios)]
    assert days
    for day in days:
        assert day + 60 + 1 <= 365, day


@pytest.mark.parametrize(
    "name",
    ["train_eval", "validation", "paper", "amsterdam_2012", "london_2012", "abbotsford_2012"],
)
def test_every_scenario_in_split_resets_a_one_day_env(name):
    """Every scenario of the split resets gl_gym/GreenLightTomato-v0 for a 1-day season.

    The env reads our WEATHER_DIR (not gl-gym's bundled copy), loads exactly the scenario asked
    for, and returns a finite observation. On a 2012 scenario this only checks the file loads:
    nothing is evaluated, and the observation is never printed.
    """
    import greenleaf_control.scenarios as scenarios

    env = _one_day_env()
    assert env.unwrapped.weather_repository.weather_data_dir == scenarios.WEATHER_DIR
    for s in scenarios.SPLITS[name]:
        obs, info = env.reset(seed=0, options={"scenario": s})
        assert info["scenario"] == s
        assert env.unwrapped.N == 96  # 1 day = 86,400 s / 900 s per step
        finite = bool(np.all(np.isfinite(gym.spaces.flatten(env.observation_space, obs))))
        assert finite, f"non-finite observation after resetting on {s}"


@pytest.mark.parametrize(
    "name",
    ["train_eval", "validation", "paper", "amsterdam_2012", "london_2012", "abbotsford_2012"],
)
def test_require_weather_passes_on_every_split(name):
    """Every split's weather is on disk, so require_weather returns None."""
    import greenleaf_control.scenarios as scenarios

    assert scenarios.require_weather(scenarios.SPLITS[name]) is None


@pytest.mark.parametrize(
    ("location", "year"),
    [
        ("Nowhere", 2010),  # an unknown location
        ("Amsterdam", 2013),  # a known location, a year we don't have
    ],
)
def test_require_weather_names_the_missing_file(location, year):
    """A missing file raises FileNotFoundError naming <WEATHER_DIR>/<location>/<year>.csv.

    PAPER, listed first, is on disk; the missing scenario after it is the one named.
    """
    import greenleaf_control.scenarios as scenarios

    missing = scenarios.WEATHER_DIR / location / f"{year}.csv"
    s = {"location": location, "growth_year": year, "start_day": 59}
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        scenarios.require_weather([scenarios.PAPER, s])


def test_require_weather_reads_weather_dir_at_call_time(tmp_path, monkeypatch):
    """require_weather looks under the WEATHER_DIR set when it's called, not at import.

    With WEATHER_DIR pointed at an empty tmp dir, PAPER's file is missing there, named by its
    tmp path; once that file exists, the call passes.
    """
    import greenleaf_control.scenarios as scenarios

    monkeypatch.setattr(scenarios, "WEATHER_DIR", tmp_path)
    missing = tmp_path / "Amsterdam" / "2010.csv"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        scenarios.require_weather([scenarios.PAPER])

    missing.parent.mkdir()
    missing.write_bytes(b"abc")
    assert scenarios.require_weather([scenarios.PAPER]) is None


def test_weather_hashes_cover_the_seven_csvs_and_are_stable():
    """weather_hashes keys each of the 7 weather files "<location>/<year>.csv" to its sha256.

    Each value is 64 lowercase hex characters, equal to hashlib's sha256 of the file's bytes,
    and two calls return the same dict.
    """
    import greenleaf_control.scenarios as scenarios

    first = scenarios.weather_hashes()
    assert set(first) == WEATHER_KEYS
    for key, digest in first.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), key
        assert digest == hashlib.sha256((WEATHER / key).read_bytes()).hexdigest(), key
    assert scenarios.weather_hashes() == first


@pytest.mark.parametrize("year", [2008, 2009, 2010, 2011, 2012])
def test_amsterdam_copy_matches_upstream_record(year):
    """Our Amsterdam/<year>.csv is byte-identical to the file gl-gym shipped.

    The reference is the sha256 in gl-gym's installed RECORD, read live: urlsafe base64 without
    padding, converted to hex here.
    """
    import greenleaf_control.scenarios as scenarios

    record = {str(f): f.hash for f in distribution("gl-gym").files if f.hash is not None}
    entry = record[f"gl_gym/data/weather/Amsterdam/{year}.csv"]
    assert entry.mode == "sha256"
    upstream = base64.urlsafe_b64decode(entry.value + "=" * (-len(entry.value) % 4)).hex()
    assert scenarios.weather_hashes()[f"Amsterdam/{year}.csv"] == upstream


def test_weather_hashes_reads_weather_dir_at_call_time(tmp_path, monkeypatch):
    """On a tmp WEATHER_DIR, weather_hashes hashes exactly its <location>/<year>.csv files.

    Known answer: the bytes b"abc" hash to sha256 ba7816bf...15ad (the FIPS 180-2 test vector).
    A .DS_Store beside them is not a weather file. Editing one file changes only its key.
    """
    import greenleaf_control.scenarios as scenarios

    monkeypatch.setattr(scenarios, "WEATHER_DIR", tmp_path)
    (tmp_path / "Here").mkdir()
    (tmp_path / "There").mkdir()
    (tmp_path / "Here" / "2001.csv").write_bytes(b"abc")
    (tmp_path / "There" / "2002.csv").write_bytes(b"x,y\n1,2\n")
    (tmp_path / "Here" / ".DS_Store").write_bytes(b"finder junk")

    before = scenarios.weather_hashes()
    assert set(before) == {"Here/2001.csv", "There/2002.csv"}
    assert before["Here/2001.csv"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )

    (tmp_path / "There" / "2002.csv").write_bytes(b"x,y\n1,3\n")
    after = scenarios.weather_hashes()
    assert after["Here/2001.csv"] == before["Here/2001.csv"]
    assert after["There/2002.csv"] != before["There/2002.csv"]
    assert after["There/2002.csv"] == hashlib.sha256(b"x,y\n1,3\n").hexdigest()


@pytest.mark.parametrize("location", ["London", "Abbotsford"])
def test_fetched_csv_has_gl_gym_format_and_plausible_ranges(location):
    """The fetched <location>/2012.csv has gl-gym's exact format and plausible weather.

    Format: the exact header; time from 0 in 300 s steps to 31,618,800 s, so 105,397 rows
    (2012 is a leap year: 366 d × 24 h = 8,784 hours, the last at 8,783 × 3,600 s);
    ?? = 0, CO2 = 400, day number = floor(time / 86,400); no NaN.
    Ranges (these catch unit slips and fill values such as -999): air temperature in
    [-30, 45] °C; RH in [0, 100] %; global radiation >= 0 W/m² with an annual max in
    [500, 1400] W/m²; wind speed in [0, 40] m/s; and on each hour, sky temperature <= air
    temperature (a clear or cloudy sky emits less than a black body at air temperature,
    so a Kelvin slip or swapped column fails).
    Local time: the fetcher asks POWER for LST, so the hour of day with the highest mean
    radiation is 11, 12 or 13 (noon, ±1 h for whether POWER stamps an hour's start or end);
    a UTC file for Abbotsford (UTC−8) would peak near hour 20 and fail.
    This is test weather: each check is reduced to a bool before it's asserted, so a failure
    names the check, never a weather value.
    """
    path = WEATHER / location / "2012.csv"
    assert path.is_file(), f"missing {path}"
    with path.open() as f:
        assert f.readline().rstrip("\n") == HEADER

    df = pd.read_csv(path)
    t = df["time"].to_numpy()
    assert len(df) == 105_397  # 31,618,800 s / 300 s + 1
    assert t[0] == 0
    assert t[-1] == 31_618_800
    hourly = df.iloc[::12]  # 3,600 s / 300 s = 12 rows per hour
    checks = {
        "no NaN": not df.isna().any().any(),
        "300 s spacing": np.all(np.diff(t) == 300),
        "?? is 0": np.all(df["??"] == 0.0),
        "CO2 is 400": np.all(df["CO2 concentration"] == 400.0),
        "day number is floor(t / 86400)": np.all(df["day number"] == np.floor(t / 86_400)),
        "air temperature in [-30, 45]": df["air temperature"].between(-30, 45).all(),
        "RH in [0, 100]": df["RH"].between(0, 100).all(),
        "global radiation >= 0": (df["global radiation"] >= 0).all(),
        "max global radiation in [500, 1400]": 500 <= df["global radiation"].max() <= 1400,
        "wind speed in [0, 40]": df["wind speed"].between(0, 40).all(),
        "sky temperature <= air temperature on the hour": (
            hourly["sky temperature"] <= hourly["air temperature"] + 1e-9
        ).all(),
        # Mean radiation per hour of day (t // 3,600 s mod 24 h) peaks at hour 11, 12 or 13.
        "radiation peaks near local noon": (
            hourly["global radiation"].groupby((hourly["time"] // 3_600) % 24).mean().idxmax()
            in {11, 12, 13}
        ),
    }
    failed = [name for name, ok in checks.items() if not bool(ok)]
    assert not failed, f"{location}/2012.csv fails: {failed}"


def test_fetch_weather_cli_takes_location_lat_lon_year():
    """scripts/fetch_weather.py --help runs offline and lists --location, --lat, --lon, --year."""
    assert FETCHER.is_file(), f"missing {FETCHER}"
    out = subprocess.run(
        [sys.executable, str(FETCHER), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    for flag in ["--location", "--lat", "--lon", "--year"]:
        assert flag in out.stdout, flag


def _sky_reference(air_c, cloud):
    """Campbell & Norman sky temperature, written from the issue's formula (sigma = 5.67e-8)."""
    t_k = air_c + 273.15
    eps_clear = (213 + 5.5 * air_c) / (5.67e-8 * t_k**4)
    eps = (1 - 0.84 * cloud) * eps_clear + 0.84 * cloud
    return (eps * t_k**4) ** 0.25 - 273.15


def test_hourly_to_frame_builds_the_300_s_csv_frame():
    """fetch_weather.hourly_to_frame(hourly) turns 26 hourly values into gl-gym's 300 s frame.

    Signature: hourly_to_frame(hourly: dict[str, np.ndarray]) -> pandas.DataFrame. The keys are
    "global radiation" (W/m²), "wind speed" (m/s), "air temperature" (°C), "RH" (%) and
    "cloud cover" (a fraction in [0, 1]); value i is hour i from Jan 1 00:00.
    Known answers: the columns are the CSV header in order; time is 0, 300, ..., 25 × 3,600 =
    90,000 s (301 rows); each passed-through column is scipy's PCHIP through the hourly points,
    so on the hour it equals the input; ?? = 0; CO2 = 400; day number is 0 before 86,400 s and
    1 from it; sky temperature on the hour is the Campbell & Norman value of that hour's air
    temperature and cloud cover.
    """
    fetch_weather = _fetcher()
    h = np.arange(26.0)
    hourly = {
        "global radiation": np.clip(800 * np.sin(np.pi * (h - 6) / 12), 0, None),
        "wind speed": 3 + 2 * np.cos(h / 4),
        "air temperature": 10 + 5 * np.sin(np.pi * (h - 9) / 12),
        "RH": 80 - 15 * np.sin(np.pi * (h - 9) / 12),
        "cloud cover": (h % 5) / 4,
    }
    df = fetch_weather.hourly_to_frame(hourly)

    assert list(df.columns) == HEADER.split(",")
    t = np.arange(0, 90_000 + 1, 300, dtype=float)
    assert len(df) == 301
    np.testing.assert_array_equal(df["time"].to_numpy(dtype=float), t)
    for col in ["global radiation", "wind speed", "air temperature", "RH"]:
        expected = PchipInterpolator(h * 3600, hourly[col])(t)
        np.testing.assert_allclose(df[col].to_numpy(), expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(df[col].to_numpy()[::12], hourly[col], rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(df["??"].to_numpy(), np.zeros(301))
    np.testing.assert_array_equal(df["CO2 concentration"].to_numpy(), np.full(301, 400.0))
    np.testing.assert_array_equal(df["day number"].to_numpy(), np.where(t < 86_400, 0.0, 1.0))
    np.testing.assert_allclose(
        df["sky temperature"].to_numpy()[::12],
        _sky_reference(hourly["air temperature"], hourly["cloud cover"]),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("air_c", "cloud", "sky_c"),
    [
        # LW = 213 + 5.5·0 = 213 W/m²; (213 / 5.67e-8)^¼ = 247.57 K = -25.58 °C
        (0.0, 0.0, -25.58),
        # ε_clear = 268 / (5.67e-8 · 283.15⁴ = 364.46) = 0.7353; ε = 0.16·0.7353 + 0.84 = 0.9577;
        # 283.15 · 0.9577^¼ = 280.10 K = 6.95 °C
        (10.0, 1.0, 6.95),
        # ε_clear = 323 / (5.67e-8 · 293.15⁴ = 418.74) = 0.7714; ε = 0.58·0.7714 + 0.42 = 0.8674;
        # 293.15 · 0.8674^¼ = 282.91 K = 9.76 °C
        (20.0, 0.5, 9.76),
    ],
)
def test_hourly_to_frame_sky_temperature_known_answer(air_c, cloud, sky_c):
    """Constant hourly weather gives a constant sky temperature, hand-computed (Campbell & Norman).

    LW = 213 + 5.5·T; ε_clear = LW / (σ·T_K⁴); ε = (1 - 0.84c)·ε_clear + 0.84c;
    T_sky = (ε·T_K⁴)^¼ - 273.15, with σ = 5.67e-8 and c the cloud fraction. Clear sky at 0 °C
    is about -25.6 °C.
    """
    fetch_weather = _fetcher()
    hourly = {
        "global radiation": np.zeros(3),
        "wind speed": np.ones(3),
        "air temperature": np.full(3, air_c),
        "RH": np.full(3, 80.0),
        "cloud cover": np.full(3, cloud),
    }
    df = fetch_weather.hourly_to_frame(hourly)

    assert len(df) == 25  # 2 × 3,600 s / 300 s + 1
    assert df["sky temperature"].to_numpy() == pytest.approx(np.full(25, sky_c), abs=0.01)

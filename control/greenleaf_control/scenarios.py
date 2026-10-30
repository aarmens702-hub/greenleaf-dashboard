"""The weather scenarios: the splits of docs/PROTOCOL.md §2 and the weather files they read.

A scenario is {"location": str, "growth_year": int, "start_day": int}; the env reads
WEATHER_DIR/<location>/<growth_year>.csv and starts the season on day start_day.

Weather data (control/data/weather/):
- Amsterdam/2008.csv to 2012.csv are copied verbatim from GreenLight-Gym2 (Bart van Laatum,
  AGPL-3.0-or-later, commit 2d3febb), gl_gym/data/weather/Amsterdam/.
- London/2012.csv and Abbotsford/2012.csv are NASA POWER hourly data (Hourly API v2.10.3,
  accessed 2026/10/28), written by scripts/fetch_weather.py: community RE, time-standard=LST,
  and WS2M (2 m) wind, the height upstream's fetcher uses. Made, from control/, by:
  uv run python scripts/fetch_weather.py --location London --lat 51.51 --lon -0.13 --year 2012
  uv run python scripts/fetch_weather.py --location Abbotsford --lat 49.05 --lon -122.33 --year 2012

Every 2012 file is the sealed test weather.
"""

import hashlib
from pathlib import Path

# control/data/weather. Functions read it at call time, so tests can point it at a temp dir.
WEATHER_DIR = Path(__file__).resolve().parents[1] / "data" / "weather"

# TRAIN: gl-gym's random sampler draws a year and a start day at every reset.
TRAIN_SAMPLER_KWARGS = {
    "locations": ["Amsterdam"],
    "growth_years": [2008, 2009, 2011],
    "start_days": list(range(45, 76)),
}


def _scenarios(location: str, years: list[int], days: list[int]) -> list[dict]:
    """Every year × start day at one location, ordered by year, then start day."""
    return [{"location": location, "growth_year": y, "start_day": d} for y in years for d in days]


_TEST_DAYS = [45, 49, 53, 57, 61, 65, 69, 73]

TRAIN_EVAL = _scenarios("Amsterdam", [2008, 2009, 2011], [45, 60, 75])
VALIDATION = _scenarios("Amsterdam", [2010], [45, 59, 75])
PAPER = {"location": "Amsterdam", "growth_year": 2010, "start_day": 59}
TEST = {
    "amsterdam_2012": _scenarios("Amsterdam", [2012], _TEST_DAYS),
    "london_2012": _scenarios("London", [2012], _TEST_DAYS),
    "abbotsford_2012": _scenarios("Abbotsford", [2012], _TEST_DAYS),
}
SPLITS = {"train_eval": TRAIN_EVAL, "validation": VALIDATION, "paper": [PAPER], **TEST}


def is_test(scenario: dict) -> bool:
    """True for any 2012 scenario, in a split or not: 2012 is the test weather at every location."""
    return bool(scenario["growth_year"] == 2012)


def weather_hashes() -> dict[str, str]:
    """sha256 of every weather file, keyed "<location>/<year>.csv". These stamp every run."""
    return {
        f"{path.parent.name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(WEATHER_DIR.glob("*/*.csv"))
    }


def require_weather(scenarios: list[dict]) -> None:
    """Raise FileNotFoundError naming the first scenario's weather file that isn't on disk."""
    for s in scenarios:
        path = WEATHER_DIR / s["location"] / f"{s['growth_year']}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing weather file {path} for scenario {s}")

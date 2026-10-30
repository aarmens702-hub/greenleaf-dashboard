"""Fetch one year of NASA POWER hourly weather and write it as a GreenLight-Gym weather CSV.

Usage, from control/:
    uv run python scripts/fetch_weather.py --location London --lat 51.51 --lon -0.13 --year 2012

It writes data/weather/<location>/<year>.csv (or --out) and refuses to overwrite a file: weather
data is immutable. The CSV has gl-gym's 9 columns at 300 s steps from Jan 1 00:00, each a PCHIP
through POWER's hourly values, with value i placed at hour i.

The request: POWER's hourly point API with
- community RE, so radiation comes in Wh/m² per hour, i.e. the hour's mean W/m²;
- time-standard LST, POWER's local solar time (UTC shifted by the longitude's 15° band: GMT for
  London, UTC-8 for Abbotsford, their civil standard times);
- wind at 2 m (WS2M), the height GreenLight-Gym2's own NASA POWER fetcher uses;
- cloud amount in percent, turned into a fraction here.

Sky temperature from air temperature T (°C) and cloud fraction c, Campbell & Norman (1998):
    LW = 213 + 5.5·T,  ε_clear = LW / (σ·T_K⁴),  ε = (1 − 0.84·c)·ε_clear + 0.84·c,
    T_sky = (ε·T_K⁴)^¼ − 273.15,  with σ = 5.67e-8 W m⁻² K⁻⁴.
Campbell, G. S. and Norman, J. M. (1998). An Introduction to Environmental Biophysics, 2nd ed.
Springer, New York.

NASA POWER's requested acknowledgment: "The data was obtained from National Aeronautics and Space
Administration (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources (POWER)
project funded through the NASA Earth Science Division."
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator

from greenleaf_control.scenarios import WEATHER_DIR

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
# POWER parameter -> (hourly_to_frame's key, the units POWER must report it in)
PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": ("global radiation", "Wh/m^2"),
    "WS2M": ("wind speed", "m/s"),
    "T2M": ("air temperature", "C"),
    "RH2M": ("RH", "%"),
    "CLOUD_AMT": ("cloud cover", "%"),
}
SIGMA = 5.67e-8  # Stefan-Boltzmann constant, W m⁻² K⁻⁴


def sky_temperature(air_c: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """Campbell & Norman sky temperature (°C) from air temperature (°C) and cloud fraction."""
    t_k = air_c + 273.15
    eps_clear = (213 + 5.5 * air_c) / (SIGMA * t_k**4)
    eps = (1 - 0.84 * cloud) * eps_clear + 0.84 * cloud
    return (eps * t_k**4) ** 0.25 - 273.15


def hourly_to_frame(hourly: dict[str, np.ndarray]) -> pd.DataFrame:
    """gl-gym's 300 s weather frame from hourly values; value i is hour i from Jan 1 00:00.

    Every weather column, sky temperature included (computed on the hour), is a PCHIP through the
    hourly points: it equals the input on the hour and doesn't overshoot between hours.
    """
    hours = 3600.0 * np.arange(len(hourly["air temperature"]))
    t = 300.0 * np.arange(12 * (len(hours) - 1) + 1)
    sky = sky_temperature(hourly["air temperature"], hourly["cloud cover"])

    def pchip(values: np.ndarray) -> np.ndarray:
        return PchipInterpolator(hours, values)(t)

    return pd.DataFrame(
        {
            "time": t,
            "global radiation": pchip(hourly["global radiation"]),
            "wind speed": pchip(hourly["wind speed"]),
            "air temperature": pchip(hourly["air temperature"]),
            "sky temperature": pchip(sky),
            "??": 0.0,
            "CO2 concentration": 400.0,
            "day number": np.floor(t / 86_400),
            "RH": pchip(hourly["RH"]),
        }
    )


def fetch_hourly(lat: float, lon: float, year: int) -> dict[str, np.ndarray]:
    """POWER's hourly values for Jan 1 00:00 to Dec 31 23:00 of year, keyed for hourly_to_frame."""
    response = requests.get(
        POWER_URL,
        params={
            "parameters": ",".join(PARAMETERS),
            "community": "RE",
            "latitude": lat,
            "longitude": lon,
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "JSON",
            "time-standard": "LST",
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    print(f"fetched {response.url} (POWER {body['header']['api']['version']})")
    hours = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="h").strftime("%Y%m%d%H")
    fill = body["header"]["fill_value"]
    hourly = {}
    for name, (key, units) in PARAMETERS.items():
        if body["parameters"][name]["units"] != units:
            raise ValueError(
                f"POWER gives {name} in {body['parameters'][name]['units']}, not {units}"
            )
        series = body["properties"]["parameter"][name]
        if list(series) != list(hours):
            raise ValueError(f"{name}: POWER's timestamps aren't the {len(hours)} hours of {year}")
        values = np.array(list(series.values()), dtype=float)
        if np.any(values == fill):
            raise ValueError(f"{name} has {np.sum(values == fill)} fill values ({fill}) in {year}")
        hourly[key] = values
    hourly["cloud cover"] = hourly["cloud cover"] / 100
    return hourly


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a year of NASA POWER hourly weather as a gl-gym weather CSV."
    )
    parser.add_argument("--location", required=True, help="directory under data/weather/")
    parser.add_argument("--lat", type=float, required=True, help="latitude, degrees north")
    parser.add_argument("--lon", type=float, required=True, help="longitude, degrees east")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--out", type=Path, help="write here, not data/weather/<location>/<year>.csv"
    )
    args = parser.parse_args(argv)

    out = args.out or WEATHER_DIR / args.location / f"{args.year}.csv"
    if out.exists():
        sys.exit(f"{out} exists; weather data is immutable, so it isn't overwritten")
    frame = hourly_to_frame(fetch_hourly(args.lat, args.lon, args.year))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out}: {len(frame)} rows, last time {frame['time'].iloc[-1]:.0f} s")


if __name__ == "__main__":
    main()

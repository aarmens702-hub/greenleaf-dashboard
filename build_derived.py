"""Build derived parquet files from raw GreenLeaf CSVs and external YAML benchmarks.

Run: python build_derived.py

Single source of truth for everything under data/derived/. Reads data/raw/ (never
modifies it) and data/external/*.yaml, emits 6 parquet files. Idempotent: rerunning
overwrites with identical output (no timestamps written into the data).
"""
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
DERIVED_DIR = ROOT / "data" / "derived"

# 7-day trailing mean feature -> source column
ROLLING_FEATURES = {
    "vpd_kpa_7d_mean": "vpd_kpa",
    "substrate_moisture_7d_mean": "substrate_moisture",
    "pest_pressure_7d_mean": "pest_pressure_index",
    "canopy_air_diff_7d_mean": "canopy_air_diff",
}


def _write(df, name):
    df.to_parquet(DERIVED_DIR / name, index=False)
    print(f"[derived] wrote {name}: {len(df)} rows, {df.shape[1]} cols")


def _join_plot_meta(df, plot_meta):
    """Left-join plot_meta onto df by plot_id, dropping plot_meta columns that df
    already carries so no _x/_y suffixes appear."""
    overlap = [c for c in plot_meta.columns if c in df.columns and c != "plot_id"]
    return df.merge(plot_meta.drop(columns=overlap), on="plot_id", how="left")


def _load_benchmarks():
    rows = []
    for path in sorted(EXTERNAL_DIR.glob("*.yaml")):
        try:
            with open(path) as fh:
                data = yaml.safe_load(fh)
            if not isinstance(data, dict):
                raise ValueError("YAML did not parse to a mapping")
        except Exception as e:
            print(f"[derived] skipping malformed YAML {path.name}: {e}")
            continue
        data["source_filename"] = path.name
        rows.append(data)
    return pd.DataFrame(rows)


def main():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    # Load all raw CSVs. market_and_input_prices is loaded here but not consumed by
    # any of the 6 outputs below — no derived file references it.
    raw = {p.stem: pd.read_csv(p) for p in sorted(RAW_DIR.glob("*.csv"))}
    farms = raw["farms"]
    greenhouses = raw["greenhouses"]
    plots = raw["plots"]
    season_summary = raw["season_summary"]
    sensor = raw["daily_sensor_readings"]
    costs = raw["daily_input_costs"]
    applications = raw["input_applications"]
    scouting = raw["scouting_observations"]

    for df in (sensor, costs, applications, scouting):
        df["date"] = pd.to_datetime(df["date"])

    # 1. plot_meta = plots + greenhouses (drop greenhouses' redundant farm_id/crop) + farms
    gh = greenhouses.drop(columns=["farm_id", "crop"])
    plot_meta = plots.merge(gh, on="greenhouse_id", how="left").merge(
        farms, on="farm_id", how="left"
    )
    _write(plot_meta, "plot_meta.parquet")

    # 2. season = season_summary + plot_meta
    season = season_summary.merge(plot_meta, on="plot_id", how="left")
    _write(season, "season.parquet")

    # 3. daily = sensor + costs (plot_id+date) + plot_meta (plot_id), then features
    costs_j = costs.drop(columns=["greenhouse_id", "farm_id"])
    daily = sensor.merge(costs_j, on=["plot_id", "date"], how="left")
    daily = _join_plot_meta(daily, plot_meta)

    daily = daily.sort_values(["plot_id", "date"]).reset_index(drop=True)
    g = daily.groupby("plot_id")["plant_stress_index"]
    daily["stress_lag_1"] = g.shift(1)
    daily["stress_lag_3"] = g.shift(3)
    daily["stress_lag_7"] = g.shift(7)
    daily["stress_lead_3"] = g.shift(-3)
    daily["stress_delta_3d_forward"] = daily["stress_lead_3"] - daily["plant_stress_index"]
    daily["spike_3d"] = (
        (daily["stress_delta_3d_forward"] >= 0.10)
        .astype(float)
        .where(daily["stress_lead_3"].notna())
    )
    daily["canopy_air_diff"] = daily["canopy_temp_c"] - daily["air_temp_c"]

    for out_col, src_col in ROLLING_FEATURES.items():
        daily[out_col] = daily.groupby("plot_id")[src_col].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
    _write(daily, "daily.parquet")

    # 4. actions = input_applications + plot_meta
    _write(_join_plot_meta(applications, plot_meta), "actions.parquet")

    # 5. scouting = scouting_observations + plot_meta
    _write(_join_plot_meta(scouting, plot_meta), "scouting.parquet")

    # 6. bc_benchmarks = one row per YAML file in data/external/
    _write(_load_benchmarks(), "bc_benchmarks.parquet")

    print("[derived] done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[derived] FAILED: {e}")
        sys.exit(1)

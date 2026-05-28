import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_derived


def test_derived():
    start = time.perf_counter()
    build_derived.main()
    elapsed = time.perf_counter() - start

    frames = {
        p.stem: pd.read_parquet(p)
        for p in sorted(build_derived.DERIVED_DIR.glob("*.parquet"))
    }

    assert len(frames["plot_meta"]) == 120
    assert len(frames["season"]) == 120
    assert len(frames["daily"]) == 25200
    assert len(frames["actions"]) == 4613
    assert len(frames["scouting"]) == 1800
    assert len(frames["bc_benchmarks"]) >= 1

    daily = frames["daily"]
    expected_features = [
        "stress_lag_1", "stress_lag_3", "stress_lag_7", "stress_lead_3",
        "stress_delta_3d_forward", "spike_3d", "canopy_air_diff",
        "vpd_kpa_7d_mean", "substrate_moisture_7d_mean",
        "pest_pressure_7d_mean", "canopy_air_diff_7d_mean",
    ]
    for col in expected_features:
        assert col in daily.columns, f"missing feature column: {col}"

    rate = daily["spike_3d"].mean()
    assert 0.15 <= rate <= 0.22, f"spike_3d base rate {rate:.4f} out of [0.15, 0.22]"

    meta_ids = set(frames["plot_meta"]["plot_id"])
    for name in ("daily", "season", "actions", "scouting"):
        orphans = set(frames[name]["plot_id"]) - meta_ids
        assert not orphans, f"{name} has plot_ids absent from plot_meta: {orphans}"

    assert elapsed < 30, f"build took {elapsed:.1f}s (>30s)"

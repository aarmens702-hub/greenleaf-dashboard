"""Tests for the safe, flexible query_data tool in app/chat/agent.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat import agent
from app.chat import data_queries as dq


def test_filtered_query():
    """Equality filter (case-insensitive) returns only matching rows."""
    out = agent.query_data("season", filters={"crop": "strawberry"})
    assert out["table"] == "season"
    assert out["n_rows_matched"] > 0
    # Every returned row is the requested crop.
    assert all(r["crop"] == "Strawberry" for r in out["result"])


def test_grouped_aggregate():
    """group_by + aggregate computes a per-group mean."""
    out = agent.query_data(
        "season", group_by=["treatment"], aggregate={"season_profit_cad": "mean"},
        sort_by="season_profit_cad", ascending=False,
    )
    assert "treatment" in out["columns_used"]
    assert "season_profit_cad" in out["columns_used"]
    assert len(out["result"]) >= 2
    # Sorted descending by mean profit.
    profits = [r["season_profit_cad"] for r in out["result"]]
    assert profits == sorted(profits, reverse=True)


def test_range_filter():
    """Operator-dict range filter keeps only rows inside the bounds."""
    out = agent.query_data(
        "season", filters={"season_roi": {">=": 0.0, "<=": 1.0}},
        aggregate={"season_roi": "max"},
    )
    assert out["result"][0]["season_roi"] <= 1.0
    assert out["n_rows_matched"] > 0


def test_crop_treatment_autojoin():
    """When a table lacks crop/treatment, they are auto-joined from plot_meta."""
    bare = dq.DAILY.drop(columns=["crop", "treatment"]).copy()
    original = agent._TABLES["daily"]
    agent._TABLES["daily"] = lambda: bare
    try:
        out = agent.query_data(
            "daily", group_by=["crop"], aggregate={"plant_stress_index": "mean"},
        )
        assert "auto-joined" in out["notes"]
        assert "crop" in out["columns_used"]
        crops = {r["crop"] for r in out["result"]}
        assert {"Strawberry", "Tomato"}.issubset(crops)
    finally:
        agent._TABLES["daily"] = original


def test_count_sum_gets_per_plot_normalization():
    """FIX 1: count/sum with group_by also returns a size-unbiased per-plot column."""
    out = agent.query_data(
        "daily", group_by=["crop", "treatment"], aggregate={"spike_3d": "sum"},
    )
    assert "spike_3d_per_plot" in out["columns_used"]
    assert "per_plot" in out["notes"]
    # Per-plot = total / distinct plots in that group; verify against a known group.
    bare = dq.DAILY[(dq.DAILY["crop"] == "Strawberry") & (dq.DAILY["treatment"] == "High Light")]
    expected = bare["spike_3d"].sum() / bare["plot_id"].nunique()
    row = next(r for r in out["result"]
               if r["crop"] == "Strawberry" and r["treatment"] == "High Light")
    assert abs(row["spike_3d_per_plot"] - round(expected, 4)) < 1e-6
    # And the per-plot basis is NOT just the size-biased total.
    assert row["spike_3d_per_plot"] < row["spike_3d"]


def test_size_fallback_gets_count_per_plot():
    """A bare group_by (no aggregate) also gets count_per_plot (rows per plot)."""
    out = agent.query_data("daily", group_by=["crop"])
    assert "count_per_plot" in out["columns_used"]


def test_mean_aggregate_not_normalized():
    """mean is already size-unbiased, so no per-plot column is added."""
    out = agent.query_data(
        "season", group_by=["treatment"], aggregate={"season_profit_cad": "mean"},
    )
    assert not any(c.endswith("_per_plot") for c in out["columns_used"])


def test_time_bucket_covers_whole_season():
    """FIX P1: time_bucket resamples by date so trends span the full season."""
    out = agent.query_data("daily", aggregate={"plant_stress_index": "mean"}, time_bucket="month")
    assert "period" in out["columns_used"]
    periods = [r["period"] for r in out["result"]]
    # Season runs Feb-Sep 2025: expect ~8 monthly buckets, first Feb, last Sep.
    assert len(periods) >= 7
    assert periods[0].startswith("2025-02")
    assert periods[-1].startswith("2025-09")


def test_time_series_not_truncated_to_20():
    """Weekly buckets exceed the old default limit of 20 and must NOT be truncated."""
    out = agent.query_data("daily", aggregate={"plant_stress_index": "mean"}, time_bucket="week")
    assert len(out["result"]) > 20            # ~30 weeks, not capped at 20
    assert "no row cap" in out["notes"]


def test_time_bucket_requires_date_column():
    with pytest.raises(ValueError):
        agent.query_data("season", aggregate={"season_profit_cad": "mean"}, time_bucket="month")


def test_invalid_table_raises_actionable():
    with pytest.raises(ValueError) as e:
        agent.query_data("not_a_table")
    msg = str(e.value)
    assert "not_a_table" in msg and "season" in msg  # lists valid tables


def test_invalid_column_raises_actionable():
    """Unknown columns raise a self-explanatory error so the model fixes it in one retry."""
    with pytest.raises(ValueError) as e:
        agent.query_data("season", filters={"made_up_column": 5})
    msg = str(e.value)
    assert "made_up_column" in msg and "Available columns" in msg


def test_invalid_agg_func_raises_actionable():
    with pytest.raises(ValueError) as e:
        agent.query_data("season", group_by=["crop"], aggregate={"season_profit_cad": "stdev"})
    msg = str(e.value)
    assert "stdev" in msg and "mean" in msg  # lists supported funcs


def test_aggregate_collision_with_groupby():
    """FIX B: aggregating a column that is also a group_by key is handled gracefully."""
    # crop is both grouped and aggregated -> drop the crop agg with a clear note,
    # keep the valid profit aggregate.
    out = agent.query_data(
        "season", group_by=["treatment", "crop"],
        aggregate={"season_profit_cad": "mean", "crop": "count"},
    )
    assert "Cannot aggregate column 'crop'" in out["notes"]
    assert "season_profit_cad" in out["columns_used"]
    assert len(out["result"]) > 0

    # When the ONLY aggregate collides, fall back to a row count per group.
    out2 = agent.query_data("season", group_by=["crop"], aggregate={"crop": "count"})
    assert "count" in out2["columns_used"]
    assert any(r.get("count", 0) > 0 for r in out2["result"])

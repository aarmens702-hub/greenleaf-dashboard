import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat import data_queries as dq


def test_season_overview():
    o = dq.season_overview()
    assert o["n_plots"] == 120
    assert o["total_profit_cad"] > 0


def test_precision_benefit_summary():
    s = dq.precision_benefit_summary()
    assert 30 <= s["precision_benefit_share_pct"] <= 90


def test_response_time_payoff():
    df = dq.response_time_payoff()
    assert len(df) >= 2
    assert df.iloc[0]["mean_stress_recovery"] < df.iloc[-1]["mean_stress_recovery"]


def test_avg_profit_by_crop():
    d = dq.avg_profit_by_crop()
    assert set(d.keys()) == {"Tomato", "Pepper", "Cucumber", "Strawberry"}


def test_treatment_summary_all():
    df = dq.treatment_summary()
    assert len(df) == 7
    assert list(df["mean_profit_cad"]) == sorted(df["mean_profit_cad"], reverse=True)


def test_treatment_summary_bad_crop():
    with pytest.raises(ValueError):
        dq.treatment_summary(crop="Dragonfruit")


def test_plot_status_ok():
    s = dq.plot_status("P0001")
    assert s["plot_id"] == "P0001"
    assert len(s["last_n_days"]) > 0


def test_plot_status_bad():
    with pytest.raises(ValueError):
        dq.plot_status("DOESNOTEXIST")


def test_top_at_risk_plots():
    min_date = dq.DAILY["date"].min().strftime("%Y-%m-%d")
    df = dq.top_at_risk_plots(min_date, k=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_top_at_risk_plots_bad_window():
    with pytest.raises(ValueError):
        dq.top_at_risk_plots("1900-01-01")


def test_greenleaf_vs_benchmark_ok():
    r = dq.greenleaf_vs_benchmark("greenhouse_tomato_annual_yield")
    assert isinstance(r["benchmark_value"], float)
    assert isinstance(r["greenleaf_value"], float)


def test_greenleaf_vs_benchmark_bad():
    with pytest.raises(ValueError):
        dq.greenleaf_vs_benchmark("does_not_exist")


def test_greenleaf_vs_benchmark_financial_gain():
    """The tomato yield metric returns a fully-labeled, internally-consistent
    financial-gain block (yield-gap monetized at GreenLeaf's own price)."""
    r = dq.greenleaf_vs_benchmark("greenhouse_tomato_annual_yield")
    fg = r["financial_gain"]
    # All labeled components present.
    for key in ("per_cycle_yield_kg_m2", "cycles_per_year_assumed",
                "annualized_yield_kg_m2", "yield_gap_kg_m2_year",
                "revenue_per_kg_cad", "financial_gap_cad_per_m2_year",
                "assumptions", "interpretation"):
        assert key in fg
    assert fg["cycles_per_year_assumed"] == 3
    assert fg["revenue_per_kg_cad"] > 0
    # Internally consistent: annualized = per-cycle * cycles.
    assert abs(fg["annualized_yield_kg_m2"] - fg["per_cycle_yield_kg_m2"] * 3) < 1e-2
    # Gap = annualized - benchmark midpoint.
    assert abs(fg["yield_gap_kg_m2_year"] - (fg["annualized_yield_kg_m2"] - r["benchmark_value"])) < 1e-2
    # Financial gap = yield gap * revenue per kg.
    assert abs(fg["financial_gap_cad_per_m2_year"]
               - fg["yield_gap_kg_m2_year"] * fg["revenue_per_kg_cad"]) < 1e-1


def test_cost_breakdown():
    c = dq.cost_breakdown()
    for key in (
        "mean_daily_energy_cost",
        "mean_daily_fertilizer_cost",
        "mean_daily_labor_cost",
        "mean_daily_pesticide_cost",
        "mean_daily_water_cost",
        "mean_daily_total_input_cost",
    ):
        assert isinstance(c[key], float)


def test_alert_response_summary():
    a = dq.alert_response_summary()
    assert 0 <= a["action_rate_pct"] <= 100

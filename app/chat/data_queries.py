"""Stage 2 of the GreenLeaf chat: 10 pure-pandas query functions over the derived
parquet files. No AI, no model. Reused by the chat (Stage 3) and the dashboard tabs.

Each parquet is read ONCE at import into a module-global DataFrame. All functions
return JSON-serializable values (numpy types converted to Python). Monetary values
round to 2 dp, percentages to 1 dp, ratios/stress to 3 dp.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data" / "derived"

# Read each parquet ONCE at module load.
SEASON = pd.read_parquet(DERIVED / "season.parquet")
DAILY = pd.read_parquet(DERIVED / "daily.parquet")
ACTIONS = pd.read_parquet(DERIVED / "actions.parquet")
SCOUTING = pd.read_parquet(DERIVED / "scouting.parquet")
PLOT_META = pd.read_parquet(DERIVED / "plot_meta.parquet")
BENCHMARKS = pd.read_parquet(DERIVED / "bc_benchmarks.parquet")


def _money(x):
    return round(float(x), 2)


def _pct(x):
    return round(float(x), 1)


def _ratio(x):
    return round(float(x), 3)


def season_overview():
    """Whole-operation summary across all plots.

    Returns: dict with farm/greenhouse/plot/crop counts and season totals.
    Example: season_overview()["n_plots"] -> 120
    """
    total_rev = SEASON["season_revenue_cad"].sum()
    total_cost = SEASON["total_cost_cad"].sum()
    total_profit = SEASON["season_profit_cad"].sum()
    crops = sorted(PLOT_META["crop"].unique().tolist())
    return {
        "n_farms": int(PLOT_META["farm_id"].nunique()),
        "n_greenhouses": int(PLOT_META["greenhouse_id"].nunique()),
        "n_plots": int(len(PLOT_META)),
        "n_crops": int(len(crops)),
        "crops": crops,
        "total_revenue_cad": _money(total_rev),
        "total_cost_cad": _money(total_cost),
        "total_profit_cad": _money(total_profit),
        "overall_roi": _ratio((total_rev - total_cost) / total_cost),
        "mean_marketable_ratio": _ratio(SEASON["marketable_ratio"].mean()),
        "mean_yield_kg_m2": _ratio(SEASON["season_yield_kg_m2"].mean()),
    }


def precision_benefit_summary():
    """Season-wide value attributed to precision management.

    Returns: dict with totals, per-plot central tendency, and the precision
    share of total profit (%).
    Example: precision_benefit_summary()["precision_benefit_share_pct"] -> 64.9
    """
    pb = SEASON["precision_benefit_cad"]
    total_precision = pb.sum()
    total_profit = SEASON["season_profit_cad"].sum()
    return {
        "n_plots": int(len(SEASON)),
        "total_precision_benefit_cad": _money(total_precision),
        "mean_per_plot_cad": _money(pb.mean()),
        "median_per_plot_cad": _money(pb.median()),
        "total_season_profit_cad": _money(total_profit),
        "precision_benefit_share_pct": _pct(100 * total_precision / total_profit),
    }


def response_time_payoff():
    """How fast response relates to short-run stress recovery.

    From DAILY where action_taken == 1, group by action_delay_days; keep delays
    with n_actions >= 5, sorted ascending.
    Returns: pd.DataFrame[action_delay_days, n_actions, mean_stress_recovery]
    (mean_stress_recovery is post_action_stress_delta_3d; more negative = better).
    Example: response_time_payoff().iloc[0]["mean_stress_recovery"] -> -0.048
    """
    acted = DAILY[DAILY["action_taken"] == 1].dropna(subset=["action_delay_days"])
    g = (
        acted.groupby("action_delay_days")
        .agg(
            n_actions=("post_action_stress_delta_3d", "size"),
            mean_stress_recovery=("post_action_stress_delta_3d", "mean"),
        )
        .reset_index()
    )
    g = g[g["n_actions"] >= 5].sort_values("action_delay_days").reset_index(drop=True)
    g["action_delay_days"] = g["action_delay_days"].astype(int)
    g["n_actions"] = g["n_actions"].astype(int)
    g["mean_stress_recovery"] = g["mean_stress_recovery"].round(3)
    return g


def avg_profit_by_crop():
    """Mean season profit per plot, by crop.

    Returns: dict {crop: mean_profit_cad}
    Example: avg_profit_by_crop()["Strawberry"] -> 678.69
    """
    s = SEASON.groupby("crop")["season_profit_cad"].mean()
    return {str(k): _money(v) for k, v in s.items()}


def treatment_summary(crop=None):
    """Per-treatment outcome table, optionally filtered to one crop.

    Args: crop (str|None) — case-insensitive crop filter.
    Returns: pd.DataFrame[treatment, n_plots, mean_yield_kg_m2,
             mean_marketable_ratio, mean_profit_cad, mean_roi], sorted by
             mean_profit_cad descending.
    Raises: ValueError if `crop` is given but matches no plots.
    Example: treatment_summary("Strawberry").iloc[0]["treatment"]
    """
    df = SEASON
    if crop is not None:
        df = df[df["crop"].str.lower() == crop.lower()]
        if df.empty:
            raise ValueError(
                f"No plots for crop '{crop}'. Available: {sorted(SEASON['crop'].unique())}"
            )
    g = (
        df.groupby("treatment")
        .agg(
            n_plots=("plot_id", "size"),
            mean_yield_kg_m2=("season_yield_kg_m2", "mean"),
            mean_marketable_ratio=("marketable_ratio", "mean"),
            mean_profit_cad=("season_profit_cad", "mean"),
            mean_roi=("season_roi", "mean"),
        )
        .reset_index()
        .sort_values("mean_profit_cad", ascending=False)
        .reset_index(drop=True)
    )
    g["n_plots"] = g["n_plots"].astype(int)
    g["mean_yield_kg_m2"] = g["mean_yield_kg_m2"].round(3)
    g["mean_marketable_ratio"] = g["mean_marketable_ratio"].round(3)
    g["mean_profit_cad"] = g["mean_profit_cad"].round(2)
    g["mean_roi"] = g["mean_roi"].round(3)
    return g


def plot_status(plot_id, last_n_days=14):
    """Single-plot snapshot: metadata, season financials, recent daily readings.

    Args: plot_id (str); last_n_days (int) — most recent N daily rows by date.
    Returns: dict with meta + season financials + 'last_n_days' (list of daily dicts).
    Raises: ValueError if plot_id is unknown.
    Example: plot_status("P0001")["crop"] -> 'Strawberry'
    """
    meta_row = PLOT_META[PLOT_META["plot_id"] == plot_id]
    if meta_row.empty:
        raise ValueError(f"plot_id '{plot_id}' not found (example valid id: 'P0001').")
    m = meta_row.iloc[0]
    daily = DAILY[DAILY["plot_id"] == plot_id].sort_values("date").tail(last_n_days)
    recent = [
        {
            "date": d["date"].strftime("%Y-%m-%d"),
            "plant_stress_index": _ratio(d["plant_stress_index"]),
            "alert_flag": int(d["alert_flag"]),
            "action_taken": int(d["action_taken"]),
            "daily_total_input_cost": _money(d["daily_total_input_cost"]),
        }
        for _, d in daily.iterrows()
    ]
    out = {
        "plot_id": plot_id,
        "crop": str(m["crop"]),
        "treatment": str(m["treatment"]),
        "greenhouse_id": str(m["greenhouse_id"]),
        "farm_id": str(m["farm_id"]),
    }
    srow = SEASON[SEASON["plot_id"] == plot_id]
    if not srow.empty:
        s = srow.iloc[0]
        out["season_revenue_cad"] = _money(s["season_revenue_cad"])
        out["total_cost_cad"] = _money(s["total_cost_cad"])
        out["season_profit_cad"] = _money(s["season_profit_cad"])
        out["season_roi"] = _ratio(s["season_roi"])
    out["last_n_days"] = recent
    return out


def top_at_risk_plots(week_start, k=5):
    """Plots most at risk in a 7-day window, for Monday-morning triage.

    Args: week_start ('YYYY-MM-DD'); window is [week_start, week_start+6 days]. k (int).
    Returns: pd.DataFrame[plot_id, crop, treatment, mean_stress, max_stress,
             any_spike_3d, n_alerts], ranked by (any_spike_3d desc, mean_stress desc).
    Raises: ValueError if the window contains no daily rows.
    Example: top_at_risk_plots("2025-04-01", k=5)
    """
    start = pd.Timestamp(week_start)
    end = start + pd.Timedelta(days=6)
    window = DAILY[(DAILY["date"] >= start) & (DAILY["date"] <= end)]
    if window.empty:
        raise ValueError(f"No daily rows in window [{start.date()} .. {end.date()}].")
    g = (
        window.groupby("plot_id")
        .agg(
            mean_stress=("plant_stress_index", "mean"),
            max_stress=("plant_stress_index", "max"),
            any_spike_3d=("spike_3d", lambda s: bool((s == 1).any())),
            n_alerts=("alert_flag", "sum"),
        )
        .reset_index()
        .merge(PLOT_META[["plot_id", "crop", "treatment"]], on="plot_id", how="left")
        .sort_values(["any_spike_3d", "mean_stress"], ascending=[False, False])
        .head(k)
        .reset_index(drop=True)
    )
    g["mean_stress"] = g["mean_stress"].round(3)
    g["max_stress"] = g["max_stress"].round(3)
    g["n_alerts"] = g["n_alerts"].astype(int)
    g["any_spike_3d"] = g["any_spike_3d"].astype(bool)
    return g[["plot_id", "crop", "treatment", "mean_stress", "max_stress", "any_spike_3d", "n_alerts"]]


# GreenLeaf-side calculations per benchmark metric. Extend as benchmarks are added.
_GREENLEAF_METRICS = {
    "greenhouse_tomato_annual_yield": lambda: SEASON.loc[
        SEASON["crop"] == "Tomato", "season_yield_kg_m2"
    ].mean(),
}


def _tomato_financial_gain(benchmark_midpoint, cycles_per_year):
    """Translate the tomato yield gap vs the BC benchmark into a $/m²/year estimate.

    Method (every assumption stated so the number is defensible, not magic):
    - GreenLeaf's measured yield is per partial-season microplot cycle; we annualize
      it by multiplying by cycles_per_year (assumed, default 3 cycles/year).
    - The yield gap is annualized GreenLeaf yield minus the benchmark midpoint.
    - We monetize the gap at GreenLeaf's OWN realized tomato revenue per kg
      (sum tomato revenue / sum tomato kg), not an external price.
    A negative financial gap means GreenLeaf is below the commercial benchmark.
    """
    tom = SEASON[SEASON["crop"] == "Tomato"]
    per_cycle = tom["season_yield_kg_m2"].mean()
    annualized = per_cycle * cycles_per_year
    yield_gap = annualized - benchmark_midpoint
    tom_kg = (tom["season_yield_kg_m2"] * tom["plot_area_m2"]).sum()
    revenue_per_kg = tom["season_revenue_cad"].sum() / tom_kg
    financial_gap = yield_gap * revenue_per_kg
    direction = "above" if yield_gap >= 0 else "below"
    return {
        "per_cycle_yield_kg_m2": _ratio(per_cycle),
        "cycles_per_year_assumed": int(cycles_per_year),
        "annualized_yield_kg_m2": _ratio(annualized),
        "benchmark_midpoint_kg_m2_year": _ratio(benchmark_midpoint),
        "yield_gap_kg_m2_year": _ratio(yield_gap),
        "revenue_per_kg_cad": _money(revenue_per_kg),
        "financial_gap_cad_per_m2_year": _money(financial_gap),
        "assumptions": (
            f"Annualized at {int(cycles_per_year)} cycles/year (assumed); gap monetized "
            "at GreenLeaf's own realized tomato revenue per kg; benchmark is the 50-70 "
            "kg/m²/year midpoint for commercial operations, not microplot trials."
        ),
        "interpretation": (
            f"Annualized GreenLeaf tomato yield (~{_ratio(annualized)} kg/m²/yr) is "
            f"{direction} the benchmark midpoint ({_ratio(benchmark_midpoint)}), implying "
            f"about {_money(financial_gap)} CAD/m²/yr in revenue terms at GreenLeaf's own "
            "price — an estimate sensitive to the cycles-per-year assumption."
        ),
    }


def greenleaf_vs_benchmark(metric_name):
    """Compare a GreenLeaf metric against an external BC/industry benchmark.

    Args: metric_name (str) — must exist in BENCHMARKS.
    Returns: dict with benchmark value/unit/source/url, greenleaf_value, and notes.
    Raises: ValueError if metric_name is not a known benchmark.
    Example: greenleaf_vs_benchmark("greenhouse_tomato_annual_yield")["benchmark_value"]
    """
    row = BENCHMARKS[BENCHMARKS["metric_name"] == metric_name]
    if row.empty:
        raise ValueError(
            f"Benchmark '{metric_name}' not found. Available: {sorted(BENCHMARKS['metric_name'].tolist())}"
        )
    b = row.iloc[0]
    financial_gain = None
    if metric_name in _GREENLEAF_METRICS:
        greenleaf_value = _ratio(_GREENLEAF_METRICS[metric_name]())
        notes = (
            "GreenLeaf value = mean season_yield_kg_m2 over Tomato plots. Note: the "
            "benchmark is annual commercial yield while GreenLeaf is partial-season "
            "microplot yield — not directly comparable without normalization."
        )
        if metric_name == "greenhouse_tomato_annual_yield":
            # cycles_per_year is read from the benchmark if present, else defaults to 3.
            cycles = (
                int(b["cycles_per_year"])
                if "cycles_per_year" in b.index and pd.notna(b["cycles_per_year"])
                else 3
            )
            financial_gain = _tomato_financial_gain(float(b["value"]), cycles)
    else:
        greenleaf_value = None
        notes = "No GreenLeaf-side calculation defined yet for this metric."
    out = {
        "metric_name": str(b["metric_name"]),
        "benchmark_value": _ratio(b["value"]),
        "benchmark_unit": str(b["unit"]),
        "benchmark_source": str(b["source_name"]),
        "benchmark_url": str(b["source_url"]),
        "greenleaf_value": greenleaf_value,
        "notes": notes,
    }
    if financial_gain is not None:
        out["financial_gain"] = financial_gain
    return out


def cost_breakdown(crop=None, treatment=None):
    """Mean daily input cost per plot-day, optionally filtered by crop/treatment.

    Args: crop (str|None), treatment (str|None) — case-insensitive filters.
    Returns: dict with the 6 mean daily cost components, filter context, n_plot_days.
    Raises: ValueError if the filters yield no rows.
    Example: cost_breakdown(crop="Tomato")["mean_daily_total_input_cost"]
    """
    df = DAILY  # DAILY already carries crop and treatment (joined in build_derived).
    if crop is not None:
        df = df[df["crop"].str.lower() == crop.lower()]
    if treatment is not None:
        df = df[df["treatment"].str.lower() == treatment.lower()]
    if df.empty:
        raise ValueError(f"No plot-days for crop={crop!r}, treatment={treatment!r}.")
    return {
        "crop_filter": crop or "all",
        "treatment_filter": treatment or "all",
        "n_plot_days": int(len(df)),
        "mean_daily_energy_cost": _money(df["daily_energy_cost"].mean()),
        "mean_daily_fertilizer_cost": _money(df["daily_fertilizer_cost"].mean()),
        "mean_daily_labor_cost": _money(df["daily_labor_cost"].mean()),
        "mean_daily_pesticide_cost": _money(df["daily_pesticide_cost"].mean()),
        "mean_daily_water_cost": _money(df["daily_water_cost"].mean()),
        "mean_daily_total_input_cost": _money(df["daily_total_input_cost"].mean()),
    }


def alert_response_summary(crop=None):
    """How alerts were handled: volume, action rate, speed, recovery.

    Args: crop (str|None) — case-insensitive crop filter.
    Returns: dict with n_alerts, n_acted_on, action_rate_pct, mean_delay_days,
             mean_recovery_3d, crop_filter.
    Raises: ValueError if `crop` is given but matches no rows.
    Example: alert_response_summary()["action_rate_pct"] -> 74.6
    """
    df = DAILY  # DAILY already carries crop (joined in build_derived).
    if crop is not None:
        df = df[df["crop"].str.lower() == crop.lower()]
        if df.empty:
            raise ValueError(f"No rows for crop '{crop}'.")
    alerts = df[df["alert_flag"] == 1]
    n_alerts = int(len(alerts))
    if n_alerts == 0:
        return {
            "n_alerts": 0,
            "n_acted_on": 0,
            "action_rate_pct": 0.0,
            "mean_delay_days": None,
            "mean_recovery_3d": None,
            "crop_filter": crop or "all",
        }
    acted = alerts[alerts["action_taken"] == 1]
    n_acted = int(len(acted))
    return {
        "n_alerts": n_alerts,
        "n_acted_on": n_acted,
        "action_rate_pct": _pct(100 * n_acted / n_alerts),
        "mean_delay_days": _ratio(acted["action_delay_days"].mean()) if n_acted else None,
        "mean_recovery_3d": _ratio(acted["post_action_stress_delta_3d"].mean()) if n_acted else None,
        "crop_filter": crop or "all",
    }

"""Financial aggregations for the Financials dashboard tab.

Reuses analysis/eda.py (single source for the per-plot merged frame + bootstrap CIs) so
every figure reproduces and matches docs/eda_findings.md. Read-only.

Findings this surfaces (from a full financial pass):
  - Profit is a PRICE/REVENUE story, not a yield story: corr(profit, revenue)=0.90,
    corr(profit, price/kg)=0.61, but corr(profit, yield)=-0.15. drivers() exposes this.
  - Plot area does not drive profit (r=-0.10) — absolute profit is fair to compare.
  - Strawberry leads on margin/price despite the lowest yield. by_crop() shows unit econ.
  - Labor is ~63% of input cost. cost_structure().
  - High Light = highest mean profit AND highest variance; 6 plots lost money. risk_return().
  - precision_benefit_cad is ~0.65 x profit floored at 0 (corr 0.996) — an ASSUMPTION;
    season vs daily cost ledgers don't reconcile (corr -0.21) -> no defensible precision ROI.
"""
from functools import lru_cache
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import eda  # noqa: E402


@lru_cache(maxsize=1)
def _daily():
    return eda.load()["daily"]


@lru_cache(maxsize=1)
def _m():
    m = eda.merged().copy()
    kg = m.season_yield_kg_m2 * m.plot_area_m2          # total kg produced
    m["profit_per_m2"] = m.season_profit_cad / m.plot_area_m2
    m["margin"] = m.season_profit_cad / m.season_revenue_cad
    m["price_per_kg"] = m.season_revenue_cad / kg
    m["cost_per_kg"] = m.total_cost_cad / kg
    return m


def headline():
    m = _m()
    return {
        "n_plots": int(len(m)),
        "total_profit": float(m.season_profit_cad.sum()),
        "mean_profit": float(m.season_profit_cad.mean()),
        "median_profit": float(m.season_profit_cad.median()),
        "mean_roi": float(m.season_roi.mean()),
        "mean_margin": float(m.margin.mean()),
        "total_revenue": float(m.season_revenue_cad.sum()),
        "total_cost": float(m.total_cost_cad.sum()),
        "n_unprofitable": int((m.season_profit_cad < 0).sum()),
        "total_precision_benefit": float(m.precision_benefit_cad.sum()),
        "benefit_share": float(m.precision_benefit_cad.sum() / m.season_profit_cad.sum()),
        "mean_benefit": float(m.precision_benefit_cad.mean()),
    }


def drivers():
    """What correlates with per-plot profit — the headline 'price not yield' insight."""
    m = _m()
    factors = {"Revenue": "season_revenue_cad", "Price / kg": "price_per_kg",
               "Total cost": "total_cost_cad", "Marketable %": "marketable_ratio",
               "Yield (kg/m²)": "season_yield_kg_m2", "Plant stress": "mean_stress"}
    rows = [(label, float(m.season_profit_cad.corr(m[col]))) for label, col in factors.items()]
    return pd.DataFrame(rows, columns=["factor", "r"]).sort_values("r")


def by_treatment():
    m = _m()
    g = (m.groupby("treatment").agg(
        profit=("season_profit_cad", "mean"), roi=("season_roi", "mean"),
        margin=("margin", "mean"), revenue=("season_revenue_cad", "mean"),
        cost=("total_cost_cad", "mean"), n=("plot_id", "size")).reset_index().sort_values("profit"))
    gap = eda.boot_mean_diff(m[m.treatment == "High Light"].season_profit_cad,
                             m[m.treatment == "Shade"].season_profit_cad)
    return g, gap


def by_crop():
    m = _m()
    g = (m.groupby("crop").agg(
        profit=("season_profit_cad", "mean"), profit_m2=("profit_per_m2", "mean"),
        margin=("margin", "mean"), price_kg=("price_per_kg", "mean"),
        yield_kgm2=("season_yield_kg_m2", "mean"), n=("plot_id", "size"))
        .reset_index().sort_values("profit"))
    return g


def cost_structure():
    daily = _daily()
    cats = {"Labor": "daily_labor_cost", "Energy": "daily_energy_cost",
            "Fertilizer": "daily_fertilizer_cost", "Water": "daily_water_cost",
            "Pesticide": "daily_pesticide_cost"}
    df = (pd.DataFrame([(k, float(daily[v].sum())) for k, v in cats.items()],
                       columns=["category", "total"]).sort_values("total"))
    return df, float(daily.daily_precision_cost.sum()), float(daily.daily_routine_cost.sum())


def risk_return():
    """Mean profit (return) vs std (risk) by treatment — for the risk-return scatter."""
    m = _m()
    return (m.groupby("treatment").agg(
        mean_profit=("season_profit_cad", "mean"), risk=("season_profit_cad", "std"),
        worst=("season_profit_cad", "min"), n=("plot_id", "size")).reset_index())


def loss_makers():
    """The plots that ran at a loss — the downside a lender cares about."""
    m = _m()
    cols = ["plot_id", "crop", "treatment", "season_revenue_cad", "total_cost_cad",
            "season_profit_cad", "season_roi"]
    return m[m.season_profit_cad < 0][cols].sort_values("season_profit_cad").reset_index(drop=True)


def top_earners(n=5):
    m = _m()
    cols = ["plot_id", "crop", "treatment", "season_revenue_cad", "total_cost_cad",
            "season_profit_cad", "season_roi"]
    return m.nlargest(n, "season_profit_cad")[cols].reset_index(drop=True)


def profit_distribution():
    return _m()[["plot_id", "season_profit_cad", "season_roi", "margin", "crop", "treatment"]].copy()


def precision_benefit_check():
    m = _m()
    pos = m[m.season_profit_cad > 0]
    return {
        "corr_benefit_profit": float(m.precision_benefit_cad.corr(m.season_profit_cad)),
        "median_ratio": float((pos.precision_benefit_cad / pos.season_profit_cad).median()),
        "corr_ledger": float(m.total_cost_cad.corr(m.total_input_cost)),
        "scatter": m[["precision_benefit_cad", "season_profit_cad", "treatment"]].copy(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(headline(), indent=2))
    print("\ndrivers:\n", drivers().round(2).to_string(index=False))
    print("\nby_crop:\n", by_crop().round(2).to_string(index=False))
    print("\nrisk_return:\n", risk_return().round(0).to_string(index=False))
    print("\nloss_makers:\n", loss_makers().round(0).to_string(index=False))

"""Traceability guard for the read-only insight helpers.

Confirms each dashboard number reproduces the validated Part-1 menu (i.e. comes straight
from the frozen analysis layer, not from UI-layer recomputation/invention)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.streamlit_app import insights as I


def test_response_dose_matches_eda():
    by = I.response_dose()["by"].set_index("delay")
    assert abs(by.loc[0, "mean"] + 0.0494) < 0.001
    assert abs(by.loc[1, "mean"] + 0.0291) < 0.001
    assert abs(by.loc[2, "mean"] + 0.0112) < 0.001
    assert abs(by.loc[0, "share"] - 0.428) < 0.01


def test_driver_substrate_is_top_env():
    di = I.driver_importance()
    env = di[di.env].reset_index(drop=True)
    assert env.iloc[0].feature == "substrate_moisture"
    assert abs(float(env.iloc[0].mean_abs) - 0.64) < 0.05


def test_cost_labor_dominates():
    cm = I.cost_mix()
    assert abs(cm["labor_pct"] - 62.97) < 0.5
    assert cm["precision_pct"] < 6.0


def test_profit_tail():
    pt = I.profit_tail()
    assert abs(pt["es5"] + 123.06) < 2
    assert abs(pt["gini_profit"] - 0.311) < 0.02
    assert len(pt["loss_makers"]) == 6
    assert round(pt["headline"]["total_profit"]) == 71654


def test_gains_curve():
    g = I.gains_curve()
    assert abs(g["top20"] - 0.442) < 0.02
    assert abs(g["recall"] - 0.759) < 0.02
    assert g["n_test"] == 4800


def test_treatment_economics_hl_ci():
    te = I.treatment_economics()
    hl = te["rows"][te["rows"].treatment == "High Light"].iloc[0]
    assert abs(hl["diff"] - 259.9) < 5
    assert hl["lo"] > 0  # CI excludes zero


def test_seasonal_timing_dominates():
    w = I.weekly_stress_trajectory()
    assert w["eta_timing"] > w["eta_treat"]    # timing dominates treatment
    assert abs(w["spread"] - 0.142) < 0.01


def test_alert_exposure_tautology_flagged():
    a = I.alert_exposure()
    assert a["corr_stress"] > 0.98             # the near-tautology we caveat


def test_delay_treatment_relief_matches_eda():
    import numpy as np
    mean, cnt = I.delay_treatment_relief()
    assert mean.shape == (3, 7) and cnt.shape == (3, 7)
    assert (cnt.values >= 30).all()            # every cell well-sampled (min ~86)
    # n-weighted row means reconcile with eda's overall by-delay (same event definition)
    overall = (mean.values * cnt.values).sum(axis=1) / cnt.values.sum(axis=1)
    for got, want in zip(overall, [-0.0494, -0.0291, -0.0112]):
        assert abs(got - want) < 0.001


def test_recommendation_adherence():
    g, overall, orelief = I.recommendation_adherence()
    assert len(g) == 3
    assert abs(overall - 0.477) < 0.01                 # ~48% acted on overall
    assert g.adhere.min() < 0.40 and g.adhere.max() > 0.50   # 34%–52% spread
    assert (g.n >= 30).all()                           # all well-sampled (min ~1,888)
    assert orelief < 0                                 # acting reduces stress (~-0.034)


def test_greenhouse_ops_well_sampled():
    g, overall = I.greenhouse_ops()
    assert len(g) == 20                                   # 20 greenhouses
    assert g.resp_pct.between(0, 100).all()
    assert (g.alert_days >= 30).all()                     # every GH well-sampled (min ~103)
    assert 0.70 < overall < 0.80                          # overall response ~74.6%
    assert g.resp_pct.max() - g.resp_pct.min() > 10       # real spread (57%–90%)


def test_clean_plot_score_is_decontaminated():
    """Step 1: dropping the profit term pulls the score's profit-correlation from -0.69 down low."""
    from analysis import eda
    cs = I.clean_plot_score()
    assert set(["date", "plot_id", "plot_health", "clean_score"]).issubset(cs.columns)
    assert 0 <= cs.clean_score.min() and cs.clean_score.max() <= 90.001
    season = eda.load()["season"][["plot_id", "season_profit_cad"]]
    perplot = cs.groupby("plot_id").clean_score.mean().reset_index().merge(season, on="plot_id")
    r = perplot.clean_score.corr(perplot.season_profit_cad)
    assert abs(r) < 0.5                              # no longer profit-contaminated (was -0.69)
    assert abs(r - 0.332) < 0.05                     # matches the reported +0.33


def test_headline_kpis():
    k = I.headline_kpis()
    assert abs(k["avg_performance"] - 75.7) < 1.0    # 0-90 profit-free score
    assert abs(k["avg_health"] - 0.635) < 0.02
    assert abs(k["avg_roi"] - 0.428) < 0.02
    assert abs(k["weekly_response_rate"] - 0.75) < 0.05
    assert abs(k["profit_per_m2"] - 36.41) < 0.5
    assert (k["week_end"] - k["week_start"]).days == 6


def test_weekly_spend_last7():
    ws = I.weekly_spend()
    assert len(ws["series"]) == 7
    assert (ws["series"].date.max() - ws["series"].date.min()).days == 6
    assert ws["total"] > 0


def test_finance_kpis():
    k = I.finance_kpis()
    assert abs(k["profit_per_m2"] - 36.41) < 0.5
    assert abs(k["avg_roi"] - 0.428) < 0.02
    assert round(k["total_precision_benefit"]) == 47034     # the 0.65×profit assumption figure
    assert abs(k["cost_per_kg"] - 12.76) < 0.5


def test_composite_risk_distinct_from_spike_model():
    """Composite risk (all_scores_complete.risk_score_90, 10-90) and the spike model must NOT be
    near-duplicates."""
    day, target = I.composite_risk_today("2025-08-15")
    assert len(day) == 120
    assert day.risk_score.is_monotonic_decreasing
    assert 10 <= day.risk_score.min() and day.risk_score.max() <= 90   # 90-scale source, not 0-100
    rec = I.risk_reconciliation("2025-08-15")
    assert abs(rec["corr"]) < 0.5                    # genuinely different signal
    assert rec["overlap_top10"] <= 3                 # rankings barely overlap


def test_profit_drivers_price_not_yield():
    d = I.profit_drivers()
    r = d["corr"].set_index("factor").r
    assert r["Revenue"] > 0.85 and r["Price / kg"] > 0.55   # price drives profit
    assert r["Yield (kg/m²)"] < 0                           # yield doesn't
    assert d["scatter"].shape == (120, 3)


def test_crop_economics_strawberry_price_premium():
    g = I.crop_economics().set_index("crop")
    assert g.loc["Strawberry", "profit"] == g["profit"].max()       # earns the most
    assert g.loc["Strawberry", "price_kg"] == g["price_kg"].max()   # at the highest price
    assert g.loc["Strawberry", "yield_kgm2"] == g["yield_kgm2"].min()  # on the lowest yield


def test_treatment_spread_high_light():
    rr = I.treatment_spread().set_index("treatment")
    assert rr.loc["High Light", "mean_profit"] == rr["mean_profit"].max()  # highest payoff
    assert rr.loc["High Light", "risk"] == rr["risk"].max()                # widest spread
    assert rr.loc["Shade", "worst"] == rr["worst"].min()                   # deepest single loss


def test_loss_diagnosis_price_not_cost():
    d = I.loss_diagnosis()
    assert d["n"] == 6
    assert d["n_below_p10_price"] == 5          # 5 of 6 in the bottom 10% of price for their crop
    assert d["med_price_pctile"] < 15           # losers' price is far below their crop's median
    assert 40 < d["med_cost_pctile"] < 60       # losers' cost is mid-pack (~52nd pct)
    assert len(I.top_earners()) == 5


def test_operational_kpis_match_expected():
    k = I.operational_kpis()
    assert abs(k["healthy_rate"] - 0.6068) < 0.002          # stress < 0.50
    assert abs(k["alert_rate"] - 0.2563) < 0.002            # alert_flag == 1
    assert abs(k["alert_response_rate"] - 0.7461) < 0.002   # action | alert
    assert abs(k["critical_rate"] - 0.0744) < 0.002         # stress >= 0.65
    assert abs(k["performance_score"] - 0.6412) < 0.002     # plot_score.final_score_scaled mean
    # bands partition the plot-days
    assert abs(k["healthy_rate"] + k["watch_rate"] + k["critical_rate"] - 1.0) < 1e-6

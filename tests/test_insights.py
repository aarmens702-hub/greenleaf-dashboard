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

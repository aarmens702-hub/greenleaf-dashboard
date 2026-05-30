"""Phase-2B risk model tests: the saved logistic scores, and its signed feature
contributions decompose the calibrated probability exactly (no fabrication)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import risk_score as rs

DATE = "2025-08-15"


def test_model_loads_and_scores_all_plots():
    rs.load_model.cache_clear()
    rs._daily.cache_clear()
    wl = rs.score_plots(DATE)
    assert len(wl) == 120                      # one row per plot on the date
    assert set(["plot_id", "crop", "treatment", "greenhouse_id",
                "spike_prob", "top_driver"]).issubset(wl.columns)
    assert wl.spike_prob.between(0, 1).all()   # calibrated probabilities
    # sorted by risk, descending
    assert wl.spike_prob.is_monotonic_decreasing
    assert wl.top_driver.notna().all()


def test_contributions_reconstruct_probability():
    """sigmoid(sum(parent contributions) + intercept) must equal predict_proba."""
    art = rs.load_model()
    clf = art["model"].named_steps["clf"]
    intercept = float(clf.intercept_[0])
    pid = rs.score_plots(DATE).iloc[0].plot_id

    contrib = rs.plot_contributions(pid, DATE)
    prob = rs.plot_probability(pid, DATE)
    logit = contrib.contribution.sum() + intercept
    assert abs(1 / (1 + np.exp(-logit)) - prob) < 1e-6


def test_top_driver_is_environmental():
    """The watch-list headline driver is an actionable environmental lever
    (not the mean-reversion baseline), per the design."""
    wl = rs.score_plots(DATE)
    top_phrase = wl.iloc[0].top_driver
    assert top_phrase != "environment not elevating risk"
    # phrase derives from an ENVIRONMENTAL feature's label
    env_labels = {rs.FEATURE_LABELS[f] for f in rs.ENVIRONMENTAL_DRIVERS}
    assert any(lbl in top_phrase for lbl in env_labels)


def test_recent_trend_window():
    pid = rs.score_plots(DATE).iloc[0].plot_id
    tr = rs.recent_trend(pid, DATE, days=21)
    assert 1 <= len(tr) <= 21
    assert list(tr.columns) == ["date", "plant_stress_index", "substrate_moisture"]
    assert tr.date.is_monotonic_increasing


def test_validation_metrics_match_findings():
    """Saved metrics reproduce docs/spike_model_findings.md (logistic ties GBT)."""
    art = rs.load_model()
    by = {r["model"]: r for r in art["validation_metrics"]}
    assert abs(by["Logistic (dashboard)"]["AUC_ROC"] - 0.7748) < 0.005
    assert abs(by["HGB (comparison)"]["AUC_ROC"] - 0.7750) < 0.005
    assert by["Logistic (dashboard)"]["AUC_PR"] > by["B2 current-stress"]["AUC_PR"]

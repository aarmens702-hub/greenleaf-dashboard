"""Phase-2B dashboard model: standardized logistic spike_3d model + per-plot scoring.

Why logistic (not the GBT): it ties the gradient-boosted tree on AUC
(see docs/spike_model_findings.md) and is *interpretable* — with features
StandardScaler-standardized, each plot's risk decomposes into signed feature
contributions (coef x standardized value), which is exactly what the operational
dashboard needs to say "low substrate moisture is the #1 factor raising this plot's risk."

Single source of truth for features / time-split / metrics is analysis/spike_model.py;
this module reuses them so the dashboard model matches the validated one.

CLI:    python analysis/risk_score.py            # (re)build + save models/risk_logistic.joblib
Dashboard API:
    load_model()                       -> artifact dict (cached)
    score_plots(as_of_date)            -> per-plot DataFrame: prob + top risk driver
    plot_contributions(plot_id, date)  -> signed parent-level contributions for one plot
    recent_trend(plot_id, date, days)  -> recent stress/moisture series for the drill-down
"""
from functools import lru_cache
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import spike_model as sm  # noqa: E402  (features, split, metric helpers)

MODEL_PATH = sm.MODELS / "risk_logistic.joblib"

# The watch-list HEADLINE driver is the top *environmental / sensor* contributor — the
# in-season lever a manager can actually act on (irrigation, venting, IPM). The raw #1
# contribution is usually an autoregressive "headroom / mean-reversion" term (a low-stress
# plot has room to rise) or current alert state — true but not actionable. Those, plus
# structural context (treatment, crop), are shown in FULL in the per-plot drill-down, so
# nothing is hidden; the watch-list just leads with the controllable cause.
ENVIRONMENTAL_DRIVERS = {
    "substrate_moisture", "substrate_moisture_7d_mean", "vpd_kpa", "vpd_kpa_7d_mean",
    "relative_humidity_pct", "air_temp_c", "canopy_temp_c", "canopy_air_diff",
    "canopy_air_diff_7d_mean", "co2_ppm", "light_ppfd", "substrate_ec", "substrate_ph",
    "pest_pressure_index", "pest_pressure_7d_mean", "pest_dev_7", "disease_risk_index",
}

# Each sensor enters the model as BOTH a current-day value and its 7-day mean. For
# substrate moisture those two carry opposite-signed coefficients (current −0.77,
# 7d-mean +0.81), so the model's real signal is the *deviation* (drying below the recent
# baseline), not the level. Reported separately they read as a self-contradictory
# "high moisture AND low moisture both raise risk". We merge each pair into one parent
# and phrase it by what the model actually keys on: a "trend" deviation when the pair is
# opposite-signed, a "level" otherwise.
ROLLING_BASE = {
    "substrate_moisture": "substrate_moisture_7d_mean",
    "vpd_kpa": "vpd_kpa_7d_mean",
    "pest_pressure_index": "pest_pressure_7d_mean",
    "canopy_air_diff": "canopy_air_diff_7d_mean",
}
PARTNER_TO_BASE = {v: k for k, v in ROLLING_BASE.items()}

# Raw feature -> plain-language label for the dashboard's "why" copy.
FEATURE_LABELS = {
    "substrate_moisture_7d_mean": "substrate moisture (7-day avg)",
    "substrate_moisture": "substrate moisture",
    "stress_lag_7": "stress one week ago", "stress_lag_3": "stress 3 days ago",
    "stress_lag_1": "stress yesterday", "plant_stress_index": "current stress",
    "stress_trend_3": "3-day stress trend",
    "pest_pressure_index": "pest pressure", "pest_pressure_7d_mean": "pest pressure (7-day avg)",
    "pest_dev_7": "pest pressure vs its 7-day avg",
    "vpd_kpa": "VPD (vapour-pressure deficit)", "vpd_kpa_7d_mean": "VPD (7-day avg)",
    "relative_humidity_pct": "humidity", "air_temp_c": "air temperature",
    "canopy_temp_c": "canopy temperature", "canopy_air_diff": "canopy-air temp gap",
    "canopy_air_diff_7d_mean": "canopy-air gap (7-day avg)",
    "co2_ppm": "CO₂", "light_ppfd": "light (PPFD)",
    "substrate_ec": "substrate EC", "substrate_ph": "substrate pH",
    "disease_risk_index": "disease risk", "alert_flag": "active alert today",
    "plant_density_plants_m2": "planting density", "plot_area_m2": "plot area",
    "sensor_density_per_100m2": "sensor density", "greenhouse_acres": "greenhouse size",
    "capacity_plants": "greenhouse capacity",
    "crop": "crop", "treatment": "treatment", "structure_type": "structure",
    "heating_system": "heating", "irrigation_system": "irrigation",
    "climate_zone": "climate zone", "region": "region",
    "production_system": "production system", "alert_type": "alert type",
}


# --------------------------------------------------------------------------- #
# Build & save
# --------------------------------------------------------------------------- #
def build_and_save():
    """Train the standardized logistic, validate vs baselines on the time-held-out
    split (reusing spike_model), and save the deployment model (refit on all valid
    rows) + validation metrics + calibration to models/risk_logistic.joblib."""
    from sklearn.calibration import calibration_curve
    from sklearn.dummy import DummyClassifier

    df = sm.load_frame()
    train, test = sm.time_split(df)
    Xtr, ytr = train[sm.FEATURES], train[sm.TARGET].astype(int)
    Xte, yte = test[sm.FEATURES], test[sm.TARGET].astype(int)

    rows = []
    b1 = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    rows.append(sm.metrics_row("B1 base-rate", yte, b1.predict_proba(Xte)[:, 1]))
    b2 = sm.build_logistic(["plant_stress_index"]).fit(Xtr, ytr)
    rows.append(sm.metrics_row("B2 current-stress", yte, b2.predict_proba(Xte)[:, 1]))
    b3 = sm.build_logistic(["alert_flag", "pest_pressure_index", "pest_pressure_7d_mean"]).fit(Xtr, ytr)
    rows.append(sm.metrics_row("B3 pest/alert", yte, b3.predict_proba(Xte)[:, 1]))
    logit = sm.build_logistic(sm.FEATURES).fit(Xtr, ytr)
    p_log = logit.predict_proba(Xte)[:, 1]
    rows.append(sm.metrics_row("Logistic (dashboard)", yte, p_log))
    hgb = sm.build_hgb().fit(Xtr, ytr)
    rows.append(sm.metrics_row("HGB (comparison)", yte, hgb.predict_proba(Xte)[:, 1]))
    val = pd.DataFrame(rows).set_index("model")

    frac_pos, mean_pred = calibration_curve(yte, p_log, n_bins=10, strategy="quantile")
    cal = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})

    # Deployment model: same standardized-logistic pipeline, refit on ALL valid rows.
    deploy = sm.build_logistic(sm.FEATURES).fit(df[sm.FEATURES], df[sm.TARGET].astype(int))

    artifact = {
        "model": deploy,
        "model_kind": "standardized logistic regression (StandardScaler + one-hot)",
        "features": sm.FEATURES,
        "numeric_features": sm.NUMERIC,
        "categorical_features": sm.CATEGORICAL,
        "excluded_leakage": sm.EXCLUDED_LEAKAGE,
        "base_rate": float(df[sm.TARGET].mean()),
        "alarm_threshold": sm.ALARM_THRESHOLD,
        "target": sm.TARGET,
        "train_end": sm.TRAIN_END, "test_start": sm.TEST_START,
        "validation_metrics": val.round(4).reset_index().to_dict(orient="records"),
        "calibration": cal.round(4).to_dict(orient="records"),
        "note": "Validation metrics are from the time-held-out split (train<=%s, "
                "test>=%s). The saved model is refit on all valid rows for scoring."
                % (sm.TRAIN_END, sm.TEST_START),
        "seed": sm.SEED,
    }
    sm.MODELS.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    return val, cal


# --------------------------------------------------------------------------- #
# Load + feature plumbing (process-cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        build_and_save()
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _daily():
    daily = pd.read_parquet(sm.DERIVED / "daily.parquet")
    return sm.add_derived(daily)


def date_range():
    d = _daily()
    return d["date"].min(), d["date"].max()


def _parent(out_name):
    """Map a ColumnTransformer output name to (raw_feature, category|None).
    'num__substrate_moisture' -> ('substrate_moisture', None)
    'cat__treatment_Shade'    -> ('treatment', 'Shade')"""
    body = out_name.split("__", 1)[1]
    if out_name.startswith("cat__"):
        for c in sm.CATEGORICAL:
            if body.startswith(c + "_"):
                return c, body[len(c) + 1:]
    return body, None


def _pair_modes(model):
    """For each current/7d-mean pair, 'trend' if the two coefficients have opposite
    signs (model keys on the deviation, e.g. moisture), else 'level'."""
    prep = model.named_steps["prep"]
    cmap = dict(zip(prep.get_feature_names_out(), model.named_steps["clf"].coef_[0]))
    modes = {}
    for base, partner in ROLLING_BASE.items():
        cc = cmap.get("num__" + base, 0.0)
        cm = cmap.get("num__" + partner, 0.0)
        modes[base] = "trend" if cc * cm < 0 else "level"
    return modes


def _phrase(base, rec, pair_modes):
    label = FEATURE_LABELS.get(base, base)
    if base == "alert_flag":
        return "active alert today"
    if rec["category"] is not None:
        return f"{label}: {rec['category']}"
    if rec["is_pair"]:
        z_cur = rec["z_cur"] if rec["z_cur"] is not None else 0.0
        z_7d = rec["z_7d"] if rec["z_7d"] is not None else 0.0
        if pair_modes.get(base) == "trend":          # deviation feature (e.g. moisture)
            return f"{label} {'below' if (z_cur - z_7d) < 0 else 'above'} recent trend"
        return f"{'high' if z_7d >= 0 else 'low'} {label}"   # level feature, smoothed
    z = rec["z"]
    if z is None:
        return label
    return f"{'high' if z >= 0 else 'low'} {label}"


def _contributions_matrix(model, X):
    """Return (contrib, names, z_values, intercept). contrib[i,j] = signed log-odds
    contribution of output-feature j for row i (= standardized value * coefficient)."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    Xt = np.asarray(prep.transform(X))
    names = list(prep.get_feature_names_out())
    coef = clf.coef_[0]
    return Xt * coef, names, Xt, float(clf.intercept_[0])


def _row_contributions(contrib_row, names, z_row, pair_modes):
    """Aggregate one row's output-feature contributions up to parent features, merging
    each current/7d-mean pair into one parent (so moisture is a single 'trend' driver,
    not two contradictory halves). Returns [feature, contribution, phrase] sorted desc."""
    agg = {}
    for j, name in enumerate(names):
        raw, category = _parent(name)
        base = PARTNER_TO_BASE.get(raw, raw)        # fold 7d-mean half into its base
        c = float(contrib_row[j])
        z = float(z_row[j])
        rec = agg.setdefault(base, {"contribution": 0.0, "category": None, "z": None,
                                    "z_cur": None, "z_7d": None,
                                    "is_pair": base in ROLLING_BASE})
        rec["contribution"] += c
        if category is not None:
            if abs(c) > 1e-12:                      # the active one-hot category
                rec["category"] = category
        elif rec["is_pair"]:
            if raw == base:
                rec["z_cur"] = z                    # current-day half
            else:
                rec["z_7d"] = z                     # 7-day-mean half
        else:
            rec["z"] = z
    out = pd.DataFrame(
        [{"feature": b, "contribution": r["contribution"], "phrase": _phrase(b, r, pair_modes)}
         for b, r in agg.items()]
    )
    return out.sort_values("contribution", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Dashboard API
# --------------------------------------------------------------------------- #
def score_plots(as_of_date):
    """Score every plot on as_of_date. Returns a DataFrame sorted by spike risk:
    plot_id, crop, treatment, greenhouse_id, spike_prob, top_driver."""
    art = load_model()
    model = art["model"]
    day = _daily()
    rows = day[day["date"] == pd.Timestamp(as_of_date)].copy()
    if rows.empty:
        return pd.DataFrame(columns=["plot_id", "crop", "treatment", "greenhouse_id",
                                     "spike_prob", "top_driver"])
    X = rows[art["features"]]
    probs = model.predict_proba(X)[:, 1]
    contrib, names, zmat, _ = _contributions_matrix(model, X)
    pair_modes = _pair_modes(model)

    drivers = []
    for i in range(len(rows)):
        rc = _row_contributions(contrib[i], names, zmat[i], pair_modes)
        env = rc[rc["feature"].isin(ENVIRONMENTAL_DRIVERS)]
        top = env.iloc[0] if len(env) else None   # top risk-raising ENVIRONMENTAL lever
        drivers.append(top["phrase"] if top is not None and top["contribution"] > 0
                       else "environment not elevating risk")
    out = pd.DataFrame({
        "plot_id": rows["plot_id"].values, "crop": rows["crop"].values,
        "treatment": rows["treatment"].values, "greenhouse_id": rows["greenhouse_id"].values,
        "spike_prob": probs, "top_driver": drivers,
    })
    return out.sort_values("spike_prob", ascending=False, ignore_index=True)


def plot_contributions(plot_id, as_of_date):
    """Signed parent-level contributions for one plot on one date (for the drill-down
    bar). Positive = pushing spike risk up; negative = pulling it down."""
    art = load_model()
    model = art["model"]
    day = _daily()
    row = day[(day["plot_id"] == plot_id) & (day["date"] == pd.Timestamp(as_of_date))]
    if row.empty:
        return pd.DataFrame(columns=["feature", "contribution", "phrase"])
    X = row[art["features"]]
    contrib, names, zmat, _ = _contributions_matrix(model, X)
    return _row_contributions(contrib[0], names, zmat[0], _pair_modes(model))


def plot_probability(plot_id, as_of_date):
    art = load_model()
    day = _daily()
    row = day[(day["plot_id"] == plot_id) & (day["date"] == pd.Timestamp(as_of_date))]
    if row.empty:
        return float("nan")
    return float(art["model"].predict_proba(row[art["features"]])[:, 1][0])


def recent_trend(plot_id, as_of_date, days=21):
    """Recent plant_stress_index + substrate_moisture for the drill-down trend chart."""
    day = _daily()
    end = pd.Timestamp(as_of_date)
    start = end - pd.Timedelta(days=days)
    sub = day[(day["plot_id"] == plot_id) & (day["date"] <= end) & (day["date"] > start)]
    return sub[["date", "plant_stress_index", "substrate_moisture"]].sort_values("date")


if __name__ == "__main__":
    val, cal = build_and_save()
    print("Saved", MODEL_PATH)
    print("\nValidation metrics (time-held-out test):")
    print(val.round(4).to_string())
    print("\nCalibration (logistic, test):")
    print(cal.round(4).to_string(index=False))

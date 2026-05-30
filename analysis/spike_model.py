"""Phase-2A: spike_3d early-warning risk model with honest baselines & validation.

OPERATIONAL early-warning for triage ("which plots should a manager watch/act on in
the next 3 days") — NOT a financial-return model. Phase-1 (docs/eda_findings.md) showed
plot-level responsiveness does not predict profit, so we make no profit claims here.

Target: spike_3d (1 if plant_stress_index rises >=0.10 over the next 3 days). Base rate
~0.18. Forward-looking, so features use ONLY at/before-prediction-day information.

Run:    python analysis/spike_model.py
Output: full metrics-vs-baselines table, permutation importances, calibration table,
        a GroupKFold secondary check; saves the deployment model to models/.

Everything seeded (42). Read-only on data/derived.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
MODELS = ROOT / "models"

TARGET = "spike_3d"
TRAIN_END = "2025-07-15"   # train: season start .. this date (inclusive)
TEST_START = "2025-08-01"  # test: this date .. season end. 2025-07-16..31 = dropped
#                            buffer (16 days >> 3-day forward horizon) so the target
#                            never straddles the train/test boundary.
ALARM_THRESHOLD = 0.18     # operational alarm threshold ~= base rate (recall-oriented)

# --- Feature design -------------------------------------------------------- #
# Current-day sensor readings + alert state (known at the start/end of prediction day).
NUMERIC_CURRENT = [
    "air_temp_c", "relative_humidity_pct", "vpd_kpa", "co2_ppm", "light_ppfd",
    "substrate_moisture", "substrate_ec", "substrate_ph", "canopy_temp_c",
    "pest_pressure_index", "disease_risk_index", "plant_stress_index",
    "canopy_air_diff", "alert_flag",
]
# Trailing lags / 7-day rolling means (already in daily.parquet; backward-looking).
NUMERIC_TRAILING = [
    "stress_lag_1", "stress_lag_3", "stress_lag_7",
    "vpd_kpa_7d_mean", "substrate_moisture_7d_mean",
    "pest_pressure_7d_mean", "canopy_air_diff_7d_mean",
]
# Derived trailing momentum features (computed below; use only past/current values).
NUMERIC_DERIVED = ["stress_trend_3", "pest_dev_7"]
# Static plot/greenhouse attributes (time-invariant; no leakage).
NUMERIC_STATIC = [
    "plant_density_plants_m2", "plot_area_m2", "sensor_density_per_100m2",
    "greenhouse_acres", "capacity_plants",
]
CATEGORICAL = [
    "crop", "treatment", "structure_type", "heating_system", "irrigation_system",
    "climate_zone", "region", "production_system", "alert_type",
]
NUMERIC = NUMERIC_CURRENT + NUMERIC_TRAILING + NUMERIC_DERIVED + NUMERIC_STATIC
FEATURES = NUMERIC + CATEGORICAL

# Explicitly EXCLUDED and why (asserted at runtime so a future edit can't sneak leakage in).
EXCLUDED_LEAKAGE = {
    # target itself
    "spike_3d": "target",
    # forward-looking (use the future, definitionally leaks the answer)
    "stress_lead_3": "forward (t+3 stress)",
    "stress_delta_3d_forward": "forward (t+3 - t stress = the target's source)",
    "post_action_stress_delta_3d": "forward (3-day post-action stress change)",
    # intervention / post-decision (leaks the response the model is meant to inform,
    # and is not known before the manager decides to act)
    "action_taken": "post-decision (whether we acted today)",
    "action_delay_days": "post-decision (needs to know when the action eventually happened)",
    "recommended_action": "post-decision text",
    "daily_precision_cost": "intervention cost",
    "daily_routine_cost": "intervention cost",
    "daily_precision_actions_count": "intervention count",
    "daily_total_actions_count": "intervention count",
    "daily_total_input_cost": "intervention cost",
    # identity (memorization / not generalizable)
    "plot_id": "identity", "greenhouse_id": "identity", "farm_id": "identity",
    "date": "time index (handled by the split, not a feature)",
}


# --------------------------------------------------------------------------- #
def add_derived(df):
    """Attach the trailing-only derived features the model expects (shared by the
    training pipeline and the dashboard scorer, so both build features identically)."""
    df = df.copy()
    df["stress_trend_3"] = df["plant_stress_index"] - df["stress_lag_3"]
    df["pest_dev_7"] = df["pest_pressure_index"] - df["pest_pressure_7d_mean"]
    df["alert_type"] = df["alert_type"].fillna("None")
    return df


def load_frame():
    daily = pd.read_parquet(DERIVED / "daily.parquet")
    daily = daily[daily[TARGET].notna()].copy()          # drop last 3 days/plot (no target)
    return add_derived(daily)


def assert_no_leakage():
    bad = [f for f in FEATURES if f in EXCLUDED_LEAKAGE]
    assert not bad, f"Leakage: excluded columns present in FEATURES: {bad}"


def time_split(df):
    train = df[df.date <= TRAIN_END]
    test = df[df.date >= TEST_START]
    return train, test


def make_preprocessor(for_linear):
    """One-hot categoricals. Linear models also impute+scale numerics; HGB keeps NaN
    (native support) and needs no scaling."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if for_linear:
        num = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    else:
        num = "passthrough"
    return ColumnTransformer([("num", num, NUMERIC), ("cat", ohe, CATEGORICAL)])


def build_hgb():
    return Pipeline([
        ("prep", make_preprocessor(for_linear=False)),
        ("clf", HistGradientBoostingClassifier(random_state=SEED, max_iter=300,
                                               learning_rate=0.05, l2_regularization=1.0,
                                               early_stopping=True, validation_fraction=0.15)),
    ])


def build_logistic(features):
    cat = [c for c in CATEGORICAL if c in features]
    num = [c for c in NUMERIC if c in features]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numpipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pre = ColumnTransformer([("num", numpipe, num), ("cat", ohe, cat)])
    return Pipeline([("prep", pre),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


# --------------------------------------------------------------------------- #
def metrics_row(name, y, p, threshold=ALARM_THRESHOLD):
    yhat = (p >= threshold).astype(int)
    # precision @ top-20% of scores (triage: act on the riskiest fifth today)
    k = max(1, int(0.20 * len(p)))
    top_idx = np.argsort(p)[::-1][:k]
    prec_at_20 = float(np.mean(np.asarray(y)[top_idx]))
    return {
        "model": name,
        "AUC_ROC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan"),
        "AUC_PR": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        f"Prec@thr{threshold}": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat, zero_division=0),
        "Prec@top20%": prec_at_20,
    }


def main():
    assert_no_leakage()
    MODELS.mkdir(exist_ok=True)
    df = load_frame()
    train, test = time_split(df)
    Xtr, ytr = train[FEATURES], train[TARGET].astype(int)
    Xte, yte = test[FEATURES], test[TARGET].astype(int)

    print("=" * 78)
    print("SPIKE_3D EARLY-WARNING MODEL — time-held-out validation")
    print("=" * 78)
    print(f"target base rate (all valid rows): {df[TARGET].mean():.4f}  (n={len(df)})")
    print(f"TRAIN  {train.date.min().date()}..{train.date.max().date()}  rows={len(train)}  base={ytr.mean():.4f}")
    print(f"BUFFER {TRAIN_END}<date<{TEST_START} dropped (no-leakage gap; 3-day forward horizon)")
    print(f"TEST   {test.date.min().date()}..{test.date.max().date()}  rows={len(test)}  base={yte.mean():.4f}")
    print(f"features: {len(FEATURES)} ({len(NUMERIC)} numeric + {len(CATEGORICAL)} categorical); "
          f"{len(EXCLUDED_LEAKAGE)} columns excluded for leakage/identity")

    # --- Baselines + models (all fit on train, evaluated on the same test set) --- #
    rows = []

    # B1: base rate (prior). AUC=0.5 reference.
    b1 = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    rows.append(metrics_row("B1 base-rate", yte, b1.predict_proba(Xte)[:, 1]))

    # B2: current plant_stress_index alone (logistic, calibrated single feature)
    b2 = build_logistic(["plant_stress_index"]).fit(Xtr, ytr)
    rows.append(metrics_row("B2 current-stress", yte, b2.predict_proba(Xte)[:, 1]))

    # B3: pest/alert state alone
    b3 = build_logistic(["alert_flag", "pest_pressure_index", "pest_pressure_7d_mean"]).fit(Xtr, ytr)
    rows.append(metrics_row("B3 pest/alert", yte, b3.predict_proba(Xte)[:, 1]))

    # Interpretable full logistic
    logit = build_logistic(FEATURES).fit(Xtr, ytr)
    rows.append(metrics_row("Logistic (all feats)", yte, logit.predict_proba(Xte)[:, 1]))

    # Primary: HGB
    hgb = build_hgb().fit(Xtr, ytr)
    p_hgb = hgb.predict_proba(Xte)[:, 1]
    rows.append(metrics_row("HGB (primary)", yte, p_hgb))

    table = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\n--- METRICS vs BASELINES (test set; threshold = %.2f) ---" % ALARM_THRESHOLD)
    print(table.round(4).to_string())
    print(f"(no-skill AUC_PR = base rate = {yte.mean():.4f}; AUC_ROC 0.5 = no skill)")

    # Direction of B2 (explains the ceiling effect for the narrative)
    b2_coef = b2.named_steps["clf"].coef_[0][0]
    print(f"\nB2 current-stress logistic coef (standardized) = {b2_coef:+.3f} "
          f"-> {'higher' if b2_coef>0 else 'LOWER'} current stress predicts MORE spikes "
          f"({'' if b2_coef>0 else 'ceiling effect: low-stress plots have headroom to rise'})")

    # --- Confusion matrix for the primary model at the alarm threshold --- #
    yhat = (p_hgb >= ALARM_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
    print(f"\nHGB confusion @thr={ALARM_THRESHOLD}: TP={tp} FP={fp} FN={fn} TN={tn} "
          f"(of {len(yte)} test plot-days; {int(yte.sum())} true spikes)")

    # --- Permutation importance (test set, scored on AUC-PR) --- #
    print("\n--- PERMUTATION IMPORTANCE (HGB, test, scoring=average_precision, n_repeats=10) ---")
    perm = permutation_importance(hgb, Xte, yte, scoring="average_precision",
                                  n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = (pd.DataFrame({"feature": FEATURES, "imp": perm.importances_mean,
                         "std": perm.importances_std})
           .sort_values("imp", ascending=False).reset_index(drop=True))
    print(imp.head(12).round(5).to_string(index=False))

    # --- Calibration (reliability) --- #
    print("\n--- CALIBRATION (HGB test, 10 quantile bins) ---")
    frac_pos, mean_pred = calibration_curve(yte, p_hgb, n_bins=10, strategy="quantile")
    cal = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
    print(cal.round(4).to_string(index=False))
    print(f"HGB Brier = {brier_score_loss(yte, p_hgb):.4f} (lower = better; base-rate Brier "
          f"= {brier_score_loss(yte, np.full(len(yte), ytr.mean())):.4f})")

    # --- Secondary check: GroupKFold by plot_id (out-of-fold, full data) --- #
    print("\n--- SECONDARY: GroupKFold(5) by plot_id (out-of-fold) ---")
    gkf = GroupKFold(n_splits=5)
    oof = cross_val_predict(build_hgb(), df[FEATURES], df[TARGET].astype(int),
                            groups=df["plot_id"], cv=gkf, method="predict_proba", n_jobs=-1)[:, 1]
    yall = df[TARGET].astype(int).values
    print(f"OOF AUC_ROC={roc_auc_score(yall, oof):.4f}  AUC_PR={average_precision_score(yall, oof):.4f} "
          f"(base rate {yall.mean():.4f})")

    # --- Save deployment artifact (refit on ALL valid rows; same pipeline) --- #
    deploy = build_hgb().fit(df[FEATURES], df[TARGET].astype(int))
    artifact = {
        "model": deploy,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "derived_features": {"stress_trend_3": "plant_stress_index - stress_lag_3",
                             "pest_dev_7": "pest_pressure_index - pest_pressure_7d_mean"},
        "excluded_leakage": EXCLUDED_LEAKAGE,
        "alarm_threshold": ALARM_THRESHOLD,
        "base_rate": float(df[TARGET].mean()),
        "target": TARGET,
        "train_end": TRAIN_END, "test_start": TEST_START,
        "validation_metrics": table.round(5).to_dict(orient="index"),
        "note": "Metrics are from the time-held-out model (train<=%s, test>=%s). "
                "This saved model is REFIT on all valid rows for deployment scoring."
                % (TRAIN_END, TEST_START),
        "seed": SEED,
    }
    joblib.dump(artifact, MODELS / "spike_model.joblib")
    (MODELS / "spike_model_features.json").write_text(json.dumps(
        {"features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
         "excluded_leakage": EXCLUDED_LEAKAGE, "alarm_threshold": ALARM_THRESHOLD,
         "base_rate": float(df[TARGET].mean()), "target": TARGET}, indent=2))
    print(f"\nsaved: {MODELS/'spike_model.joblib'} (+ spike_model_features.json)")


if __name__ == "__main__":
    main()

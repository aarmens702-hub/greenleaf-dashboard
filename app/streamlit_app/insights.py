"""Read-only insight helpers for the dashboards.

Every function COMPOSES the frozen analysis modules (analysis/eda.py, risk_score.py,
finance.py, spike_model.py) and adds only standard, transparent aggregations
(cumulative gains, Lorenz/Gini/Expected-Shortfall, EWMA, one-way variance decomposition).
No model, parquet, or computed statistic from the frozen layer is altered — every value
traces to an existing function/column. UI code calls ONLY these (never recomputes).
"""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis import eda                       # noqa: E402
from analysis import finance as fin            # noqa: E402
from analysis import risk_score as rs          # noqa: E402
from analysis import spike_model as sm         # noqa: E402


# --- small standard stats (operate only on values pulled from frozen functions) ---
def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cum = np.cumsum(x)
    if cum[-1] == 0:
        return 0.0
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def _lorenz(x):
    x = np.sort(np.asarray(x, float))
    c = np.concatenate([[0.0], np.cumsum(x)])
    pop = (np.arange(len(c)) / (len(c) - 1)).tolist()
    val = (c / c[-1]).tolist()
    return pop, val


def _es(x, q):  # expected shortfall = mean of the worst q-fraction
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil(q * len(x))))
    return float(x[:k].mean())


# --- S1 · response delay -> stress relief dose-response -----------------------
def response_dose():
    m = eda.mechanism_delay_stress(eda.load()["daily"])
    by = m["by_delay"].reset_index().rename(columns={"action_delay_days": "delay"})
    by["se"] = by["std"] / np.sqrt(by["count"])
    by["lo"] = by["mean"] - 1.96 * by["se"]
    by["hi"] = by["mean"] + 1.96 * by["se"]
    by["share"] = by["count"] / by["count"].sum()
    spike = m["spike_by_delay"].reset_index().rename(columns={"action_delay_days": "delay"})
    return {"by": by, "diff_ci": m["diff_ci"], "welch_p": float(m["welch_p"]),
            "spike": spike, "n": int(by["count"].sum())}


# --- S2 · spike-risk driver breakdown (logistic contributions) ----------------
@lru_cache(maxsize=1)
def driver_importance():
    art = rs.load_model()
    model = art["model"]
    daily = rs._daily()
    X = daily[art["features"]]
    contrib, names, _z, _b = rs._contributions_matrix(model, X)
    parents = np.array([rs.PARTNER_TO_BASE.get(rs._parent(nm)[0], rs._parent(nm)[0]) for nm in names])
    rows = []
    for base in pd.unique(parents):
        cols = np.where(parents == base)[0]
        per_row = contrib[:, cols].sum(axis=1)           # signed sum per row (== _row_contributions)
        rows.append({"feature": base, "mean_abs": float(np.abs(per_row).mean()),
                     "mean_signed": float(per_row.mean())})
    df = pd.DataFrame(rows)
    df["env"] = df.feature.isin(rs.ENVIRONMENTAL_DRIVERS)
    df["label"] = df.feature.map(lambda f: rs.FEATURE_LABELS.get(f, f))
    return df.sort_values("mean_abs", ascending=False).reset_index(drop=True)


def plot_contributions(plot_id, as_of_date):
    return rs.plot_contributions(plot_id, as_of_date)


# --- S3 · cost mix (daily ledger = canonical) ---------------------------------
def cost_mix():
    df, prec, rout = fin.cost_structure()
    total = prec + rout
    df = df.copy().sort_values("total")
    df["pct"] = df["total"] / total * 100
    return {"by_cat": df, "precision": prec, "routine": rout, "total": total,
            "precision_pct": prec / total * 100, "labor_pct": float(df["pct"].max())}


# --- S4 · profit distribution + portfolio tail --------------------------------
def profit_tail():
    pdist = fin.profit_distribution()
    profit = pdist.season_profit_cad.values
    rev = eda.merged().season_revenue_cad.values
    h = fin.headline()
    return {"dist": pdist, "headline": h, "loss_makers": fin.loss_makers(),
            "gini_profit": _gini(profit), "gini_revenue": _gini(rev),
            "lorenz_rev": _lorenz(rev), "es5": _es(profit, 0.05), "es10": _es(profit, 0.10),
            "top20_share": float(np.sort(profit)[::-1][:int(0.2 * len(profit))].sum() / profit.sum())}


# --- Y1 · early-warning gains / catch-rate (train-only fit, no leakage) --------
@lru_cache(maxsize=1)
def gains_curve():
    from sklearn.metrics import confusion_matrix
    df = sm.load_frame()
    tr, te = sm.time_split(df)
    model = sm.build_logistic(sm.FEATURES).fit(tr[sm.FEATURES], tr[sm.TARGET].astype(int))
    yte = te[sm.TARGET].astype(int).values
    p = model.predict_proba(te[sm.FEATURES])[:, 1]
    order = np.argsort(p)[::-1]
    cap = np.cumsum(yte[order]) / yte.sum()
    frac = np.arange(1, len(yte) + 1) / len(yte)

    def at(f):
        return float(cap[min(len(cap) - 1, int(np.searchsorted(frac, f)))])

    yhat = (p >= sm.ALARM_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
    return {"frac": frac.tolist(), "capture": cap.tolist(),
            "top10": at(0.10), "top20": at(0.20), "top30": at(0.30),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(tp / (tp + fp)), "recall": float(tp / (tp + fn)),
            "alarm_rate": float((tp + fp) / len(yte)), "base": float(yte.mean()),
            "n_test": int(len(yte)), "n_spikes": int(yte.sum())}


def model_metrics():
    art = rs.load_model()
    return (pd.DataFrame(art["validation_metrics"]), pd.DataFrame(art["calibration"]),
            float(art["base_rate"]))


# --- Y2 · seasonal stress trajectory by treatment -----------------------------
def weekly_stress_trajectory():
    d = eda.load()["daily"][["date", "treatment", "plant_stress_index"]].copy()
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    piv = d.groupby(["treatment", "week"]).plant_stress_index.mean().reset_index()
    season_mean = d.groupby("treatment").plant_stress_index.mean().sort_values()
    grand = d.plant_stress_index.mean()
    sst = float(((d.plant_stress_index - grand) ** 2).sum())
    # one-way eta^2 for each factor (reproducible; timing = day-of-season is the fuller signal)
    eta_timing = float(((d.groupby("date").plant_stress_index.transform("mean") - grand) ** 2).sum() / sst)
    eta_treat = float(((d.groupby("treatment").plant_stress_index.transform("mean") - grand) ** 2).sum() / sst)
    return {"piv": piv, "order": season_mean.index.tolist(), "season_mean": season_mean,
            "eta_timing": eta_timing, "eta_treat": eta_treat,
            "spread": float(season_mean.max() - season_mean.min())}


# --- Y3 · alert-burden exposure map -------------------------------------------
def alert_exposure():
    m = eda.merged()[["plot_id", "alert_days", "treatment", "mean_stress"]].copy()
    return {"plots": m.sort_values("alert_days"), "gini": _gini(m.alert_days.values),
            "top_decile_share": float(np.sort(m.alert_days.values)[::-1][:12].sum() / m.alert_days.sum()),
            "corr_stress": float(m.alert_days.corr(m.mean_stress)),
            "rng": (int(m.alert_days.min()), int(m.alert_days.max()))}


# --- Y4 · treatment economics with bootstrap CIs ------------------------------
def treatment_economics():
    g, gap = fin.by_treatment()
    _agg, diffs = eda.treatment_table(eda.merged())
    rows = []
    for _, r in g.iterrows():
        d = diffs.get(r["treatment"])
        rows.append({"treatment": r["treatment"], "profit": float(r["profit"]),
                     "n": int(r["n"]),
                     "diff": None if d is None else float(d["profit_vs_ctrl"][0]),
                     "lo": None if d is None else float(d["profit_vs_ctrl"][1]),
                     "hi": None if d is None else float(d["profit_vs_ctrl"][2])})
    return {"rows": pd.DataFrame(rows).sort_values("profit"), "hl_shade_gap": gap}


# --- Y5 · per-plot substrate-moisture drift (trimmed: moisture only) ----------
def moisture_drift(plot_id, as_of_date, days=28, lam=0.2, k_sigma=3):
    tr = rs.recent_trend(plot_id, as_of_date, days).copy()
    s = tr.substrate_moisture.values
    if len(s) == 0:
        for c in ("ewma", "upper", "lower"):
            tr[c] = []
        return tr
    base = s[: min(7, len(s))]
    mu, sd = float(base.mean()), float(base.std() or 1e-6)
    ewma, e = [], mu
    for v in s:
        e = lam * v + (1 - lam) * e
        ewma.append(e)
    se = sd * np.sqrt(lam / (2 - lam))
    tr["ewma"] = ewma
    tr["upper"] = mu + k_sigma * se
    tr["lower"] = mu - k_sigma * se
    return tr


# --- Financials honesty: precision_benefit (frozen finding) -------------------
def precision_benefit():
    return fin.precision_benefit_check()


# --- watch-list (operational core) --------------------------------------------
def watchlist(as_of_date):
    df = rs.score_plots(as_of_date).copy()
    df["spike_pct"] = df["spike_prob"] * 100
    return df


def date_range():
    return rs.date_range()


def headline():
    return fin.headline()

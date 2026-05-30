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


def delay_treatment_relief():
    """Mean post-action 3-day stress relief by response delay × treatment, with per-cell
    event counts (n). SAME event definition as eda.mechanism_delay_stress (action_taken==1,
    grouped by action_delay_days) — the delay effect is not recomputed differently.
    Columns ordered most-relief-first. Read-only; reuses eda.load()."""
    ad = eda.load()["daily"]
    ad = ad[ad.action_taken == 1]
    mean = ad.pivot_table(index="action_delay_days", columns="treatment",
                          values="post_action_stress_delta_3d", aggfunc="mean")
    cnt = ad.pivot_table(index="action_delay_days", columns="treatment",
                         values="post_action_stress_delta_3d", aggfunc="size")
    cols = mean.mean(axis=0).sort_values().index.tolist()   # most relief (most negative) first
    mean = mean.reindex(index=[0, 1, 2], columns=cols)
    cnt = cnt.reindex(index=[0, 1, 2], columns=cols).fillna(0).astype(int)
    return mean, cnt


def model_metrics():
    art = rs.load_model()
    return (pd.DataFrame(art["validation_metrics"]), pd.DataFrame(art["calibration"]),
            float(art["base_rate"]))


# --- Financials: profit drivers, crop economics, treatment spread, win/loss ---
def profit_drivers():
    """Correlation of each factor with per-plot profit (fin.drivers) + the per-plot
    price->profit scatter. Read-only; the scatter pulls price_per_kg straight from the
    frozen fin._m() frame (computed there, not recomputed here)."""
    sc = fin._m()[["price_per_kg", "season_profit_cad", "crop"]].copy()
    return {"corr": fin.drivers(), "scatter": sc}


def crop_economics():
    return fin.by_crop()


def treatment_spread():
    """Mean profit (payoff), std (spread) and worst plot by treatment (fin.risk_return).
    'Spread' = plot-to-plot std — a descriptive dispersion, not a forward forecast."""
    return fin.risk_return()


def top_earners(n=5):
    return fin.top_earners(n)


def loss_diagnosis():
    """Why the loss-makers lost: within-crop price percentile vs overall cost percentile.
    Standard rank aggregation over frozen fin._m() columns (price_per_kg, total_cost_cad)."""
    m = fin._m().copy()
    m["price_pctile_in_crop"] = m.groupby("crop").price_per_kg.rank(pct=True) * 100
    m["cost_pctile"] = m.total_cost_cad.rank(pct=True) * 100
    lo = m[m.season_profit_cad < 0]
    return {"n": int(len(lo)),
            "n_below_p10_price": int((lo.price_pctile_in_crop < 10).sum()),
            "med_price_pctile": float(lo.price_pctile_in_crop.median()),
            "med_cost_pctile": float(lo.cost_pctile.median())}


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
    arm = eda.merged().groupby("treatment").agg(p=("season_profit_cad", "mean"), a=("alert_days", "mean"))
    return {"rows": pd.DataFrame(rows).sort_values("profit"), "hl_shade_gap": gap,
            "corr_arms": float(arm.p.corr(arm.a))}


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


_PLOT_SCORE = Path(__file__).resolve().parents[2] / "data" / "raw" / "plot_score.csv"


@lru_cache(maxsize=1)
def _plot_score():
    return pd.read_csv(_PLOT_SCORE)


def recommendation_adherence():
    """Decision-support uptake: per recommended_action type, how often an action was taken
    (uptake proxy) and the mean realized 3-day stress relief WHEN it was (same daily log —
    post_action_stress_delta_3d). Read-only. Note: the input-application log does not align
    with this one (≈5% same-day overlap), so 'acted on' cannot be type-matched to the rec."""
    d = eda.load()["daily"]
    rec = d[d.recommended_action.notna()]
    rows = []
    for r in ["Increase Venting", "Increase Irrigation", "Adjust Shade / Venting"]:
        sub = rec[rec.recommended_action == r]
        acted = sub[sub.action_taken == 1]
        rows.append({"rec": r, "n": int(len(sub)), "adhere": float(sub.action_taken.mean()),
                     "n_acted": int(len(acted)),
                     "relief": float(acted.post_action_stress_delta_3d.mean())})
    out = pd.DataFrame(rows).sort_values("adhere").reset_index(drop=True)
    overall = float(rec.action_taken.mean())
    overall_relief = float(rec[rec.action_taken == 1].post_action_stress_delta_3d.mean())
    return out, overall, overall_relief


def greenhouse_ops():
    """Per-greenhouse operational allocation: alert workload, response rate, mean delay.
    Read-only; from daily.parquet (alert_flag / action_taken / action_delay_days)."""
    d = eda.load()["daily"]
    base = d.groupby("greenhouse_id").agg(plots=("plot_id", "nunique"),
                                          alert_days=("alert_flag", "sum")).reset_index()
    ad = (d[d.alert_flag == 1].groupby("greenhouse_id")
          .agg(resp=("action_taken", "mean"), delay=("action_delay_days", "mean")).reset_index())
    g = base.merge(ad, on="greenhouse_id")
    g["resp_pct"] = g["resp"] * 100
    overall = float((d[d.alert_flag == 1].action_taken.mean()))
    return g.sort_values("alert_days", ascending=False).reset_index(drop=True), overall


def operational_kpis():
    """Whole-season operational health KPIs. Stress KPIs trace to daily.parquet; the
    performance score is the mean of plot_score.csv's final_score_scaled — a 5-part
    composite (marketable yield, ROI, precision benefit, plant health, env stability).
    Stress bands: Healthy < 0.50 · Watch 0.50–0.65 · Critical ≥ 0.65."""
    d = eda.load()["daily"]
    alert_days = d[d.alert_flag == 1]
    return {
        "healthy_rate": float((d.plant_stress_index < 0.50).mean()),
        "watch_rate": float(((d.plant_stress_index >= 0.50) & (d.plant_stress_index < 0.65)).mean()),
        "critical_rate": float((d.plant_stress_index >= 0.65).mean()),
        "performance_score": float(_plot_score().final_score_scaled.mean()),
        "alert_rate": float((d.alert_flag == 1).mean()),
        "alert_response_rate": float(alert_days.action_taken.mean()),
        "n_plot_days": int(len(d)),
    }


# --- Spec helpers: profit-free daily plot score, headline KPIs, spend, composite risk ----
_DAILY_PLOT_SCORE = Path(__file__).resolve().parents[2] / "data" / "raw" / "daily_plot_score.csv"
_ALL_SCORES = Path(__file__).resolve().parents[2] / "data" / "raw" / "all_scores_complete.csv"


@lru_cache(maxsize=1)
def clean_plot_score():
    """Daily plot score recomputed WITHOUT the circular profit term (Step 1).
    Original: 0.40 H + 0.25 E + 0.20 (1-P) + 0.15 (1-C). The profit term (P = profit_at_risk,
    profit-derived) is dropped and the remaining three weights renormalized to sum to 1
    (Health 0.50, Environment 0.3125, Cost-efficiency 0.1875). Computed FRESH from the H / E / C
    component columns — never subtracted from the pre-baked total — so no profit leaks in. Same
    0-90 scaling as the source file (score = 90 * raw / raw.max(); verified to reproduce the
    original at R^2=1.0). Read-only: reads data/raw/daily_plot_score.csv, never writes."""
    df = pd.read_csv(_DAILY_PLOT_SCORE, parse_dates=["date"])
    c = df.daily_total_input_cost
    cn = (c - c.min()) / (c.max() - c.min())                       # normalized cost (C)
    raw = 0.50 * df.plot_health + 0.3125 * df.env_stability_score + 0.1875 * (1 - cn)
    df["clean_score"] = 90 * raw / raw.max()
    return df[["date", "plot_id", "crop", "plot_health", "env_stability_score",
               "daily_total_input_cost", "clean_score"]]


def headline_kpis():
    """The five headline KPIs (Step 2). Every value traces to a source column; 'week' is the
    most recent 7 calendar days in daily.parquet (stated in the UI)."""
    cs = clean_plot_score()
    d = eda.load()["daily"]
    season = eda.load()["season"]
    wk_start = d.date.max() - pd.Timedelta(days=6)
    wk = d[(d.date >= wk_start) & (d.alert_flag == 1)]
    return {
        "avg_performance": float(cs.clean_score.mean()),               # 0-90, profit-free
        "avg_health": float(cs.plot_health.mean()),                    # 0-1
        "avg_roi": float(season.season_roi.mean()),                    # fraction
        "weekly_response_rate": float(wk.action_taken.mean()),
        "week_start": wk_start, "week_end": d.date.max(), "week_n": int(len(wk)),
        "profit_per_m2": float(season.season_profit_cad.sum() / season.plot_area_m2.sum()),
    }


def weekly_spend():
    """Past 7 days of total daily input spend from the canonical daily ledger (Step 3) —
    daily_total_input_cost in daily.parquet (the reconciling cost source, not the season ledger)."""
    d = eda.load()["daily"]
    wk_start = d.date.max() - pd.Timedelta(days=6)
    s = (d[d.date >= wk_start].groupby("date").daily_total_input_cost.sum()
         .reset_index().rename(columns={"daily_total_input_cost": "spend"}))
    return {"series": s, "start": wk_start, "end": d.date.max(), "total": float(s.spend.sum())}


def composite_risk_today(as_of_date, crops=None):
    """Plots at risk TODAY by the daily COMPOSITE HEALTH-SCORE risk from all_scores_complete.csv
    (risk_score_90 on a 10-90 scale; risk_category Low/Moderate/Elevated/High/Critical, higher =
    worse). A DIFFERENT signal from the spike model's 3-day forecast, and clean of profit
    (risk_score_90 corr ~ -0.35 with profit/ROI, not circular). The 90-scale score and category are
    surfaced as `risk_score` / `risk_level` for the UI. Read-only; falls back to the nearest
    available date if as_of is missing. Optional `crops` (list) filters to those crops (None/empty =
    all 120 plots). Returns (sorted_day_df[date,plot_id,crop,risk_score,risk_level], actual_date)."""
    sc = pd.read_csv(_ALL_SCORES, parse_dates=["date"],
                     usecols=["date", "plot_id", "crop", "risk_score_90", "risk_category"])
    target = pd.Timestamp(as_of_date)
    if target not in set(sc.date):
        target = sc.date.iloc[(sc.date - target).abs().argmin()]
    day = sc[sc.date == target]
    if crops:
        day = day[day.crop.isin(crops)]
    day = (day.rename(columns={"risk_score_90": "risk_score", "risk_category": "risk_level"})
           .sort_values("risk_score", ascending=False).reset_index(drop=True))
    return day, target


def risk_reconciliation(as_of_date):
    """How the composite health-score risk (today, all_scores_complete.risk_score_90) relates to the
    spike model (3-day) — backs the honest 'these are different lenses' note (correlation + top-10
    overlap). The two are near-orthogonal here (|corr| < 0.4, top-10 overlap 0)."""
    day, target = composite_risk_today(as_of_date)
    spike = rs.score_plots(as_of_date)[["plot_id", "spike_prob"]]
    j = day.merge(spike, on="plot_id")
    topc = set(day.nlargest(10, "risk_score").plot_id)
    tops = set(spike.nlargest(10, "spike_prob").plot_id)
    return {"corr": float(j.risk_score.corr(j.spike_prob)), "date": target,
            "overlap_top10": int(len(topc & tops)),
            "levels": day.risk_level.value_counts().to_dict()}


# --- Overview tab: filterable daily-spend series --------------------------------
def overview_spend(crops=None, start=None, end=None):
    """Daily total input spend over time from the canonical daily ledger
    (daily_total_input_cost in daily.parquet — the SAME source as weekly_spend, generalized for
    the Overview's crop + date-range filters; the season ledger is not used). Summed over the
    selected crops within [start, end] inclusive. Read-only. Returns a [date, spend] frame."""
    d = eda.load()["daily"]
    if crops:
        d = d[d.crop.isin(crops)]
    if start is not None:
        d = d[d.date >= pd.Timestamp(start)]
    if end is not None:
        d = d[d.date <= pd.Timestamp(end)]
    return (d.groupby("date").daily_total_input_cost.sum()
            .reset_index().rename(columns={"daily_total_input_cost": "spend"}))


def crops_list():
    """Sorted unique crop names (for the Overview crop filter). Read-only."""
    return sorted(eda.load()["daily"].crop.unique().tolist())


# --- Overview: three graphs built from the clean all_scores_complete.csv columns ----
# Clean 0-90 score columns (small profit corr: plot_health +0.35, env_stability +0.04, risk -0.35).
# final_score_90 / plot_score_90 are deliberately EXCLUDED (corr ~ -0.69 with profit -> profit-laden,
# never shown as a health axis).
SCORE_SERIES = [("plot_health_90", "Plot health"), ("env_stability_90", "Env stability"),
                ("risk_score_90", "Composite risk")]


@lru_cache(maxsize=1)
def _all_scores():
    """The full clean per-plot-per-day score panel (all_scores_complete.csv, 24,480 rows). Cached
    for the process (one 6 MB read). Read-only; callers filter/copy, never mutate."""
    return pd.read_csv(_ALL_SCORES, parse_dates=["date"])


def _filter_scores(crops, start, end):
    df = _all_scores()
    if crops:
        df = df[df.crop.isin(crops)]
    if start is not None:
        df = df[df.date >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.date <= pd.Timestamp(end)]
    return df


def scores_timeseries(crops=None, start=None, end=None):
    """Daily mean of the clean 0-90 scores (plot_health_90, env_stability_90, risk_score_90) across
    plots, from all_scores_complete.csv. Crop + date filtered. Returns [date, <score cols>]."""
    cols = [c for c, _ in SCORE_SERIES]
    return _filter_scores(crops, start, end).groupby("date")[cols].mean().reset_index()


def scores_by_dimension(dimension="treatment", metric="plot_health_90", crops=None, start=None, end=None):
    """Mean of a clean all_scores_complete score per category of `dimension`, weighting each plot
    equally (per-plot mean first). Crop + date filtered. Returns df[category, value, n] sorted by
    value. Read-only."""
    df = _filter_scores(crops, start, end)
    pm = df.groupby(["plot_id", dimension])[metric].mean().reset_index()
    g = (pm.groupby(dimension)[metric].agg(value="mean", n="size").reset_index()
         .rename(columns={dimension: "category"}).sort_values("value"))
    return g.reset_index(drop=True)


def scores_scatter(as_of_date, crops=None):
    """Per-plot plot_health_90 vs risk_score_90 (+ env_stability_90) on a given day, from
    all_scores_complete.csv, for the health-vs-risk scatter. Crop filtered; nearest-date fallback.
    Returns (df[plot_id, crop, plot_health_90, risk_score_90, env_stability_90], actual_date)."""
    df = _all_scores()
    target = pd.Timestamp(as_of_date)
    if target not in set(df.date):
        target = df.date.iloc[(df.date - target).abs().argmin()]
    day = df[df.date == target]
    if crops:
        day = day[day.crop.isin(crops)]
    return day[["plot_id", "crop", "plot_health_90", "risk_score_90",
                "env_stability_90"]].reset_index(drop=True), target


def health_category_distribution(as_of_date, crops=None):
    """Distribution of plots across plot_health_category (Excellent/Good/Fair/Poor/Critical) on a
    given day, from all_scores_complete.csv. Higher category = healthier (Excellent best, Critical
    worst). Filterable by crop; nearest-date fallback; empty categories dropped. Returns
    (counts_df[category, n], actual_date)."""
    df = _all_scores()
    target = pd.Timestamp(as_of_date)
    if target not in set(df.date):
        target = df.date.iloc[(df.date - target).abs().argmin()]
    day = df[df.date == target]
    if crops:
        day = day[day.crop.isin(crops)]
    order = ["Excellent", "Good", "Fair", "Poor", "Critical"]
    counts = (day.plot_health_category.value_counts()
              .reindex(order).fillna(0).astype(int).rename_axis("category").reset_index(name="n"))
    return counts[counts.n > 0].reset_index(drop=True), target


def date_range():
    return rs.date_range()


def headline():
    return fin.headline()


def finance_kpis():
    """Four headline FINANCIAL KPIs for the Financials tab (read-only, reuses the frozen finance
    layer). Portfolio figures: profit/m² = total season profit ÷ total plot area; cost/kg = total
    cost ÷ total kg produced; avg ROI = mean per-plot season ROI. total_precision_benefit is the
    frozen headline figure — an ASSUMPTION (= 0.65 × profit), surfaced with that flag in the UI,
    never as a measured return."""
    m = fin._m()
    h = fin.headline()
    kg = m.season_yield_kg_m2 * m.plot_area_m2
    return {
        "profit_per_m2": float(m.season_profit_cad.sum() / m.plot_area_m2.sum()),
        "avg_roi": float(h["mean_roi"]),
        "total_precision_benefit": float(h["total_precision_benefit"]),
        "cost_per_kg": float(m.total_cost_cad.sum() / kg.sum()),
    }

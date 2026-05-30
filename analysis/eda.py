"""Phase-1 EDA: reproduce every computed number in docs/eda_findings.md.

Read-only. Reads data/derived/*.parquet (never modifies). Computes the
precision-vs-routine analysis: the action-level response->stress mechanism,
the stress->outcome chain, the confounder hunt (treatment / alert burden),
treatment effects, and the financial reconciliation.

Run:  python analysis/eda.py            # prints all numbers used in the doc
Import: from analysis.eda import load, plot_features, ...   # for notebooks

Bootstrap uses a fixed seed (SEED=42) so CIs are reproducible.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
SEED = 42


# --------------------------------------------------------------------------- #
# Loading & per-plot feature construction
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load():
    """Return dict of the 6 derived parquets.

    Cached for the process (the parquet read is the heavy I/O). The cache resets when
    this module reloads, so a build_derived rerun or a code change re-reads fresh.
    Callers treat the frames as read-only (they filter/copy, never mutate in place)."""
    names = ["season", "daily", "actions", "scouting", "plot_meta", "bc_benchmarks"]
    return {n: pd.read_parquet(DERIVED / f"{n}.parquet") for n in names}


def plot_features(daily):
    """Collapse the daily log to one row per plot with the precision/stress
    signals the analysis needs. Two distinct 'action' notions live in daily:
      - action_taken / action_delay_days / post_action_stress_delta_3d: the
        SENSOR-alert response log (5,509 response-days; delta is 0 when no action).
      - daily_precision_actions_count / daily_precision_cost: the INPUT-application
        log (799 precision applications; reconciles with actions.is_precision_action).
    They are different granularities and do NOT reconcile in volume.
    """
    g = daily.groupby("plot_id")
    pm = pd.DataFrame(
        {
            "mean_stress": g["plant_stress_index"].mean(),
            "max_stress": g["plant_stress_index"].max(),
            "alert_days": g["alert_flag"].sum(),
            "action_days": g["action_taken"].sum(),
            "precision_actions": g["daily_precision_actions_count"].sum(),
            "total_actions": g["daily_total_actions_count"].sum(),
            "precision_cost": g["daily_precision_cost"].sum(),
            "routine_cost": g["daily_routine_cost"].sum(),
            "total_input_cost": g["daily_total_input_cost"].sum(),
            "total_stress_relief": g["post_action_stress_delta_3d"].sum(),
        }
    )
    # Behaviour on alert days only
    alert = daily[daily.alert_flag == 1]
    pm["alert_response_rate"] = alert.groupby("plot_id")["action_taken"].mean()
    responded = daily[(daily.alert_flag == 1) & (daily.action_taken == 1)]
    pm["mean_resp_delay"] = responded.groupby("plot_id")["action_delay_days"].mean()
    pm["missed_alerts"] = pm["alert_days"] * (1 - pm["alert_response_rate"])
    return pm.reset_index()


def merged(d=None):
    """season joined with per-plot daily features (one row per plot)."""
    d = d or load()
    return d["season"].merge(plot_features(d["daily"]), on="plot_id", how="left")


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def boot_mean_diff(a, b, n=10000, seed=SEED):
    """Bootstrap mean(a) - mean(b) with 95% percentile CI."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = rng.choice(a, (n, len(a)), replace=True).mean(1) - rng.choice(
        b, (n, len(b)), replace=True
    ).mean(1)
    return diff.mean(), np.percentile(diff, 2.5), np.percentile(diff, 97.5)


def within_group_corr(df, x, y, grp):
    """Pooled within-group (group-mean-centered) Pearson r: 'controls for' grp."""
    d = df[[x, y, grp]].dropna().copy()
    d[x] = d[x] - d.groupby(grp)[x].transform("mean")
    d[y] = d[y] - d.groupby(grp)[y].transform("mean")
    r, p = stats.pearsonr(d[x], d[y])
    return r, p, len(d)


def partial_corr(df, x, y, z):
    """Partial Pearson r between x and y controlling (linearly) for z."""
    d = df[[x, y, z]].dropna()

    def resid(a, b):
        b1 = np.c_[np.ones(len(b)), b]
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    Z = d[z].values.reshape(-1, 1)
    return stats.pearsonr(resid(d[x].values, Z), resid(d[y].values, Z))


# --------------------------------------------------------------------------- #
# Analyses (each returns numbers; main() prints them)
# --------------------------------------------------------------------------- #
def precision_benefit_formula(season):
    """precision_benefit_cad vs 0.65*profit (floored at 0)."""
    s = season.copy()
    pred = np.clip(0.65 * s.season_profit_cad, 0, None)
    pos = s[s.season_profit_cad > 0]
    ratio = pos.precision_benefit_cad / pos.season_profit_cad
    return {
        "corr_benefit_profit": s.precision_benefit_cad.corr(s.season_profit_cad),
        "max_abs_err_vs_clip065": (s.precision_benefit_cad - pred).abs().max(),
        "ratio_min": ratio.min(),
        "ratio_median": ratio.median(),
        "ratio_max": ratio.max(),
        "ratio_std": ratio.std(),
        "n_neg_profit": int((s.season_profit_cad < 0).sum()),
        "n_zero_benefit": int((s.precision_benefit_cad == 0).sum()),
        "total_benefit": s.precision_benefit_cad.sum(),
        "total_profit": s.season_profit_cad.sum(),
    }


def mechanism_delay_stress(daily):
    """Action-level: does faster response cut post-action stress more?"""
    ad = daily[daily.action_taken == 1]
    by = ad.groupby("action_delay_days")["post_action_stress_delta_3d"].agg(
        ["mean", "std", "count"]
    )
    same = ad[ad.action_delay_days == 0].post_action_stress_delta_3d
    delayed = ad[ad.action_delay_days >= 1].post_action_stress_delta_3d
    t, p = stats.ttest_ind(same, delayed, equal_var=False)
    ci = boot_mean_diff(same, delayed)
    # Does speed change the forward stress-SPIKE rate?
    sp = ad[ad.spike_3d.notna()].groupby("action_delay_days")["spike_3d"].mean()
    return {"by_delay": by, "same_n": len(same), "delayed_n": len(delayed),
            "diff_ci": ci, "welch_t": t, "welch_p": p, "spike_by_delay": sp}


def stress_outcomes(m):
    """Plot-level: season-mean stress vs each outcome (Pearson)."""
    out = {}
    for y in ["season_profit_cad", "season_roi", "season_yield_kg_m2",
              "marketable_ratio", "season_revenue_cad", "total_cost_cad"]:
        r, p = stats.pearsonr(m.mean_stress, m[y])
        out[y] = (r, p)
    return out


def responsiveness_outcomes(m):
    """Plot-level: does responsiveness predict stress / profit?"""
    out = {}
    for x in ["alert_response_rate", "mean_resp_delay", "precision_actions",
              "total_stress_relief"]:
        out[x] = {
            "vs_mean_stress": m[x].corr(m.mean_stress),
            "vs_profit": m[x].corr(m.season_profit_cad),
            "vs_roi": m[x].corr(m.season_roi),
        }
    return out


def confounders(m):
    """The credibility section: alert burden, within-strata, benefit-vs-effort."""
    res = {}
    res["corr_alertdays_stress"] = m.alert_days.corr(m.mean_stress)
    res["corr_alertdays_profit"] = m.alert_days.corr(m.season_profit_cad)
    res["stress_profit_raw"] = m.mean_stress.corr(m.season_profit_cad)
    res["stress_profit_within_treatment"] = within_group_corr(
        m, "mean_stress", "season_profit_cad", "treatment")
    res["stress_profit_within_crop"] = within_group_corr(
        m, "mean_stress", "season_profit_cad", "crop")
    res["stress_profit_partial_alertdays"] = partial_corr(
        m, "mean_stress", "season_profit_cad", "alert_days")
    res["respdelay_profit_partial_alertdays"] = partial_corr(
        m, "mean_resp_delay", "season_profit_cad", "alert_days")
    res["missed_stress_partial_alertdays"] = partial_corr(
        m, "missed_alerts", "mean_stress", "alert_days")
    res["missed_profit_partial_alertdays"] = partial_corr(
        m, "missed_alerts", "season_profit_cad", "alert_days")
    # benefit (=0.65*profit) vs precision effort, across treatments
    t = m.groupby("treatment").agg(
        precision_benefit=("precision_benefit_cad", "mean"),
        precision_actions=("precision_actions", "mean"),
        alert_days=("alert_days", "mean"))
    res["benefit_vs_precision_actions_xtreat"] = t.precision_benefit.corr(t.precision_actions)
    res["benefit_vs_alertdays_xtreat"] = t.precision_benefit.corr(t.alert_days)
    return res


def treatment_table(m):
    """Per-treatment means + bootstrap profit/stress diff vs Control."""
    agg = m.groupby("treatment").agg(
        n=("plot_id", "size"),
        profit=("season_profit_cad", "mean"),
        roi=("season_roi", "mean"),
        yield_kgm2=("season_yield_kg_m2", "mean"),
        marketable=("marketable_ratio", "mean"),
        mean_stress=("mean_stress", "mean"),
        alert_days=("alert_days", "mean"),
        precision_actions=("precision_actions", "mean"),
        mean_resp_delay=("mean_resp_delay", "mean"),
    ).sort_values("profit", ascending=False)
    ctrl = m[m.treatment == "Control"]
    diffs = {}
    for tr in m.treatment.unique():
        if tr == "Control":
            continue
        sub = m[m.treatment == tr]
        diffs[tr] = {
            "n": len(sub),
            "profit_vs_ctrl": boot_mean_diff(sub.season_profit_cad, ctrl.season_profit_cad),
            "stress_vs_ctrl": boot_mean_diff(sub.mean_stress, ctrl.mean_stress),
        }
    return agg, diffs


def financials(m, season):
    """Defensible vs non-defensible dollar figures."""
    return {
        "total_precision_benefit": season.precision_benefit_cad.sum(),
        "total_profit": season.season_profit_cad.sum(),
        "benefit_share_of_profit": season.precision_benefit_cad.sum() / season.season_profit_cad.sum(),
        "mean_benefit_per_plot": season.precision_benefit_cad.mean(),
        "precision_cost_total": m.precision_cost.sum(),
        "routine_cost_total": m.routine_cost.sum(),
        "total_input_cost": m.total_input_cost.sum(),
        "precision_cost_share": m.precision_cost.sum() / m.total_input_cost.sum(),
        "precision_action_share": m.precision_actions.sum() / m.total_actions.sum(),
        # cross-universe reconciliation (NOT a valid ROI):
        "corr_season_vs_daily_cost": season.set_index("plot_id").total_cost_cad.corr(
            m.set_index("plot_id").total_input_cost),
    }


def scouting_decoupling(m, scouting):
    """Scouting (2026 dates) vs sensor signals (2025): are they linked?"""
    sp = scouting.groupby("plot_id").agg(
        pest_sev=("pest_severity", "mean"),
        dis_sev=("disease_severity", "mean")).reset_index()
    j = m.merge(sp, on="plot_id")
    return {
        "scout_dates": (scouting.date.min(), scouting.date.max()),
        "corr_pestsev_alertdays": j.pest_sev.corr(j.alert_days),
        "corr_pestsev_stress": j.pest_sev.corr(j.mean_stress),
        "corr_pestsev_profit": j.pest_sev.corr(j.season_profit_cad),
    }


# --------------------------------------------------------------------------- #
def main():
    d = load()
    season, daily, scouting = d["season"], d["daily"], d["scouting"]
    m = merged(d)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n=== GRAIN ===")
    print(f"season {season.shape} | daily {daily.shape} ({daily.date.min().date()}->{daily.date.max().date()})")
    print(f"scouting {scouting.shape} ({scouting.date.min().date()}->{scouting.date.max().date()}) <- note 2026 offset")

    print("\n=== precision_benefit_cad = ~0.65 * profit (floored at 0) ===")
    for k, v in precision_benefit_formula(season).items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== MECHANISM: response delay -> post-action stress delta ===")
    mech = mechanism_delay_stress(daily)
    print(mech["by_delay"].round(4).to_string())
    print(f"  same-day vs delayed diff = {mech['diff_ci'][0]:.4f} "
          f"[95% CI {mech['diff_ci'][1]:.4f}, {mech['diff_ci'][2]:.4f}]  "
          f"Welch t={mech['welch_t']:.1f} p={mech['welch_p']:.2e}")
    print("  forward spike_3d rate by delay (speed does NOT change spike rate):")
    print(mech["spike_by_delay"].round(4).to_string())

    print("\n=== stress -> outcomes (plot level) ===")
    for y, (r, p) in stress_outcomes(m).items():
        print(f"  mean_stress vs {y:20s}: r={r:+.3f} p={p:.2e}")

    print("\n=== responsiveness -> outcomes (plot level: weak/absent) ===")
    for x, v in responsiveness_outcomes(m).items():
        print(f"  {x:20s}: vs_stress {v['vs_mean_stress']:+.3f} | vs_profit {v['vs_profit']:+.3f} | vs_roi {v['vs_roi']:+.3f}")

    print("\n=== CONFOUNDERS ===")
    c = confounders(m)
    print(f"  corr(alert_days, mean_stress) = {c['corr_alertdays_stress']:+.3f}")
    print(f"  stress->profit raw            = {c['stress_profit_raw']:+.3f}")
    print(f"  stress->profit within treatment = {c['stress_profit_within_treatment'][0]:+.3f} (p={c['stress_profit_within_treatment'][1]:.2f})")
    print(f"  stress->profit within crop      = {c['stress_profit_within_crop'][0]:+.3f} (p={c['stress_profit_within_crop'][1]:.2e})")
    print(f"  stress->profit | alert_days     = {c['stress_profit_partial_alertdays'][0]:+.3f} (p={c['stress_profit_partial_alertdays'][1]:.2f})")
    print(f"  benefit vs precision_actions (x treat) = {c['benefit_vs_precision_actions_xtreat']:+.3f}")
    print(f"  benefit vs alert_days        (x treat) = {c['benefit_vs_alertdays_xtreat']:+.3f}")

    print("\n=== TREATMENT TABLE ===")
    agg, diffs = treatment_table(m)
    print(agg.round(3).to_string())
    print("  vs Control (bootstrap 95% CI):")
    for tr, v in diffs.items():
        dp, ds = v["profit_vs_ctrl"], v["stress_vs_ctrl"]
        print(f"    {tr:16s} n={v['n']:2d} profit {dp[0]:+7.1f} [{dp[1]:+7.1f},{dp[2]:+7.1f}]  stress {ds[0]:+.3f} [{ds[1]:+.3f},{ds[2]:+.3f}]")

    print("\n=== FINANCIALS ===")
    for k, v in financials(m, season).items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== SCOUTING DECOUPLING ===")
    for k, v in scouting_decoupling(m, scouting).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

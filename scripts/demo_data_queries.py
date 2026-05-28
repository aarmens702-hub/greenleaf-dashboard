"""Demo each of the 10 data-query functions with default args.

Run: .venv/bin/python scripts/demo_data_queries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat import data_queries as dq


def show(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


show("1) season_overview()")
o = dq.season_overview()
print(f"  {o['n_farms']} farms, {o['n_greenhouses']} greenhouses, {o['n_plots']} plots, crops={o['crops']}")
print(f"  revenue ${o['total_revenue_cad']:,} | cost ${o['total_cost_cad']:,} | profit ${o['total_profit_cad']:,}")
print(f"  overall ROI {o['overall_roi']} | marketable {o['mean_marketable_ratio']} | yield {o['mean_yield_kg_m2']} kg/m2")

show("2) precision_benefit_summary()")
p = dq.precision_benefit_summary()
print(f"  precision benefit ${p['total_precision_benefit_cad']:,} of ${p['total_season_profit_cad']:,} profit = {p['precision_benefit_share_pct']}%")
print(f"  per plot: mean ${p['mean_per_plot_cad']} | median ${p['median_per_plot_cad']}")

show("3) response_time_payoff()")
print(dq.response_time_payoff().to_string(index=False))

show("4) avg_profit_by_crop()")
for crop, profit in sorted(dq.avg_profit_by_crop().items(), key=lambda kv: -kv[1]):
    print(f"  {crop:<12} ${profit}")

show("5) treatment_summary()")
print(dq.treatment_summary().to_string(index=False))

show("6) plot_status('P0001')")
s = dq.plot_status("P0001")
print(f"  {s['plot_id']}: {s['crop']} / {s['treatment']} in {s['greenhouse_id']} ({s['farm_id']})")
print(f"  season profit ${s.get('season_profit_cad')} | ROI {s.get('season_roi')}")
print(f"  {len(s['last_n_days'])} recent days; latest: {s['last_n_days'][-1]}")

show("7) top_at_risk_plots(<min date>)")
min_date = dq.DAILY["date"].min().strftime("%Y-%m-%d")
print(f"  window starting {min_date}")
print(dq.top_at_risk_plots(min_date).to_string(index=False))

show("8) greenleaf_vs_benchmark('greenhouse_tomato_annual_yield')")
b = dq.greenleaf_vs_benchmark("greenhouse_tomato_annual_yield")
print(f"  benchmark: {b['benchmark_value']} {b['benchmark_unit']} ({b['benchmark_source']})")
print(f"  greenleaf: {b['greenleaf_value']}")
print(f"  notes: {b['notes']}")

show("9) cost_breakdown()")
c = dq.cost_breakdown()
print(f"  filters: crop={c['crop_filter']}, treatment={c['treatment_filter']} | n_plot_days={c['n_plot_days']:,}")
for key in ["energy", "fertilizer", "labor", "pesticide", "water", "total_input"]:
    print(f"  mean_daily_{key}_cost: ${c['mean_daily_' + key + '_cost']}")

show("10) alert_response_summary()")
a = dq.alert_response_summary()
print(f"  {a['n_alerts']:,} alerts | acted on {a['n_acted_on']:,} ({a['action_rate_pct']}%)")
print(f"  mean delay {a['mean_delay_days']} days | mean 3d recovery {a['mean_recovery_3d']}")

print("\n[demo] done")

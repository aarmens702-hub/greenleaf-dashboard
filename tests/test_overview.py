"""Overview tab: read-only helpers reconcile with the frozen layer + the clean all_scores_complete
columns, the crop/date filters are interactive, and the page renders offline with the KPI row on top
and all six panels (spend line, risk table, health donut, scores line, health-by-treatment bar,
health-vs-risk scatter)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.streamlit_app import insights as I


# --- helpers reconcile with the frozen layer ---------------------------------
def test_overview_spend_reconciles_with_cost_structure():
    """Full-season, all-crop spend must equal the frozen daily cost ledger total."""
    s_all = I.overview_spend()
    assert len(s_all) == 210                                  # one row per season day
    assert round(s_all.spend.sum()) == 404050                 # == cost_structure / total_input_cost
    assert (s_all.spend > 0).all()


def test_health_category_distribution():
    counts, _ = I.health_category_distribution("2025-09-12")
    assert list(counts.columns) == ["category", "n"]
    assert int(counts.n.sum()) == 120                              # every plot categorized once
    # categories come from all_scores_complete.plot_health_category (Excellent best, Critical worst)
    assert set(counts.category).issubset({"Excellent", "Good", "Fair", "Poor", "Critical"})


# --- filters actually change the outputs (interactivity at the data layer) ----
def test_crop_filter_changes_all_three_panels():
    all_crops = I.crops_list()
    assert all_crops == ["Cucumber", "Pepper", "Strawberry", "Tomato"]
    # G1 spend: one crop spends strictly less than all crops
    assert I.overview_spend(["Strawberry"]).spend.sum() < I.overview_spend().spend.sum()
    # G2 composite-risk table: one crop is a strict subset of all 120 plots
    day_all, _ = I.composite_risk_today("2025-09-12")
    day_straw, _ = I.composite_risk_today("2025-09-12", ["Strawberry"])
    assert len(day_all) == 120 and 0 < len(day_straw) < 120
    # G3 health donut (plot_health_category): one crop sums to fewer plots than all
    c_all, _ = I.health_category_distribution("2025-09-12")
    c_straw, _ = I.health_category_distribution("2025-09-12", ["Strawberry"])
    assert c_straw.n.sum() < c_all.n.sum()


def test_date_filter_changes_spend_and_health():
    # G1 spend: a narrow window has fewer points and less total than the full season
    win = I.overview_spend(start="2025-08-01", end="2025-08-31")
    assert len(win) == 31 and win.spend.sum() < I.overview_spend().spend.sum()
    # G3 health: the category distribution differs between an early and a late date
    early, _ = I.health_category_distribution("2025-03-15")
    late, _ = I.health_category_distribution("2025-09-12")
    assert list(early.category) != list(late.category)


def test_composite_risk_backward_compatible():
    """The added crops= param must not change the default (all-plots) behaviour."""
    day, _ = I.composite_risk_today("2025-08-15")
    assert len(day) == 120 and day.risk_score.is_monotonic_decreasing


# --- page renders offline -----------------------------------------------------
def test_overview_imports_cleanly():
    from app.streamlit_app.overview_page import render_overview
    assert callable(render_overview)


def test_render_overview_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    script = (
        f"import sys; sys.path.insert(0, r'{root}')\n"
        "from app.streamlit_app.overview_page import render_overview\n"
        "render_overview()\n"
    )
    at = AppTest.from_string(script).run(timeout=120)
    assert not at.exception, at.exception
    md = " ".join(b.value for b in at.markdown)
    # KPI row pinned at the very top
    for card in ["Avg plot performance", "Avg plot health", "Avg ROI", "Weekly response rate", "Profit per m"]:
        assert card in md, card
    assert "profit-free score" in md
    # the six panels are present and labelled
    assert "At a Glance" in md
    assert "Input spending over time" in md
    assert "Plots at risk" in md
    assert "Plot-health mix" in md
    assert "Plot scores over the season" in md       # 04 · all_scores line
    assert "Average plot health by treatment" in md  # 05 · all_scores bar
    assert "Plot health vs composite risk" in md     # 06 · all_scores scatter
    # the removed group-by / metric explorer must be gone
    assert "Compare by dimension" not in md
    assert "Group by" not in md
    # global controls: crop + date only (no group-by selectbox, no metric radio)
    assert len(at.multiselect) == 1                  # crop filter
    assert len(at.date_input) == 1                   # date range
    assert len(at.selectbox) == 0                    # group-by + metric selectors removed
    assert len(at.radio) == 0                        # metric-group radio removed
    assert len(at.dataframe) == 1                    # the composite at-risk table


def test_render_overview_filters_are_interactive():
    """Changing the crop filter must change what the panels render (not a static screenshot)."""
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    script = (
        f"import sys; sys.path.insert(0, r'{root}')\n"
        "from app.streamlit_app.overview_page import render_overview\n"
        "render_overview()\n"
    )
    at = AppTest.from_string(script).run(timeout=120)
    md_all = " ".join(b.value for b in at.markdown)
    at.multiselect[0].set_value(["Strawberry"]).run(timeout=120)
    md_straw = " ".join(b.value for b in at.markdown)
    assert md_all != md_straw                         # panels responded to the crop filter
    assert "Strawberry" in md_straw                   # scope label reflects the selection


# --- the three all_scores_complete graphs (line / bar / scatter) --------------
def test_scores_timeseries_clean_columns():
    g = I.scores_timeseries()
    assert list(g.columns) == ["date", "plot_health_90", "env_stability_90", "risk_score_90"]
    assert len(g) == 204                               # one row per scored day
    # all three scores live on the 0-90 scale; profit-laden composites are excluded
    for c in ["plot_health_90", "env_stability_90", "risk_score_90"]:
        assert 0 <= g[c].min() and g[c].max() <= 90
    assert "final_score_90" not in g.columns and "plot_score_90" not in g.columns


def test_scores_by_treatment_bar_matches_eda():
    g = I.scores_by_dimension("treatment", "plot_health_90")
    assert len(g) == 7 and int(g.n.sum()) == 120       # every plot weighted once across 7 treatments
    assert g.value.is_monotonic_increasing             # returned sorted ascending
    assert g.iloc[-1].category == "High Light"         # healthiest treatment (matches EDA)
    assert g.iloc[0].category == "Shade"               # least healthy


def test_scores_scatter_inverse_health_risk():
    df, target = I.scores_scatter("2025-09-12")
    assert len(df) == 120
    assert {"plot_id", "crop", "plot_health_90", "risk_score_90"}.issubset(df.columns)
    assert df.plot_health_90.corr(df.risk_score_90) < 0    # health and risk move inversely


def test_score_graphs_respond_to_crop_and_date():
    # crop filter changes the bar + scatter slices
    assert not I.scores_by_dimension("treatment", "plot_health_90").equals(
        I.scores_by_dimension("treatment", "plot_health_90", ["Strawberry"]))
    straw_scatter, _ = I.scores_scatter("2025-09-12", ["Strawberry"])
    assert 0 < len(straw_scatter) < 120
    # date filter narrows the time series
    assert len(I.scores_timeseries(start="2025-08-01", end="2025-08-31")) == 31

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.retrieve import retrieve


def test_strawberry_query():
    results = retrieve("is precision irrigation good for strawberries")
    assert len(results) == 5
    sources = {r["source_file"] for r in results}
    assert sources & {"precision_ag_roi_evidence.md", "crop_growing_characteristics.md"}, (
        f"expected a precision/crop doc in top-5, got {sources}"
    )


def test_heatwave_query():
    results = retrieve("how hot did the 2021 BC heatwave get")
    sources = {r["source_file"] for r in results}
    assert "bc_climate_heat.md" in sources, f"expected bc_climate_heat.md in top-5, got {sources}"


def test_scores_descending():
    results = retrieve("greenhouse energy costs")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

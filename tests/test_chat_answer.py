import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat import answer as ans


def _chunk(cid, url, title="Doc", text="some text"):
    return {
        "chunk_id": cid, "text": text, "source_file": f"{title}.md",
        "title": title, "source_name": "S", "source_url": url, "score": 0.9,
    }


@mock.patch("app.chat.answer._generate_answer")
@mock.patch("app.chat.answer.retrieve")
@mock.patch("app.chat.answer.classify")
def test_doc_only_question(mock_classify, mock_retrieve, mock_gen):
    mock_classify.return_value = {
        "needs_docs": True, "needs_data": False,
        "data_functions": [], "data_args": {}, "reasoning": "docs",
    }
    mock_retrieve.return_value = [
        _chunk(0, "https://example.com/x", "DocX"),
        _chunk(1, "https://example.com/y", "DocY"),
    ]
    mock_gen.return_value = (
        "Precision irrigation improves water-use efficiency (https://example.com/x) "
        "and yields (https://example.com/y)."
    )
    out = ans.answer("what does research say about precision irrigation")
    assert out["answer"] == mock_gen.return_value
    doc_urls = {s["source_url"] for s in out["sources"] if s["type"] == "doc"}
    assert doc_urls == {"https://example.com/x", "https://example.com/y"}
    assert out["trace"]["n_doc_chunks"] == 2
    assert out["trace"]["data_calls"] == []


@mock.patch("app.chat.answer._generate_answer", return_value="Precision benefit is large [precision_benefit_summary].")
@mock.patch("app.chat.answer.retrieve")
@mock.patch("app.chat.answer.classify")
def test_data_only_question(mock_classify, mock_retrieve, mock_gen):
    mock_classify.return_value = {
        "needs_docs": False, "needs_data": True,
        "data_functions": ["precision_benefit_summary"],
        "data_args": {"precision_benefit_summary": {}}, "reasoning": "data",
    }
    with mock.patch.dict(ans.DISPATCH, {"precision_benefit_summary": lambda **k: {"precision_benefit_share_pct": 65.6}}):
        out = ans.answer("what is the precision benefit")
    assert len(out["trace"]["data_calls"]) == 1
    assert out["trace"]["data_calls"][0]["status"] == "ok"
    mock_retrieve.assert_not_called()


@mock.patch("app.chat.answer._generate_answer")
@mock.patch("app.chat.answer.retrieve")
@mock.patch("app.chat.answer.classify")
def test_mixed_question(mock_classify, mock_retrieve, mock_gen):
    mock_classify.return_value = {
        "needs_docs": True, "needs_data": True,
        "data_functions": ["avg_profit_by_crop"], "data_args": {}, "reasoning": "both",
    }
    mock_retrieve.return_value = [_chunk(0, "https://example.com/z", "DocZ")]
    mock_gen.return_value = "Strawberry pays best [avg_profit_by_crop]; context (https://example.com/z)."
    with mock.patch.dict(ans.DISPATCH, {"avg_profit_by_crop": lambda **k: {"Strawberry": 678.69}}):
        out = ans.answer("which crop pays best and what does research say")
    mock_retrieve.assert_called_once()
    assert any(c["name"] == "avg_profit_by_crop" and c["status"] == "ok" for c in out["trace"]["data_calls"])
    assert {s["type"] for s in out["sources"]} == {"doc", "data"}


@mock.patch("app.chat.answer._generate_answer", return_value="I couldn't load that plot's data.")
@mock.patch("app.chat.answer.retrieve")
@mock.patch("app.chat.answer.classify")
def test_data_function_error_is_handled(mock_classify, mock_retrieve, mock_gen):
    mock_classify.return_value = {
        "needs_docs": False, "needs_data": True,
        "data_functions": ["plot_status"],
        "data_args": {"plot_status": {"plot_id": "DOESNOTEXIST"}}, "reasoning": "data",
    }

    def _raise(**kwargs):
        raise ValueError("plot not found")

    with mock.patch.dict(ans.DISPATCH, {"plot_status": _raise}):
        out = ans.answer("how is plot DOESNOTEXIST doing")
    assert len(out["trace"]["data_calls"]) == 1
    assert out["trace"]["data_calls"][0]["status"] == "error"
    assert out["trace"]["data_calls"][0]["error_msg"] == "plot not found"
    assert out["answer"] == "I couldn't load that plot's data."

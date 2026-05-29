"""Agent-loop tests with a MOCKED LLM (no live API calls).

We patch app.chat.agent._chat to return a scripted sequence of assistant
messages (tool calls then a final answer) and assert the loop executes tools,
adjusts, recovers from errors, and stops at max_iterations.
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.chat import agent


def _tool_msg(name, arguments="{}", call_id="c1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": name, "arguments": arguments}},
        ],
    }


def _final_msg(text):
    return {"role": "assistant", "content": text}


def test_p0_tools_are_wired_and_execute():
    """P0: the 5 newly-exposed functions are real tools and execute cleanly."""
    new_tools = ["season_overview", "avg_profit_by_crop", "plot_status",
                 "cost_breakdown", "alert_response_summary"]
    schema_names = {t["function"]["name"] for t in agent.TOOL_SCHEMAS}
    for name in new_tools:
        assert name in schema_names, f"{name} missing from TOOL_SCHEMAS"
        assert name in agent._FIXED_TOOLS
    # No-arg / optional-arg tools run; plot_status takes a plot_id.
    assert agent._execute_tool("season_overview", {})[1] == "ok"
    assert agent._execute_tool("avg_profit_by_crop", {})[1] == "ok"
    assert agent._execute_tool("cost_breakdown", {})[1] == "ok"
    assert agent._execute_tool("alert_response_summary", {})[1] == "ok"
    result, status = agent._execute_tool("plot_status", {"plot_id": "P0001"})
    assert status == "ok" and result["crop"] == "Strawberry"


def test_single_tool():
    """Model requests one tool, then answers."""
    script = [
        _tool_msg("precision_benefit_summary"),
        _final_msg("Precision benefit is large [precision_benefit_summary]."),
    ]
    with mock.patch.object(agent, "_chat", side_effect=script):
        out = agent.answer("what is the precision benefit")
    assert out["answer"].startswith("Precision benefit is large")
    assert len(out["trace"]["tool_calls"]) == 1
    assert out["trace"]["tool_calls"][0] == {
        "name": "precision_benefit_summary", "args": {}, "status": "ok",
    }
    assert out["trace"]["hit_max"] is False
    # A successful data tool becomes a data source.
    assert any(s["type"] == "data" and s["tool"] == "precision_benefit_summary"
               for s in out["sources"])


def test_multi_tool_adjust():
    """Model calls query_data, gets an empty result, ADJUSTS to a different tool, then answers."""
    script = [
        _tool_msg("query_data", '{"table": "season", "filters": {"crop": "Nonexistent"}}', "c1"),
        _tool_msg("treatment_summary", "{}", "c2"),
        _final_msg("High Light is most profitable [treatment_summary]."),
    ]
    with mock.patch.object(agent, "_chat", side_effect=script):
        out = agent.answer("which treatment is most profitable")
    names = [c["name"] for c in out["trace"]["tool_calls"]]
    assert names == ["query_data", "treatment_summary"]   # both ran, in order
    assert all(c["status"] == "ok" for c in out["trace"]["tool_calls"])
    assert out["trace"]["iterations"] == 3                # 2 tool turns + 1 answer turn
    assert out["answer"].startswith("High Light")


def test_max_iterations():
    """Model keeps requesting tools forever -> loop stops at max_iterations and still answers."""
    always_tool = lambda *a, **k: _tool_msg("precision_benefit_summary")
    with mock.patch.object(agent, "_chat", side_effect=always_tool):
        out = agent.answer("loop forever", max_iterations=4)
    assert out["trace"]["hit_max"] is True
    assert out["trace"]["iterations"] == 4
    assert len(out["trace"]["tool_calls"]) == 4
    assert isinstance(out["answer"], str) and out["answer"]   # a final answer is still returned


def test_tool_error_recovery():
    """A tool that raises -> error captured in trace, loop continues, doesn't crash."""
    script = [
        _tool_msg("search_documents", '{"query": "anything"}'),
        _final_msg("Here is what I found anyway."),
    ]
    with mock.patch.object(agent, "_chat", side_effect=script), \
         mock.patch.object(agent, "retrieve", side_effect=RuntimeError("index unavailable")):
        out = agent.answer("what does research say")
    assert len(out["trace"]["tool_calls"]) == 1
    assert out["trace"]["tool_calls"][0]["name"] == "search_documents"
    assert out["trace"]["tool_calls"][0]["status"] == "error"
    assert out["answer"] == "Here is what I found anyway."   # loop survived the tool error

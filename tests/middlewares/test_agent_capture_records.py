"""Tests for the unified agent-capture middleware (tool-records channel)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from muffin_agent.middlewares.agent_capture.records import (
    MAX_RECORDS_PER_CAPTURE,
    OUTPUT_PREVIEW,
    build_tool_records,
    merge_tool_runs,
)
from muffin_agent.middlewares.tool_result_cache.cache import get_args_hash


def _ai(tool_name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": tool_name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


@pytest.mark.unit
class TestBuildToolRecords:
    def test_ok_record_pairs_call_and_result(self):
        messages = [
            _ai("etf_equity_exposure", {"ticker": "AAPL"}, "c1"),
            ToolMessage(content='{"sector": "tech"}', tool_call_id="c1"),
        ]
        records = build_tool_records(messages, agent_name="ticker_classification")
        assert len(records) == 1
        r = records[0]
        assert r["tool"] == "etf_equity_exposure"
        assert r["agent"] == "ticker_classification"
        assert r["status"] == "ok"
        assert r["is_subagent_call"] is False
        assert '"ticker": "AAPL"' in r["args_preview"]
        assert "tech" in r["output_preview"]
        assert r["error"] is None
        # args_hash IS the tool-result-cache store key — lets the UI join a
        # tool-run to its cached payload under ("cache", tool) with no rehashing.
        assert r["args_hash"] == get_args_hash({"ticker": "AAPL"})

    def test_error_status_message(self):
        messages = [
            _ai("equity_estimates", {}, "c1"),
            ToolMessage(content="boom", tool_call_id="c1", status="error"),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert records[0]["status"] == "error"
        assert records[0]["error"] == "boom"
        assert records[0]["output_preview"] == ""  # no output preview on failure

    def test_error_prefix_content(self):
        messages = [
            _ai("equity_estimates", {}, "c1"),
            ToolMessage(content="Error (permanent): missing key", tool_call_id="c1"),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert records[0]["status"] == "error"

    def test_duplicate_blocked(self):
        messages = [
            _ai("news_company", {}, "c1"),
            ToolMessage(
                content="DUPLICATE CALL BLOCKED: previously failed", tool_call_id="c1"
            ),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert records[0]["status"] == "duplicate_blocked"
        assert "DUPLICATE" in records[0]["error"]

    def test_cache_hit_flag(self):
        messages = [
            _ai("etf_equity_exposure", {}, "c1"),
            ToolMessage(
                content="{}",
                tool_call_id="c1",
                additional_kwargs={"cache": {"hit": True}},
            ),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert records[0]["cache_hit"] is True

    def test_task_call_tagged_as_subagent(self):
        messages = [
            _ai("task", {"subagent_type": "equity-fundamentals"}, "c1"),
            ToolMessage(content="collected", tool_call_id="c1"),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert records[0]["tool"] == "task"
        assert records[0]["is_subagent_call"] is True

    def test_excluded_tools_are_skipped(self):
        messages = [
            _ai("write_todos", {"todos": []}, "c1"),
            ToolMessage(content="ok", tool_call_id="c1"),
            _ai("etf_equity_exposure", {}, "c2"),
            ToolMessage(content="{}", tool_call_id="c2"),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert [r["tool"] for r in records] == ["etf_equity_exposure"]

    def test_output_preview_capped(self):
        big = "x" * (OUTPUT_PREVIEW * 2)
        messages = [
            _ai("equity_price", {}, "c1"),
            ToolMessage(content=big, tool_call_id="c1"),
        ]
        records = build_tool_records(messages, agent_name="a")
        assert len(records[0]["output_preview"]) <= OUTPUT_PREVIEW + 1  # + ellipsis

    def test_cap_appends_truncated_marker(self):
        messages: list = []
        for i in range(MAX_RECORDS_PER_CAPTURE + 5):
            messages.append(_ai("equity_price", {"i": i}, f"c{i}"))
            messages.append(ToolMessage(content="{}", tool_call_id=f"c{i}"))
        records = build_tool_records(messages, agent_name="a")
        assert len(records) == MAX_RECORDS_PER_CAPTURE + 1
        assert records[-1]["status"] == "truncated"
        assert records[-1]["args_hash"] is None

    def test_unmatched_tool_message_ignored(self):
        messages = [ToolMessage(content="orphan", tool_call_id="unknown")]
        assert build_tool_records(messages, agent_name="a") == []


@pytest.mark.unit
class TestMergeToolRuns:
    def test_concatenates(self):
        assert merge_tool_runs([{"a": 1}], [{"b": 2}]) == [{"a": 1}, {"b": 2}]

    def test_handles_none(self):
        assert merge_tool_runs(None, [{"b": 2}]) == [{"b": 2}]
        assert merge_tool_runs([{"a": 1}], None) == [{"a": 1}]


# ── Live capture through a real agent ─────────────────────────────────────────


@tool
def fake_probe(ticker: str) -> str:
    """Return a canned payload for *ticker*.

    Args:
        ticker: The symbol to look up.
    """
    return f'{{"ticker": "{ticker}", "value": 42}}'


def _scripted_probe_agent():
    """A minimal ReAct agent scripted to call ``fake_probe`` once then finish."""
    from muffin_agent.utils.agent_builder import MuffinAgentBuilder
    from tests.integration._harness.scripted_model import (
        Script,
        ScriptedChatModel,
        final,
        tool_turn,
    )

    cursor = Script([tool_turn("fake_probe", {"ticker": "AAPL"}), final("done")])
    model = ScriptedChatModel(script=cursor)
    return (
        MuffinAgentBuilder(model, name="probe")
        .with_tool(fake_probe)
        .build_react_agent()
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_middleware_emits_tool_runs_on_agent_output():
    """Capture is unconditional: `tool_runs` surfaces on the agent's output.

    This is the propagation contract: the middleware writes `tool_runs` into
    the agent's own state, so a parent graph declaring the same channel (with a
    reducer) receives it when the compiled agent is added as a node. No config
    flag is required — graphs opt in by declaring the channel, and parents that
    don't declare it drop the records at their boundary. (The previous
    ``tool_telemetry_enabled`` gate read the ambient ``get_config()`` inside
    ``aafter_agent``, which proved unreliable on the deployed runtime.)
    """
    agent = _scripted_probe_agent()
    result = await agent.ainvoke(
        {"messages": [HumanMessage("look up AAPL")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    runs = result.get("tool_runs") or []
    assert len(runs) == 1
    assert runs[0]["tool"] == "fake_probe"
    assert runs[0]["status"] == "ok"
    assert runs[0]["agent"] == "probe"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_format_schema_call_is_not_recorded():
    """The synthetic structured-output tool call must not appear in tool_runs.

    The builder excludes the response schema's class name from capture — the
    final ``Out`` tool call is plumbing, not a data-collection step.
    """
    from pydantic import BaseModel

    from muffin_agent.utils.agent_builder import MuffinAgentBuilder
    from tests.integration._harness.scripted_model import (
        Script,
        ScriptedChatModel,
        tool_turn,
    )

    class Out(BaseModel):
        answer: str

    cursor = Script(
        [
            tool_turn("fake_probe", {"ticker": "AAPL"}),
            tool_turn("Out", {"answer": "done"}),
        ]
    )
    agent = (
        MuffinAgentBuilder(ScriptedChatModel(script=cursor), name="probe")
        .with_tool(fake_probe)
        .with_response_format(Out)
        .build_react_agent()
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("look up AAPL")]},
        config={"configurable": {"thread_id": "t1"}},
    )

    tools = [r["tool"] for r in result.get("tool_runs") or []]
    assert tools == ["fake_probe"]  # the Out schema call is excluded

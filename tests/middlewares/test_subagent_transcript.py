"""Tests for the subagent-transcript capture middleware."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from deepagents import create_deep_agent
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from muffin_agent.middlewares import (
    SubagentTranscriptMiddleware,
    SubagentTranscriptParentMiddleware,
)
from muffin_agent.middlewares.subagent_transcript import merge_subagent_runs


class _Scripted(BaseChatModel):
    """Minimal chat model that replays a scripted list of AI messages."""

    responses: list

    def bind_tools(self, *a: Any, **k: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    @property
    def _llm_type(self) -> str:
        return "scripted"


@pytest.mark.unit
def test_merge_reducer_accumulates() -> None:
    assert merge_subagent_runs({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert merge_subagent_runs(None, {"b": 2}) == {"b": 2}
    assert merge_subagent_runs({"a": 1}, None) == {"a": 1}


@pytest.mark.unit
def test_child_after_agent_captures_transcript() -> None:
    mw = SubagentTranscriptMiddleware(name="equity_fundamentals")
    state = {
        "messages": [
            HumanMessage("get AAPL fundamentals"),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "equity_income", "args": {"t": "AAPL"}, "id": "t1"}
                ],
            ),
            ToolMessage("revenue=100", tool_call_id="t1"),
            AIMessage("done"),
        ]
    }
    update = mw.after_agent(state, MagicMock())
    assert update is not None
    runs = update["subagent_runs"]
    (record,) = runs.values()
    assert record["name"] == "equity_fundamentals"
    assert record["description"] == "get AAPL fundamentals"
    types = [m["type"] for m in record["messages"]]
    assert types == ["human", "ai", "tool", "ai"]
    # the tool call is preserved for the nested timeline
    assert record["messages"][1]["tool_calls"][0]["name"] == "equity_income"


@pytest.mark.unit
def test_child_after_agent_ignores_empty() -> None:
    mw = SubagentTranscriptMiddleware(name="x")
    assert mw.after_agent({"messages": []}, MagicMock()) is None


@pytest.mark.unit
def test_transcript_merges_up_into_parent_state() -> None:
    """The child's transcript reaches parent thread state via the task tool."""
    child = create_agent(
        model=_Scripted(responses=[AIMessage("fetched revenue=100")]),
        middleware=[SubagentTranscriptMiddleware(name="equity_fundamentals")],
    )
    subagent = CompiledSubAgent(
        name="equity-fundamentals", description="fundamentals", runnable=child
    )
    parent = create_deep_agent(
        model=_Scripted(
            responses=[
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "get AAPL fundamentals",
                                "subagent_type": "equity-fundamentals",
                            },
                            "id": "c1",
                        }
                    ],
                ),
                AIMessage("final"),
            ]
        ),
        subagents=[subagent],
        middleware=[SubagentTranscriptParentMiddleware()],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "T"}}
    parent.invoke({"messages": [HumanMessage("evaluate AAPL")]}, cfg)
    runs = parent.get_state(cfg).values.get("subagent_runs")
    assert runs, "subagent_runs should be present in parent state"
    (record,) = runs.values()
    assert record["name"] == "equity_fundamentals"
    assert record["description"] == "get AAPL fundamentals"
    assert any("revenue=100" in m.get("content", "") for m in record["messages"])


@pytest.mark.unit
def test_guarded_capture_noop_when_not_subagent() -> None:
    """A deep agent's guarded child capturer stays silent at the top level."""
    agent = create_deep_agent(
        model=_Scripted(responses=[AIMessage("orchestrator answer")]),
        middleware=[
            SubagentTranscriptParentMiddleware(),
            SubagentTranscriptMiddleware(name="stock_evaluation", subagent_only=True),
        ],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "T-top"}}
    agent.invoke({"messages": [HumanMessage("go")]}, cfg)
    runs = agent.get_state(cfg).values.get("subagent_runs")
    assert not runs, "top-level deep agent must not capture its own transcript"


@pytest.mark.unit
def test_guarded_capture_fires_for_deep_agent_as_subagent() -> None:
    """A deep agent wrapped in CompiledSubAgent leaves a transcript upstream."""
    child_deep = create_deep_agent(
        model=_Scripted(responses=[AIMessage("nested deep agent worked")]),
        middleware=[
            SubagentTranscriptParentMiddleware(),
            SubagentTranscriptMiddleware(name="nested_deep", subagent_only=True),
        ],
    )
    subagent = CompiledSubAgent(
        name="nested-deep",
        description="a deep agent as a subagent",
        runnable=child_deep,
    )
    parent = create_deep_agent(
        model=_Scripted(
            responses=[
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "delegate to the nested deep agent",
                                "subagent_type": "nested-deep",
                            },
                            "id": "c1",
                        }
                    ],
                ),
                AIMessage("final"),
            ]
        ),
        subagents=[subagent],
        middleware=[SubagentTranscriptParentMiddleware()],
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "T-nested"}}
    parent.invoke({"messages": [HumanMessage("go")]}, cfg)
    runs = parent.get_state(cfg).values.get("subagent_runs")
    assert runs, "the nested deep agent's transcript should merge up"
    assert any(r["name"] == "nested_deep" for r in runs.values())

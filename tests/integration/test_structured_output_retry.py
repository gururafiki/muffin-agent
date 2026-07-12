"""E2E: the structured-output retry guard through a real ReAct loop.

Reproduces the upstream langchain bug (present through at least 1.3.13) that
failed criteria_analysis in prod (thread ``019f55f0-8a7f-...``): when the
model pairs a MALFORMED structured-output tool call with a regular tool call
in the same ``AIMessage``, ``_make_tools_to_model_edge`` exit condition 3
("a structured output tool was executed") ends the loop before the model can
read the parse-error feedback — ``structured_response`` stays ``None``.

``_StructuredOutputRetryMiddleware`` (wired automatically by
``MuffinAgentBuilder`` whenever a response format is set) jumps back to the
model from ``after_agent`` so the retry happens in-loop.
"""

from __future__ import annotations

import pytest
from langchain.agents.structured_output import AutoStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from muffin_agent.utils._structured_output_retry_middleware import (
    StructuredOutputRetryExhaustedError,
)
from muffin_agent.utils.agent_builder import MuffinAgentBuilder

from ._harness import Script, ScriptedChatModel, final, tool_turn

pytestmark = pytest.mark.asyncio


class ClassificationOutput(BaseModel):
    """Stand-in for a node-output schema (e.g. TickerClassificationNodeOutput)."""

    ticker: str
    sector: str


@tool
def write_note(text: str) -> str:
    """Write a note (stand-in for write_todos — any regular tool works)."""
    return f"noted: {text}"


def _malformed_combo_turn(i: int = 0) -> AIMessage:
    """Regular tool call + malformed structured call in ONE AIMessage.

    The malformed args mirror the prod failure: a list wrapped in
    ``{'item': [...]}`` and a missing required field.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "write_note", "args": {"text": f"note{i}"}, "id": f"n{i}"},
            {
                "name": "ClassificationOutput",
                "args": {"sector": {"item": ["tech"]}},
                "id": f"s{i}",
            },
        ],
    )


_GOOD_TURN = tool_turn(
    "ClassificationOutput", {"ticker": "AAPL", "sector": "tech"}, id="good"
)


def _build_agent(script: Script):
    return (
        MuffinAgentBuilder(ScriptedChatModel(script=script), name="retry-e2e")
        .with_tool(write_note)
        .with_response_format(AutoStrategy(schema=ClassificationOutput))
        .build_react_agent()
    )


async def test_malformed_structured_call_with_sibling_tool_call_is_retried():
    """The prod failure mode: the guard recovers the lost retry in-loop."""
    script = Script([_malformed_combo_turn(), _GOOD_TURN])
    agent = _build_agent(script)

    result = await agent.ainvoke({"messages": [HumanMessage("classify AAPL")]})

    assert result["structured_response"] == ClassificationOutput(
        ticker="AAPL", sector="tech"
    )
    # Exactly one retry: the guard jumped back to the model once.
    assert script.consumed == 2


async def test_plain_text_end_is_nudged_to_structured_output():
    """Model ends in prose without calling the schema tool: nudge + retry."""
    script = Script([final("here is my analysis in prose"), _GOOD_TURN])
    agent = _build_agent(script)

    result = await agent.ainvoke({"messages": [HumanMessage("classify AAPL")]})

    assert result["structured_response"] == ClassificationOutput(
        ticker="AAPL", sector="tech"
    )


async def test_exhaustion_raises_for_node_retry_policy_backstop():
    """Three malformed attempts: fail loudly instead of looping forever."""
    script = Script([_malformed_combo_turn(i) for i in range(3)])
    agent = _build_agent(script)

    with pytest.raises(StructuredOutputRetryExhaustedError):
        await agent.ainvoke({"messages": [HumanMessage("classify AAPL")]})

    assert script.consumed == 3

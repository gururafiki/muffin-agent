"""Tests for ``_StructuredOutputRetryMiddleware`` and its builder wiring.

The middleware recovers the in-loop structured-output retry that langchain's
``_make_tools_to_model_edge`` loses (through at least 1.3.13) when a malformed
structured-output tool call shares an ``AIMessage`` with a regular tool call
(observed in prod: criteria_analysis thread 019f55f0-..., minimax-m3 emitted
``write_todos`` + a malformed ``TickerClassificationNodeOutput`` call; the
parse-error feedback was appended but the loop exited before the model could
read it).  The real-ReAct-loop repro lives in
``tests/integration/test_structured_output_retry.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.structured_output import AutoStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from muffin_agent.utils._structured_output_retry_middleware import (
    _NUDGE_MARKER,
    StructuredOutputRetryExhaustedError,
    _StructuredOutputRetryMiddleware,
)
from muffin_agent.utils.agent_builder import MuffinAgentBuilder

_REACT_PATCH = "muffin_agent.utils.agent_builder.create_agent"
_DEEP_PATCH = "muffin_agent.utils.agent_builder.create_deep_agent"


class _SampleOutput(BaseModel):
    """Sample response schema — the structured-output tool is named after it."""

    report: str


def _parse_error_tool_message() -> ToolMessage:
    return ToolMessage(
        content="Failed to parse structured output for tool '_SampleOutput': ...",
        tool_call_id="c1",
        name="_SampleOutput",
    )


# ── _guard behaviour ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_passes_through_when_structured_response_present():
    mw = _StructuredOutputRetryMiddleware("_SampleOutput")
    state = {
        "structured_response": _SampleOutput(report="done"),
        "messages": [_parse_error_tool_message()],
    }
    assert mw._guard(state) is None


@pytest.mark.unit
def test_jumps_to_model_after_failed_parse_without_nudge():
    """A parse-error ToolMessage IS the feedback — jump without extra messages."""
    mw = _StructuredOutputRetryMiddleware("_SampleOutput")
    state = {"messages": [AIMessage(content=""), _parse_error_tool_message()]}
    update = mw._guard(state)
    assert update == {"jump_to": "model"}


@pytest.mark.unit
def test_injects_nudge_when_schema_tool_never_called():
    """Plain-text end: no feedback in the transcript, so a nudge is added."""
    mw = _StructuredOutputRetryMiddleware("_SampleOutput")
    state = {"messages": [AIMessage(content="here is prose, no tool call")]}
    update = mw._guard(state)
    assert update is not None
    assert update["jump_to"] == "model"
    (nudge,) = update["messages"]
    assert isinstance(nudge, HumanMessage)
    assert nudge.additional_kwargs[_NUDGE_MARKER] is True
    assert "_SampleOutput" in str(nudge.content)


@pytest.mark.unit
def test_nudges_count_as_attempts():
    """A second plain-text end after a nudge is attempt 2, not attempt 0."""
    mw = _StructuredOutputRetryMiddleware("_SampleOutput", max_attempts=2)
    nudge = HumanMessage(content="nudge", additional_kwargs={_NUDGE_MARKER: True})
    state = {"messages": [AIMessage(content="prose"), nudge]}
    update = mw._guard(state)
    assert update == {"jump_to": "model"}  # attempt 1 of 2 — no second nudge

    state["messages"].append(_parse_error_tool_message())
    with pytest.raises(StructuredOutputRetryExhaustedError):
        mw._guard(state)


@pytest.mark.unit
def test_raises_after_max_attempts():
    mw = _StructuredOutputRetryMiddleware("_SampleOutput", max_attempts=3)
    state = {"messages": [_parse_error_tool_message() for _ in range(3)]}
    with pytest.raises(StructuredOutputRetryExhaustedError, match="3 attempt"):
        mw._guard(state)


@pytest.mark.unit
def test_other_tool_messages_do_not_count_as_attempts():
    mw = _StructuredOutputRetryMiddleware("_SampleOutput", max_attempts=1)
    state = {
        "messages": [
            ToolMessage(content="ok", tool_call_id="t1", name="write_todos"),
            HumanMessage(content="a real user message"),
        ]
    }
    # Zero attempts so far -> nudge, not exhaustion.
    update = mw._guard(state)
    assert update is not None and update["jump_to"] == "model"


@pytest.mark.unit
def test_exhaustion_error_is_retryable_by_node_retry_policy():
    """The error must stay a direct ``Exception`` subclass.

    langgraph's ``default_retry_on`` refuses the common builtin error types
    but retries unknown ``Exception`` subclasses — that opt-in is what lets
    the node-level ``RetryPolicy(max_attempts=2)`` re-run the agent node as
    the outer backstop.
    """
    assert issubclass(StructuredOutputRetryExhaustedError, Exception)
    assert not issubclass(
        StructuredOutputRetryExhaustedError,
        (ValueError, TypeError, RuntimeError, LookupError, OSError, ArithmeticError),
    )


# ── Builder wiring ────────────────────────────────────────────────────────


def _middleware_of(mock_create) -> list:
    _, kwargs = mock_create.call_args
    return list(kwargs["middleware"])


@pytest.mark.unit
def test_builder_wires_guard_last_for_react_with_response_format():
    with patch(_REACT_PATCH, return_value=MagicMock()) as mock_create:
        (
            MuffinAgentBuilder(MagicMock(), name="t")
            .with_response_format(AutoStrategy(schema=_SampleOutput))
            .build_react_agent()
        )
    stack = _middleware_of(mock_create)
    guard = stack[-1]
    assert isinstance(guard, _StructuredOutputRetryMiddleware)
    assert guard._schema_name == "_SampleOutput"


@pytest.mark.unit
def test_builder_wires_guard_for_deep_agent():
    with patch(_DEEP_PATCH, return_value=MagicMock()) as mock_create:
        (
            MuffinAgentBuilder(MagicMock(), name="t")
            .with_response_format(AutoStrategy(schema=_SampleOutput))
            .build_deep_agent()
        )
    stack = _middleware_of(mock_create)
    assert isinstance(stack[-1], _StructuredOutputRetryMiddleware)


@pytest.mark.unit
def test_builder_skips_guard_without_response_format():
    with patch(_REACT_PATCH, return_value=MagicMock()) as mock_create:
        MuffinAgentBuilder(MagicMock(), name="t").build_react_agent()
    stack = _middleware_of(mock_create)
    assert not any(isinstance(m, _StructuredOutputRetryMiddleware) for m in stack)


@pytest.mark.unit
def test_builder_guard_runs_after_caller_middleware_registration():
    """Caller middleware must precede the guard (guard's after_agent runs first)."""

    from langchain.agents.middleware.types import AgentMiddleware

    class _CallerMiddleware(AgentMiddleware):
        pass

    caller = _CallerMiddleware()
    with patch(_REACT_PATCH, return_value=MagicMock()) as mock_create:
        (
            MuffinAgentBuilder(MagicMock(), name="t")
            .with_response_format(AutoStrategy(schema=_SampleOutput))
            .with_middleware(caller)
            .build_react_agent()
        )
    stack = _middleware_of(mock_create)
    assert stack.index(caller) < len(stack) - 1
    assert isinstance(stack[-1], _StructuredOutputRetryMiddleware)

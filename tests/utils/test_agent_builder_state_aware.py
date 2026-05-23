"""Tests for the state-aware builder methods added to ``MuffinAgentBuilder``.

Covers:

* ``with_state_schema(schema)`` — forwards to ``create_agent``.
* ``with_runtime_system_prompt_template(template)`` —
  registers ``_RuntimePromptMiddleware`` and clears the static
  ``system_prompt`` forwarded to ``create_agent``.
* Auto-trigger: setting BOTH ``with_state_schema`` and
  ``with_response_format`` wires
  ``_StructuredResponseToStateMiddleware``; setting only one does NOT
  (backward-compatible).
* ``_RuntimePromptMiddleware`` schema-aware Jinja-var filtering
  (skips ``OmitFromSchema(input=True)`` fields and reserved
  agent-state fields).
* ``_StructuredResponseToStateMiddleware`` unpacks Pydantic + dict
  structured responses into per-field state updates.
"""

from __future__ import annotations

from typing import Annotated
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import AgentState, OmitFromSchema
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from muffin_agent.utils._runtime_prompt_middleware import _RuntimePromptMiddleware
from muffin_agent.utils._structured_response_to_state_middleware import (
    _StructuredResponseToStateMiddleware,
)
from muffin_agent.utils.agent_builder import MuffinAgentBuilder

_REACT_PATCH = "muffin_agent.utils.agent_builder.create_agent"


def _react_kwargs(mock_create_agent):
    """Return kwargs forwarded to ``create_agent`` from the most recent call."""
    assert mock_create_agent.call_count == 1
    _, kwargs = mock_create_agent.call_args
    return kwargs


# ── Fixtures: sample schemas ──────────────────────────────────────────────


class _SampleAgentState(AgentState):
    """Sample state extending ``AgentState`` with input + output extras."""

    ticker: Annotated[str, OmitFromSchema(input=False, output=True)]
    decision_date: Annotated[str, OmitFromSchema(input=False, output=True)]
    report: Annotated[str, OmitFromSchema(input=True, output=False)]


class _SampleOutput(BaseModel):
    """Sample Pydantic response — field name matches a state-schema field."""

    report: str


# ── Builder integration tests ─────────────────────────────────────────────


@pytest.mark.unit
class TestWithStateSchema:
    def test_forwards_state_schema_to_create_agent(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_state_schema(
                _SampleAgentState
            ).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        assert kwargs["state_schema"] is _SampleAgentState

    def test_state_schema_defaults_to_none(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        assert kwargs["state_schema"] is None


@pytest.mark.unit
class TestWithRuntimeSystemPromptTemplate:
    def test_clears_static_system_prompt(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_system_prompt(
                "static prompt"
            ).with_runtime_system_prompt_template("some.jinja").build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        assert kwargs["system_prompt"] is None

    def test_registers_runtime_prompt_middleware(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_state_schema(
                _SampleAgentState
            ).with_runtime_system_prompt_template("some.jinja").build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_RuntimePromptMiddleware" in types_in_stack

    def test_no_runtime_middleware_when_template_unset(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_RuntimePromptMiddleware" not in types_in_stack


@pytest.mark.unit
class TestStructuredResponseAutoTrigger:
    def test_wires_unpacker_when_both_state_schema_and_response_format_set(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            (
                MuffinAgentBuilder(MagicMock())
                .with_state_schema(_SampleAgentState)
                .with_response_format(_SampleOutput)
                .build_react_agent()
            )
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_StructuredResponseToStateMiddleware" in types_in_stack

    def test_no_unpacker_with_only_response_format(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_response_format(
                _SampleOutput
            ).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_StructuredResponseToStateMiddleware" not in types_in_stack

    def test_no_unpacker_with_only_state_schema(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_state_schema(
                _SampleAgentState
            ).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_StructuredResponseToStateMiddleware" not in types_in_stack


# ── Internal middleware unit tests ────────────────────────────────────────


@pytest.mark.unit
class TestRuntimePromptMiddleware:
    def test_extracts_input_fields_skipping_omit_input(self):
        fields = _RuntimePromptMiddleware._extract_input_fields(_SampleAgentState)
        # ticker and decision_date are input-eligible; report is omitted (input=True).
        assert "ticker" in fields
        assert "decision_date" in fields
        assert "report" not in fields

    def test_extracts_skips_reserved_fields(self):
        fields = _RuntimePromptMiddleware._extract_input_fields(_SampleAgentState)
        # messages is reserved and must not appear.
        assert "messages" not in fields
        assert "structured_response" not in fields

    @pytest.mark.asyncio
    async def test_awrap_renders_template_and_sets_system_message(self):
        """Render integrates with state and writes ``request.system_message``."""
        with patch(
            "muffin_agent.utils._runtime_prompt_middleware.render_template",
            return_value="rendered: AAPL on 2026-05-23",
        ) as mock_render:
            mw = _RuntimePromptMiddleware(
                template="trading_decision/analysts/market.jinja",
                state_schema=_SampleAgentState,
                static_partials=(),
            )
            request = MagicMock()
            request.state = {
                "messages": [HumanMessage("hi")],
                "ticker": "AAPL",
                "decision_date": "2026-05-23",
                "report": "already produced",  # output-only; must NOT be passed
            }
            handler = MagicMock()

            async def fake_handler(req):
                handler(req)
                return "done"

            result = await mw.awrap_model_call(request, fake_handler)

        assert result == "done"
        assert isinstance(request.system_message, SystemMessage)
        assert request.system_message.content == "rendered: AAPL on 2026-05-23"
        # Confirm the template received only input fields (no report, no messages).
        _, kwargs = mock_render.call_args
        assert kwargs == {"ticker": "AAPL", "decision_date": "2026-05-23"}

    @pytest.mark.asyncio
    async def test_awrap_falls_back_to_all_state_fields_without_schema(self):
        with patch(
            "muffin_agent.utils._runtime_prompt_middleware.render_template",
            return_value="x",
        ) as mock_render:
            mw = _RuntimePromptMiddleware(
                template="some.jinja",
                state_schema=None,
                static_partials=(),
            )
            request = MagicMock()
            request.state = {
                "messages": [HumanMessage("hi")],
                "foo": "bar",
                "baz": 1,
            }

            async def fake_handler(_req):
                return "ok"

            await mw.awrap_model_call(request, fake_handler)
        _, kwargs = mock_render.call_args
        # `messages` reserved → excluded; foo + baz pass through.
        assert kwargs == {"foo": "bar", "baz": 1}

    @pytest.mark.asyncio
    async def test_awrap_appends_static_partials(self):
        def fake_render(name, **vars):
            if name == "main.jinja":
                return "MAIN"
            return f"PARTIAL[{name}]"

        with patch(
            "muffin_agent.utils._runtime_prompt_middleware.render_template",
            side_effect=fake_render,
        ):
            mw = _RuntimePromptMiddleware(
                template="main.jinja",
                state_schema=_SampleAgentState,
                static_partials=("p1.jinja", "p2.jinja"),
            )
            request = MagicMock()
            request.state = {"ticker": "AAPL", "decision_date": "2026-05-23"}

            async def fake_handler(_req):
                return "ok"

            await mw.awrap_model_call(request, fake_handler)

        assert "MAIN" in request.system_message.content
        assert "PARTIAL[p1.jinja]" in request.system_message.content
        assert "PARTIAL[p2.jinja]" in request.system_message.content


@pytest.mark.unit
class TestStructuredResponseToStateMiddleware:
    @pytest.mark.asyncio
    async def test_unpacks_pydantic_into_state(self):
        mw = _StructuredResponseToStateMiddleware()
        state = {"structured_response": _SampleOutput(report="all is well")}
        update = await mw.aafter_agent(state, MagicMock())
        assert update == {"report": "all is well"}

    @pytest.mark.asyncio
    async def test_unpacks_dict_into_state(self):
        mw = _StructuredResponseToStateMiddleware()
        state = {"structured_response": {"report": "from dict"}}
        update = await mw.aafter_agent(state, MagicMock())
        assert update == {"report": "from dict"}

    @pytest.mark.asyncio
    async def test_returns_none_when_structured_response_absent(self):
        mw = _StructuredResponseToStateMiddleware()
        update = await mw.aafter_agent({}, MagicMock())
        assert update is None

    @pytest.mark.asyncio
    async def test_returns_none_for_unsupported_type(self):
        mw = _StructuredResponseToStateMiddleware()
        state = {"structured_response": "raw string"}
        update = await mw.aafter_agent(state, MagicMock())
        assert update is None

"""Tests for the state-aware builder methods added to ``MuffinAgentBuilder``.

Covers:

* ``with_state_schema(schema)`` — forwards to ``create_agent``.
* ``with_input_prompt_template(template)`` — registers
  ``_InputPromptMiddleware`` and clears the static ``system_prompt`` forwarded
  to ``create_agent`` (the task goes to the first human message instead).
* Auto-trigger: setting BOTH ``with_state_schema`` and ``with_response_format``
  wires ``_StructuredResponseToStateMiddleware``; setting only one does NOT.
* ``_InputPromptMiddleware`` — schema-aware Jinja-var filtering, first-human
  seeding (and its skip-when-a-human-message-exists rule), and partial→system
  composition.
* The universal ``_EnsureUserMessageMiddleware`` guard.
* ``_StructuredResponseToStateMiddleware`` unpacks Pydantic + dict structured
  responses into per-field state updates.
"""

from __future__ import annotations

from typing import Annotated
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware.types import AgentState, OmitFromSchema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from muffin_agent.utils._ensure_user_message_middleware import (
    _EnsureUserMessageMiddleware,
)
from muffin_agent.utils._input_prompt_middleware import _InputPromptMiddleware
from muffin_agent.utils._structured_response_to_state_middleware import (
    _StructuredResponseToStateMiddleware,
)
from muffin_agent.utils.agent_builder import MuffinAgentBuilder

_REACT_PATCH = "muffin_agent.utils.agent_builder.create_agent"
_DEEP_PATCH = "muffin_agent.utils.agent_builder.create_deep_agent"
_RENDER_PATCH = "muffin_agent.utils._input_prompt_middleware.render_template"


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


class _FakeRequest:
    """Minimal ``ModelRequest`` stand-in with a working ``override``.

    The middlewares call ``request.override(...)`` (direct attribute assignment
    is deprecated), so the double returns a NEW request carrying the override
    rather than mutating in place.
    """

    def __init__(self, state=None, system_message=None, messages=None):
        self.state = state or {}
        self.system_message = system_message
        self.messages = messages if messages is not None else []

    def override(self, **kwargs):
        new = _FakeRequest(self.state, self.system_message, self.messages)
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new


async def _run_and_capture(mw, request):
    """Invoke ``awrap_model_call`` and return the request the handler received."""
    captured = {}

    async def handler(req):
        captured["req"] = req
        return "done"

    result = await mw.awrap_model_call(request, handler)
    return captured["req"], result


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
class TestDeepAgentStateAware:
    """Deep agents support the same state-aware composition as ReAct agents."""

    def test_forwards_state_schema_to_create_deep_agent(self):
        with patch(_DEEP_PATCH) as mock_create_deep:
            MuffinAgentBuilder(MagicMock()).with_state_schema(
                _SampleAgentState
            ).build_deep_agent()
        _, kwargs = mock_create_deep.call_args
        assert kwargs["state_schema"] is _SampleAgentState

    def test_input_template_clears_static_system_prompt(self):
        with patch(_DEEP_PATCH) as mock_create_deep:
            MuffinAgentBuilder(MagicMock()).with_system_prompt(
                "static prompt"
            ).with_input_prompt_template("some.jinja").build_deep_agent()
        _, kwargs = mock_create_deep.call_args
        assert kwargs["system_prompt"] is None
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_InputPromptMiddleware" in types_in_stack

    def test_static_prompt_kept_without_input_template(self):
        with patch(_DEEP_PATCH) as mock_create_deep:
            MuffinAgentBuilder(MagicMock()).with_system_prompt(
                "static prompt"
            ).build_deep_agent()
        _, kwargs = mock_create_deep.call_args
        assert kwargs["system_prompt"] == "static prompt"

    def test_wires_unpacker_when_state_schema_and_response_format_set(self):
        with patch(_DEEP_PATCH) as mock_create_deep:
            (
                MuffinAgentBuilder(MagicMock())
                .with_state_schema(_SampleAgentState)
                .with_response_format(_SampleOutput)
                .build_deep_agent()
            )
        _, kwargs = mock_create_deep.call_args
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_StructuredResponseToStateMiddleware" in types_in_stack


@pytest.mark.unit
class TestWithInputPromptTemplate:
    def test_clears_static_system_prompt(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_system_prompt(
                "static prompt"
            ).with_input_prompt_template("some.jinja").build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        assert kwargs["system_prompt"] is None

    def test_registers_input_prompt_middleware(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).with_state_schema(
                _SampleAgentState
            ).with_input_prompt_template("some.jinja").build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_InputPromptMiddleware" in types_in_stack

    def test_no_input_middleware_when_template_unset(self):
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_InputPromptMiddleware" not in types_in_stack

    def test_guard_is_wired_universally(self):
        """The system-only safety-net is on EVERY agent, no opt-in needed."""
        with patch(_REACT_PATCH) as mock_create_agent:
            MuffinAgentBuilder(MagicMock()).build_react_agent()
        kwargs = _react_kwargs(mock_create_agent)
        types_in_stack = [type(m).__name__ for m in kwargs["middleware"]]
        assert "_EnsureUserMessageMiddleware" in types_in_stack


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
class TestInputPromptMiddleware:
    def test_extracts_input_fields_skipping_omit_input(self):
        fields = _InputPromptMiddleware._extract_input_fields(_SampleAgentState)
        # ticker and decision_date are input-eligible; report is omitted (input=True).
        assert "ticker" in fields
        assert "decision_date" in fields
        assert "report" not in fields

    def test_extracts_skips_reserved_fields(self):
        fields = _InputPromptMiddleware._extract_input_fields(_SampleAgentState)
        assert "messages" not in fields
        assert "structured_response" not in fields

    @pytest.mark.asyncio
    async def test_abefore_agent_seeds_first_human_message(self):
        """The rendered template becomes the FIRST human message (input fields)."""
        with patch(
            _RENDER_PATCH, return_value="rendered: AAPL on 2026-05-23"
        ) as mock_render:
            mw = _InputPromptMiddleware(
                template="trading_decision/analysts/market.jinja",
                state_schema=_SampleAgentState,
                static_partials=(),
            )
            update = await mw.abefore_agent(
                {
                    "messages": [],
                    "ticker": "AAPL",
                    "decision_date": "2026-05-23",
                    "report": "already produced",  # output-only; must NOT be passed
                },
                MagicMock(),
            )
        assert update is not None
        (human,) = update["messages"]
        assert isinstance(human, HumanMessage)
        assert human.content == "rendered: AAPL on 2026-05-23"
        # Confirm the template received only input fields (no report, no messages).
        _, kwargs = mock_render.call_args
        assert kwargs == {"ticker": "AAPL", "decision_date": "2026-05-23"}

    @pytest.mark.asyncio
    async def test_abefore_agent_skips_when_human_message_present(self):
        """A subagent (task tool) already carries a human turn — do not seed."""
        with patch(_RENDER_PATCH, return_value="rendered") as mock_render:
            mw = _InputPromptMiddleware(
                template="main.jinja", state_schema=None, static_partials=()
            )
            update = await mw.abefore_agent(
                {"messages": [HumanMessage("do the task")], "foo": "bar"},
                MagicMock(),
            )
        assert update is None
        mock_render.assert_not_called()

    @pytest.mark.asyncio
    async def test_abefore_agent_returns_none_for_empty_render(self):
        with patch(_RENDER_PATCH, return_value="   "):
            mw = _InputPromptMiddleware(
                template="main.jinja", state_schema=None, static_partials=()
            )
            update = await mw.abefore_agent({"messages": []}, MagicMock())
        assert update is None

    @pytest.mark.asyncio
    async def test_abefore_agent_all_state_fields_without_schema(self):
        with patch(_RENDER_PATCH, return_value="x") as mock_render:
            mw = _InputPromptMiddleware(
                template="some.jinja", state_schema=None, static_partials=()
            )
            await mw.abefore_agent(
                {"messages": [], "foo": "bar", "baz": 1}, MagicMock()
            )
        _, kwargs = mock_render.call_args
        # `messages` reserved → excluded; foo + baz pass through.
        assert kwargs == {"foo": "bar", "baz": 1}

    @pytest.mark.asyncio
    async def test_awrap_composes_partials_onto_system(self):
        def fake_render(name, **vars):
            return f"PARTIAL[{name}]"

        with patch(_RENDER_PATCH, side_effect=fake_render):
            mw = _InputPromptMiddleware(
                template="main.jinja",
                state_schema=_SampleAgentState,
                static_partials=("p1.jinja", "p2.jinja"),
            )
            request = _FakeRequest(system_message=None)
            handled, _ = await _run_and_capture(mw, request)
        # The task template is NOT in the system prompt — only the partials.
        assert "PARTIAL[p1.jinja]" in handled.system_message.content
        assert "PARTIAL[p2.jinja]" in handled.system_message.content
        assert "main.jinja" not in handled.system_message.content

    @pytest.mark.asyncio
    async def test_awrap_no_partials_leaves_system_untouched(self):
        mw = _InputPromptMiddleware(
            template="main.jinja", state_schema=None, static_partials=()
        )
        base = SystemMessage(content="DEEP AGENT BASE")
        request = _FakeRequest(system_message=base)
        handled, _ = await _run_and_capture(mw, request)
        assert handled.system_message is base

    @pytest.mark.asyncio
    async def test_awrap_partials_preserve_content_blocks(self):
        """deepagents base prompts arrive as content blocks — never wipe them."""
        with patch(_RENDER_PATCH, return_value="PARTIALS"):
            mw = _InputPromptMiddleware(
                template="main.jinja",
                state_schema=None,
                static_partials=("p1.jinja",),
            )
            request = _FakeRequest(
                system_message=SystemMessage(
                    content=[{"type": "text", "text": "DEEP AGENT BASE"}]
                )
            )
            handled, _ = await _run_and_capture(mw, request)
        content = handled.system_message.content
        assert isinstance(content, list)
        assert "PARTIALS" in content[0]["text"]
        assert content[-1] == {"type": "text", "text": "DEEP AGENT BASE"}


@pytest.mark.unit
class TestEnsureUserMessageMiddleware:
    @pytest.mark.asyncio
    async def test_injects_when_no_human_message(self):
        mw = _EnsureUserMessageMiddleware()
        request = _FakeRequest(messages=[])
        handled, result = await _run_and_capture(mw, request)
        assert result == "done"
        assert len(handled.messages) == 1
        assert isinstance(handled.messages[0], HumanMessage)

    @pytest.mark.asyncio
    async def test_noop_when_human_message_present(self):
        mw = _EnsureUserMessageMiddleware()
        original = [HumanMessage("hi"), AIMessage("ok")]
        request = _FakeRequest(messages=original)
        handled, _ = await _run_and_capture(mw, request)
        # No override → same request/messages passed through untouched.
        assert handled.messages is original

    @pytest.mark.asyncio
    async def test_prepends_when_only_non_human_messages(self):
        mw = _EnsureUserMessageMiddleware()
        request = _FakeRequest(messages=[AIMessage("prior")])
        handled, _ = await _run_and_capture(mw, request)
        assert isinstance(handled.messages[0], HumanMessage)
        assert isinstance(handled.messages[1], AIMessage)


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

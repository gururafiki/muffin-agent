"""Shared test helpers for the multi_agent conference framework.

Mirrors the pattern in ``tests/agents/test_trading_decision/conftest.py`` —
participants and judges call ``ModelConfiguration.get_chat_model_for_role``
which composes the returned chat model via ``with_fallbacks`` / ``with_retry``
/ optional ``with_structured_output``. ``FakeLLM`` short-circuits all of
those by returning ``self`` so tests drive a single configured response.
"""

from __future__ import annotations

from typing import Annotated, Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict


class FakeLLM:
    """Test double for a chat model wired through the muffin resolution chain.

    ``with_fallbacks`` / ``with_retry`` / ``with_structured_output`` all
    return ``self`` so the surrounding builder calls don't break.
    ``ainvoke`` returns the response the test configured and records the
    message list it was called with (for assertions on prompt content).
    """

    def __init__(self, response: Any):
        self.response = response
        self.invocations: list[list[Any]] = []

    def with_fallbacks(self, fallbacks):  # noqa: ARG002 — accept and ignore
        return self

    def with_retry(self, **kwargs):  # noqa: ARG002
        return self

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, messages, config=None):  # noqa: ARG002
        self.invocations.append(messages)
        return self.response


def fake_model_config(response: Any) -> tuple[MagicMock, FakeLLM]:
    """Build a stub ``ModelConfiguration`` with a single shared FakeLLM."""
    fake_llm = FakeLLM(response)
    cfg = MagicMock()
    cfg.get_llm_for_role.return_value = [fake_llm]
    return cfg, fake_llm


def fake_model_config_seq(
    *responses: Any,
) -> tuple[MagicMock, list[FakeLLM]]:
    """Build a stub returning a fresh FakeLLM per ``get_llm_for_role`` call.

    Each call pops the next response from ``responses``; the last response
    is reused if the queue runs out. Returns ``(config_mock, fakes_list)``.
    """
    cfg = MagicMock()
    queue: list[Any] = list(responses) or [AIMessage("default")]
    fakes: list[FakeLLM] = []
    counter = {"i": 0}

    def _get_llm_for_role(role: str):  # noqa: ARG001
        i = min(counter["i"], len(queue) - 1)
        counter["i"] += 1
        fake = FakeLLM(queue[i])
        fakes.append(fake)
        return [fake]

    cfg.get_llm_for_role.side_effect = _get_llm_for_role
    return cfg, fakes


def ai(text: str) -> AIMessage:
    """Build an ``AIMessage`` with text content."""
    return AIMessage(content=text)


# ── Stub compiled agents for AgentParticipant tests ──────────────────────────


class _StubAgentState(TypedDict, total=False):
    """State for the canned-response stub agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    invocation_count: int


def build_counter_stub_agent(
    *, prefix: str = "stub reply"
) -> CompiledStateGraph:
    """Compile a tiny LangGraph agent that emits a counter-incremented AIMessage.

    Useful for proving per-agent state persistence across wrapper-subgraph
    invocations: invocation 1 emits ``"<prefix> #1"``, invocation 2 emits
    ``"<prefix> #2"``, etc. The counter is stored in the agent's own state
    (under its checkpointer thread). If LangGraph subgraph namespacing
    gives cross-invocation continuation, the counter persists; otherwise
    every invocation resets to 1.
    """

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        count = (state.get("invocation_count") or 0) + 1
        return {
            "messages": [AIMessage(content=f"{prefix} #{count}")],
            "invocation_count": count,
        }

    builder: StateGraph = StateGraph(_StubAgentState)
    builder.add_node("agent", _node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    # `checkpointer=True` is the langgraph sentinel for "enable per-thread
    # persistence on this subgraph when used as a parent-graph node". The
    # parent that uses this agent must itself be compiled with a real
    # checkpointer (e.g. InMemorySaver). See:
    # https://docs.langchain.com/oss/python/langgraph/use-subgraphs
    return builder.compile(checkpointer=True)


def build_echo_stub_agent(*, response_text: str) -> CompiledStateGraph:
    """Compile a tiny agent that always emits the same response.

    Used for tests that care about message flow shape rather than per-turn
    state. Carries a checkpointer so it can be slotted into ``AgentParticipant``.
    """

    async def _node(state: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
        return {"messages": [AIMessage(content=response_text)]}

    builder: StateGraph = StateGraph(_StubAgentState)
    builder.add_node("agent", _node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    # `checkpointer=True` is the langgraph sentinel for "enable per-thread
    # persistence on this subgraph when used as a parent-graph node". The
    # parent that uses this agent must itself be compiled with a real
    # checkpointer (e.g. InMemorySaver). See:
    # https://docs.langchain.com/oss/python/langgraph/use-subgraphs
    return builder.compile(checkpointer=True)


class _RecordingStubAgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]


def build_recording_stub_agent(
    recording: list[list[BaseMessage]],
    response_text: str = "stub recorded reply",
) -> CompiledStateGraph:
    """Compile a stub agent that records the messages it RECEIVES per invocation.

    ``recording`` is a list the agent appends to on each invocation,
    capturing the input messages it saw. Useful for asserting that the
    framework's ``prep`` node sliced messages correctly (only new ones
    since the agent's last turn).
    """

    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        msgs: list[BaseMessage] = state.get("messages") or []
        recording.append(list(msgs))
        return {"messages": [AIMessage(content=response_text)]}

    builder: StateGraph = StateGraph(_RecordingStubAgentState)
    builder.add_node("agent", _node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    # `checkpointer=True` is the langgraph sentinel for "enable per-thread
    # persistence on this subgraph when used as a parent-graph node". The
    # parent that uses this agent must itself be compiled with a real
    # checkpointer (e.g. InMemorySaver). See:
    # https://docs.langchain.com/oss/python/langgraph/use-subgraphs
    return builder.compile(checkpointer=True)

"""Tests for LangFuse observability wiring (``instrument_graph``)."""

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from muffin_agent.utils.observability import instrument_graph

pytestmark = pytest.mark.unit

_LANGFUSE_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")


class _State(TypedDict):
    x: int


def _compiled_graph() -> CompiledStateGraph:
    builder = StateGraph(_State)
    builder.add_node("n", lambda s: {"x": s["x"] + 1})
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    return builder.compile()


@pytest.fixture
def _no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _LANGFUSE_ENV:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def _with_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")


def test_no_credentials_is_a_noop(_no_langfuse: None) -> None:
    """Without credentials the graph is returned untouched (no callbacks)."""
    graph = _compiled_graph()
    result = instrument_graph(graph)
    assert result is graph
    assert not (result.config or {}).get("callbacks")


def test_credentials_attach_callback_and_preserve_type(_with_langfuse: None) -> None:
    """With credentials a LangFuse handler is baked in via ``with_config``.

    The result must still be a ``CompiledStateGraph`` (``Pregel.with_config``
    returns ``Self``) so LangGraph Platform autodiscovery keeps working, and the
    original graph must be left untouched.
    """
    graph = _compiled_graph()
    result = instrument_graph(graph)

    assert isinstance(result, CompiledStateGraph)
    callbacks = (result.config or {}).get("callbacks")
    assert callbacks and len(callbacks) == 1
    # LangFuse's LangChain handler subclasses BaseCallbackHandler.
    from langchain_core.callbacks import BaseCallbackHandler

    assert isinstance(callbacks[0], BaseCallbackHandler)

    # Original graph is not mutated, and the instrumented copy still runs.
    assert not (graph.config or {}).get("callbacks")
    assert result.invoke({"x": 1}) == {"x": 2}

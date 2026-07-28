"""The two fork pins must actually be installed, and must actually work.

muffin-ui reads a run's execution tree from LangGraph's checkpoints and has NO
fallback — the reconstruction that used to paper over the first bug was deleted
on purpose, because a silent fallback hides a regression instead of reporting
one. These tests are that alarm: drop a pin before its fix ships upstream and CI
fails loudly, rather than every nested transcript quietly going empty in the UI.

Both are behavioural. They exercise the actual defect against the installed
libraries, so they keep working if upstream fixes things a different way — at
which point the pin can be dropped and these still pass.

Pins live in ``pyproject.toml``; see the comments there for removal triggers.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer
from typing_extensions import TypedDict


class _SubState(TypedDict, total=False):
    messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]


class _ParentState(TypedDict, total=False):
    messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]


def _subgraph() -> Any:
    graph = StateGraph(_SubState)
    graph.add_node("work", lambda _state: {"messages": [AIMessage("sub worked")]})
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


@pytest.mark.unit
def test_langgraph_hydrates_a_nested_subgraphs_delta_channel() -> None:
    """langchain-ai/langgraph#8470 — else every deep agent's transcript is empty.

    A subgraph from ``get_subgraphs()`` is compiled WITHOUT a checkpointer; the
    parent supplies one via config. Upstream hydrated channels from
    ``self.checkpointer`` only, so a ``DeltaChannel`` had no saver to replay its
    ancestor writes and silently read empty. An agent's ``messages`` IS a
    DeltaChannel whenever the agent uses delta storage.
    """
    child = _subgraph()
    parent = StateGraph(_ParentState)
    parent.add_node("child", child)
    parent.add_edge(START, "child")
    parent.add_edge("child", END)
    app = parent.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "1"}}
    app.invoke({}, config)

    child_ns = next(
        task.state["configurable"]["checkpoint_ns"]
        for snapshot in app.get_state_history(config)
        for task in snapshot.tasks
        if task.name == "child" and isinstance(task.state, dict)
    )
    values = app.get_state(
        {"configurable": {"thread_id": "1", "checkpoint_ns": child_ns}}
    ).values

    assert [m.content for m in values.get("messages") or []] == ["sub worked"], (
        "a nested subgraph's DeltaChannel hydrated EMPTY — the langgraph fork pin "
        "is missing or was dropped before #8470 shipped. muffin-ui has no "
        "fallback, so every nested agent's transcript is now blank in the UI."
    )


@pytest.mark.unit
def test_deep_agent_declares_its_subagent_graphs() -> None:
    """langchain-ai/deepagents#5136 — else `task` sub-agents can't be drilled into.

    Namespace resolution only knows ``add_node``-registered subgraphs, and a
    ``task`` sub-agent runs inside a TOOL — so ``get_state_history`` on its
    namespace raises ``Subgraph tools not found``. The pinned fork declares the
    compiled sub-agent graphs on the tools node, which is what makes those
    namespaces resolvable.

    Asserted on the BUILT graph rather than by running one: this needs no model
    calls, and the declaration is precisely what the pin adds. Upstream
    documents the underlying behaviour as unsupported, so unlike the langgraph
    pin this one may never be upstreamed.
    """
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    agent = create_deep_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage("done")])),
        subagents=[
            {
                "name": "helper",
                "description": "Does things.",
                "runnable": _subgraph(),
            }
        ],
    )

    assert agent.nodes["tools"].subgraphs, (
        "the deep agent's tools node declares no subgraphs — the deepagents fork "
        "pin is missing or was dropped. `task` sub-agent namespaces will raise "
        "'Subgraph tools not found', and muffin-ui's sub-agent drill-down breaks."
    )

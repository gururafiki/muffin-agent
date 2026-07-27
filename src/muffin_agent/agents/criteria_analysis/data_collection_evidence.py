"""Evidence that a criterion worker actually retrieved data.

This is **not** observability. A run's execution record — every transcript, every
tool call, at every depth — lives in LangGraph's checkpoints and is read from
there (see the graph-authoring rule in ``CLAUDE.md``). Nothing here exists to be
displayed.

It exists because the ``package`` node needs a fact for a *decision*: weak models
routinely single-shot a criterion's structured output without calling a single
tool and then fabricate plausible ``data_sources``. ``_reconcile_data_sources``
compares those claims against what actually executed, and cannot ask the model —
that is the thing being checked.

So this captures the minimum that check consumes: one short label per tool call
that really completed. No transcripts, no previews of results, no tree — those
would be telemetry, and telemetry does not belong in graph state.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Generic, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import Runtime

# Framework/filesystem plumbing a deep agent calls constantly. None of it is
# data retrieval, so counting it would defeat the check: an agent that only ever
# wrote a todo list would read as "collected data".
_PLUMBING_TOOLS: frozenset[str] = frozenset(
    {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "discover_cached_tool_outputs",
        "get_tool_output_schema",
        "write_cached_tool_output_to_backend",
    }
)

_ARGS_PREVIEW_CHARS = 200


def _args_preview(args: Any) -> str:
    """Render a tool call's arguments short and searchable."""
    if not args:
        return ""
    try:
        text = json.dumps(args, default=str)
    except (TypeError, ValueError):
        text = str(args)
    return text[:_ARGS_PREVIEW_CHARS]


def executed_tool_labels(
    messages: list[AnyMessage],
    *,
    agent_name: str | None = None,
    exclude_tools: frozenset[str] = frozenset(),
) -> list[str]:
    """Label every tool call in *messages* that actually returned.

    A call is only evidence once its ``ToolMessage`` comes back — an
    ``AIMessage`` naming a tool proves intent, not retrieval, and intent is
    exactly what a hallucinating model produces.

    The deep-agent ``task`` tool is deliberately kept: its arguments name the
    sub-agent that was delegated to, which is what a claimed ``data_sources``
    entry is usually matched against.

    *exclude_tools* must carry the agent's **structured-response schema name**.
    ``AutoStrategy`` surfaces that schema as a tool, so the synthetic final call
    that emits the answer looks exactly like a tool execution — and counting it
    would mark every evaluation as data-backed, including the single-shot
    fabrications this whole pass exists to catch.
    """
    calls: dict[str, tuple[str, Any]] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                cid, name = tc.get("id"), tc.get("name") or ""
                if cid and name:
                    calls[cid] = (name, tc.get("args", {}))

    labels: list[str] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        pair = calls.get(msg.tool_call_id)
        if pair is None:
            continue
        name, args = pair
        if name in _PLUMBING_TOOLS or name in exclude_tools:
            continue
        parts = [name, _args_preview(args)]
        if agent_name:
            parts.append(agent_name)
        labels.append(" ".join(p for p in parts if p))
    return labels


class DataCollectionEvidenceState(AgentState):
    """Declares the ``executed_tools`` channel."""

    executed_tools: NotRequired[Annotated[list[str], operator.add]]


class DataCollectionEvidenceMiddleware(
    AgentMiddleware[DataCollectionEvidenceState, ContextT],
    Generic[ContextT],
):
    """Record which tools an agent really ran, for the anti-hallucination pass.

    Reducer-backed, so a nested ``task``-invoked sub-agent's labels merge up to
    the criterion worker automatically. A parent that does not declare
    ``executed_tools`` drops them at its boundary for free — which is why this
    stays scoped to the criteria worker instead of leaking run-wide.
    """

    state_schema = DataCollectionEvidenceState

    def __init__(
        self,
        name: str | None = None,
        *,
        exclude_tools: frozenset[str] = frozenset(),
    ) -> None:
        """Initialise with the agent's label and the tool names to ignore.

        *exclude_tools* must include the agent's structured-response schema name
        — see :func:`executed_tool_labels`.
        """
        self._name = name
        self._exclude_tools = exclude_tools

    def after_agent(
        self,
        state: DataCollectionEvidenceState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Emit one label per completed tool call this agent made."""
        labels = executed_tool_labels(
            list(state.get("messages") or []),
            agent_name=self._name,
            exclude_tools=self._exclude_tools,
        )
        return {"executed_tools": labels} if labels else None

"""Child + parent middleware that capture subagent transcripts to state."""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Generic, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT
from langchain_core.messages import AnyMessage
from langgraph.runtime import Runtime

# Cap a single message's text so a chatty subagent can't bloat parent state.
_MAX_TEXT = 4000


def merge_subagent_runs(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer: accumulate subagent run records across every ``task`` call."""
    return {**(left or {}), **(right or {})}


class SubagentTranscriptState(AgentState):
    """Adds the ``subagent_runs`` channel — ``{run_id: {name, description, messages}}``."""  # noqa: E501

    subagent_runs: NotRequired[Annotated[dict[str, Any], merge_subagent_runs]]


def _text(content: Any) -> str:
    """Flatten message content (str | content blocks) to capped plain text."""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        text = "".join(parts)
    else:
        text = str(content) if content is not None else ""
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "\n… (truncated)"


def _serialize(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """Trim messages to the shape the UI's message renderer understands."""
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {
            "type": getattr(m, "type", None),
            "content": _text(getattr(m, "content", "")),
        }
        name = getattr(m, "name", None)
        if name:
            d["name"] = name
        tcid = getattr(m, "tool_call_id", None)
        if tcid:
            d["tool_call_id"] = tcid
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            d["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
                for tc in tool_calls
            ]
        out.append(d)
    return out


def _first_human(messages: list[AnyMessage]) -> str:
    for m in messages:
        if getattr(m, "type", None) == "human":
            return _text(getattr(m, "content", ""))
    return ""


def _record(name: str, messages: list[AnyMessage]) -> dict[str, Any]:
    return {
        "subagent_runs": {
            uuid.uuid4().hex[:12]: {
                "name": name,
                "description": _first_human(messages),
                "messages": _serialize(messages),
            }
        }
    }


class SubagentTranscriptMiddleware(
    AgentMiddleware[SubagentTranscriptState, ContextT],
    Generic[ContextT],
):
    """Child-side: persist this subagent's transcript into ``subagent_runs``."""

    state_schema = SubagentTranscriptState

    def __init__(self, name: str | None = None) -> None:
        """Initialise with the subagent's name (used to label its transcript)."""
        self._name = name or "subagent"

    def after_agent(
        self, state: SubagentTranscriptState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Capture this subagent's transcript into ``subagent_runs`` (sync)."""
        messages = state.get("messages") or []
        if not messages:
            return None
        return _record(self._name, messages)

    async def aafter_agent(
        self, state: SubagentTranscriptState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Capture this subagent's transcript into ``subagent_runs`` (async)."""
        messages = state.get("messages") or []
        if not messages:
            return None
        return _record(self._name, messages)


class SubagentTranscriptParentMiddleware(
    AgentMiddleware[SubagentTranscriptState, ContextT],
    Generic[ContextT],
):
    """Parent-side: declare the ``subagent_runs`` channel so merged records land."""

    state_schema = SubagentTranscriptState

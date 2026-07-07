"""``AgentCaptureMiddleware`` — one message-walk capturing transcript + tool runs.

Consolidates the former ``subagent_transcript`` and ``tool_telemetry``
middlewares (which duplicated purpose and mechanism — see roadmap). One
``aafter_agent`` pass over the agent's own messages produces:

* ``subagent_runs`` — the trimmed transcript (``{run_id: {name, description,
  messages}}``), serialized with FULL fidelity (``status`` + cache-hit kept).
  Captured for ReAct subagents always; for deep agents only when task-invoked
  (a top-level orchestrator's transcript would duplicate the thread's own
  ``messages``).
* ``tool_runs`` — compact per-tool execution records (status / cache_hit /
  args / output / error previews). Captured for EVERY agent, unconditionally —
  no runtime config gate (the previous ``tool_telemetry_enabled`` gate read the
  ambient ``get_config()`` inside ``aafter_agent``, which proved unreliable on
  the deployed LangGraph runtime). Graphs opt IN by declaring a ``tool_runs``
  channel on their state; parents that don't declare it drop the records at
  their boundary for free.

Both are reducer-backed channels, so nested ``task``-invoked subagents' captures
merge up to the parent automatically, and a compiled agent added as a graph
node surfaces them on its output state.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Generic, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT
from langgraph.config import get_config
from langgraph.runtime import Runtime

from .records import DEFAULT_EXCLUDE_TOOLS, build_tool_records, merge_tool_runs
from .serialize import first_human, serialize_messages


def merge_subagent_runs(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer: accumulate subagent run records across every ``task`` call."""
    return {**(left or {}), **(right or {})}


class AgentCaptureState(AgentState):
    """Declares both capture channels: ``subagent_runs`` + ``tool_runs``."""

    subagent_runs: NotRequired[Annotated[dict[str, Any], merge_subagent_runs]]
    tool_runs: NotRequired[Annotated[list[dict[str, Any]], merge_tool_runs]]


def _running_as_subagent() -> bool:
    """Report whether this execution was invoked by the deepagents task tool.

    The task tool stamps ``configurable.ls_agent_type = "subagent"`` on the
    config it passes to the subagent runnable (and langgraph's ``ensure_config``
    keeps it ambient for the whole nested run).
    """
    try:
        config = get_config()
    except Exception:
        return False
    configurable = config.get("configurable") if isinstance(config, dict) else None
    return (
        isinstance(configurable, dict)
        and configurable.get("ls_agent_type") == "subagent"
    )


class AgentCaptureMiddleware(
    AgentMiddleware[AgentCaptureState, ContextT],
    Generic[ContextT],
):
    """Capture this agent's transcript + tool-execution records to state.

    Args:
        name: Label for this agent's records.
        transcript_subagent_only: Capture the transcript only when actually
            running as a task-invoked subagent. Used on deep agents (usually
            top-level orchestrators whose transcript would duplicate the
            thread's own ``messages``); ReAct subagents capture always.
        exclude_tools: Tool names omitted from ``tool_runs`` (framework and
            filesystem plumbing; the builder also adds the agent's structured
            response-format schema name so the synthetic final tool call isn't
            recorded as a data-collection step).
    """

    state_schema = AgentCaptureState

    def __init__(
        self,
        name: str | None = None,
        *,
        transcript_subagent_only: bool = False,
        exclude_tools: frozenset[str] = DEFAULT_EXCLUDE_TOOLS,
    ) -> None:
        """Initialise with the agent's label + capture policy."""
        self._name = name or "agent"
        self._transcript_subagent_only = transcript_subagent_only
        self._exclude_tools = exclude_tools

    def _capture(self, state: AgentCaptureState) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        updates: dict[str, Any] = {}

        records = build_tool_records(
            messages, agent_name=self._name, exclude_tools=self._exclude_tools
        )
        if records:
            updates["tool_runs"] = records

        if not self._transcript_subagent_only or _running_as_subagent():
            updates["subagent_runs"] = {
                uuid.uuid4().hex[:12]: {
                    "name": self._name,
                    "description": first_human(messages),
                    "messages": serialize_messages(messages),
                }
            }

        return updates or None

    def after_agent(
        self, state: AgentCaptureState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Capture transcript + tool runs into state (sync)."""
        return self._capture(state)

    async def aafter_agent(
        self, state: AgentCaptureState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Capture transcript + tool runs into state (async)."""
        return self._capture(state)


class AgentCaptureParentMiddleware(
    AgentMiddleware[AgentCaptureState, ContextT],
    Generic[ContextT],
):
    """Declare both capture channels so merged-up child records land."""

    state_schema = AgentCaptureState

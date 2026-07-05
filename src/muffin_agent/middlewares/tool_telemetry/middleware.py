"""``ToolTelemetryMiddleware`` — capture per-tool execution records to state.

Off by default; enabled per-run via ``ToolTelemetryConfiguration``
(``configurable.tool_telemetry_enabled``). When on, each agent's
``aafter_agent`` reconstructs one record per completed tool call from its
own message history and returns them on the ``tool_runs`` state channel.
Nested ``task``-invoked subagents' records merge up to the parent exactly
like ``subagent_runs`` (reducer-backed channel + the deepagents task tool).
Mirrors ``SubagentTranscriptMiddleware``.
"""

from __future__ import annotations

from typing import Any, Generic

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT
from langgraph.config import get_config
from langgraph.runtime import Runtime

from .config import ToolTelemetryConfiguration
from .records import (
    DEFAULT_EXCLUDE_TOOLS,
    ToolTelemetryState,
    build_tool_records,
)


def _telemetry_enabled() -> bool:
    """Read ``tool_telemetry_enabled`` from the ambient run config."""
    try:
        config = get_config()
    except Exception:
        return False
    if not isinstance(config, dict):
        return False
    try:
        return ToolTelemetryConfiguration.from_runnable_config(
            config
        ).tool_telemetry_enabled
    except Exception:
        return False


def _running_as_subagent() -> bool:
    """Report whether this run was invoked by the deepagents ``task`` tool."""
    try:
        config = get_config()
    except Exception:
        return False
    configurable = config.get("configurable") if isinstance(config, dict) else None
    return (
        isinstance(configurable, dict)
        and configurable.get("ls_agent_type") == "subagent"
    )


class ToolTelemetryMiddleware(
    AgentMiddleware[ToolTelemetryState, ContextT],
    Generic[ContextT],
):
    """Capture this agent's tool-call records into the ``tool_runs`` channel.

    Args:
        name: Label recorded on each of this agent's tool runs.
        exclude_tools: Tool names to omit (framework/filesystem plumbing).
    """

    state_schema = ToolTelemetryState

    def __init__(
        self,
        name: str | None = None,
        *,
        exclude_tools: frozenset[str] = DEFAULT_EXCLUDE_TOOLS,
    ) -> None:
        """Initialise with this agent's telemetry label + tool exclusions."""
        self._name = name or "agent"
        self._exclude_tools = exclude_tools

    def _capture(self, state: ToolTelemetryState) -> dict[str, Any] | None:
        if not _telemetry_enabled():
            return None
        messages = state.get("messages") or []
        if not messages:
            return None
        records = build_tool_records(
            messages,
            agent_name=self._name,
            is_subagent=_running_as_subagent(),
            exclude_tools=self._exclude_tools,
        )
        if not records:
            return None
        return {"tool_runs": records}

    def after_agent(
        self, state: ToolTelemetryState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Capture this agent's tool runs into ``tool_runs`` (sync)."""
        return self._capture(state)

    async def aafter_agent(
        self, state: ToolTelemetryState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Capture this agent's tool runs into ``tool_runs`` (async)."""
        return self._capture(state)


class ToolTelemetryParentMiddleware(
    AgentMiddleware[ToolTelemetryState, ContextT],
    Generic[ContextT],
):
    """Declare the ``tool_runs`` channel so merged-up child records land."""

    state_schema = ToolTelemetryState

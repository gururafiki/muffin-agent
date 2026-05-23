"""Internal middleware that unpacks the structured response into state fields.

Wired automatically by
:class:`muffin_agent.utils.agent_builder.MuffinAgentBuilder` when both
``with_state_schema(...)`` and ``with_response_format(...)`` have been
called — the unambiguous signal that the agent is meant to be added
directly to a parent graph and the Pydantic response should flow back
as per-field state updates.

Not intended for direct use by callers.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.runtime import Runtime


class _StructuredResponseToStateMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Unpack ``state.structured_response`` into individual state fields.

    Scales to N output fields — each Pydantic-model field maps to a
    same-named state field. The parent graph reads them directly
    without needing to crack open a ``structured_response`` dict.
    """

    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        structured = state.get("structured_response")
        if structured is None:
            return None
        if hasattr(structured, "model_dump"):
            return structured.model_dump()
        if isinstance(structured, dict):
            return dict(structured)
        return None

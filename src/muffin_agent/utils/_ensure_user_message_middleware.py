"""Universal guard: guarantee the model sees at least one user turn.

Some providers reject a chat request that carries only a system message and no
user turn — Ollama Cloud (e.g. ``minimax-m3:cloud``) returns HTTP 500. Muffin's
compiled-agent-as-node stages normally get their user turn from
:class:`muffin_agent.utils._input_prompt_middleware._InputPromptMiddleware`, but
this middleware is wired for EVERY agent as a belt-and-braces safety net so any
other system-only invocation (present or future) never trips the provider.

No-op for normal agents (they always carry a human message). Not for direct use.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage


class _EnsureUserMessageMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Inject a minimal HumanMessage when a model call carries no user turn."""

    _NUDGE = "Proceed with the task described in the system prompt."

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        if not any(isinstance(m, HumanMessage) for m in request.messages):
            request = request.override(
                messages=[HumanMessage(content=self._NUDGE), *request.messages]
            )
        return await handler(request)

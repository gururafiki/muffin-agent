"""Internal middleware that renders the system prompt from runtime state.

Wired automatically by
:meth:`muffin_agent.utils.agent_builder.MuffinAgentBuilder.with_runtime_system_prompt_template`.
Not intended for direct use by callers.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
)
from langchain_core.messages import AIMessage, SystemMessage

from ..prompts import render_template


class _RuntimePromptMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Render the system prompt from agent state on each model call.

    When a state schema is supplied, fields annotated with
    ``OmitFromSchema(input=True)`` are skipped — only input-eligible
    fields flow to the template. Reserved LangChain fields
    (``messages`` / ``jump_to`` / ``structured_response`` /
    ``remaining_steps``) are always skipped. Without a schema, all
    non-reserved state fields are passed.

    Templates use ``{% if field %}`` to handle absent values.
    """

    _RESERVED: frozenset[str] = frozenset(
        {"messages", "jump_to", "structured_response", "remaining_steps"}
    )

    def __init__(
        self,
        template: str,
        state_schema: type | None,
        static_partials: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._template = template
        self._static_partials = static_partials
        self._jinja_field_names: frozenset[str] | None = (
            self._extract_input_fields(state_schema)
            if state_schema is not None
            else None
        )

    @classmethod
    def _extract_input_fields(cls, state_schema: type) -> frozenset[str]:
        """Return field names that flow IN to the agent.

        Skips fields annotated with ``OmitFromSchema(input=True)`` and
        the reserved LangChain agent-state fields.
        """
        hints = get_type_hints(state_schema, include_extras=True)
        fields: set[str] = set()
        for name, type_ in hints.items():
            if name in cls._RESERVED:
                continue
            omit_input = False
            if get_origin(type_) is Annotated:
                for meta in get_args(type_)[1:]:
                    if isinstance(meta, OmitFromSchema) and meta.input:
                        omit_input = True
                        break
            if not omit_input:
                fields.add(name)
        return frozenset(fields)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        state = request.state
        if self._jinja_field_names is not None:
            jinja_vars: dict[str, Any] = {
                k: state.get(k) for k in self._jinja_field_names
            }
        else:
            jinja_vars = {k: v for k, v in state.items() if k not in self._RESERVED}
        rendered = render_template(self._template, **jinja_vars)
        for partial in self._static_partials:
            rendered = f"{rendered}\n\n{render_template(partial)}"
        request.system_message = SystemMessage(rendered)
        return await handler(request)

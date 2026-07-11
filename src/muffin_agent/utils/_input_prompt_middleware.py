"""Internal middleware that seeds the agent's first human message from state.

Wired automatically by
:meth:`muffin_agent.utils.agent_builder.MuffinAgentBuilder.with_input_prompt_template`.

Muffin's compiled-agent-as-node stages (criteria_analysis, trading analysts,
personas, specialists) receive their task via an explicit ``input_schema`` and
would otherwise be invoked with an EMPTY ``messages`` channel — the first model
call would be *system-only*. Some providers reject that (Ollama Cloud returns
HTTP 500 on a request with no user turn). This middleware renders an
input-variable Jinja template into the **first HumanMessage** (once, at agent
start) so the task is a proper user turn — never baked into the system prompt.

The static capability partials (sandbox / memory / skills instructions) still
compose onto the *system* message: they are framework instructions and carry no
user input. Not intended for direct use by callers.
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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from ..prompts import render_template


class _InputPromptMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    """Seed the first human message from agent state; keep partials in system.

    When a state schema is supplied, fields annotated with
    ``OmitFromSchema(input=True)`` are skipped — only input-eligible fields flow
    to the template. Reserved LangChain fields (``messages`` / ``jump_to`` /
    ``structured_response`` / ``remaining_steps``) are always skipped. Without a
    schema, all non-reserved state fields are passed.

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

    def _render(self, state: AgentState[Any]) -> str:
        if self._jinja_field_names is not None:
            jinja_vars: dict[str, Any] = {
                k: state.get(k) for k in self._jinja_field_names
            }
        else:
            jinja_vars = {k: v for k, v in state.items() if k not in self._RESERVED}
        return render_template(self._template, **jinja_vars)

    async def abefore_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        """Seed the rendered template as the FIRST human message (once).

        Skips when the agent was already invoked WITH a human message — e.g. a
        subagent spawned via the deepagents ``task`` tool, which passes the task
        description as a ``HumanMessage``. In that case the caller-supplied turn
        IS the input, so seeding would be wrong (and redundant).
        """
        messages = state.get("messages") or []
        if any(isinstance(m, HumanMessage) for m in messages):
            return None
        rendered = self._render(state)
        if not rendered.strip():
            return None
        return {"messages": [HumanMessage(content=rendered)]}

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage:
        """Compose the static capability partials onto the system message.

        The task/input lives in the human message (seeded by
        :meth:`abefore_agent`); only framework instructions (no user input)
        belong in the system prompt.
        """
        if self._static_partials:
            partials = "\n\n".join(render_template(p) for p in self._static_partials)
            composed = self._compose(partials, request.system_message)
            request = request.override(system_message=composed)
        return await handler(request)

    @staticmethod
    def _compose(rendered: str, existing: SystemMessage | None) -> SystemMessage:
        """Prepend *rendered* to the existing system message, never replace.

        The request may already carry content that must survive: the deepagents
        base prompt (composed as content blocks — a list) when the agent is a
        deep agent built with ``system_prompt=None``, and any addenda injected by
        outer middleware (e.g. the ``ToolKnowledgeMiddleware`` lessons block).
        Replacing the message wholesale silently deletes both.
        """
        if existing is None:
            return SystemMessage(content=rendered)
        content = existing.content
        if isinstance(content, str):
            combined = f"{rendered}\n\n{content}" if content else rendered
            return SystemMessage(content=combined)
        if isinstance(content, list) and content:
            return SystemMessage(
                content=[{"type": "text", "text": f"{rendered}\n\n"}, *content]
            )
        return SystemMessage(content=rendered)

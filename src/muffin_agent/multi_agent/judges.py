"""Judge abstraction: optional post-conference synthesiser.

Runs once after termination to produce a structured verdict from the
full transcript. Use this when the conference's value is the final
synthesis (e.g. Investment Judge producing a directional thesis from
the Bull/Bear debate). Leave ``judge=None`` when the parent graph
consumes the raw transcript itself (the risk-debate migration does
this — the Portfolio Manager lives in the parent graph and reads the
transcript directly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from ..model_config import ModelConfiguration, Role
from ..prompts import render_template
from ._formatters import render_transcript_chronological
from .state import Turn


@runtime_checkable
class Judge(Protocol):
    """Synthesise the conference transcript into a structured verdict."""

    name: str

    async def adjudicate(
        self, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        """Read the full transcript and return the verdict as a dict."""
        ...


@dataclass
class StructuredOutputJudge:
    """Judge that produces a Pydantic-validated verdict from one LLM call.

    Template vars are the same as :class:`LLMParticipant`: state pass-through
    plus ``transcript`` (chronological text rendering of the full transcript).
    Returns ``result.model_dump()`` so callers can stash the verdict in a
    plain dict state field without leaking the Pydantic class.
    """

    name: str
    system_prompt_template: str
    output_schema: type[BaseModel]
    llm_role: Role = "reasoner"
    user_prompt: str = "Render your verdict now."

    async def adjudicate(
        self, state: dict[str, Any], config: RunnableConfig
    ) -> dict[str, Any]:
        """Render the synthesis prompt + transcript and return the verdict."""
        turns: list[Turn] = state.get("transcript") or []
        llm = ModelConfiguration.get_chat_model_for_role(
            config, self.llm_role, schema=self.output_schema
        )
        template_vars: dict[str, Any] = {
            **state,
            "transcript": render_transcript_chronological(turns),
        }
        prompt = render_template(self.system_prompt_template, **template_vars)
        result = await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage(self.user_prompt)]
        )
        # `with_structured_output(schema)` returns the validated Pydantic
        # instance; model_dump() flattens to a plain dict for state storage.
        if isinstance(result, BaseModel):
            return result.model_dump()
        # Fallback for dict-schema mode (when output_schema is a JSON Schema dict).
        return dict(result) if not isinstance(result, dict) else result

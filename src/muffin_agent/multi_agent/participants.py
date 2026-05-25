"""Participant abstraction for the multi_agent conference framework.

A ``Participant`` is one speaker in a conference. Each implements:

* ``name: str`` — used as both the graph node name and the ``Turn.speaker``
  tag in the shared transcript.
* ``async def speak(state, config) -> str`` — produces one turn of free-text
  content given the (framework-normalised) conference state.

The framework normalises state before calling ``speak``: the canonical
transcript field is ``state["transcript"]`` (a ``list[Turn]``) regardless
of the parent state's actual field name.

Two concrete implementations ship today:

* :class:`LLMParticipant` (Option α — prompt-text) — the prior transcript
  is rendered into the system prompt as text. Best for short conferences;
  matches what the legacy ``risk_debate`` / ``investment_debate`` flows do.
* :class:`LLMMessageParticipant` (Option β — message-thread) — the prior
  transcript is materialised as a ``BaseMessage`` thread. The system prompt
  holds only the role description; better for long conferences (prompt-cache
  reuse) or as an adapter to plug compiled ReAct agents into a conference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..model_config import ModelConfiguration, Role
from ..prompts import render_template
from ._formatters import last_opposing_turn, render_transcript_chronological
from .state import Turn


@runtime_checkable
class Participant(Protocol):
    """One speaker in a conference."""

    name: str

    async def speak(
        self, state: dict[str, Any], config: RunnableConfig
    ) -> str:
        """Produce one turn of free-text content for this participant."""
        ...


@dataclass
class LLMParticipant:
    """Participant that issues one LLM call per turn (Option α — prompt-text).

    Template vars made available to ``system_prompt_template``:

    * ``transcript`` — chronological text rendering of the prior transcript
      (overrides the raw list of the same name).
    * ``last_opposing_turn`` — most recent ``Turn`` by a non-self speaker,
      or ``None`` on the opening turn.
    * every other key in ``state`` (so templates reference any
      domain-specific input like ``ticker``, ``query``, ``investment_judge``,
      etc. directly).
    """

    name: str
    system_prompt_template: str
    llm_role: Role = "reasoner"
    user_prompt: str = "Take your turn now."

    async def speak(
        self, state: dict[str, Any], config: RunnableConfig
    ) -> str:
        """Render the role prompt + transcript and return one LLM turn."""
        turns: list[Turn] = state.get("transcript") or []
        llm = ModelConfiguration.get_chat_model_for_role(config, self.llm_role)
        template_vars: dict[str, Any] = {
            **state,
            "transcript": render_transcript_chronological(turns),
            "last_opposing_turn": last_opposing_turn(turns, self.name),
        }
        prompt = render_template(self.system_prompt_template, **template_vars)
        response = await llm.ainvoke(
            [SystemMessage(prompt), HumanMessage(self.user_prompt)]
        )
        return str(response.content).strip()


@dataclass
class LLMMessageParticipant:
    """Participant that materialises the transcript as a message thread (Option β).

    The system prompt holds only the participant's role description (no
    transcript). Each prior ``Turn`` becomes either an ``AIMessage`` (if it's
    this participant's own past turn) or a ``HumanMessage`` prefixed with
    ``[<speaker>]:`` for any other speaker.

    Template vars: every key in ``state`` (the raw ``transcript`` list is
    passed through but most prompts won't reference it because the transcript
    arrives via the materialised messages).
    """

    name: str
    system_prompt_template: str
    llm_role: Role = "reasoner"
    user_prompt: str = "Take your turn now."

    async def speak(
        self, state: dict[str, Any], config: RunnableConfig
    ) -> str:
        """Materialise the transcript as messages and return one LLM turn."""
        turns: list[Turn] = state.get("transcript") or []
        llm = ModelConfiguration.get_chat_model_for_role(config, self.llm_role)
        prompt = render_template(self.system_prompt_template, **state)
        history: list[AIMessage | HumanMessage] = [
            AIMessage(turn["content"])
            if turn["speaker"] == self.name
            else HumanMessage(f"[{turn['speaker']}]: {turn['content']}")
            for turn in turns
        ]
        response = await llm.ainvoke(
            [SystemMessage(prompt), *history, HumanMessage(self.user_prompt)]
        )
        return str(response.content).strip()

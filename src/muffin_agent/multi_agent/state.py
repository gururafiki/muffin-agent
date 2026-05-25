"""State types for the multi_agent conference framework.

The conference uses a single shared ``messages`` field as the canonical
inter-participant conversation. Each turn appends one ``AIMessage``
tagged with ``name=<speaker>``. Participants that wrap a compiled agent
(``AgentParticipant``) maintain their own per-thread state via the
agent's own checkpointer; the conference tracks each agent's
last-seen message id in ``agent_cursors`` so each invocation only
receives messages added since the agent last returned.

`ConferenceState` is the default state schema for `build_conference_graph`.
Callers passing a wider schema (e.g. a parent graph's state) must
include matching field names or override the framework's defaults via
the ``messages_field`` / ``next_speaker_field`` / ``verdict_field`` /
``agent_cursors_field`` parameters on ``build_conference_graph``.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ConferenceState(TypedDict, total=False):
    """Default state schema for ``build_conference_graph``."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_speaker: str | None
    agent_cursors: dict[str, str]
    verdict: dict[str, Any] | None

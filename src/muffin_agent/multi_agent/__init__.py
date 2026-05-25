"""Multi-agent conference framework.

Generic builder for "put N agents with different system prompts in a room
and let them debate / collaborate until a termination condition fires".
Four pluggable abstractions — :class:`Participant`, :class:`Moderator`,
:class:`Terminator`, :class:`Judge` — plus the entry-point
:func:`build_conference_graph`.

Three participant kinds ship today:

* :class:`LLMParticipant` — single-LLM-call, transcript rendered into the
  system prompt as text (Option α).
* :class:`LLMMessageParticipant` — single-LLM-call, prior conversation
  forwarded as a ``BaseMessage`` thread (Option β).
* :class:`AgentParticipant` — wraps a compiled muffin agent (ReAct or
  deep). Added as a parent-graph node via a thin framework-generated
  subgraph; per-agent state persists across turns via the agent's own
  checkpointer.

Example::

    from muffin_agent.multi_agent import (
        build_conference_graph,
        LLMParticipant,
        AgentParticipant,
        RoundRobinModerator,
        MaxRoundsTerminator,
    )
    from langgraph.checkpoint.memory import InMemorySaver

    bull_agent = (
        MuffinAgentBuilder(model, name="bull")
        .with_system_prompt_template("debate/bull.jinja")
        .with_tool(web_search)
        .with_checkpointer(InMemorySaver())
        .build_react_agent()
    )

    participants = [
        LLMParticipant("conservative", "debate/conservative.jinja"),
        AgentParticipant("bull", bull_agent),
    ]
    graph = build_conference_graph(
        participants=participants,
        moderator=RoundRobinModerator([p.name for p in participants]),
        terminator=MaxRoundsTerminator(max_rounds=2, num_participants=2),
    )
"""

from __future__ import annotations

from ._formatters import last_opposing_message, render_messages_chronological
from .conference import build_conference_graph
from .judges import Judge, StructuredOutputJudge
from .moderators import AlternatingModerator, Moderator, RoundRobinModerator
from .participants import (
    AgentParticipant,
    LLMMessageParticipant,
    LLMParticipant,
    Participant,
)
from .state import ConferenceState
from .terminators import MaxRoundsTerminator, Terminator

__all__ = [
    "AgentParticipant",
    "AlternatingModerator",
    "ConferenceState",
    "Judge",
    "LLMMessageParticipant",
    "LLMParticipant",
    "MaxRoundsTerminator",
    "Moderator",
    "Participant",
    "RoundRobinModerator",
    "StructuredOutputJudge",
    "Terminator",
    "build_conference_graph",
    "last_opposing_message",
    "render_messages_chronological",
]

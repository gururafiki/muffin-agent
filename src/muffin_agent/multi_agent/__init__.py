"""Multi-agent conference framework.

Generic builder for "put N agents with different system prompts in a room
and let them debate / collaborate until a termination condition fires".
The four pluggable abstractions are :class:`Participant`, :class:`Moderator`,
:class:`Terminator`, and :class:`Judge` — see :func:`build_conference_graph`
for the wiring.

Example::

    from muffin_agent.multi_agent import (
        build_conference_graph,
        LLMParticipant,
        RoundRobinModerator,
        MaxRoundsTerminator,
    )

    participants = [
        LLMParticipant("aggressive", "risk_debate/aggressive.jinja"),
        LLMParticipant("conservative", "risk_debate/conservative.jinja"),
        LLMParticipant("neutral", "risk_debate/neutral.jinja"),
    ]
    graph = build_conference_graph(
        participants=participants,
        moderator=RoundRobinModerator([p.name for p in participants]),
        terminator=MaxRoundsTerminator(max_rounds=2, num_participants=3),
    )
"""

from __future__ import annotations

from ._formatters import last_opposing_turn, render_transcript_chronological
from .conference import build_conference_graph
from .judges import Judge, StructuredOutputJudge
from .moderators import AlternatingModerator, Moderator, RoundRobinModerator
from .participants import LLMMessageParticipant, LLMParticipant, Participant
from .state import ConferenceState, Turn
from .terminators import MaxRoundsTerminator, Terminator

__all__ = [
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
    "Turn",
    "build_conference_graph",
    "last_opposing_turn",
    "render_transcript_chronological",
]

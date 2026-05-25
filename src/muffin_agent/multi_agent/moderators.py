"""Moderator abstraction: decides who speaks next given current state.

A ``Moderator`` always returns a participant name (never ``None``). The
``Terminator`` decides when the conference ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@runtime_checkable
class Moderator(Protocol):
    """Decides which participant speaks next given current state."""

    def next_speaker(self, state: dict[str, Any]) -> str:
        """Return the name of the next participant to speak."""
        ...


@dataclass
class RoundRobinModerator:
    """Cycle through ``speaker_order`` in canonical order.

    Maps ``len(state['messages']) % len(speaker_order)`` to the next
    speaker, so after each full round the cycle restarts at index 0.
    """

    speaker_order: list[str]

    def next_speaker(self, state: dict[str, Any]) -> str:
        """Return the next speaker in the round-robin cycle."""
        messages: list[BaseMessage] = state.get("messages") or []
        return self.speaker_order[len(messages) % len(self.speaker_order)]


@dataclass
class AlternatingModerator:
    """Two-speaker alternation; lead count breaks ties.

    If ``speaker_a`` has more messages than ``speaker_b``, ``speaker_b``
    goes next. Otherwise (including the opening turn), ``speaker_a`` goes.
    """

    speaker_a: str
    speaker_b: str

    def next_speaker(self, state: dict[str, Any]) -> str:
        """Return whichever speaker is behind (ties resolve to ``speaker_a``)."""
        messages: list[BaseMessage] = state.get("messages") or []
        a = sum(1 for m in messages if (m.name or "") == self.speaker_a)
        b = sum(1 for m in messages if (m.name or "") == self.speaker_b)
        return self.speaker_b if a > b else self.speaker_a

"""Terminator abstraction: decides when to end the conference.

Returns ``(True, reason)`` when the conference should end, ``(False, None)``
to continue. The reason string is for observability only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage


@runtime_checkable
class Terminator(Protocol):
    """Returns ``(True, reason)`` when the conference should end."""

    def should_stop(
        self, state: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Return ``(True, reason)`` to stop or ``(False, None)`` to continue."""
        ...


@dataclass
class MaxRoundsTerminator:
    """Stop after ``max_rounds × num_participants`` total messages.

    A round = each participant has spoken once. With ``max_rounds=2``
    and 3 participants, the conference ends after exactly 6 messages.
    """

    max_rounds: int
    num_participants: int

    def should_stop(
        self, state: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Stop once the message count hits the configured budget."""
        messages: list[BaseMessage] = state.get("messages") or []
        if len(messages) >= self.max_rounds * self.num_participants:
            return True, f"max_rounds={self.max_rounds}"
        return False, None

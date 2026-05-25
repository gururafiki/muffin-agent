"""Pure functions for rendering conference message lists.

Kept separate from the participant / judge classes so they're trivial to
unit-test and to reuse from downstream prompt-builders (e.g. a Portfolio
Manager that consumes a conference message list outside the conference
subgraph).
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def render_messages_chronological(messages: list[BaseMessage]) -> str:
    """Render shared messages as ``<Speaker>: <content>`` lines.

    Uses ``msg.name`` for the speaker prefix; falls back to the message
    class name (``HumanMessage`` / ``AIMessage``) if ``name`` is absent.
    Empty list returns the empty string.
    """
    if not messages:
        return ""
    return "\n\n".join(
        f"{m.name or type(m).__name__}: {m.content}" for m in messages
    )


def last_opposing_message(
    messages: list[BaseMessage], speaker: str
) -> BaseMessage | None:
    """Return the most recent message authored by someone OTHER than ``speaker``.

    Convenience for participants who want a direct handle on the most
    recent opponent's argument for head-on rebuttal. Returns ``None`` if
    there are no opposing messages yet (e.g. the opening turn).
    """
    for m in reversed(messages):
        if (m.name or "") != speaker:
            return m
    return None

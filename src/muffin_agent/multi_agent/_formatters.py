"""Pure functions for rendering conference transcripts.

Kept separate from the participant / judge classes so they're trivial to
unit-test and to reuse from downstream prompt-builders (e.g. a Portfolio
Manager that consumes a conference transcript outside the conference
subgraph).
"""

from __future__ import annotations

from .state import Turn


def render_transcript_chronological(turns: list[Turn]) -> str:
    """Render turns as ``Speaker: content`` lines, joined with blank lines.

    Matches the format produced by the legacy ``format_risk_history`` /
    ``format_debate_history`` helpers so migrated prompts see the same shape.
    Empty list returns the empty string.
    """
    if not turns:
        return ""
    return "\n\n".join(f"{turn['speaker']}: {turn['content']}" for turn in turns)


def last_opposing_turn(turns: list[Turn], speaker: str) -> Turn | None:
    """Return the most recent turn by anyone OTHER than ``speaker``.

    Convenience for participants who want a direct handle on the most
    recent opponent's argument for head-on rebuttal. Returns ``None`` if
    there are no opposing turns yet (e.g. the opening turn of a debate).
    """
    for turn in reversed(turns):
        if turn["speaker"] != speaker:
            return turn
    return None

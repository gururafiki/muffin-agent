"""Pure debate-transcript formatters.

Two helpers serve the two distinct shapes of transcript the package
maintains today:

* :func:`format_debate_history` — Bull/Bear two-list format (legacy
  pattern; will be retired when the bull/bear debate migrates onto the
  multi_agent conference framework).
* :func:`format_risk_history` — speaker-tagged ``Turn`` list emitted by
  the risk-debate conference subgraph. Consumed by the Portfolio Manager.

Both produce the same "<Speaker name>: <content>" line shape so prompts
that interleave the two transcripts read consistently.
"""

from __future__ import annotations

from ...multi_agent import Turn, render_transcript_chronological


def format_debate_history(bull_responses: list[str], bear_responses: list[str]) -> str:
    """Interleave Bull and Bear turns into a single chronological transcript.

    Bull speaks first in the canonical opening; subsequent pairs are
    interleaved by index. Used by both researcher prompts (so each speaker
    sees the full transcript leading up to its turn) and by the Investment
    Judge prompt for synthesis.
    """
    rows: list[str] = []
    for i in range(max(len(bull_responses), len(bear_responses))):
        if i < len(bull_responses):
            rows.append(f"Bull Researcher: {bull_responses[i]}")
        if i < len(bear_responses):
            rows.append(f"Bear Researcher: {bear_responses[i]}")
    return "\n\n".join(rows)


def format_risk_history(turns: list[Turn]) -> str:
    """Render the risk-debate conference transcript chronologically.

    Thin wrapper around :func:`render_transcript_chronological` from the
    multi_agent framework — kept here so trading_decision consumers don't
    need to import from ``multi_agent`` directly. Each ``Turn``'s speaker
    name is rendered as the line prefix (e.g. ``aggressive_debator:`` /
    ``conservative_debator:`` / ``neutral_debator:``).
    """
    return render_transcript_chronological(turns)

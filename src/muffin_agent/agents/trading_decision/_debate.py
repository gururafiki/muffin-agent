"""Pure debate-transcript formatters.

The only Python helpers shared across per-role node files. Everything else
(state reads, LLM resolution, prompt rendering, routing) lives inline at
each node's call site so reads are explicit.
"""

from __future__ import annotations


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


def format_risk_history(
    aggressive_responses: list[str],
    conservative_responses: list[str],
    neutral_responses: list[str],
) -> str:
    """Interleave Aggressive, Conservative, Neutral turns chronologically.

    Round-robin order: Aggressive → Conservative → Neutral. Used by both
    the three debator prompts (each sees full transcript) and the Portfolio
    Manager prompt for synthesis.
    """
    rows: list[str] = []
    for i in range(
        max(
            len(aggressive_responses),
            len(conservative_responses),
            len(neutral_responses),
        )
    ):
        if i < len(aggressive_responses):
            rows.append(f"Aggressive Analyst: {aggressive_responses[i]}")
        if i < len(conservative_responses):
            rows.append(f"Conservative Analyst: {conservative_responses[i]}")
        if i < len(neutral_responses):
            rows.append(f"Neutral Analyst: {neutral_responses[i]}")
    return "\n\n".join(rows)

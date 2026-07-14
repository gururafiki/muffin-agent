"""Pure debate-transcript formatters.

Both debates (Bull/Bear investment debate and the 3-way risk debate) run
on the multi_agent conference framework and accumulate name-tagged
``BaseMessage`` lists. These thin wrappers render those lists into the
``"<Speaker name>: <content>"`` transcript shape the downstream Judge /
Portfolio Manager prompts expect, so trading_decision consumers don't
need to import from ``multi_agent`` directly.

* :func:`format_debate_history` — the Bull/Bear ``investment_debate_messages``
  list (consumed by the Investment Judge).
* :func:`format_risk_history` — the risk-debate ``risk_debate_messages``
  list (consumed by the Portfolio Manager).
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

from ...multi_agent import render_messages_chronological


def format_debate_history(messages: list[BaseMessage]) -> str:
    """Render the Bull/Bear investment-debate messages chronologically.

    Each message's ``name`` is rendered as the line prefix (``bull_researcher:``
    / ``bear_researcher:``). Used by the Investment Judge prompt for synthesis.
    """
    return render_messages_chronological(messages)


def format_risk_history(messages: list[BaseMessage]) -> str:
    """Render the risk-debate conference messages chronologically.

    Each message's ``name`` is rendered as the line prefix (e.g.
    ``aggressive_debator:`` / ``conservative_debator:`` /
    ``neutral_debator:``). Used by the Portfolio Manager prompt.
    """
    return render_messages_chronological(messages)

"""Tests for the debate-history formatters in ``_debate.py``."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from muffin_agent.agents.trading_decision._debate import (
    format_debate_history,
    format_risk_history,
)


@pytest.mark.unit
class TestFormatDebateHistory:
    def test_empty(self):
        assert format_debate_history([]) == ""

    def test_full_round(self):
        # The conference framework writes name-tagged AIMessages; the
        # chronological formatter renders them as "<name>: <content>".
        messages = [
            AIMessage(content="bull 1", name="bull_researcher"),
            AIMessage(content="bear 1", name="bear_researcher"),
            AIMessage(content="bull 2", name="bull_researcher"),
            AIMessage(content="bear 2", name="bear_researcher"),
        ]
        result = format_debate_history(messages)
        assert result == (
            "bull_researcher: bull 1\n\n"
            "bear_researcher: bear 1\n\n"
            "bull_researcher: bull 2\n\n"
            "bear_researcher: bear 2"
        )

    def test_partial_round(self):
        # Only Bull has opened.
        messages = [AIMessage(content="bull 1", name="bull_researcher")]
        result = format_debate_history(messages)
        assert result == "bull_researcher: bull 1"


@pytest.mark.unit
class TestFormatRiskHistory:
    def test_empty(self):
        assert format_risk_history([]) == ""

    def test_full_round(self):
        # The conference framework writes name-tagged AIMessages; the
        # chronological formatter renders them as "<name>: <content>".
        messages = [
            AIMessage(content="agg A", name="aggressive_debator"),
            AIMessage(content="cons A", name="conservative_debator"),
            AIMessage(content="neut A", name="neutral_debator"),
        ]
        result = format_risk_history(messages)
        assert result == (
            "aggressive_debator: agg A\n\n"
            "conservative_debator: cons A\n\n"
            "neutral_debator: neut A"
        )

    def test_partial_round(self):
        # Only Aggressive and Conservative have spoken.
        messages = [
            AIMessage(content="agg A", name="aggressive_debator"),
            AIMessage(content="cons A", name="conservative_debator"),
        ]
        result = format_risk_history(messages)
        assert "aggressive_debator: agg A" in result
        assert "conservative_debator: cons A" in result
        assert "neutral_debator" not in result

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
        assert format_debate_history([], []) == ""

    def test_bull_only(self):
        result = format_debate_history(["case A"], [])
        assert result == "Bull Researcher: case A"

    def test_interleaves_in_order(self):
        result = format_debate_history(["bull 1", "bull 2"], ["bear 1", "bear 2"])
        assert result == (
            "Bull Researcher: bull 1\n\n"
            "Bear Researcher: bear 1\n\n"
            "Bull Researcher: bull 2\n\n"
            "Bear Researcher: bear 2"
        )

    def test_handles_uneven_lengths(self):
        # Bull is one turn ahead.
        result = format_debate_history(["bull 1", "bull 2"], ["bear 1"])
        assert "Bull Researcher: bull 1" in result
        assert "Bear Researcher: bear 1" in result
        assert "Bull Researcher: bull 2" in result
        assert "Bear Researcher: bear 2" not in result


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

"""Tests for the debate-history formatters in ``_debate.py``."""

from __future__ import annotations

import pytest

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
        # The conference framework's chronological formatter uses each
        # participant's `name` as the line prefix — so the rendered
        # transcript reflects the graph-node names directly.
        turns = [
            {"speaker": "aggressive_debator", "content": "agg A", "round": 1},
            {"speaker": "conservative_debator", "content": "cons A", "round": 1},
            {"speaker": "neutral_debator", "content": "neut A", "round": 1},
        ]
        result = format_risk_history(turns)
        assert result == (
            "aggressive_debator: agg A\n\n"
            "conservative_debator: cons A\n\n"
            "neutral_debator: neut A"
        )

    def test_partial_round(self):
        # Only Aggressive and Conservative have spoken.
        turns = [
            {"speaker": "aggressive_debator", "content": "agg A", "round": 1},
            {"speaker": "conservative_debator", "content": "cons A", "round": 1},
        ]
        result = format_risk_history(turns)
        assert "aggressive_debator: agg A" in result
        assert "conservative_debator: cons A" in result
        assert "neutral_debator" not in result

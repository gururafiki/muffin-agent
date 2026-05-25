"""Tests for the graph-level routing functions (``graph.py``).

These functions are called by LangGraph's conditional edges. They read
state directly (no Command, no node coupling) and return the literal
name of the next node.

The 3-way risk debate no longer has its own router — it's wired through
the multi_agent conference framework whose internal routing is exercised
in ``tests/multi_agent/test_conference.py``. The bull/bear investment
debate keeps its bespoke router and is tested here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from muffin_agent.agents.trading_decision.graph import _route_investment_debate


def _config(**configurable) -> dict:
    return {"configurable": configurable}


@pytest.mark.unit
class TestRouteInvestmentDebate:
    def test_opening_routes_to_bull(self):
        # Empty state — bull opens.
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate({}) == "bull_researcher"

    def test_after_bull_routes_to_bear(self):
        state = {"investment_bull_responses": ["a"]}
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate(state) == "bear_researcher"

    def test_after_bear_routes_to_bull(self):
        state = {
            "investment_bull_responses": ["a"],
            "investment_bear_responses": ["b"],
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate(state) == "bull_researcher"

    def test_exit_to_judge_at_default_budget(self):
        # Default = 2 rounds = 4 turns.
        state = {
            "investment_bull_responses": ["a", "c"],
            "investment_bear_responses": ["b", "d"],
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(),
        ):
            assert _route_investment_debate(state) == "investment_judge"

    def test_one_round_override(self):
        state = {
            "investment_bull_responses": ["a"],
            "investment_bear_responses": ["b"],
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(max_investment_debate_rounds=1),
        ):
            assert _route_investment_debate(state) == "investment_judge"

    @pytest.mark.parametrize(
        "rounds,bulls,bears,expected",
        [
            (3, 1, 1, "bull_researcher"),
            (3, 2, 2, "bull_researcher"),
            (3, 3, 2, "bear_researcher"),
            (3, 3, 3, "investment_judge"),
        ],
    )
    def test_3_round_progression(self, rounds, bulls, bears, expected):
        state = {
            "investment_bull_responses": ["x"] * bulls,
            "investment_bear_responses": ["x"] * bears,
        }
        with patch(
            "muffin_agent.agents.trading_decision.graph._active_config",
            return_value=_config(max_investment_debate_rounds=rounds),
        ):
            assert _route_investment_debate(state) == expected
